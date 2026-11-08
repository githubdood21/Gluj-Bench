"""Deliberate local hardware qualification; run one workload at a time.
No performance ranking is asserted. All evidence/data stays in the output folder.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import time

class Host:
    def __init__(self, worker, data):
        env=os.environ.copy(); env["GLUJ_BENCH_DATA_DIR"]=str(data.resolve())
        self.process=subprocess.Popen([str(worker.resolve()),"--stdio"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8",env=env,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        self.events=queue.Queue();self.number=0;self.stderr=[]
        def output():
            for line in self.process.stdout:
                try:self.events.put(json.loads(line))
                except ValueError:self.events.put({"type":"malformed","line":line[:500]})
        def errors():
            for line in self.process.stderr:
                self.stderr.append(line.rstrip());self.stderr=self.stderr[-100:]
        threading.Thread(target=output,daemon=True).start();threading.Thread(target=errors,daemon=True).start()
    def send(self, command, arguments=None):
        self.number+=1;ident=f"qualify-{self.number}"
        self.process.stdin.write(json.dumps({"protocol":2,"id":ident,"command":command,"arguments":arguments or {}})+"\n");self.process.stdin.flush();return ident
    def request(self,command,arguments=None,cancel_after=None,busy=False,timeout=240):
        started=time.perf_counter();ident=self.send(command,arguments);cancel_id=None;cancel_sent=None;ack=None;busy_id=None;busy_seen=False;progress=0
        while time.perf_counter()-started<timeout:
            elapsed=time.perf_counter()-started
            if busy and busy_id is None and elapsed>0.25:
                busy_id=self.send("run",arguments)
            if cancel_after is not None and cancel_id is None and elapsed>=cancel_after:
                cancel_sent=time.perf_counter();cancel_id=self.send("cancel",{"request_id":ident})
            try:event=self.events.get(timeout=0.1)
            except queue.Empty:
                if self.process.poll() is not None:raise RuntimeError(f"Worker exited {self.process.returncode}: {self.stderr}")
                continue
            if event.get("id")==busy_id:
                busy_seen=event.get("error",{}).get("code")=="busy"
            if event.get("id")==cancel_id:ack=time.perf_counter()-cancel_sent
            if event.get("id")!=ident:continue
            if event.get("type")=="progress":progress+=1;continue
            if command=="run" and event.get("type")=="success" and event.get("data",{}).get("accepted"):continue
            return event,{"wall_seconds":time.perf_counter()-started,"cancel_ack_seconds":ack,"busy_rejected":busy_seen,"progress_messages":progress}
        raise TimeoutError(f"Timed out: {command} {arguments}")
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)

def cases():
    result=[]
    def add(name,benchmark,**options):result.append((name,benchmark,options))
    for policy in ("shorter","extensive"):
        common={"measurement_policy":policy}
        add(f"cpu-register-{policy}","cpu.performance.integer.i64",cpu_core_limit="1",**common)
        add(f"gpu-vector-{policy}","gpu.performance.fp32.scaling",gpu_preset="custom",dataset_mode="single",dataset_bytes=str(1<<20),**common)
        add(f"gpu-memory-{policy}","gpu.bandwidth.vram",dataset_mode="single",dataset_bytes=str(8<<20),memory_operation="read",**common)
        add(f"transfer-{policy}","gpu.bandwidth.host_link",dataset_mode="single",dataset_bytes=str(1<<20),transfer_direction="host_to_device",**common)
    for benchmark in ("cpu.performance.float.f32","cpu.performance.float.f64","cpu.performance.avx2.f64_fma","cpu.performance.extended.aes","cpu.performance.extended.popcnt","cpu.performance.prime.sieve","cpu.bandwidth.cache.l1","cpu.bandwidth.cache.l2","cpu.bandwidth.cache.l3"):
        add(benchmark,benchmark,cpu_core_limit="1")
    for direction in ("compression","decompression"):
        add(f"codec-{direction}",f"cpu.performance.{direction}.deflate",cpu_core_limit="2",codec_corpus="mixed",codec_corpus_bytes=str(1<<20),codec_block_bytes="65536",access_pattern="shuffled_blocks",access_realizations="3")
    add("cpu-buffered-ensemble","cpu.performance.avx2.f32_fma.scaling",cpu_core_limit="2",vector_preset="custom",dataset_mode="single",dataset_bytes=str(2<<20),access_pattern="shuffled_blocks",access_realizations="3")
    add("cpu-buffered-sequential","cpu.performance.avx2.f32_fma.scaling",cpu_core_limit="2",vector_preset="custom",dataset_mode="single",dataset_bytes=str(2<<20))
    add("cpu-fp64-buffered","cpu.performance.avx2.f64_fma",cpu_core_limit="1",fp64_preset="custom",dataset_mode="single",dataset_bytes=str(1<<20),access_pattern="rotating_regions")
    add("cpu-popcnt-buffered","cpu.performance.extended.popcnt",cpu_core_limit="1",popcnt_preset="custom",dataset_mode="single",dataset_bytes=str(1<<20))
    add("cpu-ascii","cpu.performance.string.ascii",cpu_core_limit="2",ascii_preset="custom",dataset_mode="single",dataset_bytes=str(1<<20),ascii_content="mixed")
    add("cpu-matrix-fixed","cpu.performance.matrix.fp32.scaling",cpu_core_limit="2",matrix_shape="square",matrix_reuse="4",matrix_k_block="128")
    add("cpu-ram-bandwidth","cpu.bandwidth.memory",cpu_core_limit="2",dataset_mode="single",dataset_bytes=str(8<<20))
    add("cpu-cache-latency","cpu.latency.cache.l1")
    add("cpu-ram-latency","cpu.latency.memory",dataset_bytes=str(128<<20))
    for precision in ("fp16","fp64"):
        add(f"gpu-{precision}",f"gpu.performance.{precision}",gpu_preset="custom",dataset_mode="single",dataset_bytes=str(1<<20),access_pattern="lane_scattered")
    for pattern in ("sequential","shuffled_blocks","rotating_regions","lane_scattered"):
        add(f"memory-{pattern}","gpu.bandwidth.cache",dataset_mode="single",dataset_bytes=str(3<<20),memory_copy_path="shader",access_pattern=pattern,access_realizations="3" if pattern=="shuffled_blocks" else "1")
    add("native-copy-sweep","gpu.bandwidth.vram",dataset_mode="sweep",dataset_bytes=str(3<<20),memory_operation="copy",memory_copy_path="native")
    add("cache-fresh-discovery","gpu.bandwidth.cache")
    for ring,submission,batch in ((1,"serial",1),(3,"batched",8),(8,"serial",1),(8,"batched",32)):
        add(f"transfer-ring{ring}-{submission}","gpu.bandwidth.host_link",dataset_mode="sweep",dataset_bytes=str(1<<20),transfer_direction="both",staging_buffers=str(ring),transfer_submission=submission,transfer_batch=str(batch))
    for share in (50,100):
        add(f"offload-{share}","gpu.performance.fp32.offload.scaling",gpu_offload_preset="custom",dataset_mode="single",dataset_bytes=str((4 if share==50 else 64)<<20),ram_offload_percent=str(share),offload_passes_per_second="10" if share==50 else "1000",arithmetic_iterations="64" if share==50 else "1024")
    for kind,key in (("fp16","gpu_matrix_preset"),("int8","gpu_int8_matrix_preset")):
        for traversal in ("sequential","shuffled_tiles","rotating_tiles"):
            add(f"matrix-{kind}-{traversal}",f"gpu.performance.matrix.{kind}.scaling",**{key:"custom","dataset_mode":"single","dataset_bytes":str(1<<20),"matrix_tile_reuse":"16","matrix_traversal":traversal,"access_realizations":"3" if traversal=="shuffled_tiles" else "1"})
    add("sustained-memory","gpu.bandwidth.vram",dataset_mode="single",dataset_bytes=str(8<<20),memory_operation="read",sustained_seconds="15")
    add("sustained-cpu","cpu.performance.integer.i64",cpu_core_limit="2",sustained_seconds="15")
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--only",default="");args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True);data=root/"data";data.mkdir(exist_ok=True)
    host=Host(args.worker,data);summary=[]
    def save(): (root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    try:
        for command in ("devices","benchmarks","families"):
            response,_=host.request(command);(root/f"{command}.json").write_text(json.dumps(response,indent=2),encoding="utf-8")
        for name,benchmark,options in cases():
            if args.only and args.only not in name:continue
            options={"measurement_policy":"shorter",**options};config={"benchmark_id":benchmark,"target_duration_ms":1,"samples":1,"options":options}
            response,timing=host.request("run",config)
            (root/f"{name.replace('.','-')}.json").write_text(json.dumps(response),encoding="utf-8")
            record=response.get("run_record",{});error=response.get("error",{});passed=response.get("type")=="result" and record.get("sample_coverage")=="complete" and bool(record.get("samples")) and response.get("run_storage",{}).get("saved") is True
            if passed:
                saved=json.loads((data/"run-history"/"runs"/(record["run_id"]+".json")).read_text(encoding="utf-8"));passed=saved==record
            row={"case":name,"benchmark":benchmark,"passed":passed,"error":error,"status":record.get("status"),"coverage":record.get("sample_coverage"),"raw_samples":len(record.get("samples",[])),"validation":record.get("validation"),**timing};summary.append(row);save();print(json.dumps(row),flush=True)
        if not args.only:
            boundaries=[
                ("reject-memory-over-budget","gpu.bandwidth.vram",{"dataset_mode":"single","dataset_bytes":str(1<<40)}),
                ("reject-transfer-over-budget","gpu.bandwidth.host_link",{"dataset_mode":"single","dataset_bytes":str(1<<40),"staging_buffers":"8"}),
                ("reject-native-irregular","gpu.bandwidth.vram",{"dataset_mode":"single","dataset_bytes":str(1<<20),"memory_copy_path":"native","access_pattern":"shuffled_blocks"}),
                ("reject-unsupported-tile","gpu.performance.matrix.fp16.scaling",{"gpu_matrix_preset":"custom","dataset_mode":"single","dataset_bytes":str(1<<20),"matrix_tile_shape":"3x3x3:fp16:fp32"}),
                ("reject-register-access","gpu.performance.fp16",{"gpu_preset":"register","dataset_mode":"register","access_pattern":"lane_scattered"}),
            ]
            for name,benchmark,options in boundaries:
                response,timing=host.request("run",{"benchmark_id":benchmark,"target_duration_ms":1,"samples":1,"options":{"measurement_policy":"shorter",**options}})
                (root/f"{name}.json").write_text(json.dumps(response),encoding="utf-8");row={"case":name,"passed":response.get("type")=="error" and response.get("error",{}).get("code")!="busy" and response.get("run_record",{}).get("result") is None,"error":response.get("error"),**timing};summary.append(row);save();print(json.dumps(row),flush=True)
            options={"measurement_policy":"extensive","cpu_core_limit":"1","sustained_seconds":"30"};response,timing=host.request("run",{"benchmark_id":"cpu.performance.integer.i64","options":options,"target_duration_ms":3000,"samples":12},cancel_after=4,busy=True)
            (root/"cancel-busy.json").write_text(json.dumps(response),encoding="utf-8");record=response.get("run_record",{});row={"case":"cancel-busy","passed":record.get("status")=="cancelled" and record.get("result") is None and bool(record.get("samples")) and timing["busy_rejected"] and timing["cancel_ack_seconds"] is not None and timing["cancel_ack_seconds"]<2,"raw_samples":len(record.get("samples",[])),**timing};summary.append(row);save();print(json.dumps(row),flush=True)
    finally:host.close()
    print(f"Completed {len(summary)} cases; {sum(not r['passed'] for r in summary)} failures.",flush=True)
    return int(any(not row['passed'] for row in summary))
if __name__=="__main__":raise SystemExit(main())

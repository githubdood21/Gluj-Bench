# Gluj-Bench user guide

Current benchmarks are synthetic tests, with throughput and latency reported for the settings you choose.

## Run your first benchmark

1. Extract the Windows x64 ZIP into a writable folder. Open `gluj-bench-ui.exe` with `gluj-bench-worker.exe` alongside it.
2. **System Information** shows the hardware and capabilities the app detected.
3. Open **Benchmarks**, choose a category, and select the GPU adapter where applicable. The adapter selector applies to all sets.
4. Choose a row in **Test sets** and click **Run**. Curated sets supply fixed workloads, precision coverage, dataset policies, and extensive sampling. Click a row to inspect its included tests. Preparation, calibration, and sweeps add time.
5. Open **Benchmark Results** and click a result row. Use the component picker to switch between CPU and GPU results.

During a run, **Cancel test** stops the active test and leaves the remaining queue in place. **Stop all tests** also clears the pending queue. Available observations from a cancelled test can be inspected in Run History.

## Create your own set

1. Choose a category and hardware in **Benchmarks**, then click **New set** in **Test sets**.
2. Name your set on the right and click **Add test**.
3. Choose a **Workload**. GPU performance offers **Vector** or **Matrix**, with supported **Precision** and **Data placement** options. Registers measure compute throughput; memory workloads expose dataset controls.
4. Choose **One dataset** or **Sweep to target** and set the size. Workload-specific controls appear where supported. **Advanced** contains secondary settings. **Shorter run** applies only to this test.
5. Click **Add to set**. Repeat to add other tests, such as FP16 and FP32 with different dataset sizes. **Cancel** returns without adding anything.
6. Click **Save set**. The collection appears under **Your sets** in the selected category. Its **Run** button executes the saved collection.

Select your saved set to rename it or change its contents. **Edit** opens one test's settings; **Apply changes** updates that test, and **Cancel** leaves it unchanged. **Remove** removes a test from the collection. Click **Save set** to persist your changes to the same entry. **Delete set** removes the saved collection, leaving recorded results intact.

Curated sets have a lock icon and are strictly read-only. Selecting one shows its tests and **Settings** for inspection. Create a **New set** to configure your own benchmarks; curated sets cannot be copied into the editor.

Each test retains its own configuration and sampling policy. Switching GPU adapters applies the selected adapter to GPU tests, with compatibility checked before running. Hover the small **?** circles for a short explanation of what a setting changes. Keyboard focus shows the same help; **Escape** dismisses it. The **Precision** hint lists formats reported as unsupported by the selected GPU, one per line. The CPU **Variant** hint lists unsupported instruction modes such as AVX2/FMA, even when only a scalar option remains. Hardware **Technical properties** retains the full details. Saved sets show their workloads, datatypes, and test count in the library.

Sequential access is the standard choice. Custom workloads offer randomized access where supported. A fixed seed in **Advanced** reproduces the access order.

## Read the results

| Readout | Meaning |
| --- | --- |
| **TOPS** | Trillions of operations per second for the selected workload. A fused multiply-add counts as two operations. |
| **GB/s** | Billions of bytes per second. Memory read, write, and copy are separate measurements. |
| **ns per access** | Time per dependent access. Lower values indicate lower latency. |
| **Median and observed range** | The middle measurement and the lowest-to-highest values recorded across observations. |

On a compute sweep, the horizontal axis shows dataset size. Hover a point to see its exact value and click it to copy the value. A dashed line marks the compute reference when available. A percentage beside the graph shows the measured drop at the largest dataset when that comparison applies.

Compare matching workloads, precision, sizes, and settings. Use **Compare with** to choose a saved baseline. For per-dataset data and individual observations, open **Details & measurements**.

## Keep and share measurements

- **Run history** opens saved and cancelled runs. Select a row to open that measurement. Its **Actions** menu lets you edit notes, pin a baseline, or archive the run.
- **Share results** creates a PNG preview. Check the results you want in the image, then use **Save PNG**. The original PNG also carries comparison summaries that can be loaded with **Import snapshot**.
- For full raw observations, check runs in **Run history** and choose **Files → Export native JSON**. Use **Import native JSON** from the same menu to load records for comparison.
- When updating or moving the app, keep `run-history/`, `results.json`, `settings.json`, and `benchmark-settings.json` beside the executables.

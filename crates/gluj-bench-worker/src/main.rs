use gluj_bench_core::{
    BenchmarkConfig, BenchmarkRegistry, CancellationToken, ProtocolCommand, RunRecord, RunStore,
    benchmark_error_response, error_response, parse_request, progress_response, result_response,
    success_response,
};
use gluj_bench_cpu::CpuBandwidthProvider;
use gluj_bench_gpu::GpuBandwidthProvider;
use serde_json::{Value, json};
use std::{
    env,
    io::{self, BufRead, Write},
    process::ExitCode,
    sync::{Arc, Mutex, mpsc},
    thread,
};

fn build_registry() -> BenchmarkRegistry {
    let mut registry = BenchmarkRegistry::default();
    registry.add(Box::new(CpuBandwidthProvider::discover()));
    registry.add(Box::new(GpuBandwidthProvider::discover()));
    registry
}

fn usage() {
    eprintln!(
        "Usage:\n  gluj-bench-worker devices [--json]\n  gluj-bench-worker benchmarks [--json]\n  gluj-bench-worker families [--json]\n  gluj-bench-worker resolve <benchmark-id> [--config <file>] [--json]\n  gluj-bench-worker run <benchmark-id> [--config <file>] [--json]\n  gluj-bench-worker --stdio"
    );
}

fn run_store() -> RunStore {
    RunStore::new(gluj_bench_core::app_data_path("run-history"))
}

/// Persist before responding. Storage failure is visible while the complete record remains
/// in the response, allowing a caller to save it elsewhere without losing measurements.
fn recorded_response(request_id: &str, record: &RunRecord) -> Value {
    let storage = run_store().save(record);
    let mut response = match &record.result {
        Some(result) => result_response(request_id, result),
        None => benchmark_error_response(
            request_id,
            record.error.clone().unwrap_or_else(|| {
                gluj_bench_core::BenchmarkError::new("incomplete_run", "The run did not complete.")
            }),
        ),
    };
    response["run_record"] = json!(record);
    response["run_storage"] = match storage {
        Ok(()) => json!({ "saved": true, "run_id": record.run_id }),
        Err(error) => json!({ "saved": false, "error": error }),
    };
    response
}

fn tag_template(record: &mut RunRecord, template: Option<&str>) {
    let Some((id, version)) = template
        .filter(|value| value.len() <= 128)
        .and_then(|value| value.rsplit_once('@'))
    else {
        return;
    };
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        || version
            .parse::<u32>()
            .ok()
            .is_none_or(|version| version == 0)
    {
        return;
    }
    if let Some(result) = &mut record.result {
        result
            .workload_metadata
            .insert("template_id".into(), id.into());
        result
            .workload_metadata
            .insert("template_version".into(), version.into());
    }
}

#[derive(Clone)]
struct ActiveRun {
    request_id: String,
    cancellation: CancellationToken,
}

fn platform_fingerprint() -> String {
    [
        std::env::consts::OS.to_owned(),
        std::env::consts::ARCH.to_owned(),
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_default(),
        std::env::var("NUMBER_OF_PROCESSORS").unwrap_or_default(),
    ]
    .join("|")
}

fn stdio_host() -> io::Result<()> {
    // Creating providers probes topology and GPU capabilities. Keep that work lazy so a cached
    // profile can be shown after only the inexpensive platform fingerprint request.
    let registry: Arc<Mutex<Option<BenchmarkRegistry>>> = Arc::new(Mutex::new(None));
    let active: Arc<Mutex<Option<ActiveRun>>> = Arc::new(Mutex::new(None));
    let (output, receiver) = mpsc::channel::<Value>();
    let writer = thread::spawn(move || -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        while let Ok(message) = receiver.recv() {
            writeln!(stdout, "{message}")?;
            stdout.flush()?;
        }
        Ok(())
    });

    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(problem) => {
                let _ = output.send(error_response(
                    "unknown",
                    "invalid_json",
                    problem.to_string(),
                ));
                continue;
            }
        };
        let request = match parse_request(&value) {
            Ok(request) => request,
            Err(error) => {
                let _ = output.send(error);
                continue;
            }
        };
        match request.command {
            ProtocolCommand::Fingerprint => {
                let _ = output.send(success_response(
                    &request.id,
                    json!({ "fingerprint": platform_fingerprint() }),
                ));
            }
            ProtocolCommand::Devices => {
                let mut registry = registry.lock().expect("registry lock");
                let registry = registry.get_or_insert_with(build_registry);
                let _ = output.send(success_response(
                    &request.id,
                    json!({ "devices": registry.devices() }),
                ));
            }
            ProtocolCommand::Benchmarks => {
                let mut registry = registry.lock().expect("registry lock");
                let registry = registry.get_or_insert_with(build_registry);
                let _ = output.send(success_response(
                    &request.id,
                    json!({ "benchmarks": registry.benchmarks(), "conditions": registry.conditions() }),
                ));
            }
            ProtocolCommand::Families => {
                let mut registry = registry.lock().expect("registry lock");
                let registry = registry.get_or_insert_with(build_registry);
                let _ = output.send(success_response(
                    &request.id,
                    json!({ "families": registry.families() }),
                ));
            }
            ProtocolCommand::Resolve {
                benchmark_id,
                config,
            } => {
                let mut registry = registry.lock().expect("registry lock");
                let registry = registry.get_or_insert_with(build_registry);
                let response = match registry.resolve(&benchmark_id, &config) {
                    Ok(condition) => {
                        success_response(&request.id, json!({ "condition": condition }))
                    }
                    Err(error) => benchmark_error_response(&request.id, error),
                };
                let _ = output.send(response);
            }
            ProtocolCommand::Cancel { request_id } => {
                let running = active.lock().expect("active lock").clone();
                match running.filter(|run| run.request_id == request_id) {
                    Some(run) => {
                        run.cancellation.cancel();
                        let _ = output.send(success_response(
                            &request.id,
                            json!({ "cancelled_request_id": request_id }),
                        ));
                    }
                    None => {
                        let _ = output.send(error_response(
                            &request.id,
                            "not_running",
                            "The requested benchmark is not running.",
                        ));
                    }
                }
            }
            ProtocolCommand::Run {
                benchmark_id,
                config,
            } => {
                let template_id = value["arguments"]["template_id"]
                    .as_str()
                    .map(str::to_owned);
                let mut current = active.lock().expect("active lock");
                if current.is_some() {
                    let _ = output.send(error_response(
                        &request.id,
                        "busy",
                        "Another benchmark is already running.",
                    ));
                    continue;
                }
                let cancellation = CancellationToken::default();
                *current = Some(ActiveRun {
                    request_id: request.id.clone(),
                    cancellation: cancellation.clone(),
                });
                drop(current);
                let _ = output.send(success_response(
                    &request.id,
                    json!({ "accepted": true, "benchmark_id": benchmark_id }),
                ));
                let registry = registry.clone();
                let active = active.clone();
                let output = output.clone();
                let request_id = request.id;
                thread::spawn(move || {
                    let mut progress = |update| {
                        let _ = output.send(progress_response(&request_id, &update));
                    };
                    let mut result = registry
                        .lock()
                        .expect("registry lock")
                        .get_or_insert_with(build_registry)
                        .run_recorded(&benchmark_id, &config, &cancellation, &mut progress);
                    tag_template(&mut result, template_id.as_deref());
                    let response = recorded_response(&request_id, &result);
                    let mut current = active.lock().expect("active lock");
                    if current
                        .as_ref()
                        .is_some_and(|run| run.request_id == request_id)
                    {
                        *current = None;
                    }
                    drop(current);
                    let _ = output.send(response);
                });
            }
        }
    }
    drop(output);
    writer
        .join()
        .map_err(|_| io::Error::other("stdout writer panicked"))?
}

fn run_cli(
    registry: &mut BenchmarkRegistry,
    id: &str,
    config: &BenchmarkConfig,
    json_output: bool,
) -> ExitCode {
    let cancellation = CancellationToken::default();
    let mut progress = |update: gluj_bench_core::ProgressUpdate| {
        eprintln!("{:>3.0}% {}", update.fraction * 100.0, update.message)
    };
    let record = registry.run_recorded(id, config, &cancellation, &mut progress);
    let response = recorded_response("cli-1", &record);
    if response["run_storage"]["saved"] == false {
        eprintln!(
            "Could not save native run: {}",
            response["run_storage"]["error"]["message"]
        );
    }
    match record
        .result
        .ok_or_else(|| record.error.expect("unsuccessful run has an error"))
    {
        Ok(result) => {
            if json_output {
                println!("{response}");
            } else {
                for metric in result.metrics {
                    println!(
                        "{}: {}",
                        metric.name,
                        format_metric(metric.value, &metric.unit)
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Err(problem) => {
            if json_output {
                println!("{response}");
            } else {
                eprintln!("{problem}");
            }
            ExitCode::from(3)
        }
    }
}

fn format_metric(value: f64, unit: &str) -> String {
    if unit == "bytes/s" {
        return format!("{:.2} GB/s", value / 1e9);
    }
    if unit == "operations/s" {
        let (scale, suffix) = if value >= 1e12 {
            (1e12, "TOPS")
        } else if value >= 1e6 {
            (1e6, "MOPS")
        } else if value >= 1e3 {
            (1e3, "KOPS")
        } else {
            (1.0, "OPS")
        };
        return format!("{} {suffix}", format_grouped_2(value / scale));
    }
    if unit == "MB/s" {
        return format!("{} MB/s", format_grouped_2(value));
    }
    if unit == "strings/s" {
        let (scale, suffix) = if value >= 1e6 {
            (1e6, "M strings/s")
        } else if value >= 1e3 {
            (1e3, "K strings/s")
        } else {
            (1.0, "strings/s")
        };
        return format!("{} {suffix}", format_grouped_2(value / scale));
    }
    if unit == "primes/s" {
        let (scale, suffix) = if value >= 1e6 {
            (1e6, "M primes/s")
        } else if value >= 1e3 {
            (1e3, "K primes/s")
        } else {
            (1.0, "primes/s")
        };
        return format!("{} {suffix}", format_grouped_2(value / scale));
    }
    format!("{value:.2} {unit}")
}

fn format_grouped_2(value: f64) -> String {
    let raw = format!("{:.2}", value.abs());
    let (integer, fraction) = raw.split_once('.').unwrap_or((&raw, "00"));
    let mut grouped = String::with_capacity(raw.len() + integer.len() / 3);
    if value.is_sign_negative() {
        grouped.push('-');
    }
    for (index, character) in integer.chars().enumerate() {
        if index != 0 && (integer.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(character);
    }
    grouped.push('.');
    grouped.push_str(fraction);
    grouped
}

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments == ["--stdio"] {
        return stdio_host()
            .map(|_| ExitCode::SUCCESS)
            .unwrap_or_else(|problem| {
                eprintln!("Worker I/O failed: {problem}");
                ExitCode::FAILURE
            });
    }
    if arguments.is_empty() {
        usage();
        return ExitCode::from(2);
    }
    let mut registry = build_registry();
    let json_output = arguments.iter().any(|argument| argument == "--json");
    match arguments[0].as_str() {
        "devices" if arguments.len() == 1 + usize::from(json_output) => {
            if json_output {
                println!(
                    "{}",
                    success_response("cli-1", json!({ "devices": registry.devices() }))
                );
            } else {
                for item in registry.devices() {
                    println!("{} | {} | {}", item.id, item.name, item.status);
                }
            }
            ExitCode::SUCCESS
        }
        "benchmarks" if arguments.len() == 1 + usize::from(json_output) => {
            if json_output {
                println!(
                    "{}",
                    success_response("cli-1", json!({ "benchmarks": registry.benchmarks() }))
                );
            } else {
                for item in registry.benchmarks() {
                    println!(
                        "{} | {}",
                        item.id,
                        if item.available {
                            "available"
                        } else {
                            &item.unavailable_reason
                        }
                    );
                }
            }
            ExitCode::SUCCESS
        }
        "families" if arguments.len() == 1 + usize::from(json_output) => {
            println!(
                "{}",
                success_response("cli-1", json!({"families": registry.families()}))
            );
            ExitCode::SUCCESS
        }
        "run" | "resolve" if arguments.len() >= 2 => {
            let config = match cli_config(&arguments[2..]) {
                Ok(config) => config,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            if arguments[0] == "run" {
                run_cli(&mut registry, &arguments[1], &config, json_output)
            } else {
                match registry.resolve(&arguments[1], &config) {
                    Ok(condition) => {
                        println!(
                            "{}",
                            success_response("cli-1", json!({"condition": condition}))
                        );
                        ExitCode::SUCCESS
                    }
                    Err(error) => {
                        println!("{}", benchmark_error_response("cli-1", error));
                        ExitCode::from(3)
                    }
                }
            }
        }
        _ => {
            usage();
            ExitCode::from(2)
        }
    }
}

fn cli_config(arguments: &[String]) -> Result<BenchmarkConfig, String> {
    let mut path = None;
    let mut json_seen = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--json" if !json_seen => {
                json_seen = true;
            }
            "--config" if path.is_none() => {
                index += 1;
                path = Some(arguments.get(index).ok_or("--config requires a filename")?);
            }
            option => return Err(format!("Unknown or repeated argument '{option}'.")),
        }
        index += 1;
    }
    match path {
        Some(path) => serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string()),
        None => Ok(BenchmarkConfig::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gluj_bench_core::BenchmarkProvider;

    #[test]
    fn template_identity_is_attached_before_native_record_storage() {
        let mut record = RunRecord::start("test", &BenchmarkConfig::default(), vec![]);
        record.result = Some(gluj_bench_core::BenchmarkResult {
            benchmark_id: "test".into(),
            device_id: "device".into(),
            elapsed_ns: 1,
            metrics: vec![],
            workload_metadata: Default::default(),
            device_metadata: Default::default(),
        });
        tag_template(&mut record, Some("gpu.vector.peak@1"));
        let metadata = &record.result.as_ref().unwrap().workload_metadata;
        assert_eq!(metadata["template_id"], "gpu.vector.peak");
        assert_eq!(metadata["template_version"], "1");
        let saved = metadata.clone();
        for invalid in [
            None,
            Some("no-version"),
            Some("test@0"),
            Some("test@bad"),
            Some("test/injected@1"),
        ] {
            tag_template(&mut record, invalid);
            assert_eq!(record.result.as_ref().unwrap().workload_metadata, saved);
        }
    }

    #[test]
    fn operation_rates_use_millions_instead_of_giga_prefixes() {
        assert_eq!(
            format_metric(127_274_779_471.0, "operations/s"),
            "127,274.78 MOPS"
        );
        assert_eq!(
            format_metric(1_800_000_000_000.0, "operations/s"),
            "1.80 TOPS"
        );
        assert_eq!(format_metric(3_744.186_882_15, "MB/s"), "3,744.19 MB/s");
        assert_eq!(
            format_metric(1_141_432_000.0, "strings/s"),
            "1,141.43 M strings/s"
        );
        assert_eq!(format_metric(12_345_678.0, "primes/s"), "12.35 M primes/s");
    }

    #[test]
    fn gpu_ids_are_vendor_neutral() {
        assert!(
            GpuBandwidthProvider::discover()
                .devices()
                .iter()
                .all(|device| device.id.starts_with("gpu:vulkan:"))
        );
    }
}

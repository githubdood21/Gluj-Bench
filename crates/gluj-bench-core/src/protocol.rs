use crate::{BenchmarkConfig, BenchmarkError, BenchmarkResult, PROTOCOL_VERSION, ProgressUpdate};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq)]
pub enum ProtocolCommand {
    Devices,
    Benchmarks,
    Run {
        benchmark_id: String,
        config: BenchmarkConfig,
    },
    Cancel {
        request_id: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolRequest {
    pub id: String,
    pub command: ProtocolCommand,
}

pub fn error_response(id: &str, code: &str, message: impl Into<String>) -> Value {
    json!({ "protocol": PROTOCOL_VERSION, "id": id, "type": "error",
        "error": { "code": code, "message": message.into() } })
}

pub fn success_response(id: &str, data: Value) -> Value {
    json!({ "protocol": PROTOCOL_VERSION, "id": id, "type": "success", "data": data })
}

pub fn progress_response(id: &str, progress: &ProgressUpdate) -> Value {
    json!({ "protocol": PROTOCOL_VERSION, "id": id, "type": "progress", "data": {
        "fraction": progress.fraction.clamp(0.0, 1.0), "phase": &progress.phase,
        "message": &progress.message } })
}

pub fn result_response(id: &str, result: &BenchmarkResult) -> Value {
    json!({ "protocol": PROTOCOL_VERSION, "id": id, "type": "result", "data": result })
}

pub fn benchmark_error_response(id: &str, error: BenchmarkError) -> Value {
    error_response(id, &error.code, error.message)
}

fn arguments<'a>(
    object: &'a Map<String, Value>,
    id: &str,
) -> Result<&'a Map<String, Value>, Value> {
    object
        .get("arguments")
        .and_then(Value::as_object)
        .ok_or_else(|| error_response(id, "invalid_request", "Command arguments are required."))
}

pub fn parse_request(value: &Value) -> Result<ProtocolRequest, Value> {
    let Some(object) = value.as_object() else {
        return Err(error_response(
            "unknown",
            "invalid_request",
            "The request must be a JSON object.",
        ));
    };
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if id.is_empty() || id == "unknown" {
        return Err(error_response(
            id,
            "invalid_request",
            "A non-empty string request id is required.",
        ));
    }
    if object.get("protocol").and_then(Value::as_u64) != Some(PROTOCOL_VERSION.into()) {
        return Err(error_response(
            id,
            "unsupported_protocol",
            format!("Protocol version {PROTOCOL_VERSION} is required."),
        ));
    }
    let Some(command) = object.get("command").and_then(Value::as_str) else {
        return Err(error_response(
            id,
            "invalid_request",
            "A string command is required.",
        ));
    };
    let command = match command {
        "devices" => ProtocolCommand::Devices,
        "benchmarks" => ProtocolCommand::Benchmarks,
        "run" => {
            let arguments = arguments(object, id)?;
            let Some(benchmark_id) = arguments.get("benchmark_id").and_then(Value::as_str) else {
                return Err(error_response(
                    id,
                    "invalid_request",
                    "A benchmark_id is required.",
                ));
            };
            let mut config = BenchmarkConfig::default();
            if let Some(duration) = arguments.get("target_duration_ms").and_then(Value::as_u64) {
                config.target_duration_ms = duration;
            }
            if let Some(samples) = arguments.get("samples").and_then(Value::as_u64) {
                config.samples = u32::try_from(samples)
                    .map_err(|_| error_response(id, "invalid_request", "samples is too large."))?;
            }
            if let Some(options) = arguments.get("options") {
                let Some(options) = options.as_object() else {
                    return Err(error_response(
                        id,
                        "invalid_request",
                        "options must be an object containing string values.",
                    ));
                };
                for (name, value) in options {
                    let Some(value) = value.as_str() else {
                        return Err(error_response(
                            id,
                            "invalid_request",
                            "Every benchmark option must be a string.",
                        ));
                    };
                    config.options.insert(name.clone(), value.to_owned());
                }
            }
            ProtocolCommand::Run {
                benchmark_id: benchmark_id.to_owned(),
                config,
            }
        }
        "cancel" => {
            let arguments = arguments(object, id)?;
            let Some(request_id) = arguments.get("request_id").and_then(Value::as_str) else {
                return Err(error_response(
                    id,
                    "invalid_request",
                    "A request_id is required.",
                ));
            };
            ProtocolCommand::Cancel {
                request_id: request_id.to_owned(),
            }
        }
        _ => {
            return Err(error_response(
                id,
                "unknown_command",
                format!("Unknown command '{command}'."),
            ));
        }
    };
    Ok(ProtocolRequest {
        id: id.to_owned(),
        command,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_version_two_run_configuration() {
        let request = parse_request(&json!({ "protocol": 2, "id": "run-1", "command": "run",
            "arguments": { "benchmark_id": "cpu.bandwidth.cache.l1", "samples": 7,
                "options": { "thread_mode": "logical_processors" } } }))
        .unwrap();
        let ProtocolCommand::Run {
            benchmark_id,
            config,
        } = request.command
        else {
            panic!("expected run")
        };
        assert_eq!(benchmark_id, "cpu.bandwidth.cache.l1");
        assert_eq!(config.samples, 7);
        assert_eq!(
            config.options.get("thread_mode").map(String::as_str),
            Some("logical_processors")
        );
    }

    #[test]
    fn rejects_old_protocol_versions() {
        let error =
            parse_request(&json!({ "protocol": 1, "id": "x", "command": "devices" })).unwrap_err();
        assert_eq!(error["error"]["code"], "unsupported_protocol");
    }

    #[test]
    fn parses_correlated_cancellation() {
        let request = parse_request(
            &json!({ "protocol": 2, "id": "cancel-1", "command": "cancel",
            "arguments": { "request_id": "run-1" } }),
        )
        .unwrap();
        assert_eq!(
            request.command,
            ProtocolCommand::Cancel {
                request_id: "run-1".into()
            }
        );
    }

    #[test]
    fn progress_is_clamped_and_correlated() {
        let response = progress_response(
            "p",
            &ProgressUpdate {
                fraction: 2.0,
                phase: "read".into(),
                message: String::new(),
            },
        );
        assert_eq!(response["id"], "p");
        assert_eq!(response["data"]["fraction"], 1.0);
    }
}

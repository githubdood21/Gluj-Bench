//! Explicit GPU memory/transfer conditions. Every curve keeps one kernel/path.
use super::*;
use crate::vulkan_bandwidth::{self, TransferOptions, VulkanBandwidthContext};
fn number(c: &BenchmarkConfig, key: &str, default: u64) -> u64 {
    c.options
        .get(key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn choice<'a>(c: &'a BenchmarkConfig, key: &str, default: &'a str) -> &'a str {
    c.options.get(key).map(String::as_str).unwrap_or(default)
}
fn sizes(c: &BenchmarkConfig, limit: u64) -> Result<Vec<u64>, BenchmarkError> {
    let mode = choice(c, "dataset_mode", "automatic");
    let requested = if mode == "automatic" {
        limit.min(64 * MIB)
    } else {
        number(c, "dataset_bytes", 0)
    };
    if requested < 65536 || requested > limit {
        return Err(BenchmarkError::new(
            "invalid_working_set",
            "Requested dataset must fit both allocation budgets and device limits; minimum is 64 KiB.",
        ));
    }
    let end = requested / 1024 * 1024;
    if mode != "sweep" {
        return Ok(vec![end]);
    }
    let mut values = Vec::new();
    let mut n = 65536;
    while n < end {
        values.push(n);
        n *= 2;
    }
    values.push(end);
    Ok(values)
}
pub(super) fn run(
    record: &AdapterRecord,
    id: &str,
    c: &BenchmarkConfig,
    cancel: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    let started = Instant::now();
    let transfer = gluj_bench_core::gpu_transfer(id);
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let vram = gluj_bench_core::vram_budget_bytes(
        record.vulkan.device_local_memory_bytes,
        gluj_bench_core::vram_budget_percent(c)?,
    )
    .saturating_sub(16 * MIB);
    let ring = number(c, "staging_buffers", 3) as usize;
    let host = gluj_bench_core::ram_allocation_budget(
        system.total_memory(),
        system.available_memory(),
        number(c, "ram_budget_percent", 20) as u32,
    );
    let host = if record.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        host.min(
            record
                .vulkan
                .host_memory_heap_bytes
                .saturating_sub(16 * MIB),
        )
    } else {
        host
    };
    let limit = record
        .vulkan
        .max_storage_buffer_range
        .min(if transfer {
            vram.min(host / ring.max(1) as u64)
        } else {
            vram.saturating_sub(4 * MIB) / 33 * 16
        })
        .min(
            if record.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                u64::MAX
            } else {
                system.available_memory() / 4
            },
        );
    let limit = if transfer {
        limit
    } else {
        limit.min(local_memory_limit(record, c)?)
    };
    let points = sizes(c, limit)?;
    let context = VulkanBandwidthContext::new(record)?;
    let mut metrics = Vec::new();
    let mut metadata = bandwidth_metadata(
        record,
        if transfer {
            "cpu_wall_clock_end_to_end"
        } else {
            "gpu_timestamp"
        },
    );
    let pattern = choice(c, "access_pattern", "sequential");
    let mode = match pattern {
        "sequential" => 0,
        "shuffled_blocks" => 1,
        "rotating_regions" => 2,
        "lane_scattered" => 3,
        _ => {
            return Err(BenchmarkError::new(
                "invalid_config",
                "Unsupported local-memory access pattern.",
            ));
        }
    };
    let native = choice(
        c,
        "memory_copy_path",
        if id == VRAM_ID { "native" } else { "shader" },
    ) == "native";
    let operation = choice(c, "memory_operation", "all");
    if !transfer && native && mode != 0 && matches!(operation, "all" | "copy") {
        return Err(BenchmarkError::new(
            "invalid_config",
            "Native copy requires sequential access; select shader copy for irregular access.",
        ));
    }
    let reals = if transfer {
        1
    } else {
        number(c, "access_realizations", 1) as u32
    };
    let root = number(c, "access_seed", gluj_bench_core::DEFAULT_ACCESS_SEED);
    let flavor = if id == CACHE_ID {
        KernelFlavor::Cache
    } else {
        KernelFlavor::Stream
    };
    let mut peak = 0;
    for (point, &size) in points.iter().enumerate() {
        progress(ProgressUpdate {
            fraction: point as f64 / points.len() as f64,
            phase: "gpu_memory_conditions".into(),
            message: format!("Measuring {} bytes with separate operation results", size),
        });
        if transfer {
            let opts = TransferOptions {
                ring,
                batch: number(c, "transfer_batch", 8) as u32,
                serial: choice(c, "transfer_submission", "batched") == "serial",
            };
            let direction = choice(c, "transfer_direction", "both");
            for (name, to_device) in [("host_to_device", true), ("device_to_host", false)] {
                if direction != "both" && direction != name {
                    continue;
                }
                let values = if to_device {
                    vulkan_bandwidth::measure_host_to_device_config(
                        &context,
                        size,
                        c.samples,
                        per_sample_duration(c),
                        cancel,
                        opts,
                    )?
                } else {
                    vulkan_bandwidth::measure_device_to_host_config(
                        &context,
                        size,
                        c.samples,
                        per_sample_duration(c),
                        cancel,
                        opts,
                    )?
                };
                let times = values
                    .iter()
                    .map(|rate| size as f64 / rate * 1e9)
                    .collect::<Vec<_>>();
                let mut duration = metric(
                    format!("working_set_{size}.{name}_mean_request_cost"),
                    times,
                );
                duration.unit = "ns".into();
                metrics.push(duration);
                metrics.push(metric(format!("working_set_{size}.{name}"), values.clone()));
                if point == points.len() - 1 {
                    metrics.push(metric(name.into(), values));
                }
            }
            peak = peak.max(size * (ring as u64 + 1));
        } else {
            for op in [
                KernelOperation::Read,
                KernelOperation::Write,
                KernelOperation::Copy,
            ] {
                if operation != "all" && operation != op.name() {
                    continue;
                }
                let mut values = Vec::new();
                let dispatch = if flavor == KernelFlavor::Cache && op == KernelOperation::Read {
                    size.max(64 * MIB)
                } else {
                    size
                };
                peak = peak.max(
                    size * 2
                        + if op == KernelOperation::Copy && native {
                            0
                        } else if op == KernelOperation::Read {
                            dispatch / 16 / op.accesses_per_invocation(flavor) * 4
                        } else {
                            4
                        },
                );
                for realization in 0..reals {
                    let seed = gluj_bench_core::derive_seed(root, 0, size, realization as u64);
                    let _real = gluj_bench_core::MeasurementScope::realization(
                        seed,
                        BTreeMap::from([
                            ("generator_revision".into(), "gpu-memory-affine-1".into()),
                            ("root_seed".into(), root.to_string()),
                            ("dataset_bytes".into(), size.to_string()),
                            ("realization".into(), realization.to_string()),
                        ]),
                    );
                    values.extend(vulkan_bandwidth::measure_gpu_operation_access(
                        &context,
                        DispatchMeasurement {
                            size,
                            dispatch_bytes: dispatch,
                            operation: op,
                            flavor,
                            precondition: true,
                            sample_count: c.samples,
                            target_per_sample: per_sample_duration(c),
                        },
                        cancel,
                        native,
                        mode,
                        seed,
                    )?);
                }
                metrics.push(metric(
                    format!("working_set_{size}.{}", op.name()),
                    values.clone(),
                ));
                if point == points.len() - 1 {
                    metrics.push(metric(op.name().into(), values));
                }
            }
        }
    }
    metadata.extend(BTreeMap::from([("kernel_revision".into(),if transfer {"host-transfer-configurable-2"}else{"local-memory-access-2"}.into()),("dataset_mode".into(),choice(c,"dataset_mode","automatic").into()),("working_set_bytes".into(),points.last().unwrap().to_string()),("resolved_dataset_bytes".into(),points.last().unwrap().to_string()),("allocated_test_buffer_bytes".into(),peak.to_string()),("allocation_budget_bytes".into(),(vram+if transfer {host}else{0}).to_string()),("result_validation".into(),"boundary/interior known source/write/checksum patterns outside timing; all used staging slots checked".into())]));
    if transfer {
        for key in [
            "transfer_direction",
            "staging_buffers",
            "transfer_batch",
            "transfer_submission",
        ] {
            metadata.insert(
                key.into(),
                choice(
                    c,
                    key,
                    match key {
                        "transfer_direction" => "both",
                        "staging_buffers" => "3",
                        "transfer_batch" => "8",
                        _ => "batched",
                    },
                )
                .into(),
            );
        }
        metadata.insert(
            "transfer_buffer_bytes".into(),
            points.last().unwrap().to_string(),
        );
        metadata.insert(
            "host_transfer_path".into(),
            "preallocated_staging_ring_native_copy".into(),
        );
        metadata.insert("request_duration_definition".into(),"mean cost per copy = observed timed submission/wait/host visibility cost divided by actual completed copies; serial waits per request, batched waits per configured group; not individual request completion latency or percentiles".into());
        metadata.insert("host_allocation_limit_bytes".into(), host.to_string());
        metadata.insert(
            "staging_memory_placement".into(),
            if record.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                "host-visible memory from a non-device-local heap; BAR/VRAM excluded"
            } else {
                "host-visible shared-memory allocation; not a discrete PCIe-link claim"
            }
            .into(),
        );
        metadata.insert("device_allocation_limit_bytes".into(), vram.to_string());
    } else {
        metadata.insert("memory_operation".into(), operation.into());
        metadata.insert(
            "memory_copy_path".into(),
            if !matches!(operation, "all" | "copy") {
                "not_applicable"
            } else if native {
                "native"
            } else {
                "shader"
            }
            .into(),
        );
        metadata.insert(
            "memory_kernel_identity".into(),
            format!(
                "{:?}; {}; each operation kept separate",
                flavor,
                if operation == "all" {
                    format!(
                        "read/write shaders; {} copy",
                        if native { "native command" } else { "shader" }
                    )
                } else {
                    format!(
                        "{operation}: {}",
                        if operation == "copy" && native {
                            "native command"
                        } else {
                            "shader"
                        }
                    )
                }
            ),
        );
        metadata.insert("access_pattern".into(), pattern.into());
        metadata.insert("access_seed".into(), root.to_string());
        metadata.insert("access_realizations".into(), reals.to_string());
        metadata.insert(
            "access_revision".into(),
            "gpu-memory-affine-1; 64-vector blocks; one output owner; no cache-miss counters"
                .into(),
        );
        metadata.insert(
            "cache_calibration_policy".into(),
            "explicit condition; no cache-capacity classification or discovery reuse".into(),
        );
        normalize_copy_presentation(&mut metrics, &mut metadata);
    }
    progress(done("GPU memory conditions completed."));
    Ok(BenchmarkResult {
        benchmark_id: id.into(),
        device_id: record.id.clone(),
        elapsed_ns: duration_ns(started.elapsed()),
        metrics,
        workload_metadata: metadata,
        device_metadata: bandwidth_device_metadata(record),
    })
}

pub(super) fn permutation(count: u32, seed: u64) -> (u32, u32) {
    let count = count.max(1);
    let bound = (u32::MAX / (count - 1).max(1)).clamp(1, 127);
    let mut mul = ((seed as u32) % bound).max(1);
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }
    while gcd(mul, count) != 1 {
        mul = if mul >= bound { 1 } else { mul + 1 };
    }
    (mul, ((seed >> 32) as u32) % count)
}

pub(super) fn local_memory_limit(
    record: &AdapterRecord,
    c: &BenchmarkConfig,
) -> Result<u64, BenchmarkError> {
    let budget = gluj_bench_core::vram_budget_bytes(
        record.vulkan.device_local_memory_bytes,
        gluj_bench_core::vram_budget_percent(c)?,
    )
    .saturating_sub(20 * MIB);
    let mut limit = record.vulkan.max_storage_buffer_range.min(budget / 33 * 16);
    if record.vulkan.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        limit = limit.min(system.available_memory() / 4);
    }
    Ok(limit / 1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_sizes_keep_alignment_endpoints_and_never_enlarge_or_clamp_over_budget() {
        let mut c = BenchmarkConfig {
            options: BTreeMap::from([
                ("dataset_mode".into(), "sweep".into()),
                ("dataset_bytes".into(), "900001".into()),
            ]),
            ..Default::default()
        };
        let values = sizes(&c, 1048576).unwrap();
        assert_eq!(values[0], 65536);
        assert_eq!(*values.last().unwrap(), 899072);
        assert!(values.windows(2).all(|p| p[0] < p[1]));
        assert!(values.iter().all(|n| n % 1024 == 0));
        c.options.insert("dataset_mode".into(), "single".into());
        assert_eq!(sizes(&c, 1048576).unwrap(), vec![899072]);
        assert!(sizes(&c, 65536).is_err());
        c.options.insert("dataset_bytes".into(), "1".into());
        assert!(sizes(&c, 1048576).is_err());
    }
    #[test]
    fn shuffled_blocks_preserve_single_writer_ownership_for_non_power_of_two_sizes() {
        for count in [1, 3, 17, 257, 4096] {
            for seed in [0, 1, 42, u64::MAX] {
                let (mul, offset) = permutation(count, seed);
                let values = (0..count)
                    .map(|i| {
                        ((u64::from(i) * u64::from(mul) + u64::from(offset)) % u64::from(count))
                            as u32
                    })
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(values.len(), count as usize);
                assert!(
                    u64::from(count - 1) * u64::from(mul) + u64::from(offset) <= u32::MAX as u64
                );
            }
        }
        let (mul, offset) = permutation(u32::MAX, u64::MAX);
        assert_eq!(mul, 1);
        assert!(u64::from(u32::MAX - 1) * u64::from(mul) + u64::from(offset) <= u32::MAX as u64);
    }
    #[test]
    fn shader_modular_addition_matches_wide_arithmetic_at_the_index_limits() {
        for count in [1, 3, 65535, 67108863, u32::MAX] {
            for seed in [0, 42, 0xfffffffe0000002a, u64::MAX] {
                let (mul, offset) = permutation(count, seed);
                for index in [0, count / 2, count - 1] {
                    let product = index
                        .checked_mul(mul)
                        .expect("bounded shader multiplication");
                    let base = product % count;
                    let actual = if base >= count - offset {
                        base - (count - offset)
                    } else {
                        base + offset
                    };
                    let expected = ((u64::from(index) * u64::from(mul) + u64::from(offset))
                        % u64::from(count)) as u32;
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

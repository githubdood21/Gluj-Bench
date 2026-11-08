//! Small, untimed boundary/interior readbacks; these do not claim full-buffer coverage.
use ash::vk;
use gluj_bench_core::BenchmarkError;

#[allow(clippy::too_many_arguments)]
pub(super) fn read_words(
    device: &ash::Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    family: u32,
    source: vk::Buffer,
    source_size: u64,
) -> Result<Vec<(usize, u32)>, BenchmarkError> {
    let _phase_timer = gluj_bench_core::phase_timer("validation");
    let indices = gluj_bench_core::validation_indices(source_size as usize / 4);
    if indices.is_empty() {
        return Err(problem("Cannot validate an empty GPU buffer."));
    }
    let size = (indices.len() * 4) as u64;
    // SAFETY: handles are local, ranges are checked, the queue is idle before host reads
    // and every successfully allocated object is destroyed on the error paths below.
    unsafe {
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
            .map_err(problem)?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let kind = (0..properties.memory_type_count).find(|index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[*index as usize]
                    .property_flags
                    .contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
        });
        let Some(kind) = kind else {
            device.destroy_buffer(buffer, None);
            return Err(problem(
                "Host-coherent validation readback memory is unavailable.",
            ));
        };
        let memory = match device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(kind),
            None,
        ) {
            Ok(memory) => memory,
            Err(error) => {
                device.destroy_buffer(buffer, None);
                return Err(problem(error));
            }
        };
        let pool = match device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(family),
            None,
        ) {
            Ok(pool) => pool,
            Err(error) => {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                return Err(problem(error));
            }
        };
        let outcome = (|| {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(problem)?;
            let command = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(problem)?[0];
            device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(problem)?;
            let before = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &before,
                &[],
                &[],
            );
            let regions: Vec<_> = indices
                .iter()
                .enumerate()
                .map(|(to, from)| {
                    vk::BufferCopy::default()
                        .src_offset((*from * 4) as u64)
                        .dst_offset((to * 4) as u64)
                        .size(4)
                })
                .collect();
            device.cmd_copy_buffer(command, source, buffer, &regions);
            let after = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &after,
                &[],
                &[],
            );
            device.end_command_buffer(command).map_err(problem)?;
            let commands = [command];
            let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
            device
                .queue_submit(queue, &submits, vk::Fence::null())
                .map_err(problem)?;
            device.queue_wait_idle(queue).map_err(problem)?;
            let mapped = device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(problem)?;
            let words = std::slice::from_raw_parts(mapped.cast::<u32>(), indices.len()).to_vec();
            device.unmap_memory(memory);
            Ok(indices.into_iter().zip(words).collect())
        })();
        let _ = device.queue_wait_idle(queue);
        device.destroy_command_pool(pool, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        outcome
    }
}
pub(super) fn problem(error: impl std::fmt::Display) -> BenchmarkError {
    BenchmarkError::new("gpu_result_validation_failed", error.to_string())
}

/// Round positive finite values to binary16, including ties-to-even.
pub(super) fn half(value: f32) -> f32 {
    if value == 0.0 {
        return value;
    }
    let magnitude = value.abs();
    let exponent = magnitude.log2().floor() as i32;
    let step = 2.0_f32.powi((exponent - 10).max(-24));
    let rounded = (magnitude / step).round_ties_even() * step;
    if rounded >= 65536.0 {
        f32::INFINITY.copysign(value)
    } else {
        rounded.copysign(value)
    }
}

pub(super) fn vector_reference(
    kind: crate::compute::ComputeKind,
    index: u32,
    loops: u32,
) -> [f32; 4] {
    use crate::compute::ComputeKind;
    if kind == ComputeKind::Fp16 {
        let base = half(((index ^ 0x9e37_79b9) & 255) as f32 * 0.001 + 0.25);
        let mut accumulators = [[0.0; 2]; 16];
        for (chain, values) in accumulators.iter_mut().enumerate() {
            for (lane, value) in values.iter_mut().enumerate() {
                let initial = half(base + half(if lane == 0 { 0.01 } else { 0.02 }));
                *value = if chain == 0 {
                    initial
                } else {
                    half(initial + half((chain as f32 * 2.0 + 1.0) / 100.0))
                };
                for _ in 0..loops {
                    *value = half(gluj_bench_core::reference_fma_f32(
                        *value,
                        if chain % 2 == 0 {
                            1.000_976_6
                        } else {
                            0.999_511_7
                        },
                        (chain + 1) as f32 / 1024.0,
                    ));
                }
            }
        }
        let sum = |lane| {
            accumulators
                .iter()
                .skip(1)
                .fold(accumulators[0][lane], |total, value| {
                    half(total + value[lane])
                })
        };
        return [
            sum(0),
            sum(1),
            half(accumulators[0][0] + accumulators[15][1]),
            half(accumulators[7][0] + accumulators[8][1]),
        ];
    }
    let mut result = [0.0; 4];
    for (lane, result) in result.iter_mut().enumerate() {
        if kind == ComputeKind::Fp64 {
            let base = f64::from((index ^ 0x9e37_79b9) & 1023) * 0.0001 + 0.25;
            let initial = base + (lane + 1) as f64 / 100.0;
            let mut total = 0.0;
            for chain in 0..8 {
                let mut value = initial + chain as f64 * 0.05;
                let multiplier = if chain % 2 == 0 {
                    1.0 + (chain / 2 + 1) as f64 * 1e-9
                } else {
                    1.0 - (chain / 2 + 1) as f64 * 1e-9
                };
                for _ in 0..loops {
                    value = value.mul_add(multiplier, (chain + 1) as f64 * 1e-9);
                }
                total += value;
            }
            *result = total as f32;
        } else {
            let base = ((index ^ 0x9e37_79b9) & 1023) as f32 * 0.0001 + 0.25;
            let initial = base + (lane + 1) as f32 / 100.0;
            let mut total = 0.0;
            for chain in 0..8 {
                let mut value = initial + chain as f32 * 0.05;
                let multiplier = if chain % 2 == 0 {
                    1.0 + (chain / 2 + 1) as f32 * 1e-6
                } else {
                    1.0 - (chain / 2 + 1) as f32 * 1e-6
                };
                for _ in 0..loops {
                    value = gluj_bench_core::reference_fma_f32(
                        value,
                        multiplier,
                        (chain + 1) as f32 * 1e-6,
                    );
                }
                total += value;
            }
            *result = total;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary16_rounding_and_reference_boundaries_are_finite() {
        assert_eq!(half(1.0), 1.0);
        assert_eq!(half(1.0 + 1.0 / 2048.0), 1.0);
        assert_eq!(half(1.0 + 3.0 / 2048.0), 1.0 + 2.0 / 1024.0);
        assert_eq!(half(2.0_f32.powi(-24)), 2.0_f32.powi(-24));
        assert_eq!(half(2.0_f32.powi(-25)), 0.0);
        for kind in [
            crate::compute::ComputeKind::Fp16,
            crate::compute::ComputeKind::Fp32,
            crate::compute::ComputeKind::Fp64,
        ] {
            for index in [0, 1, 127, 128, 255, 1023, 65535] {
                assert!(
                    vector_reference(kind, index, 1024)
                        .iter()
                        .all(|value| value.is_finite())
                );
            }
        }
    }
}

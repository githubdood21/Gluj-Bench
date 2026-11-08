//! Fixed-rate arrivals independent of GPU completion; bounded physical submissions.
use super::{VulkanComputeProfileHarness, error, timestamp_delta};
use ash::vk;
use gluj_bench_core::{BenchmarkError, CancellationToken};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};

const QUEUE_DEPTH: usize = 8;
#[derive(Clone, Copy)]
struct Schedule {
    rate: f64,
    seconds: f64,
    total: u64,
}
impl Schedule {
    fn new(rate: f64, seconds: f64) -> Self {
        // Whole arrival periods avoid crediting a start-of-window burst at low rates.
        let total = (seconds.max(1.0 / rate) * rate - 1e-9).ceil().max(1.0) as u64;
        Self {
            rate,
            seconds: total as f64 / rate,
            total,
        }
    }
    fn arrived(self, elapsed: f64) -> u64 {
        ((elapsed * self.rate).floor() as u64)
            .saturating_add(1)
            .min(self.total)
    }
    fn arrival(self, index: u64) -> f64 {
        index as f64 / self.rate
    }
}

pub(crate) struct OfferedSample {
    pub completion_delays_ns: Vec<f64>,
    pub request_window_ns: f64,
    pub offered: u64,
    pub submitted: u64,
    pub completed: u64,
    pub window_ns: f64,
    pub gpu_ns: f64,
    pub completion_ns: f64,
    pub queue_wait_ns: f64,
    pub oldest_pending_ns: f64,
    pub mean_pending_ns: f64,
    pub youngest_pending_ns: f64,
}
struct Slot {
    command: vk::CommandBuffer,
    queries: vk::QueryPool,
    fence: vk::Fence,
    request: Option<(u64, f64)>,
}
pub(crate) struct OfferedQueue<'a> {
    harness: &'a VulkanComputeProfileHarness,
    slots: Vec<Slot>,
}
impl VulkanComputeProfileHarness {
    pub fn offered_queue(
        &self,
        bytes: u64,
        loops: u32,
    ) -> Result<OfferedQueue<'_>, BenchmarkError> {
        if bytes != self.actual_working_set(bytes) {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "Offered requests require an aligned dataset within the allocated buffers.",
            ));
        }
        // SAFETY: the worker exclusively owns this queue and pool; Drop drains all submissions.
        unsafe {
            let commands = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(QUEUE_DEPTH as u32),
                )
                .map_err(|p| error("vulkan_command_buffer_failed", p))?;
            // Register every owned handle before any further fallible operation.
            let mut owner = OfferedQueue {
                harness: self,
                slots: commands
                    .into_iter()
                    .map(|command| Slot {
                        command,
                        queries: vk::QueryPool::null(),
                        fence: vk::Fence::null(),
                        request: None,
                    })
                    .collect(),
            };
            let (mode, seed, epoch) = self.access.get();
            for (index, slot) in owner.slots.iter_mut().enumerate() {
                self.configure_access(mode, seed, epoch.wrapping_add(index as u32));
                slot.queries = self
                    .device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .map_err(|p| error("vulkan_query_pool_failed", p))?;
                slot.fence = self
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|p| error("vulkan_fence_failed", p))?;
                self.record_measurement(slot.command, slot.queries, bytes, loops, 1, true)?;
            }
            self.configure_access(mode, seed, epoch);
            Ok(owner)
        }
    }
}
impl OfferedQueue<'_> {
    pub fn observe(
        &mut self,
        rate: f64,
        seconds: f64,
        cancellation: &CancellationToken,
    ) -> Result<OfferedSample, BenchmarkError> {
        if !rate.is_finite() || rate <= 0.0 || !seconds.is_finite() || seconds <= 0.0 {
            return Err(BenchmarkError::new(
                "invalid_config",
                "Offered rate and observation duration must be finite and positive.",
            ));
        }
        let mut completion_delays_ns = Vec::new();
        let schedule = Schedule::new(rate, seconds);
        let started = Instant::now();
        let (mut submitted, mut completed, mut first_pending) = (0_u64, 0_u64, 0_u64);
        let mut observed_indices = BTreeSet::new();
        let (mut gpu_ns, mut completion_ns, mut queue_wait_ns) = (0.0, 0.0, 0.0);
        loop {
            if cancellation.is_cancelled() {
                return Err(BenchmarkError::new(
                    "cancelled",
                    "The benchmark was cancelled.",
                ));
            }
            // SAFETY: independent fences/query pools; no slot is reset while its work is pending.
            unsafe {
                for slot in &mut self.slots {
                    if let Some((index, submitted_at)) = slot.request
                        && self
                            .harness
                            .device
                            .get_fence_status(slot.fence)
                            .map_err(|p| error("vulkan_fence_failed", p))?
                    {
                        let observed_at = started.elapsed().as_secs_f64();
                        if observed_at <= schedule.seconds {
                            let mut times = [0_u64; 2];
                            self.harness
                                .device
                                .get_query_pool_results(
                                    slot.queries,
                                    0,
                                    &mut times,
                                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                                )
                                .map_err(|p| error("vulkan_timestamp_failed", p))?;
                            gpu_ns += timestamp_delta(
                                times[0],
                                times[1],
                                self.harness.timestamp_valid_bits,
                            ) as f64
                                * self.harness.timestamp_period_ns;
                            completion_ns += (observed_at - schedule.arrival(index)).max(0.0) * 1e9;
                            completion_delays_ns
                                .push((observed_at - schedule.arrival(index)).max(0.0) * 1e9);
                            queue_wait_ns +=
                                (submitted_at - schedule.arrival(index)).max(0.0) * 1e9;
                            completed += 1;
                            observed_indices.insert(index);
                            while observed_indices.remove(&first_pending) {
                                first_pending += 1;
                            }
                        }
                        slot.request = None;
                    }
                }
                let now = started.elapsed().as_secs_f64();
                if now >= schedule.seconds {
                    break;
                }
                let arrived = schedule.arrived(now);
                for slot in &mut self.slots {
                    if submitted >= arrived || started.elapsed().as_secs_f64() >= schedule.seconds {
                        break;
                    }
                    if slot.request.is_some() {
                        continue;
                    }
                    self.harness
                        .device
                        .reset_fences(&[slot.fence])
                        .map_err(|p| error("vulkan_fence_failed", p))?;
                    let submitted_at = started.elapsed().as_secs_f64();
                    self.harness
                        .device
                        .queue_submit(
                            self.harness.queue,
                            &[vk::SubmitInfo::default()
                                .command_buffers(std::slice::from_ref(&slot.command))],
                            slot.fence,
                        )
                        .map_err(|p| error("vulkan_queue_submit_failed", p))?;
                    slot.request = Some((submitted, submitted_at));
                    submitted += 1;
                }
            }
            let now = started.elapsed().as_secs_f64();
            let next = schedule.arrival(submitted).min(schedule.seconds);
            let delay = if submitted < schedule.arrived(now) {
                0.00005
            } else {
                (next - now).clamp(0.00001, 0.0002)
            };
            std::thread::sleep(Duration::from_secs_f64(delay));
        }
        let window_ns = started.elapsed().as_secs_f64() * 1e9;
        let oldest_pending_ns = if completed < schedule.total {
            (schedule.seconds - schedule.arrival(first_pending)).max(0.0) * 1e9
        } else {
            0.0
        };
        let ages = (first_pending..schedule.total)
            .filter(|i| !observed_indices.contains(i))
            .map(|i| (schedule.seconds - schedule.arrival(i)).max(0.) * 1e9)
            .collect::<Vec<_>>();
        let mean_pending_ns = if ages.is_empty() {
            0.
        } else {
            ages.iter().sum::<f64>() / ages.len() as f64
        };
        let youngest_pending_ns = ages.iter().copied().reduce(f64::min).unwrap_or(0.);
        // SAFETY: drain bounded physical work before reusing query pools; drain earns no throughput credit.
        unsafe { self.harness.device.queue_wait_idle(self.harness.queue) }
            .map_err(|p| error("vulkan_device_lost", p))?;
        for slot in &mut self.slots {
            slot.request = None;
        }
        Ok(OfferedSample {
            completion_delays_ns,
            request_window_ns: schedule.seconds * 1e9,
            offered: schedule.total,
            submitted,
            completed,
            window_ns,
            gpu_ns,
            completion_ns,
            queue_wait_ns,
            oldest_pending_ns,
            mean_pending_ns,
            youngest_pending_ns,
        })
    }
}
impl Drop for OfferedQueue<'_> {
    fn drop(&mut self) {
        // SAFETY: cancellation/errors drain all outstanding work before destroying owned handles.
        unsafe {
            let device = &self.harness.device;
            let _ = device.queue_wait_idle(self.harness.queue);
            for slot in &self.slots {
                device.destroy_fence(slot.fence, None);
                device.destroy_query_pool(slot.queries, None);
            }
            device.free_command_buffers(
                self.harness.command_pool,
                &self.slots.iter().map(|s| s.command).collect::<Vec<_>>(),
            );
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arrival_demand_keeps_growing_independently_of_completion_or_queue_capacity() {
        let schedule = Schedule::new(1000.0, 0.1);
        assert_eq!(schedule.total, 100);
        assert_eq!(schedule.arrived(0.01), 11);
        assert_eq!(schedule.arrived(0.09), 91);
        assert_eq!(schedule.arrived(0.2), 100);
        assert_eq!(schedule.arrival(80), 0.08);
        assert!(schedule.arrived(0.09) > QUEUE_DEPTH as u64);
        let slow = Schedule::new(0.125, 0.05);
        assert_eq!(slow.seconds, 8.0);
        assert_eq!(slow.total, 1);
    }
}

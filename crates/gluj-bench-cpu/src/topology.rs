use gluj_bench_core::{BenchmarkError, CacheDescriptor, CacheKind};
use std::{collections::BTreeMap, mem::size_of, ptr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcessorLocation {
    pub group: u16,
    pub index: u8,
}

#[derive(Debug, Clone)]
pub struct CacheTarget {
    pub descriptor: CacheDescriptor,
    pub group: u16,
    pub mask: usize,
}

#[derive(Debug, Clone)]
pub struct CpuTopology {
    pub physical_cores: Vec<ProcessorLocation>,
    pub core_threads: BTreeMap<ProcessorLocation, Vec<ProcessorLocation>>,
    pub representative: ProcessorLocation,
    pub representative_caches: BTreeMap<u8, CacheTarget>,
    pub cache_targets: Vec<CacheTarget>,
    pub caches: Vec<CacheDescriptor>,
}

#[cfg(windows)]
mod platform {
    use super::*;
    use windows_sys::Win32::System::SystemInformation::{
        CACHE_RELATIONSHIP, CacheData, CacheUnified, GetLogicalProcessorInformationEx,
        RelationCache, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    fn records(relationship: i32) -> Result<Vec<u8>, BenchmarkError> {
        let mut length = 0u32;
        unsafe { GetLogicalProcessorInformationEx(relationship, ptr::null_mut(), &mut length) };
        if length == 0 {
            return Err(BenchmarkError::new(
                "topology_unavailable",
                "Windows returned no CPU topology data.",
            ));
        }
        let mut data = vec![0u8; length as usize];
        let ok = unsafe {
            GetLogicalProcessorInformationEx(
                relationship,
                data.as_mut_ptr()
                    .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(),
                &mut length,
            )
        };
        if ok == 0 {
            return Err(BenchmarkError::new(
                "topology_unavailable",
                "Windows CPU topology discovery failed.",
            ));
        }
        data.truncate(length as usize);
        Ok(data)
    }

    unsafe fn for_each_record(data: &[u8], mut visit: impl FnMut(*const u8, usize)) {
        let mut offset = 0usize;
        const HEADER_BYTES: usize = size_of::<i32>() + size_of::<u32>();
        while offset + HEADER_BYTES <= data.len() {
            let pointer = unsafe { data.as_ptr().add(offset) };
            let size =
                unsafe { pointer.add(size_of::<i32>()).cast::<u32>().read_unaligned() } as usize;
            if size == 0 || offset + size > data.len() {
                break;
            }
            visit(pointer, size);
            offset += size;
        }
    }

    fn first_location(group: u16, mask: usize) -> Option<ProcessorLocation> {
        (mask != 0).then(|| ProcessorLocation {
            group,
            index: mask.trailing_zeros() as u8,
        })
    }

    fn locations(group: u16, mask: usize) -> Vec<ProcessorLocation> {
        (0..usize::BITS)
            .filter(|index| mask & (1usize << index) != 0)
            .map(|index| ProcessorLocation {
                group,
                index: index as u8,
            })
            .collect()
    }

    pub fn discover() -> Result<CpuTopology, BenchmarkError> {
        let core_data = records(RelationProcessorCore)?;
        let mut physical_cores = Vec::new();
        let mut core_threads = BTreeMap::new();
        unsafe {
            for_each_record(&core_data, |pointer, size| {
                if size
                    < 8 + size_of::<
                        windows_sys::Win32::System::SystemInformation::PROCESSOR_RELATIONSHIP,
                    >()
                {
                    return;
                }
                let relation = pointer
                    .add(8)
                    .cast::<windows_sys::Win32::System::SystemInformation::PROCESSOR_RELATIONSHIP>()
                    .read_unaligned();
                let affinity_size =
                    size_of::<windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY>();
                let available_groups = 1 + (size - 8 - size_of_val(&relation)) / affinity_size;
                for index in 0..(relation.GroupCount as usize).min(available_groups) {
                    let affinity = *relation.GroupMask.as_ptr().add(index);
                    if let Some(location) = first_location(affinity.Group, affinity.Mask) {
                        physical_cores.push(location);
                        core_threads.insert(location, locations(affinity.Group, affinity.Mask));
                    }
                }
            });
        }
        physical_cores.sort_unstable();
        physical_cores.dedup();
        if physical_cores.is_empty() {
            return Err(BenchmarkError::new(
                "topology_unavailable",
                "No physical CPU cores were discovered.",
            ));
        }

        let cache_data = records(RelationCache)?;
        let mut targets = Vec::new();
        unsafe {
            for_each_record(&cache_data, |pointer, size| {
                if size < 8 + size_of::<CACHE_RELATIONSHIP>() {
                    return;
                }
                let cache: CACHE_RELATIONSHIP =
                    pointer.add(8).cast::<CACHE_RELATIONSHIP>().read_unaligned();
                if cache.Type != CacheData && cache.Type != CacheUnified {
                    return;
                }
                let kind = if cache.Type == CacheData {
                    CacheKind::Data
                } else {
                    CacheKind::Unified
                };
                let affinity_size =
                    size_of::<windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY>();
                let available_groups = 1 + (size - 8 - size_of_val(&cache)) / affinity_size;
                for index in 0..(cache.GroupCount.max(1) as usize).min(available_groups) {
                    let affinity = *cache.Anonymous.GroupMasks.as_ptr().add(index);
                    targets.push(CacheTarget {
                        descriptor: CacheDescriptor {
                            level: cache.Level,
                            kind,
                            size_bytes: cache.CacheSize as u64,
                            line_size_bytes: cache.LineSize as u32,
                            sharing_logical_processors: affinity.Mask.count_ones(),
                            instances: 1,
                        },
                        group: affinity.Group,
                        mask: affinity.Mask,
                    });
                }
            });
        }

        let representative = physical_cores
            .iter()
            .copied()
            .max_by_key(|location| {
                let mask = 1usize << location.index;
                let matching: Vec<_> = targets
                    .iter()
                    .filter(|cache| {
                        cache.group == location.group
                            && cache.mask & mask != 0
                            && cache.descriptor.level <= 3
                    })
                    .collect();
                (
                    matching
                        .iter()
                        .map(|cache| cache.descriptor.level)
                        .max()
                        .unwrap_or(0),
                    matching.len(),
                    matching
                        .iter()
                        .map(|cache| cache.descriptor.size_bytes)
                        .sum::<u64>(),
                    std::cmp::Reverse(*location),
                )
            })
            .unwrap();

        let mut representative_caches = BTreeMap::new();
        let representative_mask = 1usize << representative.index;
        for target in &targets {
            if target.group == representative.group
                && target.mask & representative_mask != 0
                && target.descriptor.level <= 3
            {
                representative_caches
                    .entry(target.descriptor.level)
                    .and_modify(|old: &mut CacheTarget| {
                        if target.descriptor.size_bytes > old.descriptor.size_bytes {
                            *old = target.clone();
                        }
                    })
                    .or_insert_with(|| target.clone());
            }
        }

        let mut aggregated: BTreeMap<(u8, CacheKind, u64, u32, u32), CacheDescriptor> =
            BTreeMap::new();
        for target in &targets {
            let descriptor = target.descriptor.clone();
            let key = (
                descriptor.level,
                descriptor.kind,
                descriptor.size_bytes,
                descriptor.line_size_bytes,
                descriptor.sharing_logical_processors,
            );
            aggregated
                .entry(key)
                .and_modify(|item| item.instances += 1)
                .or_insert(descriptor);
        }
        Ok(CpuTopology {
            physical_cores,
            core_threads,
            representative,
            representative_caches,
            cache_targets: targets,
            caches: aggregated.into_values().collect(),
        })
    }
}

#[cfg(windows)]
pub use platform::discover;

#[cfg(not(windows))]
pub fn discover() -> Result<CpuTopology, BenchmarkError> {
    Err(BenchmarkError::new(
        "backend_unavailable",
        "CPU topology discovery is currently implemented for Windows only.",
    ))
}

#[cfg(windows)]
pub struct AffinityGuard(windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY);

#[cfg(windows)]
impl AffinityGuard {
    pub fn pin(location: ProcessorLocation) -> Result<Self, BenchmarkError> {
        use windows_sys::Win32::System::{
            SystemInformation::GROUP_AFFINITY,
            Threading::{GetCurrentThread, SetThreadGroupAffinity},
        };
        let requested = GROUP_AFFINITY {
            Mask: 1usize << location.index,
            Group: location.group,
            Reserved: [0; 3],
        };
        let mut previous = GROUP_AFFINITY::default();
        let ok = unsafe { SetThreadGroupAffinity(GetCurrentThread(), &requested, &mut previous) };
        if ok == 0 {
            Err(BenchmarkError::new(
                "affinity_failed",
                "Unable to pin the benchmark thread.",
            ))
        } else {
            Ok(Self(previous))
        }
    }
}

#[cfg(windows)]
impl Drop for AffinityGuard {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadGroupAffinity};
        unsafe {
            SetThreadGroupAffinity(GetCurrentThread(), &self.0, ptr::null_mut());
        }
    }
}

#[cfg(not(windows))]
pub struct AffinityGuard;

#[cfg(not(windows))]
impl AffinityGuard {
    pub fn pin(_: ProcessorLocation) -> Result<Self, BenchmarkError> {
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processor_location_orders_by_group_then_index() {
        assert!(
            ProcessorLocation { group: 0, index: 2 } < ProcessorLocation { group: 1, index: 0 }
        );
    }
}

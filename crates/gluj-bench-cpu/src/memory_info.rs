use std::collections::BTreeMap;

pub(super) fn discover() -> BTreeMap<String, String> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::GetSystemFirmwareTable;
        const RSMB: u32 = u32::from_be_bytes(*b"RSMB");
        // Read-only firmware discovery runs in the benchmark worker, never on UI input.
        // https://learn.microsoft.com/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemfirmwaretable
        let size = unsafe { GetSystemFirmwareTable(RSMB, 0, std::ptr::null_mut(), 0) };
        if !(8..=16 * 1024 * 1024).contains(&size) {
            return BTreeMap::new();
        }
        let mut bytes = vec![0_u8; size as usize];
        // SAFETY: the supplied buffer is allocated for exactly size writable bytes.
        let read = unsafe { GetSystemFirmwareTable(RSMB, 0, bytes.as_mut_ptr().cast(), size) };
        if read < 8 || read > size {
            return BTreeMap::new();
        }
        let length = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if let Some(table) = bytes[..read as usize].get(8..8_usize.saturating_add(length)) {
            return parse(table);
        }
    }
    BTreeMap::new()
}

// SMBIOS Type 16/17: system-memory arrays and their populated modules.
// Configured speed is MT/s, not the module's advertised maximum or a measured clock.
// https://www.dmtf.org/sites/default/files/standards/documents/DSP0134_3.7.0.pdf
#[cfg(any(windows, test))]
fn parse(mut data: &[u8]) -> BTreeMap<String, String> {
    fn word(data: &[u8], offset: usize) -> Option<u16> {
        Some(u16::from_le_bytes(
            data.get(offset..offset + 2)?.try_into().ok()?,
        ))
    }
    fn dword(data: &[u8], offset: usize) -> Option<u32> {
        Some(u32::from_le_bytes(
            data.get(offset..offset + 4)?.try_into().ok()?,
        ))
    }
    let mut records = Vec::new();
    while !data.is_empty() {
        let Some(&length) = data.get(1) else {
            return BTreeMap::new();
        };
        if length < 4 || usize::from(length) > data.len() {
            return BTreeMap::new();
        }
        let record = &data[..usize::from(length)];
        if record[0] == 127 {
            break;
        }
        records.push(record);
        let tail = &data[usize::from(length)..];
        let Some(end) = tail.windows(2).position(|pair| pair == [0, 0]) else {
            return BTreeMap::new();
        };
        data = &tail[end + 2..];
    }
    let arrays: Vec<_> = records
        .iter()
        .filter(|r| r[0] == 16 && r.get(5) == Some(&3))
        .filter_map(|r| word(r, 2))
        .collect();
    let mut total = Some(0_u64);
    let mut speeds = Vec::new();
    let mut complete_speeds = true;
    let mut populated = 0;
    for record in records
        .iter()
        .filter(|r| r[0] == 17 && word(r, 4).is_some_and(|h| arrays.contains(&h)))
    {
        let Some(size) = word(record, 12) else {
            return BTreeMap::new();
        };
        if size == 0 {
            continue;
        }
        populated += 1;
        let bytes = match size {
            0xffff => None,
            0x7fff => dword(record, 28)
                .map(|v| u64::from(v & 0x7fff_ffff) * 1024 * 1024)
                .filter(|v| *v > 0),
            v if v & 0x8000 != 0 => Some(u64::from(v & 0x7fff) * 1024),
            v => Some(u64::from(v) * 1024 * 1024),
        };
        total = total.zip(bytes).and_then(|(a, b)| a.checked_add(b));
        let speed = match word(record, 32) {
            Some(0xffff) => dword(record, 88),
            Some(v) => Some(u32::from(v)),
            None => None,
        }
        .filter(|v| *v > 0 && *v < 0x8000_0000);
        if let Some(speed) = speed {
            speeds.push(speed)
        } else {
            complete_speeds = false
        }
    }
    let mut properties = BTreeMap::new();
    if populated > 0 {
        if let Some(total) = total.filter(|v| *v > 0) {
            properties.insert("installed_bytes".into(), total.to_string());
        }
        if complete_speeds {
            speeds.sort_unstable();
            speeds.dedup();
            properties.insert(
                "configured_speed_mts".into(),
                speeds
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
    }
    properties
}

#[cfg(test)]
mod tests {
    use super::*;
    fn table(size: u16, speed: u16, extended: Option<(u32, u32)>) -> Vec<u8> {
        let mut array = vec![0; 15];
        array[0] = 16;
        array[1] = 15;
        array[2] = 1;
        array[5] = 3;
        array.extend_from_slice(&[0, 0]);
        let mut module = vec![0; if extended.is_some() { 92 } else { 34 }];
        module[0] = 17;
        module[1] = module.len() as u8;
        module[4] = 1;
        module[12..14].copy_from_slice(&size.to_le_bytes());
        module[32..34].copy_from_slice(&speed.to_le_bytes());
        if let Some((size, speed)) = extended {
            module[28..32].copy_from_slice(&size.to_le_bytes());
            module[88..92].copy_from_slice(&speed.to_le_bytes());
        }
        array.extend_from_slice(&module);
        array.extend_from_slice(&[0, 0]);
        array
    }
    #[test]
    fn populated_system_memory_uses_configured_speed_and_extended_fields() {
        let mut data = table(16384, 3200, None);
        let second = table(16384, 3200, None);
        data.extend_from_slice(&second[17..]);
        let properties = parse(&data);
        assert_eq!(
            properties["installed_bytes"],
            (32_u64 * 1024 * 1024 * 1024).to_string()
        );
        assert_eq!(properties["configured_speed_mts"], "3200");
        let properties = parse(&table(0x7fff, 0xffff, Some((131072, 100000))));
        assert_eq!(
            properties["installed_bytes"],
            (128_u64 * 1024 * 1024 * 1024).to_string()
        );
        assert_eq!(properties["configured_speed_mts"], "100000");
    }
    #[test]
    fn unknown_empty_non_system_and_malformed_firmware_do_not_invent_values() {
        assert!(parse(&table(0, 3200, None)).is_empty());
        let properties = parse(&table(0xffff, 0, None));
        assert!(properties.is_empty());
        let mut data = table(16384, 3200, None);
        data[5] = 4;
        assert!(parse(&data).is_empty());
        let data = table(16384, 3200, None);
        for length in 1..data.len() {
            let _ = parse(&data[..length]);
        }
        assert!(parse(&[17, 255, 0, 0]).is_empty());
        assert!(parse(&[17, 0, 0, 0]).is_empty());
    }
}

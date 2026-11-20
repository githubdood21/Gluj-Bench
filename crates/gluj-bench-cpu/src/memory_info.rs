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

// SMBIOS Types 16/17/37: system-memory arrays, populated modules and channel topology.
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
        let tail = &data[usize::from(length)..];
        let Some(end) = tail.windows(2).position(|pair| pair == [0, 0]) else {
            return BTreeMap::new();
        };
        records.push((record, &tail[..end]));
        data = &tail[end + 2..];
    }
    let arrays: Vec<_> = records
        .iter()
        .map(|(record, _)| *record)
        .filter(|r| r[0] == 16 && r.get(5) == Some(&3))
        .filter_map(|r| word(r, 2))
        .collect();
    let mut total = Some(0_u64);
    let mut speeds = Vec::new();
    let mut complete_speeds = true;
    let mut modules = Vec::new();
    for (record, strings) in records
        .iter()
        .filter(|(r, _)| r[0] == 17 && word(r, 4).is_some_and(|h| arrays.contains(&h)))
    {
        let Some(size) = word(record, 12) else {
            return BTreeMap::new();
        };
        if size == 0 {
            continue;
        }
        let width = word(record, 10).filter(|width| !matches!(*width, 0 | u16::MAX));
        modules.push((
            word(record, 2).unwrap(),
            width,
            slot_channel(record, strings),
        ));
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
    if !modules.is_empty() {
        if let Some(width) = modules[0].1
            && modules.iter().all(|module| module.1 == Some(width))
        {
            let mut topology = BTreeMap::new();
            let mut complete = true;
            for (channel, _) in records.iter().filter(|(record, _)| record[0] == 37) {
                let Some(&count) = channel.get(6) else {
                    complete = false;
                    break;
                };
                let Some(devices) = channel.get(7..7 + usize::from(count) * 3) else {
                    complete = false;
                    break;
                };
                for device in devices.as_chunks::<3>().0 {
                    let handle = word(device, 1).unwrap();
                    if matches!(handle, 0 | u16::MAX)
                        && !modules.iter().any(|module| module.0 == handle)
                    {
                        continue;
                    }
                    if topology.insert(handle, word(channel, 2).unwrap()).is_some() {
                        complete = false;
                    }
                }
            }
            let channels = if complete
                && modules
                    .iter()
                    .all(|module| topology.contains_key(&module.0))
            {
                modules
                    .iter()
                    .map(|module| topology[&module.0])
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            } else if modules.len() == 1 {
                // A single populated module can only occupy one memory channel regardless
                // of DDR generation, locator label format, or omitted/incomplete Type 37.
                1
            } else if topology.is_empty()
                && complete
                && arrays.len() == 1
                && modules.iter().all(|module| module.2.is_some())
            {
                modules
                    .iter()
                    .filter_map(|module| module.2)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            } else {
                0
            };
            if channels > 0 {
                properties.insert("memory_channel_count".into(), channels.to_string());
                properties.insert("memory_channel_data_width_bits".into(), width.to_string());
            }
        }
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

// Some DDR3/DDR4/DDR5/LPDDRx firmware omits Type 37 but encodes the channel in Device Locator
// or Bank Locator labels such as DIMM_A2, Controller0-ChannelA-DIMM0, or P0 CHANNEL A.
// Numeric-only slot or bank labels (e.g. DIMM 2, BANK 0) carry no reliable channel information.
#[cfg(any(windows, test))]
fn slot_channel(record: &[u8], strings: &[u8]) -> Option<(u8, u8)> {
    // SMBIOS Type 17 memory types for DDR3/DDR4/DDR5 and LPDDR-LPDDR5 (DSP0134).
    let mem_type = *record.get(18)?;
    if !matches!(mem_type, 0x18 | 0x1a..=0x1e | 0x22 | 0x23) {
        return None;
    }
    [16, 17]
        .into_iter()
        .filter_map(|offset| smbios_string(strings, record.get(offset)))
        .find_map(parse_channel_label)
}

#[cfg(any(windows, test))]
fn smbios_string<'a>(strings: &'a [u8], index_byte: Option<&u8>) -> Option<&'a [u8]> {
    let index = index_byte?.checked_sub(1)?;
    strings.split(|byte| *byte == 0).nth(usize::from(index))
}

#[cfg(any(windows, test))]
fn parse_channel_label(label: &[u8]) -> Option<(u8, u8)> {
    fn find_ci(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window.eq_ignore_ascii_case(needle))
    }

    fn skip_sep(bytes: &[u8]) -> &[u8] {
        let mut rest = bytes;
        while let Some((&b, tail)) = rest.split_first() {
            if matches!(b, b'_' | b'-' | b' ') {
                rest = tail;
            } else {
                break;
            }
        }
        rest
    }

    let controller = find_ci(label, b"controller")
        .and_then(|pos| {
            let after = skip_sep(&label[pos + b"controller".len()..]);
            let len = after.iter().take_while(|b| b.is_ascii_digit()).count();
            (len > 0).then(|| std::str::from_utf8(&after[..len]).ok()?.parse::<u8>().ok())?
        })
        .unwrap_or(0);

    if let Some(pos) = find_ci(label, b"channel") {
        let after = skip_sep(&label[pos + b"channel".len()..]);
        if let Some((&ch, rest)) = after.split_first() {
            let upper = ch.to_ascii_uppercase();
            if (b'A'..=b'H').contains(&upper) && !rest.first().is_some_and(u8::is_ascii_alphabetic)
            {
                return Some((controller, upper));
            }
            if (b'0'..=b'7').contains(&ch) && !rest.first().is_some_and(u8::is_ascii_digit) {
                return Some((controller, ch));
            }
        }
    }

    let mut offset = 0;
    while let Some(rel) = find_ci(&label[offset..], b"dimm") {
        let after = skip_sep(&label[offset + rel + b"dimm".len()..]);
        if let Some((&ch, number)) = after.split_first() {
            let upper = ch.to_ascii_uppercase();
            let digit_len = number.iter().take_while(|b| b.is_ascii_digit()).count();
            if (b'A'..=b'H').contains(&upper)
                && digit_len > 0
                && !number.get(digit_len).is_some_and(u8::is_ascii_alphanumeric)
            {
                return Some((controller, upper));
            }
        }
        offset += rel + b"dimm".len();
    }

    None
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
    fn data_bus_counts_populated_channels_instead_of_modules() {
        let sample = table(8192, 3200, None);
        let memory = |slots: &[(&str, u16)]| {
            let mut data = sample[..17].to_vec();
            for (index, (locator, width)) in slots.iter().enumerate() {
                let mut module = sample[17..51].to_vec();
                module[2] = index as u8 + 10;
                module[10..12].copy_from_slice(&width.to_le_bytes());
                module[16] = 1;
                module[18] = 0x1a;
                data.extend_from_slice(&module);
                data.extend_from_slice(locator.as_bytes());
                data.extend_from_slice(&[0, 0]);
            }
            data
        };
        let two = parse(&memory(&[
            ("DIMM_A1", 64),
            ("DIMM_A2", 64),
            ("DIMM_B1", 64),
        ]));
        assert_eq!(two["memory_channel_count"], "2");
        assert_eq!(two["memory_channel_data_width_bits"], "64");
        let one = parse(&memory(&[("DIMM_A1", 64), ("DIMM_A2", 64)]));
        assert_eq!(one["memory_channel_count"], "1");
        for (slot, width) in [("DIMM 2", 64), ("DIMM_B1", 32), ("DIMM_B1", 0)] {
            assert!(
                !parse(&memory(&[("DIMM_A1", 64), (slot, width)]))
                    .contains_key("memory_channel_count")
            );
        }
        let mut explicit = memory(&[("DIMM 0", 64), ("DIMM 1", 64)]);
        explicit.extend_from_slice(&[37, 10, 20, 0, 3, 1, 1, 1, 10, 0, 0, 0]);
        explicit.extend_from_slice(&[37, 10, 21, 0, 3, 1, 1, 1, 11, 0, 0, 0]);
        assert_eq!(parse(&explicit)["memory_channel_count"], "2");

        let multi_format = |memory_type: u8, entries: &[(&str, &str, u16)]| {
            let mut data = sample[..17].to_vec();
            for (index, (locator, bank, width)) in entries.iter().enumerate() {
                let mut module = sample[17..51].to_vec();
                module[2] = 10 + index as u8;
                module[10..12].copy_from_slice(&width.to_le_bytes());
                module[16] = 1;
                module[17] = if bank.is_empty() { 0 } else { 2 };
                module[18] = memory_type;
                data.extend_from_slice(&module);
                data.extend_from_slice(locator.as_bytes());
                data.push(0);
                if !bank.is_empty() {
                    data.extend_from_slice(bank.as_bytes());
                    data.push(0);
                }
                data.push(0);
            }
            data
        };

        let intel_ddr5 = parse(&multi_format(
            0x22,
            &[
                ("Controller0-ChannelA-DIMM0", "", 64),
                ("Controller0-ChannelA-DIMM1", "", 64),
                ("Controller1-ChannelB-DIMM0", "", 64),
            ],
        ));
        assert_eq!(intel_ddr5["memory_channel_count"], "2");

        let intel_dual_controller = parse(&multi_format(
            0x22,
            &[
                ("Controller0-ChannelA-DIMM0", "", 64),
                ("Controller1-ChannelA-DIMM0", "", 64),
            ],
        ));
        assert_eq!(intel_dual_controller["memory_channel_count"], "2");

        let ddr5_single_ch = parse(&multi_format(
            0x22,
            &[
                ("Controller0-ChannelA-DIMM0", "", 64),
                ("Controller0-ChannelA-DIMM1", "", 64),
            ],
        ));
        assert_eq!(ddr5_single_ch["memory_channel_count"], "1");

        let lpddr5_quad = parse(&multi_format(
            0x23,
            &[
                ("ChannelA-DIMM0", "", 32),
                ("ChannelB-DIMM0", "", 32),
                ("ChannelC-DIMM0", "", 32),
                ("ChannelD-DIMM0", "", 32),
            ],
        ));
        assert_eq!(lpddr5_quad["memory_channel_count"], "4");
        assert_eq!(lpddr5_quad["memory_channel_data_width_bits"], "32");

        let bank_locator = parse(&multi_format(
            0x22,
            &[
                ("DIMM 0", "P0 CHANNEL A", 64),
                ("DIMM 1", "P0 CHANNEL B", 64),
            ],
        ));
        assert_eq!(bank_locator["memory_channel_count"], "2");

        assert!(
            !parse(&multi_format(
                0x22,
                &[("DIMM 0", "BANK 0", 64), ("DIMM 1", "BANK 1", 64)],
            ))
            .contains_key("memory_channel_count")
        );
    }

    #[test]
    fn single_populated_module_detects_one_channel_without_type_37_or_ddr4_label() {
        let sample = table(16384, 5600, None);
        let single = |memory_type: u8, locator: &str, width: u16| {
            let mut data = sample[..17].to_vec();
            let mut module = sample[17..51].to_vec();
            module[2] = 10;
            module[10..12].copy_from_slice(&width.to_le_bytes());
            module[16] = 1;
            module[18] = memory_type;
            data.extend_from_slice(&module);
            data.extend_from_slice(locator.as_bytes());
            data.extend_from_slice(&[0, 0]);
            data
        };

        let ddr4_generic = parse(&single(0x1a, "DIMM 0", 64));
        assert_eq!(ddr4_generic["memory_channel_count"], "1");
        assert_eq!(ddr4_generic["memory_channel_data_width_bits"], "64");

        let ddr5_single = parse(&single(0x22, "DIMM_A1", 64));
        assert_eq!(ddr5_single["memory_channel_count"], "1");
        assert_eq!(ddr5_single["memory_channel_data_width_bits"], "64");

        let lpddr5_single = parse(&single(0x23, "ChannelA-DIMM0", 32));
        assert_eq!(lpddr5_single["memory_channel_count"], "1");
        assert_eq!(lpddr5_single["memory_channel_data_width_bits"], "32");

        // Repeated empty sentinel handles in Type 37 must not suppress single-channel detection.
        let mut with_sentinel_type37 = single(0x22, "DIMM_A1", 64);
        with_sentinel_type37.extend_from_slice(&[37, 13, 20, 0, 3, 2, 2, 1, 10, 0, 0, 0, 0, 0, 0]);
        with_sentinel_type37
            .extend_from_slice(&[37, 13, 21, 0, 3, 2, 2, 0, 0, 0, 0, 0xff, 0xff, 0, 0]);
        let padded = parse(&with_sentinel_type37);
        assert_eq!(padded["memory_channel_count"], "1");
        assert_eq!(padded["memory_channel_data_width_bits"], "64");

        // Unknown width still suppresses bus metadata even for a single module.
        assert!(!parse(&single(0x22, "DIMM 0", 0)).contains_key("memory_channel_count"));
        assert!(!parse(&single(0x22, "DIMM 0", u16::MAX)).contains_key("memory_channel_count"));
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

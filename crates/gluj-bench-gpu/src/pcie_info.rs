#[cfg(windows)]
use std::ffi::OsString;
#[cfg(windows)]
use std::os::windows::ffi::OsStringExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PciBusInfo {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

impl PciBusInfo {
    pub fn format_bdf(self) -> String {
        format!(
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PciDeviceLinkSample {
    pub current_speed: Option<u32>,
    pub current_width: Option<u32>,
    pub max_speed: Option<u32>,
    pub max_width: Option<u32>,
    pub pcie_device_type: Option<u32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PcieLinkInfo {
    pub generation: Option<u32>,
    pub width: Option<u32>,
    pub current_generation: Option<u32>,
    pub current_width: Option<u32>,
    pub max_generation: Option<u32>,
    pub max_width: Option<u32>,
    pub is_integrated_endpoint: bool,
}

#[allow(dead_code)]
impl PcieLinkInfo {
    pub fn has_link(self) -> bool {
        self.generation.is_some() || self.width.is_some()
    }
}

pub(super) fn effective_link_from_path(hops: &[PciDeviceLinkSample]) -> PcieLinkInfo {
    const ROOT_COMPLEX_INTEGRATED_ENDPOINT: u32 = 9;
    let is_integrated_endpoint = hops
        .first()
        .is_some_and(|hop| hop.pcie_device_type == Some(ROOT_COMPLEX_INTEGRATED_ENDPOINT));
    if is_integrated_endpoint {
        return PcieLinkInfo {
            is_integrated_endpoint: true,
            ..Default::default()
        };
    }

    let min_nonzero = |extract: fn(&PciDeviceLinkSample) -> Option<u32>| {
        hops.iter()
            .filter_map(extract)
            .filter(|&value| value > 0)
            .min()
    };

    let current_generation = min_nonzero(|hop| hop.current_speed);
    let current_width = min_nonzero(|hop| hop.current_width);
    let max_generation = min_nonzero(|hop| hop.max_speed);
    let max_width = min_nonzero(|hop| hop.max_width);

    // Windows power management / ASPM frequently lowers CurrentLinkSpeed to Gen 1 (2.5 GT/s)
    // while the GPU is idle at startup; prefer negotiated path capability ceiling when idle at Gen 1.
    let generation = match (current_generation, max_generation) {
        (Some(1), Some(max)) if max > 1 => Some(max),
        (Some(cur), Some(max)) => Some(cur.min(max)),
        (None, Some(max)) => Some(max),
        (Some(cur), None) => Some(cur),
        (None, None) => None,
    };
    let width = current_width.or(max_width);

    PcieLinkInfo {
        generation,
        width,
        current_generation,
        current_width,
        max_generation,
        max_width,
        is_integrated_endpoint: false,
    }
}

pub(super) fn parse_ven_dev_from_instance_id(instance_id: &str) -> Option<(u32, u32)> {
    let upper = instance_id.to_ascii_uppercase();
    let ven_pos = upper.find("VEN_")?;
    let ven_hex = upper.get(ven_pos + 4..ven_pos + 8)?;
    let dev_pos = upper.find("DEV_")?;
    let dev_hex = upper.get(dev_pos + 4..dev_pos + 8)?;
    let vendor_id = u32::from_str_radix(ven_hex, 16).ok()?;
    let device_id = u32::from_str_radix(dev_hex, 16).ok()?;
    Some((vendor_id, device_id))
}

#[cfg(windows)]
#[derive(Debug, Clone)]
struct WindowsPciDevice {
    vendor_id: u32,
    device_id: u32,
    bus: Option<u32>,
    device: Option<u32>,
    function: Option<u32>,
    link: PcieLinkInfo,
}

#[cfg(windows)]
pub(super) struct WindowsPciIndex {
    devices: Vec<WindowsPciDevice>,
}

#[cfg(windows)]
impl WindowsPciIndex {
    pub fn discover() -> Self {
        Self {
            devices: enumerate_windows_pci_display_devices(),
        }
    }

    pub fn lookup(
        &self,
        vendor_id: u32,
        device_id: u32,
        pci_bus_info: Option<PciBusInfo>,
    ) -> Option<PcieLinkInfo> {
        if let Some(bdf) = pci_bus_info
            && let Some(found) = self.devices.iter().find(|entry| {
                entry.vendor_id == vendor_id
                    && entry.device_id == device_id
                    && entry.bus == Some(bdf.bus)
                    && entry.device == Some(bdf.device)
                    && entry.function == Some(bdf.function)
            })
        {
            return Some(found.link);
        }

        // Fall back to unique VID:DID match when VK_EXT_pci_bus_info is unavailable.
        let mut matches = self
            .devices
            .iter()
            .filter(|entry| entry.vendor_id == vendor_id && entry.device_id == device_id);
        let first = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        Some(first.link)
    }
}

#[cfg(not(windows))]
pub(super) struct WindowsPciIndex;

#[cfg(not(windows))]
impl WindowsPciIndex {
    pub fn discover() -> Self {
        Self
    }

    pub fn lookup(
        &self,
        _vendor_id: u32,
        _device_id: u32,
        _pci_bus_info: Option<PciBusInfo>,
    ) -> Option<PcieLinkInfo> {
        None
    }
}

#[cfg(windows)]
fn enumerate_windows_pci_display_devices() -> Vec<WindowsPciDevice> {
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        CM_GETIDLIST_FILTER_ENUMERATOR, CM_GETIDLIST_FILTER_PRESENT, CM_Get_DevNode_PropertyW,
        CM_Get_Device_ID_List_SizeW, CM_Get_Device_ID_ListW, CM_Get_Device_IDW, CM_Get_Parent,
        CM_LOCATE_DEVNODE_NORMAL, CM_Locate_DevNodeW, CR_SUCCESS,
    };
    use windows_sys::Win32::Devices::Properties::{
        DEVPKEY_Device_Address, DEVPKEY_Device_BusNumber, DEVPKEY_Device_Class,
        DEVPROP_TYPE_STRING, DEVPROP_TYPE_UINT32,
    };
    use windows_sys::Win32::Foundation::DEVPROPKEY;
    use windows_sys::core::GUID;

    // Microsoft WDK pciprop.h keys (`GUID_DEVPROPKEY_PCI_DEVICE`).
    const PCI_DEVICE_FMTID: GUID = GUID::from_u128(0x3ab22e31_8264_4b4e_9af5_a8d2d8e33e62);
    const DEVPKEY_PCI_DEVICE_TYPE: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_DEVICE_FMTID,
        pid: 1,
    };
    const DEVPKEY_PCI_CURRENT_LINK_SPEED: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_DEVICE_FMTID,
        pid: 9,
    };
    const DEVPKEY_PCI_CURRENT_LINK_WIDTH: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_DEVICE_FMTID,
        pid: 10,
    };
    const DEVPKEY_PCI_MAX_LINK_SPEED: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_DEVICE_FMTID,
        pid: 11,
    };
    const DEVPKEY_PCI_MAX_LINK_WIDTH: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_DEVICE_FMTID,
        pid: 12,
    };

    fn read_u32_prop(devinst: u32, key: &DEVPROPKEY) -> Option<u32> {
        let mut prop_type = 0_u32;
        let mut value = 0_u32;
        let mut size = std::mem::size_of::<u32>() as u32;
        let cr = unsafe {
            CM_Get_DevNode_PropertyW(
                devinst,
                key,
                &mut prop_type,
                (&mut value as *mut u32).cast::<u8>(),
                &mut size,
                0,
            )
        };
        (cr == CR_SUCCESS && prop_type == DEVPROP_TYPE_UINT32 && size == 4).then_some(value)
    }

    fn read_string_prop(devinst: u32, key: &DEVPROPKEY) -> Option<String> {
        let mut prop_type = 0_u32;
        let mut size = 0_u32;
        unsafe {
            CM_Get_DevNode_PropertyW(
                devinst,
                key,
                &mut prop_type,
                std::ptr::null_mut(),
                &mut size,
                0,
            )
        };
        if !(2..=4096).contains(&size) {
            return None;
        }
        let mut buf = vec![0_u16; (size as usize).div_ceil(2)];
        let mut actual_size = (buf.len() * 2) as u32;
        let cr = unsafe {
            CM_Get_DevNode_PropertyW(
                devinst,
                key,
                &mut prop_type,
                buf.as_mut_ptr().cast::<u8>(),
                &mut actual_size,
                0,
            )
        };
        if cr != CR_SUCCESS || prop_type != DEVPROP_TYPE_STRING {
            return None;
        }
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(
            OsString::from_wide(&buf[..len])
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn devnode_id(devinst: u32) -> Option<String> {
        let mut buf = [0_u16; 512];
        let cr = unsafe { CM_Get_Device_IDW(devinst, buf.as_mut_ptr(), buf.len() as u32, 0) };
        if cr != CR_SUCCESS {
            return None;
        }
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(
            OsString::from_wide(&buf[..len])
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn sample_devnode(devinst: u32) -> PciDeviceLinkSample {
        PciDeviceLinkSample {
            current_speed: read_u32_prop(devinst, &DEVPKEY_PCI_CURRENT_LINK_SPEED),
            current_width: read_u32_prop(devinst, &DEVPKEY_PCI_CURRENT_LINK_WIDTH),
            max_speed: read_u32_prop(devinst, &DEVPKEY_PCI_MAX_LINK_SPEED),
            max_width: read_u32_prop(devinst, &DEVPKEY_PCI_MAX_LINK_WIDTH),
            pcie_device_type: read_u32_prop(devinst, &DEVPKEY_PCI_DEVICE_TYPE),
        }
    }

    let filter: Vec<u16> = "PCI\0".encode_utf16().collect();
    let flags = CM_GETIDLIST_FILTER_ENUMERATOR | CM_GETIDLIST_FILTER_PRESENT;
    let mut list_len = 0_u32;
    let cr = unsafe { CM_Get_Device_ID_List_SizeW(&mut list_len, filter.as_ptr(), flags) };
    if cr != CR_SUCCESS || list_len == 0 {
        return Vec::new();
    }
    let mut raw_ids = vec![0_u16; list_len as usize];
    let cr =
        unsafe { CM_Get_Device_ID_ListW(filter.as_ptr(), raw_ids.as_mut_ptr(), list_len, flags) };
    if cr != CR_SUCCESS {
        return Vec::new();
    }

    let mut result = Vec::new();
    for slice in raw_ids.split(|&ch| ch == 0) {
        if slice.is_empty() {
            continue;
        }
        let id = OsString::from_wide(slice).to_string_lossy().into_owned();
        let Some((vendor_id, device_id)) = parse_ven_dev_from_instance_id(&id) else {
            continue;
        };
        let mut wide_id: Vec<u16> = slice.to_vec();
        wide_id.push(0);
        let mut devinst = 0_u32;
        if unsafe { CM_Locate_DevNodeW(&mut devinst, wide_id.as_ptr(), CM_LOCATE_DEVNODE_NORMAL) }
            != CR_SUCCESS
        {
            continue;
        }
        // Only inspect display-class devices to avoid scanning non-GPU PCI endpoints.
        if !read_string_prop(devinst, &DEVPKEY_Device_Class)
            .is_some_and(|class| class.eq_ignore_ascii_case("Display"))
        {
            continue;
        }

        let bus = read_u32_prop(devinst, &DEVPKEY_Device_BusNumber);
        let addr = read_u32_prop(devinst, &DEVPKEY_Device_Address);
        let device = addr.map(|a| (a >> 16) & 0xffff);
        let function = addr.map(|a| a & 0xffff);

        let mut hops = vec![sample_devnode(devinst)];
        let mut cursor = devinst;
        for _ in 0..8 {
            let mut parent = 0_u32;
            if unsafe { CM_Get_Parent(&mut parent, cursor, 0) } != CR_SUCCESS {
                break;
            }
            let Some(parent_id) = devnode_id(parent) else {
                break;
            };
            if !parent_id
                .get(..4)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("PCI\\"))
            {
                break;
            }
            hops.push(sample_devnode(parent));
            cursor = parent;
        }

        result.push(WindowsPciDevice {
            vendor_id,
            device_id,
            bus,
            device,
            function,
            link: effective_link_from_path(&hops),
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vendor_and_device_ids_from_pnp_instance_id() {
        assert_eq!(
            parse_ven_dev_from_instance_id(
                r"PCI\VEN_10DE&DEV_2684&SUBSYS_40901462&REV_A1\4&1A2B3C4D&0&0008"
            ),
            Some((0x10de, 0x2684))
        );
        assert_eq!(
            parse_ven_dev_from_instance_id(r"PCI\ven_1002&dev_744c\6&1234&0&0000"),
            Some((0x1002, 0x744c))
        );
        assert_eq!(parse_ven_dev_from_instance_id(r"ROOT\DISPLAY\0000"), None);
    }

    #[test]
    fn effective_link_accounts_for_idle_aspm_and_upstream_slot_bottlenecks() {
        // Gen 5 x16 GPU plugged into Gen 5 x16 slot, downclocked to Gen 1 at idle.
        let gen5_idle = effective_link_from_path(&[
            PciDeviceLinkSample {
                current_speed: Some(1),
                current_width: Some(16),
                max_speed: Some(5),
                max_width: Some(16),
                pcie_device_type: Some(0),
            },
            PciDeviceLinkSample {
                current_speed: Some(1),
                current_width: Some(16),
                max_speed: Some(5),
                max_width: Some(16),
                pcie_device_type: Some(4),
            },
        ]);
        assert_eq!(gen5_idle.generation, Some(5));
        assert_eq!(gen5_idle.width, Some(16));

        // Gen 4 x16 AMD/Intel GPU behind internal Gen 4 x16 switch in a PCIe Gen 3 x4 chipset slot.
        let switch_in_gen3_x4 = effective_link_from_path(&[
            PciDeviceLinkSample {
                current_speed: Some(4),
                current_width: Some(16),
                max_speed: Some(4),
                max_width: Some(16),
                pcie_device_type: Some(0),
            },
            PciDeviceLinkSample {
                current_speed: Some(4),
                current_width: Some(16),
                max_speed: Some(4),
                max_width: Some(16),
                pcie_device_type: Some(6),
            },
            PciDeviceLinkSample {
                current_speed: Some(3),
                current_width: Some(4),
                max_speed: Some(3),
                max_width: Some(4),
                pcie_device_type: Some(5),
            },
            PciDeviceLinkSample {
                current_speed: Some(3),
                current_width: Some(4),
                max_speed: Some(3),
                max_width: Some(4),
                pcie_device_type: Some(4),
            },
        ]);
        assert_eq!(switch_in_gen3_x4.generation, Some(3));
        assert_eq!(switch_in_gen3_x4.width, Some(4));

        // Root Complex Integrated Endpoint (Intel iGPU) reports integrated without an external PCIe link.
        let integrated = effective_link_from_path(&[PciDeviceLinkSample {
            current_speed: Some(0),
            current_width: Some(0),
            max_speed: Some(0),
            max_width: Some(0),
            pcie_device_type: Some(9),
        }]);
        assert!(integrated.is_integrated_endpoint);
        assert_eq!(integrated.generation, None);
        assert_eq!(integrated.width, None);
    }

    #[test]
    #[ignore = "inspects live PCIe link discovery on the current Windows machine"]
    fn real_gpu_pcie_discovery_smoke() {
        use gluj_bench_core::BenchmarkProvider;
        let provider = crate::GpuBandwidthProvider::discover();
        for device in provider.devices() {
            println!(
                "GPU: {} ({}) avail={}",
                device.name, device.id, device.available
            );
            for (k, v) in &device.properties {
                if k.starts_with("pcie_") || k == "device_type" {
                    println!("  {k} = {v}");
                }
            }
        }
        assert!(provider.devices().iter().any(|d| d.available));
    }
}

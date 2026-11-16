//! `--adapter "<name substring>"`: which adapter a run is on (plan 2.4, C1).
//!
//! Both devices of a run, GUST's `HeadlessDevice` and the hand-WGSL `WgpuContext`, are opened on the one adapter whose
//! `AdapterInfo::name` contains the substring (letter case ignored). A substring that matches no adapter or more than one
//! is an error that lists the adapters found; no flag keeps the choice the harness has always made (wgpu's
//! high-performance Vulkan adapter). GUST's device is opened through the public `DeviceOptions` /
//! `AdapterChoice::Index` of `gust-wgpu`, with the index of this module's own match in `HeadlessDevice::enumerate`
//! (the order `AdapterChoice::Index` counts), so the library's first-match `NameContains` is never the judge of
//! ambiguity.

use gust_wgpu::{AdapterChoice, DeviceOptions, HeadlessDevice};
use serde_json::{Value, json};

use crate::backend::BenchResult;

/// The index of the one name that contains `needle` (letter case ignored). An empty needle, no match and more than one
/// match are errors that list every adapter found, one per line with its index.
pub fn match_adapter<S: AsRef<str>>(names: &[S], needle: &str) -> Result<usize, String> {
    let listing = || {
        if names.is_empty() {
            "no adapter was found".to_owned()
        } else {
            let lines = names
                .iter()
                .enumerate()
                .map(|(i, name)| format!("  [{i}] {}", name.as_ref()))
                .collect::<Vec<_>>()
                .join("\n");
            format!("adapters found:\n{lines}")
        }
    };
    if needle.trim().is_empty() {
        return Err(format!(
            "--adapter needs a non-empty name substring; {}",
            listing()
        ));
    }
    let wanted = needle.to_lowercase();
    let hits = names
        .iter()
        .enumerate()
        .filter(|(_, name)| name.as_ref().to_lowercase().contains(&wanted))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    match hits.as_slice() {
        [one] => Ok(*one),
        [] => Err(format!(
            "--adapter `{needle}` matches no adapter; {}",
            listing()
        )),
        many => Err(format!(
            "--adapter `{needle}` matches {} adapters, not one ({}); {}",
            many.len(),
            many.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            listing()
        )),
    }
}

/// The adapter a run was told to use: the substring, its index in `HeadlessDevice::enumerate(VULKAN)` and the adapter's
/// description.
pub struct Selection {
    pub substring: String,
    pub index: usize,
    pub info: wgpu::AdapterInfo,
}

impl Selection {
    /// The run file's record of the selection (the machine block's `adapter_selection`).
    pub fn record(&self) -> Value {
        json!({
            "flag": "--adapter",
            "substring": self.substring,
            "enumeration_index": self.index,
            "name": self.info.name,
            "vendor_id": self.info.vendor,
            "device_id": self.info.device,
            "device_type": format!("{:?}", self.info.device_type),
            "backend": format!("{:?}", self.info.backend),
            "driver": self.info.driver,
            "pci_bus_id": self.info.device_pci_bus_id,
        })
    }

    /// GUST's device options on this adapter. `adapter_limits` asks for the adapter's own limits and optional features
    /// (what `DeviceOptions::default()` is); otherwise wgpu's default limits and no optional feature, what
    /// `HeadlessDevice::new` requests.
    pub fn options(&self, adapter_limits: bool) -> DeviceOptions {
        device_options(self.index, adapter_limits)
    }
}

/// [`Selection::options`] for an enumeration index.
pub fn device_options(index: usize, adapter_limits: bool) -> DeviceOptions {
    let options = DeviceOptions::default()
        .with_backends(wgpu::Backends::VULKAN)
        .with_adapter(AdapterChoice::Index(index));
    if adapter_limits {
        options
    } else {
        options
            .with_limits(gust_wgpu::LimitsPolicy::Default)
            .with_features(gust_wgpu::FeaturesPolicy::Default)
    }
}

/// One adapter description as the run file records it.
pub fn info_record(info: &wgpu::AdapterInfo) -> Value {
    json!({
        "name": info.name,
        "vendor_id": info.vendor,
        "device_id": info.device,
        "device_type": format!("{:?}", info.device_type),
        "backend": format!("{:?}", info.backend),
        "driver": info.driver,
        "pci_bus_id": info.device_pci_bus_id,
    })
}

/// Whether every labelled device is on one adapter: the same name, vendor, device, backend and PCI bus id. A mismatch is an error that
/// prints all of them. Two adapters with identical names (two identical GPUs) are told apart here by the PCI bus id, where the
/// driver reports one, but `--adapter` matches on the name alone and cannot select one of them: it reports the ambiguity.
pub fn require_same(devices: &[(&str, &wgpu::AdapterInfo)]) -> Result<(), String> {
    let key = |i: &wgpu::AdapterInfo| {
        (
            i.name.clone(),
            i.vendor,
            i.device,
            format!("{:?}", i.backend),
            i.device_pci_bus_id.clone(),
        )
    };
    if devices
        .windows(2)
        .all(|pair| key(pair[0].1) == key(pair[1].1))
    {
        return Ok(());
    }
    let lines = devices
        .iter()
        .map(|(label, info)| {
            format!(
                "  {label}: `{}` (vendor {:#x}, device {:#x}, {:?}, bus `{}`)",
                info.name, info.vendor, info.device, info.backend, info.device_pci_bus_id
            )
        })
        .collect::<Vec<_>>()
        .join(
            "
",
        );
    Err(format!(
        "the devices of the run are not on one adapter; refusing to compare:
{lines}"
    ))
}

/// Chooses the adapter whose name contains `substring` among those of `HeadlessDevice::enumerate` (Vulkan).
pub fn select(substring: &str) -> BenchResult<Selection> {
    let infos = HeadlessDevice::enumerate(wgpu::Backends::VULKAN);
    let names = infos
        .iter()
        .map(|info| info.name.as_str())
        .collect::<Vec<_>>();
    let index = match_adapter(&names, substring)?;
    Ok(Selection {
        substring: substring.to_owned(),
        index,
        info: infos[index].clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: [&str; 3] = [
        "NVIDIA GeForce RTX 5090",
        "AMD Radeon(TM) 890M Graphics",
        "llvmpipe (LLVM 19.1.1, 256 bits)",
    ];

    fn info(name: &str, vendor: u32) -> wgpu::AdapterInfo {
        let mut info = wgpu::AdapterInfo::new(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan);
        info.name = name.to_owned();
        info.vendor = vendor;
        info
    }

    #[test]
    fn devices_on_different_adapters_are_refused_with_all_of_them_printed() {
        let (a, b) = (info("A", 1), info("B", 2));
        assert!(require_same(&[("gust", &a), ("hand", &a.clone()), ("selection", &a)]).is_ok());
        let error = require_same(&[("gust", &a), ("hand", &b), ("selection", &a)]).unwrap_err();
        for text in ["gust: `A`", "hand: `B`", "selection: `A`"] {
            assert!(error.contains(text), "{error}");
        }
        // The same name on another vendor is another adapter.
        assert!(require_same(&[("gust", &a), ("hand", &info("A", 3))]).is_err());
    }

    #[test]
    fn a_substring_picks_the_one_adapter_whatever_the_letter_case() {
        assert_eq!(match_adapter(&NAMES, "5090"), Ok(0));
        assert_eq!(match_adapter(&NAMES, "890m"), Ok(1));
        assert_eq!(match_adapter(&NAMES, "RADEON"), Ok(1));
        assert_eq!(match_adapter(&NAMES, "LlvmPipe"), Ok(2));
        assert_eq!(match_adapter(&NAMES, "geforce rtx 5090"), Ok(0));
        let owned = NAMES.map(str::to_owned);
        assert_eq!(match_adapter(&owned, "890M"), Ok(1));
    }

    #[test]
    fn no_match_and_several_matches_are_errors_that_list_every_adapter() {
        let none = match_adapter(&NAMES, "arc").unwrap_err();
        assert!(none.contains("matches no adapter"), "{none}");
        for name in NAMES {
            assert!(none.contains(name), "{none}");
        }
        assert!(none.contains("[2]"), "{none}");
        // "a" is in every name above but one, so it is ambiguous; the error says which indices match.
        let many = match_adapter(&NAMES, "a").unwrap_err();
        assert!(
            many.contains("matches 2 adapters, not one (0, 1)"),
            "{many}"
        );
        assert!(many.contains(NAMES[1]), "{many}");
        let two = match_adapter(&NAMES, "(").unwrap_err();
        assert!(two.contains("matches 2 adapters, not one (1, 2)"), "{two}");
    }

    #[test]
    fn an_empty_substring_or_an_empty_list_is_an_error() {
        assert!(match_adapter(&NAMES, "").is_err());
        assert!(match_adapter(&NAMES, "  ").is_err());
        let none: [&str; 0] = [];
        let error = match_adapter(&none, "5090").unwrap_err();
        assert!(error.contains("no adapter was found"), "{error}");
    }

    #[test]
    fn the_device_options_name_the_index_and_keep_the_limit_policy_of_the_run() {
        assert_eq!(device_options(1, true).adapter, AdapterChoice::Index(1));
        let portable = device_options(2, false);
        assert_eq!(portable.adapter, AdapterChoice::Index(2));
        assert_eq!(portable.limits, gust_wgpu::LimitsPolicy::Default);
        assert_eq!(portable.features, gust_wgpu::FeaturesPolicy::Default);
        assert_ne!(
            device_options(0, true).limits,
            gust_wgpu::LimitsPolicy::Default
        );
    }
}

//! The report: the header, one table row and the line of the variant text.

use std::fmt::Write as _;

use gust_wgpu::{DeviceHandle, HeadlessDevice};

use crate::kernels::kit;
use crate::measure::Measured;
use crate::setup::{FRAME_MS, STEPS, WARMUP};

// ---- the table ----

/// What one workload size produced: a table row's pieces.
pub(crate) struct Row {
    pub(crate) workload: &'static str,
    pub(crate) size: String,
    /// A measured time and the bytes one step moves (`None` where a rate has no meaning), or the reason it has none.
    pub(crate) result: Result<(Measured, Option<f64>), String>,
    pub(crate) note: String,
}

impl Row {
    pub(crate) fn text(&self) -> String {
        match &self.result {
            Ok((measured, bytes)) => {
                let rate = bytes.map_or_else(
                    || "n/a".to_owned(),
                    |b| format!("{:.1}", b / (measured.ms * 1e-3) / 1e9),
                );
                let frame = if self.workload.starts_with("W6") {
                    "n/a".to_owned()
                } else {
                    format!("{:.1}", FRAME_MS / measured.ms)
                };
                format!(
                    "| {} | {} | {} | {:.4} | {} | {} | ok | {} |",
                    self.workload, self.size, measured.method, measured.ms, rate, frame, self.note
                )
            }
            Err(reason) => format!(
                "| {} | {} | n/a | n/a | n/a | n/a | {} | {} |",
                self.workload,
                self.size,
                if reason.starts_with("skipped") {
                    "skipped"
                } else {
                    "FAILED"
                },
                if reason.starts_with("skipped") {
                    reason.clone()
                } else {
                    format!("{reason}; {}", self.note)
                }
            ),
        }
    }

    pub(crate) fn failed(&self) -> bool {
        matches!(&self.result, Err(reason) if !reason.starts_with("skipped"))
    }
}

/// Why a buffer of `bytes` cannot be bound on this device, or `None`.
pub(crate) fn skip_reason(device: &HeadlessDevice, bytes: u64) -> Option<String> {
    let limits = device.limits();
    (bytes > limits.max_storage_buffer_binding_size || bytes > limits.max_buffer_size).then(|| {
        format!(
            "skipped: {bytes} bytes in one buffer exceed the device's storage binding or buffer limit"
        )
    })
}

/// The total bytes of Slang text of this kit's kernels and of their split variants (slice m11.g4b): what the variants
/// add to the binary. A measurement, with no target.
pub(crate) fn variant_text_line() -> String {
    let kernels = [
        &kit::w1_copy::DESCRIPTOR,
        &kit::w2_diffuse::DESCRIPTOR,
        &kit::w3_step::DESCRIPTOR,
        &kit::w4_clear::DESCRIPTOR,
        &kit::w4_count::DESCRIPTOR,
        &kit::w4_scatter::DESCRIPTOR,
        &kit::g4_linear::DESCRIPTOR,
        &kit::g4_grid::DESCRIPTOR,
        &kit::w6_plus_one::DESCRIPTOR,
        &kit::w8_deposit::DESCRIPTOR,
        &kit::w8_fold::DESCRIPTOR,
        &kit::w8_retire::DESCRIPTOR,
        &kit::w8_compact::DESCRIPTOR,
        &kit::w8_spawn::DESCRIPTOR,
        &kit::w8_bin_count::DESCRIPTOR,
        &kit::w8_bin_scatter::DESCRIPTOR,
        &kit::w8_step_count_deposit::DESCRIPTOR,
    ];
    let base: usize = kernels.iter().map(|kernel| kernel.slang_source.len()).sum();
    let split: usize = kernels
        .iter()
        .filter_map(|kernel| kernel.split)
        .map(|variant| variant.slang_source.len())
        .sum();
    format!(
        "Variant text: {} kernels, {base} bytes of base Slang text and {split} bytes of split-variant Slang text ({:.2}x)",
        kernels.len(),
        split as f64 / base as f64
    )
}

/// The adapter, the limits, the statistic and the table header, as the report starts.
pub(crate) fn header(small: bool, device: &DeviceHandle) -> String {
    let info = device.adapter_info();
    let limits = device.limits();
    let mut text = String::new();
    let _ = writeln!(
        text,
        "budget-kit ({}): a one-machine design aid, not a benchmark; no threshold is asserted",
        if small {
            "small: the first size of each workload"
        } else {
            "full"
        }
    );
    let _ = writeln!(
        text,
        "adapter {} ({:?}, {:?}, driver {} {})",
        info.name, info.device_type, info.backend, info.driver, info.driver_info
    );
    let _ = writeln!(
        text,
        "gpu timestamps: {}",
        if device.supports_gpu_timestamps() {
            "available"
        } else {
            "not available (rows fall back to the wall clock)"
        }
    );
    let _ = writeln!(
        text,
        "limits: storage binding {} bytes, buffer {} bytes, workgroups per dimension {}, invocations per workgroup {}, workgroup size x {}",
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
        limits.max_compute_workgroups_per_dimension,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x
    );
    let _ = writeln!(
        text,
        "statistic: median of {STEPS} steps after {WARMUP} warm-up steps; a step is one dispatch (W4: a clear, a count and a scan, and in its second row a scatter; W7: one batch of four kernels, a sustained run with its own statistics in the note; W8: a whole step of four batches (two of them scans) in one timing scope, a sustained run with its own statistics in the note; W8a: the same without bins, three batches (one scan); W8b and W8c: four batches as W8; W8d: the lifecycle on every 4th step only, so the steps between are three batches, and its time is the amortised mean per step over whole rounds of 4 steps, not a median; W5: a whole reduction; with GPU timestamps the pooled-buffer clears are outside the timed region, but the wall-clock fallback includes the clears, the allocation (not that of the W4 scan output or the scatter call, which are made once before the steps), the host overhead and the scalar readback; W6: submit, wait and readback); steps per frame is {FRAME_MS:.1} ms over the median; GB/s is the stated bytes over the median"
    );
    let _ = writeln!(
        text,
        "sizes are exact counts; 1M is 2^20 and 4M is 2^22 elements"
    );
    let _ = writeln!(text);
    let _ = writeln!(
        text,
        "| workload | size | timing method | ms per step | GB/s | steps per 16.7 ms | check | note |"
    );
    let _ = writeln!(text, "|---|---|---|---:|---:|---:|---|---|");
    text
}

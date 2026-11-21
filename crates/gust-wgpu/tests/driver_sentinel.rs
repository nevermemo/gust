// Slice m9.sen (D168, D169): a sentinel for the NVIDIA array defect. See `fixtures/driver_sentinel.rs` for the kernels.
include!("fixtures/driver_sentinel.rs");

// Slice m11.ld: a sentinel for the NVIDIA float-break loop defect (SEMANTICS, "Known driver defect (m11.ld)"). The
// loop has a constant bound, and its exit is a `break` on a float comparison; `s` starts at 2^(t - 126) and is halved
// each step, so the exit is due at iteration t - 1 (n = t). The start values and the thresholds come from buffers,
// built from bits on the host, so the values are exact whatever the compiler folds.
#[gust::gpu]
mod loop_break {
    // limit[i] is 1.5 * 2^-125 for lt_form and 2^-125 for eq_form. It is a per-invocation buffer, every element
    // alike, only because the typed buffers of one call share one length.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn lt_form(
        id: SV_DispatchThreadID,
        starts: StructuredBuffer<f32>,
        limit: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < out.len() {
            let mut s = starts[i];
            let thr = limit[i];
            let mut n = 0u32;
            for j in 0u32..255u32 {
                if s < thr {
                    n = j + 1u32;
                    break;
                }
                s *= 0.5f32;
            }
            out[i] = n;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn eq_form(
        id: SV_DispatchThreadID,
        starts: StructuredBuffer<f32>,
        limit: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < out.len() {
            let mut s = starts[i];
            let y = limit[i];
            let mut n = 0u32;
            for j in 0u32..255u32 {
                if s == y {
                    n = j + 1u32;
                    break;
                }
                s *= 0.5f32;
            }
            out[i] = n;
        }
    }
}

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

const COUNT: usize = 129;
// a[0] of the D168 programs; the correct result is -3 - 1 = -4 in every element.
const A0: i32 = 1;
const D168_RIGHT: i32 = -4;
// D169, written by hand from the program: out[i] reads element 3 of the copy, which holds 2048 and is shifted left
// by 10 to 2097152 when i % 4 == 3.
const D169_SHIFTED: u32 = 2_097_152;
const D169_PLAIN: u32 = 2048;

const NVIDIA: u32 = 0x10DE;

fn d168_right(_i: usize) -> i32 {
    D168_RIGHT
}

fn d169_right(i: usize) -> u32 {
    if i % 4 == 3 { D169_SHIFTED } else { D169_PLAIN }
}

#[derive(Debug, PartialEq)]
enum Verdict {
    Right,
    KnownDefect,
    Other,
}

// Judge a result table against Rust's values and the defect's recorded values, element by element.
fn judge<T: PartialEq + Copy>(
    got: &[T],
    right: impl Fn(usize) -> T,
    defect: impl Fn(usize) -> T,
) -> Verdict {
    if got.iter().enumerate().all(|(i, g)| *g == right(i)) {
        Verdict::Right
    } else if got.iter().enumerate().all(|(i, g)| *g == defect(i)) {
        Verdict::KnownDefect
    } else {
        Verdict::Other
    }
}

fn first_mismatches<T: PartialEq + Copy + std::fmt::Debug>(
    got: &[T],
    right: impl Fn(usize) -> T,
) -> String {
    let rows: Vec<String> = got
        .iter()
        .enumerate()
        .filter(|(i, g)| **g != right(*i))
        .take(8)
        .map(|(i, g)| format!("[{i}] got {g:?} want {:?}", right(i)))
        .collect();
    rows.join("; ")
}

// What the adapter is held to. Only the RTX 5090 on Vulkan was measured (D168, D169).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Device {
    // The measured device: the defect's values are expected, anything else fails.
    Measured,
    // Another NVIDIA adapter or backend: Rust's values or the defect's pass, and the verdict is noted.
    UnmeasuredNvidia,
    // Any other adapter: Rust's values.
    Other,
}

// The measured device is matched by the adapter name: a case-sensitive substring "RTX 5090", on the Vulkan backend
// only (m11.r4fix). Any other NVIDIA adapter, and this one on another backend, is `UnmeasuredNvidia`: Rust's values
// and the defect's values both pass there, and the verdict is only noted. A non-NVIDIA adapter must give Rust's values.
fn classify(info: &wgpu::AdapterInfo) -> Device {
    if info.vendor != NVIDIA {
        Device::Other
    } else if info.backend == wgpu::Backend::Vulkan && info.name.contains("RTX 5090") {
        Device::Measured
    } else {
        Device::UnmeasuredNvidia
    }
}

// Judge one program: Some(failure text) when the adapter's expectation is not met.
fn failure<T: PartialEq + Copy + std::fmt::Debug>(
    device: Device,
    verdict: &Verdict,
    got: &[T],
    right: impl Fn(usize) -> T,
) -> Option<String> {
    match (device, verdict) {
        (Device::Other, Verdict::Right)
        | (Device::Measured, Verdict::KnownDefect)
        | (Device::UnmeasuredNvidia, Verdict::Right | Verdict::KnownDefect) => None,
        (Device::Measured, Verdict::Right) => Some(
            "computes Rust's values on the measured device: the driver or the toolchain (GUST, slangc, naga) \
             changed the result"
                .to_string(),
        ),
        (Device::Other, Verdict::KnownDefect) => Some(format!(
            "a non-NVIDIA adapter shows the NVIDIA defect's values ({}); a new finding, not D168/D169",
            first_mismatches(got, &right)
        )),
        (_, Verdict::Other) => Some(format!(
            "neither Rust's values nor the recorded defect's: {}",
            first_mismatches(got, &right)
        )),
    }
}

type Judged = (&'static str, Verdict, Option<String>);

fn judged<T: PartialEq + Copy + std::fmt::Debug>(
    name: &'static str,
    device: Device,
    got: &[T],
    right: impl Fn(usize) -> T,
    defect: impl Fn(usize) -> T,
) -> Judged {
    assert_eq!(got.len(), COUNT, "{name}: result length");
    let verdict = judge(got, &right, &defect);
    let fail = failure(device, &verdict, got, &right);
    (name, verdict, fail)
}

#[test]
fn the_nvidia_array_defect_is_still_there_or_absent_elsewhere() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    let adapter = format!(
        "{} (vendor {:#06x}, driver {:?} {:?}, {:?})",
        info.name, info.vendor, info.driver, info.driver_info, info.backend
    );
    eprintln!("sentinel adapter: {adapter}");
    let kind = classify(info);
    // On the SPIR-V route (m9.rt2) the program is not naga's: the defect was measured on the WGSL route, so a device
    // whose sentinel kernels took the route is held to Rust's values (a defect there is a new finding).
    let held = |device: &HeadlessDevice| {
        if spirv_route_taken(device) {
            Device::Other
        } else {
            kind
        }
    };

    let a = device.create_typed_buffer("sentinel a", &vec![A0; COUNT], GpuBufferAccess::ReadOnly);
    let mut results: Vec<Judged> = Vec::new();

    let blank_i = vec![i32::MIN; COUNT];
    for (name, n) in [("d168_n3", 3usize), ("d168_n5", 5)] {
        let out = device.create_typed_buffer("sentinel out", &blank_i, GpuBufferAccess::ReadWrite);
        if n == 3 {
            device
                .dispatch_call(&sentinel::d168_n3::call(COUNT as u32, &a, &out))
                .unwrap();
        } else {
            device
                .dispatch_call(&sentinel::d168_n5::call(COUNT as u32, &a, &out))
                .unwrap();
        }
        let got = device.read_typed_buffer(&out).unwrap();
        // Recorded on the RTX 5090: element 0 reads as 0 wherever the index was not 0.
        results.push(judged(name, held(&device), &got, d168_right, |i| {
            if i % n == 0 { D168_RIGHT } else { 0 }
        }));
    }

    let blank_u = vec![u32::MAX; COUNT];
    let out = device.create_typed_buffer("sentinel out", &blank_u, GpuBufferAccess::ReadWrite);
    let call = sentinel::d169_len4::call(COUNT as u32, &out);
    device.dispatch_call(&call).unwrap();
    let got = device.read_typed_buffer(&out).unwrap();
    // Recorded on the RTX 5090: 0 where i % 4 == 3.
    results.push(judged("d169_len4", held(&device), &got, d169_right, |i| {
        if i % 4 == 3 { 0 } else { D169_PLAIN }
    }));

    // Every verdict is printed before the test fails, once.
    let mut failed = Vec::new();
    for (name, verdict, fail) in &results {
        eprintln!("sentinel {name}: {verdict:?} on {adapter}");
        if kind == Device::UnmeasuredNvidia {
            eprintln!(
                "NOTE: unmeasured NVIDIA adapter: record this result in D168/D169's envelope ({name}: {verdict:?})"
            );
        }
        if let Some(text) = fail {
            failed.push(format!("{name}: {text}"));
        }
    }
    if !failed.is_empty() {
        let mut message = format!(
            "driver sentinel failed on {adapter}:\n  {}\n",
            failed.join("\n  ")
        );
        if kind == Device::Measured && results.iter().any(|(_, v, _)| *v == Verdict::Right) {
            message.push_str(
                "Re-measure the programs of D168 and D169 (docs/DECISIONS.md: the shapes of the two investigations that \
                 found the defect) before touching the guard. Remove the difftest's array guard (`Gen::guard_set` and `scan_driver_hole` in \
                 tests/difftest/src/array_tests.rs, also used by closure_tests.rs) and this sentinel's expectation \
                 only when a re-measure shows the driver computes Rust's values for every program.",
            );
        }
        panic!("{message}");
    }
}

/// Whether every kernel this device built so far took the SPIR-V route (m9.rt2). `false` without the feature.
#[cfg(feature = "spirv-route")]
fn spirv_route_taken(device: &HeadlessDevice) -> bool {
    let events = device.route_events();
    !events.is_empty()
        && events
            .iter()
            .all(|event| event.route == gust_wgpu::Route::Spirv)
}

#[cfg(not(feature = "spirv-route"))]
fn spirv_route_taken(_device: &HeadlessDevice) -> bool {
    false
}

// m11.ld: the iteration at which the break is due, per invocation (Rust's value is n == t).
const LOOP_T: [u32; 6] = [100, 127, 128, 129, 200, 250];
// The values measured on the RTX 5090 (NVIDIA 610.88, Vulkan) on both routes: a `<` break due after 128 iterations is
// taken at 128; an `==` break due at iteration 128 or later never fires (n stays 0).
const LOOP_LT_DEFECT: [u32; 6] = [100, 127, 128, 128, 128, 128];
const LOOP_EQ_DEFECT: [u32; 6] = [100, 127, 0, 0, 0, 0];

#[test]
fn the_nvidia_float_break_loop_defect_is_still_there_or_absent_elsewhere() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    let adapter = format!(
        "{} (vendor {:#06x}, driver {:?} {:?}, {:?})",
        info.name, info.vendor, info.driver, info.driver_info, info.backend
    );
    eprintln!("loop sentinel adapter: {adapter}");
    let kind = classify(info);

    // 2^(t - 126) from its bits: the exponent field is t - 126 + 127 = t + 1.
    let starts: Vec<f32> = LOOP_T
        .iter()
        .map(|t| f32::from_bits((t + 1) << 23))
        .collect();
    let thr = f32::from_bits((2u32 << 23) | (1 << 22)); // 1.5 * 2^-125: exponent field 2, mantissa .5
    let y = f32::from_bits(2u32 << 23); // 2^-125
    assert_eq!(starts[0], 2.0f32.powi(100 - 126));
    assert_eq!(y, 2.0f32.powi(-125));
    assert_eq!(thr, 1.5 * 2.0f32.powi(-125));
    let thr_all = vec![thr; LOOP_T.len()];
    let y_all = vec![y; LOOP_T.len()];

    let device_starts =
        device.create_typed_buffer("loop starts", &starts, GpuBufferAccess::ReadOnly);
    let device_thr = device.create_typed_buffer("loop thr", &thr_all, GpuBufferAccess::ReadOnly);
    let device_y = device.create_typed_buffer("loop y", &y_all, GpuBufferAccess::ReadOnly);
    let blank = vec![u32::MAX; LOOP_T.len()];
    let count = LOOP_T.len() as u32;

    let out_lt = device.create_typed_buffer("loop out lt", &blank, GpuBufferAccess::ReadWrite);
    device
        .dispatch_call(&loop_break::lt_form::call(
            count,
            &device_starts,
            &device_thr,
            &out_lt,
        ))
        .unwrap();
    let gpu_lt = device.read_typed_buffer(&out_lt).unwrap();
    let out_eq = device.create_typed_buffer("loop out eq", &blank, GpuBufferAccess::ReadWrite);
    device
        .dispatch_call(&loop_break::eq_form::call(
            count,
            &device_starts,
            &device_y,
            &out_eq,
        ))
        .unwrap();
    let gpu_eq = device.read_typed_buffer(&out_eq).unwrap();

    // The CPU twin gives Rust's values everywhere.
    let cpu = CpuDevice::new();
    let twin_starts = cpu.buffer(&starts);
    let twin_thr = cpu.buffer(&thr_all);
    let twin_y = cpu.buffer(&y_all);
    let twin_lt = cpu.buffer(&blank);
    loop_break::lt_form::dispatch(&cpu, count, &twin_starts, &twin_thr, &twin_lt).unwrap();
    let twin_eq = cpu.buffer(&blank);
    loop_break::eq_form::dispatch(&cpu, count, &twin_starts, &twin_y, &twin_eq).unwrap();
    assert_eq!(twin_lt.read(), LOOP_T, "CPU twin, lt form");
    assert_eq!(twin_eq.read(), LOOP_T, "CPU twin, eq form");

    // The defect was measured on both routes (the WGSL route and, with the `spirv-route` feature and the route
    // taken, the SPIR-V route), so the measured device is held to the defect's values on either, and a failure
    // names the route.
    let route = spirv_route_taken(&device);
    let route_name = if route { "SPIR-V route" } else { "WGSL route" };
    let held = kind;
    let right = |i: usize| LOOP_T[i];
    let results = [
        judged_loop("lt_form", held, &gpu_lt, right, |i| LOOP_LT_DEFECT[i]),
        judged_loop("eq_form", held, &gpu_eq, right, |i| LOOP_EQ_DEFECT[i]),
    ];
    let mut failed = Vec::new();
    for (name, verdict, fail, got) in &results {
        eprintln!(
            "loop sentinel {name}: {verdict:?} on {adapter}; got {got:?}; spirv route taken: {route}"
        );
        if kind == Device::UnmeasuredNvidia {
            eprintln!(
                "NOTE: unmeasured NVIDIA adapter: record this result in m11.ld's envelope ({name}: {verdict:?})"
            );
        }
        if let Some(text) = fail {
            failed.push(format!("{name} ({route_name}): {text}"));
        }
    }
    if !failed.is_empty() {
        panic!(
            "float-break loop sentinel failed on {adapter}:\n  {}\nA change of the RTX 5090's values is the cue to \
             re-measure the m11.ld loop before touching the expectation.",
            failed.join("\n  ")
        );
    }
}

fn judged_loop(
    name: &'static str,
    device: Device,
    got: &[u32],
    right: impl Fn(usize) -> u32,
    defect: impl Fn(usize) -> u32,
) -> (&'static str, Verdict, Option<String>, Vec<u32>) {
    assert_eq!(got.len(), LOOP_T.len(), "{name}: result length");
    let verdict = judge(got, &right, &defect);
    let fail = failure(device, &verdict, got, &right);
    (name, verdict, fail, got.to_vec())
}

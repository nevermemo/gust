// Slice m9.sen (D168, D169): a sentinel for the NVIDIA array defect. See `fixtures/driver_sentinel.rs` for the kernels.
include!("fixtures/driver_sentinel.rs");

use gust::Run;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

const COUNT: usize = 129;
// a[0] of the D168 programs; the correct result is -3 - 1 = -4 in every element.
const A0: i32 = 1;
const D168_RIGHT: i32 = -4;
// D169, written by hand from the program: element i % 4 is shifted left by 10 (2048 << 10 = 2097152) and out[i] reads
// element 3 of the copy, so only i % 4 == 3 sees the shifted value.
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

// On NVIDIA the defect's values are expected; on any other adapter Rust's.
fn check<T: PartialEq + Copy + std::fmt::Debug>(
    name: &str,
    nvidia: bool,
    got: &[T],
    right: impl Fn(usize) -> T,
    defect: impl Fn(usize) -> T,
) {
    assert_eq!(got.len(), COUNT, "{name}: result length");
    let verdict = judge(got, &right, &defect);
    eprintln!("sentinel {name}: {verdict:?}");
    match (nvidia, verdict) {
        (false, Verdict::Right) | (true, Verdict::KnownDefect) => {}
        (true, Verdict::Right) => panic!(
            "{name}: this NVIDIA driver no longer shows D168/D169 (Rust's values came back). Re-measure the \
             defect with the investigation programs, then remove the array guard of tests/difftest (m9.d1e) and \
             this sentinel's expectation (the NVIDIA branch of `check` in crates/gust-wgpu/tests/driver_sentinel.rs)."
        ),
        (false, Verdict::KnownDefect) => panic!(
            "{name}: a non-NVIDIA adapter shows the NVIDIA defect's values ({}); that is a new finding, not D168/D169",
            first_mismatches(got, &right)
        ),
        (_, Verdict::Other) => panic!(
            "{name}: neither Rust's values nor the recorded defect's: {}",
            first_mismatches(got, &right)
        ),
    }
}

#[test]
fn the_nvidia_array_defect_is_still_there_or_absent_elsewhere() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    eprintln!(
        "sentinel adapter: {} (vendor {:#06x}, driver {:?} {:?}, {:?})",
        info.name, info.vendor, info.driver, info.driver_info, info.backend
    );
    let nvidia = info.vendor == NVIDIA;

    let a = device.create_typed_buffer("sentinel a", &vec![A0; COUNT], GpuBufferAccess::ReadOnly);

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
        check(name, nvidia, &got, d168_right, |i| {
            if i % n == 0 { D168_RIGHT } else { 0 }
        });
    }

    let blank_u = vec![u32::MAX; COUNT];
    let out = device.create_typed_buffer("sentinel out", &blank_u, GpuBufferAccess::ReadWrite);
    let call = sentinel::d169_len4::call(COUNT as u32, &out);
    device.dispatch_call(&call).unwrap();
    let got = device.read_typed_buffer(&out).unwrap();
    // Recorded on the RTX 5090: 0 where i % 4 == 3.
    check("d169_len4", nvidia, &got, d169_right, |i| {
        if i % 4 == 3 { 0 } else { D169_PLAIN }
    });
}

//! `DeviceOptions::unchecked_kernels` (m14.pftrust1, D-ruling pf.trust): the device-level opt-in that builds the WGSL
//! route's shader modules without wgpu's runtime checks.
//!
//! The default device is checked, and a kernel that reads past the end of a buffer stays inside it (wgpu's bounds
//! policy, or the driver's own robust access). A device opened with the option runs kernels that stay inside their
//! buffers, with loops and integer division, to the same results as the checked device and the CPU twin. A kernel that
//! goes out of range on the unchecked device is undefined behaviour, so no test does that, and no test here sees a
//! GPU-level difference between the two settings; the module call is observed by a unit test in `compile/modules.rs`.

mod common;

use common::unchecked_device;
use gust::cpu::CpuDevice;
use gust_wgpu::{
    BufferBinding, DeviceOptions, GpuBufferAccess, HeadlessDevice, ShaderChecks, default_device,
};

#[gust::gpu_module]
mod probe {
    // Reads `inp[i + shift[0]]`: past the end of `inp` for a large shift.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn shifted(
        id: SV_DispatchThreadID,
        inp: StructuredBuffer<uint>,
        shift: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = inp[i + shift[0]];
        }
    }

    // A loop and integer division and remainder by a divisor the host keeps non-zero; every index is in range.
    #[allow(clippy::assign_op_pattern)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn mix(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<uint>,
        b: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < out.len() {
            let mut acc = 0u32;
            for k in 0u32..16u32 {
                acc = acc + (a[i] + k) / b[i] + (a[i] + k) % b[i];
            }
            out[i] = acc;
        }
    }
}

fn run_mix(device: &HeadlessDevice, a: &[u32], b: &[u32]) -> Vec<u32> {
    let a_buf = device.create_typed_buffer("mix a", a, GpuBufferAccess::ReadOnly);
    let b_buf = device.create_typed_buffer("mix b", b, GpuBufferAccess::ReadOnly);
    let out =
        device.create_typed_buffer("mix out", &vec![0u32; a.len()], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &probe::mix::DESCRIPTOR,
            a.len() as u32,
            &[
                BufferBinding::read_only(&a_buf),
                BufferBinding::read_only(&b_buf),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    device.read_typed_buffer(&out).unwrap()
}

fn host_mix(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| (0..16u32).fold(0u32, |acc, k| acc + (a + k) / b + (a + k) % b))
        .collect()
}

#[test]
fn a_default_device_is_checked() {
    assert_eq!(ShaderChecks::default(), ShaderChecks::Checked);
    assert_eq!(
        DeviceOptions::default().shader_checks(),
        ShaderChecks::Checked
    );
    // Not through `common::device`, which the test variable can switch.
    assert_eq!(
        HeadlessDevice::new().unwrap().shader_checks(),
        ShaderChecks::Checked
    );
    assert_eq!(
        HeadlessDevice::with_options(DeviceOptions::default())
            .unwrap()
            .shader_checks(),
        ShaderChecks::Checked
    );
    assert_eq!(
        default_device().unwrap().shader_checks(),
        ShaderChecks::Checked
    );
}

#[test]
fn the_option_reaches_the_device() {
    let unchecked = unchecked_device();
    assert_eq!(unchecked.shader_checks(), ShaderChecks::Unchecked);
}

/// With the checks on, a read past the end of a read-only buffer never returns a value from outside it. What it returns
/// depends on the layer that guards it: wgpu's bounds policy is `Restrict` (the index is clamped, so the last element,
/// 64) on a device without `robustBufferAccess2`, and left to the driver on one that has it (the driver's robust access
/// returns zero). The test accepts either and nothing else. The SPIR-V route is not this path, so it is skipped there.
#[cfg(not(feature = "spirv-route"))]
#[test]
fn a_checked_device_keeps_a_read_past_the_end_of_a_buffer_inside_it() {
    let device = HeadlessDevice::new().unwrap();
    assert_eq!(device.shader_checks(), ShaderChecks::Checked);
    let inp: Vec<u32> = (1..=64).collect();
    let input = device.create_typed_buffer("shifted in", &inp, GpuBufferAccess::ReadOnly);
    // Every read is at least 100 elements past the end.
    let shift =
        device.create_typed_buffer("shifted shift", &[100u32; 64], GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("shifted out", &[7u32; 64], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &probe::shifted::DESCRIPTOR,
            64,
            &[
                BufferBinding::read_only(&input),
                BufferBinding::read_only(&shift),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    let got = device.read_typed_buffer(&out).unwrap();
    assert!(
        got.iter().all(|&v| v == 0 || v == 64),
        "a read past the end returned a value from outside the buffer: {got:?}"
    );
}

#[test]
fn an_unchecked_device_equals_the_checked_device_and_the_twin_inside_the_buffers() {
    let a: Vec<u32> = (0..300).map(|i| (i * 7919 + 13) % 100_003).collect();
    let b: Vec<u32> = (0..300).map(|i| (i * 31 + 5) % 97 + 1).collect();
    assert!(
        b.iter().all(|&d| d != 0),
        "the host keeps every divisor non-zero"
    );
    let want = host_mix(&a, &b);

    let checked = HeadlessDevice::new().unwrap();
    let unchecked = unchecked_device();
    assert_eq!(
        (checked.shader_checks(), unchecked.shader_checks()),
        (ShaderChecks::Checked, ShaderChecks::Unchecked)
    );
    assert_eq!(run_mix(&checked, &a, &b), want, "checked device");
    assert_eq!(run_mix(&unchecked, &a, &b), want, "unchecked device");

    let cpu = CpuDevice::new();
    let (ta, tb) = (cpu.buffer(&a), cpu.buffer(&b));
    let tout = cpu.buffer(&vec![0u32; a.len()]);
    probe::mix::dispatch(&cpu, a.len() as u32, &ta, &tb, &tout).unwrap();
    assert_eq!(tout.read(), want, "CPU twin");
}

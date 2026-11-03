//! Workgroup memory and the workgroup barrier on a real device (slice m7.g1): the block reduce of
//! `tests/fixtures/workgroup.rs` against an independent host sum, the local and group ids of every invocation, and the
//! barrier in the SPIR-V and the WGSL the kernel compiles to.

include!("../../../tests/fixtures/workgroup.rs");

use std::{fs, process::Command};

use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

/// Invocations per workgroup of both kernels.
const GROUP: usize = 256;

/// `OpControlBarrier` of the SPIR-V specification (unified 1.5, section 3.56.20).
const OP_CONTROL_BARRIER: u32 = 224;

/// Integer-valued `f32` inputs: every value is a whole number below 97, so every partial sum of up to 65 537 of them is
/// a whole number below 2^24 and exact in `f32` in any order. The tree reduction therefore has to equal the integer
/// sum bit for bit, and the test does not need a rounding bound.
fn input_values(count: usize) -> Vec<f32> {
    (0..count).map(|i| ((i * 7 + 3) % 97) as f32).collect()
}

/// The sum of each block of 256 input elements, in `u64` integers: independent of the kernel's tree and of `f32`.
fn block_sums(input: &[f32]) -> Vec<f32> {
    input
        .chunks(GROUP)
        .map(|block| block.iter().map(|value| *value as u64).sum::<u64>() as f32)
        .collect()
}

#[test]
fn block_reduce_matches_the_host_sum() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("workgroup adapter: {:?}", device.adapter_info());
    // One element, one short of a group, a group, one past it, and a count with a one-element last group that needs
    // 257 workgroups.
    for count in [1usize, 255, 256, 257, 65_537] {
        let input = input_values(count);
        let groups = count.div_ceil(GROUP);
        let input_buffer =
            device.create_typed_buffer("workgroup input", &input, GpuBufferAccess::ReadOnly);
        // One partial per workgroup, so its length is not the element count (`independent_length` below); the kernel
        // writes `partial[gid.x]` and `gid.x` is below the number of workgroups, which is this length.
        // The sentinel shows an unwritten partial: a workgroup that never ran would leave it.
        let partial = device.create_typed_buffer(
            "workgroup partial",
            &vec![-1.0f32; groups],
            GpuBufferAccess::ReadWrite,
        );
        device
            .dispatch_buffers(
                &workgroup::block_sum::DESCRIPTOR,
                count as u32,
                &[
                    BufferBinding::read_only(&input_buffer),
                    BufferBinding::read_write(&partial).independent_length(),
                ],
            )
            .unwrap();
        let actual = device.read_typed_buffer(&partial).unwrap();
        let expected = block_sums(&input);
        assert_eq!(actual.len(), expected.len(), "count {count}");
        for (block, (got, want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "partial[{block}] for count {count}: got {got}, want {want}"
            );
        }
        // The total over the partials is the sum of the whole input.
        let total: f32 = actual.iter().sum();
        let wanted: u64 = input.iter().map(|value| *value as u64).sum();
        assert_eq!(total as u64, wanted, "total for count {count}");
    }
}

#[test]
fn local_and_group_ids_are_the_dispatch_id_modulo_and_over_the_group_size() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    for count in [1usize, 255, 256, 257, 1000, 65_537] {
        let locals = device.create_typed_buffer(
            "workgroup locals",
            &vec![u32::MAX; count],
            GpuBufferAccess::ReadWrite,
        );
        let groups = device.create_typed_buffer(
            "workgroup groups",
            &vec![u32::MAX; count],
            GpuBufferAccess::ReadWrite,
        );
        device
            .dispatch_buffers(
                &workgroup::local_ids::DESCRIPTOR,
                count as u32,
                &[
                    BufferBinding::read_write(&locals),
                    BufferBinding::read_write(&groups),
                ],
            )
            .unwrap();
        let locals = device.read_typed_buffer(&locals).unwrap();
        let groups = device.read_typed_buffer(&groups).unwrap();
        for id in 0..count {
            assert_eq!(
                locals[id],
                (id % GROUP) as u32,
                "local id of {id}, count {count}"
            );
            assert_eq!(
                groups[id],
                (id / GROUP) as u32,
                "group id of {id}, count {count}"
            );
        }
    }
}

#[test]
fn the_kernel_with_ids_only_keeps_its_cpu_twin_and_agrees_with_the_gpu() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let count = 700usize;
    let locals = device.create_typed_buffer(
        "twin locals",
        &vec![0u32; count],
        GpuBufferAccess::ReadWrite,
    );
    let groups = device.create_typed_buffer(
        "twin groups",
        &vec![0u32; count],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &workgroup::local_ids::DESCRIPTOR,
            count as u32,
            &[
                BufferBinding::read_write(&locals),
                BufferBinding::read_write(&groups),
            ],
        )
        .unwrap();
    let cpu = gust::cpu::CpuDevice::new();
    let twin_locals = cpu.buffer(&vec![0u32; count]);
    let twin_groups = cpu.buffer(&vec![0u32; count]);
    workgroup::local_ids::dispatch(&cpu, count as u32, &twin_locals, &twin_groups).unwrap();
    assert_eq!(
        twin_locals.read(),
        device.read_typed_buffer(&locals).unwrap()
    );
    assert_eq!(
        twin_groups.read(),
        device.read_typed_buffer(&groups).unwrap()
    );
}

/// The opcode of every instruction of a SPIR-V module, walked by word count from the end of the five-word header.
fn instructions(words: &[u32]) -> Vec<u32> {
    let mut opcodes = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        assert!(count > 0, "a zero-length instruction at word {at}");
        opcodes.push(words[at] & 0xFFFF);
        at += count;
    }
    opcodes
}

#[test]
fn the_barrier_is_an_op_control_barrier_in_the_spirv_and_a_workgroup_barrier_in_the_wgsl() {
    let words = gust::slang::compile_spirv(&workgroup::block_sum::DESCRIPTOR).unwrap();
    gust::spirv::validate_structure(&words).unwrap();
    // Checked two ways: the opcode of every instruction, and the disassembler's own name for it.
    let barriers = instructions(&words)
        .into_iter()
        .filter(|opcode| *opcode == OP_CONTROL_BARRIER)
        .count();
    assert!(
        barriers >= 2,
        "the kernel has one barrier before its loop and one in it, found {barriers} OpControlBarrier"
    );
    let directory = gust::slang::TemporaryDirectory::create().unwrap();
    let artifact = directory.path().join("block_sum.spv");
    fs::write(&artifact, gust::spirv::words_as_le_bytes(&words)).unwrap();
    let status = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(&artifact)
        .status()
        .expect("spirv-val must be installed for workgroup validation");
    assert!(status.success(), "spirv-val rejected the workgroup SPIR-V");
    let listing = Command::new("spirv-dis")
        .arg(&artifact)
        .output()
        .expect("spirv-dis must be installed for workgroup validation");
    assert!(listing.status.success(), "spirv-dis failed");
    let listing = String::from_utf8_lossy(&listing.stdout);
    assert!(listing.contains("OpControlBarrier"), "{listing}");
    assert!(listing.contains("Workgroup"), "{listing}");

    let wgsl = gust::slang::compile_wgsl(&workgroup::block_sum::DESCRIPTOR).unwrap();
    assert!(wgsl.contains("workgroupBarrier()"), "{wgsl}");
    assert!(wgsl.contains("var<workgroup>"), "{wgsl}");

    // The kernel without workgroup memory has neither.
    let plain = gust::slang::compile_spirv(&workgroup::local_ids::DESCRIPTOR).unwrap();
    assert!(
        !instructions(&plain).contains(&OP_CONTROL_BARRIER),
        "local_ids has no barrier"
    );
}

#[test]
fn workgroup_memory_and_the_derived_ids_are_not_parameters_of_the_descriptor() {
    let block_sum = &workgroup::block_sum::DESCRIPTOR;
    let names: Vec<&str> = block_sum.parameters.iter().map(|p| p.name).collect();
    assert_eq!(names, ["id", "input", "partial"]);
    let local_ids = &workgroup::local_ids::DESCRIPTOR;
    let names: Vec<&str> = local_ids.parameters.iter().map(|p| p.name).collect();
    assert_eq!(names, ["id", "locals", "groups"]);
}

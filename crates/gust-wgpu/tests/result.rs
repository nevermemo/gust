include!("../../../tests/fixtures/result.rs");

use std::{fs, process::Command};

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[allow(clippy::unnecessary_literal_unwrap)]
#[gust::gpu]
mod mixed_option_result {
    fn classify(value: uint) -> Result<uint, uint> {
        if value > 0u32 { Ok(value) } else { Err(value) }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut output: RWStructuredBuffer<uint>,
    ) {
        if id.x < input.len() {
            let option: Option<uint> = Some(input[id.x]);
            let result = classify(option.unwrap_or(0u32));
            output[id.x] = result.unwrap_or(option.unwrap_or(1u32));
        }
    }
}

fn reference(value: u32) -> (u32, u32) {
    let result = if value.is_multiple_of(2) {
        Ok(value / 2)
    } else {
        Err(value)
    };
    let flags = if result.is_ok() { 1 } else { 2 } + result.unwrap_or(3) * 10;
    let mut total = result.unwrap_or(7);
    match result {
        Ok(half) => {
            total += half;
        }
        Err(error) => {
            total += error;
        }
    }
    (flags, total)
}

#[test]
fn result_lowering_executes_on_gpu() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("result adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        for (label, start) in [("ok", 0u32), ("err", 97u32)] {
            let input = (0..count)
                .map(|index| start + index as u32 * 2u32)
                .collect::<Vec<_>>();
            let input_buffer =
                device.create_typed_buffer("result input", &input, GpuBufferAccess::ReadOnly);
            let flags = device.create_typed_buffer(
                "result flags",
                &vec![0u32; count],
                GpuBufferAccess::ReadWrite,
            );
            let values = device.create_typed_buffer(
                "result values",
                &vec![0u32; count],
                GpuBufferAccess::ReadWrite,
            );
            device
                .dispatch_buffers(
                    &results::run::DESCRIPTOR,
                    count as u32,
                    &[
                        BufferBinding::read_only(&input_buffer),
                        BufferBinding::read_write(&flags),
                        BufferBinding::read_write(&values),
                    ],
                )
                .unwrap();
            let actual_flags = device.read_typed_buffer(&flags).unwrap();
            let actual_values = device.read_typed_buffer(&values).unwrap();

            // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
            let cpu = CpuDevice::new();
            let twin_input = cpu.buffer(&input);
            let twin_flags = cpu.buffer(&vec![0u32; count]);
            let twin_values = cpu.buffer(&vec![0u32; count]);
            results::run::dispatch(&cpu, count as u32, &twin_input, &twin_flags, &twin_values)
                .unwrap();
            let twin_flags = twin_flags.read();
            let twin_values = twin_values.read();
            assert_eq!(
                twin_flags, actual_flags,
                "{label} twin flags, count {count}"
            );
            assert_eq!(
                twin_values, actual_values,
                "{label} twin values, count {count}"
            );

            for index in 0..count {
                let (flags, value) = reference(input[index]);
                assert_eq!(actual_flags[index], flags, "{label} flags[{index}]");
                assert_eq!(actual_values[index], value, "{label} values[{index}]");
                assert_eq!(twin_flags[index], flags, "{label} twin flags[{index}]");
                assert_eq!(twin_values[index], value, "{label} twin values[{index}]");
            }
        }
    }
}

#[test]
fn result_compiles_to_wgsl() {
    assert!(
        gust::slang::compile_wgsl(&results::run::DESCRIPTOR)
            .unwrap()
            .contains("@compute")
    );
}

#[test]
fn result_compiles_to_spirv_and_passes_external_validation() {
    let words = gust::slang::compile_spirv(&results::run::DESCRIPTOR).unwrap();
    gust::spirv::validate_structure(&words).unwrap();
    let directory = gust::slang::TemporaryDirectory::create().unwrap();
    let artifact = directory.path().join("result.spv");
    fs::write(&artifact, gust::spirv::words_as_le_bytes(&words)).unwrap();
    let status = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(&artifact)
        .status()
        .expect("spirv-val must be installed for Result target validation");
    assert!(status.success(), "spirv-val rejected Result SPIR-V");
}

#[test]
fn option_and_result_unwrap_or_overloads_compile_together() {
    let wgsl = gust::slang::compile_wgsl(&mixed_option_result::run::DESCRIPTOR).unwrap();
    assert!(wgsl.contains("@compute"));
}

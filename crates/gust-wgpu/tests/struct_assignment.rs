use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod assignments {
    pub struct Pair {
        pub first: uint,
        pub second: uint,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Pair>,
        mut local_out: RWStructuredBuffer<Pair>,
        mut buffer_out: RWStructuredBuffer<Pair>,
        mut shadow_out: RWStructuredBuffer<Pair>,
    ) {
        let i = id.x;
        if i < input.len() {
            let mut p = input[i];
            p = Pair {
                first: p.second,
                second: p.first,
            };
            local_out[i] = p;
            buffer_out[i] = Pair {
                first: buffer_out[i].second,
                second: buffer_out[i].first,
            };
            {
                let p = Pair {
                    second: p.first,
                    first: p.second,
                };
                shadow_out[i] = p;
            }
        }
    }
}

#[test]
fn struct_assignment_reads_the_complete_rhs_before_writing() {
    let device = HeadlessDevice::new().expect("Vulkan adapter required; no silent skip");
    eprintln!("struct assignment adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let values: Vec<_> = (0..count)
            .map(|i| assignments::Pair {
                first: i as u32 + 11,
                second: i as u32 + 101,
            })
            .collect();
        let input = device.create_typed_buffer("input", &values, GpuBufferAccess::ReadOnly);
        let local = device.create_typed_buffer("local", &values, GpuBufferAccess::ReadWrite);
        let buffer = device.create_typed_buffer("buffer", &values, GpuBufferAccess::ReadWrite);
        let shadow = device.create_typed_buffer("shadow", &values, GpuBufferAccess::ReadWrite);
        device
            .dispatch_buffers(
                &assignments::run::DESCRIPTOR,
                count as u32,
                &[
                    BufferBinding::read_only(&input),
                    BufferBinding::read_write(&local),
                    BufferBinding::read_write(&buffer),
                    BufferBinding::read_write(&shadow),
                ],
            )
            .unwrap();
        let local = device.read_typed_buffer(&local).unwrap();
        let buffer = device.read_typed_buffer(&buffer).unwrap();
        let shadow = device.read_typed_buffer(&shadow).unwrap();

        // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&values);
        let twin_local = cpu.buffer(&values);
        let twin_buffer = cpu.buffer(&values);
        let twin_shadow = cpu.buffer(&values);
        assignments::run::dispatch(
            &cpu,
            count as u32,
            &twin_input,
            &twin_local,
            &twin_buffer,
            &twin_shadow,
        )
        .unwrap();
        let (twin_local, twin_buffer, twin_shadow) =
            (twin_local.read(), twin_buffer.read(), twin_shadow.read());

        for i in 0..count {
            for (label, actual, twin) in [
                ("local", &local[i], &twin_local[i]),
                ("buffer", &buffer[i], &twin_buffer[i]),
            ] {
                assert_eq!(
                    (actual.first, actual.second),
                    (values[i].second, values[i].first),
                    "{label}[{i}]"
                );
                assert_eq!(
                    (twin.first, twin.second),
                    (values[i].second, values[i].first),
                    "twin {label}[{i}]"
                );
            }
            assert_eq!(
                (shadow[i].first, shadow[i].second),
                (values[i].first, values[i].second),
                "shadow[{i}]"
            );
            assert_eq!(
                (twin_shadow[i].first, twin_shadow[i].second),
                (values[i].first, values[i].second),
                "twin shadow[{i}]"
            );
        }
    }
}

#[test]
fn struct_assignment_compiles_to_wgsl() {
    gust::slang::compile_wgsl(&assignments::run::DESCRIPTOR).unwrap();
}

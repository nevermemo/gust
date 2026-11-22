//! Two `#[gpu_module]` modules share a helper `fn`, a `const` and a `struct` through a user-side `macro_rules!` splice,
//! on the GPU and on the CPU twin (slice m11.xsplice; Gpu tier).
//!
//! `with_shared!` wraps a module and expands to `#[gust::gpu_module] mod name { <shared items> <the module's own items>
//! }`. `macro_rules!` expands before the attribute macro runs, so `#[gpu_module]` receives one ordinary inline module
//! with the shared items already inside it. The two modules below call the same helper, read the same const and use the
//! same struct; each kernel runs on the GPU and on the CPU twin against a host reference, and a non-GPU test checks
//! that each kernel's Slang text carries the shared const, struct and helper (tested, via `DESCRIPTOR.slang_source`).
//!
//! Limits of the splice. Inferred from Rust typing, not compiled here: the splice copies the *text* of the shared items
//! into each module, so `one::Acc` and `two::Acc` are two distinct Rust types with the same fields; a value of one
//! does not fit where the other is expected (E0308 mismatched types), there is no shared `impl` target outside the
//! modules, and a host helper written against one does not accept the other. The shared `fn` and `const` are a private
//! `mix` and a `FACTOR` in each module.

mod common;

use gust::{Independent, cpu::CpuDevice};
use gust_wgpu::{BufferBinding, GpuBufferAccess};

/// Wraps a module and splices the shared items in before the module's own items.
macro_rules! with_shared {
    ($vis:vis mod $name:ident { $($body:tt)* }) => {
        #[gust::gpu_module]
        $vis mod $name {
            pub const FACTOR: u32 = 3;

            pub struct Acc {
                pub total: u32,
                pub twice: u32,
            }

            fn mix(x: u32) -> u32 {
                x * FACTOR + 1
            }

            $($body)*
        }
    };
}

with_shared! {
    mod one {
        #[kernel(workgroup_size(64, 1, 1))]
        pub fn run(
            id: SV_DispatchThreadID,
            src: StructuredBuffer<u32>,
            mut out: RWStructuredBuffer<Acc>,
        ) {
            let i = id.x;
            if i < src.len() {
                let m = mix(src[i]);
                out[i] = Acc { total: m + FACTOR, twice: m * 2 };
            }
        }
    }
}

with_shared! {
    mod two {
        #[kernel(workgroup_size(32, 1, 1))]
        pub fn run(
            id: SV_DispatchThreadID,
            src: StructuredBuffer<u32>,
            mut out: RWStructuredBuffer<Acc>,
        ) {
            let i = id.x;
            if i < src.len() {
                // A different combination of the same shared items.
                let m = mix(mix(src[i]));
                out[i] = Acc { total: m, twice: FACTOR * src[i] };
            }
        }
    }
}

/// Element counts around both workgroup widths (32 and 64), so the guard matters.
const COUNTS: [usize; 4] = [1, 31, 33, 70];

fn sources(n: usize) -> Vec<u32> {
    (0..n as u32).map(|k| k * 5).collect()
}

/// Host reference, written from the shared rule `mix(x) = x * 3 + 1` and the constant 3 (not by calling the module).
fn host_mix(x: u32) -> u32 {
    x * 3 + 1
}

#[test]
fn the_shared_items_are_in_each_modules_own_slang_text() {
    // Host-side smoke test of the spliced Rust items.
    assert_eq!(one::FACTOR, 3);
    assert_eq!(two::FACTOR, 3);
    let a = one::Acc { total: 1, twice: 2 };
    let b = two::Acc { total: 1, twice: 2 };
    assert_eq!((a.total, a.twice), (b.total, b.twice));

    // Each kernel's own Slang text carries the shared const, struct and helper.
    for (module, text) in [
        ("one", one::run::DESCRIPTOR.slang_source),
        ("two", two::run::DESCRIPTOR.slang_source),
    ] {
        for name in ["FACTOR", "Acc", "mix"] {
            assert!(text.contains(name), "{module}: Slang text lacks `{name}`");
        }
    }
}

#[test]
fn two_spliced_modules_run_on_the_gpu_and_the_twin() {
    let device = common::device();
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);

    for n in COUNTS {
        let src = sources(n);
        let want_one: Vec<(u32, u32)> = src
            .iter()
            .map(|&x| (host_mix(x) + 3, host_mix(x) * 2))
            .collect();
        let want_two: Vec<(u32, u32)> = src
            .iter()
            .map(|&x| (host_mix(host_mix(x)), 3 * x))
            .collect();

        let source = device.create_typed_buffer("splice src", &src, GpuBufferAccess::ReadOnly);

        let blank_one = vec![one::Acc { total: 0, twice: 0 }; n];
        let out_one =
            device.create_typed_buffer("splice one", &blank_one, GpuBufferAccess::ReadWrite);
        device
            .dispatch_buffers(
                &one::run::DESCRIPTOR,
                n as u32,
                &[
                    BufferBinding::read_only(&source).independent_length(),
                    BufferBinding::read_write(&out_one).independent_length(),
                ],
            )
            .unwrap();
        let gpu_one: Vec<(u32, u32)> = device
            .read_typed_buffer(&out_one)
            .unwrap()
            .iter()
            .map(|a| (a.total, a.twice))
            .collect();
        assert_eq!(gpu_one, want_one, "gpu: module one, count {n}");

        let blank_two = vec![two::Acc { total: 0, twice: 0 }; n];
        let out_two =
            device.create_typed_buffer("splice two", &blank_two, GpuBufferAccess::ReadWrite);
        device
            .dispatch_buffers(
                &two::run::DESCRIPTOR,
                n as u32,
                &[
                    BufferBinding::read_only(&source).independent_length(),
                    BufferBinding::read_write(&out_two).independent_length(),
                ],
            )
            .unwrap();
        let gpu_two: Vec<(u32, u32)> = device
            .read_typed_buffer(&out_two)
            .unwrap()
            .iter()
            .map(|a| (a.total, a.twice))
            .collect();
        assert_eq!(gpu_two, want_two, "gpu: module two, count {n}");
        eprintln!(
            "splice: count={n} one_last={:?} two_last={:?}",
            gpu_one.last(),
            gpu_two.last()
        );

        // The twin: the same typed calls through `CpuDevice`; a second assertion, never GPU evidence.
        let cpu = CpuDevice::new();
        let twin_src = cpu.buffer(&src);
        let twin_one = cpu.buffer(&blank_one);
        one::run::dispatch(
            &cpu,
            n as u32,
            Independent(&twin_src),
            Independent(&twin_one),
        )
        .unwrap();
        let got: Vec<(u32, u32)> = twin_one.read().iter().map(|a| (a.total, a.twice)).collect();
        assert_eq!(got, want_one, "twin: module one, count {n}");
        let twin_two = cpu.buffer(&blank_two);
        two::run::dispatch(
            &cpu,
            n as u32,
            Independent(&twin_src),
            Independent(&twin_two),
        )
        .unwrap();
        let got: Vec<(u32, u32)> = twin_two.read().iter().map(|a| (a.total, a.twice)).collect();
        assert_eq!(got, want_two, "twin: module two, count {n}");
    }
}

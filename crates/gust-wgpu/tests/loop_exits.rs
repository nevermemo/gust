// Slice m11.loops: the workarounds for the NVIDIA float-break loop defect (SEMANTICS, "Known driver defect (m11.ld)")
// as GPU tests. The sentinel in `driver_sentinel.rs` pins the wrong values of a `break` on a float comparison; each
// form here is the advice SEMANTICS gives instead, and must give Rust's values (n == t) on every adapter, on the CPU
// twin, and on both routes (the SPIR-V route is taken under `--features spirv-route`).
//
// The inputs are the sentinel's: per invocation t in LOOP_T, `s` starts at 2^(t - 126) and is halved each step, and
// `n` is the iteration (1-based) at which `s < 1.5 * 2^-125` first holds, which is t. Start values and thresholds
// come from buffers built from bits on the host, so the values are exact whatever the compiler folds. One `#[test]`
// per form (four), each running on the adapter the process gets and printing the adapter line and the route.
#[gust::gpu]
mod loop_exits {
    // Form 1: a done flag instead of `break`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn done_flag(
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
            let mut done = false;
            for j in 0u32..255u32 {
                if !done {
                    if s < thr {
                        n = j + 1u32;
                        done = true;
                    }
                    s *= 0.5f32;
                }
            }
            out[i] = n;
        }
    }

    // Form 2: an integer exit. `while` is not in the language (GUST-E0402), so the loop keeps its constant bound
    // and breaks on an integer comparison of the exponent field `e`, which falls by one with each halving of `s`.
    // For these powers of two, `s < 1.5 * 2^-125` holds exactly when the exponent field is 2 or less.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn integer_exit(
        id: SV_DispatchThreadID,
        exponents: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < out.len() {
            let mut e = exponents[i];
            let mut n = 0u32;
            for j in 0u32..255u32 {
                if e <= 2u32 {
                    n = j + 1u32;
                    break;
                }
                e -= 1u32;
            }
            out[i] = n;
        }
    }

    // Form 3: the sentinel's `<` loop with the bound read from a buffer (255 in every element).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn bound_per_invocation(
        id: SV_DispatchThreadID,
        starts: StructuredBuffer<f32>,
        limit: StructuredBuffer<f32>,
        bounds: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < out.len() {
            let mut s = starts[i];
            let thr = limit[i];
            let bound = bounds[i];
            let mut n = 0u32;
            for j in 0u32..bound {
                if s < thr {
                    n = j + 1u32;
                    break;
                }
                s *= 0.5f32;
            }
            out[i] = n;
        }
    }

    // Form 4: the sentinel's `==` loop with a done flag.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn eq_done_flag(
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
            let mut done = false;
            for j in 0u32..255u32 {
                if !done {
                    if s == y {
                        n = j + 1u32;
                        done = true;
                    }
                    s *= 0.5f32;
                }
            }
            out[i] = n;
        }
    }
}

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// The iteration at which the exit is due, per invocation (Rust's value is n == t).
const LOOP_T: [u32; 6] = [100, 127, 128, 129, 200, 250];

/// Whether every kernel this device built so far took the SPIR-V route. `false` without the feature.
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

struct Inputs {
    starts: Vec<f32>,
    thr: Vec<f32>,
    y: Vec<f32>,
    exponents: Vec<u32>,
    bounds: Vec<u32>,
}

fn inputs() -> Inputs {
    // 2^(t - 126) from its bits: the exponent field is t - 126 + 127 = t + 1.
    let starts: Vec<f32> = LOOP_T
        .iter()
        .map(|t| f32::from_bits((t + 1) << 23))
        .collect();
    let thr = f32::from_bits((2u32 << 23) | (1 << 22)); // 1.5 * 2^-125
    let y = f32::from_bits(2u32 << 23); // 2^-125
    assert_eq!(starts[0], 2.0f32.powi(100 - 126));
    assert_eq!(thr, 1.5 * 2.0f32.powi(-125));
    assert_eq!(y, 2.0f32.powi(-125));
    let n = LOOP_T.len();
    Inputs {
        starts,
        thr: vec![thr; n],
        y: vec![y; n],
        exponents: LOOP_T.iter().map(|t| t + 1).collect(),
        bounds: vec![255; n],
    }
}

fn open(form: &str) -> (HeadlessDevice, String) {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    let adapter = format!(
        "{} (vendor {:#06x}, driver {:?} {:?}, {:?})",
        info.name, info.vendor, info.driver, info.driver_info, info.backend
    );
    eprintln!("loop exits {form}: adapter {adapter}");
    (device, adapter)
}

fn check(form: &str, device: &HeadlessDevice, adapter: &str, gpu: &[u32], cpu: &[u32]) {
    let route = if spirv_route_taken(device) {
        "SPIR-V route"
    } else {
        "WGSL route"
    };
    eprintln!("loop exits {form}: {route} on {adapter}; gpu {gpu:?}; cpu {cpu:?}");
    assert_eq!(
        cpu, LOOP_T,
        "{form}: CPU twin must give Rust's values (n == t)"
    );
    assert_eq!(
        gpu, LOOP_T,
        "{form} ({route}) on {adapter}: the form must give Rust's values (n == t); if a workaround form does not, \
         the advice in SEMANTICS (m11.ld) is wrong"
    );
}

fn blank() -> Vec<u32> {
    vec![u32::MAX; LOOP_T.len()]
}

#[test]
fn done_flag_exit_gives_rusts_values() {
    let (device, adapter) = open("done_flag");
    let inp = inputs();
    let count = LOOP_T.len() as u32;
    let starts = device.create_typed_buffer("starts", &inp.starts, GpuBufferAccess::ReadOnly);
    let thr = device.create_typed_buffer("thr", &inp.thr, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &blank(), GpuBufferAccess::ReadWrite);
    device
        .dispatch_call(&loop_exits::done_flag::call(count, &starts, &thr, &out))
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();

    let cpu = CpuDevice::new();
    let (t_starts, t_thr, t_out) = (
        cpu.buffer(&inp.starts),
        cpu.buffer(&inp.thr),
        cpu.buffer(&blank()),
    );
    loop_exits::done_flag::dispatch(&cpu, count, &t_starts, &t_thr, &t_out).unwrap();
    check("done_flag", &device, &adapter, &gpu, &t_out.read());
}

#[test]
fn integer_exit_gives_rusts_values() {
    let (device, adapter) = open("integer_exit");
    let inp = inputs();
    let count = LOOP_T.len() as u32;
    let exps = device.create_typed_buffer("exponents", &inp.exponents, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &blank(), GpuBufferAccess::ReadWrite);
    device
        .dispatch_call(&loop_exits::integer_exit::call(count, &exps, &out))
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();

    let cpu = CpuDevice::new();
    let (t_exps, t_out) = (cpu.buffer(&inp.exponents), cpu.buffer(&blank()));
    loop_exits::integer_exit::dispatch(&cpu, count, &t_exps, &t_out).unwrap();
    check("integer_exit", &device, &adapter, &gpu, &t_out.read());
}

#[test]
fn bound_per_invocation_gives_rusts_values() {
    let (device, adapter) = open("bound_per_invocation");
    let inp = inputs();
    let count = LOOP_T.len() as u32;
    let starts = device.create_typed_buffer("starts", &inp.starts, GpuBufferAccess::ReadOnly);
    let thr = device.create_typed_buffer("thr", &inp.thr, GpuBufferAccess::ReadOnly);
    let bounds = device.create_typed_buffer("bounds", &inp.bounds, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &blank(), GpuBufferAccess::ReadWrite);
    device
        .dispatch_call(&loop_exits::bound_per_invocation::call(
            count, &starts, &thr, &bounds, &out,
        ))
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();

    let cpu = CpuDevice::new();
    let (t_starts, t_thr, t_bounds, t_out) = (
        cpu.buffer(&inp.starts),
        cpu.buffer(&inp.thr),
        cpu.buffer(&inp.bounds),
        cpu.buffer(&blank()),
    );
    loop_exits::bound_per_invocation::dispatch(&cpu, count, &t_starts, &t_thr, &t_bounds, &t_out)
        .unwrap();
    check(
        "bound_per_invocation",
        &device,
        &adapter,
        &gpu,
        &t_out.read(),
    );
}

#[test]
fn eq_done_flag_exit_gives_rusts_values() {
    let (device, adapter) = open("eq_done_flag");
    let inp = inputs();
    let count = LOOP_T.len() as u32;
    let starts = device.create_typed_buffer("starts", &inp.starts, GpuBufferAccess::ReadOnly);
    let y = device.create_typed_buffer("y", &inp.y, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &blank(), GpuBufferAccess::ReadWrite);
    device
        .dispatch_call(&loop_exits::eq_done_flag::call(count, &starts, &y, &out))
        .unwrap();
    let gpu = device.read_typed_buffer(&out).unwrap();

    let cpu = CpuDevice::new();
    let (t_starts, t_y, t_out) = (
        cpu.buffer(&inp.starts),
        cpu.buffer(&inp.y),
        cpu.buffer(&blank()),
    );
    loop_exits::eq_done_flag::dispatch(&cpu, count, &t_starts, &t_y, &t_out).unwrap();
    check("eq_done_flag", &device, &adapter, &gpu, &t_out.read());
}

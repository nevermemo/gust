//! `compact` with separate keep and kids buffers (m14.lim-b, D280, check 5): a scatter that needs nine storage buffers in
//! one stage, against the same scatter with the two flags packed into one buffer (eight).
//!
//! The nine-buffer form is a TEST KERNEL of this file, not a public operation of `gust-algo` (whose `compact` takes one
//! flags buffer). The scatter is the walker example's: an element that is kept goes to slot `starts[i]`, an element
//! that has a child writes it to `starts[i] + keep`, where `starts` is the exclusive scan of `keep + kids`. Outputs are
//! sentinel-filled first and the whole buffers are compared with an independent host reference. The `twin_` test runs
//! on the CPU twin; the `gpu_` tests run on the adapter the environment selects and fail without one.

#![allow(clippy::needless_range_loop)]

use gust::cpu::CpuDevice;
use gust::{Independent, gpu_module};
use gust_wgpu::{DeviceHandle, DeviceOptions, GpuVec, HeadlessDevice};

#[gpu_module]
pub mod k {
    /// Nine storage buffers: `a`, `b`, `c`, `keep`, `kids`, `starts` and three outputs.
    #[allow(clippy::too_many_arguments)]
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn split9(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<uint>,
        b: StructuredBuffer<uint>,
        c: StructuredBuffer<uint>,
        keep: StructuredBuffer<uint>,
        kids: StructuredBuffer<uint>,
        starts: StructuredBuffer<uint>,
        mut out_a: RWStructuredBuffer<uint>,
        mut out_b: RWStructuredBuffer<uint>,
        mut out_c: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < starts.len() {
            let s = starts[i];
            let kept = keep[i];
            if kept == 1u32 {
                out_a[s] = a[i];
                out_b[s] = b[i];
                out_c[s] = c[i];
            }
            if kids[i] == 1u32 {
                let t = s + kept;
                out_a[t] = a[i] ^ 21845u32;
                out_b[t] = b[i] + 1u32;
                out_c[t] = c[i] + 2u32;
            }
        }
    }

    /// The same scatter with `keep` in bit 0 and `kids` in bit 1 of one `flags` buffer: eight storage buffers.
    #[allow(clippy::too_many_arguments)]
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn packed8(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<uint>,
        b: StructuredBuffer<uint>,
        c: StructuredBuffer<uint>,
        flags: StructuredBuffer<uint>,
        starts: StructuredBuffer<uint>,
        mut out_a: RWStructuredBuffer<uint>,
        mut out_b: RWStructuredBuffer<uint>,
        mut out_c: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < starts.len() {
            let s = starts[i];
            let f = flags[i];
            let kept = f & 1u32;
            if kept == 1u32 {
                out_a[s] = a[i];
                out_b[s] = b[i];
                out_c[s] = c[i];
            }
            if (f >> 1u32) == 1u32 {
                let t = s + kept;
                out_a[t] = a[i] ^ 21845u32;
                out_b[t] = b[i] + 1u32;
                out_c[t] = c[i] + 2u32;
            }
        }
    }
}

const N: usize = 1000;
const UNWRITTEN: u32 = 0xDEAD_BEEF;

/// The inputs and the host reference of the scatter.
struct Case {
    a: Vec<u32>,
    b: Vec<u32>,
    c: Vec<u32>,
    keep: Vec<u32>,
    kids: Vec<u32>,
    flags: Vec<u32>,
    starts: Vec<u32>,
    /// The three outputs, `2 * N` long, the slots no element reaches left as the sentinel.
    expected: [Vec<u32>; 3],
}

fn case() -> Case {
    let a: Vec<u32> = (0..N as u32).map(|i| i * 3 + 1).collect();
    let b: Vec<u32> = (0..N as u32).map(|i| i * 5 + 2).collect();
    let c: Vec<u32> = (0..N as u32).map(|i| i * 7 + 3).collect();
    let keep: Vec<u32> = (0..N as u32).map(|i| u32::from(i % 3 != 0)).collect();
    let kids: Vec<u32> = (0..N as u32).map(|i| u32::from(i % 5 == 0)).collect();
    let flags: Vec<u32> = keep.iter().zip(&kids).map(|(k, c)| k + 2 * c).collect();
    let mut starts = Vec::with_capacity(N);
    let mut total = 0;
    for i in 0..N {
        starts.push(total);
        total += keep[i] + kids[i];
    }
    let mut expected = [
        vec![UNWRITTEN; 2 * N],
        vec![UNWRITTEN; 2 * N],
        vec![UNWRITTEN; 2 * N],
    ];
    for i in 0..N {
        let s = starts[i] as usize;
        if keep[i] == 1 {
            expected[0][s] = a[i];
            expected[1][s] = b[i];
            expected[2][s] = c[i];
        }
        if kids[i] == 1 {
            let t = s + keep[i] as usize;
            expected[0][t] = a[i] ^ 21845;
            expected[1][t] = b[i] + 1;
            expected[2][t] = c[i] + 2;
        }
    }
    Case {
        a,
        b,
        c,
        keep,
        kids,
        flags,
        starts,
        expected,
    }
}

/// Which form to run.
#[derive(Clone, Copy)]
enum Form {
    Nine,
    Packed,
}

fn run_gpu(
    handle: &DeviceHandle,
    case: &Case,
    form: Form,
) -> Result<[Vec<u32>; 3], Box<dyn std::error::Error>> {
    let device: &HeadlessDevice = handle;
    let upload = |v: &[u32]| GpuVec::from_slice_on(handle, v);
    let (a, b, c) = (upload(&case.a)?, upload(&case.b)?, upload(&case.c)?);
    let starts = upload(&case.starts)?;
    let sentinel = vec![UNWRITTEN; 2 * N];
    let (out_a, out_b, out_c) = (upload(&sentinel)?, upload(&sentinel)?, upload(&sentinel)?);
    match form {
        Form::Nine => {
            let (keep, kids) = (upload(&case.keep)?, upload(&case.kids)?);
            k::split9::dispatch(
                device,
                N as u32,
                Independent(&a),
                Independent(&b),
                Independent(&c),
                Independent(&keep),
                Independent(&kids),
                Independent(&starts),
                Independent(&out_a),
                Independent(&out_b),
                Independent(&out_c),
            )?;
        }
        Form::Packed => {
            let flags = upload(&case.flags)?;
            k::packed8::dispatch(
                device,
                N as u32,
                Independent(&a),
                Independent(&b),
                Independent(&c),
                Independent(&flags),
                Independent(&starts),
                Independent(&out_a),
                Independent(&out_b),
                Independent(&out_c),
            )?;
        }
    }
    Ok([out_a.to_vec()?, out_b.to_vec()?, out_c.to_vec()?])
}

fn run_twin(case: &Case) -> Result<[Vec<u32>; 3], Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let buffer = |v: &[u32]| device.buffer(v);
    let (a, b, c) = (buffer(&case.a), buffer(&case.b), buffer(&case.c));
    let (keep, kids, starts) = (buffer(&case.keep), buffer(&case.kids), buffer(&case.starts));
    let sentinel = vec![UNWRITTEN; 2 * N];
    let (out_a, out_b, out_c) = (buffer(&sentinel), buffer(&sentinel), buffer(&sentinel));
    k::split9::dispatch(
        &device,
        N as u32,
        Independent(&a),
        Independent(&b),
        Independent(&c),
        Independent(&keep),
        Independent(&kids),
        Independent(&starts),
        Independent(&out_a),
        Independent(&out_b),
        Independent(&out_c),
    )?;
    Ok([
        out_a.read().to_vec(),
        out_b.read().to_vec(),
        out_c.read().to_vec(),
    ])
}

/// A device opened with `n` storage buffers per stage, or `None` (with the reason printed) on an adapter that offers
/// fewer (a software adapter such as lavapipe may): the test then skips by name instead of failing for a limit of the
/// machine. Any other failure to open a device is a failure.
fn raised(n: u32) -> Option<DeviceHandle> {
    match HeadlessDevice::with_options(DeviceOptions::default().with_storage_buffers(n)) {
        Ok(device) => Some(std::sync::Arc::new(device)),
        Err(gust_wgpu::Error::DeviceLimit { maximum, .. }) => {
            println!(
                "SKIPPED: this adapter offers {maximum} storage buffers per stage and the test needs {n}"
            );
            None
        }
        Err(error) => panic!("an adapter is required; this test must not skip silently: {error}"),
    }
}

#[test]
fn twin_the_nine_buffer_scatter_equals_the_reference() {
    let case = case();
    assert_eq!(run_twin(&case).unwrap(), case.expected);
}

#[test]
fn gpu_nine_buffers_on_a_raised_device_equal_the_packed_form_and_the_reference() {
    let case = case();
    let Some(handle) = raised(9) else {
        return;
    };
    let nine = run_gpu(&handle, &case, Form::Nine).unwrap();
    let packed = run_gpu(&handle, &case, Form::Packed).unwrap();
    assert_eq!(
        packed, case.expected,
        "the packed form against the reference"
    );
    assert_eq!(nine, packed, "nine buffers against the packed form");
    assert_eq!(nine, case.expected, "nine buffers against the reference");
    assert_eq!(
        nine,
        run_twin(&case).unwrap(),
        "nine buffers against the twin"
    );
}

#[test]
fn gpu_nine_buffers_on_a_default_device_are_refused_and_the_packed_form_runs() {
    let case = case();
    let handle: DeviceHandle = std::sync::Arc::new(
        HeadlessDevice::new().expect("an adapter is required; this test must not skip silently"),
    );
    let error = run_gpu(&handle, &case, Form::Nine).expect_err("nine buffers on a device of eight");
    let limit = error
        .downcast_ref::<gust::map::Error>()
        .and_then(|error| error.downcast_ref::<gust_wgpu::Error>())
        .or_else(|| error.downcast_ref::<gust_wgpu::Error>());
    match limit {
        Some(gust_wgpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) => assert_eq!(
            (*limit, *requested, *maximum),
            ("max_storage_buffers_per_shader_stage", 9, 8)
        ),
        other => panic!("expected the DeviceLimit refusal, got {other:?} from {error}"),
    }
    let packed = run_gpu(&handle, &case, Form::Packed).unwrap();
    assert_eq!(packed, case.expected);
}

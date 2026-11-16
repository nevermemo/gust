//! The CubeCL comparator (m10.b2): `cubecl-wgpu` and `cubecl-wgpu-tuned`. The same workloads written as CubeCL 0.11.4
//! kernels in Rust and run through CubeCL's wgpu runtime, on the adapter the run selected (`--adapter`).
//!
//! The runtime is not a second graphics stack of its own choosing: the bench creates the wgpu instance, adapter, device
//! and queue (the run's third wgpu device, as `wgsl-prebuilt` has its own) and hands them to CubeCL's "existing setup"
//! path (`try_init_device`), so CubeCL runs on exactly the adapter of the run. An adapter that cannot run the kernels is a
//! typed [`Refusal`]: the rows are left out with the reason (`config.backends_not_run`), never a silent other GPU.
//!
//! CubeCL is built with `default-features = false, features = ["wgpu"]`: no `spirv` feature, so its compiler emits WGSL
//! that wgpu and naga translate, the route the hand WGSL takes; no `std`/`persistence` default features, so CubeCL has no
//! autotune or kernel cache on disk, and no `ureq` (network) in the resolved graph. See `docs/BENCHMARKS.md`.
//!
//! Two rows, both outside the publishable group's kill rule:
//! - `cubecl-wgpu` (like-for-like): the default checked launch (`launch`), `CubeDim::new_1d(256)` (W9: `new_2d(16, 16)`,
//!   the hand shader's 16 by 16), line size 1, the algorithm of the hand WGSL of each workload;
//! - `cubecl-wgpu-tuned`: the same kernels through `launch_unchecked` (no bounds rewrite), and for W1 to W3 the line size
//!   4 (`Vector<f32, N>` buffers, one invocation per four elements) when the length is a multiple of four.
//!   CubeCL's autotune only applies to `Tunable` operation sets (matmul, reduce), not to a hand-launched kernel, so the
//!   tuned row has none.

use std::{cell::Cell, error::Error, fmt, rc::Rc, time::Duration};

use bytemuck::cast_slice;
use cubecl::__private::Runtime;
use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{
    AutoCompiler, RuntimeOptions, WgpuDevice, WgpuInitError, WgpuRuntime, WgpuSetup,
    try_init_device,
};
use serde_json::{Value, json};

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{self, Inputs, WORKGROUP, Workload, hist_threads, normalize_counts},
};

/// The id of the like-for-like row.
pub const LIKE_FOR_LIKE: &str = "cubecl-wgpu";
/// The id of the tuned row.
pub const TUNED: &str = "cubecl-wgpu-tuned";

/// Line size of the tuned row's vectorised kernels (W1 to W3).
const LINE_SIZE: usize = 4;

/// The adapter cannot run CubeCL's kernels of this bench. This, and only this, leaves the rows out of a run ("not run",
/// with the reason); any other error while opening the device is a failure of the run.
#[derive(Debug)]
pub struct Refusal(pub String);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for Refusal {}

/// What the capabilities the rows need are, read from an adapter: its backend, and the device limits CubeCL's launches
/// use. Pure, so the gate is tested without a GPU.
///
/// - Vulkan: the bench runs the hand-WGSL baseline on Vulkan only, and the rows must sit on the adapter of that run.
/// - 256 invocations per workgroup: every kernel but W9 launches 256-wide cubes.
/// - six storage buffers per shader stage: the widest kernel (W2, W9) binds four buffers, CubeCL adds one for the
///   launch metadata (array lengths) and wgpu reserves one for runtime-sized arrays.
pub fn check_adapter(
    name: &str,
    backend: wgpu::Backend,
    max_invocations: u32,
    max_storage_buffers: u32,
) -> Result<(), Refusal> {
    if backend != wgpu::Backend::Vulkan {
        return Err(Refusal(format!(
            "adapter {name} is on {backend:?}, not Vulkan: the CubeCL rows run beside the Vulkan hand-WGSL baseline only"
        )));
    }
    if max_invocations < WORKGROUP as u32 {
        return Err(Refusal(format!(
            "adapter {name} allows {max_invocations} invocations per workgroup, below the {WORKGROUP} of the CubeCL kernels"
        )));
    }
    if max_storage_buffers < 6 {
        return Err(Refusal(format!(
            "adapter {name} allows {max_storage_buffers} storage buffers per shader stage, below the 6 the CubeCL kernels need (four bound, one for CubeCL's launch metadata, one wgpu reserves)"
        )));
    }
    Ok(())
}

/// Whether a CubeCL initialisation error is a refusal of the adapter (a typed `Refusal`) rather than a failure of the run.
fn refusal_of(error: WgpuInitError, name: &str) -> Box<dyn Error> {
    match error {
        WgpuInitError::InvalidConfiguration { message } => Box::new(Refusal(format!(
            "CubeCL refused the setup of adapter {name}: {message}"
        ))),
        WgpuInitError::UnsupportedGraphicsApi { api } => Box::new(Refusal(format!(
            "CubeCL has no {api:?} runtime in this build (adapter {name})"
        ))),
        WgpuInitError::AdapterUnavailable { message, .. } => Box::new(Refusal(format!(
            "CubeCL found no adapter for {name}: {message}"
        ))),
        other => Box::new(other),
    }
}

/// What the run does about the CubeCL rows.
#[derive(Debug)]
pub enum Decision<T> {
    /// The runtime opened: the rows run.
    Run(T),
    /// The adapter was refused: both rows are left out, one entry each for `config.backends_not_run`.
    NotRun(Vec<Value>),
}

/// Whether the rows run, from the outcome of opening the runtime and the run's `--backends` list: a refusal leaves the rows
/// out and records the reason, unless `--backends` names one of them, which is an error; any other error propagates. Pure.
pub fn decide<T>(
    opened: Result<T, Box<dyn Error>>,
    wanted: Option<&[String]>,
) -> Result<Decision<T>, Box<dyn Error>> {
    match opened {
        Ok(opened) => Ok(Decision::Run(opened)),
        Err(error) => {
            let Some(refusal) = error.downcast_ref::<Refusal>() else {
                return Err(error);
            };
            let reason = refusal.0.clone();
            if let Some(named) = wanted.and_then(|names| {
                names
                    .iter()
                    .find(|name| name.as_str() == LIKE_FOR_LIKE || name.as_str() == TUNED)
            }) {
                return Err(format!("--backends names {named}, which cannot run: {reason}").into());
            }
            Ok(Decision::NotRun(
                [LIKE_FOR_LIKE, TUNED]
                    .into_iter()
                    .map(|id| json!({ "backend": id, "reason": reason }))
                    .collect(),
            ))
        }
    }
}

/// What a run file records about the comparator (`versions.cubecl`): the CubeCL version, the compiler it ran, and what each
/// row is, so the file says what the tuned row did.
pub fn identity() -> Value {
    json!({
        "cubecl": "0.11.4",
        "features": "default-features = false, features = [\"wgpu\"]",
        "compiler": "WGSL (no `spirv` feature): CubeCL emits WGSL, wgpu and naga translate it",
        "rows": {
            LIKE_FOR_LIKE: "default checked launch (CubeCL clamps an out-of-range buffer index to the last element), CubeDim 1d(256) (W9: 2d(16,16)), line size 1, the hand WGSL's algorithm per workload",
            TUNED: "launch_unchecked (no bounds rewrite), line size 4 (Vector<f32, 4>) for W1 to W3 when n is a multiple of 4, else line size 1; same cube dims; no autotune (it applies to Tunable operation sets, not to hand-launched kernels)",
        },
    })
}

/// The wgpu device CubeCL runs on and the client of its runtime: the run's third wgpu device on the run's adapter selection,
/// with default limits (the adapter's own buffer limits for a W7 input above 2^25 words) and `TIMESTAMP_QUERY`, as the hand
/// device is opened.
pub struct CubeclContext {
    pub client: Client,
    pub info: wgpu::AdapterInfo,
    /// The device's `max_compute_workgroups_per_dimension` (W7's grid cap).
    pub max_groups: u32,
    /// Whether the adapter has `TIMESTAMP_QUERY`, so CubeCL's profiling windows are device timestamps.
    pub timestamps: bool,
    /// Keeps the registered device alive.
    _device: WgpuDevice,
}

impl CubeclContext {
    /// Opens the runtime on the adapter whose name contains `adapter_name` (`--adapter`), or wgpu's high-performance choice.
    pub fn open_on(adapter_name: Option<&str>, adapter_limits: bool) -> BenchResult<Self> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = crate::wgsl_wgpu::WgpuContext::pick_adapter(&instance, adapter_name)?;
        let info = adapter.get_info();
        let available = adapter.limits();
        check_adapter(
            &info.name,
            info.backend,
            available.max_compute_invocations_per_workgroup,
            available.max_storage_buffers_per_shader_stage,
        )?;
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut required_limits = wgpu::Limits::default();
        if adapter_limits {
            required_limits.max_buffer_size = available.max_buffer_size;
            required_limits.max_storage_buffer_binding_size =
                available.max_storage_buffer_binding_size;
            required_limits.max_compute_workgroups_per_dimension =
                available.max_compute_workgroups_per_dimension;
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("bench CubeCL device"),
                required_features: if timestamps {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                required_limits,
                ..Default::default()
            }))?;
        let max_groups = device.limits().max_compute_workgroups_per_dimension;
        let setup = WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend: info.backend,
        };
        let options = RuntimeOptions::try_default().map_err(|e| refusal_of(e, &info.name))?;
        let device =
            try_init_device(setup, options).map_err(|error| refusal_of(error, &info.name))?;
        let client = WgpuRuntime::<AutoCompiler>::client(&device);
        Ok(Self {
            client,
            info,
            max_groups,
            timestamps,
            _device: device,
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Kernels. Each is the algorithm of the hand WGSL of the workload (bench/shaders/*.wgsl), with the same guards.
// `launch` is the like-for-like row's checked launch, `launch_unchecked` the tuned row's.
// ---------------------------------------------------------------------------------------------------------------------

/// W1, vector add: one invocation per element, the bound the output's own length.
#[cube(launch, launch_unchecked)]
fn w1_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// W1 with line size `N`: one invocation per `N` elements.
#[cube(launch_unchecked)]
fn w1_add_lines<N: Size>(a: &[Vector<f32, N>], b: &[Vector<f32, N>], out: &mut [Vector<f32, N>]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// W2, saxpy: the scalar is a one-element buffer (as in the hand WGSL), out of place.
#[allow(clippy::len_zero)] // `is_empty` is not a cube operation on a launch slice
#[cube(launch, launch_unchecked)]
fn w2_saxpy(scale: &[f32], x: &[f32], y: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() && 0usize < scale.len() {
        out[ABSOLUTE_POS] = scale[0usize] * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// W2 with line size `N`.
#[allow(clippy::len_zero)] // `is_empty` is not a cube operation on a launch slice
#[cube(launch_unchecked)]
fn w2_saxpy_lines<N: Size>(
    scale: &[f32],
    x: &[Vector<f32, N>],
    y: &[Vector<f32, N>],
    out: &mut [Vector<f32, N>],
) {
    if ABSOLUTE_POS < out.len() && 0usize < scale.len() {
        out[ABSOLUTE_POS] = Vector::new(scale[0usize]) * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// W3, Horner: the degree is the coefficient count minus one, the loop runs `degree` steps (the hand WGSL's loop is bounded
/// by 64 and leaves at the degree: the same iterations).
#[cube(launch, launch_unchecked)]
fn w3_horner(coeffs: &[f32], x: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() && ABSOLUTE_POS < x.len() && 1usize < coeffs.len() {
        let degree = coeffs.len() - 1usize;
        let v = x[ABSOLUTE_POS];
        let mut acc = coeffs[degree];
        for k in 0..degree {
            acc = acc * v + coeffs[degree - 1usize - k];
        }
        out[ABSOLUTE_POS] = acc;
    }
}

/// W3 with line size `N` (the coefficients stay scalar, broadcast to the line).
#[cube(launch_unchecked)]
fn w3_horner_lines<N: Size>(coeffs: &[f32], x: &[Vector<f32, N>], out: &mut [Vector<f32, N>]) {
    if ABSOLUTE_POS < out.len() && ABSOLUTE_POS < x.len() && 1usize < coeffs.len() {
        let degree = coeffs.len() - 1usize;
        let v = x[ABSOLUTE_POS];
        let mut acc = Vector::new(coeffs[degree]);
        for k in 0..degree {
            acc = acc * v + Vector::new(coeffs[degree - 1usize - k]);
        }
        out[ABSOLUTE_POS] = acc;
    }
}

/// W4, histogram: a workgroup owns a block of 256 * 4 contiguous keys, each lane takes four, strided by 256; the bins are
/// `u32` atomics. The keys' bound is the key buffer's length, the bin index is guarded by the bin buffer's length.
#[cube(launch, launch_unchecked)]
fn w4_histogram(keys: &[u32], bins: &[Atomic<u32>]) {
    let n = keys.len();
    let first = (ABSOLUTE_POS / 256usize) * 1024usize + ABSOLUTE_POS % 256usize;
    for k in 0..4usize {
        let index = first + k * 256usize;
        if index < n {
            let bin = keys[index] as usize;
            if bin < bins.len() {
                bins[bin].fetch_add(1u32);
            }
        }
    }
}

/// W7, first pass of the two-pass workgroup-tree sum: a grid-stride share per lane, then a tree in workgroup memory, one
/// partial per workgroup. `u32` adds wrap (as WGSL's do).
#[cube(launch, launch_unchecked)]
fn w7_blocks<F: Numeric>(x: &[F], partials: &mut [F]) {
    let n = x.len();
    let local = UNIT_POS_X as usize;
    let group = CUBE_POS_X as usize;
    let stride = CUBE_COUNT_X as usize * 256usize;
    let mut tile = Shared::<[F]>::new_slice(256usize);
    let mut acc = F::from_int(0);
    let mut i = group * 256usize + local;
    while i < n {
        acc += x[i];
        i += stride;
    }
    tile[local] = acc;
    sync_cube();
    let mut s = 128usize;
    while s > 0usize {
        if local < s {
            tile[local] = tile[local] + tile[local + s];
        }
        sync_cube();
        s /= 2usize;
    }
    if local == 0usize && group < partials.len() {
        partials[group] = tile[0usize];
    }
}

/// W7, second pass: one workgroup, each lane adds its share of the partials, then the same tree into the one output word.
#[allow(clippy::len_zero)] // `is_empty` is not a cube operation on a launch slice
#[cube(launch, launch_unchecked)]
fn w7_total<F: Numeric>(partials: &[F], out: &mut [F]) {
    let count = partials.len();
    let local = UNIT_POS_X as usize;
    let mut tile = Shared::<[F]>::new_slice(256usize);
    let mut acc = F::from_int(0);
    let mut i = local;
    while i < count {
        acc += partials[i];
        i += 256usize;
    }
    tile[local] = acc;
    sync_cube();
    let mut s = 128usize;
    while s > 0usize {
        if local < s {
            tile[local] = tile[local] + tile[local + s];
        }
        sync_cube();
        s /= 2usize;
    }
    if local == 0usize && 0usize < out.len() {
        out[0usize] = tile[0usize];
    }
}

/// W9, naive matrix product: the side `N` is the one word of `dims`, one output entry per invocation, `k` ascending.
#[allow(clippy::len_zero)] // `is_empty` is not a cube operation on a launch slice
#[cube(launch, launch_unchecked)]
fn w9_matmul(dims: &[u32], a: &[f32], b: &[f32], c: &mut [f32]) {
    if 0usize < dims.len() {
        let n = dims[0usize] as usize;
        let row = ABSOLUTE_POS_Y as usize;
        let col = ABSOLUTE_POS_X as usize;
        if row < n && col < n && row * n + col < c.len() {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The backend.
// ---------------------------------------------------------------------------------------------------------------------

struct Resident {
    workload: Workload,
    n: usize,
    out_len: usize,
    /// In binding order, the output last: `[a, b, out]` (W1), `[scale, x, y, out]` (W2), `[coeffs, x, out]` (W3),
    /// `[keys, bins]` (W4), `[x, partials, out]` (W7), `[dims, a, b, c]` (W9).
    handles: Vec<Handle>,
    /// Element counts of `handles`, as the launch declares them.
    lens: Vec<usize>,
    /// The handles (indexes) an end-to-end sample uploads from the host inputs, in the order of `input_bytes`.
    uploads: Vec<usize>,
    poison: Vec<f32>,
    /// W7: workgroups of the first pass.
    blocks: u32,
}

pub struct Cubecl {
    ctx: Rc<CubeclContext>,
    tuned: bool,
    resident: Option<Resident>,
    /// Dispatches since the bins were last cleared (a histogram accumulates).
    passes: Cell<u32>,
}

impl Cubecl {
    pub fn new(ctx: Rc<CubeclContext>, tuned: bool) -> Self {
        Self {
            ctx,
            tuned,
            resident: None,
            passes: Cell::new(0),
        }
    }

    fn resident(&self) -> &Resident {
        self.resident.as_ref().expect("prepare was called")
    }

    fn input_bytes(workload: Workload, inputs: &Inputs) -> Vec<&[u8]> {
        match workload.family() {
            "W1" | "W2" | "W9" => vec![cast_slice(&inputs.x), cast_slice(&inputs.y)],
            "W3" => vec![cast_slice(&inputs.x)],
            "W7" if workload == Workload::SumU32 => vec![cast_slice(&inputs.values)],
            "W7" => vec![cast_slice(&inputs.x)],
            _ => vec![cast_slice(&inputs.keys)],
        }
    }

    /// Whether the tuned row runs `workload` with line size `LINE_SIZE` at this length.
    fn lines(&self, workload: Workload, n: usize) -> bool {
        self.tuned && matches!(workload.family(), "W1" | "W2" | "W3") && n.is_multiple_of(LINE_SIZE)
    }

    fn write(&self, handle: &Handle, bytes: &[u8]) {
        self.ctx
            .client
            .write(handle, Bytes::from_bytes_vec(bytes.to_vec()));
    }

    fn sync(&self) -> BenchResult<()> {
        pollster::block_on(self.ctx.client.sync()).map_err(|error| format!("{error:?}"))?;
        Ok(())
    }

    /// Queues one pass of the workload (W7: its two dispatches). Not waited for.
    fn pass(&self, r: &Resident) {
        let client = &self.ctx.client;
        let h = |i: usize| r.handles[i].clone();
        let len = |i: usize| r.lens[i];
        // SAFETY (every `BufferArg::from_raw_parts` and every `launch_unchecked` below): `lens` are the exact element
        // counts of the buffers `prepare` allocated (bytes / element size), and every kernel guards its indices against
        // the lengths of its own buffers exactly as the hand WGSL does, so no launch reads or writes out of range; every
        // loop is bounded by a length or a constant. Every sample of both rows is checked against the serial reference.
        let arg = |i: usize| unsafe { BufferArg::from_raw_parts(h(i), len(i)) };
        let groups = |threads: usize| CubeCount::Static(threads.div_ceil(WORKGROUP) as u32, 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP as u32);
        match (r.workload.family(), self.tuned) {
            ("W1", false) => w1_add::launch(client, groups(r.n), dim, arg(0), arg(1), arg(2)),
            ("W1", true) if self.lines(r.workload, r.n) => unsafe {
                w1_add_lines::launch_unchecked(
                    client,
                    groups(r.n / LINE_SIZE),
                    dim,
                    LINE_SIZE,
                    BufferArg::from_raw_parts(h(0), len(0) / LINE_SIZE),
                    BufferArg::from_raw_parts(h(1), len(1) / LINE_SIZE),
                    BufferArg::from_raw_parts(h(2), len(2) / LINE_SIZE),
                );
            },
            ("W1", true) => unsafe {
                w1_add::launch_unchecked(client, groups(r.n), dim, arg(0), arg(1), arg(2));
            },
            ("W2", false) => {
                w2_saxpy::launch(client, groups(r.n), dim, arg(0), arg(1), arg(2), arg(3));
            }
            ("W2", true) if self.lines(r.workload, r.n) => unsafe {
                w2_saxpy_lines::launch_unchecked(
                    client,
                    groups(r.n / LINE_SIZE),
                    dim,
                    LINE_SIZE,
                    arg(0),
                    BufferArg::from_raw_parts(h(1), len(1) / LINE_SIZE),
                    BufferArg::from_raw_parts(h(2), len(2) / LINE_SIZE),
                    BufferArg::from_raw_parts(h(3), len(3) / LINE_SIZE),
                );
            },
            ("W2", true) => unsafe {
                w2_saxpy::launch_unchecked(
                    client,
                    groups(r.n),
                    dim,
                    arg(0),
                    arg(1),
                    arg(2),
                    arg(3),
                );
            },
            ("W3", false) => w3_horner::launch(client, groups(r.n), dim, arg(0), arg(1), arg(2)),
            ("W3", true) if self.lines(r.workload, r.n) => unsafe {
                w3_horner_lines::launch_unchecked(
                    client,
                    groups(r.n / LINE_SIZE),
                    dim,
                    LINE_SIZE,
                    arg(0),
                    BufferArg::from_raw_parts(h(1), len(1) / LINE_SIZE),
                    BufferArg::from_raw_parts(h(2), len(2) / LINE_SIZE),
                );
            },
            ("W3", true) => unsafe {
                w3_horner::launch_unchecked(client, groups(r.n), dim, arg(0), arg(1), arg(2));
            },
            ("W4", false) => {
                w4_histogram::launch(client, groups(hist_threads(r.n)), dim, arg(0), arg(1))
            }
            ("W4", true) => unsafe {
                w4_histogram::launch_unchecked(
                    client,
                    groups(hist_threads(r.n)),
                    dim,
                    arg(0),
                    arg(1),
                );
            },
            ("W7", tuned) => {
                let blocks = CubeCount::Static(r.blocks, 1, 1);
                let one = CubeCount::Static(1, 1, 1);
                match (r.workload == Workload::SumU32, tuned) {
                    (true, false) => {
                        w7_blocks::launch::<u32>(client, blocks, dim, arg(0), arg(1));
                        w7_total::launch::<u32>(client, one, dim, arg(1), arg(2));
                    }
                    (false, false) => {
                        w7_blocks::launch::<f32>(client, blocks, dim, arg(0), arg(1));
                        w7_total::launch::<f32>(client, one, dim, arg(1), arg(2));
                    }
                    (true, true) => unsafe {
                        w7_blocks::launch_unchecked::<u32>(client, blocks, dim, arg(0), arg(1));
                        w7_total::launch_unchecked::<u32>(client, one, dim, arg(1), arg(2));
                    },
                    (false, true) => unsafe {
                        w7_blocks::launch_unchecked::<f32>(client, blocks, dim, arg(0), arg(1));
                        w7_total::launch_unchecked::<f32>(client, one, dim, arg(1), arg(2));
                    },
                }
            }
            ("W9", false) => {
                let side = r.n.div_ceil(workloads::matmul::TILE) as u32;
                let dim = CubeDim::new_2d(
                    workloads::matmul::TILE as u32,
                    workloads::matmul::TILE as u32,
                );
                w9_matmul::launch(
                    client,
                    CubeCount::Static(side, side, 1),
                    dim,
                    arg(0),
                    arg(1),
                    arg(2),
                    arg(3),
                );
            }
            ("W9", true) => unsafe {
                let side = r.n.div_ceil(workloads::matmul::TILE) as u32;
                let dim = CubeDim::new_2d(
                    workloads::matmul::TILE as u32,
                    workloads::matmul::TILE as u32,
                );
                w9_matmul::launch_unchecked(
                    client,
                    CubeCount::Static(side, side, 1),
                    dim,
                    arg(0),
                    arg(1),
                    arg(2),
                    arg(3),
                );
            },
            (family, _) => unreachable!("{family} has no CubeCL kernel (`supports` leaves it out)"),
        }
    }

    /// GPU time of `repeats` back-to-back passes: CubeCL's profiling window around them, which is a device-timestamp span
    /// when the adapter has `TIMESTAMP_QUERY` (`None` otherwise: a host clock is not a GPU clock).
    fn window(&self, repeats: usize) -> BenchResult<Option<Duration>> {
        let r = self.resident();
        let client = &self.ctx.client;
        let window = client
            .profile_start()
            .map_err(|error| format!("{error:?}"))?;
        for _ in 0..repeats {
            self.pass(r);
        }
        let profile = client
            .profile_end(window)
            .map_err(|error| format!("{error:?}"))?;
        if profile.timing_method() != cubecl::profile::TimingMethod::Device {
            self.sync()?;
            return Ok(None);
        }
        Ok(pollster::block_on(profile.resolve()).map(|ticks| ticks.duration()))
    }
}

impl Backend for Cubecl {
    fn id(&self) -> &'static str {
        if self.tuned { TUNED } else { LIKE_FOR_LIKE }
    }

    fn description(&self) -> &'static str {
        if self.tuned {
            "CubeCL 0.11.4 kernels on its wgpu runtime (WGSL), tuned: launch_unchecked, line size 4 for W1 to W3 where n is a multiple of 4; W1 to W4, W7, W9"
        } else {
            "CubeCL 0.11.4 kernels on its wgpu runtime (WGSL), like for like: checked launch (out-of-range index clamped), CubeDim 256 (W9 16 by 16), line size 1; W1 to W4, W7, W9"
        }
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    /// W1 to W4, W7 and W9 (the m10.b2 scope); W5 and W6 have no CubeCL kernel here.
    fn supports(&self, workload: Workload) -> bool {
        matches!(workload.family(), "W1" | "W2" | "W3" | "W4" | "W7" | "W9")
    }

    /// The same two-pass tree as the hand WGSL (same grid, grid-stride, tree depth), so the hand tree's own bound judges it
    /// (`workloads::sum::hand_tree_bound`) instead of the serial worst case. Only for the `f32` sum, only after `prepare`.
    fn sum_tree_bound(&self, n: usize, abs_sum: f64) -> Option<f64> {
        let r = self.resident.as_ref()?;
        (r.workload == Workload::SumF32)
            .then(|| workloads::sum::hand_tree_bound(n, r.blocks, abs_sum))
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        self.resident = None;
        let n = inputs.n;
        let out_len = workload.out_len(n);
        let client = &self.ctx.client;
        let mut handles = Vec::new();
        let mut lens = Vec::new();
        let mut uploads = Vec::new();
        let dims = [n as u32];
        let fixed: Option<&[u8]> = match workload.family() {
            "W2" => Some(cast_slice(std::slice::from_ref(&inputs.scale))),
            "W3" => Some(cast_slice(&inputs.coeffs)),
            "W9" => Some(cast_slice(&dims)),
            _ => None,
        };
        if let Some(bytes) = fixed {
            handles.push(client.create_from_slice(bytes));
            lens.push(bytes.len() / 4);
        }
        for bytes in Self::input_bytes(workload, inputs) {
            uploads.push(handles.len());
            handles.push(client.create_from_slice(bytes));
            lens.push(bytes.len() / 4);
        }
        let blocks = if workload.family() == "W7" {
            let blocks = workloads::sum::hand_blocks(n, self.ctx.max_groups);
            handles.push(client.empty(blocks as usize * 4));
            lens.push(blocks as usize);
            blocks
        } else {
            0
        };
        let poison = vec![workload.poison_word(); out_len];
        handles.push(client.create_from_slice(cast_slice(&poison)));
        lens.push(out_len);
        self.sync()?;
        self.passes.set(0);
        self.resident = Some(Resident {
            workload,
            n,
            out_len,
            handles,
            lens,
            uploads,
            poison,
            blocks,
        });
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        let r = self.resident();
        let out = r.handles.last().expect("output handle");
        self.write(out, cast_slice(&r.poison));
        self.passes.set(0);
        self.sync()
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        for (index, bytes) in r.uploads.iter().zip(Self::input_bytes(r.workload, inputs)) {
            self.write(&r.handles[*index], bytes);
        }
        if r.workload.accumulates() {
            // A histogram adds into its bins: a host call starts from clean bins, so the clear is part of the sample.
            self.write(
                r.handles.last().expect("output handle"),
                cast_slice(&r.poison),
            );
        }
        self.passes.set(1);
        self.pass(r);
        self.read(out)
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        self.pass(self.resident());
        self.passes.set(self.passes.get().wrapping_add(1));
        self.sync()
    }

    /// W7: the sum of `batch` windows of one repeat each (the per-repeat spans of the hand rows and GUST); every other
    /// workload: one window around all `batch` passes.
    fn gpu_time(&mut self, batch: usize) -> BenchResult<Option<Duration>> {
        let r = self.resident();
        if !self.ctx.timestamps {
            return Ok(None);
        }
        let family = r.workload.family();
        self.passes
            .set(self.passes.get().wrapping_add(batch as u32));
        if family == "W7" {
            let mut total = Duration::ZERO;
            for _ in 0..batch {
                match self.window(1)? {
                    Some(span) => total += span,
                    None => return Ok(None),
                }
            }
            return Ok(Some(total));
        }
        self.window(batch)
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        self.read(out)
    }
}

impl Cubecl {
    /// Reads the output (waits for everything queued before it).
    fn read(&self, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        let handle = r.handles.last().expect("output handle").clone();
        let bytes = self
            .ctx
            .client
            .read_one(handle)
            .map_err(|error| format!("{error:?}"))?;
        let words = cast_slice::<u8, f32>(&bytes[..r.out_len * 4]);
        out.clear();
        out.extend_from_slice(words);
        if r.workload.accumulates() {
            normalize_counts(out, self.passes.get());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{Judging, Rounding, check_with, gen_inputs, reference, test_size};

    /// The GPU tests take turns (several Vulkan instances at once made adapter discovery fail now and then in the hand
    /// tests), and each makes its own runtime: CubeCL registers a device per call.
    fn gpu_context() -> (std::sync::MutexGuard<'static, ()>, Rc<CubeclContext>) {
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = CubeclContext::open_on(None, false)
            .expect("an adapter CubeCL can run on is needed for the CubeCL tests");
        (guard, Rc::new(ctx))
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn the_rows_have_their_ids_and_the_scope_of_the_brief() {
        assert_eq!(LIKE_FOR_LIKE, "cubecl-wgpu");
        assert_eq!(TUNED, "cubecl-wgpu-tuned");
        let identity = identity();
        assert_eq!(identity["cubecl"], "0.11.4");
        for id in [LIKE_FOR_LIKE, TUNED] {
            assert!(identity["rows"][id].is_string(), "{id} is described");
        }
        assert!(
            identity["rows"][TUNED]
                .as_str()
                .is_some_and(|text| text.contains("line size 4") && text.contains("no autotune"))
        );
    }

    #[test]
    fn the_adapter_gate_refuses_what_the_kernels_cannot_run_on() {
        let vulkan = wgpu::Backend::Vulkan;
        assert!(check_adapter("a", vulkan, 256, 8).is_ok());
        assert!(check_adapter("a", vulkan, 1024, 6).is_ok());
        for refused in [
            check_adapter("a", wgpu::Backend::Gl, 256, 8),
            check_adapter("a", vulkan, 128, 8),
            check_adapter("a", vulkan, 256, 5),
        ] {
            assert!(refused.is_err());
        }
        let reason = check_adapter("a", vulkan, 128, 8).unwrap_err().to_string();
        assert!(reason.contains("128") && reason.contains("256"), "{reason}");
    }

    /// The gate's refusal reaches `decide` as a refusal through the same `?` conversion `open_on` uses; both rows are
    /// left out with the reason, naming one with `--backends` is an error, and any other error propagates.
    #[test]
    fn a_refusal_leaves_both_rows_out_and_other_errors_fail_the_run() {
        fn open(max_invocations: u32) -> Result<u32, Box<dyn Error>> {
            check_adapter("a", wgpu::Backend::Vulkan, max_invocations, 8)?;
            Ok(1)
        }
        assert!(matches!(decide(open(256), None).unwrap(), Decision::Run(1)));
        for wanted in [None, Some(names(&["wgsl-prebuilt", "gust-gpu"]))] {
            match decide(open(64), wanted.as_deref()).unwrap() {
                Decision::NotRun(entries) => {
                    let ids = entries
                        .iter()
                        .map(|entry| entry["backend"].as_str().unwrap().to_owned())
                        .collect::<Vec<_>>();
                    assert_eq!(ids, [LIKE_FOR_LIKE, TUNED]);
                    assert!(
                        entries
                            .iter()
                            .all(|e| e["reason"].as_str().unwrap().contains("64"))
                    );
                }
                Decision::Run(_) => panic!("a refused adapter must not run the rows"),
            }
        }
        for named in [LIKE_FOR_LIKE, TUNED] {
            let wanted = names(&["gust-gpu", named]);
            let error = decide(open(64), Some(&wanted)).unwrap_err().to_string();
            assert!(
                error.contains(&format!("--backends names {named}")),
                "{error}"
            );
        }
        let other: Result<u32, Box<dyn Error>> = Err("out of device memory".into());
        assert_eq!(
            decide(other, None).unwrap_err().to_string(),
            "out of device memory"
        );
    }

    /// Every workload the rows run, end to end, as a resident dispatch and as a 16-pass batch, judged by the same checker as
    /// every other GPU backend (the poison alone fails it), on both rows. W7's `f32` sum is judged by the hand tree's depth
    /// bound (`sum_tree_bound`): the kernels are the hand WGSL's two-pass tree (same grid, grid-stride and tree), so its
    /// summation order is known, which is the case the depth-based bound is for.
    #[test]
    fn every_cubecl_workload_passes_its_bound_on_both_rows() {
        let (_gpu, ctx) = gpu_context();
        let mut ran = 0;
        for tuned in [false, true] {
            for w in Workload::ALL {
                let mut backend = Cubecl::new(Rc::clone(&ctx), tuned);
                assert_eq!(backend.id(), if tuned { TUNED } else { LIKE_FOR_LIKE });
                assert_eq!(
                    backend.supports(w),
                    matches!(w.family(), "W1" | "W2" | "W3" | "W4" | "W7" | "W9"),
                    "{}",
                    w.id()
                );
                if !backend.supports(w) {
                    continue;
                }
                ran += 1;
                let inputs = gen_inputs(w, test_size(w));
                let reference = reference(w, &inputs);
                backend.prepare(w, &inputs).unwrap();
                let judging = Judging {
                    rounding: Rounding::Either,
                    tree: backend.sum_tree_bound(inputs.n, reference.abs_sum),
                };
                assert_eq!(judging.tree.is_some(), w == Workload::SumF32);
                let mut out = Vec::new();
                backend.poison().unwrap();
                backend.read_back(&mut out).unwrap();
                assert!(
                    check_with(w, &inputs, &reference, &out, judging).is_err(),
                    "{} poison",
                    w.id()
                );
                backend.poison().unwrap();
                backend.end_to_end(&inputs, &mut out).unwrap();
                check_with(w, &inputs, &reference, &out, judging)
                    .unwrap_or_else(|e| panic!("{} tuned={tuned} end to end: {e}", w.id()));
                backend.poison().unwrap();
                backend.dispatch().unwrap();
                backend.read_back(&mut out).unwrap();
                check_with(w, &inputs, &reference, &out, judging)
                    .unwrap_or_else(|e| panic!("{} tuned={tuned} dispatch: {e}", w.id()));
                backend.poison().unwrap();
                let time = backend.gpu_time(16).unwrap();
                assert_eq!(time.is_some(), ctx.timestamps, "{} gpu time", w.id());
                backend.read_back(&mut out).unwrap();
                check_with(w, &inputs, &reference, &out, judging)
                    .unwrap_or_else(|e| panic!("{} tuned={tuned} batch: {e}", w.id()));
            }
        }
        // W1, W2, W3 (three degrees), W4 (four cases), W7 (two) and W9, on two rows.
        assert_eq!(ran, 2 * 12);
    }

    /// The tuned row's line-size-4 kernels (W1 to W3) need a length that is a multiple of four, which the odd test sizes
    /// above are not: this runs them at 4096, and checks the row takes the vector path there and the scalar one at 4103.
    #[test]
    fn the_tuned_row_runs_line_size_four_where_the_length_allows_it() {
        let (_gpu, ctx) = gpu_context();
        for w in [Workload::VectorAdd, Workload::Saxpy, Workload::Poly16] {
            let mut tuned = Cubecl::new(Rc::clone(&ctx), true);
            let plain = Cubecl::new(Rc::clone(&ctx), false);
            assert!(tuned.lines(w, 4096) && !tuned.lines(w, 4103), "{}", w.id());
            assert!(!plain.lines(w, 4096), "{}", w.id());
            let inputs = gen_inputs(w, 4096);
            let reference = reference(w, &inputs);
            tuned.prepare(w, &inputs).unwrap();
            let judging = Judging {
                rounding: Rounding::Either,
                tree: None,
            };
            let mut out = Vec::new();
            tuned.poison().unwrap();
            tuned.dispatch().unwrap();
            tuned.read_back(&mut out).unwrap();
            check_with(w, &inputs, &reference, &out, judging)
                .unwrap_or_else(|e| panic!("{} line size 4: {e}", w.id()));
        }
        // W4, W7 and W9 have no line-size variant.
        let tuned = Cubecl::new(Rc::clone(&ctx), true);
        for w in [Workload::Hist256Uniform, Workload::SumU32, Workload::MatMul] {
            assert!(!tuned.lines(w, 4096), "{}", w.id());
        }
    }

    /// The bounds policy's claim, checked without an out-of-range access: on in-range sizes the checked launch (the
    /// like-for-like row) and the unchecked one (the tuned row) give the same words for the kernels both rows run with the
    /// same operation sequence (W1, W4, the sums, W9): the clamp changes nothing in range.
    #[test]
    fn checked_and_unchecked_launches_agree_on_in_range_sizes() {
        let (_gpu, ctx) = gpu_context();
        for w in [
            Workload::VectorAdd,
            Workload::Hist256Zipf,
            Workload::Hist65536Uniform,
            Workload::SumU32,
            Workload::SumF32,
            Workload::MatMul,
        ] {
            let inputs = gen_inputs(w, test_size(w));
            let mut results = Vec::new();
            for tuned in [false, true] {
                let mut backend = Cubecl::new(Rc::clone(&ctx), tuned);
                backend.prepare(w, &inputs).unwrap();
                backend.poison().unwrap();
                backend.dispatch().unwrap();
                let mut out = Vec::new();
                backend.read_back(&mut out).unwrap();
                results.push(out.iter().map(|word| word.to_bits()).collect::<Vec<_>>());
            }
            assert_eq!(
                results[0],
                results[1],
                "{}: checked and unchecked differ",
                w.id()
            );
        }
    }
}

//! The workload set (W1 and W2 of the M10 research): inputs, the serial reference and the checker.

pub mod saxpy;
pub mod vector_add;

/// Workgroup size of every GPU kernel in this harness, GUST and hand WGSL alike.
///
/// 256 and not 64: the default wgpu limits cap a one-dimensional dispatch at 65535 workgroups, so 64-wide groups stop
/// at 4 194 240 invocations, just below 2^22. With 256 the cap is 16 776 960 (see `SIZES`).
pub const WORKGROUP: usize = 256;

/// The scalar of saxpy.
pub const SCALE: f32 = 2.5;

/// The size ladder: 2^12, 2^16, 2^20, 2^22 and the largest 256-wide one-dimensional launch the default limits allow
/// (65535 workgroups, 2^24 - 256 elements; 2^24 itself needs 65536 and is rejected). Three f32 streams of the last
/// size are 192 MiB, well past the L2 cache of the RTX 5090 (96 MB), so it is the DRAM-bound point.
pub const SIZES: [usize; 5] = [1 << 12, 1 << 16, 1 << 20, 1 << 22, 65535 * WORKGROUP];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Workload {
    VectorAdd,
    Saxpy,
}

impl Workload {
    pub const ALL: [Workload; 2] = [Workload::VectorAdd, Workload::Saxpy];

    pub fn id(self) -> &'static str {
        match self {
            Workload::VectorAdd => "W1",
            Workload::Saxpy => "W2",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Workload::VectorAdd => "vector-add",
            Workload::Saxpy => "saxpy",
        }
    }

    pub fn parse(text: &str) -> Option<Workload> {
        Workload::ALL
            .into_iter()
            .find(|w| w.id().eq_ignore_ascii_case(text) || w.name() == text)
    }

    /// Bytes moved per element by the kernel: two reads and one write of f32.
    pub fn bytes_per_element(self) -> u64 {
        12
    }
}

/// Host inputs. For vector-add `x` and `y` are the two addends and `scale` is unused.
pub struct Inputs {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub scale: f32,
}

/// Deterministic, varied, non-constant inputs in [-1, 1) so that no kernel can be folded and a wrong index shows up.
pub fn gen_inputs(n: usize) -> Inputs {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n as u64 {
        x.push(((i * 7919) % 4093) as f32 / 2048.0 - 1.0);
        y.push(((i * 104_729 + 17) % 8191) as f32 / 4096.0 - 1.0);
    }
    Inputs { x, y, scale: SCALE }
}

/// One element, the scalar reference every backend is checked against.
#[inline(always)]
pub fn element(w: Workload, scale: f32, x: f32, y: f32) -> f32 {
    match w {
        Workload::VectorAdd => vector_add::element(x, y),
        Workload::Saxpy => saxpy::element(scale, x, y),
    }
}

fn tolerance(w: Workload, scale: f32, x: f32, y: f32, want: f32) -> f32 {
    match w {
        Workload::VectorAdd => vector_add::tolerance(x, y, want),
        Workload::Saxpy => saxpy::tolerance(scale, x, y),
    }
}

/// The reference output, computed once per size by a plain serial loop.
pub fn reference(w: Workload, inputs: &Inputs) -> Vec<f32> {
    inputs
        .x
        .iter()
        .zip(&inputs.y)
        .map(|(x, y)| element(w, inputs.scale, *x, *y))
        .collect()
}

/// Checks `got` against the reference. A NaN (the poison value) never passes. Returns the first failing index and
/// the number of failing elements.
pub fn check(w: Workload, inputs: &Inputs, want: &[f32], got: &[f32]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "length {} where {} was expected",
            got.len(),
            want.len()
        ));
    }
    let mut first = None;
    let mut bad = 0usize;
    for i in 0..want.len() {
        let tol = tolerance(w, inputs.scale, inputs.x[i], inputs.y[i], want[i]);
        // Written as a positive comparison so that a NaN, which is never `<=`, counts as a failure.
        let within = (got[i] - want[i]).abs() <= tol;
        if !within {
            bad += 1;
            first.get_or_insert(i);
        }
    }
    match first {
        None => Ok(()),
        Some(i) => Err(format!(
            "{bad} of {} elements are wrong; first at {i}: got {}, want {}",
            want.len(),
            got[i],
            want[i]
        )),
    }
}

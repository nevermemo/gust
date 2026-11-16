//! The workload set (W1 to W4, W7 and W9 of the M10 plan): inputs, the serial reference and the checker.
//!
//! - W1 vector-add and W2 saxpy: memory-bandwidth bound, f32, bit-exact and two-ulp checks.
//! - W3 polynomial (Horner, degree 4, 16 and 64): compute bound, f32, checked against Higham's Horner bound.
//! - W4 histogram (`u32` atomic add, 256 and 65536 bins, uniform and Zipf keys): contended atomics, exact integers.
//! - W7 sum (`u32` wrapping and `f32`, 2^20 to 2^27 values, one output word): a reduction; `u32` exact, `f32` against
//!   the bound of the summation order the backend runs (see `sum`).
//! - W9 naive matrix product (`N x N` f32, `N` from 256 to 2048, one output entry per invocation): checked entry by
//!   entry against Higham's inner-product bound (see `matmul`).
//!
//! The output of every workload travels as `Vec<f32>`: for the `u32` workloads (the histogram, the `u32` sum) the words
//! are the integers' bit patterns (`f32::from_bits`), so the backend trait stays one type and no word is ever
//! converted as a number.

pub mod histogram;
#[cfg(not(feature = "gpu"))]
pub mod maps;
pub mod matmul;
pub mod polynomial;
pub mod saxpy;
pub mod sum;
pub mod vector_add;

/// Workgroup size of every GPU kernel in this harness, GUST and hand WGSL alike.
///
/// 256 and not 64: the default wgpu limits cap a one-dimensional dispatch at 65535 workgroups, so 64-wide groups stop
/// at 4 194 240 invocations, just below 2^22. With 256 the cap is 16 776 960 (see `SIZES`).
pub const WORKGROUP: usize = 256;

/// The scalar of saxpy.
pub const SCALE: f32 = 2.5;

/// The size ladder of W1 and W2: 2^12, 2^16, 2^20, 2^22 and the largest 256-wide one-dimensional launch the default
/// limits allow (65535 workgroups, 2^24 - 256 elements; 2^24 itself needs 65536 and is rejected). Three f32 streams of
/// the last size are 192 MiB, well past the L2 cache of the RTX 5090 (96 MB), so it is the DRAM-bound point.
pub const SIZES: [usize; 5] = [1 << 12, 1 << 16, 1 << 20, 1 << 22, 65535 * WORKGROUP];

/// W3's sizes (plan 2.1): 2^20 and 2^22 per degree.
pub const POLY_SIZES: [usize; 2] = [1 << 20, 1 << 22];

/// W4's sizes (plan 2.1): 2^20 and 2^24 keys. One invocation handles [`KEYS_PER_THREAD`] keys, so 2^24 keys are 2^22
/// invocations, inside the 65535-workgroup cap that 2^24 one-key invocations would exceed.
pub const HIST_SIZES: [usize; 2] = [1 << 20, 1 << 24];

/// W7's sizes (plan 2.1): 2^20 to 2^27 values, one power of two apart. The `f32` stream of the last size is 512 MiB, so
/// the two largest need the adapter's own storage-buffer limits (see [`needs_adapter_limits`]).
pub const SUM_SIZES: [usize; 8] = [
    1 << 20,
    1 << 21,
    1 << 22,
    1 << 23,
    1 << 24,
    1 << 25,
    1 << 26,
    1 << 27,
];

/// W9's sizes (plan 2.1): the matrix side `N`; the matrices hold `N * N` words.
pub const MATMUL_SIZES: [usize; 4] = [256, 512, 1024, 2048];

/// Whether a case needs a device opened with the adapter's own limits: the default wgpu limits cap a storage binding at
/// 128 MiB, which is 2^25 `f32` or `u32` words, so a W7 input above that cannot be bound (on GUST's `HeadlessDevice`
/// and on the hand-WGSL device alike).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn needs_adapter_limits(w: Workload, n: usize) -> bool {
    w.family() == "W7" && n > (1 << 25)
}

/// Keys one histogram invocation handles (a workgroup owns a block of `256 * 4` contiguous keys, each lane strides by 256).
pub const KEYS_PER_THREAD: usize = 4;

/// Invocations a launch of `n` keys needs.
pub fn hist_threads(n: usize) -> usize {
    n.div_ceil(KEYS_PER_THREAD)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Workload {
    VectorAdd,
    Saxpy,
    Poly4,
    Poly16,
    Poly64,
    Hist256Uniform,
    Hist256Zipf,
    Hist65536Uniform,
    Hist65536Zipf,
    SumU32,
    SumF32,
    MatMul,
}

/// What the output words are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    F32,
    U32,
}

impl Workload {
    pub const ALL: [Workload; 12] = [
        Workload::VectorAdd,
        Workload::Saxpy,
        Workload::Poly4,
        Workload::Poly16,
        Workload::Poly64,
        Workload::Hist256Uniform,
        Workload::Hist256Zipf,
        Workload::Hist65536Uniform,
        Workload::Hist65536Zipf,
        Workload::SumU32,
        Workload::SumF32,
        Workload::MatMul,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Workload::VectorAdd => "W1",
            Workload::Saxpy => "W2",
            Workload::Poly4 => "W3-d4",
            Workload::Poly16 => "W3-d16",
            Workload::Poly64 => "W3-d64",
            Workload::Hist256Uniform => "W4-256-uniform",
            Workload::Hist256Zipf => "W4-256-zipf",
            Workload::Hist65536Uniform => "W4-65536-uniform",
            Workload::Hist65536Zipf => "W4-65536-zipf",
            Workload::SumU32 => "W7-u32",
            Workload::SumF32 => "W7-f32",
            Workload::MatMul => "W9-naive",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Workload::VectorAdd => "vector-add",
            Workload::Saxpy => "saxpy",
            Workload::Poly4 => "polynomial degree 4",
            Workload::Poly16 => "polynomial degree 16",
            Workload::Poly64 => "polynomial degree 64",
            Workload::Hist256Uniform => "histogram 256 bins, uniform keys",
            Workload::Hist256Zipf => "histogram 256 bins, Zipf keys",
            Workload::Hist65536Uniform => "histogram 65536 bins, uniform keys",
            Workload::Hist65536Zipf => "histogram 65536 bins, Zipf keys",
            Workload::SumU32 => "sum u32 (wrapping)",
            Workload::SumF32 => "sum f32",
            Workload::MatMul => "matrix product, naive f32",
        }
    }

    /// One workload by id, name or lower-case id.
    pub fn parse(text: &str) -> Option<Workload> {
        Workload::ALL
            .into_iter()
            .find(|w| w.id().eq_ignore_ascii_case(text) || w.name() == text)
    }

    /// A workload list item of `--workloads`: one workload, or a family (`W3` is the three degrees, `W4` the four
    /// histograms, `W7` the two sums).
    pub fn parse_family(text: &str) -> Option<Vec<Workload>> {
        if let Some(one) = Workload::parse(text) {
            return Some(vec![one]);
        }
        let family = Workload::ALL
            .into_iter()
            .filter(|w| w.family().eq_ignore_ascii_case(text))
            .collect::<Vec<_>>();
        (!family.is_empty()).then_some(family)
    }

    /// `W1`, `W2`, `W3`, `W4`, `W7` or `W9`.
    pub fn family(self) -> &'static str {
        match self {
            Workload::VectorAdd => "W1",
            Workload::Saxpy => "W2",
            Workload::Poly4 | Workload::Poly16 | Workload::Poly64 => "W3",
            Workload::Hist256Uniform
            | Workload::Hist256Zipf
            | Workload::Hist65536Uniform
            | Workload::Hist65536Zipf => "W4",
            Workload::SumU32 | Workload::SumF32 => "W7",
            Workload::MatMul => "W9",
        }
    }

    pub fn kind(self) -> Kind {
        match self {
            Workload::SumU32 => Kind::U32,
            w if w.bins().is_some() => Kind::U32,
            _ => Kind::F32,
        }
    }

    /// Whether the output accumulates over dispatches (a histogram adds into its bins; every other workload writes its
    /// output afresh).
    pub fn accumulates(self) -> bool {
        self.bins().is_some()
    }

    /// Whether GUST and the hand-WGSL baseline run the same kernel on this workload, so that a GUST/hand pairing (the
    /// kill rule's ratios, the report's overhead rows) compares the same algorithm and the same timed work. `false` for
    /// W7: GUST runs the public `gust_algo::reduce` (v1 schedule, no GPU timestamps exposed, `u32` read-back inside the
    /// call) and the baseline runs its own tree, so no ratio between them is a GUST overhead.
    pub fn same_kernel(self) -> bool {
        self.family() != "W7"
    }

    /// The default size ladder of the workload.
    pub fn sizes(self) -> &'static [usize] {
        match self.family() {
            "W3" => &POLY_SIZES,
            "W4" => &HIST_SIZES,
            "W7" => &SUM_SIZES,
            "W9" => &MATMUL_SIZES,
            _ => &SIZES,
        }
    }

    /// Polynomial degree of a W3 workload.
    pub fn degree(self) -> Option<usize> {
        match self {
            Workload::Poly4 => Some(4),
            Workload::Poly16 => Some(16),
            Workload::Poly64 => Some(64),
            _ => None,
        }
    }

    /// Bin count of a W4 workload.
    pub fn bins(self) -> Option<usize> {
        match self {
            Workload::Hist256Uniform | Workload::Hist256Zipf => Some(256),
            Workload::Hist65536Uniform | Workload::Hist65536Zipf => Some(65536),
            _ => None,
        }
    }

    pub fn zipf(self) -> bool {
        matches!(self, Workload::Hist256Zipf | Workload::Hist65536Zipf)
    }

    /// Words of the output for `n` inputs: `n` for the element-wise workloads, the bin count for a histogram, one for a
    /// sum, `n * n` for the product of two `n x n` matrices.
    pub fn out_len(self, n: usize) -> usize {
        match self.family() {
            "W7" => 1,
            "W9" => n * n,
            _ => self.bins().unwrap_or(n),
        }
    }

    /// The word the device output is filled with before a sample (so a dispatch that does not run cannot pass): NaN
    /// for the f32 workloads, zero for the histogram (whose kernel adds into the bins, so the start must be zero) and
    /// all ones for the `u32` sum (a NaN as an f32 word, and not the sum of the reference: a test pins it).
    pub fn poison_word(self) -> f32 {
        match self {
            Workload::SumU32 => f32::from_bits(u32::MAX),
            _ => match self.kind() {
                Kind::F32 => f32::NAN,
                Kind::U32 => f32::from_bits(0),
            },
        }
    }

    /// Bytes of memory traffic per element that the achieved-bandwidth column counts. W1 and W2: two reads and a write
    /// of f32. W3: a read and a write. W4: the key read only (the atomic traffic on the bins is not counted). W7: the
    /// value read only (one word out). W9: not a per-element stream (the matrices are read many times), so none is
    /// counted.
    pub fn bytes_per_element(self) -> u64 {
        match self.family() {
            "W3" => 8,
            "W4" | "W7" => 4,
            "W9" => 0,
            _ => 12,
        }
    }

    /// Floating-point operations per element, for the GFLOP/s column (0 where it is not a flop workload). W3 counts a
    /// multiply and an add per Horner step (2 * degree). W7's `f32` sum is one add per value; W9 counts none (its `n` is
    /// the side, not an element count).
    pub fn flops_per_element(self) -> u64 {
        match self {
            Workload::Saxpy => 2,
            Workload::VectorAdd | Workload::SumF32 => 1,
            w => w.degree().map_or(0, |d| 2 * d as u64),
        }
    }
}

/// Host inputs of one (workload, size).
#[derive(Clone)]
pub struct Inputs {
    /// Elements of the workload: values for W1 to W3 and W7, keys for W4, the matrix side for W9.
    pub n: usize,
    /// W1: the first addend; W2, W3 and W7 (`f32`): the values; W9: the matrix `A`, row-major. Empty for W4 and the
    /// `u32` sum.
    pub x: Vec<f32>,
    /// W1 and W2: the second operand; W9: the matrix `B`. Empty for W3, W4 and W7.
    pub y: Vec<f32>,
    /// W2's scalar.
    pub scale: f32,
    /// W3's `degree + 1` coefficients, lowest degree first.
    pub coeffs: Vec<f32>,
    /// W4's keys, every one below `bins`.
    pub keys: Vec<u32>,
    /// W4's bin count.
    pub bins: usize,
    /// The `u32` sum's addends.
    pub values: Vec<u32>,
}

/// The reference of one (workload, size): the expected output words and, for W3, the exact values and per-element
/// bounds the check uses (computed once per case, never per sample).
#[derive(Default)]
pub struct Reference {
    pub want: Vec<f32>,
    /// W3, W7 (`f32`) and W9: the exact value in f64 (the polynomial at each `x`, the sum, each entry of the product).
    pub exact: Vec<f64>,
    /// W3, W7 (`f32`) and W9: the derived worst-case error bound at each output word (see `polynomial::exact_and_bound`,
    /// `sum::serial_bound` and `matmul::exact_and_bound`).
    pub bound: Vec<f64>,
    /// W7 (`f32`): an upper bound on `sum |x_i|`, the quantity every sum bound scales with (the schedule-specific tree
    /// bound of a backend, `Backend::sum_tree_bound`, takes it).
    pub abs_sum: f64,
    /// W7 (`f32`): the most the f64 reference can be off, `(n - 1) 2^-53 S`; added to the bound of every backend.
    pub ref_error: f64,
    /// W3, W7 (`f32`, the serial worst case) and W9: the same bounds as `bound` for a backend whose adds and multiplies
    /// may round to either neighbour (a GPU, `u = 2^-23`; see [`Rounding`]). `bound` is the round-to-nearest one (`u = 2^-24`).
    pub bound_gpu: Vec<f64>,
}

/// The unit roundoff a backend's float operations are modelled with (`docs/SEMANTICS.md` section 4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rounding {
    /// IEEE round to nearest: `u = 2^-24`. The CPU backends (host arithmetic) and `gust-twin`.
    #[default]
    Nearest,
    /// A GPU: an add or a multiply may round to either neighbour, within 1 ULP, `u = 2^-23` (the `C_GPU` of
    /// `gust_algo::bound`). Every GPU backend: GUST, its SPIR-V route and the hand WGSL.
    Either,
}

impl Rounding {
    /// The unit roundoff `u` of this rounding: `2^-24` (nearest) or `2^-23` (either neighbour).
    pub fn unit(self) -> f64 {
        match self {
            Rounding::Nearest => polynomial::U32,
            Rounding::Either => polynomial::U32_GPU,
        }
    }

    /// The rounding of a backend of this kind.
    pub fn of(kind: crate::backend::Kind) -> Rounding {
        match kind {
            crate::backend::Kind::Gpu => Rounding::Either,
            crate::backend::Kind::Cpu => Rounding::Nearest,
        }
    }
}

/// How one backend's result is judged: its rounding, and for the `f32` sum its own tree bound when it has one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Judging {
    pub rounding: Rounding,
    /// W7 (`f32`): the backend's worst-case error for the order it runs (`Backend::sum_tree_bound`); replaces the serial
    /// worst case.
    pub tree: Option<f64>,
}

/// A splitmix64 finaliser: a well-mixed word from a counter, the only generator of the key streams.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic, varied, non-constant inputs so that no kernel can be folded and a wrong index shows up. The f32
/// values are in [-1, 1); the keys are uniform or Zipf (exponent 1, bin 0 the hottest) over the workload's bins.
pub fn gen_inputs(w: Workload, n: usize) -> Inputs {
    let mut inputs = Inputs {
        n,
        x: Vec::new(),
        y: Vec::new(),
        scale: SCALE,
        coeffs: Vec::new(),
        keys: Vec::new(),
        bins: 0,
        values: Vec::new(),
    };
    match w {
        Workload::SumU32 => {
            inputs.values = sum::u32_values(n);
            return inputs;
        }
        Workload::SumF32 => {
            inputs.x = sum::f32_values(n);
            return inputs;
        }
        Workload::MatMul => {
            (inputs.x, inputs.y) = matmul::matrices(n);
            return inputs;
        }
        _ => {}
    }
    if let Some(bins) = w.bins() {
        inputs.bins = bins;
        inputs.keys = if w.zipf() {
            histogram::zipf_keys(n, bins)
        } else {
            (0..n as u64)
                .map(|i| (mix(i) >> 32) as u32 % bins as u32)
                .collect()
        };
        return inputs;
    }
    inputs.x = Vec::with_capacity(n);
    for i in 0..n as u64 {
        inputs.x.push(((i * 7919) % 4093) as f32 / 2048.0 - 1.0);
    }
    if let Some(degree) = w.degree() {
        inputs.coeffs = polynomial::coefficients(degree);
        return inputs;
    }
    inputs.y = Vec::with_capacity(n);
    for i in 0..n as u64 {
        inputs
            .y
            .push(((i * 104_729 + 17) % 8191) as f32 / 4096.0 - 1.0);
    }
    inputs
}

/// One element of W1 or W2, the scalar reference every backend is checked against.
#[inline(always)]
pub fn element(w: Workload, scale: f32, x: f32, y: f32) -> f32 {
    match w {
        Workload::VectorAdd => vector_add::element(x, y),
        _ => saxpy::element(scale, x, y),
    }
}

/// The reference of a case, computed once per size by plain serial loops.
pub fn reference(w: Workload, inputs: &Inputs) -> Reference {
    match w.family() {
        "W1" | "W2" => Reference {
            want: inputs
                .x
                .iter()
                .zip(&inputs.y)
                .map(|(x, y)| element(w, inputs.scale, *x, *y))
                .collect(),
            ..Reference::default()
        },
        "W3" => {
            let (exact, bound, bound_gpu) = polynomial::exact_and_bound(&inputs.coeffs, &inputs.x);
            Reference {
                want: exact.iter().map(|v| *v as f32).collect(),
                exact,
                bound,
                bound_gpu,
                ..Reference::default()
            }
        }
        "W7" if w == Workload::SumU32 => Reference {
            want: vec![f32::from_bits(sum::wrapping_sum(&inputs.values))],
            ..Reference::default()
        },
        "W7" => {
            let n = inputs.x.len();
            let (exact, abs) = sum::f64_sums(&inputs.x);
            let abs_sum = sum::abs_sum_upper(n, abs);
            Reference {
                want: vec![exact as f32],
                exact: vec![exact],
                bound: vec![sum::serial_bound(n, abs_sum, Rounding::Nearest)],
                bound_gpu: vec![sum::serial_bound(n, abs_sum, Rounding::Either)],
                abs_sum,
                ref_error: sum::reference_error(n, abs_sum),
            }
        }
        "W9" => {
            let (exact, bound, bound_gpu) = matmul::exact_and_bound(inputs.n, &inputs.x, &inputs.y);
            Reference {
                want: exact.iter().map(|v| *v as f32).collect(),
                exact,
                bound,
                bound_gpu,
                ..Reference::default()
            }
        }
        _ => Reference {
            want: histogram::counts(&inputs.keys, inputs.bins)
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            ..Reference::default()
        },
    }
}

fn tolerance(w: Workload, scale: f32, x: f32, y: f32, want: f32) -> f32 {
    match w {
        Workload::VectorAdd => vector_add::tolerance(x, y, want),
        _ => saxpy::tolerance(scale, x, y),
    }
}

/// Checks `got` against the reference. A NaN (the poison value) never passes. Returns the first failing index and
/// the number of failing elements.
#[cfg(test)]
pub fn check(
    w: Workload,
    inputs: &Inputs,
    reference: &Reference,
    got: &[f32],
) -> Result<(), String> {
    check_with(w, inputs, reference, got, Judging::default())
}

/// [`check`] for a result of a backend of the given rounding (a GPU's is [`Rounding::Either`]).
#[cfg_attr(not(any(test, feature = "profile-counts")), allow(dead_code))]
pub fn check_on(
    w: Workload,
    inputs: &Inputs,
    reference: &Reference,
    got: &[f32],
    rounding: Rounding,
) -> Result<(), String> {
    check_with(
        w,
        inputs,
        reference,
        got,
        Judging {
            rounding,
            tree: None,
        },
    )
}

/// [`check`] with the backend's rounding and, for the `f32` sum, the bound of its summation order: `judging.tree` is the
/// backend's own worst-case error for the order it runs (`Backend::sum_tree_bound`), which replaces the serial worst case
/// that bounds every order. The tree is ignored by every other workload. The f64 reference's own error is added in
/// both cases. W3, W9 and the serial worst case use the bound of the backend's rounding.
pub fn check_with(
    w: Workload,
    inputs: &Inputs,
    reference: &Reference,
    got: &[f32],
    judging: Judging,
) -> Result<(), String> {
    let bounds = match judging.rounding {
        Rounding::Nearest => &reference.bound,
        Rounding::Either => &reference.bound_gpu,
    };
    let want = &reference.want;
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
        let within = match w.family() {
            // Exact integers: the words are compared as bits, never as numbers.
            "W4" => got[i].to_bits() == want[i].to_bits(),
            "W7" if w == Workload::SumU32 => got[i].to_bits() == want[i].to_bits(),
            // The bound of the order of summation that ran, and the f64 reference's own error.
            "W7" => {
                let limit = judging.tree.unwrap_or(bounds[i]) + reference.ref_error;
                (f64::from(got[i]) - reference.exact[i]).abs() <= limit
            }
            // The derived Horner bound (W3) and inner-product bound (W9) against the f64 value. A positive comparison,
            // so a NaN counts as a failure.
            "W3" | "W9" => (f64::from(got[i]) - reference.exact[i]).abs() <= bounds[i],
            _ => {
                let tol = tolerance(w, inputs.scale, inputs.x[i], inputs.y[i], want[i]);
                (got[i] - want[i]).abs() <= tol
            }
        };
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
            describe(w, got[i]),
            describe(w, want[i])
        )),
    }
}

fn describe(w: Workload, word: f32) -> String {
    match w.kind() {
        Kind::F32 => word.to_string(),
        Kind::U32 => word.to_bits().to_string(),
    }
}

/// A histogram output after `passes` dispatches that each added into the same bins: every count divided by `passes`.
/// A count that is not a multiple of `passes` cannot be the sum of identical passes, so it becomes `u32::MAX` (never
/// the reference's value for a case below 2^32 keys) and fails the check, instead of passing as its remainder would.
pub fn normalize_counts(words: &mut [f32], passes: u32) {
    if passes <= 1 {
        return;
    }
    for word in words {
        let count = word.to_bits();
        *word = f32::from_bits(if count % passes == 0 {
            count / passes
        } else {
            u32::MAX
        });
    }
}

/// The descriptor of the GUST kernel a workload runs (the three W3 degrees share one kernel, the four W4 histograms
/// another).
#[cfg(feature = "gpu")]
pub fn descriptor(w: Workload) -> &'static gust::KernelDescriptor {
    match w.family() {
        "W1" => &vector_add::kernel::add::DESCRIPTOR,
        "W2" => &saxpy::kernel::saxpy::DESCRIPTOR,
        "W3" => &polynomial::kernel::horner::DESCRIPTOR,
        "W7" => algo_descriptor(w),
        "W9" => &matmul::kernel::matmul::DESCRIPTOR,
        _ => &histogram::kernel::histogram::DESCRIPTOR,
    }
}

/// The first-pass kernel of `gust-algo`'s public sum for a W7 workload (the one whose route events a W7 cell is judged
/// on). Only a build with the `algo` feature has it.
#[cfg(all(feature = "gpu", feature = "algo"))]
fn algo_descriptor(w: Workload) -> &'static gust::KernelDescriptor {
    match w {
        Workload::SumU32 => &gust_algo::kernels::sum_wrapping_u32_block::DESCRIPTOR,
        _ => &gust_algo::kernels::sum_f32_block::DESCRIPTOR,
    }
}

/// A build without the `algo` feature (`gust-bench-pre`) has no GUST W7 backend and never asks.
#[cfg(all(feature = "gpu", not(feature = "algo")))]
fn algo_descriptor(w: Workload) -> &'static gust::KernelDescriptor {
    unreachable!("{} needs the `algo` feature", w.id())
}

/// Every kernel of this harness's own set, once (W7 runs `gust-algo`'s kernels, which this crate does not own).
#[cfg(feature = "gpu")]
pub fn all_descriptors() -> [&'static gust::KernelDescriptor; 5] {
    [
        descriptor(Workload::VectorAdd),
        descriptor(Workload::Saxpy),
        descriptor(Workload::Poly4),
        descriptor(Workload::Hist256Uniform),
        descriptor(Workload::MatMul),
    ]
}

/// The size of a unit-test case of a workload: small, off the workgroup boundaries (so the guards run), and for W9 a
/// matrix side that is not a multiple of the 16-wide workgroup (the naive product is cubic).
#[cfg(test)]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn test_size(w: Workload) -> usize {
    match w.family() {
        "W4" => 5000,
        "W9" => 45,
        _ => 4096 + 7,
    }
}

/// `u32` counts as output words.
pub fn counts_as_words(counts: Vec<u32>) -> Vec<f32> {
    counts.into_iter().map(f32::from_bits).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(w: Workload, n: usize) -> (Inputs, Reference) {
        let inputs = gen_inputs(w, n);
        let reference = reference(w, &inputs);
        (inputs, reference)
    }

    #[test]
    fn families_parse_to_their_members() {
        assert_eq!(
            Workload::parse_family("W1"),
            Some(vec![Workload::VectorAdd])
        );
        assert_eq!(
            Workload::parse_family("W3"),
            Some(vec![Workload::Poly4, Workload::Poly16, Workload::Poly64])
        );
        assert_eq!(Workload::parse_family("w4").map(|v| v.len()), Some(4));
        assert_eq!(
            Workload::parse_family("W3-d16"),
            Some(vec![Workload::Poly16])
        );
        assert_eq!(Workload::parse_family("W5"), None);
        let ids = Workload::ALL.map(Workload::id);
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "ids are unique");
    }

    #[test]
    fn the_plan_sizes_and_output_lengths() {
        assert_eq!(Workload::Poly64.sizes(), [1 << 20, 1 << 22]);
        assert_eq!(Workload::Hist256Zipf.sizes(), [1 << 20, 1 << 24]);
        assert_eq!(Workload::Hist65536Uniform.out_len(1 << 20), 65536);
        assert_eq!(Workload::Poly4.out_len(100), 100);
        // 2^24 keys fit the default dispatch cap only because a thread takes four of them.
        assert!(hist_threads(1 << 24).div_ceil(WORKGROUP) <= 65535);
        assert!((1usize << 24).div_ceil(WORKGROUP) > 65535);
    }

    #[test]
    fn w1_and_w2_references_pass_and_a_perturbed_output_fails() {
        for w in [Workload::VectorAdd, Workload::Saxpy] {
            let (inputs, reference) = case(w, 4096);
            check(w, &inputs, &reference, &reference.want).unwrap();
            let mut bad = reference.want.clone();
            bad[17] += 1.0;
            let error = check(w, &inputs, &reference, &bad).unwrap_err();
            assert!(error.starts_with("1 of 4096 elements are wrong; first at 17"));
            bad[17] = f32::NAN;
            assert!(check(w, &inputs, &reference, &bad).is_err());
        }
    }

    #[test]
    fn histogram_is_exact_and_one_off_fails() {
        for w in [
            Workload::Hist256Uniform,
            Workload::Hist256Zipf,
            Workload::Hist65536Uniform,
            Workload::Hist65536Zipf,
        ] {
            let (inputs, reference) = case(w, 1 << 16);
            assert_eq!(inputs.keys.len(), 1 << 16);
            assert!(inputs.keys.iter().all(|k| (*k as usize) < inputs.bins));
            let total: u64 = reference.want.iter().map(|c| u64::from(c.to_bits())).sum();
            assert_eq!(total, 1 << 16, "{}: every key is counted once", w.id());
            check(w, &inputs, &reference, &reference.want).unwrap();
            // One count too many, one too few, two bins swapped, the zero (poison) output.
            let mut bad = reference.want.clone();
            bad[3] = f32::from_bits(bad[3].to_bits() + 1);
            assert!(check(w, &inputs, &reference, &bad).is_err());
            let mut bad = reference.want.clone();
            let hot = bad
                .iter()
                .position(|c| c.to_bits() > 0)
                .expect("a non-empty bin");
            bad[hot] = f32::from_bits(bad[hot].to_bits() - 1);
            assert!(check(w, &inputs, &reference, &bad).is_err());
            let zero = vec![f32::from_bits(0); inputs.bins];
            assert!(check(w, &inputs, &reference, &zero).is_err());
            assert!(check(w, &inputs, &reference, &reference.want[1..]).is_err());
        }
    }

    #[test]
    fn zipf_keys_are_skewed_toward_bin_zero_and_uniform_keys_are_not() {
        let zipf = reference(
            Workload::Hist256Zipf,
            &gen_inputs(Workload::Hist256Zipf, 1 << 18),
        );
        let uniform = reference(
            Workload::Hist256Uniform,
            &gen_inputs(Workload::Hist256Uniform, 1 << 18),
        );
        let count = |r: &Reference, bin: usize| r.want[bin].to_bits();
        // Zipf with exponent 1 over 256 bins: bin 0 has 1 / H_256 (about 16 %) of the keys, bin 255 a few hundredths.
        assert!(count(&zipf, 0) > (1 << 18) / 10, "{}", count(&zipf, 0));
        assert!(count(&zipf, 0) > 20 * count(&zipf, 255));
        let expected = (1u32 << 18) / 256;
        for bin in 0..256 {
            let c = count(&uniform, bin);
            assert!(c > expected / 2 && c < expected * 2, "bin {bin}: {c}");
        }
    }

    #[test]
    fn w7_and_w9_have_their_sizes_output_lengths_and_poison() {
        assert_eq!(Workload::SumF32.sizes().len(), 8);
        assert_eq!(Workload::SumU32.sizes()[0], 1 << 20);
        assert_eq!(*Workload::SumU32.sizes().last().unwrap(), 1 << 27);
        assert_eq!(Workload::MatMul.sizes(), [256, 512, 1024, 2048]);
        assert_eq!(Workload::SumF32.out_len(1 << 20), 1);
        assert_eq!(Workload::MatMul.out_len(256), 256 * 256);
        assert_eq!(
            Workload::parse_family("W7"),
            Some(vec![Workload::SumU32, Workload::SumF32])
        );
        assert_eq!(Workload::parse_family("W9"), Some(vec![Workload::MatMul]));
        assert_eq!(Workload::SumU32.kind(), Kind::U32);
        assert_eq!(Workload::SumF32.kind(), Kind::F32);
        // Only a histogram accumulates (the `u32` sum writes its word afresh).
        let accumulating = Workload::ALL.into_iter().filter(|w| w.accumulates());
        assert_eq!(accumulating.count(), 4);
        // The sizes that need the adapter's own limits: W7 inputs above 2^25 words, nothing else.
        assert!(!needs_adapter_limits(Workload::SumF32, 1 << 25));
        assert!(needs_adapter_limits(Workload::SumF32, 1 << 26));
        assert!(needs_adapter_limits(Workload::SumU32, 1 << 27));
        assert!(!needs_adapter_limits(Workload::MatMul, 2048));
        assert!(!needs_adapter_limits(Workload::VectorAdd, SIZES[4]));
        // The `u32` sum's poison is a NaN word that is no sum of a case (the reference is far from all ones) and fails.
        for n in [1usize, 2, 1 << 12] {
            let (inputs, reference) = case(Workload::SumU32, n);
            let poison = Workload::SumU32.poison_word();
            assert!(poison.is_nan());
            assert_ne!(reference.want[0].to_bits(), poison.to_bits());
            assert!(check(Workload::SumU32, &inputs, &reference, &[poison]).is_err());
            assert!(check(Workload::SumU32, &inputs, &reference, &reference.want).is_ok());
        }
    }

    #[test]
    fn normalizing_a_multi_pass_histogram_never_turns_a_wrong_count_into_a_right_one() {
        let mut words = vec![f32::from_bits(32), f32::from_bits(16), f32::from_bits(5)];
        normalize_counts(&mut words, 16);
        assert_eq!(words[0].to_bits(), 2);
        assert_eq!(words[1].to_bits(), 1);
        assert_eq!(words[2].to_bits(), u32::MAX, "5 is not a multiple of 16");
        let mut one = vec![f32::from_bits(5)];
        normalize_counts(&mut one, 1);
        assert_eq!(one[0].to_bits(), 5);
    }
}

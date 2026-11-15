//! The workload set (W1 to W4 of the M10 plan): inputs, the serial reference and the checker.
//!
//! - W1 vector-add and W2 saxpy: memory-bandwidth bound, f32, bit-exact and two-ulp checks.
//! - W3 polynomial (Horner, degree 4, 16 and 64): compute bound, f32, checked against Higham's Horner bound.
//! - W4 histogram (`u32` atomic add, 256 and 65536 bins, uniform and Zipf keys): contended atomics, exact integers.
//!
//! The output of every workload travels as `Vec<f32>`: for the `u32` histogram the words are the counts' bit patterns
//! (`f32::from_bits`), so the backend trait stays one type and no word is ever converted as a number.

pub mod histogram;
#[cfg(not(feature = "gpu"))]
pub mod maps;
pub mod polynomial;
pub mod saxpy;
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
}

/// What the output words are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    F32,
    U32,
}

impl Workload {
    pub const ALL: [Workload; 9] = [
        Workload::VectorAdd,
        Workload::Saxpy,
        Workload::Poly4,
        Workload::Poly16,
        Workload::Poly64,
        Workload::Hist256Uniform,
        Workload::Hist256Zipf,
        Workload::Hist65536Uniform,
        Workload::Hist65536Zipf,
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
        }
    }

    /// One workload by id, name or lower-case id.
    pub fn parse(text: &str) -> Option<Workload> {
        Workload::ALL
            .into_iter()
            .find(|w| w.id().eq_ignore_ascii_case(text) || w.name() == text)
    }

    /// A workload list item of `--workloads`: one workload, or a family (`W3` is the three degrees, `W4` the four
    /// histograms).
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

    /// `W1`, `W2`, `W3` or `W4`.
    pub fn family(self) -> &'static str {
        match self {
            Workload::VectorAdd => "W1",
            Workload::Saxpy => "W2",
            Workload::Poly4 | Workload::Poly16 | Workload::Poly64 => "W3",
            _ => "W4",
        }
    }

    pub fn kind(self) -> Kind {
        match self.family() {
            "W4" => Kind::U32,
            _ => Kind::F32,
        }
    }

    /// The default size ladder of the workload.
    pub fn sizes(self) -> &'static [usize] {
        match self.family() {
            "W3" => &POLY_SIZES,
            "W4" => &HIST_SIZES,
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

    /// Words of the output for `n` inputs: `n` for the element-wise workloads, the bin count for a histogram.
    pub fn out_len(self, n: usize) -> usize {
        self.bins().unwrap_or(n)
    }

    /// The word the device output is filled with before a sample (so a dispatch that does not run cannot pass): NaN
    /// for the f32 workloads, zero for the histogram (whose kernel adds into the bins, so the start must be zero).
    pub fn poison_word(self) -> f32 {
        match self.kind() {
            Kind::F32 => f32::NAN,
            Kind::U32 => f32::from_bits(0),
        }
    }

    /// Bytes of memory traffic per element that the achieved-bandwidth column counts. W1 and W2: two reads and a write
    /// of f32. W3: a read and a write. W4: the key read only (the atomic traffic on the bins is not counted).
    pub fn bytes_per_element(self) -> u64 {
        match self.family() {
            "W3" => 8,
            "W4" => 4,
            _ => 12,
        }
    }

    /// Floating-point operations per element, for the GFLOP/s column (0 where it is not a flop workload). W3 counts a
    /// multiply and an add per Horner step (2 * degree).
    pub fn flops_per_element(self) -> u64 {
        match self {
            Workload::Saxpy => 2,
            Workload::VectorAdd => 1,
            w => w.degree().map_or(0, |d| 2 * d as u64),
        }
    }
}

/// Host inputs of one (workload, size).
#[derive(Clone)]
pub struct Inputs {
    /// Elements of the workload: values for W1 to W3, keys for W4.
    pub n: usize,
    /// W1: the first addend; W2 and W3: the values. Empty for W4.
    pub x: Vec<f32>,
    /// W1 and W2: the second operand. Empty for W3 and W4.
    pub y: Vec<f32>,
    /// W2's scalar.
    pub scale: f32,
    /// W3's `degree + 1` coefficients, lowest degree first.
    pub coeffs: Vec<f32>,
    /// W4's keys, every one below `bins`.
    pub keys: Vec<u32>,
    /// W4's bin count.
    pub bins: usize,
}

/// The reference of one (workload, size): the expected output words and, for W3, the exact values and per-element
/// bounds the check uses (computed once per case, never per sample).
pub struct Reference {
    pub want: Vec<f32>,
    /// W3: the exact value of the polynomial at each `x`, in f64.
    pub exact: Vec<f64>,
    /// W3: the derived worst-case error bound at each `x` (see [`polynomial::bound`]).
    pub bound: Vec<f64>,
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
    };
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
            exact: Vec::new(),
            bound: Vec::new(),
        },
        "W3" => {
            let (exact, bound) = polynomial::exact_and_bound(&inputs.coeffs, &inputs.x);
            Reference {
                want: exact.iter().map(|v| *v as f32).collect(),
                exact,
                bound,
            }
        }
        _ => Reference {
            want: histogram::counts(&inputs.keys, inputs.bins)
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            exact: Vec::new(),
            bound: Vec::new(),
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
pub fn check(
    w: Workload,
    inputs: &Inputs,
    reference: &Reference,
    got: &[f32],
) -> Result<(), String> {
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
            // The derived Horner bound against the f64 value. A positive comparison, so a NaN counts as a failure.
            "W3" => (f64::from(got[i]) - reference.exact[i]).abs() <= reference.bound[i],
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
        _ => &histogram::kernel::histogram::DESCRIPTOR,
    }
}

/// Every kernel of the set, once.
#[cfg(feature = "gpu")]
pub fn all_descriptors() -> [&'static gust::KernelDescriptor; 4] {
    [
        descriptor(Workload::VectorAdd),
        descriptor(Workload::Saxpy),
        descriptor(Workload::Poly4),
        descriptor(Workload::Hist256Uniform),
    ]
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

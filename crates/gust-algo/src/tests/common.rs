//! Inputs, references and the backend-generic case tables shared by the twin and the GPU tests.

use crate::{
    bound::{self, C_CPU, C_GPU},
    reduce::Element,
};

/// splitmix64: a small deterministic generator, so no dependency and no flaky input.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(super) fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    /// Uniform in `[0, 1)` on the lattice of multiples of `2^-24`.
    pub(super) fn unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }
}

/// The input shapes of the float sums of plan S7.7 (the adversarial one is [`tie_ladder`]).
#[derive(Clone, Copy, Debug)]
pub(super) enum Shape {
    /// Uniform in `[0, 1)`.
    Uniform,
    /// Uniform magnitudes in `[0, 1)` with alternating signs: the result is far smaller than `S`.
    Alternating,
}

pub(super) fn f32_input(shape: Shape, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    match shape {
        Shape::Uniform => (0..n).map(|_| rng.unit_f32()).collect(),
        Shape::Alternating => (0..n)
            .map(|index| {
                let value = rng.unit_f32();
                if index % 2 == 0 { value } else { -value }
            })
            .collect(),
    }
}

/// The input that makes round-to-nearest-even lose as much as the schedule lets it, for `n = 2^24`:
/// block 0 of level 1 holds `1.0` in the first term of every lane and `2^-24` in the other 255 (each addition to `1.0`
/// is an exact tie, rounds to even and is lost); every later block holds `2^-32` (a lane sums them exactly to `2^-24`).
/// Level 2 then sees `1.0` from block 0 and `2^-24` from the 255 other blocks in every lane, and loses those the same
/// way. The computed sum is `256.0` and the exact one `256 + 130 560 * 2^-24`, about 0.0078: the error is 0.67 of
/// the CPU bound of the schedule.
pub(super) fn tie_ladder(n: usize) -> Vec<f32> {
    const TIE: f32 = 1.0 / 16_777_216.0;
    const TINY: f32 = 1.0 / 4_294_967_296.0;
    (0..n)
        .map(|index| {
            if index >= 65_536 {
                TINY
            } else if index < 256 {
                1.0
            } else {
                TIE
            }
        })
        .collect()
}

/// The independent reference of an integer sum: exact, through `i128`.
pub(super) fn exact_sum<T: Copy + Into<i128>>(values: &[T]) -> i128 {
    values.iter().map(|value| (*value).into()).sum()
}

/// What one float-sum check measured, for the logs the S7.7 probe reuses.
#[derive(Debug)]
pub(super) struct FloatCheck {
    pub(super) distance: f64,
    pub(super) bound: f64,
}

impl FloatCheck {
    pub(super) fn ratio(&self) -> f64 {
        if self.bound == 0.0 {
            0.0
        } else {
            self.distance / self.bound
        }
    }
}

/// Assert `actual` is within the per-level bound (plus the `f64` reference's own error) of the `f64` reference sum.
#[track_caller]
pub(super) fn assert_f32_sum(label: &str, values: &[f32], actual: f32, c: f64) -> FloatCheck {
    let n = u32::try_from(values.len()).unwrap();
    let reference = bound::sum_f64(values);
    let s = bound::abs_sum(values);
    let limit = bound::f32_sum_bound(n, c, s) + bound::f64_reference_error(n, s);
    let distance = (f64::from(actual) - reference).abs();
    assert!(
        distance <= limit,
        "{label}: n = {n}, |{actual} - {reference}| = {distance:e} exceeds the bound {limit:e} (ratio {})",
        distance / limit
    );
    FloatCheck {
        distance,
        bound: limit,
    }
}

/// A backend the case tables run on: the CPU twins or a GPU device.
pub(super) trait Backend {
    /// The constant of the backend's per-addition error: `C_CPU` or `C_GPU`.
    const C: f64;
    fn sum<T: Element>(&self, values: &[T]) -> T;
    fn min<T: Element>(&self, values: &[T]) -> Option<T>;
    fn max<T: Element>(&self, values: &[T]) -> Option<T>;
    /// `reduce::sum_wrapping`: the sum of `values` modulo 2^32.
    fn sum_wrapping(&self, values: &[u32]) -> u32;
    /// `scan::inclusive`, read back to the host.
    fn inclusive(&self, values: &[u32]) -> Vec<u32>;
    /// `scan::exclusive`, read back to the host.
    fn exclusive(&self, values: &[u32]) -> Vec<u32>;
}

pub(super) const CPU_C: f64 = C_CPU;
pub(super) const GPU_C: f64 = C_GPU;

/// Lengths around every boundary of the schedule: the terminal kernel (256), one block (65 536), one lane's chain
/// (256 * 256), a length that is no multiple of anything, and a million.
pub(super) const LENGTHS: [usize; 17] = [
    1, 2, 3, 255, 256, 257, 511, 1_000, 4_097, 65_535, 65_536, 65_537, 131_072, 131_073, 200_003,
    999_983, 1_000_003,
];

/// `u32` and `i32` sums, minima and maxima over [`LENGTHS`], each against the independent reference. The data are
/// chosen so the identities matter: the minimum of a vector whose smallest element is large, the maximum of a vector
/// whose largest element is small or negative, and the extremes themselves.
pub(super) fn integer_cases<B: Backend>(backend: &B, lengths: &[usize]) {
    for &n in lengths {
        let mut rng = Rng::new(0x1234 + n as u64);
        // u32 sum: small enough that no partial of any grouping overflows.
        let cap = u64::from(u32::MAX) / n as u64;
        let values: Vec<u32> = (0..n).map(|_| rng.below(cap.max(1)) as u32).collect();
        assert_eq!(
            i128::from(backend.sum(&values)),
            exact_sum(&values),
            "u32 sum, n = {n}"
        );
        // i32 sum, both signs.
        let cap = i64::from(i32::MAX) / n as i64;
        let values: Vec<i32> = (0..n)
            .map(|_| (rng.below(2 * cap as u64 + 1) as i64 - cap) as i32)
            .collect();
        assert_eq!(
            i128::from(backend.sum(&values)),
            exact_sum(&values),
            "i32 sum, n = {n}"
        );

        // u32 min with a floor far above the identity of max, max with a ceiling far below the identity of min.
        let high: Vec<u32> = (0..n)
            .map(|_| 1_000 + rng.below(4_000_000_000) as u32)
            .collect();
        assert_eq!(
            backend.min(&high),
            high.iter().copied().min(),
            "u32 min, n = {n}"
        );
        assert_eq!(
            backend.max(&high),
            high.iter().copied().max(),
            "u32 max, n = {n}"
        );
        let low: Vec<u32> = (0..n).map(|_| 1 + rng.below(1_000) as u32).collect();
        assert_eq!(
            backend.max(&low),
            low.iter().copied().max(),
            "u32 max low, n = {n}"
        );
        assert_eq!(
            backend.min(&low),
            low.iter().copied().min(),
            "u32 min low, n = {n}"
        );

        // i32: all positive (min must not return a padding identity of 0 or i32::MIN), all negative (max likewise).
        let positive: Vec<i32> = (0..n)
            .map(|_| 1 + rng.below(2_000_000_000) as i32)
            .collect();
        assert_eq!(
            backend.min(&positive),
            positive.iter().copied().min(),
            "i32 min positive, n = {n}"
        );
        assert_eq!(
            backend.max(&positive),
            positive.iter().copied().max(),
            "i32 max positive, n = {n}"
        );
        let negative: Vec<i32> = positive.iter().map(|value| -value).collect();
        assert_eq!(
            backend.max(&negative),
            negative.iter().copied().max(),
            "i32 max negative, n = {n}"
        );
        assert_eq!(
            backend.min(&negative),
            negative.iter().copied().min(),
            "i32 min negative, n = {n}"
        );

        // The extremes are results too, not only identities.
        assert_eq!(
            backend.min(&vec![u32::MAX; n]),
            Some(u32::MAX),
            "u32 min of all MAX, n = {n}"
        );
        assert_eq!(
            backend.max(&vec![0u32; n]),
            Some(0),
            "u32 max of all 0, n = {n}"
        );
        assert_eq!(
            backend.min(&vec![i32::MAX; n]),
            Some(i32::MAX),
            "i32 min of all MAX, n = {n}"
        );
        assert_eq!(
            backend.max(&vec![i32::MIN; n]),
            Some(i32::MIN),
            "i32 max of all MIN, n = {n}"
        );
        assert_eq!(backend.sum(&vec![0u32; n]), 0, "u32 sum of zeros, n = {n}");
    }
}

/// `f32` minima and maxima (exact), over the same lengths: positive-only and negative-only data so a wrong identity
/// shows, and the extreme finite values.
pub(super) fn float_min_max_cases<B: Backend>(backend: &B, lengths: &[usize]) {
    let fold_min = |values: &[f32]| values.iter().copied().fold(f32::INFINITY, f32::min);
    let fold_max = |values: &[f32]| values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for &n in lengths {
        let mut rng = Rng::new(0x5678 + n as u64);
        let positive: Vec<f32> = (0..n).map(|_| 1.0 + 1_000.0 * rng.unit_f32()).collect();
        assert_eq!(
            backend.min(&positive),
            Some(fold_min(&positive)),
            "f32 min positive, n = {n}"
        );
        assert_eq!(
            backend.max(&positive),
            Some(fold_max(&positive)),
            "f32 max positive, n = {n}"
        );
        let negative: Vec<f32> = positive.iter().map(|value| -value).collect();
        assert_eq!(
            backend.min(&negative),
            Some(fold_min(&negative)),
            "f32 min negative, n = {n}"
        );
        assert_eq!(
            backend.max(&negative),
            Some(fold_max(&negative)),
            "f32 max negative, n = {n}"
        );
        assert_eq!(
            backend.min(&vec![f32::MAX; n]),
            Some(f32::MAX),
            "f32 min of all MAX, n = {n}"
        );
        assert_eq!(
            backend.max(&vec![f32::MIN; n]),
            Some(f32::MIN),
            "f32 max of all MIN, n = {n}"
        );
        // Infinities are inside the domain: a lane without an element starts from `input[0]`, not from `f32::MAX`.
        assert_eq!(
            backend.min(&vec![f32::INFINITY; n]),
            Some(f32::INFINITY),
            "f32 min of all +inf, n = {n}"
        );
        assert_eq!(
            backend.max(&vec![f32::NEG_INFINITY; n]),
            Some(f32::NEG_INFINITY),
            "f32 max of all -inf, n = {n}"
        );
        if n == 1 {
            continue;
        }
        let mut with_infinities = positive.clone();
        with_infinities[n - 1] = f32::NEG_INFINITY;
        with_infinities[0] = f32::INFINITY;
        assert_eq!(
            backend.min(&with_infinities),
            Some(f32::NEG_INFINITY),
            "f32 min with -inf last, n = {n}"
        );
        assert_eq!(
            backend.max(&with_infinities),
            Some(f32::INFINITY),
            "f32 max with +inf first, n = {n}"
        );
    }
}

/// The empty input (section 5.1): the identity for `sum`, `None` (`Err(Empty)` on the device) for `min` and `max`.
pub(super) fn empty_cases<B: Backend>(backend: &B) {
    assert_eq!(backend.sum::<u32>(&[]), 0);
    assert_eq!(backend.sum::<i32>(&[]), 0);
    assert_eq!(backend.sum::<f32>(&[]).to_bits(), 0.0f32.to_bits());
    assert_eq!(backend.min::<u32>(&[]), None);
    assert_eq!(backend.max::<u32>(&[]), None);
    assert_eq!(backend.min::<i32>(&[]), None);
    assert_eq!(backend.max::<i32>(&[]), None);
    assert_eq!(backend.min::<f32>(&[]), None);
    assert_eq!(backend.max::<f32>(&[]), None);
}

/// The float sums of S7.7's shapes at length `n` against the `f64` reference and the per-level bound; returns the
/// checks (distance and bound) in the order uniform, alternating.
pub(super) fn float_sum_cases<B: Backend>(backend: &B, n: usize) -> Vec<(Shape, FloatCheck)> {
    [Shape::Uniform, Shape::Alternating]
        .into_iter()
        .map(|shape| {
            let values = f32_input(shape, n, 0xABCD + n as u64);
            let actual = backend.sum(&values);
            let check = assert_f32_sum(&format!("{shape:?}"), &values, actual, B::C);
            (shape, check)
        })
        .collect()
}

/// The independent reference of a scan: a sequential `u64` accumulation, which must stay inside `u32` (the domain of
/// plain `+`: no inclusive prefix passes `u32::MAX`).
pub(super) fn reference_scan(values: &[u32], exclusive: bool) -> Vec<u32> {
    let mut running = 0u64;
    values
        .iter()
        .map(|value| {
            let before = running;
            running += u64::from(*value);
            assert!(
                running <= u64::from(u32::MAX),
                "the case leaves the domain of the scan"
            );
            u32::try_from(if exclusive { before } else { running }).unwrap()
        })
        .collect()
}

/// Lengths around every boundary of the scan schedule: empty, one element, one chunk (256), the second chunk, the 256
/// chunks one level holds (65 536) and the first length past them (257 totals), 65 793 (258 totals), a length that is
/// no multiple of anything, and a million.
pub(super) const SCAN_LENGTHS: [usize; 18] = [
    0, 1, 2, 3, 255, 256, 257, 511, 512, 513, 4_097, 65_535, 65_536, 65_537, 65_793, 131_073,
    999_983, 1_000_003,
];

/// `scan::inclusive` and `scan::exclusive` over `lengths`, each against the independent reference: all ones, random
/// values that keep every prefix inside `u32`, values that sit exactly on the edge of the domain (a total of
/// `u32::MAX` made of a single element at the first, the last and the chunk-boundary positions), and the relation of
/// the two scans.
pub(super) fn scan_cases<B: Backend>(backend: &B, lengths: &[usize]) {
    for &n in lengths {
        let ones = vec![1u32; n];
        let counted: Vec<u32> = (1..=n as u32).collect();
        assert_eq!(backend.inclusive(&ones), counted, "ones inclusive, n = {n}");
        let shifted: Vec<u32> = (0..n as u32).collect();
        assert_eq!(backend.exclusive(&ones), shifted, "ones exclusive, n = {n}");

        let mut rng = Rng::new(0x5CA7 + n as u64);
        let cap = (u64::from(u32::MAX) / (n as u64).max(1)).max(1);
        let values: Vec<u32> = (0..n).map(|_| rng.below(cap) as u32).collect();
        let inclusive = backend.inclusive(&values);
        let exclusive = backend.exclusive(&values);
        assert_eq!(
            inclusive,
            reference_scan(&values, false),
            "random inclusive, n = {n}"
        );
        assert_eq!(
            exclusive,
            reference_scan(&values, true),
            "random exclusive, n = {n}"
        );
        // The two scans differ by exactly the element, and the exclusive one starts at zero.
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                exclusive[index] + value,
                inclusive[index],
                "exclusive + element = inclusive, n = {n}, index {index}"
            );
        }
        assert_eq!(exclusive.first().copied().unwrap_or(0), 0, "n = {n}");

        // A single element equal to u32::MAX: every prefix from it on is exactly u32::MAX.
        for position in [0, 255, 256, 257, n.saturating_sub(1)] {
            if position >= n {
                continue;
            }
            let mut edge = vec![0u32; n];
            edge[position] = u32::MAX;
            assert_eq!(
                backend.inclusive(&edge),
                reference_scan(&edge, false),
                "edge inclusive, n = {n}, position {position}"
            );
            assert_eq!(
                backend.exclusive(&edge),
                reference_scan(&edge, true),
                "edge exclusive, n = {n}, position {position}"
            );
        }
    }
}

/// The two deepest scans, on a schedule of five dispatches (`2^24`: the second level holds exactly 256 totals) and of
/// seven (`2^24 + 1`: 257 totals, so a third level and the add-back through it): all ones and random values below 255,
/// both scans, each against the independent reference.
pub(super) fn deep_scan_cases<B: Backend>(backend: &B) {
    for n in [1usize << 24, (1 << 24) + 1] {
        let ones = vec![1u32; n];
        let inclusive = backend.inclusive(&ones);
        assert!(
            inclusive
                .iter()
                .enumerate()
                .all(|(index, value)| *value as usize == index + 1),
            "ones inclusive, n = {n}"
        );
        let exclusive = backend.exclusive(&ones);
        assert!(
            exclusive
                .iter()
                .enumerate()
                .all(|(index, value)| *value as usize == index),
            "ones exclusive, n = {n}"
        );
        let mut rng = Rng::new(0xDEE9 + n as u64);
        let values: Vec<u32> = (0..n).map(|_| rng.below(255) as u32).collect();
        assert_eq!(
            backend.inclusive(&values),
            reference_scan(&values, false),
            "random inclusive, n = {n}"
        );
        assert_eq!(
            backend.exclusive(&values),
            reference_scan(&values, true),
            "random exclusive, n = {n}"
        );
    }
}

/// `reduce::sum_wrapping` over `lengths`: the independent reference is the exact `i128` sum reduced modulo 2^32. Full
/// range values (the total passes `u32::MAX` for every `n >= 2` that is not tiny, which the case asserts), all
/// `u32::MAX`, and the pair that wraps to zero. The empty input is `0`.
pub(super) fn wrapping_cases<B: Backend>(backend: &B, lengths: &[usize]) {
    let modulo = |total: i128| (total.rem_euclid(1i128 << 32)) as u32;
    assert_eq!(backend.sum_wrapping(&[]), 0);
    assert_eq!(backend.sum_wrapping(&[u32::MAX, 1]), 0);
    assert_eq!(backend.sum_wrapping(&[u32::MAX; 3]), u32::MAX - 2);
    for &n in lengths {
        let mut rng = Rng::new(0xA11CE + n as u64);
        let values: Vec<u32> = (0..n).map(|_| rng.next_u64() as u32).collect();
        let exact = exact_sum(&values);
        if n >= 1_000 {
            assert!(exact > i128::from(u32::MAX), "the case must pass u32::MAX");
        }
        assert_eq!(
            backend.sum_wrapping(&values),
            modulo(exact),
            "wrapping sum of random full-range values, n = {n}"
        );
        let maxima = vec![u32::MAX; n];
        assert_eq!(
            backend.sum_wrapping(&maxima),
            modulo(exact_sum(&maxima)),
            "wrapping sum of u32::MAX, n = {n}"
        );
        // Inside the domain the wrapping sum is the plain one.
        let cap = u64::from(u32::MAX) / n as u64;
        let small: Vec<u32> = (0..n).map(|_| rng.below(cap.max(1)) as u32).collect();
        assert_eq!(
            i128::from(backend.sum_wrapping(&small)),
            exact_sum(&small),
            "wrapping sum inside the domain, n = {n}"
        );
        assert_eq!(backend.sum_wrapping(&small), backend.sum(&small), "n = {n}");
    }
}

//! Reading the per-program result files, comparing the three outputs, and the discard classes of the report.
//!
//! The three parties: the interpreter (`interp`, the independent host reference), the twin (the shadow body compiled
//! by rustc and run on `CpuDevice`) and the GPU. The twin must equal the interpreter bit for bit, for every type, `f32`
//! included (R1-4). The GPU is held to the interpreter's oracle: integers and booleans bit for bit, an `f32` finite and
//! inside its `[lo, hi]` interval, which is a comparison by value (`-0.0` lies in `[0, 0]`; a GPU may return either zero).

use std::path::Path;

use crate::ast::Ty;
use crate::generator::DIFFTEST_FORMAT;
use crate::json::{self, Json};
use crate::runsrc::Planned;

/// The four discard classes of R1-4 and the generator's two extra reasons, in report order. The `domain` class is the
/// integer labels (`overflow`, `division-by-zero`, `min-over-minus-one`, `shift-amount`, `index`) plus `float-domain`
/// plus `static`.
pub const CLASSES: [&str; 8] = [
    "domain",
    "comparison-overlap",
    "unboundable",
    "float-construct-not-in-v1",
    "no-input-read",
    "no-aggregate",
    "no-call",
    "no-match",
];

/// The class of a discard reason (a `Violation::label`, `static`, `no-input-read`, `no-aggregate` or `no-call`).
pub fn discard_class(reason: &str) -> &'static str {
    match reason {
        "overflow" | "division-by-zero" | "min-over-minus-one" | "shift-amount" | "index"
        | "float-domain" | "static" => "domain",
        "comparison-overlap" => "comparison-overlap",
        "unboundable" => "unboundable",
        "float-construct-not-in-v1" => "float-construct-not-in-v1",
        "no-input-read" => "no-input-read",
        "no-aggregate" => "no-aggregate",
        "no-call" => "no-call",
        "no-match" => "no-match",
        // A label this table does not know is a new violation: it must be classified here on purpose.
        other => panic!("discard reason {other:?} has no class"),
    }
}

/// Whether `bits` is a finite `f32` inside `[lo, hi]` (by value: a negative zero is inside `[0, 0]`).
pub fn in_interval(bits: u32, (lo, hi): (f64, f64)) -> bool {
    let value = f64::from(f32::from_bits(bits));
    value.is_finite() && lo <= value && value <= hi
}

/// An output element as text: the value, and for an `f32` also its bits.
pub fn element_text(ty: Ty, bits: u32) -> String {
    match ty {
        Ty::F32 => format!("{:?} (0x{bits:08x})", f32::from_bits(bits)),
        Ty::I32 => format!("{}", bits as i32),
        Ty::Bool | Ty::U32 => format!("{bits}"),
        Ty::Vec(..) | Ty::Mat(_) | Ty::Struct(_) | Ty::Enum(_) => {
            unreachable!("the output buffer is scalar")
        }
    }
}

/// One program's result file, parsed.
#[derive(Clone, Debug)]
pub struct Raw {
    pub program: String,
    pub seed: u64,
    pub attempt: u64,
    pub index: usize,
    pub count: usize,
    pub out_ty: String,
    pub mode: String,
    pub adapter: Option<String>,
    /// The capabilities the run crate was built for (`GUST_PROFILE`), as the program itself reports them (`gust::capability::enabled`).
    pub capabilities: Vec<String>,
    pub interp: Vec<u32>,
    pub twin: Result<Vec<u32>, String>,
    /// `None` when the GPU was not used (`--cpu-only`).
    pub gpu: Option<Result<Vec<u32>, String>>,
    pub wgsl_validated: bool,
    pub twin_ns: u64,
    pub gpu_first_ns: u64,
    pub gpu_warm_ns: u64,
    /// The `k_px` outputs, present when the program ran as `pixel` too.
    pub pixel: Option<PixelRaw>,
}

/// The pixel kernel's outputs of one program (`--shapes grid,pixel`).
#[derive(Clone, Debug)]
pub struct PixelRaw {
    pub twin: Result<Vec<u32>, String>,
    /// `None` when the GPU was not used (`--cpu-only`).
    pub gpu: Option<Result<Vec<u32>, String>>,
    pub wgsl_validated: bool,
    pub twin_ns: u64,
    pub gpu_first_ns: u64,
    pub gpu_warm_ns: u64,
}

fn field<'a>(json: &'a Json, key: &str) -> Result<&'a Json, String> {
    json.get(key)
        .ok_or_else(|| format!("missing field {key:?}"))
}

fn number(json: &Json, key: &str) -> Result<u64, String> {
    field(json, key)?
        .as_u64()
        .ok_or_else(|| format!("field {key:?} is not a non-negative integer"))
}

fn text(json: &Json, key: &str) -> Result<String, String> {
    field(json, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("field {key:?} is not a string"))
}

/// One party's output: the words (with the element count checked) or the error message.
fn party(
    json: &Json,
    words_key: &str,
    error_key: &str,
    count: usize,
) -> Result<Result<Vec<u32>, String>, String> {
    let words = field(json, words_key)?;
    let error = field(json, error_key)?;
    match (words.is_null(), error.as_str()) {
        (false, None) if error.is_null() => {
            let words = words
                .as_words()
                .ok_or_else(|| format!("field {words_key:?} is not an array of u32"))?;
            if words.len() != count {
                return Err(format!(
                    "field {words_key:?} holds {} elements, the element count is {count}",
                    words.len()
                ));
            }
            Ok(Ok(words))
        }
        (true, Some(message)) => Ok(Err(message.to_owned())),
        _ => Err(format!(
            "exactly one of {words_key:?} and {error_key:?} must be set"
        )),
    }
}

/// Parses one result file. A file that does not hold the element count and the interpreter and twin outputs (or the
/// twin's error) is an error: the harness never classifies a half-written result.
pub fn parse_result(file_text: &str) -> Result<Raw, String> {
    let json = json::parse(file_text)?;
    if field(&json, "end")? != &Json::Bool(true) {
        return Err("the file is not complete".into());
    }
    // A file of another format was written by another build of the run program (the result fields change with the
    // format), so it is refused by its stamp and not by whichever field happens to be missing first.
    let format = number(&json, "format")?;
    if format != u64::from(DIFFTEST_FORMAT) {
        return Err(format!(
            "the file has format {format}, this harness reads format {DIFFTEST_FORMAT} (a stale result directory, or a run program built by another version)"
        ));
    }
    let count = usize::try_from(number(&json, "count")?).map_err(|e| e.to_string())?;
    let interp = field(&json, "interp")?
        .as_words()
        .ok_or("field \"interp\" is not an array of u32")?;
    if interp.len() != count {
        return Err(format!(
            "field \"interp\" holds {} elements, the element count is {count}",
            interp.len()
        ));
    }
    let twin = party(&json, "twin", "twin_error", count)?;
    let gpu = if field(&json, "gpu")?.is_null() && field(&json, "gpu_error")?.is_null() {
        None
    } else {
        Some(party(&json, "gpu", "gpu_error", count)?)
    };
    let mode = text(&json, "mode")?;
    if (mode == "cpu-only") != gpu.is_none() {
        return Err(format!(
            "mode {mode:?} does not match the GPU fields (a gpu-mode file must hold a GPU output or a GPU error)"
        ));
    }
    let pixel = match json.get("pixel_status") {
        None => None,
        Some(status) if status.as_str() == Some("run") => {
            let gpu = if field(&json, "pixel_gpu")?.is_null()
                && field(&json, "pixel_gpu_error")?.is_null()
            {
                None
            } else {
                Some(party(&json, "pixel_gpu", "pixel_gpu_error", count)?)
            };
            if (mode == "cpu-only") != gpu.is_none() {
                return Err(format!("mode {mode:?} does not match the pixel GPU fields"));
            }
            Some(PixelRaw {
                twin: party(&json, "pixel_twin", "pixel_twin_error", count)?,
                gpu,
                wgsl_validated: field(&json, "pixel_wgsl_validated")? == &Json::Bool(true),
                twin_ns: number(&json, "pixel_twin_ns")?,
                gpu_first_ns: number(&json, "pixel_gpu_first_ns")?,
                gpu_warm_ns: number(&json, "pixel_gpu_warm_ns")?,
            })
        }
        Some(_) => return Err("field \"pixel_status\" is not \"run\"".into()),
    };
    Ok(Raw {
        program: text(&json, "program")?,
        seed: number(&json, "seed")?,
        attempt: number(&json, "attempt")?,
        index: usize::try_from(number(&json, "index")?).map_err(|e| e.to_string())?,
        count,
        out_ty: text(&json, "out_ty")?,
        mode,
        adapter: field(&json, "adapter")?.as_str().map(str::to_owned),
        capabilities: field(&json, "capabilities")?
            .as_array()
            .and_then(|names| {
                names
                    .iter()
                    .map(|name| name.as_str().map(str::to_owned))
                    .collect()
            })
            .ok_or("field \"capabilities\" is not an array of text")?,
        interp,
        twin,
        gpu,
        wgsl_validated: field(&json, "wgsl_validated")? == &Json::Bool(true),
        twin_ns: number(&json, "twin_ns")?,
        gpu_first_ns: number(&json, "gpu_first_ns")?,
        gpu_warm_ns: number(&json, "gpu_warm_ns")?,
        pixel,
    })
}

/// Reads the result files of a run: exactly `expected.len()` files named `<program>.json`, one per planned program,
/// each complete. Anything else (a missing file, an extra file, a half-written one) is an error.
pub fn read_results(dir: &Path, expected: &[Planned]) -> Result<Vec<Raw>, String> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".json") {
            names.push(name);
        }
    }
    names.sort();
    if names.len() != expected.len() {
        return Err(format!(
            "expected exactly {} result files, found {}",
            expected.len(),
            names.len()
        ));
    }
    let mut raws = Vec::with_capacity(expected.len());
    for planned in expected {
        let program = crate::emit::module_name(&planned.program);
        let name = format!("{program}.json");
        if !names.contains(&name) {
            return Err(format!("the result file {name} is missing"));
        }
        let path = dir.join(&name);
        let content =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let raw = parse_result(&content).map_err(|e| format!("{name}: {e}"))?;
        if raw.program != program
            || raw.seed != planned.program.seed
            || raw.attempt != planned.program.attempt
            || raw.index != planned.index
            || raw.count != planned.program.count as usize
        {
            return Err(format!("{name}: the file describes a different program"));
        }
        raws.push(raw);
    }
    Ok(raws)
}

/// One disagreement, or one failure to produce an output, with enough to reproduce it.
#[derive(Clone, Debug, PartialEq)]
pub struct Mismatch {
    pub program: String,
    pub seed: u64,
    pub index: usize,
    pub attempt: u64,
    /// The shape that disagreed: `grid` or `pixel`. `pixel-vs-grid` is a `pixel` finding.
    pub shape: &'static str,
    /// `transport` (the file's interpreter output is not the harness's: a harness bug), `twin-error`,
    /// `twin-vs-interpreter`, `gpu-error`, `gpu-vs-interpreter` (integers and booleans), `gpu-outside-interval`
    /// (`f32`), `gpu-not-validated`, and for `pixel` also `pixel-vs-grid` (an integer or boolean pixel GPU output differs
    /// from the grid GPU output in a bit; an `f32` one is judged by its interval like `grid`'s, and a bit difference
    /// between the shapes is only an observation, see [`float_comparison`]).
    pub kind: &'static str,
    /// How many elements differ (0 for an error).
    pub elements: usize,
    pub first_element: Option<usize>,
    pub message: String,
}

impl Mismatch {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("program", Json::str(&self.program)),
            ("seed", Json::uint(self.seed)),
            ("index", Json::uint(self.index as u64)),
            ("attempt", Json::uint(self.attempt)),
            ("shape", Json::str(self.shape)),
            ("kind", Json::str(self.kind)),
            ("elements_differing", Json::uint(self.elements as u64)),
            (
                "first_element",
                self.first_element
                    .map_or(Json::Null, |e| Json::uint(e as u64)),
            ),
            ("message", Json::str(&self.message)),
        ])
    }
}

/// Compares the outputs of one program: the grid kernel's three, and with `--shapes grid,pixel` the pixel kernel's two
/// (twin and GPU) and the pixel GPU output against the grid GPU output. `cpu_only` says whether a GPU output is expected
/// at all.
pub fn classify(planned: &Planned, raw: &Raw, cpu_only: bool) -> Vec<Mismatch> {
    let mut found = Vec::new();
    if raw.interp != planned.expected {
        let first = raw
            .interp
            .iter()
            .zip(&planned.expected)
            .position(|(a, b)| a != b);
        found.push(mismatch(
            planned,
            raw,
            "grid",
            "transport",
            0,
            first,
            "the interpreter output in the file is not the one the harness computed".into(),
        ));
    }
    let grid = Party {
        shape: "grid",
        twin: &raw.twin,
        gpu: raw.gpu.as_ref(),
        wgsl_validated: raw.wgsl_validated,
    };
    judge(planned, raw, &grid, cpu_only, &mut found);
    if let Some(pixel) = &raw.pixel {
        let party = Party {
            shape: "pixel",
            twin: &pixel.twin,
            gpu: pixel.gpu.as_ref(),
            wgsl_validated: pixel.wgsl_validated,
        };
        judge(planned, raw, &party, cpu_only, &mut found);
        // Integers and booleans hold the same bits in both shapes. An `f32` does not have to: a device may fuse `a * b + c`
        // and round `x / y` within 2.5 ulp differently per stage (SEMANTICS section 4, D128), so each shape is judged
        // against the interval above and the difference between the two is an observation (`float_comparison`).
        if planned.program.out_ty() != Ty::F32
            && let (Some(Ok(grid_bits)), Some(Ok(pixel_bits))) = (&raw.gpu, &pixel.gpu)
        {
            let bad: Vec<usize> = (0..raw.count)
                .filter(|&k| grid_bits[k] != pixel_bits[k])
                .collect();
            if let Some(&first) = bad.first() {
                let ty = planned.program.out_ty();
                found.push(mismatch(
                    planned,
                    raw,
                    "pixel",
                    "pixel-vs-grid",
                    bad.len(),
                    Some(first),
                    format!(
                        "element {first}: grid {}, pixel {}",
                        element_text(ty, grid_bits[first]),
                        element_text(ty, pixel_bits[first])
                    ),
                ));
            }
        }
    }
    found
}

/// The distance in `f32` ulps between two bit patterns, on the line of ordered floats (`+0.0` and `-0.0` are one apart
/// at most: they map to the same point). A NaN has no distance and counts as the largest.
pub fn ulp_distance(a: u32, b: u32) -> u32 {
    if f32::from_bits(a).is_nan() || f32::from_bits(b).is_nan() {
        return u32::MAX;
    }
    let ordered = |bits: u32| -> i64 {
        if bits >> 31 == 1 {
            -i64::from(bits & 0x7fff_ffff)
        } else {
            i64::from(bits)
        }
    };
    u32::try_from((ordered(a) - ordered(b)).unsigned_abs()).unwrap_or(u32::MAX)
}

/// What the two shapes' GPU outputs of one `f32` program differ by (an observation, never a mismatch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatComparison {
    /// Elements whose bits differ.
    pub elements: usize,
    /// The first of them, with the grid and the pixel bits.
    pub first: Option<(usize, u32, u32)>,
    /// The largest distance in ulps over the elements.
    pub max_ulp: u32,
}

/// The comparison of the grid and pixel GPU outputs of an `f32` program: `None` when there is nothing to compare (not an
/// `f32` output, no pixel run, no GPU, or one of the two outputs is an error).
pub fn float_comparison(raw: &Raw) -> Option<FloatComparison> {
    if raw.out_ty != "f32" {
        return None;
    }
    let (Some(Ok(grid)), Some(pixel)) = (&raw.gpu, &raw.pixel) else {
        return None;
    };
    let Some(Ok(pixel)) = &pixel.gpu else {
        return None;
    };
    let differing: Vec<usize> = (0..raw.count.min(grid.len()).min(pixel.len()))
        .filter(|&k| grid[k] != pixel[k])
        .collect();
    Some(FloatComparison {
        elements: differing.len(),
        first: differing.first().map(|&k| (k, grid[k], pixel[k])),
        max_ulp: differing
            .iter()
            .map(|&k| ulp_distance(grid[k], pixel[k]))
            .max()
            .unwrap_or(0),
    })
}

/// One party's outputs of one shape.
struct Party<'a> {
    shape: &'static str,
    twin: &'a Result<Vec<u32>, String>,
    gpu: Option<&'a Result<Vec<u32>, String>>,
    wgsl_validated: bool,
}

fn mismatch(
    planned: &Planned,
    raw: &Raw,
    shape: &'static str,
    kind: &'static str,
    elements: usize,
    first: Option<usize>,
    message: String,
) -> Mismatch {
    let _ = planned;
    Mismatch {
        program: raw.program.clone(),
        seed: raw.seed,
        index: raw.index,
        attempt: raw.attempt,
        shape,
        kind,
        elements,
        first_element: first,
        message,
    }
}

/// Holds the twin and the GPU output of one shape to the interpreter.
fn judge(
    planned: &Planned,
    raw: &Raw,
    party: &Party<'_>,
    cpu_only: bool,
    found: &mut Vec<Mismatch>,
) {
    let ty = planned.program.out_ty();
    let make = |kind: &'static str, elements: usize, first: Option<usize>, message: String| {
        mismatch(planned, raw, party.shape, kind, elements, first, message)
    };
    match party.twin {
        Err(message) => found.push(make("twin-error", 0, None, message.clone())),
        Ok(twin) => {
            let bad: Vec<usize> = (0..raw.count)
                .filter(|&k| twin[k] != planned.expected[k])
                .collect();
            if let Some(&first) = bad.first() {
                found.push(make(
                    "twin-vs-interpreter",
                    bad.len(),
                    Some(first),
                    format!(
                        "element {first}: interpreter {}, twin {}",
                        element_text(ty, planned.expected[first]),
                        element_text(ty, twin[first])
                    ),
                ));
            }
        }
    }
    if cpu_only {
        return;
    }
    match party.gpu {
        None => found.push(make(
            "gpu-error",
            0,
            None,
            "the file holds no GPU output in gpu mode".into(),
        )),
        Some(Err(message)) => found.push(make("gpu-error", 0, None, message.clone())),
        Some(Ok(gpu)) => {
            if !party.wgsl_validated {
                found.push(make(
                    "gpu-not-validated",
                    0,
                    None,
                    "the GPU output is present but the module was not marked validated by wgpu"
                        .into(),
                ));
            }
            if ty == Ty::F32 {
                let bad: Vec<usize> = (0..raw.count)
                    .filter(|&k| !in_interval(gpu[k], planned.intervals[k]))
                    .collect();
                if let Some(&first) = bad.first() {
                    let (lo, hi) = planned.intervals[first];
                    found.push(make(
                        "gpu-outside-interval",
                        bad.len(),
                        Some(first),
                        format!(
                            "element {first}: GPU {}, interval [{lo:e}, {hi:e}], interpreter {}",
                            element_text(ty, gpu[first]),
                            element_text(ty, planned.expected[first])
                        ),
                    ));
                }
            } else {
                let bad: Vec<usize> = (0..raw.count)
                    .filter(|&k| gpu[k] != planned.expected[k])
                    .collect();
                if let Some(&first) = bad.first() {
                    found.push(make(
                        "gpu-vs-interpreter",
                        bad.len(),
                        Some(first),
                        format!(
                            "element {first}: interpreter {}, GPU {}",
                            element_text(ty, planned.expected[first]),
                            element_text(ty, gpu[first])
                        ),
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, InputBuf, Program};

    fn program(out: Ty, count: u32) -> Program {
        let result = match out {
            Ty::F32 => Expr::f32(0.0),
            Ty::I32 => Expr::i32(0),
            _ => Expr::u32(0),
        };
        Program {
            seed: 7,
            methods: Default::default(),
            attempt: 3,
            count,
            inputs: vec![InputBuf {
                ty: Ty::U32,
                data: vec![0; count as usize],
            }],
            body: Vec::new(),
            result,
        }
    }

    fn planned(out: Ty, expected: &[u32], intervals: &[(f64, f64)]) -> Planned {
        Planned {
            program: program(out, expected.len() as u32),
            index: 4,
            expected: expected.to_vec(),
            intervals: intervals.to_vec(),
        }
    }

    fn raw(
        planned: &Planned,
        twin: Result<Vec<u32>, String>,
        gpu: Option<Result<Vec<u32>, String>>,
    ) -> Raw {
        Raw {
            program: "s7_a3".into(),
            seed: 7,
            attempt: 3,
            index: 4,
            count: planned.expected.len(),
            out_ty: planned.program.out_ty().name().into(),
            mode: if gpu.is_some() { "gpu" } else { "cpu-only" }.into(),
            adapter: None,
            capabilities: Vec::new(),
            interp: planned.expected.clone(),
            twin,
            gpu,
            wgsl_validated: true,
            twin_ns: 0,
            gpu_first_ns: 0,
            gpu_warm_ns: 0,
            pixel: None,
        }
    }

    #[test]
    fn every_label_the_generator_can_produce_has_a_class() {
        use crate::interp::Violation;
        for violation in Violation::ALL {
            let class = discard_class(violation.label());
            assert!(CLASSES.contains(&class), "{violation:?}");
        }
        for reason in [
            "static",
            "no-input-read",
            "no-aggregate",
            "no-call",
            "no-match",
        ] {
            assert!(CLASSES.contains(&discard_class(reason)));
        }
    }

    #[test]
    fn the_domain_class_is_the_integer_labels_float_domain_and_static() {
        for reason in [
            "overflow",
            "division-by-zero",
            "min-over-minus-one",
            "shift-amount",
            "index",
            "float-domain",
            "static",
        ] {
            assert_eq!(discard_class(reason), "domain", "{reason}");
        }
        assert_eq!(discard_class("comparison-overlap"), "comparison-overlap");
        assert_eq!(discard_class("unboundable"), "unboundable");
        assert_eq!(
            discard_class("float-construct-not-in-v1"),
            "float-construct-not-in-v1"
        );
        assert_eq!(discard_class("no-input-read"), "no-input-read");
        assert_eq!(discard_class("no-aggregate"), "no-aggregate");
        assert_eq!(discard_class("no-call"), "no-call");
        assert_eq!(discard_class("no-match"), "no-match");
    }

    #[test]
    fn the_gpu_f32_is_held_by_value_so_either_zero_is_inside_the_interval() {
        let negative_zero = (-0.0f32).to_bits();
        let zero = 0.0f32.to_bits();
        assert_ne!(negative_zero, zero);
        // An interpreter output of +0 and a GPU that holds -0 (or the other way round) is not a mismatch.
        let p = planned(
            Ty::F32,
            &[zero, 1.5f32.to_bits()],
            &[(0.0, 0.0), (1.4, 1.6)],
        );
        let r = raw(
            &p,
            Ok(p.expected.clone()),
            Some(Ok(vec![negative_zero, 1.5f32.to_bits()])),
        );
        assert_eq!(classify(&p, &r, false), []);
        let p = planned(Ty::F32, &[negative_zero], &[(0.0, 0.0)]);
        let r = raw(&p, Ok(vec![negative_zero]), Some(Ok(vec![zero])));
        assert_eq!(classify(&p, &r, false), []);
    }

    #[test]
    fn the_twin_is_compared_bit_for_bit_so_negative_zero_against_zero_is_a_mismatch() {
        let negative_zero = (-0.0f32).to_bits();
        let zero = 0.0f32.to_bits();
        let p = planned(Ty::F32, &[zero], &[(0.0, 0.0)]);
        let found = classify(&p, &raw(&p, Ok(vec![negative_zero]), None), true);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, "twin-vs-interpreter");
        // And the other way round; equal bits are fine; integers too.
        let p = planned(Ty::F32, &[negative_zero], &[(0.0, 0.0)]);
        assert_eq!(classify(&p, &raw(&p, Ok(vec![zero]), None), true).len(), 1);
        assert_eq!(
            classify(&p, &raw(&p, Ok(vec![negative_zero]), None), true),
            []
        );
        let p = planned(Ty::I32, &[0], &[]);
        assert_eq!(
            classify(&p, &raw(&p, Ok(vec![u32::MAX]), None), true).len(),
            1
        );
    }

    #[test]
    fn a_gpu_f32_must_be_finite_and_inside_its_interval() {
        let p = planned(Ty::F32, &[1.0f32.to_bits()], &[(0.99, 1.01)]);
        let ok = |bits: u32| {
            classify(
                &p,
                &raw(&p, Ok(p.expected.clone()), Some(Ok(vec![bits]))),
                false,
            )
        };
        assert_eq!(ok(1.005f32.to_bits()), []);
        assert_eq!(ok(0.99f32.to_bits() + 1), []);
        let low = ok(0.98f32.to_bits());
        assert_eq!(low.len(), 1);
        assert_eq!(low[0].kind, "gpu-outside-interval");
        assert_eq!(low[0].first_element, Some(0));
        assert_eq!(ok(f32::INFINITY.to_bits())[0].kind, "gpu-outside-interval");
        assert_eq!(ok(f32::NAN.to_bits())[0].kind, "gpu-outside-interval");
    }

    #[test]
    fn integer_and_boolean_outputs_must_match_bit_for_bit() {
        let p = planned(Ty::U32, &[1, 2, 3], &[]);
        let r = raw(&p, Ok(vec![1, 2, 3]), Some(Ok(vec![1, 9, 8])));
        let found = classify(&p, &r, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, "gpu-vs-interpreter");
        assert_eq!((found[0].elements, found[0].first_element), (2, Some(1)));
        assert!(found[0].message.contains("interpreter 2, GPU 9"));
        assert_eq!(found[0].program, "s7_a3");
        let r = raw(&p, Ok(vec![1, 2, 4]), Some(Ok(vec![1, 2, 3])));
        let found = classify(&p, &r, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, "twin-vs-interpreter");
        assert_eq!(found[0].first_element, Some(2));
    }

    #[test]
    fn errors_of_a_party_are_mismatches_and_cpu_only_skips_the_gpu() {
        let p = planned(Ty::I32, &[1], &[]);
        let r = raw(
            &p,
            Err("twin panicked: boom".into()),
            Some(Err("dispatch: nope".into())),
        );
        let kinds: Vec<&str> = classify(&p, &r, false).iter().map(|m| m.kind).collect();
        assert_eq!(kinds, ["twin-error", "gpu-error"]);
        let r = raw(&p, Ok(vec![1]), None);
        assert_eq!(classify(&p, &r, true), []);
        let kinds: Vec<&str> = classify(&p, &r, false).iter().map(|m| m.kind).collect();
        assert_eq!(kinds, ["gpu-error"]);
        let mut r = raw(&p, Ok(vec![1]), Some(Ok(vec![1])));
        r.wgsl_validated = false;
        let kinds: Vec<&str> = classify(&p, &r, false).iter().map(|m| m.kind).collect();
        assert_eq!(kinds, ["gpu-not-validated"]);
        let mut r = raw(&p, Ok(vec![1]), None);
        r.interp = vec![2];
        let kinds: Vec<&str> = classify(&p, &r, true).iter().map(|m| m.kind).collect();
        assert_eq!(kinds, ["transport"]);
    }

    fn result_text(program: &str, count: usize, extra: &str) -> String {
        let words = vec!["1"; count].join(",");
        format!(
            "{{\"format\": {DIFFTEST_FORMAT}, \"program\": \"{program}\", \"seed\": 7, \"attempt\": 3, \"index\": 4, \"count\": {count}, \
             \"out_ty\": \"u32\", \"mode\": \"cpu-only\", \"adapter\": null, \"capabilities\": [], \"interp\": [{words}], \"twin\": [{words}], \
             \"twin_error\": null, \"twin_ns\": 5, \"gpu\": null, \"gpu_error\": null, \"wgsl_validated\": false, \
             \"gpu_first_ns\": 0, \"gpu_warm_ns\": 0{extra}, \"end\": true}}"
        )
    }

    #[test]
    fn a_result_file_must_hold_the_count_and_the_outputs() {
        let good = parse_result(&result_text("s7_a3", 2, "")).unwrap();
        assert_eq!((good.count, good.gpu.is_none(), good.twin_ns), (2, true, 5));
        // A truncated array, a missing field, a half-written file, a mode that contradicts the fields.
        let short = result_text("s7_a3", 2, "").replace("\"count\": 2", "\"count\": 3");
        assert!(parse_result(&short).unwrap_err().contains("elements"));
        let missing = result_text("s7_a3", 2, "").replace("\"interp\"", "\"interpreter\"");
        assert!(parse_result(&missing).unwrap_err().contains("interp"));
        let half = &result_text("s7_a3", 2, "")[..60];
        assert!(parse_result(half).is_err());
        let both =
            result_text("s7_a3", 2, "").replace("\"twin_error\": null", "\"twin_error\": \"x\"");
        assert!(parse_result(&both).unwrap_err().contains("exactly one"));
        let mode = result_text("s7_a3", 2, "").replace("cpu-only", "gpu");
        assert!(parse_result(&mode).unwrap_err().contains("mode"));
        // The capabilities the program reports are read, and a file without them is refused (Opus S4).
        let with = result_text("s7_a3", 2, "")
            .replace("\"capabilities\": []", "\"capabilities\": [\"subgroups\"]");
        assert_eq!(parse_result(&with).unwrap().capabilities, ["subgroups"]);
        let without = result_text("s7_a3", 2, "").replace("\"capabilities\"", "\"caps\"");
        assert!(parse_result(&without).unwrap_err().contains("capabilities"));
    }

    /// m8.x1: the format of the file is read and must be this harness's. A file of the format before the one that made
    /// `capabilities` required (6), one of a later format and one without the field are refused, each by the format.
    #[test]
    fn a_result_file_of_another_format_is_refused_by_its_format() {
        let current = result_text("s7_a3", 2, "");
        assert!(current.contains(&format!("\"format\": {DIFFTEST_FORMAT},")));
        assert!(parse_result(&current).is_ok());
        for other in [DIFFTEST_FORMAT - 1, DIFFTEST_FORMAT + 1, 2] {
            let stale = current.replace(
                &format!("\"format\": {DIFFTEST_FORMAT},"),
                &format!("\"format\": {other},"),
            );
            let error = parse_result(&stale).unwrap_err();
            assert!(
                error.contains(&format!("format {other}"))
                    && error.contains(&format!("reads format {DIFFTEST_FORMAT}")),
                "{other}: {error}"
            );
        }
        let missing = current.replace("\"format\"", "\"layout\"");
        assert!(parse_result(&missing).unwrap_err().contains("format"));
        let text = current.replace(
            &format!("\"format\": {DIFFTEST_FORMAT},"),
            &format!("\"format\": \"{DIFFTEST_FORMAT}\","),
        );
        assert!(parse_result(&text).unwrap_err().contains("format"));
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("difftest-classify-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn exactly_k_result_files_are_required_before_anything_is_classified() {
        let dir = scratch("k-files");
        let p = |attempt: u64| {
            let mut planned = planned(Ty::U32, &[1, 1], &[]);
            planned.program.attempt = attempt;
            planned
        };
        let plan = vec![p(3), p(4), p(5)];
        for planned in &plan {
            let name = crate::emit::module_name(&planned.program);
            let text = result_text(&name, 2, "").replace(
                "\"attempt\": 3",
                &format!("\"attempt\": {}", planned.program.attempt),
            );
            std::fs::write(dir.join(format!("{name}.json")), text).unwrap();
        }
        assert_eq!(read_results(&dir, &plan).unwrap().len(), 3);
        // One file dropped: the count assertion fails and nothing is classified.
        std::fs::remove_file(dir.join("s7_a4.json")).unwrap();
        let error = read_results(&dir, &plan).unwrap_err();
        assert!(
            error.contains("expected exactly 3 result files, found 2"),
            "{error}"
        );
        // The right number of files but one of them is not a planned program.
        std::fs::write(dir.join("s7_a99.json"), "{}").unwrap();
        let error = read_results(&dir, &plan).unwrap_err();
        assert!(error.contains("s7_a4.json is missing"), "{error}");
        // An extra file.
        std::fs::write(dir.join("s7_a4.json"), "{}").unwrap();
        let error = read_results(&dir, &plan).unwrap_err();
        assert!(error.contains("found 4"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn with_pixel(
        mut raw: Raw,
        twin: Result<Vec<u32>, String>,
        gpu: Option<Result<Vec<u32>, String>>,
    ) -> Raw {
        raw.pixel = Some(PixelRaw {
            twin,
            gpu,
            wgsl_validated: true,
            twin_ns: 0,
            gpu_first_ns: 0,
            gpu_warm_ns: 0,
        });
        raw
    }

    #[test]
    fn a_pixel_run_that_agrees_with_the_interpreter_and_the_grid_run_is_no_mismatch() {
        let p = planned(Ty::U32, &[1, 2, 3], &[]);
        let grid = raw(&p, Ok(vec![1, 2, 3]), Some(Ok(vec![1, 2, 3])));
        let both = with_pixel(grid, Ok(vec![1, 2, 3]), Some(Ok(vec![1, 2, 3])));
        assert_eq!(classify(&p, &both, false), []);
    }

    #[test]
    fn an_f32_pixel_output_inside_its_interval_may_differ_from_the_grid_bits() {
        // D128: each shape is judged against the interval; a bit difference between them is an observation.
        let one = 1.0f32.to_bits();
        let p = planned(Ty::F32, &[one, one], &[(0.99, 1.01), (0.99, 1.01)]);
        let grid = raw(&p, Ok(vec![one, one]), Some(Ok(vec![one, one])));
        let near = with_pixel(grid, Ok(vec![one, one]), Some(Ok(vec![one, one + 1])));
        assert_eq!(classify(&p, &near, false), []);
        assert_eq!(
            float_comparison(&near),
            Some(FloatComparison {
                elements: 1,
                first: Some((1, one, one + 1)),
                max_ulp: 1
            })
        );
        // Outside its interval, the pixel f32 is a mismatch of its own shape, whatever the grid did.
        let far = with_pixel(
            raw(&p, Ok(vec![one, one]), Some(Ok(vec![one, one]))),
            Ok(vec![one, one]),
            Some(Ok(vec![one, 1.5f32.to_bits()])),
        );
        let found = classify(&p, &far, false);
        let kinds: Vec<(&str, &str)> = found.iter().map(|m| (m.shape, m.kind)).collect();
        assert_eq!(kinds, [("pixel", "gpu-outside-interval")]);
    }

    #[test]
    fn the_ulp_distance_is_taken_on_the_ordered_floats() {
        let (one, two) = (1.0f32.to_bits(), 2.0f32.to_bits());
        assert_eq!(ulp_distance(one, one), 0);
        assert_eq!(ulp_distance(one, one + 3), 3);
        assert_eq!(ulp_distance(two, two - 1), 1);
        // Across zero: the smallest positive and the smallest negative subnormal are two apart, `0.0` and `-0.0` none.
        assert_eq!(ulp_distance(1, 0x8000_0001), 2);
        assert_eq!(ulp_distance(0, 0x8000_0000), 0);
        assert_eq!(ulp_distance(f32::NAN.to_bits(), one), u32::MAX);
        // Integer and absent comparisons are not float observations.
        let p = planned(Ty::U32, &[1], &[]);
        let both = with_pixel(
            raw(&p, Ok(vec![1]), Some(Ok(vec![1]))),
            Ok(vec![1]),
            Some(Ok(vec![1])),
        );
        assert_eq!(float_comparison(&both), None);
    }

    #[test]
    fn the_pixel_gpu_output_of_an_integer_must_hold_the_grid_bits() {
        let p = planned(Ty::U32, &[7, 8], &[]);
        let grid = raw(&p, Ok(vec![7, 8]), Some(Ok(vec![7, 8])));
        let off = with_pixel(grid, Ok(vec![7, 8]), Some(Ok(vec![7, 9])));
        let found = classify(&p, &off, false);
        let kinds: Vec<(&str, &str)> = found.iter().map(|m| (m.shape, m.kind)).collect();
        assert_eq!(
            kinds,
            [("pixel", "gpu-vs-interpreter"), ("pixel", "pixel-vs-grid")]
        );
        let pvg = &found[1];
        assert_eq!((pvg.elements, pvg.first_element), (1, Some(1)));
        assert!(pvg.message.contains("grid 8"), "{}", pvg.message);
        // The poison value left by a pass that never wrote an element: wrong against the interpreter and the grid.
        let p = planned(Ty::U32, &[7, 8], &[]);
        let grid = raw(&p, Ok(vec![7, 8]), Some(Ok(vec![7, 8])));
        let unwritten = with_pixel(
            grid,
            Ok(vec![7, 8]),
            Some(Ok(vec![7, crate::shapes::POISON])),
        );
        let found = classify(&p, &unwritten, false);
        let kinds: Vec<(&str, &str)> = found.iter().map(|m| (m.shape, m.kind)).collect();
        assert_eq!(
            kinds,
            [("pixel", "gpu-vs-interpreter"), ("pixel", "pixel-vs-grid")]
        );
    }

    #[test]
    fn a_disagreeing_grid_run_is_not_blamed_on_the_pixel_shape() {
        let p = planned(Ty::U32, &[1, 2], &[]);
        let grid = raw(&p, Ok(vec![1, 2]), Some(Ok(vec![1, 9])));
        let pixel = with_pixel(grid, Ok(vec![1, 2]), Some(Ok(vec![1, 2])));
        let found = classify(&p, &pixel, false);
        let kinds: Vec<(&str, &str)> = found.iter().map(|m| (m.shape, m.kind)).collect();
        assert_eq!(
            kinds,
            [("grid", "gpu-vs-interpreter"), ("pixel", "pixel-vs-grid")]
        );
    }

    #[test]
    fn pixel_errors_and_cpu_only_follow_the_grid_rules_under_the_pixel_shape() {
        let p = planned(Ty::I32, &[1], &[]);
        let grid = raw(&p, Ok(vec![1]), Some(Ok(vec![1])));
        let broken = with_pixel(
            grid,
            Err("pixel twin panicked".into()),
            Some(Err("pixel dispatch: no".into())),
        );
        let found = classify(&p, &broken, false);
        let kinds: Vec<(&str, &str)> = found.iter().map(|m| (m.shape, m.kind)).collect();
        assert_eq!(kinds, [("pixel", "twin-error"), ("pixel", "gpu-error")]);
        let grid = raw(&p, Ok(vec![1]), None);
        let cpu = with_pixel(grid, Ok(vec![1]), None);
        assert_eq!(classify(&p, &cpu, true), []);
        let mut off = with_pixel(
            raw(&p, Ok(vec![1]), Some(Ok(vec![1]))),
            Ok(vec![1]),
            Some(Ok(vec![1])),
        );
        off.pixel.as_mut().unwrap().wgsl_validated = false;
        let found = classify(&p, &off, false);
        let kinds: Vec<(&str, &str)> = found.iter().map(|m| (m.shape, m.kind)).collect();
        assert_eq!(kinds, [("pixel", "gpu-not-validated")]);
    }

    #[test]
    fn a_result_file_with_pixel_fields_parses_into_the_pixel_outputs() {
        let pixel = |gpu: &str, gpu_error: &str| {
            format!(
                ", \"pixel_status\": \"run\", \"pixel_twin\": [1,1], \"pixel_twin_error\": null, \"pixel_twin_ns\": 9, \
                 \"pixel_gpu\": {gpu}, \"pixel_gpu_error\": {gpu_error}, \"pixel_wgsl_validated\": false, \
                 \"pixel_gpu_first_ns\": 0, \"pixel_gpu_warm_ns\": 0"
            )
        };
        let cpu = parse_result(&result_text("s7_a3", 2, &pixel("null", "null"))).unwrap();
        let px = cpu.pixel.as_ref().unwrap();
        assert_eq!((px.twin_ns, px.gpu.is_none()), (9, true));
        // No pixel fields: the grid-only form.
        assert!(
            parse_result(&result_text("s7_a3", 2, ""))
                .unwrap()
                .pixel
                .is_none()
        );
        // A mode that contradicts the pixel GPU fields, and a status other than "run".
        let gpu_mode = |extra: &str| {
            result_text("s7_a3", 2, extra)
                .replace("cpu-only", "gpu")
                .replace("\"gpu\": null", "\"gpu\": [1,1]")
        };
        let mismatched = gpu_mode(&pixel("null", "null"));
        assert!(parse_result(&mismatched).unwrap_err().contains("mode"));
        let status =
            result_text("s7_a3", 2, &pixel("null", "null")).replace("\"run\"", "\"skipped\"");
        assert!(parse_result(&status).unwrap_err().contains("pixel_status"));
        // A pixel GPU output of the wrong length.
        let short = gpu_mode(&pixel("[1]", "null"));
        assert!(parse_result(&short).unwrap_err().contains("pixel_gpu"));
    }
}

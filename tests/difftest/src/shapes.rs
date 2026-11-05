//! Execution shapes in the batch (slice mx.p1c): `--shapes grid,pixel`.
//!
//! The generated kernel `k` is a `grid` kernel and is never touched. With `pixel` requested, every program also gets a
//! second kernel `k_px` in the same module: the same parameters and body under `#[kernel(shape = pixel)]` (no
//! `workgroup_size`, which the classifier rules out for `pixel`). Nothing here decides whether a program is
//! pixel-compatible. The pin is checked by the macro's own classifier (`classify::check_pin`); when it refuses a
//! `k_px`, the build fails with a `GUST-E072x` error naming the construct, and [`refusals`] and [`assign`] turn those
//! errors into the programs that stay `grid`-only, with the macro's message as the reason.

use std::collections::BTreeMap;

/// The name of the pixel form of a program's kernel.
pub const PIXEL_KERNEL: &str = "k_px";

/// The bit pattern a pixel output buffer starts with (a grid output starts at zero), so an element the pixel pass never
/// wrote differs from the grid result.
pub const POISON: u32 = 0xA5A5_A5A5;

/// An execution shape the batch can run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Grid,
    Pixel,
}

impl Shape {
    pub fn name(self) -> &'static str {
        match self {
            Self::Grid => "grid",
            Self::Pixel => "pixel",
        }
    }
}

/// Parses `--shapes`: `grid` or `grid,pixel`. `grid` is the oracle of the comparison and must be named; each shape at
/// most once.
pub fn parse_shapes(text: &str) -> Result<Vec<Shape>, String> {
    let mut shapes = Vec::new();
    for part in text.split(',') {
        let shape = match part.trim() {
            "grid" => Shape::Grid,
            "pixel" => Shape::Pixel,
            other => {
                return Err(format!(
                    "unknown shape {other:?}; the shapes are `grid` and `pixel`"
                ));
            }
        };
        if shapes.contains(&shape) {
            return Err(format!("shape `{}` named twice", shape.name()));
        }
        shapes.push(shape);
    }
    if !shapes.contains(&Shape::Grid) {
        return Err(
            "`--shapes` must name `grid`, the shape every other one is compared with".into(),
        );
    }
    Ok(shapes)
}

/// A program the classifier keeps off `pixel`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridOnly {
    /// The reason as the report counts it: the code and the macro's clause with the quoted constructs elided.
    pub reason: String,
    /// The macro's full message.
    pub message: String,
}

/// Which programs get a `k_px`.
#[derive(Clone, Debug, Default)]
pub struct ShapePlan {
    /// `pixel` was requested.
    pub pixel: bool,
    /// Programs (by module name) whose `k_px` the macro refused.
    pub grid_only: BTreeMap<String, GridOnly>,
}

impl ShapePlan {
    pub fn new(shapes: &[Shape]) -> Self {
        Self {
            pixel: shapes.contains(&Shape::Pixel),
            grid_only: BTreeMap::new(),
        }
    }

    /// Whether the module `module` is emitted and run with a `k_px`.
    pub fn runs_pixel(&self, module: &str) -> bool {
        self.pixel && !self.grid_only.contains_key(module)
    }

    pub fn requested(&self) -> Vec<&'static str> {
        if self.pixel {
            vec!["grid", "pixel"]
        } else {
            vec!["grid"]
        }
    }
}

/// One `GUST-E072x` error of a build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The line of `src/lib.rs` the error points at (the `#[kernel(shape = pixel)]` attribute).
    pub line: usize,
    pub code: String,
    /// The text after the code.
    pub message: String,
}

/// The `GUST-E072x` errors of a cargo build log (human format: `error: [GUST-E0727] message`, then `--> src/lib.rs:L:C`).
/// Other errors, including the ones that follow from a module the macro refused, are ignored.
pub fn refusals(log: &str) -> Vec<Refusal> {
    let mut found = Vec::new();
    let lines: Vec<&str> = log.lines().collect();
    for (at, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("error: [GUST-E072") else {
            continue;
        };
        let Some((digit, message)) = rest.split_once("] ") else {
            continue;
        };
        // The location is within the next few lines of the same diagnostic.
        let location = lines[at + 1..lines.len().min(at + 6)]
            .iter()
            .find_map(|next| {
                let (_, after) = next
                    .trim_start()
                    .strip_prefix("--> ")?
                    .split_once("lib.rs:")?;
                after.split(':').next()?.parse::<usize>().ok()
            });
        if let Some(line) = location {
            found.push(Refusal {
                line,
                code: format!("GUST-E072{digit}"),
                message: message.to_owned(),
            });
        }
    }
    found
}

/// The modules of an emitted crate source with their 1-based line ranges: `pub mod NAME {` to the closing `}` at
/// column zero.
pub fn module_ranges(source: &str) -> Vec<(String, usize, usize)> {
    let mut ranges = Vec::new();
    let mut open: Option<(String, usize)> = None;
    for (at, line) in source.lines().enumerate() {
        let number = at + 1;
        if let Some(name) = line
            .strip_prefix("pub mod ")
            .and_then(|rest| rest.strip_suffix(" {"))
        {
            open = Some((name.to_owned(), number));
        } else if line == "}"
            && let Some((name, start)) = open.take()
        {
            ranges.push((name, start, number));
        }
    }
    ranges
}

/// The programs whose `k_px` the macro refused: each refusal belongs to the module whose line range holds it. A module
/// refused twice keeps the first message.
pub fn assign(
    refusals: &[Refusal],
    ranges: &[(String, usize, usize)],
) -> BTreeMap<String, GridOnly> {
    let mut found = BTreeMap::new();
    for refusal in refusals {
        let Some((name, _, _)) = ranges
            .iter()
            .find(|(_, start, end)| (*start..=*end).contains(&refusal.line))
        else {
            continue;
        };
        found.entry(name.clone()).or_insert_with(|| GridOnly {
            reason: reason_key(&refusal.code, &refusal.message),
            message: format!("{}: {}", refusal.code, refusal.message),
        });
    }
    found
}

/// `shape `pixel` is not compatible with kernel `k`: `<construct>` <clause> (that makes the kernel `grid`-only)` becomes
/// `<code>: <clause>` with every quoted span elided, so that the same rule counts as one reason whatever the construct.
pub fn reason_key(code: &str, message: &str) -> String {
    let clause = message
        .split_once("`: `")
        .and_then(|(_, after)| after.split_once("` "))
        .map_or(message, |(_, clause)| clause);
    let clause = clause.split(" (that makes").next().unwrap_or(clause);
    let mut out = String::new();
    let mut quoted = false;
    for ch in clause.chars() {
        if ch == '`' {
            if !quoted {
                out.push_str("..");
            }
            quoted = !quoted;
        } else if !quoted {
            out.push(ch);
        }
    }
    format!("{code}: {out}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFUSED: &str = "error: [GUST-E0727] shape `pixel` is not compatible with kernel `k`: `return` returns early, and a pixel kernel has no `return` (that makes the kernel `grid`-only)\n  --> src\\lib.rs:41:15\n   |\n41 |     #[kernel(shape = pixel)]\n   |               ^^^^^^^^^^^^^\n\nerror[E0433]: failed to resolve: could not find `s1_a2` in the crate root\n --> src/lib.rs:900:9\n\nerror: [GUST-E0723] shape `pixel` is not compatible with kernel `k`: `x` is an atomic operation, which the `pixel` shape does not take (that makes the kernel `grid`-only)\n --> src/lib.rs:10:5\n";

    #[test]
    fn shapes_name_grid_and_each_shape_once() {
        assert_eq!(parse_shapes("grid").unwrap(), [Shape::Grid]);
        assert_eq!(
            parse_shapes("grid,pixel").unwrap(),
            [Shape::Grid, Shape::Pixel]
        );
        assert_eq!(
            parse_shapes("pixel,grid").unwrap(),
            [Shape::Pixel, Shape::Grid]
        );
        assert!(
            parse_shapes("pixel")
                .unwrap_err()
                .contains("must name `grid`")
        );
        assert!(parse_shapes("grid,grid").unwrap_err().contains("twice"));
        assert!(
            parse_shapes("grid,cpu")
                .unwrap_err()
                .contains("unknown shape")
        );
        assert!(parse_shapes("").is_err());
    }

    #[test]
    fn only_the_classifier_errors_of_a_build_are_refusals() {
        let found = refusals(REFUSED);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!((found[0].line, found[0].code.as_str()), (41, "GUST-E0727"));
        assert_eq!((found[1].line, found[1].code.as_str()), (10, "GUST-E0723"));
        assert!(
            found[0]
                .message
                .starts_with("shape `pixel` is not compatible")
        );
        assert_eq!(
            refusals("error[E0308]: mismatched types\n --> src/lib.rs:3:1\n"),
            []
        );
        assert_eq!(refusals("error: [GUST-E0727] no location here\n"), []);
    }

    #[test]
    fn a_refusal_belongs_to_the_module_that_holds_its_line() {
        let source = "// header\n\n/// doc\n#[gust::gpu]\npub mod s1_a0 {\n    body\n}\n\n#[gust::gpu]\npub mod s1_a1 {\n    body\n}\n";
        let ranges = module_ranges(source);
        assert_eq!(
            ranges,
            [("s1_a0".to_owned(), 5, 7), ("s1_a1".to_owned(), 10, 12)]
        );
        let at = |line: usize| {
            Refusal {
            line,
            code: "GUST-E0727".into(),
            message: "shape `pixel` is not compatible with kernel `k`: `return` returns early (that makes the kernel `grid`-only)".into(),
        }
        };
        let found = assign(&[at(11), at(6), at(2)], &ranges);
        assert_eq!(found.keys().collect::<Vec<_>>(), ["s1_a0", "s1_a1"]);
        assert_eq!(found["s1_a0"].reason, "GUST-E0727: returns early");
        assert!(
            found["s1_a0"]
                .message
                .starts_with("GUST-E0727: shape `pixel`")
        );
    }

    #[test]
    fn the_same_rule_counts_as_one_reason_whatever_the_construct() {
        let message = |construct: &str| {
            format!(
                "shape `pixel` is not compatible with kernel `k`: `{construct}` makes the write to the output `out` conditional, and a pixel output must be written on every path (that makes the kernel `grid`-only) and 1 more construct rule it out"
            )
        };
        assert_eq!(
            reason_key("GUST-E0726", &message("if a {}")),
            reason_key("GUST-E0726", &message("if b {}"))
        );
        assert_eq!(
            reason_key("GUST-E0726", &message("if a {}")),
            "GUST-E0726: makes the write to the output .. conditional, and a pixel output must be written on every path"
        );
        // A message of another form is kept whole, its quoted spans elided.
        assert_eq!(
            reason_key("GUST-E0721", "unknown `x`"),
            "GUST-E0721: unknown .."
        );
    }

    #[test]
    fn a_plan_runs_pixel_only_when_asked_and_not_refused() {
        let mut plan = ShapePlan::new(&[Shape::Grid]);
        assert!(!plan.runs_pixel("s1_a0"));
        assert_eq!(plan.requested(), ["grid"]);
        plan = ShapePlan::new(&[Shape::Grid, Shape::Pixel]);
        assert!(plan.runs_pixel("s1_a0"));
        plan.grid_only.insert(
            "s1_a0".into(),
            GridOnly {
                reason: "r".into(),
                message: "m".into(),
            },
        );
        assert!(!plan.runs_pixel("s1_a0") && plan.runs_pixel("s1_a1"));
        assert_eq!(plan.requested(), ["grid", "pixel"]);
    }
}

//! The reviewed expectations `tests/fixtures/reduce.slang`, `tests/fixtures/scan.slang` and
//! `tests/fixtures/compact.slang`: the Slang the macro emits for every kernel of `tests/fixtures/reduce.rs`,
//! `scan.rs` and `compact.rs`, in declaration order, each under a marker line (the layout of `mismatch.slang`). A
//! translation change needs an explained golden diff and the GPU tests of this crate.

use std::{env, fs, path::Path};

use gust::ModuleDescriptor;

use crate::{compact_kernels, kernels, scan_kernels};

const BLESS_VARIABLE: &str = "BLESS_GOLDEN";

fn emitted(module: &ModuleDescriptor) -> String {
    module
        .kernels
        .iter()
        .map(|kernel| {
            format!(
                "// ==== kernel {} ====\n{}",
                kernel.name,
                kernel.slang_source.replace("\r\n", "\n")
            )
        })
        .collect()
}

fn check_golden(module: &ModuleDescriptor, file: &str) {
    // With the SPIR-V route's guards in the emitted Slang (m9.rt2) the text is another one, pinned in its own file; the
    // files without the suffix stay the text of a build without the feature.
    let routed;
    let file = if gust::SPIRV_GUARDS {
        routed = file.replace(".slang", ".spirv-route.slang");
        routed.as_str()
    } else {
        file
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(file);
    let actual = emitted(module);
    let stored = fs::read_to_string(&path).unwrap_or_default();
    let expected = stored.replace("\r\n", "\n");
    if actual == expected {
        return;
    }
    if env::var_os(BLESS_VARIABLE).is_some_and(|value| value == "1") {
        // Keep the line endings the working tree already uses, so a bless diffs only real changes.
        let text = if stored.contains("\r\n") {
            actual.replace('\n', "\r\n")
        } else {
            actual
        };
        fs::write(&path, text).expect("the golden can be written");
        panic!(
            "golden blessed; review `git diff tests/fixtures/{file}` and rerun without {BLESS_VARIABLE}"
        );
    }
    let first_difference = actual
        .lines()
        .zip(expected.lines())
        .position(|(left, right)| left != right)
        .map_or_else(
            || "one text is a prefix of the other".to_owned(),
            |line| format!("line {}", line + 1),
        );
    panic!(
        "tests/fixtures/{file} differs from the emitted Slang (first difference at {first_difference}); \
         rerun with {BLESS_VARIABLE}=1 to rewrite it and review the diff"
    );
}

fn assert_each_kernel_once(module: &ModuleDescriptor) {
    let text = emitted(module);
    for kernel in module.kernels {
        assert_eq!(
            text.matches(&format!("// ==== kernel {} ====\n", kernel.name))
                .count(),
            1,
            "{}",
            kernel.name
        );
        assert!(
            text.contains(&format!("void {}(", kernel.entry_point)),
            "{}",
            kernel.entry_point
        );
    }
}

#[test]
fn reduce_golden() {
    check_golden(&kernels::MODULE_DESCRIPTOR, "reduce.slang");
}

#[test]
fn scan_golden() {
    check_golden(&scan_kernels::MODULE_DESCRIPTOR, "scan.slang");
}

#[test]
fn compact_golden() {
    check_golden(&compact_kernels::MODULE_DESCRIPTOR, "compact.slang");
}

#[test]
fn the_golden_holds_every_kernel_once() {
    // 3 element types x (sum, min, max) x (pass, final), then the wrapping `u32` sum (pass, final), then the v1 block
    // kernels: 3 element types x (sum, min, max) and the wrapping `u32` sum (m7.r4).
    assert_eq!(kernels::MODULE_DESCRIPTOR.kernels.len(), 30);
    assert_each_kernel_once(&kernels::MODULE_DESCRIPTOR);
}

#[test]
fn the_scan_golden_holds_every_kernel_once() {
    // The level-0 scan, the scan of totals, the top scan, the add-back into totals and the two final add-backs, then
    // the v1 kernels (m7.r4): the inclusive and the exclusive tile scan, the tile scan of totals, the top tile scan and
    // the add-back of a carry.
    assert_eq!(scan_kernels::MODULE_DESCRIPTOR.kernels.len(), 11);
    assert_each_kernel_once(&scan_kernels::MODULE_DESCRIPTOR);
}

#[test]
fn the_compact_golden_holds_every_kernel_once() {
    // The mask, the scatter and one gather per element type.
    assert_eq!(compact_kernels::MODULE_DESCRIPTOR.kernels.len(), 5);
    assert_each_kernel_once(&compact_kernels::MODULE_DESCRIPTOR);
}

/// The Slang cannot tell `wrapping_add` from `+` (SEMANTICS KM-10: it lowers to the plain operator), and a release
/// build of the twin wraps either way, so this reads the source: the `sum_wrapping_u32_*` kernels (pass, final, and the
/// v1 block kernel, whose tree also adds lane values) must add with `wrapping_add` and never with a plain `+` or `+=`.
#[test]
fn the_wrapping_sum_kernels_add_with_wrapping_add() {
    let source = include_str!("../../../../tests/fixtures/reduce.rs").replace("\r\n", "\n");
    // The text of one kernel: from its `pub fn` to the next `#[kernel` (or the end of the module).
    let kernel = |name: &str| {
        let from = source.find(&format!("pub fn {name}(")).unwrap();
        let rest = &source[from..];
        rest[..rest.find("#[kernel").unwrap_or(rest.len())].to_owned()
    };
    for name in ["sum_wrapping_u32_pass", "sum_wrapping_u32_final"] {
        let text = kernel(name);
        assert_eq!(
            text.matches("acc = acc.wrapping_add(input[index]);")
                .count(),
            1,
            "{name}"
        );
        assert!(!text.contains("acc +"), "a plain addition in {name}");
    }
    let block = kernel("sum_wrapping_u32_block");
    assert_eq!(
        block
            .matches("acc = acc.wrapping_add(input[index]);")
            .count(),
        1
    );
    assert_eq!(
        block
            .matches("scratch[lid.x] = scratch[lid.x].wrapping_add(scratch[lid.x + s]);")
            .count(),
        1
    );
    assert!(
        !block.contains("acc +") && !block.contains("scratch[lid.x] +="),
        "a plain addition in the wrapping block kernel"
    );
}

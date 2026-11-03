//! The reviewed expectations `tests/fixtures/reduce.slang` and `tests/fixtures/scan.slang`: the Slang the macro
//! emits for every kernel of `tests/fixtures/reduce.rs` and `tests/fixtures/scan.rs`, in declaration order, each under a
//! marker line (the layout of `mismatch.slang`). A translation change needs an explained golden diff and the GPU tests
//! of this crate.

use std::{env, fs, path::Path};

use gust::ModuleDescriptor;

use crate::{kernels, scan_kernels};

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
fn the_golden_holds_every_kernel_once() {
    // 3 element types x (sum, min, max) x (pass, final), then the wrapping `u32` sum (pass, final).
    assert_eq!(kernels::MODULE_DESCRIPTOR.kernels.len(), 20);
    assert_each_kernel_once(&kernels::MODULE_DESCRIPTOR);
}

#[test]
fn the_scan_golden_holds_every_kernel_once() {
    // The level-0 scan, the scan of totals, the top scan, the add-back into totals and the two final add-backs.
    assert_eq!(scan_kernels::MODULE_DESCRIPTOR.kernels.len(), 6);
    assert_each_kernel_once(&scan_kernels::MODULE_DESCRIPTOR);
}

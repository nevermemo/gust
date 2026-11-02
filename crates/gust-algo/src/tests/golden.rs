//! The reviewed expectation `tests/fixtures/reduce.slang`: the Slang the macro emits for every kernel of
//! `tests/fixtures/reduce.rs`, in declaration order, each under a marker line (the layout of `mismatch.slang`). A
//! translation change needs an explained golden diff and the GPU tests of this crate.

use std::{env, fs, path::Path};

use crate::kernels::MODULE_DESCRIPTOR;

const BLESS_VARIABLE: &str = "BLESS_GOLDEN";

fn emitted() -> String {
    MODULE_DESCRIPTOR
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

#[test]
fn reduce_golden() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/reduce.slang");
    let actual = emitted();
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
            "golden blessed; review `git diff tests/fixtures/reduce.slang` and rerun without {BLESS_VARIABLE}"
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
        "tests/fixtures/reduce.slang differs from the emitted Slang (first difference at {first_difference}); \
         rerun with {BLESS_VARIABLE}=1 to rewrite it and review the diff"
    );
}

#[test]
fn the_golden_holds_every_kernel_once() {
    // 3 element types x (sum, min, max) x (pass, final).
    assert_eq!(MODULE_DESCRIPTOR.kernels.len(), 18);
    let text = emitted();
    for kernel in MODULE_DESCRIPTOR.kernels {
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

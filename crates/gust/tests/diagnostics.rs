//! Slang failures that rustc and the GUST validator accept must still point at the
//! Rust line the faulty statement was written on (DX19 line table).
//!
//! The fixtures below are valid Rust and pass validation, so only `slangc` can
//! reject them. A local named `new` used to be one (a Slang syntax error where it is
//! used); the macro now renames it (`__gust_r_new`, M6 plan 2.5) and the kernel compiles, so
//! those fixtures keep their line tables and the failing tests take the emitted text
//! with the prefix deleted (`unrenamed`): the bare `new` is the same syntax error on the
//! same lines. A self-recursive helper is "recursion not allowed". The expected Rust line is
//! found by searching
//! this very file for a marker comment, so no line number is written down by hand.
//!
//! The renaming itself is covered at the end: the Slangc-tier table test compiles every
//! entry of the macro's collision table (`slang/collisions.rs`, included here by path) in
//! six name positions, with and without the rename, and `annotate_diagnostics` deleting the
//! prefix from slangc's messages is pinned. The D17 layout check on renamed names needs the native
//! reflection helper, so it is in `reflection.rs`.
//!
//! Both compile paths are covered: `slang::compile` and the two failure sites of
//! `reflect` (`reflect::reflect`, which runs `slangc`, and
//! `reflect::compile_reflected_with_helper`). The helper site runs a scripted
//! stand-in for the native reflection helper that fails with a Slang-shaped error
//! line, so this file needs only `slangc` on `PATH`, not the helper; the real
//! helper is exercised by the reflection tests.
//!
//! Two more fixtures, `HANDWRITTEN` and `HANDWRITTEN_SYNTAX`, are `KernelDescriptor` literals
//! with no macro behind them (an undeclared identifier and a syntax error in hand-written
//! Slang), so they keep failing in `slangc` even if the validator or the macro one day
//! accepts every construct above.
//!
//! The compile copy carries `#line` directives (M1.3), so the compiler's own
//! error location is the Rust file and line (`--> <file>:<line>:<column>`); the
//! tests check that raw text as well as the appended `originating Rust` lines.

use gust::{
    Access, GpuPod, Kernel, KernelDescriptor, LineSpan, ParameterDescriptor, ParameterKind,
    ResourceBinding, ResourceLayout, SourceOrigin, reflect,
    slang::{self, Error, Target},
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

const THIS_FILE: &str = include_str!("diagnostics.rs");

/// The macro's collision table, as plain data (the file has no `crate::` paths on purpose).
#[path = "../../gust-macros/src/slang/collisions.rs"]
mod collisions;

#[gust::gpu]
mod keyword_local {
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            let new = id.x;
            out[id.x] = new; // FAILS: keyword local
        }
    }
}

#[gust::gpu]
mod recursive_helper {
    fn count_down(n: uint) -> uint {
        if n == 0u32 {
            0u32
        } else {
            count_down(n - 1u32) // FAILS: recursion
        }
    }

    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            out[id.x] = count_down(id.x);
        }
    }
}

/// Valid on every target: a struct, a helper, a loop and a conditional, so the
/// directives of the compile copy fall in a struct body, a helper and nested blocks.
#[gust::gpu]
mod structured {
    pub struct Pair {
        pub low: uint,
        pub high: uint,
    }

    fn twice(value: uint) -> uint {
        value + value
    }

    #[allow(clippy::assign_op_pattern)]
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Pair>) {
        if id.x < out.len() {
            let mut total = 0u32;
            for step in 0u32..4u32 {
                total = total + twice(step);
            }
            out[id.x].low = total;
            out[id.x].high = total + 1u32;
        }
    }
}

/// Valid: a `Result` helper and a lowered `match` (scaffolding lines, nested arm
/// statements), so directives fall inside the `match` lowering.
#[gust::gpu]
mod lowered_match {
    fn half_if_even(value: uint) -> Result<uint, uint> {
        if value & 1u32 == 0u32 {
            Ok(value / 2u32)
        } else {
            Err(value)
        }
    }

    #[allow(clippy::assign_op_pattern)]
    #[kernel]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut values: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < input.len() {
            let result = half_if_even(input[i]);
            let mut total = 0u32;
            match result {
                Err(error) => {
                    total = total + error;
                }
                Ok(half) => {
                    total = total + half;
                }
            };
            values[i] = total;
        }
    }
}

/// Valid: struct literals in a local initializer and as a helper result.
#[gust::gpu]
mod struct_literal {
    pub struct Summary {
        pub total: uint,
        pub count: uint,
    }

    #[allow(clippy::let_and_return)]
    fn fold(acc: Summary, value: uint) -> Summary {
        let next = Summary {
            total: acc.total + value,
            count: acc.count + 1u32,
        };
        next
    }

    #[kernel]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<Summary>,
    ) {
        if id.x < out.len() {
            let start = Summary {
                total: 0u32,
                count: 0u32,
            };
            out[id.x] = fold(start, input[id.x]);
        }
    }
}

#[gust::gpu]
mod else_if_chain {
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            let new = id.x;
            // OWNS: else-if chain
            if id.x == 0u32 {
                out[id.x] = 1u32;
            } else if new == 1u32 {
                // FAILS: else-if condition
                out[id.x] = 2u32;
            } else {
                out[id.x] = 3u32;
            }
        }
    }
}

// Seven more Slang-only failures in other positions, so that the line table is
// exercised in loops, branches, helpers and at the top of a kernel.

#[gust::gpu]
mod keyword_in_loop {
    #[allow(clippy::assign_op_pattern)]
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            let mut total = 0u32;
            for step in 0u32..4u32 {
                let new = step;
                total = total + new; // FAILS: keyword in loop
            }
            out[id.x] = total;
        }
    }
}

#[gust::gpu]
mod keyword_in_else {
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            if id.x == 0u32 {
                out[id.x] = 1u32;
            } else {
                let new = id.x;
                out[id.x] = new; // FAILS: keyword in else
            }
        }
    }
}

#[gust::gpu]
mod keyword_at_top {
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        let new = id.x;
        out[0] = new; // FAILS: keyword at top
    }
}

#[gust::gpu]
mod keyword_in_helper {
    fn bump(value: uint) -> uint {
        let new = value;
        new + 1u32 // FAILS: keyword in helper
    }

    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            out[id.x] = bump(id.x);
        }
    }
}

#[gust::gpu]
mod recursion_with_locals {
    fn factorial(n: uint) -> uint {
        let base = n;
        let next = n - 1u32;
        if n <= 1u32 {
            1u32
        } else {
            base * factorial(next) // FAILS: recursion with locals
        }
    }

    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            out[id.x] = factorial(id.x);
        }
    }
}

#[gust::gpu]
mod keyword_in_nested_if {
    #[allow(clippy::collapsible_if)]
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            if id.x > 1u32 {
                if id.x > 2u32 {
                    let new = id.x;
                    out[id.x] = new + 1u32; // FAILS: keyword in nested if
                }
            }
        }
    }
}

#[gust::gpu]
mod recursion_in_loop {
    #[allow(clippy::assign_op_pattern)]
    fn spread(n: uint) -> uint {
        let mut total = 0u32;
        for i in 0u32..n {
            total = total + spread(i); // FAILS: recursion in loop
        }
        total
    }

    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            out[id.x] = spread(id.x);
        }
    }
}

// A hand-written descriptor (M1 risk 5). It never passes through `#[gpu]`, so no
// GUST validator can reject its Slang: the fixture keeps a failing kernel even
// when the validator learns to refuse the constructs above.

/// The Rust file the hand-written line table points at (relative, `/` only).
const HANDWRITTEN_FILE: &str = "crates/gust/tests/diagnostics.rs";

/// The faulty line of `HANDWRITTEN_SLANG` (1-based): an undeclared identifier.
const HANDWRITTEN_FAULTY_SLANG_LINE: u32 = 4;

const HANDWRITTEN_SLANG: &str = concat!(
    "[[vk::binding(0, 0)]] RWStructuredBuffer<uint> output;\n",
    "[shader(\"compute\")] [numthreads(64, 1, 1)]\n",
    "void handwritten_write(uint3 id : SV_DispatchThreadID) {\n",
    "    output[id.x] = missing_value;\n",
    "}\n",
);

const HANDWRITTEN_PARAMETERS: &[ParameterDescriptor] = &[
    ParameterDescriptor {
        name: "id",
        rust_type: "SV_DispatchThreadID",
        kind: ParameterKind::Invocation,
        access: Access::NotApplicable,
        resource_layout: None,
        binding: None,
    },
    ParameterDescriptor {
        name: "output",
        rust_type: "RWStructuredBuffer<uint>",
        kind: ParameterKind::Storage,
        access: Access::ReadWrite,
        resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
    },
];

/// Two records, outer construct first: the whole kernel maps to a line that is
/// not the marker, the faulty statement to the marked line (the innermost wins).
const HANDWRITTEN: KernelDescriptor = KernelDescriptor {
    module: "handwritten",
    name: "write",
    entry_point: "handwritten_write",
    workgroup_size: [64, 1, 1],
    parameters: HANDWRITTEN_PARAMETERS,
    slang_source: HANDWRITTEN_SLANG,
    source: SourceOrigin {
        file: HANDWRITTEN_FILE,
        lines: &[
            LineSpan {
                slang_start: 3,
                slang_end: 5,
                rust_line: line!(),
            },
            LineSpan {
                slang_start: HANDWRITTEN_FAULTY_SLANG_LINE,
                slang_end: HANDWRITTEN_FAULTY_SLANG_LINE,
                rust_line: line!(), // FAILS: handwritten
            },
        ],
    },
    artifacts: &[],
};

/// The same kernel with a file but an empty line table: nothing maps back.
const HANDWRITTEN_UNMAPPED: KernelDescriptor = KernelDescriptor {
    source: SourceOrigin {
        file: HANDWRITTEN_FILE,
        lines: &[],
    },
    ..HANDWRITTEN
};

/// A hand-written syntax error (a missing right-hand side) on line 4 of its Slang, with the same two-record table
/// shape as `HANDWRITTEN`. It stands in for the old `new` local as the syntax-error fixture: no macro behind it.
const HANDWRITTEN_SYNTAX_FAULTY_SLANG_LINE: u32 = 4;

const HANDWRITTEN_SYNTAX_SLANG: &str = concat!(
    "[[vk::binding(0, 0)]] RWStructuredBuffer<uint> output;\n",
    "[shader(\"compute\")] [numthreads(64, 1, 1)]\n",
    "void handwritten_syntax(uint3 id : SV_DispatchThreadID) {\n",
    "    output[id.x] = ;\n",
    "}\n",
);

const HANDWRITTEN_SYNTAX: KernelDescriptor = KernelDescriptor {
    entry_point: "handwritten_syntax",
    slang_source: HANDWRITTEN_SYNTAX_SLANG,
    source: SourceOrigin {
        file: HANDWRITTEN_FILE,
        lines: &[
            LineSpan {
                slang_start: 3,
                slang_end: 5,
                rust_line: line!(),
            },
            LineSpan {
                slang_start: HANDWRITTEN_SYNTAX_FAULTY_SLANG_LINE,
                slang_end: HANDWRITTEN_SYNTAX_FAULTY_SLANG_LINE,
                rust_line: line!(), // FAILS: handwritten syntax
            },
        ],
    },
    ..HANDWRITTEN
};

/// The macro renames `new` and the other colliding names (`__gust_r_<name>`), so the kernels below that were written
/// to fail on a local named `new` now compile. The failure tests need a Slang-only error on the same lines, so they
/// take the emitted text with the prefix deleted: that is the bare `new` again, a syntax error where it is used, and
/// the text keeps its line count and its line table. A source without the prefix is a bug in the test, not a pass.
fn unrenamed(kernel: &KernelDescriptor) -> KernelDescriptor {
    assert!(
        kernel.slang_source.contains(collisions::PREFIX),
        "`{}` has nothing to un-rename:\n{}",
        kernel.entry_point,
        kernel.slang_source
    );
    let text = kernel.slang_source.replace(collisions::PREFIX, "");
    KernelDescriptor {
        slang_source: Box::leak(text.into_boxed_str()),
        ..*kernel
    }
}

/// The 1-based line of this file whose code ends with the `// FAILS: <what>` marker.
fn marked_line(what: &str) -> u32 {
    line_ending_with(&format!("// FAILS: {what}"))
}

/// The 1-based line right after the `// OWNS: <what>` comment line of this file.
fn owner_line(what: &str) -> u32 {
    line_ending_with(&format!("// OWNS: {what}")) + 1
}

fn line_ending_with(marker: &str) -> u32 {
    let line = THIS_FILE
        .lines()
        .position(|line| line.trim_end().ends_with(marker))
        .unwrap_or_else(|| panic!("no line ends with `{marker}`"));
    u32::try_from(line + 1).unwrap()
}

/// The diagnostics of a failed compile, which must be an actual compiler failure.
fn failure(kernel: &KernelDescriptor, target: Target) -> String {
    match slang::compile(kernel, target) {
        Err(Error::CompilationFailed { diagnostics, .. }) => diagnostics,
        other => panic!("{target:?}: expected a Slang compile failure, got {other:?}"),
    }
}

/// The `file` and `line` of the `originating Rust source: file:line` annotation.
fn rust_source(diagnostics: &str) -> (&str, u32) {
    let line = diagnostics
        .lines()
        .find_map(|line| line.strip_prefix("originating Rust source: "))
        .unwrap_or_else(|| panic!("no Rust source annotation in:\n{diagnostics}"));
    let (file, number) = line.rsplit_once(':').expect("file:line");
    (file, number.parse().expect("line number"))
}

/// The compiler's own location is `--> <file>:<line>:<column>` in the Rust file,
/// where `file` is the one the descriptor recorded.
fn assert_compiler_location(diagnostics: &str, kernel: &KernelDescriptor, line: u32) {
    let location = format!("--> {}:{line}:", kernel.source.file);
    assert!(
        diagnostics.contains(&location),
        "no `{location}` in:\n{diagnostics}"
    );
}

fn assert_table_is_recorded(kernel: &KernelDescriptor) {
    let source = &kernel.source;
    assert!(
        source.file.ends_with("diagnostics.rs"),
        "file `{}`",
        source.file
    );
    assert!(
        !source.file.contains('\\'),
        "`{}` uses `/` only",
        source.file
    );
    // The path is never absolute and never leaks the build machine's layout.
    let manifest_dir = env!("CARGO_MANIFEST_DIR").replace('\\', "/");
    assert!(
        !std::path::Path::new(source.file).is_absolute()
            && !source.file.starts_with('/')
            && !source.file.contains(':'),
        "`{}` is relative",
        source.file
    );
    assert!(
        !source.file.contains(&manifest_dir),
        "`{}` does not embed the manifest directory",
        source.file
    );
    assert!(!source.lines.is_empty());
    for record in source.lines {
        assert!(record.slang_start >= 1 && record.slang_start <= record.slang_end);
        assert!(record.rust_line >= 1);
    }
    assert!(
        source
            .lines
            .windows(2)
            .all(|pair| pair[0].slang_start <= pair[1].slang_start),
        "records are in emission order"
    );
    assert!(
        source
            .lines
            .iter()
            .all(|record| (record.rust_line as usize) <= THIS_FILE.lines().count())
    );
}

#[test]
fn descriptors_record_the_defining_file_and_an_ordered_line_table() {
    assert_table_is_recorded(&keyword_local::run::DESCRIPTOR);
    assert_table_is_recorded(&recursive_helper::run::DESCRIPTOR);
    assert_eq!(
        <keyword_local::run as Kernel>::DESCRIPTOR.source,
        keyword_local::run::DESCRIPTOR.source
    );
}

#[test]
fn a_handwritten_syntax_error_names_the_rust_line() {
    let expected = marked_line("handwritten syntax");
    for target in [Target::Wgsl, Target::Spirv] {
        let diagnostics = failure(&HANDWRITTEN_SYNTAX, target);
        let (file, line) = rust_source(&diagnostics);
        assert_eq!(file, HANDWRITTEN_FILE, "{target:?}: {diagnostics}");
        assert_eq!(line, expected, "{target:?}: {diagnostics}");
        assert_compiler_location(&diagnostics, &HANDWRITTEN_SYNTAX, expected);
        assert!(
            diagnostics.contains("error[E20001]") || diagnostics.contains("syntax error"),
            "{target:?}: the failure is a syntax error: {diagnostics}"
        );
    }
}

#[test]
fn a_slang_only_syntax_error_in_a_kernel_names_the_rust_line() {
    let expected = marked_line("keyword local");
    let failing = unrenamed(&keyword_local::run::DESCRIPTOR);
    for target in [Target::Wgsl, Target::Spirv] {
        let diagnostics = failure(&failing, target);
        let (file, line) = rust_source(&diagnostics);
        assert!(
            file.ends_with("diagnostics.rs"),
            "{target:?}: {diagnostics}"
        );
        assert_eq!(line, expected, "{target:?}: {diagnostics}");
        assert_compiler_location(&diagnostics, &failing, expected);
        assert!(
            diagnostics.contains("originating Rust kernel: keyword_local::run"),
            "{target:?}: {diagnostics}"
        );
        assert!(
            diagnostics.contains("originating Rust construct: assignment #"),
            "{target:?}: {diagnostics}"
        );
    }
}

#[test]
fn a_slang_only_recursion_error_in_a_helper_names_the_rust_line() {
    let expected = marked_line("recursion");
    for target in [Target::Wgsl, Target::Spirv] {
        let diagnostics = failure(&recursive_helper::run::DESCRIPTOR, target);
        assert!(
            diagnostics.contains("recursion"),
            "{target:?}: the failure is Slang's recursion rule: {diagnostics}"
        );
        let (file, line) = rust_source(&diagnostics);
        assert!(
            file.ends_with("diagnostics.rs"),
            "{target:?}: {diagnostics}"
        );
        assert_eq!(line, expected, "{target:?}: {diagnostics}");
        assert_compiler_location(&diagnostics, &recursive_helper::run::DESCRIPTOR, expected);
    }
}

/// Ten deliberate Slang-only failures with the Rust line each must name. The
/// plan's kill rule (M1.3) is that fewer than nine of ten landing on the right
/// Rust line stops the slice; here every one must, on both targets.
fn deliberate_failures() -> Vec<(&'static str, KernelDescriptor, u32)> {
    vec![
        (
            "keyword local",
            unrenamed(&keyword_local::run::DESCRIPTOR),
            marked_line("keyword local"),
        ),
        (
            "recursion",
            recursive_helper::run::DESCRIPTOR,
            marked_line("recursion"),
        ),
        (
            "else-if chain",
            unrenamed(&else_if_chain::run::DESCRIPTOR),
            owner_line("else-if chain"),
        ),
        (
            "keyword in loop",
            unrenamed(&keyword_in_loop::run::DESCRIPTOR),
            marked_line("keyword in loop"),
        ),
        (
            "keyword in else",
            unrenamed(&keyword_in_else::run::DESCRIPTOR),
            marked_line("keyword in else"),
        ),
        (
            "keyword at top",
            unrenamed(&keyword_at_top::run::DESCRIPTOR),
            marked_line("keyword at top"),
        ),
        (
            "keyword in helper",
            unrenamed(&keyword_in_helper::run::DESCRIPTOR),
            marked_line("keyword in helper"),
        ),
        (
            "recursion with locals",
            recursion_with_locals::run::DESCRIPTOR,
            marked_line("recursion with locals"),
        ),
        (
            "keyword in nested if",
            unrenamed(&keyword_in_nested_if::run::DESCRIPTOR),
            marked_line("keyword in nested if"),
        ),
        (
            "recursion in loop",
            recursion_in_loop::run::DESCRIPTOR,
            marked_line("recursion in loop"),
        ),
    ]
}

#[test]
fn ten_deliberate_failures_land_on_the_right_rust_line_on_each_target() {
    let failures = deliberate_failures();
    assert_eq!(failures.len(), 10);
    let mut right = 0;
    let mut wrong = Vec::new();
    for target in [Target::Wgsl, Target::Spirv] {
        for (name, kernel, expected) in &failures {
            let diagnostics = failure(kernel, target);
            let annotated = rust_source(&diagnostics);
            let compiler = format!("--> {}:{expected}:", kernel.source.file);
            if annotated.1 == *expected && diagnostics.contains(&compiler) {
                right += 1;
            } else {
                wrong.push(format!(
                    "{name} on {target:?}: expected line {expected}, got {}\n{diagnostics}",
                    annotated.1
                ));
            }
        }
    }
    println!(
        "{right} of {} deliberate failures on the right Rust line",
        failures.len() * 2
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn directives_change_neither_the_artifacts_nor_the_exported_text() {
    // `structured`: a struct body, a helper and nested blocks. `lowered_match`: the
    // multi-line `match` lowering. `struct_literal`: struct values built in place.
    let kernels = [
        ("structured", structured::run::DESCRIPTOR, "struct Pair"),
        (
            "lowered_match",
            lowered_match::run::DESCRIPTOR,
            "__gust_match_",
        ),
        (
            "struct_literal",
            struct_literal::run::DESCRIPTOR,
            "struct Summary",
        ),
    ];
    let directory = slang::TemporaryDirectory::create().unwrap();
    for (name, kernel, shape) in kernels {
        // The shape under test is really in the emitted text, and has records.
        assert!(
            kernel.slang_source.contains(shape),
            "{name}: no `{shape}` in:\n{}",
            kernel.slang_source
        );
        assert!(
            kernel.source.lines.len() >= 6,
            "{name}: {:?}",
            kernel.source.lines
        );
        assert!(!kernel.slang_source.contains("#line"));
        for target in [Target::Wgsl, Target::Spirv] {
            let with_directives = slang::compile(&kernel, target).unwrap();
            let clean =
                slang::compile_source(kernel.slang_source, kernel.entry_point, target).unwrap();
            assert!(!clean.is_empty());
            assert_eq!(with_directives, clean, "{name} {target:?}");
        }
        // The exported text is the descriptor's own, byte for byte.
        let path = directory.path().join(format!("{name}.slang"));
        slang::write_source(&kernel, &path).unwrap();
        assert_eq!(
            fs::read_to_string(path).unwrap(),
            kernel.slang_source,
            "{name}"
        );
    }
}

#[test]
fn a_descriptor_without_a_line_table_reports_no_rust_source() {
    let mut descriptor = unrenamed(&keyword_local::run::DESCRIPTOR);
    descriptor.source = gust::SourceOrigin::UNKNOWN;
    let diagnostics = failure(&descriptor, Target::Wgsl);
    assert!(
        !diagnostics.contains("originating Rust source"),
        "{diagnostics}"
    );
    // The compiler read the clean text: its location is in the generated Slang.
    assert!(diagnostics.contains("kernel.slang:"), "{diagnostics}");
    assert!(!diagnostics.contains("diagnostics.rs"), "{diagnostics}");
    // The kernel and construct attribution of DX19 is unchanged.
    assert!(
        diagnostics.contains("originating Rust kernel: keyword_local::run"),
        "{diagnostics}"
    );
}

#[test]
fn an_error_in_an_else_if_condition_names_the_line_of_the_owning_if() {
    let owner = owner_line("else-if chain");
    // The `else if` header is the line above its marker comment.
    let else_if = marked_line("else-if condition") - 1;
    assert_eq!(
        else_if,
        owner + 2,
        "the fixture's else-if sits two lines down"
    );
    let failing = unrenamed(&else_if_chain::run::DESCRIPTOR);
    let table = else_if_chain::run::DESCRIPTOR.source;
    // The table has no record of its own for the `else if` header: the owning
    // `if` statement's record covers the whole chain, so no record names the
    // else-if's Rust line.
    assert!(
        table.lines.iter().all(|record| record.rust_line != else_if),
        "an `else if` header must not get its own record: {:?}",
        table.lines
    );
    let chain = table
        .lines
        .iter()
        .find(|record| record.rust_line == owner)
        .expect("a record for the owning if");
    assert!(chain.slang_end > chain.slang_start);
    for target in [Target::Wgsl, Target::Spirv] {
        let diagnostics = failure(&failing, target);
        let (file, line) = rust_source(&diagnostics);
        assert!(
            file.ends_with("diagnostics.rs"),
            "{target:?}: {diagnostics}"
        );
        assert_eq!(line, owner, "{target:?}: {diagnostics}");
        assert_ne!(line, else_if, "{target:?}: {diagnostics}");
        assert_compiler_location(&diagnostics, &failing, owner);
        // The reflection path reports the same line.
        let diagnostics = reflect_failure(&failing, target);
        assert_eq!(
            rust_source(&diagnostics).1,
            owner,
            "{target:?}: {diagnostics}"
        );
    }
}

// ---------------------------------------------------------------------------
// The reflection path (DX19: both compile paths annotate alike).

/// The diagnostics of a failed `reflect::reflect` (it runs `slangc` itself).
fn reflect_failure(kernel: &KernelDescriptor, target: Target) -> String {
    match reflect::reflect(kernel, target) {
        Err(reflect::Error::Compiler(Error::CompilationFailed { diagnostics, .. })) => diagnostics,
        other => panic!("{target:?}: expected a reflect compile failure, got {other:?}"),
    }
}

/// The diagnostics of a failed `reflect::compile_reflected_with_helper` whose
/// helper is a script that fails with a `kernel.slang(<slang_line>)` error.
fn helper_failure(kernel: &KernelDescriptor, target: Target, slang_line: u32) -> String {
    let helper = scripted_failing_helper(slang_line);
    let result = reflect::compile_reflected_with_helper(kernel, target, &helper);
    let _ = fs::remove_file(&helper);
    match result {
        Err(reflect::Error::Compiler(Error::CompilationFailed { diagnostics, .. })) => diagnostics,
        other => panic!("{target:?}: expected a helper compile failure, got {other:?}"),
    }
}

/// An executable that prints a Slang-shaped error at `slang_line` of
/// `kernel.slang` to stderr and exits with status 1, like the real helper does
/// for a Slang compile error.
fn scripted_failing_helper(slang_line: u32) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&directory).unwrap();
    let name = format!(
        "scripted-helper-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let message = format!("kernel.slang({slang_line}): error 1: scripted failure");
    #[cfg(windows)]
    {
        let path = directory.join(format!("{name}.bat"));
        fs::write(
            &path,
            format!("@echo off\r\necho {message} 1>&2\r\nexit /b 1\r\n"),
        )
        .unwrap();
        path
    }
    #[cfg(not(windows))]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = directory.join(format!("{name}.sh"));
        fs::write(&path, format!("#!/bin/sh\necho '{message}' >&2\nexit 1\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }
}

/// The first Slang line generated from the given Rust line of this file.
fn slang_line_of(kernel: &KernelDescriptor, rust_line: u32) -> u32 {
    kernel
        .source
        .lines
        .iter()
        .rev()
        .find(|record| record.rust_line == rust_line)
        .unwrap_or_else(|| panic!("no record for Rust line {rust_line}"))
        .slang_start
}

fn without_source(kernel: &KernelDescriptor) -> KernelDescriptor {
    let mut descriptor = *kernel;
    descriptor.source = SourceOrigin::UNKNOWN;
    descriptor
}

/// A known origin: the kernel and the Rust source line, the kernel exactly once.
fn assert_known_origin(diagnostics: &str, kernel: &str, rust_line: u32, context: &str) {
    assert_eq!(
        diagnostics.matches("originating Rust kernel: ").count(),
        1,
        "{context}: {diagnostics}"
    );
    assert!(
        diagnostics.contains(&format!("originating Rust kernel: {kernel}")),
        "{context}: {diagnostics}"
    );
    let (file, line) = rust_source(diagnostics);
    assert!(file.ends_with("diagnostics.rs"), "{context}: {diagnostics}");
    assert_eq!(line, rust_line, "{context}: {diagnostics}");
}

/// An unknown origin gets exactly the text from before the line table: the
/// compiler output, then the kernel name as the last line, and no construct or
/// source line. `preceding` are the prefixes of the compiler's trailer lines
/// just before the kernel line, in order.
fn assert_unknown_origin(diagnostics: &str, kernel: &str, preceding: &[&str], context: &str) {
    let lines: Vec<&str> = diagnostics.lines().collect();
    assert_eq!(
        lines.last().copied(),
        Some(format!("originating Rust kernel: {kernel}").as_str()),
        "{context}: {diagnostics}"
    );
    let trailer = &lines[lines.len() - 1 - preceding.len()..lines.len() - 1];
    for (line, prefix) in trailer.iter().zip(preceding) {
        assert!(line.starts_with(prefix), "{context}: {diagnostics}");
    }
    assert_eq!(
        diagnostics.matches("originating Rust").count(),
        1,
        "no construct or source line: {context}: {diagnostics}"
    );
}

#[test]
fn the_reflect_path_names_the_kernel_construct_and_rust_line() {
    let keyword_local = unrenamed(&keyword_local::run::DESCRIPTOR);
    let cases = [
        (
            &keyword_local,
            "keyword_local::run",
            marked_line("keyword local"),
        ),
        // The error is in a helper above the first kernel marker, which has a
        // Rust line but no construct: the kernel is still named, once.
        (
            &recursive_helper::run::DESCRIPTOR,
            "recursive_helper::run",
            marked_line("recursion"),
        ),
    ];
    for target in [Target::Wgsl, Target::Spirv] {
        for (kernel, name, expected) in cases {
            let diagnostics = reflect_failure(kernel, target);
            assert_known_origin(&diagnostics, name, expected, &format!("{target:?} {name}"));
        }
        let diagnostics = reflect_failure(&keyword_local, target);
        assert!(
            diagnostics.contains("originating Rust construct: assignment #"),
            "{target:?}: {diagnostics}"
        );
    }
}

#[test]
fn the_reflect_path_without_a_line_table_keeps_the_text_from_before_the_table() {
    let keyword_local = unrenamed(&keyword_local::run::DESCRIPTOR);
    for target in [Target::Wgsl, Target::Spirv] {
        for (kernel, name) in [
            (&keyword_local, "keyword_local::run"),
            (&recursive_helper::run::DESCRIPTOR, "recursive_helper::run"),
        ] {
            let diagnostics = reflect_failure(&without_source(kernel), target);
            assert_unknown_origin(
                &diagnostics,
                name,
                &["compiler exit: "],
                &format!("{target:?} {name}"),
            );
        }
    }
}

#[test]
fn the_helper_path_names_the_kernel_construct_and_rust_line() {
    for target in [Target::Wgsl, Target::Spirv] {
        let kernel = &keyword_local::run::DESCRIPTOR;
        let expected = marked_line("keyword local");
        let diagnostics = helper_failure(kernel, target, slang_line_of(kernel, expected));
        assert_known_origin(
            &diagnostics,
            "keyword_local::run",
            expected,
            &format!("{target:?}"),
        );
        assert!(
            diagnostics.contains("originating Rust construct: assignment #"),
            "{target:?}: {diagnostics}"
        );
        assert!(diagnostics.contains("reflection helper: "), "{diagnostics}");

        // A helper above the first kernel marker: a Rust line, but no construct.
        let kernel = &recursive_helper::run::DESCRIPTOR;
        let expected = marked_line("recursion");
        let diagnostics = helper_failure(kernel, target, slang_line_of(kernel, expected));
        assert_known_origin(
            &diagnostics,
            "recursive_helper::run",
            expected,
            &format!("{target:?}"),
        );
    }
}

#[test]
fn the_helper_path_without_a_line_table_keeps_the_text_from_before_the_table() {
    for target in [Target::Wgsl, Target::Spirv] {
        let kernel = &keyword_local::run::DESCRIPTOR;
        // The Slang line comes from the table, which the descriptor then drops.
        let line = slang_line_of(kernel, marked_line("keyword local"));
        let diagnostics = helper_failure(&without_source(kernel), target, line);
        assert_unknown_origin(
            &diagnostics,
            "keyword_local::run",
            &["compiler exit: ", "reflection helper: "],
            &format!("{target:?}"),
        );
        assert!(
            diagnostics.contains("kernel.slang(") && diagnostics.contains("scripted failure"),
            "{target:?}: {diagnostics}"
        );
    }
}

// ---------------------------------------------------------------------------
// The hand-written descriptor (M1 risk 5): a failing fixture that never passes
// through the macro, so no validator change can remove it.

#[test]
fn a_handwritten_descriptor_fixture_is_what_the_tests_assume() {
    let source = HANDWRITTEN.source;
    assert_eq!(source.file, HANDWRITTEN_FILE);
    // The hand-written path tracks the file the macro records.
    assert_eq!(HANDWRITTEN_FILE, keyword_local::run::DESCRIPTOR.source.file);
    assert_table_is_recorded(&HANDWRITTEN);
    assert_eq!(source.lines.len(), 2);
    // The faulty Slang line is the undeclared identifier.
    let faulty = HANDWRITTEN
        .slang_source
        .lines()
        .nth(HANDWRITTEN_FAULTY_SLANG_LINE as usize - 1);
    assert_eq!(faulty, Some("    output[id.x] = missing_value;"));
    // The outer record is another Rust line: only the innermost record names the marker.
    let marker = marked_line("handwritten");
    assert_eq!(source.lines[1].rust_line, marker);
    assert_ne!(source.lines[0].rust_line, marker);
    assert_eq!(
        slang_line_of(&HANDWRITTEN, marker),
        HANDWRITTEN_FAULTY_SLANG_LINE
    );
    assert_eq!(HANDWRITTEN_UNMAPPED.source.file, HANDWRITTEN_FILE);
    assert!(HANDWRITTEN_UNMAPPED.source.lines.is_empty());
    assert_eq!(HANDWRITTEN_UNMAPPED.slang_source, HANDWRITTEN.slang_source);
}

#[test]
fn a_handwritten_descriptor_failure_names_the_marked_rust_line_on_every_path() {
    let expected = marked_line("handwritten");
    for target in [Target::Wgsl, Target::Spirv] {
        let context = format!("{target:?}");
        // `slang::compile` (real slangc).
        let diagnostics = failure(&HANDWRITTEN, target);
        assert!(
            diagnostics.contains("undefined identifier 'missing_value'"),
            "{context}: the failure is the undeclared identifier: {diagnostics}"
        );
        let (file, line) = rust_source(&diagnostics);
        assert_eq!(file, HANDWRITTEN_FILE, "{context}: {diagnostics}");
        assert_eq!(line, expected, "{context}: {diagnostics}");
        assert_compiler_location(&diagnostics, &HANDWRITTEN, expected);
        // No `// @rust kernel:` marker was written by hand: no construct is named.
        assert!(
            !diagnostics.contains("originating Rust construct"),
            "{context}: {diagnostics}"
        );

        // `reflect::reflect` (real slangc).
        let diagnostics = reflect_failure(&HANDWRITTEN, target);
        assert_known_origin(&diagnostics, "handwritten::write", expected, &context);
        assert_compiler_location(&diagnostics, &HANDWRITTEN, expected);
        assert!(
            diagnostics.contains("undefined identifier 'missing_value'"),
            "{context}: {diagnostics}"
        );

        // `reflect::compile_reflected_with_helper` with the scripted stand-in
        // helper (not the native helper): it reports the faulty Slang line from
        // the table, and the annotation must map it back.
        let line = slang_line_of(&HANDWRITTEN, expected);
        let diagnostics = helper_failure(&HANDWRITTEN, target, line);
        assert_known_origin(&diagnostics, "handwritten::write", expected, &context);
        assert!(diagnostics.contains("reflection helper: "), "{diagnostics}");
    }
}

#[test]
fn a_handwritten_descriptor_without_a_line_table_reports_the_slang_line() {
    let location = format!("kernel.slang:{HANDWRITTEN_FAULTY_SLANG_LINE}:");
    for target in [Target::Wgsl, Target::Spirv] {
        let context = format!("{target:?}");
        let kernel = &HANDWRITTEN_UNMAPPED;

        let diagnostics = failure(kernel, target);
        assert!(
            diagnostics.contains(&location),
            "{context}: no `{location}` in:\n{diagnostics}"
        );
        assert!(
            diagnostics.contains("undefined identifier 'missing_value'"),
            "{context}: {diagnostics}"
        );
        assert!(
            !diagnostics.contains("originating Rust source"),
            "{context}: {diagnostics}"
        );
        assert!(
            !diagnostics.contains("diagnostics.rs"),
            "{context}: {diagnostics}"
        );

        // `reflect::reflect`: the compiler text, then the kernel name, and no source line.
        let diagnostics = reflect_failure(kernel, target);
        assert!(
            diagnostics.contains(&location),
            "{context}: no `{location}` in:\n{diagnostics}"
        );
        assert_unknown_origin(
            &diagnostics,
            "handwritten::write",
            &["compiler exit: "],
            &context,
        );

        // The scripted helper reports the same Slang line.
        let diagnostics = helper_failure(kernel, target, HANDWRITTEN_FAULTY_SLANG_LINE);
        assert_unknown_origin(
            &diagnostics,
            "handwritten::write",
            &["compiler exit: ", "reflection helper: "],
            &context,
        );
        assert!(
            diagnostics.contains(&format!("kernel.slang({HANDWRITTEN_FAULTY_SLANG_LINE})")),
            "{context}: {diagnostics}"
        );
    }
}

// ---------------------------------------------------------------------------
// Collision-only identifier renaming (M6 plan 2.5). The table is the macro's own data file; the macro's Unit tests
// pin that it is emitted at every name site, and these tests compile it with slangc.

/// The seven kernels above that declare a local named `new`.
fn kernels_with_a_local_named_new() -> [(&'static str, KernelDescriptor); 7] {
    [
        ("keyword_local", keyword_local::run::DESCRIPTOR),
        ("else_if_chain", else_if_chain::run::DESCRIPTOR),
        ("keyword_in_loop", keyword_in_loop::run::DESCRIPTOR),
        ("keyword_in_else", keyword_in_else::run::DESCRIPTOR),
        ("keyword_at_top", keyword_at_top::run::DESCRIPTOR),
        ("keyword_in_helper", keyword_in_helper::run::DESCRIPTOR),
        (
            "keyword_in_nested_if",
            keyword_in_nested_if::run::DESCRIPTOR,
        ),
    ]
}

#[test]
fn a_local_named_new_is_emitted_renamed_and_compiles_on_each_target() {
    for (name, kernel) in kernels_with_a_local_named_new() {
        assert!(
            kernel.slang_source.contains("var __gust_r_new = "),
            "{name}: the local is emitted as __gust_r_new:\n{}",
            kernel.slang_source
        );
        for target in [Target::Wgsl, Target::Spirv] {
            slang::compile(&kernel, target)
                .unwrap_or_else(|error| panic!("{name} on {target:?}: {error:?}"));
        }
        // And the bare `new` is what slangc refuses, on the same lines (the rest of this file relies on it).
        for target in [Target::Wgsl, Target::Spirv] {
            failure(&unrenamed(&kernel), target);
        }
    }
}

/// slangc itself, with no GUST annotation: the raw message text.
fn raw_slangc(source: &str, entry: &str, target: Target) -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&directory).unwrap();
    let stem = format!(
        "raw-slangc-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let input = directory.join(format!("{stem}.slang"));
    let output = directory.join(format!("{stem}.out"));
    fs::write(&input, source).unwrap();
    let target_name = match target {
        Target::Wgsl => "wgsl",
        Target::Spirv => "spirv",
    };
    let result = std::process::Command::new("slangc")
        .arg(&input)
        .args([
            "-target",
            target_name,
            "-entry",
            entry,
            "-stage",
            "compute",
            "-o",
        ])
        .arg(&output)
        .output()
        .expect("slangc runs");
    let _ = fs::remove_file(&input);
    let _ = fs::remove_file(&output);
    assert!(
        !result.status.success(),
        "the raw compile was meant to fail"
    );
    format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    )
}

/// An error that names the renamed local reaches the user with the Rust spelling, on both compile paths, and its
/// Rust line is the one it had before.
#[test]
fn slangc_messages_name_the_rust_spelling_not_the_prefix() {
    assert_eq!(collisions::PREFIX, "__gust_r_");
    let kernel = keyword_local::run::DESCRIPTOR;
    let text = kernel
        .slang_source
        .replacen("var __gust_r_new = ", "var __gust_r_nex = ", 1);
    assert_ne!(text, kernel.slang_source);
    let broken = KernelDescriptor {
        slang_source: Box::leak(text.clone().into_boxed_str()),
        ..kernel
    };
    let expected = marked_line("keyword local");
    for target in [Target::Wgsl, Target::Spirv] {
        // Without the annotation the compiler says `__gust_r_new`, so the assertions below are not vacuous.
        let raw = raw_slangc(&text, kernel.entry_point, target);
        assert!(raw.contains("'__gust_r_new'"), "{target:?}: {raw}");
        for (path, diagnostics) in [
            ("compile", failure(&broken, target)),
            ("reflect", reflect_failure(&broken, target)),
        ] {
            let context = format!("{path} on {target:?}: {diagnostics}");
            assert!(
                diagnostics.contains("undefined identifier 'new'"),
                "{context}"
            );
            assert!(!diagnostics.contains("__gust_r_"), "{context}");
            assert_eq!(rust_source(&diagnostics).1, expected, "{context}");
        }
    }
}

/// Runs `work` over `jobs` on a few threads and returns the results in job order.
fn run_parallel<T: Sync, R: Send>(jobs: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else {
                        break;
                    };
                    let result = work(job);
                    results.lock().unwrap().push((index, result));
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

/// Whether slangc takes `source`; `Err` carries its first error line.
fn slangc_takes(source: &str, target: Target) -> Result<(), String> {
    match slang::compile_source(source, "gpu_table_run", target) {
        Ok(_) => Ok(()),
        Err(Error::CompilationFailed { diagnostics, .. }) => Err(diagnostics
            .lines()
            .find(|line| line.contains("error"))
            .unwrap_or("no error line")
            .to_owned()),
        Err(other) => panic!("{target:?}: slangc did not run: {other:?}"),
    }
}

const BOTH_TARGETS: [Target; 2] = [Target::Wgsl, Target::Spirv];

/// (i) Every entry, renamed, in all six positions on wgsl, and on spirv in its recorded failing positions and the
/// local position (slangc's two targets agreed on every probed name and position, S6.3 and S6.3b; this keeps the
/// run short).
#[test]
fn every_table_entry_compiles_renamed_in_every_position() {
    let mut jobs = Vec::new();
    for entry in collisions::COLLISIONS {
        for position in collisions::Position::ALL {
            jobs.push((entry.name, position, Target::Wgsl));
            if position == collisions::Position::Local || entry.fails_in.contains(&position) {
                jobs.push((entry.name, position, Target::Spirv));
            }
        }
    }
    assert!(jobs.len() > collisions::COLLISIONS.len() * 6);
    let results = run_parallel(&jobs, |&(name, position, target)| {
        let spelled = collisions::emit_name(name);
        slangc_takes(&collisions::position_source(position, &spelled), target)
    });
    let failures: Vec<String> = jobs
        .iter()
        .zip(results)
        .filter_map(|((name, position, target), result)| {
            result
                .err()
                .map(|message| format!("{name} as {} on {target:?}: {message}", position.label()))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// (ii) The measured class-1 record, re-measured on every run: the bare name must fail in exactly the recorded
/// positions on wgsl, and in each recorded position on spirv too. The record is the probe's, not the table's, so a
/// dropped or mis-recorded table entry fails here (and a Slang upgrade that stops rejecting a name turns the row into
/// a documented retirement, D22).
#[test]
fn class_1_names_fail_bare_in_exactly_their_recorded_positions() {
    let mut jobs = Vec::new();
    for (name, labels) in collisions::MEASURED_CLASS1 {
        assert!(
            collisions::collides(name),
            "the measured class-1 name `{name}` is not renamed: it is not in the table"
        );
        for position in collisions::Position::ALL {
            let recorded = labels.contains(&position.probe_label());
            jobs.push((*name, position, Target::Wgsl, recorded));
            if recorded {
                jobs.push((*name, position, Target::Spirv, recorded));
            }
        }
    }
    let results = run_parallel(&jobs, |&(name, position, target, _)| {
        slangc_takes(&collisions::position_source(position, name), target)
    });
    let wrong: Vec<String> = jobs
        .iter()
        .zip(results)
        .filter_map(|((name, position, target, recorded), result)| {
            let failed = result.is_err();
            (failed != *recorded).then(|| {
                format!(
                    "{name} as {} on {target:?}: recorded {}, slangc {}",
                    position.label(),
                    if *recorded { "failing" } else { "fine" },
                    if failed { "fails" } else { "takes it" },
                )
            })
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Whether the bare name of a class-2 entry captures silently (the program compiles and means something else).
fn silent_capture(name: &str) -> bool {
    collisions::SILENT_CAPTURES
        .iter()
        .any(|(silent, _, _)| *silent == name)
}

/// (iii) Class 2: the bare name captures the construct GUST emits and fails; the renamed one compiles. Every emitted
/// entry has such a text but `__GustResult`, which E0901 denies to users.
#[test]
fn class_2_names_capture_the_emitted_construct_until_they_are_renamed() {
    let mut jobs = Vec::new();
    for entry in collisions::COLLISIONS.iter().filter(|entry| entry.emitted) {
        for target in BOTH_TARGETS {
            let bare = collisions::capture_source(entry.name, entry.name);
            if bare.is_none() {
                assert_eq!(
                    entry.name, "__GustResult",
                    "no capture text for {}",
                    entry.name
                );
                continue;
            }
            jobs.push((entry.name, target, false));
            jobs.push((entry.name, target, true));
        }
    }
    assert!(jobs.len() >= 50, "{}", jobs.len());
    let results = run_parallel(&jobs, |&(name, target, renamed)| {
        let spelled = if renamed {
            collisions::emit_name(name).into_owned()
        } else {
            name.to_owned()
        };
        slangc_takes(
            &collisions::capture_source(name, &spelled).expect("checked above"),
            target,
        )
    });
    let wrong: Vec<String> = jobs
        .iter()
        .zip(results)
        .filter_map(
            |((name, target, renamed), result)| match (renamed, result) {
                (true, Err(message)) => Some(format!("{name} renamed on {target:?}: {message}")),
                (false, Ok(())) if !silent_capture(name) => Some(format!(
                    "{name} bare on {target:?}: no capture, the entry is dead weight"
                )),
                _ => None,
            },
        )
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// (iii-b) `true` and `false` are captured silently: the bare program compiles, and slangc folds the condition to the
/// other value. The tests above only see failures, so this one compares the generated code.
#[test]
fn true_and_false_captured_by_a_local_silently_change_the_generated_code() {
    let wgsl = |name: &str, spelled: &str| -> String {
        let source = collisions::capture_source(name, spelled).expect("a capture text");
        let bytes = slang::compile_source(&source, "gpu_table_run", Target::Wgsl)
            .unwrap_or_else(|error| panic!("{name} as `{spelled}`: {error:?}"));
        String::from_utf8(bytes).unwrap()
    };
    assert_eq!(collisions::SILENT_CAPTURES.len(), 2);
    for (name, bare_has, renamed_has) in collisions::SILENT_CAPTURES {
        let bare = wgsl(name, name);
        let renamed = wgsl(name, &collisions::emit_name(name));
        assert!(
            bare.contains(bare_has) && !bare.contains(renamed_has),
            "bare `{name}` holds the other value:\n{bare}"
        );
        assert!(
            renamed.contains(renamed_has) && !renamed.contains(bare_has),
            "renamed `{name}` keeps the literal:\n{renamed}"
        );
    }
}

/// The statement keywords and `inout` that GUST writes into every program are not looked up in scope: a Rust name
/// of that spelling declared as a local, a helper, a struct or a global leaves the generated code byte for byte what
/// it is without it. (`var`, `true`, `false` and the types are different, and are in the table.) A struct or global
/// of a table name fails at its declaration, which is class 1, so those positions are only tried for the others.
#[test]
fn the_statement_keywords_gust_writes_cannot_be_captured_by_a_user_name() {
    const KERNEL: &str = "[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid gpu_table_run(uint3 gid : SV_DispatchThreadID)\n";
    let program = |declaration: &str, local: &str| {
        format!(
            "void bump(inout uint a, uint b) {{ a = a ^ b; }}\n{declaration}RWStructuredBuffer<uint> ob;\n{KERNEL}{{ {local}\
             uint total = 0u; for (var i = 0u; i < 2u; i++) {{ if (i == 1u) {{ continue; }} else {{ bump(total, i); }} \
             if (i > 5u) {{ break; }} }} ob[gid.x] = total; if (total > 9u) {{ return; }} }}\n"
        )
    };
    let wgsl = |source: &str| -> String {
        let bytes = slang::compile_source(source, "gpu_table_run", Target::Wgsl)
            .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
        String::from_utf8(bytes).unwrap()
    };
    let baseline = wgsl(&program("", ""));
    for name in ["if", "else", "for", "break", "continue", "return", "inout"] {
        let mut declarations = vec![
            ("local", program("", &format!("var {name} = 1u; "))),
            (
                "helper",
                program(&format!("uint {name}() {{ return 1u; }}\n"), ""),
            ),
        ];
        if !collisions::collides(name) {
            declarations.push((
                "struct",
                program(&format!("struct {name} {{ uint v; }};\n"), ""),
            ));
            declarations.push((
                "global",
                program(
                    &format!("[[vk::binding(1, 0)]] RWStructuredBuffer<uint> {name};\n"),
                    "",
                ),
            ));
        }
        for (position, source) in declarations {
            assert_eq!(wgsl(&source), baseline, "`{name}` as a {position}");
        }
    }
}

/// Class 3 stays empty: the Rust keywords that are not in the table (spelled `r#..`, the only way to write most of
/// them) are plain names to slangc in all six positions, so nothing else needs a rename.
#[test]
fn rust_keywords_outside_the_table_compile_bare_in_every_position() {
    let keywords = [
        "as", "async", "await", "const", "dyn", "else", "extern", "fn", "impl", "in", "let",
        "loop", "match", "mod", "move", "mut", "pub", "ref", "static", "trait", "type", "unsafe",
        "use", "where", "abstract", "become", "box", "final", "macro", "override", "priv",
        "typeof", "unsized", "virtual", "yield", "gen", "union", "auto",
    ];
    let mut jobs = Vec::new();
    for name in keywords {
        assert!(!collisions::collides(name), "{name} is in the table");
        for position in collisions::Position::ALL {
            jobs.push((name, position));
        }
    }
    let results = run_parallel(&jobs, |&(name, position)| {
        slangc_takes(&collisions::position_source(position, name), Target::Wgsl)
    });
    let failures: Vec<String> = jobs
        .iter()
        .zip(results)
        .filter_map(|((name, position), result)| {
            result
                .err()
                .map(|message| format!("{name} as {}: {message}", position.label()))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The `vectors` golden compiles with the pinned slangc to both targets (m6.m1). `compile_source` passes the same
/// `-matrix-layout-column-major` flag as every compile of a kernel. The golden holds the integer `dot`, which Slang
/// emits as `OpSDot` or `OpUDot` with `Capability DotProduct` (slangc 2026.13.1 and 2026.20), and naga's SPIR-V reader
/// refuses that capability. Since slice m6.m2 the macro lowers `.dot` in such a module to `__gust_dot`, whose integer
/// overloads are a sum of products, so the SPIR-V is pinned to declare `Shader` and nothing else (a change of Slang
/// or of the lowering that brings a capability back shows here). The naga halves (WGSL and SPIR-V with empty
/// capabilities) were run by hand: they need a `naga` dependency in the manifests, which this slice may not touch.
#[test]
fn the_vectors_golden_compiles_to_wgsl_and_spirv() {
    let golden = include_str!("../../../tests/fixtures/vectors.slang").replace("\r\n", "\n");
    let wgsl = slang::compile_source(&golden, "gpu_vectors_run", Target::Wgsl).unwrap();
    let wgsl = String::from_utf8(wgsl).unwrap();
    assert!(wgsl.contains("@compute"), "{wgsl}");
    let spirv = slang::compile_source(&golden, "gpu_vectors_run", Target::Spirv).unwrap();
    // `Shader` is capability 1; `DotProduct` is 6019 and `DotProductInputAll` 6016.
    assert_eq!(spirv_capabilities(&spirv), [1]);
}

/// The `OpCapability` operands of a SPIR-V module, found by walking the instructions after the header.
fn spirv_capabilities(spirv: &[u8]) -> Vec<u32> {
    assert!(
        spirv.len().is_multiple_of(4) && spirv.len() > 20,
        "{} bytes",
        spirv.len()
    );
    let words: Vec<u32> = spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    assert_eq!(words[0], 0x0723_0203, "not SPIR-V");
    let mut capabilities = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let (opcode, length) = (words[at] & 0xffff, (words[at] >> 16) as usize);
        assert!(length > 0, "malformed SPIR-V at word {at}");
        if opcode == 17 {
            capabilities.push(words[at + 1]);
        }
        at += length;
    }
    capabilities
}

/// The `products` golden (every operand pair of the product table written as `a * b`, the `*=` rows, and the literal
/// operands, lowered by the macro) compiles with the pinned slangc to both targets, and its SPIR-V declares `Shader`
/// and nothing else. A pair the table lists but Slang cannot resolve fails here, before it can be a slangc error in a
/// user kernel (M6 plan 2.1, the coverage test); `products_golden_covers_every_row_of_the_table` in the macro
/// crate pins that the golden holds every row.
/// The naga halves were run by hand (no `naga` dependency in the manifests).
#[test]
fn the_products_golden_compiles_to_wgsl_and_spirv() {
    let golden = include_str!("../../../tests/fixtures/products.slang").replace("\r\n", "\n");
    let wgsl = slang::compile_source(&golden, "gpu_products_run", Target::Wgsl).unwrap();
    assert!(String::from_utf8(wgsl).unwrap().contains("@compute"));
    let spirv = slang::compile_source(&golden, "gpu_products_run", Target::Spirv).unwrap();
    assert_eq!(spirv_capabilities(&spirv), [1]);
}

/// The `coretests` golden compiles with the pinned slangc to both targets (m5.d3). The golden holds every kernel's
/// source under a `// ==== kernel <name> ====` marker, each with its own prelude (so the whole file is not one
/// Slang module); each section is compiled on its own for its `gpu_coretests_<name>` entry. The naga halves are
/// not part of this test: the GPU test of m5.d2 builds the WGSL through wgpu.
#[test]
fn the_coretests_golden_compiles_to_wgsl_and_spirv() {
    let golden = include_str!("../../../tests/fixtures/coretests.slang").replace("\r\n", "\n");
    let sections: Vec<(&str, &str)> = golden
        .split("// ==== kernel ")
        .skip(1)
        .map(|section| section.split_once(" ====\n").unwrap())
        .collect();
    let names: Vec<&str> = sections.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, ["ints", "uints", "floats", "bools", "casts"]);
    for (name, section) in &sections {
        let entry = format!("gpu_coretests_{name}");
        let wgsl = slang::compile_source(section, &entry, Target::Wgsl).unwrap();
        let wgsl = String::from_utf8(wgsl).unwrap();
        assert!(wgsl.contains("@compute"), "{name}: {wgsl}");
        let spirv = slang::compile_source(section, &entry, Target::Spirv).unwrap();
        assert!(
            spirv.len().is_multiple_of(4) && spirv.len() > 20,
            "{name}: {} bytes",
            spirv.len()
        );
        let magic = u32::from_le_bytes(spirv[..4].try_into().unwrap());
        assert_eq!(magic, 0x0723_0203, "{name}: not SPIR-V");
    }
}

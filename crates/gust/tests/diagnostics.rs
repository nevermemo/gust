//! Slang failures that rustc and the GUST validator accept must still point at the
//! Rust line the faulty statement was written on (DX19 line table).
//!
//! The fixtures below are valid Rust and pass validation, so only `slangc` can
//! reject them: a local named `new` is a Slang syntax error where it is used, and
//! a self-recursive helper is "recursion not allowed". The expected Rust line is
//! found by searching
//! this very file for a marker comment, so no line number is written down by hand.
//!
//! Both compile paths are covered: `slang::compile` and the two failure sites of
//! `reflect` (`reflect::reflect`, which runs `slangc`, and
//! `reflect::compile_reflected_with_helper`). The helper site runs a scripted
//! stand-in for the native reflection helper that fails with a Slang-shaped error
//! line, so this file needs only `slangc` on `PATH`, not the helper; the real
//! helper is exercised by the reflection tests.
//!
//! One more fixture, `HANDWRITTEN`, is a `KernelDescriptor` literal with no macro
//! behind it (an undeclared identifier in hand-written Slang), so it keeps failing
//! in `slangc` even if the validator one day rejects every construct above.
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
fn a_slang_only_syntax_error_in_a_kernel_names_the_rust_line() {
    let expected = marked_line("keyword local");
    for target in [Target::Wgsl, Target::Spirv] {
        let diagnostics = failure(&keyword_local::run::DESCRIPTOR, target);
        let (file, line) = rust_source(&diagnostics);
        assert!(
            file.ends_with("diagnostics.rs"),
            "{target:?}: {diagnostics}"
        );
        assert_eq!(line, expected, "{target:?}: {diagnostics}");
        assert_compiler_location(&diagnostics, &keyword_local::run::DESCRIPTOR, expected);
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
            keyword_local::run::DESCRIPTOR,
            marked_line("keyword local"),
        ),
        (
            "recursion",
            recursive_helper::run::DESCRIPTOR,
            marked_line("recursion"),
        ),
        (
            "else-if chain",
            else_if_chain::run::DESCRIPTOR,
            owner_line("else-if chain"),
        ),
        (
            "keyword in loop",
            keyword_in_loop::run::DESCRIPTOR,
            marked_line("keyword in loop"),
        ),
        (
            "keyword in else",
            keyword_in_else::run::DESCRIPTOR,
            marked_line("keyword in else"),
        ),
        (
            "keyword at top",
            keyword_at_top::run::DESCRIPTOR,
            marked_line("keyword at top"),
        ),
        (
            "keyword in helper",
            keyword_in_helper::run::DESCRIPTOR,
            marked_line("keyword in helper"),
        ),
        (
            "recursion with locals",
            recursion_with_locals::run::DESCRIPTOR,
            marked_line("recursion with locals"),
        ),
        (
            "keyword in nested if",
            keyword_in_nested_if::run::DESCRIPTOR,
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
    let mut descriptor = keyword_local::run::DESCRIPTOR;
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
        let diagnostics = failure(&else_if_chain::run::DESCRIPTOR, target);
        let (file, line) = rust_source(&diagnostics);
        assert!(
            file.ends_with("diagnostics.rs"),
            "{target:?}: {diagnostics}"
        );
        assert_eq!(line, owner, "{target:?}: {diagnostics}");
        assert_ne!(line, else_if, "{target:?}: {diagnostics}");
        assert_compiler_location(&diagnostics, &else_if_chain::run::DESCRIPTOR, owner);
        // The reflection path reports the same line.
        let diagnostics = reflect_failure(&else_if_chain::run::DESCRIPTOR, target);
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
    let cases = [
        (
            &keyword_local::run::DESCRIPTOR,
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
        let diagnostics = reflect_failure(&keyword_local::run::DESCRIPTOR, target);
        assert!(
            diagnostics.contains("originating Rust construct: assignment #"),
            "{target:?}: {diagnostics}"
        );
    }
}

#[test]
fn the_reflect_path_without_a_line_table_keeps_the_text_from_before_the_table() {
    for target in [Target::Wgsl, Target::Spirv] {
        for (kernel, name) in [
            (&keyword_local::run::DESCRIPTOR, "keyword_local::run"),
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

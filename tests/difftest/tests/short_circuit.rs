//! `&&` and `||` short-circuit: a right operand that would read out of bounds is skipped when the left operand decides
//! the result (m5.c1b, R1-4).
//!
//! The program is `out[i] = (i < 24) && (a[i + 40] == 7)` (and the `||` twin, `(i >= 24) || (a[i + 40] == 7)`) over
//! 64 elements with a 64-element input (every buffer has the length of the dispatch), so the right operand would read
//! out of bounds for the elements 24 and above. The interpreter skips it (its domain tracking would otherwise reject
//! the program), the CPU twin skips it (Rust `&&`), and the GPU must neither fault nor fail validation.
//!
//! A value comparison cannot show that the GPU skipped the read: wgpu's robust buffer access makes an out-of-bounds read
//! return a value, and the operands are pure, so evaluating both sides gives the same result. The GPU test therefore
//! also reads the WGSL that the kernel compiles to and checks that the buffer read sits inside the right operand of a
//! WGSL `&&`/`||` (which the WGSL specification defines as short-circuit) and was not hoisted out of it into a statement
//! of its own or a `select` (which evaluates both sides). The CPU test needs neither slangc nor a GPU; the GPU test is
//! `#[ignore]`d (Gpu tier) and runs on the machine's adapter:
//! `cargo test --locked --manifest-path tests/difftest/Cargo.toml --test short_circuit -- --ignored`.

use std::path::{Path, PathBuf};

use difftest::ast::{BinOp, Expr, InputBuf, Program, Ty};
use difftest::classify::classify;
use difftest::emit::module_name;
use difftest::emit::package_name;
use difftest::harness::{Options, execute_plan};
use difftest::interp::run_full;
use difftest::runsrc::Planned;
use difftest::shapes::Shape;

const COUNT: u32 = 64;
const LEN: usize = 64;
/// The elements below this index may read; the others would read out of bounds.
const BOUND: u32 = 24;
const OFFSET: u32 = 40;

fn input() -> InputBuf {
    InputBuf {
        ty: Ty::U32,
        data: (0..LEN as u32)
            .map(|i| if i % 3 == 0 { 7 } else { i + 100 })
            .collect(),
    }
}

fn program(seed: u64, or: bool) -> Program {
    let index = Expr::bin(BinOp::Add, Expr::Gid, Expr::u32(OFFSET));
    let element = Expr::input(0, Ty::U32, index);
    let seven = Expr::bin(BinOp::Eq, element, Expr::u32(7));
    let result = if or {
        Expr::bin(
            BinOp::Or,
            Expr::bin(BinOp::Ge, Expr::Gid, Expr::u32(BOUND)),
            seven,
        )
    } else {
        Expr::bin(
            BinOp::And,
            Expr::bin(BinOp::Lt, Expr::Gid, Expr::u32(BOUND)),
            seven,
        )
    };
    Program {
        seed,
        attempt: if or { 2 } else { 1 },
        count: COUNT,
        inputs: vec![input()],
        body: Vec::new(),
        result,
    }
}

fn planned(program: Program, index: usize) -> Planned {
    let run =
        run_full(&program).expect("the skipped right operand must keep the program in the domain");
    Planned {
        program,
        index,
        expected: run.values,
        intervals: run.intervals,
    }
}

fn target_dir() -> PathBuf {
    let dir = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("difftest")
            .join("target"),
    };
    std::fs::create_dir_all(&dir).expect("create the target directory");
    std::path::absolute(dir).expect("absolute target directory")
}

fn options(cpu_only: bool) -> Options {
    Options {
        seeds: Vec::new(),
        cpu_only,
        shapes: vec![Shape::Grid],
        target_dir: target_dir(),
        repo_root: std::path::absolute(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .expect("absolute repository root"),
    }
}

/// The expected output by hand: `i < 24 && a[i + 40] == 7`.
fn by_hand() -> Vec<u32> {
    (0..COUNT)
        .map(|i| u32::from(i < BOUND && (i + OFFSET).is_multiple_of(3)))
        .collect()
}

fn run_both(seed: u64, cpu_only: bool, wgsl_dir: Option<&Path>) {
    let plan = vec![
        planned(program(seed, false), 0),
        planned(program(seed, true), 1),
    ];
    // The `||` program is true from element 24 on (nothing is read there) and below it where the element is 7.
    let or_by_hand: Vec<u32> = (0..COUNT)
        .map(|i| u32::from(i >= BOUND || (i + OFFSET).is_multiple_of(3)))
        .collect();
    assert_eq!(plan[0].expected, by_hand());
    assert_eq!(plan[1].expected, or_by_hand);

    // Its own key, never the one that `difftest-run --seeds <seed>` would use for the same number.
    let name = format!("short-circuit-{seed}");
    let package = package_name(&name).unwrap();
    let env: Vec<(&str, &Path)> = wgsl_dir
        .map(|d| ("DIFFTEST_WGSL_OUT", d))
        .into_iter()
        .collect();
    let (raws, _) = execute_plan(&options(cpu_only), &package, &name, &plan, &env).unwrap();
    assert_eq!(raws.len(), 2);
    for (planned, raw) in plan.iter().zip(&raws) {
        let found = classify(planned, raw, cpu_only);
        assert!(
            found.is_empty(),
            "{}: {found:?}",
            module_name(&planned.program)
        );
        assert_eq!(raw.twin.as_ref(), Ok(&planned.expected));
        if !cpu_only {
            assert_eq!(raw.gpu.as_ref(), Some(&Ok(planned.expected.clone())));
            assert!(raw.wgsl_validated);
        }
    }
}

#[test]
fn the_twin_skips_an_out_of_bounds_right_operand() {
    run_both(9001, true, None);
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// What guards the one read of the input buffer `a_0` in the WGSL of the kernel: the text on the left of a `&&` or
/// `||` that has the read on its right, or the condition of the `if` (or of the `if` that the `else` belongs to)
/// whose block holds the read. Slang lowers `&&` and `||` of a statement context to such an `if` and `else`, which
/// is also a short circuit, so both shapes count.
fn guard_of_the_read(wgsl: &str) -> Result<String, String> {
    let lines: Vec<&str> = wgsl.lines().collect();
    let reads: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].contains("a_0["))
        .collect();
    let [read] = reads[..] else {
        return Err(format!(
            "expected one read of a_0, found {}:
{wgsl}",
            reads.len()
        ));
    };
    let line = lines[read];
    for operator in ["&&", "||"] {
        if let Some(at) = line.find(operator) {
            let (left, right) = line.split_at(at);
            if right.contains("a_0[") && !left.contains("a_0[") {
                return Ok(format!("{operator} operand: {}", left.trim()));
            }
        }
    }
    let depth = indent(line);
    let open = (0..read)
        .rev()
        .find(|&i| !lines[i].trim().is_empty() && indent(lines[i]) < depth)
        .ok_or("the read is not inside a block")?;
    let mut header = open.checked_sub(1).ok_or("the block has no header line")?;
    let mut arm = "then";
    if lines[header].trim_start().starts_with("else") {
        arm = "else";
        let level = indent(lines[header]);
        header = (0..header)
            .rev()
            .find(|&i| indent(lines[i]) == level && lines[i].trim_start().starts_with("if"))
            .ok_or("an else without its if")?;
    }
    Ok(format!("{arm} arm of {}", lines[header].trim()))
}

#[test]
#[ignore = "Gpu tier: needs a Vulkan adapter, slangc and the native reflection helper"]
fn the_gpu_skips_an_out_of_bounds_right_operand() {
    let dir = target_dir().join("difftest-wgsl-short-circuit");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    run_both(9002, false, Some(&dir));

    for or in [false, true] {
        let module = module_name(&program(9002, or));
        let path = dir.join(format!("{module}.wgsl"));
        let wgsl =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let guard = guard_of_the_read(&wgsl).unwrap_or_else(|e| panic!("{module}: {e}"));
        println!("{module}: the read of a is in the {guard}");
        // The guard is the left operand (the comparison with the bound), never the read itself, and nothing evaluates
        // both operands unconditionally.
        assert!(
            guard.contains(&format!("u32({BOUND})")),
            "{module}: {guard}"
        );
        assert!(!guard.contains("a_0["), "{module}: {guard}");
        assert!(!wgsl.contains("select("), "{module}: {wgsl}");
    }
}

#[test]
fn the_structural_check_tells_a_guarded_read_from_a_hoisted_one() {
    let guarded = "{\n    if(i_0 < (_S1.x))\n    {\n        var _S2 : bool;\n        if(i_0 < u32(24))\n        {\n            _S2 = a_0[i_0 + u32(40)] == u32(7);\n        }\n        else\n        {\n            _S2 = false;\n        }\n    }\n}\n";
    assert_eq!(
        guard_of_the_read(guarded).unwrap(),
        "then arm of if(i_0 < u32(24))"
    );
    // Both operands evaluated first and combined afterwards: the read sits in the outer block, whose guard is the
    // bound check of the kernel, not the left operand.
    let hoisted = "{\n    if(i_0 < (_S1.x))\n    {\n        var _S2 : bool = a_0[i_0 + u32(40)] == u32(7);\n        var _S3 : bool = i_0 < u32(24);\n    }\n}\n";
    assert!(!guard_of_the_read(hoisted).unwrap().contains("u32(24)"));
    // A read in an operator expression: the guard is the text on its left.
    let operator = "{\n    var _S2 : bool = (i_0 < u32(24)) && (a_0[i_0] == u32(7));\n}\n";
    assert_eq!(
        guard_of_the_read(operator).unwrap(),
        "&& operand: var _S2 : bool = (i_0 < u32(24))"
    );
    assert!(guard_of_the_read("{\n}\n").is_err());
}

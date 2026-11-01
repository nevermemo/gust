//! The Rust-source emitter: AST -> a `#[gust::gpu]` kernel crate.
//!
//! Every nested operation is parenthesised (an operand that is not an atom gets parentheses), so the text does
//! not depend on operator precedence, and the dialect's traps (`a as i32 < b` parses as a generic) cannot occur.
//! The tests re-parse the emitted text with Rust's own precedence (`reparse.rs`) and require the same AST back,
//! which is what catches a dropped parenthesis; `cargo check` of a generated crate (tests/roundtrip.rs) proves the
//! text is accepted by the `#[gpu]` validator and by rustc.
//!
//! The emitted text uses `\n` line endings whatever the platform, so the same seed gives the same bytes everywhere.

use std::fmt::Write as _;
use std::io;
use std::path::Path;

use crate::ast::{Expr, Program, Stmt, Ty, UnOp};
use crate::generator::DIFFTEST_FORMAT;

/// The lockfile every generated crate is built with (`--locked`). It is generated from `template/Cargo.toml`,
/// which `manifest("../../..", CRATE_NAME)` reproduces byte for byte (a unit test checks both).
pub const TEMPLATE_LOCK: &str = include_str!("../template/Cargo.lock");

/// The package name of the template crate, which is the one root package in `TEMPLATE_LOCK`. A generated crate gets its
/// own name (`CRATE_NAME` plus a suffix): crates that share a target directory and a package name also share a
/// fingerprint (cargo hashes a workspace member's path relative to its own workspace root, which is `.` for every
/// generated crate), and one crate's build then makes another look fresh.
pub const CRATE_NAME: &str = "difftest-case";

/// Input buffer names, by buffer index.
pub const BUFFER_NAMES: [&str; 3] = ["a", "b", "c"];

const INDENT: &str = "    ";

/// The module name of a program: seed and attempt identify it.
pub fn module_name(program: &Program) -> String {
    format!("s{}_a{}", program.seed, program.attempt)
}

/// The exact decimal expansion of a finite `f32`, with a decimal point and the suffix `f32` (`0.5f32`, `-1048576.0f32`,
/// `0.000030517578125f32`). Every `f32` is a dyadic rational, so its decimal expansion is finite and rustc, the
/// re-parser and Slang all read it back to the same bits (no shortest-round-trip digits that could sit on a rounding
/// boundary). Spelling: an exact decimal, not `f32::from_bits`, because the dialect has no function calls.
pub fn f32_literal(bits: u32) -> String {
    let value = f32::from_bits(bits);
    assert!(value.is_finite(), "non-finite f32 literal");
    let negative = bits >> 31 == 1;
    let exponent_field = ((bits >> 23) & 0xFF) as i32;
    let fraction = bits & 0x7F_FFFF;
    // value = mantissa * 2^exponent
    let (mantissa, exponent) = if exponent_field == 0 {
        (fraction, -149)
    } else {
        (fraction | 0x80_0000, exponent_field - 150)
    };
    // Decimal digits of `mantissa * 2^exponent`: scale by 2 for a positive exponent, by 5 and move the point for a
    // negative one (2^-k = 5^k / 10^k).
    let mut digits: Vec<u8> = mantissa.to_string().bytes().map(|b| b - b'0').collect();
    let mut places = 0;
    let (factor, times) = if exponent >= 0 {
        (2, exponent)
    } else {
        (5, -exponent)
    };
    for _ in 0..times {
        let mut carry = 0;
        for digit in digits.iter_mut().rev() {
            let product = *digit * factor + carry;
            *digit = product % 10;
            carry = product / 10;
        }
        while carry > 0 {
            digits.insert(0, carry % 10);
            carry /= 10;
        }
    }
    if exponent < 0 {
        places = -exponent as usize;
    }
    while digits.len() <= places {
        digits.insert(0, 0);
    }
    let split = digits.len() - places;
    let integer: String = digits[..split]
        .iter()
        .map(|d| char::from(b'0' + d))
        .collect();
    let integer = integer.trim_start_matches('0');
    let integer = if integer.is_empty() { "0" } else { integer };
    let fractional: String = digits[split..]
        .iter()
        .map(|d| char::from(b'0' + d))
        .collect();
    let fractional = fractional.trim_end_matches('0');
    let fractional = if fractional.is_empty() {
        "0"
    } else {
        fractional
    };
    format!(
        "{}{integer}.{fractional}f32",
        if negative { "-" } else { "" }
    )
}

fn literal(ty: Ty, bits: u32) -> String {
    match ty {
        Ty::F32 => f32_literal(bits),
        Ty::Bool => (bits != 0).to_string(),
        Ty::U32 => format!("{bits}u32"),
        Ty::I32 => {
            let value = bits as i32;
            if value < 0 {
                format!("-{}i32", value.unsigned_abs())
            } else {
                format!("{value}i32")
            }
        }
    }
}

/// Whether the text of `expr` needs no parentheses as an operand.
fn is_atom(expr: &Expr) -> bool {
    match expr {
        Expr::Lit(Ty::I32, bits) => (*bits as i32) >= 0,
        Expr::Lit(Ty::F32, bits) => bits >> 31 == 0,
        Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) | Expr::Input { .. } => true,
        Expr::Un { .. } | Expr::Bin { .. } | Expr::Cast { .. } => false,
    }
}

fn operand(expr: &Expr) -> String {
    if is_atom(expr) {
        emit_expr(expr)
    } else {
        format!("({})", emit_expr(expr))
    }
}

/// The text of an expression at statement level (no outer parentheses).
pub fn emit_expr(expr: &Expr) -> String {
    match expr {
        Expr::Lit(ty, bits) => literal(*ty, *bits),
        Expr::Var(var, _) => format!("v{var}"),
        Expr::Gid => "i".to_owned(),
        Expr::Len(buf) => format!("{}.len()", BUFFER_NAMES[*buf]),
        Expr::Input { buf, index, .. } => format!("{}[{}]", BUFFER_NAMES[*buf], emit_expr(index)),
        Expr::Un { op, e, .. } => {
            let symbol = match op {
                UnOp::Neg => "-",
                UnOp::Not => "!",
            };
            format!("{symbol}{}", operand(e))
        }
        Expr::Bin { op, l, r, .. } => format!("{} {} {}", operand(l), op.symbol(), operand(r)),
        Expr::Cast { to, e } => format!("{} as {}", operand(e), to.name()),
    }
}

fn emit_stmt(stmt: &Stmt, level: usize, out: &mut String) {
    let pad = INDENT.repeat(level);
    match stmt {
        Stmt::Let {
            var,
            ty,
            mutable,
            init,
        } => {
            let keyword = if *mutable { "let mut" } else { "let" };
            let _ = writeln!(
                out,
                "{pad}{keyword} v{var}: {} = {};",
                ty.name(),
                emit_expr(init)
            );
        }
        Stmt::Assign { var, value } => {
            let _ = writeln!(out, "{pad}v{var} = {};", emit_expr(value));
        }
        Stmt::If { cond, then, els } => {
            let _ = writeln!(out, "{pad}if {} {{", emit_expr(cond));
            emit_block(then, level + 1, out);
            if els.is_empty() {
                let _ = writeln!(out, "{pad}}}");
            } else {
                let _ = writeln!(out, "{pad}}} else {{");
                emit_block(els, level + 1, out);
                let _ = writeln!(out, "{pad}}}");
            }
        }
        Stmt::For { var, trips, body } => {
            let _ = writeln!(out, "{pad}for v{var} in 0u32..{trips}u32 {{");
            emit_block(body, level + 1, out);
            let _ = writeln!(out, "{pad}}}");
        }
    }
}

fn emit_block(stmts: &[Stmt], level: usize, out: &mut String) {
    for stmt in stmts {
        emit_stmt(stmt, level, out);
    }
}

/// The kernel crate's text for one program: a documented `#[gust::gpu]` module with the kernel `k`.
pub fn emit_module(program: &Program) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "/// Seed {}, attempt {}: {} elements, output {}.",
        program.seed,
        program.attempt,
        program.count,
        program.out_ty().name()
    );
    let _ = writeln!(out, "#[gust::gpu]");
    let _ = writeln!(out, "pub mod {} {{", module_name(program));
    let _ = writeln!(out, "{INDENT}#[kernel(workgroup_size(64, 1, 1))]");
    let _ = writeln!(out, "{INDENT}pub fn k(");
    let _ = writeln!(out, "{INDENT}{INDENT}id: SV_DispatchThreadID,");
    for (k, input) in program.inputs.iter().enumerate() {
        let _ = writeln!(
            out,
            "{INDENT}{INDENT}{}: StructuredBuffer<{}>,",
            BUFFER_NAMES[k],
            input.ty.dialect_name()
        );
    }
    let _ = writeln!(
        out,
        "{INDENT}{INDENT}mut out: RWStructuredBuffer<{}>,",
        program.out_ty().dialect_name()
    );
    let _ = writeln!(out, "{INDENT}) {{");
    let _ = writeln!(out, "{INDENT}{INDENT}let i = id.x;");
    let _ = writeln!(out, "{INDENT}{INDENT}if i < out.len() {{");
    emit_block(&program.body, 3, &mut out);
    let pad = INDENT.repeat(3);
    if program.result.ty() == Ty::Bool {
        let _ = writeln!(out, "{pad}if {} {{", emit_expr(&program.result));
        let _ = writeln!(out, "{pad}{INDENT}out[i] = 1u32;");
        let _ = writeln!(out, "{pad}}} else {{");
        let _ = writeln!(out, "{pad}{INDENT}out[i] = 0u32;");
        let _ = writeln!(out, "{pad}}}");
    } else {
        let _ = writeln!(out, "{pad}out[i] = {};", emit_expr(&program.result));
    }
    let _ = writeln!(out, "{INDENT}{INDENT}}}");
    let _ = writeln!(out, "{INDENT}}}");
    let _ = writeln!(out, "}}");
    out
}

/// The `src/lib.rs` of a generated crate holding all `programs`.
///
/// The crate-level `allow` is needed because rustc's constant propagation denies, by default, an overflowing or
/// zero-dividing operation it can see statically, even in a branch that no input reaches; the interpreter's
/// domain filter is per input, not per path.
pub fn emit_crate_source(programs: &[Program]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// Generated by difftest (DIFFTEST_FORMAT {DIFFTEST_FORMAT}). Do not edit; regenerate from the seed."
    );
    let _ = writeln!(
        out,
        "#![allow(arithmetic_overflow, unconditional_panic, unused_variables, unused_assignments, unused_mut, unused_parens, dead_code)]"
    );
    for program in programs {
        out.push('\n');
        out.push_str(&emit_module(program));
    }
    out
}

/// The `Cargo.toml` of a generated crate. `root` is the TOML string content of the repository root (a path with
/// forward slashes, or `../../..` for the committed template).
pub fn manifest(root: &str, package: &str) -> String {
    format!(
        "# Generated by difftest. Do not edit.\n\
[package]\n\
name = \"{package}\"\n\
version = \"0.0.0\"\n\
edition = \"2024\"\n\
publish = false\n\
\n\
[lib]\n\
path = \"src/lib.rs\"\n\
\n\
[dependencies]\n\
gust = {{ path = \"{root}/crates/gust\" }}\n\
gust-wgpu = {{ path = \"{root}/crates/gust-wgpu\", features = [\"slang-runtime\"] }}\n\
\n\
# `--cfg gust_dual` and `gust_profile` are set by cargo xtask and by gust's build script, not by this manifest.\n\
[lints.rust]\n\
unexpected_cfgs = {{ level = \"warn\", check-cfg = ['cfg(gust_dual)', 'cfg(gust_profile, values(\"dev\", \"native\", \"web\", \"cpu\"))'] }}\n\
\n\
# A workspace of its own, so a parent directory's manifest never claims the crate.\n\
[workspace]\n"
    )
}

/// A path as the content of a TOML basic string: forward slashes, no Windows verbatim prefix, nothing that would
/// need escaping. A path that cannot be written safely is an error, never a silently altered path.
pub fn path_for_toml(path: &Path) -> Result<String, String> {
    let text = path
        .to_str()
        .ok_or_else(|| "the repository path is not valid UTF-8".to_owned())?;
    let mut text = text.replace('\\', "/");
    if let Some(rest) = text.strip_prefix("//?/UNC/") {
        text = format!("//{rest}");
    } else if let Some(rest) = text.strip_prefix("//?/") {
        text = rest.to_owned();
    }
    while text.len() > 1 && text.ends_with('/') {
        text.pop();
    }
    if text.is_empty() {
        return Err("the repository path is empty".into());
    }
    if text.chars().any(|c| c == '"' || c.is_control()) {
        return Err(format!(
            "the repository path {text:?} cannot be written into a TOML string"
        ));
    }
    Ok(text)
}

/// The package name of a generated crate: `CRATE_NAME` and a suffix of lowercase letters, digits and hyphens.
pub fn package_name(suffix: &str) -> Result<String, String> {
    if suffix.is_empty()
        || !suffix
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err(format!(
            "package suffix {suffix:?} must be lowercase letters, digits and hyphens"
        ));
    }
    Ok(format!("{CRATE_NAME}-{suffix}"))
}

/// The template lockfile for a crate named `package`: `TEMPLATE_LOCK` (as `\n` text) with the root package's name
/// replaced, which is the only difference `--locked` has to accept.
pub fn lock_for(package: &str) -> Result<String, String> {
    if package.is_empty()
        || !package
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err(format!(
            "package name {package:?} is not lowercase letters, digits and hyphens"
        ));
    }
    let lock = TEMPLATE_LOCK.replace("\r\n", "\n");
    let root = format!("[[package]]\nname = \"{CRATE_NAME}\"\nversion = \"0.0.0\"\n");
    if lock.matches(&root).count() != 1 {
        return Err("the template lockfile does not hold the root package exactly once".into());
    }
    Ok(lock.replacen(
        &root,
        &format!("[[package]]\nname = \"{package}\"\nversion = \"0.0.0\"\n"),
        1,
    ))
}

/// Writes a generated crate named `package` into `dir` (created, with `src/`): `Cargo.toml` for the repository at
/// `repo_root`, the template lockfile for that name, and `src/lib.rs`. Existing files are overwritten.
pub fn write_crate(
    dir: &Path,
    package: &str,
    programs: &[Program],
    repo_root: &Path,
) -> io::Result<()> {
    let root = path_for_toml(repo_root).map_err(io::Error::other)?;
    let lock = lock_for(package).map_err(io::Error::other)?;
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(dir.join("Cargo.toml"), manifest(&root, package))?;
    std::fs::write(dir.join("Cargo.lock"), lock)?;
    std::fs::write(dir.join("src").join("lib.rs"), emit_crate_source(programs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::ast::{BinOp, InputBuf};
    use crate::generator::generate_batch;
    use crate::reparse::reparse;

    fn var(n: u32, ty: Ty) -> Expr {
        Expr::Var(n, ty)
    }

    fn u(v: u32) -> Expr {
        Expr::u32(v)
    }

    #[test]
    fn f32_literals_are_exact_decimals_that_read_back_to_the_same_bits() {
        // Exact text of a few values.
        for (value, text) in [
            (0.5f32, "0.5f32"),
            (1.0, "1.0f32"),
            (-1.0, "-1.0f32"),
            (1_048_576.0, "1048576.0f32"),
            (0.1, "0.100000001490116119384765625f32"),
            (-2.5, "-2.5f32"),
            (0.0, "0.0f32"),
            (2f32.powi(-20), "0.00000095367431640625f32"),
        ] {
            assert_eq!(f32_literal(value.to_bits()), text);
        }
        // Read-back: edges of the bounded input range and of the interval range, then a pseudo-random sweep over
        // every exponent.
        let mut bits: Vec<u32> = [
            2f32.powi(-100),
            2f32.powi(100),
            2f32.powi(-20),
            2f32.powi(20),
            f32::MIN_POSITIVE,
            f32::MAX,
            16_777_216.0,
            1.0 - f32::EPSILON / 2.0,
            1.0 + f32::EPSILON,
            0.3,
            1.0 / 3.0,
            1e-7,
        ]
        .iter()
        .flat_map(|v| [v.to_bits(), (-v).to_bits()])
        .collect();
        let mut prng = crate::prng::Prng::new(0xF32);
        for _ in 0..5000 {
            let bits32 = prng.next_u32();
            if f32::from_bits(bits32).is_finite() {
                bits.push(bits32);
            }
        }
        for bits32 in bits {
            let text = f32_literal(bits32);
            let digits = text.strip_suffix("f32").expect("suffix");
            let back: f32 = digits.parse().expect("parses");
            assert_eq!(back.to_bits(), bits32, "{text}");
        }
    }

    #[test]
    fn operands_are_parenthesised_unless_they_are_atoms() {
        let a0 = Expr::input(0, Ty::U32, Expr::Gid);
        let cases: Vec<(Expr, &str)> = vec![
            (u(7), "7u32"),
            (Expr::i32(-5), "-5i32"),
            (Expr::i32(5), "5i32"),
            (Expr::bool(true), "true"),
            (Expr::i32_min(), "(-2147483647i32) - 1i32"),
            (a0.clone(), "a[i]"),
            (Expr::input(1, Ty::I32, Expr::Len(1)), "b[b.len()]"),
            (Expr::bin(BinOp::Add, a0.clone(), u(1)), "a[i] + 1u32"),
            // The shift is an operand of `+`: its parentheses are what keeps `(a << b) + c` from reading `a << (b + c)`.
            (
                Expr::bin(
                    BinOp::Add,
                    Expr::bin(BinOp::Shl, var(0, Ty::U32), u(3)),
                    var(1, Ty::U32),
                ),
                "(v0 << 3u32) + v1",
            ),
            (
                Expr::bin(
                    BinOp::Shl,
                    var(0, Ty::U32),
                    Expr::bin(BinOp::Add, var(1, Ty::U32), u(3)),
                ),
                "v0 << (v1 + 3u32)",
            ),
            // A cast operand of `<` must be parenthesised or `i32 < b` parses as a generic argument list.
            (
                Expr::bin(
                    BinOp::Lt,
                    Expr::cast(Ty::I32, var(0, Ty::U32)),
                    Expr::i32(0),
                ),
                "(v0 as i32) < 0i32",
            ),
            (Expr::cast(Ty::U32, Expr::i32(-1)), "(-1i32) as u32"),
            (
                Expr::cast(
                    Ty::U32,
                    Expr::bin(BinOp::Add, var(0, Ty::I32), Expr::i32(1)),
                ),
                "(v0 + 1i32) as u32",
            ),
            (Expr::negated(var(0, Ty::I32)), "-v0"),
            (Expr::negated(Expr::negated(var(0, Ty::I32))), "-(-v0)"),
            (
                Expr::inverted(Expr::bin(BinOp::Lt, u(1), u(2))),
                "!(1u32 < 2u32)",
            ),
            (
                Expr::bin(
                    BinOp::And,
                    Expr::bin(BinOp::Or, Expr::bool(true), Expr::bool(false)),
                    Expr::bool(true),
                ),
                "(true || false) && true",
            ),
        ];
        for (expr, expected) in cases {
            assert_eq!(emit_expr(&expr), expected, "{expr:?}");
        }
    }

    fn tiny_program() -> Program {
        Program {
            seed: 9,
            attempt: 4,
            count: 3,
            inputs: vec![
                InputBuf {
                    ty: Ty::U32,
                    data: vec![1, 2, 3],
                },
                InputBuf {
                    ty: Ty::I32,
                    data: vec![1, 2, 3],
                },
            ],
            body: vec![
                Stmt::Let {
                    var: 0,
                    ty: Ty::U32,
                    mutable: true,
                    init: Expr::input(0, Ty::U32, Expr::Gid),
                },
                Stmt::For {
                    var: 1,
                    trips: 3,
                    body: vec![Stmt::If {
                        cond: Expr::bin(BinOp::Gt, var(1, Ty::U32), u(0)),
                        then: vec![Stmt::Assign {
                            var: 0,
                            value: Expr::bin(
                                BinOp::Add,
                                var(0, Ty::U32),
                                Expr::bin(BinOp::Shl, var(1, Ty::U32), u(2)),
                            ),
                        }],
                        els: Vec::new(),
                    }],
                },
            ],
            result: Expr::bin(BinOp::Lt, var(0, Ty::U32), u(100)),
        }
    }

    #[test]
    fn a_module_has_exactly_this_text() {
        let expected = "\
/// Seed 9, attempt 4: 3 elements, output u32.
#[gust::gpu]
pub mod s9_a4 {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn k(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<uint>,
        b: StructuredBuffer<int>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < out.len() {
            let mut v0: u32 = a[i];
            for v1 in 0u32..3u32 {
                if v1 > 0u32 {
                    v0 = v0 + (v1 << 2u32);
                }
            }
            if v0 < 100u32 {
                out[i] = 1u32;
            } else {
                out[i] = 0u32;
            }
        }
    }
}
";
        let program = tiny_program();
        program.check().unwrap();
        assert_eq!(emit_module(&program), expected);
    }

    #[test]
    fn an_integer_result_is_stored_directly_and_an_else_is_written_when_present() {
        let mut program = tiny_program();
        program.result = var(0, Ty::U32);
        program.body.push(Stmt::If {
            cond: Expr::bool(true),
            then: vec![Stmt::Assign {
                var: 0,
                value: u(1),
            }],
            els: vec![Stmt::Assign {
                var: 0,
                value: u(2),
            }],
        });
        program.check().unwrap();
        let text = emit_module(&program);
        assert!(text.contains("        mut out: RWStructuredBuffer<uint>,\n"));
        assert!(text.contains(
            "            if true {\n                v0 = 1u32;\n            } else {\n                v0 = 2u32;\n            }\n            out[i] = v0;\n"
        ));
    }

    #[test]
    fn the_crate_source_lists_every_module_after_the_header() {
        let program = tiny_program();
        let text = emit_crate_source(&[program.clone(), program]);
        assert!(text.starts_with(&format!(
            "// Generated by difftest (DIFFTEST_FORMAT {DIFFTEST_FORMAT})."
        )));
        assert_eq!(text.matches("#[gust::gpu]").count(), 2);
        assert!(text.lines().nth(1).unwrap().starts_with("#![allow("));
        assert!(!text.contains('\r'));
    }

    #[test]
    fn emitted_text_reparses_to_the_same_ast_for_a_hand_written_program() {
        let program = tiny_program();
        assert_eq!(
            reparse(&emit_module(&program), &program.inputs),
            Ok((program.body.clone(), program.result.clone()))
        );
    }

    #[test]
    fn emitted_text_reparses_to_the_same_ast_for_every_seed_of_the_corpus() {
        for seed in 1..=8 {
            let batch = generate_batch(seed).unwrap();
            for program in &batch.programs {
                program.check().unwrap();
                let text = emit_module(program);
                assert_eq!(
                    reparse(&text, &program.inputs),
                    Ok((program.body.clone(), program.result.clone())),
                    "seed {seed} attempt {}:\n{text}",
                    program.attempt
                );
            }
        }
    }

    #[test]
    fn the_reparser_reads_rust_precedence_not_the_intent() {
        // Control for the test above: text without the parentheses of `(v0 << 3) + v1` is a different tree, so the
        // comparison would fail if the emitter dropped them.
        let inputs = vec![InputBuf {
            ty: Ty::U32,
            data: vec![0],
        }];
        let text = |expr: &str| {
            format!(
                "pub mod m {{\n let i = id.x;\n if i < out.len() {{\n let v0: u32 = 1u32;\n let v1: u32 = 2u32;\n out[i] = {expr};\n }}\n}}\n"
            )
        };
        let tree = |expr: &str| reparse(&text(expr), &inputs).unwrap().1;
        let grouped = tree("(v0 << 3u32) + v1");
        let ungrouped = tree("v0 << 3u32 + v1");
        assert_ne!(grouped, ungrouped);
        assert_eq!(
            ungrouped,
            Expr::bin(
                BinOp::Shl,
                var(0, Ty::U32),
                Expr::bin(BinOp::Add, u(3), var(1, Ty::U32))
            )
        );
        // `*` binds tighter than `+`.
        assert_eq!(
            tree("v0 + v1 * 2u32"),
            Expr::bin(
                BinOp::Add,
                var(0, Ty::U32),
                Expr::bin(BinOp::Mul, var(1, Ty::U32), u(2))
            )
        );
        // Unlike C, `&` binds tighter than `==` in Rust.
        assert_eq!(
            tree("v0 & v1 == 1u32"),
            Expr::bin(
                BinOp::Eq,
                Expr::bin(BinOp::BitAnd, var(0, Ty::U32), var(1, Ty::U32)),
                u(1)
            )
        );
    }

    #[test]
    fn the_template_manifest_is_the_emitted_manifest_with_the_relative_root() {
        let template =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/template/Cargo.toml"))
                .unwrap()
                .replace("\r\n", "\n");
        assert_eq!(template, manifest("../../..", CRATE_NAME));
    }

    /// (name, version) -> checksum of every registry package in a lockfile text.
    fn registry_packages(lock: &str) -> BTreeMap<(String, String), String> {
        let lock = lock.replace("\r\n", "\n");
        let mut packages = BTreeMap::new();
        for block in lock.split("[[package]]\n").skip(1) {
            let field = |key: &str| {
                block.lines().find_map(|line| {
                    line.strip_prefix(key)
                        .and_then(|rest| rest.strip_prefix(" = \""))
                        .and_then(|rest| rest.strip_suffix('"'))
                        .map(str::to_owned)
                })
            };
            if let (Some(name), Some(version), Some(checksum)) =
                (field("name"), field("version"), field("checksum"))
            {
                packages.insert((name, version), checksum);
            }
        }
        packages
    }

    #[test]
    fn the_template_lock_pins_the_same_registry_packages_as_the_root_lock() {
        let read = |path: &str| {
            std::fs::read_to_string(format!("{}/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap()
        };
        let root = registry_packages(&read("../../Cargo.lock"));
        let template = registry_packages(&read("template/Cargo.lock"));
        assert!(template.keys().any(|(name, _)| name == "wgpu"));
        for (package, checksum) in &template {
            assert_eq!(
                root.get(package),
                Some(checksum),
                "{package:?} differs from the root Cargo.lock"
            );
        }
        // The lock embedded into the emitter is the committed file.
        assert_eq!(registry_packages(TEMPLATE_LOCK), template);
    }

    #[test]
    fn paths_are_written_safely_into_the_manifest() {
        let ok = |text: &str| path_for_toml(Path::new(text));
        assert_eq!(ok("C:\\work\\repo").unwrap(), "C:/work/repo");
        assert_eq!(ok("C:/work/repo/").unwrap(), "C:/work/repo");
        assert_eq!(ok("\\\\?\\C:\\work\\repo").unwrap(), "C:/work/repo");
        assert_eq!(
            ok("\\\\?\\UNC\\host\\share\\repo").unwrap(),
            "//host/share/repo"
        );
        assert_eq!(ok("/opt/repo").unwrap(), "/opt/repo");
        assert_eq!(ok("../..").unwrap(), "../..");
        assert_eq!(ok("/").unwrap(), "/");
        assert!(ok("").is_err());
        assert!(ok("C:/we\"ird").is_err());
        assert!(ok("C:/new\nline").is_err());
    }

    #[test]
    fn the_manifest_names_the_package_and_its_dependencies() {
        let text = manifest("/r", "difftest-case-x");
        assert!(text.contains("name = \"difftest-case-x\"\n"));
        assert!(text.contains("gust = { path = \"/r/crates/gust\" }"));
        assert!(text.contains(
            "gust-wgpu = { path = \"/r/crates/gust-wgpu\", features = [\"slang-runtime\"] }"
        ));
        assert!(text.ends_with("[workspace]\n"));
    }

    #[test]
    fn the_lock_for_a_package_differs_from_the_template_only_in_the_root_name() {
        let template = TEMPLATE_LOCK.replace("\r\n", "\n");
        assert_eq!(lock_for(CRATE_NAME).unwrap(), template);
        let named = lock_for("difftest-case-ok").unwrap();
        let (a, b): (Vec<&str>, Vec<&str>) = (template.lines().collect(), named.lines().collect());
        assert_eq!(a.len(), b.len());
        let differing: Vec<(&str, &str)> = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x != y)
            .map(|(x, y)| (*x, *y))
            .collect();
        assert_eq!(
            differing,
            [("name = \"difftest-case\"", "name = \"difftest-case-ok\"")]
        );
        assert_eq!(package_name("ok").unwrap(), "difftest-case-ok");
        assert!(package_name("").is_err());
        assert!(package_name("Bad_Name").is_err());
        assert!(lock_for("bad name\"").is_err());
        assert!(lock_for("").is_err());
    }
}

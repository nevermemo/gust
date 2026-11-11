//! The program forms of the NVIDIA array mismatch (plans/m9/arrbug.md and arrbug2.md, D168, D169, driver 610.88), kept
//! covered on the host side. This is NOT GPU coverage: the GPU is deliberately not run here.
//!
//! The shrunk seed 40 program (attempt 5000008) is
//!
//! ```text
//! let v8: S0 = S0 { f0: 15, f1: 0 };
//! let mut v17: [i32; 3] = [-3 - a[0]; 3];
//! v17[(i % 3) as usize] -= v8.f1;
//! out[i] = v17[0] ^ 7;
//! ```
//!
//! With `a[0] = -5` every element is `2 ^ 7 = 5` by the Rust Reference (the interpreter) and by the CPU twin. The RTX 5090
//! driver gives 7 where `i % 3 != 0`.
//!
//! The shrunk seed 53 program (the wide run of fix round 1: an array that is not uniform, written after an earlier
//! dynamic store and a copy) is
//!
//! ```text
//! let x: u32 = 2048;
//! let mut v46: [u32; 4] = [x, x, 1073742547, x];
//! v46[(i % v46.len()) as usize] <<= 10;
//! let mut v48: [u32; 4] = v46;
//! v48[(i % 4) as usize] ^= 0;
//! out[i] = v48[3];
//! ```
//!
//! By the Rust Reference (the interpreter) and the CPU twin it is 2097152 where `i % 4 == 3` and 2048 elsewhere; the
//! RTX 5090 driver gives 0 where `i % 4 == 3`. The generator does not write either form while that driver bug stands
//! (DIFFTEST.md, "Known driver hole"), and this test keeps the forms checked on the interpreter and the twin. It needs cargo and the registry, not a GPU:
//! `cargo test --locked --manifest-path tests/difftest/Cargo.toml --test array_known_driver`.

use std::path::{Path, PathBuf};

use difftest::ast::{
    ArrTy, BinOp, Elem, Expr, InputBuf, Methods, Place, Program, Stmt, StructDef, Ty,
};
use difftest::classify::classify;
use difftest::emit::{module_name, package_name};
use difftest::harness::{Options, execute_plan};
use difftest::interp::run_full;
use difftest::runsrc::Planned;
use difftest::shapes::Shape;

const COUNT: u32 = 64;
const S0: Ty = Ty::Struct(0);

fn seed_40_program() -> Program {
    let arr = Ty::Array(ArrTy::new(Elem::I32, 3));
    let v17 = |index: Expr| Place {
        index: Some(Box::new(index)),
        ..Place::var(17, arr)
    };
    let program = Program {
        seed: 40,
        attempt: 5_000_008,
        count: COUNT,
        methods: Methods {
            structs: vec![StructDef {
                fields: vec![Ty::U32, Ty::I32],
            }],
            fns: Vec::new(),
            enums: Vec::new(),
        },
        store: None,
        inputs: vec![InputBuf {
            ty: Ty::I32,
            data: (0..COUNT).map(|k| (k as i32 - 5) as u32).collect(),
        }],
        body: vec![
            Stmt::Let {
                var: 8,
                ty: S0,
                mutable: true,
                init: Expr::StructLit {
                    ty: S0,
                    fields: vec![Expr::u32(15), Expr::i32(0)],
                    order: vec![0, 1],
                },
            },
            Stmt::Let {
                var: 17,
                ty: arr,
                mutable: true,
                init: Expr::Repeat {
                    ty: arr,
                    e: Box::new(Expr::bin(
                        BinOp::Sub,
                        Expr::i32(-3),
                        Expr::input(0, Ty::I32, Expr::u32(0)),
                    )),
                },
            },
            Stmt::Set {
                place: v17(Expr::bin(BinOp::Rem, Expr::Gid, Expr::u32(3))),
                ty: Ty::I32,
                op: Some(BinOp::Sub),
                value: Expr::Read {
                    place: Place::var(8, S0).field(1),
                    ty: Ty::I32,
                },
            },
        ],
        result: Expr::bin(
            BinOp::BitXor,
            Expr::Read {
                place: v17(Expr::u32(0)),
                ty: Ty::I32,
            },
            Expr::i32(7),
        ),
    };
    program
        .check()
        .unwrap_or_else(|error| panic!("the hand program is refused: {error}"));
    program
}

fn options() -> Options {
    let dir = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("difftest")
            .join("target"),
    };
    std::fs::create_dir_all(&dir).expect("create the target directory");
    Options {
        seeds: Vec::new(),
        cpu_only: true,
        shapes: vec![Shape::Grid],
        profile: "dev".to_owned(),
        export_dir: None,
        target_dir: std::path::absolute(dir).expect("absolute target directory"),
        repo_root: std::path::absolute(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .expect("absolute repository root"),
    }
}

/// The seed 53 minimal program of arrbug2.md.
fn seed_53_program() -> Program {
    let arr = Ty::Array(ArrTy::new(Elem::U32, 4));
    let at = |var: u32, index: Expr| Place {
        index: Some(Box::new(index)),
        ..Place::var(var, arr)
    };
    let x = || Expr::Var(1, Ty::U32);
    let program = Program {
        seed: 53,
        attempt: 5_000_015,
        count: COUNT,
        methods: Methods {
            structs: Vec::new(),
            fns: Vec::new(),
            enums: Vec::new(),
        },
        store: None,
        inputs: vec![InputBuf {
            ty: Ty::U32,
            data: (0..COUNT).collect(),
        }],
        body: vec![
            Stmt::Let {
                var: 1,
                ty: Ty::U32,
                mutable: false,
                init: Expr::u32(2048),
            },
            Stmt::Let {
                var: 46,
                ty: arr,
                mutable: true,
                init: Expr::ArrayLit {
                    ty: arr,
                    elems: vec![x(), x(), Expr::u32(1_073_742_547), x()],
                },
            },
            Stmt::Set {
                place: at(
                    46,
                    Expr::bin(
                        BinOp::Rem,
                        Expr::Gid,
                        Expr::ArrLen {
                            place: Place::var(46, arr),
                        },
                    ),
                ),
                ty: Ty::U32,
                op: Some(BinOp::Shl),
                value: Expr::u32(10),
            },
            Stmt::Let {
                var: 48,
                ty: arr,
                mutable: true,
                init: Expr::Var(46, arr),
            },
            Stmt::Set {
                place: at(48, Expr::bin(BinOp::Rem, Expr::Gid, Expr::u32(4))),
                ty: Ty::U32,
                op: Some(BinOp::BitXor),
                value: Expr::u32(0),
            },
        ],
        result: Expr::Read {
            place: at(48, Expr::u32(3)),
            ty: Ty::U32,
        },
    };
    program
        .check()
        .unwrap_or_else(|error| panic!("the hand program is refused: {error}"));
    program
}

fn seed_53_expected() -> Vec<u32> {
    (0..COUNT)
        .map(|i| if i % 4 == 3 { 2_097_152 } else { 2048 })
        .collect()
}

#[test]
fn the_interpreter_gives_five_everywhere() {
    let run = run_full(&seed_40_program()).unwrap();
    assert_eq!(run.values, vec![5; COUNT as usize]);
}

#[test]
fn the_interpreter_gives_the_seed_53_values() {
    let run = run_full(&seed_53_program()).unwrap();
    assert_eq!(run.values, seed_53_expected());
}

fn twin_agrees(name: &str, program: Program) {
    let run = run_full(&program).unwrap();
    let plan = vec![Planned {
        program,
        index: 0,
        expected: run.values,
        intervals: run.intervals,
    }];
    let package = package_name(name).unwrap();
    let (raws, _) = execute_plan(&options(), &package, name, &plan, &[]).unwrap();
    assert_eq!(raws.len(), 1);
    let found = classify(&plan[0], &raws[0], true);
    assert!(
        found.is_empty(),
        "{}: {found:?}",
        module_name(&plan[0].program)
    );
    assert_eq!(raws[0].twin.as_ref(), Ok(&plan[0].expected));
}

#[test]
fn the_cpu_twin_agrees_with_the_interpreter() {
    twin_agrees("array-known-driver", seed_40_program());
}

#[test]
fn the_cpu_twin_agrees_with_the_interpreter_on_seed_53() {
    twin_agrees("array-known-driver-53", seed_53_program());
}

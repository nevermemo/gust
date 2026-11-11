//! The program form of the NVIDIA array mismatch (plans/m9/arrbug.md, driver 610.88), kept covered on the host side.
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
//! driver gives 7 where `i % 3 != 0`, so the GPU is deliberately not run here: the generator does not write the form
//! while that driver bug stands (DIFFTEST.md, "Known driver hole"), and this test keeps the form checked on the
//! interpreter and the twin. It needs cargo and the registry, not a GPU:
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

fn program() -> Program {
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

#[test]
fn the_interpreter_gives_five_everywhere() {
    let run = run_full(&program()).unwrap();
    assert_eq!(run.values, vec![5; COUNT as usize]);
}

#[test]
fn the_cpu_twin_agrees_with_the_interpreter() {
    let program = program();
    let run = run_full(&program).unwrap();
    let plan = vec![Planned {
        program,
        index: 0,
        expected: run.values,
        intervals: run.intervals,
    }];
    let name = "array-known-driver";
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

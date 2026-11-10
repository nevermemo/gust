//! Tests of the vector and matrix programs (slice m6.d): the arithmetic of the interpreter against hand-computed
//! values, the integer domain and the float intervals per component, the `Mat2` guard, the emitted text and the
//! corpus of the vector space.

use crate::ast::{ASSIGN_COMP, Expr};
use crate::ast::{BinOp, InputBuf, Lane, Program, Stmt, Ty, operator_counts};
use crate::emit::emit_module;
use crate::generator::{
    Space, VECTOR_ATTEMPT_BASE, discard_reason, generate_attempt, generate_batch, generate_batch_in,
};
use crate::interp::{Violation, run_covered, run_full, static_violation};
use crate::reparse::reparse;

const V2: Ty = Ty::Vec(Lane::F32, 2);
const V4: Ty = Ty::Vec(Lane::F32, 4);

fn f(value: f32) -> Expr {
    Expr::f32(value)
}

fn vec2(a: f32, b: f32) -> Expr {
    Expr::vec_new(V2, vec![f(a), f(b)])
}

fn vec4(a: f32, b: f32, c: f32, d: f32) -> Expr {
    Expr::vec_new(V4, vec![f(a), f(b), f(c), f(d)])
}

/// `Mat2::from_cols(c0, c1)`.
fn mat2(c0: Expr, c1: Expr) -> Expr {
    Expr::Make {
        ty: Ty::Mat(2),
        rows: false,
        args: vec![c0, c1],
    }
}

fn mul(l: Expr, r: Expr) -> Expr {
    Expr::agg_bin(BinOp::Mul, l, r)
}

/// A program over one `f32` input buffer holding `data` (one element per entry).
fn program(data: &[f32], body: Vec<Stmt>, result: Expr) -> Program {
    Program {
        seed: 0,
        methods: Default::default(),
        attempt: VECTOR_ATTEMPT_BASE,
        count: data.len() as u32,
        inputs: vec![InputBuf {
            ty: Ty::F32,
            data: data.iter().map(|v| v.to_bits()).collect(),
        }],
        body,
        result,
    }
}

fn input() -> Expr {
    Expr::input(0, Ty::F32, Expr::Gid)
}

/// The `f32` result of a closed expression, for one element.
fn value(result: Expr) -> f32 {
    let program = program(&[1.0], Vec::new(), result);
    f32::from_bits(run_full(&program).expect("in the domain").values[0])
}

fn violation(result: Expr, data: &[f32]) -> Violation {
    run_full(&program(data, Vec::new(), result))
        .expect_err("outside the domain")
        .violation
}

#[test]
fn dot_is_the_left_to_right_sum_of_the_products() {
    let dot = Expr::dot(vec4(1.0, 2.0, 3.0, 4.0), vec4(5.0, 6.0, 7.0, 8.0));
    assert_eq!(value(dot), 70.0);
    let dot2 = Expr::dot(vec2(1.5, -2.0), vec2(4.0, 0.25));
    assert_eq!(value(dot2), 5.5);
}

#[test]
fn a_matrix_times_a_vector_uses_the_columns() {
    // m = [[1, 3], [2, 4]] (columns (1, 2) and (3, 4)); m * (5, 6) = 5 * (1, 2) + 6 * (3, 4) = (23, 34). A row-major
    // reading would give (1 * 5 + 2 * 6, 3 * 5 + 4 * 6) = (17, 39): this test is what an operand mix-up breaks.
    let m = || mat2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    assert_eq!(value(Expr::comp(mul(m(), vec2(5.0, 6.0)), 0)), 23.0);
    assert_eq!(value(Expr::comp(mul(m(), vec2(5.0, 6.0)), 1)), 34.0);
}

#[test]
fn a_matrix_times_a_matrix_multiplies_each_column() {
    let a = || mat2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    let b = || mat2(vec2(5.0, 6.0), vec2(7.0, 8.0));
    // (a * b) columns: a * (5, 6) = (23, 34) and a * (7, 8) = (1 * 7 + 3 * 8, 2 * 7 + 4 * 8) = (31, 46).
    let column = |unit: Expr, k: u8| value(Expr::comp(mul(mul(a(), b()), unit), k));
    assert_eq!(
        [column(vec2(1.0, 0.0), 0), column(vec2(1.0, 0.0), 1)],
        [23.0, 34.0]
    );
    assert_eq!(
        [column(vec2(0.0, 1.0), 0), column(vec2(0.0, 1.0), 1)],
        [31.0, 46.0]
    );
    // Not commutative: (b * a) first column is b * (1, 2) = (1 * 5 + 2 * 7, 1 * 6 + 2 * 8) = (19, 22).
    let flipped = |k: u8| value(Expr::comp(mul(mul(b(), a()), vec2(1.0, 0.0)), k));
    assert_eq!([flipped(0), flipped(1)], [19.0, 22.0]);
}

#[test]
fn from_rows_and_transpose_swap_rows_and_columns() {
    let rows = Expr::Make {
        ty: Ty::Mat(2),
        rows: true,
        args: vec![vec2(1.0, 2.0), vec2(3.0, 4.0)],
    };
    // Rows (1, 2) and (3, 4): the columns are (1, 3) and (2, 4); times (1, 0) is the first column.
    assert_eq!(value(Expr::comp(mul(rows.clone(), vec2(1.0, 0.0)), 1)), 3.0);
    let transposed = Expr::Transpose {
        e: Box::new(mat2(vec2(1.0, 2.0), vec2(3.0, 4.0))),
    };
    assert_eq!(value(Expr::comp(mul(transposed, vec2(1.0, 0.0)), 1)), 3.0);
    // Applied twice it is the identity: the same first column (1, 3) as `rows` has.
    let twice = Expr::Transpose {
        e: Box::new(Expr::Transpose { e: Box::new(rows) }),
    };
    assert_eq!(value(Expr::comp(mul(twice, vec2(1.0, 0.0)), 1)), 3.0);
}

#[test]
fn a_four_wide_product_is_the_scalar_sum_of_its_row() {
    // The fourth row of m * v is m03 * v0 + m13 * v1 + m23 * v2 + m33 * v3, in that order.
    let column = |k: u32| {
        let k = k as f32;
        vec4(k + 1.0, k + 2.0, k + 3.0, k + 4.0)
    };
    let m = || Expr::Make {
        ty: Ty::Mat(4),
        rows: false,
        args: (0..4).map(column).collect(),
    };
    let v = || vec4(0.5, -1.0, 2.0, 0.25);
    let expected = 4.0 * 0.5 - 5.0 + 6.0 * 2.0 + 7.0 * 0.25;
    assert_eq!(value(Expr::comp(mul(m(), v()), 3)), expected);
    // The interpreter's interval for it is that of the scalar expression, bit for bit.
    let term = |k: u8, c: f32| Expr::bin(BinOp::Mul, Expr::comp(column(u32::from(k)), 3), f(c));
    let scalar = [0.5, -1.0, 2.0, 0.25]
        .iter()
        .enumerate()
        .map(|(k, c)| term(k as u8, *c))
        .reduce(|sum, next| Expr::bin(BinOp::Add, sum, next))
        .unwrap();
    let vector = Expr::comp(mul(m(), v()), 3);
    let (a, b) = (
        run_full(&program(&[1.0], Vec::new(), vector)).unwrap(),
        run_full(&program(&[1.0], Vec::new(), scalar)).unwrap(),
    );
    assert_eq!(a, b);
}

#[test]
fn swizzles_pick_components_in_the_order_written() {
    let v = || vec4(1.0, 2.0, 3.0, 4.0);
    assert_eq!(
        value(Expr::comp(Expr::swizzle(v(), vec![3, 2, 1, 0]), 0)),
        4.0
    );
    assert_eq!(
        value(Expr::comp(Expr::swizzle(v(), vec![3, 2, 1, 0]), 3)),
        1.0
    );
    let narrowed = Expr::swizzle(v(), vec![2, 1]);
    assert_eq!(value(Expr::comp(narrowed, 0)), 3.0);
    let widened = Expr::swizzle(vec2(7.0, 8.0), vec![1, 0, 0, 1]);
    assert_eq!(value(Expr::comp(widened, 2)), 7.0);
}

#[test]
fn a_scalar_operand_is_used_for_every_component_and_the_order_of_a_quotient_is_kept() {
    let scaled = Expr::agg_bin(BinOp::Mul, f(3.0), vec2(2.0, 5.0));
    assert_eq!(value(Expr::comp(scaled, 1)), 15.0);
    let divided = Expr::agg_bin(BinOp::Div, vec2(9.0, 4.0), f(2.0));
    assert_eq!(value(Expr::comp(divided, 1)), 2.0);
    let negated = Expr::AggNeg {
        e: Box::new(vec2(9.0, 4.0)),
    };
    assert_eq!(value(Expr::comp(negated, 0)), -9.0);
    let splat = Expr::Splat {
        ty: V4,
        e: Box::new(f(2.5)),
    };
    assert_eq!(value(Expr::comp(splat, 3)), 2.5);
}

fn uvec2(a: u32, b: u32) -> Expr {
    Expr::vec_new(Ty::Vec(Lane::U32, 2), vec![Expr::u32(a), Expr::u32(b)])
}

fn ivec2(a: Expr, b: Expr) -> Expr {
    Expr::vec_new(Ty::Vec(Lane::I32, 2), vec![a, b])
}

/// The integer domain (D38) is tracked per component: the violation is that of the component that leaves it, and a
/// vector is outside the domain if any component is.
#[test]
fn integer_vectors_keep_the_domain_per_component() {
    let sum = |a: Expr, b: Expr| Expr::agg_bin(BinOp::Add, a, b);
    let comp = |e: Expr, k: u8| Expr::comp(e, k);
    let read = |e: Expr| Expr::cast(Ty::F32, e);
    // Component 0 is fine, component 1 overflows.
    let overflow = read(comp(sum(uvec2(1, u32::MAX), uvec2(1, 1)), 0));
    assert_eq!(violation(overflow, &[1.0]), Violation::Overflow);
    // Both fine: the same expression with room in component 1.
    let fits = read(comp(sum(uvec2(1, 5), uvec2(1, 1)), 1));
    assert_eq!(value(fits), 6.0);
    // `-MIN` of one component.
    let negated = Expr::AggNeg {
        e: Box::new(ivec2(Expr::i32(1), Expr::i32_min())),
    };
    assert_eq!(
        violation(read(comp(negated, 0)), &[1.0]),
        Violation::Overflow
    );
    // A zero divisor in one component, and `MIN / -1`.
    let divided = Expr::agg_bin(
        BinOp::Div,
        ivec2(Expr::i32(4), Expr::i32(4)),
        ivec2(Expr::i32(2), Expr::i32(0)),
    );
    assert_eq!(
        violation(read(comp(divided, 0)), &[1.0]),
        Violation::DivisionByZero
    );
    let min_over = Expr::agg_bin(
        BinOp::Div,
        ivec2(Expr::i32(4), Expr::i32_min()),
        ivec2(Expr::i32(2), Expr::i32(-1)),
    );
    assert_eq!(
        violation(read(comp(min_over, 0)), &[1.0]),
        Violation::MinOverMinusOne
    );
    // `dot` sums the products: 3 * 4 + 5 * 6 = 42, and an overflowing product is a violation.
    let dot = Expr::dot(uvec2(3, 5), uvec2(4, 6));
    assert_eq!(value(read(dot)), 42.0);
    let big = Expr::dot(uvec2(65_536, 1), uvec2(65_536, 1));
    assert_eq!(violation(read(big), &[1.0]), Violation::Overflow);
}

#[test]
fn float_vectors_keep_the_range_rule_per_component() {
    let huge = Expr::agg_bin(BinOp::Mul, vec2(1.0, 1e30), vec2(1.0, 1e30));
    assert_eq!(
        violation(Expr::comp(huge, 0), &[1.0]),
        Violation::FloatDomain
    );
    // A quotient by an input that is zero for some element: the divisor interval reaches zero (float.rs), which is a
    // float-domain discard, as for a scalar quotient.
    let quotient = Expr::agg_bin(BinOp::Div, vec2(1.0, 2.0), input());
    assert_eq!(
        violation(Expr::comp(quotient, 0), &[1.0, 0.0]),
        Violation::FloatDomain
    );
}

/// A vector assigned component by component keeps the other components.
#[test]
fn a_component_assignment_changes_one_component() {
    let v = Expr::Var(0, V2);
    let body = vec![
        Stmt::Let {
            var: 0,
            ty: V2,
            mutable: true,
            init: vec2(1.0, 2.0),
        },
        Stmt::AssignComp {
            var: 0,
            index: 1,
            value: input(),
        },
    ];
    let sum = Expr::bin(BinOp::Add, Expr::comp(v.clone(), 0), Expr::comp(v, 1));
    let p = program(&[10.0, 20.0], body, sum);
    p.check().unwrap();
    let run = run_full(&p).unwrap();
    assert_eq!(run.values, [11.0f32.to_bits(), 21.0f32.to_bits()]);
}

/// SEMANTICS section 4: a constant `Mat2` crashes naga's SPIR-V front end. The generator never writes one, and
/// `Program::check` refuses one.
#[test]
fn a_mat2_without_an_input_read_is_refused_and_a_mat4_is_not() {
    let constant = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    let result = Expr::comp(mul(constant, vec2(1.0, 0.0)), 0);
    let message = program(&[1.0], Vec::new(), result).check().unwrap_err();
    assert!(message.contains("Mat2"), "{message}");
    // With an input read among its components it is accepted.
    let read = Expr::vec_new(V2, vec![input(), f(2.0)]);
    let ok = Expr::comp(mul(mat2(read, vec2(3.0, 4.0)), vec2(1.0, 0.0)), 0);
    program(&[1.0], Vec::new(), ok).check().unwrap();
    // A constant `Mat4` is fine (the crash is Mat2 and Mat3).
    let column = || vec4(1.0, 2.0, 3.0, 4.0);
    let mat4 = Expr::Make {
        ty: Ty::Mat(4),
        rows: false,
        args: vec![column(), column(), column(), column()],
    };
    let ok4 = Expr::comp(mul(mat4, column()), 0);
    program(&[1.0], Vec::new(), ok4).check().unwrap();
}

/// The guard wants a direct `f32` read: `(in1[i] * 0u32) as f32` is an input read in the tree, but slangc can fold it
/// to a constant, so it does not make a `Mat2` safe (review of m6.d, nit 5).
#[test]
fn a_mat2_whose_only_input_read_is_behind_a_cast_is_refused() {
    let folded = Expr::cast(
        Ty::F32,
        Expr::bin(BinOp::Mul, Expr::input(1, Ty::U32, Expr::Gid), Expr::u32(0)),
    );
    let column = Expr::vec_new(V2, vec![folded, f(2.0)]);
    let result = Expr::comp(mul(mat2(column, vec2(3.0, 4.0)), vec2(1.0, 0.0)), 0);
    let mut p = program(&[1.0], Vec::new(), result);
    p.inputs.push(InputBuf {
        ty: Ty::U32,
        data: vec![5],
    });
    let message = p.check().unwrap_err();
    assert!(
        message.contains("Mat2") && message.contains("direct"),
        "{message}"
    );
    // The same matrix with a direct `f32` read in the other column is accepted.
    let read = Expr::vec_new(V2, vec![input(), f(2.0)]);
    let folded = Expr::cast(
        Ty::F32,
        Expr::bin(BinOp::Mul, Expr::input(1, Ty::U32, Expr::Gid), Expr::u32(0)),
    );
    let column = Expr::vec_new(V2, vec![folded, f(2.0)]);
    let ok = Expr::comp(mul(mat2(column, read), vec2(1.0, 0.0)), 0);
    let mut p = program(&[1.0], Vec::new(), ok);
    p.inputs.push(InputBuf {
        ty: Ty::U32,
        data: vec![5],
    });
    p.check().unwrap();
}

#[test]
fn check_refuses_ill_typed_vector_programs() {
    let int_vec = Ty::Vec(Lane::I32, 2);
    // A vector result, a vector buffer, a mixed-lane operator, a swizzle past the width, `Mat * Vec` of another width.
    let result_is_a_vector = program(&[1.0], Vec::new(), vec2(1.0, 2.0));
    assert!(result_is_a_vector.check().unwrap_err().contains("result"));
    let mut buffer = program(&[1.0], Vec::new(), input());
    buffer.inputs[0].ty = V2;
    assert!(buffer.check().is_err());
    let mixed = Expr::AggBin {
        op: BinOp::Add,
        ty: V2,
        l: Box::new(vec2(1.0, 2.0)),
        r: Box::new(ivec2(Expr::i32(1), Expr::i32(2))),
    };
    assert!(
        program(&[1.0], Vec::new(), Expr::comp(mixed, 0))
            .check()
            .is_err()
    );
    let wide = Expr::swizzle(vec2(1.0, 2.0), vec![0, 3]);
    assert!(
        program(&[1.0], Vec::new(), Expr::comp(wide, 0))
            .check()
            .is_err()
    );
    let components = Expr::vec_new(int_vec, vec![Expr::i32(1)]);
    assert!(
        program(&[1.0], Vec::new(), Expr::comp(components, 0))
            .check()
            .is_err()
    );
}

#[test]
fn a_closed_vector_expression_outside_the_domain_is_a_static_violation_even_in_dead_code() {
    let dead = |init: Expr, ty: Ty| {
        static_violation(&Program {
            body: vec![Stmt::If {
                cond: Expr::bool(false),
                then: vec![Stmt::Let {
                    var: 0,
                    ty,
                    mutable: false,
                    init,
                }],
                els: Vec::new(),
            }],
            ..program(&[1.0], Vec::new(), input())
        })
    };
    let ty = Ty::Vec(Lane::U32, 2);
    let sum = Expr::agg_bin(BinOp::Add, uvec2(u32::MAX, 0), uvec2(1, 0));
    assert_eq!(dead(sum, ty), Some(Violation::Overflow));
    let in_domain = Expr::agg_bin(BinOp::Add, uvec2(1, 0), uvec2(1, 0));
    assert_eq!(dead(in_domain, ty), None);
    // A zero divisor that is closed makes the operation static even if the dividend is not.
    let quotient = Expr::agg_bin(
        BinOp::Div,
        Expr::Splat {
            ty: V2,
            e: Box::new(input()),
        },
        Expr::Splat {
            ty: V2,
            e: Box::new(f(0.0)),
        },
    );
    assert_eq!(dead(quotient, V2), Some(Violation::DivisionByZero));
    // A float constant out of range inside a constructor.
    let huge = Expr::agg_bin(BinOp::Mul, vec2(1e30, 1.0), vec2(1e30, 1.0));
    assert_eq!(dead(huge, V2), Some(Violation::FloatDomain));
}

#[test]
fn the_interpreter_counts_the_vector_operators_it_executes_by_the_names_the_syntax_count_uses() {
    let body = vec![
        Stmt::Let {
            var: 0,
            ty: V4,
            mutable: true,
            init: Expr::Splat {
                ty: V4,
                e: Box::new(input()),
            },
        },
        Stmt::AssignComp {
            var: 0,
            index: 2,
            value: f(3.0),
        },
    ];
    let v = || Expr::Var(0, V4);
    let result = Expr::dot(widen(Expr::swizzle(v(), vec![0, 1])), v());
    let p = program(&[1.0, 2.0, 3.0], body, result);
    p.check().unwrap();
    let (run, coverage) = run_covered(&p).unwrap();
    assert_eq!(run.values.len(), 3);
    for name in ["splat Vec4", "assign comp", "swizzle Vec4->2", "dot Vec4"] {
        assert_eq!(
            coverage.operators[name], 3,
            "{name}: {:?}",
            coverage.operators
        );
    }
    let written = operator_counts(&p);
    assert_eq!(written["splat Vec4"], 1);
    assert_eq!(written[ASSIGN_COMP], 1);
}

/// A `Vec2` as a `Vec4`: `v.xyxy()`.
fn widen(e: Expr) -> Expr {
    Expr::swizzle(e, vec![0, 1, 0, 1])
}

/// Text: every new form is written, and read back as the same AST by Rust's own precedence (`reparse.rs`).
#[test]
fn the_emitted_text_of_every_vector_form_reads_back_as_the_same_program() {
    let v = |n| Expr::Var(n, V4);
    let m = Expr::Var(1, Ty::Mat(4));
    let w = Expr::Var(2, Ty::Vec(Lane::I32, 2));
    let body = vec![
        Stmt::Let {
            var: 0,
            ty: V4,
            mutable: true,
            init: vec4(1.0, -2.0, 0.5, 4.0),
        },
        Stmt::Let {
            var: 1,
            ty: Ty::Mat(4),
            mutable: false,
            init: Expr::Make {
                ty: Ty::Mat(4),
                rows: true,
                args: vec![v(0), v(0), Expr::swizzle(v(0), vec![3, 2, 1, 0]), v(0)],
            },
        },
        Stmt::Let {
            var: 2,
            ty: Ty::Vec(Lane::I32, 2),
            mutable: true,
            init: ivec2(Expr::i32(-3), Expr::cast(Ty::I32, input())),
        },
        Stmt::AssignComp {
            var: 0,
            index: 3,
            value: Expr::bin(BinOp::Mul, Expr::comp(v(0), 1), input()),
        },
        Stmt::AssignComp {
            var: 2,
            index: 0,
            value: Expr::comp(w.clone(), 1),
        },
    ];
    // Negation of a component, of a product, a transpose, a dot, a cast of a component, a negated swizzle.
    let negated = Expr::AggNeg {
        e: Box::new(mul(
            Expr::Transpose {
                e: Box::new(m.clone()),
            },
            Expr::agg_bin(
                BinOp::Add,
                v(0),
                Expr::Splat {
                    ty: V4,
                    e: Box::new(f(1.0)),
                },
            ),
        )),
    };
    let result = Expr::bin(
        BinOp::Add,
        Expr::negated(Expr::comp(negated.clone(), 2)),
        Expr::bin(
            BinOp::Mul,
            Expr::dot(widen(Expr::swizzle(negated, vec![1, 0])), v(0)),
            Expr::cast(Ty::F32, Expr::comp(w, 0)),
        ),
    );
    let p = program(&[1.0, 2.0], body, result);
    p.check().unwrap();
    let text = emit_module(&p);
    for needle in [
        "Vec4::new(1.0f32, -2.0f32, 0.5f32, 4.0f32)",
        "Mat4::from_rows(",
        "Vec4::splat(1.0f32)",
        ".wzyx()",
        ".transpose()",
        "v0.w = ",
        "v2.x = v2.y;",
        "IVec2::new(-3i32, ",
    ] {
        assert!(text.contains(needle), "{needle} is not in\n{text}");
    }
    let (body, result) = reparse(&text, &p.inputs).unwrap();
    assert_eq!(body, p.body, "{text}");
    assert_eq!(result, p.result, "{text}");
}

/// The vector part of every committed seed: determinism, typing, the `Mat2` guard on every attempt (accepted or not),
/// and what the corpus reaches.
#[test]
fn the_vector_space_is_deterministic_and_never_writes_a_constant_mat2() {
    assert_eq!(
        generate_attempt(3, VECTOR_ATTEMPT_BASE + 2),
        generate_attempt(3, VECTOR_ATTEMPT_BASE + 2)
    );
    assert_ne!(
        generate_attempt(3, VECTOR_ATTEMPT_BASE + 2),
        generate_attempt(3, VECTOR_ATTEMPT_BASE + 3)
    );
    let mut mat2s = 0;
    for seed in 1..=4 {
        for attempt in VECTOR_ATTEMPT_BASE..VECTOR_ATTEMPT_BASE + 150 {
            let p = generate_attempt(seed, attempt);
            // `check` holds the `Mat2` guard, among the typing rules.
            p.check().unwrap_or_else(|e| {
                panic!("seed {seed} attempt {attempt}: {e}\n{}", emit_module(&p))
            });
            mat2s += operator_counts(&p)
                .iter()
                .filter(|(name, _)| name.contains("Mat2"))
                .count();
        }
    }
    assert!(
        mat2s > 20,
        "only {mat2s} programs with a Mat2 operator: the guard was barely exercised"
    );
}

/// The vector programs of seeds 1 to 8 pass `check` and the domain, and together use 20 named forms and the
/// prefixes `swizzle `, `transpose ` and `/ `. It does not say that every form is written: see
/// `the_generator_writes_every_binary_row_...` below.
#[test]
fn the_vector_programs_of_a_batch_are_in_the_domain_and_seeds_1_to_8_use_twenty_listed_forms() {
    let mut names = std::collections::BTreeSet::new();
    for seed in 1..=8 {
        let batch = generate_batch(seed).unwrap();
        for p in &batch.programs
            [crate::generator::SCALAR_PROGRAMS_PER_SEED..crate::generator::VECTOR_END]
        {
            p.check().unwrap();
            assert_eq!(discard_reason(p), None, "seed {seed} attempt {}", p.attempt);
            assert!(!p.out_ty().is_aggregate());
            names.extend(operator_counts(p).into_keys());
        }
    }
    for wanted in [
        "new Vec4",
        "new Vec2",
        "new IVec2",
        "new UVec4",
        "splat Vec4",
        "comp Vec4",
        "comp Vec2",
        "dot Vec4",
        "dot Vec2",
        "from_cols Mat4",
        "from_cols Mat2",
        "from_rows Mat4",
        "* Mat4 Vec4",
        "* Mat2 Vec2",
        "+ Vec4 Vec4",
        "- Vec2 Vec2",
        "* Vec4 f32",
        "* f32 Vec2",
        "-Vec4",
        ASSIGN_COMP,
    ] {
        assert!(
            names.contains(wanted),
            "no vector program uses `{wanted}`: {names:?}"
        );
    }
    for prefix in ["swizzle ", "transpose ", "/ "] {
        assert!(
            names.iter().any(|n| n.starts_with(prefix)),
            "no vector program uses `{prefix}`"
        );
    }
}

#[test]
fn the_float_space_alone_still_writes_no_vectors() {
    for seed in 1..=4 {
        for p in generate_batch_in(Space::Float, seed).unwrap().programs {
            assert!(!p.has_aggregate());
        }
    }
}

/// Seeds `1..=REACH_SEEDS` write every form of [`crate::ast::writable_aggregate_forms`]. Measured 2026-11-02: seeds 1
/// to 68 still miss `- Mat4 Mat4`, 72 do not, over the first five vector programs of a seed, which is what a batch holds
/// again (five, as in format 8). The bound is a round number just above the first one that works. Seeds 1 to 8, the
/// gated batch, miss about half of the forms (see DIFFTEST.md, Coverage).
const REACH_SEEDS: u64 = 72;

/// The coverage the GPU batch of seeds 1 to 8 does not have: CPU only, over the generator alone. If a row of
/// `agg_bin_ty`, a negation or a swizzle shape becomes unreachable (a weight of zero, a guard that is always false),
/// this fails and names it.
#[test]
fn the_generator_writes_every_binary_row_negation_and_swizzle_shape_within_the_first_seeds() {
    let mut written = std::collections::BTreeSet::new();
    for seed in 1..=REACH_SEEDS {
        for p in &generate_batch_in(Space::Vector, seed).unwrap().programs[..5] {
            written.extend(operator_counts(p).into_keys());
        }
    }
    let missing: Vec<String> = crate::ast::writable_aggregate_forms()
        .difference(&written)
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "seeds 1 to {REACH_SEEDS} never write {} of the forms: {missing:?}",
        missing.len()
    );
}

/// How many of the [`crate::ast::writable_aggregate_forms`] the programs of the 64-seed batch itself write (the batch
/// holds five vector programs per seed, as in format 8). Measured in slice m9.d1b, round r1: see DIFFTEST.md,
/// "Coverage of the older spaces". The test above guards the generator over its own space; this one guards the batch
/// that the GPU runs.
const BATCH_FORMS_AT_64_SEEDS: usize = 71;

#[test]
fn the_gated_batch_of_64_seeds_writes_the_vector_forms_that_it_wrote_when_measured() {
    let mut written = std::collections::BTreeSet::new();
    for seed in 1..=64 {
        for p in &crate::generator::generate_batch(seed).unwrap().programs {
            written.extend(operator_counts(p).into_keys());
        }
    }
    let forms = crate::ast::writable_aggregate_forms();
    let hit = forms.intersection(&written).count();
    eprintln!("BATCH-FORMS {hit} of {}", forms.len());
    assert!(
        hit >= BATCH_FORMS_AT_64_SEEDS,
        "the 64-seed batch writes {hit} of {} vector forms",
        forms.len()
    );
}

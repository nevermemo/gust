//! `if`/`else` as the whole initializer of a `let` (slice m12.ifx): the Slang each form lowers to, and the forms that
//! stay refused with GUST-E0811.

use crate::{
    diagnostics::{
        Code,
        testing::{assert_code_for, parse_str},
    },
    expand::kernel_options,
    slang::emit_kernel,
    validate::validate_module,
};

/// A kernel whose body is `body`, over a `uint` output buffer.
fn kernel(items: &str, body: &str) -> String {
    format!(
        "mod m {{ {items} #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {{ {body} }} }}"
    )
}

fn lower(source: &str) -> syn::Result<String> {
    let module: syn::ItemMod = parse_str(source)?;
    validate_module(&module)?;
    let function = module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "run" => Some(function),
            _ => None,
        })
        .unwrap();
    let options = kernel_options(function)?.expect("kernel has #[kernel] attribute");
    Ok(emit_kernel(&module, function, options)?
        .source
        .replace("\r\n", "\n"))
}

fn accepted(body: &str) -> String {
    let source = kernel("", body);
    lower(&source).unwrap_or_else(|error| panic!("{error}: {source}"))
}

fn refused(items: &str, body: &str) {
    let source = kernel(items, body);
    let error = lower(&source).expect_err(&source);
    assert_code_for(&error, Code::UnsupportedExpression, &source);
}

#[test]
fn a_two_arm_if_declares_the_local_and_assigns_it_in_each_branch() {
    let text = accepted("let w: u32 = if id.x > 3u32 { 1u32 } else { 2u32 }; out[id.x] = w;");
    assert!(
        text.contains(
            "    uint w;\n    if ((id.x > uint(3)))\n    {\n        w = uint(1);\n    }\n    else\n    {\n        w = uint(2);\n    }\n"
        ),
        "{text}"
    );
    // The type of an untyped `let` comes from the tails.
    let untyped = accepted("let w = if id.x > 3u32 { 1u32 } else { 2u32 }; out[id.x] = w;");
    assert_eq!(untyped, text);
    let float =
        accepted("let w = if id.x > 3u32 { 0.5f32 } else { 2.0f32 }; out[id.x] = w as u32;");
    assert!(float.contains("    float w;\n    if ("), "{float}");
    let boolean =
        accepted("let w = if id.x > 3u32 { id.x < 9u32 } else { false }; out[id.x] = w as u32;");
    assert!(boolean.contains("    bool w;\n    if ("), "{boolean}");
    let cast =
        accepted("let w = if id.x > 3u32 { id.x as i32 } else { -1i32 }; out[id.x] = w as u32;");
    assert!(cast.contains("    int w;\n    if ("), "{cast}");
}

#[test]
fn an_else_if_chain_is_one_assignment_in_each_branch() {
    let text = accepted(
        "let w: u32 = if id.x == 0u32 { 10u32 } else if id.x == 1u32 { 20u32 } else { 30u32 }; out[id.x] = w;",
    );
    assert!(
        text.contains("    uint w;\n    if ((id.x == uint(0)))\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "    else\n    {\n        if ((id.x == uint(1)))\n        {\n            w = uint(20);\n        }\n        else\n        {\n            w = uint(30);\n        }\n    }\n"
        ),
        "{text}"
    );
    assert_eq!(text.matches("w = ").count(), 3, "{text}");
}

#[test]
fn a_branch_runs_its_statements_before_its_value() {
    let text = accepted(
        "let w: u32 = if id.x > 3u32 { let t = id.x * 2u32; out[0u32] = t; t + 1u32 } else { 0u32 }; out[id.x] = w;",
    );
    assert!(
        text.contains(
            "    {\n        var t = (id.x * uint(2));\n        out[uint(0)] = t;\n        w = (t + uint(1));\n    }\n"
        ),
        "{text}"
    );
}

#[test]
fn a_vector_if_assigns_a_vector_local() {
    let typed = accepted(
        "let v: Vec2 = if id.x > 3u32 { Vec2::new(1.0, 2.0) } else { Vec2::splat(0.0) }; out[id.x] = v.x as u32;",
    );
    assert!(typed.contains("    float2 v;\n    if ("), "{typed}");
    assert!(typed.contains("v = float2(1.0, 2.0);"), "{typed}");
    let untyped = accepted(
        "let v = if id.x > 3u32 { Vec2::new(1.0, 2.0) } else { Vec2::splat(0.0) }; out[id.x] = v.x as u32;",
    );
    assert_eq!(untyped, typed);
}

#[test]
fn a_nested_if_value_assigns_the_same_local() {
    let text = accepted(
        "let w: u32 = if id.x > 3u32 { if id.x > 7u32 { 1u32 } else { 2u32 } } else { 3u32 }; out[id.x] = w;",
    );
    assert!(
        text.contains("    uint w;\n    if ((id.x > uint(3)))\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "    {\n        if ((id.x > uint(7)))\n        {\n            w = uint(1);\n        }\n        else\n        {\n            w = uint(2);\n        }\n    }\n    else\n    {\n        w = uint(3);\n    }\n"
        ),
        "{text}"
    );
}

#[test]
fn an_if_that_names_its_local_assigns_a_temporary() {
    let text = accepted(
        "let w = 5u32; { let w: u32 = if w > 3u32 { w + 1u32 } else { w }; out[id.x] = w; }",
    );
    assert!(text.contains("uint __gust_if_"), "{text}");
    assert!(text.contains("uint w = __gust_if_"), "{text}");
}

#[test]
fn an_if_without_else_is_not_a_value() {
    refused("", "let w: u32 = if id.x > 3u32 { 1u32 }; out[id.x] = w;");
    // The `else` of an `else if` is needed as well.
    refused(
        "",
        "let w: u32 = if id.x > 3u32 { 1u32 } else if id.x > 1u32 { 2u32 }; out[id.x] = w;",
    );
}

#[test]
fn an_if_let_is_not_a_value() {
    refused(
        "",
        "let o = Some(id.x); let w: u32 = if let Some(v) = o { v } else { 0u32 }; out[id.x] = w;",
    );
}

#[test]
fn an_if_value_anywhere_but_the_whole_let_initializer_is_refused() {
    // A call argument, an operand, an assignment, a returned value, a struct field.
    refused(
        "fn f(a: u32) -> u32 { a }",
        "out[id.x] = f(if id.x > 3u32 { 1u32 } else { 2u32 });",
    );
    refused(
        "",
        "out[id.x] = 1u32 + if id.x > 3u32 { 1u32 } else { 2u32 };",
    );
    refused("", "out[id.x] = if id.x > 3u32 { 1u32 } else { 2u32 };");
    refused(
        "fn f(a: u32) -> u32 { return if a > 3u32 { 1u32 } else { 2u32 }; }",
        "out[id.x] = f(id.x);",
    );
    refused(
        "pub struct P { pub a: u32 }",
        "let p = P { a: if id.x > 3u32 { 1u32 } else { 2u32 } }; out[id.x] = p.a;",
    );
    // Not the whole initializer.
    refused(
        "",
        "let w: u32 = 1u32 + if id.x > 3u32 { 1u32 } else { 2u32 }; out[id.x] = w;",
    );
}

#[test]
fn an_array_struct_or_matrix_valued_if_is_refused() {
    refused(
        "",
        "let a: [u32; 2] = if id.x > 3u32 { [1u32, 2u32] } else { [3u32, 4u32] }; out[id.x] = a[0];",
    );
    refused(
        "pub struct P { pub a: u32 }",
        "let p: P = if id.x > 3u32 { P { a: 1u32 } } else { P { a: 2u32 } }; out[id.x] = p.a;",
    );
    refused(
        "",
        "let m: Mat2 = if id.x > 3u32 { Mat2::IDENTITY } else { Mat2::IDENTITY }; out[id.x] = 1u32;",
    );
    // An untyped `let` whose branches show no type has nothing to declare.
    refused(
        "fn f(a: u32) -> u32 { a }",
        "let w = if id.x > 3u32 { f(1u32) } else { f(2u32) }; out[id.x] = w;",
    );
}

#[test]
fn an_untyped_if_takes_the_vector_beside_a_scalar_and_never_the_scalar() {
    // The vector side gives the type, on either side of the operator.
    for tail in [
        "Vec2::new(1.0, 2.0) * 0.5f32",
        "0.5f32 * Vec2::new(1.0, 2.0)",
    ] {
        let text = accepted(&format!(
            "let v = if id.x > 3u32 {{ {tail} }} else {{ Vec2::splat(1.0) }}; out[id.x] = v.x as u32;"
        ));
        assert!(text.contains("    float2 v;\n    if ("), "{text}");
    }
    // Two operands of the same type give it.
    let text = accepted("let w = if id.x > 3u32 { 1u32 + 2u32 } else { 0u32 }; out[id.x] = w;");
    assert!(text.contains("    uint w;\n    if ("), "{text}");
    // A scalar beside an operand that shows nothing gives nothing: `v * 0.5f32` is a vector when `v` is one.
    refused(
        "",
        "let v = Vec3::new(1.0, 2.0, 3.0); let w = if id.x > 3u32 { v * 0.5f32 } else { v }; out[id.x] = w.x as u32;",
    );
    // A matrix result is never inferred, so it reaches the refusal of a matrix value.
    refused(
        "",
        "let m = if id.x > 3u32 { 2.0f32 * Mat2::IDENTITY } else { Mat2::IDENTITY }; out[id.x] = 1u32;",
    );
}

#[test]
fn a_branch_may_end_in_continue_or_break_without_a_semicolon() {
    for jump in ["continue", "break"] {
        let text = accepted(&format!(
            "for i in 0u32..4u32 {{ let w: u32 = if i > 1u32 {{ {jump} }} else {{ i }}; out[i] = w; }}"
        ));
        assert!(text.contains(&format!("        {jump};\n")), "{text}");
        assert!(text.contains("w = i;"), "{text}");
    }
}

#[test]
fn a_branch_may_end_in_return_continue_or_break_with_a_semicolon() {
    let text = accepted("let w: u32 = if id.x > 3u32 { return; } else { 5u32 }; out[id.x] = w;");
    assert!(text.contains("        return;\n"), "{text}");
    assert!(text.contains("w = uint(5);"), "{text}");
    let text = accepted(
        "for i in 0u32..4u32 { let w: u32 = if i > 1u32 { continue; } else { i }; out[i] = w; }",
    );
    assert!(text.contains("        continue;\n"), "{text}");
}

#[test]
fn an_else_if_let_in_the_middle_of_a_chain_is_refused() {
    refused(
        "",
        "let o = Some(id.x); let w: u32 = if id.x > 9u32 { 1u32 } else if let Some(v) = o { v } else { 0u32 }; out[id.x] = w;",
    );
}

#[test]
fn a_tuple_pattern_does_not_take_an_if_value() {
    refused(
        "",
        "let (a, b) = if id.x > 3u32 { (1u32, 2u32) } else { (3u32, 4u32) }; out[id.x] = a + b;",
    );
}

#[test]
fn a_qualified_annotation_and_an_if_inside_a_block_tail_are_refused() {
    // The validator refuses the qualified spelling first, with its own code (GUST-E0324).
    let source = kernel(
        "",
        "let v: gust::Vec2 = if id.x > 3u32 { Vec2::new(1.0, 2.0) } else { Vec2::splat(0.0) }; out[id.x] = 1u32;",
    );
    let error = lower(&source).expect_err(&source);
    assert_code_for(&error, Code::QualifiedMathPath, &source);
    refused(
        "",
        "let w: u32 = if id.x > 3u32 { { if id.x > 7u32 { 1u32 } else { 2u32 } } } else { 3u32 }; out[id.x] = w;",
    );
}

#[test]
fn a_shadow_in_the_same_scope_declares_the_name_twice() {
    // The macro does not refuse it (as for every same-scope shadow): the emitted text declares `x` twice in one Slang
    // scope, which slangc refuses (E30200, SEMANTICS section 9). The shadow belongs in a block of its own.
    let source = kernel(
        "",
        "let x = 1u32; let x: u32 = if id.x > 3u32 { x } else { 2u32 }; out[id.x] = x;",
    );
    let text = lower(&source).unwrap_or_else(|error| panic!("{error}: {source}"));
    assert!(text.contains("var x = uint(1);"), "{text}");
    assert!(text.contains("uint x = __gust_if_"), "{text}");
}

//! A value `match` as the whole initializer of an untyped `let` (slice m13.matchu): the Slang each form lowers to, and
//! the forms whose arms show no type, which stay GUST-E0419. The type follows the rule of an untyped value `if`
//! (slice m12.ifx, `value_if.rs`).

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

fn accepted(items: &str, body: &str) -> String {
    let source = kernel(items, body);
    lower(&source).unwrap_or_else(|error| panic!("{error}: {source}"))
}

fn refused(items: &str, body: &str) {
    let source = kernel(items, body);
    let error = lower(&source).expect_err(&source);
    assert_code_for(&error, Code::ValueMatchForm, &source);
}

/// The untyped `let` lowers exactly as the typed one, and starts with the declaration `declaration`.
fn same_as_typed(declaration: &str, ty: &str, arms: &str, use_of_w: &str) -> String {
    let untyped = accepted("", &format!("let w = match id.x {{ {arms} }}; {use_of_w}"));
    let typed = accepted(
        "",
        &format!("let w: {ty} = match id.x {{ {arms} }}; {use_of_w}"),
    );
    assert_eq!(untyped, typed);
    assert!(
        untyped.contains(&format!(
            "    {declaration} w;\n    // @gust construct: match #0\n    switch (id.x)\n"
        )),
        "{untyped}"
    );
    untyped
}

#[test]
fn an_untyped_integer_match_declares_the_local_and_assigns_it_in_every_arm() {
    let text = same_as_typed("uint", "u32", "0 => 1u32, _ => 2u32", "out[id.x] = w;");
    assert!(
        text.contains(
            "case 0:\n        {\n            w = uint(1);\n            break;\n        }"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "default:\n        {\n            w = uint(2);\n            break;\n        }"
        ),
        "{text}"
    );
}

#[test]
fn an_untyped_match_takes_i32_f32_and_bool_from_its_arms() {
    same_as_typed(
        "int",
        "i32",
        "0 => 1i32, _ => -2i32",
        "out[id.x] = w as u32;",
    );
    same_as_typed(
        "float",
        "f32",
        "0 => 0.5f32, _ => 2.0f32",
        "out[id.x] = w as u32;",
    );
    same_as_typed(
        "bool",
        "bool",
        "0 => id.x < 9u32, _ => false",
        "out[id.x] = w as u32;",
    );
    // An `as` cast shows its target type.
    same_as_typed(
        "int",
        "i32",
        "0 => id.x as i32, _ => 7",
        "out[id.x] = w as u32;",
    );
}

#[test]
fn an_untyped_vector_match_assigns_a_vector_local() {
    let text = same_as_typed(
        "float2",
        "Vec2",
        "0 => Vec2::new(1.0, 2.0), _ => Vec2::splat(0.0)",
        "out[id.x] = w.x as u32;",
    );
    assert!(text.contains("w = float2(1.0, 2.0);"), "{text}");
}

#[test]
fn the_first_arm_that_shows_a_type_gives_it() {
    // The first arm shows nothing (a call), the second is a block whose tail does.
    let text = accepted(
        "fn f(a: u32) -> u32 { a }",
        "let w = match id.x { 0 => f(1u32), 1 => { let t = id.x * 2u32; t + 1u32 } _ => { 9u32 } }; out[id.x] = w;",
    );
    assert!(text.contains("    uint w;\n"), "{text}");
    // A block arm gives its tail's type, after its statements.
    let block = accepted(
        "",
        "let w = match id.x { 0 => { let t = id.x * 2u32; out[0u32] = t; 5i32 } _ => 6i32 }; out[id.x] = w as u32;",
    );
    assert!(block.contains("    int w;\n"), "{block}");
    assert!(
        block.contains("        {\n            var t = (id.x * uint(2));\n            out[uint(0)] = t;\n            w = int(5);\n            break;\n        }"),
        "{block}"
    );
}

#[test]
fn a_nested_match_gives_its_arms_type_and_assigns_the_same_local() {
    let text = accepted(
        "",
        "let w = match id.x { 0 => match id.x { 0 => 1i32, _ => 2i32 }, _ => 3i32 }; out[id.x] = w as u32;",
    );
    assert!(text.contains("    int w;\n"), "{text}");
    assert_eq!(text.matches("switch (id.x)").count(), 2, "{text}");
    assert_eq!(text.matches("w = int(").count(), 3, "{text}");
    // The type may come from the nested value only.
    let inner = accepted(
        "fn f(a: u32) -> u32 { a }",
        "let w = match id.x { 0 => f(1u32), _ => { match id.x { 1 => 7u32, _ => f(2u32) } } }; out[id.x] = w;",
    );
    assert!(inner.contains("    uint w;\n"), "{inner}");
}

#[test]
fn a_value_if_in_an_arm_is_refused_typed_or_not() {
    // The lowering is the typed one, and a value `if` is the whole initializer of a `let` only (GUST-E0811), so an
    // arm that holds one is refused with or without the annotation.
    for annotation in ["", ": u32"] {
        let source = kernel(
            "",
            &format!(
                "let w{annotation} = match id.x {{ 0 => if id.x > 3u32 {{ 1u32 }} else {{ 2u32 }}, _ => 3u32 }}; out[id.x] = w;"
            ),
        );
        let error = lower(&source).expect_err(&source);
        assert_code_for(&error, Code::UnsupportedExpression, &source);
    }
}

#[test]
fn an_arm_that_returns_shows_no_type_and_the_others_give_it() {
    for first in ["{ return; }", "{ out[0u32] = 1u32; return; }"] {
        let text = accepted(
            "",
            &format!("let w = match id.x {{ 0 => {first} 1 => 4u32, _ => 5u32 }}; out[id.x] = w;"),
        );
        assert!(text.contains("    uint w;\n"), "{text}");
        assert!(text.contains("            return;\n"), "{text}");
        assert!(text.contains("w = uint(4);"), "{text}");
    }
    // A `match` whose arms all return shows nothing.
    refused(
        "",
        "let w = match id.x { 0 => { return; } _ => { return; } }; out[id.x] = 1u32;",
    );
}

#[test]
fn an_untyped_match_that_names_its_local_assigns_a_temporary() {
    let text = accepted(
        "",
        "let k = id.x; { let k = match k { 0 => 10u32, _ => k + 1u32 }; out[id.x] = k; }",
    );
    assert!(text.contains("uint __gust_match_"), "{text}");
    assert!(text.contains("uint k = __gust_match_"), "{text}");
}

#[test]
fn an_untyped_enum_match_takes_its_type_from_the_arms() {
    let text = accepted(
        "#[derive(PartialEq, Eq)] pub enum Mode { A, B = 3, C }",
        "let m = Mode::A; let w = match m { Mode::A => 1u32, Mode::B => 2u32, Mode::C => 4u32 }; out[id.x] = w;",
    );
    assert!(text.contains("    uint w;\n"), "{text}");
    assert!(text.contains("switch (m)"), "{text}");
}

#[test]
fn an_untyped_match_takes_the_vector_beside_a_scalar_and_never_the_scalar() {
    for arm in [
        "Vec2::new(1.0, 2.0) * 0.5f32",
        "0.5f32 * Vec2::new(1.0, 2.0)",
    ] {
        let text = accepted(
            "",
            &format!(
                "let v = match id.x {{ 0 => {arm}, _ => Vec2::splat(1.0) }}; out[id.x] = v.x as u32;"
            ),
        );
        assert!(text.contains("    float2 v;\n"), "{text}");
    }
    // Two operands of the same type give it.
    let text = accepted(
        "",
        "let w = match id.x { 0 => 1u32 + 2u32, _ => 0u32 }; out[id.x] = w;",
    );
    assert!(text.contains("    uint w;\n"), "{text}");
}

#[test]
fn a_match_whose_arms_show_no_type_needs_the_typed_form() {
    // Calls, unsuffixed literals, a path, a struct literal, a scalar times an unknown (`v` may be a vector).
    refused(
        "fn f(a: u32) -> u32 { a }",
        "let w = match id.x { 0 => f(1u32), _ => f(2u32) }; out[id.x] = w;",
    );
    refused(
        "",
        "let w = match id.x { 0 => 1, _ => 2 }; out[id.x] = w as u32;",
    );
    refused(
        "",
        "let v = Vec3::new(1.0, 2.0, 3.0); let w = match id.x { 0 => 0.5f32 * v, _ => v }; out[id.x] = w.x as u32;",
    );
    refused(
        "",
        "let v = Vec3::new(1.0, 2.0, 3.0); let w = match id.x { 0 => v * 0.5f32, _ => v }; out[id.x] = w.x as u32;",
    );
    refused(
        "pub struct P { pub a: u32 }",
        "let p = match id.x { 0 => P { a: 1u32 }, _ => P { a: 2u32 } }; out[id.x] = p.a;",
    );
    // The typed form is the way.
    let text = accepted(
        "fn f(a: u32) -> u32 { a }",
        "let w: u32 = match id.x { 0 => f(1u32), _ => f(2u32) }; out[id.x] = w;",
    );
    assert!(text.contains("    uint w;\n"), "{text}");
}

#[test]
fn an_arm_that_breaks_or_continues_is_refused_typed_or_not() {
    // The validator takes a `return` as the end of a value arm; a `break` or a `continue` is not one (GUST-E0518 with
    // the `;`, GUST-E0405 without it), so such an arm never reaches the inference and shows no type to it.
    for jump in ["break", "continue"] {
        for (arm, code) in [
            (format!("{{ {jump}; }}"), Code::MatchArmBlock),
            (format!("{{ {jump} }}"), Code::JumpAsValue),
        ] {
            for annotation in ["", ": u32"] {
                let source = kernel(
                    "",
                    &format!(
                        "for i in 0u32..4u32 {{ let w{annotation} = match i {{ 0 => {arm} 1 => 4u32, _ => 5u32 }}; out[i] = w; }}"
                    ),
                );
                let error = lower(&source).expect_err(&source);
                assert_code_for(&error, code, &source);
            }
        }
    }
}

#[test]
fn a_value_match_as_the_tail_of_a_value_if_branch_is_refused_typed_or_not() {
    // A value `match` is the whole initializer of a `let` only: the tail of a branch is a statement `match`, whose arms
    // are blocks (GUST-E0518).
    for annotation in ["", ": u32"] {
        let source = kernel(
            "",
            &format!(
                "let w{annotation} = if id.x > 3u32 {{ match id.x {{ 0 => 1u32, _ => 2u32 }} }} else {{ 3u32 }}; out[id.x] = w;"
            ),
        );
        let error = lower(&source).expect_err(&source);
        assert_code_for(&error, Code::MatchArmBlock, &source);
    }
}

#[test]
fn suffixing_one_arm_is_enough_to_leave_out_the_type() {
    let text = accepted(
        "",
        "let w = match id.x { 0 => 1u32, _ => 2 }; out[id.x] = w;",
    );
    assert!(text.contains("    uint w;\n"), "{text}");
}

//! Subgroup operations in the macro (m8.s, M8 D140): which calls the validator accepts and where (GUST-E0745, E0747,
//! E0751, and E0415/E0416 as for a barrier), what each one lowers to in Slang, the derived operations, the classifier's
//! `Why::Subgroup`, and the golden of `tests/fixtures/subgroup.rs`.

use std::collections::HashSet;

use quote::ToTokens as _;

use crate::{
    diagnostics::{
        Code,
        testing::{assert_code, parse_str},
    },
    expand::{
        Environment, Profile, expand_in, kernel_attribute_facts, kernel_options,
        variant::{self},
    },
    profile::GRID,
    regression_tests::assert_golden,
    slang::emit_kernel,
    validate::{
        classify::{Shape, Why, classify},
        helper_names, subgroup_operation, validate_module,
    },
};

const FIXTURE: &str = include_str!("../../../../tests/fixtures/subgroup.rs");
const GOLDEN: &str = include_str!("../../../../tests/fixtures/subgroup.slang");

const PARAMETERS: &str =
    "id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>";

/// A module with the portable kernel `k` and, beside it, `variant_body` as the body of `#[variant(of = k,
/// requires(subgroups))] pub fn k_subgroups`.
fn pair(variant_body: &str) -> String {
    format!(
        "mod m {{
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn k({PARAMETERS}) {{
        if id.x < data.len() {{ out[id.x] = data[id.x]; }}
    }}
    #[variant(of = k, requires(subgroups))]
    pub fn k_subgroups({PARAMETERS}) {{
        {variant_body}
    }}
}}"
    )
}

/// The result of lowering and validating `source`.
fn check(source: &str) -> syn::Result<syn::ItemMod> {
    let module: syn::ItemMod = parse_str(source).unwrap();
    let (lowered, _) = variant::lower(module)?;
    validate_module(&lowered)?;
    Ok(lowered)
}

fn rejected(source: &str) -> syn::Error {
    match check(source) {
        Ok(_) => panic!("expected an error for\n{source}"),
        Err(error) => error,
    }
}

/// The Slang of the function `name` of an accepted module.
fn slang_of(source: &str, name: &str) -> String {
    let lowered = check(source).unwrap_or_else(|error| panic!("{error}\n{source}"));
    let function = kernel_named(&lowered, name);
    let options = kernel_options(&function).unwrap().unwrap();
    emit_kernel(&lowered, &function, options)
        .unwrap()
        .source
        .replace("\r\n", "\n")
}

fn kernel_named(module: &syn::ItemMod, name: &str) -> syn::ItemFn {
    module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == name => Some(function.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no function {name}"))
}

/// A subgroup body that holds `statement` after loading `x` (a `u32`) and `p` (a `bool`).
fn body(statement: &str) -> String {
    format!(
        "let mut x = 0u32;
        if id.x < data.len() {{ x = data[id.x]; }}
        let p = x > 3u32;
        {statement}
        if id.x < data.len() {{ out[id.x] = x; }}"
    )
}

// ---------------------------------------------------------------------------------------------------------------
// The fixture

#[test]
fn the_subgroup_fixture_matches_the_golden() {
    let module: syn::ItemMod = parse_str(FIXTURE.replacen("#[gust::gpu]", "", 1).trim()).unwrap();
    let (lowered, plan) = variant::lower(module).unwrap();
    validate_module(&lowered).unwrap();
    for kernel in [
        "partial_sums",
        "count_if",
        "lanes_u32",
        "lanes_i32",
        "lanes_f32",
        "votes",
    ] {
        assert!(plan.kernel(kernel).is_some(), "{kernel}");
    }
    let mut actual = String::new();
    for item in &lowered.content.as_ref().unwrap().1 {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let Some(options) = kernel_options(function).unwrap() else {
            continue;
        };
        actual.push_str(&format!("// ==== kernel {} ====\n", function.sig.ident));
        actual.push_str(
            &emit_kernel(&lowered, function, options)
                .unwrap()
                .source
                .replace("\r\n", "\n"),
        );
    }
    assert_golden(&actual, "subgroup.slang", &GOLDEN.replace("\r\n", "\n"));
}

#[test]
fn the_golden_lowers_every_operation_of_the_dialect() {
    let golden = GOLDEN.replace("\r\n", "\n");
    for fragment in [
        "WaveActiveSum(",
        "WaveActiveProduct(",
        "WaveActiveMin(",
        "WaveActiveMax(",
        "__gust_subgroup_and(",
        "__gust_subgroup_or(",
        "__gust_subgroup_xor(",
        "WaveActiveAllTrue(",
        "WaveActiveAnyTrue(",
        "WaveActiveBallot(",
        "WaveReadLaneFirst(",
        "WaveBroadcastLaneAt(",
        "WaveReadLaneAt(",
        "WaveShuffle(",
        "WavePrefixSum(",
        "WavePrefixProduct(",
        "WaveGetLaneCount()",
        "WaveGetLaneIndex()",
        "__gust_subgroup_inclusive_add(",
        "__gust_subgroup_inclusive_mul(",
        "__gust_subgroup_count(",
        "WaveReadLaneFirst(WaveGetLaneIndex())",
    ] {
        assert!(golden.contains(fragment), "{fragment}");
    }
    // Each kernel carries only the helpers it calls.
    let votes = golden
        .split("// ==== kernel ")
        .find(|part| part.starts_with("votes_subgroups "))
        .unwrap();
    assert!(
        votes.contains("uint __gust_subgroup_count(bool value)"),
        "{votes}"
    );
    assert!(!votes.contains("__gust_subgroup_and"), "{votes}");
    let portable = golden
        .split("// ==== kernel ")
        .find(|part| part.starts_with("lanes_u32 "))
        .unwrap();
    assert!(!portable.contains("Wave"), "{portable}");
}

#[test]
fn the_variants_lower_to_attributes_that_require_subgroups() {
    let module: syn::ItemMod = parse_str(FIXTURE.replacen("#[gust::gpu]", "", 1).trim()).unwrap();
    let (lowered, _) = variant::lower(module).unwrap();
    for item in &lowered.content.as_ref().unwrap().1 {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let name = function.sig.ident.to_string();
        let attribute = function
            .attrs
            .iter()
            .filter(|attribute| attribute.path().is_ident("kernel"))
            .map(|attribute| attribute.to_token_stream().to_string())
            .collect::<String>();
        let is_variant = name.ends_with("_subgroups");
        assert_eq!(
            attribute.contains("subgroups"),
            is_variant,
            "{name}: {attribute}"
        );
        assert_eq!(
            attribute.contains("unsupported"),
            is_variant,
            "{name}: {attribute}"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// What is accepted, and what it lowers to

/// (operation, the call written in a body that holds `x: u32` and `p: bool`, a fragment of its Slang).
const OPERATIONS: &[(&str, &str, &str)] = &[
    ("subgroup_add", "subgroup_add(x)", "WaveActiveSum(x)"),
    ("subgroup_mul", "subgroup_mul(x)", "WaveActiveProduct(x)"),
    ("subgroup_min", "subgroup_min(x)", "WaveActiveMin(x)"),
    ("subgroup_max", "subgroup_max(x)", "WaveActiveMax(x)"),
    ("subgroup_and", "subgroup_and(x)", "__gust_subgroup_and(x)"),
    ("subgroup_or", "subgroup_or(x)", "__gust_subgroup_or(x)"),
    ("subgroup_xor", "subgroup_xor(x)", "__gust_subgroup_xor(x)"),
    ("subgroup_all", "subgroup_all(p)", "WaveActiveAllTrue(p)"),
    ("subgroup_any", "subgroup_any(p)", "WaveActiveAnyTrue(p)"),
    (
        "subgroup_ballot",
        "subgroup_ballot(p)",
        "WaveActiveBallot(p)",
    ),
    (
        "subgroup_broadcast_first",
        "subgroup_broadcast_first(x)",
        "WaveReadLaneFirst(x)",
    ),
    (
        "subgroup_broadcast",
        "subgroup_broadcast(x, 2u32)",
        "WaveBroadcastLaneAt(x, uint(2))",
    ),
    (
        "subgroup_shuffle",
        "subgroup_shuffle(x, subgroup_lane() ^ 1u32)",
        "WaveShuffle(x, __gust_xor(WaveGetLaneIndex(), uint(1)))",
    ),
    (
        "subgroup_exclusive_add",
        "subgroup_exclusive_add(x)",
        "WavePrefixSum(x)",
    ),
    (
        "subgroup_exclusive_mul",
        "subgroup_exclusive_mul(x)",
        "WavePrefixProduct(x)",
    ),
    ("subgroup_size", "subgroup_size()", "WaveGetLaneCount()"),
    ("subgroup_lane", "subgroup_lane()", "WaveGetLaneIndex()"),
    (
        "subgroup_elect",
        "subgroup_elect()",
        "(WaveGetLaneIndex() == WaveReadLaneFirst(WaveGetLaneIndex()))",
    ),
    (
        "subgroup_count",
        "subgroup_count(p)",
        "__gust_subgroup_count(p)",
    ),
    (
        "subgroup_inclusive_add",
        "subgroup_inclusive_add(x)",
        "__gust_subgroup_inclusive_add(x)",
    ),
    (
        "subgroup_inclusive_mul",
        "subgroup_inclusive_mul(x)",
        "__gust_subgroup_inclusive_mul(x)",
    ),
];

#[test]
fn the_table_names_every_operation_of_the_dialect_once() {
    assert_eq!(OPERATIONS.len(), 21);
    let unique: HashSet<_> = OPERATIONS.iter().map(|row| row.0).collect();
    assert_eq!(unique.len(), OPERATIONS.len());
    for (operation, call, _) in OPERATIONS {
        let call: syn::ExprCall = parse_str(call).unwrap();
        assert_eq!(subgroup_operation(&call), Some(*operation));
    }
    // A name near an operation is none.
    for name in ["subgroup_sum()", "subgroup_add_all(x)", "wave_add(x)"] {
        let call: syn::ExprCall = parse_str(name).unwrap();
        assert_eq!(subgroup_operation(&call), None, "{name}");
    }
}

#[test]
fn each_operation_is_accepted_in_a_subgroup_body_and_lowers_to_its_intrinsic() {
    for (operation, call, fragment) in OPERATIONS {
        let source = pair(&body(&format!("let r = {call};")));
        let slang = slang_of(&source, "k_subgroups");
        assert!(slang.contains(fragment), "{operation}: {fragment}\n{slang}");
    }
}

#[test]
fn the_helpers_are_emitted_for_the_operations_a_kernel_calls_and_no_others() {
    let and = slang_of(&pair(&body("let r = subgroup_and(x);")), "k_subgroups");
    assert!(
        and.contains("uint __gust_subgroup_and(uint value) { return WaveActiveBitAnd(value); }"),
        "{and}"
    );
    // The signed overload reduces the bits of the unsigned value (S3: DX12's compiler has no signed reduction).
    assert!(
        and.contains(
            "int __gust_subgroup_and(int value) { return asint(WaveActiveBitAnd(asuint(value))); }"
        ),
        "{and}"
    );
    assert!(!and.contains("__gust_subgroup_or"), "{and}");
    assert!(!and.contains("__gust_subgroup_count"), "{and}");
    let scan = slang_of(
        &pair(&body("let r = subgroup_inclusive_mul(x);")),
        "k_subgroups",
    );
    for ty in ["uint", "int", "float"] {
        assert!(
            scan.contains(&format!(
                "{ty} __gust_subgroup_inclusive_mul({ty} value) {{ return WavePrefixProduct(value) * value; }}"
            )),
            "{scan}"
        );
    }
    let plain = slang_of(&pair(&body("let r = subgroup_add(x);")), "k_subgroups");
    assert!(!plain.contains("__gust_subgroup"), "{plain}");
    // The portable body of the pair is untouched.
    let portable = slang_of(&pair(&body("let r = subgroup_add(x);")), "k");
    assert!(!portable.contains("Wave"), "{portable}");
}

#[test]
fn a_literal_lane_is_a_broadcast_and_any_other_lane_is_a_shuffle() {
    let literal = slang_of(
        &pair(&body("let r = subgroup_broadcast(x, 3u32);")),
        "k_subgroups",
    );
    assert!(
        literal.contains("WaveBroadcastLaneAt(x, uint(3))"),
        "{literal}"
    );
    // slangc requires a compile-time constant for a broadcast (E40012, S1), so anything else is a read of the lane.
    for lane in ["subgroup_size() / 2u32", "x", "(1u32 + 2u32)"] {
        let source = pair(&body(&format!("let r = subgroup_broadcast(x, {lane});")));
        let slang = slang_of(&source, "k_subgroups");
        assert!(slang.contains("WaveReadLaneAt(x, "), "{lane}: {slang}");
        assert!(!slang.contains("WaveBroadcastLaneAt"), "{lane}: {slang}");
    }
}

#[test]
fn operations_combine_with_arithmetic_comparisons_and_casts() {
    let source = pair(&body(
        "let a = subgroup_add(x) + subgroup_max(x) * 2u32;
        let b = subgroup_add(x) > 4u32;
        let c = subgroup_add(x) as f32;
        let d = subgroup_add(subgroup_min(x));
        let mut e = 0u32;
        e = subgroup_xor(x);
        e += subgroup_add(x);",
    ));
    check(&source).unwrap();
}

#[test]
fn a_top_level_literal_loop_may_hold_operations() {
    let source = pair(&body(
        "for step in 0u32..4u32 {
            let s = subgroup_add(x + step);
            x = s;
        }",
    ));
    check(&source).unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// GUST-E0745: only in a body that requires subgroups

#[test]
fn a_portable_body_may_not_call_an_operation() {
    let source = "mod m {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn k(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x = subgroup_add(data[id.x]);
        out[id.x] = x;
    }
}";
    let error = rejected(source);
    assert_code(&error, Code::SubgroupNotAllowed);
    assert!(
        error
            .to_string()
            .contains("`subgroup_add` is a subgroup operation"),
        "{error}"
    );
}

#[test]
fn a_helper_may_not_call_an_operation() {
    let source = pair(&body("let r = double(x);")).replace(
        "mod m {",
        "mod m {
    fn double(v: u32) -> u32 { subgroup_add(v) }",
    );
    let error = rejected(&source);
    assert_code(&error, Code::SubgroupNotAllowed);
    assert!(
        error
            .to_string()
            .contains("a helper may be called under a condition"),
        "{error}"
    );
}

#[test]
fn a_function_may_not_take_the_name_of_an_operation() {
    let source = pair(&body("let r = x;")).replace(
        "mod m {",
        "mod m {
    fn subgroup_add(v: u32) -> u32 { v }",
    );
    let error = rejected(&source);
    assert_code(&error, Code::SubgroupNotAllowed);
    assert!(
        error
            .to_string()
            .contains("is the name of a subgroup operation"),
        "{error}"
    );
}

#[test]
fn a_kernel_that_requires_subgroups_may_call_them_without_a_variant() {
    let source = "mod m {
    #[kernel(workgroup_size(64, 1, 1), requires(subgroups))]
    pub fn k(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x = subgroup_add(data[id.x]);
        out[id.x] = x;
    }
}";
    check(source).unwrap();
}

#[test]
fn the_alias_vulkan_is_the_capability() {
    let source = "mod m {
    #[kernel(workgroup_size(64, 1, 1), requires(vulkan))]
    pub fn k(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x = subgroup_add(data[id.x]);
        out[id.x] = x;
    }
}";
    check(source).unwrap();
}

#[test]
fn an_operation_takes_its_arguments() {
    for call in [
        "subgroup_add()",
        "subgroup_add(x, x)",
        "subgroup_size(x)",
        "subgroup_shuffle(x)",
    ] {
        let error = rejected(&pair(&body(&format!("let r = {call};"))));
        assert_code(&error, Code::BuiltinArguments);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GUST-E0747: where

#[test]
fn an_operation_in_an_unapproved_position_is_refused() {
    for (what, statement) in [
        ("under an if", "if p { let r = subgroup_add(x); }"),
        (
            "in the else of an if",
            "if p { x = 1u32; } else { let r = subgroup_add(x); }",
        ),
        ("under &&", "let r = p && subgroup_all(p);"),
        ("under ||", "let r = p || subgroup_any(p);"),
        ("in an index", "let r = data[subgroup_add(x)];"),
        (
            "as a method receiver",
            "let r = subgroup_add(x).wrapping_add(1u32);",
        ),
        ("as a statement of its own", "subgroup_add(x);"),
        (
            "in a loop with a run-time bound",
            "for step in 0u32..x { let r = subgroup_add(step); }",
        ),
        (
            "in a nested loop",
            "for a in 0u32..2u32 { for b in 0u32..2u32 { let r = subgroup_add(a + b); } }",
        ),
        (
            "in the value of an if",
            "let r = if p { subgroup_add(x) } else { 0u32 };",
        ),
        ("in an index", "let r = x + 1u32; out[subgroup_add(x)] = r;"),
    ] {
        let error = rejected(&pair(&body(statement)));
        assert!(
            error.to_string().contains("[GUST-E0747]"),
            "{what}: {error}"
        );
        assert_code(&error, Code::SubgroupPosition);
    }
}

#[test]
fn a_while_loop_is_refused_for_its_own_sake() {
    let error = rejected(&pair(&body(
        "while x < 4u32 { let r = subgroup_add(x); x += 1u32; }",
    )));
    assert_code(&error, Code::WhileNotSupported);
}

// ---------------------------------------------------------------------------------------------------------------
// GUST-E0415 and E0416: the early exits of a kernel that holds an operation

#[test]
fn a_kernel_with_an_operation_has_no_return() {
    for statement in [
        "let r = subgroup_add(x); return;",
        "if p { return; } let r = subgroup_add(x);",
    ] {
        let error = rejected(&pair(&body(statement)));
        assert_code(&error, Code::ReturnInBarrierKernel);
        assert!(
            error
                .to_string()
                .contains("a kernel that calls a subgroup operation has no `return`"),
            "{error}"
        );
    }
}

#[test]
fn a_jump_in_a_loop_that_holds_an_operation_is_refused() {
    for jump in ["break;", "continue;"] {
        let error = rejected(&pair(&body(&format!(
            "for step in 0u32..4u32 {{ let s = subgroup_add(x + step); if s > 3u32 {{ {jump} }} }}"
        ))));
        assert_code(&error, Code::JumpInBarrierLoop);
    }
    // A loop without one may jump.
    check(&pair(&body(
        "for step in 0u32..4u32 { if step == 2u32 { break; } x += step; } let r = subgroup_add(x);",
    )))
    .unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// GUST-E0751: whole subgroups

#[test]
fn the_invocation_count_of_a_body_with_operations_is_a_multiple_of_64() {
    let with_size = |size: &str, statement: &str| {
        pair(&body(statement)).replace(
            "#[kernel(workgroup_size(64, 1, 1))]",
            &format!("#[kernel(workgroup_size({size}))]"),
        )
    };
    for size in ["64, 1, 1", "128, 1, 1", "8, 8, 1", "16, 2, 2", "256, 1, 1"] {
        check(&with_size(size, "let r = subgroup_add(x);")).unwrap();
    }
    for size in ["32, 1, 1", "63, 1, 1", "100, 1, 1", "1, 1, 1", "8, 4, 1"] {
        let error = rejected(&with_size(size, "let r = subgroup_add(x);"));
        assert_code(&error, Code::SubgroupInvocations);
        assert!(
            error.to_string().contains("multiple of 64"),
            "{size}: {error}"
        );
    }
    // A body with no operation has no such rule: the portable body of the pair keeps its size.
    check(&with_size("32, 1, 1", "let r = x;")).unwrap();
}

/// m9.0 (Codex, M8 sanity pass): `workgroup_size` has no upper bound, and the product of the three dimensions was taken
/// in `u64`, so dimensions near `u32::MAX` overflowed it and the macro panicked (overflow checks are on in a dev build
/// of a proc macro) instead of reporting E0751. The product is exact in `u128`.
#[test]
fn huge_workgroup_dimensions_are_the_coded_error_with_the_exact_product() {
    let max = u128::from(u32::MAX);
    let with_size = |size: &str| {
        pair(&body("let r = subgroup_add(x);")).replace(
            "#[kernel(workgroup_size(64, 1, 1))]",
            &format!("#[kernel(workgroup_size({size}))]"),
        )
    };
    // None is a multiple of 64, so all are refused. The first (2^66 - 2^35 + 4) and the second (about 2^96) are over
    // `u64::MAX`, which the old product overflowed; the third, (2^32 - 1)^2, still fits a `u64` and is the control.
    for (size, product) in [
        ("4294967295, 4294967295, 4", max * max * 4),
        ("4294967295, 4294967295, 4294967295", max * max * max),
        ("4294967295, 1, 4294967295", max * max),
    ] {
        let error = rejected(&with_size(size));
        assert_code(&error, Code::SubgroupInvocations);
        let text = error.to_string();
        assert!(
            text.contains(&format!("`workgroup_size({size})` has {product}")),
            "{size}: {text}"
        );
        assert!(text.contains("multiple of 64"), "{size}: {text}");
    }
    // A product that is a multiple of 64 and over `u64` is accepted: 4294967232 is 2^32 - 64, which 64 divides.
    check(&with_size("4294967232, 4294967232, 64")).unwrap();
    check(&with_size("4294967232, 4294967232, 4294967232")).unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// The classifier

#[test]
fn a_subgroup_body_is_grid_only() {
    let lowered = check(&pair(&body("let r = subgroup_add(x);"))).unwrap();
    let helpers = helper_names(&lowered);
    let subgroup = kernel_named(&lowered, "k_subgroups");
    let (_, _, facts) = kernel_attribute_facts(&GRID, &subgroup).unwrap().unwrap();
    let classification = classify(&GRID, &helpers, &subgroup, &facts);
    assert!(!classification.allows(Shape::Pixel));
    let exclusion = classification
        .exclusions
        .iter()
        .find(|exclusion| exclusion.why == Why::Subgroup)
        .unwrap_or_else(|| panic!("{:?}", classification.exclusions));
    assert_eq!(exclusion.why.code(), Code::ShapeExcluded);
    assert!(
        exclusion
            .message("k_subgroups")
            .contains("subgroup operation"),
        "{}",
        exclusion.message("k_subgroups")
    );
    // The call is not a callee the ledger does not know: nothing is `Unrecognised`.
    assert!(
        !classification
            .exclusions
            .iter()
            .any(|exclusion| exclusion.why == Why::Unrecognised),
        "{:?}",
        classification.exclusions
    );
}

// ---------------------------------------------------------------------------------------------------------------
// The expansion

fn expand(source: &str, value: &str) -> String {
    let profile = Profile::parse(value).unwrap_or_else(|| panic!("{value:?} is a profile"));
    let environment = Environment {
        manifest_dir: None,
        export_dir: None,
        package: Some("demo".to_owned()),
        reported_file: None,
    };
    expand_in(parse_str(source).unwrap(), &environment, profile)
        .unwrap()
        .to_string()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

#[test]
fn the_fixture_expands_under_every_profile_and_the_selected_body_follows_the_capability() {
    let module = FIXTURE.replacen("#[gust::gpu]", "", 1);
    let portable = expand(&module, "dev");
    assert!(
        portable.contains("entry_point:\"gpu_subgroup_partial_sums\""),
        "{portable}"
    );
    assert!(!portable.contains("entry_point:\"gpu_subgroup_partial_sums_subgroups\""));
    assert!(portable.contains("variant:\"\",requires:&[],"));
    let wave = expand(&module, "dev+subgroups");
    assert!(
        wave.contains("entry_point:\"gpu_subgroup_partial_sums_subgroups\""),
        "{wave}"
    );
    assert!(wave.contains("requires:&[::gust::capability::Capability::Subgroups]"));
    // The subgroup operations are in the crate only when the build selected the body that has them.
    assert!(wave.contains("WaveActiveSum"), "{wave}");
    assert!(!portable.contains("WaveActiveSum"), "{portable}");
    // The `cpu` form is the portable body; the subgroup bodies have none (D42).
    expand(&module, "cpu");
}

/// m9.0, the audit of the other places that read the workgroup dimensions (the descriptor, the artifact record, the
/// entry point and the local and group ids of the Slang, the CPU twin's id arithmetic, the portable floor and the
/// workgroup-memory sum): a kernel without subgroup operations at the largest dimensions expands, under every profile
/// that has a form for it, without a panic, and carries the dimensions as the numbers they are.
#[test]
fn a_kernel_with_the_largest_workgroup_dimensions_expands_without_a_panic() {
    let source = "mod m {
    #[kernel(workgroup_size(4294967295, 4294967295, 4294967295))]
    pub fn k(id: SV_DispatchThreadID, local: LocalId, group: GroupId, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = local.x + group.y + local.z;
    }
}";
    for profile in ["dev", "native", "web", "dev+subgroups", "cpu"] {
        let text = expand(source, profile);
        assert!(
            text.contains("workgroup_size:[4294967295u32,4294967295u32,4294967295u32]"),
            "{profile}: {text}"
        );
    }
    // The twin divides by the sizes as literals, and the Slang does the same in `uint` arithmetic.
    let twin = expand(source, "dev");
    assert!(twin.contains("__gust_id.x%4294967295u32"), "{twin}");
    assert!(twin.contains("__gust_id.y/4294967295u32"), "{twin}");
    let slang = slang_of(source, "k");
    assert!(
        slang.contains("[numthreads(4294967295, 4294967295, 4294967295)]"),
        "{slang}"
    );
    assert!(
        slang.contains("uint3 local = uint3(")
            && slang.contains("% 4294967295u")
            && slang.contains("/ 4294967295u"),
        "{slang}"
    );
}

#[test]
fn the_cpu_form_of_the_kernel_with_a_subgroup_body_is_the_portable_body() {
    let module = FIXTURE.replacen("#[gust::gpu]", "", 1);
    let cpu = expand(&module, "cpu");
    assert!(!cpu.contains("Wave"), "{cpu}");
}

/// m8.b1 review N10: when `variant::lower` cannot re-spell an own attribute option, the fallback leaves it as written
/// and the next check reports it. One case: `requires(subgroups)` with `cpu = "lockstep"` and a `requires()` variant.
#[test]
fn an_own_requires_with_a_cpu_option_that_lower_cannot_respell_is_unknown_cpu_value() {
    let source = "mod m {
    #[kernel(workgroup_size(64, 1, 1), requires(subgroups), cpu = \"lockstep\")]
    pub fn k(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
    #[variant(of = k, requires())]
    pub fn k_portable(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
}";
    let error = rejected(source);
    assert_code(&error, Code::UnknownCpuValue);
}

//! Variants in the macro (m8.b1, M8 D139): `#[variant(of = k, requires(..))]`, the selection of a body, the
//! exhaustiveness against the declared profiles, GUST-E0740 to GUST-E0744 and GUST-E0750, and the schema 4 record.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use proc_macro2::{Span, TokenStream};

use crate::{
    diagnostics::{
        Code,
        testing::{assert_code, assert_codes, assert_rendered_code, parse_str},
    },
    expand::{
        Capability, CapabilitySet, Environment, Profile, content_key, expand_in, kernel_options,
        variant::{self, Choice, KernelBodies, VariantBody},
    },
    regression_tests::assert_golden,
    slang::emit_kernel,
};

const FIXTURE: &str = include_str!("../../../../tests/fixtures/variant.rs");
const GOLDEN: &str = include_str!("../../../../tests/fixtures/variant.slang");

const PARAMETERS: &str =
    "id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>";
const WORKGROUP: &str = "#[kernel(workgroup_size(64, 1, 1))]";
const NEEDS_SUBGROUPS: &str = "#[kernel(workgroup_size(64, 1, 1), requires(subgroups))]";
const MANIFEST: &str = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n";

/// A module with the kernel `sum` (attribute `kernel_attribute`) and `rest` beside it.
fn module_text(kernel_attribute: &str, rest: &str) -> String {
    format!(
        "mod reduce {{
    {kernel_attribute}
    pub fn sum({PARAMETERS}) {{
        if id.x < out.len() {{ out[id.x] = data[id.x]; }}
    }}
    {rest}
}}"
    )
}

/// A function of the same signature as `sum`, under `attribute`.
fn function_text(attribute: &str, name: &str) -> String {
    format!(
        "{attribute}
    pub fn {name}({PARAMETERS}) {{
        if id.x < out.len() {{ out[id.x] = data[id.x] + 0u32; }}
    }}"
    )
}

/// The module `reduce`: `sum` and its variant `sum_subgroups`.
fn pair() -> String {
    module_text(
        WORKGROUP,
        &function_text("#[variant(of = sum, requires(subgroups))]", "sum_subgroups"),
    )
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A scratch crate directory with `manifest` as its `Cargo.toml` and room for artifacts and records. Removed when
/// dropped.
struct Crate(PathBuf);

impl Crate {
    fn new(name: &str, manifest: &str) -> Self {
        let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("gust-m8b1-{}-{serial}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("Cargo.toml"), manifest).unwrap();
        Self(directory)
    }

    /// A crate whose manifest declares `profiles` (TOML text of the array), or none when `None`.
    fn declaring(name: &str, profiles: Option<&str>) -> Self {
        let table = profiles.map_or_else(String::new, |profiles| {
            format!("\n[package.metadata.gust]\nprofiles = {profiles}\n")
        });
        Self::new(name, &format!("{MANIFEST}{table}"))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Crate {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn environment(directory: Option<&Path>, export: Option<&Path>) -> Environment {
    Environment {
        manifest_dir: directory.map(Path::to_path_buf),
        export_dir: export.map(Path::to_path_buf),
        package: Some("demo".to_owned()),
        reported_file: None,
    }
}

fn expand(source: &str, profile: Profile) -> syn::Result<TokenStream> {
    expand_in(
        parse_str(source).unwrap(),
        &environment(None, None),
        profile,
    )
}

fn expand_declaring(
    source: &str,
    profiles: Option<&str>,
    profile: Profile,
) -> syn::Result<TokenStream> {
    let scratch = Crate::declaring("declared", profiles);
    expand_in(
        parse_str(source).unwrap(),
        &environment(Some(scratch.path()), None),
        profile,
    )
}

/// The tokens as text without whitespace, so assertions do not depend on token spacing.
fn squashed(tokens: &TokenStream) -> String {
    tokens
        .to_string()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

fn profile(value: &str) -> Profile {
    Profile::parse(value).unwrap_or_else(|| panic!("{value:?} is a profile"))
}

/// The error of an expansion that must fail.
fn rejected(result: syn::Result<TokenStream>) -> syn::Error {
    match result {
        Ok(tokens) => panic!("expected an error, got {tokens}"),
        Err(error) => error,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The fixture

fn fixture_module() -> String {
    FIXTURE.replacen("#[gust::gpu]", "", 1)
}

#[test]
fn the_variant_fixture_matches_the_golden() {
    let module: syn::ItemMod = parse_str(fixture_module().trim()).unwrap();
    let (lowered, plan) = variant::lower(module).unwrap();
    crate::validate::validate_module(&lowered).unwrap();
    assert!(plan.kernel("double").is_some());
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
    assert_golden(&actual, "variant.slang", &GOLDEN.replace("\r\n", "\n"));
}

#[test]
fn the_golden_has_one_entry_point_per_body_and_the_variant_has_the_workgroup_of_its_kernel() {
    let golden = GOLDEN.replace("\r\n", "\n");
    assert_eq!(golden.matches("void gpu_variant_double(").count(), 1);
    assert_eq!(
        golden.matches("void gpu_variant_double_subgroups(").count(),
        1
    );
    // The variant wrote no options: the workgroup size is the kernel's.
    assert_eq!(
        golden.matches("[numthreads(64, 1, 1)]").count(),
        2,
        "{golden}"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Selection and the descriptor

#[test]
fn a_build_without_the_capability_gets_the_portable_body() {
    for value in [None, Some("dev"), Some("native"), Some("web"), Some("cpu")] {
        let profile = Profile::from_value(value);
        let text = squashed(&expand(&pair(), profile).unwrap());
        assert!(
            text.contains("entry_point:\"gpu_reduce_sum\""),
            "{value:?}: {text}"
        );
        assert!(
            text.contains("variant:\"\",requires:&[],"),
            "{value:?}: {text}"
        );
        assert!(
            !text.contains("entry_point:\"gpu_reduce_sum_subgroups\""),
            "{value:?}: {text}"
        );
    }
}

#[test]
fn a_build_with_the_capability_gets_the_variant_in_the_descriptor_of_the_kernel() {
    for value in ["dev+subgroups", "native+subgroups"] {
        let text = squashed(&expand(&pair(), profile(value)).unwrap());
        assert!(
            text.contains("entry_point:\"gpu_reduce_sum_subgroups\""),
            "{value}: {text}"
        );
        assert!(
            text.contains(
                "variant:\"sum_subgroups\",requires:&[::gust::capability::Capability::Subgroups],"
            ),
            "{value}: {text}"
        );
        // The kernel keeps its name and its handle; the variant has none of its own.
        assert!(text.contains("name:stringify!(sum)"), "{value}: {text}");
        assert_eq!(text.matches("structsum;").count(), 1, "{text}");
        assert!(!text.contains("structsum_subgroups"), "{value}: {text}");
        assert!(
            !text.contains("entry_point:\"gpu_reduce_sum\""),
            "{value}: {text}"
        );
    }
}

#[test]
fn only_the_selected_bodys_slang_is_in_the_crate() {
    let portable = squashed(&expand(&pair(), profile("native")).unwrap());
    let subgroups = squashed(&expand(&pair(), profile("native+subgroups")).unwrap());
    // The two bodies differ in `+ 0u32`, which only the variant has.
    assert!(!portable.contains("+uint(0)"), "{portable}");
    assert!(subgroups.contains("+uint(0)"), "{subgroups}");
    assert!(
        !subgroups.contains("voidgpu_reduce_sum(uint3"),
        "{subgroups}"
    );
}

#[test]
fn the_twin_and_the_cpu_form_come_from_the_portable_body_whichever_is_selected() {
    for value in ["native", "native+subgroups"] {
        let text = squashed(&expand(&pair(), profile(value)).unwrap());
        // The variant function is defined once (as a type-checking shadow) and never called; the twin calls the
        // kernel's own shadow.
        assert_eq!(
            text.matches("__gpu_typecheck_sum_subgroups(").count(),
            1,
            "{value}: {text}"
        );
        assert!(
            text.matches("__gpu_typecheck_sum(").count() >= 2,
            "{value}: {text}"
        );
    }
    // Under the CPU target the kernel's own body is the one named.
    let cpu = squashed(&expand(&pair(), Profile::Cpu).unwrap());
    assert!(cpu.contains("entry_point:\"gpu_reduce_sum\""), "{cpu}");
}

#[test]
fn a_variant_is_the_kernels_signature_proved_by_rustc() {
    let text = squashed(&expand(&pair(), Profile::Other).unwrap());
    assert!(
        text.contains("const_:()={let_=[__gpu_typecheck_sum,__gpu_typecheck_sum_subgroups];};"),
        "{text}"
    );
    // A module without variants has no proof.
    let plain = squashed(&expand(&module_text(WORKGROUP, ""), Profile::Other).unwrap());
    assert!(!plain.contains("let_=["), "{plain}");
}

#[test]
fn a_module_without_variants_is_the_expansion_it_always_was() {
    let text = squashed(&expand(&module_text(WORKGROUP, ""), Profile::Other).unwrap());
    assert!(
        text.contains("variant:\"\",requires:&[],target:::gust::target::Target::Gpu"),
        "{text}"
    );
    assert!(text.contains("entry_point:\"gpu_reduce_sum\""), "{text}");
}

#[test]
fn the_kernel_may_carry_requires_and_a_portable_variant_may_have_the_empty_set() {
    let source = module_text(
        NEEDS_SUBGROUPS,
        &function_text("#[variant(of = sum, requires())]", "sum_portable"),
    );
    // No profile has the capability: the portable variant (an empty set) is the body.
    let text = squashed(&expand(&source, Profile::Other).unwrap());
    assert!(
        text.contains("entry_point:\"gpu_reduce_sum_portable\""),
        "{text}"
    );
    assert!(
        text.contains("variant:\"sum_portable\",requires:&[],"),
        "{text}"
    );
    // With the capability the kernel's own body is the larger set.
    let text = squashed(&expand(&source, profile("native+subgroups")).unwrap());
    assert!(text.contains("entry_point:\"gpu_reduce_sum\""), "{text}");
    assert!(
        text.contains("variant:\"\",requires:&[::gust::capability::Capability::Subgroups],"),
        "{text}"
    );
}

#[test]
fn the_aliases_are_sets_vulkan_is_subgroups_and_webgpu_is_none() {
    let source = module_text(
        WORKGROUP,
        &function_text("#[variant(of = sum, requires(vulkan))]", "sum_vulkan"),
    );
    let text = squashed(&expand(&source, profile("native+subgroups")).unwrap());
    assert!(
        text.contains(
            "variant:\"sum_vulkan\",requires:&[::gust::capability::Capability::Subgroups],"
        )
    );
    // `webgpu` is the empty set: the same set as the kernel's own body, so a tie (GUST-E0743).
    let source = module_text(
        WORKGROUP,
        &function_text("#[variant(of = sum, requires(webgpu))]", "sum_webgpu"),
    );
    assert_code(
        &rejected(expand(&source, Profile::Other)),
        Code::VariantAmbiguous,
    );
    // Names add up: `requires(vulkan, subgroups, webgpu)` is one capability.
    let source = module_text(
        WORKGROUP,
        &function_text(
            "#[variant(of = sum, requires(vulkan, subgroups, webgpu))]",
            "both",
        ),
    );
    let text = squashed(&expand(&source, profile("native+subgroups")).unwrap());
    assert!(
        text.contains("variant:\"both\",requires:&[::gust::capability::Capability::Subgroups],")
    );
}

/// `KernelBodies` for `sum` with an own set and variants given as bits.
struct BodiesBuilder(KernelBodies);

impl BodiesBuilder {
    fn new(own: u8) -> Self {
        Self(KernelBodies {
            kernel: "sum".to_owned(),
            span: Span::call_site(),
            own: CapabilitySet::from_bits(own),
            variants: Vec::new(),
        })
    }

    fn variant(&mut self, name: &str, requires: u8) {
        self.0.variants.push(VariantBody {
            name: name.to_owned(),
            requires: CapabilitySet::from_bits(requires),
        });
    }

    fn build(&self) -> KernelBodies {
        self.0.clone()
    }
}

#[test]
fn several_variants_are_chosen_by_the_largest_set_the_profile_contains() {
    let none = BodiesBuilder::new(0b00);
    let mut many = BodiesBuilder::new(0b00);
    many.variant("a", 0b01);
    many.variant("b", 0b10);
    many.variant("c", 0b11);
    let many = many.build();
    assert_eq!(
        none.build().choose(CapabilitySet::from_bits(0b11)),
        Choice::Body(0)
    );
    assert_eq!(many.choose(CapabilitySet::from_bits(0b11)), Choice::Body(3));
    assert_eq!(many.choose(CapabilitySet::from_bits(0b01)), Choice::Body(1));
    assert_eq!(many.choose(CapabilitySet::from_bits(0b10)), Choice::Body(2));
    assert_eq!(many.choose(CapabilitySet::from_bits(0b00)), Choice::Body(0));
    assert_eq!(many.function(0), "sum");
    assert_eq!(many.function(3), "c");
    assert_eq!(many.first_tie(2), None);
    // A kernel whose own body needs a capability has no body without it.
    let needs = BodiesBuilder::new(0b01).build();
    assert_eq!(needs.choose(CapabilitySet::from_bits(0)), Choice::NoneFits);
    assert_eq!(needs.choose(CapabilitySet::from_bits(1)), Choice::Body(0));
}

#[test]
fn two_bodies_of_the_same_size_that_a_profile_can_have_are_a_tie() {
    let mut bodies = BodiesBuilder::new(0b00);
    bodies.variant("a", 0b01);
    bodies.variant("b", 0b10);
    let bodies = bodies.build();
    assert_eq!(
        bodies.choose(CapabilitySet::from_bits(0b11)),
        Choice::Tie(1, 2)
    );
    // The one capability this release knows cannot tell them apart, so only two can.
    assert_eq!(bodies.first_tie(1), None);
    assert_eq!(
        bodies.first_tie(2),
        Some((1, 2, CapabilitySet::from_bits(0b11)))
    );
    // Equal sets tie as soon as the set is available.
    let mut equal = BodiesBuilder::new(0b01);
    equal.variant("a", 0b01);
    let equal = equal.build();
    assert_eq!(
        equal.first_tie(1),
        Some((0, 1, CapabilitySet::from_bits(0b01)))
    );
}

#[test]
fn a_set_names_its_members_in_the_order_of_the_capabilities() {
    assert_eq!(CapabilitySet::EMPTY.names(), Vec::<&str>::new());
    assert_eq!(CapabilitySet::from_bits(1).names(), ["subgroups"]);
    assert_eq!(
        CapabilitySet::from_bits(1),
        CapabilitySet::EMPTY.with(Capability::Subgroups)
    );
    assert!(CapabilitySet::EMPTY.is_subset_of(CapabilitySet::from_bits(1)));
    assert!(!CapabilitySet::from_bits(1).is_subset_of(CapabilitySet::EMPTY));
    assert_eq!(Capability::ALL, [Capability::Subgroups]);
}

// ---------------------------------------------------------------------------------------------------------------
// GUST-E0740 to GUST-E0743: the attribute

#[test]
fn a_malformed_variant_attribute_is_e0740() {
    let cases = [
        // No arguments, no `of`, an unknown option, a duplicate option, a value that is not a name.
        ("#[variant]", "bare"),
        ("#[variant(requires(subgroups))]", "no_of"),
        ("#[variant(of = sum, tier = 3)]", "tier"),
        ("#[variant(of = sum, of = sum)]", "of_twice"),
        (
            "#[variant(of = sum, requires(subgroups), requires(subgroups))]",
            "requires_twice",
        ),
        ("#[variant(of = \"sum\")]", "string"),
        // The options of the kernel are the kernel's.
        ("#[variant(of = sum, workgroup_size(8, 1, 1))]", "workgroup"),
        ("#[variant(of = sum, shape = pixel)]", "shape"),
        ("#[variant(of = sum, cpu = \"unsupported\")]", "cpu"),
    ];
    for (attribute, name) in cases {
        let source = module_text(WORKGROUP, &function_text(attribute, name));
        let error = rejected(expand(&source, Profile::Other));
        assert_code(&error, Code::VariantAttribute);
    }
    let source = module_text(
        WORKGROUP,
        &function_text("#[variant(of = sum, workgroup_size(8, 1, 1))]", "w"),
    );
    let error = rejected(expand(&source, Profile::Other));
    assert_eq!(
        error.to_string(),
        "[GUST-E0740] #[variant] takes `of = <kernel>` and an optional `requires(..)`; the workgroup size, the shape and the CPU form are the kernel's"
    );
}

#[test]
fn a_variant_stands_on_a_function_of_a_module_and_is_not_a_kernel() {
    let on_struct = module_text(
        WORKGROUP,
        "#[variant(of = sum)]\n    pub struct Stray { pub x: u32 }",
    );
    assert_code(
        &rejected(expand(&on_struct, Profile::Other)),
        Code::VariantAttribute,
    );
    let both = module_text(
        WORKGROUP,
        &function_text(
            "#[kernel]\n    #[variant(of = sum, requires(subgroups))]",
            "both",
        ),
    );
    assert_code(
        &rejected(expand(&both, Profile::Other)),
        Code::VariantAttribute,
    );
    let twice = module_text(
        WORKGROUP,
        &function_text(
            "#[variant(of = sum, requires(subgroups))]\n    #[variant(of = sum)]",
            "twice",
        ),
    );
    assert_code(
        &rejected(expand(&twice, Profile::Other)),
        Code::VariantAttribute,
    );
}

#[test]
fn a_variant_has_the_signature_of_its_kernel_token_for_token() {
    let renamed = "#[variant(of = sum, requires(subgroups))]
    pub fn other(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x];
    }";
    let error = rejected(expand(&module_text(WORKGROUP, renamed), Profile::Other));
    assert_code(&error, Code::VariantAttribute);
    assert!(
        error.to_string().starts_with(
            "[GUST-E0740] variant `other` does not have the signature of kernel `sum`"
        ),
        "{error}"
    );
    let other_type = "#[variant(of = sum, requires(subgroups))]
    pub fn other(id: SV_DispatchThreadID, data: StructuredBuffer<i32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 0u32;
    }";
    assert_code(
        &rejected(expand(&module_text(WORKGROUP, other_type), Profile::Other)),
        Code::VariantAttribute,
    );
    let fewer = "#[variant(of = sum, requires(subgroups))]
    pub fn other(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 0u32;
    }";
    assert_code(
        &rejected(expand(&module_text(WORKGROUP, fewer), Profile::Other)),
        Code::VariantAttribute,
    );
}

#[test]
fn a_variant_on_a_gpu_function_is_e0740_and_the_function_comes_back_without_it() {
    let item: TokenStream = "#[variant(of = other)] pub fn doubled(x: f32) -> f32 { x * 2.0 }"
        .parse()
        .unwrap();
    let text = crate::gpu_impl(TokenStream::new(), item).to_string();
    assert_rendered_code(&text, Code::VariantAttribute);
    assert_eq!(text.matches("compile_error").count(), 1, "{text}");
    let squashed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        squashed.contains("pubfndoubled(x:f32)->f32{x*2.0}"),
        "{text}"
    );
    assert!(!squashed.contains("#[variant(of=other)]pubfn"), "{text}");
}

#[test]
fn a_rejected_module_with_variants_goes_back_with_one_error_only() {
    // The variants return as kernels (renamed, attribute-free), the stray `#[variant]` is dropped: rustc has no
    // unknown attribute to add a second error for.
    let item: TokenStream = module_text(
        WORKGROUP,
        &format!(
            "{}\n    #[variant(of = missing)]\n    pub struct Stray {{ pub x: u32 }}",
            function_text("#[variant(of = sum, requires(subgroups))]", "sum_subgroups")
        ),
    )
    .parse()
    .unwrap();
    let text = crate::gpu_impl(TokenStream::new(), item).to_string();
    assert_rendered_code(&text, Code::VariantAttribute);
    let squashed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        squashed.contains("fn__gpu_typecheck_sum_subgroups("),
        "{text}"
    );
    assert!(!squashed.contains("#[variant(of=sum"), "{text}");
    assert!(
        !squashed.contains("#[variant(of=missing)]pubstruct"),
        "{text}"
    );
}

#[test]
fn an_unknown_capability_or_alias_is_e0741() {
    for requires in [
        "subgroup",
        "f16",
        "atomics64",
        "Subgroups",
        "gpu",
        "subgroups, simd",
    ] {
        let source = module_text(
            WORKGROUP,
            &function_text(&format!("#[variant(of = sum, requires({requires}))]"), "v"),
        );
        let error = rejected(expand(&source, Profile::Other));
        assert_code(&error, Code::UnknownCapability);
    }
    let source = module_text(
        WORKGROUP,
        &function_text("#[variant(of = sum, requires(subgroup))]", "v"),
    );
    let error = rejected(expand(&source, Profile::Other));
    assert_eq!(
        error.to_string(),
        "[GUST-E0741] unknown capability `subgroup` in `requires(..)`; the capabilities are `subgroups`, the aliases `vulkan` (`subgroups`) and `webgpu` (none)"
    );
    // The kernel's own `requires(..)` is read the same way.
    let source = module_text("#[kernel(workgroup_size(64, 1, 1), requires(f16))]", "");
    assert_code(
        &rejected(expand(&source, Profile::Other)),
        Code::UnknownCapability,
    );
}

#[test]
fn a_variant_of_nothing_is_e0742() {
    let source = module_text(
        WORKGROUP,
        &function_text(
            "#[variant(of = prod, requires(subgroups))]",
            "sum_subgroups",
        ),
    );
    let error = rejected(expand(&source, Profile::Other));
    assert_code(&error, Code::VariantTarget);
    assert_eq!(
        error.to_string(),
        "[GUST-E0742] #[variant(of = prod)] on `sum_subgroups` names no #[kernel] function of this module"
    );
    // A helper is no kernel.
    let source = module_text(
        WORKGROUP,
        &format!(
            "fn helper(x: u32) -> u32 {{ x }}\n    {}",
            function_text("#[variant(of = helper, requires(subgroups))]", "v")
        ),
    );
    assert_code(
        &rejected(expand(&source, Profile::Other)),
        Code::VariantTarget,
    );
    // A variant of a variant is no kernel either.
    let source = module_text(
        WORKGROUP,
        &format!(
            "{}\n    {}",
            function_text("#[variant(of = sum, requires(subgroups))]", "first"),
            function_text("#[variant(of = first)]", "second")
        ),
    );
    let error = rejected(expand(&source, Profile::Other));
    assert_code(&error, Code::VariantTarget);
    assert!(
        error.to_string().ends_with("names a variant, not a kernel"),
        "{error}"
    );
}

#[test]
fn two_bodies_with_one_set_are_e0743() {
    // Two variants with one set (one spelled by its alias).
    let source = module_text(
        WORKGROUP,
        &format!(
            "{}\n    {}",
            function_text("#[variant(of = sum, requires(subgroups))]", "first"),
            function_text("#[variant(of = sum, requires(vulkan))]", "second")
        ),
    );
    let error = rejected(expand(&source, Profile::Other));
    assert_code(&error, Code::VariantAmbiguous);
    assert_eq!(
        error.to_string(),
        "[GUST-E0743] kernel `sum` has two bodies that a device with {subgroups} cannot tell apart: `first` requires {subgroups} and `second` requires {subgroups}; give them sets of different sizes, or one a superset of the other"
    );
    // A variant with the empty set beside a kernel with the empty set.
    let source = module_text(WORKGROUP, &function_text("#[variant(of = sum)]", "again"));
    assert_code(
        &rejected(expand(&source, Profile::Other)),
        Code::VariantAmbiguous,
    );
    // It is an error in every build, the CPU target included.
    assert_code(
        &rejected(expand(&source, Profile::Cpu)),
        Code::VariantAmbiguous,
    );
}

#[test]
fn the_first_fault_in_source_order_is_the_one_reported() {
    let source = module_text(
        WORKGROUP,
        &format!(
            "{}\n    {}",
            function_text("#[variant(of = sum, requires(nothing))]", "first"),
            function_text("#[variant(of = missing)]", "second")
        ),
    );
    assert_codes(
        &rejected(expand(&source, Profile::Other)),
        &[Code::UnknownCapability],
    );
}

// ---------------------------------------------------------------------------------------------------------------
// GUST-E0744 and GUST-E0750: the declared profiles

#[test]
fn a_declared_gpu_profile_without_a_body_is_e0744() {
    let source = module_text(NEEDS_SUBGROUPS, "");
    // The default list has `native` and `web`, which have no capability.
    let error = rejected(expand_declaring(&source, None, Profile::Other));
    assert_code(&error, Code::ProfileNotCovered);
    assert_eq!(
        error.to_string(),
        "[GUST-E0744] kernel `sum` has no body for the declared profile `native` (the default list; the manifest has no `profiles` key): `sum` requires {subgroups}; declare a profile with the capabilities in `[package.metadata.gust] profiles`, or add a variant that requires less"
    );
    // An explicit list says nothing of the kind.
    let error = rejected(expand_declaring(
        &source,
        Some(r#"["native"]"#),
        Profile::Other,
    ));
    assert_eq!(
        error.to_string(),
        "[GUST-E0744] kernel `sum` has no body for the declared profile `native`: `sum` requires {subgroups}; declare a profile with the capabilities in `[package.metadata.gust] profiles`, or add a variant that requires less"
    );
    // An explicit list that has the capability on every GPU entry is covered.
    assert!(
        expand_declaring(
            &source,
            Some(r#"["native+subgroups"]"#),
            profile("native+subgroups")
        )
        .is_ok()
    );
    // `web` cannot have the capability, so a list that declares it needs a body without.
    let error = rejected(expand_declaring(
        &source,
        Some(r#"["native+subgroups", "web"]"#),
        profile("native+subgroups"),
    ));
    assert_code(&error, Code::ProfileNotCovered);
    assert!(
        error.to_string().contains("declared profile `web`"),
        "{error}"
    );
    // The portable variant covers every entry.
    let covered = module_text(
        NEEDS_SUBGROUPS,
        &function_text("#[variant(of = sum, requires())]", "sum_portable"),
    );
    assert!(expand_declaring(&covered, None, Profile::Other).is_ok());
    // The check is made in every build, the CPU target included.
    assert_code(
        &rejected(expand_declaring(&source, None, Profile::Cpu)),
        Code::ProfileNotCovered,
    );
}

#[test]
fn a_kernel_without_variants_or_requirements_is_covered_by_any_list() {
    let source = module_text(WORKGROUP, "");
    for profiles in [
        None,
        Some(r#"["native"]"#),
        Some(r#"["native+subgroups", "web", "cpu"]"#),
    ] {
        assert!(
            expand_declaring(&source, profiles, Profile::Other).is_ok(),
            "{profiles:?}"
        );
    }
}

#[test]
fn a_build_that_no_body_can_use_is_e0750() {
    let source = module_text(NEEDS_SUBGROUPS, "");
    let declared = Some(r#"["native+subgroups"]"#);
    // `dev` has no capability: the kernel is declared for `native+subgroups`, not built for it.
    for (value, text) in [
        (Profile::from_value(Some("dev")), "`dev`"),
        (Profile::from_value(Some("native")), "`native`"),
        (
            Profile::Other,
            "(`GUST_PROFILE` unset or unknown; no capability)",
        ),
    ] {
        let error = rejected(expand_declaring(&source, declared, value));
        assert_code(&error, Code::NoBodySelectable);
        assert_eq!(
            error.to_string(),
            format!(
                "[GUST-E0750] kernel `sum` has no body that the build profile {text} can use: `sum` requires {{subgroups}}; build with a profile that provides them, or add a variant that requires less"
            )
        );
    }
    // The profile with the capability builds the kernel's own body; the CPU target builds no GPU body at all.
    for value in [
        profile("dev+subgroups"),
        profile("native+subgroups"),
        Profile::Cpu,
    ] {
        assert!(
            expand_declaring(&source, declared, value).is_ok(),
            "{value:?}"
        );
    }
}

#[test]
fn the_cpu_entry_of_an_explicit_list_needs_a_cpu_form() {
    let gpu_only = module_text(
        "#[kernel(workgroup_size(64, 1, 1), cpu = \"unsupported\")]",
        "",
    );
    // `cpu = "unsupported"` is the explicit arm that meets the default list's CPU entry (Q4), with no manifest and
    // with a manifest that has no list.
    assert!(expand(&gpu_only, Profile::Other).is_ok());
    assert!(expand_declaring(&gpu_only, None, Profile::Other).is_ok());
    // A list that does not name `cpu` does not need it.
    assert!(expand_declaring(&gpu_only, Some(r#"["native", "web"]"#), Profile::Other).is_ok());
    // An explicit list that names it does, even when it equals the default list.
    for profiles in [r#"["cpu"]"#, r#"["native", "web", "cpu"]"#] {
        let error = rejected(expand_declaring(&gpu_only, Some(profiles), Profile::Other));
        assert_code(&error, Code::ProfileNotCovered);
        let text = error.to_string();
        assert!(text.contains("remove `cpu = \"unsupported\"`"), "{text}");
        assert!(text.contains("drop `cpu` from `profiles`"), "{text}");
    }
    // Every build checks it, a CPU build included; a kernel with a twin meets the entry.
    assert_code(
        &rejected(expand_declaring(
            &gpu_only,
            Some(r#"["cpu"]"#),
            Profile::Cpu,
        )),
        Code::ProfileNotCovered,
    );
    let with_twin = module_text(WORKGROUP, "");
    assert!(expand_declaring(&with_twin, Some(r#"["cpu"]"#), Profile::Other).is_ok());
}

// ---------------------------------------------------------------------------------------------------------------
// The export records and the artifacts

/// What export mode wrote for `source` under `profile`: the `(entry point, record)` of every `*.kernel.json`, sorted.
fn exported(source: &str, profile: Profile) -> Vec<(String, String)> {
    exported_declaring(source, None, profile)
}

/// [`exported`] for a crate whose manifest declares `profiles` (TOML text of the array), or none when `None`.
fn exported_declaring(
    source: &str,
    profiles: Option<&str>,
    profile: Profile,
) -> Vec<(String, String)> {
    let scratch = Crate::declaring("export", profiles);
    let export = scratch.path().join("exports");
    expand_in(
        parse_str(source).unwrap(),
        &environment(Some(scratch.path()), Some(&export)),
        profile,
    )
    .unwrap();
    let mut records = Vec::new();
    for entry in fs::read_dir(export.join("demo")).unwrap() {
        let path = entry.unwrap().path();
        if path.to_string_lossy().ends_with(".kernel.json") {
            let text = fs::read_to_string(&path).unwrap();
            let entry_point = text
                .lines()
                .find_map(|line| line.trim().strip_prefix("\"entry_point\": \""))
                .map(|rest| rest.trim_end_matches(['"', ',']).to_owned())
                .unwrap();
            records.push((entry_point, text));
        }
    }
    records.sort();
    records
}

#[test]
fn every_body_writes_its_record_whatever_the_profile_selects() {
    let under_native = exported(&pair(), profile("native"));
    let under_subgroups = exported(&pair(), profile("native+subgroups"));
    assert_eq!(under_native, under_subgroups);
    assert_eq!(
        under_native
            .iter()
            .map(|(entry, _)| entry.as_str())
            .collect::<Vec<_>>(),
        ["gpu_reduce_sum", "gpu_reduce_sum_subgroups"]
    );
    // The portable body keeps its schema, byte for byte: no `requires`, no `variant`.
    let portable = &under_native[0].1;
    assert!(portable.contains("\"schema\": 1,\n"), "{portable}");
    assert!(
        !portable.contains("requires") && !portable.contains("\"variant\""),
        "{portable}"
    );
    assert!(portable.contains("\"name\": \"sum\",\n"), "{portable}");
    // The variant is schema 4, names the kernel it belongs to, and says which body it is.
    let record = &under_native[1].1;
    assert!(record.contains("\"schema\": 4,\n"), "{record}");
    assert!(record.contains("\"name\": \"sum\",\n"), "{record}");
    assert!(
        record.contains("\"requires\": [\n    \"subgroups\"\n  ],\n"),
        "{record}"
    );
    assert!(
        record.contains("\"variant\": \"sum_subgroups\",\n"),
        "{record}"
    );
    assert!(
        record.contains("\"workgroup_size\": [64, 1, 1]"),
        "{record}"
    );
}

#[test]
fn a_kernel_that_requires_writes_schema_4_with_the_empty_variant_and_a_portable_variant_the_empty_set()
 {
    let source = module_text(
        NEEDS_SUBGROUPS,
        &function_text("#[variant(of = sum, requires())]", "sum_portable"),
    );
    let records = exported(&source, Profile::Other);
    assert_eq!(records.len(), 2);
    let own = &records[0].1;
    assert!(own.contains("\"entry_point\": \"gpu_reduce_sum\""), "{own}");
    assert!(
        own.contains("\"schema\": 4,\n") && own.contains("\"variant\": \"\",\n"),
        "{own}"
    );
    assert!(
        own.contains("\"requires\": [\n    \"subgroups\"\n  ],\n"),
        "{own}"
    );
    let portable = &records[1].1;
    assert!(
        portable.contains("\"schema\": 4,\n") && portable.contains("\"requires\": [],\n"),
        "{portable}"
    );
    assert!(
        portable.contains("\"variant\": \"sum_portable\",\n"),
        "{portable}"
    );
}

#[test]
fn a_module_without_variants_exports_what_it_always_did() {
    let records = exported(&module_text(WORKGROUP, ""), Profile::Other);
    assert_eq!(records.len(), 1);
    assert!(records[0].1.contains("\"schema\": 1,\n"));
    assert!(!records[0].1.contains("requires") && !records[0].1.contains("variant"));
}

#[test]
fn the_content_key_of_a_body_is_its_entry_point_and_its_clean_slang() {
    let scratch = Crate::declaring("keys", None);
    let export = scratch.path().join("exports");
    expand_in(
        parse_str(&pair()).unwrap(),
        &environment(Some(scratch.path()), Some(&export)),
        Profile::Other,
    )
    .unwrap();
    let mut seen = 0;
    for entry in fs::read_dir(export.join("demo")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Some(key) = name.strip_suffix(".slang") else {
            continue;
        };
        let slang = fs::read_to_string(&path).unwrap();
        let entry_point = if slang.contains("void gpu_reduce_sum_subgroups(") {
            "gpu_reduce_sum_subgroups"
        } else {
            "gpu_reduce_sum"
        };
        assert_eq!(content_key(entry_point, &slang), key);
        seen += 1;
    }
    assert_eq!(seen, 2);
}

#[test]
fn only_the_selected_bodys_artifact_is_embedded() {
    let scratch = Crate::declaring("artifacts", None);
    let export = scratch.path().join("exports");
    expand_in(
        parse_str(&pair()).unwrap(),
        &environment(Some(scratch.path()), Some(&export)),
        Profile::Other,
    )
    .unwrap();
    // Give both bodies a complete pair of artifacts (the bytes are not read by the expansion).
    let mut keys = Vec::new();
    let artifacts = scratch.path().join("gust-artifacts");
    fs::create_dir_all(&artifacts).unwrap();
    for entry in fs::read_dir(export.join("demo")).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if let Some(key) = name.strip_suffix(".kernel.json") {
            fs::write(artifacts.join(format!("{key}.wgsl")), "").unwrap();
            fs::write(artifacts.join(format!("{key}.wgsl.reflection.json")), "{}").unwrap();
            keys.push(key.to_owned());
        }
    }
    assert_eq!(keys.len(), 2);
    for (value, selected) in [
        ("native", "gpu_reduce_sum"),
        ("native+subgroups", "gpu_reduce_sum_subgroups"),
    ] {
        let text = squashed(
            &expand_in(
                parse_str(&pair()).unwrap(),
                &environment(Some(scratch.path()), None),
                profile(value),
            )
            .unwrap(),
        );
        let embedded: Vec<&String> = keys
            .iter()
            .filter(|key| text.contains(&format!("{key}.wgsl\"")))
            .collect();
        assert_eq!(embedded.len(), 1, "{value}: {text}");
        assert!(
            text.contains(&format!("entry_point:\"{selected}\"")),
            "{value}: {text}"
        );
        // One artifact in the descriptor, and one `require_artifact(true)` for the selected body only.
        assert_eq!(
            text.matches("::gust::Artifact{").count(),
            1,
            "{value}: {text}"
        );
        assert_eq!(
            text.matches("require_artifact(true)").count(),
            1,
            "{value}: {text}"
        );
        assert_eq!(
            text.matches("require_artifact(false)").count(),
            0,
            "{value}: {text}"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Fix round 1 (review of m8.b1): export mode, the lowered attribute, the portable body, the nits

#[test]
fn export_mode_writes_every_body_of_a_kernel_the_build_profile_cannot_use() {
    // `cargo gust build` collects under a fixed `GUST_PROFILE=dev`; a crate that declares only `native+subgroups` and
    // has a kernel with no portable body is valid, and its record is written (M1).
    let source = module_text(NEEDS_SUBGROUPS, "");
    let declared = Some(r#"["native+subgroups"]"#);
    let records = exported_declaring(&source, declared, profile("dev"));
    assert_eq!(records.len(), 1);
    let record = &records[0].1;
    assert_eq!(records[0].0, "gpu_reduce_sum");
    assert!(record.contains("\"schema\": 4,\n"), "{record}");
    assert!(record.contains("\"variant\": \"\",\n"), "{record}");
    assert!(
        record.contains("\"requires\": [\n    \"subgroups\"\n  ],\n"),
        "{record}"
    );
    // Every build tool profile gets the same records, and the same crate outside export mode is still E0750.
    assert_eq!(
        exported_declaring(&source, declared, Profile::Other),
        records
    );
    assert_code(
        &rejected(expand_declaring(&source, declared, profile("dev"))),
        Code::NoBodySelectable,
    );
}

/// The `#[kernel(..)]` attribute of the function `name` of the lowered module, as text without whitespace.
fn lowered_kernel_attribute(source: &str, name: &str) -> String {
    let (module, _) = variant::lower(parse_str(source).unwrap()).unwrap();
    for item in &module.content.as_ref().unwrap().1 {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        if function.sig.ident == name {
            let attributes: TokenStream = function
                .attrs
                .iter()
                .filter(|attribute| attribute.path().is_ident("kernel"))
                .map(quote::ToTokens::to_token_stream)
                .collect();
            return squashed(&attributes);
        }
    }
    panic!("no function `{name}`");
}

#[test]
fn a_lowered_variant_has_the_layout_of_its_kernel_and_its_own_requirement_and_no_twin() {
    assert_eq!(
        lowered_kernel_attribute(&pair(), "sum_subgroups"),
        "#[kernel(workgroup_size(64,1,1),cpu=\"unsupported\",requires(subgroups))]"
    );
    // The kernel keeps its own attribute word for word.
    assert_eq!(
        lowered_kernel_attribute(&pair(), "sum"),
        "#[kernel(workgroup_size(64,1,1))]"
    );
    // A kernel's `shape`, `workgroup_size` (in the order written) pass on; its `cpu` does not repeat.
    let pinned = module_text(
        "#[kernel(shape = grid, cpu = \"unsupported\", workgroup_size(32, 1, 1))]",
        &function_text("#[variant(of = sum, requires(subgroups))]", "sum_subgroups"),
    );
    assert_eq!(
        lowered_kernel_attribute(&pinned, "sum_subgroups"),
        "#[kernel(shape=grid,workgroup_size(32,1,1),cpu=\"unsupported\",requires(subgroups))]"
    );
    assert!(expand(&pinned, Profile::Other).is_ok());
    // A kernel with a bare `#[kernel]` has the default layout, which the variant leaves unwritten.
    let bare = module_text(
        "#[kernel]",
        &function_text("#[variant(of = sum, requires(subgroups))]", "sum_subgroups"),
    );
    assert_eq!(
        lowered_kernel_attribute(&bare, "sum_subgroups"),
        "#[kernel(cpu=\"unsupported\",requires(subgroups))]"
    );
}

#[test]
fn a_portable_variant_of_a_kernel_that_requires_lowers_with_no_requirement() {
    let source = module_text(
        NEEDS_SUBGROUPS,
        &function_text("#[variant(of = sum, requires())]", "sum_portable"),
    );
    // Not the kernel's `requires(subgroups)`: the body is portable, and a subgroup operation in it would be refused
    // (m8.s reads the attribute).
    assert_eq!(
        lowered_kernel_attribute(&source, "sum_portable"),
        "#[kernel(workgroup_size(64,1,1),cpu=\"unsupported\")]"
    );
    assert_eq!(
        lowered_kernel_attribute(&source, "sum"),
        "#[kernel(workgroup_size(64,1,1),requires(subgroups))]"
    );
}

#[test]
fn a_variant_may_use_a_barrier_of_a_kernel_that_has_a_cpu_twin() {
    let barrier = "#[variant(of = sum, requires(subgroups))]
    pub fn sum_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        workgroup_barrier();
        if id.x < out.len() { out[id.x] = data[id.x]; }
    }";
    // `sum` is twin-able (no `cpu = \"unsupported\"`), yet its variant has no twin and so no E0717.
    assert!(expand(&module_text(WORKGROUP, barrier), Profile::Other).is_ok());
    assert!(
        expand(
            &module_text(WORKGROUP, barrier),
            profile("native+subgroups")
        )
        .is_ok()
    );
    // The same body as a kernel of its own is E0717: the test would fail on the old lowering.
    let as_kernel = barrier.replace("#[variant(of = sum, requires(subgroups))]", WORKGROUP);
    assert_code(
        &rejected(expand(&module_text(WORKGROUP, &as_kernel), Profile::Other)),
        Code::BarrierKernelCpu,
    );
}

#[test]
fn the_twin_and_the_cpu_descriptor_come_from_the_empty_set_body_not_the_kernels_own() {
    let source = module_text(
        NEEDS_SUBGROUPS,
        &function_text("#[variant(of = sum, requires())]", "sum_portable"),
    );
    for value in ["native", "native+subgroups"] {
        let text = squashed(&expand(&source, profile(value)).unwrap());
        // The twin calls the portable variant; the kernel's own body (which needs subgroups) is only defined.
        assert!(
            text.matches("__gpu_typecheck_sum_portable(").count() >= 2,
            "{value}: {text}"
        );
        assert_eq!(
            text.matches("__gpu_typecheck_sum(").count(),
            1,
            "{value}: {text}"
        );
        assert!(
            text.contains("for::gust::cpu::CpuDevice"),
            "{value}: {text}"
        );
    }
    // The `cpu` target builds the portable body: its descriptor, and an ungated twin that calls it.
    let cpu = squashed(&expand(&source, Profile::Cpu).unwrap());
    assert!(
        cpu.contains("entry_point:\"gpu_reduce_sum_portable\""),
        "{cpu}"
    );
    assert!(!cpu.contains("entry_point:\"gpu_reduce_sum\""), "{cpu}");
    assert!(
        cpu.matches("__gpu_typecheck_sum_portable(").count() >= 2,
        "{cpu}"
    );
    assert_eq!(cpu.matches("__gpu_typecheck_sum(").count(), 1, "{cpu}");
}

#[test]
fn a_kernel_with_no_empty_set_body_has_no_twin_and_no_cpu_form() {
    let source = module_text(NEEDS_SUBGROUPS, "");
    let declared = Some(r#"["native+subgroups"]"#);
    let text = squashed(&expand_declaring(&source, declared, profile("native+subgroups")).unwrap());
    assert!(!text.contains("for::gust::cpu::CpuDevice"), "{text}");
    assert_eq!(text.matches("__gpu_typecheck_sum(").count(), 1, "{text}");
    // Under the CPU target the build is still valid (it builds no GPU body), with no twin.
    let cpu = squashed(&expand_declaring(&source, declared, Profile::Cpu).unwrap());
    assert!(!cpu.contains("for::gust::cpu::CpuDevice"), "{cpu}");
    // A list that names `cpu` needs a CPU form, whatever the kernel's `cpu` option says (E0744).
    let error = rejected(expand_declaring(
        &source,
        Some(r#"["native+subgroups", "cpu"]"#),
        profile("native+subgroups"),
    ));
    assert_code(&error, Code::ProfileNotCovered);
    assert_eq!(
        error.to_string(),
        "[GUST-E0744] kernel `sum` has no CPU form, but `[package.metadata.gust] profiles` declares `cpu`: `sum` requires {subgroups}, and the CPU form is the body that requires nothing; add a variant with `requires()`, or drop `cpu` from `profiles`"
    );
    // The empty-set variant is the CPU form.
    let portable = module_text(
        NEEDS_SUBGROUPS,
        &function_text("#[variant(of = sum, requires())]", "sum_portable"),
    );
    assert!(
        expand_declaring(
            &portable,
            Some(r#"["native+subgroups", "cpu"]"#),
            profile("native+subgroups")
        )
        .is_ok()
    );
}

#[test]
fn a_second_requires_on_a_kernel_is_e0740_like_on_a_variant() {
    let source = module_text(
        "#[kernel(workgroup_size(64, 1, 1), requires(subgroups), requires(subgroups))]",
        "",
    );
    let error = rejected(expand(&source, Profile::Other));
    assert_code(&error, Code::VariantAttribute);
    assert_eq!(
        error.to_string(),
        "[GUST-E0740] duplicate `requires` option"
    );
}

#[test]
fn a_variant_attribute_anywhere_but_on_a_function_of_the_module_is_e0740_and_one_error() {
    let strays = [
        "#[variant(of = sum)]\n    macro_rules! m { () => {}; }",
        "pub struct S;\n    impl S {\n        #[variant(of = sum)]\n        pub fn f(&self) {}\n    }",
        "pub trait T {\n        #[variant(of = sum)]\n        fn f(&self);\n    }",
        "#[variant(of = sum)]\n    extern crate core;",
        "#[variant(of = sum)]\n    pub trait Alias = Clone;",
        "extern \"C\" {\n        #[variant(of = sum)]\n        fn f();\n    }",
    ];
    for stray in strays {
        let source = module_text(WORKGROUP, stray);
        assert_code(
            &rejected(expand(&source, Profile::Other)),
            Code::VariantAttribute,
        );
        // And the module goes back with no `#[variant]` left for rustc to ask about.
        let item: TokenStream = source.parse().unwrap();
        let text = crate::gpu_impl(TokenStream::new(), item).to_string();
        assert_rendered_code(&text, Code::VariantAttribute);
        assert_eq!(text.matches("compile_error").count(), 1, "{stray}: {text}");
        let flat: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(!flat.contains("[variant("), "{stray}: {text}");
    }
}

#[test]
fn the_texts_of_the_attribute_errors_are_pinned() {
    // A function that is both.
    let both = module_text(
        WORKGROUP,
        &function_text(
            "#[kernel]\n    #[variant(of = sum, requires(subgroups))]",
            "both",
        ),
    );
    assert_eq!(
        rejected(expand(&both, Profile::Other)).to_string(),
        "[GUST-E0740] `both` is both a #[kernel] and a #[variant]: a function is the kernel or one of its variants"
    );
    // A second `#[variant]` attribute.
    let twice = module_text(
        WORKGROUP,
        &function_text(
            "#[variant(of = sum, requires(subgroups))]\n    #[variant(of = sum)]",
            "twice",
        ),
    );
    assert_eq!(
        rejected(expand(&twice, Profile::Other)).to_string(),
        "[GUST-E0740] a function has one #[variant] attribute"
    );
    // No kernel named.
    let bare = module_text(
        WORKGROUP,
        &function_text("#[variant(requires(subgroups))]", "v"),
    );
    assert_eq!(
        rejected(expand(&bare, Profile::Other)).to_string(),
        "[GUST-E0740] #[variant] needs the kernel it replaces: `#[variant(of = kernel, requires(subgroups))]`"
    );
}

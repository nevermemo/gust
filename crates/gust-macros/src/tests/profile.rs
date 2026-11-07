//! The stage profile (mx.p0, D63): the `grid` profile gives the text the emitter wrote before the profile existed, and
//! every hook of the interface is read by the call site that owns it. The second half runs a probe profile that
//! differs from `grid` at each hook and shows the emitter, the validator and the option parser follow it.

use crate::{
    diagnostics::testing::parse_str,
    expand::{kernel_attribute_in, kernel_options, parameter_bindings},
    profile::{
        BindingRules, BodyChecks, Builtin, ContractTier, ExtraBinding, GRID, KernelFacts,
        OptionRules, OwnIndexWrite, ReflectionKind, Shape, StageProfile, TwinHarness,
    },
    slang::{ParameterFlavor, emit_kernel, emit_kernel_in, parameter_flavor_in, type_name},
    validate::{validate_module, validate_module_in},
};

const VECTOR_ADD: &str = "mod vector_add {
    #[kernel(workgroup_size(32, 2, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        local: LocalId,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i] + local.x as f32;
        }
    }
}";

fn module(source: &str) -> syn::ItemMod {
    parse_str(source).unwrap()
}

fn kernel(module: &syn::ItemMod) -> &syn::ItemFn {
    module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if kernel_options(function).unwrap().is_some() => {
                Some(function)
            }
            _ => None,
        })
        .unwrap()
}

#[test]
fn the_grid_profile_answers_with_the_rules_the_emitter_held() {
    assert_eq!(GRID.shape(), Shape::Grid);
    assert_eq!(GRID.slang_stage(), "compute");
    assert_eq!(GRID.twin(), TwinHarness::Linear);
    assert_eq!(GRID.reflection(), ReflectionKind::Compute);
    assert_eq!(GRID.contract_tier(), ContractTier::Exact);
    for (name, expected) in [
        ("Invocation", Some(Builtin::Global)),
        ("SV_DispatchThreadID", Some(Builtin::Global)),
        ("UVec3", Some(Builtin::Global)),
        ("LocalId", Some(Builtin::Local)),
        ("SV_GroupThreadID", Some(Builtin::Local)),
        ("GroupId", Some(Builtin::Group)),
        ("SV_GroupID", Some(Builtin::Group)),
        ("StructuredBuffer<f32>", None),
        ("u32", None),
    ] {
        let ty: syn::Type = parse_str(name).unwrap();
        assert_eq!(GRID.invocation(&ty), expected, "{name}");
    }
    let rules = GRID.binding_rules();
    assert!(rules.gets_slot(ParameterFlavor::StorageRead));
    assert!(rules.gets_slot(ParameterFlavor::StorageReadWrite));
    assert!(rules.gets_slot(ParameterFlavor::Uniform));
    assert!(!rules.gets_slot(ParameterFlavor::Invocation));
    assert!(!rules.gets_slot(ParameterFlavor::Workgroup));
    let options = GRID.options();
    assert!(options.workgroup_size && options.cpu);
    assert_eq!(options.default_workgroup, [64, 1, 1]);
    let mut checks = BodyChecks {
        workgroup_memory: true,
        barrier: true,
        atomics: true,
        forbidden: Vec::new(),
    };
    GRID.validate_body(&mut checks);
    assert!(checks.forbidden.is_empty());
    let resources = vec!["a".to_owned(), "out".to_owned()];
    let facts = KernelFacts {
        entry_point: "gpu_m_k",
        workgroup_size: [8, 4, 1],
        invocation: "id",
        resources: &resources,
        derived_ids: &[],
        next_binding: 2,
    };
    assert_eq!(
        GRID.entry_signature(&facts),
        "[numthreads(8, 4, 1)]\nvoid gpu_m_k(uint3 id : SV_DispatchThreadID)"
    );
    assert_eq!(
        GRID.prologue(&facts),
        "    uint __gpu_len_a;\n    uint __gpu_stride_a;\n    a.GetDimensions(__gpu_len_a, __gpu_stride_a);\n\
         \x20   uint __gpu_len_out;\n    uint __gpu_stride_out;\n\
         \x20   out.GetDimensions(__gpu_len_out, __gpu_stride_out);\n"
    );
    assert_eq!(GRID.epilogue(&facts), "");
    assert!(GRID.extra_bindings(&facts).is_empty());
    assert_eq!(
        GRID.lower_output_write(&OwnIndexWrite {
            buffer: "out",
            value: "1.0"
        }),
        None
    );
}

#[test]
fn the_entry_wrappers_are_the_grid_profile() {
    let module = module(VECTOR_ADD);
    validate_module(&module).unwrap();
    validate_module_in(&GRID, &module).unwrap();
    let kernel = kernel(&module);
    let options = kernel_options(kernel).unwrap().unwrap();
    let wrapped = emit_kernel(&module, kernel, options).unwrap().source;
    let direct = emit_kernel_in(&GRID, &module, kernel, options)
        .unwrap()
        .source;
    assert_eq!(wrapped, direct);
    assert!(wrapped.contains(
        "[shader(\"compute\")]\n[numthreads(32, 2, 1)]\n\
         void gpu_vector_add_add(uint3 id : SV_DispatchThreadID)\n{\n"
    ));
    assert!(wrapped.contains("    uint3 local = uint3(id.x % 32u, id.y % 2u, id.z % 1u);\n"));
}

/// A profile that differs from `grid` at every hook the call sites read.
struct Probe;

impl StageProfile for Probe {
    fn shape(&self) -> Shape {
        Shape::Grid
    }

    fn slang_stage(&self) -> &'static str {
        "fragment"
    }

    fn invocation(&self, ty: &syn::Type) -> Option<Builtin> {
        match type_name(ty)?.as_str() {
            "Pixel" => Some(Builtin::Global),
            _ => None,
        }
    }

    fn binding_rules(&self) -> BindingRules {
        BindingRules::new(
            "the probe stage takes no value parameters",
            "the probe stage requires a `Pixel` parameter",
            "the probe stage requires exactly one `Pixel` parameter",
            &[
                ParameterFlavor::StorageRead,
                ParameterFlavor::StorageReadWrite,
            ],
        )
    }

    fn options(&self) -> OptionRules {
        OptionRules {
            workgroup_size: false,
            cpu: true,
            default_workgroup: [8, 8, 1],
            expected: "the probe stage takes the `cpu` option only",
        }
    }

    fn validate_body(&self, cx: &mut BodyChecks) {
        if cx.atomics {
            cx.forbidden
                .push("the probe stage has no atomics".to_owned());
        }
        if cx.barrier {
            cx.forbidden
                .push("the probe stage has no barriers".to_owned());
        }
    }

    fn entry_signature(&self, k: &KernelFacts<'_>) -> String {
        format!(
            "float4 {}(float4 {} : SV_Position) : SV_Target",
            k.entry_point, k.invocation
        )
    }

    fn prologue(&self, k: &KernelFacts<'_>) -> String {
        format!("    // prologue after binding {}\n", k.next_binding)
    }

    fn epilogue(&self, _k: &KernelFacts<'_>) -> String {
        "    return float4(0.0, 0.0, 0.0, 1.0);\n".to_owned()
    }

    fn lower_output_write(&self, _write: &OwnIndexWrite<'_>) -> Option<String> {
        None
    }

    fn extra_bindings(&self, _k: &KernelFacts<'_>) -> Vec<ExtraBinding> {
        vec![ExtraBinding {
            name: "__probe".to_owned(),
            slang_type: "uint".to_owned(),
        }]
    }

    fn twin(&self) -> TwinHarness {
        TwinHarness::Linear
    }

    fn reflection(&self) -> ReflectionKind {
        ReflectionKind::Compute
    }

    fn contract_tier(&self) -> ContractTier {
        ContractTier::Exact
    }
}

const PROBE_KERNEL: &str = "mod shade {
    #[kernel(cpu = \"unsupported\")]
    pub fn paint(px: Pixel, a: StructuredBuffer<float>, mut out: RWStructuredBuffer<float>) {
        out[px.x] = a[px.x];
    }
}";

#[test]
fn the_emitter_reads_the_stage_the_entry_signature_the_statements_and_the_bindings_from_the_profile()
 {
    let module = module(PROBE_KERNEL);
    validate_module_in(&Probe, &module).unwrap();
    let kernel = kernel_attribute_in(&Probe, kernel_ref(&module))
        .unwrap()
        .unwrap();
    // The default workgroup size is the profile's.
    assert_eq!(kernel.0.workgroup_size, [8, 8, 1]);
    let source = emit_kernel_in(&Probe, &module, kernel_ref(&module), kernel.0)
        .unwrap()
        .source;
    assert!(source.contains("[shader(\"fragment\")]\n"), "{source}");
    assert!(!source.contains("numthreads"), "{source}");
    assert!(!source.contains("SV_DispatchThreadID"), "{source}");
    assert!(
        source.contains(
            "float4 gpu_shade_paint(float4 px : SV_Position) : SV_Target\n{\n\
             \x20   // prologue after binding 3\n"
        ),
        "{source}"
    );
    assert!(
        source.contains("[[vk::binding(2, 0)]] ConstantBuffer<uint> __probe;\n"),
        "{source}"
    );
    assert!(
        source.ends_with("    return float4(0.0, 0.0, 0.0, 1.0);\n}\n"),
        "{source}"
    );
}

/// A profile that leaves a kind of resource out of `slotted` (the probe's uniform) declares none of it, and the slots
/// of the others are the descriptor's: both come from `gets_slot`.
#[test]
fn the_emitter_numbers_the_slots_the_descriptor_numbers() {
    let module = module(
        "mod shade {
            #[repr(C, align(16))]
            pub struct P { pub a: f32, pub b: f32, pub c: f32, pub d: f32 }
            #[kernel(cpu = \"unsupported\")]
            pub fn paint(
                px: Pixel,
                a: StructuredBuffer<float>,
                u: Uniform<P>,
                mut out: RWStructuredBuffer<float>,
            ) {
                out[px.x] = a[px.x];
            }
        }",
    );
    let kernel = kernel_ref(&module);
    let options = kernel_attribute_in(&Probe, kernel).unwrap().unwrap().0;
    let source = emit_kernel_in(&Probe, &module, kernel, options)
        .unwrap()
        .source;
    assert!(!source.contains("ConstantBuffer<P>"), "{source}");
    assert!(
        source.contains("[[vk::binding(0, 0)]] StructuredBuffer<float> a;\n"),
        "{source}"
    );
    assert!(
        source.contains("[[vk::binding(1, 0)]] RWStructuredBuffer<float> out;\n"),
        "{source}"
    );
    // The profile's own extra binding takes the next slot, after the two resources.
    assert!(
        source.contains("[[vk::binding(2, 0)]] ConstantBuffer<uint> __probe;\n"),
        "{source}"
    );
    let bindings = parameter_bindings(&Probe, &kernel.sig.inputs).unwrap();
    assert_eq!(
        bindings,
        [
            ("px".to_owned(), None),
            ("a".to_owned(), Some(0)),
            ("u".to_owned(), None),
            ("out".to_owned(), Some(1)),
        ]
    );
    // `grid` slots the uniform too: the same module numbers it 1 and the output 2.
    let grid = parameter_bindings(&GRID, &kernel.sig.inputs).unwrap();
    assert!(grid.contains(&("u".to_owned(), Some(1))), "{grid:?}");
}

fn kernel_ref(module: &syn::ItemMod) -> &syn::ItemFn {
    module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function)
                if function.attrs.iter().any(|a| a.path().is_ident("kernel")) =>
            {
                Some(function)
            }
            _ => None,
        })
        .unwrap()
}

#[test]
fn the_validator_asks_the_profile_for_the_invocation_and_the_forbidden_constructs() {
    // `Pixel` is the invocation of the probe and nothing to `grid`, which finds no dispatch-thread parameter.
    let module = module(PROBE_KERNEL);
    let error = validate_module(&module).unwrap_err().to_string();
    assert!(
        error.contains("exactly one dispatch-thread parameter"),
        "{error}"
    );
    assert_eq!(
        parameter_flavor_in(&Probe, &parse_str("Pixel").unwrap()),
        ParameterFlavor::Invocation
    );
    assert_eq!(
        parameter_flavor_in(&Probe, &parse_str("SV_DispatchThreadID").unwrap()),
        ParameterFlavor::Value
    );
    // A barrier is refused by the probe and accepted by `grid`.
    let barrier = self::module(
        "mod shade {
            #[kernel(cpu = \"unsupported\")]
            pub fn paint(px: Pixel, mut out: RWStructuredBuffer<float>) {
                out[px.x] = 1.0;
                workgroup_barrier();
            }
        }",
    );
    let error = validate_module_in(&Probe, &barrier)
        .unwrap_err()
        .to_string();
    assert!(error.contains("the probe stage has no barriers"), "{error}");
    let grid_barrier = self::module(
        "mod shade {
            #[kernel(cpu = \"unsupported\")]
            pub fn paint(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
                out[id.x] = 1.0;
                workgroup_barrier();
            }
        }",
    );
    validate_module_in(&GRID, &grid_barrier).unwrap();
}

#[test]
fn the_option_parser_and_the_value_refusal_follow_the_profile() {
    let sized = self::module(
        "mod shade {
            #[kernel(workgroup_size(8, 8, 1))]
            pub fn paint(px: Pixel, mut out: RWStructuredBuffer<float>) {
                out[px.x] = 1.0;
            }
        }",
    );
    let error = kernel_attribute_in(&Probe, kernel_ref(&sized))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("the probe stage takes the `cpu` option only"),
        "{error}"
    );
    assert!(
        kernel_attribute_in(&GRID, kernel_ref(&sized))
            .unwrap()
            .is_some()
    );
    let valued = self::module(
        "mod shade {
            #[kernel(cpu = \"unsupported\")]
            pub fn paint(px: Pixel, scale: f32, mut out: RWStructuredBuffer<float>) {
                out[px.x] = scale;
            }
        }",
    );
    let options = kernel_attribute_in(&Probe, kernel_ref(&valued))
        .unwrap()
        .unwrap()
        .0;
    let error = emit_kernel_in(&Probe, &valued, kernel_ref(&valued), options)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("the probe stage takes no value parameters"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// The `pixel` profile (slice mx.p1a, D61, D63, D120).
// ---------------------------------------------------------------------------------------------------------------

mod pixel {
    use super::*;
    use crate::{
        expand::{KernelOptions, Profile, expand_with_directories},
        profile::{DerivedId, PIXEL},
        regression_tests::assert_golden,
    };

    const FIXTURE: &str = include_str!("../../../../tests/fixtures/pixel.rs");
    const GOLDEN: &str = include_str!("../../../../tests/fixtures/pixel.slang");

    /// The fixture module as `#[gpu]` receives it (the attribute is the macro's own).
    fn fixture_module() -> String {
        FIXTURE.replacen("#[gust::gpu]", "", 1)
    }

    fn facts<'a>(resources: &'a [String], derived: &'a [DerivedId]) -> KernelFacts<'a> {
        KernelFacts {
            entry_point: "gpu_m_k",
            workgroup_size: [1, 1, 1],
            invocation: "id",
            resources,
            derived_ids: derived,
            next_binding: 2,
        }
    }

    #[test]
    fn the_pixel_profile_answers_with_the_fragment_rules() {
        assert_eq!(PIXEL.shape(), Shape::Pixel);
        assert_eq!(PIXEL.slang_stage(), "fragment");
        assert_eq!(PIXEL.twin(), TwinHarness::Linear);
        assert_eq!(PIXEL.reflection(), ReflectionKind::Fragment);
        assert_eq!(PIXEL.contract_tier(), ContractTier::Exact);
        // A fragment has the global id only.
        for (name, expected) in [
            ("Invocation", Some(Builtin::Global)),
            ("SV_DispatchThreadID", Some(Builtin::Global)),
            ("UVec3", Some(Builtin::Global)),
            ("LocalId", None),
            ("SV_GroupThreadID", None),
            ("GroupId", None),
            ("SV_GroupID", None),
            ("StructuredBuffer<f32>", None),
        ] {
            let ty: syn::Type = parse_str(name).unwrap();
            assert_eq!(PIXEL.invocation(&ty), expected, "{name}");
        }
        let rules = PIXEL.binding_rules();
        assert!(!rules.extent_calls, "a pixel call is one-dimensional");
        assert!(GRID.binding_rules().extent_calls);
        for flavor in [
            ParameterFlavor::StorageRead,
            ParameterFlavor::StorageReadWrite,
            ParameterFlavor::Uniform,
            ParameterFlavor::SampledTexture,
            ParameterFlavor::Sampler,
        ] {
            assert!(rules.gets_slot(flavor), "{flavor:?}");
        }
        assert!(!rules.gets_slot(ParameterFlavor::Invocation));
        assert!(!rules.gets_slot(ParameterFlavor::Workgroup));
        let options = PIXEL.options();
        assert!(!options.workgroup_size && options.cpu);
        assert_eq!(options.default_workgroup, [1, 1, 1]);
        assert!(options.expected.contains("shape = pixel"));
    }

    /// `validate_body` is judged on facts, so a double with the facts set shows what the profile refuses. Through
    /// `#[gpu]` the classifier refuses the same constructs on a `pixel` pin first (GUST-E0723).
    #[test]
    fn the_pixel_profile_refuses_workgroup_memory_barriers_and_atomics() {
        let mut clean = BodyChecks::default();
        PIXEL.validate_body(&mut clean);
        assert!(clean.forbidden.is_empty());
        for (mut checks, word) in [
            (
                BodyChecks {
                    workgroup_memory: true,
                    ..BodyChecks::default()
                },
                "workgroup memory",
            ),
            (
                BodyChecks {
                    barrier: true,
                    ..BodyChecks::default()
                },
                "barrier",
            ),
            (
                BodyChecks {
                    atomics: true,
                    ..BodyChecks::default()
                },
                "atomic",
            ),
        ] {
            PIXEL.validate_body(&mut checks);
            assert_eq!(checks.forbidden.len(), 1, "{word}");
            assert!(checks.forbidden[0].contains(word), "{:?}", checks.forbidden);
            assert!(checks.forbidden[0].starts_with("a `pixel` kernel"));
        }
    }

    #[test]
    fn the_pixel_profile_writes_the_fragment_entry_the_linear_index_and_no_epilogue() {
        let resources = vec!["a".to_owned(), "out".to_owned()];
        let facts = facts(&resources, &[]);
        assert_eq!(
            PIXEL.entry_signature(&facts),
            "void gpu_m_k(float4 __gust_pos : SV_Position)"
        );
        assert_eq!(
            PIXEL.prologue(&facts),
            "    uint __gust_linear = ((uint(__gust_pos.y) * 8192u) + uint(__gust_pos.x));\n\
             \x20   uint3 id = uint3(__gust_linear, 0u, 0u);\n\
             \x20   uint __gpu_len_a;\n    uint __gpu_stride_a;\n    a.GetDimensions(__gpu_len_a, __gpu_stride_a);\n\
             \x20   uint __gpu_len_out;\n    uint __gpu_stride_out;\n\
             \x20   out.GetDimensions(__gpu_len_out, __gpu_stride_out);\n"
        );
        assert_eq!(PIXEL.epilogue(&facts), "");
        // No uniform for the row width: nothing is added to the kernel's own bindings (option S, D127).
        assert!(PIXEL.extra_bindings(&facts).is_empty());
        // The write at the own index is the plain index assignment: no user expression is rewritten.
        assert_eq!(
            PIXEL.lower_output_write(&OwnIndexWrite {
                buffer: "out",
                value: "1.0"
            }),
            None
        );
    }

    #[test]
    fn the_pixel_prologue_is_the_fragment_index_and_then_the_lines_the_grid_prologue_reads_lengths_with()
     {
        // The length reads are shared text: the same lines, in the same order, after the two index lines.
        let resources = vec!["a".to_owned(), "b".to_owned(), "out".to_owned()];
        let facts = facts(&resources, &[]);
        let grid = GRID.prologue(&facts);
        let pixel = PIXEL.prologue(&facts);
        assert!(pixel.ends_with(&grid), "{pixel}\n{grid}");
        assert_eq!(pixel.lines().count(), grid.lines().count() + 2);
    }

    /// The emitted Slang of every kernel of the fixture, in the layout of the other goldens.
    fn emitted_fixture() -> String {
        let module: syn::ItemMod = parse_str(fixture_module().trim()).unwrap();
        validate_module(&module).unwrap();
        let mut actual = String::new();
        for item in &module.content.as_ref().unwrap().1 {
            let syn::Item::Fn(function) = item else {
                continue;
            };
            if kernel_options(function).unwrap().is_none() {
                continue;
            }
            let options = KernelOptions {
                workgroup_size: PIXEL.options().default_workgroup,
            };
            actual.push_str(&format!("// ==== kernel {} ====\n", function.sig.ident));
            actual.push_str(
                &emit_kernel_in(&PIXEL, &module, function, options)
                    .unwrap()
                    .source
                    .replace("\r\n", "\n"),
            );
        }
        actual
    }

    #[test]
    fn kernels_of_the_pixel_fixture_match_the_golden() {
        assert_golden(
            &emitted_fixture(),
            "pixel.slang",
            &GOLDEN.replace("\r\n", "\n"),
        );
    }

    #[test]
    fn the_pixel_golden_is_a_fragment_with_storage_outputs_and_no_render_target() {
        let golden = GOLDEN.replace("\r\n", "\n");
        assert_eq!(golden.matches("[shader(\"fragment\")]").count(), 3);
        assert!(!golden.contains("[shader(\"compute\")]"));
        assert!(!golden.contains("numthreads"));
        assert!(
            !golden.contains("SV_Target"),
            "an output is not a render target (D120)"
        );
        assert!(!golden.contains("SV_DispatchThreadID"));
        assert_eq!(
            golden
                .matches("void gpu_pixel_shade(float4 __gust_pos : SV_Position)")
                .count(),
            1
        );
        // The outputs are storage buffers with the slots `grid` gives them, in parameter order.
        for text in [
            "[[vk::binding(0, 0)]] StructuredBuffer<float> input;",
            "[[vk::binding(1, 0)]] RWStructuredBuffer<float> scaled;",
            "[[vk::binding(2, 0)]] RWStructuredBuffer<uint> flags;",
            "[[vk::binding(0, 0)]] Texture2D<float4> image;",
            "[[vk::binding(1, 0)]] SamplerState bilinear;",
            "[[vk::binding(2, 0)]] RWStructuredBuffer<float4> colours;",
        ] {
            assert!(golden.contains(text), "{text} in {golden}");
        }
        // The row width is a constant of the text: no uniform, and no guard against a padding pixel.
        assert_eq!(golden.matches("* 8192u) + uint(__gust_pos.x)").count(), 3);
        assert!(!golden.contains("ConstantBuffer"), "{golden}");
        assert!(!golden.contains("__gust_pixel"), "{golden}");
        assert!(!golden.contains("return"), "{golden}");
        // The user's body is unchanged: the own-index writes stay plain assignments.
        assert!(golden.contains("scaled[i] = "), "{golden}");
        assert!(golden.contains("steps[id.x] = "), "{golden}");
    }

    #[test]
    fn the_expansion_of_the_pixel_fixture_describes_pixel_kernels_with_one_dimensional_calls() {
        let item: proc_macro2::TokenStream = fixture_module().parse().unwrap();
        let text: String =
            crate::gpu_impl_with(proc_macro2::TokenStream::new(), item, Profile::Other)
                .to_string()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
        assert_eq!(
            text.matches("shape:::gust::Shape::Pixel,").count(),
            3,
            "{text}"
        );
        assert!(!text.contains("shape:::gust::Shape::Grid,"), "{text}");
        assert_eq!(
            text.matches("workgroup_size:[1u32,1u32,1u32],").count(),
            3,
            "{text}"
        );
        for form in ["call_extent", "submit_extent", "dispatch_extent"] {
            assert!(!text.contains(form), "{form} in {text}");
        }
        for form in ["pubfncall<", "pubfnsubmit<", "pubfndispatch<"] {
            assert!(text.contains(form), "{form} in {text}");
        }
        // An output is a storage parameter with a binding and read-write access, as under `grid`.
        assert!(text.contains("::gust::ParameterKind::Storage"), "{text}");
        assert!(text.contains("::gust::Access::ReadWrite"), "{text}");
        // Every kernel is built for the GPU target until routing exists (m8.r1), whatever its shape.
        assert_eq!(
            text.matches("target:::gust::target::Target::Gpu,").count(),
            3,
            "{text}"
        );
        assert!(!text.contains("Target::Cpu"), "{text}");
    }

    /// A module that holds a `grid` kernel with an explicit workgroup size beside a `pixel` kernel (slice mx.p1b).
    const MIXED: &str = "mod mixed {
    fn twice(x: f32) -> f32 {
        x * 2.0f32
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn grid_kernel(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = twice(input[i]);
        }
    }

    #[kernel(shape = pixel)]
    pub fn pixel_kernel(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        out[i] = twice(input[i]);
    }
}";

    fn function<'a>(module: &'a syn::ItemMod, name: &str) -> &'a syn::ItemFn {
        module
            .content
            .as_ref()
            .unwrap()
            .1
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(function) if function.sig.ident == name => Some(function),
                _ => None,
            })
            .unwrap()
    }

    /// Before the fix the pixel emitter parsed every sibling's options under the pixel profile, only to ask whether it
    /// was a kernel, and a sibling `workgroup_size(..)` (which the pixel profile refuses) failed the pixel kernel with
    /// E0707. A sibling is recognised by its attribute now; its options are the business of its own profile.
    #[test]
    fn a_module_may_hold_an_explicit_workgroup_grid_kernel_beside_a_pixel_kernel() {
        let module: syn::ItemMod = parse_str(MIXED).unwrap();
        validate_module(&module).unwrap();
        let pixel = emit_kernel_in(
            &PIXEL,
            &module,
            function(&module, "pixel_kernel"),
            KernelOptions {
                workgroup_size: PIXEL.options().default_workgroup,
            },
        )
        .unwrap()
        .source;
        assert!(pixel.contains("[shader(\"fragment\")]"), "{pixel}");
        assert!(pixel.contains("float twice(float x)"), "{pixel}");
        assert!(pixel.contains("void gpu_mixed_pixel_kernel("), "{pixel}");
        assert!(!pixel.contains("grid_kernel"), "{pixel}");
        assert!(!pixel.contains("numthreads"), "{pixel}");
        let grid = emit_kernel_in(
            &GRID,
            &module,
            function(&module, "grid_kernel"),
            KernelOptions {
                workgroup_size: [64, 1, 1],
            },
        )
        .unwrap()
        .source;
        assert!(grid.contains("[numthreads(64, 1, 1)]"), "{grid}");
        assert!(grid.contains("[shader(\"compute\")]"), "{grid}");
        assert!(grid.contains("float twice(float x)"), "{grid}");
        assert!(grid.contains("void gpu_mixed_grid_kernel("), "{grid}");
        assert!(!grid.contains("pixel_kernel"), "{grid}");
        assert!(!grid.contains("fragment"), "{grid}");
    }

    /// The same module through the whole macro: both kernels are described, one of each shape, with their own sizes.
    #[test]
    fn the_expansion_of_a_mixed_module_describes_both_kernels() {
        let item: proc_macro2::TokenStream = MIXED.parse().unwrap();
        let text: String =
            crate::gpu_impl_with(proc_macro2::TokenStream::new(), item, Profile::Other)
                .to_string()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
        assert!(!text.contains("compile_error"), "{text}");
        assert_eq!(
            text.matches("shape:::gust::Shape::Pixel,").count(),
            1,
            "{text}"
        );
        assert_eq!(
            text.matches("shape:::gust::Shape::Grid,").count(),
            1,
            "{text}"
        );
        assert_eq!(
            text.matches("workgroup_size:[64u32,1u32,1u32],").count(),
            1,
            "{text}"
        );
        assert_eq!(
            text.matches("workgroup_size:[1u32,1u32,1u32],").count(),
            1,
            "{text}"
        );
    }

    #[test]
    fn a_pixel_kernels_export_record_is_schema_3_with_its_shape() {
        let root = std::env::temp_dir().join(format!("gust-macros-pixel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let module: syn::ItemMod = parse_str(fixture_module().trim()).unwrap();
        expand_with_directories(module, None, Some(&root), Some("demo")).unwrap();
        let mut records = 0;
        for entry in std::fs::read_dir(root.join("demo")).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            if !name.ends_with(".json") {
                continue;
            }
            let text = std::fs::read_to_string(root.join("demo").join(&name)).unwrap();
            assert!(text.contains("\"schema\": 3"), "{name}: {text}");
            assert!(text.contains("\"shape\": \"pixel\""), "{name}: {text}");
            assert!(text.contains("\"workgroup_size\""), "{name}: {text}");
            records += 1;
        }
        assert_eq!(records, 3);
        let _ = std::fs::remove_dir_all(&root);
    }
}

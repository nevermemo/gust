//! The stage profile (mx.p0, D63): the `grid` profile gives the text the emitter wrote before the profile existed, and
//! every hook of the interface is read by the call site that owns it. The second half runs a probe profile that
//! differs from `grid` at each hook and shows the emitter, the validator and the option parser follow it.

use crate::{
    diagnostics::testing::parse_str,
    expand::{kernel_attribute_in, kernel_options},
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

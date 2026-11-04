//! The `grid` profile: one compute thread per element of the dispatch grid. Its rules are the ones the emitter, the
//! validator and the twin harness held before the profile existed; every text below is the text they wrote.

use std::fmt::Write as _;

use super::{
    BindingRules, Builtin, ContractTier, ExtraBinding, KernelFacts, OptionRules, OwnIndexWrite,
    ReflectionKind, Shape, StageProfile, TwinHarness,
};
use crate::slang::ParameterFlavor;

/// The profile of `grid` kernels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GridProfile;

/// The one `grid` profile; a kernel without a shape pin has it.
pub(crate) static GRID: GridProfile = GridProfile;

impl StageProfile for GridProfile {
    fn shape(&self) -> Shape {
        Shape::Grid
    }

    fn slang_stage(&self) -> &'static str {
        "compute"
    }

    fn invocation(&self, ty: &syn::Type) -> Option<Builtin> {
        match crate::slang::type_name(ty)?.as_str() {
            "Invocation" | "SV_DispatchThreadID" | "UVec3" => Some(Builtin::Global),
            "LocalId" | "SV_GroupThreadID" => Some(Builtin::Local),
            "GroupId" | "SV_GroupID" => Some(Builtin::Group),
            _ => None,
        }
    }

    fn binding_rules(&self) -> BindingRules {
        BindingRules::new(
            "Slang compute entry points currently accept resources and SV_DispatchThreadID only",
            "a Slang kernel requires an `SV_DispatchThreadID` (or legacy `Invocation`) parameter",
            "a GPU kernel requires exactly one dispatch-thread parameter (`LocalId` and `GroupId` are derived from it, \
             so a kernel that takes either needs it)",
            &[
                ParameterFlavor::StorageRead,
                ParameterFlavor::StorageReadWrite,
                ParameterFlavor::Uniform,
                ParameterFlavor::SampledTexture,
                ParameterFlavor::StorageTexture,
                ParameterFlavor::WriteTexture,
                ParameterFlavor::Sampler,
            ],
        )
    }

    fn options(&self) -> OptionRules {
        OptionRules {
            workgroup_size: true,
            cpu: true,
            default_workgroup: [64, 1, 1],
            expected: "expected `workgroup_size(x, y, z)`, `cpu = \"unsupported\"` or `shape = grid`",
        }
    }

    fn validate_body(&self, _cx: &mut super::BodyChecks) {
        // A `grid` kernel may declare workgroup memory, call barriers and use atomics: nothing is refused.
    }

    fn entry_signature(&self, k: &KernelFacts<'_>) -> String {
        let [x, y, z] = k.workgroup_size;
        format!(
            "[numthreads({x}, {y}, {z})]\nvoid {}(uint3 {} : SV_DispatchThreadID)",
            k.entry_point, k.invocation
        )
    }

    fn prologue(&self, k: &KernelFacts<'_>) -> String {
        let mut text = String::new();
        for resource in k.resources {
            writeln!(text, "    uint __gpu_len_{resource};").unwrap();
            writeln!(text, "    uint __gpu_stride_{resource};").unwrap();
            writeln!(
                text,
                "    {resource}.GetDimensions(__gpu_len_{resource}, __gpu_stride_{resource});"
            )
            .unwrap();
        }
        // `LocalId` and `GroupId` come from the dispatch-thread id and the workgroup size: a dispatch-thread id is
        // `group * size + local` in every dimension, so `id % size` and `id / size` are the local and the group id.
        // The entry point keeps its one semantic parameter, so the reflection and the descriptor stay as they are.
        let [x, y, z] = k.workgroup_size;
        let invocation = k.invocation;
        for id in k.derived_ids {
            let name = &id.name;
            let operator = if id.group { '/' } else { '%' };
            writeln!(
                text,
                "    uint3 {name} = uint3({invocation}.x {operator} {x}u, {invocation}.y {operator} {y}u, \
                 {invocation}.z {operator} {z}u);"
            )
            .unwrap();
        }
        text
    }

    fn epilogue(&self, _k: &KernelFacts<'_>) -> String {
        String::new()
    }

    fn lower_output_write(&self, _write: &OwnIndexWrite<'_>) -> Option<String> {
        None
    }

    fn extra_bindings(&self, _k: &KernelFacts<'_>) -> Vec<ExtraBinding> {
        Vec::new()
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

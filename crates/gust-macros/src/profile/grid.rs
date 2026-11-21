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
        let mut text = super::length_reads(k.resources);
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

/// The name of the hidden uniform of a split variant, and of its entry point's dispatch-thread parameter. Both start
/// with `__gust_`, which E0901 denies to users, so neither can clash with a name of the kernel.
pub(crate) const LAUNCH_BASE: &str = "__gust_launch_base";
pub(crate) const LAUNCH_ID: &str = "__gust_launch_id";

/// The profile of the split variant of a `grid` kernel (m11.g4a): the kernel's own text with one hidden uniform at the
/// next free binding, whose `x`, `y` and `z` are added to the dispatch-thread id before anything reads it. The host
/// launches a dispatch past the per-dimension workgroup limit as several smaller ones, each with the id of its first
/// invocation in that uniform. Everything but the entry signature, the prologue and the extra binding is `GRID`'s.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GridSplitProfile;

/// The one split profile.
pub(crate) static GRID_SPLIT: GridSplitProfile = GridSplitProfile;

impl StageProfile for GridSplitProfile {
    fn shape(&self) -> Shape {
        GRID.shape()
    }

    fn slang_stage(&self) -> &'static str {
        GRID.slang_stage()
    }

    fn invocation(&self, ty: &syn::Type) -> Option<Builtin> {
        GRID.invocation(ty)
    }

    fn binding_rules(&self) -> BindingRules {
        GRID.binding_rules()
    }

    fn options(&self) -> OptionRules {
        GRID.options()
    }

    fn validate_body(&self, cx: &mut super::BodyChecks) {
        GRID.validate_body(cx);
    }

    fn entry_signature(&self, k: &KernelFacts<'_>) -> String {
        let [x, y, z] = k.workgroup_size;
        format!(
            "[numthreads({x}, {y}, {z})]\nvoid {}(uint3 {LAUNCH_ID} : SV_DispatchThreadID)",
            k.entry_point
        )
    }

    fn prologue(&self, k: &KernelFacts<'_>) -> String {
        // The base is a whole number of workgroups on every axis, so `id % size` and `id / size` of the sum are the
        // local and group ids of an unsplit launch. The line comes first: the length reads and the derived ids, and
        // the workgroup zeroing, read the id.
        format!(
            "    uint3 {} = {LAUNCH_ID} + uint3({LAUNCH_BASE}.x, {LAUNCH_BASE}.y, {LAUNCH_BASE}.z);\n{}",
            k.invocation,
            GRID.prologue(k)
        )
    }

    fn epilogue(&self, k: &KernelFacts<'_>) -> String {
        GRID.epilogue(k)
    }

    fn lower_output_write(&self, write: &OwnIndexWrite<'_>) -> Option<String> {
        GRID.lower_output_write(write)
    }

    fn extra_bindings(&self, _k: &KernelFacts<'_>) -> Vec<ExtraBinding> {
        vec![ExtraBinding {
            declaration: "struct __GustLaunchBase { uint x; uint y; uint z; uint pad; };\n"
                .to_owned(),
            name: LAUNCH_BASE.to_owned(),
            slang_type: "__GustLaunchBase".to_owned(),
        }]
    }

    fn twin(&self) -> TwinHarness {
        GRID.twin()
    }

    fn reflection(&self) -> ReflectionKind {
        GRID.reflection()
    }

    fn contract_tier(&self) -> ContractTier {
        GRID.contract_tier()
    }
}

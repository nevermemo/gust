//! The `pixel` profile (slice mx.p1a; decisions D61, D63, D120): one fragment per element of a full-screen pass.
//!
//! Under D120 a `pixel` output is not a render target. It stays a storage buffer (`RWStructuredBuffer`) bound like a
//! `grid` output, and the fragment writes its own element of it; the entry point returns nothing. The host runs the
//! pass with one never-written 8-bit attachment whose write mask is 0 (mx.p1b), so no colour value is stored anywhere.
//!
//! The element index is the fragment's position in a grid of constant row width, [`ROW_WIDTH`]:
//! `__gust_linear = uint(y) * 8192u + uint(x)`. The host scissors each draw to the live pixels (the full rows, then
//! the tail row), so no fragment runs past the last element: there is no padding pixel, no guard and no row-width
//! uniform. The constant is part of the shader's text, and so of the artifact's key; a call is therefore at most
//! `2^26` elements on every device (mx.p1b refuses a larger one by name). The invocation parameter of the Rust kernel
//! is a local `uint3(__gust_linear, 0, 0)`, so `id.x` is the element index and `id.y` and `id.z` are 0.

use super::{
    BindingRules, BodyChecks, Builtin, ContractTier, ExtraBinding, KernelFacts, OptionRules,
    OwnIndexWrite, ReflectionKind, Shape, StageProfile, TwinHarness,
};
use crate::slang::ParameterFlavor;

/// The width of a row of fragments, in elements: WebGPU's baseline `maxTextureDimension2D`, which every device has.
/// The attachment of a call is this wide and `ceil(n / ROW_WIDTH)` rows high.
pub(crate) const ROW_WIDTH: u32 = 8192;

/// The profile of `pixel` kernels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PixelProfile;

/// The one `pixel` profile; a kernel pinned `shape = pixel` has it.
pub(crate) static PIXEL: PixelProfile = PixelProfile;

impl StageProfile for PixelProfile {
    fn shape(&self) -> Shape {
        Shape::Pixel
    }

    fn slang_stage(&self) -> &'static str {
        "fragment"
    }

    /// A fragment has the global id only: `LocalId` and `GroupId` name nothing in it.
    fn invocation(&self, ty: &syn::Type) -> Option<Builtin> {
        match crate::slang::type_name(ty)?.as_str() {
            "Invocation" | "SV_DispatchThreadID" | "UVec3" => Some(Builtin::Global),
            _ => None,
        }
    }

    fn binding_rules(&self) -> BindingRules {
        BindingRules::new(
            "Slang fragment entry points currently accept resources and SV_DispatchThreadID only",
            "a Slang kernel requires an `SV_DispatchThreadID` (or legacy `Invocation`) parameter",
            "a `pixel` kernel requires exactly one dispatch-thread parameter (`LocalId` and `GroupId` do not exist in \
             a fragment shader)",
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
        .one_dimensional()
    }

    fn options(&self) -> OptionRules {
        OptionRules {
            workgroup_size: false,
            cpu: true,
            default_workgroup: [1, 1, 1],
            expected: "expected `cpu = \"unsupported\"` or `shape = pixel`",
        }
    }

    fn validate_body(&self, cx: &mut BodyChecks) {
        if cx.workgroup_memory {
            cx.forbidden.push(
                "a `pixel` kernel cannot declare workgroup memory: a fragment shader has none"
                    .to_owned(),
            );
        }
        if cx.barrier {
            cx.forbidden.push(
                "a `pixel` kernel cannot call a workgroup barrier: a fragment shader has none"
                    .to_owned(),
            );
        }
        if cx.atomics {
            cx.forbidden
                .push("a `pixel` kernel cannot use atomic operations".to_owned());
        }
    }

    fn entry_signature(&self, k: &KernelFacts<'_>) -> String {
        format!("void {}(float4 __gust_pos : SV_Position)", k.entry_point)
    }

    fn prologue(&self, k: &KernelFacts<'_>) -> String {
        let mut text = format!(
            "    uint __gust_linear = ((uint(__gust_pos.y) * {ROW_WIDTH}u) + uint(__gust_pos.x));\n    uint3 {} = \
             uint3(__gust_linear, 0u, 0u);\n",
            k.invocation
        );
        text.push_str(&super::length_reads(k.resources));
        text
    }

    fn epilogue(&self, _k: &KernelFacts<'_>) -> String {
        String::new()
    }

    /// An output is written in place at the kernel's own index, so the plain index assignment stands.
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
        ReflectionKind::Fragment
    }

    fn contract_tier(&self) -> ContractTier {
        ContractTier::Exact
    }
}

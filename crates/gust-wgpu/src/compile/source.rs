use crate::Error;
use crate::cache;
#[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
use crate::passthrough;
use gust::KernelDescriptor;

/// The feature the SPIR-V route needs (m9.rt2): `PASSTHROUGH_SHADERS`, when the feature is on, the adapter is Vulkan
/// and has it. An adapter without it still opens; its kernels fall back to WGSL and `HeadlessDevice::route_events`
/// says so.
pub(crate) fn spirv_route_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    #[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
    {
        // Vulkan only: DX12, Metal and GL advertise the feature but take no SPIR-V (review M1).
        if passthrough::backend_reason(adapter.get_info().backend).is_some() {
            return wgpu::Features::empty();
        }
        adapter.features() & wgpu::Features::PASSTHROUGH_SHADERS
    }
    #[cfg(not(all(feature = "spirv-route", not(target_arch = "wasm32"))))]
    {
        let _ = adapter;
        wgpu::Features::empty()
    }
}

/// Where the WGSL of a kernel comes from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ShaderSource {
    /// The kernel's embedded WGSL artifact, to be verified before use.
    Precompiled(&'static gust::Artifact),
    /// The run-time compiler (the `slang-runtime` feature).
    RunTime,
}

/// The shader source policy (D45): a kernel with a WGSL artifact uses it, whether or not the run-time compiler is
/// available; a kernel without one is compiled at run time only when `run_time_compiler` says this build holds the
/// compiler, and is otherwise `Error::NoArtifact`. `run_time_compiler` is `cfg!(feature = "slang-runtime")` in
/// production and an argument so that a test can reach both builds from the one it runs in.
pub(crate) fn select_shader_source(
    kernel: &KernelDescriptor,
    run_time_compiler: bool,
) -> Result<ShaderSource, Error> {
    if let Some(artifact) = cache::preferred_artifact(kernel) {
        return Ok(ShaderSource::Precompiled(artifact));
    }
    if run_time_compiler {
        return Ok(ShaderSource::RunTime);
    }
    Err(no_artifact(kernel))
}

/// Whether a split variant has a shader source in this build: a precompiled WGSL artifact of its own (D209), or the
/// run-time compiler (D208). The same policy as any kernel's (D45): an artifact is used whether or not the compiler is
/// there, and a variant with neither is `Error::NoArtifact`, which the planner reads as "this build cannot run the
/// pieces" and answers with the launch's own refusal. `run_time_compiler` is `cfg!(feature = "slang-runtime")` in
/// production and an argument so that a test can reach both builds from the one it runs in.
pub(crate) fn split_variant_has_a_shader_source(
    variant: &KernelDescriptor,
    run_time_compiler: bool,
) -> bool {
    select_shader_source(variant, run_time_compiler).is_ok()
}

/// The error of a kernel that has no WGSL artifact in a build without the run-time compiler.
pub(crate) fn no_artifact(kernel: &KernelDescriptor) -> Error {
    Error::NoArtifact {
        kernel: kernel.qualified_name().to_string(),
        profile: gust::profile::NAME,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use gust::{
        Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
        ResourceLayout,
    };

    const WGSL_ARTIFACT: gust::Artifact =
        gust::Artifact::new(gust::Format::Wgsl, "0123456789abcdef", b"", "");
    const SPIRV_ARTIFACT: gust::Artifact =
        gust::Artifact::new(gust::Format::Spirv, "0123456789abcdef", b"", "");

    #[test]
    fn a_kernel_with_a_wgsl_artifact_uses_it_with_or_without_the_run_time_compiler() {
        let kernel = DESCRIPTOR.with_artifacts(&[WGSL_ARTIFACT]);
        for run_time_compiler in [false, true] {
            let Ok(ShaderSource::Precompiled(artifact)) =
                select_shader_source(&kernel, run_time_compiler)
            else {
                panic!("a WGSL artifact is always preferred");
            };
            assert_eq!(artifact, &WGSL_ARTIFACT);
        }
    }

    #[test]
    fn a_kernel_without_an_artifact_compiles_at_run_time_only_where_the_compiler_exists() {
        assert!(DESCRIPTOR.artifacts.is_empty());
        assert!(matches!(
            select_shader_source(&DESCRIPTOR, true),
            Ok(ShaderSource::RunTime)
        ));
    }

    #[test]
    fn a_split_variant_is_planned_when_it_has_an_artifact_or_the_build_has_the_run_time_compiler() {
        // m11.psplit-a (D209): the planner's own decision, for the four cases. An artifact is a shader source whether or
        // not the compiler is there; without one, only the compiler is; with neither the launch keeps its refusal.
        assert!(DESCRIPTOR.artifacts.is_empty());
        assert!(split_variant_has_a_shader_source(&DESCRIPTOR, true));
        assert!(!split_variant_has_a_shader_source(&DESCRIPTOR, false));
        let precompiled = DESCRIPTOR.with_artifacts(&[WGSL_ARTIFACT]);
        for run_time_compiler in [false, true] {
            assert!(split_variant_has_a_shader_source(
                &precompiled,
                run_time_compiler
            ));
        }
        // An artifact of another format is no shader source of this host's: the same as none.
        let spirv_only = DESCRIPTOR.with_artifacts(&[SPIRV_ARTIFACT]);
        assert!(split_variant_has_a_shader_source(&spirv_only, true));
        assert!(!split_variant_has_a_shader_source(&spirv_only, false));
    }

    #[test]
    fn a_kernel_without_an_artifact_in_a_build_without_the_compiler_is_no_artifact() {
        // The `slang-runtime` feature is always on in tests (the self dev-dependency), so this is the one place the
        // off-state policy runs: through the argument that stands for the feature.
        const SPIRV: gust::Artifact = WGSL_ARTIFACT.with_format(gust::Format::Spirv);
        let spirv_only = DESCRIPTOR.with_artifacts(&[SPIRV]);
        for kernel in [&DESCRIPTOR, &spirv_only] {
            let error = select_shader_source(kernel, false)
                .expect_err("no WGSL artifact and no compiler is an error, not a fallback");
            let Error::NoArtifact {
                kernel: name,
                profile,
            } = &error
            else {
                panic!("{error:?}");
            };
            assert_eq!(name, &kernel.qualified_name().to_string());
            assert_eq!(*profile, gust::profile::NAME);
            let message = error.to_string();
            assert!(message.contains(&format!("`{name}`")), "{message}");
            assert!(message.contains("cargo gust build"), "{message}");
            assert!(message.contains("slang-runtime"), "{message}");
        }
    }

    const PARAMETERS: &[ParameterDescriptor] = &[ParameterDescriptor {
        name: "out",
        rust_type: "StorageMut<f32>",
        kind: ParameterKind::Storage,
        access: Access::ReadWrite,
        resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
    }];

    const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
        "test",
        "kernel",
        "gpu_test_kernel",
        [64, 1, 1],
        PARAMETERS,
        r#"
            [[vk::binding(0, 0)]] RWStructuredBuffer<float> out;
            [shader("compute")]
            [numthreads(64, 1, 1)]
            void gpu_test_kernel(uint3 id : SV_DispatchThreadID) { if (id.x < 1) out[id.x] = 0.0; }
        "#,
    );
}

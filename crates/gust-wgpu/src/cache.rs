use std::{collections::HashMap, sync::Arc};

use gust::{Artifact, Format, KernelDescriptor, Shape};

/// Observable state of one device's compiled-kernel cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineCacheStats {
    /// The number of compiled kernels held in the cache.
    pub entries: usize,
    /// The number of lookups that found a compiled kernel in the cache.
    pub hits: u64,
    /// The number of lookups that compiled the kernel and stored it.
    pub misses: u64,
}

/// The format of the artifact in a [`KernelCacheKey`]. `gust::Format` is not `Hash`, and `gust` is not this crate's
/// to change, so the key carries its own copy of the two cases (`Format` is `#[non_exhaustive]`, so a format that this
/// release does not know is `Other`; the key only ever holds a WGSL one, see `preferred_artifact`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ArtifactFormat {
    Wgsl,
    Spirv,
    Other,
}

impl From<Format> for ArtifactFormat {
    fn from(format: Format) -> Self {
        match format {
            Format::Wgsl => Self::Wgsl,
            Format::Spirv => Self::Spirv,
            _ => Self::Other,
        }
    }
}

/// The artifact this backend runs: the first WGSL artifact of the descriptor. SPIR-V artifacts are for other
/// backends and are ignored here, in the cache key as well as in the shader selection.
pub(crate) fn preferred_artifact(kernel: &KernelDescriptor) -> Option<&'static Artifact> {
    kernel
        .artifacts
        .iter()
        .find(|artifact| artifact.format == Format::Wgsl)
}

/// What identifies one compiled kernel in the cache of a device (R4.5). The fields are addresses and lengths of the
/// descriptor's static slices, not their contents: two descriptors that share every slice share a pipeline, and
/// descriptors that differ in any field below never share one. The artifact and its reflection JSON are in the key
/// because a cache hit skips `verify_artifact`: an artifact that was never verified must not be able to borrow the
/// pipeline of one that was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct KernelCacheKey {
    entry_point: &'static str,
    slang_address: usize,
    slang_bytes: usize,
    parameters_address: usize,
    parameters: usize,
    /// The shape (mx.k): the stage of the entry point, and so what kind of pipeline the entry point makes. Two
    /// descriptors that differ only in it never share a pipeline.
    shape: Shape,
    workgroup_size: [u32; 3],
    artifact_address: usize,
    artifact_bytes: usize,
    artifact_key: Option<&'static str>,
    reflection_address: usize,
    reflection_bytes: usize,
    format: Option<ArtifactFormat>,
}

impl KernelCacheKey {
    pub(crate) fn new(kernel: &KernelDescriptor) -> Self {
        let artifact = preferred_artifact(kernel);
        Self {
            entry_point: kernel.entry_point,
            slang_address: kernel.slang_source.as_ptr() as usize,
            slang_bytes: kernel.slang_source.len(),
            parameters_address: kernel.parameters.as_ptr() as usize,
            parameters: kernel.parameters.len(),
            shape: kernel.shape,
            workgroup_size: kernel.workgroup_size,
            artifact_address: artifact.map_or(0, |artifact| artifact.bytes.as_ptr() as usize),
            artifact_bytes: artifact.map_or(0, |artifact| artifact.bytes.len()),
            artifact_key: artifact.map(|artifact| artifact.key),
            reflection_address: artifact
                .map_or(0, |artifact| artifact.reflection.as_ptr() as usize),
            reflection_bytes: artifact.map_or(0, |artifact| artifact.reflection.len()),
            format: artifact.map(|artifact| artifact.format.into()),
        }
    }
}

/// The pipeline of a cached kernel: a compute pipeline for a `grid` kernel, a full-screen render pipeline for a
/// `pixel` one (slice mx.p1b). The cache key holds the shape, so one key has one kind.
pub(crate) enum Pipeline {
    Compute(wgpu::ComputePipeline),
    Render(wgpu::RenderPipeline),
}

impl Pipeline {
    /// The compute pipeline of a `grid` kernel. The v1 one-shot functions call it after refusing every other shape.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn compute(&self) -> &wgpu::ComputePipeline {
        match self {
            Self::Compute(pipeline) => pipeline,
            Self::Render(_) => unreachable!("the v1 dispatch functions refuse a non-grid kernel"),
        }
    }
}

pub(crate) struct CachedKernel {
    pub(crate) _shader: wgpu::ShaderModule,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) _pipeline_layout: wgpu::PipelineLayout,
    pub(crate) pipeline: Pipeline,
}

#[derive(Default)]
pub(crate) struct PipelineCache {
    pub(crate) kernels: HashMap<KernelCacheKey, Arc<CachedKernel>>,
    /// The vertex module of every `pixel` pipeline of the device (`shapes/fullscreen.wgsl`), created on first use.
    pub(crate) fullscreen: Option<wgpu::ShaderModule>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    const BYTES: &[u8] = b"@compute fn main() {}";
    const REFLECTION: &str = "{\"schemaVersion\":1}";

    const BASE: Artifact = Artifact {
        format: Format::Wgsl,
        key: "0123456789abcdef",
        bytes: BYTES,
        reflection: REFLECTION,
    };

    const ARTIFACTS: &[Artifact] = &[BASE];

    const KERNEL: KernelDescriptor = KernelDescriptor::new(
        "cache_test",
        "kernel",
        "cache_test_kernel",
        [64, 1, 1],
        &[],
        "void cache_test_kernel() {}",
    )
    .with_artifacts(ARTIFACTS);

    fn with(artifact: Artifact) -> KernelDescriptor {
        KERNEL.with_artifacts(Box::leak(Box::new([artifact])))
    }

    fn key(kernel: &KernelDescriptor) -> KernelCacheKey {
        KernelCacheKey::new(kernel)
    }

    fn leaked_copy(bytes: &[u8]) -> &'static [u8] {
        Box::leak(bytes.to_vec().into_boxed_slice())
    }

    #[test]
    fn the_same_descriptor_has_the_same_key() {
        assert_eq!(key(&KERNEL), key(&KERNEL));
        assert_eq!(key(&with(BASE)), key(&with(BASE)));
    }

    #[test]
    fn descriptors_that_differ_only_in_the_artifact_address_have_different_keys() {
        // Same length, same key text, same reflection, same content: only the address of the bytes differs.
        let copy = Artifact {
            bytes: leaked_copy(BYTES),
            ..BASE
        };
        assert_eq!(copy.bytes, BASE.bytes);
        assert_ne!(copy.bytes.as_ptr(), BASE.bytes.as_ptr());
        assert_ne!(key(&with(BASE)), key(&with(copy)));
    }

    #[test]
    fn descriptors_that_differ_only_in_the_artifact_length_have_different_keys() {
        // Same address (a prefix of the same slice).
        let shorter = Artifact {
            bytes: &BYTES[..BYTES.len() - 1],
            ..BASE
        };
        assert_eq!(shorter.bytes.as_ptr(), BASE.bytes.as_ptr());
        assert_ne!(key(&with(BASE)), key(&with(shorter)));
    }

    #[test]
    fn descriptors_that_differ_only_in_the_artifact_key_have_different_keys() {
        let renamed = Artifact {
            key: "fedcba9876543210",
            ..BASE
        };
        assert_ne!(key(&with(BASE)), key(&with(renamed)));
    }

    #[test]
    fn descriptors_with_the_same_artifact_bytes_and_different_reflection_json_have_different_keys()
    {
        // R2-5: the reflection JSON is part of the identity. Same bytes, same length of JSON, different address.
        let copy: &'static str = Box::leak(REFLECTION.to_string().into_boxed_str());
        let moved = Artifact {
            reflection: copy,
            ..BASE
        };
        assert_eq!(moved.reflection, BASE.reflection);
        assert_ne!(moved.reflection.as_ptr(), BASE.reflection.as_ptr());
        assert_ne!(key(&with(BASE)), key(&with(moved)));

        // Different length, same address.
        let shorter = Artifact {
            reflection: &REFLECTION[..REFLECTION.len() - 1],
            ..BASE
        };
        assert_eq!(shorter.reflection.as_ptr(), BASE.reflection.as_ptr());
        assert_ne!(key(&with(BASE)), key(&with(shorter)));
    }

    #[test]
    fn descriptors_that_differ_only_in_the_shape_have_different_keys() {
        let pixel = KERNEL
            .with_shape(Shape::Pixel)
            .with_compatible_shapes(&[Shape::Pixel]);
        assert_ne!(key(&KERNEL), key(&pixel));
        assert_eq!(key(&pixel), key(&pixel));
    }

    #[test]
    fn a_descriptor_with_an_artifact_differs_from_one_without() {
        let bare = KERNEL.with_artifacts(&[]);
        assert_ne!(key(&bare), key(&KERNEL));
    }

    #[test]
    fn a_spirv_artifact_is_not_part_of_the_key_of_this_backend() {
        // This backend runs WGSL: a descriptor that adds a SPIR-V artifact compiles the same pipeline.
        let spirv = Artifact {
            format: Format::Spirv,
            ..BASE
        };
        let bare = KERNEL.with_artifacts(&[]);
        assert_eq!(key(&with(spirv)), key(&bare));
        assert!(preferred_artifact(&with(spirv)).is_none());
        assert_eq!(preferred_artifact(&KERNEL), Some(&BASE));
    }
}

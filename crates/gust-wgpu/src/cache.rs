use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
};

use gust::{
    Artifact, Format, KernelDescriptor, ParameterDescriptor, Shape, capability::Capability,
};

/// Observable state of one device's compiled-kernel cache.
///
/// [`HeadlessDevice::pipeline_cache_stats`](crate::HeadlessDevice::pipeline_cache_stats) returns one. The first
/// dispatch of a kernel compiles it (a miss), unless
/// [`HeadlessDevice::precompile`](crate::HeadlessDevice::precompile) created it first (that call is the miss, and the
/// dispatch a hit, wherever the descriptor is stored); the next dispatch of the same kernel finds it (a hit):
///
/// ```
/// use gust::gpu_module;
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
///
/// #[gpu_module]
/// mod double {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let out = device.create_typed_buffer("out", &[1.0f32, 2.0], GpuBufferAccess::ReadWrite);
///     let stats = device.pipeline_cache_stats();
///     assert_eq!((stats.entries, stats.hits, stats.misses), (0, 0, 0));
///
///     double::run::dispatch(&device, 2, &out)?;
///     double::run::dispatch(&device, 2, &out)?;
///     let stats = device.pipeline_cache_stats();
///     assert_eq!((stats.entries, stats.hits, stats.misses), (1, 1, 1));
///     assert_eq!(device.read_typed_buffer(&out)?, [4.0, 8.0]);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
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

/// What a lookup finds first in the cache of a device (R4.5): the address of the descriptor's static slices, the fast
/// path. Two descriptors that share every slice share a pipeline here, and descriptors that differ in any field below
/// never do. A descriptor at an address not seen before falls through to [`KernelContentKey`], which is what
/// makes two copies of one descriptor one entry. The artifact and its reflection JSON are in the key because a cache
/// hit skips `verify_artifact`: an artifact whose content was never verified must not be able to borrow the
/// pipeline of one that was. The one field that is by content is `requires`: the set of capabilities (D142), small, and
/// compared and hashed in order.
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
    /// The capabilities the selected body needs (M8, D142), by content: the same pipeline never serves a descriptor that
    /// differs only in what it requires, and a device only ever caches kernels whose `requires` it was opened with.
    requires: &'static [Capability],
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
            requires: kernel.requires,
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

/// What a compiled kernel is, by content: everything that changes its shader module or its pipeline (the ruling of
/// m14.bzpre). The address key above is the fast path and an alias of this one: a descriptor reached through
/// another address (a `const` is copied to one address per use, so `Kernel::DESCRIPTOR` and the typed call's own
/// reference do not share one) finds the pipeline here once, and the alias then serves its later lookups without
/// hashing. Equality compares every field in full, after the stored hash: the hash is a filter over the fields that
/// tell descriptors apart (`ParameterDescriptor` is not `Hash`, so a parameter counts by name and binding), never the
/// decision. The hash is computed once, in [`KernelContentKey::new`], which runs outside the cache's lock, so no
/// content is hashed while the lock is held. The labels of a descriptor (`module`, `name`, `variant`) are not part of
/// it: a descriptor that differs only in them reuses the first one's pipeline, with the first one's wgpu labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KernelContentKey {
    /// First, so that the derived `Eq` compares it before any content.
    hash: u64,
    entry_point: &'static str,
    slang_source: &'static str,
    parameters: &'static [ParameterDescriptor],
    shape: Shape,
    workgroup_size: [u32; 3],
    requires: &'static [Capability],
    artifact: Option<ArtifactContent>,
}

/// The preferred artifact of a descriptor, by content (`gust::Artifact` is not `Hash`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArtifactContent {
    format: ArtifactFormat,
    key: &'static str,
    bytes: &'static [u8],
    reflection: &'static str,
}

impl Hash for KernelContentKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl KernelContentKey {
    pub(crate) fn new(kernel: &KernelDescriptor) -> Self {
        let artifact = preferred_artifact(kernel).map(|artifact| ArtifactContent {
            format: artifact.format.into(),
            key: artifact.key,
            bytes: artifact.bytes,
            reflection: artifact.reflection,
        });
        let mut key = Self {
            hash: 0,
            entry_point: kernel.entry_point,
            slang_source: kernel.slang_source,
            parameters: kernel.parameters,
            shape: kernel.shape,
            workgroup_size: kernel.workgroup_size,
            requires: kernel.requires,
            artifact,
        };
        key.hash = key.content_hash();
        key
    }

    /// The hash over the fields that tell descriptors apart. A parameter counts by name and binding only.
    fn content_hash(&self) -> u64 {
        let mut state = std::collections::hash_map::DefaultHasher::new();
        self.entry_point.hash(&mut state);
        self.slang_source.hash(&mut state);
        self.parameters.len().hash(&mut state);
        for parameter in self.parameters {
            parameter.name.hash(&mut state);
            parameter
                .binding
                .map(|binding| (binding.group, binding.binding))
                .hash(&mut state);
        }
        self.shape.hash(&mut state);
        self.workgroup_size.hash(&mut state);
        self.requires.hash(&mut state);
        if let Some(artifact) = &self.artifact {
            artifact.format.hash(&mut state);
            artifact.key.hash(&mut state);
            artifact.bytes.hash(&mut state);
            artifact.reflection.hash(&mut state);
        }
        state.finish()
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
    /// The addresses seen so far, each an alias of an entry of `contents`: the lookup of the dispatch path, which
    /// hashes nothing but the key's own fields. Aliases are not entries.
    pub(crate) kernels: HashMap<KernelCacheKey, Arc<CachedKernel>>,
    /// The compiled kernels by content: the entries `PipelineCacheStats::entries` counts.
    pub(crate) contents: HashMap<KernelContentKey, Arc<CachedKernel>>,
    /// The vertex module of every `pixel` pipeline of the device (`shapes/fullscreen.wgsl`), created on first use.
    pub(crate) fullscreen: Option<wgpu::ShaderModule>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::{Access, GpuPod, ParameterKind, ResourceBinding, ResourceLayout};

    const BYTES: &[u8] = b"@compute fn main() {}";
    const REFLECTION: &str = "{\"schemaVersion\":1}";

    const BASE: Artifact = Artifact::new(Format::Wgsl, "0123456789abcdef", BYTES, REFLECTION);

    const ARTIFACTS: &[Artifact] = &[BASE];

    const PARAMS: &[ParameterDescriptor] = &[ParameterDescriptor {
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

    const KERNEL: KernelDescriptor = KernelDescriptor::new(
        "cache_test",
        "kernel",
        "cache_test_kernel",
        [64, 1, 1],
        PARAMS,
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
        let copy = BASE.with_bytes(leaked_copy(BYTES));
        assert_eq!(copy.bytes, BASE.bytes);
        assert_ne!(copy.bytes.as_ptr(), BASE.bytes.as_ptr());
        assert_ne!(key(&with(BASE)), key(&with(copy)));
    }

    #[test]
    fn descriptors_that_differ_only_in_the_artifact_length_have_different_keys() {
        // Same address (a prefix of the same slice).
        let shorter = BASE.with_bytes(&BYTES[..BYTES.len() - 1]);
        assert_eq!(shorter.bytes.as_ptr(), BASE.bytes.as_ptr());
        assert_ne!(key(&with(BASE)), key(&with(shorter)));
    }

    #[test]
    fn descriptors_that_differ_only_in_the_artifact_key_have_different_keys() {
        let renamed = BASE.with_key("fedcba9876543210");
        assert_ne!(key(&with(BASE)), key(&with(renamed)));
    }

    #[test]
    fn descriptors_with_the_same_artifact_bytes_and_different_reflection_json_have_different_keys()
    {
        // R2-5: the reflection JSON is part of the identity. Same bytes, same length of JSON, different address.
        let copy: &'static str = Box::leak(REFLECTION.to_string().into_boxed_str());
        let moved = BASE.with_reflection(copy);
        assert_eq!(moved.reflection, BASE.reflection);
        assert_ne!(moved.reflection.as_ptr(), BASE.reflection.as_ptr());
        assert_ne!(key(&with(BASE)), key(&with(moved)));

        // Different length, same address.
        let shorter = BASE.with_reflection(&REFLECTION[..REFLECTION.len() - 1]);
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
    fn descriptors_that_differ_only_in_what_they_require_have_different_keys() {
        let subgroups = KERNEL.with_requires(&[Capability::Subgroups]);
        assert_ne!(key(&KERNEL), key(&subgroups));
        assert_eq!(key(&subgroups), key(&subgroups));
        // By content: another slice with the same capabilities is the same pipeline.
        let again = KERNEL.with_requires(Box::leak(Box::new([Capability::Subgroups])));
        assert_eq!(key(&subgroups), key(&again));
        // And the empty set is the default.
        assert_eq!(key(&KERNEL), key(&KERNEL.with_requires(&[])));
    }

    #[test]
    fn a_descriptor_with_an_artifact_differs_from_one_without() {
        let bare = KERNEL.with_artifacts(&[]);
        assert_ne!(key(&bare), key(&KERNEL));
    }

    #[test]
    fn a_spirv_artifact_is_not_part_of_the_key_of_this_backend() {
        // This backend runs WGSL: a descriptor that adds a SPIR-V artifact compiles the same pipeline.
        let spirv = BASE.with_format(Format::Spirv);
        let bare = KERNEL.with_artifacts(&[]);
        assert_eq!(key(&with(spirv)), key(&bare));
        assert!(preferred_artifact(&with(spirv)).is_none());
        assert_eq!(preferred_artifact(&KERNEL), Some(&BASE));
    }

    fn content(kernel: &KernelDescriptor) -> KernelContentKey {
        KernelContentKey::new(kernel)
    }

    fn hash_of(key: &KernelContentKey) -> u64 {
        let mut state = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut state);
        state.finish()
    }

    /// `KERNEL` with `PARAMS` replaced by the same parameter with `access` changed.
    fn with_access(access: Access) -> KernelDescriptor {
        let parameters: &'static [ParameterDescriptor] =
            Box::leak(Box::new([ParameterDescriptor {
                access,
                ..PARAMS[0]
            }]));
        KERNEL.with_parameters(parameters)
    }

    fn with_source(source: &'static str) -> KernelDescriptor {
        KernelDescriptor::new(
            "cache_test",
            "kernel",
            "cache_test_kernel",
            [64, 1, 1],
            PARAMS,
            source,
        )
        .with_artifacts(ARTIFACTS)
    }

    #[test]
    fn two_copies_of_one_descriptor_have_one_content_key_and_two_address_keys() {
        // Fresh copies of the source, the parameter list and the artifact: other addresses, the same content.
        let source: &'static str = Box::leak(String::from("void cache_test_kernel() {}").into());
        let parameters: &'static [ParameterDescriptor] =
            Box::leak(PARAMS.to_vec().into_boxed_slice());
        let artifact = Artifact::new(
            Format::Wgsl,
            "0123456789abcdef",
            leaked_copy(BYTES),
            Box::leak(String::from(REFLECTION).into_boxed_str()),
        );
        let copy = with_source(source)
            .with_parameters(parameters)
            .with_artifacts(Box::leak(Box::new([artifact])));
        assert_ne!(key(&KERNEL), key(&copy));
        assert_eq!(content(&KERNEL), content(&copy));
        assert_eq!(hash_of(&content(&KERNEL)), hash_of(&content(&copy)));
    }

    #[test]
    fn the_content_key_tells_apart_whatever_changes_the_pipeline() {
        let base = content(&KERNEL);
        let differs = |kernel: KernelDescriptor| assert_ne!(base, content(&kernel));
        differs(KERNEL.with_entry_point("other"));
        differs(KERNEL.with_requires(&[Capability::Subgroups]));
        differs(KERNEL.with_artifacts(&[]));
        differs(with(BASE.with_key("fedcba9876543210")));
        differs(with(BASE.with_bytes(b"@compute fn other() {}")));
        differs(with(BASE.with_reflection("{\"schemaVersion\":2}")));
        differs(with_source("void cache_test_kernel() { }"));
        differs(KERNEL.with_workgroup_size([32, 1, 1]));
        differs(KERNEL.with_shape(Shape::Pixel));
        differs(KERNEL.with_parameters(&[]));
        differs(with_access(Access::ReadOnly));
        let moved = ParameterDescriptor {
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ..PARAMS[0]
        };
        differs(KERNEL.with_parameters(Box::leak(Box::new([moved]))));
        // Names that only label the kernel do not change the pipeline.
        assert_eq!(base, content(&KERNEL.with_name("renamed")));
    }

    #[test]
    fn a_hash_match_is_decided_by_content_not_by_the_hash() {
        // `access` is not in the hash (a parameter counts by name and binding), so these two collide, and only the
        // full comparison tells them apart: the layout entry they make differs in `read_only`.
        let read_write = content(&with_access(Access::ReadWrite));
        let read_only = content(&with_access(Access::ReadOnly));
        assert_eq!(read_write.hash, read_only.hash);
        assert_eq!(hash_of(&read_write), hash_of(&read_only));
        assert_ne!(read_write, read_only);
        // In a map the two stay two entries, and each finds its own.
        let mut map = HashMap::new();
        map.insert(read_write, 1);
        map.insert(read_only, 2);
        assert_eq!(map.len(), 2);
        assert_eq!(map[&read_write], 1);
        assert_eq!(map[&read_only], 2);
    }
}

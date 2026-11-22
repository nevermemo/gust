use crate::cache::{CachedKernel, KernelCacheKey, Pipeline};
#[cfg(not(feature = "slang-runtime"))]
use crate::no_artifact;
use crate::{Error, HeadlessDevice, ShaderSource, render, select_shader_source};
use gust::{Access, KernelDescriptor, ParameterDescriptor, ParameterKind, Shape};
use std::sync::Arc;

impl HeadlessDevice {
    pub(crate) fn cached_kernel(
        &self,
        kernel: &KernelDescriptor,
        resources: &[&ParameterDescriptor],
    ) -> Result<Arc<CachedKernel>, Error> {
        let key = KernelCacheKey::new(kernel);
        {
            let mut cache = self.lock_pipeline_cache();
            if let Some(cached) = cache.kernels.get(&key).cloned() {
                cache.hits += 1;
                return Ok(cached);
            }
        }

        let compiled = Arc::new(self.compile_kernel(kernel, resources)?);
        let mut cache = self.lock_pipeline_cache();
        if let Some(cached) = cache.kernels.get(&key).cloned() {
            cache.hits += 1;
            return Ok(cached);
        }
        cache.misses += 1;
        cache.kernels.insert(key, Arc::clone(&compiled));
        Ok(compiled)
    }

    /// Compile and reflect `kernel` through the native helper, then cross-check it against the descriptor.
    #[cfg(feature = "slang-runtime")]
    fn compile_at_run_time(
        kernel: &KernelDescriptor,
    ) -> Result<gust::reflect::CompiledReflection, Error> {
        let compiled = gust::reflect::compile_reflected(kernel, gust::slang::Target::Wgsl)
            .map_err(Error::Reflection)?;
        let _ = compiled
            .reflection
            .cross_check_kernel(kernel)
            .map_err(Error::Reflection)?;
        Ok(compiled)
    }

    /// Without `slang-runtime` this crate holds no run-time Slang bridge: the only kernels it can run are the ones whose
    /// precompiled artifact it verifies. `select_shader_source` never sends a kernel here in this build (it
    /// answers `NoArtifact` first); the same error is returned in case a caller does.
    #[cfg(not(feature = "slang-runtime"))]
    fn compile_at_run_time(
        kernel: &KernelDescriptor,
    ) -> Result<gust::reflect::CompiledReflection, Error> {
        Err(no_artifact(kernel))
    }

    fn compile_kernel(
        &self,
        kernel: &KernelDescriptor,
        resources: &[&ParameterDescriptor],
    ) -> Result<CachedKernel, Error> {
        let pixel = match kernel.shape {
            Shape::Grid => false,
            Shape::Pixel => {
                // Without fragment-stage storage writes there is no pipeline to build (D120).
                render::require_fragment_storage(kernel, self.fragment_storage)?;
                true
            }
            // `Shape` is `#[non_exhaustive]`: a shape this backend does not know has no pipeline either.
            _ => return Err(render::unknown_shape(kernel)),
        };
        let mut from_artifact = false;
        let compiled = match select_shader_source(kernel, cfg!(feature = "slang-runtime"))? {
            // A verification failure is the answer: there is no fallback to the run-time compiler.
            ShaderSource::Precompiled(artifact) => {
                from_artifact = true;
                gust::artifact::verify_artifact(kernel, artifact)
                    .map_err(Error::Reflection)?
                    .compiled
            }
            ShaderSource::RunTime => Self::compile_at_run_time(kernel)?,
        };
        let wgsl = std::str::from_utf8(&compiled.artifact)
            .expect("WGSL validated by the native compiler bridge");
        let label = kernel.qualified_name().to_string();
        // A texture or sampler entry follows the device's layout variant (`texture.rs`); a buffer entry is as ever.
        let layout_entries = resources
            .iter()
            .map(|parameter| {
                let ty = match parameter.kind {
                    ParameterKind::Texture | ParameterKind::Sampler => {
                        self.texture_binding_type(parameter)?
                    }
                    _ => wgpu::BindingType::Buffer {
                        ty: if parameter.kind == ParameterKind::Uniform {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: parameter.access == Access::ReadOnly,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                };
                Ok(wgpu::BindGroupLayoutEntry {
                    binding: parameter
                        .binding
                        .expect("validated resource binding")
                        .binding,
                    visibility: if pixel {
                        wgpu::ShaderStages::FRAGMENT
                    } else {
                        wgpu::ShaderStages::COMPUTE
                    },
                    ty,
                    count: None,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{label} bind group layout")),
                    entries: &layout_entries,
                });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label} pipeline layout")),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let shader =
            self.shader_module(kernel, &label, wgsl, &layout_entries, pixel, from_artifact);
        let pipeline = if pixel {
            let vertex = self.fullscreen_module();
            Pipeline::Render(render::build_pipeline(
                &self.device,
                &label,
                &pipeline_layout,
                &vertex,
                &shader,
                kernel.entry_point,
            ))
        } else {
            Pipeline::Compute(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{label} pipeline")),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(kernel.entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ))
        };
        Ok(CachedKernel {
            _shader: shader,
            bind_group_layout,
            _pipeline_layout: pipeline_layout,
            pipeline,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::KernelCacheKey;
    use crate::{BufferBinding, GpuBufferAccess, HeadlessDevice};
    use gust::{
        Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
        ResourceLayout,
    };

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

    #[test]
    fn pipeline_cache_identity_includes_entrypoint() {
        let other = DESCRIPTOR.with_entry_point("gpu_other_entry");
        assert_ne!(
            KernelCacheKey::new(&DESCRIPTOR),
            KernelCacheKey::new(&other)
        );
        assert_eq!(
            KernelCacheKey::new(&DESCRIPTOR),
            KernelCacheKey::new(&DESCRIPTOR)
        );
    }

    #[test]
    fn pipeline_cache_identity_ignores_the_diagnostic_source_origin() {
        // The line table is diagnostic metadata only (DX19): two descriptors that
        // differ only in `source` must share one cached pipeline.
        const LINES: &[gust::LineSpan] = &[gust::LineSpan {
            slang_start: 3,
            slang_end: 3,
            rust_line: 9,
        }];
        let located = DESCRIPTOR.with_source(gust::SourceOrigin {
            file: "crates/demo/src/lib.rs",
            lines: LINES,
        });
        assert_ne!(located.source, DESCRIPTOR.source);
        assert_eq!(
            KernelCacheKey::new(&DESCRIPTOR),
            KernelCacheKey::new(&located)
        );
    }

    #[test]
    fn shared_source_entrypoints_execute_distinct_pipelines() {
        const SOURCE: &str = r#"
            [[vk::binding(0, 0)]] RWStructuredBuffer<float> out;
            [shader("compute")] [numthreads(64, 1, 1)]
            void first(uint3 id : SV_DispatchThreadID) { if (id.x == 0) out[0] = 11.0; }
            [shader("compute")] [numthreads(64, 1, 1)]
            void second(uint3 id : SV_DispatchThreadID) { if (id.x == 0) out[0] = 22.0; }
        "#;
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let buffer =
            device.create_typed_buffer("entrypoint identity", &[0.0], GpuBufferAccess::ReadWrite);
        for (entry, expected) in [
            ("first", 11.0),
            ("second", 22.0),
            ("first", 11.0),
            ("second", 22.0),
        ] {
            let descriptor = DESCRIPTOR.with_entry_point(entry).with_slang_source(SOURCE);
            device
                .dispatch_buffers(&descriptor, 1, &[BufferBinding::read_write(&buffer)])
                .unwrap();
            assert_eq!(device.read_typed_buffer(&buffer).unwrap(), [expected]);
        }
        let stats = device.pipeline_cache_stats();
        assert_eq!((stats.entries, stats.misses, stats.hits), (2, 2, 2));
    }
}

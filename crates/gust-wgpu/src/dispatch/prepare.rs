use crate::texture;
use crate::texture::TextureCounts;
use crate::{
    BindGroupSlot, BufferBinding, BufferDispatch, Error, HeadlessDevice, INVOCATION_CAP_LIMIT,
    LAUNCH_BASE_BYTES, Launch, LaunchPlan, PreparedDispatch, SplitLaunch, SplitPiece,
    check_binding_indices, check_binding_size, check_kernel_capabilities, check_kernel_limits,
    check_kernel_requires, render, split_pieces, split_variant_has_a_shader_source,
    split_variant_matches, validate_dispatch_extent, validate_kernel, workgroup_counts,
};
use gust::{BindingKind, Extent, KernelDescriptor, ParameterDescriptor, ParameterKind, Shape};
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

/// The last whole launch of a batch's preparation: its pipeline entry, the entries its bind group was built from, and
/// the group.
type LastGroup<'a> = (
    Arc<crate::cache::CachedKernel>,
    Vec<wgpu::BindGroupEntry<'a>>,
    wgpu::BindGroup,
);

/// Whether two bind group entry lists bind the same resources at the same binding indices: the same buffers with the
/// same offsets and sizes, the same texture views and the same samplers. Anything else (a resource kind this crate
/// never binds included) is a difference.
fn same_entries(first: &[wgpu::BindGroupEntry<'_>], second: &[wgpu::BindGroupEntry<'_>]) -> bool {
    first.len() == second.len()
        && first.iter().zip(second).all(|(a, b)| {
            a.binding == b.binding
                && match (&a.resource, &b.resource) {
                    (wgpu::BindingResource::Buffer(x), wgpu::BindingResource::Buffer(y)) => {
                        x.buffer == y.buffer && x.offset == y.offset && x.size == y.size
                    }
                    (
                        wgpu::BindingResource::TextureView(x),
                        wgpu::BindingResource::TextureView(y),
                    ) => x == y,
                    (wgpu::BindingResource::Sampler(x), wgpu::BindingResource::Sampler(y)) => {
                        x == y
                    }
                    _ => false,
                }
        })
}

impl HeadlessDevice {
    /// The staged graph's preparation (`graph.rs` reads `workgroup_count`, the x count): one-dimensional dispatches
    /// only. A dispatch with a y or z extent other than 1 is refused with [`Error::Extent`] instead of being encoded
    /// as its x count alone, which would silently run a fraction of it.
    ///
    /// `slot` is the node's kept bind group ([`BindGroupSlot`]): it is reused when the pipeline entry is the one it
    /// was built for, and replaced otherwise.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn prepare_node(
        &self,
        dispatch: &BufferDispatch<'_>,
        slot: &BindGroupSlot,
    ) -> Result<Vec<PreparedDispatch>, Error> {
        if dispatch.extent().y != 1 || dispatch.extent().z != 1 {
            // A `pixel` kernel has no extent launch at all: the refusal is its own, not the graph's.
            render::check_pixel_extent(dispatch.kernel, dispatch.extent())?;
            return Err(Error::Extent {
                extent: dispatch.extent().to_array().map(u64::from),
                reason: "a staged graph runs one-dimensional dispatches only (use dispatch_batch)",
            });
        }
        self.prepare_dispatches(std::slice::from_ref(dispatch), false, Some(slot))
    }

    /// The batch functions' preparation: every dispatch over its whole [`Extent`].
    pub(crate) fn prepare_extent_batch(
        &self,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<Vec<PreparedDispatch>, Error> {
        self.prepare_dispatches(dispatches, false, None)
    }

    /// Prepare one indirect dispatch: every check of `prepare_node` except the workgroup count,
    /// which only the GPU reads, from the arguments buffer. `dispatch.extent()` must be a
    /// nonzero placeholder; the prepared `workgroup_count` is meaningless (0) and never encoded.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn prepare_indirect(
        &self,
        dispatch: &BufferDispatch<'_>,
        slot: &BindGroupSlot,
    ) -> Result<Vec<PreparedDispatch>, Error> {
        self.prepare_dispatches(std::slice::from_ref(dispatch), true, Some(slot))
    }

    /// `slot`, given only for one dispatch (a staged graph's node), keeps the bind group of a whole launch from one
    /// call to the next; split pieces and every other caller build theirs each time.
    fn prepare_dispatches(
        &self,
        dispatches: &[BufferDispatch<'_>],
        indirect: bool,
        slot: Option<&BindGroupSlot>,
    ) -> Result<Vec<PreparedDispatch>, Error> {
        let validated = dispatches
            .iter()
            .map(|dispatch| {
                let resources = validate_dispatch_extent(
                    self.id,
                    self.float32_filterable(),
                    dispatch.kernel,
                    dispatch.extent(),
                    dispatch.bindings,
                    dispatch.samplers,
                )?;
                let binding_bytes = dispatch
                    .bindings
                    .iter()
                    .map(BufferBinding::limit_bytes)
                    .collect::<Vec<_>>();
                let plan = if indirect {
                    self.check_indirect_launch(
                        dispatch.kernel,
                        &resources,
                        binding_bytes.iter().copied(),
                    )?;
                    LaunchPlan::Whole([0; 3])
                } else {
                    match self.check_launch_extent(
                        dispatch.kernel,
                        dispatch.extent(),
                        &resources,
                        binding_bytes.iter().copied(),
                    ) {
                        Ok(workgroups) => LaunchPlan::Whole(workgroups),
                        // A launch past a workgroup limit may still run, as pieces (m11.g4b); every other
                        // refusal, and a launch that cannot be split, keeps this very error.
                        Err(refused) => {
                            LaunchPlan::Split(self.plan_split(dispatch, &binding_bytes, refused)?)
                        }
                    }
                };
                if indirect || !dispatch.extent().is_empty() {
                    texture::check_texture_dimensions(
                        &self.limits,
                        dispatch.kernel,
                        dispatch.bindings,
                    )?;
                }
                Ok((resources, plan))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let mut prepared = Vec::with_capacity(dispatches.len());
        let mut last: Option<LastGroup<'_>> = None;
        for (dispatch, (resources, plan)) in dispatches.iter().zip(validated) {
            if dispatch.extent().is_empty() {
                continue;
            }
            let workgroups = match plan {
                LaunchPlan::Whole(workgroups) => workgroups,
                LaunchPlan::Split(split) => {
                    // Split pieces keep their own groups, and break the run of consecutive whole launches.
                    last = None;
                    prepared.push(self.prepare_split(dispatch, split)?);
                    continue;
                }
            };
            let cached_kernel = self.cached_kernel(dispatch.kernel, &resources)?;
            let launch = if dispatch.kernel.shape == Shape::Pixel {
                Launch::Pixel {
                    elements: dispatch.extent().x,
                }
            } else {
                Launch::Compute { workgroups }
            };
            if let Some(kept) = slot.and_then(|slot| slot.reuse(&cached_kernel)) {
                last = None;
                prepared.push(PreparedDispatch {
                    cached_kernel,
                    bind_group: kept,
                    launch,
                    pieces: Vec::new(),
                });
                continue;
            }
            let label = dispatch.kernel.qualified_name().to_string();
            // The bound parameters in declaration order; a sampler parameter takes the next sampler binding and every
            // other one the next resource binding (`validate_dispatch_extent` counted both).
            let mut resource_bindings = dispatch.bindings.iter();
            let mut sampler_bindings = dispatch.samplers.iter();
            let bind_group_entries = resources
                .iter()
                .map(|parameter| wgpu::BindGroupEntry {
                    binding: parameter
                        .binding
                        .expect("validated resource binding")
                        .binding,
                    resource: if parameter.kind == ParameterKind::Sampler {
                        sampler_bindings
                            .next()
                            .expect("validated sampler binding")
                            .resource()
                    } else {
                        resource_bindings
                            .next()
                            .expect("validated resource binding")
                            .resource()
                    },
                })
                .collect::<Vec<_>>();
            // The previous whole launch of this call, when it ran the same pipeline entry over exactly the same
            // resources (its bind group is then the same group): one group serves both passes. A dispatch over an
            // empty extent in between is skipped above and does not break the run (it is never encoded).
            let bind_group = match &last {
                Some((kernel, entries, group))
                    if Arc::ptr_eq(kernel, &cached_kernel)
                        && same_entries(entries, &bind_group_entries) =>
                {
                    group.clone()
                }
                _ => self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("{label} batched bind group")),
                    layout: &cached_kernel.bind_group_layout,
                    entries: &bind_group_entries,
                }),
            };
            if let Some(slot) = slot {
                slot.keep(&cached_kernel, &bind_group);
            }
            last = Some((
                Arc::clone(&cached_kernel),
                bind_group_entries,
                bind_group.clone(),
            ));
            prepared.push(PreparedDispatch {
                cached_kernel,
                bind_group,
                launch,
                pieces: Vec::new(),
            });
        }
        Ok(prepared)
    }

    /// Decide whether a launch that the plain checks refused (`refused`) can run as pieces of the kernel's split
    /// variant (m11.g4b), and plan them. Only a `grid` launch refused for a workgroup limit (`refused` is that
    /// [`Error::DeviceLimit`]) qualifies, and only when the kernel's `split` is the one the `#[gpu_module]` macro makes
    /// ([`split_variant_matches`]), the variant has a shader source in this build (its own precompiled artifact, D209,
    /// or the run-time compiler, D208: [`split_variant_has_a_shader_source`]), and the variant passes every limit the base passed (one
    /// more uniform, its binding index, the hidden uniform's 16 bytes against the uniform binding size, the
    /// capabilities). Anything else returns `refused` unchanged, before anything
    /// is compiled. A launch the planner itself refuses (the invocation cap, too many pieces) returns that error.
    fn plan_split(
        &self,
        dispatch: &BufferDispatch<'_>,
        binding_bytes: &[(u64, BindingKind)],
        refused: Error,
    ) -> Result<SplitLaunch, Error> {
        let kernel = dispatch.kernel;
        let workgroup_limit = matches!(
            &refused,
            Error::DeviceLimit { limit, .. }
                if limit.starts_with("max_compute_workgroups_per_dimension")
                    || *limit == INVOCATION_CAP_LIMIT
        );
        let Some(variant) = kernel.split else {
            return Err(refused);
        };
        if !workgroup_limit
            || self.limits.max_compute_workgroups_per_dimension == 0
            || kernel.shape != Shape::Grid
            || !split_variant_matches(kernel, variant)
            || !split_variant_has_a_shader_source(variant, cfg!(feature = "slang-runtime"))
        {
            return Err(refused);
        }
        let Ok(resources) = validate_kernel(variant) else {
            return Err(refused);
        };
        let count = |kind: ParameterKind| {
            resources
                .iter()
                .filter(|parameter| parameter.kind == kind)
                .count()
        };
        if check_kernel_limits(
            &self.limits,
            variant,
            count(ParameterKind::Storage),
            count(ParameterKind::Uniform),
            TextureCounts::of(&resources),
        )
        .is_err()
            || check_kernel_capabilities(
                self.capabilities,
                self.subgroup_sizes,
                &self.adapter_info.name,
                variant,
            )
            .is_err()
            || check_binding_indices(&self.limits, variant, &resources).is_err()
            // The hidden launch-base uniform is a binding of its own against the uniform binding size limit, whatever
            // the caller's bindings are.
            || check_binding_size(
                &self.limits,
                variant,
                LAUNCH_BASE_BYTES,
                BindingKind::Uniform,
            )
            .is_err()
        {
            return Err(refused);
        }
        let pieces = split_pieces(&self.limits, kernel, dispatch.extent())?;
        for &(bytes, kind) in binding_bytes {
            check_binding_size(&self.limits, kernel, bytes, kind)?;
        }
        Ok(SplitLaunch {
            variant,
            resources,
            pieces,
        })
    }

    /// Build the prepared dispatch of a split launch: the variant's pipeline (the only one compiled), one uniform
    /// buffer holding every piece's base at a stride of `max(16, min_uniform_buffer_offset_alignment)`, and one bind
    /// group per piece that binds its own 16 bytes of it.
    fn prepare_split(
        &self,
        dispatch: &BufferDispatch<'_>,
        split: SplitLaunch,
    ) -> Result<PreparedDispatch, Error> {
        let SplitLaunch {
            variant,
            resources,
            pieces,
        } = split;
        let cached_kernel = self.cached_kernel(variant, &resources)?;
        let label = dispatch.kernel.qualified_name().to_string();
        let stride =
            // The device's own alignment: a lowered `with_limits` value only emulates a weaker target, and wgpu
            // validates the offsets against what the device was opened with.
            u64::from(self.device.limits().min_uniform_buffer_offset_alignment).max(LAUNCH_BASE_BYTES);
        let mut contents = vec![0_u8; pieces.len() * stride as usize];
        for (index, piece) in pieces.iter().enumerate() {
            let at = index * stride as usize;
            for (axis, base) in piece.base.iter().enumerate() {
                contents[at + 4 * axis..at + 4 * axis + 4].copy_from_slice(&base.to_le_bytes());
            }
        }
        let bases = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("{label} launch bases")),
            contents: &contents,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // The variant's resources are the kernel's and then the hidden uniform, which has no caller binding.
        let own = resources.len() - 1;
        let mut resource_resources = dispatch.bindings.iter().map(BufferBinding::resource);
        let mut samplers = dispatch.samplers.iter().map(|sampler| sampler.resource());
        let mut entries = Vec::with_capacity(resources.len());
        for parameter in &resources[..own] {
            entries.push(wgpu::BindGroupEntry {
                binding: parameter
                    .binding
                    .expect("validated resource binding")
                    .binding,
                resource: if parameter.kind == ParameterKind::Sampler {
                    samplers.next().expect("validated sampler binding")
                } else {
                    resource_resources
                        .next()
                        .expect("validated resource binding")
                },
            });
        }
        let hidden = resources[own]
            .binding
            .expect("validated resource binding")
            .binding;
        let mut groups = Vec::with_capacity(pieces.len());
        for (index, piece) in pieces.iter().enumerate() {
            let mut piece_entries = entries.clone();
            piece_entries.push(wgpu::BindGroupEntry {
                binding: hidden,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &bases,
                    offset: index as u64 * stride,
                    size: std::num::NonZeroU64::new(LAUNCH_BASE_BYTES),
                }),
            });
            groups.push(SplitPiece {
                bind_group: self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("{label} split bind group")),
                    layout: &cached_kernel.bind_group_layout,
                    entries: &piece_entries,
                }),
                workgroups: piece.workgroups,
            });
        }
        let first = groups.first().expect("a split launch has a piece");
        Ok(PreparedDispatch {
            bind_group: first.bind_group.clone(),
            launch: Launch::Compute {
                workgroups: first.workgroups,
            },
            cached_kernel,
            pieces: groups,
        })
    }

    /// Check one non-empty launch against the stored limits and return its workgroup
    /// count. Runs after `validate_*` and before the pipeline-cache lookup, so no
    /// compilation, cache hit or wgpu object can precede a limit error. A launch of
    /// zero elements does no GPU work and is checked only against the kernel's `requires`.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn check_launch(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        resources: &[&ParameterDescriptor],
        binding_bytes: impl IntoIterator<Item = (u64, BindingKind)>,
    ) -> Result<u32, Error> {
        self.check_launch_extent(
            kernel,
            Extent::linear(element_count),
            resources,
            binding_bytes,
        )
        .map(|workgroups| workgroups[0])
    }

    /// `check_launch` over an [`Extent`]: the workgroups along x, y and z. A launch with a zero axis does no GPU
    /// work, is checked only against the kernel's `requires` and has `[0, 0, 0]` workgroups.
    fn check_launch_extent(
        &self,
        kernel: &KernelDescriptor,
        extent: Extent,
        resources: &[&ParameterDescriptor],
        binding_bytes: impl IntoIterator<Item = (u64, BindingKind)>,
    ) -> Result<[u32; 3], Error> {
        if extent.is_empty() {
            // No work, but the kernel's `requires` is a fact of the descriptor: an empty launch is refused as a full one.
            self.check_requires(kernel)?;
            return Ok([0; 3]);
        }
        self.check_launch_limits(kernel, Some(extent), resources, binding_bytes)
    }

    /// The kernel's `requires` against the capabilities this device was opened with (`check_kernel_requires`), for the
    /// launches that return early when they have no work.
    pub(crate) fn check_requires(&self, kernel: &KernelDescriptor) -> Result<(), Error> {
        check_kernel_requires(self.capabilities, &self.adapter_info.name, kernel)
    }

    /// `check_launch` for an indirect dispatch: every check except the workgroup count, which the
    /// host never sees (the GPU reads it from the arguments buffer), so a lowered
    /// `max_compute_workgroups_per_dimension` never rejects an indirect launch.
    pub(crate) fn check_indirect_launch(
        &self,
        kernel: &KernelDescriptor,
        resources: &[&ParameterDescriptor],
        binding_bytes: impl IntoIterator<Item = (u64, BindingKind)>,
    ) -> Result<(), Error> {
        render::refuse_indirect(kernel)?;
        self.check_launch_limits(kernel, None, resources, binding_bytes)
            .map(|_| ())
    }

    /// The limit checks in their fixed order. `extent` is `None` for an indirect launch,
    /// which skips the workgroup counts and returns zeros.
    fn check_launch_limits(
        &self,
        kernel: &KernelDescriptor,
        extent: Option<Extent>,
        resources: &[&ParameterDescriptor],
        binding_bytes: impl IntoIterator<Item = (u64, BindingKind)>,
    ) -> Result<[u32; 3], Error> {
        let count = |kind: ParameterKind| {
            resources
                .iter()
                .filter(|parameter| parameter.kind == kind)
                .count()
        };
        check_kernel_limits(
            &self.limits,
            kernel,
            count(ParameterKind::Storage),
            count(ParameterKind::Uniform),
            TextureCounts::of(resources),
        )?;
        check_kernel_capabilities(
            self.capabilities,
            self.subgroup_sizes,
            &self.adapter_info.name,
            kernel,
        )?;
        check_binding_indices(&self.limits, kernel, resources)?;
        let workgroups = match (kernel.shape, extent) {
            // A `pixel` launch has no workgroups: the cap and the attachment are its limits (the adapter's ability to
            // write storage from a fragment first, since without it there is no pipeline to build).
            (Shape::Pixel, Some(extent)) => {
                render::require_fragment_storage(kernel, self.fragment_storage)?;
                render::check_pixel_launch(&self.limits, kernel, extent.x)?;
                [0; 3]
            }
            (_, Some(extent)) => workgroup_counts(&self.limits, kernel, extent)?,
            (_, None) => [0; 3],
        };
        for (bytes, kind) in binding_bytes {
            check_binding_size(&self.limits, kernel, bytes, kind)?;
        }
        Ok(workgroups)
    }
}

#[cfg(test)]
mod tests {
    use crate::{BufferBinding, BufferDispatch, GpuBufferAccess, HeadlessDevice};

    #[gust::gpu_module]
    mod ops {
        #[kernel(workgroup_size(64, 1, 1))]
        pub fn add_one(
            id: SV_DispatchThreadID,
            src: StructuredBuffer<uint>,
            mut dst: RWStructuredBuffer<uint>,
        ) {
            let i = id.x;
            if i < dst.len() && i < src.len() {
                dst[i] = src[i] + 1u32;
            }
        }

        #[kernel(workgroup_size(64, 1, 1))]
        pub fn add_two(
            id: SV_DispatchThreadID,
            src: StructuredBuffer<uint>,
            mut dst: RWStructuredBuffer<uint>,
        ) {
            let i = id.x;
            if i < dst.len() && i < src.len() {
                dst[i] = src[i] + 2u32;
            }
        }
    }

    /// The bind groups of a batch of dispatches over `pairs` (source, destination), in order.
    fn groups_of(
        device: &HeadlessDevice,
        pairs: &[(&crate::GpuBuffer<u32>, &crate::GpuBuffer<u32>)],
    ) -> Vec<wgpu::BindGroup> {
        let bindings: Vec<[BufferBinding<'_>; 2]> = pairs
            .iter()
            .map(|(src, dst)| {
                [
                    BufferBinding::read_only(src),
                    BufferBinding::read_write(dst),
                ]
            })
            .collect();
        let dispatches: Vec<_> = bindings
            .iter()
            .map(|pair| BufferDispatch::new(&ops::add_one::DESCRIPTOR, 3, pair))
            .collect();
        device
            .prepare_extent_batch(&dispatches)
            .unwrap()
            .into_iter()
            .map(|prepared| prepared.bind_group)
            .collect()
    }

    #[test]
    fn identical_consecutive_dispatches_share_one_bind_group() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let src = device.create_typed_buffer("src", &[1u32, 2, 3], GpuBufferAccess::ReadWrite);
        let dst = device.create_typed_buffer("dst", &[0u32; 3], GpuBufferAccess::ReadWrite);
        let groups = groups_of(&device, &[(&src, &dst); 4]);
        assert_eq!(groups.len(), 4);
        assert!(groups.iter().all(|group| *group == groups[0]));
    }

    #[test]
    fn a_dispatch_that_differs_in_one_binding_gets_its_own_bind_group() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let src = device.create_typed_buffer("src", &[1u32, 2, 3], GpuBufferAccess::ReadWrite);
        let dst = device.create_typed_buffer("dst", &[0u32; 3], GpuBufferAccess::ReadWrite);
        let other = device.create_typed_buffer("other", &[0u32; 3], GpuBufferAccess::ReadWrite);
        let groups = groups_of(&device, &[(&src, &dst), (&src, &dst), (&src, &other)]);
        assert_eq!(groups[0], groups[1]);
        assert_ne!(groups[1], groups[2]);
        // Only consecutive passes share: a return to the first bindings builds a new group.
        let groups = groups_of(&device, &[(&src, &dst), (&src, &other), (&src, &dst)]);
        assert_ne!(groups[0], groups[1]);
        assert_ne!(groups[1], groups[2]);
        assert_ne!(groups[0], groups[2]);
    }

    #[test]
    fn a_batch_that_shares_its_bind_group_gives_the_same_results() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let src = device.create_typed_buffer("src", &[1u32, 2, 3], GpuBufferAccess::ReadWrite);
        let dst = device.create_typed_buffer("dst", &[0u32; 3], GpuBufferAccess::ReadWrite);
        let other = device.create_typed_buffer("other", &[0u32; 3], GpuBufferAccess::ReadWrite);
        let first = [
            BufferBinding::read_only(&src),
            BufferBinding::read_write(&dst),
        ];
        let second = [
            BufferBinding::read_only(&dst),
            BufferBinding::read_write(&other),
        ];
        let call = BufferDispatch::new(&ops::add_one::DESCRIPTOR, 3, &first);
        let then = BufferDispatch::new(&ops::add_one::DESCRIPTOR, 3, &second);
        device.dispatch_batch(&[call, call, call, then]).unwrap();
        assert_eq!(device.read_typed_buffer(&dst).unwrap(), [2, 3, 4]);
        assert_eq!(device.read_typed_buffer(&other).unwrap(), [3, 4, 5]);
    }

    #[test]
    fn two_kernels_over_the_same_buffers_do_not_share_a_bind_group() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let src = device.create_typed_buffer("src", &[1u32, 2, 3], GpuBufferAccess::ReadWrite);
        let dst = device.create_typed_buffer("dst", &[0u32; 3], GpuBufferAccess::ReadWrite);
        let bindings = [
            BufferBinding::read_only(&src),
            BufferBinding::read_write(&dst),
        ];
        let one = BufferDispatch::new(&ops::add_one::DESCRIPTOR, 3, &bindings);
        let two = BufferDispatch::new(&ops::add_two::DESCRIPTOR, 3, &bindings);
        let groups: Vec<_> = device
            .prepare_extent_batch(&[one, one, two])
            .unwrap()
            .into_iter()
            .map(|prepared| prepared.bind_group)
            .collect();
        assert_eq!(groups[0], groups[1]);
        assert_ne!(groups[1], groups[2]);
    }
}

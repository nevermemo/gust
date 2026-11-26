use crate::Error;
use crate::texture::TextureCounts;
use gust::capability::Capability;
use gust::{BindingKind, Extent, KernelDescriptor, ParameterDescriptor, Shape};

pub(crate) fn device_limit(
    kernel: &KernelDescriptor,
    limit: &'static str,
    requested: u64,
    maximum: u64,
) -> Error {
    Error::DeviceLimit {
        kernel: kernel.qualified_name().to_string(),
        limit,
        requested,
        maximum,
    }
}

/// Limits a pipeline for `kernel` must respect: each workgroup axis against its own limit (x, then y, then z), the
/// product of the three (a checked product: [`Error::Extent`] when it overflows) against
/// `max_compute_invocations_per_workgroup`, then the buffer counts, then the texture and sampler counts
/// (`max_sampled_textures_per_shader_stage`, `max_storage_textures_per_shader_stage` and
/// `max_samplers_per_shader_stage`; the texture dimension is `texture::check_texture_dimensions`'s).
pub(crate) fn check_kernel_limits(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    storage_buffers: usize,
    uniform_buffers: usize,
    textures: TextureCounts,
) -> Result<(), Error> {
    // A `pixel` kernel has no workgroup: the three workgroup limits are a compute stage's.
    let compute = kernel.shape != Shape::Pixel;
    let size = kernel.workgroup_size.map(u64::from);
    if compute {
        for (limit, requested, maximum) in [
            (
                "max_compute_workgroup_size_x",
                size[0],
                u64::from(limits.max_compute_workgroup_size_x),
            ),
            (
                "max_compute_workgroup_size_y",
                size[1],
                u64::from(limits.max_compute_workgroup_size_y),
            ),
            (
                "max_compute_workgroup_size_z",
                size[2],
                u64::from(limits.max_compute_workgroup_size_z),
            ),
        ] {
            if requested > maximum {
                return Err(device_limit(kernel, limit, requested, maximum));
            }
        }
    }
    let invocations = if compute {
        size[0]
            .checked_mul(size[1])
            .and_then(|plane| plane.checked_mul(size[2]))
            .ok_or(Error::Extent {
                extent: size,
                reason: "the workgroup size multiplies to more invocations than a u64 holds",
            })?
    } else {
        0
    };
    for (limit, requested, maximum) in [
        (
            "max_compute_invocations_per_workgroup",
            invocations,
            u64::from(limits.max_compute_invocations_per_workgroup),
        ),
        (
            "max_storage_buffers_per_shader_stage",
            storage_buffers as u64,
            u64::from(limits.max_storage_buffers_per_shader_stage),
        ),
        (
            "max_uniform_buffers_per_shader_stage",
            uniform_buffers as u64,
            u64::from(limits.max_uniform_buffers_per_shader_stage),
        ),
        (
            "max_sampled_textures_per_shader_stage",
            textures.sampled as u64,
            u64::from(limits.max_sampled_textures_per_shader_stage),
        ),
        (
            "max_storage_textures_per_shader_stage",
            textures.storage as u64,
            u64::from(limits.max_storage_textures_per_shader_stage),
        ),
        (
            "max_samplers_per_shader_stage",
            textures.samplers as u64,
            u64::from(limits.max_samplers_per_shader_stage),
        ),
    ] {
        if requested > maximum {
            return Err(device_limit(kernel, limit, requested, maximum));
        }
    }
    Ok(())
}

/// What a kernel's `requires` asks of a device (M8, D142), checked beside [`check_kernel_limits`]: every capability in
/// `kernel.requires` is one the device was opened with (`capabilities`), else [`Error::MissingCapability`] naming the
/// kernel and `adapter`; and a kernel that requires subgroups has a workgroup of a whole number of the largest subgroups
/// wgpu reports for the adapter (`subgroup_sizes`), else [`Error::SubgroupSize`]. Only the largest size is checked, and the
/// macro already requires 64 invocations at a time (E0751): the lane count of one dispatch is the shader's to read.
///
/// "Full subgroups" rests on driver behaviour and S3's measurements, not on a guarantee of wgpu: wgpu 30.0.1 allows a
/// varying subgroup size and never sets `REQUIRE_FULL_SUBGROUPS` (D140). The product of the workgroup's axes is checked,
/// on the ground that invocations are packed into subgroups in linear order.
pub(crate) fn check_kernel_capabilities(
    capabilities: &[Capability],
    subgroup_sizes: Option<(u32, u32)>,
    adapter: &str,
    kernel: &KernelDescriptor,
) -> Result<(), Error> {
    check_kernel_requires(capabilities, adapter, kernel)?;
    if kernel.requires.contains(&Capability::Subgroups)
        && let Some((_, maximum)) = subgroup_sizes
        && maximum != 0
    {
        let invocations = kernel.workgroup_size.iter().fold(1_u64, |product, &axis| {
            product.saturating_mul(u64::from(axis))
        });
        if invocations % u64::from(maximum) != 0 {
            return Err(Error::SubgroupSize {
                kernel: kernel.qualified_name().to_string(),
                invocations,
                subgroup_max: maximum,
            });
        }
    }
    Ok(())
}

/// The first half of [`check_kernel_capabilities`]: every capability in `kernel.requires` is one the device was opened
/// with. It is a fact of the descriptor against the device, not of the work, so it also judges an empty launch (as
/// [`Error::KernelTarget`] does); the subgroup-size half needs a launch.
pub(crate) fn check_kernel_requires(
    capabilities: &[Capability],
    adapter: &str,
    kernel: &KernelDescriptor,
) -> Result<(), Error> {
    for &required in kernel.requires {
        if !capabilities.contains(&required) {
            return Err(Error::MissingCapability {
                capability: required,
                adapter: adapter.to_owned(),
                kernel: Some(kernel.qualified_name().to_string()),
            });
        }
    }
    Ok(())
}

/// Every binding index must be below `max_bindings_per_bind_group`: wgpu rejects the bind group layout
/// (an uncaptured-error panic) when an index is equal to or above it, however few bindings there are.
/// `requested` is the highest index plus one, so it compares with the maximum like the other limits.
pub(crate) fn check_binding_indices(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    resources: &[&ParameterDescriptor],
) -> Result<(), Error> {
    let maximum = u64::from(limits.max_bindings_per_bind_group);
    let requested = resources
        .iter()
        .filter_map(|parameter| parameter.binding)
        .map(|binding| u64::from(binding.binding) + 1)
        .max()
        .unwrap_or(0);
    if requested > maximum {
        return Err(device_limit(
            kernel,
            "max_bindings_per_bind_group",
            requested,
            maximum,
        ));
    }
    Ok(())
}

/// The limit the workgroups along y name in an [`Error::DeviceLimit`]; the one along x keeps the unsuffixed name of the
/// one-dimensional dispatch, the one along z is `max_compute_workgroups_per_dimension (z)`.
const WORKGROUPS_Y_LIMIT: &str = "max_compute_workgroups_per_dimension (y)";
const WORKGROUPS_Z_LIMIT: &str = "max_compute_workgroups_per_dimension (z)";

/// The name of the invocation cap in an [`Error::DeviceLimit`]; the CPU twin reports the same name
/// (`gust::cpu::Error::DeviceLimit`).
pub(crate) const INVOCATION_CAP_LIMIT: &str = "global_invocation_id";

/// Workgroups along x, y and z needed for `extent`, each within `max_compute_workgroups_per_dimension` (the error
/// names the axis: `max_compute_workgroups_per_dimension` for x, with ` (y)` and ` (z)` for the others), and the launch
/// within the invocation cap: every axis rounded up to whole workgroups, multiplied, at most 2^32
/// ([`Extent::MAX_INVOCATIONS`]), the rule of the CPU twin. A kernel without a zero workgroup axis
/// (`validate_kernel`) and a non-empty extent are assumed.
pub(crate) fn workgroup_counts(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    extent: Extent,
) -> Result<[u32; 3], Error> {
    let size = kernel.workgroup_size;
    let maximum = limits.max_compute_workgroups_per_dimension;
    let counts = [
        extent.x.div_ceil(size[0]),
        extent.y.div_ceil(size[1]),
        extent.z.div_ceil(size[2]),
    ];
    for (limit, count) in [
        ("max_compute_workgroups_per_dimension", counts[0]),
        (WORKGROUPS_Y_LIMIT, counts[1]),
        (WORKGROUPS_Z_LIMIT, counts[2]),
    ] {
        if count > maximum {
            return Err(device_limit(
                kernel,
                limit,
                u64::from(count),
                u64::from(maximum),
            ));
        }
    }
    // `rounded_total` is in `u128`: three rounded axes are at most 2^96, so it cannot overflow. The error carries a
    // `u64`, so a launch beyond it saturates (as the CPU twin's does).
    let invocations = extent
        .rounded_total(size)
        .expect("a workgroup size without a zero axis");
    if invocations > u128::from(Extent::MAX_INVOCATIONS) {
        return Err(device_limit(
            kernel,
            INVOCATION_CAP_LIMIT,
            u64::try_from(invocations).unwrap_or(u64::MAX),
            Extent::MAX_INVOCATIONS,
        ));
    }
    Ok(counts)
}

/// One binding of `bytes` bytes against `max_storage_buffer_binding_size`, or `max_uniform_buffer_binding_size` for a
/// uniform.
pub(crate) fn check_binding_size(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    bytes: u64,
    kind: BindingKind,
) -> Result<(), Error> {
    // A texture has no byte size in a binding: its limit is the dimension (`texture::check_texture_dimensions`).
    if kind == BindingKind::Texture {
        return Ok(());
    }
    let (limit, maximum) = if kind == BindingKind::Uniform {
        (
            "max_uniform_buffer_binding_size",
            limits.max_uniform_buffer_binding_size,
        )
    } else {
        (
            "max_storage_buffer_binding_size",
            limits.max_storage_buffer_binding_size,
        )
    };
    if bytes > maximum {
        return Err(device_limit(kernel, limit, bytes, maximum));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::TextureCounts;
    use crate::{Error, validate_kernel};
    use gust::capability::Capability;
    use gust::target::Target;
    use gust::{
        Access, BindingKind, Extent, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind,
        ResourceBinding, ResourceLayout,
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

    /// A kernel that requires subgroups, with a workgroup of `size` invocations along x.
    fn subgroup_kernel(size: u32) -> KernelDescriptor {
        DESCRIPTOR
            .with_workgroup_size([size, 1, 1])
            .with_requires(&[Capability::Subgroups])
    }

    #[test]
    fn a_requires_capability_the_device_was_not_opened_with_is_missing_capability() {
        let kernel = subgroup_kernel(64);
        // A device opened without the capability refuses the kernel, naming the kernel and the adapter.
        let error = check_kernel_capabilities(&[], Some((32, 32)), "Fake Adapter", &kernel)
            .expect_err("a subgroup kernel on a device without the capability");
        match &error {
            Error::MissingCapability {
                capability,
                adapter,
                kernel: Some(name),
            } => {
                assert_eq!(*capability, Capability::Subgroups);
                assert_eq!(adapter, "Fake Adapter");
                assert_eq!(name, &kernel.qualified_name().to_string());
            }
            other => panic!("expected MissingCapability with a kernel, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` requires the `subgroups` capability, which this device (adapter `Fake Adapter`) was not opened or adopted with; pass the capability to `DeviceOptions::with_capabilities` or to `adopt`"
        );
        // With the capability the same kernel passes, and a kernel with no `requires` never needs one.
        assert!(
            check_kernel_capabilities(
                &[Capability::Subgroups],
                Some((32, 32)),
                "Fake Adapter",
                &kernel
            )
            .is_ok()
        );
        assert!(check_kernel_capabilities(&[], None, "Fake Adapter", &DESCRIPTOR).is_ok());
    }

    #[test]
    fn a_subgroup_kernel_needs_whole_subgroups_of_the_adapters_largest_size() {
        let have = [Capability::Subgroups];
        // 64 invocations are whole subgroups of 32 and of 64, and so are 128; 96 are not whole subgroups of 64.
        for (size, range, whole) in [
            (64, (32, 32), true),
            (64, (32, 64), true),
            (128, (32, 64), true),
            (96, (32, 32), true),
            (96, (32, 64), false),
            (48, (32, 32), false),
            (64, (8, 8), true),
            (32, (32, 64), false),
        ] {
            let result = check_kernel_capabilities(&have, Some(range), "A", &subgroup_kernel(size));
            assert_eq!(result.is_ok(), whole, "{size} invocations, range {range:?}");
        }
        let error = check_kernel_capabilities(&have, Some((32, 64)), "A", &subgroup_kernel(96))
            .unwrap_err();
        match &error {
            Error::SubgroupSize {
                kernel,
                invocations,
                subgroup_max,
            } => {
                assert_eq!(kernel, "test::kernel");
                assert_eq!((*invocations, *subgroup_max), (96, 64));
            }
            other => panic!("expected SubgroupSize, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` runs 96 invocations per workgroup, which is not a multiple of 64, the largest subgroup size wgpu reports for this adapter; a subgroup kernel needs full subgroups"
        );
        // The invocation count is the product of the axes, not the x size.
        let tile = DESCRIPTOR
            .with_workgroup_size([8, 8, 1])
            .with_requires(&[Capability::Subgroups]);
        assert!(check_kernel_capabilities(&have, Some((32, 64)), "A", &tile).is_ok());
        // A kernel without `requires` is never judged by subgroup sizes, and an adapter without a range cannot judge.
        assert!(
            check_kernel_capabilities(
                &have,
                Some((32, 64)),
                "A",
                &DESCRIPTOR.with_workgroup_size([48, 1, 1])
            )
            .is_ok()
        );
        assert!(check_kernel_capabilities(&have, None, "A", &subgroup_kernel(48)).is_ok());
    }

    #[test]
    fn a_kernel_built_for_the_cpu_is_refused_before_anything_else() {
        assert!(validate_kernel(&DESCRIPTOR).is_ok());
        let cpu = DESCRIPTOR.with_target(Target::Cpu);
        let error = validate_kernel(&cpu).expect_err("a CPU-target kernel on a GPU device");
        match &error {
            Error::KernelTarget { kernel, target } => {
                assert_eq!(kernel, "test::kernel");
                assert_eq!(*target, Target::Cpu);
            }
            other => panic!("expected KernelTarget, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` is built for the `cpu` target and cannot run on a GPU device; run it on a `CpuDevice`, or route its tag to `gpu`"
        );
        // First: a CPU kernel with an empty source is still the target's error, not the source's.
        let empty = cpu.with_slang_source("");
        assert!(matches!(
            validate_kernel(&empty),
            Err(Error::KernelTarget { .. })
        ));
    }

    #[test]
    fn uniform_limits_are_checked_under_their_own_names() {
        let limits = wgpu::Limits {
            max_uniform_buffer_binding_size: 64,
            max_uniform_buffers_per_shader_stage: 2,
            ..wgpu::Limits::default()
        };
        assert!(check_binding_size(&limits, &DESCRIPTOR, 64, BindingKind::Uniform).is_ok());
        assert!(matches!(
            check_binding_size(&limits, &DESCRIPTOR, 65, BindingKind::Uniform),
            Err(Error::DeviceLimit {
                limit: "max_uniform_buffer_binding_size",
                requested: 65,
                maximum: 64,
                ..
            })
        ));
        // The same bytes as a storage binding are within the (much larger) storage limit.
        assert!(check_binding_size(&limits, &DESCRIPTOR, 65, BindingKind::Storage).is_ok());
        assert!(check_kernel_limits(&limits, &DESCRIPTOR, 1, 2, TextureCounts::NONE).is_ok());
        assert!(matches!(
            check_kernel_limits(&limits, &DESCRIPTOR, 1, 3, TextureCounts::NONE),
            Err(Error::DeviceLimit {
                limit: "max_uniform_buffers_per_shader_stage",
                requested: 3,
                maximum: 2,
                ..
            })
        ));
        // A uniform does not count against the storage buffer limit.
        assert!(check_kernel_limits(&limits, &DESCRIPTOR, 8, 2, TextureCounts::NONE).is_ok());
    }

    #[test]
    fn binding_size_is_checked_against_the_stored_limit() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 16,
            ..wgpu::Limits::default()
        };
        assert!(check_binding_size(&limits, &DESCRIPTOR, 16, BindingKind::Storage).is_ok());
        assert!(matches!(
            check_binding_size(&limits, &DESCRIPTOR, 17, BindingKind::Storage),
            Err(Error::DeviceLimit {
                kernel,
                limit: "max_storage_buffer_binding_size",
                requested: 17,
                maximum: 16,
            }) if kernel == "test::kernel"
        ));
    }

    #[test]
    fn workgroup_count_rounds_up_and_stops_at_the_limit() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..wgpu::Limits::default()
        };
        let count =
            |elements: u32| workgroup_counts(&limits, &DESCRIPTOR, Extent::linear(elements));
        // An empty extent never reaches the count (the dispatch is skipped first); its x count is 0.
        assert_eq!(count(0).unwrap(), [0, 1, 1]);
        assert_eq!(count(1).unwrap(), [1, 1, 1]);
        assert_eq!(count(192).unwrap(), [3, 1, 1]);
        assert!(matches!(
            count(193),
            Err(Error::DeviceLimit {
                limit: "max_compute_workgroups_per_dimension",
                requested: 4,
                maximum: 3,
                ..
            })
        ));
    }

    /// A 16 x 16 kernel for the multi-dimensional unit tests below.
    const TILE: KernelDescriptor = DESCRIPTOR.with_workgroup_size([16, 16, 1]);

    #[test]
    fn workgroup_counts_round_every_axis_and_name_the_axis_that_is_over_the_limit() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..wgpu::Limits::default()
        };
        let counts = |x, y, z| workgroup_counts(&limits, &TILE, Extent::new(x, y, z));
        assert_eq!(counts(15, 17, 1).unwrap(), [1, 2, 1]);
        assert_eq!(counts(48, 48, 3).unwrap(), [3, 3, 3]);
        let over = |result: Result<[u32; 3], Error>| match result {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => (limit, requested, maximum),
            other => panic!("expected a DeviceLimit, got {other:?}"),
        };
        assert_eq!(
            over(counts(49, 16, 1)),
            ("max_compute_workgroups_per_dimension", 4, 3)
        );
        assert_eq!(
            over(counts(16, 49, 1)),
            ("max_compute_workgroups_per_dimension (y)", 4, 3)
        );
        assert_eq!(
            over(counts(16, 16, 4)),
            ("max_compute_workgroups_per_dimension (z)", 4, 3)
        );
        // The first axis over the limit is the one named: x before y before z.
        assert_eq!(
            over(counts(49, 49, 4)),
            ("max_compute_workgroups_per_dimension", 4, 3)
        );
    }

    #[test]
    fn the_invocation_cap_is_the_rounded_product_at_most_2_pow_32() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: u32::MAX,
            ..wgpu::Limits::default()
        };
        let counts = |x, y, z| workgroup_counts(&limits, &TILE, Extent::new(x, y, z));
        let cap = |result: Result<[u32; 3], Error>| match result {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => Some((limit, requested, maximum)),
            Ok(_) => None,
            other => panic!("expected a DeviceLimit or Ok, got {other:?}"),
        };
        // Exactly 2^32 after rounding is accepted (not "below"), even when the raw product is smaller.
        assert_eq!(counts(65_536, 65_536, 1).unwrap(), [4096, 4096, 1]);
        assert_eq!(counts(65_536, 65_535, 1).unwrap(), [4096, 4096, 1]);
        // A raw product below 2^32 (65537 * 65535) that rounds up above it is refused.
        assert_eq!(
            cap(counts(65_537, 65_535, 1)),
            Some(("global_invocation_id", 65_552 * 65_536, 1 << 32))
        );
        assert_eq!(
            cap(counts(65_537, 65_536, 1)),
            Some(("global_invocation_id", 65_552 * 65_536, 1 << 32))
        );
        // The rounded product is far beyond a u64 here, and saturates in the error.
        assert_eq!(
            cap(counts(u32::MAX, u32::MAX, u32::MAX)),
            Some(("global_invocation_id", u64::MAX, 1 << 32))
        );
    }

    #[test]
    fn kernel_limits_name_the_first_exceeded_limit() {
        let mut limits = wgpu::Limits::default();
        assert!(check_kernel_limits(&limits, &DESCRIPTOR, 8, 0, TextureCounts::NONE).is_ok());
        assert!(matches!(
            check_kernel_limits(&limits, &DESCRIPTOR, 9, 0, TextureCounts::NONE),
            Err(Error::DeviceLimit {
                limit: "max_storage_buffers_per_shader_stage",
                requested: 9,
                maximum: 8,
                ..
            })
        ));
        limits.max_compute_invocations_per_workgroup = 32;
        assert!(matches!(
            check_kernel_limits(&limits, &DESCRIPTOR, 1, 0, TextureCounts::NONE),
            Err(Error::DeviceLimit {
                limit: "max_compute_invocations_per_workgroup",
                requested: 64,
                maximum: 32,
                ..
            })
        ));
        limits.max_compute_workgroup_size_x = 32;
        assert!(matches!(
            check_kernel_limits(&limits, &DESCRIPTOR, 1, 0, TextureCounts::NONE),
            Err(Error::DeviceLimit {
                limit: "max_compute_workgroup_size_x",
                requested: 64,
                maximum: 32,
                ..
            })
        ));
    }

    #[test]
    fn storage_buffers_are_checked_against_a_raised_limit() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 16,
            ..wgpu::Limits::default()
        };
        for count in [9, 10, 16] {
            assert!(
                check_kernel_limits(&limits, &DESCRIPTOR, count, 0, TextureCounts::NONE).is_ok(),
                "{count} storage buffers against 16"
            );
        }
        let error = check_kernel_limits(&limits, &DESCRIPTOR, 17, 0, TextureCounts::NONE)
            .expect_err("17 against 16");
        assert!(matches!(
            error,
            Error::DeviceLimit {
                limit: "max_storage_buffers_per_shader_stage",
                requested: 17,
                maximum: 16,
                ..
            }
        ));
        // Against the default 8, 9 is still refused.
        assert!(matches!(
            check_kernel_limits(
                &wgpu::Limits::default(),
                &DESCRIPTOR,
                9,
                0,
                TextureCounts::NONE
            ),
            Err(Error::DeviceLimit {
                requested: 9,
                maximum: 8,
                ..
            })
        ));
    }

    #[test]
    fn kernel_limits_check_the_y_and_z_sizes_and_the_checked_product() {
        let limits = wgpu::Limits::default();
        let kernel = |size| DESCRIPTOR.with_workgroup_size(size);
        let check = |limits: &wgpu::Limits, size| {
            check_kernel_limits(limits, &kernel(size), 1, 0, TextureCounts::NONE)
        };
        let named = |result: Result<(), Error>| match result {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => (limit, requested, maximum),
            other => panic!("expected a DeviceLimit, got {other:?}"),
        };
        assert!(check(&limits, [16, 16, 1]).is_ok());
        assert!(check(&limits, [1, 1, 64]).is_ok());
        assert_eq!(
            named(check(&limits, [1, 257, 1])),
            ("max_compute_workgroup_size_y", 257, 256)
        );
        assert_eq!(
            named(check(&limits, [1, 1, 65])),
            ("max_compute_workgroup_size_z", 65, 64)
        );
        // 16 * 16 * 2 = 512 invocations: every axis is within its limit and the product is not.
        assert_eq!(
            named(check(&limits, [16, 16, 2])),
            ("max_compute_invocations_per_workgroup", 512, 256)
        );
        // The axes are checked in the order x, y, z, before the product.
        assert_eq!(
            named(check(&limits, [257, 257, 65])),
            ("max_compute_workgroup_size_x", 257, 256)
        );
        assert_eq!(
            named(check(&limits, [1, 257, 65])),
            ("max_compute_workgroup_size_y", 257, 256)
        );
        // A product beyond a u64 is an `Error::Extent`, not a wrapped number.
        let wide = wgpu::Limits {
            max_compute_workgroup_size_x: u32::MAX,
            max_compute_workgroup_size_y: u32::MAX,
            max_compute_workgroup_size_z: u32::MAX,
            ..limits
        };
        assert!(matches!(
            check(&wide, [u32::MAX, u32::MAX, u32::MAX]),
            Err(Error::Extent { extent, .. }) if extent == [u64::from(u32::MAX); 3]
        ));
    }

    #[test]
    fn device_limit_error_text_names_kernel_limit_and_remedy() {
        let error = device_limit(
            &DESCRIPTOR,
            "max_compute_workgroups_per_dimension",
            65536,
            65535,
        );
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` needs 65536 for `max_compute_workgroups_per_dimension`, the limit is 65535; \
             split the work into several dispatches, or spread it over y and z with a multi-dimensional extent"
        );
        let error = device_limit(&DESCRIPTOR, "max_storage_buffer_binding_size", 20, 16);
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` needs 20 for `max_storage_buffer_binding_size`, the limit is 16"
        );
        // The y and z counts name their axis and get a remedy of their own; neither mentions a roadmap.
        for axis in ["y", "z"] {
            let limit: &'static str = Box::leak(
                format!("max_compute_workgroups_per_dimension ({axis})").into_boxed_str(),
            );
            let text = device_limit(&DESCRIPTOR, limit, 4, 3).to_string();
            assert_eq!(
                text,
                format!(
                    "kernel `test::kernel` needs 4 for `{limit}`, the limit is 3; split the work into several \
                     dispatches, or use fewer invocations along that axis"
                )
            );
            assert!(!text.contains("roadmap"), "{text}");
        }
        // A workgroup size with a zero axis says so; it does not claim the backend is one-dimensional.
        assert_eq!(
            Error::UnsupportedWorkgroupSize([16, 0, 1]).to_string(),
            "workgroup size [16, 0, 1] has a zero axis; every axis must be at least 1"
        );
    }
}

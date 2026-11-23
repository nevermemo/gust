use crate::cache::{CachedKernel, Pipeline};
use crate::{Error, INVOCATION_CAP_LIMIT, device_limit, render};
use gust::{Extent, KernelDescriptor, ParameterDescriptor, Shape};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// What one prepared dispatch launches: compute workgroups for a `grid` kernel, a full-screen pass over `elements`
/// fragments for a `pixel` kernel (slice mx.p1b).
#[derive(Clone, Copy)]
pub(crate) enum Launch {
    /// The workgroups along x, y and z; `[count, 1, 1]` for a one-dimensional dispatch.
    Compute { workgroups: [u32; 3] },
    /// The element count `n` of a `pixel` call: `render::PixelPlan::new(n)` says how it is drawn.
    Pixel { elements: u32 },
}

pub(crate) struct PreparedDispatch {
    pub(crate) cached_kernel: Arc<CachedKernel>,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) launch: Launch,
    /// The pieces of a split `grid` launch (m11.g4b), in dispatch order: empty for a launch that runs whole. A split
    /// launch's `bind_group` is its first piece's and `launch` holds the workgroups of that piece.
    pub(crate) pieces: Vec<SplitPiece>,
}

/// The bind group of one dispatch node of a staged graph, kept from one execution of the graph to the next.
///
/// A bind group is fixed by the pipeline's layout and the resources bound. A node's bindings are borrowed by the graph
/// for its whole life, so no buffer or texture of it can be replaced or dropped while the slot exists: the resources
/// are the same on every execution, and what can differ is the pipeline (another device, or a cache that compiled the
/// kernel again). The slot therefore keeps the pipeline entry the group was built for and hands the group out only
/// for that very entry (`Arc::ptr_eq`); any other entry builds a group and replaces the stored one.
#[derive(Default)]
pub(crate) struct BindGroupSlot {
    kept: Mutex<Option<(Arc<CachedKernel>, wgpu::BindGroup)>>,
}

impl BindGroupSlot {
    fn lock(&self) -> MutexGuard<'_, Option<(Arc<CachedKernel>, wgpu::BindGroup)>> {
        // Nothing runs under the lock that can leave the pair half-written.
        self.kept.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The kept group, if it was built for `kernel`.
    pub(crate) fn reuse(&self, kernel: &Arc<CachedKernel>) -> Option<wgpu::BindGroup> {
        self.lock()
            .as_ref()
            .filter(|(built_for, _)| Arc::ptr_eq(built_for, kernel))
            .map(|(_, group)| group.clone())
    }

    /// Keep `group`, built for `kernel`, for the next execution.
    pub(crate) fn keep(&self, kernel: &Arc<CachedKernel>, group: &wgpu::BindGroup) {
        *self.lock() = Some((Arc::clone(kernel), group.clone()));
    }

    /// The kept group, for a test that compares it across executions.
    #[cfg(test)]
    pub(crate) fn kept_group(&self) -> Option<wgpu::BindGroup> {
        self.lock().as_ref().map(|(_, group)| group.clone())
    }
}

/// One piece of a split launch: a bind group whose hidden launch-base uniform holds the piece's base, and the
/// workgroups of the piece along x, y and z.
pub(crate) struct SplitPiece {
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) workgroups: [u32; 3],
}

/// The query slots a timed pass writes: the first pass of a batch writes `begin`, the last writes `end`.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) struct PassTimestamps<'a> {
    pub(crate) query_set: &'a wgpu::QuerySet,
    pub(crate) begin: Option<u32>,
    pub(crate) end: Option<u32>,
}

impl PreparedDispatch {
    /// The workgroups along x, which is all the staged graph reports (it runs one-dimensional dispatches only); 0 for a
    /// `pixel` launch, which has none. For a split launch (which the graph never prepares) it is the first piece's.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn workgroup_count(&self) -> u32 {
        match self.launch {
            Launch::Compute { workgroups } => workgroups[0],
            Launch::Pixel { .. } => 0,
        }
    }

    /// Encode this dispatch into `encoder` as one pass of its own: a compute pass for `grid`, a render pass for `pixel`
    /// (a pass per call, because one call's output may be the next call's input and a pass is one usage scope).
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        timestamps: Option<PassTimestamps<'_>>,
    ) -> Result<(), Error> {
        match (&self.cached_kernel.pipeline, self.launch) {
            (Pipeline::Compute(pipeline), Launch::Compute { workgroups }) => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: timestamps.map(|stamps| wgpu::ComputePassTimestampWrites {
                        query_set: stamps.query_set,
                        beginning_of_pass_write_index: stamps.begin,
                        end_of_pass_write_index: stamps.end,
                    }),
                });
                pass.set_pipeline(pipeline);
                if self.pieces.is_empty() {
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                } else {
                    // The pieces of a split launch run in order inside this one pass: x innermost, then y, then z.
                    for piece in &self.pieces {
                        pass.set_bind_group(0, &piece.bind_group, &[]);
                        pass.dispatch_workgroups(
                            piece.workgroups[0],
                            piece.workgroups[1],
                            piece.workgroups[2],
                        );
                    }
                }
                Ok(())
            }
            (Pipeline::Render(pipeline), Launch::Pixel { elements }) => render::encode_pixel_pass(
                device,
                encoder,
                pipeline,
                &self.bind_group,
                elements,
                label,
                timestamps,
            ),
            _ => unreachable!("a kernel's shape fixes both its pipeline and its launch"),
        }
    }
}

/// The most pieces one launch may be cut into (m11.g4b): each piece is one `dispatch_workgroups` with a bind group of
/// its own, so a launch that would need more (a tiny `max_compute_workgroups_per_dimension` against a large extent) is
/// refused as [`Error::DeviceLimit`] with the limit named `split dispatches`.
pub const MAX_SPLIT_DISPATCHES: usize = 4096;

/// The size of the launch-base uniform of a split variant: four `u32` (`x`, `y`, `z`, padding).
pub(crate) const LAUNCH_BASE_BYTES: u64 = 16;

/// What the plain checks left for a launch: run whole with these workgroups, or as pieces.
pub(crate) enum LaunchPlan {
    Whole([u32; 3]),
    Split(SplitLaunch),
}

/// A launch cut into pieces: the kernel's split variant, its validated resources (the kernel's, then the hidden
/// uniform) and the pieces in dispatch order.
pub(crate) struct SplitLaunch {
    pub(crate) variant: &'static KernelDescriptor,
    pub(crate) resources: Vec<&'static ParameterDescriptor>,
    pub(crate) pieces: Vec<PiecePlan>,
}

/// One piece of a split launch: `base` is the id of its first invocation (a whole number of workgroups per axis, so
/// local and group ids are the unsplit launch's), `workgroups` how many workgroups it runs along each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PiecePlan {
    pub(crate) base: [u32; 3],
    pub(crate) workgroups: [u32; 3],
}

/// Cut a launch over `extent` into pieces of at most `max_compute_workgroups_per_dimension` workgroups per axis, in
/// dispatch order (x innermost, then y, then z). The invocation cap comes first, the rule of [`workgroup_counts`]
/// and the CPU twin ([`INVOCATION_CAP_LIMIT`]); then more than [`MAX_SPLIT_DISPATCHES`] pieces is refused. A kernel
/// without a zero workgroup axis and a non-empty extent are assumed.
pub(crate) fn split_pieces(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    extent: Extent,
) -> Result<Vec<PiecePlan>, Error> {
    let size = kernel.workgroup_size;
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
    let maximum = limits.max_compute_workgroups_per_dimension.max(1);
    let counts = [
        extent.x.div_ceil(size[0]),
        extent.y.div_ceil(size[1]),
        extent.z.div_ceil(size[2]),
    ];
    let per_axis = counts.map(|count| u64::from(count.div_ceil(maximum)));
    let total = per_axis[0] * per_axis[1] * per_axis[2];
    if total > MAX_SPLIT_DISPATCHES as u64 {
        return Err(device_limit(
            kernel,
            "split dispatches",
            total,
            MAX_SPLIT_DISPATCHES as u64,
        ));
    }
    let mut pieces = Vec::with_capacity(total as usize);
    for z in 0..counts[2].div_ceil(maximum) {
        for y in 0..counts[1].div_ceil(maximum) {
            for x in 0..counts[0].div_ceil(maximum) {
                let first = [x * maximum, y * maximum, z * maximum];
                pieces.push(PiecePlan {
                    // The base is below the rounded axis, which the cap keeps within `u32`.
                    base: [first[0] * size[0], first[1] * size[1], first[2] * size[2]],
                    workgroups: [
                        maximum.min(counts[0] - first[0]),
                        maximum.min(counts[1] - first[1]),
                        maximum.min(counts[2] - first[2]),
                    ],
                });
            }
        }
    }
    Ok(pieces)
}

/// Whether `variant` is the split variant the `#[gpu]` macro makes for `kernel`: the same kernel (module, name, entry
/// point, shape, workgroup size, `requires`, target), the kernel's parameters and then one more, the hidden launch-base
/// uniform at a binding past every binding of the kernel, no split of its own, and no artifact or WGSL artifacts only
/// (a precompiled variant is the same kernel text under a key of its own, D209; an artifact of another format is not
/// one this host runs). `KernelDescriptor::with_split` is public for the macro only; this check keeps a hand-set `split`
/// that is not shaped like the macro's from being run. It cannot read the Slang: the run-time compiler's reflection, or
/// the verification of the artifact (`verify_artifact`, which holds the stored reflection to the variant's parameters),
/// is what a text that lacks the uniform fails against when the variant is compiled.
pub(crate) fn split_variant_matches(kernel: &KernelDescriptor, variant: &KernelDescriptor) -> bool {
    let Some((hidden, own)) = variant.parameters.split_last() else {
        return false;
    };
    let past_every_binding = kernel
        .parameters
        .iter()
        .filter_map(|parameter| parameter.binding)
        .all(|binding| {
            hidden.binding.is_some_and(|hidden| {
                hidden.group == binding.group && hidden.binding > binding.binding
            })
        });
    kernel.shape == Shape::Grid
        && variant.shape == Shape::Grid
        && variant.module == kernel.module
        && variant.name == kernel.name
        && variant.entry_point == kernel.entry_point
        && variant.workgroup_size == kernel.workgroup_size
        && variant.requires == kernel.requires
        && variant.target == kernel.target
        && variant.split.is_none()
        && variant
            .artifacts
            .iter()
            .all(|artifact| artifact.format == gust::Format::Wgsl)
        && own == kernel.parameters
        && past_every_binding
        && hidden.binding.is_some_and(|binding| {
            *hidden == KernelDescriptor::__gust_launch_base_parameter(binding.binding)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use gust::{
        Access, Extent, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
        Shape,
    };

    const WGSL_ARTIFACT: gust::Artifact =
        gust::Artifact::new(gust::Format::Wgsl, "0123456789abcdef", b"", "");
    const SPIRV_ARTIFACT: gust::Artifact =
        gust::Artifact::new(gust::Format::Spirv, "0123456789abcdef", b"", "");

    // ---- m11.g4b: the planner and the check of a split variant, on pure data (no device). ----

    fn limits_of(workgroups: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_compute_workgroups_per_dimension: workgroups,
            ..wgpu::Limits::default()
        }
    }

    fn planner_kernel(size: [u32; 3]) -> KernelDescriptor {
        KernelDescriptor::new("m", "k", "m_k", size, &[], "")
    }

    /// Every piece is inside the rounded extent, within the limit, non-empty and aligned to the workgroup, the volumes
    /// add up to the rounded extent, and (for a modest count) no two pieces overlap.
    fn assert_tiles(pieces: &[PiecePlan], size: [u32; 3], extent: Extent, maximum: u32) {
        let rounded = [
            u64::from(extent.x.div_ceil(size[0])) * u64::from(size[0]),
            u64::from(extent.y.div_ceil(size[1])) * u64::from(size[1]),
            u64::from(extent.z.div_ceil(size[2])) * u64::from(size[2]),
        ];
        let mut volume = 0u128;
        for piece in pieces {
            for axis in 0..3 {
                assert!(piece.workgroups[axis] >= 1 && piece.workgroups[axis] <= maximum);
                assert_eq!(piece.base[axis] % size[axis], 0);
                let end = u64::from(piece.base[axis])
                    + u64::from(piece.workgroups[axis]) * u64::from(size[axis]);
                assert!(end <= rounded[axis], "{piece:?} past the rounded extent");
            }
            volume += piece
                .workgroups
                .iter()
                .map(|&w| u128::from(w))
                .product::<u128>();
        }
        let expected = (0..3)
            .map(|axis| u128::from(rounded[axis]) / u128::from(size[axis]))
            .product::<u128>();
        assert_eq!(volume, expected, "the pieces add up to the rounded extent");
        if pieces.len() <= 64 {
            for (i, a) in pieces.iter().enumerate() {
                for b in &pieces[i + 1..] {
                    let overlap = (0..3).all(|axis| {
                        let a_end = a.base[axis] + a.workgroups[axis] * size[axis];
                        let b_end = b.base[axis] + b.workgroups[axis] * size[axis];
                        a.base[axis] < b_end && b.base[axis] < a_end
                    });
                    assert!(!overlap, "{a:?} overlaps {b:?}");
                }
            }
        }
        // x innermost, then y, then z: the bases are strictly increasing in (z, y, x) order.
        assert!(
            pieces
                .windows(2)
                .all(|w| (w[0].base[2], w[0].base[1], w[0].base[0])
                    < (w[1].base[2], w[1].base[1], w[1].base[0]))
        );
    }

    #[test]
    fn the_planner_cuts_a_near_2_pow_32_launch_into_257_pieces() {
        let kernel = planner_kernel([256, 1, 1]);
        let extent = Extent::new(u32::MAX, 1, 1);
        let pieces = split_pieces(&limits_of(65_535), &kernel, extent).unwrap();
        assert_eq!(pieces.len(), 257);
        for (k, piece) in pieces.iter().enumerate().take(256) {
            assert_eq!(piece.workgroups, [65_535, 1, 1]);
            assert_eq!(u64::from(piece.base[0]), k as u64 * 65_535 * 256);
        }
        // 2^24 workgroups = 256 * 65535 + 256.
        assert_eq!(pieces[256].workgroups, [256, 1, 1]);
        assert_eq!(u64::from(pieces[256].base[0]), 256 * 65_535 * 256);
        assert_tiles(&pieces, [256, 1, 1], extent, 65_535);
    }

    #[test]
    fn the_planner_takes_4096_pieces_and_refuses_4097() {
        let kernel = planner_kernel([8, 1, 1]);
        let limits = limits_of(1);
        let ok = Extent::linear(8 * 4096);
        let pieces = split_pieces(&limits, &kernel, ok).unwrap();
        assert_eq!(pieces.len(), MAX_SPLIT_DISPATCHES);
        assert_eq!(pieces[4095].base, [8 * 4095, 0, 0]);
        assert_tiles(&pieces, [8, 1, 1], ok, 1);
        // 4097 pieces by x alone, and 4097 as a product over two axes (a prime count cannot be split evenly).
        for extent in [Extent::linear(8 * 4097), Extent::new(8 * 17, 241, 1)] {
            match split_pieces(&limits, &kernel, extent) {
                Err(Error::DeviceLimit {
                    limit,
                    requested,
                    maximum,
                    ..
                }) => assert_eq!(
                    (limit, requested, maximum),
                    ("split dispatches", 4097, 4096)
                ),
                other => panic!("expected the piece cap, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_last_piece_of_an_axis_can_be_one_workgroup() {
        let kernel = planner_kernel([8, 1, 1]);
        let extent = Extent::linear(8 * 9);
        let pieces = split_pieces(&limits_of(4), &kernel, extent).unwrap();
        assert_eq!(
            pieces,
            [
                PiecePlan {
                    base: [0, 0, 0],
                    workgroups: [4, 1, 1]
                },
                PiecePlan {
                    base: [32, 0, 0],
                    workgroups: [4, 1, 1]
                },
                PiecePlan {
                    base: [64, 0, 0],
                    workgroups: [1, 1, 1]
                },
            ]
        );
        // A launch that fits is one piece of its own size.
        let one = split_pieces(&limits_of(4), &kernel, Extent::linear(8 * 3)).unwrap();
        assert_eq!(
            one,
            [PiecePlan {
                base: [0; 3],
                workgroups: [3, 1, 1]
            }]
        );
    }

    #[test]
    fn the_planner_tiles_a_non_multiple_extent_on_every_axis_in_order() {
        let kernel = planner_kernel([4, 4, 4]);
        // 21 x 18 x 19 are 6 x 5 x 5 workgroups of 4: pieces of (4, 2) x (4, 1) x (4, 1), x innermost.
        let extent = Extent::new(21, 18, 19);
        let pieces = split_pieces(&limits_of(4), &kernel, extent).unwrap();
        let mut expected = Vec::new();
        for (z, nz) in [(0, 4), (16, 1)] {
            for (y, ny) in [(0, 4), (16, 1)] {
                for (x, nx) in [(0, 4), (16, 2)] {
                    expected.push(PiecePlan {
                        base: [x, y, z],
                        workgroups: [nx, ny, nz],
                    });
                }
            }
        }
        assert_eq!(pieces, expected);
        assert_tiles(&pieces, [4, 4, 4], extent, 4);
        // The same on y alone and on z alone.
        for extent in [Extent::new(4, 21, 4), Extent::new(4, 4, 21)] {
            let pieces = split_pieces(&limits_of(4), &kernel, extent).unwrap();
            assert_eq!(pieces.len(), 2);
            assert_tiles(&pieces, [4, 4, 4], extent, 4);
        }
    }

    #[test]
    fn the_planner_refuses_a_launch_past_the_invocation_cap_first() {
        let kernel = planner_kernel([8, 1, 1]);
        match split_pieces(&limits_of(1), &kernel, Extent::new(u32::MAX, 2, 1)) {
            Err(Error::DeviceLimit { limit, .. }) => assert_eq!(limit, "global_invocation_id"),
            other => panic!("expected the cap, got {other:?}"),
        }
    }

    const BASE_PARAMETER: ParameterDescriptor = ParameterDescriptor {
        name: "data",
        rust_type: "u32",
        kind: ParameterKind::Storage,
        access: Access::ReadWrite,
        resource_layout: None,
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
    };
    const BASE_PARAMETERS: &[ParameterDescriptor] = &[BASE_PARAMETER];
    const VARIANT_PARAMETERS: &[ParameterDescriptor] = &[
        BASE_PARAMETER,
        KernelDescriptor::__gust_launch_base_parameter(1),
    ];
    const MISPLACED_PARAMETERS: &[ParameterDescriptor] = &[
        BASE_PARAMETER,
        KernelDescriptor::__gust_launch_base_parameter(0),
    ];
    const BARE_VARIANT_PARAMETERS: &[ParameterDescriptor] = &[BASE_PARAMETER, BASE_PARAMETER];

    fn base_kernel() -> KernelDescriptor {
        KernelDescriptor::new("m", "k", "m_k", [8, 1, 1], BASE_PARAMETERS, "base")
    }

    fn variant_with(parameters: &'static [ParameterDescriptor]) -> KernelDescriptor {
        KernelDescriptor::new("m", "k", "m_k", [8, 1, 1], parameters, "variant")
    }

    #[test]
    fn a_variant_shaped_like_the_macros_is_accepted() {
        assert!(split_variant_matches(
            &base_kernel(),
            &variant_with(VARIANT_PARAMETERS)
        ));
    }

    #[test]
    fn a_variant_set_by_hand_that_is_not_shaped_like_the_macros_is_refused() {
        let base = base_kernel();
        // No hidden uniform, the hidden uniform on a binding the kernel already uses, and one that is no launch base.
        for parameters in [
            BASE_PARAMETERS,
            MISPLACED_PARAMETERS,
            BARE_VARIANT_PARAMETERS,
        ] {
            assert!(!split_variant_matches(&base, &variant_with(parameters)));
        }
        let good = variant_with(VARIANT_PARAMETERS);
        let mut other_entry = good;
        other_entry.entry_point = "other";
        let other_size =
            KernelDescriptor::new("m", "k", "m_k", [16, 1, 1], VARIANT_PARAMETERS, "v");
        let spirv_artifact = good.with_artifacts(&[SPIRV_ARTIFACT]);
        let mut pixel = good;
        pixel.shape = Shape::Pixel;
        for bad in [other_entry, other_size, spirv_artifact, pixel] {
            assert!(!split_variant_matches(&base, &bad));
        }
        // A WGSL artifact of its own does not make it another thing (D209), and neither does having none.
        let with_artifact = good.with_artifacts(&[WGSL_ARTIFACT]);
        assert!(split_variant_matches(&base, &with_artifact));
        assert!(split_variant_matches(&base, &good));
        // A variant with a variant of its own is no variant.
        static NESTED: KernelDescriptor =
            KernelDescriptor::new("m", "k", "m_k", [8, 1, 1], VARIANT_PARAMETERS, "variant");
        let nested = good.with_split(Some(&NESTED));
        assert!(!split_variant_matches(&base, &nested));
        // A pixel kernel never takes one, whatever it carries.
        let mut pixel_base = base_kernel();
        pixel_base.shape = Shape::Pixel;
        assert!(!split_variant_matches(&pixel_base, &good));
    }

    #[test]
    fn a_kernel_without_a_parameter_has_a_variant_at_binding_zero() {
        let base = KernelDescriptor::new("m", "k", "m_k", [8, 1, 1], &[], "base");
        const ONLY: &[ParameterDescriptor] = &[KernelDescriptor::__gust_launch_base_parameter(0)];
        assert!(split_variant_matches(&base, &variant_with(ONLY)));
    }
}

//! The `pixel` shape on the GPU (slice mx.p1b, decisions D61, D120 and D127).
//!
//! A `pixel` kernel is a Slang fragment entry whose outputs are `RWStructuredBuffer`s at the grid slots. It runs as a
//! render pass over a full-screen triangle (`shapes/fullscreen.wgsl`) with one never-written `r8unorm` colour
//! attachment whose write mask is empty (D120): the attachment only gives the rasteriser a frame to cover, and the
//! fragment index is `y * PIXEL_ROW_WIDTH + x` of that frame (Option S). A call of `n` elements uses
//! `ceil(n / 8192)` rows and at most two scissored draws, the full rows and then the tail row, so no fragment outside
//! `0..n` is ever launched as a real invocation (the scissor probe measured 0 differing words on the RTX 5090 and the
//! Radeon 890M with `LoadOp::Load` and `StoreOp::Discard`). Helper invocations of a partly covered 2x2 quad do run,
//! but they cannot store, so they change nothing; the count test at the end of this file counts the fragments that
//! really ran with an atomic (a stray store cannot be seen: robustBufferAccess2 drops an out-of-bounds store).
//!
//! This module holds the pipeline, the pass, the plan of a call and the refusals that belong to the shape. The
//! dispatch, the batch and the graph reach it through `PreparedDispatch::encode` and the `check_*` functions.

use gust::{Extent, KernelDescriptor, PIXEL_MAX_ELEMENTS, PIXEL_ROW_WIDTH, Shape};

use crate::{Error, PassTimestamps};

/// The vertex stage of every `pixel` pipeline: one triangle that covers the whole attachment.
const FULLSCREEN_WGSL: &str = include_str!("shapes/fullscreen.wgsl");

/// The entry point of [`FULLSCREEN_WGSL`].
const FULLSCREEN_ENTRY: &str = "gust_fullscreen_vs";

/// Why the one-shot v1 functions (`dispatch_f32`, `dispatch_u32`) refuse a `pixel` kernel.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const ONE_SHOT_REASON: &str = "the one-shot v1 functions run compute passes; call a pixel kernel through the typed call or a batch";

/// The vertex module shared by every `pixel` pipeline of a device.
pub(crate) fn fullscreen_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("GUST full-screen vertex stage"),
        source: wgpu::ShaderSource::Wgsl(FULLSCREEN_WGSL.into()),
    })
}

/// The render pipeline of a `pixel` kernel: the shared full-screen vertex stage, the kernel's fragment entry, and one
/// `r8unorm` target that nothing can write (no blend, empty write mask, D120).
pub(crate) fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    vertex: &wgpu::ShaderModule,
    fragment: &wgpu::ShaderModule,
    fragment_entry: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label} pipeline")),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: vertex,
            entry_point: Some(FULLSCREEN_ENTRY),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: fragment,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: ATTACHMENT_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// The format of the attachment: one byte a texel, the cheapest colour format every adapter renders to.
const ATTACHMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// How a call of `n` elements covers the attachment (Option S): `rows` rows of [`PIXEL_ROW_WIDTH`] texels, drawn as the
/// full rows (one scissored draw) and the tail row (a second one over its first `tail` texels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PixelPlan {
    /// The attachment height: `ceil(n / PIXEL_ROW_WIDTH)`.
    pub(crate) rows: u32,
    /// The rows that are full: `n / PIXEL_ROW_WIDTH`.
    pub(crate) full_rows: u32,
    /// The texels in the row after the full ones: `n % PIXEL_ROW_WIDTH`.
    pub(crate) tail: u32,
}

impl PixelPlan {
    pub(crate) const fn new(elements: u32) -> Self {
        Self {
            rows: elements.div_ceil(PIXEL_ROW_WIDTH),
            full_rows: elements / PIXEL_ROW_WIDTH,
            tail: elements % PIXEL_ROW_WIDTH,
        }
    }

    /// The scissor rectangles `[x, y, width, height]`, one per draw, in draw order: zero, one or two of them.
    pub(crate) fn scissors(&self) -> impl Iterator<Item = [u32; 4]> {
        let full = (self.full_rows > 0).then_some([0, 0, PIXEL_ROW_WIDTH, self.full_rows]);
        let tail = (self.tail > 0).then_some([0, self.full_rows, self.tail, 1]);
        full.into_iter().chain(tail)
    }
}

/// Encode one `pixel` call as a render pass of its own: a fresh attachment (it is never written, only discarded), the
/// bind group, and the scissored draws of [`PixelPlan`]. The attachment is dropped here; wgpu keeps it alive until the
/// submission that uses it has finished. It is `PIXEL_ROW_WIDTH` bytes a row (64 MiB at the cap); caching one per
/// device is a follow-up for the performance test.
///
/// The attachment is created inside an out-of-memory error scope (native), so an adapter without the memory for it is
/// [`Error::OutOfMemory`] named `GUST pixel attachment`, and nothing is encoded for that call; wgpu would otherwise
/// report it to the uncaptured-error handler, which panics. On `wasm32` no scope is pushed (it cannot be awaited in a
/// synchronous function, and a scope nobody reads would swallow the browser's own report), as for the buffers.
pub(crate) fn encode_pixel_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    elements: u32,
    label: &str,
    timestamps: Option<PassTimestamps<'_>>,
) -> Result<(), Error> {
    let plan = PixelPlan::new(elements);
    let attachment = create_attachment(device, &plan)?;
    let view = attachment.create_view(&wgpu::TextureViewDescriptor::default());
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            // The form the scissor probe measured. wgpu zero-fills a never-used attachment before its first pass, so
            // `Load` reads zeros (a clear of up to 64 MiB per call); `Discard` stores nothing; no draw writes it.
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Discard,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: timestamps.map(|stamps| wgpu::RenderPassTimestampWrites {
            query_set: stamps.query_set,
            beginning_of_pass_write_index: stamps.begin,
            end_of_pass_write_index: stamps.end,
        }),
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    for [x, y, width, height] in plan.scissors() {
        pass.set_scissor_rect(x, y, width, height);
        pass.draw(0..3, 0..1);
    }
    Ok(())
}

/// The label of the attachment, which [`Error::OutOfMemory`] carries.
const ATTACHMENT_LABEL: &str = "GUST pixel attachment";

/// The bytes of the attachment of `plan`: `PIXEL_ROW_WIDTH` rows of one byte a texel.
#[cfg(not(target_arch = "wasm32"))]
fn attachment_bytes(plan: &PixelPlan) -> u64 {
    u64::from(PIXEL_ROW_WIDTH) * u64::from(plan.rows)
}

/// The attachment of one call, created inside an out-of-memory scope on native.
fn create_attachment(device: &wgpu::Device, plan: &PixelPlan) -> Result<wgpu::Texture, Error> {
    #[cfg(not(target_arch = "wasm32"))]
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let attachment = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(ATTACHMENT_LABEL),
        size: wgpu::Extent3d {
            width: PIXEL_ROW_WIDTH,
            height: plan.rows,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ATTACHMENT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    #[cfg(not(target_arch = "wasm32"))]
    crate::device::finish_popped_scope(ATTACHMENT_LABEL, attachment_bytes(plan), scope.pop())?;
    Ok(attachment)
}

fn shape_unsupported(kernel: &KernelDescriptor, reason: &'static str) -> Error {
    Error::ShapeUnsupported {
        kernel: kernel.qualified_name().to_string(),
        shape: kernel.shape,
        reason,
    }
}

/// The refusal of a shape this backend has no path for (`Shape` is `#[non_exhaustive]`).
pub(crate) fn unknown_shape(kernel: &KernelDescriptor) -> Error {
    shape_unsupported(kernel, "this backend runs the grid and pixel shapes only")
}

/// Refuses every kernel that is not `grid`, with `reason`: the callers that only know compute passes (the v1 one-shot
/// functions, the generated host snippet).
pub(crate) fn require_grid(kernel: &KernelDescriptor, reason: &'static str) -> Result<(), Error> {
    if kernel.shape == Shape::Grid {
        Ok(())
    } else {
        Err(shape_unsupported(kernel, reason))
    }
}

/// Refuses an indirect launch of a `pixel` kernel: a render pass has no indirect workgroup count (D127).
pub(crate) fn refuse_indirect(kernel: &KernelDescriptor) -> Result<(), Error> {
    if kernel.shape == Shape::Pixel {
        Err(shape_unsupported(
            kernel,
            "a pixel kernel has no indirect launch",
        ))
    } else {
        Ok(())
    }
}

/// Refuses a `pixel` call over a two- or three-dimensional extent: its fragment index is the attachment's, not the
/// call's y and z. The twin refuses the same (`cpu::Error::PixelExtent`).
pub(crate) fn check_pixel_extent(kernel: &KernelDescriptor, extent: Extent) -> Result<(), Error> {
    if kernel.shape == Shape::Pixel && (extent.y != 1 || extent.z != 1) {
        Err(shape_unsupported(
            kernel,
            "a pixel call is one-dimensional (y and z must be 1)",
        ))
    } else {
        Ok(())
    }
}

/// The `pixel` limits of one non-empty launch of `elements` against the limits in force: the cap
/// ([`PIXEL_MAX_ELEMENTS`], the same name and variant as the twin's), then the width and the rows of the attachment
/// against `max_texture_dimension_2d` (the baseline 8192 covers both; only a lowered limit refuses).
pub(crate) fn check_pixel_launch(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    elements: u32,
) -> Result<(), Error> {
    let device_limit = |limit: &'static str, requested: u32, maximum: u32| Error::DeviceLimit {
        kernel: kernel.qualified_name().to_string(),
        limit,
        requested: u64::from(requested),
        maximum: u64::from(maximum),
    };
    if elements > PIXEL_MAX_ELEMENTS {
        return Err(device_limit(
            "pixel_max_elements",
            elements,
            PIXEL_MAX_ELEMENTS,
        ));
    }
    let maximum = limits.max_texture_dimension_2d;
    for side in [PIXEL_ROW_WIDTH, PixelPlan::new(elements).rows] {
        if side > maximum {
            return Err(device_limit("max_texture_dimension_2d", side, maximum));
        }
    }
    Ok(())
}

/// Refuses a `pixel` kernel on an adapter whose fragment stage cannot write storage buffers
/// (`DownlevelFlags::FRAGMENT_WRITABLE_STORAGE`); without it wgpu would panic at pipeline creation.
pub(crate) fn require_fragment_storage(
    kernel: &KernelDescriptor,
    supported: bool,
) -> Result<(), Error> {
    if supported {
        Ok(())
    } else {
        Err(shape_unsupported(
            kernel,
            "this adapter cannot write storage buffers from a fragment stage (FRAGMENT_WRITABLE_STORAGE)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use gust::{KernelDescriptor, SourceOrigin};

    use super::*;

    const PIXEL: KernelDescriptor = KernelDescriptor {
        module: "render_test",
        name: "pixel",
        entry_point: "render_test_pixel",
        shape: Shape::Pixel,
        compatible_shapes: &[Shape::Pixel],
        workgroup_size: [1, 1, 1],
        parameters: &[],
        slang_source: "void render_test_pixel() {}",
        source: SourceOrigin::UNKNOWN,
        artifacts: &[],
    };

    const GRID: KernelDescriptor = KernelDescriptor {
        shape: Shape::Grid,
        compatible_shapes: &[Shape::Grid],
        workgroup_size: [64, 1, 1],
        name: "grid",
        ..PIXEL
    };

    fn scissors(elements: u32) -> Vec<[u32; 4]> {
        PixelPlan::new(elements).scissors().collect()
    }

    #[test]
    fn a_plan_has_no_draw_for_nothing_one_for_a_whole_row_and_two_with_a_tail() {
        assert_eq!(PixelPlan::new(0).rows, 0);
        assert!(scissors(0).is_empty());
        // Less than a row: a tail row only.
        assert_eq!(scissors(1), [[0, 0, 1, 1]]);
        assert_eq!(scissors(8191), [[0, 0, 8191, 1]]);
        // Whole rows only.
        assert_eq!(scissors(8192), [[0, 0, 8192, 1]]);
        assert_eq!(scissors(3 * 8192), [[0, 0, 8192, 3]]);
        // Whole rows and a tail: two draws, the tail on the row after the full ones.
        assert_eq!(scissors(8193), [[0, 0, 8192, 1], [0, 1, 1, 1]]);
        assert_eq!(scissors(2 * 8192 + 5), [[0, 0, 8192, 2], [0, 2, 5, 1]]);
    }

    #[test]
    fn a_plan_covers_exactly_n_texels_in_at_most_two_draws_and_rows_round_up() {
        for elements in [
            0,
            1,
            2,
            8191,
            8192,
            8193,
            16_385,
            100_003,
            1_000_003,
            PIXEL_MAX_ELEMENTS - 1,
            PIXEL_MAX_ELEMENTS,
        ] {
            let plan = PixelPlan::new(elements);
            let covered: u64 = scissors(elements)
                .iter()
                .map(|[_, _, width, height]| u64::from(*width) * u64::from(*height))
                .sum();
            assert_eq!(covered, u64::from(elements), "n = {elements}");
            assert!(scissors(elements).len() <= 2, "n = {elements}");
            assert!(u64::from(plan.rows) * u64::from(PIXEL_ROW_WIDTH) >= u64::from(elements));
            assert!(plan.rows == 0 || (plan.rows - 1) * PIXEL_ROW_WIDTH < elements);
        }
        assert_eq!(PixelPlan::new(PIXEL_MAX_ELEMENTS).rows, 8192);
    }

    #[test]
    fn the_cap_and_the_attachment_have_their_own_named_limits() {
        let limits = wgpu::Limits::default();
        assert!(check_pixel_launch(&limits, &PIXEL, PIXEL_MAX_ELEMENTS).is_ok());
        match check_pixel_launch(&limits, &PIXEL, PIXEL_MAX_ELEMENTS + 1) {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => {
                assert_eq!(limit, "pixel_max_elements");
                assert_eq!(requested, u64::from(PIXEL_MAX_ELEMENTS) + 1);
                assert_eq!(maximum, u64::from(PIXEL_MAX_ELEMENTS));
            }
            other => panic!("expected the cap error, got {other:?}"),
        }
        // A lowered texture dimension refuses the width first, then the rows.
        let narrow = wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..wgpu::Limits::default()
        };
        match check_pixel_launch(&narrow, &PIXEL, 1) {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => {
                assert_eq!(limit, "max_texture_dimension_2d");
                assert_eq!((requested, maximum), (8192, 4096));
            }
            other => panic!("expected the width error, got {other:?}"),
        }
    }

    /// A real out-of-memory needs an exhausted adapter, which the tests never create (as for the buffers, see the
    /// `device` tests), so this pins the two halves: the attachment of a normal call is made through the scope and
    /// comes out valid and sized by the plan, and what the scope would catch becomes `Error::OutOfMemory` carrying the
    /// attachment's label and byte size.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_attachment_is_made_in_a_scope_and_an_out_of_memory_is_an_error() {
        let device = crate::HeadlessDevice::new().expect("a Vulkan adapter");
        let plan = PixelPlan::new(3 * PIXEL_ROW_WIDTH + 5);
        let attachment = create_attachment(&device.device, &plan).expect("a small attachment fits");
        assert_eq!(
            (attachment.width(), attachment.height()),
            (PIXEL_ROW_WIDTH, 4)
        );
        assert_eq!(attachment_bytes(&plan), 4 * u64::from(PIXEL_ROW_WIDTH));
        let cap = PixelPlan::new(PIXEL_MAX_ELEMENTS);
        let caught = wgpu::Error::OutOfMemory {
            source: Box::new(std::fmt::Error),
        };
        match crate::device::scope_result(ATTACHMENT_LABEL, attachment_bytes(&cap), Some(caught)) {
            Err(Error::OutOfMemory { label, bytes }) => {
                assert_eq!(label, "GUST pixel attachment");
                assert_eq!(bytes, 64 << 20);
            }
            other => panic!("expected OutOfMemory, got {other:?}"),
        }
    }

    #[test]
    fn only_the_pixel_shape_has_the_pixel_refusals() {
        let wide = Extent::new(4, 2, 1);
        assert!(matches!(
            check_pixel_extent(&PIXEL, wide),
            Err(Error::ShapeUnsupported {
                shape: Shape::Pixel,
                ..
            })
        ));
        assert!(check_pixel_extent(&PIXEL, Extent::linear(4)).is_ok());
        assert!(check_pixel_extent(&GRID, wide).is_ok());
        assert!(refuse_indirect(&PIXEL).is_err());
        assert!(refuse_indirect(&GRID).is_ok());
        assert!(require_grid(&GRID, ONE_SHOT_REASON).is_ok());
        assert!(require_grid(&PIXEL, ONE_SHOT_REASON).is_err());
    }

    #[test]
    fn a_missing_fragment_storage_capability_is_a_named_refusal() {
        assert!(require_fragment_storage(&PIXEL, true).is_ok());
        match require_fragment_storage(&PIXEL, false) {
            Err(Error::ShapeUnsupported {
                kernel,
                shape,
                reason,
            }) => {
                assert_eq!(kernel, "render_test::pixel");
                assert_eq!(shape, Shape::Pixel);
                assert!(reason.contains("FRAGMENT_WRITABLE_STORAGE"));
            }
            other => panic!("expected ShapeUnsupported, got {other:?}"),
        }
    }

    /// A fragment stage that counts its own invocations. An atomic add by a helper invocation does nothing (WGSL), so
    /// the count is the number of fragments that ran and were not helpers: a stray fragment outside the scissor would
    /// raise it. (A stray store cannot be seen the way a stray add can: robustBufferAccess2 drops an out-of-bounds
    /// store, so a stray `out[i]` with `i >= n` leaves no trace.)
    const COUNTER_WGSL: &str = "
        @group(0) @binding(0) var<storage, read_write> count: atomic<u32>;
        @fragment fn count_fragment() { _ = atomicAdd(&count, 1u); }
    ";

    /// The sizes of the count test: the ends of a tail row, one and two rows, and `8192k + 1` and `8192k + 2` (a tail
    /// row of one or two texels), up to the cap.
    fn counted_sizes() -> Vec<u32> {
        let mut sizes = vec![1, 2, 8191, 8192, 8193];
        for rows in [1u32, 2, 7, 1000] {
            sizes.extend([PIXEL_ROW_WIDTH * rows + 1, PIXEL_ROW_WIDTH * rows + 2]);
        }
        sizes.push(PIXEL_MAX_ELEMENTS);
        sizes
    }

    /// The count of fragment invocations equals `n` for every size, through `build_pipeline` and `encode_pixel_pass`
    /// (the pass of every `pixel` call). On a software adapter the sizes over a million are skipped, with a printed
    /// branch; any other adapter runs all of them, 2^26 included.
    #[test]
    fn a_pixel_pass_launches_exactly_n_fragments() {
        let device = crate::HeadlessDevice::with_options(crate::DeviceOptions::default())
            .expect("Vulkan adapter required; this test must not silently skip");
        if !device.fragment_storage {
            eprintln!(
                "BRANCH fragment count: the adapter cannot write storage from a fragment stage; nothing to count"
            );
            return;
        }
        let software = device.adapter_info.device_type == wgpu::DeviceType::Cpu;
        let counter = device.create_typed_buffer(
            "fragment count",
            &[0u32],
            crate::GpuBufferAccess::ReadWrite,
        );
        let layout = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fragment count layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("fragment count pipeline layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
        let fragment = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fragment count shader"),
                source: wgpu::ShaderSource::Wgsl(COUNTER_WGSL.into()),
            });
        let vertex = fullscreen_module(&device.device);
        let pipeline = build_pipeline(
            &device.device,
            "fragment count",
            &pipeline_layout,
            &vertex,
            &fragment,
            "count_fragment",
        );
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fragment count bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counter.raw.as_entire_binding(),
            }],
        });
        let mut ran = 0;
        for n in counted_sizes() {
            if software && n > 1_000_003 {
                eprintln!("BRANCH fragment count: n = {n} skipped, the adapter is of type Cpu");
                continue;
            }
            device
                .queue
                .write_buffer(&counter.raw, 0, &0u32.to_le_bytes());
            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("fragment count encoder"),
                    });
            encode_pixel_pass(
                &device.device,
                &mut encoder,
                &pipeline,
                &bind_group,
                n,
                "fragment count pass",
                None,
            )
            .expect("the attachment fits");
            device.queue.submit([encoder.finish()]);
            let count = device.read_typed_buffer(&counter).unwrap();
            assert_eq!(
                count,
                [n],
                "the pass launched {} fragments for n = {n}",
                count[0]
            );
            eprintln!("COUNT n = {n}: {} fragments", count[0]);
            ran += 1;
        }
        assert!(ran >= 9, "the small sizes must run on every adapter");
        eprintln!("BRANCH fragment count: {ran} sizes run, count == n for each");
    }
}

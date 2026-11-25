//! A caller's render pass reads a batch's buffers through a stable view (m14.s2v, decision D261; Gpu tier).
//!
//! The batch is `y = x * 3 + gain`, then `y = y * 2 + 1` in place, over `u32` values small enough never to overflow, so
//! a result equals the host reference bit for bit and survives the conversion to `f32`. The reader is a real render
//! pass of the caller: its **vertex stage** reads the view as read-only storage (one point instance per element) and
//! the fragment stage writes the value into an offscreen `R32Float` target that the test reads back. A pixel past the
//! end of the live elements shows `-2.0` (the vertex stage compares its instance with `arrayLength`), a pixel no point
//! covers keeps the clear value `-1.0`.
//!
//! The expectations are the host reference, the frame's ticket and the CPU twin, never one another alone.

mod common;

use std::sync::Arc;

use common::{device_logged, drive};
use gust::batch::{Batch, D1, Encoded, Error, Param, Stage, Ticket, View};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{HeadlessDevice, wgpu};

const N: u32 = 1000;
/// A length whose buffer is larger than it (the pool's classes are exact up to 64 KiB, then powers of two), and not a
/// multiple of `WIDTH`, so a pass over its end has live and past-the-end instances.
const LONG: u32 = 20_000;
/// The width of the offscreen target: one pixel per instance drawn (at least `N`, a multiple of 256 bytes a row).
const WIDTH: u32 = 1024;
const CLEAR: f32 = -1.0;
const PAST_THE_END: f32 = -2.0;

const SHADER: &str = "
@group(0) @binding(0) var<storage, read> data: array<u32>;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) value: f32,
};

@vertex
fn vs(@builtin(instance_index) i: u32) -> Out {
    var out: Out;
    let x = (f32(i % 1024u) + 0.5) / 1024.0 * 2.0 - 1.0;
    out.position = vec4<f32>(x, 0.0, 0.0, 1.0);
    if (i >= arrayLength(&data)) {
        out.value = -2.0;
    } else {
        out.value = f32(data[i]);
    }
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return vec4<f32>(in.value, 0.0, 0.0, 1.0);
}
";

struct Handles {
    gain: Param<u32>,
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
}

macro_rules! define_build {
    ($build:ident, $device:ty, $depth:tt) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("viewed", on(device); in_flight($depth); |b| {
                let gain = b.param::<u32>(1);
                let x = b.input::<u32>(N);
                let y = gpu_task!(N, |i| -> u32 { x[i] * 3u32 + gain.get() });
                let _double = gpu_task!(&mut y, |i| { y[i] = y[i] * 2u32 + 1u32; });
                Handles { gain, x, y }
            })
        }
    };
}

define_build!(build_gpu, HeadlessDevice, 1);
define_build!(build_gpu2, HeadlessDevice, 2);
define_build!(build_cpu, CpuDevice, 1);

fn data(seed: u32) -> Vec<u32> {
    (0..N).map(|i| (i * 7 + seed) % 1_000).collect()
}

fn reference(x: &[u32], gain: u32) -> Vec<u32> {
    x.iter().map(|v| (v * 3 + gain) * 2 + 1).collect()
}

/// The target a pass over `values` must leave: the values, then the past-the-end pixels.
fn expected(values: &[u32]) -> Vec<f32> {
    expected_at(values, 0)
}

/// The target a pass over the elements `base..base + WIDTH` of `values` must leave.
fn expected_at(values: &[u32], base: usize) -> Vec<f32> {
    let mut pixels: Vec<f32> = values
        .iter()
        .skip(base)
        .take(WIDTH as usize)
        .map(|&v| v as f32)
        .collect();
    pixels.resize(WIDTH as usize, PAST_THE_END);
    pixels
}

/// The caller's side: the pipeline of the reading pass and the way to record it and read it back.
struct Rig {
    device: Arc<HeadlessDevice>,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Rig {
    fn new(device: &Arc<HeadlessDevice>) -> Self {
        let raw = device.wgpu_device();
        let layout = raw.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view reader layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = raw.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("view reader pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view reader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = raw.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("view reader pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            device: Arc::clone(device),
            pipeline,
            layout,
        }
    }

    /// A bind group over the view, built once by the caller.
    fn bind(&self, view: &View<HeadlessDevice, u32>) -> wgpu::BindGroup {
        self.device
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("view reader bind group"),
                layout: &self.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(view.buffer().as_binding()),
                }],
            })
    }

    /// [`pass_at`](Self::pass_at) over the first `WIDTH` elements.
    fn pass(&self, encoder: &mut wgpu::CommandEncoder, bind: &wgpu::BindGroup) -> wgpu::Buffer {
        self.pass_at(encoder, bind, 0)
    }

    /// Record a pass that draws `WIDTH` point instances, those of elements `base..base + WIDTH` (`base` is a multiple
    /// of `WIDTH`), over the bind group into its own target, and the copy of the target into the returned staging
    /// buffer, into `encoder`.
    fn pass_at(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind: &wgpu::BindGroup,
        base: u32,
    ) -> wgpu::Buffer {
        let raw = self.device.wgpu_device();
        let target = raw.create_texture(&wgpu::TextureDescriptor {
            label: Some("view reader target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("view reader pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(CLEAR),
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..1, base..base + WIDTH);
        }
        let staging = raw.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view reader staging"),
            size: u64::from(WIDTH) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH * 4),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        staging
    }

    /// The pixels of a staging buffer, after the submission that carried its copy.
    fn read(&self, staging: &wgpu::Buffer) -> Vec<f32> {
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device
            .wgpu_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        let pixels = mapped
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        drop(mapped);
        staging.unmap();
        pixels
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    /// The caller's own submission, which GUST does not count.
    fn submit(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.device.wgpu_queue().submit([encoder.finish()])
    }
}

/// One frame by `submit()`: the writes and sets that are given, a ticket of `y`, the values it resolves to.
fn submit_frame(
    batch: &mut Batch<HeadlessDevice>,
    h: &Handles,
    x: Option<&[u32]>,
    gain: Option<u32>,
) -> Vec<u32> {
    let mut frame = drive(batch.begin()).unwrap();
    if let Some(x) = x {
        frame.write(&h.x, x).unwrap();
    }
    if let Some(gain) = gain {
        frame.set(&h.gain, gain).unwrap();
    }
    let ticket = frame.ticket(&h.y).unwrap();
    let _submitted = frame.submit().unwrap();
    drive(ticket).unwrap()
}

/// One frame recorded into the caller's encoder; the caller submits the encoder and reports it.
fn encode_frame(
    batch: &mut Batch<HeadlessDevice>,
    h: &Handles,
    x: Option<&[u32]>,
    gain: Option<u32>,
    encoder: &mut wgpu::CommandEncoder,
) -> (Encoded<HeadlessDevice>, Ticket<u32>) {
    let mut frame = drive(batch.begin()).unwrap();
    if let Some(x) = x {
        frame.write(&h.x, x).unwrap();
    }
    if let Some(gain) = gain {
        frame.set(&h.gain, gain).unwrap();
    }
    let ticket = frame.ticket(&h.y).unwrap();
    (frame.encode(encoder).unwrap(), ticket)
}

fn assert_pixels(label: &str, found: &[f32], wanted: &[f32]) {
    assert_eq!(found.len(), wanted.len(), "{label}: pixel count");
    let differing: Vec<usize> = (0..found.len())
        .filter(|&i| found[i].to_bits() != wanted[i].to_bits())
        .collect();
    assert!(
        differing.is_empty(),
        "{label}: {} pixel(s) differ, the first at {} ({} instead of {})",
        differing.len(),
        differing.first().copied().unwrap_or(0),
        differing.first().map_or(0.0, |&i| found[i]),
        differing.first().map_or(0.0, |&i| wanted[i]),
    );
}

/// Test 1, the full form. One caller encoder holds pass P0, `encode`, pass P1, and one queue submission carries them:
/// P1 reads the frame (equal to its ticket, the host reference and the CPU twin), P0 reads the frame before. Each pass
/// reads the stage result `y` and the input `x`, so P0 also shows that this frame's `write` has not run yet.
#[test]
fn a_pass_after_encode_reads_the_frame_and_one_before_it_the_frame_before() {
    let device = Arc::new(device_logged("batch_view full form"));
    let rig = Rig::new(&device);
    let (mut batch, h) = build_gpu(&device).unwrap();
    let before_views = device.gust_submissions();
    let view = batch.view(&h.y).unwrap();
    assert_eq!(
        device.gust_submissions(),
        before_views,
        "`view` makes no submission"
    );
    assert_eq!(
        (view.len(), view.generation()),
        (N as usize, batch.generation())
    );
    let bind = rig.bind(&view);
    let x_view = batch.view(&h.x).unwrap();
    let x_bind = rig.bind(&x_view);

    let x0 = data(1);
    let first = submit_frame(&mut batch, &h, Some(&x0), Some(2));
    assert_eq!(first, reference(&x0, 2));

    let x1 = data(5);
    let counted = device.gust_submissions();
    let mut encoder = rig.encoder();
    let p0 = rig.pass(&mut encoder, &bind);
    let p0_x = rig.pass(&mut encoder, &x_bind);
    let (encoded, ticket) = encode_frame(&mut batch, &h, Some(&x1), Some(7), &mut encoder);
    let p1 = rig.pass(&mut encoder, &bind);
    let p1_x = rig.pass(&mut encoder, &x_bind);
    let index = rig.submit(encoder);
    let _submitted = encoded.submitted(&index);
    assert_eq!(
        device.gust_submissions(),
        counted,
        "the encode form and the caller's one submission are none of GUST's"
    );
    let second = drive(ticket).unwrap();
    assert_eq!(
        second,
        reference(&x1, 7),
        "the ticket is the host reference"
    );
    assert_pixels("P1 after encode", &rig.read(&p1), &expected(&second));
    assert_pixels("P0 before encode", &rig.read(&p0), &expected(&first));
    assert_pixels("P1 input after encode", &rig.read(&p1_x), &expected(&x1));
    assert_pixels("P0 input before encode", &rig.read(&p0_x), &expected(&x0));

    // The CPU twin, through its own encoder, ends in the same values.
    let cpu = Arc::new(CpuDevice::new());
    let (mut twin, ht) = build_cpu(&cpu).unwrap();
    let mut frame = drive(twin.begin()).unwrap();
    frame.write(&ht.x, &x0).unwrap();
    frame.set(&ht.gain, 2).unwrap();
    let _first = frame.submit().unwrap();
    let mut cpu_encoder = gust::batch::BatchDevice::new_encoder(&*cpu);
    let mut frame = drive(twin.begin()).unwrap();
    frame.write(&ht.x, &x1).unwrap();
    frame.set(&ht.gain, 7).unwrap();
    let twin_ticket = frame.ticket(&ht.y).unwrap();
    let encoded = frame.encode(&mut cpu_encoder).unwrap();
    let job = cpu_encoder.run(&cpu).unwrap();
    let _submitted = encoded.submitted(&job);
    assert_eq!(drive(twin_ticket).unwrap(), second, "twin equals GPU");
}

/// Test 2: one bind group, built once, serves three frames while the generation stays: each frame's P1 is exact.
#[test]
fn one_bind_group_serves_three_frames_while_the_generation_holds() {
    let device = Arc::new(device_logged("batch_view bind group"));
    let rig = Rig::new(&device);
    let (mut batch, h) = build_gpu(&device).unwrap();
    let view = batch.view(&h.y).unwrap();
    let generation = batch.generation();
    let bind = rig.bind(&view);
    for (k, (seed, gain)) in [(2_u32, 3_u32), (3, 11), (4, 17)].into_iter().enumerate() {
        let x = data(seed);
        let mut encoder = rig.encoder();
        let (encoded, ticket) = encode_frame(&mut batch, &h, Some(&x), Some(gain), &mut encoder);
        let pass = rig.pass(&mut encoder, &bind);
        let index = rig.submit(encoder);
        let _submitted = encoded.submitted(&index);
        assert_eq!(batch.generation(), generation, "frame {k}");
        assert_eq!(view.generation(), batch.generation(), "frame {k}");
        let values = drive(ticket).unwrap();
        assert_eq!(values, reference(&x, gain), "frame {k} ticket");
        assert_pixels(
            &format!("frame {k} P1"),
            &rig.read(&pass),
            &expected(&values),
        );
    }
}

/// Test 3, `in_flight(2)`: two frames are encoded and submitted in order before anything is read, and each pass reads
/// the frame of its own encoder, not the later one, for the result `y` and for the input `x`. (A second `begin` is
/// refused while the first `Encoded` is unreported, so B is begun after A is reported, without waiting for A to run,
/// which is the overlap `in_flight(2)` allows; a pass recorded into B's encoder before `encode_frame` reads A's frame.)
#[test]
fn with_two_in_flight_each_pass_reads_its_own_frame() {
    let device = Arc::new(device_logged("batch_view in_flight(2)"));
    let rig = Rig::new(&device);
    let (mut batch, h) = build_gpu2(&device).unwrap();
    let view = batch.view(&h.y).unwrap();
    let bind = rig.bind(&view);
    let x_bind = rig.bind(&batch.view(&h.x).unwrap());
    let (xa, xb) = (data(6), data(9));

    let mut encoder_a = rig.encoder();
    let (encoded_a, ticket_a) = encode_frame(&mut batch, &h, Some(&xa), Some(4), &mut encoder_a);
    let pass_a = rig.pass(&mut encoder_a, &bind);
    let pass_a_x = rig.pass(&mut encoder_a, &x_bind);
    let index_a = rig.submit(encoder_a);
    let _submitted_a = encoded_a.submitted(&index_a);

    let mut encoder_b = rig.encoder();
    let pass_b_before = rig.pass(&mut encoder_b, &bind);
    let pass_b_x_before = rig.pass(&mut encoder_b, &x_bind);
    let (encoded_b, ticket_b) = encode_frame(&mut batch, &h, Some(&xb), Some(8), &mut encoder_b);
    let pass_b = rig.pass(&mut encoder_b, &bind);
    let pass_b_x = rig.pass(&mut encoder_b, &x_bind);
    let index_b = rig.submit(encoder_b);
    let _submitted_b = encoded_b.submitted(&index_b);

    let (values_a, values_b) = (drive(ticket_a).unwrap(), drive(ticket_b).unwrap());
    assert_eq!(values_a, reference(&xa, 4));
    assert_eq!(values_b, reference(&xb, 8));
    assert_ne!(
        values_a, values_b,
        "the two frames must differ for this test to mean anything"
    );
    assert_pixels("frame A pass", &rig.read(&pass_a), &expected(&values_a));
    assert_pixels("frame B pass", &rig.read(&pass_b), &expected(&values_b));
    assert_pixels("frame A input", &rig.read(&pass_a_x), &expected(&xa));
    assert_pixels("frame B input", &rig.read(&pass_b_x), &expected(&xb));
    assert_pixels(
        "frame B pass before encode",
        &rig.read(&pass_b_before),
        &expected(&values_a),
    );
    assert_pixels(
        "frame B input before encode",
        &rig.read(&pass_b_x_before),
        &expected(&xa),
    );
}

/// Test 4: a dropped `Encoded` and its encoder, then a frame that does not `write`: its pass reads the last frame that
/// reached the queue, with that frame's input, and this frame's own `set`.
#[test]
fn after_a_dropped_encoded_a_frame_without_a_write_reads_the_last_frame_that_reached_the_queue() {
    let device = Arc::new(device_logged("batch_view dropped Encoded"));
    let rig = Rig::new(&device);
    let (mut batch, h) = build_gpu(&device).unwrap();
    let view = batch.view(&h.y).unwrap();
    let bind = rig.bind(&view);

    let x0 = data(10);
    let first = submit_frame(&mut batch, &h, Some(&x0), Some(2));
    assert_eq!(first, reference(&x0, 2));

    // The frame that never reaches the queue, with the pass recorded in its encoder: all of it is dropped.
    let x_lost = data(99);
    let mut lost = rig.encoder();
    let (encoded, ticket) = encode_frame(&mut batch, &h, Some(&x_lost), Some(9), &mut lost);
    let _lost_pass = rig.pass(&mut lost, &bind);
    drop(encoded);
    drop(lost);
    assert!(matches!(drive(ticket), Err(Error::NotSubmitted { .. })));

    // The next frame writes nothing and sets another gain; its pass is recorded after `encode`.
    let mut encoder = rig.encoder();
    let (encoded, ticket) = encode_frame(&mut batch, &h, None, Some(4), &mut encoder);
    let pass = rig.pass(&mut encoder, &bind);
    let index = rig.submit(encoder);
    let _submitted = encoded.submitted(&index);
    let values = drive(ticket).unwrap();
    assert_eq!(
        values,
        reference(&x0, 4),
        "the lost frame's write and set did not reach the queue"
    );
    assert_ne!(values, first, "the frame's own set shows");
    assert_pixels(
        "pass after the dropped Encoded",
        &rig.read(&pass),
        &expected(&values),
    );
}

/// Test 5, the size class: the buffer behind a length of 20000 is larger, and the binding gives the shader
/// `arrayLength == len` (the pixels past the end show the past-the-end value, never the class's tail). A buffer of no
/// element binds one zeroed placeholder element.
#[test]
fn the_binding_covers_the_live_elements_and_not_the_size_class() {
    let device = Arc::new(device_logged("batch_view size class"));
    let rig = Rig::new(&device);
    let (mut batch, handle) =
        gpu_batch!("long", on(&device); |b| { let a = b.input::<u32>(LONG); a }).unwrap();
    let view = batch.view(&handle).unwrap();
    let live_bytes = u64::from(LONG) * 4;
    assert!(
        view.buffer().as_wgpu_buffer().size() > live_bytes,
        "this length must be below its size class, or the test shows nothing"
    );
    assert_eq!(
        view.buffer()
            .as_binding()
            .size
            .map(std::num::NonZeroU64::get),
        Some(live_bytes)
    );
    let values: Vec<u32> = (0..LONG).map(|i| (i * 7 + 12) % 1_000).collect();
    let mut frame = drive(batch.begin()).unwrap();
    frame.write(&handle, &values).unwrap();
    let ticket = frame.ticket(&handle).unwrap();
    let _submitted = frame.submit().unwrap();
    assert_eq!(drive(ticket).unwrap(), values);

    // The pass covers the last elements and the instances after them: the live ones show their values, the rest the
    // past-the-end value, never the (zeroed) tail of the size class.
    let base = (LONG / WIDTH) * WIDTH;
    let bind = rig.bind(&view);
    let mut encoder = rig.encoder();
    let pass = rig.pass_at(&mut encoder, &bind, base);
    let _index = rig.submit(encoder);
    let pixels = rig.read(&pass);
    assert_pixels(
        "a pass over the end of a length below its size class",
        &pixels,
        &expected_at(&values, base as usize),
    );
    assert!(
        pixels[(LONG - base) as usize..]
            .iter()
            .all(|&p| p == PAST_THE_END)
    );

    // No element: one placeholder element, zeroed, so `arrayLength` is 1 and the caller reads `len()` for the count.
    let (empty, handle) =
        gpu_batch!("empty", on(&device); |b| { let a = b.input::<u32>(0u32); a }).unwrap();
    let empty_view = empty.view(&handle).unwrap();
    assert!(empty_view.is_empty());
    assert_eq!(
        empty_view
            .buffer()
            .as_binding()
            .size
            .map(std::num::NonZeroU64::get),
        Some(4)
    );
    let bind = rig.bind(&empty_view);
    let mut encoder = rig.encoder();
    let pass = rig.pass(&mut encoder, &bind);
    let _index = rig.submit(encoder);
    assert_pixels(
        "a pass over an empty buffer",
        &rig.read(&pass),
        &expected(&[0]),
    );
}

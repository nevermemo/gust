//! A window that presents from the device GUST runs on.
//!
//! The example owns the `wgpu::Instance`, the surface and the adapter, asks `DeviceOptions::device_request` for the
//! features and limits GUST needs, adds its own (`FLOAT32_BLENDABLE` where offered), opens the device itself and
//! hands it to GUST with `HeadlessDevice::adopt`. It then renders 60 frames: each runs one GUST kernel on the shared
//! device, reads the result back, and clears the window to the colour the kernel computed. On success it prints
//! `GUST_OK frames=60 adapter=<name>` and exits with code 0; any failure prints the cause and exits non-zero.
//!
//! `GUST_WINDOW_ADAPTER=<text>` picks the adapter whose name contains the text (ignoring case), among those that can
//! present to the window; without it the high-performance adapter that can present is used.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gust::gpu_module;
use gust_wgpu::wgpu;
use gust_wgpu::{DeviceOptions, GpuBuffer, GpuBufferAccess, HeadlessDevice};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// The frames to present before the run counts as passed.
const FRAMES: u32 = 60;
/// The run fails if the frames are not presented within this time (a window that is never visible presents none).
const DEADLINE: Duration = Duration::from_secs(60);

#[gpu_module]
mod shade {
    /// `out[i] = input[i] * 0.5 + 0.25`: a colour the host can work out by hand.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn colour(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 0.5 + 0.25;
        }
    }
}

/// Everything that exists once the window does.
struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    gust: HeadlessDevice,
    input: GpuBuffer<f32>,
    out: GpuBuffer<f32>,
    adapter: String,
}

struct App {
    instance: wgpu::Instance,
    gpu: Option<Gpu>,
    frames: u32,
    started: Instant,
    failure: Option<String>,
}

impl App {
    fn start(&self, event_loop: &ActiveEventLoop) -> Result<Gpu, String> {
        let attributes = Window::default_attributes()
            .with_title("GUST window share")
            .with_inner_size(LogicalSize::new(320.0, 240.0));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .map_err(|error| format!("could not create the window: {error}"))?,
        );
        let surface = self
            .instance
            .create_surface(Arc::clone(&window))
            .map_err(|error| format!("could not create the surface: {error}"))?;
        let adapter = pick_adapter(&self.instance, &surface)?;

        // What GUST asks of the adapter, ORed with what this renderer needs.
        let options = DeviceOptions::default();
        let request = options
            .device_request(&adapter)
            .map_err(|error| format!("device_request: {error}"))?;
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("window-share device"),
                required_features: request.features
                    | (adapter.features() & wgpu::Features::FLOAT32_BLENDABLE),
                required_limits: request
                    .limits
                    .or_better_values_from(&wgpu::Limits::default()),
                ..Default::default()
            }),
        )
        .map_err(|error| format!("could not open the device: {error}"))?;
        let gust = HeadlessDevice::adopt(
            &adapter,
            device.clone(),
            queue.clone(),
            options.capabilities,
        )
        .map_err(|error| format!("adopt: {error}"))?;

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("the surface does not support this adapter")?;
        surface.configure(&device, &config);

        let input =
            gust.create_typed_buffer("colour input", &[0.0_f32; 4], GpuBufferAccess::ReadOnly);
        let out = gust.create_typed_buffer("colour out", &[0.0_f32; 4], GpuBufferAccess::ReadWrite);
        let adapter = adapter.get_info().name;
        Ok(Gpu {
            window,
            surface,
            config,
            device,
            queue,
            gust,
            input,
            out,
            adapter,
        })
    }

    /// One frame: a GUST kernel on the shared device, then a clear pass of the surface to the kernel's colour.
    /// Returns whether a frame was presented (`false`: the surface was not ready, try again).
    fn draw(&mut self) -> Result<bool, String> {
        let Some(gpu) = &mut self.gpu else {
            return Ok(false);
        };
        let t = self.frames as f32 / (FRAMES - 1) as f32;
        let sent = [t, 1.0 - t, 0.5, 1.0];
        gpu.gust
            .write_typed_buffer(&gpu.input, &sent)
            .map_err(|error| format!("write: {error}"))?;
        shade::colour::dispatch(&gpu.gust, 4, &gpu.input, &gpu.out)
            .map_err(|error| format!("dispatch: {error}"))?;
        let colour = gpu
            .gust
            .read_typed_buffer(&gpu.out)
            .map_err(|error| format!("read: {error}"))?;
        for (got, sent) in colour.iter().zip(sent) {
            let expected = sent * 0.5 + 0.25;
            if (got - expected).abs() > 1e-6 {
                return Err(format!(
                    "frame {}: the kernel gave {got}, the host worked out {expected}",
                    self.frames
                ));
            }
        }

        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // Present this one, and match the surface again before the next.
                gpu.surface.configure(&gpu.device, &gpu.config);
                frame
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("the surface reported a validation error".to_owned());
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear to the kernel's colour"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(colour[0]),
                        g: f64::from(colour[1]),
                        b: f64::from(colour[2]),
                        a: f64::from(colour[3]),
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        gpu.queue.submit([encoder.finish()]);
        gpu.window.pre_present_notify();
        gpu.queue.present(frame);
        Ok(true)
    }
}

/// The adapter that presents to `surface`: the one named by `GUST_WINDOW_ADAPTER`, else the high-performance one.
fn pick_adapter(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> Result<wgpu::Adapter, String> {
    match std::env::var("GUST_WINDOW_ADAPTER") {
        Ok(text) => {
            let text = text.to_lowercase();
            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
                .into_iter()
                .find(|adapter| {
                    adapter.get_info().name.to_lowercase().contains(&text)
                        && adapter.is_surface_supported(surface)
                })
                .ok_or_else(|| {
                    format!("no adapter whose name contains `{text}` can present to the window")
                })
        }
        Err(_) => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(surface),
            ..Default::default()
        }))
        .map_err(|error| format!("no adapter can present to the window: {error}")),
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        match self.start(event_loop) {
            Ok(gpu) => {
                println!("adapter: {}", gpu.adapter);
                gpu.window.request_redraw();
                self.gpu = Some(gpu);
            }
            Err(error) => {
                self.failure = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if self.frames < FRAMES {
                    self.failure = Some(format!(
                        "the window was closed after {} frames",
                        self.frames
                    ));
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width;
                    gpu.config.height = size.height;
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::RedrawRequested => {
                if self.started.elapsed() > DEADLINE {
                    self.failure = Some(format!("only {} frames in {DEADLINE:?}", self.frames));
                    event_loop.exit();
                    return;
                }
                match self.draw() {
                    Ok(true) => self.frames += 1,
                    Ok(false) => {}
                    Err(error) => {
                        self.failure = Some(error);
                        event_loop.exit();
                        return;
                    }
                }
                if self.frames >= FRAMES {
                    event_loop.exit();
                } else if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn run() -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|error| format!("no event loop: {error}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut descriptor = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
        event_loop.owned_display_handle(),
    ));
    descriptor.backends = wgpu::Backends::VULKAN;
    let mut app = App {
        instance: wgpu::Instance::new(descriptor),
        gpu: None,
        frames: 0,
        started: Instant::now(),
        failure: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|error| format!("the event loop failed: {error}"))?;
    if let Some(failure) = app.failure {
        return Err(failure);
    }
    match &app.gpu {
        Some(gpu) if app.frames == FRAMES => {
            println!("GUST_OK frames={} adapter={}", app.frames, gpu.adapter);
            Ok(())
        }
        _ => Err(format!("stopped after {} of {FRAMES} frames", app.frames)),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("GUST_FAIL {error}");
            ExitCode::FAILURE
        }
    }
}

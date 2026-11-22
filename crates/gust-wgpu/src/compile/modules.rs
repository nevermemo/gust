use crate::{HeadlessDevice, render};
#[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
use crate::{Route, RouteEvent};
use gust::KernelDescriptor;

impl HeadlessDevice {
    /// The shader module a kernel's pipeline is built from. Without the `spirv-route` feature it is always the kernel's
    /// WGSL through naga. With it, slangc's SPIR-V through passthrough when every condition holds (see
    /// [`spirv_module`](Self::spirv_module)), else the same WGSL; either way the route is recorded in
    /// [`route_events`](Self::route_events).
    pub(crate) fn shader_module(
        &self,
        kernel: &KernelDescriptor,
        label: &str,
        wgsl: &str,
        layout_entries: &[wgpu::BindGroupLayoutEntry],
        pixel: bool,
        from_artifact: bool,
    ) -> wgpu::ShaderModule {
        #[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
        {
            match self.spirv_module(kernel, label, layout_entries, pixel, from_artifact) {
                Ok(module) => {
                    self.note_route(kernel, Route::Spirv, None);
                    return module;
                }
                Err(reason) => self.note_route(kernel, Route::Wgsl, Some(reason)),
            }
        }
        #[cfg(not(all(feature = "spirv-route", not(target_arch = "wasm32"))))]
        let _ = (kernel, layout_entries, pixel, from_artifact);
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{label} Slang-generated WGSL")),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            })
    }

    /// The SPIR-V route (m9.rt2, D172, D174): slangc's SPIR-V for the kernel as a passthrough module, or the reason
    /// this kernel stays on the WGSL route. Checked in this order: a precompiled WGSL artifact (its bytes are the
    /// contract and no SPIR-V artifact exists), a backend that is not Vulkan, a build without the run-time compiler, a
    /// device without `PASSTHROUGH_SHADERS`, a SPIR-V compile error, a module that does not fit the pipeline layout
    /// ([`passthrough::check_bindings`]), a binding that cannot be renumbered.
    #[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
    fn spirv_module(
        &self,
        kernel: &KernelDescriptor,
        label: &str,
        layout_entries: &[wgpu::BindGroupLayoutEntry],
        pixel: bool,
        from_artifact: bool,
    ) -> Result<wgpu::ShaderModule, String> {
        if from_artifact {
            return Err("precompiled WGSL artifact: its bytes are the contract".to_owned());
        }
        if let Some(reason) = passthrough::backend_reason(self.adapter_info.backend) {
            return Err(reason.to_owned());
        }
        #[cfg(not(feature = "slang-runtime"))]
        {
            let _ = (kernel, label, layout_entries, pixel);
            Err("no slang-runtime: no SPIR-V to compile".to_owned())
        }
        #[cfg(feature = "slang-runtime")]
        {
            if !self
                .device
                .features()
                .contains(wgpu::Features::PASSTHROUGH_SHADERS)
            {
                return Err("device lacks PASSTHROUGH_SHADERS".to_owned());
            }
            let spirv = gust::reflect::compile_reflected(kernel, gust::slang::Target::Spirv)
                .map_err(|error| format!("SPIR-V compile: {error}"))?;
            let mut words = passthrough::words(&spirv.artifact).ok_or_else(|| {
                "SPIR-V compile: byte length is not a multiple of four".to_owned()
            })?;
            let scan = passthrough::scan(&words).map_err(|error| format!("scan: {error}"))?;
            passthrough::check_bindings(&scan, layout_entries, kernel.entry_point, pixel)
                .map_err(|error| format!("binding check: {error}"))?;
            passthrough::densify_bindings(&mut words, layout_entries)
                .map_err(|error| format!("densify: {error}"))?;
            let [x, y, z] = spirv.workgroup_size;
            // SAFETY: the words are slangc's own module for this kernel (the Spirv suite runs `spirv-val` on every
            // example), its entry point and bindings were checked against the pipeline layout above, and the bindings
            // were renumbered to the layout's dense numbering, which passthrough would not do.
            Ok(unsafe {
                self.device.create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough {
                        label: Some(&format!("{label} slangc SPIR-V")),
                        entry_points: vec![wgpu::PassthroughShaderEntryPoint {
                            name: kernel.entry_point.into(),
                            workgroup_size: (x, y, z),
                        }]
                        .into(),
                        spirv: Some(words.into()),
                        ..Default::default()
                    },
                )
            })
        }
    }

    #[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
    fn note_route(&self, kernel: &KernelDescriptor, route: Route, reason: Option<String>) {
        // A verification aid: with GUST_SPIRV_ROUTE_LOG=<file> every event is appended as one line, so a run of generated
        // crates (the difftest) can show which route its kernels took.
        if let Some(path) = std::env::var_os("GUST_SPIRV_ROUTE_LOG") {
            use std::io::Write as _;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                // One `write_all` of the whole line (m9.rt2 review, R4): parallel test threads append to the same log,
                // and `writeln!` on an unbuffered file issues several writes per line, which can tear a line.
                let line = format!(
                    "{} {:?} {}\n",
                    kernel.qualified_name(),
                    route,
                    reason.as_deref().unwrap_or("")
                );
                let _ = file.write_all(line.as_bytes());
            }
        }
        self.route_events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(RouteEvent {
                kernel: kernel.qualified_name().to_string(),
                route,
                reason,
            });
    }

    /// The route each pipeline of this device was built from so far, in build order, one event per kernel compile (the
    /// pipeline cache means a kernel appears once per device unless its compile failed). `Route::Wgsl` carries the
    /// reason the SPIR-V route was not taken. Only with the `spirv-route` feature (m9.rt2).
    /// With the environment variable `GUST_SPIRV_ROUTE_LOG=<file>` set, every event is also appended to that file as a line
    /// `<kernel> <Spirv|Wgsl> <reason>`, which is how `check-difftest` shows the route of generated crates.
    #[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
    #[cfg_attr(
        docsrs,
        doc(cfg(all(feature = "spirv-route", not(target_arch = "wasm32"))))
    )]
    pub fn route_events(&self) -> Vec<RouteEvent> {
        self.route_events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The vertex module every `pixel` pipeline of this device shares, created on first use.
    pub(crate) fn fullscreen_module(&self) -> wgpu::ShaderModule {
        let mut cache = self.lock_pipeline_cache();
        cache
            .fullscreen
            .get_or_insert_with(|| render::fullscreen_module(&self.device))
            .clone()
    }
}

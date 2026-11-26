#[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
use crate::passthrough;
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
        self.wgsl_module(label, wgsl)
    }

    /// The WGSL route's module call, the one place the device's [`ShaderChecks`](crate::ShaderChecks) reaches wgpu.
    fn wgsl_module(&self, label: &str, wgsl: &str) -> wgpu::ShaderModule {
        // SAFETY: `create_shader_module_trusted` is unsafe for the runtime checks it can turn off. They are on unless the
        // caller opened this device with `DeviceOptions::unchecked_kernels` (`unsafe`, `# Safety` there), which is the
        // caller's promise about every kernel of the device; `shader_checks` is private and has no other writer. With
        // `Checked` the call is what `create_shader_module` does. wgpu's WebGPU backend ignores the checks argument.
        unsafe {
            self.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("{label} Slang-generated WGSL")),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                },
                self.noted(self.shader_checks.runtime_checks()),
            )
        }
    }

    /// `checks`, unchanged. In a test build it also records them as what the module call was given: it wraps the
    /// argument of the call itself, so a test sees the value wgpu received, not a copy.
    fn noted(&self, checks: wgpu::ShaderRuntimeChecks) -> wgpu::ShaderRuntimeChecks {
        #[cfg(test)]
        {
            *self
                .last_module_checks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(checks);
        }
        checks
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

#[cfg(test)]
mod tests {
    use crate::{DeviceOptions, HeadlessDevice, ShaderChecks};

    /// The setting changes the wgpu runtime checks of a module, all of them and nothing else: `Checked` is wgpu's own
    /// `checked()` (what `create_shader_module` uses), `Unchecked` is `unchecked()`. No adapter needed.
    #[test]
    fn the_setting_selects_all_wgpu_runtime_checks_on_or_off() {
        let on = ShaderChecks::Checked.runtime_checks();
        let off = ShaderChecks::Unchecked.runtime_checks();
        let flags = |c: wgpu::ShaderRuntimeChecks| {
            [
                c.bounds_checks,
                c.force_loop_bounding,
                c.ray_query_initialization_tracking,
                c.task_shader_dispatch_tracking,
                c.mesh_shader_primitive_indices_clamp,
                c.int_div_checks,
            ]
        };
        assert_eq!(flags(on), [true; 6]);
        assert_eq!(flags(off), [false; 6]);
        assert_eq!(flags(on), flags(wgpu::ShaderRuntimeChecks::checked()));
        assert_eq!(flags(off), flags(wgpu::ShaderRuntimeChecks::unchecked()));
    }

    /// The module call itself: a device opened checked builds its WGSL module with every wgpu check on, and one opened
    /// with `unchecked_kernels` with every check off. The call records the checks it passed to wgpu (test builds only),
    /// so this sees the funnel, not the accessor. Needs an adapter. What an unchecked module does on the GPU is not
    /// observed: out of range it is undefined behaviour, so no test can see a difference there.
    #[test]
    fn the_module_call_passes_the_devices_setting_to_wgpu() {
        const WGSL: &str = "@compute @workgroup_size(1) fn main() {}";
        let flags = |device: &HeadlessDevice| {
            let _ = device.wgsl_module("probe", WGSL);
            let c = device
                .last_module_checks
                .lock()
                .unwrap()
                .expect("the call recorded its checks");
            [
                c.bounds_checks,
                c.force_loop_bounding,
                c.ray_query_initialization_tracking,
                c.task_shader_dispatch_tracking,
                c.mesh_shader_primitive_indices_clamp,
                c.int_div_checks,
            ]
        };
        let checked = HeadlessDevice::new().expect("an adapter is required");
        assert_eq!(flags(&checked), [true; 6]);
        // SAFETY: the only module this device builds is an empty compute entry point.
        let options = unsafe { DeviceOptions::default().unchecked_kernels() };
        let unchecked = HeadlessDevice::with_options(options).expect("an adapter is required");
        assert_eq!(flags(&unchecked), [false; 6]);
    }

    /// Only the `unsafe` constructor turns the checks off, and the default is checked.
    #[test]
    fn options_are_checked_unless_unchecked_kernels_is_called() {
        assert_eq!(ShaderChecks::default(), ShaderChecks::Checked);
        assert_eq!(
            DeviceOptions::default().shader_checks(),
            ShaderChecks::Checked
        );
        // SAFETY: the options are only inspected here; no device is opened with them.
        let options = unsafe { DeviceOptions::default().unchecked_kernels() };
        assert_eq!(options.shader_checks(), ShaderChecks::Unchecked);
        assert_ne!(options, DeviceOptions::default());
    }
}

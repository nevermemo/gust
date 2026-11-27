//! Creating kernels ahead of use: [`HeadlessDevice::precompile`].

use std::time::Duration;

use gust::{KernelDescriptor, ParameterKind};

use crate::texture::TextureCounts;
use crate::{
    Error, HeadlessDevice, check_binding_indices, check_kernel_capabilities, check_kernel_limits,
    validate_kernel,
};

/// What the pipeline cache counted for one descriptor of a [`HeadlessDevice::precompile`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecompileOutcome {
    /// The device's pipeline cache already held the kernel (counted as a hit). A kernel that another thread created
    /// while this call was creating it reads `Hit` too: the cache kept the other thread's and counted this lookup a
    /// hit.
    Hit,
    /// The kernel was not in the cache and this call created it (counted as a miss).
    Compiled,
}

/// The report of one descriptor of a [`HeadlessDevice::precompile`] call.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Precompiled {
    /// The kernel's qualified name (`module::kernel`).
    pub kernel: String,
    /// Whether this call created the kernel or found it.
    pub outcome: PrecompileOutcome,
    /// The time this call took on the CPU for this kernel: its checks, the artifact check (or the `slangc` run, with
    /// the `slang-runtime` feature) and the `wgpu` calls that create the module, layouts and pipeline. A hit is a
    /// lookup. A driver or a browser may defer part of its own compile past the create call, so a short time does not
    /// prove the pipeline is fully built. `None` on `wasm32`, where this crate has no clock.
    pub elapsed: Option<Duration>,
}

impl HeadlessDevice {
    /// Create the named kernels in this device's pipeline cache now, instead of at their first dispatch.
    ///
    /// A "precompiled artifact" is the build-time WGSL a descriptor carries; this call creates the *device's* pipeline
    /// from it (or from the run-time compiler's output, with the `slang-runtime` feature). Each descriptor, in the
    /// order given, gets the device-level checks of a launch (the workgroup size and the buffer counts against the
    /// limits, the kernel's `requires`, the binding indices, a `pixel` kernel's fragment storage), then its shader
    /// module, bind group layout, pipeline layout and pipeline. A later dispatch of the same descriptor (a batch, a
    /// staged graph) is a cache hit and creates none of these. Exactly the named descriptors are created: the split
    /// variant (the form a launch past a workgroup limit runs) is a descriptor of its own, `KernelDescriptor::split`,
    /// and is named like any other. In a build without the run-time compiler a split variant without its own artifact
    /// is refused here with `Error::NoArtifact`; name it only when the build embeds its artifact or has
    /// `slang-runtime`.
    ///
    /// "The same descriptor" is the same content, wherever it is stored: the cache finds a kernel by the address of its
    /// descriptor's data first and, for an address it has not seen, by the content (entry point, source text,
    /// parameters, workgroup size, shape, `requires` and the artifact), so `module::kernel::DESCRIPTOR` here and the
    /// descriptor a typed call (`kernel::dispatch`, `call`, `submit`) reads through `Kernel` are one entry, and the
    /// later call is a hit.
    ///
    /// The call stops at the first error, which is the one any dispatch of that descriptor would give before its extent
    /// and buffers are checked; the earlier entries stay in the cache and their report is dropped with the error. It
    /// returns one [`Precompiled`] per descriptor, in the order given. A descriptor already in the cache is a hit,
    /// counted like any lookup, and a repeat of a call is hits only. [`HeadlessDevice::clear_pipeline_cache`] undoes
    /// it.
    ///
    /// "Created" is the promise, not "warm": nothing is submitted, so a driver may still do work at the first
    /// submission, and in the browser the objects are created but the browser may still be compiling. Bind groups are
    /// made at use, as they need buffers. A batch stage sealed over an empty extent compiles at its first non-empty
    /// frame; this call does not reach it. With the `slang-runtime` feature a kernel without an artifact runs `slangc`
    /// here, which is slow: use precompiled artifacts for timed work.
    ///
    /// ```
    /// use gust::gpu_module;
    /// use gust_wgpu::{HeadlessDevice, PrecompileOutcome};
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
    ///     // The kernel and, when it has one, its split variant.
    ///     let mut kernels = vec![&double::run::DESCRIPTOR];
    ///     if let Some(split) = double::run::DESCRIPTOR.split {
    ///         kernels.push(split);
    ///     }
    ///     let report = device.precompile(&kernels)?;
    ///     assert_eq!(report.len(), kernels.len());
    ///     assert!(report.iter().all(|entry| entry.outcome == PrecompileOutcome::Compiled));
    ///
    ///     let stats = device.pipeline_cache_stats();
    ///     assert_eq!((stats.entries, stats.misses, stats.hits), (2, 2, 0));
    ///     Ok(())
    /// }
    /// ```
    pub fn precompile(&self, kernels: &[&KernelDescriptor]) -> Result<Vec<Precompiled>, Error> {
        kernels
            .iter()
            .map(|kernel| self.precompile_one(kernel))
            .collect()
    }

    fn precompile_one(&self, kernel: &KernelDescriptor) -> Result<Precompiled, Error> {
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        let resources = validate_kernel(kernel)?;
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
            TextureCounts::of(&resources),
        )?;
        check_kernel_capabilities(
            self.capabilities,
            self.subgroup_sizes,
            &self.adapter_info.name,
            kernel,
        )?;
        check_binding_indices(&self.limits, kernel, &resources)?;
        let (_, outcome) = self.cached_kernel_outcome(kernel, &resources)?;
        #[cfg(not(target_arch = "wasm32"))]
        let elapsed = Some(started.elapsed());
        #[cfg(target_arch = "wasm32")]
        let elapsed = None;
        Ok(Precompiled {
            kernel: kernel.qualified_name().to_string(),
            outcome,
            elapsed,
        })
    }
}

#[cfg(test)]
mod tests {
    use gust::{KernelDescriptor, target::Target};

    use crate::{Error, HeadlessDevice};

    const KERNEL: KernelDescriptor = KernelDescriptor::new(
        "precompile_unit",
        "kernel",
        "precompile_unit_kernel",
        [64, 1, 1],
        &[],
        "[shader(\"compute\")] [numthreads(64, 1, 1)] void precompile_unit_kernel() {}",
    );

    #[test]
    fn a_kernel_built_for_the_cpu_is_refused_before_anything_is_created() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let cpu = KERNEL.with_target(Target::Cpu);
        let error = device.precompile(&[&cpu]).unwrap_err();
        assert!(
            matches!(
                &error,
                Error::KernelTarget {
                    target: Target::Cpu,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(device.pipeline_cache_stats().entries, 0);
    }

    #[test]
    fn a_zero_workgroup_axis_is_refused_before_anything_is_created() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let zero = KernelDescriptor::new(
            "precompile_unit",
            "zero",
            "precompile_unit_zero",
            [64, 0, 1],
            &[],
            "void precompile_unit_zero() {}",
        );
        let error = device.precompile(&[&zero]).unwrap_err();
        assert!(
            matches!(&error, Error::UnsupportedWorkgroupSize([64, 0, 1])),
            "{error:?}"
        );
        assert_eq!(device.pipeline_cache_stats().entries, 0);
    }
}

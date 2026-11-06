//! The target a kernel is built for (D21, D65, D137).
//!
//! A kernel is built for one target in one build: the GPU, or the CPU (D51). `KernelDescriptor::target` records
//! which. The `#[gpu]` macro decides it per kernel at expansion; until routing exists (M8, slice m8.r1) every kernel
//! it emits is [`Target::Gpu`].

use crate::Kernel;

/// The target a kernel is built for. The enum is `#[non_exhaustive]`: a later release may add one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Target {
    /// The kernel runs on a GPU, through a precompiled artifact or the run-time Slang bridge.
    Gpu,
    /// The kernel runs on the CPU, through the sequential `CpuDevice` (D51).
    Cpu,
}

impl Target {
    /// Every target this release knows, in a fixed order.
    pub const ALL: &'static [Target] = &[Target::Gpu, Target::Cpu];

    /// The target's name in a route, a record and a manifest: `gpu` or `cpu`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }

    /// The target named `name` (the inverse of [`Target::name`]), or `None` for any other text.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|target| target.name() == name)
    }
}

/// Marks a kernel that this build runs on a GPU.
///
/// The `#[gpu]` macro implements it for every kernel it builds for [`Target::Gpu`]; the GPU runtime bounds its typed
/// entry points on it, so a kernel routed to the CPU cannot be handed to a GPU device. A hand-written [`Kernel`]
/// implementation that runs on a GPU device adds one line, `impl gust::target::GpuKernel for MyKernel {}`.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "`HeadlessDevice` cannot run the kernel `{Self}`: it is routed to the CPU in this build",
    label = "this kernel has no GPU form in this build",
    note = "call it on a `CpuDevice`, or route its tag to `gpu`",
    note = "a hand-written `Kernel` impl that runs on a GPU adds `impl gust::target::GpuKernel for {Self} {{}}`"
)]
pub trait GpuKernel: Kernel {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_target_has_a_name_that_reads_back() {
        for (target, name) in [(Target::Gpu, "gpu"), (Target::Cpu, "cpu")] {
            assert_eq!(target.name(), name);
            assert_eq!(Target::from_name(name), Some(target));
        }
        assert_eq!(Target::ALL.len(), 2);
        for other in [
            "",
            "Gpu",
            "CPU",
            "native",
            "web",
            "dev",
            "gpu ",
            "cpu+subgroups",
        ] {
            assert_eq!(Target::from_name(other), None, "{other:?}");
        }
    }
}

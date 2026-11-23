//! The build profile the expansion knows: target, capabilities and the CPU-twin switch.

use super::*;

/// What the expansion knows about the build profile (D44, R1-9, M8 D138): the target `GUST_PROFILE` names and the
/// capabilities after it. The macro cannot infer a profile (the build script of `gust` owns the effective one and its
/// release inference), so `target` is `None` when the variable is unset or its value is one the build script refuses.
/// The profile is an argument, never read from the process inside the expander, so expansion tests run in parallel
/// threads without touching the environment.
///
/// `Profile::Cpu` and `Profile::Other` are the two values the expander has always branched on, kept as constants:
/// `Cpu` is exactly `GUST_PROFILE=cpu`, `Other` is an unset or refused variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    /// The target, `None` when the variable is unset (the build script infers it) or refused.
    pub target: Option<ProfileTarget>,
    /// The capabilities after the target.
    pub capabilities: CapabilitySet,
}

/// The target of a [`Profile`]: the four names `GUST_PROFILE` starts with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileTarget {
    /// `dev`: no artifact is required and the run-time bridge is allowed.
    Dev,
    /// `native`: strict, for the native GPU runtime.
    Native,
    /// `web`: strict, for the browser.
    Web,
    /// `cpu`: the production CPU target (D21).
    Cpu,
}

impl ProfileTarget {
    /// The target's name in `GUST_PROFILE` and in a manifest.
    pub(super) const fn text(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Native => "native",
            Self::Web => "web",
            Self::Cpu => "cpu",
        }
    }
}

/// A capability of a device (M8). This is the macro's copy of `gust::capability::Capability`, which it cannot depend
/// on; the names are asserted equal by the `GRAMMAR` table of both crates' tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// Subgroup (wave) operations.
    Subgroups,
}

impl Capability {
    /// Every capability this release knows, in the order of `gust::capability::Capability::ALL`.
    pub const ALL: &'static [Capability] = &[Capability::Subgroups];

    /// The path of the capability in the expansion: `::gust::capability::Capability::Subgroups`.
    pub(super) fn tokens(self) -> TokenStream {
        match self {
            Self::Subgroups => quote!(::gust::capability::Capability::Subgroups),
        }
    }

    /// The capability named `name` in a profile (`subgroups`); `f16` and `atomics64` are reserved and unknown here.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "subgroups" => Some(Self::Subgroups),
            _ => None,
        }
    }

    /// The name of the capability in a profile and in a manifest (`subgroups`): the inverse of [`Self::from_name`].
    pub const fn name(self) -> &'static str {
        match self {
            Self::Subgroups => "subgroups",
        }
    }

    pub(super) const fn bit(self) -> u8 {
        match self {
            Self::Subgroups => 1,
        }
    }
}

/// A set of [`Capability`], small enough that [`Profile`] stays `Copy`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapabilitySet(u8);

impl CapabilitySet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// Whether `capability` is in the set.
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & capability.bit() != 0
    }

    /// The set with `capability` added.
    #[must_use]
    pub const fn with(self, capability: Capability) -> Self {
        Self(self.0 | capability.bit())
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set whose members are the bits of `bits`, one per capability in the order of [`Capability::ALL`] (bit 0 is
    /// `Capability::ALL[0]`). The tests of the selection use bits no capability has yet, to make sets of several.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// The set of the members of both.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every member of the set is in `other`.
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    /// The number of members.
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// The names of the members, in the order of [`Capability::ALL`].
    pub fn names(self) -> Vec<&'static str> {
        Capability::ALL
            .iter()
            .filter(|capability| self.contains(**capability))
            .map(|capability| capability.name())
            .collect()
    }

    /// The paths of the members in the expansion, in the order of [`Capability::ALL`].
    pub(super) fn tokens(self) -> Vec<TokenStream> {
        Capability::ALL
            .iter()
            .filter(|capability| self.contains(**capability))
            .map(|capability| capability.tokens())
            .collect()
    }
}

#[allow(non_upper_case_globals)]
impl Profile {
    /// `GUST_PROFILE=cpu`: the production CPU target (D21). The CPU impl is ungated, no artifact is looked up or
    /// embedded, and nothing of the artifact directory is tracked.
    pub const Cpu: Self = Self {
        target: Some(ProfileTarget::Cpu),
        capabilities: CapabilitySet::EMPTY,
    };

    /// An unset variable, or a value the build script refuses: the CPU impl is a twin behind
    /// `cfg(any(test, gust_dual))` and artifacts are embedded as before.
    pub const Other: Self = Self {
        target: None,
        capabilities: CapabilitySet::EMPTY,
    };
}

impl Profile {
    /// The profile for the value of `GUST_PROFILE` (`None` when unset). A value the grammar refuses is [`Profile::Other`]:
    /// the build script rejects it with the real error, so the macro never reports it.
    pub fn from_value(value: Option<&str>) -> Self {
        value.and_then(Self::parse).unwrap_or(Self::Other)
    }

    /// The grammar of `GUST_PROFILE`: `<target>(+<capability>)*`, exact and lower-case, with no trimming. `None` for
    /// what the build script refuses: an unknown target, an empty, unknown, reserved (`f16`, `atomics64`) or repeated
    /// capability, any capability on `cpu`, and `subgroups` on `web`. The build script (`crates/gust/build.rs`) holds
    /// the mirror of this function with the real messages; `GRAMMAR` in the tests of both is the same table.
    pub fn parse(value: &str) -> Option<Self> {
        let mut pieces = value.split('+');
        let target = match pieces.next()? {
            "dev" => ProfileTarget::Dev,
            "native" => ProfileTarget::Native,
            "web" => ProfileTarget::Web,
            "cpu" => ProfileTarget::Cpu,
            _ => return None,
        };
        let mut capabilities = CapabilitySet::EMPTY;
        for name in pieces {
            let capability = Capability::from_name(name)?;
            if capabilities.contains(capability) {
                return None;
            }
            capabilities = capabilities.with(capability);
        }
        let refused = match target {
            ProfileTarget::Cpu => !capabilities.is_empty(),
            ProfileTarget::Web => capabilities.contains(Capability::Subgroups),
            ProfileTarget::Dev | ProfileTarget::Native => false,
        };
        (!refused).then_some(Self {
            target: Some(target),
            capabilities,
        })
    }

    /// Whether this is the production CPU target (`GUST_PROFILE=cpu`).
    pub fn is_cpu(self) -> bool {
        self == Self::Cpu
    }
}

/// Whether a kernel gets a CPU twin (roadmap M3): the `cpu` option of `#[kernel]`. It is kept beside
/// [`KernelOptions`], not inside it, because only the expander reads it; the Slang emitter never does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CpuTwin {
    /// The default: the kernel runs on `gust::cpu::CpuDevice` under `cfg(any(test, gust_dual))`, or in every build
    /// under `GUST_PROFILE=cpu`.
    #[default]
    Emitted,
    /// `cpu = "unsupported"`: no twin is emitted.
    Unsupported,
}

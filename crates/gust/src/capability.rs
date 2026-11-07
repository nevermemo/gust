//! Optional device capabilities a kernel body may require (M8, D23, D138).
//!
//! A [`Capability`] is a feature of a GPU that not every adapter has and that a kernel variant can depend on. A
//! kernel's `requires` set (`KernelDescriptor::requires`) names the capabilities its selected body needs; the
//! portable body requires none. The names are the spellings of `GUST_PROFILE`'s suffix and of the `#[variant]`
//! attribute.

/// A device capability. The enum is `#[non_exhaustive]`: a later release may add one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Capability {
    /// Subgroup (wave) operations: the lanes of one subgroup exchange values without shared memory.
    Subgroups,
}

impl Capability {
    /// Every capability this release knows, in a fixed order.
    pub const ALL: &'static [Capability] = &[Capability::Subgroups];

    /// The capability's name in a profile, an attribute, a record and a manifest: `subgroups`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Subgroups => "subgroups",
        }
    }

    /// The capability named `name` (the inverse of [`Capability::name`]), or `None` for any other text.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|capability| capability.name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_capability_has_a_name_that_reads_back() {
        assert!(!Capability::ALL.is_empty());
        for &capability in Capability::ALL {
            assert_eq!(Capability::from_name(capability.name()), Some(capability));
        }
        assert_eq!(Capability::Subgroups.name(), "subgroups");
        // Only the exact lower-case names read back; `f16` and `atomics64` are reserved, not capabilities yet.
        for other in [
            "",
            "Subgroups",
            "SUBGROUPS",
            "subgroup",
            "f16",
            "atomics64",
            "subgroups ",
        ] {
            assert_eq!(Capability::from_name(other), None, "{other:?}");
        }
    }

    #[test]
    fn the_names_are_distinct() {
        for (index, left) in Capability::ALL.iter().enumerate() {
            for right in &Capability::ALL[index + 1..] {
                assert_ne!(left.name(), right.name());
            }
        }
    }
}

//! User enums as 32-bit words (plan m14.s2, slice s2e2a, D157).
//!
//! A buffer or a parameter holds an enum as one `u32` word. [`GpuEnum`] is the two-way conversion between an enum and
//! its word, and `#[derive(GpuEnum)]` writes it for a field-less `#[repr(u32)]` enum. The derive also writes
//! `TryFrom<u32>`, which gives [`EnumWordError`] for a word that is no variant. Both are a `match` over the
//! discriminants: no `transmute` and no `as` from a word, so an invalid word can never become a value of the enum.

use core::fmt;

/// An enum that a GPU buffer or parameter holds as one `u32` word.
///
/// Implement it with `#[derive(GpuEnum)]` (the derive of the same name, re-exported next to this trait), never by hand:
/// the derive checks the shape (field-less, `#[repr(u32)]`, literal or implicit discriminants, no generics) and writes
/// both directions as a `match` over the discriminants. The enum must also derive `Clone`, `Copy` and `PartialEq`:
/// the derive does not see them, so the supertraits of this trait enforce them as rustc's own error at the enum. A
/// `cfg` on a variant is applied by rustc before the derive runs, so it is not checked. A hand-written `TryFrom<u32>`
/// or `From<u32>` for the enum conflicts with the derived one (E0119).
///
/// ```
/// use gust::GpuEnum;
///
/// #[derive(Clone, Copy, Debug, PartialEq, GpuEnum)]
/// #[repr(u32)]
/// enum Mode {
///     Linear = 0,
///     Curve = 1,
///     Step = 5,
/// }
///
/// assert_eq!(Mode::Step.to_word(), 5);
/// assert_eq!(Mode::from_word(1), Some(Mode::Curve));
/// assert_eq!(Mode::from_word(2), None);
/// assert_eq!(Mode::try_from(2).unwrap_err().word(), 2);
/// ```
///
/// An enum that does not derive `PartialEq` (or `Clone` and `Copy`) is rustc's error, from the supertraits:
///
/// ```compile_fail,E0277
/// #[derive(Clone, Copy, gust::GpuEnum)]
/// #[repr(u32)]
/// enum Mode {
///     Linear,
/// }
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a GPU enum",
    label = "add `#[derive(GpuEnum)]` to the enum",
    note = "the enum must be field-less and `#[repr(u32)]`, and derive `Clone`, `Copy` and `PartialEq`"
)]
pub trait GpuEnum: Copy + PartialEq + 'static {
    /// The variant whose discriminant is `word`, or `None` when no variant has it.
    fn from_word(word: u32) -> Option<Self>;

    /// The discriminant of the variant, as the word a buffer holds.
    fn to_word(self) -> u32;
}

/// The error of `TryFrom<u32>` for a [`GpuEnum`]: the word is no variant of the enum. It carries the word.
///
/// The enum is not named, because the `Result` that holds the error already has the type it was converted to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumWordError {
    word: u32,
}

impl EnumWordError {
    /// The error for `word`. The derive of [`GpuEnum`] calls it; call it only for a word you just found to be no variant.
    #[must_use]
    pub const fn new(word: u32) -> Self {
        Self { word }
    }

    /// The word that is no variant.
    #[must_use]
    pub const fn word(self) -> u32 {
        self.word
    }
}

impl fmt::Display for EnumWordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "the word {} is no variant of the enum",
            self.word
        )
    }
}

impl core::error::Error for EnumWordError {}

#[cfg(test)]
mod tests {
    use crate::{EnumWordError, GpuEnum};

    /// Explicit discriminants with gaps, then implicit ones after an explicit one.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, GpuEnum)]
    #[repr(u32)]
    enum Gapped {
        Low = 0,
        Mid = 3,
        Next,
        High = 10,
    }

    /// The largest word is a variant: `u32::MAX` must not be taken for "no variant".
    #[derive(Clone, Copy, Debug, PartialEq, Eq, GpuEnum)]
    #[repr(u32)]
    enum Edge {
        First,
        Last = 4_294_967_295,
    }

    const GAPPED: [(Gapped, u32); 4] = [
        (Gapped::Low, 0),
        (Gapped::Mid, 3),
        (Gapped::Next, 4),
        (Gapped::High, 10),
    ];

    #[test]
    fn every_variant_round_trips_through_its_word() {
        for (variant, word) in GAPPED {
            assert_eq!(variant.to_word(), word);
            assert_eq!(variant.to_word(), variant as u32);
            assert_eq!(Gapped::from_word(word), Some(variant));
            assert_eq!(Gapped::try_from(word), Ok(variant));
        }
    }

    #[test]
    fn a_word_that_is_no_variant_is_refused_with_the_word() {
        for word in [1, 2, 5, 9, 11, 1 << 31, u32::MAX] {
            assert_eq!(Gapped::from_word(word), None, "{word}");
            let error = Gapped::try_from(word).unwrap_err();
            assert_eq!(error, EnumWordError::new(word));
            assert_eq!(error.word(), word);
        }
    }

    #[test]
    fn the_largest_word_is_a_variant_where_one_has_it() {
        assert_eq!(Edge::Last.to_word(), u32::MAX);
        assert_eq!(Edge::Last.to_word(), Edge::Last as u32);
        assert!(Edge::First == Edge::from_word(0).unwrap());
        assert_eq!(Edge::from_word(u32::MAX), Some(Edge::Last));
        assert_eq!(Edge::try_from(0), Ok(Edge::First));
        assert_eq!(Edge::from_word(1), None);
    }

    #[test]
    fn the_error_names_the_word() {
        assert_eq!(
            EnumWordError::new(7).to_string(),
            "the word 7 is no variant of the enum"
        );
    }
}

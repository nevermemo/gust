//! User structs as the rows of a batch (plan m14.s2, slice s2s, part a, D279).
//!
//! A row is a free-standing `#[repr(C)]` struct of scalar, vector and matrix fields. [`GpuStruct`] marks one that
//! `#[derive(GpuStruct)]` has checked; the derive also writes the [`GpuPod`](crate::GpuPod) impl, so the host half of a
//! batch element is the blanket `Element` impl for any `GpuPod` type and needs nothing of its own.

use crate::GpuPod;

/// A struct that `#[derive(GpuStruct)]` checked: `#[repr(C)]`, named fields, each an `f32`, `u32`, `i32`, vector or
/// matrix of `gust::math` that a buffer may hold, and no byte of padding.
///
/// Implement it with the derive of the same name (re-exported next to this trait), never by hand: the derive writes the
/// [`GpuPod`] impl with its layout assertions, and a hand-written marker would claim a check that was never made. The
/// struct must also derive `Clone` and `Copy`: the derive does not see them, so the supertraits of [`GpuPod`] enforce
/// them as rustc's own error at the struct. Write the pad fields a layout needs yourself (`_pad0: u32`); a gap is
/// GUST-E0208 and the message says how many to add and where.
///
/// ```
/// use gust::{GpuPod, GpuStruct, prelude::Vec2};
///
/// #[derive(Clone, Copy, GpuStruct)]
/// #[repr(C)]
/// struct Row {
///     pos: Vec2,
///     life: f32,
///     flags: u32,
/// }
///
/// let layout = <Row as GpuPod>::LAYOUT;
/// assert_eq!((layout.size, layout.alignment, layout.stride), (16, 8, 16));
/// ```
///
/// A struct that does not derive `Copy` is rustc's error, from the supertraits:
///
/// ```compile_fail,E0277
/// use gust::GpuStruct;
///
/// #[derive(Clone, GpuStruct)]
/// #[repr(C)]
/// struct Row {
///     life: f32,
/// }
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a GPU struct",
    label = "add `#[derive(GpuStruct)]` to the struct",
    note = "the struct must be `#[repr(C)]`, without padding, of scalar, vector and matrix fields, and derive `Clone` and `Copy`"
)]
pub trait GpuStruct: GpuPod + Sealed {}

/// The supertrait that keeps `GpuStruct` for the derive: not for hand-written impls.
#[doc(hidden)]
pub trait Sealed {}

#[cfg(test)]
mod tests {
    // The fields of these structs are read through `LAYOUT` and the byte view only.
    #![allow(dead_code)]

    use core::mem::{align_of, offset_of, size_of};

    use crate::batch::Element;
    use crate::math::{IVec2, IVec4, Mat2, Mat4, UVec2, UVec4, Vec2, Vec4};
    use crate::{GpuPod, GpuStruct, TypeLayoutKind};

    /// Layout (i): two `Vec2`, an `f32` and a `u32`, 24 bytes.
    #[derive(Clone, Copy, GpuStruct)]
    #[repr(C)]
    struct Packed {
        a: Vec2,
        b: Vec2,
        c: f32,
        d: u32,
    }

    /// Layout (ii): explicit pad fields, 32 bytes.
    #[derive(Clone, Copy, GpuStruct)]
    #[repr(C)]
    struct Padded {
        a: Vec4,
        b: Vec2,
        _pad0: f32,
        _pad1: f32,
    }

    /// Layout (iii): a matrix and scalars up to a multiple of 16, 80 bytes.
    #[derive(Clone, Copy, GpuStruct)]
    #[repr(C)]
    struct Matrixed {
        m: Mat4,
        a: f32,
        b: u32,
        c: i32,
        _pad: u32,
    }

    /// One field alone, and a field of each remaining kind.
    #[derive(Clone, Copy, GpuStruct)]
    #[repr(C)]
    struct Single {
        only: UVec4,
    }

    #[derive(Clone, Copy, GpuStruct)]
    #[repr(C)]
    struct Mixed {
        a: IVec2,
        b: i32,
        _pad: i32,
    }

    /// Every accepted field type once (rustc checks the derive's table against them), no padding.
    #[derive(Clone, Copy, GpuStruct)]
    #[repr(C)]
    struct All {
        a: Vec4,
        b: IVec4,
        c: UVec2,
        d: IVec2,
        e: Vec2,
        f: Mat2,
        g: f32,
        h: u32,
        i: i32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
        j: UVec4,
    }
    fn needs_struct<T: GpuStruct>() {}

    /// The reported layout: the size, alignment and stride of the struct and the name and offset of every field, equal
    /// to what rustc computed (`offset_of!`).
    fn assert_layout<T: GpuPod>(size: usize, alignment: usize, offsets: &[(&str, usize)]) {
        let layout = T::LAYOUT;
        assert_eq!(layout.size as usize, size_of::<T>());
        assert_eq!(layout.size as usize, size);
        assert_eq!(layout.alignment as usize, align_of::<T>());
        assert_eq!(layout.alignment as usize, alignment);
        assert_eq!(layout.stride as usize, size);
        assert!(layout.is_storage_v2());
        let TypeLayoutKind::Struct(fields) = layout.kind else {
            panic!("a struct layout");
        };
        let reported: Vec<(&str, usize)> = fields
            .iter()
            .map(|field| (field.name, field.offset as usize))
            .collect();
        assert_eq!(reported, offsets);
    }

    #[test]
    fn the_derive_reports_the_std430_layout_of_a_row_with_no_padding() {
        needs_struct::<Packed>();
        assert_layout::<Packed>(24, 8, &[("a", 0), ("b", 8), ("c", 16), ("d", 20)]);
        assert_eq!(offset_of!(Packed, c), 16);
    }

    #[test]
    fn the_derive_reports_the_layout_of_a_row_with_explicit_pad_fields() {
        needs_struct::<Padded>();
        assert_layout::<Padded>(32, 16, &[("a", 0), ("b", 16), ("_pad0", 24), ("_pad1", 28)]);
    }

    #[test]
    fn the_derive_reports_the_layout_of_a_matrix_row() {
        needs_struct::<Matrixed>();
        assert_layout::<Matrixed>(
            80,
            16,
            &[("m", 0), ("a", 64), ("b", 68), ("c", 72), ("_pad", 76)],
        );
    }

    #[test]
    fn the_derive_reports_the_layout_of_single_field_and_mixed_rows() {
        needs_struct::<Single>();
        needs_struct::<Mixed>();
        assert_layout::<Single>(16, 16, &[("only", 0)]);
        assert_layout::<Mixed>(16, 8, &[("a", 0), ("b", 8), ("_pad", 12)]);
    }

    #[test]
    fn every_accepted_field_type_compiles_with_the_derives_table() {
        needs_struct::<All>();
        assert_layout::<All>(
            112,
            16,
            &[
                ("a", 0),
                ("b", 16),
                ("c", 32),
                ("d", 40),
                ("e", 48),
                ("f", 56),
                ("g", 72),
                ("h", 76),
                ("i", 80),
                ("_pad0", 84),
                ("_pad1", 88),
                ("_pad2", 92),
                ("j", 96),
            ],
        );
    }

    #[test]
    fn a_derived_row_is_a_host_element_through_the_blanket_impl() {
        let rows = [
            Packed {
                a: Vec2::new(1.0, 2.0),
                b: Vec2::new(3.0, 4.0),
                c: 5.0,
                d: 6,
            },
            Packed {
                a: Vec2::new(7.0, 8.0),
                b: Vec2::new(9.0, 10.0),
                c: 11.0,
                d: 12,
            },
        ];
        let packed = <Packed as Element>::pack(&rows);
        assert_eq!(packed.len(), 2);
        let back = <Packed as Element>::unpack(packed.into_owned()).unwrap();
        assert_eq!(
            crate::pod_slice_as_bytes(&back),
            crate::pod_slice_as_bytes(&rows)
        );
        // The device holds `LAYOUT.size` bytes per row: the byte view of the pack is rows end to end.
        assert_eq!(
            crate::pod_slice_as_bytes(&rows).len(),
            2 * <Packed as GpuPod>::LAYOUT.stride as usize
        );
    }
}

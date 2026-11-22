use crate::Error;
use gust::{BindingKind, GpuPod, ResourceLayout};

/// Whether `layout` has no padding byte anywhere: a struct's field sizes add up to its size (recursively), a matrix's
/// columns are packed. [`gust::TypeLayout::is_storage_v2`] accepts padded structs, which are not `GpuPod` types:
/// `pod_slice_as_bytes` would read their uninitialised padding.
pub(crate) const fn padding_free(layout: gust::TypeLayout) -> bool {
    match layout.kind {
        gust::TypeLayoutKind::Scalar(_) | gust::TypeLayoutKind::Vector { .. } => true,
        gust::TypeLayoutKind::Matrix {
            rows,
            cols,
            column_stride,
            ..
        } => column_stride == rows * 4 && column_stride * cols == layout.size,
        gust::TypeLayoutKind::Struct(fields) => {
            let mut sum = 0_u32;
            let mut index = 0;
            while index < fields.len() {
                let field = fields[index].ty;
                let Some(next) = sum.checked_add(field.size) else {
                    return false;
                };
                if !padding_free(field) {
                    return false;
                }
                sum = next;
                index += 1;
            }
            sum == layout.size
        }
        // Elements back to back: padding-free when the element is, its stride is its size and the sizes add up.
        gust::TypeLayoutKind::Array { element, count } => {
            if element.stride != element.size || !padding_free(*element) {
                return false;
            }
            match element.size.checked_mul(count) {
                Some(total) => total == layout.size,
                None => false,
            }
        }
        // A kind this release does not know cannot be shown padding-free, so it is not a `GpuPod` layout.
        _ => false,
    }
}

/// Panic unless `T`'s `GpuPod::LAYOUT` is StorageV2 (StorageV1 is its scalar sub-case), has no padding, and agrees
/// with Rust's size and alignment. The message keeps the StorageV1 wording that tests pin.
/// Every typed allocation (`create_typed_buffer`, pool buffers) runs it first, so a hand-written
/// `GpuPod` that lies, or that describes padding, is refused at creation instead of failing later in wgpu or at
/// dispatch (or uploading uninitialised bytes).
pub(crate) fn assert_pod_matches_storage_layout<T: GpuPod>() {
    assert!(
        T::LAYOUT.is_storage_v2()
            && padding_free(T::LAYOUT)
            && T::LAYOUT.size as usize == size_of::<T>()
            && T::LAYOUT.alignment as usize == align_of::<T>(),
        "GpuPod implementation must match the Rust StorageV1 representation \
         (StorageV2 for vectors and matrices, without padding)"
    );
}

/// Whether `T` can be a uniform: a `GpuPod` type whose layout is `Std140V1`, agrees with Rust's size and alignment, and
/// has no padding byte. The padding rule is the one that keeps `pod_slice_as_bytes` sound: a `Std140V1` struct is
/// rounded up to 16 bytes, so a struct whose fields do not already fill a multiple of 16 has trailing padding that
/// Rust leaves uninitialised, and the upload would read it. The remedy is in the message: spell the padding out as
/// fields. (Copying the value into a zeroed buffer and skipping the gaps would also be sound; it is not done, so that
/// the bytes on the GPU are exactly the bytes of the Rust value.)
pub(crate) fn check_uniform_type<T: GpuPod>() -> Result<(), Error> {
    let layout = T::LAYOUT;
    let refuse = |reason: &'static str| {
        Err(Error::InvalidUniformType {
            type_name: layout.name,
            reason,
        })
    };
    if !layout.is_std140_v1() {
        return refuse(
            "its layout is not Std140V1 (a constant buffer takes scalars, 2- and 4-wide vectors, 4x4 matrices and \
             structs of them, each struct rounded up to 16 bytes; 3-wide vectors and 2x2 matrices do not qualify)",
        );
    }
    if layout.size as usize != size_of::<T>() || layout.alignment as usize != align_of::<T>() {
        return refuse(
            "its Std140V1 size or alignment differs from the Rust type's; a struct is rounded up to 16 bytes in a \
             constant buffer, so add explicit padding fields (and `repr(C, align(16))` if needed) until the Rust size \
             is the Std140V1 size",
        );
    }
    if !padding_free(layout) {
        return refuse(
            "it has padding bytes, which Rust leaves uninitialised; add explicit padding fields so that the fields \
             fill the whole struct (a Std140V1 struct is rounded up to 16 bytes)",
        );
    }
    Ok(())
}

/// Whether `layout` is exactly `wgpu::util::DispatchIndirectArgs`: three `u32`s.
pub(crate) fn is_indirect_args_layout(layout: gust::TypeLayout) -> bool {
    use gust::{ScalarKind, TypeLayoutKind};
    layout.size == 12
        && layout.stride == 12
        && matches!(layout.kind, TypeLayoutKind::Struct(fields) if fields.len() == 3
            && fields.iter().all(|field| matches!(field.ty.kind, TypeLayoutKind::Scalar(ScalarKind::U32))))
}

/// The resource layout a typed buffer of element layout `element` presents: `StorageV1` when the element is a scalar
/// or a struct of scalars (the macro describes those parameters as `StorageV1`), else `StorageV2` (vectors, matrices).
pub(crate) fn buffer_resource_layout(element: gust::TypeLayout) -> ResourceLayout {
    if element.is_storage_v1() {
        ResourceLayout::storage_v1(element)
    } else {
        ResourceLayout::storage_v2(element)
    }
}

/// The resource layout a binding of `kind` and element layout `element` presents: a storage buffer's
/// (`buffer_resource_layout`), or `Std140V1` for a uniform.
pub(crate) fn binding_resource_layout(
    kind: BindingKind,
    element: gust::TypeLayout,
) -> ResourceLayout {
    if kind == BindingKind::Uniform {
        ResourceLayout::std140_v1(element)
    } else {
        buffer_resource_layout(element)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use gust::GpuPod;

    /// A hand-written `GpuPod` whose layout claims alignment 8 for a four-byte value.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LyingAlignment(u32);

    // SAFETY: deliberately dishonest, to exercise the runtime assertion; it is never uploaded.
    unsafe impl GpuPod for LyingAlignment {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout {
            name: "LyingAlignment",
            size: 4,
            alignment: 8,
            stride: 4,
            kind: gust::TypeLayoutKind::Scalar(gust::ScalarKind::U32),
        };
    }

    /// A hand-written `GpuPod` whose layout is a valid StorageV1 struct of two `u32`s (size 8)
    /// while the Rust type is four bytes.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LyingSize(u32);

    // SAFETY: deliberately dishonest, to exercise the runtime assertion; it is never uploaded.
    unsafe impl GpuPod for LyingSize {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout {
            name: "LyingSize",
            size: 8,
            alignment: 4,
            stride: 8,
            kind: gust::TypeLayoutKind::Struct(&[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: gust::TypeLayout::scalar("u32", gust::ScalarKind::U32),
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 4,
                    ty: gust::TypeLayout::scalar("u32", gust::ScalarKind::U32),
                },
            ]),
        };
    }

    /// Right size (4) and alignment (4), but the only field sits at offset 2, so the layout is not
    /// StorageV1: only the `is_storage_v1()` clause of the assertion can refuse it.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LyingOffset(u32);

    // SAFETY: deliberately dishonest, to exercise the runtime assertion; it is never uploaded.
    unsafe impl GpuPod for LyingOffset {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout {
            name: "LyingOffset",
            size: 4,
            alignment: 4,
            stride: 4,
            kind: gust::TypeLayoutKind::Struct(&[gust::StructFieldLayout {
                name: "a",
                offset: 2,
                ty: gust::TypeLayout::scalar("u32", gust::ScalarKind::U32),
            }]),
        };
    }

    #[test]
    fn the_layout_assertion_accepts_honest_types() {
        assert_pod_matches_storage_layout::<u32>();
        assert_pod_matches_storage_layout::<f32>();
    }

    #[test]
    #[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
    fn the_layout_assertion_refuses_a_layout_that_is_not_storage_v1() {
        assert_pod_matches_storage_layout::<LyingAlignment>();
    }

    #[test]
    #[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
    fn the_layout_assertion_refuses_a_bad_offset_with_the_right_size_and_alignment() {
        assert_eq!(LyingOffset::LAYOUT.size as usize, size_of::<LyingOffset>());
        assert_eq!(
            LyingOffset::LAYOUT.alignment as usize,
            align_of::<LyingOffset>()
        );
        assert!(!LyingOffset::LAYOUT.is_storage_v1());
        assert_pod_matches_storage_layout::<LyingOffset>();
    }

    #[test]
    #[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
    fn the_layout_assertion_refuses_a_size_that_differs_from_rust() {
        assert!(LyingSize::LAYOUT.is_storage_v1());
        assert_pod_matches_storage_layout::<LyingSize>();
    }

    /// An honest hand-written `GpuPod` over vectors: `Vec4` (16, align 16) then two `Vec2` at 16 and 24, size 32, no
    /// padding.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct HonestBody {
        a: gust::math::Vec4,
        b: gust::math::Vec2,
        c: gust::math::Vec2,
    }

    // SAFETY: `repr(C)` over three `repr(C)` vector types, no padding (16 + 8 + 8 = 32), offsets 0, 16, 24.
    unsafe impl GpuPod for HonestBody {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "HonestBody",
            32,
            16,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: <gust::math::Vec4 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 16,
                    ty: <gust::math::Vec2 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "c",
                    offset: 24,
                    ty: <gust::math::Vec2 as GpuPod>::LAYOUT,
                },
            ],
        );
    }

    /// A hand-written `GpuPod` that honestly describes padding: a `u32` then a `Vec2` (align 8) at offset 8, so
    /// bytes 4 to 7 are padding. Size 16 and alignment 8 are what Rust reports and what `is_storage_v2` accepts.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct PaddedPair {
        a: u32,
        b: gust::math::Vec2,
    }

    // SAFETY: deliberately breaks the no-padding rule, to exercise the runtime assertion; it is never uploaded.
    unsafe impl GpuPod for PaddedPair {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "PaddedPair",
            16,
            8,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: gust::TypeLayout::scalar("u32", gust::ScalarKind::U32),
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 8,
                    ty: <gust::math::Vec2 as GpuPod>::LAYOUT,
                },
            ],
        );
    }

    #[test]
    fn the_layout_assertion_accepts_vectors_matrices_and_a_padding_free_struct_of_vectors() {
        assert_pod_matches_storage_layout::<gust::math::Vec2>();
        assert_pod_matches_storage_layout::<gust::math::Vec4>();
        assert_pod_matches_storage_layout::<gust::math::Mat2>();
        assert_pod_matches_storage_layout::<gust::math::Mat4>();
        assert_pod_matches_storage_layout::<HonestBody>();
    }

    #[test]
    #[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
    fn the_layout_assertion_refuses_a_struct_with_padding_even_when_the_layout_is_storage_v2() {
        assert!(PaddedPair::LAYOUT.is_storage_v2());
        assert_eq!(PaddedPair::LAYOUT.size as usize, size_of::<PaddedPair>());
        assert_eq!(
            PaddedPair::LAYOUT.alignment as usize,
            align_of::<PaddedPair>()
        );
        assert!(!padding_free(PaddedPair::LAYOUT));
        assert_pod_matches_storage_layout::<PaddedPair>();
    }

    /// A padding-free `Std140V1` struct: `Vec4` then three `f32`s and one `f32` of explicit padding, size 32.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct FullStd140 {
        a: gust::math::Vec4,
        b: [f32; 4],
    }

    // SAFETY: `repr(C)`, a `Vec4` (16 bytes, align 16) then four `f32`s at 16, 20, 24 and 28: no padding, size 32.
    unsafe impl GpuPod for FullStd140 {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "FullStd140",
            32,
            16,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: <gust::math::Vec4 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "b0",
                    offset: 16,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
                gust::StructFieldLayout {
                    name: "b1",
                    offset: 20,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
                gust::StructFieldLayout {
                    name: "b2",
                    offset: 24,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
                gust::StructFieldLayout {
                    name: "b3",
                    offset: 28,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
            ],
        );
    }

    /// An honest `Std140V1` layout and honest Rust size and alignment, but padded: a `Vec4` then one `f32`, so bytes 20
    /// to 31 are padding.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct PaddedStd140 {
        a: gust::math::Vec4,
        b: f32,
    }

    // SAFETY: deliberately breaks the no-padding rule, to exercise the uniform check; it is never uploaded.
    unsafe impl GpuPod for PaddedStd140 {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "PaddedStd140",
            32,
            16,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: <gust::math::Vec4 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 16,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
            ],
        );
    }

    /// A struct of one `f32`: its `Std140V1` size is 16 but Rust's is 4.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct OneFloat {
        a: f32,
    }

    // SAFETY: deliberately dishonest about the size, to exercise the uniform check; it is never uploaded.
    unsafe impl GpuPod for OneFloat {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "OneFloat",
            16,
            16,
            &[gust::StructFieldLayout {
                name: "a",
                offset: 0,
                ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
            }],
        );
    }

    fn uniform_refusal<T: GpuPod>() -> (&'static str, &'static str) {
        match check_uniform_type::<T>() {
            Err(Error::InvalidUniformType { type_name, reason }) => (type_name, reason),
            other => panic!("expected an InvalidUniformType error, got {other:?}"),
        }
    }

    #[test]
    fn a_uniform_type_is_a_padding_free_std140_v1_type_with_rusts_size_and_alignment() {
        use gust::math::{Mat4, Vec2, Vec4};
        assert!(check_uniform_type::<f32>().is_ok());
        assert!(check_uniform_type::<Vec2>().is_ok());
        assert!(check_uniform_type::<Vec4>().is_ok());
        assert!(check_uniform_type::<Mat4>().is_ok());
        assert!(check_uniform_type::<FullStd140>().is_ok());
        assert!(check_uniform_type::<HonestBody>().is_ok());
    }

    #[test]
    fn a_padded_uniform_type_is_refused_with_the_fix_in_the_message() {
        let (name, reason) = uniform_refusal::<PaddedStd140>();
        assert_eq!(name, "PaddedStd140");
        assert!(reason.contains("padding bytes"), "{reason}");
        assert!(reason.contains("explicit padding fields"), "{reason}");
        let text = check_uniform_type::<PaddedStd140>()
            .unwrap_err()
            .to_string();
        assert!(
            text.starts_with("`PaddedStd140` cannot be a uniform: "),
            "{text}"
        );
    }

    #[test]
    fn a_uniform_type_whose_layout_is_not_std140_v1_or_not_rusts_is_refused() {
        use gust::math::Mat2;
        // StorageV2 only: a 2x2 matrix, a 4-aligned struct of scalars (not rounded to 16), a padded storage struct.
        for (name, reason) in [
            uniform_refusal::<Mat2>(),
            uniform_refusal::<LyingSize>(),
            uniform_refusal::<PaddedPair>(),
        ] {
            assert!(reason.contains("not Std140V1"), "{name}: {reason}");
        }
        let (name, reason) = uniform_refusal::<OneFloat>();
        assert_eq!(name, "OneFloat");
        assert!(reason.contains("size or alignment"), "{reason}");
    }
}

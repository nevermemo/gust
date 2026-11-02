//! The vector and matrix types of `gust::math`, as the macro knows them (slice m6.m1).
//!
//! The macro has no type information, so everything here is by name: one table gives each spelling its Slang type, its
//! Rust size and alignment (the `#[repr(C, align(..))]` layout of `gust::math`, the same numbers as `StorageV2`) and
//! whether it may be a struct field or a resource element. The 3-wide types are locals and helper values only.

/// What a type name means to the macro.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MathType {
    /// The Rust spelling, as the prelude exports it.
    pub(crate) name: &'static str,
    /// The Slang spelling.
    pub(crate) slang: &'static str,
    /// Size in bytes of the Rust type.
    pub(crate) size: u32,
    /// Alignment in bytes of the Rust type.
    pub(crate) alignment: u32,
    /// Whether the type is a matrix (it has `IDENTITY`, `from_cols` and `from_rows`, and no `new` or `splat`).
    pub(crate) matrix: bool,
    /// Whether the type may be a struct field or a resource element: false for the 3-wide types, which have
    /// stride 16 in Slang's buffers but size 12 in Rust.
    pub(crate) buffer: bool,
}

const fn vector(
    name: &'static str,
    slang: &'static str,
    size: u32,
    alignment: u32,
    buffer: bool,
) -> MathType {
    MathType {
        name,
        slang,
        size,
        alignment,
        matrix: false,
        buffer,
    }
}

const fn matrix(
    name: &'static str,
    slang: &'static str,
    size: u32,
    alignment: u32,
    buffer: bool,
) -> MathType {
    MathType {
        name,
        slang,
        size,
        alignment,
        matrix: true,
        buffer,
    }
}

/// Every vector and matrix spelling, in the order the documentation lists them.
pub(crate) const MATH_TYPES: &[MathType] = &[
    vector("Vec2", "float2", 8, 8, true),
    vector("Vec3", "float3", 12, 4, false),
    vector("Vec4", "float4", 16, 16, true),
    vector("IVec2", "int2", 8, 8, true),
    vector("IVec4", "int4", 16, 16, true),
    vector("UVec2", "uint2", 8, 8, true),
    vector("UVec3", "uint3", 12, 4, false),
    vector("UVec4", "uint4", 16, 16, true),
    matrix("Mat2", "float2x2", 16, 8, true),
    matrix("Mat3", "float3x3", 36, 4, false),
    matrix("Mat4", "float4x4", 64, 16, true),
];

/// The scalar spellings the macro maps itself; each has size and alignment 4.
pub(crate) const SCALARS: &[&str] = &["f32", "float", "i32", "int", "u32", "uint"];

/// The table entry for an unrawed type name.
pub(crate) fn math_type(name: &str) -> Option<&'static MathType> {
    MATH_TYPES.iter().find(|entry| entry.name == name)
}

/// Whether a struct may not be named `name`: the dialect maps these names to Slang types itself, so a user struct
/// of the name would be emitted as the builtin while rustc resolves the user's type (E0203).
pub(crate) fn is_reserved_type_name(name: &str) -> bool {
    matches!(name, "bool" | "float" | "int" | "uint") || math_type(name).is_some()
}

/// Size and alignment of a scalar spelling.
pub(crate) fn scalar_layout(name: &str) -> Option<(u32, u32)> {
    SCALARS.contains(&name).then_some((4, 4))
}

/// The math type and member of a `Type::member` path that names a constructor or constant `gust::math` has: two
/// unqualified segments, no arguments, the first a vector or matrix spelling (raw or not) and the member one of its
/// associated items (`new` and `splat` of a vector, `from_cols`, `from_rows` and `IDENTITY` of a matrix).
pub(crate) fn associated_item(path: &syn::Path) -> Option<(&'static MathType, &'static str)> {
    use syn::ext::IdentExt as _;

    if path.leading_colon.is_some() || path.segments.len() != 2 {
        return None;
    }
    let (first, second) = (&path.segments[0], &path.segments[1]);
    if !first.arguments.is_empty() || !second.arguments.is_empty() {
        return None;
    }
    let math = math_type(&first.ident.unraw().to_string())?;
    let members: &[&'static str] = if math.matrix {
        &["IDENTITY", "from_cols", "from_rows"]
    } else {
        &["new", "splat"]
    };
    let member = members
        .iter()
        .copied()
        .find(|member| second.ident == member)?;
    Some((math, member))
}

/// The number of columns (and rows) of a square matrix, or the width of a vector.
pub(crate) fn width(ty: &MathType) -> usize {
    ty.name
        .chars()
        .last()
        .and_then(|digit| digit.to_digit(10))
        .map_or(0, |digit| digit as usize)
}

/// Whether `name` is a swizzle method: two to four of `x`, `y`, `z`, `w`.
pub(crate) fn is_swizzle(name: &str) -> bool {
    (2..=4).contains(&name.len())
        && name
            .bytes()
            .all(|byte| matches!(byte, b'x' | b'y' | b'z' | b'w'))
}

/// The matrix and vector methods other than swizzles, with their argument counts.
pub(crate) fn method_arity(name: &str) -> Option<usize> {
    match name {
        "dot" | "mul_vec2" | "mul_vec3" | "mul_vec4" | "mul_mat2" | "mul_mat3" | "mul_mat4" => {
            Some(1)
        }
        "transpose" => Some(0),
        _ => None,
    }
}

/// Size and alignment of a field type known by name: a scalar, a vector or a matrix. Module structs are the
/// caller's business.
pub(crate) fn known_layout(name: &str) -> Option<(u32, u32)> {
    scalar_layout(name).or_else(|| math_type(name).map(|entry| (entry.size, entry.alignment)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_by_family_and_has_no_duplicates() {
        let mut names: Vec<&str> = MATH_TYPES.iter().map(|entry| entry.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), MATH_TYPES.len());
    }

    /// The table is checked against itself only (this crate cannot depend on `gust`): sizes are whole alignments and
    /// the 3-wide types are the ones that are not buffer types. That the numbers equal `size_of` and `align_of` of
    /// `gust::math` is enforced where the types are used, by the `const` assertions of `gpu_pod_impl`.
    #[test]
    fn table_sizes_are_whole_alignments_and_only_three_wide_types_are_not_buffer_types() {
        for entry in MATH_TYPES {
            if !entry.buffer {
                assert!(entry.name.contains('3'), "{}", entry.name);
                continue;
            }
            assert!(entry.alignment.is_power_of_two(), "{}", entry.name);
            assert_eq!(entry.size % entry.alignment, 0, "{}", entry.name);
        }
    }

    #[test]
    fn swizzle_names_are_two_to_four_xyzw_letters() {
        assert!(is_swizzle("xy"));
        assert!(is_swizzle("wzyx"));
        assert!(is_swizzle("xxxx"));
        assert!(!is_swizzle("x"));
        assert!(!is_swizzle("xyzwx"));
        assert!(!is_swizzle("xr"));
        assert!(!is_swizzle("XY"));
    }
}

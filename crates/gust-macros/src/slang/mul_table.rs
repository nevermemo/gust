//! The product table behind `*` and `*=` in a module that names a matrix (slice m6.m2, M6 plan 2.1).
//!
//! This file is plain data and string templates. Three things read it: the Slang overload text (`slang/mod.rs`), the
//! generator of the `Mul` and `MulAssign` impls of `gust::math` (`mul_impls.rs`), and the macro-crate tests, which
//! check that the `products` golden holds every row (`tests/fixtures/products.rs`, compiled by slangc in
//! `crates/gust/tests/diagnostics.rs`).
//!
//! One row is one operand pair of `lhs * rhs`. The Slang side is the overload `__gust_mul(lhs, rhs)`: a module that
//! names `Mat2`, `Mat3` or `Mat4` lowers every `*` to that call (the macro sees syntax and not types, and Slang's own
//! `*` is component-wise on two matrices where Rust's `Mat * Mat` is the matrix product). The Rust side is the `Mul`
//! impl of the same pair, and a `MulAssign` impl for every row whose output is its left type, except the products.

/// How a row is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Two scalars: Rust's own operator, so there is no impl, only the overload that makes `3 * 2` resolve.
    Scalar,
    /// Component-wise on vectors, and a matrix scaled by a scalar: Slang's `*` and a loop over the components.
    Ew,
    /// A matrix times a matrix or a vector: Slang's `mul(a, b)`, the Rust methods `mul_matN` and `mul_vecN`. No
    /// `MulAssign` (a reader cannot tell the operand order of `m *= n`).
    MatMul,
}

/// One operand pair, by Rust spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub left: &'static str,
    pub right: &'static str,
    pub output: &'static str,
    pub kind: Kind,
}

impl Row {
    const fn scalar(name: &'static str) -> Self {
        Self {
            left: name,
            right: name,
            output: name,
            kind: Kind::Scalar,
        }
    }

    const fn ew(left: &'static str, right: &'static str, output: &'static str) -> Self {
        Self {
            left,
            right,
            output,
            kind: Kind::Ew,
        }
    }

    const fn product(left: &'static str, right: &'static str, output: &'static str) -> Self {
        Self {
            left,
            right,
            output,
            kind: Kind::MatMul,
        }
    }

    /// Whether the row gets a `MulAssign` impl: the output is the left type, and the row is not a matrix product.
    pub fn assigns(&self) -> bool {
        self.kind == Kind::Ew && self.output == self.left && !is_scalar(self.left)
    }
}

/// Every operand pair: 3 scalar rows, 8 vector types times 3, and 3 matrices times 4.
pub const MUL_PAIRS: &[Row] = &[
    // Scalars with themselves: Rust's own operators, so the generated impls skip these rows.
    Row::scalar("f32"),
    Row::scalar("i32"),
    Row::scalar("u32"),
    // Vectors with themselves and with their scalar on both sides.
    Row::ew("Vec2", "Vec2", "Vec2"),
    Row::ew("Vec2", "f32", "Vec2"),
    Row::ew("f32", "Vec2", "Vec2"),
    Row::ew("Vec3", "Vec3", "Vec3"),
    Row::ew("Vec3", "f32", "Vec3"),
    Row::ew("f32", "Vec3", "Vec3"),
    Row::ew("Vec4", "Vec4", "Vec4"),
    Row::ew("Vec4", "f32", "Vec4"),
    Row::ew("f32", "Vec4", "Vec4"),
    Row::ew("IVec2", "IVec2", "IVec2"),
    Row::ew("IVec2", "i32", "IVec2"),
    Row::ew("i32", "IVec2", "IVec2"),
    Row::ew("IVec4", "IVec4", "IVec4"),
    Row::ew("IVec4", "i32", "IVec4"),
    Row::ew("i32", "IVec4", "IVec4"),
    Row::ew("UVec2", "UVec2", "UVec2"),
    Row::ew("UVec2", "u32", "UVec2"),
    Row::ew("u32", "UVec2", "UVec2"),
    Row::ew("UVec3", "UVec3", "UVec3"),
    Row::ew("UVec3", "u32", "UVec3"),
    Row::ew("u32", "UVec3", "UVec3"),
    Row::ew("UVec4", "UVec4", "UVec4"),
    Row::ew("UVec4", "u32", "UVec4"),
    Row::ew("u32", "UVec4", "UVec4"),
    // Matrices: a product with itself and with its vector, and the scalar on both sides.
    Row::product("Mat2", "Mat2", "Mat2"),
    Row::product("Mat2", "Vec2", "Vec2"),
    Row::ew("Mat2", "f32", "Mat2"),
    Row::ew("f32", "Mat2", "Mat2"),
    Row::product("Mat3", "Mat3", "Mat3"),
    Row::product("Mat3", "Vec3", "Vec3"),
    Row::ew("Mat3", "f32", "Mat3"),
    Row::ew("f32", "Mat3", "Mat3"),
    Row::product("Mat4", "Mat4", "Mat4"),
    Row::product("Mat4", "Vec4", "Vec4"),
    Row::ew("Mat4", "f32", "Mat4"),
    Row::ew("f32", "Mat4", "Mat4"),
];

/// The Slang spelling of a Rust spelling in the table.
pub fn slang(name: &str) -> &'static str {
    match name {
        "f32" => "float",
        "i32" => "int",
        "u32" => "uint",
        "Vec2" => "float2",
        "Vec3" => "float3",
        "Vec4" => "float4",
        "IVec2" => "int2",
        "IVec4" => "int4",
        "UVec2" => "uint2",
        "UVec3" => "uint3",
        "UVec4" => "uint4",
        "Mat2" => "float2x2",
        "Mat3" => "float3x3",
        "Mat4" => "float4x4",
        _ => panic!("{name} is not in the product table"),
    }
}

/// Whether a spelling is `f32`, `i32` or `u32`.
pub fn is_scalar(name: &str) -> bool {
    matches!(name, "f32" | "i32" | "u32")
}

/// Whether a spelling is a matrix.
pub fn is_matrix(name: &str) -> bool {
    name.starts_with("Mat")
}

/// The number of components of a vector, or of the columns (and rows) of a matrix; 1 for a scalar.
pub fn width(name: &str) -> usize {
    if is_scalar(name) {
        return 1;
    }
    name.chars()
        .last()
        .and_then(|digit| digit.to_digit(10))
        .map_or(1, |digit| digit as usize)
}

/// The line of Slang for one row: the overload `__gust_mul` for its operand pair.
pub fn overload(row: &Row) -> String {
    let body = if row.kind == Kind::MatMul {
        "mul(a, b)"
    } else {
        "a * b"
    };
    format!(
        "{} __gust_mul({} a, {} b) {{ return {body}; }}",
        slang(row.output),
        slang(row.left),
        slang(row.right)
    )
}

/// The overload table a module that names a matrix gets after the prelude: one line per row, then a blank line.
/// Slangc drops the overloads a kernel does not use (measured on `plans/m6/probe/r1/gm2.slang`).
pub fn overload_text() -> String {
    let mut text = String::new();
    for row in MUL_PAIRS {
        text.push_str(&overload(row));
        text.push('\n');
    }
    text.push('\n');
    text
}

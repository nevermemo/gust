//! The negation table behind unary `-` in a module that names a matrix (slice m6.n, M6 plan 2.1).
//!
//! Plain data and string templates, in the style of [`super::mul_table`], and under the same gate: a module that names
//! `Mat2`, `Mat3` or `Mat4` lowers every unary `-` on a non-literal to the overloaded call `__gust_neg(a)`, and the
//! overloads follow `__gust_mul`'s. The macro sees syntax and not types, and slangc's WGSL for `-m` on a matrix is
//! `mat4x4<f32>(0) - m`, which naga refuses (the m6.g1 review, finding 6): so a matrix is negated as `m * -1.0` (WGSL:
//! `m[0] * v[0]` per column against a matrix of `-1.0`; SPIR-V: `OpVectorTimesScalar` by `-1.0` per column; both pass
//! naga and `spirv-val`, probed in `plans/m6/probe/m6n`). Under IEEE arithmetic a product with `-1.0` is bit-identical
//! to negation for every non-NaN value, `-0` included, which `0.0 - m` is not (`0.0 - 0.0` is `+0`). A device that
//! drops the sign of a zero (Vulkan does not promise it without `SignedZeroInfNanPreserve`) or flushes a subnormal in
//! a multiply gives a different lane than the twin's sign flip (D39); `vectors.rs` of `gust-wgpu` records what the
//! RTX 5090 does.
//!
//! The scalar and vector rows are Slang's own `-a`. Slangc's WGSL for a vector `-v` is `vec3<f32>(0) - v` (it passes
//! naga; by the IEEE rules a `+0` lane comes out `+0` where the twin has `-0`, an edge-case difference left as it is,
//! D39; `-v` means the same in every module, so it is not scaled), the SPIR-V is `OpFNegate`.
//!
//! `gust::math` has a `Neg` for `Vec2`, `Vec3`, `Vec4`, `IVec2`, `IVec4` and the three matrices (`vector_neg!` and
//! `matrix_ops!`; the unsigned vectors have none, as Rust has no unary minus on `u32`), and every one has a row here.

/// How a row is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegKind {
    /// Slang's own `-a`.
    Plain,
    /// `a * -1.0`: a matrix, for which slangc's `-a` is not valid WGSL.
    Scale,
}

/// One operand type, by Rust spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegRow {
    pub operand: &'static str,
    pub kind: NegKind,
}

impl NegRow {
    const fn plain(operand: &'static str) -> Self {
        Self {
            operand,
            kind: NegKind::Plain,
        }
    }

    const fn scale(operand: &'static str) -> Self {
        Self {
            operand,
            kind: NegKind::Scale,
        }
    }
}

/// Every operand type: 2 scalars, 5 vectors, 3 matrices. (`u32` has no unary minus in Rust.)
pub const NEG_ROWS: &[NegRow] = &[
    NegRow::plain("f32"),
    NegRow::plain("i32"),
    NegRow::plain("Vec2"),
    NegRow::plain("Vec3"),
    NegRow::plain("Vec4"),
    NegRow::plain("IVec2"),
    NegRow::plain("IVec4"),
    NegRow::scale("Mat2"),
    NegRow::scale("Mat3"),
    NegRow::scale("Mat4"),
];

/// The line of Slang for one row: the overload `__gust_neg` for its operand type.
pub fn overload(row: &NegRow) -> String {
    let body = match row.kind {
        NegKind::Plain => "-a",
        NegKind::Scale => "a * -1.0",
    };
    let slang = super::mul_table::slang(row.operand);
    format!("{slang} __gust_neg({slang} a) {{ return {body}; }}")
}

/// The overload table a module that names a matrix gets after the `__gust_mul` table: one line per row, then a blank
/// line.
pub fn overload_text() -> String {
    let mut text = String::new();
    for row in NEG_ROWS {
        text.push_str(&overload(row));
        text.push('\n');
    }
    text.push('\n');
    text
}

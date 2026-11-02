# Direct emission rules

These are design constraints for extending the existing dialect, not a claim that every construct is currently supported.

| Rust source | Emission rule / rejection condition |
|---|---|
| `u32`, `i32`, `f32` | Use deliberate target-width mapping (`uint`, `int`, `float`); reject unsupported widths. |
| `usize`, unsuffixed literals | Require a documented restricted typing/range rule; host pointer width is not GPU index width. |
| Binary/unary expressions | Parenthesize structurally or use an explicit precedence table; preserve signed comparisons, short circuit and unary minus. |
| Casts | Check Rust versus Slang behavior, especially float-to-int, truncation and overflow. Reject unsupported semantics. |
| Locals and shadowing | Maintain lexical scopes; mangle collisions and reserved target names deterministically. |
| Function calls | Resolve allowlisted module helpers/intrinsics, arity and types; do not stringify an arbitrary Rust path. |
| Method syntax | Map only known receiver/intrinsic pairs; `.len()` is not an arbitrary shader method. |
| Index read/write | Enforce buffer access, index type and host length contract; write only to writable resources. |
| `if`, blocks, tail expressions | Rust expressions may yield values; emit a compatible value/temporary or reject. Do not lose the final expression or branch. |
| Loops, break/continue, returns | Support only explicitly implemented forms; preserve exits and return values. |
| Vector and matrix types (`gust::math`) | By name, from the prelude only: `Vec2/3/4` to `float2/3/4`, `IVec2/4` to `int2/4`, `UVec2/3/4` to `uint2/3/4`, `Mat2/3/4` to `float2x2/3x3/4x4` (`crates/gust-macros/src/math_types.rs` is the one table). A struct, or a qualified path (`super::Mat4`, E0324), may not reuse the names (E0203). `Vec3`, `UVec3` and `Mat3` are locals and helper values only (E0204 as a struct field or resource element). A struct whose fields leave implicit padding is E0208. |
| Vector and matrix constructors | `Vec4::new(a, b, c, d)` is `float4(a, b, c, d)`, `Vec4::splat(s)` is `float4(s)`, `Mat4::IDENTITY` is the rows constructor of the identity. Slang's matrix constructor takes rows and Rust matrices are column-major, so `Mat4::from_rows(r0, r1, r2, r3)` is `float4x4(r0, r1, r2, r3)` and `Mat4::from_cols(c0, c1, c2, c3)` is `transpose(float4x4(c0, c1, c2, c3))`. These are the only multi-segment paths accepted; any other is E0308 (value) or E0302 (call), with the argument count checked (E0304). |
| Vector and matrix methods | A swizzle `v.xy()` (two to four of `xyzw`) is `v.xy`; `a.dot(b)` is `dot(a, b)`; `m.mul_vec4(v)` and `a.mul_mat4(b)` are `mul(m, v)` and `mul(a, b)` (M x v, written order); `m.transpose()` is `transpose(m)`. No `*` on a matrix is lowered here (slice m6.m2 owns it); the golden is written with methods. |

Use a typed emission function per node family returning an error for unsupported nodes. Passing syn tokens through Display/ToTokens and replacing strings is not a lowering algorithm. Preserve evaluation order with temporaries where target rules differ; do not duplicate an expression with side effects. Separate Rust host expansion from Slang text emission.

Keep a source-span association for diagnostics using the project's existing mapping or emitted comments; generated line numbers must not be presented as original Rust locations without a mapping. Keep helper ordering and bindings stable.

For numeric changes test integer boundaries, division/shift edge cases and floats with an explicit NaN/infinity/rounding policy. Independent host reference arithmetic can differ from GPU arithmetic; do not use tolerance to conceal a semantic error. Layout comes from the host/shader ABI, not Rust field spelling.

Source guidance checked 2026-09-07: [Rust expressions](https://doc.rust-lang.org/reference/expressions.html), [Slang compilation](https://github.com/shader-slang/slang/blob/master/docs/user-guide/08-compiling.md). Project-specific mapping rules above must be confirmed against existing tests.

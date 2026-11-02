# GUST diagnostic codes

Every rejection `#[gpu]` reports starts with a stable code:

```text
error: [GUST-E0401] `loop` is not supported: its trip count is not fixed at entry, ...
```

## Stability

**Codes are stable; messages are not.** A code keeps its meaning and is never reused for
something else. The message text after it may be reworded, so match on the code, not on
the words. Renumbering or retiring a code needs a row in [DECISIONS.md](DECISIONS.md).
Search this page for the code to find what it rejects and what to write instead.
Decision: D31.

The promise starts with the first crates.io release (D33). Until then a code may still be
split or renumbered, and each change is listed here:

- `f20d313`: E0602 narrowed from any wrong shape of an atomic's first argument to "not a
  reference"; the other shapes moved to the new codes E0608, E0609 and E0610.

Tests assert the code, not the wording. Every code appears once on this page and once in
the macro's code list; unit tests fail if the two disagree. A code counts as tested only
when a test assertion passed for it: the test binary runs again and the code helpers
(`assert_code`, `assert_codes`, `assert_rendered_code`) record what they checked, so a
`Code::X` that is only named in a test does not count.

Clippy keeps uncoded errors out. The workspace `clippy.toml` bars `syn::Error::new` and
`new_spanned`, the `error` methods of `ParseBuffer`, `StepCursor`, `ParseNestedMeta` and
`Lookahead1`, the `syn` parse functions (`parse`, `parse2`, `parse_str`, `parse_file`,
`Parser::parse*`, `Attribute::` and `MetaList::` `parse_args`, `parse_args_with`,
`parse_nested_meta`) and `parse_macro_input!`.
Only `diagnostics.rs` allows them, and it offers coded wrappers (`error`, `wrap`, `parse2`,
`parse_args_with`, `parse_nested_meta`); CI runs clippy with `-D warnings`, so any other use
fails the build. Clippy resolves the real item, so an alias or a renamed import is caught
too. Unit tests fail if `clippy.toml` drops an entry, if another source file names the
lints, their groups (`clippy::style`, `clippy::all`) or `allow(warnings`, if this crate's
manifest has lints in any form (a `[lints` table however spaced or quoted, or
`lints.workspace = true`), if the root manifest has lints other than `[workspace.lints.rust]`
with the single key `unexpected_cfgs` (the check-cfg declaration of `cfg(gust_dual)`; a
package-level `[lints`, a `clippy` or other group, another key, and the dotted or inline
spellings are all offenders), if the root `clippy.toml` is missing, if one sits in `crates/` or
`crates/gust-macros` (clippy takes the first it finds walking up), or if `CLIPPY_CONF_DIR` is
set. Not checked: `rustflags` in a `.cargo/config.toml` (for example
`-A clippy::disallowed_methods`) would silence the bar; keep such flags out of the repository.
The root `unexpected_cfgs` table cannot reach this crate: a workspace lints table applies
only to members that opt in with `lints.workspace = true`, this crate does not, and its own
manifest may have no lints at all, so the test above fails if it ever tried.
The limits of the guard: clippy cannot bar one type's `From` impl, so the
conversion `syn::Error::from(LexError)` is only checked as text (`LexError` and
`Error::from(` outside `diagnostics.rs`). It also does not see a `syn::Error` that arrives
through another `syn` call (for example `Parse::parse`, `base10_parse`, `LitStr::parse`)
and is propagated with `?`, nor one converted with `.into()`. Review new code
that handles a `syn::Result` by hand with that in mind.

The subset is deliberately small: the macro translates Rust syntax directly to Slang
without an intermediate representation (D02), and rejects what it cannot lower with a
proven meaning rather than guessing. Each entry names the decision or task that
records why. The same code is used wherever the same message is produced, whether the
validator or the emitter finds it; different messages have different codes, and a message
only the emitter produces has its own code in E08xx.

Each entry also says whether `#[gpu]` can produce the code. Most are reached through
`#[gpu]`. A few checks in the emitter and the expander are second lines of defence: the
validator rejects the same shape first, so the code appears only if a module skips
validation. Those say "Unreachable through `#[gpu]`: defence in depth", and a test keeps
this page and the macro's list of such codes in agreement.

| Family | Area |
| --- | --- |
| E01xx | Module and item shape: inline module, allowed items, structs, signatures, attributes |
| E02xx | Types: unsupported primitives, casts |
| E03xx | Expressions and calls: names, built-ins, paths, bindings |
| E04xx | Control flow and loops |
| E05xx | `Option` and `Result` |
| E06xx | Atomics |
| E07xx | Macro entry and `#[kernel]` options |
| E08xx | Emitter-only rejections |
| E09xx | Reserved names and shadowing |

## Slang compiler errors and Rust lines

The codes below are the macro's. An error that `slangc` reports later, in the emitted
Slang, is mapped back to a Rust line when the kernel's source file is known: the compile
copy of the Slang carries `#line <rust_line> "<file>"` directives, and the runtime reads
the location back from the compiler's message. A source file whose name fails the
directive whitelist (`directive_file_is_usable` in `crates/gust/src/slang/mod.rs`) gets
no Rust-line attribution: its errors name Slang lines of `kernel.slang` only. The macro
and the runtime check the same characters and the same structure. The macro records a path
only if every `/`-separated component is non-empty, is not made of dots only, and consists
of ASCII letters and digits, `_ . + -`, and non-ASCII characters with the Unicode
Alphabetic property; the directive check applies exactly that to each component, so no path
the macro records is refused, and hand-written origins with an empty, dot-only or leading
`/` component are refused as well. A path under a non-ASCII directory such as
`crates/ü/src/k.rs` therefore gets `#line` directives; slangc echoes a UTF-8 name back
unchanged (probed with two-, three- and four-byte characters). A decomposed accent (U+0301,
U+0308) is not Alphabetic, so a name in NFD form, as macOS file systems store accented
names, gets no `#line`. The name `kernel.slang` itself is excluded. The reason is Slang
string quoting: the file name is written into the directive's string literal, and a quote,
a backslash, a control character or a line break would end that string or the directive,
so no such name is ever injected. A file named like the compile copy would also make
Rust and Slang locations indistinguishable.

**Known limitation: slangc refuses a zero divisor that it can fold, in code that never
runs** (found by the differential test, slice m5.c2; a limitation, not a defect to fix
in the translation). slangc 2026.20 evaluates an integer `/` or `%` at compile time when
it can fold the divisor to a constant, and it rejects a zero with
`error[E30002]: divide by zero`, in a branch that no invocation takes as well. The error
comes through as an ordinary compile error of the kernel, mapped to its Rust line, so a
valid Rust kernel can fail to compile. What it folds was measured on the shapes that GUST
emits (each probe was a division under a branch keyed on a buffer value):

- constants, the counter of a constant-bound `for` at its first iteration only (the
  counter is 0; `v - 3`, zero only in the last iteration, compiles), and the locals that
  are set from them, with wrapping arithmetic: `(v - 1) + 1` and `(v + 65536) * 65536`
  fold to zero, and so do shifts of the counter;
- a product with a zero factor, whatever the other factor is: `i * 0`, `0 * i` and
  `int(a[i]) * 0` all fail, so a value read from a buffer does not help;
- the difference of two identical operands that read no buffer: `i - i`,
  `(i + 1) - (i + 1)` and `len - len` fail, and they compose (`i * (i - i)`, `0 / (i + 1)`),
  while `a[i] - a[i]` is not folded;
- a condition on a known value picks its branch: with `d = 5; if v < 1 { d = 0 }`, the
  division `7 / d` fails, and `if v < 1 { 7 / v }` fails, but `if v > 0 { 7 / v }` and
  `d = 5; if v > 0 { d = 0 }` compile;
- it does not fold `!` and `^` (GUST lowers them to the calls `__gust_not` and
  `__gust_xor`), `x & 0`, `x ^ x`, `x % 1`, `0 % x`, `i - 1`, `i + (0 - i)`, or anything
  that is known only through a buffer read.

The smallest reproducer is `for i in 0u32..4u32 { if !true { let r = i % i; } }` (emitted
as `if (__gust_not(true)) { uint r = (i % i); }` in `for (var i = uint(0); i < end; i++)`
and refused at the `%`). The workaround is a divisor that is non-zero by construction,
for example `i % (i | 1)` (probed: it compiles, in the reproducer's own shape and in
`let d = i; 7 / (d | 1)`), or a non-zero literal. A value read from a buffer is no
workaround: `int(a[i]) * 0` is folded, and a plain `7 / a[i]` compiles but may be zero
at run time, which is the integer edge case of D39 and KM-01 in
[SEMANTICS](SEMANTICS.md). The differential generator refuses the programs that this model
says slangc folds (DIFFTEST_FORMAT 4, [DIFFTEST](development/DIFFTEST.md)).

## E01xx: module and item shape

### GUST-E0101 inline module required

`#[gpu]` needs the module body in the attribute's own item.
Reached through `#[gpu]`.
Rejected: `#[gpu] mod shaders;`.
Write instead: `#[gpu] mod shaders { ... }`.
Decision: D02.

### GUST-E0102 unsupported module item

A GPU module holds functions and plain structs only.
Reached through `#[gpu]`.
Rejected: `const`, `static`, `enum`, `impl`, `trait`, `use`, nested `mod`, type aliases.
Write instead: pass constants as uniform or value parameters, or write them inline.
Decision: D02.

### GUST-E0103 generic struct

GPU structs cannot take generic parameters or a `where` clause.
Reached through `#[gpu]`.
Rejected: `struct Pair<T> { first: T }`.
Write instead: one concrete struct per payload type.
Decision: D02.

### GUST-E0104 struct needs named fields

A GPU struct is laid out field by field, so each field needs a name.
Reached through `#[gpu]`.
Rejected: `struct Pair(f32, f32);` and `struct Marker;`.
Write instead: `struct Pair { first: f32, second: f32 }`.
Decision: D17.

### GUST-E0105 empty struct

A GPU struct needs at least one field to have a layout.
Reached through `#[gpu]`.
Rejected: `struct Empty {}`.
Write instead: give it a field, or remove it.
Decision: D17.

### GUST-E0106 manual repr

GPU structs receive `repr(C)` from the macro.
Reached through `#[gpu]`.
Rejected: `#[repr(C)] struct Params { ... }`.
Write instead: remove the attribute.
Decision: D17.

### GUST-E0107 manual Clone or Copy

The macro derives `Clone` and `Copy` for GPU structs.
Reached through `#[gpu]`.
Rejected: `#[derive(Clone, Copy)] struct Params { ... }`.
Write instead: remove those two derives; others such as `Debug` and `PartialEq` are allowed.
Decision: D17.

### GUST-E0108 malformed derive

The argument list of `#[derive(..)]` on a GPU struct does not parse. The message after
the code is `syn`'s own wording.
Reached through `#[gpu]`.
Rejected: `#[derive = "Debug"]`.
Write instead: `#[derive(Debug)]`.
Decision: D02.

### GUST-E0109 kernel dispatch parameter

A `#[kernel]` function takes exactly one dispatch-thread parameter.
Reached through `#[gpu]`.
Rejected: `#[kernel] fn run() {}` and a kernel with two `SV_DispatchThreadID` parameters.
Write instead: `#[kernel] fn run(id: SV_DispatchThreadID) { ... }`.
Decision: D02.

### GUST-E0110 kernel return type

A kernel returns nothing; results go through buffers.
Reached through `#[gpu]`.
Rejected: `#[kernel] fn run(id: SV_DispatchThreadID) -> u32 { ... }`.
Write instead: write results to a `RWStructuredBuffer` parameter.
Decision: D02.

### GUST-E0111 helper resource parameter

Helper functions take value parameters only.
Reached through `#[gpu]`.
Rejected: `fn helper(b: StructuredBuffer<u32>) -> u32 { ... }`.
Write instead: read the buffer in the kernel and pass the value to the helper.
Decision: D02.

### GUST-E0112 async function

Async has no meaning on the GPU.
Reached through `#[gpu]`.
Rejected: `async fn f() {}`. Async is CPU-only.
Write instead: a plain `fn`.
Decision: D02.

### GUST-E0113 unsafe function

The subset has no unsafe code.
Reached through `#[gpu]`.
Rejected: `unsafe fn f() {}`.
Write instead: a plain `fn`.
Decision: D02.

### GUST-E0114 extern or generic function

Functions are plain, non-generic Rust functions with the default ABI.
Reached through `#[gpu]`.
Rejected: `extern "C" fn f() {}` and `fn f<T>(x: T) {}`.
Write instead: a plain, non-generic `fn`.
Decision: D02.

### GUST-E0115 unsupported attribute

An attribute with no proven Rust-to-Slang meaning cannot be silently dropped.
Allowed attributes: `doc`, `allow`, `warn`, `deny`, `forbid`, `derive` (on structs),
and `kernel`.
Reached through `#[gpu]`.
Rejected: `#[cfg(..)]`, `#[inline]`, `#[must_use]` and every other attribute.
Write instead: remove the attribute.
Decision: D02.

### GUST-E0116 `self` parameter

GPU functions are free functions; there is no receiver to lower.
Reached through `#[gpu]`.
Rejected: `fn f(&self) -> u32`.
Write instead: pass the value as an ordinary parameter.
Decision: D02.

### GUST-E0117 parameter needs an identifier

Every parameter is a named binding that the emitted Slang can refer to.
Reached through `#[gpu]`.
Rejected: `fn f((a, b): (u32, u32))`, `fn f(_: u32)`.
Write instead: a plain `name: Type` parameter.
Decision: D02.

### GUST-E0118 POD struct needs named fields

Every struct in a `#[gpu]` module becomes a plain-old-data GPU type, laid out field by field, so every field needs a name.
Unreachable through `#[gpu]`: defence in depth. The validator rejects the same struct as E0104 first.
Rejected: `struct P(u32, u32);`.
Write instead: `struct P { a: u32, b: u32 }`.
Decision: D02.

## E02xx: types

### GUST-E0201 unsupported primitive type

GPU code uses the four-byte scalars only.
Reached through `#[gpu]`.
Rejected: `i8`, `i16`, `i64`, `i128`, `isize`, `u8`, `u16`, `u64`, `u128`, `usize`,
`f16`, `f64`, `f128`, `char`, `str`.
Write instead: `f32`/`float`, `i32`/`int`, `u32`/`uint`, `bool`.
Decision: D06.

### GUST-E0202 unsupported cast target

`as` casts are proven only between the 32-bit scalars.
Reached through `#[gpu]`.
Rejected: `x as f64`, `x as usize`, `x as bool`, `x as Pair`.
Write instead: cast to `f32`/`float`, `i32`/`int` or `u32`/`uint`.
Decision: D22 (numeric semantics).

### GUST-E0203 reserved type name

A struct named like a type the dialect maps to a Slang type itself would be emitted as that builtin while rustc resolves the user's struct.
Reached through `#[gpu]`.
Rejected: `struct Mat4 { .. }`, `struct r#Vec2 { .. }` (the comparison is on the unraw text), `struct float { .. }`; the reserved names are `bool`, `float`, `int`, `uint`, `Vec2`, `Vec3`, `Vec4`, `IVec2`, `IVec4`, `UVec2`, `UVec3`, `UVec4`, `Mat2`, `Mat3` and `Mat4`.
Write instead: another name, such as `Transform` or `Particle`.
Decision: D17, D31 (M6 plan R1.1).

### GUST-E0204 3-wide type in a buffer

`Vec3`, `UVec3` and `Mat3` have stride 16 in a Slang buffer but size 12 (36) in Rust, so no Rust type matches them in every buffer position.
Reached through `#[gpu]`.
Rejected: a struct field `v: Vec3`, `m: Mat3`; a resource element `StructuredBuffer<Vec3>`, `RWStructuredBuffer<UVec3>`, `Storage<Mat3>`.
Write instead: `Vec4` or three scalars in buffers; the 3-wide types stay allowed as locals, helper parameters and helper results.
Decision: D17 (M6 plan F4, F5).

### GUST-E0205 `Mat2` or `bool` in a uniform struct

A 2x2 matrix has column stride 16 in a constant buffer but 8 in Rust, and a `bool` has no layout a Rust type shares with the GPU's, so a struct a uniform reaches may hold neither.
Reached through `#[gpu]`.
Rejected: `struct Params { m: Mat2, v: Vec2 }` or `struct Params { on: bool, .. }` when a kernel has a `Uniform<Params>` parameter, also when the struct is a field of the uniform's struct. The same `Mat2` stays fine as a storage element and in a struct no uniform reaches.
Write instead: a `Mat4`, a `Vec4` holding the four entries, or a `u32` that is 0 or 1.
Decision: D17, D40 (M6 plan 2.3, F7).

### GUST-E0206 uniform element is not a module struct

A uniform holds one struct declared in the same `#[gpu]` module, so the macro can compute its constant-buffer layout and declare it `#[repr(C, align(16))]`.
Reached through `#[gpu]`.
Rejected: `Uniform<f32>`, `ConstantBuffer<Vec4>`, `Uniform<Mat4>`, `Uniform<super::Outside>`, `Uniform<Missing>`.
Write instead: a struct of the module with the value as a field, `struct Params { scale: f32, .. }` and `params: Uniform<Params>`.
Decision: D17, D40 (M6 plan 2.3).

### GUST-E0207 struct used by a uniform and a storage buffer

A struct a uniform reaches (as the parameter's element or as a field of a struct that is) is declared `#[repr(C, align(16))]` with a constant-buffer layout, which differs from the storage layout of the same fields, and copying between the two crashes slangc on WGSL.
Reached through `#[gpu]`. The check covers every kernel of the module together: one struct may be the uniform of one kernel and the storage element of another.
Rejected: `Uniform<S>` with `StructuredBuffer<S>`, `Uniform<Params>` with `StructuredBuffer<Inner>` where `Params` has a field `Inner`.
Write instead: declare a separate struct for each use. Two uniforms may share a struct.
Decision: D17, D40 (M6 plan 2.3, F7).

### GUST-E0208 implicit padding in a struct

A struct whose fields leave padding (before a field, or at the tail) cannot be a `GpuPod` type: `pod_slice_as_bytes` would read the padding bytes, which are uninitialised.
Reached through `#[gpu]`. The check covers scalars, vectors, matrices and the module's own structs; a struct with a field type the macro cannot see into is left to the `const` assertions of the expansion, which fail compilation instead.
Rejected: `struct S { a: f32, v: Vec2 }` (`v` would sit at offset 8, after four padding bytes), `struct S { v: Vec2, a: f32 }` (the size is 16, the fields end at 12).
Write instead: order the fields by decreasing alignment and fill what is left with explicit fields, `struct S { v: Vec2, a: f32, pad: f32 }`.
Decision: D17 (M6 plan R1-A, m6.a1 review).

### GUST-E0209 implicit padding in a uniform struct

A struct a uniform reaches has a constant-buffer layout in which every struct is rounded up to 16 bytes (a nested struct also starts at a multiple of 16), so fields that are padding-free in Rust can still leave a gap before a field or a tail. GUST never inserts padding: the upload copies the bytes of the Rust value, and padding bytes are uninitialised. The message names the field or struct and the bytes of explicit padding to write.
Reached through `#[gpu]`. A gap or tail that `#[repr(C)]` has too is GUST-E0208 first. A struct with a field type the macro cannot see into is left to the `const` assertions of the expansion, which fail compilation instead.
Rejected: `struct Params { a: f32 }` (ends at byte 4, a constant buffer rounds it up to 16), `struct Params { a: f32, b: f32, c: f32 }`, `struct Params { a: f32, inner: Inner }` where `Inner` is four `f32` (`inner` would sit at byte 16, 12 bytes after `a` ends), each when a `Uniform<Params>` parameter reaches it. The same structs are fine as storage elements.
Write instead: explicit padding fields, `struct Params { a: f32, pad0: f32, pad1: f32, pad2: f32 }`, or order the fields so that whole 16-byte rows are filled.
Decision: D17, D40 (M6 plan 2.3, F7).

## E03xx: expressions and calls

### GUST-E0301 CPU-only name

Standard-library and thread names have no GPU meaning.
Reached through `#[gpu]`.
Rejected: `Vec`, `String`, `Box`, `Rc`, `Arc`, `HashMap`, `HashSet`, `std::..`,
`core::..`, `alloc::..`, `println`, `dbg`, `panic`, `thread`, `spawn`, `File`,
`TcpStream`.
Write instead: fixed-size values and buffers.
Decision: D02.

### GUST-E0302 call to something other than a helper

Only helpers declared in the same GPU module can be called.
Reached through `#[gpu]`.
Rejected: `foo(x)` where `foo` is outside the module, `(f)(x)`.
Write instead: declare `fn foo` inside the module.
Decision: D02.

### GUST-E0303 type arguments on a built-in

Built-in methods are not generic.
Reached through `#[gpu]`.
Rejected: `ctx.global_id::<u32>()`.
Write instead: `ctx.global_id()`.
Decision: D02.

### GUST-E0304 arguments on a built-in

`.global_id()` and `.len()` take no arguments. A swizzle (`.xy()`) and `.transpose()` take none, `.dot(v)`, `.mul_vecN(v)` and
`.mul_matN(m)` take one, `Vec4::new(..)` and `Mat4::from_cols(..)` take as many as the type has components or columns, and `splat` takes one.
Reached through `#[gpu]`.
Rejected: `out.len(1)`, `v.xy(1)`, `m.transpose(m)`, `Vec4::new(1.0, 2.0)`.
Write instead: `out.len()`, `v.xy()`, `m.transpose()`, `Vec4::new(1.0, 2.0, 3.0, 4.0)`.
Decision: D02.

### GUST-E0305 unsupported method

Methods other than `.global_id()`, `.len()`, `.is_some()`, `.is_none()`, `.is_ok()`,
`.is_err()`, `.unwrap_or(default)` and the vector and matrix methods (swizzles `.xy()`..`.wzyx()`, `.dot(v)`, `.mul_vec2/3/4(v)`,
`.mul_mat2/3/4(m)`, `.transpose()`) have no lowering.
Integer `.dot` (slice m6.m2): Slang's `dot` on an integer vector is `OpSDot` or `OpUDot` in SPIR-V, which needs
`Capability DotProduct`, and naga's SPIR-V reader refuses it (`UnsupportedCapability(DotProduct)`; measured on
slangc 2026.20). The macro sees no types (the invocation id is a `UVec3` and `id.xy()` a `UVec2`, neither named by a
token), so a module that contains a `dot` lowers every `.dot(v)` to the overloaded helper `__gust_dot(a, b)`. Its
integer overloads are the sum of the component products in written order, and its float overloads call Slang's `dot`
(the SPIR-V of a float-only module is byte-identical, and the WGSL calls a helper whose body is the plain `dot`). The
SPIR-V needs no capability beyond `Shader`: it passes naga's reader with empty capabilities and `spirv-val`. A module
without a `dot` keeps its text unchanged.
Reached through `#[gpu]`.
Rejected: `x.abs()`, `v.iter()`.
Write instead: write the operation with operators and helpers.
Decision: D02, D14.

### GUST-E0306 `.len()` receiver

`.len()` works on a storage parameter named directly.
Reached through `#[gpu]`.
Rejected: `id.len()`, `buffers.out.len()`.
Write instead: `out.len()` where `out` is a buffer parameter.
Decision: D02.

### GUST-E0307 `.global_id()` receiver

`.global_id()` works on the dispatch-thread parameter named directly.
Reached through `#[gpu]`.
Rejected: `other.global_id()`.
Write instead: `id.global_id()` where `id` is the kernel's dispatch-thread parameter.
Decision: D02.

### GUST-E0308 associated constant or foreign path

A path used as a value must be one local, parameter or helper name, or `Mat2::IDENTITY`, `Mat3::IDENTITY` or `Mat4::IDENTITY`.
Reached through `#[gpu]`.
Rejected: `f32::INFINITY`, `u32::MAX`, `core::f32::consts::PI`, `Vec4::ZERO`.
Write instead: write the literal, for example `4294967295u32`.
Decision: D06.

### GUST-E0309 const block

A `const { .. }` block has no lowering.
Reached through `#[gpu]`.
Rejected: `const { 1u32 }`.
Write instead: write the value directly.
Decision: D02.

### GUST-E0310 struct update syntax

Struct update syntax would copy fields the translator cannot see.
Reached through `#[gpu]`.
Rejected: `Pair { first: 9u32, ..old }`.
Write instead: list every field: `Pair { first: 9u32, second: old.second }`.
Decision: D07.

### GUST-E0311 `?` operator

Early return through `?` has no lowering.
Reached through `#[gpu]`.
Rejected: `value?`.
Write instead: `if let Some(x) = value { .. }` or `unwrap_or`.
Decision: D14.

### GUST-E0312 unsafe block

The subset has no unsafe code.
Reached through `#[gpu]`.
Rejected: `unsafe { .. }`.
Write instead: safe code only.
Decision: D02.

### GUST-E0313 macro in expression position

Macros cannot run inside a GPU module. This code is for a macro used as an expression;
a macro used as a statement is E0805.
Reached through `#[gpu]`.
Rejected: `let v = vec![..];`, `let n = len!(x);`.
Write instead: plain expressions.
Decision: D02.

### GUST-E0314 closure

Closures capture an environment the translator does not model.
Reached through `#[gpu]`.
Rejected: `|x| x + 1u32`.
Write instead: a helper function in the module.
Decision: D02.

### GUST-E0315 await

Async has no meaning on the GPU.
Reached through `#[gpu]`.
Rejected: `value.await`.
Write instead: GPU code is synchronous; do async work on the host.
Decision: D02.

### GUST-E0316 reference

References are not part of the value model.
Reached through `#[gpu]`.
Rejected: `&x`, `&mut x`. The one exception is the first argument of an atomic call,
`&mut buffer[index]` (see E06xx).
Write instead: pass and return values.
Decision: D02.

### GUST-E0317 division by a literal zero

Slang makes integer division by a literal zero a compile error while rustc treats it as
a runtime panic, so the boundary rejects it. Float division by zero is allowed.
Reached through `#[gpu]`.
Rejected: `x / 0u32`, `x % 0`.
Write instead: guard the divisor before dividing.
Decision: D22.

### GUST-E0318 reference or `@` binding

Binding modes other than a plain move or copy have no lowering.
Reached through `#[gpu]`.
Rejected: `let ref x = v;`, `let x @ _ = v;`, `for ref i in ..`, a `ref` parameter.
Write instead: a plain `let x = v;`.
Decision: D02.

### GUST-E0319 `let` pattern must be one identifier

A `let` binds one name, optionally typed.
Reached through `#[gpu]`.
Rejected: `let (a, b) = pair;`, `let Pair { first, .. } = p;`.
Write instead: `let a = pair.first;`.
Decision: D02.

### GUST-E0320 let-else

`let .. else` needs control-flow lowering the translator does not have.
Reached through `#[gpu]`.
Rejected: `let v = x else { return; };`. A refutable pattern such as `let Some(x) = o else { .. };` is E0319.
Write instead: `if let Some(x) = o { .. }`.
Decision: D02.

### GUST-E0321 uniform used as a whole value

A uniform parameter is read one field at a time. Copying or passing the whole struct out of a constant buffer is the construct that crashes slangc on WGSL, and the CPU twin would accept what the GPU cannot compile.
Reached through `#[gpu]`.
Rejected: `*params`, `let p = params;`, `(params)` alone, `params` as a function argument. `params.len()` is GUST-E0306, and a helper taking a uniform is GUST-E0111.
Write instead: read the fields, `params.scale`, `params.view.mul_vec4(p)`, or copy a field into a local.
Decision: D17, D40 (M6 plan 2.3, F7).

### GUST-E0323 `*=` on a place with a call

In a module that names `Mat2`, `Mat3` or `Mat4`, every `*` is the overloaded product `__gust_mul`, and `x *= y` is
written `x = __gust_mul(x, y)`, which names the place twice. Rust evaluates the place of `*=` once, so a place that is
more than names, fields, literals, indexing, arithmetic, `.global_id()` and `.len()` (a helper call, an `if` or a block
in an index) could not be lowered without evaluating it twice, and the macro refuses it. A module that names no matrix
keeps the plain `*=` and never meets this code.
Reached through `#[gpu]`.
Rejected: `out[pick(i)] *= s;` in a module that names a matrix, where `pick` is a helper and `s` a scalar.
Write instead: `out[pick(i)] = out[pick(i)] * m;`, or bind the index first: `let j = pick(i); out[j] *= m;`.
Decision: D17 (M6 plan 2.1, choice 12).

### GUST-E0324 qualified math path

The vector and matrix types are the prelude's; a module path to one is not mapped by its last segment, so a user type of the same name in another module cannot be mistaken for it.
Reached through `#[gpu]`.
Rejected: `super::Mat4`, `crate::Vec2`, `::Mat4`, `gust::math::Vec4::new(..)`, in a type, a struct literal or a call.
Write instead: the bare name, `Mat4`, `Vec4::new(..)`, `Mat4::IDENTITY`. A `use` item inside the module is still E0102.
Decision: D17, D31 (M6 plan R1.1-2).

## E04xx: control flow and loops

A GPU thread must always terminate and must never wait on another workgroup, so the
only loop is `for i in start..end` (D18).

### GUST-E0401 `loop`

A `loop` has no fixed trip count.
Reached through `#[gpu]`.
Rejected: `loop { .. }`.
Write instead: `for i in 0u32..n { .. break; }`.
Decision: D18.

### GUST-E0402 `while`

A `while` loop has no fixed trip count.
Reached through `#[gpu]`.
Rejected: `while cond { .. }`.
Write instead: `for i in 0u32..n { if !cond { break; } .. }`.
Decision: D18.

### GUST-E0403 loop label

Labelled jumps have no lowering.
Reached through `#[gpu]`.
Rejected: `'outer: for ..`, `break 'outer;`, `continue 'x;`.
Write instead: a flag variable or a helper function that returns early.
Decision: D18.

### GUST-E0404 `break` or `continue` outside a loop

A jump needs an enclosing `for` loop.
Reached through `#[gpu]`.
Rejected: `break;` or `continue;` with no enclosing `for`.
Write instead: remove it, or move it inside the loop.
Decision: D18.

### GUST-E0405 `break` or `continue` as a value

They are statements only.
Reached through `#[gpu]`.
Rejected: `let v = break;`.
Write instead: `break;` on its own line.
Decision: D18.

### GUST-E0406 `break` with a value

Only `loop` can break with a value, and `loop` is not supported.
Reached through `#[gpu]`.
Rejected: `break 3u32;`.
Write instead: assign to a `let mut` local, then `break;`.
Decision: D18.

### GUST-E0407 mutable loop variable

Slang's counter would diverge from Rust's fresh per-iteration binding.
Reached through `#[gpu]`.
Rejected: `for mut i in 0u32..n { i += 1u32; }`.
Write instead: `for i in 0u32..n { let mut j = i; j += 1u32; }`.
Decision: D18.

### GUST-E0408 loop variable pattern

The loop variable is one name.
Reached through `#[gpu]`.
Rejected: `for (a, b) in 0u32..n`.
Write instead: a single identifier or `_`.
Decision: D18.

### GUST-E0409 loop iterable

Only an exclusive integer range is iterable.
Reached through `#[gpu]`.
Rejected: `for v in input`, `for v in input.iter()`.
Write instead: `for i in 0u32..input.len() { let v = input[i]; .. }`.
Decision: D18.

### GUST-E0410 inclusive range

When `end` is the type maximum the Slang counter never exceeds it and the loop would
not terminate.
Reached through `#[gpu]`.
Rejected: `for i in 0u32..=n`.
Write instead: `for i in 0u32..n + 1u32` (mind overflow at the type maximum).
Decision: D18.

### GUST-E0411 open range

The trip count must be fixed at loop entry.
Reached through `#[gpu]`.
Rejected: `for i in 0u32..`, `for i in ..n`.
Write instead: give both bounds.
Decision: D18.

### GUST-E0412 unsuffixed integer-literal bound

Slang would type a bare literal `int` where rustc may infer `u32`.
Reached through `#[gpu]`.
Rejected: `for i in 0..n`, `for i in 0u32..(-4)`.
Write instead: `for i in 0u32..n`.
Decision: D18.

### GUST-E0413 range as a value

Ranges exist only as the iterable of a `for`.
Reached through `#[gpu]`.
Rejected: `let r = 0u32..4u32;`.
Write instead: use the bounds directly.
Decision: D18.

## E05xx: Option and Result

`Option<T>` lowers to Slang `Optional<T>` and `Result<T, T>` to a tagged struct, bounded
to locals and helper signatures (D14).

### GUST-E0501 Option layout

`Option` has no proven storage layout, so it cannot sit in a struct field or a buffer
element type.
Reached through `#[gpu]`.
Rejected: `struct S { slot: Option<u32> }`, `RWStructuredBuffer<Option<u32>>`.
Write instead: keep `Option` values in locals and helper functions.
Decision: D14, D09.

### GUST-E0502 Option payload

The payload is a single supported scalar or module struct.
Reached through `#[gpu]`.
Rejected: `Option<Option<u32>>`, `Option<(u32, u32)>`, `Option<Storage<u32>>`.
Write instead: `Option<u32>` or `Option<MyStruct>`.
Decision: D14.

### GUST-E0503 Result layout

As E0501, for `Result`.
Reached through `#[gpu]`.
Rejected: `struct S { r: Result<u32, u32> }`, `RWStructuredBuffer<Result<u32, u32>>`.
Write instead: keep `Result` values in locals and helper functions.
Decision: D14, D09.

### GUST-E0504 Result shape

Slang cannot infer an absent type parameter, so both payloads must be the same type.
Reached through `#[gpu]`.
Rejected: `Result<u32, i32>`, `Result<Option<u32>, Option<u32>>`.
Write instead: `Result<u32, u32>`.
Decision: D14.

### GUST-E0505 `Some` arguments

`Some` wraps exactly one value.
Reached through `#[gpu]`.
Rejected: `Some(1u32, 2u32)`.
Write instead: `Some(1u32)`.
Decision: D14.

### GUST-E0506 `Ok` or `Err` arguments

`Ok` and `Err` wrap exactly one value.
Reached through `#[gpu]`.
Rejected: `Ok()`, `Err(1u32, 2u32)`.
Write instead: `Ok(1u32)`.
Decision: D14.

### GUST-E0507 variant used as a value

`Some`, `Ok` and `Err` exist only as calls.
Reached through `#[gpu]`.
Rejected: `let f = Some;`, `Ok` or `Err` without a call.
Write instead: call it with one argument.
Decision: D14.

### GUST-E0508 predicate arguments

The `is_*` predicates take no arguments.
Reached through `#[gpu]`.
Rejected: `o.is_some(1u32)`.
Write instead: `o.is_some()`.
Decision: D14.

### GUST-E0509 `unwrap_or` arguments

`unwrap_or` takes the default as its one argument.
Reached through `#[gpu]`.
Rejected: `o.unwrap_or(1u32, 2u32)`.
Write instead: `o.unwrap_or(1u32)`.
Decision: D14.

### GUST-E0510 `unwrap` and `expect`

GPU code cannot panic.
Reached through `#[gpu]`.
Rejected: `o.unwrap()`, `r.expect("msg")`.
Write instead: `o.unwrap_or(default)` or `if let Some(x) = o { .. }`.
Decision: D14.

### GUST-E0511 `if let` pattern

Only `Some(identifier)` and `Ok(identifier)` have a lowering.
Reached through `#[gpu]`.
Rejected: `if let Some(_) = o`, `if let None = o`, `if let Some(1) = o`.
Write instead: `if let Some(x) = o { .. }` or test with `is_none()`.
Decision: D14.

### GUST-E0512 `if let` position

`if let` is the whole condition of an `if`, not part of a larger expression.
Reached through `#[gpu]`.
Rejected: `if let Some(v) = o && v > 0u32 { .. }`.
Write instead: nest an `if` inside the `if let`.
Decision: D14.

### GUST-E0513 Result match as a value

A `match` on a `Result` is a statement, not an expression with a value.
Reached through `#[gpu]`.
Rejected: `let x = match r { Ok(v) => { v } Err(e) => { e } };`.
Write instead: assign inside the arms to a `let mut` local, and end the match with `;`.
Decision: D14.

### GUST-E0514 Result match needs a terminating semicolon

A statement `match` on a `Result` ends with `;`.
Reached through `#[gpu]`.
Rejected: `match r { .. }` without a `;` in statement position.
Write instead: `match r { .. };`.
Decision: D14.

### GUST-E0515 match scrutinee

Only a known `Result<T, T>` local, parameter or same-module helper call can be matched.
Reached through `#[gpu]`.
Rejected: `match 1u32 { .. }`, a match on an `Option`.
Write instead: bind the value to a `Result<T, T>` local first, or use `if let` for `Option`.
Decision: D14.

### GUST-E0516 match arm attribute

A match arm carries no attributes.
Reached through `#[gpu]`.
Rejected: `#[allow(unused)] Ok(x) => { .. }`.
Write instead: remove the attribute.
Decision: D14.

### GUST-E0517 match guard

A match arm has no guard.
Reached through `#[gpu]`.
Rejected: `Ok(x) if x > 0u32 => { .. }`.
Write instead: test inside the arm body.
Decision: D14.

### GUST-E0518 match arm block

Each arm body is a block.
Reached through `#[gpu]`.
Rejected: `Ok(x) => x`.
Write instead: `Ok(x) => { .. }`.
Decision: D14.

### GUST-E0519 match binding

Bindings are plain identifiers.
Reached through `#[gpu]`.
Rejected: `Ok(mut x)`, `Ok(ref x)`, `Ok(x @ _)`.
Write instead: `Ok(x)` and copy into a `let mut` local when needed.
Decision: D14.

### GUST-E0520 match arm count

Exactly one `Ok` arm and one `Err` arm.
Reached through `#[gpu]`.
Rejected: a match with only `Ok`, or two `Ok` arms.
Write instead: one arm each.
Decision: D14.

### GUST-E0521 match pattern

Arm patterns are `Ok(identifier)` and `Err(identifier)`.
Reached through `#[gpu]`.
Rejected: `_ => { .. }`, `Ok((x, y)) => { .. }`.
Write instead: `Ok(x) => { .. }` and `Err(e) => { .. }`.
Decision: D14.

## E06xx: atomics

Atomics are statement calls on `&mut buffer[index]` of a `RWStructuredBuffer<u32>` or
`RWStructuredBuffer<i32>` (T10, see [the task](development/tasks/T10-atomics.md)).
The first argument has three shapes to get right, each with its own code: a reference
(E0602), a mutable one (E0608), to an indexed element (E0609) of a named buffer (E0610).

### GUST-E0601 atomic argument count

`atomic_add`, `atomic_min`, `atomic_max` and `atomic_exchange` take 2 arguments;
`atomic_compare_exchange` takes 3.
Reached through `#[gpu]`.
Rejected: `atomic_add(&mut c[0])`, `atomic_compare_exchange(&mut c[0], 1u32)`.
Write instead: pass the right number of arguments.
Decision: T10.

### GUST-E0602 atomic first argument is not a reference

The first argument of an atomic operation is a reference to a buffer element.
Reached through `#[gpu]`.
Rejected: `atomic_add(counter, 1u32)`, `atomic_add(0u32, 1u32)`.
Write instead: `atomic_add(&mut c[0], 1u32)`.
Decision: T10.

### GUST-E0603 atomic buffer parameter

The buffer must be a read-write buffer parameter.
Reached through `#[gpu]`.
Rejected: an atomic on a `StructuredBuffer` parameter or an unknown name.
Write instead: declare the parameter as `mut c: RWStructuredBuffer<u32>`.
Decision: T10.

### GUST-E0604 atomic element type

Atomics work on 32-bit integer elements only.
Reached through `#[gpu]`.
Rejected: an atomic on `RWStructuredBuffer<f32>`.
Write instead: use a `u32` or `i32` buffer.
Decision: T10.

### GUST-E0605 signed `atomic_min` or `atomic_max`

Slang and WGSL support them for unsigned elements only.
Reached through `#[gpu]`.
Rejected: `atomic_min(&mut c[0], 1i32)`.
Write instead: use a `u32` buffer.
Decision: T10.

### GUST-E0606 atomic as a value

Atomics are statements; the old value is not returned.
Reached through `#[gpu]`.
Rejected: `let old = atomic_add(&mut c[0], 1u32);`, `if atomic_compare_exchange(..) { .. }`.
Write instead: call it as a statement on its own line.
Decision: T10.

### GUST-E0607 direct access to an atomic buffer

In a kernel that uses a buffer atomically, ordinary reads and writes of it are rejected.
Reached through `#[gpu]`.
Rejected: `out[id.x] = c[0];` after `atomic_add(&mut c[0], 1u32);`.
Write instead: use a separate non-atomic buffer, or read after the dispatch on the host.
Decision: T10.

### GUST-E0608 atomic reference is not mutable

The reference must be `&mut`, since the operation writes the element.
Reached through `#[gpu]`.
Rejected: `atomic_add(&c[0], 1u32)`.
Write instead: `atomic_add(&mut c[0], 1u32)`.
Decision: T10.

### GUST-E0609 atomic operand is not a buffer element

The reference must point at an indexed element, not a whole value.
Reached through `#[gpu]`.
Rejected: `atomic_add(&mut c, 1u32)`, `atomic_add(&mut local, 1u32)`.
Write instead: `atomic_add(&mut c[0], 1u32)`.
Decision: T10.

### GUST-E0610 atomic receiver is not a named buffer

The indexed value is a buffer parameter named directly.
Reached through `#[gpu]`.
Rejected: `atomic_add(&mut s.items[0], 1u32)`, `atomic_add(&mut (c)[0], 1u32)`.
Write instead: `atomic_add(&mut c[0], 1u32)`.
Decision: T10.

## E07xx: macro entry and `#[kernel]` options

### GUST-E0701 `#[gpu]` takes no arguments

The attribute has no options yet.
Reached through `#[gpu]`.
Rejected: `#[gpu(debug)] mod shaders { ... }`.
Write instead: `#[gpu] mod shaders { ... }`.
Decision: D02.

### GUST-E0702 item is not a module, or does not parse as one

`#[gpu]` goes on an inline module whose body parses as Rust. The message after the code is
`syn`'s own parse error.
Reached through `#[gpu]`.
Rejected: `#[gpu] fn f() {}`, a module with a syntax error inside, or tokens that do not
parse as an item.
Write instead: put the function inside `#[gpu] mod shaders { ... }` and fix the syntax error.
Decision: D02.

### GUST-E0703 duplicate `#[kernel]`

A function is a kernel once.
Reached through `#[gpu]`.
Rejected: two `#[kernel]` attributes on one function.
Write instead: one `#[kernel]` or `#[kernel(workgroup_size(x, y, z))]`.
Decision: D02.

### GUST-E0704 duplicate `workgroup_size`

The workgroup size is stated once.
Reached through `#[gpu]`.
Rejected: `#[kernel(workgroup_size(8, 8, 1), workgroup_size(4, 4, 1))]`.
Write instead: state `workgroup_size` once.
Decision: D02.

### GUST-E0705 wrong workgroup dimension count

A workgroup has three dimensions.
Reached through `#[gpu]`.
Rejected: `#[kernel(workgroup_size(8, 8, 1, 1))]`.
Write instead: exactly three dimensions; use `1` for unused ones.
Decision: D02.

### GUST-E0706 zero workgroup dimension

A dimension of zero would dispatch no threads.
Reached through `#[gpu]`.
Rejected: `#[kernel(workgroup_size(0, 1, 1))]`.
Write instead: every dimension at least `1`.
Decision: D02.

### GUST-E0707 unknown `#[kernel]` option

`workgroup_size` and `cpu` are the only `#[kernel]` options.
Reached through `#[gpu]`.
Rejected: `#[kernel(threads(64))]`.
Write instead: `#[kernel(workgroup_size(x, y, z))]` or `#[kernel(cpu = "unsupported")]`.
Decision: D02, D26.

### GUST-E0708 malformed `#[kernel]` attribute

Reported for attribute syntax that `syn` rejects before an option is recognised: a
missing comma, a dimension that is not an integer literal, fewer than three dimensions,
a `cpu` value that is not a string literal, a `cpu` without a value.
The message after the code is `syn`'s own.
Reached through `#[gpu]`.
Rejected: `#[kernel(workgroup_size(8, 8))]`, `#[kernel(workgroup_size(a, 1, 1))]`, `#[kernel(cpu = 1)]`,
`#[kernel(cpu)]`.
Write instead: `#[kernel(workgroup_size(8, 8, 1))]` or `#[kernel(cpu = "unsupported")]`.
Decision: D02.

### GUST-E0709 unknown `cpu` value

`cpu` opts a kernel out of its CPU twin, which the macro otherwise emits under `cfg(test)` or
`--cfg gust_dual`. The only value is `"unsupported"`.
Reached through `#[gpu]`.
Rejected: `#[kernel(cpu = "off")]`.
Fixture: `tests/ui/kernel-cpu-bad-value` (`cargo xtask check-ui`, `code: none` plus the message).
Write instead: `#[kernel(cpu = "unsupported")]`, or no `cpu` option to keep the twin.
Decision: D26, D40.

### GUST-E0710 duplicate `cpu`

The `cpu` option is stated once.
Reached through `#[gpu]`.
Rejected: `#[kernel(cpu = "unsupported", cpu = "unsupported")]`.
Fixture: `tests/ui/kernel-cpu-duplicate` (`cargo xtask check-ui`, `code: none` plus the message).
Write instead: state `cpu` once.
Decision: D26, D40.

### GUST-E0711 half an artifact pair

A kernel's precompiled artifact is two files in the kernel crate's `gust-artifacts/`
directory, `<key>.wgsl` and `<key>.wgsl.reflection.json`; only one of them exists. The macro
never turns that into an empty artifact list silently. Export mode (`GUST_EXPORT_DIR`, which
`cargo gust build` sets) treats a half pair as absent, so the build tool can repair it.
Reached through `#[gpu]`.
Rejected: a build outside `cargo gust build` while `gust-artifacts/` holds only one of the two files.
Write instead: delete the file that is left, then run `cargo gust build`, which writes both.
Decision: D45, D46.

### GUST-E0712 export records not written

Export mode (`GUST_EXPORT_DIR` set) writes `<key>.slang` and `<key>.kernel.json` for each
kernel, and the write failed: the directory or a file cannot be created, or a record held
CPU twin text (a bug in the macro; records never hold a twin), or a record with the same key but other bytes already exists.
Reached through `#[gpu]`.
Rejected: `GUST_EXPORT_DIR` pointing at a path that is a file, or at a read-only directory.
Also rejected: two kernels of one package with the same 64-bit key and different text; the second one fails here instead of overwriting the first one's record (the message names both kernels).
Write instead: point `GUST_EXPORT_DIR` at a writable directory, or let `cargo gust build` choose it.
Decision: D45, D46.

### GUST-E0713 export package name

Export mode names the record directory after `CARGO_PKG_NAME`, and the value is missing or is
not a plain cargo package name (ASCII letters, digits, `-` and `_`), so it could climb out of the
export directory.
Reached through `#[gpu]`.
Rejected: `GUST_EXPORT_DIR` set while `CARGO_PKG_NAME` is unset or contains a path separator.
Write instead: build through cargo, which sets `CARGO_PKG_NAME`.
Decision: D45, D46.

## E08xx: emitter-only rejections

These are checked by the Slang emitter after the validator has accepted the module. The
validator does not check everything the emitter does, so most are reached through
`#[gpu]` (a kernel parameter that is not a resource, a `let` without an initializer, a
tuple expression, a struct literal in the wrong place). Those marked "Unreachable through
`#[gpu]`" are a second line of defence that keeps the emitter from producing wrong Slang if
the validator ever lets the shape through; unit tests call the emitter directly for them.
A rule that the validator and the emitter both enforce with the same message shares the
validator's code (for example E0116, E0117 and E0511); the codes below are for messages
only the emitter has.

### GUST-E0801 unsupported entry point parameter

Slang compute entry points accept resources and the dispatch thread id only.
Reached through `#[gpu]`.
Rejected: a kernel parameter whose type is not a resource or `SV_DispatchThreadID`.
Write instead: pass data through a buffer.
Decision: D02.

### GUST-E0802 kernel needs a dispatch id parameter

An emitted kernel has a dispatch-thread parameter to index its work.
Unreachable through `#[gpu]`: defence in depth. The validator rejects the same kernel as E0109 first.
Rejected: a `#[kernel]` function with no `SV_DispatchThreadID` (or legacy `Invocation`) parameter.
Write instead: add `id: SV_DispatchThreadID`.
Decision: D02.

### GUST-E0803 `let` needs an initializer

Slang locals are always initialised.
Reached through `#[gpu]`.
Rejected: `let x: u32;`.
Write instead: `let x: u32 = 0;`.
Decision: D02.

### GUST-E0804 item inside a function

Functions, structs and constants are declared in the module, not in a body.
Reached through `#[gpu]`.
Rejected: a `fn`, `struct` or `const` declared inside a kernel body.
Write instead: declare it in the module.
Decision: D02.

### GUST-E0805 macro statement

A macro invoked as a statement. The validator does not see statement macros, so the emitter
reports them; a macro used as an expression is E0313.
Reached through `#[gpu]`.
Rejected: `asm!("...");`.
Write instead: plain Rust statements.
Decision: D02.

### GUST-E0806 struct literal field needs a name

Struct literals name their fields.
Reached through `#[gpu]`.
Rejected: a tuple-style struct literal field.
Write instead: `Pair { first: a, second: b }`.
Decision: D02.

### GUST-E0807 unsupported `let` binding form

The emitter's own check of the `let` pattern.
Unreachable through `#[gpu]`: defence in depth. The validator applies the same check to every `let` and reports E0319 first.
Rejected: `let (a, b) = pair;`.
Write instead: one identifier, optionally typed: `let a: u32 = pair.first;`.
Decision: D02.

### GUST-E0808 atomic operand is not a mutable reference

The emitter's own check of an atomic call's first argument.
Unreachable through `#[gpu]`: defence in depth. The validator reports E0602 or E0608 first.
Rejected: `atomic_add(counter, 1u32)`.
Write instead: `atomic_add(&mut counter[0], 1u32)`.
Decision: T10.

### GUST-E0809 atomic operand is not a buffer element

The emitter's own check of an atomic call's reference.
Unreachable through `#[gpu]`: defence in depth. The validator reports E0609 first.
Rejected: `atomic_add(&mut counter, 1u32)`.
Write instead: `atomic_add(&mut counter[0], 1u32)`.
Decision: T10.

### GUST-E0810 struct literal in an unsupported position

A struct literal is a `let` initializer or the right side of an assignment.
Reached through `#[gpu]`.
Rejected: passing `Pair { first: 1, second: 2 }` directly as an argument.
Write instead: bind it to a local first, or copy a buffer element and update its fields.
Decision: D02.

### GUST-E0811 unsupported expression

An expression form the emitter has no lowering for.
Reached through `#[gpu]`.
Rejected: a tuple expression `(1u32, 2u32)`, an array repeat `[0u32; 4]`, a block used as a value `{ 1u32 }`.
Write instead: the supported expression forms listed in the validator codes above.
Decision: D02.

### GUST-E0812 unsupported literal

GPU literals are booleans and 32-bit numbers.
Reached through `#[gpu]`.
Rejected: `'a'`, `"text"`, `1u64`.
Write instead: `bool`, `i32`, `u32` or `f32` literals.
Decision: D02.

### GUST-E0813 type must be a named path

A type in a signature or `let` is a name the emitted Slang can use.
Reached through `#[gpu]`.
Rejected: tuple and reference types in a position that needs a named type, such as `(u32, u32)` or `&u32`.
Write instead: a named type such as `u32` or a struct from the module.
Decision: D02.

### GUST-E0814 `Result` needs a payload type

The emitter's own check of a `Result` type.
Unreachable through `#[gpu]`: defence in depth. The validator accepts `Result` only as `Result<T, T>` and reports E0504 first.
Rejected: `Result<>`.
Write instead: `Result<u32, u32>`.
Decision: D02.

### GUST-E0815 resource element type must be a path

The emitter's own check of a resource's element type.
Unreachable through `#[gpu]`: defence in depth. The emitter rejects a resource type that is not a path as E0813 first.
Rejected: `StructuredBuffer<*const u32>`.
Write instead: `StructuredBuffer<u32>`.
Decision: D02.

### GUST-E0816 resource needs one element type

A resource names its element type.
Reached through `#[gpu]`.
Rejected: `StructuredBuffer` with no generic arguments.
Write instead: `StructuredBuffer<u32>`.
Decision: D02.

### GUST-E0817 resource element type missing

A resource's generic argument is a type, not a lifetime.
Reached through `#[gpu]`.
Rejected: `StructuredBuffer<'a>`, a resource with only a lifetime argument.
Write instead: `StructuredBuffer<u32>`.
Decision: D02.

## E09xx: reserved names and shadowing

### GUST-E0901 reserved prefix

Names starting with `__gust_`, `__gpu_` or `__Gust` are generated shader symbols (`__GustResult`, the
generics of the typed call, and the `__gust_r_` prefix of renamed identifiers, see below). The check reads the
unrawed name, so `r#__gust_r_new` is rejected too.
Reached through `#[gpu]`.
Rejected: `fn __gust_not(x: u32) -> u32`, a parameter named `__gpu_len_out`, a local named `__GustCount`.
Write instead: any other name.
Decision: D02.

### Renamed identifiers (not a diagnostic)

A few Rust names are valid in the Rust twin but cannot be Slang identifiers, or would capture a name the generated
Slang uses: `new`, `class`, `float4`, `mul`, `uint`, `none`, `var`, `true`, `false` and the rest of the collision table in
`crates/gust-macros/src/slang/collisions.rs`. The macro emits such a name as `__gust_r_<name>` at every site (local,
parameter, helper, struct, field, kernel parameter) and leaves every other name as written; a raw prefix (`r#new`) is
stripped first. Nothing is rejected, so there is no code, and the Rust source is unchanged. A raw `r#true` or `r#false` is renamed because, left alone, it would silently change what the `true` and `false` literals that GUST writes after it mean (slangc takes the program and stores the other value); a local named `var` would break every local after it.

- **Slangc messages** have the prefix deleted before they reach you, so an error about `__gust_r_new` says `new`. The
  rename only ever writes the prefix at the start of a name, so the deletion (at a word boundary, in front of a name) undoes it; a name that merely contains the text, such as `a__gust_r_b`, is left as written. The line mapping is unchanged.
- **naga and wgpu messages are not de-renamed.** slangc has already mangled the name, so a shader-creation error names
  `_gust_r_new_0` (a leading underscore, a trailing counter). Read `_gust_r_<name>_N` as the Rust name `<name>`.
- **Exported Slang, artifacts and reflection** keep the renamed text, because it is the compiled text. Kernel
  descriptors carry the emitted names (`ParameterDescriptor.name`, `StructFieldLayout.name`), so a layout-check
  message about a renamed parameter or field names `__gust_r_<name>`, which you can search for in the exported
  `.slang`. `rust_type` keeps the Rust spelling.

Decision: M6 plan section 2.5 (renaming is a build-time text rule; the decision row lands with the M6 close).

### GUST-E0902 shadowed parameter

A local cannot reuse the name of a resource or dispatch-thread parameter, because the
emitted prologue and length symbols are keyed by it.
Reached through `#[gpu]`.
Rejected: `let out = 1u32;` when `out` is a buffer parameter, `for id in 0u32..n`.
Write instead: pick a different local name.
Decision: D02.

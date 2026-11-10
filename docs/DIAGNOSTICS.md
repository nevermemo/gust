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

This page is not part of the published packages. If you installed GUST from crates.io, read it on GitHub
(<https://github.com/nevermemo/gust/blob/main/docs/DIAGNOSTICS.md>) and search it for the code in your error. The
codes come from the `#[gpu]` macro of the package `gust-compute-macros`, which you get by adding `gust-compute` (see
"Install from crates.io" in the [README](../README.md)).

The promise starts with the first crates.io release, 0.1.0 (D33). Until then a code may still be
split or renumbered, and each change is listed here:

- Slice mx.p1a: E0730 ("pinned shape not available yet", D63) is retired: the `pixel` profile exists, so a
  `shape = pixel` pin is built as a fragment entry. The number is never reused. E0723 lost its 8-output and
  32-byte bounds (D120).
- Slice mx.p1b: a `pixel` kernel runs on `gust-wgpu` as a render pass. `Error::ShapeUnsupported` gained a `reason`
  field and refuses the pixel cases that have no path (an extent call, an indirect launch, the one-shot v1 functions,
  the generated host snippet, an adapter without fragment-stage storage writes); the new `Error::ShapeIndependentOutput`
  refuses a read-write output bound `Independent`. The twin has `cpu::Error::PixelExtent` and
  `cpu::Error::PixelIndependentOutput`, and a call over 2^26 elements is a `DeviceLimit` named `pixel_max_elements`.
  These are run-time errors, not macro codes.
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

**Retired limitation: slangc refused a zero divisor that it could fold, in code that never
runs** (found by the differential test, slice m5.c2; cured by slice m6.dv, D74). slangc
2026.20 evaluates an integer `/` or `%` at compile time when it can fold the divisor to a
constant, and it rejects a zero with `error[E30002]: divide by zero`, in a branch that no
invocation takes as well (the counter of an unrolled `for` at its first iteration, a
product with a zero factor such as `i * 0`, the difference `i - i`, wrapping arithmetic,
a condition on known values). The smallest reproducer is
`for i in 0u32..4u32 { if !true { let r = i % i; } }`, which used to be refused at the `%`.
The macro now lowers every `/` and `%` whose divisor is not a literal to the call
`__gust_div(a, b)` or `__gust_rem(a, b)` (`/=` and `%=` to `__gust_div_assign` and
`__gust_rem_assign`, with the place `inout`), a helper whose body is the plain operator and
which slangc does not fold; the reproducer and the other shapes compile on WGSL and on
SPIR-V (the Slangc-tier test `every_division_shape_slangc_folded_compiles_on_each_target`),
and the differential generator no longer refuses them (DIFFTEST_FORMAT 5,
[DIFFTEST](development/DIFFTEST.md)). A divisor that is zero at run time keeps the results
of D39 and KM-01 to KM-03 in [SEMANTICS](SEMANTICS.md); that was measured with divisors read
from a buffer. A closed zero that is live (`let zero = 0u32; 5 / zero`) goes through the helper and
compiles, but no test executes it, and the differential generator discards it. What stays: a literal zero divisor
is GUST's own error E0317 below, and slangc's behaviour for a zero divisor on the SPIR-V
route is undefined in Slang and was not executed (the WGSL route, which the GPU tests run,
is defined by WGSL).

## E01xx: module and item shape

### GUST-E0101 inline module required

`#[gpu]` needs the module body in the attribute's own item.
Reached through `#[gpu]`.
Rejected: `#[gpu] mod shaders;`.
Write instead: `#[gpu] mod shaders { ... }`.
Decision: D02.

### GUST-E0102 unsupported module item

A GPU module holds functions, plain structs, field-less enums (GUST-E0122) and the inherent `impl` blocks of its structs (several blocks for one struct are allowed).
Reached through `#[gpu]`.
Rejected: `const`, `static`, `trait`, `use`, nested `mod`, type aliases; an `impl` that is an `unsafe impl`, a trait impl (`impl Clone for S`), an associated `const` or an associated type, a `Self` anywhere but inside an `impl` block (inside one it is the struct of the block, in a type, a struct literal or a path such as `Self::make()`). A path call of a function that has a receiver (`S::area(c)`; write `c.area()`) or of an associated function that no struct has (`S::nowhere()`) is GUST-E0302, not this code. Methods and associated functions are accepted since M9 slices m9.a1 and m9.a2.
Write instead: pass constants as uniform or value parameters, or write them inline.
Decision: D02, D133 (methods), D154.

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

A `#[kernel]` function takes exactly one dispatch-thread parameter. A kernel that takes `LocalId` or `GroupId` needs it too, since those ids are derived from the dispatch id and the workgroup size.
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
Rejected: `#[cfg(..)]`, `#[inline]`, `#[must_use]` and every other attribute. On an `impl` block only `doc`, `allow`, `warn`, `deny` and `forbid` are accepted (`#[derive(..)]` and `#[kernel]` are errors in rustc there, `#[cfg(..)]` would let the Slang and the twin disagree about the methods).
Write instead: remove the attribute.
Decision: D02.

### GUST-E0116 `self` parameter

A free function has no receiver. A method takes `&self`, `&mut self` or `self` (a `&mut self` method is lowered with the Slang attribute `[mutating]` and can be called on a local place only, GUST-E0333 for a storage, uniform, workgroup or invocation-id place). A function of an `impl` block without a receiver is an associated function (`static` in Slang). The receivers `mut self` and the typed forms (`self: Box<Self>`, `self: &Self`) are refused at once, where they stand, with this code.
Reached through `#[gpu]`.
Rejected: `fn f(&self) -> u32` outside an `impl`, `impl S { fn bump(mut self) {} }`, `impl S { fn area(self: Box<Self>) {} }`.
Write instead: `&self` for a method that reads, `&mut self` for a method that changes its receiver, `self` for a method that consumes a copy; no `mut` on the receiver. In a `&mut self` method `self` is used only as the base of a field (`self.n`) or the receiver of a call (`self.bump()`): `let c = self;` binds the reference in Rust and copies in Slang, so it is this code too; write through `self.field` directly (a `self` or `&self` method may copy it, as in Rust).
Fixture: `tests/ui/method-self-mut-binding` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D02, D154 (M9 plan 2.1).

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

### GUST-E0119 `Workgroup` parameter on a helper

Workgroup memory belongs to a kernel: a kernel's `Workgroup<[T; N]>` parameter becomes a `groupshared` array of its entry point, and a helper (or a map function) may be called under a condition, so it can neither declare the array nor receive it.
Reached through `#[gpu]`.
Rejected: `fn helper(mut s: Workgroup<[f32; 64]>) { .. }`.
Write instead: declare the array on the kernel and pass the helper the values it needs, `helper(s[lid.x])`.
Decision: D42 (M7 plan 4.2 rule 4, slice m7.g1).

### GUST-E0120 recursive call

Recursion has no GPU lowering, and the dialect keeps its call graph acyclic. The check builds a conservative graph of the module from names alone, without types: a call `f(..)` is an edge to the helper `f`, a call `x.m(..)` is an edge to every method named `m` of every `impl` block, and a path call `S::f(..)` or `Self::f(..)` is an edge to the function `f` of the blocks of `S`. A cycle is reported at the call that closes it, with the chain (`a -> b -> a`). Because the receiver's type is not known, a method that calls another struct's method of the same name is a cycle too (the message says so); rename one of them. The graph carries no uniformity: every callee is validated as a helper (value parameters only, no barrier, subgroup operation, atomic or buffer access), which keeps D154 closed.
Reached through `#[gpu]`.
Rejected: `fn f(n: u32) -> u32 { f(n - 1u32) }`, `fn a() { b() } fn b() { a() }`, `impl S { fn go(&self) { self.go() } }`, and a helper that calls a method that calls the helper.
Write instead: a `for` loop with a fixed trip count that does the same work.
Fixture: `tests/ui/method-recursion-direct`, `tests/ui/method-recursion-mutual` and `tests/ui/method-recursion-through` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D162 (M9 plan 1 item 3, slice m9.a2).

### GUST-E0121 `impl` for a type the module does not define

An `impl` block of a GPU module must be an inherent block (`impl S { .. }`) of a struct declared in the same module, so that the methods become members of the generated Slang struct. A trait impl with a blanket form (`impl<T> Anything for T`), an impl for a type that is not a struct of the module (a built-in vector or matrix, a tuple, a path to another module, an undeclared name) is rejected with this code. Several inherent blocks for one struct are accepted (slice m9.a2), and so is the path form `impl self::S`; an attribute on an `impl` block (`#[cfg(..)]`, `#[inline]`) is GUST-E0115, and a trait impl for a module struct is GUST-E0102. A generic impl or method is GUST-E0114.
Reached through `#[gpu]`.
Rejected: `impl Elsewhere { .. }`, `impl Mat4 { .. }`, `impl<T> Anything for T { }`, `impl (u32, u32) { }`.
Write instead: declare the struct in the same `#[gpu]` module and put the methods in its `impl` block.
Fixture: `tests/ui/method-impl-foreign` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D154 (M9 plan 2.1, slice m9.a1).

An `impl` block for an enum of the module (`impl Mode { .. }`) is this code too, with the message "methods on an enum are not supported": enums have no methods in M9 (slice m9.e1); write a free function that takes the enum.

### GUST-E0122 enum shape outside the contract

A `#[gpu]` module accepts a field-less `enum` whose variants become the values of a Slang `enum E : uint { A, B = 3, C };`. The macro adds `#[derive(Clone, Copy)]` and `#[repr(u32)]` itself (an explicit `#[repr(u32)]` or `#[repr(i32)]` is allowed, and `i32` makes a signed Slang enum); the user derives `PartialEq` (and `Eq`), because `==`, `!=` and `match` compare enum values and the CPU twin must be able to do the same. Discriminants are integer literals (`B = 3`, and `-2` with `#[repr(i32)]`); a variant without one counts up from the one before, in Rust and in Slang alike. An enum casts with `as` to its repr type only (`as u32`, or `as i32` for the signed enum). The check is by name: an enum parameter, a local declared with the enum, a variant path (`Mode::A as u32`) or a call of a helper that returns the enum; a cast of any other expression is not tracked.
Reached through `#[gpu]`.
Rejected: a variant with data (`A(u32)`, `A { x: u32 }`); a discriminant that is not an integer literal (`A = 1 + 2`, `A = 1.5`) or is negative without `#[repr(i32)]`; a missing `PartialEq` derive; a generic enum or one with no variants; a repr other than `u32` or `i32` (`#[repr(u8)]`, `#[repr(u32, C)]`); `Clone` or `Copy` derived by hand; an attribute on a variant other than documentation and lints; `m as i32`, `m as f32` or `Mode::A as i32` on a `repr(u32)` enum, and `s as u32` on a `repr(i32)` one.
Write instead: the shape above; for a value with data, use a struct and an integer tag, and convert at the cast.
Name collisions (not checked by the macro, loud in slangc, and the same for a struct): an enum named like a Slang modifier that the emitter does not rename, or like a function of the module, gives a slangc error at the declaration or a use, not a silent change; rename the enum. Names that every identifier shares (`class`, `new` ..) are renamed by the emitter and are no clash.
Fixture: `tests/ui/enum-data-variant`, `tests/ui/enum-discriminant`, `tests/ui/enum-no-partialeq` and `tests/ui/enum-cast-type` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D157 (M9 plan 2.2, slice m9.e1).

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
Rejected: a struct field `v: Vec3`, `m: Mat3`; a resource element `StructuredBuffer<Vec3>`, `RWStructuredBuffer<UVec3>`, `Storage<Mat3>`; a parameter or the result of a `#[gpu]` map function (`#[gpu] fn f(v: Vec3) -> f32`), whose arguments and results are buffer elements.
Write instead: `Vec4` or three scalars in buffers; the 3-wide types stay allowed as locals, helper parameters and helper results (also inside the body of a map function).
Decision: D17 (M6 plan F4, F5), D60.

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
Reached through `#[gpu]`. A gap that a field's own alignment causes (and so `#[repr(C)]` has too) is GUST-E0208 first, as is a tail past the largest field alignment; a gap caused only by the 16-byte alignment of a nested struct, and a tail that only the rounding up to 16 causes, are GUST-E0209. A struct with a field type the macro cannot see into is left to the `const` assertions of the expansion, which fail compilation instead.
Rejected: `struct Params { a: f32 }` (ends at byte 4, a constant buffer rounds it up to 16), `struct Params { a: f32, b: f32, c: f32 }`, `struct Params { a: f32, inner: Inner }` where `Inner` is four `f32` (`inner` would sit at byte 16, 12 bytes after `a` ends), each when a `Uniform<Params>` parameter reaches it. The same structs are fine as storage elements.
Write instead: explicit padding fields, `struct Params { a: f32, pad0: f32, pad1: f32, pad2: f32 }`, or order the fields so that whole 16-byte rows are filled.
Decision: D17, D40 (M6 plan 2.3, F7).

### GUST-E0210 malformed `Workgroup` type

`Workgroup<[T; N]>` is the type of a kernel parameter and declares `N` elements of workgroup memory (`groupshared` in Slang) that every invocation of one workgroup shares. `N` is a nonzero integer literal, so the size is known when the macro runs, and `T` is `f32`, `i32`, `u32`, `bool` or a vector type (`Vec2`, `Vec3`, `Vec4`, `IVec2`, `IVec4`, `UVec2`, `UVec3`, `UVec4`); a matrix, a struct and a nested array are not accepted. The type stands nowhere but as the type of a kernel parameter: not as a local, a field, a helper parameter or a result. The macro sees no types, so `Workgroup`, `LocalId`, `GroupId`, `SV_GroupThreadID` and `SV_GroupID` are taken by their spelling, as `UVec3` is: a user type of one of these names is read as the dialect's type.
Reached through `#[gpu]`.
Rejected: `Workgroup<f32>`, `Workgroup<[f32; 0]>`, `Workgroup<[f32; LEN]>`, `Workgroup<[f64; 4]>`, `Workgroup<[Mat4; 4]>`, `Workgroup<[Item; 4]>`, `let s: Workgroup<[f32; 4]> = ..;`, `fn f() -> Workgroup<[f32; 4]>`.
Write instead: `mut scratch: Workgroup<[f32; 256]>` as a kernel parameter, with a literal length and a supported element.
Decision: D42 (M7 plan 4.1, slice m7.g1).

### GUST-E0211 enum in a struct field, a buffer, a uniform, workgroup memory, an `Option` or a `Result`

An enum is a value of the invocation and is never stored: it has no `GpuPod` layout (a stored word could be none of its variants), and `Option` and `Result` have no proven layout for it. The enum lives in locals, parameters and results of helpers; store its integer (`mode as u32`) and match on it where it is read.
Reached through `#[gpu]`.
Rejected: `struct S { m: Mode }`, `StructuredBuffer<Mode>`, `RWStructuredBuffer<Mode>`, a uniform or `Workgroup<[Mode; N]>` parameter, `Option<Mode>` and `Result<Mode, Mode>` in a result or a local (a struct field of `Option` type is GUST-E0501 first).
Write instead: a `u32` (or `i32`) field or element holding `mode as u32`, and a helper that turns it back into the enum.
Fixture: `tests/ui/enum-in-field` and `tests/ui/enum-in-buffer` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D157 (M9 plan 2.2, slice m9.e1).

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
`.wrapping_add(b)`, `.wrapping_sub(b)` and `.wrapping_mul(b)` take one argument and `.wrapping_neg()` takes none
(slice m7.w). A method that is not a built-in at all is not this code: with arguments it is E0325, without
arguments E0305.
Reached through `#[gpu]`.
Rejected: `out.len(1)`, `v.xy(1)`, `m.transpose(m)`, `Vec4::new(1.0, 2.0)`, `x.wrapping_add()`, `x.wrapping_neg(1)`.
Write instead: `out.len()`, `v.xy()`, `m.transpose()`, `Vec4::new(1.0, 2.0, 3.0, 4.0)`, `x.wrapping_add(y)`,
`x.wrapping_neg()`.
Decision: D02.

### GUST-E0305 unsupported method

Methods other than `.global_id()`, `.len()`, `.is_some()`, `.is_none()`, `.is_ok()`,
`.is_err()`, `.unwrap_or(default)`, the wrapping integer methods (`.wrapping_add(b)`, `.wrapping_sub(b)`,
`.wrapping_mul(b)`, `.wrapping_neg()`; slice m7.w) and the vector and matrix methods (swizzles `.xy()`..`.wzyx()`,
`.dot(v)`, `.mul_vec2/3/4(v)`, `.mul_mat2/3/4(m)`, `.transpose()`) have no lowering. This code is the one for a method called without arguments;
the same unknown method called with arguments is E0325. The message lists the accepted methods.
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
Rejected: `x / 0u32`, `x % 0`, `x /= 0`, `x %= 0`, and the same zeros negated or in parentheses (`x / (0)`,
`x % -0`), including a zero a `macro_rules!` hands over in an invisible group.
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
Rejected: `*params`, `let p = params;`, `(params)` alone, `params` as a function argument, `params.0` (the wrapper's own field, which reaches the whole struct), and `mut params` (a constant buffer is read-only). `params.len()` is GUST-E0306, and a helper taking a uniform is GUST-E0111.
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

### GUST-E0325 method is not in the dialect

A method call with arguments whose name is not one the dialect accepts. The message names the method and lists the
accepted ones: `.global_id()` and `.len()`, the Option and Result methods (`.is_some()`, `.is_none()`, `.is_ok()`,
`.is_err()`, `.unwrap_or(default)`), the wrapping integer methods `.wrapping_add(b)`, `.wrapping_sub(b)`,
`.wrapping_mul(b)` and `.wrapping_neg()`, and the vector and matrix methods. Before slice m7.w this case was E0304 with
the text "GPU built-ins do not accept arguments or type arguments", which blamed a built-in that was never called.
`wrapping_div` and `wrapping_rem` are deliberately absent: a divisor of zero (and `MIN / -1`) has no wrapped result, so
a "wrapping" division would still leave the domain (KM-01 to KM-03 in [SEMANTICS](SEMANTICS.md)). The same unknown
method called without arguments is E0305; a built-in called with arguments it does not take is E0304. A type
argument on such a name (`x.checked_add::<u32>(y)`) is this code as well, not E0303.
Reached through `#[gpu]`.
Rejected: `x.wrapping_div(y)`, `x.wrapping_rem(y)`, `x.pow(2)`, `x.min(y)`, `x.checked_add(y)`, `x.saturating_sub(y)`.
Write instead: the operator or an `if` (`if y != 0 { x / y } else { 0 }`), or one of the accepted methods.
Decision: D22 (the dialect boundary), D87 (the wrapping methods and this code).

### GUST-E0326 unsuffixed integer-literal `let`

An unannotated `let` whose initializer is made only of integer literals without a suffix gives the local no type of
its own. Slang types it `int`, while rustc infers the type from the later uses (`u32` for `d /= y` with `y: u32`), and
the two differ for a division, a comparison, a shift or a cast of the local (`d /= y` with `y >= 2^31` is 0 in Rust and
`0xFFFFFFF9` on the GPU). The error points at the first literal and asks for a suffix or an annotation. Covered: a bare
literal, through parentheses and unary `-` or `!` (`let x = -(7);`); arithmetic and bitwise operators whose operands are
all such literals (`let x = 1 + 2;`); a shift whose left operand is one (`let x = 1 << 3;`, `let x = 1 << n;`, because
the result takes the left operand's type); `Some(..)`, `Ok(..)` or `Err(..)` of such a literal; and `X.unwrap_or(Y)`
with both literal-only (`Some(7).unwrap_or(0)`). The same check runs on the scrutinee of an `if let`
(`if let Some(mut v) = Some(7)`), where the span is again the literal. Not covered, because the initializer has a type
of its own or is not an integer: a suffix, a cast, a name, a field, an index, a call or a method anywhere in the
operands (`let x = 3 * i;`; the one literal-only method is `unwrap_or` above), and a comparison or a `bool`. A float
literal (`let x = 0.0;`) is not covered either, although rustc falls back to `f64` for it where Slang has `float`: see
the float paragraph in [SEMANTICS](SEMANTICS.md). An `if` or a `match` is not a value in the dialect
(`let x = if c { 1 } else { 2 };` is E0811 and a `match` as a value is E0513), so there are no literal arms to cover.
`while let` and `let .. else` are E0402 and E0320. Slice m6.z2 chose the error alone over inferring the type from the
uses (D86).
Reached through `#[gpu]`.
Rejected: `let mut acc = 0;`, `let x = -1;`, `let x = 1 + 2;`, `let x = 1 << 3;`, `let o = Some(7);`,
`let d = Some(7).unwrap_or(0);`, `if let Some(v) = Some(7) { .. }`.
Write instead: `let mut acc = 0u32;`, `let mut acc: u32 = 0;`, `let x: i32 = -1;`.
Decision: D86.

### GUST-E0327 `workgroup_barrier` is a statement with no arguments

`workgroup_barrier();` makes every invocation of the workgroup wait until all of them have reached it, and makes the workgroup memory they wrote visible to the others (`GroupMemoryBarrierWithGroupSync()` in Slang, `workgroupBarrier()` in WGSL, `OpControlBarrier` in SPIR-V). It is called as a statement of its own, ending in `;`, with no arguments; it is not a value, a function pointer or part of an expression.
Reached through `#[gpu]`.
Rejected: `workgroup_barrier(1u32);`, `let x = workgroup_barrier();`, `let f = workgroup_barrier;`, `if c { workgroup_barrier() }` (no `;`).
Write instead: `workgroup_barrier();`. Where the barrier may stand is GUST-E0414.
Decision: D42 (M7 plan 4.1, slice m7.g1).

### GUST-E0328 method name clashes with a built-in

A method of a module struct becomes a member function of the generated Slang struct and is called as `recv.name(args)`, so its name must not be one that a call already means something else for: a method of the dialect (`len`, `global_id`, `is_some`, `unwrap`, `wrapping_add`, `dot`, `mul_vec4`, `transpose`, the texture methods `load`, `store`, `sample`, `width`, `height` and every other built-in method, GUST-E0325), a vector swizzle (`xy`, `xyz`, `wzyx` and every other two to four letters of `x`, `y`, `z`, `w`), a word GUST reserves as a method name because it begins a member declaration of a Slang struct or names the struct (`This`, `init`, `property`, `subscript`, `this`), or the name of a field of its own struct (the getter pattern `fn radius(&self)` next to a field `radius`, where `s.radius` and `s.radius()` would collide as Slang members). Reserved Slang names that every identifier shares are renamed by the emitter and are no clash.
A helper function and a method must not share a name, and a helper must not be named like a field of the struct whose method calls it: inside the Slang struct an unqualified call finds a member before a free function (measured on slangc 2026.21), so the call would reach the member or fail. The rule is a check of the module, the same for a name in the call of a method and in the call of a helper.
Reached through `#[gpu]`.
Rejected: `impl S { fn len(&self) -> u32 { .. } }`, `fn xy(&self) -> f32`, `fn init(&self) -> u32`, `fn v(&self) -> u32` in `struct S { v: u32 }`.
Write instead: another name, `length`, `radius_value`, `get_v`.
Fixture: `tests/ui/method-name-clash` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D154 (M9 plan 2.1, slice m9.a1).

### GUST-E0333 mutating method on a shared place

A method that takes `&mut self` or `self` would write through its receiver. On a storage buffer element, a uniform or a workgroup array that write is shared with the other invocations, and no statement of the kernel shows it, so a call whose receiver is rooted in such a parameter is rejected at the call. The validator decides from the first identifier of the receiver place (`out` in `out[i].bump()`), by the method name: a name that a `&mut self` or `self` method of any struct of the module has is a mutating call. A `&self` method may be called on any place, the callee reads a copy of the element (spike M9.s0 P1b). A `&mut self` method is accepted on a `let mut` local (and a field or element of one) only, so the same code covers a receiver that is not a place (`make().bump()`, a literal) and a root that is a by-value parameter (Slang passes it as a read-only value; write `let mut copy = c;` first).
Reached through `#[gpu]`.
Rejected: `out[0].bump()` and `out[i].bump()` for a storage parameter `out`, `u.bump()` for a uniform, `scratch[l].bump()` for a workgroup array, when a method `bump` takes `&mut self`; `lid.bump()` and `gid.bump()` (invocation ids) too. A texture receiver (`tex.bump()`) is GUST-E0735 instead, because the texture methods are checked first.
Write instead: copy the element to a local, call the method on the local and store the result, `let mut c = out[i]; c.bump(); out[i] = c;`.
Fixture: `tests/ui/method-mut-on-storage` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D154 (M9 plan 2.1, X1 of spike M9.s0, slice m9.a1).

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

### GUST-E0414 barrier at a position not every invocation reaches

A control barrier is undefined unless every invocation of the workgroup reaches it the same number of times. The dialect cannot prove that for arbitrary control flow, so it admits two positions in a kernel that uses workgroup memory or a barrier: a statement of the kernel body, and a statement of the body of a `for` loop that is itself a statement of the kernel body and whose range is two integer literals (`for round in 0u32..8u32`). Every invocation runs that loop for the same trip count. A barrier anywhere else is rejected: in an `if` or a block, in a loop whose bounds are not two literals, in a loop nested in another loop, and in a helper, a method or a map function (a helper or a method may be called under a condition). `while` and `loop` are rejected for their own sake (GUST-E0402, GUST-E0401), so a barrier in one is reported by that code and not by this one.
Reached through `#[gpu]`. The check runs when the rest of the function has been accepted, so a function holds one report, not two.
Rejected: `if lid.x < 8u32 { workgroup_barrier(); }`, `for r in 0u32..4u32 { if r == 1u32 { workgroup_barrier(); } }`, `for r in 0u32..4u32 { for q in 0u32..4u32 { workgroup_barrier(); } }`, `for r in 0u32..n { workgroup_barrier(); }`, `fn sync() { workgroup_barrier(); }`, `impl S { fn sync(&self) { workgroup_barrier(); } }`.
Fixture: `tests/ui/barrier-in-if`, `tests/ui/barrier-in-nested-for`, `tests/ui/barrier-in-helper`, and `tests/ui/barrier-in-while` (GUST-E0402; the `while` is rejected first) (`cargo xtask check-ui`, `code: none` plus the message).
Write instead: do the guarded work in an `if` and call the barrier after it, at the top level of the kernel body or of a literal-range loop.
Decision: D42 (M7 plan 4.2 rule 1, slice m7.g1).

### GUST-E0415 `return` in a kernel with workgroup memory or a barrier

An invocation that returns early never reaches the barriers the others wait at, so a kernel that uses workgroup memory or a barrier has no `return`: its body ends by falling off the end. A `return` in a helper that such a kernel calls is fine; the helper contains no barrier. The `?` operator is an early exit too, so the barrier pass also rejects it in such a kernel with this code; today GUST-E0311 rejects `?` in every function first, so the pass is the default that holds if a later slice admits `?` (tested by calling the pass directly). A kernel that calls a subgroup operation (M8, D140) is held to the same rule, because the operation is a rendezvous of the subgroup too; its message then leads with the subgroup case.
Reached through `#[gpu]`.
Rejected: `workgroup_barrier(); return;`, `if lid.x == 0u32 { return; }` in a kernel that has a `Workgroup` parameter, and a `return` in an arm of an enum `match` of such a kernel (a `return` in an arm of a helper is legal).
Fixture: `tests/ui/return-in-barrier-kernel` and `tests/ui/x2-return-in-match-arm` (`cargo xtask check-ui`, `code: none` plus the message).
Write instead: guard the rest of the work with `if`.
Decision: D42 (M7 plan 4.2 rule 2, slice m7.g1).

### GUST-E0416 `break` or `continue` in a loop that holds a barrier

A `break` or `continue` lets some invocations leave a loop, or skip the rest of an iteration, while the others wait at a barrier inside it. The rule covers every unlabelled `break` and `continue`, conditional or not, whose innermost loop contains a `workgroup_barrier()`. A loop without a barrier may use its own `break` and `continue`. A labelled `break` or `continue` is rejected in a kernel with workgroup memory or a barrier by the barrier pass with this code, whichever loop holds the barrier; today a loop label is GUST-E0403 in every kernel and that error comes first, so the pass is the default that holds if a later slice admits labels (tested by calling the pass directly). The `?` operator is GUST-E0415 in such a kernel and GUST-E0311 in every function; no second report is made. The same holds for a loop that holds a subgroup operation (M8, D140).
Reached through `#[gpu]`.
Rejected: `for r in 0u32..4u32 { if r == 2u32 { break; } workgroup_barrier(); }`, `for r in 0u32..4u32 { if r == 2u32 { continue; } workgroup_barrier(); }`.
Fixture: `tests/ui/break-in-barrier-loop`, `tests/ui/continue-in-barrier-loop`, `tests/ui/labelled-break-in-barrier-kernel` (GUST-E0403) and `tests/ui/question-mark-in-barrier-kernel` (GUST-E0311) (`cargo xtask check-ui`, `code: none` plus the message).
Write instead: run the loop to its end and guard the work in each iteration with `if`.
Decision: D42 (M7 plan 4.2 rule 2, slice m7.g1).

### GUST-E0417 `break` or `continue` in an arm of an enum `match`

A statement `match` on an enum is a Slang `switch`, and in Slang a `break` ends the `switch`, not the loop around it, so a `break` written in an arm and aimed at an enclosing `for` loop would mean something else on the GPU than in Rust. A `continue` is refused for symmetry with `break`, not because it would go wrong: measured, Slang's `continue` inside a `switch` does continue the loop around it (the refusal is conservative, and can be lifted by a later slice). The rule covers an unlabelled `break` or `continue` whose innermost enclosing construct is an arm, however deep the `if`s inside it; a `for` loop inside the arm is its own target, and a jump after the `match` is the loop's.
Reached through `#[gpu]`.
Rejected: `for k in 0u32..4u32 { match m { Mode::A => { break; } _ => { } } }`, the same with `continue;`, and a `break` in an arm of a `match` nested in an arm.
Write instead: set a flag in the arm and test it after the `match`, `if done { break; }`, or move the `match` into a helper that returns.
Fixture: `tests/ui/match-break-in-arm` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D157 (M9 plan 2.2, slice m9.e1).

### GUST-E0418 guard on an enum `match` arm

A statement `match` on an enum lowers to a `switch`, which has no guards, and a guard (`Mode::A if x > 1 => ..`) would make the coverage of the arm depend on a run-time value. The arm takes no `if`.
Reached through `#[gpu]`.
Rejected: `match m { Mode::A if x > 1u32 => { .. } _ => { .. } }`.
Write instead: test the condition inside the arm with an `if`.
Fixture: `tests/ui/match-guard` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D157 (M9 plan 2.2, slice m9.e1).

### GUST-E0419 value `match` outside a typed `let`

A `match` that gives a value is a Slang `switch` whose arms assign a local, and Slang needs the local declared with its type before the `switch`. So the value form is accepted in one place only, the initializer of a `let` with a type (`let x: T = match e { .. };`). The same `match` as the initializer of an untyped `let`, as an argument, an operand, the value of an assignment, a `return`, a `for` bound, a condition or the tail of a function with a result type (`fn f(v: u32) -> u32 { match v { 0 => 1u32, _ => 2u32 } }`) is refused with this code; a tail `match` whose arms `return` or are unit blocks is the statement form and stays accepted. A `match` on a `Result` in these places keeps GUST-E0513. Each arm gives a value: an expression, or a block whose last line is an expression without `;` (or a `return`), anything else is GUST-E0518.
Reached through `#[gpu]`.
Rejected: `let x = match v { 0 => 1u32, _ => 2u32 };`, `out[i] = match v { .. };`, `f(match v { .. })`, `1u32 + match v { .. }`, and `match v { .. }` as the tail of a function that gives a `u32`.
Write instead: `let x: u32 = match v { 0 => 1u32, _ => 2u32 };` and use `x`.
Fixtures: `tests/ui/match-value-untyped`, `tests/ui/match-value-position` and `tests/ui/match-value-tail` (`cargo xtask check-ui`, `code: none` plus the message); the control case is `tests/ui/match-value-ok`.
Decision: D157 (M9 plan 2.2, slice m9.e2).

### GUST-E0421 `&mut self` call whose evaluation order differs from Rust's

A call of a `&mut self` method is refused where the GPU would evaluate it in another order than Rust (a method that only reads, a `self` method and a helper have no effect to reorder). Three places:
- the start or the end of a `for` range: the emitted Slang loop evaluates the end bound before the start bound (spike M9.s0 P9, measured on the WGSL and the SPIR-V target), where Rust evaluates the start first (`for i in c.n..c.next()`);
- anywhere in the place (the assignee) of `=` or of a compound assignment: Rust evaluates the assigned value before the place, Slang evaluates a call inside the place first (`out[c.next()] = c.next()` writes another element, `out[d.bump()] = d.n` stores the new value; review m9.a2);
- the value of `*=` in a module that names `Mat2`, `Mat3` or `Mat4`: it is lowered to `place = __gust_mul(place, value)`, which reads the place before the call and loses the method's write.
- the value of `/=`, `%=` or `^=` on an indexed place: these three lower to `inout` helpers (`__gust_div_assign`, `__gust_rem_assign`, `__gust_xor_assign`) and Slang computes the index before the call, so `out[c.n] /= c.next()` changes another element than Rust (a place with no index, `e.n /= e.bump()`, has no address to compute early; no other operator lowers through a helper).

Forms that already match Rust stay accepted: `out[c.n] = c.next()`, `out[c.n] += c.next()`, `c.n += c.bump()`, `e.n /= e.bump()`, a struct literal that holds a call, and `*=` with a call in a module without a matrix.
Reached through `#[gpu]`.
Rejected: `for i in 0u32..c.next() { .. }`, `for i in c.n..c.next() { .. }`, `out[c.next()] = v;`, `out[c.next()] += v;`, `out[c.n] /= c.next();` (also `%=`, `^=`), `c.n *= c.bump();` (with a matrix in the module), when `next` and `bump` take `&mut self`.
Write instead: call the method before the statement and use the value, `let n = c.next(); for i in 0u32..n { .. }` or `let k = c.next(); out[k] = v;`.
Fixture: `tests/ui/for-bound-mutating` and `tests/ui/method-mutating-in-assignee` (`cargo xtask check-ui`, `code: none` plus the message).
Decision: D154 (M9 plan 2.1, spike M9.s0 P9, slice m9.a2).

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

A `match` on a `Result` is a statement, not an expression with a value (an enum or integer `match` that gives a value is GUST-E0419 outside a typed `let`). For the statement form of an enum or integer `match` this also fires when an arm ends in a value (an expression without `;`, such as `Mode::A => { c.reset() }`): end the arm's last statement with `;`.
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

Only a known `Result<T, T>` local, parameter or same-module helper call can be matched in a `Result` match, and only a local, a parameter, a call or a variant of a known enum in an enum `match` (a larger expression is bound to a local first); an integer `match` takes any `u32` or `i32` expression.
Reached through `#[gpu]`.
Rejected: a match on an `Option`, `match a + b { E::A => { .. } .. }` (an enum pattern on a larger expression) and a `match` of `_` alone on something that is not an enum.
Write instead: bind the value to a `Result<T, T>` local first, or use `if let` for `Option`.
Decision: D14.

### GUST-E0516 match arm attribute

A match arm carries no attributes, in a `Result` match or in an enum `match`.
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

Each arm body is a block, in a `Result` match and in a statement `match` on an enum or an integer; in a value `match` an arm is an expression or a block that ends in an expression.
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

In a `Result` match the arm patterns are `Ok(identifier)` and `Err(identifier)`; in an enum `match` they are `E::A`, `E::A | E::B` and `_` (the enum's own variants, each named once, none after `_`); in an integer `match` on a `u32` or `i32` they are integer literals (`0`, `1 | 2`, `3u32`, `-1i32`, within 32 bits, each value once) and a final `_`, which is required. A pattern of another shape is refused with this code: a range, a binding, a constant (`u32::MAX`), a `bool`, float or character literal, a literal of another integer type, a value out of range, a repeated or unreachable arm, a missing `_` in an integer `match`, and a mix of variants and literals.
Reached through `#[gpu]`.
Rejected: `_ => { .. }` in a `Result` match, `Ok((x, y)) => { .. }`, in an enum `match` a variant of another enum, a variant named twice or after `_`, a binding or a literal, and in an integer `match` `0u32..=3u32`, `n`, `u32::MAX` or no `_`.
Fixtures: `tests/ui/match-pattern-range`, `tests/ui/match-pattern-binding`, `tests/ui/match-pattern-const` and `tests/ui/match-pattern-repeat` (`cargo xtask check-ui`, `code: none` plus the message).
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

### GUST-E0701 `#[gpu]` takes one argument, `route = "<tag>"`

The attribute has one option, `route = "<tag>"` (build-time routing, M8: GUST-E0748, GUST-E0749); anything else, a second argument included, is this error.
Reached through `#[gpu]`.
Rejected: `#[gpu(debug)] mod shaders { ... }`, `#[gpu(route = 5)]`, `#[gpu(route = "a", "b")]`.
Write instead: `#[gpu] mod shaders { ... }`, or `#[gpu(route = "physics")] mod shaders { ... }` with the tag declared in `Cargo.toml`.
Decision: D02, D65.

### GUST-E0702 item is not a module or a function, or does not parse as one

`#[gpu]` goes on an inline module whose body parses as Rust, or on a free scalar function (the
map calls of D60, GUST-E0714). The message after the code is `syn`'s own parse error when the
tokens are not an item.
Reached through `#[gpu]`.
Rejected: `#[gpu] struct S;`, `#[gpu] const X: u32 = 1;`, `#[gpu] use ..;`, a module with a syntax
error inside, or tokens that do not parse as an item.
Write instead: put the item inside `#[gpu] mod shaders { ... }` and fix the syntax error.
Decision: D02, D60.

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

`workgroup_size`, `cpu` and `shape` are the only `#[kernel]` options.
Reached through `#[gpu]`.
Rejected: `#[kernel(threads(64))]`.
Write instead: `#[kernel(workgroup_size(x, y, z))]`, `#[kernel(cpu = "unsupported")]` or `#[kernel(shape = grid)]`.
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
`--cfg gust_dual`. The only value is `"unsupported"`. `"lockstep"` would name a CPU twin that runs a barrier kernel
one workgroup at a time on scoped threads; GUST defines no CPU strategy for barrier kernels (D118), so it is rejected
with the same code as any other unknown value, with a message of its own that says that.
Reached through `#[gpu]`.
Rejected: `#[kernel(cpu = "off")]`, `#[kernel(cpu = "lockstep")]`.
Fixture: `tests/ui/kernel-cpu-bad-value` and `tests/ui/kernel-cpu-lockstep` (`cargo xtask check-ui`, `code: none` plus the message).
Write instead: `#[kernel(cpu = "unsupported")]`, or no `cpu` option to keep the twin.
Decision: D26, D40, D118.

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
Also rejected (a known limitation): two `par_for!` calls with equal bodies that start on the same line of the same file with different line tables, which is what one `macro_rules!` definition makes when it expands the call twice (the expansions are reported on the line of the call inside the definition), because both would write one call-site record. Change the body of one, or move the call out of the macro into a function that the macro calls.
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

### GUST-E0714 map function signature

`#[gpu]` on a free function generates a grid-stride kernel and the element-wise calls of the function (`saxpy::map`, `saxpy::map_on`, `saxpy::map_gpu`, `saxpy::map_async`, ..), so its signature must be one the generator can make a kernel of: at least one parameter, each a plain identifier of type `f32`, `i32`, `u32`, `bool` or a vector or matrix buffer element (`Vec2`, `Vec4`, `IVec2`, `IVec4`, `UVec2`, `UVec4`, `Mat2`, `Mat4`), by value, a result of such a type, and no `#[kernel]` attribute. The body follows the rules of a helper function, so its own errors have the codes of those rules.
Reached through `#[gpu]`.
Rejected: `#[gpu] fn f() -> f32`, `#[gpu] fn f(x: f32)` (no result), `x: f64`, `x: &f32`, `x: Option<f32>`, a tuple or array, a result of `()`, `#[gpu] #[kernel] fn ..`.
Write instead: a function from scalars and vectors to one value; write a `#[kernel]` inside `#[gpu] mod name { .. }` for anything else. A 3-wide type is GUST-E0204.
Related, not a GUST code: under a GPU profile (`dev`, `native`, ..) the generated `map`, `map_on`, `map_gpu`, `map_gpu_on`, `map_async` and `map_on_async` name `::gust_wgpu`, so the crate that writes `#[gpu]` on a function must depend on `gust-wgpu` directly. Without it rustc reports E0433 (failed to resolve: use of unresolved module or unlinked crate `gust_wgpu`) at the attribute. Add `gust-wgpu` to that crate's dependencies. Under `GUST_PROFILE=cpu` the functions name only `::gust::cpu`, nothing GPU is named and no `gust-wgpu` is needed. An argument is also typed by the function's own parameter: a `bool` parameter takes `bool`s and a `u32` parameter takes `u32`s, so passing one for the other is rustc's E0277 (the trait bound `MapArg<u32, ..>` or `MapArg<bool, ..>` is not satisfied).
Decision: D60.

### GUST-E0715 map function name collides with the generated module

The module `#[gpu]` generates for `fn name` is named `name` and defines `map`, `map_on`, `map_gpu`, `map_gpu_on`, `map_async`, `map_on_async`, `map_kernel`, `DESCRIPTOR`, `MODULE_DESCRIPTOR` and `OWN_INDEX`, and holds the function itself under its own name, so the function may not have one of those names and no parameter may have the function's name.
Reached through `#[gpu]`.
Rejected: `#[gpu] fn map(x: f32) -> f32`, `#[gpu] fn DESCRIPTOR(..)`, `#[gpu] fn f(f: f32) -> f32`.
Write instead: another name for the function or the parameter.
Decision: D60.

### GUST-E0716 `#[gpu]` function inside a `#[gpu]` module

`#[gpu]` on a function generates a module beside the function, and a `#[gpu]` module has no place for one: its functions are kernels and helpers.
Reached through `#[gpu]`.
Rejected: `#[gpu] mod m { #[gpu] pub fn f(x: f32) -> f32 { x } }`, also spelled `#[gust::gpu]`.
Write instead: put the function outside the module (a map function is plain Rust that the generated kernel calls), or drop the inner attribute and use it as a helper.
Decision: D60.

### GUST-E0717 barrier kernel without `cpu = "unsupported"`

A kernel that declares `Workgroup<[T; N]>` memory or calls `workgroup_barrier()` cannot run on the CPU twin: the twin runs the invocations one after the other, and a barrier needs all of them in flight together. GUST defines no CPU strategy for such a kernel (D118), so the kernel says so, with `cpu = "unsupported"`. It then has no `Run` impl for `CpuDevice` in any profile, so under `GUST_PROFILE=cpu` it behaves like every other unsupported kernel. `cpu = "lockstep"` is not a strategy either and is GUST-E0709. The message names the construct and the fix.
With variants (M8, D139) the rule is about the body the twin is built from, the portable body (the one whose set is empty). A body that requires a capability, the kernel's own included, has no twin and may use a barrier. A `requires()` variant is the portable body, so a barrier in it is this error, and the fix is `cpu = "unsupported"` on the kernel (`#[kernel(.., requires(..), cpu = "unsupported")]`), not on the variant: on a variant that option is GUST-E0740. The message names the variant, and it cannot tell the two cases apart.
Reached through `#[gpu]`.
Rejected: `#[kernel(workgroup_size(256, 1, 1))]` on a kernel with a `Workgroup` parameter or a `workgroup_barrier();` call.
Fixture: `tests/ui/barrier-kernel-no-cpu-strategy` (`cargo xtask check-ui`, `code: none` plus the message).
Write instead: `#[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]`.
Decision: D42 (M7 plan 4.2 rule 5, slice m7.g1), D118.

### GUST-E0718 workgroup memory over the limit

All `Workgroup` parameters of one kernel together may take 16384 bytes (16 KiB), the WebGPU default limit of `maxComputeWorkgroupStorageSize`. An element takes 4 bytes for a scalar or a `bool`, 8 for a 2-wide vector and 16 for a 3-wide or 4-wide one (the 3-wide types take 16 in workgroup memory). The macro adds the sizes up, so the check runs when the module expands, not at pipeline creation. A device with a smaller limit is a runtime question and not this code's.
Reached through `#[gpu]`.
Rejected: `Workgroup<[f32; 4097]>`, `Workgroup<[Vec4; 1025]>`, `Workgroup<[f32; 4096]>` with a second `Workgroup<[u32; 1]>`.
Fixture: `tests/ui/workgroup-over-16kib` (`cargo xtask check-ui`, `code: none` plus the message).
Write instead: shorter arrays, a smaller element type, or fewer arrays; 16384 bytes exactly is accepted.
A crate that sets `limits = "portable"` gets GUST-E0746 for the same kernel, with the same count and the same limit (M8, D143): the floor check runs first, so E0718 is reported only by a crate without the setting.
Decision: D42 (M7 plan 4.1, slice m7.g1).

### GUST-E0719 `par_for!` call that is not one of its forms

A `par_for!` call is `par_for!([workgroup_size(x, y, z);] PATTERN in RANGES, [bindings] { body })`. The pattern is `i`, `(x, y)` or `(x, y, z)`, each name an identifier or `_` and each named once; the ranges are `0..n` for one variable or a tuple of them, one per variable, each starting at the literal 0 and ending before an extent expression; the bindings are a bracketed list; the body is one block and ends the call. A `workgroup_size` option has the codes of the same option on a kernel (E0704, E0705, E0706). The hidden options `on(device);` and `descriptor;` come before the pattern and are read the same way.
Reached through `#[gpu]`, raised by `par_for!` only (the function-like macro that shares the validator of `#[gpu]`).
Rejected: `par_for!(i in 0..n, [])` (no body), `par_for!((x) in 0..n, [] { })`, `par_for!((x, y) in 0..n, [] { })`, `par_for!(i in 1..n, [] { })`, `par_for!(i in 0..=n, [] { })`, `par_for!((x, x) in (0..w, 0..h), [] { })`, `par_for!(i in 0..n, [] { } { })`.
Write instead: one of the forms of the plan (section 3.1), for example `par_for!((x, y) in (0..w, 0..h), [out: &mut [f32], w: u32] { out[y * w + x] = 1.0; })`.
Decision: M7 plan 3.1 (slice m7.p).

### GUST-E0720 `par_for!` binding that cannot be one

A binding is `name: Type` with one of: `&[T]` (a buffer the body reads), `&mut [T]` (a buffer it writes, read back after the loop), `&GpuVec<T>` and `&mut GpuVec<T>` (a resident buffer, bound where it is), or a scalar `u32`, `i32`, `f32` or `bool` (a field of the extent record). `T` is a buffer element: `f32`, `i32`, `u32` or a 2- or 4-wide vector or a square matrix; a `bool` buffer is a `u32` buffer, and a 3-wide type is GUST-E0204. A name is used once among the loop variables and the bindings, and is not one the expansion defines in the block of the call (`run`, `launch`, `launch_default`, `DESCRIPTOR`; the others start with the reserved `__gust` prefix): a binding reads the variable of its own name, so such a name would read the expansion's item. Under `GUST_PROFILE=cpu` there is no GPU API, so a `GpuVec` binding is refused.
Reached through `#[gpu]`, raised by `par_for!` only (the function-like macro that shares the validator of `#[gpu]`).
Rejected: `a: Vec<f32>`, `a: &Vec<f32>`, `a: &[bool]`, `a: &mut f32`, `a: u64`, `a: &[f32], a: &[f32]`, `i: u32` in a loop over `i`, `launch: &mut [f32]`, `a: &GpuVec<f32>` under `GUST_PROFILE=cpu`.
Fixtures: `tests/ui/par-for-binding`, `tests/ui/par-for-reserved-name` (`cargo xtask check-ui`).
Write instead: a slice or a `GpuVec` for a buffer, a scalar of the four types for a value, and distinct names that the expansion does not use (rename the variable of the call); the binding is read from the variable of that name at the call (`&a[..]`, `&mut out[..]`, `&*v`, or the scalar itself).
Decision: M7 plan 3.1 (slice m7.p).

### GUST-E0721 unknown kernel shape

The value of `#[kernel(shape = ..)]` is `grid` or `pixel`; any other name is unknown, and the names are case-sensitive.
Reached through `#[gpu]`.
Rejected: `#[kernel(shape = vulkan)]`, `#[kernel(shape = Grid)]`, `#[kernel(shape = compute)]`.
Fixture: `tests/ui/shape-unknown` (`cargo xtask check-ui`).
Write instead: `shape = grid` or `shape = pixel`, or no `shape` at all (the default is `grid`).
Decision: D63 (plan section 5.2, slice mx.c).

### GUST-E0722 malformed `shape` option

The `shape` option is given at most once, with a value that is a bare name, and `cpu` is a target (`GUST_PROFILE=cpu`, D51), never a shape.
Reached through `#[gpu]`.
Rejected: `#[kernel(shape = cpu)]` (the message says to build with `GUST_PROFILE=cpu`), `#[kernel(shape = grid, shape = pixel)]`, `#[kernel(shape = "pixel")]`, `#[kernel(shape = 1)]`, `#[kernel(shape)]`.
Fixtures: `tests/ui/shape-cpu-pin`, `tests/ui/shape-twice`, `tests/ui/shape-not-a-name` (`cargo xtask check-ui`).
Write instead: one `shape = grid` or `shape = pixel`; to run a kernel on the CPU, build with `GUST_PROFILE=cpu`.
Decision: D63 (plan section 5.2, slice mx.c).

### GUST-E0723 pinned shape excluded by the kernel's resources or attribute

A `shape` pin names a shape the kernel cannot take because of what it declares: workgroup memory, a workgroup barrier, an invocation id other than the global id (`LocalId`, `GroupId`), an atomic operation, a storage texture, an explicit `workgroup_size`, no output buffer, an output element `pixel` does not take (only `u32`, `i32`, `f32` and their 2- and 4-wide vectors are), or a construct the classifier has no rule for (D94). There is no bound on the number of outputs here: D120 made an output a fragment storage write, so the render-target bounds of D62 (8 outputs, 32 bytes) are gone, and the limit that binds is the device's number of storage buffers per shader stage, checked when the kernel runs. D120 also removed the original reason for refusing `bool`, a struct and a 3-wide vector as outputs (they had no colour-target format); the exclusion stays as mx.c fixed it, and widening it is a candidate for slice mx.p1c or later. The classifier is syntactic and conservative: what it does not recognise makes the kernel `grid`-only. The message names the construct that excludes the shape.
Reached through `#[gpu]`.
Rejected: `#[kernel(shape = pixel, workgroup_size(64, 1, 1))]`, `shape = pixel` on a kernel with a `Workgroup<[f32; 64]>` parameter or a struct output.
Fixtures (each pins its reason with a second `message:`): `tests/ui/shape-pixel-workgroup-size`, `tests/ui/shape-pixel-barrier`, `tests/ui/shape-pixel-atomic`, `tests/ui/shape-pixel-struct-output`, `tests/ui/shape-pixel-no-output`, `tests/ui/shape-pixel-storage-texture` (`cargo xtask check-ui`).
Write instead: drop the pin (the kernel stays `grid`), or remove the construct the message names.
Decision: D62, D63 (plan section 5.1 and 5.3, slice mx.c).

### GUST-E0724 pinned shape excluded by a mutable invocation

A `pixel` pin on a kernel whose invocation parameter is `mut`, or that assigns to the invocation or a field of it: the invocation id of a pixel is fixed.
Reached through `#[gpu]`.
Rejected: `#[kernel(shape = pixel)] fn k(mut id: SV_DispatchThreadID, ..)`, `id.x = 3;`.
Fixture: `tests/ui/shape-pixel-mutable-invocation` (`cargo xtask check-ui`).
Write instead: declare the invocation without `mut` and compute any other index into a local.
Decision: D63 (plan section 5.1, slice mx.c).

### GUST-E0725 pinned shape excluded by the own index

A `pixel` pin on a kernel that writes an output at an index other than the element's own (`out[i] = v` with `i` the own index), or whose own-index alias is not one immutable `let i = id.x;` (or `id.global_id().x`) that is never bound again. The own index is `id.x`, `id.global_id().x`, or that one alias.
Reached through `#[gpu]`.
Rejected: `out[id.x + 1u32] = v`, `out[0u32] = v`, `let mut i = id.x;`, a second alias, `let i = id.x; for i in 0..3 { .. }`.
Fixtures: `tests/ui/shape-pixel-other-index`, `tests/ui/shape-pixel-second-alias` (`cargo xtask check-ui`).
Write instead: write each output at the element's own index only, and bind the index once.
Decision: D63 (plan section 5.1, slice mx.c).

### GUST-E0726 pinned shape excluded by a conditional write

A `pixel` pin on a kernel that does not write every output on every path. The rule is ordinary definite assignment: a block writes if some statement does, `if` with `else` if both arms do, `match` if every arm does, a loop never, and an `if` without `else` only when its condition is a conjunction of `own < out.len()` comparisons (which a live pixel always satisfies).
Reached through `#[gpu]`.
Rejected: `if input[id.x] > 0.0 { out[id.x] = 1.0; }`, a write in a `for` loop, a guard on a length that is not an output's.
Fixture: `tests/ui/shape-pixel-conditional-write` (`cargo xtask check-ui`).
Write instead: write the output in both arms, or in front of the branch, or guard with `if i < out.len()` only.
Decision: D63 (plan section 5.1, slice mx.c).

### GUST-E0727 pinned shape excluded by a return

A `pixel` pin on a kernel with a `return` in its body: a pixel kernel is written to its end.
Reached through `#[gpu]`.
Rejected: `if id.x >= out.len() { return; }`.
Fixture: `tests/ui/shape-pixel-return` (`cargo xtask check-ui`).
Write instead: wrap the writes in `if id.x < out.len() { .. }`.
Decision: D63 (plan section 5.1, slice mx.c).

### GUST-E0728 pinned shape excluded by a use of an output

A `pixel` pin on a kernel that uses an output other than as `out[own] = v` and `out.len()`: it reads an element, assigns to it with a compound operator, writes one component of it, rebinds the buffer, passes it on, or never writes a `StorageMut` buffer (an input read through a mutable binding, which should be a plain `Storage` buffer).
Reached through `#[gpu]`.
Rejected: `out[id.x] = out[id.x] + 1.0;`, `out[id.x] += v;`, `out[id.x].x = 1.0;`, an unwritten `mut` buffer.
Fixtures: `tests/ui/shape-pixel-output-read`, `tests/ui/shape-pixel-output-compound` (`cargo xtask check-ui`).
Write instead: compute the whole element into a local from the inputs and write it once; declare a buffer the kernel only reads without `mut`.
Decision: D63 (plan section 5.1, slice mx.c).

### GUST-E0729 construct the kernel's shape does not take

The execution shape of a kernel (its stage profile) refuses a construct the kernel declares: workgroup memory, a barrier or an atomic operation, as the profile decides. The message after the code is the profile's own sentence. The `grid` profile refuses nothing. The `pixel` profile (slice mx.p1a) refuses workgroup memory, barriers and atomics, but a `shape = pixel` pin on such a kernel is already refused by the classifier (GUST-E0723) before the profile runs, so the code is raised only by a profile test double.
Unreachable through `#[gpu]`: the `grid` profile refuses no construct, and the classifier refuses the `pixel` ones first.
Rejected: a barrier in a kernel of a profile whose `validate_body` forbids barriers.
Write instead: remove the construct the message names, or give the kernel a shape that takes it.
Decision: D63 (slice mx.p0, mx.c).

### GUST-E0731 texture or sampler type outside the dialect

A texture parameter with a texel type or a format the dialect does not have: `Texture2D<T>` takes `f32`, `u32`, `i32` and their 2- and 4-wide vectors (a 3-wide texel has no texture format); `StorageTexture2D<F>` takes `R32Float`, `R32Uint` or `R32Sint`, the formats a kernel may read and write; `WriteTexture2D<F>` takes the nine formats `R32`, `Rg32` and `Rgba32` in `Float`, `Uint` and `Sint`; `Sampler` takes no type arguments. The message lists the accepted arguments.
Reached through `#[gpu]`.
Rejected: `Texture2D<Vec3>`, `Texture2D<bool>`, `Texture2D` without an argument, `StorageTexture2D<Rgba32Float>`, `WriteTexture2D<Rgba8Unorm>`, `WriteTexture2D<f32>`, `Sampler<f32>`.
Fixture: `tests/ui/texture-bad-format` (a 3-wide texel; the other spellings are also Rust type errors and have unit tests only); the accepted forms are `tests/ui/texture-ok` and `tests/fixtures/textures.rs`.
Write instead: a texel type or format from the lists above; `WriteTexture2D` for a format a kernel may only write.
Decision: D62, D64 (slice mx.t3).

### GUST-E0732 texture method with a wrong coordinate

A texture method call with the wrong number of arguments, or with a coordinate that is visibly not the right type. A texel address (`load`, `store`) is two `u32`; a sample coordinate (`sample`) is two `f32`. The check is syntactic: a float literal or a float cast as an address, an integer literal or an integer cast as a sample coordinate, a negative address, a suffix of another type. Typing the rest is the Slang compiler's job.
Reached through `#[gpu]`.
Rejected: `image.load(0u32)`, `image.load(1.5f32, 0u32)`, `image.load(-1, 0u32)`, `image.store(0u32, 0u32)`, `image.sample(s, 1u32, 0.5f32)`, `image.width(1u32)`.
No `check-ui` fixture: rustc reports its own type error beside the macro's, and `check-ui` pins exactly one error. The unit tests of `crates/gust-macros/src/tests/regression.rs` assert it.
Write instead: `load(x, y)` and `store(x, y, v)` with `u32` addresses, `sample(sampler, u, v)` with `f32` coordinates.
Decision: D62 (slice mx.t3).

### GUST-E0733 sample without a sampler

The first argument of `sample` is not a `Sampler` parameter of the kernel. A sampler is passed by its parameter name: it cannot be built, stored in a local or passed from a helper.
Reached through `#[gpu]`.
Rejected: `image.sample(id.x, 0.5f32, 0.5f32)`, `image.sample(0.5f32, 0.5f32, 0.5f32)`, `image.sample(image, ..)`.
Fixture: `tests/ui/texture-sample-argument` (`cargo xtask check-ui`).
Write instead: declare a `filter: Sampler` parameter and call `image.sample(filter, u, v)`.
Decision: D64 (slice mx.t3).

### GUST-E0734 sample of integer texels

A `sample` on a texture whose texels are integers (`u32`, `i32` and their vectors): integer textures are never filtered.
Reached through `#[gpu]`.
Rejected: `ids.sample(filter, 0.5f32, 0.5f32)` on a `Texture2D<u32>`, `Texture2D<UVec4>` or `Texture2D<IVec2>`.
No `check-ui` fixture: rustc reports its own type error beside the macro's, and `check-ui` pins exactly one error. The unit tests of `crates/gust-macros/src/tests/regression.rs` assert it.
Write instead: read the texel with `ids.load(x, y)`, or sample a texture of `f32`, `Vec2` or `Vec4` texels.
Decision: D62 (slice mx.t3).

### GUST-E0735 method a texture does not have

A method call on a texture parameter that the texture's access does not have, or a name that is no texture method. A sampled `Texture2D` has `load`, `sample`, `width` and `height`; a `StorageTexture2D` has `load`, `store`, `width` and `height`; a `WriteTexture2D` has `store`, `width` and `height`. The message names the missing method and the texture that has it.
Reached through `#[gpu]`.
Rejected: `image.store(..)` on a `Texture2D`, `out.load(..)` on a `WriteTexture2D`, `image.sample(..)` on a `StorageTexture2D`, `image.gather(..)`, `image.depth()`.
Fixture: `tests/ui/texture-method` (`image.clone()`, which is valid Rust, so only the macro refuses it; the other spellings are also rustc errors and have unit tests only).
Write instead: use the access that has the method: `StorageTexture2D` to read and write, `WriteTexture2D` to write, `Texture2D` to read and sample.
Decision: D62, D64 (slice mx.t3).

### GUST-E0736 texture or sampler used as a value

A texture or sampler parameter used other than as the receiver of a texture method (or the sampler argument of `sample`), or a texture or sampler type anywhere but a kernel parameter: copied to a local, passed to a helper, called with a method of its own, or written as the type of a `let` or a struct field.
Reached through `#[gpu]`.
Rejected: `let alias = image;`, `let size = image.width() + image;`, `filter.width()`, `let alias: Texture2D<Vec4> = ..;`.
Fixture: `tests/ui/texture-as-value` (`cargo xtask check-ui`).
Write instead: call the method on the parameter by name; a helper function takes the texel values (`image.load(x, y)`) or the coordinates, not the texture.
Decision: D62, D64 (slice mx.t3).

### GUST-E0740 malformed `#[variant]`

A `#[variant]` attribute the macro cannot read, or one in a place it does not belong (M8, D139): no `of = <kernel>`, an option other than `of` and `requires(..)`, an option given twice (a second `requires(..)` on a `#[kernel]` too), a value that is not a name, a `#[variant]` that also carries `#[kernel]`, two `#[variant]` attributes on one function, a variant whose signature is not its kernel's token for token (the generics, the parameter names, patterns and types, and the return type), or `#[variant]` on anything but a function of the module (a struct, a constant, a `#[gpu]` function, a macro, a function inside an `impl`, a `trait` or an `extern` block). A variant has no options of its own: the workgroup size and the shape are its kernel's, and it is lowered with its own `requires(..)`, so the kernel's `requires` is not its own. A variant with a non-empty set never has a CPU twin (it is lowered with `cpu = "unsupported"`); a `requires()` variant is the portable body, the body the kernel's twin is built from, and it is lowered with the kernel's `cpu` option, so it is judged as the twin's body (GUST-E0717).
Reached through `#[gpu]` on a module, and on a function (the attribute is refused there and the function goes back without it).
Rejected: `#[variant]`, `#[variant(requires(subgroups))]`, `#[variant(of = sum, tier = 3)]`, `#[variant(of = "sum")]`, `#[variant(of = sum, workgroup_size(8, 1, 1))]`, a variant that renames a parameter or changes a buffer's element type, `#[variant(of = sum)]` on a `struct`.
Fixture: `tests/ui/variant-malformed` (`cargo xtask check-ui`); the other forms are pinned by `crates/gust-macros/src/tests/variant.rs`. `tests/ui/variant-ok` compiles a pair, which also makes rustc prove the two signatures equal; `tests/ui/variant-ok-subgroups` builds under `GUST_PROFILE=dev+subgroups`, so rustc also compiles the selected variant and a kernel's own body that requires a capability.
Write instead: `#[variant(of = kernel_name, requires(subgroups))]` on a function with the kernel's signature and no other attribute; give the workgroup size, the shape and the CPU form to the `#[kernel]`.
Decision: D139.

### GUST-E0741 unknown capability in `requires(..)`

A name in `requires(..)`, on a `#[variant]` or a `#[kernel]`, that is not a capability or an alias. The capabilities of this release are `subgroups`; the aliases are `vulkan` (the set `subgroups`) and `webgpu` (the empty set). `f16` and `atomics64` are reserved names that this release does not support, so they are unknown here too, as in `GUST_PROFILE`.
Reached through `#[gpu]`.
Rejected: `requires(subgroup)`, `requires(Subgroups)`, `requires(f16)`, `requires(subgroups, simd)`.
Fixture: `tests/ui/variant-unknown-capability` (`cargo xtask check-ui`).
Write instead: `requires(subgroups)`, or `requires(vulkan)`; an empty `requires()` (or `webgpu`) is the portable body.
Decision: D139.

### GUST-E0742 variant of nothing

`#[variant(of = name)]` where `name` is no `#[kernel]` function of the same module: a function that does not exist, a helper that has no `#[kernel]`, or another variant (a variant belongs to a kernel, and the kernel is the one that names the handle and the descriptor).
Reached through `#[gpu]`.
Rejected: `#[variant(of = prod)]` with no `prod` kernel; `#[variant(of = helper)]`; `#[variant(of = first)]` where `first` is itself a variant.
Fixture: `tests/ui/variant-no-kernel` (`cargo xtask check-ui`).
Write instead: name the kernel; the kernel and its variants are in one `#[gpu]` module.
Decision: D139.

### GUST-E0743 two bodies a device cannot tell apart

Two bodies of one kernel (the kernel's own and its variants) for which some device has both sets: neither requires more than the other, and the device has both. The selection takes the body with the largest set the profile contains, so equal sets (`vulkan` and `subgroups` are the same set) and sets neither of which contains the other are a tie. The tie is checked over every subset of the capabilities, so it is an error in every build, the CPU build included, and not only in a build whose profile has the sets.
Reached through `#[gpu]`.
Rejected: two variants with `requires(subgroups)` (or one with `requires(vulkan)`); a variant with `requires()` beside a kernel whose own set is empty (it would be the same body twice).
Fixture: `tests/ui/variant-ambiguous` (`cargo xtask check-ui`).
Write instead: give the bodies sets of different sizes, or one a superset of the other; a body with a smaller set is the fallback of the one with the larger set.
Decision: D139.

### GUST-E0744 declared profile not covered

A profile the crate declares in `[package.metadata.gust] profiles` that a kernel has no body for (M8, D139, D23). The list is the exhaustiveness rule: for every declared GPU profile (`native`, `web`, each with its capabilities; a `dev` build is checked as `native`), every kernel needs a body whose `requires` the profile contains. Without a list the default is `["native", "web", "cpu"]`. The `cpu` entry is met by the kernel's CPU form, which is the portable body's (the body whose set is empty: the kernel's own, or its `requires()` variant): a kernel with `cpu = "unsupported"` has none, and so has a kernel every body of which requires a capability, and that is an error only when the crate wrote the list (an explicit list, even one equal to the default, is stricter than the absent key), because under the default list the kernel's own `cpu = "unsupported"` is the arm that meets the entry (and a kernel with no portable body is already refused at `native`). The message says when the profile is the default list (the manifest has no `profiles` key). The check runs in every build, the CPU build included.
The same code covers a route tag's set (m8.r1, D65): a `#[gpu(route = "tag")]` module whose tag lists `cpu` in `targets` (`[package.metadata.gust.routes]`) while one of its kernels has no CPU form (`cpu = "unsupported"`, or every body requires a capability) is refused at that kernel, in every build and whatever the tag is routed to, because the declaration is false. A tag with no `targets` key has no such declaration: its set is `gpu` and `cpu` for a kernel with a CPU form and `gpu` alone for one without (GUST-E0749 judges the route against it).
Reached through `#[gpu]`.
Rejected: a kernel with `requires(subgroups)` and no variant under the default list (`native` has no capability); the same under `profiles = ["native+subgroups", "web"]`; `cpu = "unsupported"` under `profiles = ["native", "cpu"]`; a kernel with `requires(subgroups)` and no `requires()` variant under `profiles = ["native+subgroups", "cpu"]`; `cpu = "unsupported"` in a module routed by a tag declared `targets = ["gpu", "cpu"]`.
Fixtures: `tests/ui/variant-not-covered` and `tests/ui/variant-cpu-unsupported` (`cargo xtask check-ui`); the tag case is `tests/ui/typed-call-routed-no-cpu-form`.
Write instead: add a variant that requires less (`requires()` covers every profile), declare profiles with the capability, or drop `cpu` from `profiles` (or the `cpu = "unsupported"` from the kernel); for a tag, give the kernel a CPU form or drop `cpu` from the tag's `targets`.
Decision: D139, D138, D65.

### GUST-E0745 subgroup operation outside a body that requires subgroups

A subgroup operation (`subgroup_add`, `subgroup_ballot`, `subgroup_size` and the rest of the 21 in `docs/SEMANTICS.md`) is called where it has no meaning (M8, D140): in a portable body (a `#[kernel]` without `requires(subgroups)`, which also runs on the CPU target and on a device with no subgroup operations), or in a helper function. A helper is called under conditions the helper cannot see, so whether every invocation of the subgroup reaches the operation is not decidable inside it; the operation belongs in the kernel body. A function of the module may not take the name of an operation either, because a call of that name is the operation. The message names the operation and says which of the three it is.
Reached through `#[gpu]`.
Rejected: `subgroup_add(x)` in the body of a `#[kernel]` with no `requires`; `subgroup_add(v)` in a `fn` helper; `fn subgroup_add(v: u32) -> u32` in a GPU module.
Fixtures: `tests/ui/subgroup-portable-body` and `tests/ui/subgroup-in-helper` (`cargo xtask check-ui`); the unit tests are in `crates/gust-macros/src/tests/subgroup.rs`.
Write instead: put the operation in a `#[variant(of = k, requires(subgroups))]` body beside the portable one, and call it from the kernel body itself.
Decision: D140, D139.

### GUST-E0746 kernel over the portable floor

A kernel of a crate that sets `limits = "portable"` in `[package.metadata.gust]` is over one of the limits of `gust::limits::PORTABLE`, which are WebGPU's defaults (M8, D143, D36). The check is opt-in: a crate without the setting is never checked against the table here, and gets the launch checks of `gust-compute-wgpu` (D36) only. The macro counts what the source of each kernel decides, in this order, and reports the first that is over: the `workgroup_size` per axis (`max_compute_workgroup_size_x`, `_y` at 256 and `_z` at 64; a kernel with no option has the grid default of 64 x 1 x 1) and their product (`max_compute_invocations_per_workgroup`, 256); the storage buffers (`max_storage_buffers_per_shader_stage`, 8, read-only and read-write parameters together), uniform buffers (`max_uniform_buffers_per_shader_stage`, 12), sampled textures (`max_sampled_textures_per_shader_stage`, 16), storage textures (`max_storage_textures_per_shader_stage`, 4, read-write and write-only together) and samplers (`max_samplers_per_shader_stage`, 16); and the workgroup memory (`max_compute_workgroup_storage_size`, 16384 bytes, counted as E0718 counts it). A `shape = pixel` kernel has no workgroup, so only the resource counts apply to it.
Message: `kernel `NAME` is over the portable floor: its <what> is <value>, over `<limit>` = <max>`, with the limit named as `wgpu` names it and both numbers (the two workgroup memory numbers carry "bytes"), then a sentence on the setting. It is spanned over the kernel's signature. Every kernel of the module is checked, a `#[variant]` body included.
E0718 is folded into this code for such a crate: the check runs before the validator, and the floor's workgroup memory limit is E0718's 16 KiB, so a portable crate gets E0746 where an ordinary crate gets E0718, with the same count. For a crate without the setting E0718 is unchanged.
What it cannot count: a `Workgroup` array whose length is not an integer literal, or whose element is a type the macro cannot size (a user struct), counts as 0 bytes (as in E0718); a kernel attribute that does not parse is reported by its own code (E0703 to E0709) and not here; the limits that depend on a launch (binding and buffer sizes, workgroups per dimension, texture dimensions, bindings per group) are the runtime's (D36, D84). Only `#[kernel]` functions of a `#[gpu]` module are checked: the `#[gpu]` function (an element-wise map) and `par_for!` go through other entry points and are not. A malformed `[package.metadata.gust]` table is E0753, reported as before; until it is fixed the floor is not checked, so a kernel over the workgroup memory limit in such a crate still gets E0718 from the validator, before E0753.
A kernel routed to the CPU (`#[gpu(route = "tag")]`, GUST-E0748) is held to this floor too. The check runs in `#[gpu]` before the route is resolved and does not consult it: a route is a per-build choice that a variable flips, so the source must stay valid for the GPU in every build (m8.r1).
The floor does not settle D84's two meanings of "default": `LimitsPolicy::Adapter` raises only `max_buffer_size`, `max_storage_buffer_binding_size` and `max_compute_workgroups_per_dimension`, none of which is in the table, so every GUST device opens with exactly these values for the limits the table holds.
Reached through `#[gpu]` on a module.
Rejected (each with `limits = "portable"`): `workgroup_size(257, 1, 1)`, `workgroup_size(1, 257, 1)`, `workgroup_size(1, 1, 65)`, `workgroup_size(17, 16, 1)` (272 invocations), 9 storage buffers, 13 uniform buffers, 17 sampled textures, 5 storage textures, 17 samplers, `Workgroup<[f32; 4097]>` (16388 bytes).
Fixtures: `tests/ui/limits-workgroup-size-x`, `limits-workgroup-size-y`, `limits-workgroup-size-z`, `limits-invocations`, `limits-storage-buffers`, `limits-uniform-buffers`, `limits-sampled-textures`, `limits-storage-textures`, `limits-samplers` and `limits-workgroup-memory` fail; `limits-at-floor` (every limit reached exactly) and `limits-not-portable` (an over-the-floor kernel in a crate without the setting) compile (`cargo xtask check-ui`). The unit tests are in `crates/gust-macros/src/validate/limits.rs`.
Write instead: a smaller workgroup, fewer bindings, shorter `Workgroup` arrays, or remove `limits = "portable"` if the crate targets only devices with higher limits.
Decision: D143.

### GUST-E0747 subgroup operation in a position where not every invocation reaches it

A subgroup operation combines the invocations of a subgroup, so it is a rendezvous like `workgroup_barrier()` (M8, D140): the macro is the only uniformity authority (the GPU compiler does not check it), and it approves a call only as the whole value of a `let`, an assignment or a compound assignment, or as an operand of arithmetic, a comparison, a cast or a unary operator there (and as the argument of another operation), at the top level of the kernel body or of a top-level `for a..b` loop with two literal bounds. Refused: under an `if` (a condition or either arm), under `&&` or `||` (short-circuit), in an index, as a method receiver, as a statement of its own, in a loop whose bounds are not literals, in a loop inside a loop, and as the value of an `if`. A call under a condition that differs per invocation would combine only the active invocations, and what that gives is a documented row, not a guarantee (D39, `docs/SEMANTICS.md`).
Reached through `#[gpu]`.
Rejected: `if p { let r = subgroup_add(x); }`, `let r = p && subgroup_all(p);`, `data[subgroup_add(x)]`, `for s in 0u32..x { .. subgroup_add(s) .. }`.
A `match` is not an approved position either: an operation in an arm of a statement or value `match` is refused with this code, and so is one in its scrutinee (`let x: u32 = match subgroup_add(v) { .. };`), although every invocation evaluates the scrutinee. That over-refusal is deliberate (M9, slice m9.e2): bind the result to a local first, `let s = subgroup_add(v);` then `match s { .. }`.
Fixtures: `tests/ui/subgroup-in-if` and `tests/ui/subgroup-under-and` (`cargo xtask check-ui`).
Write instead: compute the operation unconditionally at the top level of the body, and use its result under the condition.
Decision: D140.

### GUST-E0748 route tag ill-named or not declared

The `route = "<tag>"` of `#[gpu]` names a tag that cannot be used (M8, D65 to D67, m8.r1). A tag is declared in the `[package.metadata.gust.routes]` table of the `Cargo.toml` of the crate that uses it, and in M8 **only that crate uses it**: the macro cannot read a dependency's manifest, so a tag declared by another crate is "not declared" here (the app routes a dependency's tag through `gust.toml`, m8.r2). The code covers: a tag that is not well-formed (segments of `[a-z0-9_]+` joined by single `.`, with no `__` anywhere, because `__` spells the `.` in the route variable `GUST_ROUTE__<package>__<tag>`, and no segment that begins or ends with `_`, because `a_.b` and `a._b` would name one variable; no empty segment, so no leading, trailing or doubled `.`); a key of the routes table that is not a well-formed tag (checked when a module is routed, as the tag itself is); and a tag that the crate's table does not declare, with the declared ones named. The error is spanned over the string literal.
Reached through `#[gpu]` on a module and on a function.
Rejected: `#[gpu(route = "Physics")]`, `#[gpu(route = "a__b")]`, `#[gpu(route = "a_.b")]`, `#[gpu(route = "a..b")]`, `#[gpu(route = "")]`; `#[gpu(route = "render")]` in a crate whose table declares `physics`, or has no table.
Fixtures: `tests/ui/typed-call-routed-undeclared` and `tests/ui/typed-call-routed-ill-named` (`cargo xtask check-ui`); the unit tests are in `crates/gust-macros/src/tests/routes.rs`, which also pins the macro's copy of `gust::routes::variable_name` and `is_valid_tag` against the golden table of `crates/gust/src/routes.rs`.
Write instead: declare the tag in `[package.metadata.gust.routes]` of the crate that uses it (`physics = {}`, a dotted tag as a quoted key), and name it with lower-case segments joined by `.`.
Decision: D65, D66, D67.

### GUST-E0749 route target not allowed

A `#[gpu(route = "tag")]` module or function resolves to a target that the build cannot honour (M8, D65 to D67, m8.r1). The target is resolved once per module, nearest first: `GUST_PROFILE=cpu`; the variable `GUST_ROUTE__<package>__<tag>` (`-` in the package name is `_`, each `.` of the tag is `__`; an empty variable is unset); the tag's `default` in `Cargo.toml`, then each dotted ancestor's (`physics.broadphase`, then `physics`); then the GPU. A source `default = "gpu"` yields to `GUST_PROFILE=cpu` (a crate with the usual default still builds for the CPU); a variable does not. The error is reported for: a variable whose value is not `gpu` or `cpu`; a variable that says `gpu` under `GUST_PROFILE=cpu`, because the CPU build holds no GPU form; and a resolved target that is outside the tag's set, judged per kernel and spanned at the kernel. The set is the tag's `targets` as written, else `gpu` and `cpu` for a kernel with a CPU form and `gpu` alone for a kernel without (`cpu = "unsupported"`, or every body requires a capability). Under `GUST_PROFILE=cpu` the set is not consulted (the profile decides, as it does for an untagged GPU-only kernel, which has no impl in that build); a `targets` that lists `cpu` for a kernel with no CPU form is GUST-E0744.
A CPU-routed module is a CPU body in any build: `impl Run for CpuDevice` ungated, no artifact looked up or required, `Target::Cpu` in its descriptor and no `gust::target::GpuKernel` impl, so a `HeadlessDevice` call of its kernel fails at the call with `HeadlessDevice cannot run the kernel ..: it is routed to the CPU in this build`. A GPU-routed module, and an untagged one in a build other than `GUST_PROFILE=cpu`, keeps the GPU form, and `CpuDevice` calls need the dual build or the twin as before (D26); under `GUST_PROFILE=cpu` every module, tagged or not, is a CPU body with `Target::Cpu` and no `GpuKernel` impl. Flipping a route therefore changes which device the kernel's call sites compile against.
Reached through `#[gpu]` on a module and on a function.
Rejected: `GUST_ROUTE__demo__physics=tpu`; `GUST_ROUTE__demo__physics=gpu` with `GUST_PROFILE=cpu`; `physics = { targets = ["gpu"] }` with the tag routed to `cpu`; `physics = { targets = ["cpu"] }` with nothing routing it (the GPU); a `cpu = "unsupported"` kernel in a module whose tag has no `targets` and is routed to `cpu`.
Fixtures: `tests/ui/typed-call-routed-outside-set` and `tests/ui/typed-call-routed-cpu-profile-gpu` (`cargo xtask check-ui`); `typed-call-routed-cpu`, `typed-call-routed-default`, `typed-call-routed-gpu` and `typed-call-routed-headless` show the accepted routes and the call-site consequence; check-dual E4 and check-cpu-target E-j run a routed module.
Write instead: route the tag to a target in its set (unset the variable, or set it to `gpu` or `cpu`), or change the tag's `targets`.
Decision: D65, D66, D67, D137.

Related, not a GUST code: a routed build whose profile `cargo gust` and cargo disagree on stops at the route tag of the kernel crate (m8.z). The driver resolves the routes of a `gust.toml` for the profile its command line and environment name (`--release`, `--target wasm32-..`, `GUST_PROFILE`) and sets `GUST_ROUTED_PROFILE` to it; the `#[gpu]` expansion of a module with a route tag compares it with the profile `gust` was built for (`gust` infers it from cargo's `PROFILE` and the target architecture) and the crate fails to compile, as a const evaluation error (E0080, "evaluation panicked") at the tag's literal, with a message that names both profiles (it begins "gust: `cargo gust` resolved the routes of this build for the profile X (GUST_ROUTED_PROFILE), but this build of gust is Y"), when X and Y differ. The causes are a target from `build.target` in `.cargo/config.toml` or from `CARGO_BUILD_TARGET`, a custom profile that inherits `release`, a `GUST_PROFILE` set only in the `[env]` table of the config, and a routed kernel crate built for the host as a build dependency of a `wasm32` release build (it infers `native`; set `GUST_PROFILE=web` in the environment so that both copies infer `web`). Write instead: give `cargo gust` the `--release` and `--target` of the build on its own command line, and set `GUST_PROFILE` in the process environment. Plain cargo sets no `GUST_ROUTED_PROFILE` and is not checked (D67); `unset GUST_ROUTED_PROFILE` if it is set in a shell that runs plain cargo (the driver itself removes a stale value when no `gust.toml` applies). The check lives in the routed crate and not in the build script of `gust`, so a switch between plain cargo and `cargo gust` rebuilds the routed crates and the application and not the `gust` cone. A `GUST_ROUTED_PROFILE` that is none of `dev`, `native`, `web` and `cpu` (a typo, text with a brace, an empty value; set by hand, since the driver never writes one) is a compile error of its own at the same literal and begins "gust: GUST_ROUTED_PROFILE is "X", which is not a build profile; the valid profiles are dev, native, web, cpu." (m9.0).

### GUST-E0750 no body for the build profile

The build's own profile (`GUST_PROFILE`, or no capability when it is unset or unknown) has no body to use for a kernel: the kernel's own `requires(..)`, and every variant's, contain a capability the profile lacks. The CPU target never reports it, because it builds the portable body (the kernel's own when no body is portable). Neither does `cargo gust build`: it collects under a fixed `GUST_PROFILE=dev` and writes the record of every body, so a kernel with no body for `dev` is exported through its own body (that build is never shipped). A crate that declares the profile it builds for gets E0744 first; this one is the build that is outside the declared list. Plain `cargo check`, rust-analyzer and docs.rs set no `GUST_PROFILE`, so they see no capability: a crate whose kernel has no body without a capability gets this error there until the profile is set for them.
Reached through `#[gpu]`.
Rejected: a kernel with `requires(subgroups)` and no variant, in a crate that declares `profiles = ["native+subgroups"]` only, built under `GUST_PROFILE=dev` or `native`.
Fixture: `tests/ui/variant-no-body` (`cargo xtask check-ui`).
Write instead: build with a profile that provides the capabilities (`GUST_PROFILE=native+subgroups`, `dev+subgroups`), set `GUST_PROFILE` in the `[env]` table of the crate's `.cargo/config.toml` so that every tool that builds the crate sees it (`[env]` `GUST_PROFILE = "dev+subgroups"`), or add a variant that requires less.
Decision: D139.

### GUST-E0751 subgroup operations in a workgroup of partial subgroups

A kernel that calls a subgroup operation has a `workgroup_size` whose product (x times y times z) is not a multiple of 64 (M8, D140). The lane count of a subgroup is a property of the pipeline (32 on the RTX 5090, 64 on the Radeon 890M, 8 on lavapipe, 4 on WARP in the certified runs), so no workgroup size is correct for every device, but a multiple of 64 is a whole number of subgroups on each of them: an invocation count that is not leaves a part-filled subgroup whose operations combine fewer invocations than the kernel counts on. The rule applies to a body that calls an operation; a `requires(subgroups)` body that calls none, and a portable body, may have any size. The message names the kernel and its size, and is spanned over the signature.
Reached through `#[gpu]`.
Rejected: `workgroup_size(32, 1, 1)`, `(63, 1, 1)`, `(100, 1, 1)` and `(8, 4, 1)` with a subgroup operation in the body; `(64, 1, 1)`, `(128, 1, 1)`, `(8, 8, 1)` and `(16, 2, 2)` are accepted.
Fixtures: `tests/ui/subgroup-workgroup-size` and `tests/ui/subgroup-workgroup-size-overflow` (`cargo xtask check-ui`); the second is `workgroup_size(4294967295, 4294967295, 4)`, whose product is over what a `u64` holds, and the message gives it exactly (73786976260478468100; the product is taken in `u128`, m9.0).
Write instead: a `workgroup_size` whose product is a multiple of 64.
Decision: D140.

### GUST-E0753 malformed `[package.metadata.gust]` table

The `Cargo.toml` of a crate that uses `#[gpu]` or `par_for!` has a `[package.metadata.gust]` table the macro cannot read (M8, D138): an unknown key, a value of the wrong kind, a `profiles` entry that is not a build profile (`<target>[+<capability>]`; `web+subgroups` and a capability on `cpu` are not), an empty `profiles` list, an entry listed twice, a `dev` entry (a dev build is checked as `native` with the same capabilities, so the list says `native...`), `limits` other than `"portable"`, a route that is not a table, names a target other than `gpu` and `cpu`, or has an empty `targets` list (reported at the key), a dotted tag written as a header table instead of a quoted key, or a manifest that is not valid TOML. The accepted forms are inline tables, dotted keys and headers (`[package.metadata.gust.routes.physics]`, the form `cargo package` writes), with basic or literal strings; the `gust` tables are the only part of the manifest that is read, and only `[package.metadata.gust]` (not `[workspace.metadata]`).
Reached through `#[gpu]` on a module and on a function (an element-wise map), and through `par_for!`: all three read the manifest of the crate.
Message: `Cargo.toml:LINE:COLUMN: <what is wrong>`, with the 1-based line and the column in characters of the offending key or value. The path is always the bare file name.
What the error can point at: a proc macro has no span inside `Cargo.toml`, so rustc underlines the identifier of the `#[gpu]` module, the name of the `#[gpu]` function, or the `par_for!` call, not the TOML line; the line and column are in the message only. It cannot name one kernel, because the table belongs to the crate: every `#[gpu]` module, every `#[gpu]` function and every `par_for!` call of the crate reports the same error once. A crate with none of the three never has its manifest read, so the error never appears for it. The first problem found stops the read; which one that is, when a table has several, is not specified.
Rejected: `profile = ["native"]` (the key is `profiles`), `profiles = ["web+subgroups"]`, `profiles = []`, `profiles = ["native", "native"]`, `profiles = ["dev"]`, `limits = "device"`, `physics = "gpu"` under `routes`, `targets = ["gpu", "tpu"]`, `physics = { targets = [] }`.
Fixtures: `tests/ui/profile-manifest-malformed` and `tests/ui/profile-manifest-empty-targets` (`cargo xtask check-ui`); the exact line, column and message of each case are pinned by `crates/gust-macros/src/tests/manifest.rs`, with the corpus of source forms and the manifest `cargo package` wrote.
Write instead: the keys are `profiles` (an array of profile strings), `limits = "portable"` and `routes` (a table of tags, each `{ targets = ["gpu", "cpu"], default = "gpu" }`); a dotted tag is a quoted key, `"physics.broadphase" = {}`; omit `targets` for the default instead of writing an empty list.
Decision: D138.

### Numbers in the M8 range that are not assigned, and the M8 errors that are not macro codes

E0737, E0738, E0739 and E0752 are not assigned and have no heading: the range was set aside for M8's slices, and they
ended without needing them (E0752 was the planned refusal of a CPU route on a web profile; spike S0 ended in the
answer that the route is allowed, D151). A number is never reused for something else, so they stay free for a later
check of the same family, and a code appears on this page only when the macro reports it.

The driver (`cargo gust build`, `check`, `run`, `test` with a `gust.toml`, D151) reports its mistakes as plain errors
with the file, the line and the tag (an unknown tag, a target outside the tag's `targets`, a route that names a
capability profile, two packages that give one variable name, `--workspace` over a member with a `gust.toml`); they
have no code, because the driver is a tool and not the macro, and they happen before cargo builds.

Three run-time errors of `gust-compute-wgpu` belong to M8 and are values of `Error`, not macro codes:
`MissingCapability { capability, adapter, kernel }` (at open, when the device options request a capability the adapter
does not offer, `kernel` is `None`; at launch, when a kernel's `requires` names a capability the device was not opened
with; on `wasm32` every non-empty request is refused at open), `KernelTarget` (a kernel passed by its descriptor whose
target is not the GPU; the typed entry points refuse the same case at compile time with E0277) and `SubgroupSize` (a
subgroup kernel whose workgroup is not a multiple of the largest subgroup size wgpu reports for the adapter, which
GUST-E0751 checks against 64 at compile time). Decisions: D140, D142, D145.

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
Rejected: a tuple expression `(1u32, 2u32)`, an array repeat `[0u32; 4]`, a block used as a value `{ 1u32 }`, the dereference `*x` (`*self` included: no value of the dialect is a reference, so write `self.field` or use a local copy).
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

# The newer language features

The dialect has grown from "functions and structs of scalars" to a useful part of everyday Rust: methods,
field-less enums with `match`, tuples, fixed-size arrays, closures that capture nothing, traits with static dispatch,
and generic helpers with trait bounds. Each is checked by rustc as ordinary Rust and then translated to Slang, and each
runs on the GPU and on the CPU twin. This chapter says what each form is and where its limits are, with one program that
uses most of them. The full forms, with every rejected shape and its code, are in the README
(["Current subset"](https://github.com/nevermemo/gust/blob/main/README.md)), in
[DIAGNOSTICS](https://github.com/nevermemo/gust/blob/main/docs/DIAGNOSTICS.md) and in sections 9 to 14 and 16 of
[SEMANTICS](https://github.com/nevermemo/gust/blob/main/docs/SEMANTICS.md).

## What is accepted

- **Methods and `impl` blocks.** `impl S { .. }` of a struct of the module, in any number of blocks, with `&self`,
  `&mut self` and `self` methods and associated functions without a receiver (`S::make(..)`, `Self::make(..)`). A
  method becomes a member of the Slang struct. A `&mut self` method is callable on a `let mut` local (or a field or
  element of one) only, never on a storage, uniform, workgroup or invocation-id place (GUST-E0333). No inherent `impl`
  for an enum or a type outside the module, or with generics (GUST-E0121, E0114); no `mut self` or typed receiver
  (GUST-E0116).
- **Field-less enums and `match`.** `enum E { A, B = 3 }` with a `u32` or `#[repr(i32)]` repr; you derive `PartialEq`
  yourself. Compare with `==`, cast with `as` to the repr type, and match as a Slang `switch`: a statement `match` on an
  enum or an integer, and a value `match` as the initializer of a `let` (typed, or untyped when an arm shows the type). No enum with data (GUST-E0122), no enum
  in a struct field, buffer or uniform (GUST-E0211), no guard (GUST-E0418), no `break` or `continue` in an arm
  (GUST-E0417).
- **Tuples.** `(a, b)` of two to eight elements as a local, a parameter or a result of a helper or method, with `t.0`
  and `let (a, b) = ..;`. Not stored in a struct field, buffer, uniform or workgroup memory (GUST-E0212), and no tuple
  comparison (GUST-E0213).
- **Fixed-size arrays.** `[T; N]` with `N` a literal above zero: a typed local, parameter or result, literals and
  repeats `[e; N]`, indexing with `as usize`, `.len()`, and a struct field or buffer element for the stored lengths. No
  nested arrays or arrays of tuples (GUST-E0217), no `for x in a`, and no array in a uniform (GUST-E0215). An index out
  of range is outside the defined domain.
- **Closures that capture nothing.** `let f = |x: f32, y: f32| -> f32 { .. };` with every parameter typed, an explicit
  result type and a block body, called as `f(a, b)` after its `let`. It uses its parameters, its own locals, and the
  helpers and methods of the module; it is not a value (not passed, returned or stored). Each becomes a hoisted Slang
  function (GUST-E0329 to E0332 for the rejected forms).
- **Traits with static dispatch.** `trait T { fn m(&self) -> R; .. }` (methods with a receiver and no body) and
  `impl T for S` of a struct or field-less enum of the module. A trait becomes a Slang `interface` and each impl an
  `extension`; a call is a call of the concrete method. No `dyn`, no `impl Trait`, no default bodies, associated items,
  supertraits or generic traits (GUST-E0124, E0126, E0218).
- **Generic helpers and methods with trait bounds.** `fn f<T: A + B>(t: T) -> T`, a `where` clause, and generic methods
  of an `impl`. Every type parameter needs at least one trait of the module as a bound (GUST-E0125); type arguments are
  inferred (no turbofish, GUST-E0123) and `T` is a whole parameter, result or `let` type. Slang specialises them
  statically; the CPU twin keeps the generic as plain Rust. No generic kernels, structs, enums or trait methods, and no
  arithmetic on `T`.

## Arrays and closures in a few lines

This kernel computes a four-tap window with a closure and a local array. It is a hand-written listing for this chapter,
not a gallery entry, and the doctest checks that the macro accepts it:

```rust,no_run
# use gust::gpu_module;
#[gpu_module]
mod window {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sum4(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        if i < out.len() && i + 4u32 <= input.len() {
            let weighted = |x: f32, w: f32| -> f32 { x * w };
            let mut taps: [f32; 4] = [0.0f32; 4];
            for k in 0u32..4u32 {
                taps[k as usize] = weighted(input[i + k], 0.25f32);
            }
            out[i] = taps[0] + taps[1] + taps[2] + taps[3];
        }
    }
}
```

## One kernel with all of it (G7)

G7, `state-machine`, is a tokenizer: a stream of ASCII symbols is cut into records of 16, and one invocation scans one
record with two finite-state machines at once, one that finds words and one that finds numbers. It uses an enum with
`match` (`State`, `Class`), a tuple result (`transition`, `totals`), two traits with static dispatch (`Classify`,
`Rules`), a generic method with two bounds (`Machine::feed`, instantiated twice) and methods with `&mut self` and
`&self`. The workarounds in the source are the limits above, shown in use: an enum cannot be a struct field, so
`Machine` keeps its state as the integer `State as u32`; a struct literal is a `let` initializer or the right side of an
assignment, so the arguments of `feed` are bound to locals first; a struct needs a field, so the two rule types carry an
unused `tag`.

```rust,no_run
# use gust::gpu_module;

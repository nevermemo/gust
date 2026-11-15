# The authoring model

A GUST kernel is Rust source that rustc checks and the `#[gpu]` macro then translates, node by node, into Slang. There
is no second language to learn and no intermediate representation in between: the macro reads the Rust syntax tree and
writes Slang text, and `slangc` compiles that text for the GPU. This chapter is about what you can write inside a
`#[gpu]` module, how a kernel is declared, and what happens when you write something the dialect does not accept.

## A `#[gpu]` module

`#[gpu]` goes on an inline module (`mod name { .. }`, not `mod name;`). The module holds named-field structs, helper
functions and kernels (and, see chapter 8, impl blocks, enums and traits). It imports the shader prelude for you,
so the names below are in scope without a `use`:

- the scalar types `float`, `int` and `uint` (the Rust spellings `f32`, `i32` and `u32` mean the same and are also
  accepted), booleans, and structs built from them;
- `SV_DispatchThreadID`, the type of the parameter that says which invocation this is;
- `StructuredBuffer<T>` for a buffer a kernel only reads, `RWStructuredBuffer<T>` for one it writes (the parameter is
  declared `mut`), and `Uniform<T>` for one small constant value.

These are lightweight shadow types written in Rust, so that rustc can parse, resolve and type-check the body as it would
any other function. The Slang text is made from the same syntax tree; the Rust function is kept as a private shadow
that is never called by your program.

A kernel is a `pub fn` with `#[kernel(workgroup_size(x, y, z))]`. It has one dispatch-id parameter and then its
resources, and it returns nothing. A helper is a plain function of the module that kernels call; it takes values, not
resources. For every public kernel the macro defines a zero-sized host handle with the kernel's name, inside the
module: `module::kernel`. Its `DESCRIPTOR` holds the generated Slang source and the entry point, and its typed
`call`, `submit` and `dispatch` functions are how the host runs it ([chapter 4][ch4]).

## A kernel with a uniform

The first listing is gallery entry G2, `saxpy-uniform` in the repository's `gallery/` directory. It computes
`out[i] = a * x[i] + y[i]`, where `a` is one value that every invocation reads:

```rust,no_run
# use gust::gpu;

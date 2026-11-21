```

`atomic_add`, `atomic_exchange` and `atomic_compare_exchange`, and the unsigned-only `atomic_min` and `atomic_max`, are
statement calls on an element of an `RWStructuredBuffer<u32>` (or `<i32>`, which has no min or max); only the old value of `atomic_add` may be bound, as the whole initializer of a `let` (`let slot = atomic_add(&mut counts[cell], 1u32);`), and which invocation gets which old value is not specified, so compare sets, never per-invocation values; the receiver
is written `&mut buffer[index]`, the one reference the dialect admits. The sum of `atomic_add` does not depend on the
order in which invocations arrive, so the GPU result equals the host's count exactly, and the gallery program asserts
that. An atomic whose result does depend on the order (a contended `atomic_exchange` or `atomic_compare_exchange`) has a
set of allowed outcomes on a GPU, and the CPU twin, which runs invocations one at a time in ascending order, produces
one member of that set. That is a documented limitation, not a bug that will be chased (the project's D39 rule for
edge-case differences between the CPU and the GPU): see section 6 of
[SEMANTICS](https://github.com/nevermemo/gust/blob/main/docs/SEMANTICS.md) and its known-mismatch ledger.
Atomics order only atomic operations on the same location; do not use one to publish ordinary data to another
invocation. [Chapter 7][`crate::guide::semantics_and_the_twin`] returns to the contract.

## The accepted subset

The dialect is a subset of Rust that translates to Slang directly. This is the working summary; the "Current subset"
section of the [README](https://github.com/nevermemo/gust/blob/main/README.md) is the complete, current list.

- **Types:** `float`/`f32`, `int`/`i32`, `uint`/`u32`, booleans, named-field structs (nested, padding-free for buffers),
  the vector and matrix types of `gust::math` (`Vec2` to `Vec4`, `Mat2` to `Mat4`), and `Option<T>` and `Result<T, T>`
  as local and helper values. Fixed-size arrays, tuples, enums, closures, methods, traits and generics are the language
  of [chapter 8][`crate::guide::m9_language`].
- **Statements and expressions:** `let` (with type annotations), arithmetic, comparison and logical operators,
  assignment, indexing, field access, `if`/`else`, casts, `return`, calls to helpers of the same module, and bounded
  `for i in start..end` loops with `break` and `continue`. Integer literals take 32-bit suffixes (`1u32`);
  `wrapping_add`, `wrapping_sub`, `wrapping_mul` and `wrapping_neg` are how a kernel asks for wrapping arithmetic.
- **Resources:** structured buffers, uniforms, the textures and samplers of
  [chapter 5][`crate::guide::gpu_concepts`], and workgroup memory.
- **Not accepted:** allocation, the standard library, macros, `while` and `loop`, inclusive ranges, `?`, `unwrap` and
  `expect`, `unsafe`, recursion, `dyn` and `impl Trait`, references (apart from an atomic's receiver), arbitrary method
  calls, and calls to functions outside the module. These limits describe the implemented translator, not Slang.

The macro also writes compile-time assertions that every buffer element and struct field has the layout Rust says it
has, so a mismatch between the Rust type and the buffer layout is a rustc error that names the field, not a wrong answer
at run time.

## Diagnostics

Every rejection the macro reports starts with a stable code, for example:

```text
error: [GUST-E0401] `loop` is not supported: its trip count is not fixed at entry, ...
```

**Codes are stable; messages are not.** A code keeps its meaning and is never reused, while the words after it may be
reworded, so match on the code. The codes are grouped by what they are about:

| Range | About |
| --- | --- |
| E01xx | module and item shape |
| E02xx | types |
| E03xx | expressions and calls |
| E04xx | control flow and loops |
| E05xx | `Option` and `Result` |
| E06xx | atomics |
| E07xx | macro entry and `#[kernel]` options |
| E08xx | rejections only the emitter can make |
| E09xx | reserved names and shadowing |

[DIAGNOSTICS](https://github.com/nevermemo/gust/blob/main/docs/DIAGNOSTICS.md) lists every code with what it rejects
and what to write instead; search it for the code in your error. This block is a kernel that the macro refuses, so the
doctest expects it not to compile (the compile failure is the point; the doctest checks that the build fails, not which
code it reports):

```rust,compile_fail
# use gust::gpu;
#[gpu]
mod rejected {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn spin(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        let i = id.x;
        loop {
            if i < out.len() {
                out[i] = 1u32;
            }
            break;
        }
    }
}
```

The next kernel differs from that one only in using a bounded `for` loop (which needs no `break`) where the other
has the `loop`. It is compiled by the doctest (`no_run`: nothing runs), so the pair shows that the `loop` is what the
macro refuses:

```rust,no_run
# use gust::gpu;
#[gpu]
mod accepted {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn once(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        let i = id.x;
        for _round in 0u32..1u32 {
            if i < out.len() {
                out[i] = 1u32;
            }
        }
    }
}
```

Because the body is ordinary Rust to rustc, a type error in a kernel is an ordinary rustc error with an ordinary span.
An error that `slangc` reports later, in the emitted Slang, is mapped back to the Rust line when the kernel's source
file is known.

## The twin, in one paragraph

In a test build of the crate that defines the kernel (or with `--cfg gust_dual`), the macro also emits a CPU twin that
runs the Rust shadow body on `gust::cpu::CpuDevice`, which is how the gallery's tests check a kernel without a GPU. A
production build contains no twin. A kernel that uses workgroup memory or a barrier has none and says
`#[kernel(.., cpu = "unsupported")]` ([chapter 5][`crate::guide::gpu_concepts`]). What the twin proves and what it
cannot is the subject of [chapter 7][`crate::guide::semantics_and_the_twin`].

## Where to go next

[Chapter 4][ch4] runs these kernels. [Chapter 5][`crate::guide::gpu_concepts`] adds shapes, textures, workgroup
memory, barriers and subgroups. [Chapter 8][`crate::guide::m9_language`] covers the newer language features.

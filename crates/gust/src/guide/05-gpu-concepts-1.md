# GPU concepts in Rust

A GPU runs thousands of invocations of one kernel at once, in no order you control. Most of what is new when you write
a kernel comes from that, and the dialect has a construct for each part of it. This chapter names the concepts and shows
the kernel half of each in Rust; the host half, which needs a device, is in the sibling chapter
[`gust_wgpu::guide::gpu_concepts_running`][ch5-running]. The sources of truth are sections 4 to 6 of
[SEMANTICS](https://github.com/nevermemo/gust/blob/main/docs/SEMANTICS.md) and the README sections on shapes, textures,
workgroup memory and capability variants.

## Invocations, workgroups and the dispatch

A dispatch launches `element_count` invocations, grouped into workgroups of the size the kernel declares in
`workgroup_size(x, y, z)`. The call rounds the launch up to whole workgroups, which is why a kernel guards its tail
(`if i < out.len()`). Inside a kernel:

- `SV_DispatchThreadID` is the global invocation id (`id.x`, `id.y`, `id.z`);
- `LocalId` is the invocation's place inside its workgroup and `GroupId` is the workgroup's place in the dispatch; they
  are `id % size` and `id / size` per axis.

Dispatches may be one-, two- or three-dimensional.

## Two memory rules the twin cannot check

A kernel is inside the contract only if every index into a buffer is below its `len()`, and no two invocations of one
dispatch write the same location or read what another writes, except through atomics. On a GPU a violation of either is
a dynamic error whose outcome is open, and a race is not reliably detected. The CPU twin hides both: it runs
invocations one at a time in ascending order and so produces one definite answer. Every reference value in the gallery
is therefore order-independent by construction (each written index has one writer, or the kernel uses atomics), and the
race-free rule is yours to keep. Section 5 of SEMANTICS has the details, and chapter 7 the whole contract.

## Shapes: `grid` and `pixel`, and textures

A kernel has a shape. `grid` is the compute dispatch used so far and the default. `pixel` runs the body in the fragment
stage instead: the host draws a full-screen triangle, each fragment computes one element and stores its outputs, which
stay ordinary buffers, at its own index. It is computation, not a graphics API; GUST owns no blend, depth or raster
state. A kernel opts in with `#[kernel(shape = pixel)]`. A `pixel` kernel has no workgroup size, no guard and no
padding invocation, and it is limited to one-dimensional calls, at most 2^26 elements, and an adapter whose fragment
stage can write storage buffers. A kernel with workgroup memory, a barrier, atomics, a storage texture, or an output
written at another index is `grid`-only, and a pin the kernel cannot take is a build error (GUST-E0723 to E0728).

Textures and samplers are kernel parameters too:

| Parameter | Access | Methods |
| --- | --- | --- |
| `Texture2D<T>` | sampled | `load`, `sample` (float `T` only), `width`, `height` |
| `StorageTexture2D<F>` | read and write, `F` is `R32Float`, `R32Uint` or `R32Sint` | `load`, `store`, `width`, `height` |
| `WriteTexture2D<F>` | write only, nine formats | `store`, `width`, `height` |
| `Sampler` | none | the first argument of `sample` |

G5, `image-threshold`, writes the same decision twice. `binarize` is a `grid` kernel on a storage texture, in place, and
guards against the texture's own size, which it reads like a buffer's length. `flag` is a `pixel` kernel: it runs once
for each element of a one-dimensional call, reads the texel through a sampler at its centre, and writes one flag.

```rust,no_run
# use gust::gpu;

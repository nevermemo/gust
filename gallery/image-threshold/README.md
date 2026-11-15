# image-threshold (gallery G5)

A brightness threshold over a small image, written twice: once as a `grid` kernel on a **storage texture** and once as a
`pixel` kernel that reads a **sampled texture through a sampler**. Both decide the same thing (is the texel brighter than
one half?), and the program checks both against a host loop, texel for texel.

What it shows:

- A **storage texture** (`StorageTexture2D<R32Float>`): the `grid` kernel `binarize` loads a texel and stores 1.0 or 0.0
  back to the same address, in place. The texture is made with `TextureUse::STORAGE`, written with `write_texture`, run
  with `dispatch_extent` (an `Extent` of width by height, not a count) and read back with `read_texture`. The dispatch is a
  whole number of 8 by 8 workgroups, so it can be larger than the image; the kernel reads the image size with `width()` and
  `height()` and guards against it.
- A **sampled texture and a sampler** (`Texture2D<f32>` and `Sampler`): the `pixel` kernel `flag` samples the texture at
  the centre of its own texel with a nearest sampler. The texture is made with `TextureUse::SAMPLED`, and the sampler is a
  value of its own (`create_sampler`, with a `Filter` and an `Address` mode), so the same kernel could be given a
  different sampler.
- The **`pixel` shape beside `grid`**: `#[kernel(shape = pixel)]` runs the body once for each element `0..n` of a
  one-dimensional call, as a fragment shader, with no workgroup size and no guard (the element index is the whole story).
  Its output is a `RWStructuredBuffer` written at the element's own index on every path, and the call takes a count, not an
  extent. The classifier (SEMANTICS, "The `pixel` shape at run time") admits a kernel as `pixel` only when it fits those
  rules, which is why the `grid` kernel, which writes a texture, stays `grid`.

The input is `((7 x + 13 y) % 17) / 16`: sixteenths, so every value is exact in `f32`, texels of exactly one half occur
(the comparison is strict, so they are not bright), and the image has both bright and dark texels.

Tests (`cargo test`, CPU twin only): both kernels on `CpuDevice` against a plain host loop, **exact**, at 1 by 1, 7 by 3,
8 by 8, 9 by 9, 37 by 21, 64 by 33, 100 by 100 and 200 by 50 texels (at every size but 8 by 8 the last row or column of
8 by 8 workgroups is partial; the twin has no rows, so the `pixel` partial row is the GPU's); a hand-computed first row; a
guard that the image is neither all bright nor all dark. `main` runs the same check on the GPU at 37 by 21 and at 200 by
50, whose 10 000 texels pass the 8192-wide rows of a `pixel` call and end in a partial second row, and asserts both
kernels against the host before it prints.

Why there is no tolerance: every texel is a multiple of 1/16, the comparison with one half is exact, a nearest read at a
texel centre is that texel, and the output is 0 or 1. A **linear** sampler would interpolate and need a stated bound
(SEMANTICS, section 4).

Edge cases (D39, documented and not chased): the position of a nearest read is the texel centre, which is never near a
texel edge, so ledger row KM-14 (a nearest read next to an edge) does not apply, and every address is inside the image, so
KM-12 (an address outside the texture) does not either. A `pixel` call covers at most 2^26 elements, and its elements run
in no defined order on a GPU and in ascending order on the twin; this body reads one texel and writes one flag, so the order
does not matter.

Commands, from this directory (the tools are set up as for `hello-gust`):

```sh
cargo test                        # the CPU twin against the host loop; no GPU, no Slang
cargo run --features gust-dev     # development: compiles the kernels at run time through Slang, needs an adapter
cargo gust build                  # compiles the kernels with Slang and writes gust-artifacts/ (commit them)
cargo gust check                  # fails when gust-artifacts/ no longer matches the sources
cargo build --release             # no Slang needed: the kernels are embedded from gust-artifacts/
cargo run --release               # needs a Vulkan compute adapter
```

It prints `image-threshold: 37 x 21 texels, 366 bright by the pixel kernel, 366 by the grid kernel` (the image is fixed, so
the counts are too) and a line `adapter: <name>` with the GPU it ran on.

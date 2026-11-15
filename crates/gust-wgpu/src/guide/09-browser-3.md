```

The page logs one console line, and that line is what the repository's runner reads. On success it is

```text
GUST_OK <vendor> <architecture> | selected add.map_kernel=<key>:-:0 | map_async exact against the host reference
```

The adapter words are what the browser says about its adapter (`GPUAdapter.info`), or `unknown`. `selected` names the
key of the artifact that the build embedded, which the runner compares with the committed
`gust-artifacts/manifest.json`. The last part is logged only after all 1000 results equal the reference bit for bit. Any
error or panic logs `GUST_FAIL <reason>` instead.

## Building and running it yourself

A browser build is a `web` build (`GUST_PROFILE=web`, [chapter 6][`crate::guide::profiles_artifacts_routing`]). That is
a strict profile: the kernel is the WGSL artifact committed in `gust-artifacts/`, and a browser build never runs Slang.
The crate is a `cdylib`; `wasm-bindgen --target web` turns it into a module that `index.html` loads. You need the
`wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`), a `wasm-bindgen` command line tool of
exactly the version that `Cargo.lock` pins for the `wasm-bindgen` crate, and a browser with WebGPU. From the entry's
directory:

```sh
GUST_PROFILE=web cargo build --release --target wasm32-unknown-unknown --locked
wasm-bindgen --target web --no-typescript --out-name web_hello --out-dir site \
    ../target/wasm32-unknown-unknown/release/web_hello.wasm
cp index.html site/
python -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8000/` or `http://localhost:8000/`. WebGPU needs a secure context, and those two addresses count
as one over plain HTTP; a page served from another host name needs HTTPS. Serve on the loopback address only: the
files are a development build of your own program, not something to put on a network.

The same crate is a plain library on the host. Its `tests` run the kernel on the CPU twin against the reference for 1,
63, 64, 65 and 1000 elements and check the reference by hand on a few values, so `cargo test` needs no browser, no GPU
and no Slang. The browser glue sits in a target-scoped dependency table, so a host build compiles none of it.

## What the repository checks, and what it cannot

- **On the host (`cargo test` in the entry).** The twin against the reference, exact. A twin run is never GPU evidence
  ([chapter 7][`gust::guide::semantics_and_the_twin`]).
- **`cargo xtask check-gallery`.** Builds the crate natively with the other entries, compares its `gust-artifacts/` with
  a fresh Slang build and, in the GPU part, builds it for `wasm32-unknown-unknown`, runs `wasm-bindgen`, serves the
  output from a loopback server and an ephemeral port, opens it in headless Chrome and requires the `GUST_OK` line
  above within 90 seconds. A missing Chrome, `wasm-bindgen` or wasm32 standard library fails the part, and nothing is
  skipped.
- **Continuous integration.** The `wasm` job checks that the browser surface of the crates and the page's own code
  compile and lint clean for `wasm32-unknown-unknown` under the `web` profile, and that wgpu's WebGPU feature is on for
  `wasm32` and off for the host. It does not run anything in a browser: the CI runners have no WebGPU adapter. The
  browser run is part of `cargo xtask check-full`, on a machine that has Chrome and `wasm-bindgen`.

So a green CI run says the page compiles for the browser, and says nothing about whether it ran there. What the browser
run has shown is narrow. It is run in headless Chrome (`--headless=new`, no WebGPU flag) on an NVIDIA RTX 5090 under
Windows, and that is the only configuration this entry has been run in. The WebGPU backend that Chrome chose there was
not recorded. Chrome and Edge ship WebGPU on desktop, and whether Firefox or Safari run the page was not tried. On
Linux, Chrome may need extra flags (`--enable-unsafe-webgpu` and a Vulkan feature flag; not tried), and a machine with
no GPU can use the software adapter (SwiftShader), which the check labels as software evidence; `cargo xtask check-web`
accepts extra Chrome arguments for this, and the gallery run passes none. The adapter line names the adapter a second
request returned, which is expected to be the one `map_async` used. The exact comparison, not the label, is what shows
that a device ran the kernel.

## One kernel is not every kernel: Tint

The `web` profile compiles a kernel to WGSL only, and in the browser that WGSL goes through the browser's own compiler
(Tint, in Chrome), not through wgpu's naga as on the desktop. `add` is the smallest kernel, so the page shows only
that it runs there. The newer language features were run once. The kernel of the
[`state-machine`](https://github.com/nevermemo/gust/blob/main/gallery/state-machine/README.md) entry (the enum and
`match`, tuples, traits, the generic `feed` and the methods of [chapter 8][`gust::guide::m9_language`]) went through
the same runner as a scratch copy of the kernel and its committed WGSL artifact, built for `wasm32-unknown-unknown`
under `GUST_PROFILE=web`, with 1000 records run through `submit` and an awaited read-back. Headless Chrome 156 on an
NVIDIA Blackwell GPU accepted the artifact in Tint, and every record equalled the host reference. That is one adapter
and one Chrome version, measured on 2026-11-15, and the entry's README records it; the scratch crate is not in the
repository, and the `state-machine` entry stays a native program. Nothing else is claimed: another kernel needs its own
run.

What the browser backend lacks is the blocking calls and the generated `map`, as above, and subgroups. The "Platforms"
section of the [README](https://github.com/nevermemo/gust/blob/main/README.md) says what has run. A `web+subgroups`
build is refused at build time, and a device opened with a capability in a browser is `Error::MissingCapability`,
because the pinned wgpu cannot request the feature there; a web build runs the portable body of a kernel.

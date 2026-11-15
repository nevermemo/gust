# web-hello (gallery G9)

The sum of `hello-gust` in a browser: one `#[gpu]` map function, `add(a, b) = a + b`, run on the browser's GPU through
`map_async`, and compared in the page, element by element and bit for bit, with a plain Rust host reference.

What it shows: the browser path. A browser cannot block, so the blocking calls (`dispatch`, `map`) do not exist on
`wasm32`; `add::map_async(&a, &b).await` opens the default device (`gust_wgpu::default_device_async`), uploads the
slices, runs the generated kernel and awaits the read-back. The crate is a `cdylib` for `wasm32-unknown-unknown`, built
under `GUST_PROFILE=web` (the strict profile: the kernel is the WGSL artifact committed in `gust-artifacts/`, a browser
build never runs Slang), turned into a module by `wasm-bindgen --target web` and loaded by `index.html`. On the host the
same crate is a plain library: its `tests` run the kernel on the CPU twin against the host reference, so `cargo test`
needs neither a browser nor a GPU nor Slang. The browser glue is in the target-scoped `[target.'cfg(target_arch =
"wasm32")'.dependencies]` table, so a host build compiles none of it.

The page logs one console line, which is what the runner reads:

```text
GUST_OK <vendor> <architecture> | selected add.map_kernel=<key>:-:0 | map_async exact against the host reference
```

The adapter words are what the browser says about its adapter (`GPUAdapter.info`); `selected` names the key of the
artifact the build embedded, which the runner compares with `gust-artifacts/manifest.json`; the last part is only logged
after all 1000 results equal the host reference bit for bit. Any error or panic logs `GUST_FAIL <reason>` instead.

## Build and run it yourself

Needs the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`), a `wasm-bindgen` CLI of exactly
the version `Cargo.lock` pins for the `wasm-bindgen` crate (`cargo install --locked wasm-bindgen-cli --version 0.2.127`),
and a browser with WebGPU. From this directory:

```sh
GUST_PROFILE=web cargo build --release --target wasm32-unknown-unknown --locked
wasm-bindgen --target web --no-typescript --out-name web_hello --out-dir site \
    ../target/wasm32-unknown-unknown/release/web_hello.wasm
cp index.html site/
python -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8000/` (or `http://localhost:8000/`): WebGPU needs a secure context, and `localhost` and
`127.0.0.1` count as one over plain HTTP; a page served from another host name needs HTTPS. The page prints the same
line it logs, and the browser's console shows it. Serve on the loopback address only: the files are a development build
of your own program, not something to expose on a network.

Browsers: the check runs headless Chrome (`--headless=new`, no WebGPU flag) on an NVIDIA RTX 5090 under Windows; that is
the only configuration this entry has been run in, and the WebGPU backend Chrome chose there was not recorded. Chrome and Edge ship WebGPU on desktop; whether
Firefox or Safari run the page was not tried here. On Linux, Chrome may need `--enable-unsafe-webgpu` and a Vulkan
feature flag (not tried here), and a machine with no GPU can use the software adapter (SwiftShader), which the check
labels `Gpu (software ...)`. `cargo xtask check-web` takes `--chrome-arg` for such flags; the gallery run passes none.

## Tests and the check

`cargo test` (host, CPU twin only): the kernel through `add::map_on` on `CpuDevice` against the host reference, exact, for
1, 63, 64, 65 and 1000 elements; the four program values; three values of the reference counted by hand. The sums are
small integers, exact in `f32` on any target, so every comparison is exact and no tolerance exists. The host reference
is a plain Rust loop that never runs the kernel; the twin is never GPU evidence.

`cargo xtask check-gallery` (from the repository root) builds this crate natively with the other entries (the no-Slang
part: build, `cargo test`, anchors), compares its `gust-artifacts/` with a fresh Slang build (the Slang part), and in
the GPU part builds it for `wasm32-unknown-unknown`, runs `wasm-bindgen`, serves the output from a server on `127.0.0.1`
and an ephemeral port, opens it in headless Chrome and requires the `GUST_OK` line above. It is the runner of `cargo
xtask check-web`, called with this page's constants. A missing Chrome, `wasm-bindgen` or wasm32 standard library fails the
part, and so does a page that logs no result within 90 seconds; nothing is skipped. CI compiles and lints the page's wasm32 code (a clippy step of the `wasm` job) but does not run the GPU part (its
runners have no WebGPU adapter); the browser run belongs to `cargo xtask check-full`, which needs Chrome and
`wasm-bindgen` for `check-web` already.

## Limits

- The `web` profile compiles the kernel to WGSL only, and the WGSL goes through the browser's compiler (Tint in Chrome),
  not the one the desktop run uses (wgpu's Naga). `add` is the smallest kernel, and the page proves only that it runs
  there: other kernels need their own run, and `docs/SEMANTICS.md` lists what the browser backend lacks (no subgroups, no
  blocking calls).
- The adapter line names the adapter a second request returned, which is expected to be the one `map_async` used; the
  exact result comparison, not the label, is what proves a GPU ran.

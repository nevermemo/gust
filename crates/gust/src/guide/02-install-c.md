```

These three listings are byte-identical to the anchored regions `imports`, `kernel` and `main` of the repository's
`gallery/hello-gust/src/main.rs`, and `cargo xtask check-gallery` fails when they drift apart. That gallery crate also
holds the `tests` module that runs the kernel on the CPU twin and compares it with a plain Rust host reference.

## Run it while you develop

```sh
cargo run --features gust-dev
```

The first call of the kernel compiles it through Slang, so this needs the tools set up above and a Vulkan compute
adapter. It prints `hello-gust: [11.0, 22.0, 33.0, 44.0]` and a line `adapter: <name>` with the GPU it ran on.

## Build it for release

```sh
cargo gust build -p hello-gust    # compiles the kernels with Slang, verifies them and writes gust-artifacts/
cargo build --release
```

A release build with `GUST_PROFILE` unset is the strict `native` profile: every `#[gpu]` kernel needs its artifact,
and the binary holds no Slang bridge. Commit `gust-artifacts/`; `cargo gust check` fails when it no longer matches the
sources.

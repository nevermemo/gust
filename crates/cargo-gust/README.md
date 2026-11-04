# cargo-gust

The command-line tool of GUST (`cargo gust ...`). It collects the `#[gpu]` kernels of a crate, compiles them with
Slang and writes the committed `gust-artifacts/` directory that a production build runs without any compiler. It also
installs and checks the pinned Slang release.

GUST is an experimental Rust to Slang compute subsystem. It is pre-1.0: expect the tool and its output to change
between 0.x releases.

```sh
cargo install cargo-gust --locked --version "~0.1"
cargo gust doctor              # reports the Slang install and the native reflection helper
cargo gust slang install       # installs the pinned Slang release
cargo gust slang build-helper  # builds the native reflection helper (needs a C++ compiler)
cargo gust build               # writes gust-artifacts/ for the kernels of this crate
```

This is a binary-only crate, so it has no docs.rs page; `cargo gust --help` lists the commands. The library it works
with is `gust-compute` (library name `gust`): <https://github.com/nevermemo/gust/blob/main/README.md>. A first program,
from the install to a release build, is in the root README's section "Install from crates.io"
(<https://github.com/nevermemo/gust/blob/main/README.md#install-from-cratesio>). Minimum supported Rust: 1.90.

The tests of this crate read files of the repository, so they run from a checkout, not from this package.

Licensed under either of MIT or Apache-2.0 at your option (`LICENSE-MIT`, `LICENSE-APACHE`).

# cargo-gust

The command-line tool of GUST (`cargo gust ...`). It collects the `#[gpu]` kernels of a crate, compiles them with
Slang and writes the committed `gust-artifacts/` directory that a production build runs without any compiler. It also
installs and checks the pinned Slang release.

GUST is an experimental Rust to Slang compute subsystem. It is pre-1.0 and follows SemVer for 0.x: a break bumps the
minor version of all four packages, a patch only adds or fixes, so expect the tool to change between 0.x minors.

What this crate promises (the full text is the root README's section
[Stability](https://github.com/nevermemo/gust/blob/v0.2.0/README.md#stability)): the `cargo gust` commands and the lines
they print, which scripts and CI read; the file formats it writes and reads (`gust-artifacts/manifest.json` schema 4
(`split_of` links a split artifact to its body), record files schema 5, reflection records, and a newer release reads
every older one); `gust.toml` and the
`[package.metadata.gust]` table; and the environment variables `GUST_PROFILE`, `GUST_SLANG_HOME`,
`GUST_SLANG_REFLECT`, `GUST_ROUTE__*`, `GUST_CPU_THREADS` (read by the CPU runtime of `gust-compute`, not by this tool),
`GUST_EXPORT_DIR` and `GUST_ROUTED_PROFILE` (the last two are set by `cargo gust` for the crates' macros; do not set
them yourself): seven in all, as in the root README. It does not promise the internal `cargo-gust-bridge` feature, any other
`GUST_*` variable, or the rules for finding Slang that it shares with the run-time bridge. The pinned Slang release is
named in the CHANGELOG; a pin bump that changes an artifact's content key is a minor release.

```sh
cargo install cargo-gust --locked --version "~0.2"
cargo gust doctor              # reports the Slang install and the native reflection helper
cargo gust slang install       # installs the pinned Slang release
cargo gust slang build-helper  # builds the native reflection helper (needs a C++ compiler)
cargo gust build               # writes gust-artifacts/ for the kernels of this crate
```

This is a binary-only crate, so it has no docs.rs page; `cargo gust --help` lists the commands. The library it works
with is `gust-compute` (library name `gust`): <https://github.com/nevermemo/gust/blob/v0.2.0/README.md>. A first program,
from the install to a release build, is in the root README's section "Install from crates.io"
(<https://github.com/nevermemo/gust/blob/v0.2.0/README.md#install-from-cratesio>). Minimum supported Rust: 1.90.

The tests of this crate read files of the repository, so they run from a checkout, not from this package.

Licensed under either of MIT or Apache-2.0 at your option (`LICENSE-MIT`, `LICENSE-APACHE`).

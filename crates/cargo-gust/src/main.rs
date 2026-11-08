//! `cargo gust`: the export tool for GUST kernels (D45, D46).
//!
//! A `#[gpu]` module is compiled to Slang by the macro; this tool turns the Slang into WGSL (and SPIR-V) ahead of
//! time and commits the result, so that a production build needs no Slang at all. The pieces:
//!
//! 1. **Collect.** `cargo check -p <crate>` runs in an isolated target directory (`<target>/gust-export`) with
//!    `GUST_EXPORT_DIR` set to a directory that is new for the run and `GUST_PROFILE=dev`. In export mode the macro
//!    writes `<key>.slang` (the clean Slang text) and `<key>.kernel.json` (see `record.rs`) for every kernel. The
//!    record directory is deleted afterwards, so a kernel removed from the source leaves nothing behind. Collecting
//!    zero kernels is an error.
//! 2. **Compile and verify.** The native Slang helper compiles the *clean* text (no `#line` directives, so the
//!    helper's `sourceHash` is the hash a program recomputes at run time), the layouts are cross-checked against the
//!    descriptor (D17), and the result is verified the way `gust::artifact::verify_artifact` does at run time. SPIR-V
//!    is also run through `spirv-val --target-env vulkan1.2` when that tool is on `PATH`; the manifest records
//!    `validated: true|false`, never a silent pass.
//! 3. **Write.** Files are staged in `gust-artifacts/.staging-<pid>/` and moved into place only after every kernel
//!    verified; `manifest.json` is written last; files that are not part of the new set are pruned and reported.
//!    A failed export writes nothing.
//!
//! # Commands
//!
//! - `cargo gust init` runs `cargo clean -p <crate>` once on the normal target directory and then creates
//!   `gust-artifacts/manifest.json` (an empty manifest), because under `GUST_PROFILE=dev` a crate expanded before it
//!   had a manifest is not re-expanded when artifacts appear. The clean comes first, so a failed clean leaves no
//!   manifest and the command can be repeated; the error then says to run `cargo clean -p <crate>` yourself, and
//!   until then `dev` builds can be stale. `build` does the same before it writes the first manifest. Other target directories (rust-analyzer's, per-lane ones, a
//!   `CARGO_TARGET_DIR` set elsewhere) stay stale until their next re-expansion of that crate; strict builds are
//!   safe because a failed build is never cached.
//! - `cargo gust build [-p <crate>] [--format wgsl|spirv]... [--manifest-path <path>]` exports the crate. WGSL is
//!   always built (it is what the macro embeds); `--format spirv` adds SPIR-V, which is listed in the manifest but not
//!   embedded. A plain `build` keeps every format the existing manifest already lists (it prints a line saying so),
//!   so it never drops committed SPIR-V; a format is dropped only by deleting it from the manifest and the directory
//!   yourself.
//! - `cargo gust check [--recompile] [--allow-slang-mismatch]` collects again and fails unless the directory is
//!   exactly what a build would give: every collected key has its files, every manifest entry and every file is a
//!   collected key (orphans fail), the SHA-256 values match, the stored artifacts verify as a program would verify
//!   them, and the manifest's Slang build is the installed helper's. `--recompile` also recompiles and compares the
//!   bytes (S3). It prints one line per problem; the exit code is 1 when there is one. A plain `check` does not run
//!   `slangc -version`: it compares the helper's tag with the manifest's, so a matching older Slang passes. Only
//!   `build` and `check --recompile` require the pinned `slangc` and a helper of the same build.
//! - `cargo gust run [cargo arguments]` and `cargo gust test [cargo arguments]` are `cargo run` and `cargo test` with the
//!   route variables of the package's `gust.toml` set (M8, D65 to D67; see [`routes`]). They are the driver's commands
//!   that compile a binary. `-p`, `--manifest-path`, `--release`, `--profile` and `--target` are read to find the
//!   package and the build profile, and `--features`, `--all-features`, `--no-default-features`, `--locked`, `--frozen`,
//!   `--offline` and `--config` also reach the `cargo metadata` call that reads the declarations; every argument is
//!   passed to cargo unchanged. `--workspace` and `--all` are refused when a member they select (less `--exclude`) has a
//!   `gust.toml`, and pass through when none has one. `build` and `check` set the same variables on the `cargo check` that collects the
//!   kernels, and read the `gust.toml` beside the package they work on. With no `gust.toml` beside the package's manifest
//!   all four behave as before, and an old `gust-artifacts/routes.json` is removed. Otherwise the driver validates the
//!   file against the tags that the crates of the build declare in `[package.metadata.gust.routes]`, sets one
//!   `GUST_ROUTE__<package>__<tag>` per declared tag and writes the resolved table (tag, target, deciding level) to
//!   `gust-artifacts/routes.json` (`build`, `run` and `test`, after the toolchain check in `build`; `check` leaves it alone
//!   and does not count it as a stray). The one-line-per-tag summary and the notes on a preset `GUST_ROUTE__*` variable go
//!   to standard error. An error names `gust.toml`, the line and the tag, and cargo does not build (`cargo metadata` runs
//!   to read the declarations).
//! - `cargo gust doctor` shows the located Slang and where its licence notices are, the helper (refused when it was
//!   built against another Slang than the located `slangc`), `spirv-val`, the profile a build here would get and the
//!   install command.
//! - `cargo gust slang install|locate|build-helper` is the pinned-Slang manager (see below).
//!
//! The same sources and the same Slang give the same bytes: no timestamp, no path, sorted keys, LF line ends. Commit
//! `gust-artifacts/` with the crate and mark it `-text` in `.gitattributes` (`**/gust-artifacts/** -text`) so that git
//! never rewrites a line ending, which would break the hashes stored inside the reflection files.
//!
//! # `manifest.json`, schema 1
//!
//! See [`manifest`](crate::manifest) for the full shape. In short:
//!
//! ```json
//! { "schema": 1,
//!   "slang": "<the Slang build tag the helper reports>",
//!   "kernels": [ { "key": "<FNV-1a 64, 16 hex>", "module": "..", "name": "..", "entry_point": "..",
//!                  "clean_sha256": "<SHA-256 of the clean Slang text>",
//!                  "formats": [ { "format": "wgsl", "file": "<key>.wgsl", "sha256": "..", "reflection_sha256": ".." },
//!                               { "format": "spirv", "file": "<key>.spv", "sha256": "..", "reflection_sha256": "..",
//!                                 "validated": true } ] } ] }
//! ```
//!
//! A kernel whose storage-buffer or uniform layout the macro could not describe (`layout: null` in its record) is
//! built anyway: the layout is filled from the helper's reflection (a vector or matrix element too), the
//! descriptor-side half of the D17 cross-check is skipped for that parameter only, and the kernel's entry lists it as
//! `"export_unchecked": ["<param>", ...]` (`build` prints a note, `check` reports the same list). D17 still holds at
//! run time, where `verify_artifact` checks the stored reflection against the descriptor. The macro describes the
//! element of a `Uniform<T>` and any storage element whose types are all scalars, vectors, matrices and structs of
//! this module, so those are checked in full at export and are not listed.
//!
//! The Slang version recorded is the build tag the helper writes into every reflection JSON (`compiler`): the build
//! that actually compiled. With `--allow-slang-mismatch` that can differ from the pin (`SLANG_PIN` in
//! `gust::toolchain`), and then the manifest says so by carrying the other tag; `build` and `locate` print a note
//! every time.
//!
//! # The pinned Slang
//!
//! Authors who export need Slang; consumers and production builds never call these commands (D46, section 2.5 of the
//! M4 plan). The pin is the version the M4 Slang bump (m4.r) validated with the GPU, Slangc, Spirv and Reflect
//! suites, check-artifacts, check-precompiled and check-web; check-full ran at the M4 close (c964222).
//!
//! - `cargo gust slang install` downloads the release archive for this platform from Slang's GitHub release with
//!   `curl -fsSL`, checks its SHA-256 before anything is extracted, extracts it with `tar -xf` into
//!   `<cache>/<version>.partial-<pid>` and renames that to `<cache>/<version>`; on any failure the partial directory is
//!   removed. The archive layout is kept, so the release's `LICENSE`, `LICENSES/` and `third-party-notices/` stay
//!   beside the binaries (`doctor` and `install` list them).
//!   The cache is `GUST_SLANG_HOME` when set, else `<CARGO_HOME or ~/.cargo>/gust/slang`. To install without a network,
//!   put the release archive in `<cache>/archives/` and run the command again; its digest is checked the same way.
//! - `cargo gust slang locate [--allow-slang-mismatch]` prints the `slangc` a build would use: the cache
//!   (`GUST_SLANG_HOME` when set, else the default), then a `slangc` on `PATH`. A `PATH` slangc counts when its
//!   `-version` is the pin exactly or a build listed in `VALIDATED_BUILDS` (the release prints `2026.21`; the list is
//!   empty, since 2026.21 was validated with the release itself). Anything else is an error naming the install
//!   command and the download URL; a `PATH` slangc of another version is used only with the flag, and is printed.
//! - `cargo gust slang build-helper [--force] [--sdk-root <dir> --out-dir <dir>]` builds `gust-slang-reflect` from
//!   `assets/slang-layout.cpp` (a copy of `scripts/probes/slang-layout.cpp`, kept equal by a test) with the first of
//!   `cl`, `clang++`, `c++`, `g++` found (`CXX` overrides), into `<cache>/helper/<version>/` by default. `build`,
//!   `check --recompile` and `doctor` use that helper when `GUST_SLANG_REFLECT` is not set, and refuse a helper whose
//!   Slang build tag differs from the located `slangc -version`. `cargo xtask build-slang-reflect` runs this command.
//!   An existing helper is kept only when the build stamp beside it (`gust-slang-reflect.stamp`, written after a good
//!   build) holds the SHA-256 of the packaged probe source, and it prints the schema of that source and the Slang tag
//!   of the SDK's own `slangc -version`; no file time plays a part. Any other helper in the output directory is
//!   rebuilt, and the build says why. Two cargo-gust versions with the same probe source leave each other's helper
//!   alone. `--force` rebuilds a current helper too.
//!
//! # Installing
//!
//! The tool needs `gust`'s run-time Slang bridge (the `slang-runtime` feature), which a strict profile refuses, and a
//! plain `cargo install` builds in release, which infers the `native` profile. The tool therefore enables the internal
//! `cargo-gust-bridge` feature of `gust-compute` (D108): with it, a release build without `GUST_PROFILE` is built as `dev`
//! (the build prints a `cargo:warning` saying so), and no variable is needed:
//!
//! ```sh
//! cargo install cargo-gust --locked --version ~0.1
//! cargo gust slang install        # the pinned Slang into <CARGO_HOME>/gust/slang/<pin>, SHA-256 checked
//! cargo gust slang build-helper   # the native reflection helper, compiled against that Slang
//! cargo gust doctor               # names every gap, and the versions of cargo-gust and gust-compute
//! ```
//!
//! Where the run-time bridge and this tool look for `slangc` and the helper is `gust::toolchain`'s one rule (D109):
//! `GUST_SLANG_REFLECT` for the helper, then the managed cache (`GUST_SLANG_HOME`, else
//! `<CARGO_HOME or ~/.cargo>/gust/slang`), then, in a GUST checkout only, `target/slang-reflect`. There is no
//! current-directory lookup.
//!
//! # Limits
//!
//! Only kernels in the package's library and binary targets are collected (`cargo check -p`), not tests or examples.
//! The `#[gpu]` macro names records by their 64-bit key. In export mode it never overwrites an existing record whose
//! bytes differ: the second kernel with that key fails to expand (E0712, naming both kernels when it can read the
//! first one's name, else the key), and rewriting identical bytes is fine. The tool still rejects a record whose text
//! does not match its key, and two records that share a key and differ in text, as defence in depth.
//!
//! # Environment
//!
//! `GUST_EXPORT_DIR` belongs to `cargo gust build` and `check`, which set it for their own `cargo check`. Leave it
//! unset everywhere else. A stray `GUST_EXPORT_DIR` under `dev` turns a half pair (a manifest without its artifacts)
//! into a silent empty embed instead of a build error, because export mode treats the pair as absent. `cargo gust
//! doctor` reads `GUST_PROFILE` from the process environment only, not from `.cargo/config.toml` `[env]`.

mod check;
mod cli;
mod collect;
mod commands;
mod doctor;
mod export;
mod fill;
mod helper;
mod json;
mod manifest;
mod msvc;
mod record;
mod routes;
mod sha256;
mod slang;
mod store;
mod toolchain;
mod tools;

#[cfg(test)]
mod testdata;

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut arguments = Vec::new();
    for argument in std::env::args_os() {
        match argument.into_string() {
            Ok(argument) => arguments.push(argument),
            Err(_) => {
                eprintln!("cargo-gust: an argument is not valid UTF-8");
                return ExitCode::from(2);
            }
        }
    }
    let command = match cli::parse(&cli::strip_shim(&arguments)) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("cargo-gust: {error}\n\n{}", cli::USAGE);
            return ExitCode::from(2);
        }
    };
    let outcome = match command {
        cli::Command::Help => {
            print!("{}", cli::USAGE);
            Ok(0)
        }
        cli::Command::Version => {
            println!("cargo-gust {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        cli::Command::Init(target) => commands::init(&target),
        cli::Command::Build(options) => commands::build(&options),
        cli::Command::Check(options) => commands::check(&options),
        cli::Command::Run(options) => commands::cargo_with_routes("run", &options),
        cli::Command::Test(options) => commands::cargo_with_routes("test", &options),
        cli::Command::Doctor => commands::doctor(),
        cli::Command::Slang(command) => commands::slang(&command),
    };
    match outcome {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(2)),
        Err(error) => {
            eprintln!("cargo-gust: error: {error}");
            ExitCode::from(2)
        }
    }
}

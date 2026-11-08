# Release checklist: publishing the GUST crates to crates.io

This is the procedure for the first release (0.1.0) and the template for later ones. Publishing, tagging, the GitHub
release and yanking are the owner's acts (D115); the coordinator prepares everything up to a green gate and the
commands below. The crate names are the published ones (D48, D106): `gust-compute-macros`, `gust-compute`,
`gust-compute-wgpu` and `cargo-gust`. This is the release checklist of the roadmap row MR ([ROADMAP](../ROADMAP.md));
the decisions behind it are D106 to D115 in [DECISIONS](../DECISIONS.md).

## Who does what

The coordinator prepares, before step 0:

- the rename of the packages (mr.n) and the last code slice, merged, with the closing commit `<C>` recorded;
- `cargo xtask check-full` and the full `cargo xtask check-release` green on `<C>`, recorded in [STATUS](STATUS.md);
- `CHANGELOG.md` on `<C>`: the 0.1.0 heading reads `## 0.1.0` (no "not yet published") and its text matches the tree.

The owner does, in this order:

- step 0, once: require the checks `unit`, `msrv` and `release` on `main` (owner decision Q4; `slang` later);
- step 1: confirm the commit and the clean tree, and run the full gate again on that checkout;
- steps 2 to 4: `cargo login`, `cargo publish`, `cargo yank`;
- steps 5 and 6: the post-publish consumer (the coordinator may run it for the record), the tag, the GitHub release.

The coordinator then records the outcome (mr.z): docs.rs status (step 7), the report of step 8 in STATUS, the DECISIONS
rows and the ROADMAP row MR.

Never put a registry token in the repository, in a file the agents read, in a command line, or in a chat. `cargo login`
asks for it on the owner's terminal and stores it in the owner's cargo home.

## What the gate proves

`cargo xtask check-release` (D113) is the release gate. CI runs `--check-only` on every pull request and push (the job
`release`); `check-full` and the owner's step 1 run the full command. It works on the packages that `cargo metadata`
says are publishable, so it needs no edit when a package is renamed.

- R1: one `cargo package --locked` call for all four packages, with verification (cargo builds each unpacked package).
- R2: the file list of each package against the allow-list: `src/**` without test modules, `README.md`,
  `LICENSE-MIT`, `LICENSE-APACHE`, `build.rs` for `gust-compute`, `assets/**` for `cargo-gust`, no test data.
- R3 and R4: the metadata fields, one version for the four, every internal dependency exact (`=0.1.0`), and every
  other workspace member `publish = false`.
- R5: `cargo doc --no-deps` of the library crates with rustdoc warnings denied, on stable and, when a nightly
  toolchain is installed, with the docs.rs features and `--cfg docsrs`.
- R6: each `.crate` file under 2 MiB.
- R7: the README recipe ("Install from crates.io") built word for word from the unpacked `.crate` files, plus a
  `cargo install` of the unpacked `cargo-gust`. The full mode also runs `cargo gust build`, builds the release form
  and runs both on the GPU. `--check-only` prints what it did not run.
- R8: `cargo publish --dry-run --locked` for the four packages in one call.

It cannot prove that the crates.io index accepts the upload, that docs.rs builds the pages, or that the owner's token
has the right scope; steps 2, 3 and 7 are where those show.

## Steps

Run from a clean checkout of `<C>`, not from a worktree that lanes use. Step 1 needs the pinned Slang and the reflection
helper, and `cargo xtask` does not read the managed cache for either: the `slangc` of the pin must be first on `PATH`,
and the helper is the file `GUST_SLANG_REFLECT` names or the one `cargo xtask doctor` builds into `target/slang-reflect`
(through `cargo gust slang build-helper`), which needs `SLANG_SDK` (the managed `<cache>/<pin>` works) or
`VULKAN_SDK`, and a C++ compiler. The full `check-release` then uses that variable, or that built file.

```sh
# 0. On GitHub: branch protection on main requires the checks unit, msrv and release (owner decision Q4). Once.

# 1. Confirm the commit, the clean tree and the full gate (needs Slang, the helper, a GPU and the registry)
git status --short --branch            # clean, on main at <C>
cargo xtask doctor                     # checks the slangc and helper pins, builds a missing helper (no cargo-gust here)
cargo --locked xtask check-release     # the full gate: packaging, docs, dry run, the README recipe built and run

# 2. Log in once: a token from https://crates.io/settings/tokens, scope publish-update, for the four crates
cargo login

# 3. Publish in dependency order; cargo waits for the index between crates
cargo publish --locked -p gust-compute-macros
cargo publish --locked -p gust-compute
cargo publish --locked -p gust-compute-wgpu
cargo publish --locked -p cargo-gust
#    One call is possible too (cargo 1.90 or newer): cargo publish --locked --workspace. The per-crate form can be
#    stopped between crates, which is why it is the default.

# 4. Yank the 0.0.0 placeholders now that 0.1.0 is up (owner decision Q7)
cargo yank --version 0.0.0 gust-compute-macros
cargo yank --version 0.0.0 gust-compute
cargo yank --version 0.0.0 gust-compute-wgpu
cargo yank --version 0.0.0 cargo-gust
```

### 5. Post-publish verification

From an empty directory, on a machine with no checkout and no `slangc` on `PATH`, repeat the README as a user would.
This is the same recipe that R7 built from the packaged files, now from crates.io.

```sh
cargo install cargo-gust --locked --version "~0.1"
cargo gust slang install
cargo gust slang build-helper          # prints the helper path
cargo gust slang locate                # prints the slangc that cargo gust and the run-time compile will use
#    No environment variable is set: the run-time bridge (D109) finds the managed slangc and helper in the cache itself.
#    R7 exports GUST_SLANG_REFLECT and needs slangc on PATH, so this is the first run of that variable-free path: a
#    failure here is a report to the coordinator, not a surprise.
cargo new --bin hello-gust && cd hello-gust
#    replace Cargo.toml and src/main.rs with the README's two blocks of "Install from crates.io", word for word
cargo run --features gust-dev          # development path: run-time compile through the managed toolchain
cargo gust build -p hello-gust         # release path: writes gust-artifacts/
cargo build --release && ./target/release/hello-gust
cargo gust doctor
```

Both runs must print `hello-gust: [11.0, 22.0, 33.0, 44.0]`. Note the adapter that ran them. The two blocks are
the ones that R7 builds from the packaged files; if they do not build against the published crates, the gate and
the registry disagree, so stop and tell the coordinator.

```sh
# 6. Tag the published commit after all four crates are up, then the GitHub release (owner decision Q6; public acts)
git tag -a v0.1.0 -m "GUST 0.1.0: first public release"
git push origin v0.1.0
#    Create the GitHub release v0.1.0 with the text of the 0.1.0 entry of CHANGELOG.md.
```

7. docs.rs: <https://docs.rs/gust-compute>, <https://docs.rs/gust-compute-macros> and
   <https://docs.rs/gust-compute-wgpu> build green. `cargo-gust` is binary-only and has no docs.rs page; its
   `documentation` field points at its README. A docs.rs failure is a finding for the next patch release, not a reason
   to yank.
8. Report to the coordinator: the four versions, the yank, the tag, the required checks, the docs.rs status and the
   adapter of step 5. The coordinator records them in STATUS (mr.z).

## If a step fails

crates.io keeps every uploaded version forever: nothing published can be retracted, only yanked.

- A transient failure (network, an index delay) with no change needed in the repository:
  resume from the command that failed.
  Cargo refuses a version that is already up, so running the whole block again would stop at the first crate that is
  already published.
- A failure that needs a change (a packaging error, a broken crate): fix it on `release/0.1`, release 0.1.1 from
  there, then forward-port the fix to `main`. The four packages share one version and pin each other exactly (D107),
  so 0.1.1 is published for the remaining crates and for the ones already up, and the procedure restarts at step 1.
- Yank a version only when that crate is broken for users, and for the 0.0.0 placeholders. Never yank to fix a name or
  a metadata mistake; a patch release fixes those.

## Release lines

`main` is the development line (`0.2.0-dev`). 0.1.0 is not published yet (the owner's deferral, 2026-11-07) and there
is no `v0.1.0` tag; the 0.1 line is the branch `release/0.1`, and steps 1 to 8 for 0.1.0 run on its tip, not on `main`.
A 0.1.x fix is made on `release/0.1`, released from there and forward-ported to `main`.

## After 0.1.0

A later release repeats steps 1 to 8 with the new version: bump `[workspace.package]` and the three exact pins in
`[workspace.dependencies]` together, add the `CHANGELOG.md` entry, run the full gate on the commit, and publish in the
same order. A breaking change bumps the minor version of all four (D111).

### Releasing 0.2.0 from `main`

Besides the bump, change every `0.2.0-dev` requirement to the released form (a caret requirement does not match a
pre-release, so the development line names the pre-release and the release must not):

- the `toml` block of the README's "A first program" and `xtask/src/release/hello-gust/Cargo.toml` (the same text,
  which a unit test compares) go from `"0.2.0-dev"` to `"0.2"`, and the sentence under that block is reworded;
- the README's two install lines (`cargo install cargo-gust --locked --version 0.2.0-dev`, at "Try it" and "Set up the
  tools") become `--version "~0.2"`, and so do the `--version "~0.1"` lines of `crates/gust/README.md` (line 18) and
  `crates/cargo-gust/README.md` (line 11) and the doc comment `--version ~0.1` of `crates/cargo-gust/src/main.rs`
  (line 130); the note at the top of the README that names `0.2.0-dev` is removed. The install hint of
  `crates/cargo-gust/src/doctor.rs` and its test need no edit: since m8.x1 the hint derives `~0.x` from the package
  version (`CARGO_PKG_VERSION_MAJOR` and `_MINOR`), so the bump changes it;
- the README's version literals of 0.1.0: the renamed-dependency example in "Renamed dependencies and name clashes"
  (`version = "0.1"`, becomes `"0.2"`), the "Not supported in 0.1.0" bullet of "Platforms" and the "known limit of
  0.1.0" sentence of "Textures and samplers" (each says which release the statement was last checked for: re-check,
  then name 0.2.0), and the sentence of "Install from crates.io" that calls 0.1.0 the first release (true as it
  stands; read it once);
- `tests/precompiled/kernels/Cargo.toml`: the `version = "0.2.0-dev"` of its `gust` path dependency becomes `"0.2.0"`;
- `CHANGELOG.md`: "Unreleased (0.2.0)" becomes "0.2.0" and loses its sentence about the workspace version; once 0.1.0
  is published, the "(not yet published)" heading note and the deferral sentences of its entry are removed. The
  "Compatibility with 0.1.0" part (the `semver` report and the list of crates) is a record of one commit: rerun the job
  on the release commit and update its run id and counts;
- the `semver` job's baseline, as the next section says.

R7 cannot see most of this, with one exception: when the workspace version has no pre-release part, it fails if the
`hello-gust` fixture still names a pre-release requirement. That is the guard against a missed edit; search the
repository for `0.2.0-dev` and for `~0.1` as well.

### The semver baseline

The `semver` job of `.github/workflows/ci.yml` reports what a branch breaks against the line below it, for
`gust-compute` and `gust-compute-wgpu`. The baseline is the git ref `release/0.1` until a version of that line is
published (`--baseline-rev`); after that it is the latest published version of the line below `main` (the registry
version or its `v0.1.*` tag). When `main` is released as 0.2.0 and a later `main` is 0.3.0-dev, the baseline moves to
the 0.2 line in the same way. Move it in the commit that publishes, and keep the job's comment true. The job runs on
pull requests and on pushes to `main` only; on a release branch a patch release must not break anything, so a job added
there would use `--release-type patch` against the latest `vX.Y.*` tag and its findings would fail it.

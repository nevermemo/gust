//! `cargo xtask check-gallery`: the example gallery (`gallery/`, plan M10 section 3.1) in three parts. Every part that
//! runs is named in the output, the parts that do not run are named with the reason, and a missing tool or adapter
//! fails a part (nothing is skipped silently). With no flag all three run; `--slang`, `--no-slang` and `--gpu` select
//! parts explicitly (CI runs the first two in separate jobs).
//!
//! - **Slang part** (needs the pinned Slang and its helper): builds `cargo-gust`, copies the gallery to a scratch
//!   directory without the committed `gust-artifacts/`, runs `cargo gust build` there, `cargo gust check` on the fresh
//!   artifacts, and compares them byte for byte with the committed ones; `cargo gust check` also runs on the committed
//!   artifacts in place.
//! - **No-Slang part**: `cargo build --release` and `cargo test` in the environment of `check-precompiled` E1 (no
//!   variable that names Slang, no `PATH` folder with `slangc` or `gust-slang-reflect`, `GUST_SLANG_HOME` an empty
//!   directory so the default cache is not consulted), which is asserted Slang-free, including a lookup of `slangc` by the
//!   shell; the entry and the library crates are cleaned first, and `cargo tree -e features` must show no
//!   `slang-runtime`. The tests run once per entry (`cargo test -p <entry>`), and each entry's output is judged on its
//!   own ([`check_entry_tests`]): none failed, no ignored test but the ones its rule names, and at least its number of
//!   CPU-twin tests (a test whose name contains `twin`) reported `ok`, so an entry whose twin tests were ignored or
//!   deleted fails by name instead of hiding in a total. The children run without `GUST_PROFILE`. Then the anchor check of every gallery source and the guide-listing sync check (a packaged listing
//!   equals the anchored region of its gallery file, both with LF line ends).
//! - **GPU part**: `cargo run --release` per entry on an adapter; the output must contain the entry's expected line and an
//!   `adapter:` line, which is printed. It reuses the release build of the no-Slang part when that ran before in the same
//!   target directory; run alone (as `check-full` does) it builds with whatever the environment holds, and says nothing
//!   about Slang. An entry with a `profile_run` (`variants`) runs a second time under that `GUST_PROFILE`, and the output of
//!   that run must contain its own expected line; an entry with `via_driver` (`routed`) runs through `cargo gust run`.
//!   An entry with `browser` (`web-hello`) is a wasm32 page: the part builds it under `GUST_PROFILE=web`, runs
//!   `wasm-bindgen` and opens it in headless Chrome through the runner of `check-web` (one loopback server on an
//!   ephemeral port, the page's `GUST_OK` line read from Chrome's console). That needs Chrome, a `wasm-bindgen` CLI of
//!   the locked version and the wasm32 standard library, and the part fails when one is missing: it is never skipped.
//!
//! Children never inherit `GUST_PROFILE`, `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS`: a gallery crate is built the way a
//! user builds it (release is the strict `native` profile, tests are `dev`).

use std::{
    collections::BTreeSet,
    env,
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use crate::{
    precompiled::{StripReport, child_environment, directory_has_slang_tool, slang_free_problems},
    process::{exit_code, set_workspace_profile, without_colors},
    web::{Example, Options as WebOptions, evidence_label, run_example},
    workspace::{cargo_target_dir, workspace_root},
};

/// The gallery workspace, relative to the repository root. It is a workspace root of its own.
pub(crate) const GALLERY_DIR: &str = "gallery";

/// One gallery entry: its directory under [`GALLERY_DIR`] (also the package name) and the line its `main` prints.
pub(crate) struct Entry {
    pub(crate) dir: &'static str,
    pub(crate) expected_line: &'static str,
    /// A second GPU run of the entry under another `GUST_PROFILE`, for an entry whose behaviour depends on the profile
    /// (`variants` runs its subgroup body under `native+subgroups`): the profile and the line that run prints.
    pub(crate) profile_run: Option<ProfileRun>,
    /// Whether the GPU part runs the entry through `cargo gust run` (so its `gust.toml` decides the routes) instead of plain
    /// `cargo run` (`routed`).
    pub(crate) via_driver: bool,
    /// Whether the entry is a browser module (`web-hello`): a library built for `wasm32-unknown-unknown` with an
    /// `index.html`, no `main`. Its GPU run is the page, through the runner of `check-web` ([`crate::web::run_example`]),
    /// and `expected_line` is the last part of the page's `GUST_OK` line, not a line a `main` prints.
    pub(crate) browser: bool,
}

/// The second GPU run of an [`Entry`]: `cargo run --release` with `GUST_PROFILE` set to `profile`.
pub(crate) struct ProfileRun {
    pub(crate) profile: &'static str,
    pub(crate) expected_line: &'static str,
}

/// Every entry of the gallery. A unit test and the check itself compare this list with the `members` of
/// `gallery/Cargo.toml`, so an entry cannot be added to one and forgotten in the other.
pub(crate) const ENTRIES: [Entry; 10] = [
    Entry {
        dir: "hello-gust",
        expected_line: "hello-gust: [11.0, 22.0, 33.0, 44.0]",
        profile_run: None,
        via_driver: false,
        browser: false,
    },
    Entry {
        dir: "saxpy-uniform",
        expected_line: "saxpy-uniform: a=2 [12.0, 24.0, 36.0, 48.0], a=3 [13.0, 26.0, 39.0, 52.0]",
        profile_run: None,
        via_driver: false,
        browser: false,
    },
    Entry {
        dir: "histogram",
        expected_line: "histogram: 100000 keys in 8 bins [35376, 14623, 11255, 9474, 8348, 7546, 6933, 6445]",
        profile_run: None,
        via_driver: false,
        browser: false,
    },
    Entry {
        dir: "particles",
        expected_line: "particles: 1000 particles, 16 steps, 1129 bounces, energy 20272.84375",
        profile_run: None,
        via_driver: false,
        browser: false,
    },
    Entry {
        dir: "image-threshold",
        expected_line: "image-threshold: 37 x 21 texels, 366 bright by the pixel kernel, 366 by the grid kernel",
        profile_run: None,
        via_driver: false,
        browser: false,
    },
    Entry {
        dir: "reduce-scan",
        expected_line: "reduce-scan: 100000 values in 8 sizes, sum 450000, scan ends at 450000 with checksum 22499850000",
        profile_run: None,
        via_driver: false,
        browser: false,
    },
    Entry {
        dir: "state-machine",
        expected_line: "state-machine: 1000 records of 16 symbols, 1927 words (longest 7), 933 numbers (longest 7)",
        profile_run: None,
        via_driver: false,
        browser: false,
    },
    Entry {
        dir: "routed",
        expected_line: "routed: 1000 samples, tidy on Cpu, smooth on Gpu, checksum 691792",
        profile_run: None,
        via_driver: true,
        browser: false,
    },
    Entry {
        dir: "variants",
        expected_line: "variants: body=portable, 100000 values, sum 12499996, 59761 over 100, max 250",
        // The portable body runs under the default profile; the subgroup body is the build of `native+subgroups` (a strict
        // profile, so it builds from the committed artifacts, which hold the artifacts of both bodies).
        profile_run: Some(ProfileRun {
            profile: "native+subgroups",
            expected_line: "variants: body=subgroups, 100000 values, sum 12499996, 59761 over 100, max 250",
        }),
        via_driver: false,
        browser: false,
    },
    Entry {
        dir: "web-hello",
        // The last part of the page's `GUST_OK` line (`gallery/web-hello/src/lib.rs` logs it; a unit test finds it there).
        expected_line: WEB_HELLO.closing,
        profile_run: None,
        via_driver: false,
        browser: true,
    },
];

/// The browser page of the gallery: `web-hello` through the runner of `check-web` (plan M10 section 3.1, G9).
pub(crate) const WEB_HELLO: Example = Example {
    dir: "gallery/web-hello",
    workspace: GALLERY_DIR,
    package: "web-hello",
    out_name: "web_hello",
    manifest: "gallery/web-hello/gust-artifacts/manifest.json",
    closing: "map_async exact against the host reference",
    // A gallery crate is built the way a user builds it: no flag variables of the caller.
    cleared_env: &["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"],
};

/// A guide listing: a packaged file (path relative to the repository root) that must be byte-identical to the region
/// `anchor` of the gallery file `source` (also relative to the root). Each chapter of the guide adds its own.
pub(crate) struct Listing {
    pub(crate) packaged: &'static str,
    pub(crate) source: &'static str,
    pub(crate) anchor: &'static str,
}

/// The guide listings: chapter 2 of `gust::guide` includes G1's three regions from `crates/gust`, and the compile check of
/// the whole program in `gust_wgpu::guide` from its own copies in `crates/gust-wgpu` (a packaged crate cannot include a
/// file outside its directory). Chapters 3, 5 and 8 of `gust::guide` include the kernel region of G2, G3, G5 to G7 and G10
/// from `crates/gust`; chapters 4 to 6 of `gust_wgpu::guide` include the regions of the whole programs of G2, G4 to G6,
/// G8 and G10 from `crates/gust-wgpu`; chapter 9 includes the regions of G9 (`web-hello`) from `crates/gust-wgpu`.
pub(crate) const LISTINGS: [Listing; 43] = [
    Listing {
        packaged: "crates/gust/src/guide/listings/hello-gust-imports.rs",
        source: "gallery/hello-gust/src/main.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust/src/guide/listings/hello-gust-kernel.rs",
        source: "gallery/hello-gust/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust/src/guide/listings/hello-gust-main.rs",
        source: "gallery/hello-gust/src/main.rs",
        anchor: "main",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/hello-gust-imports.rs",
        source: "gallery/hello-gust/src/main.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/hello-gust-kernel.rs",
        source: "gallery/hello-gust/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/hello-gust-main.rs",
        source: "gallery/hello-gust/src/main.rs",
        anchor: "main",
    },
    Listing {
        packaged: "crates/gust/src/guide/listings/saxpy-kernel.rs",
        source: "gallery/saxpy-uniform/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust/src/guide/listings/histogram-kernel.rs",
        source: "gallery/histogram/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust/src/guide/listings/image-threshold-kernel.rs",
        source: "gallery/image-threshold/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust/src/guide/listings/reduce-scan-kernel.rs",
        source: "gallery/reduce-scan/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust/src/guide/listings/variants-kernel.rs",
        source: "gallery/variants/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust/src/guide/listings/state-machine-kernel.rs",
        source: "gallery/state-machine/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/saxpy-imports.rs",
        source: "gallery/saxpy-uniform/src/main.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/saxpy-kernel.rs",
        source: "gallery/saxpy-uniform/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/saxpy-params.rs",
        source: "gallery/saxpy-uniform/src/main.rs",
        anchor: "params",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/saxpy-main.rs",
        source: "gallery/saxpy-uniform/src/main.rs",
        anchor: "main",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/particles-imports.rs",
        source: "gallery/particles/src/main.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/particles-kernel.rs",
        source: "gallery/particles/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/particles-setup.rs",
        source: "gallery/particles/src/main.rs",
        anchor: "setup",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/particles-energy.rs",
        source: "gallery/particles/src/main.rs",
        anchor: "energy",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/particles-total.rs",
        source: "gallery/particles/src/main.rs",
        anchor: "total",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/particles-host.rs",
        source: "gallery/particles/src/main.rs",
        anchor: "host",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/particles-main.rs",
        source: "gallery/particles/src/main.rs",
        anchor: "main",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/image-threshold-imports.rs",
        source: "gallery/image-threshold/src/main.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/image-threshold-kernel.rs",
        source: "gallery/image-threshold/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/image-threshold-host.rs",
        source: "gallery/image-threshold/src/main.rs",
        anchor: "host",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/image-threshold-main.rs",
        source: "gallery/image-threshold/src/main.rs",
        anchor: "main",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/reduce-scan-imports.rs",
        source: "gallery/reduce-scan/src/main.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/reduce-scan-kernel.rs",
        source: "gallery/reduce-scan/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/reduce-scan-host.rs",
        source: "gallery/reduce-scan/src/main.rs",
        anchor: "host",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/reduce-scan-main.rs",
        source: "gallery/reduce-scan/src/main.rs",
        anchor: "main",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/variants-imports.rs",
        source: "gallery/variants/src/main.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/variants-kernel.rs",
        source: "gallery/variants/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/variants-host.rs",
        source: "gallery/variants/src/main.rs",
        anchor: "host",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/variants-main.rs",
        source: "gallery/variants/src/main.rs",
        anchor: "main",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/routed-imports.rs",
        source: "gallery/routed/src/main.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/routed-kernel.rs",
        source: "gallery/routed/src/main.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/routed-host.rs",
        source: "gallery/routed/src/main.rs",
        anchor: "host",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/routed-main.rs",
        source: "gallery/routed/src/main.rs",
        anchor: "main",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/web-hello-imports.rs",
        source: "gallery/web-hello/src/lib.rs",
        anchor: "imports",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/web-hello-kernel.rs",
        source: "gallery/web-hello/src/lib.rs",
        anchor: "kernel",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/web-hello-reference.rs",
        source: "gallery/web-hello/src/lib.rs",
        anchor: "reference",
    },
    Listing {
        packaged: "crates/gust-wgpu/src/guide/listings/web-hello-map-async.rs",
        source: "gallery/web-hello/src/lib.rs",
        anchor: "map-async",
    },
];

/// The variables removed from every gallery child (see the module comment).
const CLEARED: [&str; 3] = ["GUST_PROFILE", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"];

/// The parts of the check that run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Parts {
    pub(crate) slang: bool,
    pub(crate) no_slang: bool,
    pub(crate) gpu: bool,
}

impl Parts {
    pub(crate) const ALL: Self = Self {
        slang: true,
        no_slang: true,
        gpu: true,
    };

    /// The parts the arguments select: none given means all three; `--slang`, `--no-slang` and `--gpu` select exactly
    /// those named (a repeated flag is the same as one). Anything else is an error.
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        if args.is_empty() {
            return Ok(Self::ALL);
        }
        let mut parts = Self {
            slang: false,
            no_slang: false,
            gpu: false,
        };
        for argument in args {
            match argument.as_str() {
                "--slang" => parts.slang = true,
                "--no-slang" => parts.no_slang = true,
                "--gpu" => parts.gpu = true,
                other => {
                    return Err(format!(
                        "check-gallery: unknown argument `{other}` (it takes --slang, --no-slang and --gpu; none means all three)"
                    ));
                }
            }
        }
        Ok(parts)
    }

    /// The names of the parts that run, in the order they run.
    pub(crate) fn run_names(self) -> Vec<&'static str> {
        [
            (self.slang, "slang"),
            (self.no_slang, "no-slang"),
            (self.gpu, "gpu"),
        ]
        .into_iter()
        .filter_map(|(selected, name)| selected.then_some(name))
        .collect()
    }

    /// The names of the parts that do not run.
    pub(crate) fn not_run_names(self) -> Vec<&'static str> {
        [
            (self.slang, "slang"),
            (self.no_slang, "no-slang"),
            (self.gpu, "gpu"),
        ]
        .into_iter()
        .filter_map(|(selected, name)| (!selected).then_some(name))
        .collect()
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Anchors and listings

const ANCHOR_START: &str = "// ANCHOR:";
const ANCHOR_END: &str = "// ANCHOR_END:";

/// The anchor names of `text`, in file order, after checking that the anchors are well formed: every `// ANCHOR: name`
/// has exactly one later `// ANCHOR_END: name`, a name is used once, an end has a start, and a name is not empty or
/// padded. Regions may sit beside each other or inside one another (different names).
pub(crate) fn anchor_names(text: &str) -> Result<Vec<String>, String> {
    let mut open: Vec<String> = Vec::new();
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        let line = line.trim();
        if let Some(name) = line.strip_prefix(ANCHOR_START) {
            let name = anchor_argument(name, number)?;
            if !seen.insert(name.clone()) {
                return Err(format!("line {number}: anchor `{name}` is declared twice"));
            }
            open.push(name.clone());
            order.push(name);
        } else if let Some(name) = line.strip_prefix(ANCHOR_END) {
            let name = anchor_argument(name, number)?;
            match open.iter().position(|candidate| *candidate == name) {
                Some(position) => {
                    open.remove(position);
                }
                None => {
                    return Err(format!(
                        "line {number}: ANCHOR_END `{name}` has no open ANCHOR (missing, closed twice, or before its start)"
                    ));
                }
            }
        }
    }
    if let Some(name) = open.first() {
        return Err(format!("anchor `{name}` is never closed"));
    }
    Ok(order)
}

fn anchor_argument(rest: &str, number: usize) -> Result<String, String> {
    // One space after the colon, then the name: `// ANCHOR:name` would not match the line the extraction looks for.
    let name = if rest.is_empty() {
        rest
    } else {
        rest.strip_prefix(' ').ok_or_else(|| {
            format!(
                "line {number}: write one space after the colon of an anchor line (got `{rest}`)"
            )
        })?
    };
    if name.is_empty() || name != name.trim() || name.contains(char::is_whitespace) {
        return Err(format!(
            "line {number}: an anchor name is one word with no spaces (got `{name}`)"
        ));
    }
    Ok(name.to_owned())
}

/// The lines strictly between `// ANCHOR: name` and `// ANCHOR_END: name` of `text`, each ended by `\n` (line ends of
/// the source are normalised, so a CRLF checkout gives the same region). Other anchor lines inside the region are
/// removed, so nested regions give the same text as the same lines outside. Fails when the anchors of `text` are not
/// well formed ([`anchor_names`]) or `name` is not among them.
pub(crate) fn anchored_region(text: &str, name: &str) -> Result<String, String> {
    if !anchor_names(text)?.iter().any(|found| found == name) {
        return Err(format!("no anchor `{name}`"));
    }
    let mut inside = false;
    let mut region = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == format!("{ANCHOR_START} {name}") {
            inside = true;
        } else if trimmed == format!("{ANCHOR_END} {name}") {
            inside = false;
        } else if inside && !trimmed.starts_with(ANCHOR_START) && !trimmed.starts_with(ANCHOR_END) {
            region.push_str(line.trim_end_matches('\r'));
            region.push('\n');
        }
    }
    Ok(region)
}

/// `bytes` with every CRLF replaced by LF (a lone CR stays).
pub(crate) fn lf_line_ends(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
            continue;
        }
        out.push(*byte);
    }
    out
}

/// Whether a packaged listing equals the anchored region. Both sides have LF line ends before the comparison (the
/// region already has them; the packaged file is normalised here), so a checkout with `core.autocrlf=true` gives the
/// same verdict as one without; every other byte must be the same.
pub(crate) fn listing_matches(packaged: &[u8], region: &str) -> bool {
    lf_line_ends(packaged) == lf_line_ends(region.as_bytes())
}

/// Checks every listing of `listings` against its gallery file under `root`; returns how many were checked.
pub(crate) fn check_listings(root: &Path, listings: &[Listing]) -> Result<usize, String> {
    for listing in listings {
        let source = fs::read_to_string(root.join(listing.source))
            .map_err(|error| format!("listing source {}: {error}", listing.source))?;
        let region = anchored_region(&source, listing.anchor)
            .map_err(|error| format!("{}: {error}", listing.source))?;
        let packaged = fs::read(root.join(listing.packaged))
            .map_err(|error| format!("packaged listing {}: {error}", listing.packaged))?;
        if !listing_matches(&packaged, &region) {
            return Err(format!(
                "packaged listing {} is not byte-identical to region `{}` of {}; copy the region (the lines between its ANCHOR lines) into the packaged file",
                listing.packaged, listing.anchor, listing.source
            ));
        }
    }
    Ok(listings.len())
}

/// The Rust files under `directory`, recursively (never `target`), sorted.
fn rust_files(directory: &Path, found: &mut Vec<PathBuf>) -> Result<(), String> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    entries.sort();
    for path in entries {
        if path.is_dir() {
            if path.file_name() != Some(OsStr::new("target")) {
                rust_files(&path, found)?;
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
    Ok(())
}

/// Checks that the anchors of every gallery source file are well formed; returns the files and anchors counted.
fn check_gallery_anchors(root: &Path) -> Result<(usize, usize), String> {
    let mut files = Vec::new();
    for entry in &ENTRIES {
        rust_files(&root.join(GALLERY_DIR).join(entry.dir), &mut files)?;
    }
    let mut anchors = 0;
    for file in &files {
        let text = fs::read_to_string(file)
            .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
        let relative = file.strip_prefix(root).unwrap_or(file);
        anchors += anchor_names(&text)
            .map_err(|error| format!("{}: {error}", relative.display()))?
            .len();
    }
    Ok((files.len(), anchors))
}

// ---------------------------------------------------------------------------------------------------------------------
// The workspace and the list of entries

/// The quoted names of the `members = [...]` array of a gallery `Cargo.toml`.
pub(crate) fn workspace_members(manifest: &str) -> Result<Vec<String>, String> {
    let rest = manifest
        .lines()
        .skip_while(|line| !line.trim_start().starts_with("members"))
        .collect::<Vec<_>>()
        .join("\n");
    let list = rest
        .split_once('[')
        .and_then(|(_, after)| after.split_once(']'))
        .map(|(list, _)| list)
        .ok_or("gallery/Cargo.toml has no `members = [...]` array")?;
    let mut members = Vec::new();
    for line in list.lines() {
        let line = line.split('#').next().unwrap_or_default();
        members.extend(line.split('"').skip(1).step_by(2).map(str::to_owned));
    }
    Ok(members)
}

/// The files an entry must have (relative to its directory): a program has a `main`, a browser module (`web-hello`) a
/// library and the page that loads it.
pub(crate) fn required_files(entry: &Entry) -> &'static [&'static str] {
    if entry.browser {
        &["Cargo.toml", "README.md", "src/lib.rs", "index.html"]
    } else {
        &["Cargo.toml", "README.md", "src/main.rs"]
    }
}

/// Fails unless [`ENTRIES`] and the workspace members are the same set, and every entry has its files.
fn check_entries(root: &Path) -> Result<(), String> {
    let manifest = fs::read_to_string(root.join(GALLERY_DIR).join("Cargo.toml"))
        .map_err(|error| format!("cannot read gallery/Cargo.toml: {error}"))?;
    let members = workspace_members(&manifest)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    let listed = ENTRIES
        .iter()
        .map(|entry| entry.dir.to_owned())
        .collect::<BTreeSet<_>>();
    if members != listed {
        return Err(format!(
            "gallery/Cargo.toml members {members:?} and the ENTRIES of xtask/src/gallery.rs {listed:?} differ"
        ));
    }
    for entry in &ENTRIES {
        for file in required_files(entry) {
            let path = root.join(GALLERY_DIR).join(entry.dir).join(file);
            if !path.is_file() {
                return Err(format!("gallery/{}/{file} is missing", entry.dir));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------
// The environment of the children

/// Tools that must stay reachable for a build: the cargo and rust toolchain and a linker. A `PATH` folder that holds
/// one of them and also a Slang tool cannot be dropped without breaking the build.
const BUILD_TOOLS: [&str; 8] = [
    "cargo", "rustc", "cc", "gcc", "clang", "link", "ld", "rust-lld",
];

/// Whether `folder` holds the file `name`, with or without the executable suffix.
pub(crate) fn folder_holds(folder: &Path, name: &str) -> bool {
    folder.join(name).is_file() || folder.join(format!("{name}.exe")).is_file()
}

/// The problems of dropping `dropped` folders from `PATH`: a folder that also holds a build tool (`holds` says
/// whether a folder holds a file). Dropping `/usr/bin` because a `slangc` lives there would leave a build that cannot
/// find its linker, so the part refuses instead of running it.
pub(crate) fn shared_folder_problems(
    dropped: &[PathBuf],
    holds: &dyn Fn(&Path, &str) -> bool,
) -> Vec<String> {
    let mut problems = Vec::new();
    for folder in dropped {
        for tool in BUILD_TOOLS {
            if holds(folder, tool) {
                problems.push(format!(
                    "the PATH folder {} holds a Slang tool and also `{tool}`: dropping it would leave a build that cannot find `{tool}`; move the Slang tools to a folder of their own and run again",
                    folder.display()
                ));
            }
        }
    }
    problems
}

/// A problem with the directory that `GUST_SLANG_HOME` will name: it must be an existing, empty directory, so that
/// the default cache (`<CARGO_HOME>/gust/slang`) is not consulted and nothing is found in its place.
pub(crate) fn slang_home_problem(directory: &Path) -> Option<String> {
    match fs::read_dir(directory) {
        Err(error) => Some(format!(
            "the scratch Slang home {} cannot be read: {error}",
            directory.display()
        )),
        Ok(mut entries) => entries.next().map(|_| {
            format!(
                "the scratch Slang home {} is not empty",
                directory.display()
            )
        }),
    }
}

/// Whether the shell of the platform finds `slangc` with the `PATH` of `environment`: `Ok(())` when it does not. The
/// logic of `slangc_cannot_be_found` in `precompiled.rs` (private there, and that file is not this slice's), run
/// through a shell because `Command::new("slangc")` would fall back to the parent's `PATH` on Windows.
/// On Unix the shell is `/bin/sh` by absolute path, because std would look up a bare `sh` in the child's own (stripped)
/// `PATH`.
fn slangc_cannot_be_found(environment: &[(OsString, OsString)]) -> Result<(), String> {
    let mut probe = if cfg!(windows) {
        let mut probe = Command::new("cmd");
        probe.args(["/d", "/c", "where", "slangc"]);
        probe
    } else {
        let mut probe = Command::new("/bin/sh");
        probe.args(["-c", "command -v slangc"]);
        probe
    };
    probe
        .env_clear()
        .envs(environment.iter().map(|(name, value)| (name, value)))
        .stdin(Stdio::null());
    let output = probe
        .output()
        .map_err(|error| format!("could not run the lookup of `slangc`: {error}"))?;
    let code = output.status.code();
    // `where` answers 1 when it finds nothing and 2 for an error; `command -v` answers non-zero when it finds nothing.
    let absent = if cfg!(windows) {
        code == Some(1)
    } else {
        !output.status.success()
    };
    if absent {
        return Ok(());
    }
    if output.status.success() {
        return Err(format!(
            "`slangc` is found with the stripped environment: {}",
            String::from_utf8_lossy(&output.stdout).trim()
        ));
    }
    Err(format!(
        "the lookup of `slangc` failed (exit {}): {}",
        exit_code(output.status),
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

/// The variables of a child, as `(name, value)` pairs.
pub(crate) type Environment = [(OsString, OsString)];

/// The environment of the no-Slang children: the parent's, stripped as `check-precompiled` E1 strips it
/// (`child_environment`: no variable that names Slang, `VULKAN_SDK`, `VK_SDK_PATH`, `GUST_EXPORT_DIR`, no `PATH` folder
/// that holds `slangc` or `gust-slang-reflect`, `GUST_PROFILE=native`), then asserted Slang-free by
/// `slang_free_problems` and by `find` (an independent lookup of `slangc` with the child's `PATH`), and only then
/// given `GUST_SLANG_HOME`, the empty directory `home`, which keeps the toolchain away from the default cache.
/// `has_tool`, `holds` and `find` are parameters so that a test can make each check fail. Returns the environment and
/// what was stripped.
pub(crate) fn no_slang_environment(
    parent: &[(OsString, OsString)],
    home: &Path,
    has_tool: &dyn Fn(&Path) -> bool,
    holds: &dyn Fn(&Path, &str) -> bool,
    find: &dyn Fn(&Environment) -> Result<(), String>,
) -> Result<(Vec<(OsString, OsString)>, StripReport), String> {
    let mut report = StripReport::default();
    let mut environment = child_environment(parent, has_tool, &mut report);
    let mut problems = shared_folder_problems(&report.directories, holds);
    problems.extend(slang_free_problems(&environment, has_tool));
    problems.extend(slang_home_problem(home));
    if let Err(error) = find(&environment) {
        problems.push(error);
    }
    if !problems.is_empty() {
        return Err(format!(
            "the environment of the no-Slang children is not Slang-free: {}",
            problems.join("; ")
        ));
    }
    environment.push((
        OsString::from("GUST_SLANG_HOME"),
        home.as_os_str().to_owned(),
    ));
    Ok((environment, report))
}

/// The cargo to run: `$CARGO` when set (as `cargo xtask` does), else the first `cargo` of the current `PATH`, resolved
/// to a full path before the `PATH` of a child is changed.
fn cargo_program() -> Result<PathBuf, String> {
    if let Some(cargo) = env::var_os("CARGO").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(cargo));
    }
    let current = env::var_os("PATH").unwrap_or_default();
    for folder in env::split_paths(&current) {
        for name in ["cargo", "cargo.exe"] {
            let candidate = folder.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err("cargo was not found on PATH".to_owned())
}

/// The target directory of the gallery builds: one folder beside the other xtask scratch output, away from the
/// repository workspace's own build and not tracked by git. The no-Slang part and the GPU part share it.
fn gallery_target(root: &Path, name: &str) -> PathBuf {
    root.join(cargo_target_dir(root)).join("gallery").join(name)
}

struct Done {
    success: bool,
    code: String,
    stdout: String,
    stderr: String,
}

impl Done {
    fn everything(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

/// The last `count` lines of `text`.
fn tail(text: &str, count: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    lines[lines.len().saturating_sub(count)..].join("\n")
}

fn exec(mut command: Command, label: &str) -> Result<Done, String> {
    println!("  $ {label}");
    let output = command
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("could not launch {label}: {error}"))?;
    Ok(Done {
        success: output.status.success(),
        code: exit_code(output.status),
        stdout: without_colors(&String::from_utf8_lossy(&output.stdout)),
        stderr: without_colors(&String::from_utf8_lossy(&output.stderr)),
    })
}

fn require(done: Done, label: &str) -> Result<Done, String> {
    if done.success {
        Ok(done)
    } else {
        Err(format!(
            "{label} failed (exit {}):\n{}",
            done.code,
            tail(&done.everything(), 30)
        ))
    }
}

/// A cargo command in `directory` that builds a gallery crate the way a user does: no profile or flag variables, a
/// target directory of its own, no colour.
fn gallery_cargo(directory: &Path, target: &Path) -> Result<Command, String> {
    let mut command = Command::new(cargo_program()?);
    command
        .current_dir(directory)
        .env("CARGO_TARGET_DIR", target)
        .env("CARGO_TERM_COLOR", "never");
    for name in CLEARED {
        command.env_remove(name);
    }
    Ok(command)
}

// ---------------------------------------------------------------------------------------------------------------------
// The parts

/// Copies `source` to `destination` without `target` folders and without the committed `gust-artifacts/` of the
/// entries; `../../crates/` in a manifest becomes the absolute path of the repository's `crates/` so that the copy,
/// which sits elsewhere, finds the library crates.
fn copy_gallery(root: &Path, source: &Path, destination: &Path) -> Result<(), String> {
    let crates = format!("{}/crates/", root.to_string_lossy().replace('\\', "/"));
    copy_tree(source, destination, &crates)
}

fn copy_tree(source: &Path, destination: &Path, crates: &str) -> Result<(), String> {
    fs::create_dir_all(destination)
        .map_err(|error| format!("cannot create {}: {error}", destination.display()))?;
    for entry in fs::read_dir(source)
        .map_err(|error| format!("cannot read {}: {error}", source.display()))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name();
        let from = entry.path();
        let to = destination.join(&name);
        if from.is_dir() {
            if name == "target" || name == "gust-artifacts" {
                continue;
            }
            copy_tree(&from, &to, crates)?;
        } else if name == "Cargo.toml" {
            let text = fs::read_to_string(&from)
                .map_err(|error| format!("cannot read {}: {error}", from.display()))?;
            fs::write(&to, text.replace("../../crates/", crates))
                .map_err(|error| format!("cannot write {}: {error}", to.display()))?;
        } else {
            fs::copy(&from, &to)
                .map_err(|error| format!("cannot copy {}: {error}", from.display()))?;
        }
    }
    Ok(())
}

/// The files under `directory` as `(relative path with /, bytes)`, sorted.
fn read_tree(directory: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    fn walk(base: &Path, directory: &Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<(), String> {
        for entry in fs::read_dir(directory)
            .map_err(|error| format!("cannot read {}: {error}", directory.display()))?
        {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.is_dir() {
                walk(base, &path, out)?;
            } else {
                let relative = path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                let bytes = fs::read(&path)
                    .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
                out.push((relative, bytes));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(directory, directory, &mut files)?;
    files.sort();
    Ok(files)
}

/// The first difference between two artifact trees, or `None` when they are the same files with the same bytes.
pub(crate) fn tree_difference(
    committed: &[(String, Vec<u8>)],
    fresh: &[(String, Vec<u8>)],
) -> Option<String> {
    // A driver output (`routes.json`, written by `cargo gust build` and `run` for the profile of the run) is not an artifact:
    // cargo-gust's own check skips it too.
    let artifacts = |files: &[(String, Vec<u8>)]| -> Vec<(String, Vec<u8>)> {
        files
            .iter()
            .filter(|(name, _)| name.rsplit('/').next() != Some("routes.json"))
            .cloned()
            .collect()
    };
    let (committed, fresh) = (&artifacts(committed), &artifacts(fresh));
    let names = |files: &[(String, Vec<u8>)]| {
        files
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>()
    };
    if names(committed) != names(fresh) {
        return Some(format!(
            "the file lists differ: committed {:?}, fresh {:?}",
            names(committed),
            names(fresh)
        ));
    }
    committed
        .iter()
        .zip(fresh)
        .find(|(left, right)| left.1 != right.1)
        .map(|(left, _)| format!("{} differs from the fresh build", left.0))
}

/// Builds `cargo-gust` in the repository workspace and returns the binary.
fn build_driver(root: &Path, cargo: &Path) -> Result<PathBuf, String> {
    let mut build = Command::new(cargo);
    build
        .current_dir(root)
        .args(["build", "--locked", "--color", "never", "-p", "cargo-gust"]);
    set_workspace_profile(&mut build);
    require(
        exec(build, "cargo build -p cargo-gust")?,
        "cargo build -p cargo-gust",
    )?;
    let binary = root
        .join(cargo_target_dir(root))
        .join("debug")
        .join(format!("cargo-gust{}", env::consts::EXE_SUFFIX));
    if binary.is_file() {
        Ok(binary)
    } else {
        Err(format!(
            "cargo built cargo-gust, but {} does not exist",
            binary.display()
        ))
    }
}

/// The Slang part.
fn slang_part(root: &Path) -> Result<String, String> {
    let cargo = cargo_program()?;
    let driver = build_driver(root, &cargo)?;
    let gallery = root.join(GALLERY_DIR);
    let scratch = root
        .join(cargo_target_dir(root))
        .join("gallery")
        .join("scratch");
    if scratch.exists() {
        fs::remove_dir_all(&scratch)
            .map_err(|error| format!("cannot remove {}: {error}", scratch.display()))?;
    }
    copy_gallery(root, &gallery, &scratch)?;
    let target = gallery_target(root, "slang");
    let cargo_path = cargo_program()?;
    let driver_command = |directory: &Path, verb: &str, entry: &Entry| -> Command {
        let mut command = Command::new(&driver);
        command
            .current_dir(directory)
            .env("CARGO_TARGET_DIR", &target)
            .env("CARGO_TERM_COLOR", "never")
            .env("CARGO", &cargo_path)
            .args(["gust", verb, "-p", entry.dir]);
        for name in CLEARED {
            command.env_remove(name);
        }
        command
    };
    let mut lines = Vec::new();
    for entry in &ENTRIES {
        // The committed artifacts, in place, against the sources.
        let label = format!("cargo gust check -p {} (committed artifacts)", entry.dir);
        require(
            exec(driver_command(&gallery, "check", entry), &label)?,
            &label,
        )?;
        // A fresh build in the scratch copy (which has no artifacts), checked, then compared byte for byte.
        let label = format!("cargo gust build -p {} (scratch copy)", entry.dir);
        require(
            exec(driver_command(&scratch, "build", entry), &label)?,
            &label,
        )?;
        let label = format!("cargo gust check -p {} (fresh artifacts)", entry.dir);
        require(
            exec(driver_command(&scratch, "check", entry), &label)?,
            &label,
        )?;
        let committed = read_tree(&gallery.join(entry.dir).join("gust-artifacts"))?;
        let fresh = read_tree(&scratch.join(entry.dir).join("gust-artifacts"))?;
        if committed.is_empty() {
            return Err(format!(
                "gallery/{}/gust-artifacts is empty or missing",
                entry.dir
            ));
        }
        if let Some(difference) = tree_difference(&committed, &fresh) {
            return Err(format!(
                "gallery/{}/gust-artifacts: {difference}; run `cargo gust build -p {}` in gallery/ and commit the result",
                entry.dir, entry.dir
            ));
        }
        lines.push(format!(
            "{}: {} artifact file(s) byte-identical to a fresh build",
            entry.dir,
            committed.len()
        ));
    }
    Ok(lines.join("\n"))
}

/// A cargo command for the no-Slang part: only `environment`, with the target directory and no colour.
fn no_slang_cargo(
    directory: &Path,
    target: &Path,
    environment: &[(OsString, OsString)],
) -> Result<Command, String> {
    let mut command = Command::new(cargo_program()?);
    command
        .env_clear()
        .envs(environment.iter().map(|(name, value)| (name, value)))
        .current_dir(directory)
        .env("CARGO_TARGET_DIR", target)
        .env("CARGO_TERM_COLOR", "never");
    Ok(command)
}

/// Whether the output of `cargo tree -e features` shows the run-time Slang bridge.
pub(crate) fn tree_has_slang_runtime(output: &str) -> bool {
    output.contains("slang-runtime")
}

/// The no-Slang part.
fn no_slang_part(root: &Path) -> Result<String, String> {
    let gallery = root.join(GALLERY_DIR);
    let target = gallery_target(root, "no-slang");
    let home = root
        .join(cargo_target_dir(root))
        .join("gallery")
        .join("empty-slang-home");
    if home.exists() {
        fs::remove_dir_all(&home)
            .map_err(|error| format!("cannot remove {}: {error}", home.display()))?;
    }
    fs::create_dir_all(&home)
        .map_err(|error| format!("cannot create {}: {error}", home.display()))?;
    let parent = env::vars_os().collect::<Vec<_>>();
    let (checked_environment, report) = no_slang_environment(
        &parent,
        &home,
        &directory_has_slang_tool,
        &folder_holds,
        &slangc_cannot_be_found,
    )?;
    // The environment was checked with `GUST_PROFILE=native`; every child runs without the `CLEARED` variables (a plain
    // `cargo build --release`, as the README says), so the GPU part reuses this very build.
    let environment = checked_environment
        .iter()
        .filter(|(name, _)| {
            !CLEARED
                .iter()
                .any(|cleared| name.to_string_lossy().eq_ignore_ascii_case(cleared))
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut lines = vec![format!(
        "Slang absent: {} variable(s) and {} PATH folder(s) taken out, the environment checked Slang-free and `slangc` not found by the shell; GUST_SLANG_HOME is an empty directory",
        report.variables.len(),
        report.directories.len()
    )];
    // A build that an earlier run made with Slang present must not stand in for this one: clean the entries and the
    // library crates whose build scripts and macros read the environment (the E1 list of `check-precompiled`).
    let mut clean = no_slang_cargo(&gallery, &target, &environment)?;
    clean.args(["clean", "--release"]);
    for entry in &ENTRIES {
        clean.args(["-p", entry.dir]);
    }
    clean.args(["-p", "gust-compute", "-p", "gust-compute-macros"]);
    require(
        exec(
            clean,
            "cargo clean --release -p <entries> -p gust-compute -p gust-compute-macros",
        )?,
        "cargo clean",
    )?;
    let mut build = no_slang_cargo(&gallery, &target, &environment)?;
    build.args(["build", "--release", "--locked", "--workspace"]);
    require(
        exec(build, "cargo build --release --locked (no Slang)")?,
        "cargo build --release",
    )?;
    lines.push("cargo build --release: ok".to_owned());
    for entry in &ENTRIES {
        let mut tree = no_slang_cargo(&gallery, &target, &environment)?;
        tree.args(["tree", "--locked", "-e", "features", "-p", entry.dir]);
        let label = format!("cargo tree -e features -p {}", entry.dir);
        let done = require(exec(tree, &label)?, &label)?;
        if tree_has_slang_runtime(&done.stdout) {
            return Err(format!(
                "{label} shows the `slang-runtime` feature in a build without `gust-dev`"
            ));
        }
        lines.push(format!(
            "{}: no slang-runtime in the release build",
            entry.dir
        ));
    }
    // One call per entry, so that each entry's tests are judged on their own (a total would let one entry lose its twin
    // tests while the others still pass).
    let mut total = 0;
    for entry in &ENTRIES {
        let mut test = no_slang_cargo(&gallery, &target, &environment)?;
        test.args(["test", "--locked", "-p", entry.dir]);
        let label = format!("cargo test --locked -p {} (no Slang)", entry.dir);
        let done = require(exec(test, &label)?, &label)?;
        let passed = crate::process::sum_test_results(&done.stdout);
        if passed.passed == 0 {
            return Err(format!(
                "cargo test ran no test in gallery entry `{}`",
                entry.dir
            ));
        }
        let twins = check_entry_tests(entry.dir, &done.stdout)?;
        total += passed.passed;
        lines.push(format!(
            "{}: {} test(s) passed, {twins} of them CPU-twin tests",
            entry.dir, passed.passed
        ));
    }
    lines.push(format!("cargo test: {total} test(s) passed"));
    let (files, anchors) = check_gallery_anchors(root)?;
    lines.push(format!(
        "anchors: {anchors} in {files} source file(s), all well formed"
    ));
    let listings = check_listings(root, &LISTINGS)?;
    lines.push(format!(
        "guide listings: {listings} listings, all byte-identical to their anchored regions"
    ));
    Ok(lines.join("\n"))
}

/// What the no-Slang part requires of one entry's tests (`cargo test -p <dir>`): the least number of CPU-twin tests (a
/// test whose name contains `twin`) that must run and pass, and the exact number of ignored tests. The gallery promises
/// that every entry's program is checked against a CPU twin without a GPU, so an entry may not lose its twin tests,
/// whatever the total says. The one ignored test of the gallery is the GPU test of `reduce-scan` (it needs an adapter
/// and runs in the GPU part). A new twin test raises nothing; a new entry needs a rule (a unit test compares the list
/// with [`ENTRIES`]).
pub(crate) struct TwinRule {
    pub(crate) dir: &'static str,
    pub(crate) min_twin: usize,
    pub(crate) ignored: usize,
}

/// One [`TwinRule`] per entry of [`ENTRIES`], in the same order.
pub(crate) const TWIN_RULES: [TwinRule; 10] = [
    TwinRule {
        dir: "hello-gust",
        min_twin: 2,
        ignored: 0,
    },
    TwinRule {
        dir: "saxpy-uniform",
        min_twin: 3,
        ignored: 0,
    },
    TwinRule {
        dir: "histogram",
        min_twin: 2,
        ignored: 0,
    },
    TwinRule {
        dir: "particles",
        min_twin: 2,
        ignored: 0,
    },
    TwinRule {
        dir: "image-threshold",
        min_twin: 3,
        ignored: 0,
    },
    TwinRule {
        dir: "reduce-scan",
        min_twin: 1,
        ignored: 1,
    },
    TwinRule {
        dir: "state-machine",
        min_twin: 2,
        ignored: 0,
    },
    TwinRule {
        dir: "routed",
        min_twin: 2,
        ignored: 0,
    },
    TwinRule {
        dir: "variants",
        min_twin: 2,
        ignored: 0,
    },
    TwinRule {
        dir: "web-hello",
        min_twin: 2,
        ignored: 0,
    },
];

/// Judges the standard output of `cargo test -p <entry>` for one entry against its [`TwinRule`]: returns the number of
/// twin tests that passed, or an error that names the entry. Reads the `test NAME ... STATUS` lines of libtest.
pub(crate) fn check_entry_tests(entry: &str, output: &str) -> Result<usize, String> {
    let rule = TWIN_RULES
        .iter()
        .find(|rule| rule.dir == entry)
        .ok_or_else(|| {
            format!("gallery entry `{entry}` has no test rule (add one to TWIN_RULES)")
        })?;
    let (mut passed_twins, mut failed, mut ignored) = (0_usize, Vec::new(), Vec::new());
    for line in output.lines() {
        let Some(rest) = line.trim().strip_prefix("test ") else {
            continue;
        };
        let Some((name, status)) = rest.rsplit_once(" ... ") else {
            continue;
        };
        let status = status.trim();
        if status == "ok" {
            if name.contains("twin") {
                passed_twins += 1;
            }
        } else if status.starts_with("FAILED") {
            failed.push(name.to_owned());
        } else if status.starts_with("ignored") {
            ignored.push(name.to_owned());
        }
    }
    if !failed.is_empty() {
        return Err(format!(
            "gallery entry `{entry}`: test(s) failed: {}",
            failed.join(", ")
        ));
    }
    if ignored.len() != rule.ignored {
        return Err(format!(
            "gallery entry `{entry}`: {} ignored test(s) ({}), expected {}; its CPU-twin tests must run",
            ignored.len(),
            ignored.join(", "),
            rule.ignored
        ));
    }
    if passed_twins < rule.min_twin {
        return Err(format!(
            "gallery entry `{entry}`: {passed_twins} CPU-twin test(s) (a test whose name contains `twin`) ran and passed, \
             expected at least {}",
            rule.min_twin
        ));
    }
    Ok(passed_twins)
}

/// What an entry's output must contain: its expected line, as a whole line.
pub(crate) fn output_has_line(output: &str, expected: &str) -> bool {
    output.lines().any(|line| line.trim_end() == expected)
}

/// The GPU part: every entry's default run first, then the second runs under another profile (so the build scripts of
/// `gust` rebuild once for the profile, whatever the order of [`ENTRIES`]). An entry with `via_driver` runs through
/// `cargo gust run`, so its `gust.toml` is the route table of the run. A second run under `native+subgroups` needs an
/// adapter with subgroup operations: on one without them (or with a device that cannot be opened with them) the run
/// fails, and so does the part; it is never skipped. An entry with `browser` (`web-hello`) is not run by cargo: its page is built
/// for `wasm32-unknown-unknown`, served from a loopback server and run in headless Chrome through the runner of
/// `check-web` (`web::run_example`), so the part also needs Chrome, a `wasm-bindgen` CLI of the locked version and the
/// wasm32 target; a missing one fails the part, and so does a page that does not log its `GUST_OK` line.
fn gpu_part(root: &Path, browser: bool) -> Result<String, String> {
    let cargo = cargo_program()?;
    let gallery = root.join(GALLERY_DIR);
    let target = gallery_target(root, "no-slang");
    // The driver is built only when an entry runs through it.
    let driver = if ENTRIES.iter().any(|entry| entry.via_driver) {
        Some(build_driver(root, &cargo)?)
    } else {
        None
    };
    let mut lines = Vec::new();
    let runs = ENTRIES
        .iter()
        .map(|entry| (entry, None, entry.expected_line))
        .chain(ENTRIES.iter().filter_map(|entry| {
            entry
                .profile_run
                .as_ref()
                .map(|run| (entry, Some(run.profile), run.expected_line))
        }));
    for (entry, profile, expected_line) in runs {
        if entry.browser {
            lines.push(if browser {
                browser_run(root, entry)?
            } else {
                browser_skipped_line(entry)
            });
            continue;
        }
        let mut run = match (&driver, entry.via_driver) {
            (Some(driver), true) => {
                let mut command = Command::new(driver);
                command
                    .current_dir(&gallery)
                    .env("CARGO_TARGET_DIR", &target)
                    .env("CARGO_TERM_COLOR", "never")
                    .env("CARGO", &cargo);
                for name in CLEARED {
                    command.env_remove(name);
                }
                command.args([
                    "gust",
                    "run",
                    "--release",
                    "--locked",
                    "-q",
                    "-p",
                    entry.dir,
                ]);
                command
            }
            _ => {
                let mut command = gallery_cargo(&gallery, &target)?;
                command.args(["run", "--release", "--locked", "-q", "-p", entry.dir]);
                command
            }
        };
        let tool = if entry.via_driver {
            "cargo gust run"
        } else {
            "cargo run"
        };
        let label = match profile {
            None => format!("{tool} --release -p {}", entry.dir),
            Some(profile) => format!("GUST_PROFILE={profile} {tool} --release -p {}", entry.dir),
        };
        if let Some(profile) = profile {
            run.env("GUST_PROFILE", profile);
        }
        let done = require(exec(run, &label)?, &label)?;
        if !output_has_line(&done.stdout, expected_line) {
            return Err(format!(
                "{label} printed\n{}\nbut the line `{expected_line}` is expected",
                tail(&done.stdout, 10)
            ));
        }
        let Some(adapter) = adapter_name(&done.stdout) else {
            return Err(format!(
                "{label} printed no `{ADAPTER_PREFIX}` line, so the output does not say which GPU it ran on:\n{}",
                tail(&done.stdout, 10)
            ));
        };
        lines.push(format!(
            "{}{}: printed `{expected_line}` on adapter `{adapter}`",
            entry.dir,
            profile.map_or_else(String::new, |profile| format!(" (GUST_PROFILE={profile})"))
        ));
    }
    Ok(lines.join("\n"))
}

/// What the part prints for a browser entry it was asked to leave out (`check-full --skip-web`), so that the skip is
/// visible in the output and not silent.
pub(crate) fn browser_skipped_line(entry: &Entry) -> String {
    format!(
        "{}: browser run SKIPPED on request (check-full --skip-web); no evidence for the page in a browser",
        entry.dir
    )
}

/// The page of a browser entry in headless Chrome, through the runner of `check-web`. The page's `GUST_OK` line must name the
/// adapter words, the kernel its build selected (compared with the committed manifest) and end with the entry's
/// `expected_line`; the returned line names the Chrome version and the adapter.
fn browser_run(root: &Path, entry: &Entry) -> Result<String, String> {
    let example = match entry.dir {
        "web-hello" => &WEB_HELLO,
        other => {
            return Err(format!(
                "gallery/{other} is a browser entry with no page registered"
            ));
        }
    };
    let label = format!(
        "{} page in headless Chrome ({} through the runner of check-web)",
        entry.dir, example.dir
    );
    println!("  $ {label}");
    let run = run_example(
        root,
        example,
        &WebOptions::default(),
        &gallery_target(root, "web"),
        Instant::now(),
    )
    .map_err(|error| {
        // The runner's messages name the flags of `check-web`; this check has none.
        let note = if error.contains("--chrome") || error.contains("--wasm-bindgen") {
            "
  check-gallery takes no --chrome or --wasm-bindgen flag: it looks for Chrome in its standard install locations and for a `wasm-bindgen` CLI of the locked version on PATH"
        } else {
            ""
        };
        format!("{label}: {error}{note}")
    })?;
    Ok(format!(
        "{}: the page logged `GUST_OK {}`, evidence: {}",
        entry.dir,
        run.label,
        evidence_label(run.chrome_version.as_deref(), &run.adapter)
    ))
}

/// The line of an entry that names the adapter it ran on.
const ADAPTER_PREFIX: &str = "adapter: ";

/// The adapter named by the first `adapter: <name>` line of `output`, if it names one.
pub(crate) fn adapter_name(output: &str) -> Option<&str> {
    output
        .lines()
        .find_map(|line| line.trim_end().strip_prefix(ADAPTER_PREFIX))
        .map(str::trim)
        .filter(|name| !name.is_empty())
}

/// `cargo xtask check-gallery [--slang] [--no-slang] [--gpu]`.
pub(crate) fn check_gallery(args: &[String]) -> Result<(), String> {
    let parts = Parts::parse(args)?;
    run_parts(&workspace_root()?, parts, true)?;
    println!(
        "check-gallery passed (parts run: {})",
        parts.run_names().join(", ")
    );
    Ok(())
}

/// The GPU part alone, which `check-full` runs (`verify.rs`). `browser` is false only for `check-full --skip-web`: the
/// browser entries are then left out, the output says so and `check-full` records the skip.
pub(crate) fn check_gallery_gpu_report(browser: bool) -> Result<String, String> {
    run_parts(
        &workspace_root()?,
        Parts {
            slang: false,
            no_slang: false,
            gpu: true,
        },
        browser,
    )
}

/// A part of the check, as [`run_parts`] calls it.
type PartFn<'a> = dyn Fn(&Path) -> Result<String, String> + 'a;

fn run_parts(root: &Path, parts: Parts, browser: bool) -> Result<String, String> {
    println!(
        "check-gallery: parts that run: {}",
        parts.run_names().join(", ")
    );
    if !parts.not_run_names().is_empty() {
        println!(
            "check-gallery: parts not run (not selected): {}",
            parts.not_run_names().join(", ")
        );
    }
    check_entries(root)?;
    let mut report = vec![format!(
        "check-gallery: parts run: {}",
        parts.run_names().join(", ")
    )];
    if !parts.not_run_names().is_empty() {
        report.push(format!(
            "check-gallery: parts not run (not selected): {}",
            parts.not_run_names().join(", ")
        ));
    }
    let gpu = |root: &Path| gpu_part(root, browser);
    let parts_to_run: [(bool, &str, &PartFn<'_>); 3] = [
        (parts.slang, "slang", &slang_part),
        (parts.no_slang, "no-slang", &no_slang_part),
        (parts.gpu, "gpu", &gpu),
    ];
    for (selected, name, part) in parts_to_run {
        if !selected {
            continue;
        }
        println!("check-gallery: {name} part");
        let summary = part(root).map_err(|error| format!("check-gallery {name} part: {error}"))?;
        println!("{summary}");
        report.push(format!("[{name}] ok\n{summary}"));
    }
    Ok(report.join("\n"))
}

#[cfg(test)]
#[path = "tests/gallery.rs"]
mod gallery_tests;

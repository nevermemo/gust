//! The crate's declarations and the build profile's grammar (m8.a, D138).
//!
//! The corpus reads one set of declarations from every source form a manifest can take, including the form `cargo
//! package` rewrites a manifest into (the registry's copy of a dependent's `physics = { .. }` arrives as a header), and
//! pins each GUST-E0753 case to its exact line, column and message.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{
    diagnostics::{
        Code,
        testing::{assert_code, parse_str},
    },
    expand::{
        Capability, CapabilitySet, Environment, Identity, Profile, ProfileTarget,
        expand_map_with_directories, expand_validated, expand_with_directories,
    },
    manifest::{self, Declarations, Limits, ManifestError, Route},
    profile::GRID,
};

// ---------------------------------------------------------------------------------------------------------------
// The grammar of GUST_PROFILE

/// A parsed value: its target and capabilities.
type Parsed<'a> = (&'a str, &'a [&'a str]);

/// The grammar of `GUST_PROFILE`: every row is (value, the target and the capabilities it parses to, or `None` when
/// the build script refuses it). This is the table of `crates/gust/build.rs` (its test module), pasted (rows verbatim), so the
/// two copies of the grammar cannot drift apart unnoticed.
const GRAMMAR: &[(&str, Option<Parsed<'static>>)] = &[
    ("dev", Some(("dev", &[]))),
    ("native", Some(("native", &[]))),
    ("web", Some(("web", &[]))),
    ("cpu", Some(("cpu", &[]))),
    ("dev+subgroups", Some(("dev", &["subgroups"]))),
    ("native+subgroups", Some(("native", &["subgroups"]))),
    ("", None),
    ("bogus", None),
    ("Native", None),
    (" native", None),
    ("native ", None),
    ("+subgroups", None),
    ("native+", None),
    ("native++subgroups", None),
    ("native+subgroups+", None),
    ("native+ subgroups", None),
    ("native+Subgroups", None),
    ("native+subgroup", None),
    ("native+subgroups+subgroups", None),
    ("web+subgroups", None),
    ("cpu+subgroups", None),
    ("cpu+simd", None),
    ("native+gpu", None),
    ("native+f16", None),
    ("dev+f16", None),
    ("native+atomics64", None),
    ("dev+subgroups+f16", None),
];

fn target_named(name: &str) -> ProfileTarget {
    match name {
        "dev" => ProfileTarget::Dev,
        "native" => ProfileTarget::Native,
        "web" => ProfileTarget::Web,
        "cpu" => ProfileTarget::Cpu,
        other => panic!("{other}"),
    }
}

#[test]
fn the_grammar_matches_the_build_script() {
    for (value, parsed) in GRAMMAR {
        let found = Profile::parse(value);
        match parsed {
            Some((target, names)) => {
                let profile = found.unwrap_or_else(|| panic!("{value:?} must parse"));
                assert_eq!(profile.target, Some(target_named(target)), "{value:?}");
                let mut wanted = CapabilitySet::EMPTY;
                for name in *names {
                    wanted = wanted.with(Capability::from_name(name).expect("a known name"));
                }
                assert_eq!(profile.capabilities, wanted, "{value:?}");
            }
            None => assert_eq!(found, None, "{value:?} must be refused"),
        }
    }
}

#[test]
fn a_refused_or_unset_value_is_other_and_only_the_exact_cpu_is_cpu() {
    assert_eq!(Profile::from_value(None), Profile::Other);
    for (value, parsed) in GRAMMAR {
        let profile = Profile::from_value(Some(value));
        assert_eq!(profile == Profile::Cpu, *value == "cpu", "{value:?}");
        assert_eq!(profile.is_cpu(), *value == "cpu", "{value:?}");
        if parsed.is_none() {
            assert_eq!(profile, Profile::Other, "{value:?}");
        }
    }
    // The capabilities are kept: `dev+subgroups` is not `dev`, and neither is `Other`.
    let dev = Profile::from_value(Some("dev"));
    let subgroups = Profile::from_value(Some("dev+subgroups"));
    assert_ne!(dev, subgroups);
    assert!(subgroups.capabilities.contains(Capability::Subgroups));
    assert!(!dev.capabilities.contains(Capability::Subgroups));
    assert_eq!(Profile::Other.target, None);
}

// ---------------------------------------------------------------------------------------------------------------
// The corpus: one set of declarations, every source form

/// One route as the tests write it: its tag, its targets and its default.
type RouteRow<'a> = (&'a str, Option<&'a [&'a str]>, Option<&'a str>);

fn routes(entries: &[RouteRow<'_>]) -> Vec<Route> {
    entries
        .iter()
        .map(|(tag, targets, default)| Route {
            tag: (*tag).to_owned(),
            targets: targets.map(|names| names.iter().map(|name| (*name).to_owned()).collect()),
            default: default.map(str::to_owned),
        })
        .collect()
}

/// What every form of the corpus below declares. The routes come back in the order of the tags' names.
fn declared() -> Declarations {
    Declarations {
        profiles: Some(
            ["native", "native+subgroups", "web", "cpu"]
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
        ),
        limits: Some(Limits::Portable),
        routes: routes(&[
            ("physics", Some(&["gpu", "cpu"]), Some("gpu")),
            ("physics.broadphase", None, None),
            ("render", None, Some("cpu")),
        ]),
    }
}

const PACKAGE: &str = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n";

/// The inline-table form, as a person writes it.
const INLINE: &str = r#"
[package.metadata.gust]
profiles = ["native", "native+subgroups", "web", "cpu"]
limits = "portable"

[package.metadata.gust.routes]
physics = { targets = ["gpu", "cpu"], default = "gpu" }
"physics.broadphase" = {}
render = { default = "cpu" }
"#;

/// Dotted keys, literal strings, comments, a multi-line array and a trailing comma.
const DOTTED: &str = r#"
[package]
name = "demo"
metadata.gust.profiles = [
    'native',   # the default GPU build
    'native+subgroups',
    'web',
    'cpu',
]
metadata.gust.limits = 'portable'
metadata.gust.routes.physics = { targets = ['gpu', 'cpu'], default = 'gpu' }
metadata.gust.routes."physics.broadphase" = {}
metadata.gust.routes.'render'.default = 'cpu'
"#;

/// Headers, quoted keys and an empty header table.
const HEADERS: &str = r#"
[package.metadata.gust]
"profiles" = ["native", "native+subgroups", "web", "cpu"]
'limits' = "portable"

[package.metadata.gust.routes.physics]
targets = ["gpu", "cpu"]
default = "gpu"

[package.metadata.gust.routes."physics.broadphase"]

[package.metadata.gust.routes.render]
default = "cpu"
"#;

/// The source manifest of the scratch crate that `cargo package` rewrote into [`PACKAGED`], verbatim. To measure again:
/// put this text in `Cargo.toml` of a crate with an empty `src/lib.rs`, run `cargo package --allow-dirty --no-verify`
/// there, and compare the `Cargo.toml` inside the `.crate` with [`PACKAGED`] (cargo 1.98.0 (797e8a9bc 2026-08-05) wrote it,
/// and wrote it again identically on the second measurement, in fix round 1).
const PACKAGED_FROM: &str = r#"[package]
name = "m8a-corpus"
version = "0.1.0"
edition = "2024"
description = "Corpus of the m8.a manifest reader"
license = "MIT OR Apache-2.0"

[package.metadata.gust]
profiles = ["native", "native+subgroups", "web", "cpu"]
limits = "portable"

[package.metadata.gust.routes]
physics = { targets = ["gpu", "cpu"], default = "gpu" }
"physics.broadphase" = {}
render = { default = "cpu" }
"#;

/// The manifest `cargo package` wrote for the crate of [`PACKAGED_FROM`] (cargo 1.98.0 (797e8a9bc 2026-08-05),
/// `cargo package --allow-dirty --no-verify`, then the packaged `Cargo.toml` as it is in the `.crate`): arrays are
/// exploded one entry a line, `physics = { .. }` became a header, and the quoted tag became a quoted header. This is the
/// form a dependent's `#[gpu]` module sees after a registry download.
const PACKAGED: &str = r#"# THIS FILE IS AUTOMATICALLY GENERATED BY CARGO
#
# When uploading crates to the registry Cargo will automatically
# "normalize" Cargo.toml files for maximal compatibility
# with all versions of Cargo and also rewrite `path` dependencies
# to registry (e.g., crates.io) dependencies.
#
# If you are reading this file be aware that the original Cargo.toml
# will likely look very different (and much more reasonable).
# See Cargo.toml.orig for the original contents.

[package]
edition = "2024"
name = "m8a-corpus"
version = "0.1.0"
build = false
autolib = false
autobins = false
autoexamples = false
autotests = false
autobenches = false
description = "Corpus of the m8.a manifest reader"
readme = false
license = "MIT OR Apache-2.0"

[package.metadata.gust]
profiles = [
    "native",
    "native+subgroups",
    "web",
    "cpu",
]
limits = "portable"

[package.metadata.gust.routes.physics]
targets = [
    "gpu",
    "cpu",
]
default = "gpu"

[package.metadata.gust.routes."physics.broadphase"]

[package.metadata.gust.routes.render]
default = "cpu"

[lib]
name = "m8a_corpus"
path = "src/lib.rs"
"#;

#[test]
fn every_source_form_declares_the_same_thing() {
    for (name, text) in [
        ("inline", format!("{PACKAGE}{INLINE}")),
        ("dotted", DOTTED.to_owned()),
        ("headers", format!("{PACKAGE}{HEADERS}")),
        ("packaged by cargo", PACKAGED.to_owned()),
    ] {
        assert_eq!(manifest::read(&text), Ok(declared()), "{name}");
    }
}

#[test]
fn the_source_of_the_packaged_manifest_reads_the_same_as_the_packaged_one() {
    assert_eq!(manifest::read(PACKAGED_FROM), Ok(declared()));
    assert_eq!(manifest::read(PACKAGED_FROM), manifest::read(PACKAGED));
}

#[test]
fn the_form_cargo_package_writes_is_the_one_a_dependent_sees() {
    // The point of the corpus: the packaged text has no inline table at all, so a reader that accepted only the inline
    // form would raise E0753 on every published crate with routes.
    assert!(!PACKAGED.contains('{'));
    assert!(PACKAGED.contains("[package.metadata.gust.routes.physics]"));
    assert!(PACKAGED.contains("[package.metadata.gust.routes.\"physics.broadphase\"]"));
}

#[test]
fn an_absent_key_differs_from_a_present_one() {
    // Nothing declared at all, and a manifest with other metadata: no declarations.
    for text in [
        PACKAGE.to_owned(),
        format!("{PACKAGE}[package.metadata.docs.rs]\nall-features = true\n"),
        format!("{PACKAGE}[package.metadata]\n"),
        format!("{PACKAGE}[workspace.metadata.gust]\nprofiles = [\"cpu\"]\n"),
    ] {
        assert_eq!(manifest::read(&text), Ok(Declarations::default()), "{text}");
    }
    let empty_table = format!("{PACKAGE}[package.metadata.gust]\n");
    assert_eq!(manifest::read(&empty_table), Ok(Declarations::default()));
    // An empty list declares nothing, so it is refused (D138); a list equal to the default is present and stricter than
    // none.
    let empty_list = format!("{PACKAGE}[package.metadata.gust]\nprofiles = []\n");
    assert!(manifest::read(&empty_list).is_err());
    let default_list =
        format!("{PACKAGE}[package.metadata.gust]\nprofiles = [\"native\", \"web\", \"cpu\"]\n");
    let read = manifest::read(&default_list).unwrap();
    assert_eq!(
        read.profiles,
        Some(vec![
            "native".to_owned(),
            "web".to_owned(),
            "cpu".to_owned()
        ])
    );
    assert_ne!(read.profiles, Declarations::default().profiles);
}

#[test]
fn only_the_gust_table_of_the_package_is_read() {
    // Other tables with the same key names, and the rest of the manifest, are not looked at.
    let text = format!(
        "{PACKAGE}[dependencies]\nserde = {{ version = \"1\", features = [\"derive\"] }}\n\
         [package.metadata.other]\nprofiles = 3\n[package.metadata.gust]\nlimits = \"portable\"\n\
         [target.'cfg(unix)'.dependencies]\nlibc = \"0.2\"\n"
    );
    let read = manifest::read(&text).unwrap();
    assert_eq!(read.limits, Some(Limits::Portable));
    assert_eq!(read.profiles, None);
    assert!(read.routes.is_empty());
}

// ---------------------------------------------------------------------------------------------------------------
// GUST-E0753: the exact place and message of each case

/// The error `manifest::read` gives for `body` placed under the standard `[package]` header (4 lines, so `body`'s
/// first line is line 5).
fn failure(body: &str) -> ManifestError {
    manifest::read(&format!("{PACKAGE}{body}")).unwrap_err()
}

#[test]
fn a_wrong_value_names_its_line_and_column() {
    let cases: &[(&str, usize, usize, &str)] = &[
        (
            "[package.metadata.gust]\nprofiles = \"native\"\n",
            6,
            12,
            "profiles must be an array of strings, found string",
        ),
        (
            "[package.metadata.gust]\nprofiles = [\"native\", 3]\n",
            6,
            23,
            "each entry of profiles must be a string, found integer",
        ),
        (
            "[package.metadata.gust]\nprofiles = [\"native\", \"web+subgroups\"]\n",
            6,
            23,
            "`web+subgroups` is not a build profile; write <target>[+<capability>] with the target dev, native, web or cpu and the capability subgroups (web and cpu take none)",
        ),
        (
            "[package.metadata.gust]\nlimits = \"device\"\n",
            6,
            10,
            "limits is \"device\", but the one value is \"portable\"",
        ),
        (
            "[package.metadata.gust]\nlimits = true\n",
            6,
            10,
            "limits must be the string \"portable\", found boolean",
        ),
        (
            "[package.metadata.gust]\nprofile = [\"native\"]\n",
            6,
            1,
            "unknown key `profile` in [package.metadata.gust]; the keys are profiles, limits, routes",
        ),
        (
            "[package.metadata]\ngust = 3\n",
            6,
            8,
            "[package.metadata.gust] must be a table, found integer",
        ),
        (
            "[package.metadata.gust]\nroutes = [\"physics\"]\n",
            6,
            10,
            "routes must be a table, found array",
        ),
        (
            "[package.metadata.gust.routes]\nphysics = \"gpu\"\n",
            6,
            11,
            "the route `physics` must be a table such as { targets = [\"gpu\", \"cpu\"], default = \"gpu\" }, found string",
        ),
        (
            "[package.metadata.gust.routes]\nphysics = { targets = [\"gpu\", \"tpu\"] }\n",
            6,
            31,
            "`tpu` is not a route target; the targets are gpu and cpu",
        ),
        (
            "[package.metadata.gust.routes]\nphysics = { default = \"both\" }\n",
            6,
            23,
            "`both` is not a route target; the targets are gpu and cpu",
        ),
        (
            "[package.metadata.gust.routes]\nphysics = { default = 1 }\n",
            6,
            23,
            "default must be \"gpu\" or \"cpu\", found integer",
        ),
        (
            "[package.metadata.gust.routes]\nphysics = { target = [\"gpu\"] }\n",
            6,
            13,
            "unknown key `target` in the route `physics`; the keys are targets and default",
        ),
        (
            "[package.metadata.gust.routes.physics.broadphase]\n",
            5,
            39,
            "`physics.broadphase` is a table inside the route `physics`; a dotted tag is written as a quoted key, \"physics.broadphase\"",
        ),
        // The list of profiles (D138): empty, repeated, and a dev profile are each refused at the entry.
        (
            "[package.metadata.gust]\nprofiles = []\n",
            6,
            1,
            "an empty list declares no build; omit the key for the default",
        ),
        (
            "[package.metadata.gust]\nprofiles = [\"native\", \"native\"]\n",
            6,
            23,
            "`native` is listed twice in profiles",
        ),
        (
            "[package.metadata.gust]\nprofiles = [\"native+subgroups\", \"web\", \"native+subgroups\"]\n",
            6,
            40,
            "`native+subgroups` is listed twice in profiles",
        ),
        (
            "[package.metadata.gust]\nprofiles = [\"dev\"]\n",
            6,
            13,
            "a dev build is checked as `native` with the same capabilities; list `native...`",
        ),
        (
            "[package.metadata.gust]\nprofiles = [\"native\", \"dev+subgroups\"]\n",
            6,
            23,
            "a dev build is checked as `native` with the same capabilities; list `native...`",
        ),
        // Wrong shapes that the corpus did not have (m8.a review, N6).
        (
            "[package.metadata.gust.routes]\nphysics = { targets = \"gpu\" }\n",
            6,
            23,
            "targets must be an array of strings, found string",
        ),
        (
            "[[package.metadata.gust]]\n",
            5,
            1,
            "[package.metadata.gust] must be a table, found array",
        ),
        (
            "[[package.metadata.gust.routes.physics]]\n",
            5,
            1,
            "the route `physics` must be a table such as { targets = [\"gpu\", \"cpu\"], default = \"gpu\" }, found array",
        ),
        (
            "[package.metadata.gust]\nprofiles.workspace = true\n",
            6,
            1,
            "profiles must be an array of strings, found table",
        ),
    ];
    for (body, line, column, message) in cases {
        let error = failure(body);
        assert_eq!(
            (error.line, error.column, error.message.as_str()),
            (*line, *column, *message),
            "{body}"
        );
        assert_eq!(
            error.to_string(),
            format!("Cargo.toml:{line}:{column}: {message}"),
            "{body}"
        );
    }
}

#[test]
fn a_column_counts_characters_and_a_crlf_manifest_reads_the_same() {
    // A column counts characters, not bytes (the `\u{e9}` before the error is two bytes); a CRLF file has the same
    // lines and columns.
    let body = "[package.metadata.gust]\nprofiles = [\"\u{e9}\", 3]\n";
    let error = failure(body);
    assert_eq!((error.line, error.column), (6, 18), "{error}");
    let crlf = format!("{PACKAGE}{body}").replace('\n', "\r\n");
    let crlf_error = manifest::read(&crlf).unwrap_err();
    assert_eq!((crlf_error.line, crlf_error.column), (6, 18));
    assert_eq!(crlf_error.message, error.message);
    let ok = format!("{PACKAGE}{INLINE}").replace('\n', "\r\n");
    assert_eq!(manifest::read(&ok), Ok(declared()));
}

#[test]
fn text_that_is_not_toml_is_an_error_with_a_place_and_a_message_from_the_parser() {
    let error = manifest::read("[package\nname = 1\n").unwrap_err();
    assert_eq!(error.line, 1, "{error}");
    assert!(error.message.starts_with("not valid TOML: "), "{error}");
    // A duplicate key is TOML's error, not ours, and keeps its place.
    let duplicate =
        failure("[package.metadata.gust]\nlimits = \"portable\"\nlimits = \"portable\"\n");
    assert_eq!(duplicate.line, 7, "{duplicate}");
    assert!(
        duplicate.message.starts_with("not valid TOML: "),
        "{duplicate}"
    );
}

#[test]
fn an_error_never_names_a_directory() {
    // The message begins with the bare file name; the path of the crate is never in it.
    let error = failure("[package.metadata.gust]\nprofile = 1\n");
    assert!(error.to_string().starts_with("Cargo.toml:"), "{error}");
    assert!(!error.to_string().contains('/') && !error.to_string().contains('\\'));
}

// ---------------------------------------------------------------------------------------------------------------
// Through the expansion: the error at the module, the tracking of the file

/// A scratch crate directory below the test's temporary directory, with `manifest` as its `Cargo.toml` (none when
/// `None`). Removed when dropped.
struct Crate(PathBuf);

impl Crate {
    fn new(name: &str, manifest: Option<&str>) -> Self {
        let directory =
            std::env::temp_dir().join(format!("gust-m8a-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        if let Some(text) = manifest {
            fs::write(directory.join("Cargo.toml"), text).unwrap();
        }
        Self(directory)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Crate {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const KERNEL: &str = "mod layout {
    #[kernel]
    pub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {
        let i = id.x;
    }
}";

fn expand_in_crate(directory: &Path) -> syn::Result<proc_macro2::TokenStream> {
    expand_with_directories(
        parse_str(KERNEL).unwrap(),
        Some(directory),
        None,
        Some("demo"),
    )
}

#[test]
fn a_malformed_table_is_e0753_at_the_module_with_the_place_in_the_message() {
    let scratch = Crate::new(
        "malformed",
        Some(&format!(
            "{PACKAGE}[package.metadata.gust]\nlimits = \"device\"\n"
        )),
    );
    let error = expand_in_crate(scratch.path()).unwrap_err();
    assert_code(&error, Code::GustMetadata);
    assert_eq!(
        error.to_string(),
        "[GUST-E0753] Cargo.toml:6:10: limits is \"device\", but the one value is \"portable\""
    );
}

#[test]
fn a_manifest_that_is_not_there_declares_nothing_and_tracks_nothing() {
    let scratch = Crate::new("absent", None);
    let text = expand_in_crate(scratch.path()).unwrap().to_string();
    assert!(!text.contains("include_str"), "{text}");
    // No directory at all (the unit tests of the emitter) is the same.
    let text = expand_with_directories(parse_str(KERNEL).unwrap(), None, None, None)
        .unwrap()
        .to_string();
    assert!(!text.contains("include_str"), "{text}");
}

#[test]
fn a_manifest_is_tracked_by_the_compiler_and_its_path_is_never_in_the_expansion() {
    let scratch = Crate::new("tracked", Some(&format!("{PACKAGE}{INLINE}")));
    let text = expand_in_crate(scratch.path()).unwrap().to_string();
    let item = ":: core :: include_str ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/Cargo.toml\")) ;";
    assert_eq!(text.matches(item).count(), 1, "{text}");
    let directory = scratch.path().to_string_lossy().replace('\\', "/");
    assert!(!text.replace('\\', "/").contains(&directory), "{text}");
    // A manifest without any gust table is tracked too: adding one later must re-expand the module.
    let plain = Crate::new("plain", Some(PACKAGE));
    let text = expand_in_crate(plain.path()).unwrap().to_string();
    assert_eq!(text.matches(item).count(), 1, "{text}");
}

#[test]
fn the_declarations_are_read_in_a_cpu_build_too() {
    let scratch = Crate::new(
        "cpu",
        Some(&format!("{PACKAGE}[package.metadata.gust]\nprofiles = 3\n")),
    );
    let environment = Environment {
        manifest_dir: Some(scratch.path().to_path_buf()),
        export_dir: None,
        package: Some("demo".to_owned()),
        reported_file: None,
    };
    let error = expand_validated(
        parse_str(KERNEL).unwrap(),
        &environment,
        Profile::Cpu,
        &GRID,
        &proc_macro2::TokenStream::new(),
        Identity::Module,
    )
    .unwrap_err();
    assert_code(&error, Code::GustMetadata);
}

#[test]
fn the_mirror_of_the_capability_names_is_the_one_of_gust() {
    // `gust::capability::Capability::ALL` has `subgroups` and nothing else; the macro's copy knows the same names.
    assert_eq!(
        Capability::from_name("subgroups"),
        Some(Capability::Subgroups)
    );
    for other in ["", "f16", "atomics64", "Subgroups", "subgroup"] {
        assert_eq!(Capability::from_name(other), None, "{other:?}");
    }
}

#[test]
fn a_gpu_fn_reads_the_manifest_too_so_a_crate_with_only_maps_gets_e0753_at_the_function_name() {
    // The map path goes through `expand_validated` like a module does (M1 of the review): a crate whose only GPU item
    // is a `#[gpu] fn` and whose manifest is malformed reports GUST-E0753, at the function's name.
    let source = "\n\n    pub fn doubled(x: f32) -> f32 { x * 2.0 }";
    let function: syn::ItemFn = parse_str(source).unwrap();
    let scratch = Crate::new(
        "map-malformed",
        Some(&format!(
            "{PACKAGE}[package.metadata.gust]\nlimits = \"device\"\n"
        )),
    );
    let error = expand_map_with_directories(function.clone(), Some(scratch.path())).unwrap_err();
    assert_code(&error, Code::GustMetadata);
    assert_eq!(
        error.to_string(),
        "[GUST-E0753] Cargo.toml:6:10: limits is \"device\", but the one value is \"portable\""
    );
    let start = error.span().start();
    assert_eq!(start, function.sig.ident.span().start());
    assert_eq!((start.line, start.column), (3, 11));
    // A well-formed manifest is tracked once on the map path too, with no directory in the expansion.
    let fine = Crate::new("map-tracked", Some(&format!("{PACKAGE}{INLINE}")));
    let text = expand_map_with_directories(function, Some(fine.path()))
        .unwrap()
        .to_string();
    let item = ":: core :: include_str ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/Cargo.toml\")) ;";
    assert_eq!(text.matches(item).count(), 1, "{text}");
    let directory = fine.path().to_string_lossy().replace('\\', "/");
    assert!(!text.replace('\\', "/").contains(&directory), "{text}");
}

#[test]
fn the_name_of_a_capability_reads_back_to_the_capability() {
    assert_eq!(Capability::Subgroups.name(), "subgroups");
    assert_eq!(
        Capability::from_name(Capability::Subgroups.name()),
        Some(Capability::Subgroups)
    );
}

/// The rows of the `GRAMMAR` table in the Rust `source`, trimmed, from the line of `const GRAMMAR` to the closing `];`.
fn grammar_rows(source: &str) -> Vec<String> {
    let mut rows = Vec::new();
    let mut inside = false;
    for line in source.lines().map(str::trim) {
        if line.starts_with("const GRAMMAR") {
            inside = true;
        } else if inside && line == "];" {
            return rows;
        } else if inside {
            rows.push(line.to_owned());
        }
    }
    panic!("no complete GRAMMAR table");
}

#[test]
fn the_grammar_table_is_the_one_in_the_build_script_row_for_row() {
    // The build script of `gust` cannot be called from here (the macro does not depend on it), so its test table and
    // this one are compared as text: adding a row, a capability or a refusal to one and not the other fails here.
    let build_script = include_str!("../../../gust/build.rs");
    let mine = include_str!("manifest.rs");
    let theirs = grammar_rows(build_script);
    assert!(theirs.len() > 20, "{theirs:?}");
    assert_eq!(grammar_rows(mine), theirs);
}

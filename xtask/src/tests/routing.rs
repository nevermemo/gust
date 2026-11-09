use std::fs;

use super::{
    CASES, ERRORS, FIXTURE_DIR, FLIPS, Options, STATES, TAGS, Via, flip_target, probe_lines,
    rewrite_manifest, routes_rows, tail,
};
use crate::workspace::workspace_root;

fn args(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

/// A file of the repository, with LF endings.
fn repository_file(path: &str) -> String {
    let root = workspace_root().expect("the repository root");
    fs::read_to_string(root.join(path))
        .unwrap_or_else(|error| panic!("{path}: {error}"))
        .replace("\r\n", "\n")
}

#[test]
fn options_name_the_cases_and_check_only_leaves_out_the_gpu() {
    assert_eq!(Options::parse(&[]).unwrap().cases, CASES);
    assert_eq!(
        Options::parse(&args(&["--check-only"])).unwrap().cases,
        [
            "plain",
            "resolution",
            "errors",
            "test-run",
            "guard",
            "flip",
            "lint"
        ]
    );
    // Cases run in their own order whatever the order given, and a repeat is one.
    assert_eq!(
        Options::parse(&args(&[
            "--case", "flip", "--case", "plain", "--case", "flip"
        ]))
        .unwrap()
        .cases,
        ["plain", "flip"]
    );
    assert_eq!(
        Options::parse(&args(&["--case", "gpu"])).unwrap().cases,
        ["gpu"]
    );
    for (words, phrase) in [
        (&["--case"][..], "needs a case name"),
        (&["--case", "nope"][..], "unknown case `nope`"),
        (
            &["--check-only", "--case", "gpu"][..],
            "leaves out the gpu case",
        ),
        (&["--quick"][..], "not `--quick`"),
    ] {
        let error = Options::parse(&args(words)).unwrap_err();
        assert!(error.contains(phrase), "{words:?}: {error}");
    }
}

#[test]
fn the_golden_states_are_consistent() {
    assert_eq!(STATES.len(), 7);
    let mut sorted = TAGS;
    sorted.sort_unstable();
    assert_eq!(sorted, TAGS, "routes.json lists tags in order");
    for state in &STATES {
        for (target, level) in state.rows {
            assert!(["gpu", "cpu"].contains(&target), "{}", state.name);
            assert!(
                ["profile", "entry", "default", "fallback"].contains(&level),
                "{}",
                state.name
            );
        }
        // The profile `cpu` decides every tag and no other state has that level; an entry exists exactly when a
        // state's entries are not empty.
        let profile_rows = state
            .rows
            .iter()
            .filter(|(_, level)| *level == "profile")
            .count();
        assert_eq!(
            profile_rows,
            if state.profile == Some("cpu") { 6 } else { 0 }
        );
        let has_entry = state.rows.iter().any(|(_, level)| *level == "entry");
        assert_eq!(has_entry, !state.entries.is_empty(), "{}", state.name);
        assert_eq!(probe_lines(state).len(), 3);
    }
    let first = probe_lines(&STATES[0]);
    assert_eq!(
        first,
        [
            "physics scale gpu",
            "physics.broadphase pairs gpu",
            "fusion blend cpu"
        ]
    );
}

#[test]
fn the_error_states_name_a_file_and_line_and_the_fixture_declares_their_tags() {
    assert_eq!(ERRORS.len(), 9);
    for case in &ERRORS {
        assert!(
            case.needles
                .iter()
                .any(|needle| needle.starts_with("gust.toml:")),
            "{}",
            case.name
        );
        assert!(!case.entries.is_empty(), "{}", case.name);
    }
    // The tags the states and errors use are the ones the fixture declares.
    let manifest = repository_file(&format!("{FIXTURE_DIR}/route-kernels/Cargo.toml"));
    for tag in TAGS {
        assert!(
            manifest.contains(&format!("\n{tag} = "))
                || manifest.contains(&format!("\n\"{tag}\" = ")),
            "{tag}: {manifest}"
        );
    }
}

#[test]
fn the_flip_goes_both_ways_twice_in_one_command() {
    let pairs: Vec<(Via, Via)> = FLIPS.windows(2).map(|pair| (pair[0], pair[1])).collect();
    for wanted in [
        (Via::Unset, Via::Entry),
        (Via::Entry, Via::Unset),
        (Via::Entry, Via::Default),
        (Via::Default, Via::Entry),
    ] {
        assert!(pairs.contains(&wanted), "{wanted:?} in {pairs:?}");
    }
    let unset_to_set = pairs
        .iter()
        .filter(|pair| **pair == (Via::Unset, Via::Entry))
        .count();
    assert_eq!(unset_to_set, 2);
    // No two neighbours give the same target: every state is a flip a stale build would get wrong.
    for pair in &pairs {
        assert_ne!(flip_target(pair.0), flip_target(pair.1), "{pair:?}");
    }
}

#[test]
fn the_fixture_manifests_are_rewritten_to_the_repository_path() {
    let manifest = "gust = { package = \"gust-compute\", path = \"../../../crates/gust\", default-features = false }\nroute-kernels = { path = \"../route-kernels\" }\n";
    let rewritten = rewrite_manifest(manifest, "/r/gust/");
    assert!(
        rewritten.contains("path = \"/r/gust/crates/gust\""),
        "{rewritten}"
    );
    assert!(
        rewritten.contains("path = \"../route-kernels\""),
        "{rewritten}"
    );
    // Every manifest of the fixture reaches the repository only through that spelling.
    for path in [
        "route-kernels/Cargo.toml",
        "probe/Cargo.toml",
        "cpu-app/Cargo.toml",
        "gpu-app/Cargo.toml",
    ] {
        let text = repository_file(&format!("{FIXTURE_DIR}/{path}"));
        let rewritten = rewrite_manifest(&text, "/r");
        assert!(!rewritten.contains("../../../"), "{path}: {rewritten}");
    }
}

#[test]
fn routes_rows_reads_the_driver_table() {
    let text = r#"{"profile":"dev","routes":[
        {"from":null,"level":"fallback","package":"p","tag":"a","target":"gpu","variable":"GUST_ROUTE__p__a"},
        {"from":"p::a","level":"entry","package":"p","tag":"a.b","target":"cpu","variable":"GUST_ROUTE__p__a__b"}],"schema":1}"#;
    assert_eq!(
        routes_rows(text).unwrap(),
        [
            ("a".to_owned(), "gpu".to_owned(), "fallback".to_owned()),
            ("a.b".to_owned(), "cpu".to_owned(), "entry".to_owned())
        ]
    );
    assert!(routes_rows("{}").unwrap_err().contains("no `routes` array"));
    assert!(
        routes_rows(r#"{"routes":[{"tag":"a"}]}"#)
            .unwrap_err()
            .contains("no `target`")
    );
    assert!(routes_rows("[").is_err());
}

#[test]
fn tail_keeps_the_last_lines() {
    assert_eq!(tail("a\nb\nc\nd", 2), "c\nd");
    assert_eq!(tail("a", 5), "a");
}

#[test]
fn ci_runs_check_routing_in_a_job_of_its_own_with_no_gpu_and_no_lavapipe() {
    let ci = repository_file(".github/workflows/ci.yml");
    let start = ci.find("\n  routing:\n").expect(&ci);
    let rest = &ci[start + 1..];
    let end = rest
        .match_indices("\n  ")
        .find(|(index, _)| {
            rest[index + 3..]
                .chars()
                .next()
                .is_some_and(|c| c != ' ' && c != '#' && c != '\n')
        })
        .map_or(rest.len(), |(index, _)| index);
    let job = &rest[..end];
    assert!(
        job.contains("        run: cargo --locked xtask check-routing --check-only\n"),
        "{job}"
    );
    assert!(job.contains("    runs-on: ubuntu-latest\n"), "{job}");
    // The lint case runs rustfmt and clippy, so the job installs both.
    assert!(job.contains(" --component rustfmt,clippy\n"), "{job}");
    assert!(job.contains("      - uses: actions/checkout@"), "{job}");
    let code: String = job
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    for forbidden in ["lavapipe", "mesa", "vulkan", "slang"] {
        assert!(
            !code.to_lowercase().contains(forbidden),
            "{forbidden}: {job}"
        );
    }
    // One step in the whole workflow, and not in `unit`, which keeps its budget.
    let steps = ci
        .lines()
        .filter(|line| line.trim_start().starts_with("run:") && line.contains("check-routing"))
        .count();
    assert_eq!(steps, 1, "{ci}");
    let unit = ci.split("\n  msrv:").next().unwrap();
    assert!(!unit.contains("check-routing"), "{unit}");
}

/// m8.z (review r1, SF1): the plain state of the flip carries the `GUST_ROUTED_PROFILE` the driver gives, so that only the
/// route variables differ between its states. Without it the unset-to-set direction also changed the variable, which
/// made a rebuild whether or not the route tracking worked.
#[test]
fn the_flip_gives_its_plain_state_the_routed_profile_of_the_driver() {
    let source = repository_file("xtask/src/routing.rs");
    assert!(
        source.contains("Via::Unset => self.plain_probe(Some(\"dev\"))?,"),
        "the plain state of the flip must set GUST_ROUTED_PROFILE=dev"
    );
    // The plain case is plain cargo as a user runs it: no such variable.
    assert!(source.contains("let done = self.plain_probe(None)?;"));
    // The probe's other callers pass nothing.
    assert_eq!(source.matches("self.plain_probe(").count(), 2);
}

/// m8.z (review r1, SF2): the guard runs through the real driver, in `--check-only` too (it needs no GPU), with a custom
/// profile that inherits `release` and no change to the fixture's manifests.
#[test]
fn the_guard_case_runs_the_driver_with_a_profile_that_inherits_release() {
    assert!(CASES.contains(&"guard"));
    assert!(
        Options::parse(&args(&["--check-only"]))
            .unwrap()
            .cases
            .contains(&"guard")
    );
    let source = repository_file("xtask/src/routing.rs");
    let body = &source[source.find("    fn guard(&self)").expect("fn guard")..];
    let body = &body[..body.find("    /// R5").expect("the next case")];
    for needed in [
        "self.driver_command(\"run\"",
        "\"--profile\", \"dist\"",
        "CARGO_PROFILE_DIST_INHERITS\", \"release\"",
        "resolved the routes of this build for the profile `dev` (GUST_ROUTED_PROFILE)",
        "this build of gust is `native`",
        "if done.success",
    ] {
        assert!(body.contains(needed), "{needed}");
    }
    // The driver's own value wins over an inherited one, and the fixture's manifests are left alone.
    assert!(body.contains(".env(ROUTED_PROFILE_VARIABLE, \"web\")"));
    assert!(!body.contains("fs::write"));
}

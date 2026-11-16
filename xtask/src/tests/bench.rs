use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{
    ATTESTATION, DOCS_PAGE, HEADER, JUDGE_SOURCE, MIRROR_DIR, MIRROR_PREFIX, Options, RESULTS_DIR,
    block_of, check_mirrors, functional_verdict, load_results, render, render_tables,
    replace_block, sample_limits, slugify, thresholds_block,
};
use crate::workspace::workspace_root;

fn args(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gust-xtask-bench-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// An aggregate the harness would call publishable, with two points of two backends, the first carrying every mark.
fn aggregate(publishable: bool, banner: bool) -> String {
    format!(
        r#"{{
  "banner": {},
  "publishable": {publishable},
  "publishable_reason": "x",
  "route": "wgsl",
  "build": {{ "commit": "0123456789abcdef0123" }},
  "adapter": {{ "name": "Test GPU", "driver": "1.2.3", "driver_info": "build 7" }},
  "meta": {{ "os": "Test OS 1" }},
  "thresholds": {{ "version": 1 }},
  "runs": [
    {{ "run_seed": 11, "window": {{ "started_utc": "2026-11-20T08:00:00Z", "ended_utc": "2026-11-20T08:30:00Z" }} }},
    {{ "run_seed": 22, "window": {{ "started_utc": "2026-11-20T08:40:00Z", "ended_utc": "2026-11-20T09:10:00Z" }} }},
    {{ "run_seed": 33, "window": {{ "started_utc": "2026-11-20T09:20:00Z", "ended_utc": "2026-11-21T09:50:00Z" }} }}
  ],
  "above_dram_peak": [ {{ "workload": "W1", "n": 1024, "backend": "gust-gpu", "metric": "end_to_end" }} ],
  "kill_rule_reproducibility": {{ "offenders": [ {{ "workload": "W1", "n": 1024, "backend": "gust-gpu", "quantity": "end_to_end" }} ] }},
  "points": [
    {{ "workload": "W1", "n": 1024, "backend": "gust-gpu", "metric": "end_to_end", "median_ms": 0.5, "min_run_median_ms": 0.4, "max_run_median_ms": 0.6,
      "two_state_suspect": true, "samples_per_run": [30, 30, 12] }},
    {{ "workload": "W1", "n": 1024, "backend": "wgsl-prebuilt", "metric": "end_to_end", "median_ms": 12.5, "min_run_median_ms": 12.0, "max_run_median_ms": 13.0,
      "two_state_suspect": false, "samples_per_run": [30, 30, 30] }}
  ]
}}"#,
        if banner {
            "\"shared machine, not publishable\""
        } else {
            "null"
        }
    )
}

/// Files `body` as `<dir>/results/<slug>/<name>.json`, with a sidecar when asked.
fn file_as(dir: &Path, slug: &str, name: &str, body: &str, sidecar: bool) -> PathBuf {
    let folder = dir.join("results").join(slug);
    fs::create_dir_all(&folder).unwrap();
    let path = folder.join(format!("{name}.json"));
    fs::write(&path, body).unwrap();
    if sidecar {
        fs::write(folder.join(format!("{name}.samples.json")), "{}").unwrap();
    }
    path
}

fn file(dir: &Path, body: &str, sidecar: bool) -> PathBuf {
    file_as(dir, "test-gpu", "0123456789ab-wgsl", body, sidecar)
}

fn attested(_: &Path) -> Option<String> {
    Some(format!("{ATTESTATION} three runs, quiet windows"))
}

#[test]
fn options_need_exactly_one_mode() {
    assert!(Options::parse(&args(&[])).is_err());
    assert!(Options::parse(&args(&["--functional", "--render"])).is_err());
    assert!(
        Options::parse(&args(&["--functional", "--quick"]))
            .unwrap()
            .functional
    );
    assert!(Options::parse(&args(&["--functional", "--check"])).is_err());
    let options = Options::parse(&args(&["--render", "--check", "--results", "d"])).unwrap();
    assert!(options.render && options.check);
    assert_eq!(options.results.as_deref(), Some("d"));
    assert!(Options::parse(&args(&["--render", "--bogus"])).is_err());
    assert!(Options::parse(&args(&["--render", "--results"])).is_err());
}

#[test]
fn a_functional_file_passes_only_when_it_is_what_it_claims() {
    let good = r#"{"functional": true, "publishable": false, "failures": [], "adapter": {"name": "llvmpipe"},
        "cells": [{"checks_passed": 3}, {"checks_passed": 2}]}"#;
    assert_eq!(functional_verdict(good).unwrap(), "llvmpipe");
    assert!(
        functional_verdict(&good.replace("\"functional\": true", "\"functional\": false")).is_err()
    );
    assert!(
        functional_verdict(&good.replace("\"publishable\": false", "\"publishable\": true"))
            .is_err()
    );
    let failed = good.replace(
        "\"failures\": []",
        "\"failures\": [\"W1 gust-gpu: mismatch\"]",
    );
    let error = functional_verdict(&failed).unwrap_err();
    assert!(error.contains("mismatch"), "{error}");
    let none = good.replace(
        "{\"checks_passed\": 3}, {\"checks_passed\": 2}",
        "{\"checks_passed\": 0}",
    );
    assert!(
        functional_verdict(&none)
            .unwrap_err()
            .contains("no correctness check")
    );
    assert!(functional_verdict("not json").is_err());
}

#[test]
fn a_publishable_attested_aggregate_with_a_sidecar_is_rendered_with_its_conditions_and_marks() {
    let dir = scratch("render");
    file(&dir, &aggregate(true, false), true);
    let filed = load_results(&dir.join("results"), &attested).unwrap();
    assert_eq!(filed.len(), 1);
    let tables = render_tables(&filed, &limits_of(JUDGE_FIXTURE));
    assert!(
        tables.contains("### Test GPU, route wgsl, commit 0123456789ab"),
        "{tables}"
    );
    for part in [
        "Adapter Test GPU; driver 1.2.3 build 7; OS Test OS 1;",
        "3 runs from 2026-11-20 to 2026-11-21; seeds 11, 22, 33; thresholds version 1.",
        "**S** the quantity has two states",
        "**F** fewer than 30 samples",
    ] {
        assert!(tables.contains(part), "{part}: {tables}");
    }
    // The first backend carries two-state, short-sample, above-peak and offender marks; the second none.
    assert!(
        tables.contains("| 1024 | gust-gpu | 0.5000 (0.4000 to 0.6000) **SFDK** | - | - | - |"),
        "{tables}"
    );
    assert!(
        tables.contains("| 1024 | wgsl-prebuilt | 12.50 (12.00 to 13.00) | - | - | - |"),
        "{tables}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_f_mark_and_its_legend_follow_the_sample_constants_of_the_harness_source() {
    let dir = scratch("limits");
    file(&dir, &aggregate(true, false), true);
    let filed = load_results(&dir.join("results"), &attested).unwrap();
    let raised = JUDGE_FIXTURE
        .replace("SAMPLES_TARGET: usize = 30", "SAMPLES_TARGET: usize = 40")
        .replace("SAMPLE_CAP_SECS: f64 = 8.0", "SAMPLE_CAP_SECS: f64 = 9.0");
    let tables = render_tables(&filed, &limits_of(&raised));
    assert!(
        tables.contains(
            "**F** fewer than 40 samples were collected in at least one run (the 9.0 s cap stopped it)"
        ),
        "{tables}"
    );
    // 30 samples is short of a target of 40, so the second backend carries the mark too.
    assert!(
        tables.contains("| 1024 | wgsl-prebuilt | 12.50 (12.00 to 13.00) **F**"),
        "{tables}"
    );
    assert!(sample_limits("pub const SAMPLES_TARGET: usize = 30;").is_err());
    let _ = fs::remove_dir_all(&dir);
}

/// (case, publishable, banner, has a sidecar, attestation lookup, expected message)
type Refusal = (
    &'static str,
    bool,
    bool,
    bool,
    Option<&'static str>,
    &'static str,
);

#[test]
fn a_file_that_is_not_publishable_attested_or_complete_is_refused_with_its_reason() {
    let dir = scratch("refuse");
    // (case, publishable, banner, has a sidecar, attestation lookup, expected message)
    let cases: [Refusal; 5] = [
        (
            "flag",
            false,
            false,
            true,
            Some(ATTESTATION),
            "not publishable",
        ),
        ("banner", true, true, true, Some(ATTESTATION), "banner"),
        ("sidecar", true, false, false, Some(ATTESTATION), "sidecar"),
        (
            "attestation",
            true,
            false,
            true,
            Some(""),
            "carries no `Quiet-window attestation:` line",
        ),
        ("uncommitted", true, false, true, None, "no commit adds it"),
    ];
    for (name, publishable, banner, sidecar, line, expect) in cases {
        let case = dir.join(name);
        fs::create_dir_all(&case).unwrap();
        file(&case, &aggregate(publishable, banner), sidecar);
        let lookup = move |_: &Path| line.map(str::to_owned);
        let error = load_results(&case.join("results"), &lookup).unwrap_err();
        assert!(error.contains(expect), "{name}: {error}");
        assert!(error.contains("0123456789ab-wgsl.json"), "{name}: {error}");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_loose_file_and_a_path_that_disagrees_with_its_contents_are_refused() {
    let dir = scratch("paths");
    let good = aggregate(true, false);
    let loose = dir.join("loose");
    fs::create_dir_all(loose.join("results")).unwrap();
    fs::write(loose.join("results").join("run.json"), &good).unwrap();
    let error = load_results(&loose.join("results"), &attested).unwrap_err();
    assert!(
        error.contains("directly under the results directory"),
        "{error}"
    );
    for (name, slug, stem, expect) in [
        (
            "slug",
            "other-gpu",
            "0123456789ab-wgsl",
            "belongs under `test-gpu`",
        ),
        (
            "route",
            "test-gpu",
            "0123456789ab-spirv",
            "named for route `spirv`",
        ),
        (
            "commit",
            "test-gpu",
            "fedcba987654-wgsl",
            "named for commit `fedcba987654`",
        ),
        ("shape", "test-gpu", "nodash", "not <commit>-<route>.json"),
        (
            "short",
            "test-gpu",
            "0123456789-wgsl",
            "fewer than the 12 characters",
        ),
        ("tiny", "test-gpu", "0-wgsl", "fewer than the 12 characters"),
    ] {
        let case = dir.join(name);
        fs::create_dir_all(&case).unwrap();
        file_as(&case, slug, stem, &good, true);
        let error = load_results(&case.join("results"), &attested).unwrap_err();
        assert!(error.contains(expect), "{name}: {error}");
    }
    assert_eq!(
        slugify("NVIDIA GeForce RTX 5090"),
        "nvidia-geforce-rtx-5090"
    );
    assert_eq!(slugify(" Test  GPU (x) "), "test-gpu-x");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn no_results_say_none_are_filed() {
    let dir = scratch("none");
    assert!(
        load_results(&dir.join("absent"), &attested)
            .unwrap()
            .is_empty()
    );
    let text = render_tables(&[], &limits_of(JUDGE_FIXTURE));
    assert!(text.starts_with("No results are filed."), "{text}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn blocks_are_replaced_and_read_back() {
    let page = "a\n<!-- x:begin -->\nold\n<!-- x:end -->\nb\n";
    let new = replace_block(page, "x", "new").unwrap();
    assert_eq!(new, "a\n<!-- x:begin -->\nnew\n<!-- x:end -->\nb\n");
    assert_eq!(block_of(&new, "x"), Some("new"));
    assert_eq!(replace_block(&new, "x", "new").unwrap(), new);
    assert!(replace_block("no markers", "x", "n").is_err());
    assert!(replace_block("<!-- x:end -->\n<!-- x:begin -->", "x", "n").is_err());
}

/// The constants of the harness's `judge.rs`, as `thresholds_block` reads them.
const JUDGE_FIXTURE: &str = "pub const THRESHOLDS_VERSION: u32 = 1;\npub const T_GPU_PERCENT: f64 = 5.0;\n\
    pub const T_CPU_FRACTION: Option<f64> = None;\npub const N_ATTEMPTS: u32 = 5;\npub const MIN_RUNS: usize = 3;\n\
    pub const SESSION_MAX_HOURS: f64 = 12.0;\npub const SAMPLES_TARGET: usize = 30;\n\
    pub const SAMPLE_CAP_SECS: f64 = 8.0;\npub const SAMPLES_FLOOR: usize = 5;\n";

fn limits_of(judge: &str) -> super::SampleLimits {
    sample_limits(judge).unwrap()
}

#[test]
fn the_thresholds_block_is_read_from_the_harness_source() {
    let block = thresholds_block(JUDGE_FIXTURE).unwrap();
    assert!(block.contains("| 5.0 |"), "{block}");
    assert!(block.contains("not set"), "{block}");
    assert!(
        block.contains(
            "| Samples per quantity | 30 under a cap of 8.0 s per quantity, never below 5 |"
        ),
        "{block}"
    );
    assert!(block.contains("| Thresholds version | 1 |"), "{block}");
    let set = thresholds_block(&JUDGE_FIXTURE.replace("None", "Some(0.5)")).unwrap();
    assert!(set.contains("Some(0.5)"), "{set}");
    assert!(thresholds_block("pub const T_GPU_PERCENT: f64 = 5.0;").is_err());
}

fn page_with(tables: &str) -> String {
    format!(
        "# t\n\n<!-- bench-header:begin -->\n{HEADER}\n<!-- bench-header:end -->\n\n\
         <!-- bench-thresholds:begin -->\n{}\n<!-- bench-thresholds:end -->\n\n\
         <!-- bench-tables:begin -->\n{tables}\n<!-- bench-tables:end -->\n",
        thresholds_block(JUDGE_FIXTURE).unwrap()
    )
}

/// A scratch repository root: the page, the harness's constants and no results directory.
fn scratch_root(name: &str, tables: &str) -> (PathBuf, PathBuf) {
    let root = scratch(name);
    fs::create_dir_all(root.join(DOCS_PAGE).parent().unwrap()).unwrap();
    fs::create_dir_all(root.join(JUDGE_SOURCE).parent().unwrap()).unwrap();
    fs::write(root.join(JUDGE_SOURCE), JUDGE_FIXTURE).unwrap();
    fs::write(root.join(DOCS_PAGE), page_with(tables)).unwrap();
    let page = root.join(DOCS_PAGE);
    (root, page)
}

#[test]
fn check_fails_on_a_stale_tables_block_and_render_repairs_it() {
    let (root, page) = scratch_root("drift", "stale text");
    let check = Options::parse(&args(&["--render", "--check"])).unwrap();
    let error = render(&root, &check, &attested).unwrap_err();
    assert!(error.contains("out of date"), "{error}");
    assert!(
        fs::read_to_string(&page).unwrap().contains("stale text"),
        "--check writes nothing"
    );
    render(
        &root,
        &Options::parse(&args(&["--render"])).unwrap(),
        &attested,
    )
    .unwrap();
    let repaired = fs::read_to_string(&page).unwrap();
    assert!(!repaired.contains("stale text"));
    assert_eq!(
        block_of(&repaired, "bench-tables"),
        Some(render_tables(&[], &limits_of(JUDGE_FIXTURE)).trim())
    );
    render(&root, &check, &attested).unwrap();
    // A refused result file fails the check as well.
    file(&root.join("bench"), &aggregate(false, true), true);
    let error = render(&root, &check, &attested).unwrap_err();
    assert!(error.contains("not publishable"), "{error}");
    let _ = fs::remove_dir_all(&root);
}

/// The checked-in page: its header and thresholds blocks are the canonical ones and its tables block is what the
/// renderer makes of `bench/results` (empty today), so a drift between the page, the header in this module, the
/// harness's constants and the filed results fails here.
#[test]
fn the_checked_in_page_matches_the_header_the_thresholds_and_the_filed_results() {
    let root = workspace_root().expect("the repository root");
    let page = fs::read_to_string(root.join(DOCS_PAGE))
        .expect("docs/BENCHMARKS.md")
        .replace("\r\n", "\n");
    assert_eq!(block_of(&page, "bench-header"), Some(HEADER.trim()));
    let judge = fs::read_to_string(root.join(JUDGE_SOURCE)).expect("the harness's judge.rs");
    assert_eq!(
        block_of(&page, "bench-thresholds"),
        Some(thresholds_block(&judge).unwrap().trim())
    );
    let filed = load_results(&root.join(RESULTS_DIR), &attested).expect("the filed results load");
    assert_eq!(
        block_of(&page, "bench-tables"),
        Some(render_tables(&filed, &limits_of(&judge)).trim())
    );
    assert_eq!(
        check_mirrors(&root).expect("the guide pages are readable"),
        Vec::<String>::new(),
        "a guide page that mirrors the header differs from it"
    );
    // check_mirrors finds nothing to check when the chapter is renamed or removed; require the chapter itself.
    assert!(
        root.join(MIRROR_DIR)
            .join(format!("{MIRROR_PREFIX}.md"))
            .is_file(),
        "the guide chapter that mirrors the header is missing"
    );
    assert!(HEADER.contains("not in its authors' recommended configuration"));
    assert!(HEADER.contains("corrections are welcome as issues against this repository"));
}

/// The aggregate of `aggregate()` as the harness writes it for another route and group.
fn aggregate_of(route: &str, group: &str, backend: &str, reference: Option<&str>) -> String {
    let rule = reference.map_or(String::new(), |reference| {
        format!(r#", "kill_rule": {{ "group": "{group}", "reference": "{reference}" }}"#)
    });
    let mut text = aggregate(true, false)
        .replacen(
            "\"route\": \"wgsl\",",
            &format!("\"route\": \"{route}\", \"group\": \"{group}\","),
            1,
        )
        .replace(
            "\"backend\": \"gust-gpu\"",
            &format!("\"backend\": \"{backend}\""),
        );
    // The rule goes into every run: a timed spirv run must carry one.
    for seed in [11, 22, 33] {
        text = text.replacen(
            &format!("{{ \"run_seed\": {seed},"),
            &format!("{{ \"run_seed\": {seed}, \"publishable\": true{rule},"),
            1,
        );
    }
    if group == "spirv" {
        // The group's own reference beside the GUST row, as the harness writes it.
        text = text.replace(
            "\"backend\": \"wgsl-prebuilt\"",
            "\"backend\": \"wgsl-prebuilt-unchecked\"",
        );
    }
    text
}

/// D193: the `spirv` group is a section of its own, labelled "checks off on both sides", with its rows apart from the
/// publishable group's; a legacy file (the SPIR-V route in the publishable group) is read and labelled, not refused.
#[test]
fn the_spirv_group_is_its_own_labelled_section_and_a_pre_d193_file_is_legacy_not_refused() {
    let dir = scratch("spirv-group");
    file_as(
        &dir,
        "test-gpu",
        "0123456789ab-spirv",
        &aggregate_of(
            "spirv",
            "spirv",
            "gust-gpu-spirv",
            Some("wgsl-prebuilt-unchecked"),
        ),
        true,
    );
    file(&dir, &aggregate(true, false), true);
    let filed = load_results(&dir.join("results"), &attested).unwrap();
    assert_eq!(filed.len(), 2);
    let tables = render_tables(&filed, &limits_of(JUDGE_FIXTURE));
    assert!(
        tables.contains(
            "### Test GPU, route spirv, group spirv (checks off on both sides), commit 0123456789ab"
        ),
        "{tables}"
    );
    assert!(
        tables.contains("Group `spirv` (D193): checks off on both sides."),
        "{tables}"
    );
    assert!(
        tables.contains("### Test GPU, route wgsl, commit 0123456789ab"),
        "{tables}"
    );
    // The spirv section holds the spirv rows and the publishable one does not.
    let spirv_at = tables.find("route spirv").unwrap();
    let wgsl_at = tables.find("route wgsl").unwrap();
    let (first, second) = if spirv_at < wgsl_at {
        (&tables[spirv_at..wgsl_at], &tables[wgsl_at..])
    } else {
        (&tables[wgsl_at..spirv_at], &tables[spirv_at..])
    };
    let (spirv, wgsl) = if spirv_at < wgsl_at {
        (first, second)
    } else {
        (second, first)
    };
    assert!(spirv.contains("| gust-gpu-spirv |") && !wgsl.contains("gust-gpu-spirv"));
    let _ = fs::remove_dir_all(&dir);

    // Legacy: the SPIR-V route's file in the publishable group, and a publishable file with a spirv point.
    let dir = scratch("legacy");
    file_as(
        &dir,
        "test-gpu",
        "0123456789ab-spirv",
        &aggregate_of("spirv", "publishable", "gust-gpu-spirv", None),
        true,
    );
    let filed = load_results(&dir.join("results"), &attested).unwrap();
    assert!(filed[0].legacy);
    let tables = render_tables(&filed, &limits_of(JUDGE_FIXTURE));
    assert!(
        tables.contains("route spirv, legacy (pre-D193), commit"),
        "{tables}"
    );
    assert!(tables.contains("Legacy file (pre-D193)"), "{tables}");
    let _ = fs::remove_dir_all(&dir);
    let dir = scratch("legacy-point");
    file(
        &dir,
        &aggregate_of("wgsl", "publishable", "gust-gpu-spirv", None),
        true,
    );
    assert!(load_results(&dir.join("results"), &attested).unwrap()[0].legacy);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_spirv_group_file_must_come_from_the_spirv_route_with_its_own_rule_and_other_groups_are_not_filed()
 {
    for (name, body, reason) in [
        (
            "0123456789ab-wgsl",
            aggregate_of("wgsl", "spirv", "gust-gpu", None),
            "the spirv group is the SPIR-V route's",
        ),
        (
            "0123456789ab-spirv",
            aggregate_of("spirv", "spirv", "gust-gpu-spirv", Some("wgsl-prebuilt")),
            "not the spirv group's",
        ),
        (
            "0123456789ab-wgsl",
            aggregate_of("wgsl", "ablation", "gust-gpu", None),
            "not a table group",
        ),
        // S3: a spirv file holds only its own rows.
        (
            "0123456789ab-spirv",
            aggregate_of(
                "spirv",
                "spirv",
                "gust-gpu-spirv",
                Some("wgsl-prebuilt-unchecked"),
            )
            .replace(
                "wgsl-prebuilt-unchecked\", \"metric",
                "wgsl-prebuilt\", \"metric",
            ),
            "not the spirv group's (gust-gpu-spirv",
        ),
        (
            "0123456789ab-spirv",
            aggregate_of(
                "spirv",
                "spirv",
                "gust-gpu",
                Some("wgsl-prebuilt-unchecked"),
            ),
            "`gust-gpu`, which is not the spirv group's",
        ),
        // A publishable file holds no checks-off row.
        (
            "0123456789ab-wgsl",
            aggregate_of("wgsl", "publishable", "gust-gpu", None)
                .replace("\"wgsl-prebuilt\"", "\"wgsl-prebuilt-unchecked\""),
            "checks-off row `wgsl-prebuilt-unchecked`",
        ),
        // A spirv file with no runs, or a timed run without a record, is refused.
        (
            "0123456789ab-spirv",
            {
                let body = aggregate_of(
                    "spirv",
                    "spirv",
                    "gust-gpu-spirv",
                    Some("wgsl-prebuilt-unchecked"),
                );
                let start = body.find("\"runs\": [").unwrap();
                let end = body[start..].find("],").unwrap() + start;
                format!("{}\"runs\": []{}", &body[..start], &body[end + 1..])
            },
            "has no runs",
        ),
        (
            "0123456789ab-spirv",
            aggregate_of("spirv", "spirv", "gust-gpu-spirv", None),
            "run of a spirv file has a null kill rule",
        ),
    ] {
        let dir = scratch("bad-group");
        file_as(&dir, "test-gpu", name, &body, true);
        let error = load_results(&dir.join("results"), &attested).unwrap_err();
        assert!(error.contains(reason), "{reason}: {error}");
        let _ = fs::remove_dir_all(&dir);
    }
}

/// R2: functional runs are never filed, and a null rule or an unpublishable run in a spirv file is refused outright.
#[test]
fn a_spirv_file_of_functional_or_unpublishable_runs_is_refused() {
    let ruled = aggregate_of(
        "spirv",
        "spirv",
        "gust-gpu-spirv",
        Some("wgsl-prebuilt-unchecked"),
    );
    for (body, reason) in [
        (
            ruled.replace("{ \"run_seed\":", "{ \"functional\": true, \"run_seed\":"),
            "holds a functional run",
        ),
        (
            ruled.replace("\"publishable\": true", "\"publishable\": false"),
            "not publishable",
        ),
        // F2 (b3d re-check): a missing or null per-run `publishable` is refused for a spirv file as well.
        (
            ruled.replace(", \"publishable\": true", ""),
            "not publishable",
        ),
        (
            ruled.replace("\"publishable\": true", "\"publishable\": null"),
            "not publishable",
        ),
        (
            aggregate_of("spirv", "spirv", "gust-gpu-spirv", None).replace(
                "{ \"run_seed\":",
                "{ \"functional\": false, \"kill_rule\": null, \"run_seed\":",
            ),
            "null kill rule",
        ),
    ] {
        let dir = scratch("spirv-functional");
        file_as(&dir, "test-gpu", "0123456789ab-spirv", &body, true);
        let error = load_results(&dir.join("results"), &attested).unwrap_err();
        assert!(error.contains(reason), "{reason}: {error}");
        let _ = fs::remove_dir_all(&dir);
    }
}

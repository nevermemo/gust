use std::path::Path;

use super::{
    DEFAULT_SEEDS, DifftestStep, FULL_SHAPES, Mode, Options, PROGRAMS_PER_SEED,
    SHORT_CIRCUIT_GPU_TEST, batch_args, judge_batch, parse_report, parse_seeds, parse_shapes,
    random_seeds, report_problems, seed_list, short_circuit_args, short_circuit_problems, steps,
    with_full_shapes,
};

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

fn options(args: &[&str]) -> Options {
    Options::parse_with(&owned(args), 7).unwrap()
}

/// A repository file with LF endings.
fn repo_file(text: &str) -> String {
    text.replace("\r\n", "\n")
}

// ---- seeds and options ----

#[test]
fn the_default_is_the_whole_corpus_on_the_adapter() {
    let options = options(&[]);
    assert_eq!(options.seeds, (1..=8).collect::<Vec<u64>>());
    assert_eq!(options.mode, Mode::Gpu);
    assert_eq!(options.report, None);
    assert!(!options.random);
    assert_eq!(options.expected_programs(), 128);
}

#[test]
fn the_default_seeds_are_the_seeds_of_the_corpus_file() {
    let corpus = repo_file(include_str!("../../../tests/difftest/seeds.txt"));
    let mut seeds = Vec::new();
    for line in corpus.lines().filter(|line| !line.starts_with('#')) {
        let mut columns = line.split_whitespace();
        seeds.push(columns.next().unwrap().parse::<u64>().unwrap());
        // The second column is the programs kept per seed: the constant this command multiplies by.
        assert_eq!(
            columns.next().unwrap().parse::<u64>().unwrap(),
            PROGRAMS_PER_SEED,
            "{line}"
        );
    }
    assert_eq!(parse_seeds(DEFAULT_SEEDS).unwrap(), seeds);
}

#[test]
fn the_unit_batch_is_four_seeds_cpu_only_and_the_adapter_batch_is_eight() {
    let unit = options(&["--cpu-only", "--seeds", "1-4"]);
    assert_eq!(unit.seeds, [1, 2, 3, 4]);
    assert_eq!(unit.mode, Mode::CpuOnly);
    assert_eq!(unit.expected_programs(), 64);
    let per_pull_request = options(&["--seeds", "1-2"]);
    assert_eq!(per_pull_request.mode, Mode::Gpu);
    assert_eq!(per_pull_request.expected_programs(), 32);
}

#[test]
fn seeds_parse_as_the_harness_parses_them() {
    assert_eq!(parse_seeds("3").unwrap(), [3]);
    assert_eq!(parse_seeds("1,3,5-6").unwrap(), [1, 3, 5, 6]);
    assert_eq!(parse_seeds("6,1,1,2-3").unwrap(), [1, 2, 3, 6]);
    assert_eq!(parse_seeds(" 2 - 4 ").unwrap(), [2, 3, 4]);
    for bad in ["", "x", "4-1", "1-", "-1", "1,,2", "1-5000", "random:2"] {
        assert!(parse_seeds(bad).is_err(), "{bad:?}");
    }
    assert!(parse_seeds("0-4095").is_ok());
    assert!(parse_seeds("0-4096").is_err());
    assert_eq!(seed_list(&[1, 2, 10]), "1,2,10");
}

#[test]
fn random_seeds_are_distinct_sorted_reproducible_and_clear_of_the_corpus() {
    let drawn = random_seeds("5", 42).unwrap();
    assert_eq!(drawn.len(), 5);
    assert!(drawn.windows(2).all(|pair| pair[0] < pair[1]), "{drawn:?}");
    assert!(
        drawn
            .iter()
            .all(|seed| (10_000..1_000_010_000).contains(seed)),
        "{drawn:?}"
    );
    assert_eq!(drawn, random_seeds("5", 42).unwrap());
    assert_ne!(drawn, random_seeds("5", 43).unwrap());
    let parsed = Options::parse_with(&owned(&["--seeds", "random:5"]), 42).unwrap();
    assert!(parsed.random);
    assert_eq!(parsed.seeds, drawn);
    assert_eq!(parsed.expected_programs(), 80);
    // The 64 allowed seeds are 64 different numbers even when the generator is asked for all of them.
    assert_eq!(random_seeds("64", 0).unwrap().len(), 64);
    for bad in ["0", "65", "-1", "x", "", "2.5"] {
        assert!(random_seeds(bad, 1).is_err(), "random:{bad}");
    }
}

#[test]
fn arguments_are_checked() {
    for bad in [
        &["--seeds"][..],
        &["--seeds", "1", "--seeds", "2"],
        &["--report"],
        &["--report", ""],
        &["--report", "a", "--report", "b"],
        &["--seeds", "random:2,5"],
        &["--shrink", "1:0"],
        &["--gpu"],
        &["1-4"],
    ] {
        assert!(Options::parse_with(&owned(bad), 1).is_err(), "{bad:?}");
    }
    let kept = options(&["--report", "out/report.json", "--cpu-only"]);
    assert_eq!(
        kept.report.as_deref(),
        Some(Path::new("out/report.json")),
        "{kept:?}"
    );
    assert_eq!(kept.mode, Mode::CpuOnly);
}

// ---- the step list ----

/// Mutation (c): the Gpu tier runs the batch and then the `short_circuit` GPU test, and only the Gpu tier does.
#[test]
fn the_gpu_tier_runs_the_batch_and_the_short_circuit_gpu_test() {
    assert_eq!(
        steps(Mode::Gpu),
        [DifftestStep::Batch, DifftestStep::ShortCircuitGpu]
    );
    assert_eq!(steps(Mode::CpuOnly), [DifftestStep::Batch]);
}

#[test]
fn the_short_circuit_call_runs_the_ignored_gpu_test_of_the_harness_crate() {
    let args = short_circuit_args();
    assert_eq!(
        args,
        owned(&[
            "test",
            "--locked",
            "--manifest-path",
            "tests/difftest/Cargo.toml",
            "--test",
            "short_circuit",
            "--",
            "--ignored"
        ])
    );
    // The name this command requires to be reported `ok` is a real test of that file, and it is the ignored one.
    let source = repo_file(include_str!(
        "../../../tests/difftest/tests/short_circuit.rs"
    ));
    let at = source
        .find(&format!("fn {SHORT_CIRCUIT_GPU_TEST}()"))
        .expect("the GPU test exists under that name");
    let attributes = &source[..at];
    let attributes = &attributes[attributes.rfind("\n\n").unwrap_or(0)..];
    assert!(attributes.contains("#[test]") && attributes.contains("#[ignore"));
}

#[test]
fn the_batch_call_is_locked_and_names_the_seeds_the_mode_the_report_and_the_target() {
    let args = batch_args(
        &options(&["--seeds", "1,3-4", "--cpu-only"]),
        Path::new("T"),
        Path::new("R.json"),
    );
    assert_eq!(
        args,
        owned(&[
            "run",
            "--locked",
            "--manifest-path",
            "tests/difftest/Cargo.toml",
            "--bin",
            "difftest-run",
            "--",
            "--seeds",
            "1,3,4",
            "--cpu-only",
            "--report",
            "R.json",
            "--target-dir",
            "T"
        ])
    );
    let gpu = batch_args(&options(&[]), Path::new("T"), Path::new("R.json"));
    assert!(!gpu.contains(&"--cpu-only".to_owned()));
    assert!(gpu.contains(&"1,2,3,4,5,6,7,8".to_owned()));
}

// ---- the report ----

/// A report with the keys and shapes `difftest-run` writes (`harness::build_report`), `programs` kept per seed.
fn report_json(
    mode: &str,
    seeds: &[u64],
    executed: u64,
    programs: u64,
    adapter: Option<&str>,
    mismatches: &[(u64, &str)],
    passed: bool,
) -> String {
    let list = |items: Vec<String>| items.join(",");
    let adapter = adapter.map_or("null".to_owned(), |name| format!("{name:?}"));
    let per_seed = list(
        seeds
            .iter()
            .map(|seed| {
                format!(
                    "{{\"seed\":{seed},\"programs\":{programs},\"attempts\":40,\"discarded\":24}}"
                )
            })
            .collect(),
    );
    let found = list(
        mismatches
            .iter()
            .map(|(seed, kind)| {
                format!(
                    "{{\"program\":\"k_{seed}_3\",\"seed\":{seed},\"index\":3,\"attempt\":9,\"kind\":\"{kind}\",\
                     \"elements_differing\":2,\"first_element\":5,\
                     \"message\":\"element 5: gpu 1, twin 0\\nsecond line\"}}"
                )
            })
            .collect(),
    );
    format!(
        "{{\"tool\":\"difftest-run\",\"format\":3,\"mode\":\"{mode}\",\"passed\":{passed},\"seeds\":[{}],\
         \"programs_requested\":{},\"programs_executed\":{executed},\"adapter\":{adapter},\"per_seed\":[{per_seed}],\
         \"discards\":{{\"total\":{},\"by_class\":{{\"domain\":20,\"comparison-overlap\":1,\"unboundable\":3,\
         \"float-construct-not-in-v1\":0,\"no-input-read\":0}},\"by_reason\":{{\"overflow\":20}}}},\
         \"mismatch_count\":{},\"mismatches\":[{found}],\"coverage\":{{}},\
         \"timing\":{{\"generation_ms\":1.5,\"cargo_build_s\":0.8,\"cargo_run_s\":23.0}}}}",
        list(seeds.iter().map(u64::to_string).collect()),
        PROGRAMS_PER_SEED * seeds.len() as u64,
        24 * seeds.len(),
        mismatches.len()
    )
}

fn good(mode: &str, seeds: &[u64]) -> String {
    let executed = PROGRAMS_PER_SEED * seeds.len() as u64;
    let adapter = (mode == "gpu").then_some("NVIDIA GeForce RTX 5090, DiscreteGpu, Vulkan, 610.88");
    report_json(mode, seeds, executed, PROGRAMS_PER_SEED, adapter, &[], true)
}

fn problems(options: &Options, exit: Option<i32>, report: Result<String, String>) -> Vec<String> {
    judge_batch(options, exit, report).problems
}

#[test]
fn a_report_is_read_by_its_fields() {
    let report = parse_report(&good("gpu", &[1, 2])).unwrap();
    assert!(report.passed);
    assert_eq!(report.mode, "gpu");
    assert_eq!(report.seeds, [1, 2]);
    assert_eq!(
        (report.programs_requested, report.programs_executed),
        (32, 32)
    );
    assert_eq!(
        report.adapter.as_deref(),
        Some("NVIDIA GeForce RTX 5090, DiscreteGpu, Vulkan, 610.88")
    );
    assert_eq!(report.per_seed, [(1, 16, 40, 24), (2, 16, 40, 24)]);
    assert_eq!(report.discards_total, 48);
    assert_eq!(report.discards_by_class.len(), 5);
    assert_eq!(report.mismatch_count, 0);
    assert!((report.cargo_run_s - 23.0).abs() < 1e-9);
    // Fields, not bytes: other whitespace between the members reads the same.
    let spaced = good("gpu", &[1, 2])
        .replace(",\"", " ,\n  \"")
        .replace("\":", "\" : ");
    assert_eq!(parse_report(&spaced).unwrap(), report);
}

#[test]
fn a_report_that_cannot_be_read_proves_nothing() {
    let report = good("gpu", &[1]);
    for broken in [
        String::new(),
        "[]".to_owned(),
        report.replace("\"difftest-run\"", "\"other\""),
        report.replace("\"programs_executed\"", "\"programs_done\""),
        report.replace("\"programs_executed\":16", "\"programs_executed\":\"16\""),
        report.replace("\"programs_executed\":16", "\"programs_executed\":16.5"),
        report.replace("\"passed\":true", "\"passed\":\"yes\""),
        report.replace("\"adapter\":\"NVIDIA", "\"adapter\":7,\"x\":\"NVIDIA"),
        report.replace("\"timing\"", "\"timings\""),
        format!("{report} trailing"),
        report[..report.len() / 2].to_owned(),
    ] {
        assert!(parse_report(&broken).is_err(), "{broken}");
        let found = problems(&options(&["--seeds", "1"]), Some(0), Ok(broken));
        assert!(found[0].starts_with("no usable report"), "{found:?}");
    }
    let unreadable = problems(&options(&[]), Some(0), Err("report.json: gone".into()));
    assert!(unreadable[0].contains("gone"), "{unreadable:?}");
}

#[test]
fn a_report_that_matches_the_request_passes() {
    for (args, mode, seeds) in [
        (
            &["--cpu-only", "--seeds", "1-4"][..],
            "cpu-only",
            &[1u64, 2, 3, 4][..],
        ),
        (&["--seeds", "1-2"], "gpu", &[1, 2]),
        (&[], "gpu", &[1, 2, 3, 4, 5, 6, 7, 8]),
    ] {
        let judgement = judge_batch(&options(args), Some(0), Ok(good(mode, seeds)));
        assert!(judgement.problems.is_empty(), "{:?}", judgement.problems);
        let text = judgement.summary.join("\n");
        assert!(text.contains(&format!("mode {mode}")), "{text}");
        assert!(
            text.contains(&format!("programs_executed {}", 16 * seeds.len())),
            "{text}"
        );
        assert!(
            text.contains("seed 1: 16 programs from 40 attempts (24 discarded)"),
            "{text}"
        );
        assert!(text.contains("cargo run 23.0 s"), "{text}");
    }
}

/// Mutation (a): a harness that reports one program fewer fails, and the message names the count.
#[test]
fn one_program_fewer_fails_and_names_the_count() {
    let options = options(&["--seeds", "1-2"]);
    let fewer = report_json(
        "gpu",
        &[1, 2],
        31,
        PROGRAMS_PER_SEED,
        Some("lavapipe"),
        &[],
        true,
    );
    let found = problems(&options, Some(0), Ok(fewer));
    assert!(
        found
            .iter()
            .any(|p| p == "programs_executed is 31, expected 16 x 2 seeds = 32"),
        "{found:?}"
    );
    // Even a report that claims a pass and names one short seed is caught per seed.
    let short_seed = report_json("gpu", &[1, 2], 31, 15, Some("lavapipe"), &[], true);
    let found = problems(&options, Some(0), Ok(short_seed));
    assert!(
        found
            .iter()
            .any(|p| p == "seed 1: 15 programs, expected 16"),
        "{found:?}"
    );
    assert!(
        found.iter().any(|p| p.contains("programs_executed is 31")),
        "{found:?}"
    );
    // More than asked for fails too: the count must be equal, not a lower bound.
    let more = report_json(
        "gpu",
        &[1, 2],
        33,
        PROGRAMS_PER_SEED,
        Some("lavapipe"),
        &[],
        true,
    );
    assert!(
        problems(&options, Some(0), Ok(more))
            .iter()
            .any(|p| p.contains("programs_executed is 33, expected 16 x 2 seeds = 32"))
    );
    // The requested count is checked too: a harness that planned fewer than asked.
    let planned =
        good("gpu", &[1, 2]).replace("\"programs_requested\":32", "\"programs_requested\":16");
    assert!(
        problems(&options, Some(0), Ok(planned))
            .iter()
            .any(|p| p.contains("programs_requested is 16"))
    );
    // And the per-seed table has to add up to the executed count.
    let table = good("gpu", &[1, 2]).replace(
        "\"programs\":16,\"attempts\":40,\"discarded\":24}]",
        "\"programs\":9,\"attempts\":40,\"discarded\":24}]",
    );
    assert!(
        problems(&options, Some(0), Ok(table))
            .iter()
            .any(|p| p.contains("per_seed holds 25 programs but programs_executed is 32")),
    );
    // And it has to name the seeds that ran, one entry each: the counts alone add up in all of these.
    let first = "{\"seed\":1,\"programs\":16,\"attempts\":40,\"discarded\":24}";
    let second = first.replace("\"seed\":1", "\"seed\":2");
    let table = format!("\"per_seed\":[{first},{second}]");
    assert!(good("gpu", &[1, 2]).contains(&table));
    for (label, wrong) in [
        ("another seed", table.replace("\"seed\":2", "\"seed\":3")),
        ("a seed twice", table.replace("\"seed\":2", "\"seed\":1")),
        (
            "the seeds swapped",
            format!("\"per_seed\":[{second},{first}]"),
        ),
    ] {
        let report = good("gpu", &[1, 2]).replace(&table, &wrong);
        let found = problems(&options, Some(0), Ok(report));
        assert!(
            found
                .iter()
                .any(|p| p.starts_with("per_seed names seeds [")),
            "{label}: {found:?}"
        );
    }
}

/// Mutation (b): one mismatch in the report fails and names the seed, the program and the kind.
#[test]
fn one_mismatch_fails_and_names_the_seed() {
    let options = options(&["--seeds", "1-4", "--cpu-only"]);
    let report = report_json(
        "cpu-only",
        &[1, 2, 3, 4],
        64,
        PROGRAMS_PER_SEED,
        None,
        &[(3, "twin-vs-interpreter")],
        false,
    );
    let found = problems(&options, Some(1), Ok(report));
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(
        found[0],
        "seed 3 program k_3_3 (index 3, attempt 9): twin-vs-interpreter: 2 element(s) differ: element 5: gpu 1, twin 0"
    );
    // Exit status 1 with a report that lists nothing is a failure of its own.
    let silent = problems(&options, Some(1), Ok(good("cpu-only", &[1, 2, 3, 4])));
    assert_eq!(silent, ["difftest-run exited with status 1"]);
    // A passing exit status does not excuse a mismatch in the report, and the count has to match the list.
    let report = report_json(
        "gpu",
        &[1],
        16,
        16,
        Some("a"),
        &[(1, "gpu-vs-interpreter"), (1, "gpu-vs-twin")],
        true,
    );
    let found = problems(&self::options(&["--seeds", "1"]), Some(0), Ok(report));
    assert_eq!(found.len(), 2, "{found:?}");
    let lying = good("gpu", &[1]).replace("\"mismatch_count\":0", "\"mismatch_count\":1");
    let found = problems(&self::options(&["--seeds", "1"]), Some(0), Ok(lying));
    assert!(
        found
            .iter()
            .any(|p| p.contains("mismatch_count is 1 but the report lists 0")),
        "{found:?}"
    );
    let flag = good("gpu", &[1]).replace("\"passed\":true", "\"passed\":false");
    let found = problems(&self::options(&["--seeds", "1"]), Some(1), Ok(flag));
    assert_eq!(found, ["the report says passed: false and names no cause"]);
}

#[test]
fn the_report_must_be_for_the_mode_the_seeds_and_the_adapter_that_were_asked_for() {
    let gpu = options(&["--seeds", "1-2"]);
    // A cpu-only report cannot stand in for a gpu run, and a gpu report cannot stand in for a cpu-only one.
    let found = problems(&gpu, Some(0), Ok(good("cpu-only", &[1, 2])));
    assert!(
        found
            .iter()
            .any(|p| p.contains("mode `cpu-only`, this run asked for `gpu`")),
        "{found:?}"
    );
    assert!(
        found.iter().any(|p| p == "a gpu report names no adapter"),
        "{found:?}"
    );
    let found = problems(
        &options(&["--seeds", "1-2", "--cpu-only"]),
        Some(0),
        Ok(good("gpu", &[1, 2])),
    );
    assert!(
        found
            .iter()
            .any(|p| p.starts_with("a cpu-only report names the adapter")),
        "{found:?}"
    );
    assert!(
        found
            .iter()
            .any(|p| p.contains("mode `gpu`, this run asked for `cpu-only`")),
        "{found:?}"
    );
    // Other seeds than asked for.
    let found = problems(&gpu, Some(0), Ok(good("gpu", &[2, 3])));
    assert!(
        found
            .iter()
            .any(|p| p == "the report ran seeds [2,3], this run asked for [1,2]"),
        "{found:?}"
    );
    // A gpu report without an adapter, or with an empty name, is not GPU evidence.
    let no_adapter = report_json("gpu", &[1, 2], 32, 16, None, &[], true);
    assert_eq!(
        problems(&gpu, Some(0), Ok(no_adapter)),
        ["a gpu report names no adapter"]
    );
    let empty = report_json("gpu", &[1, 2], 32, 16, Some(" "), &[], true);
    assert_eq!(
        problems(&gpu, Some(0), Ok(empty)),
        ["a gpu report names an empty adapter"]
    );
}

#[test]
fn the_exit_status_of_the_harness_counts_as_much_as_its_report() {
    let options = options(&["--seeds", "1"]);
    let report = good("gpu", &[1]);
    assert!(problems(&options, Some(0), Ok(report.clone())).is_empty());
    let found = problems(&options, Some(2), Ok(report.clone()));
    assert_eq!(found.len(), 1);
    assert!(
        found[0].contains("status 2") && found[0].contains("the harness failed"),
        "{found:?}"
    );
    assert_eq!(
        problems(&options, Some(1), Ok(report.clone())),
        ["difftest-run exited with status 1"]
    );
    assert_eq!(
        problems(&options, None, Ok(report)),
        ["difftest-run was stopped by a signal"]
    );
    // Status 2 with no report: both findings.
    let found = problems(&options, Some(2), Err("report.json: no such file".into()));
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(
        found[0].starts_with("no usable report") && found[1].contains("status 2"),
        "{found:?}"
    );
}

#[test]
fn report_problems_is_empty_for_a_good_report_and_does_not_depend_on_the_summary() {
    let report = parse_report(&good("cpu-only", &[1, 2, 3, 4])).unwrap();
    assert!(report_problems(&report, &options(&["--cpu-only", "--seeds", "1-4"])).is_empty());
}

// ---- the short_circuit run ----

fn cargo_test_output(name: &str, result: &str, summary: &str) -> String {
    format!(
        "\nrunning 1 test\ntest {name} ... {result}\n\ntest result: {summary}; 0 measured; 2 filtered out; \
         finished in 3.11s\n\n"
    )
}

#[test]
fn the_short_circuit_run_must_report_the_gpu_test_ok() {
    let args = short_circuit_args();
    let ok = cargo_test_output(
        SHORT_CIRCUIT_GPU_TEST,
        "ok",
        "ok. 1 passed; 0 failed; 0 ignored",
    );
    assert!(short_circuit_problems(&args, &ok).is_empty());
    // Colour codes (CI forces colour) do not hide the `ok`.
    let coloured = ok.replace("... ok", "... \u{1b}[32mok\u{1b}[0m");
    assert!(short_circuit_problems(&args, &coloured).is_empty());
    // The test filtered out or ignored again: nothing ran.
    let none = "\nrunning 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; \
                finished in 0.00s\n";
    let found = short_circuit_problems(&args, none);
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(found[0].contains("ran 0 tests"), "{found:?}");
    assert!(found[1].contains(SHORT_CIRCUIT_GPU_TEST), "{found:?}");
    // Another test ran and passed, the GPU one did not run.
    let other = cargo_test_output(
        "the_twin_skips_an_out_of_bounds_right_operand",
        "ok",
        "ok. 1 passed; 0 failed; 0 ignored",
    );
    let found = short_circuit_problems(&args, &other);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains("was not reported ok"), "{found:?}");
    // A failure.
    let failed = cargo_test_output(
        SHORT_CIRCUIT_GPU_TEST,
        "FAILED",
        "FAILED. 0 passed; 1 failed; 0 ignored",
    );
    let found = short_circuit_problems(&args, &failed);
    assert!(
        found.iter().any(|p| p.contains("1 test(s) failed")),
        "{found:?}"
    );
    assert!(
        found.iter().any(|p| p.contains("was not reported ok")),
        "{found:?}"
    );
}

// ---- shapes (slice mx.p1c) ----

#[test]
fn shapes_name_grid_and_each_shape_once() {
    assert_eq!(parse_shapes("grid"), Ok(false));
    assert_eq!(parse_shapes("grid,pixel"), Ok(true));
    assert_eq!(parse_shapes("pixel,grid"), Ok(true));
    assert_eq!(parse_shapes(FULL_SHAPES), Ok(true));
    for bad in [
        "pixel",
        "",
        "grid,grid",
        "grid,pixel,pixel",
        "grid,cpu",
        "Grid",
    ] {
        assert!(parse_shapes(bad).is_err(), "{bad:?}");
    }
    assert!(options(&["--shapes", "grid,pixel"]).pixel);
    assert!(!options(&["--shapes", "grid"]).pixel);
    for bad in [
        &["--shapes"][..],
        &["--shapes", "grid", "--shapes", "grid,pixel"],
        &["--shapes", "pixel"],
    ] {
        assert!(Options::parse_with(&owned(bad), 1).is_err(), "{bad:?}");
    }
}

/// The command, the Gpu suite and ci.yml stay grid-only; only the Full mode of `verify` and one step of the
/// `gpu-software` workflow (seeds 1-2, on both of its lists) ask for pixel.
#[test]
fn only_the_full_mode_asks_for_pixel_by_default() {
    // The command line: no `--shapes`, the call the harness gets is the one it always got.
    assert!(!options(&[]).pixel);
    let plain = batch_args(&options(&[]), Path::new("T"), Path::new("R.json"));
    assert!(!plain.iter().any(|arg| arg == "--shapes"), "{plain:?}");
    // `verify --mode full` reaches `check_difftest_report(&[])`, which adds the shapes; named shapes are kept.
    let full = Options::parse_with(&with_full_shapes(&[]), 1).unwrap();
    assert!(full.pixel);
    assert_eq!(full.mode, Mode::Gpu);
    assert_eq!(full.expected_programs(), 128);
    let named = with_full_shapes(&owned(&["--shapes", "grid"]));
    assert!(!Options::parse_with(&named, 1).unwrap().pixel);
    // The call carries the shapes to the harness, once, before the mode flag.
    let args = batch_args(&full, Path::new("T"), Path::new("R.json"));
    let at = args.iter().position(|arg| arg == "--shapes").unwrap();
    assert_eq!(args[at + 1], "grid,pixel");
    assert_eq!(args.iter().filter(|arg| *arg == "--shapes").count(), 1);
    // `verify` is the one caller of `check_difftest_report`, and it passes no arguments.
    let verify = repo_file(include_str!("../verify.rs"));
    assert_eq!(verify.matches("check_difftest_report(&[])").count(), 1);
    // And `check_difftest_report` hands the harness the arguments with the shapes added, not the bare ones: the only other
    // caller of `difftest_summary` is the command line, which passes `args` as typed.
    let source = repo_file(include_str!("../difftest.rs"));
    assert_eq!(
        source
            .matches("difftest_summary(&with_full_shapes(args))")
            .count(),
        1
    );
    assert_eq!(source.matches("difftest_summary(args)").count(), 1);
    let report_fn = source
        .split("pub(crate) fn check_difftest_report")
        .nth(1)
        .unwrap();
    let body = report_fn
        .split(
            "
}
",
        )
        .next()
        .unwrap();
    assert!(
        body.contains("difftest_summary(&with_full_shapes(args))"),
        "{body}"
    );
    // ci.yml never runs the shapes. gpu-software runs them in exactly one step (slice mx.z): seeds 1-2 with
    // `--shapes grid,pixel`, a step with no `if:`, so both of its lists have it. Its plain difftest step stays
    // grid-only, and a comment may name the flag (only command lines count).
    assert!(
        !repo_file(include_str!("../../../.github/workflows/ci.yml")).contains("--shapes"),
        "ci.yml runs --shapes"
    );
    let software = repo_file(include_str!("../../../.github/workflows/gpu-software.yml"));
    let shaped: Vec<&str> = software
        .lines()
        .filter(|line| !line.trim_start().starts_with('#') && line.contains("--shapes"))
        .collect();
    assert_eq!(
        shaped,
        ["          cargo --locked xtask check-difftest --seeds 1-2 --shapes grid,pixel \\"],
        "gpu-software runs --shapes in one step: seeds 1-2, grid,pixel"
    );
    let header = "      - name: Cross-shape differential test (pixel against grid, 2 seeds)";
    let step: Vec<&str> = software
        .lines()
        .skip_while(|line| *line != header)
        .enumerate()
        .take_while(|(at, line)| *at == 0 || !line.starts_with("      - "))
        .map(|(_, line)| line)
        .collect();
    assert!(!step.is_empty(), "no step {header}");
    assert!(step.iter().any(|line| line.contains("--shapes grid,pixel")));
    assert!(
        !step.iter().any(|line| line.starts_with("        if:")),
        "{step:?}"
    );
}

/// A report as `good` builds it, with the `shapes` section the harness writes.
fn shaped(mode: &str, seeds: &[u64], pixel: u64, grid_only: &[(&str, u64)]) -> String {
    let programs = PROGRAMS_PER_SEED * seeds.len() as u64;
    let total: u64 = grid_only.iter().map(|entry| entry.1).sum();
    let reasons = grid_only
        .iter()
        .map(|(reason, count)| format!("{reason:?}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    // The f32 observation: up to six programs compared, two of them with a bit difference (3 elements, 2 ulp at most).
    let compared = pixel.min(6);
    let (differing, elements, ulp) = if compared > 0 { (2, 3, 2) } else { (0, 0, 0) };
    let section = format!(
        "\"shapes\":{{\"requested\":[\"grid\",\"pixel\"],\"float_pixel_vs_grid\":{{\"programs_compared\":{compared},\
         \"programs_with_difference\":{differing},\"elements_differing\":{elements},\"max_ulp\":{ulp},\"first\":null}},\
         \"programs_by_shape\":{{\"grid\":{programs},\"pixel\":{pixel}}},\
         \"grid_only\":{{\"total\":{total},\"by_reason\":{{{reasons}}},\"programs\":[]}}}},\"mismatch_count\""
    );
    good(mode, seeds).replacen("\"mismatch_count\"", &section, 1)
}

fn pixel_options(args: &[&str]) -> Options {
    let mut all = vec!["--shapes", "grid,pixel"];
    all.extend_from_slice(args);
    options(&all)
}

#[test]
fn a_pixel_report_that_matches_the_request_passes_and_says_what_stayed_grid_only() {
    let report = shaped(
        "gpu",
        &[1, 2],
        29,
        &[(
            "GUST-E0726: makes the write to the output .. conditional",
            3,
        )],
    );
    let parsed = parse_report(&report).unwrap();
    let shapes = parsed.shapes.as_ref().unwrap();
    assert_eq!(shapes.requested, ["grid", "pixel"]);
    assert_eq!(
        (
            shapes.grid_programs,
            shapes.pixel_programs,
            shapes.grid_only_total
        ),
        (32, 29, 3)
    );
    let judgement = judge_batch(&pixel_options(&["--seeds", "1-2"]), Some(0), Ok(report));
    assert!(judgement.problems.is_empty(), "{:?}", judgement.problems);
    let text = judgement.summary.join("\n");
    assert!(
        text.contains(
            "29 programs ran as pixel and were compared with their grid run, 3 stayed grid-only"
        ),
        "{text}"
    );
    assert!(text.contains("grid-only 3 x GUST-E0726"), "{text}");
    assert!(
        text.contains("f32 pixel vs grid (an observation, not a mismatch): 2 of 6 programs differ in a bit, 3 elements, at most 2 ulp"),
        "{text}"
    );
    // A grid-only request on a grid-only report still passes without the section (the form of a report written before
    // `--shapes`) and with it.
    let old = good("gpu", &[1]);
    assert!(parse_report(&old).unwrap().shapes.is_none());
    assert!(problems(&options(&["--seeds", "1"]), Some(0), Ok(old)).is_empty());
}

#[test]
fn a_pixel_request_fails_unless_the_report_proves_the_pixel_runs() {
    let ask = pixel_options(&["--seeds", "1"]);
    let found = |report: String| problems(&ask, Some(0), Ok(report));
    // No section at all: the pixel half cannot have run.
    let none = found(good("gpu", &[1]));
    assert!(
        none.iter().any(|p| p.contains("no `shapes` section")),
        "{none:?}"
    );
    // The section is for another request.
    let grid = shaped("gpu", &[1], 0, &[]).replace("[\"grid\",\"pixel\"]", "[\"grid\"]");
    assert!(
        found(grid)
            .iter()
            .any(|p| p.contains("ran shapes [grid], this run asked for [grid,pixel]"))
    );
    // Every program grid-only: nothing was compared.
    let vacuous = found(shaped("gpu", &[1], 0, &[("GUST-E0723: r", 16)]));
    assert_eq!(vacuous.len(), 1, "{vacuous:?}");
    assert!(vacuous[0].contains("no program ran as pixel"));
    // The pixel and grid-only programs must add up to the programs of the run.
    let lost = found(shaped("gpu", &[1], 12, &[("GUST-E0723: r", 3)]));
    assert!(
        lost.iter().any(|p| p.contains(
            "12 pixel programs and 3 grid-only programs do not add up to the 16 programs"
        )),
        "{lost:?}"
    );
    // The reasons must add up to the grid-only count, and the grid run must cover every program.
    let reasons = shaped("gpu", &[1], 13, &[("GUST-E0723: r", 3)])
        .replace("\"GUST-E0723: r\":3", "\"GUST-E0723: r\":2");
    assert!(
        found(reasons)
            .iter()
            .any(|p| p.contains("counts 3 programs but its reasons add up to 2"))
    );
    let grid_count = shaped("gpu", &[1], 16, &[]).replace("\"grid\":16", "\"grid\":15");
    assert!(
        found(grid_count)
            .iter()
            .any(|p| p.contains("programs_by_shape.grid is 15, expected 16"))
    );
    // The f32 observation must be there and make sense; a difference in it is no problem by itself (D128).
    let ask_report = |report: String| problems(&ask, Some(0), Ok(report));
    let missing =
        shaped("gpu", &[1], 16, &[]).replacen("\"float_pixel_vs_grid\"", "\"unrelated\"", 1);
    assert!(
        ask_report(missing)
            .iter()
            .any(|p| p.contains("no `float_pixel_vs_grid` observation")),
        "an observation is required"
    );
    let too_many =
        shaped("gpu", &[1], 16, &[]).replace("\"programs_compared\":6", "\"programs_compared\":17");
    assert!(
        ask_report(too_many)
            .iter()
            .any(|p| p.contains("float_pixel_vs_grid is inconsistent"))
    );
    let silent = shaped("gpu", &[1], 16, &[])
        .replace("\"elements_differing\":3", "\"elements_differing\":0");
    assert!(
        ask_report(silent)
            .iter()
            .any(|p| p.contains("float_pixel_vs_grid is inconsistent"))
    );
    assert!(ask_report(shaped("gpu", &[1], 16, &[])).is_empty());
    // A grid-only request must not be given a report that ran pixel.
    let extra = shaped("gpu", &[1], 16, &[]).replace("[\"grid\",\"pixel\"]", "[\"grid\"]");
    let ask_grid = options(&["--seeds", "1"]);
    let found_grid = problems(&ask_grid, Some(0), Ok(extra));
    assert!(
        found_grid
            .iter()
            .any(|p| p.contains("a grid-only run reports 16 pixel programs")),
        "{found_grid:?}"
    );
}

#[test]
fn a_pixel_mismatch_names_its_shape_and_a_grid_one_does_not() {
    let ask = pixel_options(&["--seeds", "1"]);
    let report = shaped("gpu", &[1], 16, &[]).replace(
        "\"mismatch_count\":0,\"mismatches\":[]",
        "\"mismatch_count\":1,\"mismatches\":[{\"program\":\"s1_a4\",\"seed\":1,\"index\":3,\"attempt\":4,\
         \"shape\":\"pixel\",\"kind\":\"pixel-vs-grid\",\"elements_differing\":2,\"first_element\":5,\
         \"message\":\"element 5: grid 1, pixel 0\"}]",
    );
    let found = problems(&ask, Some(1), Ok(report));
    assert_eq!(
        found,
        [
            "seed 1 program s1_a4 (index 3, attempt 4, shape pixel): pixel-vs-grid: 2 element(s) differ: element 5: grid 1, pixel 0"
        ]
    );
}

// ---- reports written by the real harness ----

const REAL_CPU_ONLY: &str = include_str!("difftest_report_cpu_only.json");
const REAL_GPU: &str = include_str!("difftest_report_gpu.json");

/// Two reports `difftest-run` wrote (seeds 1 to 4 `--cpu-only`; seeds 1 and 2 on the RTX 5090), with the
/// pretty-printing and the number formats of the real tool, parse into the fields this command judges and pass its
/// rules.
#[test]
fn reports_written_by_the_real_harness_parse_and_pass() {
    let cpu = parse_report(REAL_CPU_ONLY).unwrap();
    assert_eq!((cpu.mode.as_str(), cpu.passed), ("cpu-only", true));
    assert_eq!(cpu.seeds, [1, 2, 3, 4]);
    assert_eq!((cpu.programs_requested, cpu.programs_executed), (64, 64));
    assert_eq!(cpu.adapter, None);
    assert_eq!(
        cpu.per_seed,
        [
            (1, 16, 67, 51),
            (2, 16, 51, 35),
            (3, 16, 63, 47),
            (4, 16, 102, 86)
        ]
    );
    assert_eq!(cpu.discards_total, 219);
    assert_eq!(
        cpu.discards_by_class
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        [
            "domain",
            "comparison-overlap",
            "unboundable",
            "float-construct-not-in-v1",
            "no-input-read"
        ]
    );
    assert_eq!(cpu.mismatch_count, 0);
    assert!(cpu.cargo_build_s >= 0.0 && cpu.cargo_run_s > 0.0);
    let found = judge_batch(
        &options(&["--cpu-only", "--seeds", "1-4"]),
        Some(0),
        Ok(REAL_CPU_ONLY.to_owned()),
    );
    assert!(found.problems.is_empty(), "{:?}", found.problems);

    let gpu = parse_report(REAL_GPU).unwrap();
    assert_eq!((gpu.mode.as_str(), gpu.passed), ("gpu", true));
    assert_eq!((gpu.programs_requested, gpu.programs_executed), (32, 32));
    assert_eq!(
        gpu.adapter.as_deref(),
        Some("NVIDIA GeForce RTX 5090 | DiscreteGpu | Vulkan | NVIDIA 610.88")
    );
    let found = judge_batch(
        &options(&["--seeds", "1-2"]),
        Some(0),
        Ok(REAL_GPU.to_owned()),
    );
    assert!(found.problems.is_empty(), "{:?}", found.problems);
    // The same real reports fail when the request does not match them.
    let found = judge_batch(&options(&[]), Some(0), Ok(REAL_GPU.to_owned()));
    assert!(
        found
            .problems
            .iter()
            .any(|p| p == "programs_executed is 32, expected 16 x 8 seeds = 128"),
        "{:?}",
        found.problems
    );
    let found = judge_batch(
        &options(&["--seeds", "1-4"]),
        Some(0),
        Ok(REAL_CPU_ONLY.to_owned()),
    );
    assert!(
        found.problems.iter().any(|p| p.contains("mode `cpu-only`")),
        "{:?}",
        found.problems
    );
}

const REAL_PIXEL: &str = include_str!("difftest_report_pixel.json");

/// A report `difftest-run --seeds 1-8 --shapes grid,pixel` wrote on the RTX 5090 (slice mx.p1c, 2026-11-05: 128
/// programs, every one also ran as a pixel kernel, none stayed grid-only) parses into the shape accounting and passes the
/// rules of a pixel request; it fails a grid-only request (the report says pixel) and a pixel request when read as a
/// report without it.
#[test]
fn a_report_written_with_the_pixel_shape_parses_and_passes() {
    let report = parse_report(REAL_PIXEL).unwrap();
    assert_eq!((report.mode.as_str(), report.passed), ("gpu", true));
    assert_eq!(
        (report.programs_requested, report.programs_executed),
        (128, 128)
    );
    let found = judge_batch(
        &options(&["--seeds", "1-8", "--shapes", "grid,pixel"]),
        Some(0),
        Ok(REAL_PIXEL.to_owned()),
    );
    assert!(found.problems.is_empty(), "{:?}", found.problems);
    // Asked for the grid only, the same report is the wrong report.
    let found = judge_batch(
        &options(&["--seeds", "1-8"]),
        Some(0),
        Ok(REAL_PIXEL.to_owned()),
    );
    assert!(!found.problems.is_empty(), "{:?}", found.problems);
    // The grid-only reports above carry no shapes block with pixel programs.
    let found = judge_batch(
        &options(&["--seeds", "1-2", "--shapes", "grid,pixel"]),
        Some(0),
        Ok(REAL_GPU.to_owned()),
    );
    assert!(!found.problems.is_empty(), "{:?}", found.problems);
}

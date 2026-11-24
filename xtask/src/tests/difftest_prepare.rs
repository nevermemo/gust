use std::{ffi::OsString, fs, path::Path};

use super::{
    BUILD_ORDER, DifftestStep, Mode, Options, Prepared, ROUTE_VARIABLE, Stage, check_prepared,
    profile_batch_args, refuse_route_split, source_state, stamp_refusal, steps,
};
use crate::workspace::workspace_root;

fn parse(args: &[&str]) -> Result<Options, String> {
    let owned: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
    Options::parse_with(&owned, 7)
}

#[test]
fn the_stage_flags_parse_exclude_each_other_and_take_the_grid_run_only() {
    assert_eq!(parse(&[]).unwrap().stage, Stage::Whole);
    assert_eq!(parse(&["--prepare"]).unwrap().stage, Stage::Prepare);
    assert_eq!(parse(&["--runs-only"]).unwrap().stage, Stage::RunsOnly);
    let seeds = parse(&["--seeds", "1-2", "--prepare"]).unwrap().seeds;
    assert_eq!(seeds, [1, 2]);
    for args in [
        &["--prepare", "--runs-only"][..],
        &["--prepare", "--prepare"],
        &["--runs-only", "--cpu-only"],
        &["--prepare", "--seeds", "random:2"],
        &["--runs-only", "--shapes", "grid,pixel"],
    ] {
        let error = parse(args).unwrap_err();
        assert!(
            error.contains("--prepare") && error.contains("--runs-only"),
            "{args:?}: {error}"
        );
        assert!(error.contains("\nusage:"), "{error}");
    }
    // The grid shape alone is the plain run, so it is no refusal.
    assert!(parse(&["--prepare", "--shapes", "grid"]).is_ok());
}

/// `--prepare` is dispatched before the steps, so it has none of them; `--runs-only` runs the steps of the Gpu tier.
#[test]
fn prepare_is_dispatched_before_the_steps_and_the_gpu_steps_are_unchanged() {
    let source = include_str!("../difftest.rs").replace("\r\n", "\n");
    let summary = source
        .split("fn difftest_summary(")
        .nth(1)
        .and_then(|rest| rest.split("\nfn run_steps").next())
        .expect("difftest_summary");
    let prepare = summary.find("return build_batches(").unwrap();
    assert!(prepare < summary.find("run_steps(").unwrap());
    assert!(summary[..prepare].contains("Stage::Prepare"));
    // The calls that pass the executable on: `run_batch` its prepared one, `build_batches` the one to build.
    assert!(source.contains("prebuilt.map(|exe| (\"--run-exe\", exe))"));
    assert!(source.contains("Some((\"--prepare-exe\", &exe))"));
    assert_eq!(
        steps(Mode::Gpu),
        [
            DifftestStep::Batch,
            DifftestStep::ShortCircuitGpu,
            DifftestStep::ProfileArtifacts
        ]
    );
    // Both profile batches are built, `dev` last (the build state `short_circuit` finds).
    assert_eq!(BUILD_ORDER, ["dev+subgroups", "dev"]);
}

/// The property the split exists for: `--runs-only` hands the harness the prepared executable and `--prepare` has it
/// build one; the whole run does neither.
#[test]
fn the_split_hands_the_harness_the_executable_to_run_or_to_build() {
    let options = parse(&["--seeds", "1-2"]).unwrap();
    let (target, report) = (Path::new("t"), Path::new("t/r.json"));
    let prepared = Prepared::new(target, "n1".to_owned());
    let (dev_records, sub_records) = prepared.records();
    assert!(dev_records.ends_with("n1/dev") && sub_records.ends_with("n1/dev-subgroups"));
    let args = |profile: &str, export: Option<&Path>, exe: Option<(&str, &Path)>| {
        profile_batch_args(&options, target, report, profile, export, exe)
    };
    let exe = prepared.exe("dev+subgroups");
    let run = args(
        "dev+subgroups",
        Some(&sub_records),
        Some(("--run-exe", &exe)),
    )
    .unwrap();
    let at = run
        .iter()
        .position(|arg| arg == "--run-exe")
        .expect("--run-exe");
    assert_eq!(Path::new(&run[at + 1]), exe);
    assert!(!run.contains(&"--prepare-exe".to_owned()));
    assert!(
        run.windows(2)
            .any(|pair| { pair[0] == "--export-dir" && Path::new(&pair[1]) == sub_records })
    );
    assert!(
        run.windows(2)
            .any(|pair| pair == ["--profile", "dev+subgroups"])
    );
    let build = args(
        "dev",
        Some(&dev_records),
        Some(("--prepare-exe", &prepared.exe("dev"))),
    )
    .unwrap();
    assert!(
        build.contains(&"--prepare-exe".to_owned()) && !build.contains(&"--run-exe".to_owned())
    );
    let whole = args("dev", None, None).unwrap();
    assert!(
        !whole.contains(&"--run-exe".to_owned()) && !whole.contains(&"--prepare-exe".to_owned())
    );
    assert!(
        args("dev+subgroups", None, None)
            .unwrap_err()
            .contains("record directory")
    );
}

#[test]
fn the_split_refuses_a_route_run() {
    let lookup = |value: &'static str| {
        move |name: &str| (name == ROUTE_VARIABLE).then(|| OsString::from(value))
    };
    let error = refuse_route_split(lookup("1")).unwrap_err();
    assert!(
        error.contains("--prepare") && error.contains(ROUTE_VARIABLE),
        "{error}"
    );
    assert!(refuse_route_split(lookup("0")).is_ok());
    assert!(refuse_route_split(|_: &str| None).is_ok());
}

#[test]
fn a_stamp_refusal_names_the_value_before_and_the_value_now() {
    let before = "head=aaa\ntree=t1\nseeds=1,2\nsha256 of the dev executable=ab\n";
    let refusal = |now: &str| stamp_refusal(before, now, "executables", "check-difftest");
    assert_eq!(refusal(before), None);
    let message = refusal(&before.replace("head=aaa", "head=bbb")).unwrap();
    assert!(
        message.contains("head was aaa") && message.contains("is bbb now"),
        "{message}"
    );
    assert!(
        message.contains("prepared executables") && message.contains("check-difftest --prepare"),
        "{message}"
    );
    let message = refusal(&before.replace("seeds=1,2", "seeds=1,3")).unwrap();
    assert!(
        message.contains("seeds was 1,2") && message.contains("is 1,3 now"),
        "{message}"
    );
    let message = stamp_refusal("head=aaa\n", "head=aaa\ntree=t1\n", "x", "y").unwrap();
    assert!(message.contains("tree was (none)"), "{message}");
}

#[test]
fn runs_only_refuses_without_a_stamp_and_with_one_that_differs_from_now() {
    let root = workspace_root().unwrap();
    let target =
        std::env::temp_dir().join(format!("xtask-difftest-prepared-{}", std::process::id()));
    let _ = fs::remove_dir_all(&target);
    let options = parse(&["--seeds", "1-2", "--runs-only"]).unwrap();
    let error = check_prepared(&options, &root, &target).err().unwrap();
    assert!(
        error.contains("no stamp") && error.contains("--prepare"),
        "{error}"
    );

    let prepared = Prepared::new(&target, "n7".to_owned());
    fs::create_dir_all(&prepared.dir).unwrap();
    for profile in BUILD_ORDER {
        fs::write(prepared.exe(profile), b"0123456789").unwrap();
    }
    let stamp = prepared
        .stamp_text(&options, &source_state(&root).unwrap())
        .unwrap();
    assert!(stamp.contains("seeds=1,2\n") && stamp.contains("records=n7\n"));
    assert!(stamp.contains("sha256 of the dev executable="));
    fs::write(prepared.stamp_file(), &stamp).unwrap();
    let checked = check_prepared(&options, &root, &target).unwrap();
    assert!(
        checked.records().0.ends_with("n7/dev"),
        "the nonce comes from the stamp"
    );

    // Other seeds, another tree, another HEAD and another executable are each refused, with both values named.
    let other = parse(&["--seeds", "1-3", "--runs-only"]).unwrap();
    let error = check_prepared(&other, &root, &target).err().unwrap();
    assert!(
        error.contains("seeds was 1,2") && error.contains("is 1,2,3 now"),
        "{error}"
    );
    for key in ["head", "tree"] {
        let edited: Vec<String> = stamp
            .lines()
            .map(|line| match line.strip_prefix(&format!("{key}=")) {
                Some(_) => format!("{key}=stale"),
                None => line.to_owned(),
            })
            .collect();
        fs::write(prepared.stamp_file(), edited.join("\n") + "\n").unwrap();
        let error = check_prepared(&options, &root, &target).err().unwrap();
        assert!(error.contains(&format!("{key} was stale")), "{error}");
    }
    // A same-size executable with other bytes is caught by the hash.
    fs::write(prepared.stamp_file(), &stamp).unwrap();
    fs::write(prepared.exe("dev"), b"9876543210").unwrap();
    let error = check_prepared(&options, &root, &target).err().unwrap();
    assert!(
        error.contains("sha256 of the dev executable was"),
        "{error}"
    );
    let _ = fs::remove_dir_all(&target);
}

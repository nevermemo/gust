//! The backend groups of a run and the kill rule of the publishable group (plan R1.2-3).
//!
//! - `publishable`: the backends whose comparison the published table rests on: gust-gpu (or gust-gpu-spirv in the
//!   `spirv-route` build), wgsl-prebuilt and wgsl-percall, sampled together in seeded random order (`harness`). The CPU
//!   backends run beside the group, outside the rotation, as before.
//! - `ablation`: wgsl-percall and wgsl-prebuilt plus the switch ablations (`wgsl-percall+bind-group-reuse`, ..., and
//!   `wgsl-percall+all-on`). No GUST backend: the ablation rows are compared only against the wgsl-percall and
//!   wgsl-prebuilt of their own group, never against numbers from the publishable group.
//!
//! The group a sample came from is written into the run file (`group` at the top level and in every cell), and the
//! aggregate refuses to mix groups.
//!
//! One deviation from the plan's single group: the route (WGSL or SPIR-V) is a build-time property of gust-wgpu (the
//! `spirv-route` feature changes the code the macros emit), so a process holds one GUST route. The two GUST routes are
//! two builds, each with its own wgsl-prebuilt and wgsl-percall in its own paired group; `gust-gpu-spirv` is its own
//! backend id. The two routes are compared through their ratios to the wgsl-prebuilt of their own runs.

// The kill rule judges GPU cells: the GPU-less `cpu` build has no use for it.
#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

use serde_json::{Value, json};

use crate::workloads::Workload;

/// The paired-ratio window of the kill rule: more than 10 % off 1.0 reopens candidate C.
pub const KILL_TOLERANCE: f64 = 0.10;

/// The quantities the rule reads: the GPU-timestamp ones (the host-clock ones include driver and queue overhead that
/// is not the kernel).
pub const KILL_QUANTITIES: [&str; 2] = ["gpu_single", "gpu_batch_per_dispatch"];

/// The GUST backends the rule judges against wgsl-prebuilt.
pub const GUST_BACKENDS: [&str; 2] = ["gust-gpu", "gust-gpu-spirv"];

/// The control of the rule's record: same kernel text as wgsl-prebuilt, so its ratio is the noise of the method.
pub const CONTROL_BACKEND: &str = "wgsl-percall";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Publishable,
    Ablation,
}

impl Group {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "publishable" => Some(Self::Publishable),
            "ablation" => Some(Self::Ablation),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Publishable => "publishable",
            Self::Ablation => "ablation",
        }
    }
}

/// A `publishable` run keeps the whole group: `--backends` may not leave out wgsl-prebuilt, wgsl-percall or the
/// GUST backend of the build (`gust_id`), because the paired ratios and the kill rule need all three. Use
/// `--group ablation` for any subset.
pub fn require_full_set(wanted: &[String], gust_id: &str) -> Result<(), String> {
    let missing = [gust_id, "wgsl-prebuilt", "wgsl-percall"]
        .into_iter()
        .filter(|id| !wanted.iter().any(|w| w == id))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "--backends leaves out {} of the publishable group (gust-gpu or its route twin, wgsl-prebuilt, wgsl-percall); a publishable run keeps all three (use --group ablation for a subset)",
            missing.join(", ")
        ))
    }
}

/// The kill rule over the cells of a run: a gust-gpu (or gust-gpu-spirv) GPU-time paired ratio to wgsl-prebuilt more
/// than 10 % off 1.0 on any cell marks the run "C re-opened". Reads `paired_vs_prebuilt.median_ratio` of
/// [`KILL_QUANTITIES`]. Every GUST cell and quantity without a ratio (a failed or dropped backend, a missing metric)
/// is counted in `unpaired`, and so is every (workload, size) that has a hand-WGSL cell but no GUST cell. The rule
/// judged the run only if `checked > 0 && unpaired == 0` (`judged`); otherwise `c_reopened` is `null`, never `false`.
/// A workload whose GUST and hand-WGSL paths are not the same kernel (`Workload::same_kernel`, W7: GUST runs the public
/// `gust_algo::reduce`, which exposes no GPU timestamps, so its cell has no paired GPU-time ratio by construction) is
/// outside the rule, on the GUST and the hand-WGSL side alike. The exclusion is recorded, not silent: `excluded` lists
/// every such cell of the rule's backends (`backend`, `workload`, `n`, `reason`), so a reader of the run file sees W7 was
/// present and left out on purpose (coordinator decision D190, a deviation from the plan's "any cell").
/// `control` carries the same ratio for wgsl-percall (same kernel text as wgsl-prebuilt, so its distance from 1.0 is
/// the noise floor of the method); it does not enter the rule. Returns the record that goes into the run file.
pub fn kill_rule(cells: &[Value]) -> Value {
    let mut checked = 0usize;
    let mut unpaired = 0usize;
    let mut offenders = Vec::new();
    let mut with_gust = std::collections::BTreeSet::new();
    let mut with_wgsl = std::collections::BTreeSet::new();
    let mut control = Vec::new();
    let mut excluded = Vec::new();
    for cell in cells {
        let backend = cell["backend"].as_str().unwrap_or("");
        let key = (
            cell["workload"].as_str().unwrap_or("").to_owned(),
            cell["n"].as_u64().unwrap_or(0),
        );
        // A workload that is not a same-kernel pairing (W7: GUST runs `gust-algo`'s sum and the hand backends a reduction of
        // their own, and `gust-algo` has no public GPU timestamps, so a W7 GUST cell has no paired GPU-time ratio by
        // construction) is outside the rule, on the GUST and the hand-WGSL side alike; it would otherwise count as
        // `unpaired` and keep every run that includes it from being judged. The cell is listed in `excluded`.
        if Workload::parse(&key.0).is_some_and(|w| !w.same_kernel()) {
            if GUST_BACKENDS.contains(&backend)
                || backend == "wgsl-prebuilt"
                || backend == CONTROL_BACKEND
            {
                excluded.push(json!({
                    "backend": backend, "workload": key.0, "n": key.1,
                    "reason": "no same-kernel GUST/hand pairing: GUST runs the library's sum, which exposes no GPU timestamps",
                }));
            }
            continue;
        }
        if backend == CONTROL_BACKEND {
            for quantity in KILL_QUANTITIES {
                if let Some(ratio) =
                    cell["metrics"][quantity]["paired_vs_prebuilt"]["median_ratio"].as_f64()
                {
                    control.push((key.clone(), quantity, ratio));
                }
            }
        }
        if backend == "wgsl-prebuilt" {
            with_wgsl.insert(key.clone());
        }
        if !GUST_BACKENDS.contains(&backend) {
            continue;
        }
        with_gust.insert(key);
        for quantity in KILL_QUANTITIES {
            match cell["metrics"][quantity]["paired_vs_prebuilt"]["median_ratio"].as_f64() {
                Some(ratio) => {
                    checked += 1;
                    // The small slack keeps a ratio of exactly 1.10 inside (1.10 - 1.0 is not exactly 0.10 in f64).
                    if (ratio - 1.0).abs() > KILL_TOLERANCE + 1e-12 {
                        offenders.push(json!({
                            "backend": backend, "workload": cell["workload"], "n": cell["n"],
                            "quantity": quantity, "median_ratio": ratio,
                        }));
                    }
                }
                None => unpaired += 1,
            }
        }
    }
    // A hand-WGSL cell with no GUST cell beside it: the backend was dropped or never ran there.
    let missing = with_wgsl.difference(&with_gust).count();
    unpaired += missing * KILL_QUANTITIES.len();
    let judged = checked > 0 && unpaired == 0;
    // The reproducibility clause (R1.3-4): an offender counts only when it recurs in 3/3 seeds and the control is within
    // tolerance on that cell in that run. One run file sees one seed, so each offender carries the control's ratio on
    // its own cell here, and `aggregate` counts the seeds (`kill_rule_reproducibility`).
    for offender in &mut offenders {
        let control_ratio = control
            .iter()
            .find(|((workload, n), quantity, _)| {
                offender["workload"].as_str() == Some(workload.as_str())
                    && offender["n"].as_u64() == Some(*n)
                    && offender["quantity"].as_str() == Some(*quantity)
            })
            .map(|(_, _, ratio)| *ratio);
        offender["control_median_ratio"] = json!(control_ratio);
        offender["control_in_tolerance"] =
            json!(control_ratio.map(|ratio| (ratio - 1.0).abs() <= KILL_TOLERANCE + 1e-12));
    }
    let control_offenders = control
        .iter()
        .filter(|(_, _, ratio)| (ratio - 1.0).abs() > KILL_TOLERANCE + 1e-12)
        .count();
    let control_worst = control
        .iter()
        .map(|(_, _, ratio)| (ratio - 1.0).abs())
        .fold(0.0_f64, f64::max);
    json!({
        "rule": "gust-gpu/wgsl-prebuilt GPU-time paired ratio more than 10 % off 1.0 on any cell reopens candidate C",
        "tolerance": KILL_TOLERANCE,
        "quantities": KILL_QUANTITIES,
        "checked": checked,
        "unpaired": unpaired,
        "judged": judged,
        "c_reopened": if judged { json!(!offenders.is_empty()) } else { Value::Null },
        "excluded": excluded,
        "offenders": offenders,
        "reproducibility_clause": "an offender counts when it recurs in 3/3 seeds with `control_in_tolerance` true on that cell in each of those runs; a single-seed hit is listed, not counted (plan R1.3-4; `aggregate` counts the seeds and needs at least 3 runs with distinct seeds)",
        "control": {
            "note": "wgsl-percall / wgsl-prebuilt, same quantities and method, same cells; same kernel text, so the distance from 1.0 is the noise floor of the rule for gpu_single. For gpu_batch_per_dispatch the control also differs in per-pass bind groups (a fresh one per pass, as GUST does; wgsl-prebuilt reuses one), so it is not a pure noise floor there. Not an input of the rule.",
            "backend": CONTROL_BACKEND,
            "checked": control.len(),
            "outside_tolerance": control_offenders,
            "worst_abs_deviation": if control.is_empty() { Value::Null } else { json!(control_worst) },
            "ratios": control.iter().map(|((workload, n), quantity, ratio)| json!({
                "workload": workload, "n": n, "quantity": quantity, "median_ratio": ratio,
            })).collect::<Vec<_>>(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(backend: &str, single: Option<f64>, batch: Option<f64>) -> Value {
        let metric = |ratio: Option<f64>| match ratio {
            Some(r) => json!({ "paired_vs_prebuilt": { "median_ratio": r } }),
            None => json!({}),
        };
        json!({
            "backend": backend, "workload": "W1", "n": 1048576,
            "metrics": { "gpu_single": metric(single), "gpu_batch_per_dispatch": metric(batch) },
        })
    }

    #[test]
    fn a_ratio_more_than_ten_percent_off_reopens_c_and_one_inside_does_not() {
        let inside = kill_rule(&[cell("gust-gpu", Some(1.09), Some(0.91))]);
        assert_eq!(inside["c_reopened"], false);
        assert_eq!(inside["checked"], 2);
        let slow = kill_rule(&[cell("gust-gpu", Some(1.11), Some(1.0))]);
        assert_eq!(slow["c_reopened"], true);
        assert_eq!(slow["offenders"][0]["quantity"], "gpu_single");
        // A faster GUST is just as far off: the rule is two-sided.
        let fast = kill_rule(&[cell("gust-gpu-spirv", Some(1.0), Some(0.89))]);
        assert_eq!(fast["c_reopened"], true);
        assert_eq!(fast["offenders"][0]["backend"], "gust-gpu-spirv");
        assert_eq!(fast["offenders"][0]["quantity"], "gpu_batch_per_dispatch");
    }

    #[test]
    fn the_boundary_is_inside_and_one_bad_cell_of_many_is_enough() {
        assert_eq!(
            kill_rule(&[cell("gust-gpu", Some(1.10), Some(0.90))])["c_reopened"],
            false
        );
        let cells = [
            cell("gust-gpu", Some(1.0), Some(1.0)),
            cell("gust-gpu", Some(1.02), Some(1.0)),
            cell("gust-gpu", Some(1.0), Some(1.2)),
        ];
        let record = kill_rule(&cells);
        assert_eq!(record["c_reopened"], true);
        assert_eq!(record["offenders"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn only_gust_backends_are_judged_and_an_unpaired_cell_is_counted_not_flagged() {
        // wgsl-percall at 1.5x is not the rule's business; the reference against itself is 1.0.
        let record = kill_rule(&[
            cell("wgsl-percall", Some(1.5), Some(1.5)),
            cell("wgsl-prebuilt", Some(1.0), Some(1.0)),
            cell("gust-gpu", None, Some(1.0)),
        ]);
        assert_eq!(record["checked"], 1);
        assert_eq!(record["unpaired"], 1);
        // Something was unpaired: nothing was judged, so the answer is null, not false.
        assert_eq!(record["judged"], false);
        assert!(record["c_reopened"].is_null());
        // The control is recorded beside the rule and does not enter it.
        assert_eq!(record["control"]["checked"], 2);
        assert_eq!(record["control"]["outside_tolerance"], 2);
    }

    fn named(backend: &str, n: u64, single: Option<f64>, batch: Option<f64>) -> Value {
        let mut value = cell(backend, single, batch);
        value["n"] = json!(n);
        value
    }

    /// W7 is a different algorithm on each side (gust-algo's sum against a hand reduction) and gust-algo has no GPU
    /// timestamps, so a W7 cell is outside the rule on both sides: it neither counts as unpaired nor is judged.
    #[test]
    fn w7_cells_are_outside_the_rule_on_both_sides() {
        let w7 = |backend: &str, single: Option<f64>| {
            let mut value = named(backend, 1 << 20, single, single);
            value["workload"] = json!("W7-f32");
            value
        };
        let mut cells = vec![
            w7("gust-gpu", None),
            w7("wgsl-prebuilt", Some(1.0)),
            w7("wgsl-percall", Some(1.7)),
            named("gust-gpu", 1, Some(1.0), Some(1.0)),
            named("wgsl-prebuilt", 1, Some(1.0), Some(1.0)),
        ];
        let record = kill_rule(&cells);
        assert_eq!(record["checked"], 2, "only the W1 cell is judged");
        assert_eq!(record["unpaired"], 0, "the W7 cells are not unpaired");
        assert_eq!(record["judged"], true);
        assert_eq!(record["c_reopened"], false);
        assert_eq!(
            record["control"]["checked"], 0,
            "a W7 control is not recorded"
        );
        // The exclusion is listed, not silent: every W7 cell of the rule's backends and of the control, with its reason.
        let listed = record["excluded"].as_array().expect("excluded is a list");
        assert_eq!(listed.len(), 3);
        assert!(
            listed
                .iter()
                .all(|e| e["workload"] == "W7-f32" && e["n"] == (1u64 << 20))
        );
        assert!(listed.iter().any(|e| e["backend"] == "gust-gpu"));
        assert!(
            listed[0]["reason"]
                .as_str()
                .is_some_and(|r| r.contains("timestamps"))
        );
        // A W7 GUST cell with a wild ratio would not reopen C either.
        cells[0] = w7("gust-gpu", Some(3.0));
        assert_eq!(kill_rule(&cells)["c_reopened"], false);
        // A run of W7 alone judges nothing, and says null.
        let only = kill_rule(&cells[..3]);
        assert_eq!(only["judged"], false);
        assert!(only["c_reopened"].is_null());
    }

    /// A run that has a W7 GUST cell (no ratio by construction) is still judged, on its other cells, and says what it
    /// left out (decision D190).
    #[test]
    fn a_run_with_a_w7_gust_cell_is_judged_and_lists_the_exclusion() {
        let mut w7 = named("gust-gpu", 1 << 20, None, None);
        w7["workload"] = json!("W7-u32");
        let record = kill_rule(&[
            w7,
            named("gust-gpu", 1, Some(1.0), Some(1.0)),
            named("wgsl-prebuilt", 1, Some(1.0), Some(1.0)),
        ]);
        assert_eq!(record["judged"], true);
        assert_eq!(record["c_reopened"], false, "judged, not null");
        assert_eq!(record["unpaired"], 0);
        assert_eq!(
            record["excluded"],
            json!([{
                "backend": "gust-gpu", "workload": "W7-u32", "n": 1 << 20,
                "reason": "no same-kernel GUST/hand pairing: GUST runs the library's sum, which exposes no GPU timestamps",
            }])
        );
        // A run with nothing excluded lists nothing.
        let plain = kill_rule(&[named("gust-gpu", 1, Some(1.0), Some(1.0))]);
        assert_eq!(plain["excluded"], json!([]));
    }

    #[test]
    fn no_gpu_cells_judges_nothing_and_says_null() {
        let record = kill_rule(&[]);
        assert_eq!(record["checked"], 0);
        assert_eq!(record["judged"], false);
        assert!(record["c_reopened"].is_null());
        assert!(record["control"]["worst_abs_deviation"].is_null());
    }

    #[test]
    fn a_dropped_gust_backend_is_unpaired_and_nothing_is_judged() {
        // The GUST cell of the second size is missing altogether.
        let record = kill_rule(&[
            named("gust-gpu", 1, Some(1.0), Some(1.0)),
            named("wgsl-prebuilt", 1, Some(1.0), Some(1.0)),
            named("wgsl-prebuilt", 2, Some(1.0), Some(1.0)),
        ]);
        assert_eq!(record["checked"], 2);
        assert_eq!(record["unpaired"], 2);
        assert!(record["c_reopened"].is_null());
        // A GUST cell whose metrics are missing (the backend failed there).
        let failed = kill_rule(&[named("gust-gpu", 1, None, None)]);
        assert_eq!(failed["unpaired"], 2);
        assert!(failed["c_reopened"].is_null());
    }

    #[test]
    fn a_judged_pass_is_false_and_a_judged_fire_is_true_with_the_control_beside_it() {
        let pass = kill_rule(&[
            named("gust-gpu", 1, Some(1.02), Some(0.95)),
            named("wgsl-prebuilt", 1, Some(1.0), Some(1.0)),
            named("wgsl-percall", 1, Some(1.2), Some(1.0)),
        ]);
        assert_eq!(pass["judged"], true);
        assert_eq!(pass["c_reopened"], false);
        // The control is outside the window but does not change the rule.
        assert_eq!(pass["control"]["outside_tolerance"], 1);
        let worst = pass["control"]["worst_abs_deviation"].as_f64().unwrap();
        assert!((worst - 0.2).abs() < 1e-9);
        let fire = kill_rule(&[
            named("gust-gpu", 1, Some(1.25), Some(1.0)),
            named("wgsl-prebuilt", 1, Some(1.0), Some(1.0)),
        ]);
        assert_eq!(fire["judged"], true);
        assert_eq!(fire["c_reopened"], true);
    }

    #[test]
    fn a_publishable_run_needs_the_whole_group_in_its_backends() {
        let names = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let full = names(&["gust-gpu", "wgsl-prebuilt", "wgsl-percall", "cpu-serial"]);
        assert!(require_full_set(&full, "gust-gpu").is_ok());
        let no_percall = names(&["gust-gpu", "wgsl-prebuilt"]);
        let message = require_full_set(&no_percall, "gust-gpu").unwrap_err();
        assert!(message.contains("wgsl-percall"), "{message}");
        // The route twin of the other build does not stand in for this build's GUST backend.
        let wrong_route = names(&["gust-gpu-spirv", "wgsl-prebuilt", "wgsl-percall"]);
        assert!(require_full_set(&wrong_route, "gust-gpu").is_err());
        assert!(require_full_set(&wrong_route, "gust-gpu-spirv").is_ok());
    }

    #[test]
    fn groups_parse_and_name_round_trip() {
        for group in [Group::Publishable, Group::Ablation] {
            assert_eq!(Group::parse(group.name()), Some(group));
        }
        assert_eq!(Group::parse("other"), None);
    }
}

#[cfg(test)]
mod reproducibility_tests {
    use super::*;

    fn cell(backend: &str, batch: f64) -> Value {
        json!({
            "backend": backend, "workload": "W1", "n": 1048576,
            "metrics": {
                "gpu_single": { "paired_vs_prebuilt": { "median_ratio": 1.0 } },
                "gpu_batch_per_dispatch": { "paired_vs_prebuilt": { "median_ratio": batch } },
            },
        })
    }

    #[test]
    fn an_offender_carries_the_control_ratio_of_its_own_cell() {
        let record = kill_rule(&[
            cell("gust-gpu", 1.2),
            cell("wgsl-prebuilt", 1.0),
            cell("wgsl-percall", 1.02),
        ]);
        let offender = &record["offenders"][0];
        assert_eq!(offender["quantity"], "gpu_batch_per_dispatch");
        assert_eq!(offender["control_median_ratio"], 1.02);
        assert_eq!(offender["control_in_tolerance"], true);
        let noisy = kill_rule(&[
            cell("gust-gpu", 1.2),
            cell("wgsl-prebuilt", 1.0),
            cell("wgsl-percall", 1.15),
        ]);
        assert_eq!(noisy["offenders"][0]["control_in_tolerance"], false);
    }
}

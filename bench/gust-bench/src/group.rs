//! The backend groups of a run and the kill rules of the table groups (plan R1.2-3, D193).
//!
//! - `publishable`: the backends whose comparison the published table rests on: gust-gpu, wgsl-prebuilt and
//!   wgsl-percall, sampled together in seeded random order (`harness`). The CPU backends run beside the group, outside
//!   the rotation, as before. The WGSL build's group; the `spirv-route` build does not run it (D193).
//! - `spirv` (D193, m10.b3d): the `spirv-route` build's group. `gust-gpu-spirv` skips wgpu's runtime shader checks (its
//!   own guards, D174), so it is compared with hand WGSL that skips them too: `wgsl-prebuilt-unchecked` (the
//!   reference of the group's ratios) and `wgsl-percall-unchecked` (the control), in a seeded rotation of its own with
//!   its own kill rule. Its rows are never compared with the publishable group's.
//! - `ablation`: wgsl-percall and wgsl-prebuilt plus the switch ablations (`wgsl-percall+bind-group-reuse`, ..., and
//!   `wgsl-percall+all-on`). No GUST backend: the ablation rows are compared only against the wgsl-percall and
//!   wgsl-prebuilt of their own group, never against numbers from the publishable group.
//!
//! The group a sample came from is written into the run file (`group` at the top level and in every cell), and the
//! aggregate refuses to mix groups.
//!
//! One deviation from the plan's single group: the route (WGSL or SPIR-V) is a build-time property of gust-wgpu (the
//! `spirv-route` feature changes the code the macros emit), so a process holds one GUST route. The two GUST routes are
//! two builds, each with its own hand rows in its own paired group (`publishable` and `spirv`); `gust-gpu-spirv` is its
//! own backend id. The two routes are not compared with each other (D193): each is judged against hand WGSL that has
//! the same runtime checks as it.

// The kill rule judges GPU cells: the GPU-less `cpu` build has no use for it.
#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

use serde_json::{Value, json};

use crate::workloads::Workload;

/// The paired-ratio window of the kill rule: more than 10 % off 1.0 reopens candidate C.
pub const KILL_TOLERANCE: f64 = 0.10;

/// The quantities the rule reads: the GPU-timestamp ones (the host-clock ones include driver and queue overhead that
/// is not the kernel).
pub const KILL_QUANTITIES: [&str; 2] = ["gpu_single", "gpu_batch_per_dispatch"];

/// The backend ids of the `spirv` group.
pub const SPIRV_GUST: &str = "gust-gpu-spirv";
pub const SPIRV_REFERENCE: &str = "wgsl-prebuilt-unchecked";
pub const SPIRV_CONTROL: &str = "wgsl-percall-unchecked";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Publishable,
    Spirv,
    Ablation,
}

impl Group {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "publishable" => Some(Self::Publishable),
            "spirv" => Some(Self::Spirv),
            "ablation" => Some(Self::Ablation),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Publishable => "publishable",
            Self::Spirv => "spirv",
            Self::Ablation => "ablation",
        }
    }

    /// Whether the group's rows can make a published table (the ablation group is method evidence only).
    pub fn is_table(self) -> bool {
        self != Self::Ablation
    }

    /// The backend every other one of the group is paired against (the denominator of `paired_vs_prebuilt`).
    pub fn reference(self) -> &'static str {
        match self {
            Self::Spirv => SPIRV_REFERENCE,
            Self::Publishable | Self::Ablation => "wgsl-prebuilt",
        }
    }

    /// The control of the group's kill rule: same kernel text as the reference, so its ratio is the noise of the method.
    pub fn control(self) -> &'static str {
        match self {
            Self::Spirv => SPIRV_CONTROL,
            Self::Publishable | Self::Ablation => "wgsl-percall",
        }
    }

    /// The GUST backend the group's kill rule judges.
    pub fn gust_backend(self) -> &'static str {
        match self {
            Self::Spirv => SPIRV_GUST,
            Self::Publishable | Self::Ablation => "gust-gpu",
        }
    }
}

/// The group a build runs when `--group` is not given: the `spirv-route` build's is `spirv`.
pub fn default_group(spirv_route: bool) -> Group {
    if spirv_route {
        Group::Spirv
    } else {
        Group::Publishable
    }
}

/// The group against the build's route (D193): the SPIR-V route is judged in `spirv` and only there, the WGSL route in
/// `publishable`; the ablation group is for either.
pub fn check_route(group: Group, spirv_route: bool) -> Result<(), String> {
    match (group, spirv_route) {
        (Group::Publishable, true) => Err(
            "the SPIR-V route is judged in its own group (D193): use --group spirv in the spirv-route build; the publishable group is the WGSL build's"
                .to_owned(),
        ),
        (Group::Spirv, false) => Err(
            "--group spirv needs the spirv-route build (the SPIR-V route is a build-time property); the WGSL build runs the publishable group"
                .to_owned(),
        ),
        _ => Ok(()),
    }
}

/// A table group keeps the whole group: `--backends` may not leave out the group's reference, control or GUST backend
/// (`gust_id`, the build's own), because the paired ratios and the kill rule need all three. Use `--group ablation`
/// for any subset.
pub fn require_full_set(group: Group, wanted: &[String], gust_id: &str) -> Result<(), String> {
    if group == Group::Ablation {
        return Ok(());
    }
    let missing = [gust_id, group.reference(), group.control()]
        .into_iter()
        .filter(|id| !wanted.iter().any(|w| w == id))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "--backends leaves out {} of the {} group ({}, {}, {}); a run of it keeps all three (use --group ablation for a subset)",
            missing.join(", "),
            group.name(),
            gust_id,
            group.reference(),
            group.control()
        ))
    }
}

/// The kill rule over the cells of a run of a table group: the group's GUST backend's GPU-time paired ratio to the
/// group's reference more than 10 % off 1.0 on any cell marks the run. Reads `paired_vs_prebuilt.median_ratio` of
/// [`KILL_QUANTITIES`]. Every GUST cell and quantity without a ratio (a failed or dropped backend, a missing metric)
/// is counted in `unpaired`, and so is every (workload, size) that has a reference cell but no GUST cell. The rule
/// judged the run only if `checked > 0 && unpaired == 0` (`judged`); otherwise `c_reopened` is `null`, never `false`.
///
/// What the mark means depends on the group. In `publishable` the rule is two-sided: an offender in either direction
/// is "C re-opened". In `spirv` (plan R1.3-3, D193) a slower offender (ratio above 1.10) re-opens C
/// (`c_reopened`), a faster one (below 0.90) does not: it sets `reading_step`, which asks for a reading of the
/// cell's cause (as the b1q/b1c W3 reading did). `reading_step` is `null` in `publishable`, and `null` when not judged.
///
/// A workload whose GUST and hand-WGSL paths are not the same kernel (`Workload::same_kernel`, W7: GUST runs the public
/// `gust_algo::reduce`, which exposes no GPU timestamps, so its cell has no paired GPU-time ratio by construction) is
/// outside the rule, on the GUST and the hand-WGSL side alike. The exclusion is recorded, not silent: `excluded` lists
/// every such cell of the rule's backends (`backend`, `workload`, `n`, `reason`), so a reader of the run file sees W7 was
/// present and left out on purpose (coordinator decision D190, a deviation from the plan's "any cell").
/// W6-graph is left out the same way for a different reason (`Workload::gust_gpu_timestamps`: `StagedGraph` records no GPU
/// timestamps, so the form has no GPU-time quantities); W6-hand stays in the rule (D191).
/// `control` carries the same ratio for the group's control (same kernel text as the reference, so its distance from
/// 1.0 is the noise floor of the method); it does not enter the rule. Returns the record that goes into the run file.
pub fn kill_rule(group: Group, cells: &[Value]) -> Value {
    let gust_backend = group.gust_backend();
    let reference = group.reference();
    let control_backend = group.control();
    let mut checked = 0usize;
    let mut unpaired = 0usize;
    let mut offenders = Vec::new();
    let mut with_gust = std::collections::BTreeSet::new();
    let mut with_reference = std::collections::BTreeSet::new();
    let mut control = Vec::new();
    let mut excluded = Vec::new();
    for cell in cells {
        let backend = cell["backend"].as_str().unwrap_or("");
        let key = (
            cell["workload"].as_str().unwrap_or("").to_owned(),
            cell["n"].as_u64().unwrap_or(0),
        );
        let of_the_rule =
            backend == gust_backend || backend == reference || backend == control_backend;
        // A workload that is not a same-kernel pairing (W7: GUST runs `gust-algo`'s sum and the hand backends a reduction of
        // their own, and `gust-algo` has no public GPU timestamps, so a W7 GUST cell has no paired GPU-time ratio by
        // construction) is outside the rule, on the GUST and the hand-WGSL side alike; it would otherwise count as
        // `unpaired` and keep every run that includes it from being judged. The cell is listed in `excluded`.
        if Workload::parse(&key.0).is_some_and(|w| !w.same_kernel()) {
            if of_the_rule {
                excluded.push(json!({
                    "backend": backend, "workload": key.0, "n": key.1,
                    "reason": "no same-kernel GUST/hand pairing: GUST runs the library's sum, which exposes no GPU timestamps",
                }));
            }
            continue;
        }
        // A workload whose GUST path records no GPU timestamps (W6-graph: `StagedGraph` runs through `execute_graph`,
        // which records none, so the form has no GPU-time samples) has no GPU-time ratio by construction. It is outside
        // the rule on both sides, through its own predicate: it stays `same_kernel`, so its dispatch and end-to-end
        // ratios are still judged by the report. Listed in `excluded`, not silent.
        if Workload::parse(&key.0).is_some_and(|w| !w.gust_gpu_timestamps()) {
            if of_the_rule {
                excluded.push(json!({
                    "backend": backend, "workload": key.0, "n": key.1,
                    "reason": "StagedGraph runs through execute_graph, which records no GPU timestamps, so the graph form has no GPU-time samples and no GPU-time pairing",
                }));
            }
            continue;
        }
        if backend == control_backend {
            for quantity in KILL_QUANTITIES {
                if let Some(ratio) =
                    cell["metrics"][quantity]["paired_vs_prebuilt"]["median_ratio"].as_f64()
                {
                    control.push((key.clone(), quantity, ratio));
                }
            }
        }
        if backend == reference {
            with_reference.insert(key.clone());
        }
        if backend != gust_backend {
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
                            "direction": if ratio > 1.0 { "slower" } else { "faster" },
                        }));
                    }
                }
                None => unpaired += 1,
            }
        }
    }
    // A reference cell with no GUST cell beside it: the backend was dropped or never ran there.
    let missing = with_reference.difference(&with_gust).count();
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
    let slower = offenders.iter().any(|o| o["direction"] == "slower");
    let faster = offenders.iter().any(|o| o["direction"] == "faster");
    let (c_reopened, reading_step) = match (judged, group) {
        (false, _) => (Value::Null, Value::Null),
        (true, Group::Spirv) => (json!(slower), json!(faster)),
        (true, _) => (json!(slower || faster), Value::Null),
    };
    let rule = match group {
        Group::Spirv => {
            "gust-gpu-spirv/wgsl-prebuilt-unchecked GPU-time paired ratio more than 10 % above 1.0 on any cell reopens candidate C; more than 10 % below 1.0 opens a reading step (R1.3-3, D193); checks are off on both sides"
        }
        _ => {
            "gust-gpu/wgsl-prebuilt GPU-time paired ratio more than 10 % off 1.0 on any cell reopens candidate C"
        }
    };
    json!({
        "group": group.name(),
        "rule": rule,
        "reference": reference,
        "tolerance": KILL_TOLERANCE,
        "quantities": KILL_QUANTITIES,
        "checked": checked,
        "unpaired": unpaired,
        "judged": judged,
        "c_reopened": c_reopened,
        "reading_step": reading_step,
        "excluded": excluded,
        "offenders": offenders,
        "reproducibility_clause": "an offender counts when it recurs in 3/3 seeds with `control_in_tolerance` true on that cell in each of those runs; a single-seed hit is listed, not counted (plan R1.3-4; `aggregate` counts the seeds and needs at least 3 runs with distinct seeds)",
        "control": {
            "note": "the group's control / its reference, same quantities and method, same cells; same kernel text, so the distance from 1.0 is the noise floor of the rule for gpu_single. For gpu_batch_per_dispatch the control also differs in per-pass bind groups (a fresh one per pass, as GUST does; the reference reuses one), so it is not a pure noise floor there. Not an input of the rule.",
            "backend": control_backend,
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

    fn kill_rule(cells: &[Value]) -> Value {
        super::kill_rule(Group::Publishable, cells)
    }

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
        let fast = kill_rule(&[cell("gust-gpu", Some(1.0), Some(0.89))]);
        assert_eq!(fast["c_reopened"], true);
        assert_eq!(fast["offenders"][0]["backend"], "gust-gpu");
        assert_eq!(fast["offenders"][0]["direction"], "faster");
        assert!(
            fast["reading_step"].is_null(),
            "the publishable rule has no reading step"
        );
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

    /// A full run containing W6-graph (a GUST cell with no GPU-time metrics, by construction) is still judged on its
    /// other cells, lists the W6-graph exclusion with its reason, and keeps W6-hand in the rule.
    #[test]
    fn a_run_with_w6_graph_is_judged_and_lists_the_exclusion() {
        let w6 = |workload: &str, backend: &str, single: Option<f64>| {
            let mut value = named(backend, 1 << 20, single, single);
            value["workload"] = json!(workload);
            value
        };
        let record = kill_rule(&[
            w6("W6-graph", "gust-gpu", None),
            w6("W6-graph", "wgsl-prebuilt", Some(1.0)),
            w6("W6-hand", "gust-gpu", Some(1.0)),
            w6("W6-hand", "wgsl-prebuilt", Some(1.0)),
            named("gust-gpu", 1, Some(1.0), Some(1.0)),
            named("wgsl-prebuilt", 1, Some(1.0), Some(1.0)),
        ]);
        assert_eq!(record["unpaired"], 0, "W6-graph is not left unpaired");
        assert_eq!(record["checked"], 4, "W1 and W6-hand, two quantities each");
        assert_eq!(record["judged"], true);
        assert_eq!(record["c_reopened"], false, "judged, not null");
        let listed = record["excluded"].as_array().expect("excluded is a list");
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().all(|e| e["workload"] == "W6-graph"));
        assert!(
            listed[0]["reason"]
                .as_str()
                .is_some_and(|r| r.contains("execute_graph") && r.contains("no GPU timestamps"))
        );
        // W6-hand is in the rule: a wild ratio there reopens C.
        let wild = kill_rule(&[
            w6("W6-graph", "gust-gpu", None),
            w6("W6-graph", "wgsl-prebuilt", Some(1.0)),
            w6("W6-hand", "gust-gpu", Some(1.5)),
            w6("W6-hand", "wgsl-prebuilt", Some(1.0)),
        ]);
        assert_eq!(wild["c_reopened"], true);
        // W6-graph keeps `same_kernel`, so the report still judges its dispatch and end-to-end ratios.
        assert!(Workload::PipelineGraph.same_kernel());
        assert!(!Workload::PipelineGraph.gust_gpu_timestamps());
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
    fn a_table_run_needs_the_whole_group_in_its_backends() {
        let names = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let full = names(&["gust-gpu", "wgsl-prebuilt", "wgsl-percall", "cpu-serial"]);
        assert!(require_full_set(Group::Publishable, &full, "gust-gpu").is_ok());
        let no_percall = names(&["gust-gpu", "wgsl-prebuilt"]);
        let message = require_full_set(Group::Publishable, &no_percall, "gust-gpu").unwrap_err();
        assert!(message.contains("wgsl-percall"), "{message}");
        // The route twin of the other build does not stand in for this build's GUST backend.
        let wrong_route = names(&["gust-gpu-spirv", "wgsl-prebuilt", "wgsl-percall"]);
        assert!(require_full_set(Group::Publishable, &wrong_route, "gust-gpu").is_err());
        // The spirv group has its own three: the checked hand rows do not stand in for the unchecked ones.
        assert!(require_full_set(Group::Spirv, &wrong_route, "gust-gpu-spirv").is_err());
        let spirv = names(&[
            "gust-gpu-spirv",
            "wgsl-prebuilt-unchecked",
            "wgsl-percall-unchecked",
            "serial",
        ]);
        assert!(require_full_set(Group::Spirv, &spirv, "gust-gpu-spirv").is_ok());
        let no_control = names(&["gust-gpu-spirv", "wgsl-prebuilt-unchecked"]);
        let message = require_full_set(Group::Spirv, &no_control, "gust-gpu-spirv").unwrap_err();
        assert!(message.contains("wgsl-percall-unchecked"), "{message}");
        assert!(message.contains("spirv"), "{message}");
        // The ablation group takes any subset.
        assert!(require_full_set(Group::Ablation, &names(&["wgsl-percall"]), "gust-gpu").is_ok());
    }

    #[test]
    fn groups_parse_and_name_round_trip() {
        for group in [Group::Publishable, Group::Spirv, Group::Ablation] {
            assert_eq!(Group::parse(group.name()), Some(group));
        }
        assert_eq!(Group::parse("other"), None);
        assert!(
            Group::Publishable.is_table() && Group::Spirv.is_table() && !Group::Ablation.is_table()
        );
        assert_eq!(Group::Spirv.reference(), "wgsl-prebuilt-unchecked");
        assert_eq!(Group::Spirv.control(), "wgsl-percall-unchecked");
        assert_eq!(Group::Publishable.reference(), "wgsl-prebuilt");
    }

    #[test]
    fn the_route_decides_the_group() {
        assert_eq!(default_group(true), Group::Spirv);
        assert_eq!(default_group(false), Group::Publishable);
        let refused = check_route(Group::Publishable, true).unwrap_err();
        assert!(
            refused.contains("D193") && refused.contains("--group spirv"),
            "{refused}"
        );
        assert!(
            check_route(Group::Spirv, false)
                .unwrap_err()
                .contains("spirv-route")
        );
        for ok in [
            (Group::Spirv, true),
            (Group::Publishable, false),
            (Group::Ablation, true),
            (Group::Ablation, false),
        ] {
            assert!(check_route(ok.0, ok.1).is_ok());
        }
    }

    fn spirv_rule(cells: &[Value]) -> Value {
        super::kill_rule(Group::Spirv, cells)
    }

    /// The spirv group judges `gust-gpu-spirv` against the unchecked prebuilt row, with the unchecked percall row as
    /// the control. Slower re-opens C, faster is a reading step and not a re-opening (R1.3-3).
    #[test]
    fn the_spirv_group_splits_slower_from_faster() {
        let rows = |ratio: f64, batch: f64, control: f64| {
            vec![
                cell("gust-gpu-spirv", Some(ratio), Some(batch)),
                cell("wgsl-prebuilt-unchecked", Some(1.0), Some(1.0)),
                cell("wgsl-percall-unchecked", Some(control), Some(1.0)),
            ]
        };
        let pass = spirv_rule(&rows(1.05, 0.95, 1.01));
        assert_eq!(pass["group"], "spirv");
        assert_eq!(pass["reference"], "wgsl-prebuilt-unchecked");
        assert_eq!(pass["judged"], true);
        assert_eq!(pass["c_reopened"], false);
        assert_eq!(pass["reading_step"], false);
        assert_eq!(pass["control"]["backend"], "wgsl-percall-unchecked");
        assert_eq!(pass["control"]["checked"], 2);
        let slower = spirv_rule(&rows(1.2, 1.0, 1.01));
        assert_eq!(slower["c_reopened"], true);
        assert_eq!(slower["reading_step"], false);
        assert_eq!(slower["offenders"][0]["direction"], "slower");
        assert_eq!(slower["offenders"][0]["control_in_tolerance"], true);
        let faster = spirv_rule(&rows(1.0, 0.7, 1.01));
        assert_eq!(faster["c_reopened"], false, "faster does not re-open C");
        assert_eq!(faster["reading_step"], true);
        assert_eq!(faster["offenders"][0]["direction"], "faster");
        // Both, on different quantities: both marks.
        let both = spirv_rule(&rows(1.2, 0.7, 1.0));
        assert_eq!(both["c_reopened"], true);
        assert_eq!(both["reading_step"], true);
        // The boundary is inside, as in the other group.
        let edge = spirv_rule(&rows(1.10, 0.90, 1.0));
        assert_eq!(edge["c_reopened"], false);
        assert_eq!(edge["reading_step"], false);
    }

    #[test]
    fn the_spirv_group_leaves_unpaired_cells_unjudged_and_ignores_the_other_groups_rows() {
        // A reference cell with no GUST cell, a GUST cell without ratios: nothing judged, both marks null.
        let missing = spirv_rule(&[named("wgsl-prebuilt-unchecked", 1, Some(1.0), Some(1.0))]);
        assert_eq!(missing["unpaired"], 2);
        assert!(missing["c_reopened"].is_null() && missing["reading_step"].is_null());
        let failed = spirv_rule(&[named("gust-gpu-spirv", 1, None, None)]);
        assert_eq!(failed["unpaired"], 2);
        assert!(failed["c_reopened"].is_null());
        // The checked hand rows and gust-gpu are not this group's rows.
        let foreign = spirv_rule(&[
            named("gust-gpu", 1, Some(3.0), Some(3.0)),
            named("wgsl-prebuilt", 1, Some(1.0), Some(1.0)),
            named("wgsl-percall", 1, Some(1.0), Some(1.0)),
        ]);
        assert_eq!(foreign["checked"], 0);
        assert_eq!(foreign["control"]["checked"], 0);
        assert!(foreign["c_reopened"].is_null());
        // And the publishable rule does not judge gust-gpu-spirv cells.
        let publishable = kill_rule(&[
            named("gust-gpu-spirv", 1, Some(3.0), Some(3.0)),
            named("wgsl-prebuilt", 1, Some(1.0), Some(1.0)),
        ]);
        assert_eq!(publishable["checked"], 0);
        assert_eq!(
            publishable["unpaired"], 2,
            "the reference cell has no GUST cell of this group"
        );
        assert!(publishable["c_reopened"].is_null());
    }

    /// D190 and D191 hold in the spirv group the same way: the W7 and W6-graph cells of the group's rows are listed
    /// as excluded and neither counted nor judged.
    #[test]
    fn the_spirv_group_applies_the_d190_and_d191_exclusions() {
        let on = |workload: &str, backend: &str, ratio: Option<f64>| {
            let mut value = named(backend, 1 << 20, ratio, ratio);
            value["workload"] = json!(workload);
            value
        };
        let record = spirv_rule(&[
            on("W7-f32", "gust-gpu-spirv", None),
            on("W7-f32", "wgsl-prebuilt-unchecked", Some(1.0)),
            on("W7-f32", "wgsl-percall-unchecked", Some(1.7)),
            on("W6-graph", "gust-gpu-spirv", None),
            on("W6-graph", "wgsl-prebuilt-unchecked", Some(1.0)),
            named("gust-gpu-spirv", 1, Some(1.0), Some(1.0)),
            named("wgsl-prebuilt-unchecked", 1, Some(1.0), Some(1.0)),
        ]);
        assert_eq!(record["judged"], true);
        assert_eq!(record["unpaired"], 0);
        assert_eq!(record["checked"], 2);
        assert_eq!(record["c_reopened"], false);
        assert_eq!(record["control"]["checked"], 0);
        let listed = record["excluded"].as_array().expect("excluded is a list");
        assert_eq!(listed.len(), 5);
        assert!(
            listed
                .iter()
                .any(|e| e["backend"] == "gust-gpu-spirv" && e["workload"] == "W7-f32")
        );
        assert!(
            listed
                .iter()
                .any(|e| e["backend"] == "wgsl-prebuilt-unchecked" && e["workload"] == "W6-graph")
        );
    }
}

#[cfg(test)]
mod reproducibility_tests {
    use super::*;

    fn kill_rule(cells: &[Value]) -> Value {
        super::kill_rule(Group::Publishable, cells)
    }

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

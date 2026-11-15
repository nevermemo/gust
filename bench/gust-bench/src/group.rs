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

/// The paired-ratio window of the kill rule: more than 10 % off 1.0 reopens candidate C.
pub const KILL_TOLERANCE: f64 = 0.10;

/// The quantities the rule reads: the GPU-timestamp ones (the host-clock ones include driver and queue overhead that
/// is not the kernel).
pub const KILL_QUANTITIES: [&str; 2] = ["gpu_single", "gpu_batch_per_dispatch"];

/// The GUST backends the rule judges against wgsl-prebuilt.
pub const GUST_BACKENDS: [&str; 2] = ["gust-gpu", "gust-gpu-spirv"];

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

/// The kill rule over the cells of a run: a gust-gpu (or gust-gpu-spirv) GPU-time paired ratio to wgsl-prebuilt more
/// than 10 % off 1.0 on any cell marks the run "C re-opened". Reads `paired_vs_prebuilt.median_ratio` of
/// [`KILL_QUANTITIES`]. A GUST cell with no paired ratio is counted in `unpaired` (the rule could not judge it) and
/// does not reopen anything by itself. Returns the record that goes into the run file.
pub fn kill_rule(cells: &[Value]) -> Value {
    let mut checked = 0usize;
    let mut unpaired = 0usize;
    let mut offenders = Vec::new();
    for cell in cells {
        let backend = cell["backend"].as_str().unwrap_or("");
        if !GUST_BACKENDS.contains(&backend) {
            continue;
        }
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
                None => {
                    if cell["metrics"][quantity].is_object() {
                        unpaired += 1;
                    }
                }
            }
        }
    }
    json!({
        "rule": "gust-gpu/wgsl-prebuilt GPU-time paired ratio more than 10 % off 1.0 on any cell reopens candidate C",
        "tolerance": KILL_TOLERANCE,
        "quantities": KILL_QUANTITIES,
        "checked": checked,
        "unpaired": unpaired,
        "c_reopened": !offenders.is_empty(),
        "offenders": offenders,
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
        assert_eq!(record["c_reopened"], false);
        assert_eq!(record["checked"], 1);
        assert_eq!(record["unpaired"], 1);
        // Nothing to judge is not a pass either: `checked` is 0 and the record says so.
        assert_eq!(kill_rule(&[])["checked"], 0);
    }

    #[test]
    fn groups_parse_and_name_round_trip() {
        for group in [Group::Publishable, Group::Ablation] {
            assert_eq!(Group::parse(group.name()), Some(group));
        }
        assert_eq!(Group::parse("other"), None);
    }
}

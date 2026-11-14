//! The quiet-window sampler (plan 2.4 items 3 and 4), **recording only**.
//!
//! Before each cell the harness waits a settle interval, then samples for a window at a fixed rate:
//!
//! - GPU-engine utilisation from the Windows performance counter `\GPU Engine(*)\Utilization Percentage` (PDH). The
//!   instance names carry the owning process and the adapter LUID, so one source serves every adapter.
//! - The process table (pid, parent pid, name, CPU time, creation time) at the window's start and at its end.
//!   `Win32_Process` carries the same fields; this reads them through ToolHelp and `GetProcessTimes`, which spawns no
//!   helper process of its own (a `powershell` call would itself be a foreign process in the table it reports).
//! - The harness's PID ancestry (harness, its shell-like parent chain, the children of that chain), computed from the
//!   start table and recorded; no process is dropped from the record, only flagged by the ancestry block.
//!
//! Nothing here judges anything: there are no thresholds and the `publishable` flag stays false (b1b sets thresholds
//! from these records). Everything goes to the `<results>.samples.json` sidecar. On a platform without the counters
//! the record says so and carries no samples.

#[cfg(windows)]
mod win;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

/// Settle interval before a window (plan 2.4: B1 saw 0 to 47 % utilisation between cells from its own work).
pub const SETTLE: Duration = Duration::from_secs(3);
/// Length of the sampling window.
pub const WINDOW: Duration = Duration::from_secs(2);
/// Samples per second in the window.
pub const RATE_HZ: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub settle: Duration,
    pub window: Duration,
    pub rate_hz: u32,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            settle: SETTLE,
            window: WINDOW,
            rate_hz: RATE_HZ,
        }
    }
}

impl Timing {
    /// Samples taken in one window (8 for the defaults).
    pub fn samples(&self) -> usize {
        (self.window.as_secs_f64() * f64::from(self.rate_hz)).round() as usize
    }
}

/// A DXGI adapter locally unique id, in the two halves PDH prints (`luid_0x<high>_0x<low>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Luid {
    pub high: u32,
    pub low: u32,
}

impl Luid {
    pub fn text(self) -> String {
        format!("0x{:08X}_0x{:08X}", self.high, self.low)
    }
}

/// One parsed `\GPU Engine(<instance>)` name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Engine {
    pub pid: u32,
    pub luid: Luid,
    pub phys: u32,
    pub eng: u32,
    pub engtype: String,
}

fn parse_hex(text: &str) -> Option<u32> {
    let digits = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))?;
    (!digits.is_empty() && digits.len() <= 8)
        .then(|| u32::from_str_radix(digits, 16).ok())
        .flatten()
}

fn parse_dec(text: &str) -> Option<u32> {
    (!text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()))
        .then(|| text.parse().ok())
        .flatten()
}

/// Parses `pid_1234_luid_0x00000000_0x0001A2B3_phys_0_eng_0_engtype_3D`. The engine type is everything after
/// `engtype_` (it can hold underscores). A name that does not have exactly this shape is `None`: the caller records
/// it as unparsed instead of dropping it.
pub fn parse_instance(name: &str) -> Option<Engine> {
    let parts = name.split('_').collect::<Vec<_>>();
    if parts.len() < 11 {
        return None;
    }
    let keys = [
        (0, "pid"),
        (2, "luid"),
        (5, "phys"),
        (7, "eng"),
        (9, "engtype"),
    ];
    if keys
        .iter()
        .any(|(index, key)| !parts[*index].eq_ignore_ascii_case(key))
    {
        return None;
    }
    let engtype = parts[10..].join("_");
    if engtype.is_empty() {
        return None;
    }
    Some(Engine {
        pid: parse_dec(parts[1])?,
        luid: Luid {
            high: parse_hex(parts[3])?,
            low: parse_hex(parts[4])?,
        },
        phys: parse_dec(parts[6])?,
        eng: parse_dec(parts[8])?,
        engtype,
    })
}

/// One row of the process table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proc {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
    /// Kernel plus user time in 100 ns units; `None` when the process cannot be opened for query.
    pub cpu_100ns: Option<u64>,
    /// Creation time as a FILETIME (100 ns since 1601); guards against a reused parent pid.
    pub created: Option<u64>,
}

/// Names (lower case, no `.exe`) the session-shell walk climbs through: the harness itself, cargo and the shells
/// plan 2.4 names. The walk stops at the first ancestor that is not one of these, so it never reaches the desktop
/// session's root and excludes nothing it should not.
const SHELLISH: [&str; 9] = [
    "gust-bench",
    "cargo",
    "bash",
    "sh",
    "pwsh",
    "powershell",
    "cmd",
    "env",
    "git-bash",
];

fn stem(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    lower.strip_suffix(".exe").unwrap_or(&lower).to_owned()
}

/// What the PID-ancestry rule excludes, from one process table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ancestry {
    /// The harness first, then its parents upward, while they are shell-like.
    pub chain: Vec<u32>,
    /// Direct children of any chain member (the harness's own helper processes, the other children of the shell),
    /// that are not themselves in the chain.
    pub children: Vec<u32>,
    /// The first ancestor that is not shell-like (not excluded), and why the walk ended.
    pub stopped_at: Option<u32>,
    pub reason: &'static str,
}

/// The harness (`own`), its parent chain up to the session shell, and their children.
///
/// A parent counts only if it exists in the table, is not already in the chain, and was not created after its child
/// (a pid can be reused, so a parent pid can name an unrelated younger process).
pub fn ancestry(table: &[Proc], own: u32) -> Ancestry {
    let by_pid = table.iter().map(|p| (p.pid, p)).collect::<BTreeMap<_, _>>();
    let mut chain = vec![own];
    let mut stopped_at = None;
    let mut reason = "harness not in the process table";
    if let Some(&start) = by_pid.get(&own) {
        let mut current = start;
        reason = "reached the root of the process tree";
        while let Some(&parent) = by_pid.get(&current.ppid) {
            if chain.contains(&parent.pid) {
                reason = "parent pid cycle";
                break;
            }
            if let (Some(parent_created), Some(child_created)) = (parent.created, current.created)
                && parent_created > child_created
            {
                reason = "parent pid reused by a younger process";
                break;
            }
            if !SHELLISH.contains(&stem(&parent.name).as_str()) {
                stopped_at = Some(parent.pid);
                reason = "first ancestor that is not a shell or cargo";
                break;
            }
            chain.push(parent.pid);
            current = parent;
        }
    }
    let members = chain.iter().copied().collect::<BTreeSet<_>>();
    let children = table
        .iter()
        .filter(|p| {
            members.contains(&p.ppid)
                && !members.contains(&p.pid)
                && p.pid != p.ppid
                && by_pid
                    .get(&p.ppid)
                    .is_some_and(|parent| match (parent.created, p.created) {
                        (Some(parent_created), Some(created)) => parent_created <= created,
                        _ => true,
                    })
        })
        .map(|p| p.pid)
        .collect();
    Ancestry {
        chain,
        children,
        stopped_at,
        reason,
    }
}

/// A DXGI adapter as the sampler sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DxgiAdapter {
    pub index: u32,
    pub description: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub luid: Luid,
    pub software: bool,
}

/// The outcome of mapping the adapter under test to a LUID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mapping {
    Mapped(Luid),
    /// The mapping failed; the sampler records every adapter's activity (plan 2.4's stricter fallback).
    Failed(String),
}

/// Maps the wgpu adapter (PCI vendor and device id, name) to one non-software DXGI adapter. Two adapters with the
/// same ids are told apart by name; anything still ambiguous or absent is a failure, never a guess.
pub fn map_adapter(adapters: &[DxgiAdapter], vendor: u32, device: u32, name: &str) -> Mapping {
    let by_id = adapters
        .iter()
        .filter(|a| !a.software && a.vendor_id == vendor && a.device_id == device)
        .collect::<Vec<_>>();
    match by_id.as_slice() {
        [one] => Mapping::Mapped(one.luid),
        [] => Mapping::Failed(format!(
            "no DXGI adapter has vendor {vendor:#06x} device {device:#06x}"
        )),
        many => {
            let named = many
                .iter()
                .filter(|a| a.description.eq_ignore_ascii_case(name))
                .collect::<Vec<_>>();
            match named.as_slice() {
                [one] => Mapping::Mapped(one.luid),
                _ => Mapping::Failed(format!(
                    "{} DXGI adapters share vendor {vendor:#06x} device {device:#06x} and the name does not decide",
                    many.len()
                )),
            }
        }
    }
}

/// `<results>.json` becomes `<results>.samples.json`; a path without that extension gets `.samples.json` appended.
pub fn sidecar_path(results: &str) -> PathBuf {
    let path = Path::new(results);
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
    {
        path.with_extension("samples.json")
    } else {
        PathBuf::from(format!("{results}.samples.json"))
    }
}

/// One PDH reading of every engine instance.
pub struct Reading {
    pub t_ms: f64,
    pub engines: Vec<(Engine, f64)>,
    zero: usize,
    invalid: usize,
    unparsed: Vec<String>,
}

impl Reading {
    fn json(&self) -> Value {
        json!({
            "t_ms": self.t_ms,
            "instances": self.engines.len() + self.zero + self.invalid + self.unparsed.len(),
            "zero_instances": self.zero,
            "invalid_instances": self.invalid,
            "unparsed_instances": self.unparsed,
            // Only instances with a non-zero value are listed; the zero ones are counted above.
            "engines": self.engines.iter().map(|(e, pct)| json!([
                e.pid, e.luid.text(), e.phys, e.eng, e.engtype, pct,
            ])).collect::<Vec<_>>(),
        })
    }
}

fn reading_from(t_ms: f64, raw: Vec<(String, f64, bool)>) -> Reading {
    let mut reading = Reading {
        t_ms,
        engines: Vec::new(),
        zero: 0,
        invalid: 0,
        unparsed: Vec::new(),
    };
    for (name, value, valid) in raw {
        if !valid {
            reading.invalid += 1;
            continue;
        }
        match parse_instance(&name) {
            Some(_) if value == 0.0 => reading.zero += 1,
            Some(engine) => reading.engines.push((engine, value)),
            None => reading.unparsed.push(name),
        }
    }
    reading
}

fn process_rows(table: &[Proc]) -> Value {
    Value::Array(
        table
            .iter()
            .map(|p| json!([p.pid, p.ppid, p.name, p.cpu_100ns, p.created]))
            .collect(),
    )
}

/// The sampler: opened once per run, used once per cell.
pub struct Sampler {
    timing: Timing,
    own_pid: u32,
    adapters: Result<Vec<DxgiAdapter>, String>,
    mapping: Mapping,
    adapter_name: String,
}

impl Sampler {
    pub fn open(timing: Timing, adapter: &wgpu::AdapterInfo) -> Self {
        #[cfg(windows)]
        let adapters = win::dxgi_adapters();
        #[cfg(not(windows))]
        let adapters = Err("the GPU-engine counters exist on Windows only".to_owned());
        let mapping = match &adapters {
            Ok(list) => map_adapter(list, adapter.vendor, adapter.device, &adapter.name),
            Err(reason) => Mapping::Failed(format!("DXGI enumeration failed: {reason}")),
        };
        Self {
            timing,
            own_pid: std::process::id(),
            adapters,
            mapping,
            adapter_name: adapter.name.clone(),
        }
    }

    pub fn mapped_luid(&self) -> Option<Luid> {
        match self.mapping {
            Mapping::Mapped(luid) => Some(luid),
            Mapping::Failed(_) => None,
        }
    }

    /// Run-level facts for the sidecar header.
    pub fn header(&self) -> Value {
        let (mapped, reason) = match &self.mapping {
            Mapping::Mapped(luid) => (Some(luid.text()), None),
            Mapping::Failed(reason) => (None, Some(reason.clone())),
        };
        let dxgi = self.adapters.as_ref().map_or(Value::Null, |list| {
            Value::Array(
                list.iter()
                    .map(|a| {
                        json!({
                            "index": a.index, "description": a.description,
                            "vendor_id": a.vendor_id, "device_id": a.device_id,
                            "luid": a.luid.text(), "software": a.software,
                        })
                    })
                    .collect(),
            )
        });
        json!({
            "schema": 1,
            "source": "PDH \\GPU Engine(*)\\Utilization Percentage; ToolHelp process table with GetProcessTimes",
            "judgement": "none: recording only, thresholds are set later from these records",
            "settle_s": self.timing.settle.as_secs_f64(),
            "window_s": self.timing.window.as_secs_f64(),
            "rate_hz": self.timing.rate_hz,
            "samples_per_window": self.timing.samples(),
            "adapter_under_test": self.adapter_name,
            "adapter_luid": mapped,
            "luid_mapping": if reason.is_some() { "failed: every adapter is recorded" } else { "mapped" },
            "luid_mapping_failure": reason,
            "dxgi_adapters": dxgi,
            "dxgi_error": self.adapters.as_ref().err(),
            "harness_pid": self.own_pid,
            "engines_row": ["pid", "luid", "phys", "eng", "engtype", "percent"],
            "process_row": ["pid", "ppid", "name", "cpu_100ns", "created_filetime"],
            "note": "engine rows list non-zero instances only; zero instances are counted per sample",
        })
    }

    /// Settle, then one window, for the cell `cell` (a JSON object naming backend, workload and size).
    pub fn sample_cell(&self, cell: Value) -> Value {
        #[cfg(windows)]
        {
            self.sample_windows(cell)
        }
        #[cfg(not(windows))]
        {
            json!({ "cell": cell, "unavailable": "the GPU-engine counters exist on Windows only" })
        }
    }

    #[cfg(windows)]
    fn sample_windows(&self, cell: Value) -> Value {
        thread::sleep(self.timing.settle);
        let window_started = Instant::now();
        let smi = crate::meta::gpu_snapshot();
        let start_table = win::process_table();
        let pdh = win::Pdh::open();
        let mut readings = Vec::new();
        let mut pdh_error = None;
        match &pdh {
            Err(reason) => pdh_error = Some(reason.clone()),
            Ok(pdh) => match pdh.collect() {
                Err(reason) => pdh_error = Some(reason),
                Ok(()) => {
                    let primed = Instant::now();
                    let period = Duration::from_secs_f64(1.0 / f64::from(self.timing.rate_hz));
                    for k in 1..=self.timing.samples() {
                        let due = primed + period.mul_f64(k as f64);
                        if let Some(wait) = due.checked_duration_since(Instant::now()) {
                            thread::sleep(wait);
                        }
                        let t_ms = primed.elapsed().as_secs_f64() * 1000.0;
                        match pdh.collect().and_then(|()| pdh.read()) {
                            Ok(raw) => readings.push(reading_from(t_ms, raw)),
                            Err(reason) => {
                                pdh_error = Some(reason);
                                break;
                            }
                        }
                    }
                }
            },
        }
        let end_table = win::process_table();
        let ancestry_json = match &start_table {
            Ok(table) => {
                let a = ancestry(table, self.own_pid);
                let name_of = |pid: u32| {
                    table
                        .iter()
                        .find(|p| p.pid == pid)
                        .map_or(Value::Null, |p| json!(p.name))
                };
                json!({
                    "chain": a.chain.iter().map(|pid| json!({ "pid": pid, "name": name_of(*pid) })).collect::<Vec<_>>(),
                    "children": a.children,
                    "stopped_at": a.stopped_at.map(|pid| json!({ "pid": pid, "name": name_of(pid) })),
                    "reason": a.reason,
                })
            }
            Err(reason) => json!({ "error": reason }),
        };
        json!({
            "cell": cell,
            "window_wall_s": window_started.elapsed().as_secs_f64(),
            "adapter_luid": self.mapped_luid().map(Luid::text),
            "pdh": {
                "error": pdh_error,
                "samples": readings.iter().map(Reading::json).collect::<Vec<_>>(),
            },
            "nvidia_smi": smi,
            "processes_start": start_table.as_ref().map_or(Value::Null, |t| process_rows(t)),
            "processes_start_error": start_table.as_ref().err(),
            "processes_end": end_table.as_ref().map_or(Value::Null, |t| process_rows(t)),
            "processes_end_error": end_table.as_ref().err(),
            "ancestry": ancestry_json,
        })
    }
}

/// The largest per-sample total utilisation of `pid` on each adapter LUID over the readings (the statistic plan 2.4
/// uses is a maximum over the window).
pub fn max_by_luid(readings: &[Reading], pid: u32) -> BTreeMap<Luid, f64> {
    let mut max = BTreeMap::new();
    for reading in readings {
        let mut sums: BTreeMap<Luid, f64> = BTreeMap::new();
        for (engine, pct) in &reading.engines {
            if engine.pid == pid {
                *sums.entry(engine.luid).or_default() += pct;
            }
        }
        for (luid, sum) in sums {
            let slot = max.entry(luid).or_insert(0.0);
            *slot = f64::max(*slot, sum);
        }
    }
    max
}

#[cfg(windows)]
impl Sampler {
    /// Samples the GPU-engine counters for `duration` at the sampler's rate, without settling or a process table.
    pub fn observe(&self, duration: Duration) -> Result<Vec<Reading>, String> {
        let pdh = win::Pdh::open()?;
        pdh.collect()?;
        let primed = Instant::now();
        let period = Duration::from_secs_f64(1.0 / f64::from(self.timing.rate_hz));
        let count = (duration.as_secs_f64() * f64::from(self.timing.rate_hz)).round() as u32;
        let mut readings = Vec::new();
        for k in 1..=count {
            if let Some(wait) = (primed + period * k).checked_duration_since(Instant::now()) {
                thread::sleep(wait);
            }
            let t_ms = primed.elapsed().as_secs_f64() * 1000.0;
            pdh.collect()?;
            readings.push(reading_from(t_ms, pdh.read()?));
        }
        Ok(readings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAME: &str = "pid_8124_luid_0x00000000_0x0001A2B3_phys_0_eng_3_engtype_3D";

    #[test]
    fn parses_an_engine_instance_name() {
        let engine = parse_instance(NAME).expect("parses");
        assert_eq!(
            engine,
            Engine {
                pid: 8124,
                luid: Luid {
                    high: 0,
                    low: 0x0001_A2B3
                },
                phys: 0,
                eng: 3,
                engtype: "3D".to_owned(),
            }
        );
        assert_eq!(engine.luid.text(), "0x00000000_0x0001A2B3");
    }

    #[test]
    fn engine_type_keeps_its_underscores_and_a_high_luid_half_survives() {
        let engine =
            parse_instance("pid_4_luid_0xFFFFFFFF_0x0000DEAD_phys_1_eng_0_engtype_Video_Decode_2")
                .expect("parses");
        assert_eq!(engine.pid, 4);
        assert_eq!(engine.luid.high, 0xFFFF_FFFF);
        assert_eq!(engine.luid.low, 0xDEAD);
        assert_eq!(engine.phys, 1);
        assert_eq!(engine.engtype, "Video_Decode_2");
        // Keys are matched without regard to case; the engine type keeps the case it had.
        assert!(parse_instance("PID_7_LUID_0x0_0x1_PHYS_0_ENG_0_ENGTYPE_Copy").is_some());
    }

    #[test]
    fn rejects_malformed_instance_names() {
        for bad in [
            "",
            "_Total",
            "pid_12_luid_0x0_0x1_phys_0_eng_0",
            "pid_12_luid_0x0_0x1_phys_0_eng_0_engtype_",
            "pid_x_luid_0x0_0x1_phys_0_eng_0_engtype_3D",
            "pid_12_luid_0_0x1_phys_0_eng_0_engtype_3D",
            "pid_12_luid_0x_0x1_phys_0_eng_0_engtype_3D",
            "pid_12_luid_0x123456789_0x1_phys_0_eng_0_engtype_3D",
            "pid_-3_luid_0x0_0x1_phys_0_eng_0_engtype_3D",
            "proc_12_luid_0x0_0x1_phys_0_eng_0_engtype_3D",
            "pid_12_luid_0x0_0x1_phys_0_eng_0_type_3D_x",
        ] {
            assert_eq!(parse_instance(bad), None, "{bad:?}");
        }
    }

    fn proc(pid: u32, ppid: u32, name: &str, created: u64) -> Proc {
        Proc {
            pid,
            ppid,
            name: name.to_owned(),
            cpu_100ns: Some(0),
            created: Some(created),
        }
    }

    #[test]
    fn ancestry_climbs_shells_stops_at_the_first_other_process_and_adds_children() {
        let table = [
            proc(1, 0, "System", 1),
            proc(10, 1, "claude.exe", 10),
            proc(20, 10, "bash.exe", 20),
            proc(30, 20, "bash.exe", 30),
            proc(40, 30, "cargo.exe", 40),
            proc(50, 40, "gust-bench.exe", 50),
            proc(51, 50, "nvidia-smi.exe", 51),
            // A sibling under the same shell and an unrelated process under the stopping ancestor.
            proc(60, 30, "rustc.exe", 60),
            proc(70, 10, "python.exe", 70),
            // A grandchild of the shell is not a child of a chain member.
            proc(61, 60, "link.exe", 61),
        ];
        let a = ancestry(&table, 50);
        assert_eq!(a.chain, vec![50, 40, 30, 20]);
        assert_eq!(a.stopped_at, Some(10));
        assert_eq!(a.reason, "first ancestor that is not a shell or cargo");
        assert_eq!(a.children, vec![51, 60]);
    }

    #[test]
    fn ancestry_ignores_a_reused_parent_pid_and_survives_cycles() {
        // pid 20 is named as the parent of 50 but was created after it: a reused pid, not a parent.
        let table = [
            proc(20, 1, "bash.exe", 900),
            proc(50, 20, "gust-bench.exe", 100),
        ];
        let a = ancestry(&table, 50);
        assert_eq!(a.chain, vec![50]);
        assert_eq!(a.reason, "parent pid reused by a younger process");
        assert!(a.children.is_empty());

        let cycle = [
            proc(50, 20, "gust-bench.exe", 100),
            proc(20, 50, "bash.exe", 50),
        ];
        let a = ancestry(&cycle, 50);
        assert_eq!(a.chain, vec![50, 20]);
        assert_eq!(a.reason, "parent pid cycle");

        let a = ancestry(&[], 50);
        assert_eq!(a.chain, vec![50]);
        assert_eq!(a.reason, "harness not in the process table");
    }

    fn adapter(index: u32, description: &str, vendor: u32, device: u32, low: u32) -> DxgiAdapter {
        DxgiAdapter {
            index,
            description: description.to_owned(),
            vendor_id: vendor,
            device_id: device,
            luid: Luid { high: 0, low },
            software: false,
        }
    }

    #[test]
    fn maps_the_adapter_by_ids_then_name_and_fails_rather_than_guesses() {
        let list = [
            adapter(0, "NVIDIA GeForce RTX 5090", 0x10DE, 0x2B85, 0x100),
            adapter(1, "AMD Radeon(TM) 890M Graphics", 0x1002, 0x150E, 0x200),
            DxgiAdapter {
                software: true,
                ..adapter(2, "Microsoft Basic Render Driver", 0x1414, 0x8C, 0x300)
            },
        ];
        assert_eq!(
            map_adapter(&list, 0x10DE, 0x2B85, "NVIDIA GeForce RTX 5090"),
            Mapping::Mapped(Luid {
                high: 0,
                low: 0x100
            })
        );
        assert!(matches!(
            map_adapter(&list, 0x1414, 0x8C, "Microsoft Basic Render Driver"),
            Mapping::Failed(_)
        ));
        assert!(matches!(
            map_adapter(&list, 0x10DE, 0x1234, "x"),
            Mapping::Failed(_)
        ));
        // Two identical cards: the name decides only when it is unique.
        let twins = [
            adapter(0, "Card A", 1, 2, 0x10),
            adapter(1, "Card B", 1, 2, 0x20),
        ];
        assert_eq!(
            map_adapter(&twins, 1, 2, "card b"),
            Mapping::Mapped(Luid { high: 0, low: 0x20 })
        );
        assert!(matches!(
            map_adapter(&twins, 1, 2, "Card C"),
            Mapping::Failed(_)
        ));
    }

    #[test]
    fn sidecar_path_replaces_or_appends() {
        assert_eq!(
            sidecar_path("out/run1.json"),
            PathBuf::from("out/run1.samples.json")
        );
        assert_eq!(
            sidecar_path("run1.JSON"),
            PathBuf::from("run1.samples.json")
        );
        assert_eq!(sidecar_path("run1"), PathBuf::from("run1.samples.json"));
    }

    #[test]
    fn default_timing_is_a_three_second_settle_and_a_two_second_window_at_4_hz() {
        let timing = Timing::default();
        assert_eq!(timing.settle, Duration::from_secs(3));
        assert_eq!(timing.window, Duration::from_secs(2));
        assert_eq!(timing.rate_hz, 4);
        assert_eq!(timing.samples(), 8);
    }

    #[test]
    fn a_reading_counts_zero_invalid_and_unparsed_instances_and_lists_the_rest() {
        let reading = reading_from(
            250.0,
            vec![
                (NAME.to_owned(), 12.5, true),
                (
                    "pid_9_luid_0x0_0x1_phys_0_eng_0_engtype_Copy".to_owned(),
                    0.0,
                    true,
                ),
                (
                    "pid_9_luid_0x0_0x1_phys_0_eng_1_engtype_3D".to_owned(),
                    7.0,
                    false,
                ),
                ("garbage".to_owned(), 3.0, true),
            ],
        );
        let json = reading.json();
        assert_eq!(json["instances"], 4);
        assert_eq!(json["zero_instances"], 1);
        assert_eq!(json["invalid_instances"], 1);
        assert_eq!(json["unparsed_instances"], json!(["garbage"]));
        assert_eq!(
            json["engines"],
            json!([[8124, "0x00000000_0x0001A2B3", 0, 3, "3D", 12.5]])
        );
    }

    #[test]
    fn max_by_luid_sums_a_pids_engines_per_sample_and_takes_the_maximum() {
        let a = Luid { high: 0, low: 1 };
        let b = Luid { high: 0, low: 2 };
        let engine = |pid, luid, eng| Engine {
            pid,
            luid,
            phys: 0,
            eng,
            engtype: "3D".to_owned(),
        };
        let reading = |engines| Reading {
            t_ms: 0.0,
            engines,
            zero: 0,
            invalid: 0,
            unparsed: Vec::new(),
        };
        let readings = [
            reading(vec![
                (engine(5, a, 0), 10.0),
                (engine(5, a, 1), 5.0),
                (engine(6, a, 0), 90.0),
            ]),
            reading(vec![(engine(5, a, 0), 20.0), (engine(5, b, 0), 1.0)]),
        ];
        let max = max_by_luid(&readings, 5);
        assert_eq!(max.get(&a), Some(&20.0));
        assert_eq!(max.get(&b), Some(&1.0));
        assert_eq!(max_by_luid(&readings, 7).len(), 0);
    }
}

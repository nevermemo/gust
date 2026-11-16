//! Run metadata and machine-state snapshots. Nothing here records a path or a user name: only product names,
//! versions and counts.

use std::process::Command;

use serde_json::{Value, json};

/// Runs `program args` and returns its trimmed stdout, or `None` when it cannot run or fails.
fn capture(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn cpu_name() -> String {
    if cfg!(windows) {
        let key = r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0";
        if let Some(text) = capture("reg", &["query", key, "/v", "ProcessorNameString"])
            && let Some(line) = text.lines().find(|l| l.contains("ProcessorNameString"))
            && let Some((_, name)) = line.split_once("REG_SZ")
        {
            return name.trim().to_owned();
        }
    } else if let Some(text) = capture("sh", &["-c", "grep -m1 'model name' /proc/cpuinfo"])
        && let Some((_, name)) = text.split_once(':')
    {
        return name.trim().to_owned();
    }
    "unknown".to_owned()
}

fn os_name() -> String {
    if cfg!(windows) {
        capture("cmd", &["/C", "ver"]).unwrap_or_else(|| "Windows".to_owned())
    } else {
        capture("uname", &["-sr"]).unwrap_or_else(|| std::env::consts::OS.to_owned())
    }
}

/// The active power scheme's name (not its GUID).
fn power_plan() -> String {
    if cfg!(windows)
        && let Some(text) = capture("powercfg", &["/getactivescheme"])
        && let (Some(open), Some(close)) = (text.rfind('('), text.rfind(')'))
        && open < close
    {
        return text[open + 1..close].to_owned();
    }
    "unknown".to_owned()
}

fn commit() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let output = Command::new("git")
        .args(["-c", "safe.directory=*", "rev-parse", "--short=12", "HEAD"])
        .current_dir(manifest_dir)
        .output();
    match output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        _ => "unknown".to_owned(),
    }
}

/// What git says about the source tree this binary is run from (plan 2.4 item 1). Every field is `None` when git cannot
/// say: the judgement treats that as "not publishable", never as a pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GitState {
    /// `git rev-parse HEAD`, the full id.
    pub head: Option<String>,
    /// Whether `git status --porcelain --untracked-files=all` lists anything (a user setting such as
    /// `status.showUntrackedFiles=no` cannot hide an untracked file from it).
    pub dirty: Option<bool>,
    /// `git rev-parse origin/main`.
    pub origin_main: Option<String>,
    /// `git merge-base --is-ancestor HEAD origin/main`: exit 0 is yes, exit 1 is no, anything else is unknown.
    pub ancestor_of_origin_main: Option<bool>,
}

impl GitState {
    /// The first 12 characters of HEAD, or `unknown`.
    pub fn short(&self) -> String {
        self.head.as_deref().map_or_else(
            || "unknown".to_owned(),
            |head| head.chars().take(12).collect(),
        )
    }
}

/// One git call: its arguments (after `git -c safe.directory=*`) in, its exit code and trimmed standard output out
/// (`None` when git could not be run).
pub type GitCall<'a> = dyn Fn(&[&str]) -> Option<(i32, String)> + 'a;

/// Reads the state through `git`.
pub fn git_state_with(git: &GitCall<'_>) -> GitState {
    let ok = |args: &[&str]| git(args).filter(|(code, _)| *code == 0).map(|(_, out)| out);
    GitState {
        head: ok(&["rev-parse", "HEAD"]).filter(|id| !id.is_empty()),
        dirty: ok(&["status", "--porcelain", "--untracked-files=all"]).map(|out| !out.is_empty()),
        origin_main: ok(&["rev-parse", "origin/main"]).filter(|id| !id.is_empty()),
        ancestor_of_origin_main: git(&["merge-base", "--is-ancestor", "HEAD", "origin/main"])
            .and_then(|(code, _)| match code {
                0 => Some(true),
                1 => Some(false),
                _ => None,
            }),
    }
}

/// The commit this binary was compiled at (`build.rs`), `None` when git could not say. A run whose HEAD differs is not
/// publishable: the binary is not the code of the commit it would be labelled with.
pub fn build_head() -> Option<String> {
    option_env!("GUST_BENCH_BUILD_HEAD")
        .filter(|head| !head.is_empty())
        .map(str::to_owned)
}

/// Whether tracked files had uncommitted changes when this binary was compiled (`build.rs`), `None` when git could not
/// say. A run whose build was dirty is not publishable: the binary is code that no commit holds.
pub fn build_dirty() -> Option<bool> {
    build_dirty_from(option_env!("GUST_BENCH_BUILD_DIRTY"))
}

/// The meaning of the `GUST_BENCH_BUILD_DIRTY` value: `1` dirty, `0` clean, anything else (empty, unset) unknown.
pub fn build_dirty_from(value: Option<&str>) -> Option<bool> {
    match value {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// The git state of the checkout this crate is built in.
pub fn git_state() -> GitState {
    git_state_with(&|args| {
        let output = Command::new("git")
            .args(["-c", "safe.directory=*"])
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()?;
        Some((
            output.status.code()?,
            String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        ))
    })
}

/// Whether the source moved between two reads of [`git_state`] (a run starts and ends with one each): a different
/// HEAD, or a tree that is not clean at the end. `None` when it did not.
pub fn source_change(start: &GitState, end: &GitState) -> Option<String> {
    if start.head != end.head {
        return Some("HEAD moved during the run".to_owned());
    }
    (end.dirty != Some(false)).then(|| "the tree was not clean at the end of the run".to_owned())
}

/// Seconds since the Unix epoch (0 when the clock is before it).
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `2026-11-15T08:05:00Z` for a Unix time (proleptic Gregorian, days-to-civil by Howard Hinnant's algorithm).
pub fn utc_text(unix_s: u64) -> String {
    let days = (unix_s / 86_400) as i64;
    let rest = unix_s % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

/// The cargo features this binary was built with, as the results file records them.
pub fn build_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "gpu") {
        features.push("gpu");
    }
    if cfg!(feature = "slang-runtime") {
        features.push("slang-runtime");
    }
    if cfg!(feature = "spirv-route") {
        features.push("spirv-route");
    }
    // The counting allocator is active in a `profile-counts` build, so a timing run from it would be tainted.
    if cfg!(feature = "profile-counts") {
        features.push("profile-counts");
    }
    if cfg!(feature = "wgpu-trace") {
        features.push("wgpu-trace");
    }
    // A third device on the adapter and one more backend in the rotation (m10.b4), so a run from it is not a publishable
    // table until that has been measured not to change the other backends.
    if cfg!(feature = "rust-gpu") {
        features.push("rust-gpu");
    }
    // A fourth device on the adapter and two more backends in the rotation (m10.b2), for the same reason.
    if cfg!(feature = "cubecl") {
        features.push("cubecl");
    }
    features
}

/// The parts of the machine and build record that do not need a GPU.
fn machine_common(rayon_threads: usize) -> serde_json::Map<String, Value> {
    let value = json!({
        "os": os_name(),
        "cpu": cpu_name(),
        "logical_cores": std::thread::available_parallelism().map_or(0, |n| n.get()),
        "rayon_threads_n": rayon_threads,
        "power_plan": power_plan(),
        "versions": {
            "rustc": capture("rustc", &["--version"]),
            "wgpu": if cfg!(feature = "gpu") { json!("30.0.2") } else { Value::Null },
            "rayon": "1.12.0",
            "gust_commit": commit(),
            "gust_bench": env!("CARGO_PKG_VERSION"),
        },
        "build": {
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "gust_profile": gust::profile::NAME,
            "gust_dual": cfg!(gust_dual),
            "debug_assertions": cfg!(debug_assertions),
            "features": build_features(),
        },
    });
    #[cfg_attr(not(any(feature = "rust-gpu", feature = "cubecl")), allow(unused_mut))]
    let mut map = match value {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    // The comparator's identity (m10.b4): the rust-gpu version, the toolchain and target that built the module, and the
    // sha256 of the module this binary embeds.
    #[cfg(feature = "rust-gpu")]
    if let Some(Value::Object(versions)) = map.get_mut("versions") {
        versions.insert("rust_gpu".to_owned(), crate::rustgpu_module::identity());
    }
    // The CubeCL comparator (m10.b2): its version, compiler and what each row is.
    #[cfg(feature = "cubecl")]
    if let Some(Value::Object(versions)) = map.get_mut("versions") {
        versions.insert("cubecl".to_owned(), crate::cubecl_backend::identity());
    }
    map
}

/// Everything about the machine and the build that a reader needs to reproduce the table.
#[cfg(feature = "gpu")]
pub fn machine(
    adapter: &wgpu::AdapterInfo,
    timestamps: bool,
    rayon_threads: usize,
    limits: &wgpu::Limits,
    hand_limits: &wgpu::Limits,
) -> Value {
    let mut map = machine_common(rayon_threads);
    map.insert(
        "adapter".to_owned(),
        json!({
            "name": adapter.name,
            "vendor_id": adapter.vendor,
            "device_id": adapter.device,
            "device_type": format!("{:?}", adapter.device_type),
            "backend": format!("{:?}", adapter.backend),
            "driver": adapter.driver,
            "driver_info": adapter.driver_info,
            "gpu_timestamps": timestamps,
        }),
    );
    map.insert(
        "nvidia_driver".to_owned(),
        json!(capture(
            "nvidia-smi",
            &["--query-gpu=driver_version", "--format=csv,noheader"]
        )),
    );
    map.insert(
        "wgpu_limits".to_owned(),
        json!({
            "max_compute_workgroups_per_dimension": limits.max_compute_workgroups_per_dimension,
            "max_storage_buffer_binding_size": limits.max_storage_buffer_binding_size,
            "max_compute_invocations_per_workgroup": limits.max_compute_invocations_per_workgroup,
        }),
    );
    // The hand-WGSL device is a second device on the same adapter; both are opened with the same limit policy, and
    // both grants are recorded so a reader can see they match.
    map.insert(
        "wgpu_limits_hand_wgsl".to_owned(),
        json!({
            "max_compute_workgroups_per_dimension": hand_limits.max_compute_workgroups_per_dimension,
            "max_storage_buffer_binding_size": hand_limits.max_storage_buffer_binding_size,
            "max_compute_invocations_per_workgroup": hand_limits.max_compute_invocations_per_workgroup,
        }),
    );
    Value::Object(map)
}

/// The machine record of a build without a GPU runtime (`GUST_PROFILE=cpu`): no adapter, no limits.
#[cfg(not(feature = "gpu"))]
pub fn machine_cpu(rayon_threads: usize) -> Value {
    let mut map = machine_common(rayon_threads);
    map.insert("adapter".to_owned(), Value::Null);
    Value::Object(map)
}

/// `nvidia-smi --query-gpu=utilization.gpu,clocks.sm,clocks.mem` plus a few more fields, as numbers.
pub fn gpu_snapshot() -> Value {
    let fields = "utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used";
    let query = format!("--query-gpu={fields}");
    let Some(text) = capture("nvidia-smi", &[&query, "--format=csv,noheader,nounits"]) else {
        return Value::Null;
    };
    let values = text
        .lines()
        .next()
        .unwrap_or("")
        .split(',')
        .map(|v| v.trim().parse::<f64>().ok())
        .collect::<Vec<_>>();
    let names = [
        "utilization_pct",
        "sm_mhz",
        "mem_mhz",
        "temp_c",
        "power_w",
        "mem_used_mib",
    ];
    let mut map = serde_json::Map::new();
    for (name, value) in names.iter().zip(values) {
        map.insert((*name).to_owned(), json!(value));
    }
    Value::Object(map)
}

/// The GPU clock state that decides whether a timing is a boost-clock timing: SM and memory clock, performance
/// state and the PCIe link generation (current, and what the GPU supports now) and width.
pub fn gpu_clocks() -> Value {
    let fields = "clocks.sm,clocks.mem,pstate,pcie.link.gen.gpucurrent,pcie.link.gen.max,pcie.link.width.current,clocks.max.sm,clocks.max.mem";
    let query = format!("--query-gpu={fields}");
    capture("nvidia-smi", &[&query, "--format=csv,noheader,nounits"])
        .map_or(Value::Null, |text| parse_clocks(&text))
}

fn parse_clocks(text: &str) -> Value {
    let fields = text
        .lines()
        .next()
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    let number = |index: usize| fields.get(index).and_then(|v| v.parse::<f64>().ok());
    json!({
        "sm_mhz": number(0), "mem_mhz": number(1),
        "pstate": fields.get(2).filter(|v| !v.is_empty()),
        "pcie_gen_now": number(3), "pcie_gen_max": number(4), "pcie_width": number(5),
        "max_sm_mhz": number(6), "max_mem_mhz": number(7),
    })
}

/// Counts of processes that compete for the shared machine, and the number of processes `nvidia-smi` lists as GPU
/// clients (on Windows that list includes the desktop, so it is a coarse signal).
pub fn load_snapshot() -> Value {
    const WATCHED: [&str; 7] = [
        "cargo.exe",
        "rustc.exe",
        "slangc.exe",
        "gust-slang-reflect.exe",
        "lms.exe",
        "ollama.exe",
        "vmmem",
    ];
    let mut counts = serde_json::Map::new();
    if cfg!(windows)
        && let Some(text) = capture("tasklist", &["/FO", "CSV", "/NH"])
    {
        for watched in WATCHED {
            let count = text
                .lines()
                .filter(|line| {
                    line.trim_start_matches('"')
                        .to_ascii_lowercase()
                        .starts_with(watched)
                })
                .count();
            counts.insert(watched.to_owned(), json!(count));
        }
    }
    let gpu_apps = capture(
        "nvidia-smi",
        &[
            "--query-compute-apps=process_name,used_memory",
            "--format=csv,noheader",
        ],
    )
    .map(|text| text.lines().filter(|line| !line.trim().is_empty()).count());
    json!({ "process_counts": counts, "gpu_compute_app_count": gpu_apps })
}

/// The provenance fields every results file carries at its top level (plan 2.5 item 8): the route, the build and the
/// adapter under test. The route is read from the build (`gust_wgpu::SPIRV_ROUTE`), never from a flag. `git` is the
/// state read at the start of the run.
#[cfg(feature = "gpu")]
pub fn provenance(adapter: &wgpu::AdapterInfo, git: &GitState) -> Value {
    let route = if gust_wgpu::SPIRV_ROUTE {
        "spirv"
    } else {
        "wgsl"
    };
    provenance_of(
        route,
        json!({
            "name": adapter.name, "backend": format!("{:?}", adapter.backend),
            "driver": adapter.driver, "driver_info": adapter.driver_info,
        }),
        git,
    )
}

/// The provenance of a build without a GPU runtime: route `cpu`, no adapter.
#[cfg(not(feature = "gpu"))]
pub fn provenance_cpu(git: &GitState) -> Value {
    provenance_of("cpu", Value::Null, git)
}

fn provenance_of(route: &str, adapter: Value, git: &GitState) -> Value {
    let release = !cfg!(debug_assertions);
    let mut cfgs = Vec::new();
    if cfg!(gust_dual) {
        cfgs.push("gust_dual");
    }
    json!({
        "route": route,
        "build": {
            "release": release,
            "profile": if release { "release" } else { "debug" },
            "gust_profile": gust::profile::NAME,
            "features": build_features(),
            "cfgs": cfgs,
            "commit": git.short(),
            "tree_dirty": git.dirty,
        },
        "adapter": adapter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_dirty_flag_is_unknown_unless_the_build_said_one_way_or_the_other() {
        assert_eq!(build_dirty_from(Some("0")), Some(false));
        assert_eq!(build_dirty_from(Some("1")), Some(true));
        assert_eq!(build_dirty_from(Some("")), None);
        assert_eq!(build_dirty_from(Some("maybe")), None);
        assert_eq!(build_dirty_from(None), None);
    }

    /// What git answers, as the closure of `git_state_with` sees it: the arguments after `git -c safe.directory=*`.
    fn git_says(
        head: Option<(i32, &str)>,
        status: Option<(i32, &str)>,
        origin: Option<(i32, &str)>,
        ancestor: Option<i32>,
    ) -> GitState {
        git_state_with(&|args| match args {
            ["rev-parse", "HEAD"] => head.map(|(c, o)| (c, o.to_owned())),
            ["status", "--porcelain", "--untracked-files=all"] => {
                status.map(|(c, o)| (c, o.to_owned()))
            }
            ["rev-parse", "origin/main"] => origin.map(|(c, o)| (c, o.to_owned())),
            ["merge-base", "--is-ancestor", "HEAD", "origin/main"] => {
                ancestor.map(|c| (c, String::new()))
            }
            other => panic!("unexpected git call {other:?}"),
        })
    }

    #[test]
    fn the_git_state_reads_head_the_tree_and_the_ancestry_and_never_guesses() {
        let clean = git_says(
            Some((0, "0123456789abcdef0123")),
            Some((0, "")),
            Some((0, "fedcba9876543210fedc")),
            Some(0),
        );
        assert_eq!(clean.head.as_deref(), Some("0123456789abcdef0123"));
        assert_eq!(clean.short(), "0123456789ab");
        assert_eq!(clean.dirty, Some(false));
        assert_eq!(clean.origin_main.as_deref(), Some("fedcba9876543210fedc"));
        assert_eq!(clean.ancestor_of_origin_main, Some(true));
        // Anything in `git status --porcelain` is dirty; exit 1 of merge-base is "not an ancestor"; exit 128 (no
        // origin/main) and a git that cannot run are unknown, never yes.
        let dirty = git_says(
            Some((0, "abc")),
            Some((0, "?? untracked.txt")),
            Some((128, "")),
            Some(1),
        );
        assert_eq!(dirty.dirty, Some(true));
        assert_eq!(dirty.origin_main, None);
        assert_eq!(dirty.ancestor_of_origin_main, Some(false));
        let unknown = git_says(Some((128, "")), Some((128, "")), None, Some(128));
        assert_eq!(
            unknown,
            GitState {
                head: None,
                dirty: None,
                origin_main: None,
                ancestor_of_origin_main: None
            }
        );
        assert_eq!(unknown.short(), "unknown");
        let missing = git_says(None, None, None, None);
        assert_eq!(missing.ancestor_of_origin_main, None);
    }

    #[test]
    fn a_source_change_is_a_moved_head_or_a_tree_that_is_not_clean_at_the_end() {
        let at = |head: &str, dirty| GitState {
            head: Some(head.to_owned()),
            dirty,
            ..GitState::default()
        };
        assert_eq!(
            source_change(&at("a", Some(false)), &at("a", Some(false))),
            None
        );
        assert!(
            source_change(&at("a", Some(false)), &at("b", Some(false)))
                .unwrap()
                .contains("HEAD moved")
        );
        assert!(
            source_change(&at("a", Some(false)), &at("a", Some(true)))
                .unwrap()
                .contains("not clean")
        );
        assert!(source_change(&at("a", Some(false)), &at("a", None)).is_some());
    }

    #[test]
    fn utc_text_formats_known_instants() {
        assert_eq!(utc_text(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_text(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_text(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(utc_text(1_794_729_900), "2026-11-15T08:05:00Z");
        assert_eq!(utc_text(4_102_444_799), "2099-12-31T23:59:59Z");
    }

    #[test]
    fn clocks_parse_into_numbers_and_a_pstate() {
        let value = parse_clocks("2655, 14001, P0, 4, 5, 16, 3090, 14001\n");
        assert_eq!(value["sm_mhz"], 2655.0);
        assert_eq!(value["pstate"], "P0");
        assert_eq!(value["pcie_gen_now"], 4.0);
        assert_eq!(value["pcie_width"], 16.0);
        assert_eq!(value["max_mem_mhz"], 14001.0);
        let bad = parse_clocks("[N/A], x, P8");
        assert!(bad["sm_mhz"].is_null() && bad["pstate"].is_string());
        assert!(parse_clocks("")["sm_mhz"].is_null());
    }

    #[test]
    fn provenance_pins_route_build_and_adapter() {
        let value = provenance_of(
            "wgsl",
            json!({ "name": "Test GPU", "backend": "Vulkan", "driver": "555.55", "driver_info": "driver 1" }),
            &GitState {
                head: Some("0123456789abcdef".to_owned()),
                dirty: Some(true),
                origin_main: Some("fedcba".to_owned()),
                ancestor_of_origin_main: Some(false),
            },
        );
        assert_eq!(value["build"]["commit"], "0123456789ab");
        assert_eq!(value["build"]["tree_dirty"], true);
        assert_eq!(value["route"], "wgsl");
        assert_eq!(value["build"]["release"], !cfg!(debug_assertions));
        assert_eq!(
            value["build"]["profile"],
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
        assert_eq!(value["build"]["features"], json!(build_features()));
        assert_eq!(value["build"]["gust_profile"], gust::profile::NAME);
        assert_eq!(value["adapter"]["name"], "Test GPU");
        assert_eq!(value["adapter"]["backend"], "Vulkan");
        assert_eq!(value["adapter"]["driver"], "555.55");
        let mut keys = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, ["adapter", "build", "route"]);
    }

    /// The features list names what the binary was built with: a `cpu`-profile build (no `gpu`) says so, and a run of
    /// one build never reads as the other.
    #[test]
    fn the_features_list_follows_the_build() {
        let features = build_features();
        assert_eq!(features.contains(&"gpu"), cfg!(feature = "gpu"));
        assert_eq!(
            features.contains(&"slang-runtime"),
            cfg!(feature = "slang-runtime")
        );
        assert_eq!(
            features.contains(&"spirv-route"),
            cfg!(feature = "spirv-route")
        );
        assert_eq!(
            features.contains(&"profile-counts"),
            cfg!(feature = "profile-counts")
        );
        assert_eq!(
            features.contains(&"wgpu-trace"),
            cfg!(feature = "wgpu-trace")
        );
    }
}

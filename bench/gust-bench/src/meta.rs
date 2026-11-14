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

/// Whether the working tree has uncommitted or untracked changes (`None` when git cannot say). A commit named
/// without this would read as clean.
fn tree_dirty() -> Option<bool> {
    let output = Command::new("git")
        .args(["-c", "safe.directory=*", "status", "--porcelain"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| !output.stdout.iter().all(u8::is_ascii_whitespace))
}

/// Everything about the machine and the build that a reader needs to reproduce the table.
pub fn machine(
    adapter: &wgpu::AdapterInfo,
    timestamps: bool,
    rayon_threads: usize,
    limits: &wgpu::Limits,
) -> Value {
    json!({
        "adapter": {
            "name": adapter.name,
            "vendor_id": adapter.vendor,
            "device_id": adapter.device,
            "device_type": format!("{:?}", adapter.device_type),
            "backend": format!("{:?}", adapter.backend),
            "driver": adapter.driver,
            "driver_info": adapter.driver_info,
            "gpu_timestamps": timestamps,
        },
        "os": os_name(),
        "cpu": cpu_name(),
        "logical_cores": std::thread::available_parallelism().map_or(0, |n| n.get()),
        "rayon_threads_n": rayon_threads,
        "power_plan": power_plan(),
        "nvidia_driver": capture("nvidia-smi", &["--query-gpu=driver_version", "--format=csv,noheader"]),
        "versions": {
            "rustc": capture("rustc", &["--version"]),
            "wgpu": "30.0.2",
            "rayon": "1.12.0",
            "gust_commit": commit(),
            "gust_bench": env!("CARGO_PKG_VERSION"),
        },
        "build": {
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "gust_dual": cfg!(gust_dual),
            "debug_assertions": cfg!(debug_assertions),
        },
        "wgpu_limits": {
            "max_compute_workgroups_per_dimension": limits.max_compute_workgroups_per_dimension,
            "max_storage_buffer_binding_size": limits.max_storage_buffer_binding_size,
            "max_compute_invocations_per_workgroup": limits.max_compute_invocations_per_workgroup,
        },
    })
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

/// Why no file is publishable yet. The judgement (plan 2.4) is b1b's; the per-run file, the aggregate and the sampler
/// sidecar all say `publishable: false` with exactly this text.
pub const PUBLISHABLE: bool = false;

/// The reason that goes with [`PUBLISHABLE`].
pub const PUBLISHABLE_REASON: &str =
    "no judgement is implemented yet: the sampler records the quiet window, thresholds come later";

/// The provenance fields every results file carries at its top level (plan 2.5 item 8): the route, the build and the
/// adapter under test. The route is read from the build (`gust_wgpu::SPIRV_ROUTE`), never from a flag.
pub fn provenance(adapter: &wgpu::AdapterInfo) -> Value {
    let route = if gust_wgpu::SPIRV_ROUTE {
        "spirv"
    } else {
        "wgsl"
    };
    provenance_of(
        route,
        &adapter.name,
        &format!("{:?}", adapter.backend),
        &adapter.driver,
        &adapter.driver_info,
        (&commit(), tree_dirty()),
    )
}

fn provenance_of(
    route: &str,
    name: &str,
    backend: &str,
    driver: &str,
    driver_info: &str,
    (commit, dirty): (&str, Option<bool>),
) -> Value {
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
            "features": ["slang-runtime"],
            "cfgs": cfgs,
            "commit": commit,
            "tree_dirty": dirty,
        },
        "adapter": { "name": name, "backend": backend, "driver": driver, "driver_info": driver_info },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape of the three provenance fields is pinned: a reader (and the renderer of b1b) relies on exactly
    /// these keys in every results file.
    /// Every output (per-run file, aggregate, sidecar) takes both fields from these constants, so pinning them pins the
    /// three outputs; the aggregate and sidecar tests check their own copies.
    #[test]
    fn no_file_is_publishable_and_the_reason_is_not_empty() {
        const { assert!(!PUBLISHABLE) };
        assert!(PUBLISHABLE_REASON.len() > 20);
    }

    #[test]
    fn provenance_pins_route_build_and_adapter() {
        let value = provenance_of(
            "wgsl",
            "Test GPU",
            "Vulkan",
            "555.55",
            "driver 1",
            ("0123456789ab", Some(true)),
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
        assert_eq!(value["build"]["features"], json!(["slang-runtime"]));
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
}

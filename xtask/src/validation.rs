use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
};

use crate::{process::CheckRecord, time::now_utc, verify::VerifyMode};

pub(crate) struct ArtifactRecord {
    pub(crate) file: String,
    pub(crate) sha256: String,
}

/// The source state a validation run checked: the commit `HEAD` pointed to and whether the
/// working tree differed from it. Each is `None` ("null" in the record) when git cannot say,
/// for instance without git, outside a repository, or on a repository git refuses to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GitStamp {
    pub(crate) commit: Option<String>,
    /// True when `git status --porcelain --untracked-files=all` lists anything: modified,
    /// staged or untracked (not ignored) files, whatever `status.showUntrackedFiles` says. A run on a dirty tree is evidence about that tree, not the commit.
    pub(crate) dirty: Option<bool>,
}

impl GitStamp {
    /// The stamp of a run without git information.
    #[cfg(test)]
    pub(crate) const UNKNOWN: Self = Self {
        commit: None,
        dirty: None,
    };

    /// The stamp of the repository at `root`, taken now.
    pub(crate) fn read(root: &Path) -> Self {
        Self::read_with(root, "git")
    }

    fn read_with(root: &Path, git: &str) -> Self {
        let run = |args: &[&str]| -> Option<String> {
            let output = Command::new(git)
                .args(args)
                .current_dir(root)
                .stdin(Stdio::null())
                .stderr(Stdio::null())
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
        };
        let commit = run(&["rev-parse", "HEAD"])
            .map(|text| text.trim().to_owned())
            .filter(|text| {
                matches!(text.len(), 40 | 64) && text.chars().all(|c| c.is_ascii_hexdigit())
            });
        // `--untracked-files=all` overrides a `status.showUntrackedFiles=no` setting, which
        // would otherwise report a tree with new files as clean.
        let dirty = run(&["status", "--porcelain", "--untracked-files=all"])
            .map(|text| !text.trim().is_empty());
        Self { commit, dirty }
    }

    fn commit_json(&self) -> String {
        self.commit
            .as_deref()
            .map_or("null".to_owned(), json_string)
    }

    fn dirty_json(&self) -> &'static str {
        match self.dirty {
            Some(true) => "true",
            Some(false) => "false",
            None => "null",
        }
    }
}

/// What a validation run was: the mode, when it started, and the source state it checked.
pub(crate) struct RunInfo<'a> {
    pub(crate) mode: VerifyMode,
    pub(crate) started: &'a str,
    pub(crate) git: &'a GitStamp,
    /// The adapter the run's GPU steps used; `None` ("null" in the record) when the run failed before the adapter
    /// step could say.
    pub(crate) adapter: Option<&'a AdapterRecord>,
}

/// The Vulkan adapter `HeadlessDevice` selected, as `crates/gust-wgpu/examples/adapter_info.rs` printed it. Any
/// Vulkan adapter satisfies the GPU tiers, a software one (lavapipe) included, so the record says which ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdapterRecord {
    pub(crate) name: String,
    /// Decimal digits (a `u32` in wgpu).
    pub(crate) vendor: String,
    /// Decimal digits (a `u32` in wgpu).
    pub(crate) device: String,
    /// The `Debug` name of wgpu's `DeviceType`, for instance `DiscreteGpu` or `Cpu`.
    pub(crate) device_type: String,
    pub(crate) backend: String,
    pub(crate) driver: String,
    pub(crate) driver_info: String,
}

impl AdapterRecord {
    /// Reads the standard output of the `adapter_info` example: exactly one JSON object line. A missing or blank
    /// name, device type or backend, a non-numeric vendor or device id, or any other shape is an error, so an empty
    /// run can never be recorded as an adapter.
    pub(crate) fn parse(stdout: &str) -> Result<Self, String> {
        let mut lines = stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty());
        let (Some(line), None) = (lines.next(), lines.next()) else {
            return Err(
                "adapter_info must print exactly one JSON line on stdout (no adapter?)".to_owned(),
            );
        };
        let value =
            crate::json::parse(line).map_err(|error| format!("adapter_info output: {error}"))?;
        let text = |key: &str| -> Result<String, String> {
            value
                .get(key)
                .and_then(crate::json::Json::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("adapter_info output has no string `{key}`"))
        };
        let named = |key: &str| -> Result<String, String> {
            let field = text(key)?;
            if field.trim().is_empty() {
                Err(format!("adapter_info output has an empty `{key}`"))
            } else {
                Ok(field)
            }
        };
        let id = |key: &str| -> Result<String, String> {
            match value.get(key) {
                Some(crate::json::Json::Number(digits))
                    if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) =>
                {
                    Ok(digits.clone())
                }
                _ => Err(format!("adapter_info output has no numeric `{key}`")),
            }
        };
        Ok(Self {
            name: named("name")?,
            vendor: id("vendor")?,
            device: id("device")?,
            device_type: named("device_type")?,
            backend: named("backend")?,
            driver: text("driver")?,
            driver_info: text("driver_info")?,
        })
    }

    /// The `"adapter"` member of the record (or `null`), ending in a comma and a newline.
    fn json(adapter: Option<&Self>, redacted: &dyn Fn(&str) -> String) -> String {
        let Some(adapter) = adapter else {
            return "  \"adapter\": null,\n".to_owned();
        };
        let text = |value: &str| json_string(&redacted(value));
        format!(
            "  \"adapter\": {{\n    \"name\": {},\n    \"vendor\": {},\n    \"device\": {},\n    \"device_type\": {},\n    \"backend\": {},\n    \"driver\": {},\n    \"driver_info\": {}\n  }},\n",
            text(&adapter.name),
            adapter.vendor,
            adapter.device,
            text(&adapter.device_type),
            text(&adapter.backend),
            text(&adapter.driver),
            text(&adapter.driver_info),
        )
    }
}

pub(crate) fn write_validation(
    root: &Path,
    run: &RunInfo,
    failure: Option<&String>,
    checks: &[CheckRecord],
    artifacts: &[ArtifactRecord],
) -> Result<(), String> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    write_validation_with_home(root, &home, run, failure, checks, artifacts)
}

/// `write_validation` with the home directory passed in, so tests need not touch the
/// process environment.
fn write_validation_with_home(
    root: &Path,
    home: &str,
    run: &RunInfo,
    failure: Option<&String>,
    checks: &[CheckRecord],
    artifacts: &[ArtifactRecord],
) -> Result<(), String> {
    let path = root.join(".ai/VALIDATION.json");
    let prefixes = redaction_prefixes(&root.display().to_string(), home);
    let redacted = |text: &str| redact(text, &prefixes);
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"schema_version\": 1,\n");
    json.push_str(&format!(
        "  \"started_utc\": {},\n",
        json_string(run.started)
    ));
    json.push_str(&format!(
        "  \"finished_utc\": {},\n",
        json_string(&now_utc())
    ));
    json.push_str(&format!(
        "  \"mode\": {},\n",
        json_string(run.mode.as_str())
    ));
    json.push_str(&format!(
        "  \"git_commit\": {},\n  \"git_dirty\": {},\n",
        run.git.commit_json(),
        run.git.dirty_json()
    ));
    json.push_str(&AdapterRecord::json(run.adapter, &redacted));
    json.push_str(&format!("  \"passed\": {},\n", failure.is_none()));
    match failure {
        Some(error) => json.push_str(&format!(
            "  \"failure\": {},\n",
            json_string(&redacted(error))
        )),
        None => json.push_str("  \"failure\": null,\n"),
    }
    json.push_str("  \"scope\": \"Debug native Vulkan wgpu execution of Slang-generated WGSL; SPIR-V export validation is separate.\",\n");
    json.push_str("  \"untested\": [\"browser WebGPU\", \"Metal\", \"DXIL\", \"other GPU vendors\", \"declared minimum Rust version\"],\n");
    json.push_str("  \"checks\": [\n");
    for (index, check) in checks.iter().enumerate() {
        if index != 0 {
            json.push_str(",\n");
        }
        json.push_str("    {\n");
        json.push_str(&format!(
            "      \"program\": {},\n",
            json_string(&redacted(&check.program))
        ));
        json.push_str("      \"arguments\": [");
        for (arg_index, arg) in check.arguments.iter().enumerate() {
            if arg_index != 0 {
                json.push_str(", ");
            }
            json.push_str(&json_string(&redacted(arg)));
        }
        json.push_str("],\n");
        match check.exit_code {
            Some(code) => json.push_str(&format!("      \"exit_code\": {code},\n")),
            None => json.push_str("      \"exit_code\": null,\n"),
        }
        json.push_str(&format!(
            "      \"stdout\": {},\n",
            json_string(&redacted(&check.stdout))
        ));
        json.push_str(&format!(
            "      \"stderr\": {}\n",
            json_string(&redacted(&check.stderr))
        ));
        json.push_str("    }");
    }
    json.push_str("\n  ],\n");
    json.push_str("  \"spirv_artifacts\": [\n");
    for (index, artifact) in artifacts.iter().enumerate() {
        if index != 0 {
            json.push_str(",\n");
        }
        json.push_str("    {\n");
        json.push_str(&format!(
            "      \"file\": {},\n",
            json_string(&redacted(&artifact.file))
        ));
        json.push_str(&format!(
            "      \"sha256\": {},\n",
            json_string(&artifact.sha256)
        ));
        json.push_str("      \"validation_target\": \"vulkan1.2\",\n");
        json.push_str("      \"status\": \"passed\"\n");
        json.push_str("    }");
    }
    json.push_str("\n  ]\n}\n");
    fs::write(path, json).map_err(|error| error.to_string())
}

/// Machine-specific path prefixes and their placeholders, longest first. Command output
/// spells the workspace root with backslashes, forward slashes, the `\\?\` and `\\?\UNC\`
/// verbatim prefixes, and doubled backslashes, so each spelling gets its own entry. Trailing
/// separators are dropped so a home spelled with one redacts like a home spelled without.
fn redaction_prefixes(root: &str, home: &str) -> Vec<(String, &'static str)> {
    let mut prefixes = Vec::new();
    for (path, placeholder) in [(root, "<workspace>"), (home, "~")] {
        // `\\?\UNC\server\share` is `\\server\share`; any other verbatim path drops `\\?\`.
        let plain = match path.strip_prefix(r"\\?\UNC\") {
            Some(unc) => format!(r"\\{unc}"),
            None => path.strip_prefix(r"\\?\").unwrap_or(path).to_owned(),
        };
        let backslash = plain.replace('/', r"\");
        let backslash = backslash.trim_end_matches('\\');
        if backslash.is_empty() {
            continue;
        }
        let verbatim = match backslash.strip_prefix(r"\\") {
            Some(unc) => format!(r"\\?\UNC\{unc}"),
            None => format!(r"\\?\{backslash}"),
        };
        for spelling in [
            verbatim,
            backslash.replace('\\', r"\\"),
            backslash.to_owned(),
            backslash.replace('\\', "/"),
        ] {
            prefixes.push((spelling, placeholder));
        }
    }
    prefixes.sort_by_key(|(spelling, _)| std::cmp::Reverse(spelling.len()));
    prefixes.dedup();
    prefixes
}

/// Replaces every prefix spelling with its placeholder. Windows paths ignore case, and the
/// workspace root comes from the current directory while examples print the on-disk
/// spelling, so the match ignores case there.
fn redact(text: &str, prefixes: &[(String, &str)]) -> String {
    redact_with(text, prefixes, cfg!(windows))
}

fn redact_with(text: &str, prefixes: &[(String, &str)], ignore_case: bool) -> String {
    let mut out = text.to_owned();
    for (spelling, placeholder) in prefixes {
        out = replace_spelling(&out, spelling, placeholder, ignore_case);
    }
    out
}

fn replace_spelling(text: &str, spelling: &str, placeholder: &str, ignore_case: bool) -> String {
    if spelling.is_empty() {
        return text.to_owned();
    }
    if !ignore_case {
        return text.replace(spelling, placeholder);
    }
    // ASCII lowercasing keeps every byte offset, so matches found in the lowercased copy
    // splice into the original text.
    let haystack = text.to_ascii_lowercase();
    let needle = spelling.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut from = 0;
    while let Some(found) = haystack[from..].find(&needle) {
        let start = from + found;
        out.push_str(&text[from..start]);
        out.push_str(placeholder);
        from = start + needle.len();
    }
    out.push_str(&text[from..]);
    out
}

#[cfg(test)]
#[path = "tests/validation_redaction.rs"]
mod validation_redaction_tests;

#[cfg(test)]
#[path = "tests/validation_write.rs"]
mod validation_write_tests;

pub(crate) fn compact_validation_summary(text: &str) -> String {
    let field = |name: &str| -> Option<String> {
        let pattern = format!("\"{name}\":");
        let line = text
            .lines()
            .find(|line| line.trim_start().starts_with(&pattern))?;
        Some(
            line.split_once(':')?
                .1
                .trim()
                .trim_end_matches(',')
                .trim_matches('"')
                .to_owned(),
        )
    };
    let finished = field("finished_utc").unwrap_or_else(|| "unknown".to_owned());
    let mode = field("mode").unwrap_or_else(|| "unknown".to_owned());
    let passed = field("passed").unwrap_or_else(|| "unknown".to_owned());
    let commit = field("git_commit").unwrap_or_else(|| "unknown".to_owned());
    let dirty = field("git_dirty").unwrap_or_else(|| "unknown".to_owned());
    // Only a software adapter gets a tier word, appended last: a hardware record, a record without an adapter and
    // a record from before the adapter member keep the line they always had.
    let tier = if field("device_type").as_deref() == Some(SOFTWARE_DEVICE_TYPE) {
        " tier=Gpu (software)"
    } else {
        ""
    };
    format!("finished={finished} mode={mode} passed={passed} commit={commit} dirty={dirty}{tier}")
}

/// The `device_type` of the adapter record for a software Vulkan driver such as lavapipe: the `Debug` name of
/// wgpu's `DeviceType::Cpu`. Evidence from such an adapter is `Gpu (software)`, never hardware evidence (D52).
const SOFTWARE_DEVICE_TYPE: &str = "Cpu";

pub(crate) fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = sha256(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut data = input.to_vec();
    let bit_len = (data.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = H0;
    let (chunks, remainder) = data.as_chunks::<64>();
    debug_assert!(remainder.is_empty());
    for chunk in chunks {
        let mut w = [0u32; 64];
        let (words, remainder) = chunk.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        for (i, word) in words.iter().take(16).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

//! `gust-artifacts/manifest.json`, schema 1 (D45).
//!
//! The manifest is the tool-owned index of a kernel crate's committed artifacts. It is required once a crate uses
//! artifacts: `cargo gust init` creates an empty one, the `#[gpu]` macro includes it so that cargo re-expands the
//! crate when it changes, and `cargo gust build` rewrites it last. It holds no timestamp and no path, only file names,
//! so the same sources and the same Slang give the same bytes (S3).
//!
//! ```json
//! {
//!   "kernels": [
//!     {
//!       "clean_sha256": "<SHA-256 of the clean Slang text, 64 hex digits>",
//!       "entry_point": "gpu_vector_add_add",
//!       "export_unchecked": ["<parameter>"],
//!       "formats": [
//!         {
//!           "file": "<key>.wgsl",
//!           "format": "wgsl",
//!           "reflection_sha256": "<SHA-256 of <key>.wgsl.reflection.json>",
//!           "sha256": "<SHA-256 of <key>.wgsl>"
//!         },
//!         {
//!           "file": "<key>.spv",
//!           "format": "spirv",
//!           "reflection_sha256": "<SHA-256 of <key>.spv.reflection.json>",
//!           "sha256": "<SHA-256 of <key>.spv>",
//!           "validated": true
//!         }
//!       ],
//!       "key": "<FNV-1a 64 content key, 16 hex digits>",
//!       "module": "vector_add",
//!       "name": "add"
//!     }
//!   ],
//!   "schema": 1,
//!   "slang": "<the Slang build tag the helper reported, or null while there are no kernels>"
//! }
//! ```
//!
//! - Object keys are sorted, `kernels` are sorted by `key` and `formats` by format (`wgsl`, then `spirv`); the text
//!   uses two-space indentation, LF line ends and one final newline. `check` reports a manifest that is not exactly
//!   this text.
//! - `export_unchecked` is present only when it is not empty. It names the storage parameters whose element type the
//!   macro could not describe from the module's own syntax (`layout: null` in the kernel record: a `GpuPod` type from
//!   outside the module, an alias, a nested type from elsewhere). For them the stored reflection supplied the
//!   layout, so the descriptor-side half of the D17 cross-check was skipped at export; the other half (binding,
//!   access, and the layouts of every other parameter) was not. D17 still holds at run time, where
//!   `verify_artifact` checks the real `GpuPod::LAYOUT`. The names are in parameter order.
//! - `file` is always `<key>.wgsl` or `<key>.spv`; the reflection file beside it is `<file>.reflection.json`.
//! - `validated` is present for `spirv` only: `true` when `spirv-val --target-env vulkan1.2` accepted the module and
//!   `false` when the tool was not on `PATH` (never a silent pass). A failed validation writes nothing.
//! - `slang` is the build tag the native helper wrote into every reflection JSON of the set.
//! - The SHA-256 values protect the committed files from accidents the FNV-1a hashes inside the reflection JSON
//!   (D17 schema 1) are too weak to be trusted with; neither defends against an adversary (R4.4).

use std::collections::HashSet;

use gust::Format;

use crate::json::{self, Json};

/// The file name of the manifest inside `gust-artifacts/`.
pub(crate) const MANIFEST_FILE: &str = "manifest.json";

/// The only manifest schema this tool reads and writes.
pub(crate) const SCHEMA: u32 = 1;

/// The manifest of a kernel crate's `gust-artifacts/` directory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Manifest {
    /// The Slang build tag of the set; `None` while the manifest has no kernels.
    pub(crate) slang: Option<String>,
    pub(crate) kernels: Vec<KernelEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KernelEntry {
    pub(crate) key: String,
    pub(crate) module: String,
    pub(crate) name: String,
    pub(crate) entry_point: String,
    pub(crate) clean_sha256: String,
    /// Storage parameters whose layout came from the reflection (see the module docs), in parameter order.
    pub(crate) export_unchecked: Vec<String>,
    pub(crate) formats: Vec<FormatEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FormatEntry {
    pub(crate) format: Format,
    pub(crate) file: String,
    pub(crate) sha256: String,
    pub(crate) reflection_sha256: String,
    /// `Some` for SPIR-V only.
    pub(crate) validated: Option<bool>,
}

/// `wgsl` or `spirv`, as the manifest names a format.
pub(crate) fn format_name(format: Format) -> &'static str {
    match format {
        Format::Wgsl => "wgsl",
        Format::Spirv => "spirv",
    }
}

/// The position of a format in the manifest and in listings: WGSL first.
pub(crate) fn format_rank(format: Format) -> u8 {
    match format {
        Format::Wgsl => 0,
        Format::Spirv => 1,
    }
}

/// The artifact file name of a kernel in a format: `<key>.wgsl` or `<key>.spv`.
pub(crate) fn artifact_file(key: &str, format: Format) -> String {
    match format {
        Format::Wgsl => format!("{key}.wgsl"),
        Format::Spirv => format!("{key}.spv"),
    }
}

/// The helper's reflection JSON stored beside an artifact file.
pub(crate) fn reflection_file(artifact_file: &str) -> String {
    format!("{artifact_file}.reflection.json")
}

impl FormatEntry {
    pub(crate) fn reflection_file(&self) -> String {
        reflection_file(&self.file)
    }
}

impl KernelEntry {
    /// `module::name`.
    pub(crate) fn qualified_name(&self) -> String {
        format!("{}::{}", self.module, self.name)
    }
}

impl Manifest {
    /// The manifest `cargo gust init` writes: schema 1, no kernels.
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// The manifest as text. Kernels are sorted by key and formats by [`format_rank`], whatever order they were added
    /// in, so equal contents always give equal bytes.
    pub(crate) fn render(&self) -> String {
        let mut kernels: Vec<&KernelEntry> = self.kernels.iter().collect();
        kernels.sort_by(|a, b| a.key.cmp(&b.key));
        let kernels = kernels
            .into_iter()
            .map(|kernel| {
                let mut formats: Vec<&FormatEntry> = kernel.formats.iter().collect();
                formats.sort_by_key(|format| format_rank(format.format));
                let mut members = vec![
                    ("key", Json::text(&kernel.key)),
                    ("module", Json::text(&kernel.module)),
                    ("name", Json::text(&kernel.name)),
                    ("entry_point", Json::text(&kernel.entry_point)),
                    ("clean_sha256", Json::text(&kernel.clean_sha256)),
                    (
                        "formats",
                        Json::Array(formats.into_iter().map(format_json).collect()),
                    ),
                ];
                if !kernel.export_unchecked.is_empty() {
                    members.push((
                        "export_unchecked",
                        Json::Array(
                            kernel
                                .export_unchecked
                                .iter()
                                .map(|name| Json::text(name))
                                .collect(),
                        ),
                    ));
                }
                Json::object(members)
            })
            .collect();
        Json::object(vec![
            ("schema", Json::number(u64::from(SCHEMA))),
            (
                "slang",
                self.slang.as_deref().map_or(Json::Null, Json::text),
            ),
            ("kernels", Json::Array(kernels)),
        ])
        .to_pretty()
    }

    /// Read a manifest. Strict: an unknown key, a file name that is not the one derived from the key and format, a
    /// repeated key or a `validated` on the wrong format is an error. Whether the text is in canonical form is not
    /// checked here (`render` of the result compares with the text).
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let root = json::parse(text)?;
        only_keys(&root, &["schema", "slang", "kernels"], "manifest")?;
        if root.field("schema", "manifest")?.as_u32() != Some(SCHEMA) {
            return Err(format!(
                "manifest: unsupported schema (this tool reads schema {SCHEMA})"
            ));
        }
        let slang = {
            let value = root.field("slang", "manifest")?;
            if value.is_null() {
                None
            } else {
                Some(
                    value
                        .as_str()
                        .ok_or("manifest: `slang` must be a string or null")?
                        .to_owned(),
                )
            }
        };
        let mut seen = HashSet::new();
        let mut kernels = Vec::new();
        for (index, kernel) in root
            .field("kernels", "manifest")?
            .as_array()
            .ok_or("manifest: `kernels` must be an array")?
            .iter()
            .enumerate()
        {
            let entry = parse_kernel(kernel, &format!("kernels[{index}]"))?;
            if !seen.insert(entry.key.clone()) {
                return Err(format!("manifest: key `{}` appears twice", entry.key));
            }
            kernels.push(entry);
        }
        Ok(Self { slang, kernels })
    }

    /// The entry for `key`.
    pub(crate) fn entry(&self, key: &str) -> Option<&KernelEntry> {
        self.kernels.iter().find(|kernel| kernel.key == key)
    }
}

fn format_json(format: &FormatEntry) -> Json {
    let mut members = vec![
        ("format", Json::text(format_name(format.format))),
        ("file", Json::text(&format.file)),
        ("sha256", Json::text(&format.sha256)),
        ("reflection_sha256", Json::text(&format.reflection_sha256)),
    ];
    if let Some(validated) = format.validated {
        members.push(("validated", Json::Bool(validated)));
    }
    Json::object(members)
}

fn only_keys(value: &Json, allowed: &[&str], context: &str) -> Result<(), String> {
    for key in value.keys() {
        if !allowed.contains(&key) {
            return Err(format!("{context}: unknown key `{key}`"));
        }
    }
    Ok(())
}

fn text(value: &Json, key: &str, context: &str) -> Result<String, String> {
    value
        .field(key, context)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{context}: `{key}` must be a string"))
}

fn is_hex(text: &str, length: usize) -> bool {
    text.len() == length
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex_field(value: &Json, key: &str, length: usize, context: &str) -> Result<String, String> {
    let found = text(value, key, context)?;
    if is_hex(&found, length) {
        Ok(found)
    } else {
        Err(format!(
            "{context}: `{key}` must be {length} lower-case hex digits"
        ))
    }
}

fn parse_kernel(value: &Json, context: &str) -> Result<KernelEntry, String> {
    only_keys(
        value,
        &[
            "key",
            "module",
            "name",
            "entry_point",
            "clean_sha256",
            "export_unchecked",
            "formats",
        ],
        context,
    )?;
    let export_unchecked = match value.get("export_unchecked") {
        None => Vec::new(),
        Some(list) => {
            let names = list
                .as_array()
                .filter(|names| !names.is_empty())
                .ok_or_else(|| {
                    format!("{context}: `export_unchecked` must be a non-empty array (omit it when empty)")
                })?
                .iter()
                .map(|name| {
                    name.as_str().map(str::to_owned).ok_or_else(|| {
                        format!("{context}: `export_unchecked` must hold strings")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(index) = (1..names.len()).find(|&i| names[..i].contains(&names[i])) {
                return Err(format!(
                    "{context}: `export_unchecked` names `{}` twice",
                    names[index]
                ));
            }
            names
        }
    };
    let key = hex_field(value, "key", 16, context)?;
    let formats = value
        .field("formats", context)?
        .as_array()
        .ok_or_else(|| format!("{context}: `formats` must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, format)| parse_format(format, &key, &format!("{context}.formats[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    for (index, format) in formats.iter().enumerate() {
        if formats[..index]
            .iter()
            .any(|other| other.format == format.format)
        {
            return Err(format!(
                "{context}: format `{}` appears twice",
                format_name(format.format)
            ));
        }
    }
    Ok(KernelEntry {
        module: text(value, "module", context)?,
        name: text(value, "name", context)?,
        entry_point: text(value, "entry_point", context)?,
        clean_sha256: hex_field(value, "clean_sha256", 64, context)?,
        key,
        export_unchecked,
        formats,
    })
}

fn parse_format(value: &Json, key: &str, context: &str) -> Result<FormatEntry, String> {
    only_keys(
        value,
        &["format", "file", "sha256", "reflection_sha256", "validated"],
        context,
    )?;
    let format = match text(value, "format", context)?.as_str() {
        "wgsl" => Format::Wgsl,
        "spirv" => Format::Spirv,
        other => return Err(format!("{context}: unknown format `{other}`")),
    };
    let file = text(value, "file", context)?;
    if file != artifact_file(key, format) {
        return Err(format!(
            "{context}: `file` must be `{}`, found `{file}`",
            artifact_file(key, format)
        ));
    }
    let validated = match (format, value.get("validated")) {
        (Format::Spirv, Some(flag)) => Some(
            flag.as_bool()
                .ok_or_else(|| format!("{context}: `validated` must be a boolean"))?,
        ),
        (Format::Spirv, None) => return Err(format!("{context}: spirv needs `validated`")),
        (Format::Wgsl, None) => None,
        (Format::Wgsl, Some(_)) => {
            return Err(format!("{context}: only spirv entries have `validated`"));
        }
    };
    Ok(FormatEntry {
        format,
        file,
        sha256: hex_field(value, "sha256", 64, context)?,
        reflection_sha256: hex_field(value, "reflection_sha256", 64, context)?,
        validated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sample_kernel(key: &str, name: &str, with_spirv: bool) -> KernelEntry {
        let mut formats = vec![FormatEntry {
            format: Format::Wgsl,
            file: artifact_file(key, Format::Wgsl),
            sha256: "a".repeat(64),
            reflection_sha256: "b".repeat(64),
            validated: None,
        }];
        if with_spirv {
            formats.push(FormatEntry {
                format: Format::Spirv,
                file: artifact_file(key, Format::Spirv),
                sha256: "c".repeat(64),
                reflection_sha256: "d".repeat(64),
                validated: Some(true),
            });
        }
        KernelEntry {
            key: key.to_owned(),
            module: "demo".to_owned(),
            name: name.to_owned(),
            entry_point: format!("gpu_demo_{name}"),
            clean_sha256: "e".repeat(64),
            export_unchecked: Vec::new(),
            formats,
        }
    }

    #[test]
    fn the_empty_manifest_is_schema_one_with_no_kernels_and_no_slang() {
        assert_eq!(
            Manifest::empty().render(),
            "{\n  \"kernels\": [],\n  \"schema\": 1,\n  \"slang\": null\n}\n"
        );
    }

    #[test]
    fn kernels_and_formats_are_sorted_whatever_order_they_arrive_in() {
        let mut spirv_first = sample_kernel("00000000000000bb", "b", true);
        spirv_first.formats.reverse();
        let manifest = Manifest {
            slang: Some("2026.13.1-test".to_owned()),
            kernels: vec![
                sample_kernel("00000000000000ff", "z", false),
                spirv_first,
                sample_kernel("0000000000000011", "a", false),
            ],
        };
        let text = manifest.render();
        let keys: Vec<usize> = ["0000000000000011", "00000000000000bb", "00000000000000ff"]
            .iter()
            .map(|key| text.find(&format!("\"key\": \"{key}\"")).unwrap())
            .collect();
        assert!(keys[0] < keys[1] && keys[1] < keys[2], "{text}");
        assert!(
            text.find("\"format\": \"wgsl\"").unwrap()
                < text.find("\"format\": \"spirv\"").unwrap()
        );
        assert!(text.ends_with("}\n"));
        assert!(!text.contains('\r'));
        // The same contents added in the opposite order give the same bytes.
        let mut reversed = manifest.clone();
        reversed.kernels.reverse();
        assert_eq!(reversed.render(), text);
    }

    #[test]
    fn the_manifest_text_is_pinned_byte_for_byte() {
        let manifest = Manifest {
            slang: Some("2026.13.1-test".to_owned()),
            kernels: vec![sample_kernel("0123456789abcdef", "run", true)],
        };
        let expected = concat!(
            "{\n",
            "  \"kernels\": [\n",
            "    {\n",
            "      \"clean_sha256\": \"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\",\n",
            "      \"entry_point\": \"gpu_demo_run\",\n",
            "      \"formats\": [\n",
            "        {\n",
            "          \"file\": \"0123456789abcdef.wgsl\",\n",
            "          \"format\": \"wgsl\",\n",
            "          \"reflection_sha256\": \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",\n",
            "          \"sha256\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n",
            "        },\n",
            "        {\n",
            "          \"file\": \"0123456789abcdef.spv\",\n",
            "          \"format\": \"spirv\",\n",
            "          \"reflection_sha256\": \"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\",\n",
            "          \"sha256\": \"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\",\n",
            "          \"validated\": true\n",
            "        }\n",
            "      ],\n",
            "      \"key\": \"0123456789abcdef\",\n",
            "      \"module\": \"demo\",\n",
            "      \"name\": \"run\"\n",
            "    }\n",
            "  ],\n",
            "  \"schema\": 1,\n",
            "  \"slang\": \"2026.13.1-test\"\n",
            "}\n"
        );
        assert_eq!(manifest.render(), expected);
        assert_eq!(Manifest::parse(expected).unwrap(), manifest);
    }

    #[test]
    fn export_unchecked_is_written_only_when_not_empty_and_round_trips() {
        let mut kernel = sample_kernel("0123456789abcdef", "run", false);
        let plain = Manifest {
            slang: Some("s".to_owned()),
            kernels: vec![kernel.clone()],
        };
        assert!(!plain.render().contains("export_unchecked"));
        kernel.export_unchecked = vec!["pairs".to_owned(), "out".to_owned()];
        let manifest = Manifest {
            slang: Some("s".to_owned()),
            kernels: vec![kernel],
        };
        let text = manifest.render();
        // Parameter order is kept (not sorted), and the key sits between `entry_point` and `formats`.
        assert!(
            text.contains(
                "      \"entry_point\": \"gpu_demo_run\",\n      \"export_unchecked\": [\n        \"pairs\",\n        \"out\"\n      ],\n      \"formats\": ["
            ),
            "{text}"
        );
        assert_eq!(Manifest::parse(&text).unwrap(), manifest);
        for (what, bad) in [
            (
                "empty list",
                text.replace("\"pairs\",\n        \"out\"", ""),
            ),
            ("not strings", text.replace("\"pairs\"", "7")),
            ("a repeat", text.replace("\"out\"", "\"pairs\"")),
        ] {
            let error = Manifest::parse(&bad).unwrap_err();
            assert!(error.contains("export_unchecked"), "{what}: {error}");
        }
    }

    #[test]
    fn parse_is_strict() {
        let good = Manifest {
            slang: Some("s".to_owned()),
            kernels: vec![sample_kernel("0123456789abcdef", "run", true)],
        }
        .render();
        assert!(Manifest::parse(&good).is_ok());
        for (what, bad) in [
            ("schema 2", good.replace("\"schema\": 1", "\"schema\": 2")),
            (
                "unknown top-level key",
                good.replace("\"schema\"", "\"extra\": 0,\n  \"schema\""),
            ),
            (
                "a file that is not derived from the key",
                good.replace("0123456789abcdef.wgsl", "../escape.wgsl"),
            ),
            (
                "validated missing on spirv",
                good.replace(",\n          \"validated\": true", ""),
            ),
            (
                "validated on wgsl",
                good.replace(
                    "\"file\": \"0123456789abcdef.wgsl\"",
                    "\"validated\": true,\n          \"file\": \"0123456789abcdef.wgsl\"",
                ),
            ),
            ("short hash", good.replace(&"e".repeat(64), "abc")),
            (
                "upper-case hash",
                good.replace(&"e".repeat(64), &"E".repeat(64)),
            ),
            (
                "unknown format",
                good.replace("\"format\": \"wgsl\"", "\"format\": \"glsl\""),
            ),
        ] {
            assert!(Manifest::parse(&bad).is_err(), "{what}");
        }
        let twice = Manifest {
            slang: None,
            kernels: vec![
                sample_kernel("0123456789abcdef", "a", false),
                sample_kernel("0123456789abcdef", "b", false),
            ],
        }
        .render();
        assert!(
            Manifest::parse(&twice)
                .unwrap_err()
                .contains("appears twice")
        );
    }

    #[test]
    fn file_names_derive_from_the_key_and_the_format() {
        assert_eq!(
            artifact_file("0123456789abcdef", Format::Wgsl),
            "0123456789abcdef.wgsl"
        );
        assert_eq!(
            artifact_file("0123456789abcdef", Format::Spirv),
            "0123456789abcdef.spv"
        );
        assert_eq!(
            reflection_file("0123456789abcdef.spv"),
            "0123456789abcdef.spv.reflection.json"
        );
    }
}

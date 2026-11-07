//! `gust-artifacts/manifest.json`, schema 1, schema 2 once a kernel is not a `grid` kernel (D45, D89), or schema 3 once a
//! kernel has variants or needs a capability (M8, D139).
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
//! - `export_unchecked` is present only when it is not empty. It names the storage and uniform parameters whose
//!   element type the macro could not describe from the module's own syntax (`layout: null` in the kernel record: a
//!   `GpuPod` type from outside the module, an alias, a nested type from elsewhere). For them the stored reflection
//!   supplied the layout, so the descriptor-side half of the D17 cross-check was skipped at export; the other half
//!   (binding, access, and the layouts of every other parameter) was not. D17 still holds at run time, where
//!   `verify_artifact` checks the real `GpuPod::LAYOUT`. The names are in parameter order. A uniform or a vector
//!   element that the macro does describe is not listed: its layout is in the record and is checked against the
//!   reflection at export (m7.lv).
//! - `file` is always `<key>.wgsl` or `<key>.spv`; the reflection file beside it is `<file>.reflection.json`.
//! - `validated` is present for `spirv` only: `true` when `spirv-val --target-env vulkan1.2` accepted the module and
//!   `false` when the tool was not on `PATH` (never a silent pass). A failed validation writes nothing.
//! - `slang` is the build tag the native helper wrote into every reflection JSON of the set.
//! - `schema` is 1 unless a kernel's shape is not `grid` (mx.k, D89). Then it is 2, and every kernel carries `"shape":
//!   "grid" | "pixel"` (the last key of its object). A manifest of `grid` kernels only is
//!   schema 1 and has no `shape` key anywhere, byte for byte what it was before shapes existed; `cargo gust init`
//!   writes schema 1. Schema 1 reads as all-`grid`; a `shape` in a schema 1 manifest is an unknown key. A schema 2
//!   entry without `shape` reads as `grid`, but `render` writes it, so such a file is not canonical.
//! - Schema 3 (M8, D139) is schema 2's grammar plus `variants` on the entry of a kernel with a variant or a required
//!   capability: `"variants": [{"key": "<16 hex digits>", "requires": ["<capability>"], "variant": "<function>"}]`,
//!   sorted by `key`, one item for each body of the kernel and nothing else. Every body has its own entry in `kernels`
//!   (its key, files and hashes, and the kernel's `module` and `name`, with its own `entry_point`), because each is
//!   compiled and committed on its own; the entry that carries `variants` is the one of the kernel's own body, whose
//!   item has `"variant": ""` and the capabilities the own body needs (none, for a kernel with a portable body). The
//!   item of a variant has the variant function's name. `shape` is written on every entry when some kernel is not
//!   `grid`, and then only; schema 3 is written exactly when some entry has `variants`, so a crate without variants
//!   keeps the bytes it had. A reader of schemas 1 and 2 refuses schema 3, and `variants` in a manifest that says schema
//!   1 or 2 is an unknown key. `requires` names are the capabilities of `gust::capability::Capability`, in its order,
//!   none twice; one this tool does not know is refused. The reader holds the links to the same standard as the files:
//!   an item names an entry of the manifest with the kernel's `module` and `name`, the own body is listed once and is
//!   the entry that carries the list, a variant's name appears once, and every entry with the kernel's `module` and
//!   `name` is listed. Two entries with one `module` and `name` and no `variants` list among them are refused as well,
//!   at any schema.
//! - The SHA-256 values protect the committed files from accidents the FNV-1a hashes inside the reflection JSON
//!   (D17 schema 1) are too weak to be trusted with; neither defends against an adversary (R4.4).

use std::collections::HashSet;

use gust::{Format, Shape, capability::Capability};

use crate::{
    json::{self, Json},
    record::capabilities_named,
};

/// The file name of the manifest inside `gust-artifacts/`.
pub(crate) const MANIFEST_FILE: &str = "manifest.json";

/// The manifest schema of a crate whose kernels are all `grid`: what `cargo gust init` writes.
pub(crate) const SCHEMA: u32 = 1;

/// The manifest schema of a crate with a kernel whose shape is not `grid` (mx.k, D89): schema 1 plus `shape` per
/// kernel.
pub(crate) const SHAPE_SCHEMA: u32 = 2;

/// The manifest schema of a crate with a kernel that has variants or needs a capability (M8, D139): schema 2's grammar
/// plus `variants` on the entry of such a kernel.
pub(crate) const VARIANT_SCHEMA: u32 = 3;

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
    /// The kernel's shape (`grid` for every kernel of a schema 1 manifest).
    pub(crate) shape: Shape,
    pub(crate) clean_sha256: String,
    /// Storage parameters whose layout came from the reflection (see the module docs), in parameter order.
    pub(crate) export_unchecked: Vec<String>,
    pub(crate) formats: Vec<FormatEntry>,
    /// The bodies of the kernel, own body included, sorted by key; empty for every entry but that of the own body of a
    /// kernel with a variant or a required capability (schema 3).
    pub(crate) variants: Vec<VariantEntry>,
}

/// One body of a kernel in the `variants` list of the entry of its own body (schema 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariantEntry {
    /// The key of the body, which is an entry of the manifest too.
    pub(crate) key: String,
    /// The capabilities the body needs, in the order of `Capability::ALL`.
    pub(crate) requires: Vec<Capability>,
    /// The variant function's name; `""` for the kernel's own body.
    pub(crate) variant: String,
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
        other => unreachable!("a format cargo-gust does not know: {other:?}"),
    }
}

/// The position of a format in the manifest and in listings: WGSL first.
pub(crate) fn format_rank(format: Format) -> u8 {
    match format {
        Format::Wgsl => 0,
        Format::Spirv => 1,
        other => unreachable!("a format cargo-gust does not know: {other:?}"),
    }
}

/// The artifact file name of a kernel in a format: `<key>.wgsl` or `<key>.spv`.
pub(crate) fn artifact_file(key: &str, format: Format) -> String {
    match format {
        Format::Wgsl => format!("{key}.wgsl"),
        Format::Spirv => format!("{key}.spv"),
        other => unreachable!("a format cargo-gust does not know: {other:?}"),
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

    /// The schema [`Manifest::render`] writes: 3 when a kernel has `variants`, else 2 when a kernel's shape is not
    /// `grid`, else 1.
    pub(crate) fn schema(&self) -> u32 {
        if self
            .kernels
            .iter()
            .any(|kernel| !kernel.variants.is_empty())
        {
            VARIANT_SCHEMA
        } else if self.has_shaped_kernel() {
            SHAPE_SCHEMA
        } else {
            SCHEMA
        }
    }

    /// Whether a kernel's shape is not `grid`: then every entry carries its `shape`, whatever the schema.
    fn has_shaped_kernel(&self) -> bool {
        self.kernels
            .iter()
            .any(|kernel| kernel.shape != Shape::Grid)
    }

    /// The manifest as text. Kernels are sorted by key and formats by [`format_rank`], whatever order they were added
    /// in, so equal contents always give equal bytes.
    pub(crate) fn render(&self) -> String {
        let schema = self.schema();
        let with_shape = self.has_shaped_kernel();
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
                if with_shape {
                    members.push(("shape", Json::text(kernel.shape.name())));
                }
                if !kernel.variants.is_empty() {
                    members.push(("variants", variants_json(&kernel.variants)));
                }
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
            ("schema", Json::number(u64::from(schema))),
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
        let schema = match root.field("schema", "manifest")?.as_u32() {
            Some(schema @ (SCHEMA | SHAPE_SCHEMA | VARIANT_SCHEMA)) => schema,
            _ => {
                return Err(format!(
                    "manifest: unsupported schema (this tool reads schemas {SCHEMA}, {SHAPE_SCHEMA} and {VARIANT_SCHEMA})"
                ));
            }
        };
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
            let entry = parse_kernel(kernel, schema, &format!("kernels[{index}]"))?;
            if !seen.insert(entry.key.clone()) {
                return Err(format!("manifest: key `{}` appears twice", entry.key));
            }
            kernels.push(entry);
        }
        check_variant_links(&kernels)?;
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

fn parse_kernel(value: &Json, schema: u32, context: &str) -> Result<KernelEntry, String> {
    // `shape` is a schema 2 key and `variants` a schema 3 one: under an older number each is as unknown as any other, so
    // a reader of schema 1 and this one agree on what a schema 1 file is.
    let mut allowed = vec![
        "key",
        "module",
        "name",
        "entry_point",
        "clean_sha256",
        "export_unchecked",
        "formats",
    ];
    if schema >= SHAPE_SCHEMA {
        allowed.push("shape");
    }
    if schema >= VARIANT_SCHEMA {
        allowed.push("variants");
    }
    only_keys(value, &allowed, context)?;
    let shape = match value.get("shape") {
        None => Shape::Grid,
        Some(found) => {
            let name = found
                .as_str()
                .ok_or_else(|| format!("{context}: `shape` must be a string"))?;
            Shape::from_name(name).ok_or_else(|| {
                format!("{context}: unknown shape `{name}` (expected `grid` or `pixel`)")
            })?
        }
    };
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
    let variants = match value.get("variants") {
        None => Vec::new(),
        Some(list) => parse_variants(list, &format!("{context}.variants"))?,
    };
    Ok(KernelEntry {
        module: text(value, "module", context)?,
        name: text(value, "name", context)?,
        entry_point: text(value, "entry_point", context)?,
        shape,
        clean_sha256: hex_field(value, "clean_sha256", 64, context)?,
        key,
        export_unchecked,
        formats,
        variants,
    })
}

/// `variants` as JSON: one object per body, sorted by key.
fn variants_json(variants: &[VariantEntry]) -> Json {
    let mut sorted: Vec<&VariantEntry> = variants.iter().collect();
    sorted.sort_by(|a, b| a.key.cmp(&b.key));
    Json::Array(
        sorted
            .into_iter()
            .map(|body| {
                Json::object(vec![
                    ("key", Json::text(&body.key)),
                    (
                        "requires",
                        Json::Array(
                            body.requires
                                .iter()
                                .map(|capability| Json::text(capability.name()))
                                .collect(),
                        ),
                    ),
                    ("variant", Json::text(&body.variant)),
                ])
            })
            .collect(),
    )
}

/// The `variants` list of one entry: non-empty (an entry without variants omits the key), each item a key, a capability
/// set and a variant name, and no key twice. Whether the items fit the other entries is [`check_variant_links`].
fn parse_variants(value: &Json, context: &str) -> Result<Vec<VariantEntry>, String> {
    let items = value
        .as_array()
        .filter(|items| !items.is_empty())
        .ok_or_else(|| {
            format!("{context} must be a non-empty array (omit it when the kernel has no variants)")
        })?;
    let mut variants: Vec<VariantEntry> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let at = format!("{context}[{index}]");
        only_keys(item, &["key", "requires", "variant"], &at)?;
        let requires = item
            .field("requires", &at)?
            .as_array()
            .ok_or_else(|| format!("{at}: `requires` must be an array"))?;
        let body = VariantEntry {
            key: hex_field(item, "key", 16, &at)?,
            requires: capabilities_named(requires, &format!("{at}.requires"))?,
            variant: text(item, "variant", &at)?,
        };
        if variants.iter().any(|other| other.key == body.key) {
            return Err(format!("{at}: key `{}` appears twice", body.key));
        }
        variants.push(body);
    }
    Ok(variants)
}

/// The links of the `variants` lists to the entries (schema 3). A list belongs to the entry of the kernel's own body:
/// it holds one item for that entry's key with `variant` `""` (and for no other key), one for each other body of the
/// kernel, each a different function, and every item is an entry of the manifest with the kernel's `module` and `name`
/// that has no list of its own. Every entry with that `module` and `name` is in the list, and no two entries carry one
/// for the same kernel.
fn check_variant_links(kernels: &[KernelEntry]) -> Result<(), String> {
    for (index, owner) in kernels.iter().enumerate() {
        if owner.variants.is_empty() {
            continue;
        }
        let context = format!("kernels[{index}] (`{}`)", owner.qualified_name());
        let own: Vec<&VariantEntry> = owner
            .variants
            .iter()
            .filter(|body| body.variant.is_empty())
            .collect();
        match own.as_slice() {
            [only] if only.key == owner.key => {}
            [only] => {
                return Err(format!(
                    "{context}: `variants` lists the own body (`variant` \"\") under key {}, but this entry is key {}",
                    only.key, owner.key
                ));
            }
            [] => {
                return Err(format!(
                    "{context}: `variants` does not list the kernel's own body (key {}, `variant` \"\")",
                    owner.key
                ));
            }
            _ => {
                return Err(format!(
                    "{context}: `variants` lists the kernel's own body (`variant` \"\") more than once"
                ));
            }
        }
        for (position, body) in owner.variants.iter().enumerate() {
            if !body.variant.is_empty()
                && owner.variants[..position]
                    .iter()
                    .any(|other| other.variant == body.variant)
            {
                return Err(format!(
                    "{context}: `variants` names the variant `{}` twice",
                    body.variant
                ));
            }
            let at = format!("{context}.variants[{position}]");
            let Some(entry) = kernels.iter().find(|entry| entry.key == body.key) else {
                return Err(format!(
                    "{at}: key {} is not an entry of this manifest",
                    body.key
                ));
            };
            if entry.module != owner.module || entry.name != owner.name {
                return Err(format!(
                    "{at}: key {} is the entry of `{}`, not of `{}`",
                    body.key,
                    entry.qualified_name(),
                    owner.qualified_name()
                ));
            }
            if entry.key != owner.key && !entry.variants.is_empty() {
                return Err(format!(
                    "{at}: key {} has a `variants` list of its own; only the entry of the kernel's own body has one",
                    body.key
                ));
            }
        }
        for (other_index, other) in kernels.iter().enumerate() {
            if other.module != owner.module || other.name != owner.name {
                continue;
            }
            if other_index != index && !other.variants.is_empty() {
                return Err(format!(
                    "{context}: kernels[{other_index}] has the same module and name and a `variants` list as well"
                ));
            }
            if !owner.variants.iter().any(|body| body.key == other.key) {
                return Err(format!(
                    "{context}: the entry with key {} has the kernel's module and name but is not in `variants`",
                    other.key
                ));
            }
        }
    }
    // Two or more entries of one kernel and no list anywhere: the loop above never looked at them.
    for (index, entry) in kernels.iter().enumerate() {
        let mates = || {
            kernels
                .iter()
                .enumerate()
                .filter(|(_, mate)| mate.module == entry.module && mate.name == entry.name)
        };
        if mates().any(|(_, mate)| !mate.variants.is_empty()) {
            continue;
        }
        if let Some((first, _)) = mates().find(|(other_index, _)| *other_index != index) {
            return Err(format!(
                "kernels[{index}] (`{}`): kernels[{first}] has the same module and name, and no entry of the kernel has a `variants` list",
                entry.qualified_name()
            ));
        }
    }
    Ok(())
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
        (other, _) => return Err(format!("{context}: unsupported format {other:?}")),
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

    /// `Format` is `#[non_exhaustive]`, so a gust release that adds a format compiles here; this walks
    /// `Format::ALL` so that the `unreachable!` arms above fail a test, not a user's export.
    #[test]
    fn every_format_gust_has_is_named_ranked_and_filed() {
        let mut names = Vec::new();
        let mut ranks = Vec::new();
        let mut files = Vec::new();
        for &format in Format::ALL {
            names.push(format_name(format));
            ranks.push(format_rank(format));
            let file = artifact_file("0123456789abcdef", format);
            assert!(file.starts_with("0123456789abcdef."), "{file}");
            files.push(file);
        }
        names.sort_unstable();
        names.dedup();
        ranks.sort_unstable();
        ranks.dedup();
        files.sort_unstable();
        files.dedup();
        assert_eq!(
            names.len(),
            Format::ALL.len(),
            "format_name is not injective"
        );
        assert_eq!(
            ranks.len(),
            Format::ALL.len(),
            "format_rank is not injective"
        );
        assert_eq!(
            files.len(),
            Format::ALL.len(),
            "artifact_file is not injective"
        );
    }

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
            shape: Shape::Grid,
            clean_sha256: "e".repeat(64),
            export_unchecked: Vec::new(),
            formats,
            variants: Vec::new(),
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
            ("schema 4", good.replace("\"schema\": 1", "\"schema\": 4")),
            (
                "a shape under schema 1",
                good.replace(
                    "\"name\": \"run\"",
                    "\"name\": \"run\",\n      \"shape\": \"grid\"",
                ),
            ),
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

    /// A manifest with one pixel kernel among grid kernels: the only way to make a schema 2 manifest today, since the
    /// macro expands no pixel kernel yet.
    fn manifest_with_a_pixel_kernel() -> Manifest {
        let mut pixel = sample_kernel("0123456789abcdef", "run", false);
        pixel.shape = Shape::Pixel;
        Manifest {
            slang: Some("2026.13.1-test".to_owned()),
            kernels: vec![sample_kernel("00000000000000ff", "grid_one", false), pixel],
        }
    }

    #[test]
    fn a_grid_manifest_is_schema_1_with_no_shape_and_a_pixel_kernel_makes_it_schema_2() {
        // Every manifest written before shapes existed: schema 1, no `shape`, byte for byte (also pinned above).
        let grid = Manifest {
            slang: Some("2026.13.1-test".to_owned()),
            kernels: vec![
                sample_kernel("0123456789abcdef", "run", true),
                sample_kernel("00000000000000ff", "other", false),
            ],
        };
        assert_eq!(grid.schema(), SCHEMA);
        let grid_text = grid.render();
        assert!(grid_text.contains("\"schema\": 1") && !grid_text.contains("shape"));
        assert_eq!(Manifest::empty().schema(), SCHEMA);
        // One non-grid kernel moves the number, and then every kernel names its shape.
        let mixed = manifest_with_a_pixel_kernel();
        assert_eq!(mixed.schema(), SHAPE_SCHEMA);
        let text = mixed.render();
        assert!(text.contains("\"schema\": 2"), "{text}");
        assert_eq!(text.matches("\"shape\": \"pixel\"").count(), 1, "{text}");
        assert_eq!(text.matches("\"shape\": \"grid\"").count(), 1, "{text}");
        // `shape` is the last key of the entry, after `name`.
        assert!(
            text.contains("\"name\": \"run\",\n      \"shape\": \"pixel\"\n    }"),
            "{text}"
        );
        assert_eq!(Manifest::parse(&text).unwrap(), mixed);
        // Apart from the number and the key, the entries are the schema 1 entries.
        let without = text
            .replace("\"schema\": 2", "\"schema\": 1")
            .replace(",\n      \"shape\": \"grid\"", "")
            .replace(",\n      \"shape\": \"pixel\"", "");
        let mut as_grid = mixed.clone();
        as_grid.kernels[1].shape = Shape::Grid;
        assert_eq!(without, as_grid.render());
    }

    #[test]
    fn a_schema_1_manifest_reads_as_all_grid_and_a_schema_2_entry_without_a_shape_is_grid() {
        let good = Manifest {
            slang: Some("s".to_owned()),
            kernels: vec![sample_kernel("0123456789abcdef", "run", true)],
        }
        .render();
        let parsed = Manifest::parse(&good).unwrap();
        assert!(
            parsed
                .kernels
                .iter()
                .all(|kernel| kernel.shape == Shape::Grid)
        );
        // A schema 2 file whose entry has no `shape` reads as grid; it is not canonical, since `render` of a manifest
        // with no pixel kernel is schema 1 (`check` reports that as a hand edit).
        let lenient = good.replace("\"schema\": 1", "\"schema\": 2");
        let parsed = Manifest::parse(&lenient).unwrap();
        assert_eq!(parsed.kernels[0].shape, Shape::Grid);
        assert_ne!(parsed.render(), lenient);
    }

    #[test]
    fn a_bad_shape_in_a_schema_2_manifest_is_named() {
        let text = manifest_with_a_pixel_kernel().render();
        for (value, wanted) in [
            ("\"cpu\"", "unknown shape `cpu`"),
            ("\"Pixel\"", "unknown shape `Pixel`"),
            ("3", "`shape` must be a string"),
        ] {
            let bad = text.replace("\"shape\": \"pixel\"", &format!("\"shape\": {value}"));
            let error = Manifest::parse(&bad).unwrap_err();
            assert!(
                error.contains(wanted) && error.contains("kernels["),
                "{value}: {error}"
            );
        }
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

    // ---- schema 3: variants (M8, D139) ----

    const OWN_KEY: &str = "0123456789abcdef";
    const VARIANT_KEY: &str = "00000000000000aa";
    const OTHER_KEY: &str = "00000000000000ff";

    fn body(key: &str, requires: &[Capability], variant: &str) -> VariantEntry {
        VariantEntry {
            key: key.to_owned(),
            requires: requires.to_vec(),
            variant: variant.to_owned(),
        }
    }

    /// A kernel `demo::run` with a portable own body and a subgroups variant (two entries, the own one carrying the
    /// list), beside the unrelated kernel `demo::other`.
    fn variant_manifest() -> Manifest {
        let mut own = sample_kernel(OWN_KEY, "run", false);
        own.variants = vec![
            body(OWN_KEY, &[], ""),
            body(VARIANT_KEY, &[Capability::Subgroups], "run_subgroups"),
        ];
        let mut variant = sample_kernel(VARIANT_KEY, "run", false);
        variant.entry_point = "gpu_demo_run_subgroups".to_owned();
        Manifest {
            slang: Some("2026.13.1-test".to_owned()),
            kernels: vec![own, variant, sample_kernel(OTHER_KEY, "other", false)],
        }
    }

    /// The manifest as the reader gives it back from `render`: kernels and list items in key order.
    fn canonical(manifest: &Manifest) -> Manifest {
        let mut sorted = manifest.clone();
        sorted.kernels.sort_by(|a, b| a.key.cmp(&b.key));
        for kernel in &mut sorted.kernels {
            kernel.variants.sort_by(|a, b| a.key.cmp(&b.key));
            kernel
                .formats
                .sort_by_key(|format| format_rank(format.format));
        }
        sorted
    }

    /// Render `manifest` and read it back: the refusal of the reader.
    fn refusal(manifest: &Manifest) -> String {
        Manifest::parse(&manifest.render()).unwrap_err()
    }

    #[test]
    fn variants_make_a_schema_3_manifest_and_only_the_own_body_carries_the_list() {
        let manifest = variant_manifest();
        assert_eq!(manifest.schema(), VARIANT_SCHEMA);
        let text = manifest.render();
        assert!(text.contains("\"schema\": 3,"), "{text}");
        assert_eq!(text.matches("\"variants\"").count(), 1, "{text}");
        // Every grid kernel: no `shape` key, as in schema 1.
        assert!(!text.contains("shape"), "{text}");
        // The list is the last key of the own body's entry, sorted by key, and an empty `requires` stays `[]`.
        let owner = concat!(
            "      \"key\": \"0123456789abcdef\",\n",
            "      \"module\": \"demo\",\n",
            "      \"name\": \"run\",\n",
            "      \"variants\": [\n",
            "        {\n",
            "          \"key\": \"00000000000000aa\",\n",
            "          \"requires\": [\n",
            "            \"subgroups\"\n",
            "          ],\n",
            "          \"variant\": \"run_subgroups\"\n",
            "        },\n",
            "        {\n",
            "          \"key\": \"0123456789abcdef\",\n",
            "          \"requires\": [],\n",
            "          \"variant\": \"\"\n",
            "        }\n",
            "      ]\n",
            "    }\n"
        );
        assert!(text.contains(owner), "{text}");
        // The variant has an entry of its own (files, hashes, its own entry point) with the kernel's module and name.
        assert!(
            text.contains("\"entry_point\": \"gpu_demo_run_subgroups\""),
            "{text}"
        );
        assert_eq!(Manifest::parse(&text).unwrap(), canonical(&manifest));
        // The kernels are sorted by key whatever order they come in, the list items too.
        let mut shuffled = manifest.clone();
        shuffled.kernels.reverse();
        shuffled.kernels[0].variants.reverse();
        assert_eq!(shuffled.render(), text);
    }

    #[test]
    fn a_manifest_without_variants_keeps_its_schema_and_its_bytes() {
        let grid = Manifest {
            slang: Some("2026.13.1-test".to_owned()),
            kernels: vec![sample_kernel(OWN_KEY, "run", true)],
        };
        assert_eq!(grid.schema(), SCHEMA);
        assert!(!grid.render().contains("variants"));
        let pixel = manifest_with_a_pixel_kernel();
        assert_eq!(pixel.schema(), SHAPE_SCHEMA);
        assert!(!pixel.render().contains("variants"));
        // Taking the variants away from the schema 3 manifest gives the schema 1 manifest of the same kernels.
        let mut plain = variant_manifest();
        for kernel in &mut plain.kernels {
            kernel.variants.clear();
        }
        assert_eq!(plain.schema(), SCHEMA);
        let text = variant_manifest().render();
        assert!(
            !plain.render().contains("variants") && plain.render().contains("\"schema\": 1,"),
            "{text}"
        );
        assert_eq!(Manifest::empty().render().matches("schema").count(), 1);
    }

    #[test]
    fn a_schema_3_manifest_with_a_pixel_kernel_writes_every_shape_and_reads_back() {
        let mut manifest = variant_manifest();
        manifest.kernels[1].shape = Shape::Pixel;
        let text = manifest.render();
        assert!(text.contains("\"schema\": 3,"), "{text}");
        assert_eq!(text.matches("\"shape\"").count(), 3, "{text}");
        assert_eq!(text.matches("\"shape\": \"pixel\"").count(), 1, "{text}");
        assert_eq!(Manifest::parse(&text).unwrap(), canonical(&manifest));
    }

    #[test]
    fn a_kernel_that_needs_a_capability_lists_only_itself() {
        // A kernel with one body that requires a capability and no variant: a list of one item.
        let mut own = sample_kernel(OWN_KEY, "run", false);
        own.variants = vec![body(OWN_KEY, &[Capability::Subgroups], "")];
        let manifest = Manifest {
            slang: Some("s".to_owned()),
            kernels: vec![own],
        };
        let text = manifest.render();
        assert!(text.contains("\"schema\": 3,"), "{text}");
        assert!(
            text.contains(concat!(
                "          \"key\": \"0123456789abcdef\",\n",
                "          \"requires\": [\n",
                "            \"subgroups\"\n",
                "          ],\n",
                "          \"variant\": \"\"\n"
            )),
            "{text}"
        );
        assert_eq!(Manifest::parse(&text).unwrap(), manifest);
    }

    #[test]
    fn a_schema_3_manifest_without_a_list_reads_but_is_not_canonical() {
        let good = Manifest {
            slang: Some("s".to_owned()),
            kernels: vec![sample_kernel(OWN_KEY, "run", true)],
        }
        .render();
        let lenient = good.replace("\"schema\": 1", "\"schema\": 3");
        let parsed = Manifest::parse(&lenient).unwrap();
        assert_eq!(parsed.schema(), SCHEMA);
        assert_ne!(parsed.render(), lenient);
    }

    #[test]
    fn variants_under_schema_1_or_2_are_an_unknown_key_and_the_schema_list_names_three() {
        let text = variant_manifest().render();
        for older in ["\"schema\": 1", "\"schema\": 2"] {
            let error = Manifest::parse(&text.replace("\"schema\": 3", older)).unwrap_err();
            assert!(
                error.contains("kernels[2]: unknown key `variants`"),
                "{error}"
            );
        }
        let error = Manifest::parse(&text.replace("\"schema\": 3", "\"schema\": 4")).unwrap_err();
        assert_eq!(
            error,
            "manifest: unsupported schema (this tool reads schemas 1, 2 and 3)"
        );
    }

    #[test]
    fn a_malformed_variants_list_is_refused_with_its_path() {
        let good = variant_manifest().render();
        for (what, bad, wanted) in [
            (
                "an empty list",
                good.replace(
                    &good[good.find("\"variants\": [").unwrap()..good.rfind("      ]\n").unwrap()],
                    "\"variants\": [",
                ),
                "kernels[2].variants must be a non-empty array (omit it when the kernel has no variants)",
            ),
            (
                "an unknown capability",
                good.replace("\"subgroups\"", "\"wave\""),
                "kernels[2].variants[0].requires: unknown capability `wave` (this tool knows `subgroups`)",
            ),
            (
                "a capability twice",
                good.replace(
                    "\"subgroups\"\n",
                    "\"subgroups\",\n            \"subgroups\"\n",
                ),
                "kernels[2].variants[0].requires: `subgroups` appears twice",
            ),
            (
                "a capability that is not text",
                good.replace("\"subgroups\"\n", "3\n"),
                "kernels[2].variants[0].requires: expected capability names (strings)",
            ),
            (
                "a missing requires",
                good.replace("          \"requires\": [],\n", ""),
                "kernels[2].variants[1]: missing `requires`",
            ),
            (
                "an unknown key in an item",
                good.replace(
                    "          \"variant\": \"run_subgroups\"",
                    "          \"extra\": 1,\n          \"variant\": \"run_subgroups\"",
                ),
                "kernels[2].variants[0]: unknown key `extra`",
            ),
            (
                "a short key",
                good.replace(
                    "\"key\": \"00000000000000aa\",\n          \"requires\"",
                    "\"key\": \"aa\",\n          \"requires\"",
                ),
                "kernels[2].variants[0]: `key` must be 16 lower-case hex digits",
            ),
            (
                "a variant that is not text",
                good.replace("\"variant\": \"run_subgroups\"", "\"variant\": 7"),
                "kernels[2].variants[0]: `variant` must be a string",
            ),
        ] {
            let error = Manifest::parse(&bad).unwrap_err();
            assert!(error.contains(wanted), "{what}: {error}");
        }
    }

    #[test]
    fn the_links_between_the_list_and_the_entries_are_checked() {
        // The own body missing from the list.
        let mut manifest = variant_manifest();
        manifest.kernels[0].variants.remove(0);
        assert_eq!(
            refusal(&manifest),
            "kernels[2] (`demo::run`): `variants` does not list the kernel's own body (key 0123456789abcdef, `variant` \"\")"
        );
        // The own item under another key.
        let mut manifest = variant_manifest();
        manifest.kernels[0].variants[0].key = OTHER_KEY.to_owned();
        assert_eq!(
            refusal(&manifest),
            "kernels[2] (`demo::run`): `variants` lists the own body (`variant` \"\") under key 00000000000000ff, but this entry is key 0123456789abcdef"
        );
        // Two own items.
        let mut manifest = variant_manifest();
        manifest.kernels[0].variants.push(body(OTHER_KEY, &[], ""));
        assert_eq!(
            refusal(&manifest),
            "kernels[2] (`demo::run`): `variants` lists the kernel's own body (`variant` \"\") more than once"
        );
        // A variant named twice (a third entry of the kernel with the same function name).
        let mut manifest = variant_manifest();
        let mut third = sample_kernel("00000000000000bb", "run", false);
        third.entry_point = "gpu_demo_run_subgroups".to_owned();
        manifest.kernels.push(third);
        manifest.kernels[0].variants.push(body(
            "00000000000000bb",
            &[Capability::Subgroups],
            "run_subgroups",
        ));
        assert!(
            refusal(&manifest).ends_with("`variants` names the variant `run_subgroups` twice"),
            "{}",
            refusal(&manifest)
        );
        // A key that is no entry.
        let mut manifest = variant_manifest();
        manifest.kernels[0].variants[1].key = "00000000000000cc".to_owned();
        assert_eq!(
            refusal(&manifest),
            "kernels[2] (`demo::run`).variants[0]: key 00000000000000cc is not an entry of this manifest"
        );
        // A key that is the entry of another kernel.
        let mut manifest = variant_manifest();
        manifest.kernels[0].variants[1].key = OTHER_KEY.to_owned();
        assert_eq!(
            refusal(&manifest),
            "kernels[2] (`demo::run`).variants[0]: key 00000000000000ff is the entry of `demo::other`, not of `demo::run`"
        );
        // A listed body with a list of its own.
        let mut manifest = variant_manifest();
        // (a third body whose key sorts after the own body's, so that the own body's entry is checked first)
        let late = "fedcba9876543210";
        let mut third = sample_kernel(late, "run", false);
        third.entry_point = "gpu_demo_run_late".to_owned();
        third.variants = vec![body(late, &[], "late")];
        manifest.kernels.push(third);
        manifest.kernels[0].variants.push(body(late, &[], "late"));
        assert_eq!(
            refusal(&manifest),
            "kernels[2] (`demo::run`).variants[2]: key fedcba9876543210 has a `variants` list of its own; only the entry of the kernel's own body has one"
        );
        // An entry of the kernel that the list leaves out.
        let mut manifest = variant_manifest();
        manifest.kernels[0].variants.remove(1);
        assert_eq!(
            refusal(&manifest),
            "kernels[2] (`demo::run`): the entry with key 00000000000000aa has the kernel's module and name but is not in `variants`"
        );
        // Two entries of one kernel that both carry a list.
        let mut manifest = variant_manifest();
        manifest.kernels[0].variants.remove(1);
        manifest.kernels[1].variants = vec![body(VARIANT_KEY, &[], "")];
        assert_eq!(
            refusal(&manifest),
            "kernels[0] (`demo::run`): kernels[2] has the same module and name and a `variants` list as well"
        );
        // Two entries of one kernel and no list anywhere (S1): the kernels sort by key, so the variant is kernels[0].
        let mut manifest = variant_manifest();
        for kernel in &mut manifest.kernels {
            kernel.variants.clear();
        }
        assert_eq!(
            refusal(&manifest),
            "kernels[0] (`demo::run`): kernels[2] has the same module and name, and no entry of the kernel has a `variants` list"
        );
    }
}

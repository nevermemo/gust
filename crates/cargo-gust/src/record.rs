//! The kernel records the `#[gpu]` macro writes in export mode (D46), and the `KernelDescriptor` the tool rebuilds
//! from them.
//!
//! For every kernel the macro writes `<key>.slang` (the clean Slang text, exact bytes) and `<key>.kernel.json`
//! (schema 1, or schema 2 for a `par_for!` call) under `<GUST_EXPORT_DIR>/<CARGO_PKG_NAME>/`. The JSON holds what
//! the tool needs to rebuild the
//! descriptor without a compiler: `schema`, `key`, `module`, `name`, `entry_point`, `workgroup_size` (three numbers),
//! `parameters` (each `name`, `rust_type`, `kind` of `invocation`, `storage`, `uniform` or `value`, `access` of
//! `read_only`, `read_write` or `not_applicable`, `binding` as `{"group": 0, "binding": n}` or null, and `layout`:
//! null or `{"rules": "storage_v1", "element_stride": n, "element": <layout>}`) and `source` (`file`, and `lines`
//! as `[slang_start, slang_end, rust_line]` triples). A `<layout>` is `name`, `size`, `alignment`, `stride`, `kind`
//! (`scalar` with `scalar` of `u32`, `i32` or `f32`, or `struct` with `fields` of `name`, `offset` and a nested
//! `type`). Unknown keys are ignored; a missing or mistyped one is an error that names its path.
//!
//! Schema 2 (R1.1 M5, slice m7.p) is the record of a `par_for!` call: the same keys without `source`, because one body
//! can stand at any number of call sites and the record must not depend on them. Each call site writes a sidecar
//! `<key>.<site16>.site.json` (`schema` 2, `key`, `file`, `lines` as above). The reader takes a schema 2 record's
//! `file` and `lines` from its first sidecar by file name, so the descriptor it builds has the origin of the first
//! call site.

use std::{fs, path::Path};

use gust::{
    Access, KernelDescriptor, LineSpan, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout, ScalarKind, ShaderLayoutRules, SourceOrigin, StructFieldLayout, TypeLayout,
    TypeLayoutKind, artifact::content_key,
};

use crate::{
    json::{self, Json},
    sha256::sha256_hex,
};

/// The schema of a record that carries its own `source`: a `#[gpu]` module or function.
pub(crate) const RECORD_SCHEMA: u32 = 1;
/// The schema of a `par_for!` record and of its sidecars: no `source` in the record, one sidecar per call site.
pub(crate) const SITE_RECORD_SCHEMA: u32 = 2;

/// The suffix of the file that holds a record's JSON.
pub(crate) const JSON_SUFFIX: &str = ".kernel.json";
/// The suffix of the file that holds a record's clean Slang text.
pub(crate) const SLANG_SUFFIX: &str = ".slang";
/// The suffix of a call site's sidecar: `<key>.<16 hex digits>.site.json`.
pub(crate) const SITE_SUFFIX: &str = ".site.json";

/// One collected kernel: its record plus the clean Slang text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Record {
    /// [`content_key`] of the entry point and the clean text, as 16 lower-case hex digits.
    pub(crate) key: String,
    pub(crate) module: String,
    pub(crate) name: String,
    pub(crate) entry_point: String,
    pub(crate) workgroup_size: [u32; 3],
    pub(crate) parameters: Vec<ParameterRecord>,
    /// The recorded Rust source file (relative, `/` separators); empty when unknown. For a schema 2 record: the
    /// file of its first call site.
    pub(crate) file: String,
    /// `[slang_start, slang_end, rust_line]`. For a schema 2 record: the lines of its first call site.
    pub(crate) lines: Vec<[u32; 3]>,
    /// The clean Slang text, byte for byte.
    pub(crate) slang: String,
}

/// One call site of a `par_for!` body: the source file of the call and the line table of its copy of the body.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Site {
    file: String,
    lines: Vec<[u32; 3]>,
}

impl Site {
    /// Read a sidecar from its text; `key` is the key of the record it must belong to.
    fn parse(json_text: &str, key: &str) -> Result<Self, String> {
        let root = json::parse(json_text)?;
        if root.field("schema", "site")?.as_u32() != Some(SITE_RECORD_SCHEMA) {
            return Err(format!(
                "site: unsupported schema (a site record is schema {SITE_RECORD_SCHEMA})"
            ));
        }
        let found = string(&root, "key", "site")?;
        if found != key {
            return Err(format!(
                "site: key `{found}` is not the key of its kernel record (`{key}`)"
            ));
        }
        Ok(Self {
            file: string(&root, "file", "site")?,
            lines: parse_lines(array(&root, "lines", "site")?, "site.lines")?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParameterRecord {
    pub(crate) name: String,
    pub(crate) rust_type: String,
    pub(crate) kind: ParameterKind,
    pub(crate) access: Access,
    /// `(group, binding)`.
    pub(crate) binding: Option<(u32, u32)>,
    pub(crate) layout: Option<ResourceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResourceRecord {
    pub(crate) element_stride: u32,
    pub(crate) element: LayoutRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayoutRecord {
    pub(crate) name: String,
    pub(crate) size: u32,
    pub(crate) alignment: u32,
    pub(crate) stride: u32,
    pub(crate) kind: LayoutKindRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LayoutKindRecord {
    Scalar(ScalarKind),
    Struct(Vec<FieldRecord>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FieldRecord {
    pub(crate) name: String,
    pub(crate) offset: u32,
    pub(crate) ty: LayoutRecord,
}

/// The key of a kernel as the macro and the tool compute it.
pub(crate) fn key_of(entry_point: &str, slang: &str) -> String {
    format!("{:016x}", content_key(entry_point, slang))
}

impl Record {
    /// [`Record::parse_any`] without the schema.
    #[cfg(test)]
    pub(crate) fn parse(json_text: &str, slang: &str) -> Result<Self, String> {
        Self::parse_any(json_text, slang).map(|(record, _)| record)
    }

    /// Read a record from the text of its `.kernel.json` and its `.slang`, and give its schema.
    ///
    /// Also checks that the stored key is the key of the entry point and the Slang text, so a `.slang` and a `.json`
    /// that belong to different kernels (two kernels whose keys collide would overwrite each other's files) are an
    /// error, not a silent mix. A schema 2 record comes back without a source origin: its call sites are in its
    /// sidecars ([`read_records`] attaches the first).
    fn parse_any(json_text: &str, slang: &str) -> Result<(Self, u32), String> {
        let root = json::parse(json_text)?;
        let schema = match root.field("schema", "record")?.as_u32() {
            Some(schema @ (RECORD_SCHEMA | SITE_RECORD_SCHEMA)) => schema,
            _ => {
                return Err(format!(
                    "record: unsupported schema (this tool reads schemas {RECORD_SCHEMA} and {SITE_RECORD_SCHEMA}); regenerate it with a matching gust-macros"
                ));
            }
        };
        let key = string(&root, "key", "record")?;
        let entry_point = string(&root, "entry_point", "record")?;
        let expected = key_of(&entry_point, slang);
        if key != expected {
            return Err(format!(
                "record: key `{key}` is not the key of its entry point and Slang text (`{expected}`)"
            ));
        }
        let size = root.field("workgroup_size", "record")?;
        let size = size
            .as_array()
            .filter(|items| items.len() == 3)
            .ok_or("record: `workgroup_size` must be three numbers")?;
        let mut workgroup_size = [0_u32; 3];
        for (slot, item) in workgroup_size.iter_mut().zip(size) {
            *slot = item
                .as_u32()
                .ok_or("record: `workgroup_size` must be three numbers")?;
        }
        let parameters = array(&root, "parameters", "record")?
            .iter()
            .enumerate()
            .map(|(index, parameter)| parse_parameter(parameter, &format!("parameters[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let (file, lines) = if schema == SITE_RECORD_SCHEMA {
            // The call sites are the sidecars' business; a record that names a source would differ between two calls
            // with equal bodies, which is what schema 2 exists to avoid.
            if root.get("source").is_some() {
                return Err(
                    "record: a schema 2 record has no `source`; its call sites are in its `.site.json` files"
                        .to_owned(),
                );
            }
            (String::new(), Vec::new())
        } else {
            let source = root.field("source", "record")?;
            (
                string(source, "file", "source")?,
                parse_lines(array(source, "lines", "source")?, "source.lines")?,
            )
        };
        let record = Self {
            key,
            module: string(&root, "module", "record")?,
            name: string(&root, "name", "record")?,
            entry_point,
            workgroup_size,
            parameters,
            file,
            lines,
            slang: slang.to_owned(),
        };
        Ok((record, schema))
    }

    /// Take the origin of a schema 2 record from its call sites, in the order of their sidecars' file names: the
    /// first one's.
    fn with_origin_of(mut self, schema: u32, sites: &[Site]) -> Result<Self, String> {
        if schema != SITE_RECORD_SCHEMA {
            if sites.is_empty() {
                return Ok(self);
            }
            return Err(format!(
                "record `{}` is schema {schema}, which has no call sites, but a site file of its key exists",
                self.key
            ));
        }
        let Some(first) = sites.first() else {
            return Err(format!(
                "record `{}` is schema 2 but has no `{}.<16 hex digits>{SITE_SUFFIX}` file: a par_for! kernel needs at least one call site",
                self.key, self.key
            ));
        };
        self.file.clone_from(&first.file);
        self.lines.clone_from(&first.lines);
        Ok(self)
    }

    /// `module::name`, as diagnostics and the manifest name the kernel.
    pub(crate) fn qualified_name(&self) -> String {
        format!("{}::{}", self.module, self.name)
    }

    /// SHA-256 of the clean Slang text (D45), as 64 hex digits.
    pub(crate) fn clean_sha256(&self) -> String {
        sha256_hex(self.slang.as_bytes())
    }

    /// The storage parameters whose element layout the macro could not write (`layout: null`), in parameter order.
    /// The manifest lists them as `export_unchecked`.
    pub(crate) fn unchecked_parameters(&self) -> Vec<String> {
        self.parameters
            .iter()
            .filter(|parameter| {
                parameter.kind == ParameterKind::Storage && parameter.layout.is_none()
            })
            .map(|parameter| parameter.name.clone())
            .collect()
    }

    /// A copy whose unchecked parameters have the layout the reflection JSON reports (see [`crate::fill`]). A record
    /// without unchecked parameters comes back unchanged and the JSON is not read.
    pub(crate) fn completed(&self, reflection_json: &str) -> Result<Self, String> {
        let mut completed = self.clone();
        for parameter in &mut completed.parameters {
            if parameter.kind == ParameterKind::Storage && parameter.layout.is_none() {
                parameter.layout = Some(
                    crate::fill::reflected_layout(reflection_json, &parameter.name)
                        .map_err(|error| format!("{}: {error}", self.qualified_name()))?,
                );
            }
        }
        Ok(completed)
    }

    /// Rebuild the kernel's descriptor. Strings and tables are leaked: the tool is a short-lived process and the
    /// descriptor type wants `'static` data.
    ///
    /// A storage parameter without an element layout (the macro could not describe its type) gets no resource layout:
    /// compile with this descriptor, then [`Record::completed`] and build the descriptor again to cross-check.
    pub(crate) fn descriptor(&self) -> KernelDescriptor {
        let mut parameters = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            parameters.push(ParameterDescriptor {
                name: leak(&parameter.name),
                rust_type: leak(&parameter.rust_type),
                kind: parameter.kind,
                access: parameter.access,
                resource_layout: parameter.layout.as_ref().map(|layout| ResourceLayout {
                    rules: ShaderLayoutRules::StorageV1,
                    element: layout.element.type_layout(),
                    element_stride: layout.element_stride,
                }),
                binding: parameter
                    .binding
                    .map(|(group, binding)| ResourceBinding { group, binding }),
            });
        }
        let lines: Vec<LineSpan> = self
            .lines
            .iter()
            .map(|[slang_start, slang_end, rust_line]| LineSpan {
                slang_start: *slang_start,
                slang_end: *slang_end,
                rust_line: *rust_line,
            })
            .collect();
        KernelDescriptor {
            module: leak(&self.module),
            name: leak(&self.name),
            entry_point: leak(&self.entry_point),
            workgroup_size: self.workgroup_size,
            parameters: Box::leak(parameters.into_boxed_slice()),
            slang_source: leak(&self.slang),
            source: if self.file.is_empty() && lines.is_empty() {
                SourceOrigin::UNKNOWN
            } else {
                SourceOrigin {
                    file: leak(&self.file),
                    lines: Box::leak(lines.into_boxed_slice()),
                }
            },
            artifacts: &[],
        }
    }
}

fn leak(text: &str) -> &'static str {
    Box::leak(text.to_owned().into_boxed_str())
}

impl LayoutRecord {
    fn type_layout(&self) -> TypeLayout {
        TypeLayout {
            name: leak(&self.name),
            size: self.size,
            alignment: self.alignment,
            stride: self.stride,
            kind: match &self.kind {
                LayoutKindRecord::Scalar(scalar) => TypeLayoutKind::Scalar(*scalar),
                LayoutKindRecord::Struct(fields) => {
                    let fields: Vec<StructFieldLayout> = fields
                        .iter()
                        .map(|field| StructFieldLayout {
                            name: leak(&field.name),
                            offset: field.offset,
                            ty: field.ty.type_layout(),
                        })
                        .collect();
                    TypeLayoutKind::Struct(Box::leak(fields.into_boxed_slice()))
                }
            },
        }
    }
}

fn string(value: &Json, key: &str, context: &str) -> Result<String, String> {
    value
        .field(key, context)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{context}: `{key}` must be a string"))
}

fn number(value: &Json, key: &str, context: &str) -> Result<u32, String> {
    value
        .field(key, context)?
        .as_u32()
        .ok_or_else(|| format!("{context}: `{key}` must be a non-negative integer"))
}

fn array<'a>(value: &'a Json, key: &str, context: &str) -> Result<&'a [Json], String> {
    value
        .field(key, context)?
        .as_array()
        .ok_or_else(|| format!("{context}: `{key}` must be an array"))
}

/// The `[slang_start, slang_end, rust_line]` triples of a line table.
fn parse_lines(lines: &[Json], context: &str) -> Result<Vec<[u32; 3]>, String> {
    lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let context = format!("{context}[{index}]");
            let triple = line
                .as_array()
                .filter(|items| items.len() == 3)
                .ok_or(format!(
                    "{context}: expected [slang_start, slang_end, rust_line]"
                ))?;
            let mut numbers = [0_u32; 3];
            for (slot, item) in numbers.iter_mut().zip(triple) {
                *slot = item
                    .as_u32()
                    .ok_or(format!("{context}: expected three numbers"))?;
            }
            Ok(numbers)
        })
        .collect()
}

fn parse_parameter(value: &Json, context: &str) -> Result<ParameterRecord, String> {
    let kind = match string(value, "kind", context)?.as_str() {
        "invocation" => ParameterKind::Invocation,
        "storage" => ParameterKind::Storage,
        "uniform" => ParameterKind::Uniform,
        "value" => ParameterKind::Value,
        other => return Err(format!("{context}: unknown parameter kind `{other}`")),
    };
    let access = match string(value, "access", context)?.as_str() {
        "read_only" => Access::ReadOnly,
        "read_write" => Access::ReadWrite,
        "not_applicable" => Access::NotApplicable,
        other => return Err(format!("{context}: unknown access `{other}`")),
    };
    let binding = value.field("binding", context)?;
    let binding = if binding.is_null() {
        None
    } else {
        let at = format!("{context}.binding");
        Some((
            number(binding, "group", &at)?,
            number(binding, "binding", &at)?,
        ))
    };
    let layout = value.field("layout", context)?;
    let layout = if layout.is_null() {
        None
    } else {
        let at = format!("{context}.layout");
        let rules = string(layout, "rules", &at)?;
        if rules != "storage_v1" {
            return Err(format!("{at}: unknown layout rules `{rules}`"));
        }
        Some(ResourceRecord {
            element_stride: number(layout, "element_stride", &at)?,
            element: parse_layout(layout.field("element", &at)?, &format!("{at}.element"))?,
        })
    };
    Ok(ParameterRecord {
        name: string(value, "name", context)?,
        rust_type: string(value, "rust_type", context)?,
        kind,
        access,
        binding,
        layout,
    })
}

fn parse_layout(value: &Json, context: &str) -> Result<LayoutRecord, String> {
    let kind = match string(value, "kind", context)?.as_str() {
        "scalar" => LayoutKindRecord::Scalar(match string(value, "scalar", context)?.as_str() {
            "u32" => ScalarKind::U32,
            "i32" => ScalarKind::I32,
            "f32" => ScalarKind::F32,
            other => return Err(format!("{context}: unknown scalar `{other}`")),
        }),
        "struct" => LayoutKindRecord::Struct(
            array(value, "fields", context)?
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let at = format!("{context}.fields[{index}]");
                    Ok(FieldRecord {
                        name: string(field, "name", &at)?,
                        offset: number(field, "offset", &at)?,
                        ty: parse_layout(field.field("type", &at)?, &format!("{at}.type"))?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        other => return Err(format!("{context}: unknown layout kind `{other}`")),
    };
    Ok(LayoutRecord {
        name: string(value, "name", context)?,
        size: number(value, "size", context)?,
        alignment: number(value, "alignment", context)?,
        stride: number(value, "stride", context)?,
        kind,
    })
}

/// A temporary the macro's record writer leaves behind when it cannot remove it (`.<file>.<pid>.<n>.tmp`).
fn is_macro_temporary(name: &str) -> bool {
    name.starts_with('.') && name.ends_with(".tmp")
}

/// Read every record in `directory` (`<GUST_EXPORT_DIR>/<package>/`), sorted by key.
///
/// A `.kernel.json` without its `.slang`, a `.slang` without its `.kernel.json` and any other file are errors:
/// the directory is the macro's output and nothing else, so anything else means a mixed-up collection. Dot-prefixed
/// `*.tmp` files are leftovers of the macro's atomic write and are ignored.
/// A missing directory is an empty collection (the caller reports zero kernels).
pub(crate) fn read_records(directory: &Path) -> Result<Vec<Record>, String> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("cannot read the record directory: {error}")),
    };
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read the record directory: {error}"))?;
        let name = entry.file_name().into_string().map_err(|name| {
            format!("record directory holds a file with a non-UTF-8 name: {name:?}")
        })?;
        if is_macro_temporary(&name) {
            continue;
        }
        names.push(name);
    }
    names.sort();
    let mut stems: Vec<&str> = Vec::new();
    let mut site_files: Vec<(&str, &str)> = Vec::new();
    for name in &names {
        if let Some(stem) = name.strip_suffix(SITE_SUFFIX) {
            // `<key>.<16 hex digits>.site.json`; the key is 16 hex digits too.
            let Some((key, site)) = stem.split_once('.') else {
                return Err(format!(
                    "record directory holds an unexpected file `{name}`"
                ));
            };
            if !is_hex16(key) || !is_hex16(site) {
                return Err(format!(
                    "record directory holds an unexpected file `{name}`"
                ));
            }
            site_files.push((key, name));
            if !stems.contains(&key) {
                stems.push(key);
            }
            continue;
        }
        let stem = name
            .strip_suffix(JSON_SUFFIX)
            .or_else(|| name.strip_suffix(SLANG_SUFFIX))
            .ok_or_else(|| format!("record directory holds an unexpected file `{name}`"))?;
        if !stems.contains(&stem) {
            stems.push(stem);
        }
    }
    let mut records = Vec::new();
    for stem in stems {
        let slang_name = format!("{stem}{SLANG_SUFFIX}");
        let json_name = format!("{stem}{JSON_SUFFIX}");
        for needed in [&slang_name, &json_name] {
            if !names.contains(needed) {
                return Err(format!(
                    "record `{stem}` is incomplete: `{needed}` is missing"
                ));
            }
        }
        let slang = read_text(&directory.join(&slang_name), &slang_name)?;
        let json = read_text(&directory.join(&json_name), &json_name)?;
        let (record, schema) =
            Record::parse_any(&json, &slang).map_err(|error| format!("{json_name}: {error}"))?;
        if record.key != stem {
            return Err(format!(
                "{json_name}: the file name does not match the record key `{}`",
                record.key
            ));
        }
        let mut sites = Vec::new();
        for (_, site_name) in site_files.iter().filter(|(key, _)| *key == stem) {
            let text = read_text(&directory.join(site_name), site_name)?;
            sites.push(Site::parse(&text, stem).map_err(|error| format!("{site_name}: {error}"))?);
        }
        let record = record
            .with_origin_of(schema, &sites)
            .map_err(|error| format!("{json_name}: {error}"))?;
        records.push(record);
    }
    records.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(records)
}

/// Sixteen lower-case hex digits: a record key or a call site's hash.
fn is_hex16(text: &str) -> bool {
    text.len() == 16
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_text(path: &Path, shown: &str) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read `{shown}`: {error}"))?;
    String::from_utf8(bytes).map_err(|_| format!("`{shown}` is not UTF-8"))
}

#[cfg(test)]
#[path = "record_tests.rs"]
mod tests;

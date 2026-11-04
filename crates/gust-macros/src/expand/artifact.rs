//! The expansion's side of precompiled artifacts (M4, D45) and export mode (D46).
//!
//! For every kernel the macro computes the content key of its clean Slang text ([`content_key`]), looks for the
//! artifact pair `gust-artifacts/<key>.wgsl` and `gust-artifacts/<key>.wgsl.reflection.json` under the expanding
//! crate's `CARGO_MANIFEST_DIR` ([`lookup`]), and embeds a complete pair with `include_bytes!` and `include_str!`
//! ([`artifact_tokens`]). The emitted paths are `concat!(env!("CARGO_MANIFEST_DIR"), "/gust-artifacts/..")`, so the
//! build machine's directory never enters the expansion and a Windows path with backslashes and spaces needs no
//! escaping. `gust-artifacts/manifest.json` is included once per module expansion when it exists
//! ([`manifest_tokens`]), so that cargo tracks it and re-expands the crate when it changes.
//!
//! Export mode (`GUST_EXPORT_DIR` set and non-empty) additionally writes, under
//! `<GUST_EXPORT_DIR>/<CARGO_PKG_NAME>/`, the clean text as `<key>.slang` (exact bytes) and `<key>.kernel.json`
//! ([`KernelRecord::json`], schema 1; schema 2 for a `par_for!` call; schema 3 only for a kernel whose shape is not
//! `grid`, which no expansion produces yet). The records hold what `cargo gust build` needs to rebuild the kernel's
//! `KernelDescriptor` without a compiler: module, name, entry point, workgroup size, the parameters with kind, access,
//! binding and the complete element layout tree, and the source file with its line table. They never hold CPU twin
//! text (D45): [`write_records`] refuses a record that does.
//!
//! Everything here reads the environment through [`Environment`], so tests pass their own directories instead of
//! changing the process environment.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use proc_macro2::{Span, TokenStream};
use quote::quote;

use crate::{
    diagnostics::{self, Code},
    slang::LineRecord,
};

/// The directory of a kernel crate that holds its committed artifacts.
const DIRECTORY: &str = "gust-artifacts";

/// The manifest of that directory.
const MANIFEST: &str = "manifest.json";

/// The version of the `kernel.json` layout of a `#[gpu]` module or function: it carries the kernel's `source`.
const SCHEMA: u64 = 1;

/// The version of the `kernel.json` layout of a `par_for!` call (R1.1 M5): the same record without `source`, so equal
/// bodies at two call sites write the same bytes, plus one [`SiteRecord`] sidecar per call site.
const SITE_SCHEMA: u64 = 2;

/// The version of the `kernel.json` layout of a kernel whose shape is not `grid` (mx.k, D89): schema 2's grammar
/// (`source` optional: present for a `#[gpu]` module or function, absent for a `par_for!` call) plus `shape`. The
/// number moves so that a reader of schema 1 and 2 refuses the record instead of reading a pixel kernel as `grid`. A
/// `grid` kernel keeps schema 1 or 2 and has no `shape` key, byte for byte.
const SHAPE_SCHEMA: u64 = 3;

/// What the expansion reads from the process environment.
#[derive(Clone, Debug, Default)]
pub(crate) struct Environment {
    /// `CARGO_MANIFEST_DIR`: the expanding crate.
    pub manifest_dir: Option<PathBuf>,
    /// `GUST_EXPORT_DIR`, when set and non-empty: export mode.
    pub export_dir: Option<PathBuf>,
    /// `CARGO_PKG_NAME`: names the record directory.
    pub package: Option<String>,
    /// The file rustc reports for the module, in place of the span's own. `from_process` leaves it unset (the span
    /// knows its file); the fallback spans of a unit test have none, so a test that needs a source file sets this.
    pub reported_file: Option<String>,
}

impl Environment {
    pub(crate) fn from_process() -> Self {
        let variable = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
        Self {
            manifest_dir: variable("CARGO_MANIFEST_DIR").map(PathBuf::from),
            export_dir: variable("GUST_EXPORT_DIR").map(PathBuf::from),
            package: variable("CARGO_PKG_NAME").and_then(|name| name.into_string().ok()),
            reported_file: None,
        }
    }

    /// The directory that export mode writes this crate's records to, `None` outside export mode. A package name
    /// that is not a plain cargo name is refused: it names a directory, so it must not climb out of the export
    /// directory.
    pub(super) fn record_directory(&self) -> Result<Option<PathBuf>, String> {
        let Some(export) = &self.export_dir else {
            return Ok(None);
        };
        match self.package.as_deref() {
            Some(package)
                if !package.is_empty()
                    && package
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')) =>
            {
                Ok(Some(export.join(package)))
            }
            _ => Err(
                "gust: GUST_EXPORT_DIR is set but CARGO_PKG_NAME is missing or is not a plain package name, so the record directory cannot be named"
                    .to_owned(),
            ),
        }
    }
}

/// The key of a kernel's artifacts: FNV-1a 64 over the entry point, one NUL byte and the clean Slang text exactly as
/// the macro holds it (no `#line` directives, no line-ending normalisation), as 16 lower-case hex digits.
///
/// This is a byte-for-byte copy of `gust::artifact::content_key`; both crates assert the same golden strings
/// (`expand::artifact::tests::the_key_matches_the_golden_values` and `gust::artifact::tests`). The macro cannot call
/// the original: `gust` depends on this crate.
pub(crate) fn content_key(entry_point: &str, clean_slang: &str) -> String {
    let hash = [entry_point.as_bytes(), &[0], clean_slang.as_bytes()]
        .iter()
        .flat_map(|part| part.iter())
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
    format!("{hash:016x}")
}

/// What `gust-artifacts/` holds for one key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Lookup {
    /// Neither file exists (or there is no manifest directory to look in).
    Absent,
    /// Both files exist.
    Present,
    /// Exactly one exists: `left` is that file and `missing` the other, both relative to the crate. Outside export
    /// mode this is an error, never a silent absent pair; export mode treats it as absent so that `cargo gust build`
    /// can write the pair again.
    Incomplete { left: String, missing: String },
}

/// The file name of the WGSL artifact of `key`.
fn wgsl_name(key: &str) -> String {
    format!("{key}.wgsl")
}

/// The file name of the reflection JSON that belongs to the WGSL artifact of `key`.
fn reflection_name(key: &str) -> String {
    format!("{key}.wgsl.reflection.json")
}

/// A path inside `gust-artifacts/`, as the emitted `concat!` appends it to `CARGO_MANIFEST_DIR` and as messages name
/// it: always `/`, never a machine path.
fn relative_path(name: &str) -> String {
    format!("{DIRECTORY}/{name}")
}

fn is_file(directory: &Path, name: &str) -> bool {
    fs::metadata(directory.join(DIRECTORY).join(name)).is_ok_and(|metadata| metadata.is_file())
}

/// Looks for the artifact pair of `key` in the crate at `manifest_dir`.
pub(super) fn lookup(manifest_dir: Option<&Path>, key: &str) -> Lookup {
    let Some(directory) = manifest_dir else {
        return Lookup::Absent;
    };
    let (wgsl, reflection) = (wgsl_name(key), reflection_name(key));
    match (is_file(directory, &wgsl), is_file(directory, &reflection)) {
        (true, true) => Lookup::Present,
        (false, false) => Lookup::Absent,
        (true, false) => Lookup::Incomplete {
            left: relative_path(&wgsl),
            missing: relative_path(&reflection),
        },
        (false, true) => Lookup::Incomplete {
            left: relative_path(&reflection),
            missing: relative_path(&wgsl),
        },
    }
}

/// Whether the crate at `manifest_dir` has `gust-artifacts/manifest.json`.
pub(super) fn manifest_present(manifest_dir: Option<&Path>) -> bool {
    manifest_dir.is_some_and(|directory| is_file(directory, MANIFEST))
}

/// The `include_bytes!` or `include_str!` argument for a file of `gust-artifacts/`: the crate's own directory, read
/// by the compiler, plus a literal tail.
fn embedded_path(name: &str) -> TokenStream {
    let tail = format!("/{}", relative_path(name));
    quote!(::core::concat!(::core::env!("CARGO_MANIFEST_DIR"), #tail))
}

/// The `artifacts:` field of a kernel descriptor: the one-element slice for a complete pair, `&[]` otherwise.
pub(super) fn artifact_tokens(key: &str, lookup: &Lookup) -> TokenStream {
    if *lookup != Lookup::Present {
        return quote!(&[]);
    }
    let bytes = embedded_path(&wgsl_name(key));
    let reflection = embedded_path(&reflection_name(key));
    quote! {
        &[::gust::Artifact {
            format: ::gust::Format::Wgsl,
            key: #key,
            bytes: ::core::include_bytes!(#bytes),
            reflection: ::core::include_str!(#reflection),
        }]
    }
}

/// The include of `gust-artifacts/manifest.json`, one per module expansion, or nothing when the file does not exist.
/// Its only purpose is that cargo tracks the file: any change to the manifest re-expands the module.
pub(super) fn manifest_tokens(present: bool) -> TokenStream {
    if !present {
        return TokenStream::new();
    }
    let path = embedded_path(MANIFEST);
    quote! {
        const _: &[::core::primitive::u8] = ::core::include_bytes!(#path);
    }
}

/// A coded `compile_error!` at `span`, for the failures of the build directory (a half pair, an export write, a
/// package name). It is an item, not a `syn::Error`, because the expansion goes on and the kernel stays usable.
pub(super) fn error_item(code: Code, span: Span, message: &str) -> TokenStream {
    diagnostics::error_at(code, span, message).to_compile_error()
}

/// The message of the half-pair error ([`Code::HalfArtifactPair`]). It must work: `cargo gust build` expands this
/// crate too (in export mode, which treats a half pair as absent), so the way out is to delete the file that is left
/// and build again.
pub(super) fn incomplete_message(left: &str, missing: &str) -> String {
    format!(
        "gust: this kernel has half of its precompiled artifact pair: `{missing}` is missing. Delete the file that is left, `{left}`, then run `cargo gust build` to write both files again"
    )
}

// ---- layouts ----

/// The size, alignment and field structure of one element type: a `gust::TypeLayout`. Its stride is its size: the
/// macro describes only layouts that are whole multiples of their alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    name: String,
    size: u32,
    alignment: u32,
    kind: LayoutKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LayoutKind {
    /// `u32`, `i32` or `f32`.
    Scalar(&'static str),
    /// `count` (2 or 4) scalars of one kind: `Vec2`, `Vec4`, `IVec2`, `IVec4`, `UVec2`, `UVec4`.
    Vector {
        scalar: &'static str,
        count: u32,
    },
    /// A square column-major `f32` matrix (`Mat2`, `Mat4`): `cols` columns of `rows` scalars, `column_stride` bytes
    /// apart.
    Matrix {
        scalar: &'static str,
        rows: u32,
        cols: u32,
        column_stride: u32,
    },
    Struct(Vec<LayoutField>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LayoutField {
    name: String,
    offset: u32,
    layout: Layout,
}

/// How deep a struct may nest before the layout is given up (a recursive struct is a Rust error, but the macro runs
/// before that error is found).
const MAX_NESTING: usize = 32;

/// The rules a record states for a resource element: the three `gust::ShaderLayoutRules` a kernel descriptor uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rules {
    /// Scalars and padding-free structs of them, alignment 4.
    StorageV1,
    /// A storage element that holds a vector or a matrix: the alignment of each field, no padding.
    StorageV2,
    /// A constant buffer's element (`Uniform<T>`): structs rounded up to 16 (D57, D80).
    Std140V1,
}

impl Rules {
    /// The spelling in the record's `layout.rules`.
    fn name(self) -> &'static str {
        match self {
            Self::StorageV1 => "storage_v1",
            Self::StorageV2 => "storage_v2",
            Self::Std140V1 => "std140_v1",
        }
    }
}

/// The layout of a resource's element and the rules it follows: a `gust::ResourceLayout`. The element stride is the
/// element's size (the stride of a storage element, the size of a constant buffer's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResourceLayout {
    pub rules: Rules,
    element: Layout,
}

/// The layout of a storage parameter's element type, with `StorageV1` rules when the element holds no vector and no
/// matrix and `StorageV2` rules when it does (the choice the descriptor makes at run time with `is_storage_v1`), or
/// `None` when [`element_layout`] cannot describe the type.
pub(super) fn storage_resource(
    element: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
) -> Option<ResourceLayout> {
    let element = element_layout(element, structs)?;
    let rules = if element.holds_math_type() {
        Rules::StorageV2
    } else {
        Rules::StorageV1
    };
    Some(ResourceLayout { rules, element })
}

/// The layout of a uniform parameter's element type: `Std140V1`, from the numbers of [`crate::uniform::std140_struct`],
/// which are the ones the `const` assertions of the expansion make rustc check against the real type (D80). `None`
/// when the element is not a module struct whose fields are all scalars, vectors, `Mat4` or such structs (a `Mat2`
/// has no constant-buffer layout: E0205). `structs` is keyed by unraw name, as [`crate::uniform::module_structs`] is.
pub(super) fn uniform_resource(
    element: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
) -> Option<ResourceLayout> {
    let element = std140_layout_at(element, structs, 0)?;
    matches!(element.kind, LayoutKind::Struct(_)).then_some(ResourceLayout {
        rules: Rules::Std140V1,
        element,
    })
}

/// The StorageV1 or StorageV2 layout of a storage element type, derived from the module's own structs, or `None` when
/// the macro cannot describe the type: anything but a scalar (`u32`, `i32`, `f32` or their `uint`, `int`, `float`
/// aliases), a vector or matrix of `gust::math` that may be a buffer element (`Vec2`, `Vec4`, `IVec2`, `IVec4`,
/// `UVec2`, `UVec4`, `Mat2`, `Mat4`) or a struct of this module whose fields are such types. Structs are padding free,
/// so their offsets are the running sum of the field sizes (each at a multiple of its own alignment) and their
/// alignment is the largest field alignment, which is what the assertions the macro emits (`gpu_pod_impl`) make rustc
/// check against the real type. A struct whose fields would need padding is no layout here (the assertions reject it
/// at compile time). The layout is the macro's claim; `cargo gust build` checks it against the Slang reflection (D17)
/// and the runtime checks it again.
pub(super) fn element_layout(
    element: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
) -> Option<Layout> {
    layout_at(element, structs, 0)
}

/// The name of a bare path type (one segment, no arguments, no `<T as ..>`, no leading `::`).
fn bare_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.leading_colon.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let segment = &path.path.segments[0];
    if !segment.arguments.is_empty() {
        return None;
    }
    Some(segment.ident.to_string())
}

/// The layout of a scalar, vector or matrix spelling that may be a buffer element, `None` for anything else (a struct,
/// the 3-wide types, an unknown name).
fn leaf_layout(name: &str) -> Option<Layout> {
    let scalar = |canonical: &'static str| {
        Some(Layout {
            name: canonical.to_owned(),
            size: 4,
            alignment: 4,
            kind: LayoutKind::Scalar(canonical),
        })
    };
    match name {
        "f32" | "float" => return scalar("f32"),
        "i32" | "int" => return scalar("i32"),
        "u32" | "uint" => return scalar("u32"),
        _ => {}
    }
    let math = crate::math_types::math_type(name).filter(|math| math.buffer)?;
    let width = u32::try_from(crate::math_types::width(math)).ok()?;
    let kind = if math.matrix {
        LayoutKind::Matrix {
            scalar: "f32",
            rows: width,
            cols: width,
            // Columns are `rows`-wide vectors packed one after the other: the size is `cols` column strides.
            column_stride: math.size.checked_div(width)?,
        }
    } else {
        LayoutKind::Vector {
            scalar: if name.starts_with('I') {
                "i32"
            } else if name.starts_with('U') {
                "u32"
            } else {
                "f32"
            },
            count: width,
        }
    };
    Some(Layout {
        name: name.to_owned(),
        size: math.size,
        alignment: math.alignment,
        kind,
    })
}

fn layout_at(
    ty: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<Layout> {
    if depth > MAX_NESTING {
        return None;
    }
    let name = bare_name(ty)?;
    if let Some(leaf) = leaf_layout(&name) {
        return Some(leaf);
    }
    let syn::Fields::Named(fields) = &structs.get(&name)?.fields else {
        return None;
    };
    let mut offset = 0_u32;
    let mut alignment = 1_u32;
    let mut described = Vec::new();
    for field in &fields.named {
        let layout = layout_at(&field.ty, structs, depth + 1)?;
        // No padding: a field sits at the sum of the sizes before it, and that must be where its alignment puts it.
        if layout.alignment == 0 || !offset.is_multiple_of(layout.alignment) {
            return None;
        }
        alignment = alignment.max(layout.alignment);
        let size = layout.size;
        described.push(LayoutField {
            name: crate::slang::emit_ident(field.ident.as_ref()?),
            offset,
            layout,
        });
        offset = offset.checked_add(size)?;
    }
    if described.is_empty() || !offset.is_multiple_of(alignment) {
        return None;
    }
    Some(Layout {
        name,
        size: offset,
        alignment,
        kind: LayoutKind::Struct(described),
    })
}

/// [`element_layout`] for a constant buffer. `Mat2` is no field of one (its column stride is 16 in a constant buffer
/// and 8 in Rust).
fn std140_layout_at(
    ty: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<Layout> {
    use syn::ext::IdentExt as _;

    if depth > MAX_NESTING {
        return None;
    }
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let name = path.path.get_ident()?.unraw().to_string();
    if let Some(leaf) = leaf_layout(&name) {
        return (name != "Mat2").then_some(leaf);
    }
    let item = structs.get(&name)?;
    let syn::Fields::Named(fields) = &item.fields else {
        return None;
    };
    let model = crate::uniform::std140_struct(item, structs, 0)?;
    if fields.named.is_empty() || model.offsets.len() != fields.named.len() {
        return None;
    }
    let mut described = Vec::new();
    for (index, field) in fields.named.iter().enumerate() {
        let layout = std140_layout_at(&field.ty, structs, depth + 1)?;
        // The model's numbers and the field's own agree, or the macro describes nothing.
        if model.sizes[index] != layout.size {
            return None;
        }
        described.push(LayoutField {
            name: crate::slang::emit_ident(field.ident.as_ref()?),
            offset: model.offsets[index],
            layout,
        });
    }
    Some(Layout {
        name,
        size: model.size,
        alignment: model.alignment,
        kind: LayoutKind::Struct(described),
    })
}

impl Layout {
    /// Whether a vector or a matrix is this layout or is inside it.
    fn holds_math_type(&self) -> bool {
        match &self.kind {
            LayoutKind::Scalar(_) => false,
            LayoutKind::Vector { .. } | LayoutKind::Matrix { .. } => true,
            LayoutKind::Struct(fields) => fields.iter().any(|field| field.layout.holds_math_type()),
        }
    }

    fn json(&self) -> Json {
        let mut members = vec![
            ("name", Json::string(&self.name)),
            ("size", Json::Number(self.size.into())),
            ("alignment", Json::Number(self.alignment.into())),
            ("stride", Json::Number(self.size.into())),
        ];
        match &self.kind {
            LayoutKind::Scalar(scalar) => {
                members.push(("kind", Json::string("scalar")));
                members.push(("scalar", Json::string(scalar)));
            }
            LayoutKind::Vector { scalar, count } => {
                members.push(("kind", Json::string("vector")));
                members.push(("scalar", Json::string(scalar)));
                members.push(("count", Json::Number((*count).into())));
            }
            LayoutKind::Matrix {
                scalar,
                rows,
                cols,
                column_stride,
            } => {
                members.push(("kind", Json::string("matrix")));
                members.push(("scalar", Json::string(scalar)));
                members.push(("rows", Json::Number((*rows).into())));
                members.push(("cols", Json::Number((*cols).into())));
                members.push(("column_stride", Json::Number((*column_stride).into())));
            }
            LayoutKind::Struct(fields) => {
                members.push(("kind", Json::string("struct")));
                members.push((
                    "fields",
                    Json::Array(
                        fields
                            .iter()
                            .map(|field| {
                                Json::object(vec![
                                    ("name", Json::string(&field.name)),
                                    ("offset", Json::Number(field.offset.into())),
                                    ("type", field.layout.json()),
                                ])
                            })
                            .collect(),
                    ),
                ));
            }
        }
        Json::object(members)
    }
}

// ---- the kernel record ----

/// One parameter of a kernel as the record holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ParameterRecord {
    pub name: String,
    pub rust_type: String,
    /// `invocation`, `storage`, `uniform` or `value`.
    pub kind: &'static str,
    /// `read_only`, `read_write` or `not_applicable`.
    pub access: &'static str,
    /// The binding in descriptor set 0; `None` for a parameter without a resource.
    pub binding: Option<u32>,
    /// The element layout of a storage or uniform parameter; `None` for other parameters and for an element type the
    /// macro cannot describe (a storage or uniform parameter with no layout: `cargo gust build` fills it from the Slang
    /// reflection and lists the parameter as `export_unchecked` in the manifest).
    pub layout: Option<ResourceLayout>,
}

/// What `<key>.kernel.json` (schema 1) says about one kernel.
pub(super) struct KernelRecord<'a> {
    pub key: &'a str,
    pub module: &'a str,
    pub name: &'a str,
    pub entry_point: &'a str,
    pub workgroup_size: [u32; 3],
    pub parameters: &'a [ParameterRecord],
    /// The recorded source file (see `recorded_path`); empty when unknown, in which case the line table is empty too.
    pub file: &'a str,
    pub lines: &'a [LineRecord],
    /// The name of the kernel's shape when it is not `grid` (`None` for every kernel today). A record with a shape is
    /// schema 3 and carries the `shape` key; one without keeps the schema of its kind.
    pub shape: Option<&'a str>,
}

impl KernelRecord<'_> {
    /// The record as JSON text: schema 1, keys sorted, two-space indentation, LF line ends, a final newline, and no
    /// timestamp or machine path, so the same kernel always gives the same bytes.
    ///
    /// Keys: `schema`, `key`, `module`, `name`, `entry_point`, `workgroup_size` (three numbers), `parameters` (each
    /// with `name`, `rust_type`, `kind`, `access`, `binding` (`{"group": 0, "binding": n}` or null) and `layout`: null,
    /// or `{"rules": <rules>, "element_stride": n, "element": <layout>}` with `<rules>` one of `storage_v1`,
    /// `storage_v2` (a storage element that holds a vector or a matrix) and `std140_v1` (a uniform)), and `source`
    /// (`file` and `lines`, each line `[slang_start, slang_end, rust_line]`). A `<layout>` is `name`, `size`,
    /// `alignment`, `stride`, `kind` (`scalar` with `scalar`; `vector` with `scalar` and `count`; `matrix` with
    /// `scalar`, `rows`, `cols` and `column_stride`; or `struct` with `fields`: `name`, `offset`, `type` with the
    /// nested `<layout>`). A record whose layouts are all scalars and structs of them with `storage_v1` rules is the
    /// same bytes it was before vectors and uniforms had layouts. A record whose `shape` is set (mx.k) is schema 3 and
    /// has one more key, `shape` (the shape's name); the record of a `grid` kernel has none.
    pub(super) fn json(&self) -> String {
        self.write_json(SCHEMA, true)
    }

    /// The record of a `par_for!` call (schema 2): every key of [`KernelRecord::json`] except `source`. The origin of
    /// the call sites is in their [`SiteRecord`]s, so the bytes depend on the clean Slang and the entry point only and
    /// two equal bodies write the same record.
    pub(super) fn json_without_source(&self) -> String {
        self.write_json(SITE_SCHEMA, false)
    }

    fn write_json(&self, schema: u64, with_source: bool) -> String {
        let schema = if self.shape.is_some() {
            SHAPE_SCHEMA
        } else {
            schema
        };
        let parameters = self
            .parameters
            .iter()
            .map(|parameter| {
                let layout = parameter.layout.as_ref().map_or(Json::Null, |layout| {
                    Json::object(vec![
                        ("rules", Json::string(layout.rules.name())),
                        ("element_stride", Json::Number(layout.element.size.into())),
                        ("element", layout.element.json()),
                    ])
                });
                let binding = parameter.binding.map_or(Json::Null, |binding| {
                    Json::object(vec![
                        ("group", Json::Number(0)),
                        ("binding", Json::Number(binding.into())),
                    ])
                });
                Json::object(vec![
                    ("name", Json::string(&parameter.name)),
                    ("rust_type", Json::string(&parameter.rust_type)),
                    ("kind", Json::string(parameter.kind)),
                    ("access", Json::string(parameter.access)),
                    ("binding", binding),
                    ("layout", layout),
                ])
            })
            .collect();
        let [x, y, z] = self.workgroup_size;
        let mut members = vec![
            ("schema", Json::Number(schema)),
            ("key", Json::string(self.key)),
            ("module", Json::string(self.module)),
            ("name", Json::string(self.name)),
            ("entry_point", Json::string(self.entry_point)),
            (
                "workgroup_size",
                Json::Array(vec![
                    Json::Number(x.into()),
                    Json::Number(y.into()),
                    Json::Number(z.into()),
                ]),
            ),
            ("parameters", Json::Array(parameters)),
        ];
        if let Some(shape) = self.shape {
            members.push(("shape", Json::string(shape)));
        }
        if with_source {
            members.push((
                "source",
                Json::object(vec![
                    ("file", Json::string(self.file)),
                    ("lines", lines_json(self.lines)),
                ]),
            ));
        }
        render(Json::object(members))
    }
}

/// A line table as JSON: one `[slang_start, slang_end, rust_line]` triple per record.
fn lines_json(lines: &[LineRecord]) -> Json {
    Json::Array(
        lines
            .iter()
            .map(|line| {
                Json::Array(vec![
                    Json::Number(line.slang_start.into()),
                    Json::Number(line.slang_end.into()),
                    Json::Number(line.rust_line.into()),
                ])
            })
            .collect(),
    )
}

/// A document as text: sorted keys, two-space indentation, LF line ends and a final newline.
fn render(document: Json) -> String {
    let mut text = String::new();
    document.write(&mut text, 0);
    text.push('\n');
    text
}

/// What `<key>.<site16>.site.json` (schema 2) says about one `par_for!` call: where its body is in the Rust source and
/// which Slang lines came from which Rust lines (R1.1 M5). It is written beside the kernel record, whose bytes do not
/// depend on it, so two calls with equal bodies share one record and one artifact and keep a line table each (D33).
pub(super) struct SiteRecord<'a> {
    pub key: &'a str,
    /// The recorded source file (see `recorded_path`); empty when unknown, in which case the line table is empty too.
    pub file: &'a str,
    pub lines: &'a [LineRecord],
}

impl SiteRecord<'_> {
    /// The sidecar as JSON text: schema 2, keys sorted (`file`, `key`, `lines`, `schema`), two-space indentation, LF
    /// line ends, a final newline.
    pub(super) fn json(&self) -> String {
        render(Json::object(vec![
            ("schema", Json::Number(SITE_SCHEMA)),
            ("key", Json::string(self.key)),
            ("file", Json::string(self.file)),
            ("lines", lines_json(self.lines)),
        ]))
    }
}

/// The 16 lower-case hex digits that tell the call sites of one body apart: FNV-1a 64 over the recorded file, one NUL
/// byte and the decimal first line of the call. The file is the relative one `recorded_path` gives, so the name does
/// not depend on the build machine's directory, and the line is the call's, so moving other code does not matter to a
/// call that stays where it is. Two calls on one line of one file with equal bodies share a name, and so a sidecar: their
/// bytes must be equal too (see [`write_site_records`]).
pub(super) fn site_hash(file: &str, call_line: u32) -> String {
    let line = call_line.to_string();
    let hash = [file.as_bytes(), &[0], line.as_bytes()]
        .iter()
        .flat_map(|part| part.iter())
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
    format!("{hash:016x}")
}

/// The file name of a call site's sidecar.
pub(super) fn site_file_name(key: &str, file: &str, call_line: u32) -> String {
    format!("{key}.{}.site.json", site_hash(file, call_line))
}

// ---- export ----

/// Text that belongs to a CPU twin or to the dual build (D26, D45). A record is the compiler's input, so none of it
/// may appear in one. These are the forms the items `cpu_twin_items` emits take (in source and in token-stream
/// spelling): the `cfg` that gates them, the paths into `gust::cpu`, and the `__gpu_` and `__gust_` names, which the
/// validator reserves (E0901), so no user identifier can match them.
///
/// The list holds no bare `gust_dual` or `CpuDevice`: those are valid user identifiers (a local, a helper, a struct),
/// and a record of such a kernel would be refused as "a bug in gust-macros". Every entry here is a form that no
/// identifier can spell: it has a path into `gust::cpu`, a `cfg` call, or a reserved prefix. Choice (D45 follow-up):
/// the guard is tightened rather than the names reserved, because reserving `CpuDevice` and `gust_dual` would reject
/// programs that are valid today for a leak that these forms already catch.
pub(crate) const TWIN_TEXT: &[&str] = &[
    "cfg(any(test, gust_dual))",
    "cfg (any (test , gust_dual))",
    "cfg(gust_dual)",
    "cfg (gust_dual)",
    "gust::cpu::",
    "gust :: cpu ::",
    "__gpu_typecheck_",
    "__gust_call",
    "__gust_id",
];

/// The first piece of CPU twin text in `text`, if any.
fn twin_text(text: &str) -> Option<&'static str> {
    TWIN_TEXT
        .iter()
        .copied()
        .find(|marker| text.contains(marker))
}

/// Writes `<key>.slang` (the clean text, exact bytes) and `<key>.kernel.json` into `directory`, creating it. A record
/// that holds CPU twin text is refused before anything is written. A file that already exists with other bytes is kept
/// and refused: two kernels of the package have one key (R4.4); an identical rewrite is fine. The error names the path
/// or both kernels.
pub(super) fn write_records(
    directory: &Path,
    key: &str,
    slang: &str,
    json: &str,
) -> Result<(), String> {
    write_files(directory, key, slang, json, None)
}

/// [`write_records`] for a `par_for!` call (schema 2): the same two files, whose bytes do not depend on the call site,
/// and the sidecar `site_name` holding `site_json`, which does. The sidecar goes through [`write_once`] like the rest:
/// the same call expanded again (a library and a binary target) writes the same bytes, and two calls that share a
/// name but not the bytes (two calls on one line of one file, equal bodies, different line tables) are refused with a
/// message that says so.
pub(super) fn write_site_records(
    directory: &Path,
    key: &str,
    slang: &str,
    json: &str,
    site_name: &str,
    site_json: &str,
) -> Result<(), String> {
    write_files(directory, key, slang, json, Some((site_name, site_json)))
}

fn write_files(
    directory: &Path,
    key: &str,
    slang: &str,
    json: &str,
    site: Option<(&str, &str)>,
) -> Result<(), String> {
    let site_text = site.map_or("", |(_, text)| text);
    for (what, text) in [
        ("the Slang text", slang),
        ("the kernel record", json),
        ("the site record", site_text),
    ] {
        if let Some(marker) = twin_text(text) {
            return Err(format!(
                "gust: {what} of kernel `{key}` contains CPU twin text (`{marker}`); records never hold a twin (D45). This is a bug in gust-macros"
            ));
        }
    }
    fs::create_dir_all(directory).map_err(|error| {
        format!(
            "gust: export mode could not create `{}`: {error}",
            directory.display()
        )
    })?;
    for (name, text) in [
        (format!("{key}.slang"), slang),
        (format!("{key}.kernel.json"), json),
    ] {
        let path = directory.join(name);
        let existing = write_once(&path, text.as_bytes()).map_err(|error| {
            format!(
                "gust: export mode could not write `{}`: {error}",
                path.display()
            )
        })?;
        if existing.is_some() {
            let old = fs::read_to_string(directory.join(format!("{key}.kernel.json"))).ok();
            return Err(collision_message(
                key,
                record_name(json).as_deref(),
                old.as_deref().and_then(record_name).as_deref(),
                path.extension()
                    .is_some_and(|extension| extension == "slang"),
            ));
        }
    }
    if let Some((name, text)) = site {
        let path = directory.join(name);
        let existing = write_once(&path, text.as_bytes()).map_err(|error| {
            format!(
                "gust: export mode could not write `{}`: {error}",
                path.display()
            )
        })?;
        if existing.is_some() {
            return Err(format!(
                "gust: two `par_for!` calls with equal bodies start on the same line of the same file but have different line tables, so they would share the site record `{name}`. This happens when one `macro_rules!` definition expands the call twice (the expansions share the line of the call inside the definition): change the body of one of them, or move the call out of the macro into a function that the macro calls. This is a known limitation (docs/DIAGNOSTICS.md, GUST-E0712)"
            ));
        }
    }
    Ok(())
}

/// Creates `path` with `bytes` unless it exists. `Ok(None)` when the file was created or already holds exactly
/// `bytes` (the same kernel expanded again, for example in a library and a binary target); `Ok(Some(old))` when it
/// holds other bytes, which it keeps.
///
/// Every file in a record directory is new for the run (the directory is named by a per-run nonce), so another
/// content under one key means two kernels of the package share that key (R4.4). The file appears complete or not
/// at all: it is written beside the target and hard linked into place, so a parallel expansion that finds it can
/// compare it without seeing a half-written file.
/// Where hard links are not available (FAT, exFAT) the fallback is a plain write, which is neither exclusive nor atomic.
fn write_once(path: &Path, bytes: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let file_name = path
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    let temporary = path.with_file_name(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&temporary, bytes)?;
    let linked = fs::hard_link(&temporary, path);
    // If this fails the dot-prefixed `.tmp` file stays; `cargo gust` ignores such files when it reads the records.
    let _ = fs::remove_file(&temporary);
    match linked {
        Ok(()) => Ok(None),
        Err(link_error) => match fs::read(path) {
            Ok(existing) if existing == bytes => Ok(None),
            Ok(existing) => Ok(Some(existing)),
            // The path is there but cannot be read (a directory): that is the error to report.
            Err(read_error) if link_error.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(read_error)
            }
            // Hard links are not available here and nothing is in the way: write the file directly.
            Err(_) => fs::write(path, bytes).map(|()| None),
        },
    }
}

/// `module::name` of a record's JSON text, read without a JSON parser (the writer is ours: keys sorted, two-space
/// indentation, so the top-level `module` and `name` start their lines), or `None` when it cannot be read.
fn record_name(json: &str) -> Option<String> {
    let value = |key: &str| {
        let marker = format!("\n  \"{key}\": \"");
        let start = json.find(&marker)? + marker.len();
        let length = json[start..].find('"')?;
        Some(json[start..start + length].to_owned())
    };
    Some(format!("{}::{}", value("module")?, value("name")?))
}

/// The message of the duplicate-key error ([`Code::ExportWriteFailed`]).
fn collision_message(
    key: &str,
    new: Option<&str>,
    old: Option<&str>,
    slang_differs: bool,
) -> String {
    let name = |kernel: Option<&str>| {
        kernel.map_or_else(
            || "a kernel whose name could not be read".to_owned(),
            |kernel| format!("`{kernel}`"),
        )
    };
    let (new, old) = (name(new), name(old));
    if slang_differs {
        format!(
            "gust: kernels {new} and {old} have the same artifact key {key} but different Slang text: the 64-bit FNV-1a keys collided (R4.4). Change the text of one of the kernels"
        )
    } else {
        format!(
            "gust: kernels {new} and {old} have the same entry point and Slang text, so they would share the artifact key {key} (R4.4). Rename one of the kernels"
        )
    }
}

// ---- a small JSON writer ----

/// JSON for the record. std only: the macro's dependency allow-list is fixed. Objects are sorted by key when written,
/// so the order members were added in never reaches the output.
enum Json {
    Null,
    Number(u64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(&'static str, Json)>),
}

impl Json {
    fn string(text: &str) -> Self {
        Self::String(text.to_owned())
    }

    fn object(members: Vec<(&'static str, Json)>) -> Self {
        Self::Object(members)
    }

    fn write(&self, out: &mut String, indent: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Number(number) => out.push_str(&number.to_string()),
            Self::String(text) => write_string(out, text),
            Self::Array(items) if items.is_empty() => out.push_str("[]"),
            // Numbers stay on one line: a workgroup size or one line-table entry.
            Self::Array(items) if items.iter().all(|item| matches!(item, Self::Number(_))) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    item.write(out, indent);
                }
                out.push(']');
            }
            Self::Array(items) => {
                out.push_str("[\n");
                for (index, item) in items.iter().enumerate() {
                    out.push_str(&"  ".repeat(indent + 1));
                    item.write(out, indent + 1);
                    out.push_str(if index + 1 < items.len() { ",\n" } else { "\n" });
                }
                out.push_str(&"  ".repeat(indent));
                out.push(']');
            }
            Self::Object(members) if members.is_empty() => out.push_str("{}"),
            Self::Object(members) => {
                let mut sorted: Vec<&(&'static str, Json)> = members.iter().collect();
                sorted.sort_by(|a, b| a.0.cmp(b.0));
                out.push_str("{\n");
                for (index, (key, value)) in sorted
                    .iter()
                    .map(|member| (member.0, &member.1))
                    .enumerate()
                {
                    out.push_str(&"  ".repeat(indent + 1));
                    write_string(out, key);
                    out.push_str(": ");
                    value.write(out, indent + 1);
                    out.push_str(if index + 1 < sorted.len() {
                        ",\n"
                    } else {
                        "\n"
                    });
                }
                out.push_str(&"  ".repeat(indent));
                out.push('}');
            }
        }
    }
}

/// A JSON string literal: `"`, `` and control characters escaped, everything else as UTF-8.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A scratch directory under the system temp directory whose name contains a space (the case that broke path
    /// handling before), removed on drop.
    pub(in crate::expand) struct Scratch(pub PathBuf);

    impl Scratch {
        pub(in crate::expand) fn new(name: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("gust m4c {} {serial} {name}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    impl Scratch {
        /// Writes `contents` to `gust-artifacts/<name>` of the scratch crate.
        pub(in crate::expand) fn artifact(&self, name: &str, contents: &str) {
            let directory = self.0.join(DIRECTORY);
            fs::create_dir_all(&directory).unwrap();
            fs::write(directory.join(name), contents).unwrap();
        }
    }

    /// The same pairs, with the same expected strings, as `gust::artifact::tests`. The expected values were computed
    /// independently of both implementations (a node script over the UTF-8 bytes), so they pin the definition: FNV-1a 64
    /// over the entry point, one NUL byte and the text, with no line-ending normalisation.
    pub(in crate::expand) const GOLDEN_KEYS: &[(&str, &str, &str)] = &[
        ("foo", "bar", "8a5f40d8243888aa"),
        ("", "", "af63bd4c8601b7df"),
        (
            "gpu_vector_add_add",
            "// Generated by GUST from Rust syntax. Slang is the compiler backend.\n\n[shader(\"compute\")]\nvoid gpu_vector_add_add() {}\n",
            "6cdc27171dd8a55e",
        ),
        // CRLF stays CRLF: rustc turns CRLF in a literal into LF before the macro sees it, so the key hashes what it
        // is given.
        ("gpu_demo_run", "uint a;\r\nuint b;\r\n", "0513779dad82ba6c"),
        (
            "gpu_\u{fc}n\u{ef}_\u{e9}",
            "// \u{fc}n\u{ef} \u{65e5}\u{672c} \u{1f600}\n",
            "f3edb671848d99cc",
        ),
        // One body under two shapes (mx.k, D89): the same entry point and body line, the stage attribute and the
        // signature differ, and so do the keys. The formula has no shape in it.
        (
            "gpu_demo_run",
            "[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid gpu_demo_run(uint3 id : SV_DispatchThreadID)\n{\n    uint i = id.x;\n}\n",
            "4be1a569519007e0",
        ),
        (
            "gpu_demo_run",
            "[shader(\"fragment\")]\nOut gpu_demo_run(float4 pos : SV_Position)\n{\n    uint i = id.x;\n}\n",
            "76d7388ee1e8f383",
        ),
    ];

    #[test]
    fn the_key_matches_the_golden_values() {
        for (entry_point, text, expected) in GOLDEN_KEYS {
            assert_eq!(&content_key(entry_point, text), expected, "{entry_point:?}");
        }
    }

    #[test]
    fn one_body_under_two_shapes_gives_two_keys() {
        let [
            ..,
            (grid_entry, grid, grid_key),
            (pixel_entry, pixel, pixel_key),
        ] = GOLDEN_KEYS
        else {
            panic!("the two-shapes pair is the end of GOLDEN_KEYS");
        };
        assert_eq!(grid_entry, pixel_entry);
        assert!(grid.contains("[shader(\"compute\")]") && pixel.contains("[shader(\"fragment\")]"));
        assert_ne!(grid_key, pixel_key);
        assert_ne!(
            content_key(grid_entry, grid),
            content_key(pixel_entry, pixel)
        );
    }

    #[test]
    fn the_key_separates_the_entry_point_from_the_text_with_a_nul() {
        assert_ne!(content_key("ab", "c"), content_key("a", "bc"));
        assert_ne!(content_key("foo", "bar"), content_key("foobar", ""));
    }

    #[test]
    fn lookup_distinguishes_absent_present_and_a_half_pair() {
        let scratch = Scratch::new("lookup");
        let key = "0123456789abcdef";
        let dir = Some(scratch.0.as_path());
        assert_eq!(lookup(dir, key), Lookup::Absent);
        assert_eq!(lookup(None, key), Lookup::Absent);
        scratch.artifact(&wgsl_name(key), "wgsl");
        assert_eq!(
            lookup(dir, key),
            Lookup::Incomplete {
                left: "gust-artifacts/0123456789abcdef.wgsl".to_owned(),
                missing: "gust-artifacts/0123456789abcdef.wgsl.reflection.json".to_owned()
            }
        );
        scratch.artifact(&reflection_name(key), "{}");
        assert_eq!(lookup(dir, key), Lookup::Present);
        // Another key is unaffected, and a directory of the right name is not a file.
        assert_eq!(lookup(dir, "ffffffffffffffff"), Lookup::Absent);
        fs::create_dir_all(scratch.0.join(DIRECTORY).join("aaaaaaaaaaaaaaaa.wgsl")).unwrap();
        scratch.artifact("aaaaaaaaaaaaaaaa.wgsl.reflection.json", "{}");
        assert_eq!(
            lookup(dir, "aaaaaaaaaaaaaaaa"),
            Lookup::Incomplete {
                left: "gust-artifacts/aaaaaaaaaaaaaaaa.wgsl.reflection.json".to_owned(),
                missing: "gust-artifacts/aaaaaaaaaaaaaaaa.wgsl".to_owned()
            }
        );
        let other = Scratch::new("lookup-only-wgsl-half");
        other.artifact(&reflection_name(key), "{}");
        assert_eq!(
            lookup(Some(other.0.as_path()), key),
            Lookup::Incomplete {
                left: "gust-artifacts/0123456789abcdef.wgsl.reflection.json".to_owned(),
                missing: "gust-artifacts/0123456789abcdef.wgsl".to_owned()
            }
        );
    }

    #[test]
    fn the_manifest_is_present_only_as_a_file() {
        let scratch = Scratch::new("manifest");
        assert!(!manifest_present(Some(scratch.0.as_path())));
        assert!(!manifest_present(None));
        fs::create_dir_all(scratch.0.join(DIRECTORY).join(MANIFEST)).unwrap();
        assert!(!manifest_present(Some(scratch.0.as_path())));
        fs::remove_dir_all(scratch.0.join(DIRECTORY).join(MANIFEST)).unwrap();
        scratch.artifact(MANIFEST, "{}");
        assert!(manifest_present(Some(scratch.0.as_path())));
    }

    #[test]
    fn a_present_pair_embeds_with_crate_relative_paths() {
        let tokens = artifact_tokens("52e85c5f03f72272", &Lookup::Present).to_string();
        assert_eq!(
            tokens,
            ":: gust :: Artifact { format : :: gust :: Format :: Wgsl , key : \"52e85c5f03f72272\" , bytes : :: core :: include_bytes ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/gust-artifacts/52e85c5f03f72272.wgsl\")) , reflection : :: core :: include_str ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/gust-artifacts/52e85c5f03f72272.wgsl.reflection.json\")) , }]"
                .replacen(":: gust :: Artifact", "& [:: gust :: Artifact", 1)
        );
    }

    #[test]
    fn absent_and_half_pairs_embed_nothing() {
        for lookup in [
            Lookup::Absent,
            Lookup::Incomplete {
                left: "gust-artifacts/y".to_owned(),
                missing: "gust-artifacts/x".to_owned(),
            },
        ] {
            assert_eq!(
                artifact_tokens("52e85c5f03f72272", &lookup).to_string(),
                "& []"
            );
        }
    }

    /// The emitted paths hold no directory of the build machine, so a `CARGO_MANIFEST_DIR` with backslashes and a
    /// space is never escaped (or escaped twice) by the macro: rustc's own `env!` supplies it to `concat!`.
    #[test]
    fn the_emitted_paths_never_contain_the_manifest_directory() {
        let scratch = Scratch::new("windows path");
        let directory = scratch.0.display().to_string();
        assert!(directory.contains(' '), "{directory}");
        let key = "52e85c5f03f72272";
        scratch.artifact(&wgsl_name(key), "wgsl");
        scratch.artifact(&reflection_name(key), "{}");
        scratch.artifact(MANIFEST, "{}");
        assert_eq!(lookup(Some(scratch.0.as_path()), key), Lookup::Present);
        let tokens = format!(
            "{} {}",
            artifact_tokens(key, &Lookup::Present),
            manifest_tokens(manifest_present(Some(scratch.0.as_path())))
        );
        assert!(!tokens.contains(&directory), "{tokens}");
        assert!(
            !tokens.contains(&directory.replace('\\', "\\\\")),
            "{tokens}"
        );
        assert!(!tokens.contains('\\'), "{tokens}");
        // Every literal the paths are built from is a plain tail starting with `/`.
        for tail in [
            "\"/gust-artifacts/52e85c5f03f72272.wgsl\"",
            "\"/gust-artifacts/52e85c5f03f72272.wgsl.reflection.json\"",
            "\"/gust-artifacts/manifest.json\"",
        ] {
            assert_eq!(tokens.matches(tail).count(), 1, "{tail} in {tokens}");
        }
        assert_eq!(tokens.matches("env ! (\"CARGO_MANIFEST_DIR\")").count(), 3);
    }

    #[test]
    fn the_manifest_include_exists_only_with_the_manifest() {
        assert_eq!(manifest_tokens(false).to_string(), "");
        assert_eq!(
            manifest_tokens(true).to_string(),
            "const _ : & [:: core :: primitive :: u8] = :: core :: include_bytes ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/gust-artifacts/manifest.json\")) ;"
        );
    }

    #[test]
    fn the_record_directory_is_named_by_the_package_and_only_in_export_mode() {
        let environment = |export: Option<&str>, package: Option<&str>| Environment {
            manifest_dir: None,
            export_dir: export.map(PathBuf::from),
            package: package.map(str::to_owned),
            reported_file: None,
        };
        assert_eq!(environment(None, Some("demo")).record_directory(), Ok(None));
        assert_eq!(
            environment(Some("records"), Some("demo-kernels_2")).record_directory(),
            Ok(Some(Path::new("records").join("demo-kernels_2")))
        );
        for bad in [
            None,
            Some(""),
            Some(".."),
            Some("a/b"),
            Some("a\\b"),
            Some("a b"),
        ] {
            assert!(
                environment(Some("records"), bad)
                    .record_directory()
                    .is_err(),
                "{bad:?}"
            );
        }
    }

    fn structs_of(source: &str) -> Vec<syn::ItemStruct> {
        let file: syn::File = crate::diagnostics::testing::parse_str(source).unwrap();
        file.items
            .into_iter()
            .filter_map(|item| match item {
                syn::Item::Struct(item) => Some(item),
                _ => None,
            })
            .collect()
    }

    fn layout_of(source: &str, element: &str) -> Option<Layout> {
        let items = structs_of(source);
        let structs: HashMap<String, &syn::ItemStruct> = items
            .iter()
            .map(|item| (item.ident.to_string(), item))
            .collect();
        let ty: syn::Type = crate::diagnostics::testing::parse_str(element).unwrap();
        element_layout(&ty, &structs)
    }

    #[test]
    fn scalars_and_their_aliases_have_the_canonical_layout() {
        for (written, canonical) in [
            ("f32", "f32"),
            ("float", "f32"),
            ("i32", "i32"),
            ("int", "i32"),
            ("u32", "u32"),
            ("uint", "u32"),
        ] {
            let layout = layout_of("", written).unwrap();
            assert_eq!(layout.name, canonical);
            assert_eq!((layout.size, layout.alignment), (4, 4));
            assert_eq!(layout.kind, LayoutKind::Scalar(canonical));
        }
    }

    fn storage_of(source: &str, element: &str) -> Option<ResourceLayout> {
        let items = structs_of(source);
        let structs: HashMap<String, &syn::ItemStruct> = items
            .iter()
            .map(|item| (item.ident.to_string(), item))
            .collect();
        let ty: syn::Type = crate::diagnostics::testing::parse_str(element).unwrap();
        storage_resource(&ty, &structs)
    }

    fn uniform_of(source: &str, element: &str) -> Option<ResourceLayout> {
        let items = structs_of(source);
        let structs: HashMap<String, &syn::ItemStruct> = items
            .iter()
            .map(|item| (item.ident.to_string(), item))
            .collect();
        let ty: syn::Type = crate::diagnostics::testing::parse_str(element).unwrap();
        uniform_resource(&ty, &structs)
    }

    /// `(name, size, alignment, offset)` of the fields of a struct layout.
    fn field_shape(layout: &Layout) -> Vec<(&str, u32, u32, u32)> {
        let LayoutKind::Struct(fields) = &layout.kind else {
            panic!("{layout:?}");
        };
        fields
            .iter()
            .map(|field| {
                (
                    field.name.as_str(),
                    field.layout.size,
                    field.layout.alignment,
                    field.offset,
                )
            })
            .collect()
    }

    #[test]
    fn vectors_and_matrices_have_the_layout_of_gust_math() {
        for (written, size, alignment, kind) in [
            (
                "Vec2",
                8,
                8,
                LayoutKind::Vector {
                    scalar: "f32",
                    count: 2,
                },
            ),
            (
                "Vec4",
                16,
                16,
                LayoutKind::Vector {
                    scalar: "f32",
                    count: 4,
                },
            ),
            (
                "IVec2",
                8,
                8,
                LayoutKind::Vector {
                    scalar: "i32",
                    count: 2,
                },
            ),
            (
                "IVec4",
                16,
                16,
                LayoutKind::Vector {
                    scalar: "i32",
                    count: 4,
                },
            ),
            (
                "UVec2",
                8,
                8,
                LayoutKind::Vector {
                    scalar: "u32",
                    count: 2,
                },
            ),
            (
                "UVec4",
                16,
                16,
                LayoutKind::Vector {
                    scalar: "u32",
                    count: 4,
                },
            ),
            (
                "Mat2",
                16,
                8,
                LayoutKind::Matrix {
                    scalar: "f32",
                    rows: 2,
                    cols: 2,
                    column_stride: 8,
                },
            ),
            (
                "Mat4",
                64,
                16,
                LayoutKind::Matrix {
                    scalar: "f32",
                    rows: 4,
                    cols: 4,
                    column_stride: 16,
                },
            ),
        ] {
            let layout = layout_of("", written).unwrap();
            assert_eq!(layout.name, written);
            assert_eq!(
                (layout.size, layout.alignment),
                (size, alignment),
                "{written}"
            );
            assert_eq!(layout.kind, kind, "{written}");
        }
        // The 3-wide types are not buffer elements, and a vector spelled with arguments is not a layout.
        for element in ["Vec3", "UVec3", "Mat3", "Vec4<f32>", "gust::Vec4", "bool"] {
            assert_eq!(layout_of("", element), None, "{element}");
        }
    }

    #[test]
    fn a_storage_element_with_a_vector_has_running_offsets_the_largest_alignment_and_storage_v2_rules()
     {
        let source = "struct P { a: Vec4, b: f32, c: u32, d: Vec2 } struct Q { m: Mat4, p: P }";
        let layout = layout_of(source, "P").unwrap();
        assert_eq!((layout.size, layout.alignment), (32, 16));
        assert_eq!(
            field_shape(&layout),
            [
                ("a", 16, 16, 0),
                ("b", 4, 4, 16),
                ("c", 4, 4, 20),
                ("d", 8, 8, 24)
            ]
        );
        let nested = layout_of(source, "Q").unwrap();
        assert_eq!((nested.size, nested.alignment), (96, 16));
        assert_eq!(field_shape(&nested), [("m", 64, 16, 0), ("p", 32, 16, 64)]);
        // The rules follow the shape: a vector anywhere in the element makes it StorageV2.
        assert_eq!(storage_of(source, "P").unwrap().rules, Rules::StorageV2);
        assert_eq!(storage_of(source, "Q").unwrap().rules, Rules::StorageV2);
        assert_eq!(storage_of("", "Vec4").unwrap().rules, Rules::StorageV2);
        assert_eq!(storage_of("", "Mat2").unwrap().rules, Rules::StorageV2);
        assert_eq!(storage_of("", "f32").unwrap().rules, Rules::StorageV1);
        assert_eq!(
            storage_of("struct S { a: u32, b: f32 }", "S")
                .unwrap()
                .rules,
            Rules::StorageV1
        );
    }

    #[test]
    fn a_storage_struct_that_would_need_padding_is_not_described() {
        // The expansion's assertions reject all three at compile time; the record says nothing about them.
        let source = "struct Gap { a: f32, v: Vec2 } struct Tail { v: Vec4, a: f32 } struct Inner { v: Vec2, a: f32 } \
                      struct Outer { a: f32, i: Inner }";
        for element in ["Gap", "Tail", "Inner", "Outer"] {
            assert_eq!(layout_of(source, element), None, "{element}");
        }
    }

    /// The `uniforms` fixture's structs (`tests/fixtures/uniforms.rs`), whose numbers the constant-buffer assertions of
    /// the expansion check.
    const UNIFORM_STRUCTS: &str = "struct Light { color: Vec4, intensity: f32, falloff: f32, pad0: f32, pad1: f32 } \
         struct Frame { view: Mat4, tint: Vec4, light: Light, offset: Vec2, scale: f32, count: u32 } \
         struct Gain { gain: f32, pad0: f32, pad1: f32, pad2: f32 }";

    #[test]
    fn a_uniform_has_the_std140_layout_of_the_macros_own_model() {
        let resource = uniform_of(UNIFORM_STRUCTS, "Frame").unwrap();
        assert_eq!(resource.rules, Rules::Std140V1);
        let frame = &resource.element;
        assert_eq!(
            (frame.name.as_str(), frame.size, frame.alignment),
            ("Frame", 128, 16)
        );
        assert_eq!(
            field_shape(frame),
            [
                ("view", 64, 16, 0),
                ("tint", 16, 16, 64),
                ("light", 32, 16, 80),
                ("offset", 8, 8, 112),
                ("scale", 4, 4, 120),
                ("count", 4, 4, 124),
            ]
        );
        let LayoutKind::Struct(fields) = &frame.kind else {
            panic!("{frame:?}");
        };
        assert_eq!(
            field_shape(&fields[2].layout),
            [
                ("color", 16, 16, 0),
                ("intensity", 4, 4, 16),
                ("falloff", 4, 4, 20),
                ("pad0", 4, 4, 24),
                ("pad1", 4, 4, 28),
            ]
        );
        let gain = uniform_of(UNIFORM_STRUCTS, "Gain").unwrap().element;
        assert_eq!((gain.size, gain.alignment), (16, 16));
        // The same numbers as the model the const assertions use, field by field.
        let items = structs_of(UNIFORM_STRUCTS);
        let by_name: HashMap<String, &syn::ItemStruct> = items
            .iter()
            .map(|item| (item.ident.to_string(), item))
            .collect();
        let model = crate::uniform::std140_struct(by_name["Frame"], &by_name, 0).unwrap();
        assert_eq!((model.size, model.alignment), (frame.size, frame.alignment));
        let offsets: Vec<u32> = field_shape(frame).iter().map(|field| field.3).collect();
        assert_eq!(offsets, model.offsets);
    }

    #[test]
    fn what_a_constant_buffer_cannot_hold_has_no_uniform_layout() {
        let source = "struct M { m: Mat2, a: f32 } struct B { a: bool } struct E {} struct T(u32) ; \
                      struct V3 { v: Vec3 } struct Fine { a: f32, b: f32, c: f32, d: f32 }";
        for element in [
            "M", "B", "E", "T", "V3", "Unknown", "f32", "Vec4", "Mat4", "Vec<u32>",
        ] {
            assert!(uniform_of(source, element).is_none(), "{element}");
        }
        assert!(uniform_of(source, "Fine").is_some());
    }

    /// The record's field names are the emitted ones (M6 plan 2.5), so they match what slangc reflects.
    #[test]
    fn layout_field_names_are_the_emitted_names() {
        let source = "struct S { r#new: float, plain: float, r#match: uint }";
        let layout = layout_of(source, "S").unwrap();
        let LayoutKind::Struct(fields) = &layout.kind else {
            panic!("{layout:?}");
        };
        let names: Vec<_> = fields.iter().map(|field| field.name.as_str()).collect();
        assert_eq!(names, ["__gust_r_new", "plain", "match"]);
    }

    #[test]
    fn structs_are_padding_free_with_running_offsets_and_nest() {
        let source = "struct Pair { a: uint, b: float } struct Outer { p: Pair, w: int, q: Pair }";
        let layout = layout_of(source, "Outer").unwrap();
        assert_eq!((layout.name.as_str(), layout.size), ("Outer", 20));
        let LayoutKind::Struct(fields) = &layout.kind else {
            panic!("{layout:?}");
        };
        let shape: Vec<(&str, u32, u32)> = fields
            .iter()
            .map(|field| (field.name.as_str(), field.offset, field.layout.size))
            .collect();
        assert_eq!(shape, [("p", 0, 8), ("w", 8, 4), ("q", 12, 8)]);
    }

    #[test]
    fn types_the_macro_cannot_describe_have_no_layout() {
        let source = "struct Bad { a: uint, b: [f32; 4] } struct Loop { a: Loop } struct Gen { a: Vec<u32> } struct Empty {} struct Tuple(u32);";
        for element in [
            "Bad",
            "Loop",
            "Gen",
            "Empty",
            "Tuple",
            "Unknown",
            "super::Outside",
            "::Outside",
            "Vec<u32>",
            "bool",
            "&u32",
        ] {
            assert_eq!(layout_of(source, element), None, "{element}");
        }
    }

    fn sample_parameters() -> Vec<ParameterRecord> {
        let pair = storage_of("struct Pair { a: uint, b: float }", "Pair").unwrap();
        vec![
            ParameterRecord {
                name: "id".to_owned(),
                rust_type: "SV_DispatchThreadID".to_owned(),
                kind: "invocation",
                access: "not_applicable",
                binding: None,
                layout: None,
            },
            ParameterRecord {
                name: "pairs".to_owned(),
                rust_type: "StructuredBuffer < Pair >".to_owned(),
                kind: "storage",
                access: "read_only",
                binding: Some(0),
                layout: Some(pair),
            },
            ParameterRecord {
                name: "out".to_owned(),
                rust_type: "RWStructuredBuffer < float >".to_owned(),
                kind: "storage",
                access: "read_write",
                binding: Some(1),
                layout: storage_of("", "float"),
            },
        ]
    }

    /// The parameters of the `uniforms` fixture's `run` (`tests/fixtures/uniforms.rs`): a `Uniform<Frame>` and two
    /// `Vec4` buffers.
    fn uniform_parameters() -> Vec<ParameterRecord> {
        let parameter =
            |name: &str, rust_type: &str, kind, access, binding, layout| ParameterRecord {
                name: name.to_owned(),
                rust_type: rust_type.to_owned(),
                kind,
                access,
                binding,
                layout,
            };
        vec![
            parameter(
                "id",
                "SV_DispatchThreadID",
                "invocation",
                "not_applicable",
                None,
                None,
            ),
            parameter(
                "frame",
                "Uniform < Frame >",
                "uniform",
                "read_only",
                Some(0),
                uniform_of(UNIFORM_STRUCTS, "Frame"),
            ),
            parameter(
                "points",
                "StructuredBuffer < Vec4 >",
                "storage",
                "read_only",
                Some(1),
                storage_of("", "Vec4"),
            ),
            parameter(
                "out",
                "RWStructuredBuffer < Vec4 >",
                "storage",
                "read_write",
                Some(2),
                storage_of("", "Vec4"),
            ),
        ]
    }

    /// The uniform kernel's record as `write` prints it.
    fn uniform_json() -> String {
        let parameters = uniform_parameters();
        KernelRecord {
            key: "0123456789abcdef",
            module: "uniforms",
            name: "run",
            entry_point: "gpu_uniforms_run",
            workgroup_size: [64, 1, 1],
            parameters: &parameters,
            file: "src/lib.rs",
            lines: &[LineRecord {
                slang_start: 3,
                slang_end: 9,
                rust_line: 12,
            }],
            shape: None,
        }
        .json()
    }

    #[test]
    fn the_uniform_and_vector_record_matches_the_golden() {
        assert_golden(
            &uniform_json(),
            "uniform.kernel.json",
            &include_str!("../../tests/golden/uniform.kernel.json").replace("\r\n", "\n"),
        );
    }

    /// The rules of each parameter as the JSON states them, so that a uniform written with storage rules (or the other
    /// way round) cannot pass on the strength of a golden that was blessed with the mistake.
    #[test]
    fn the_rules_in_the_record_follow_the_parameter_kind_and_the_element() {
        let json = uniform_json();
        let rules: Vec<&str> = json
            .lines()
            .filter_map(|line| line.trim().strip_prefix("\"rules\": \""))
            .map(|rest| rest.trim_end_matches(['"', ',']))
            .collect();
        // Parameters in order: the uniform, then the two `Vec4` buffers; JSON keys are sorted but parameters are not.
        assert_eq!(rules, ["std140_v1", "storage_v2", "storage_v2"], "{json}");
        let sample = sample_json();
        assert_eq!(sample.matches("\"rules\": \"storage_v1\"").count(), 2);
        assert!(!sample.contains("std140") && !sample.contains("storage_v2"));
    }

    pub(in crate::expand) fn sample_json() -> String {
        sample_written_by(|record| record.json())
    }

    /// The sample kernel as `write` prints it.
    fn sample_written_by(write: impl Fn(&KernelRecord<'_>) -> String) -> String {
        let parameters = sample_parameters();
        let record = KernelRecord {
            key: "0123456789abcdef",
            module: "demo",
            name: "run",
            entry_point: "gpu_demo_run",
            workgroup_size: [64, 1, 1],
            parameters: &parameters,
            file: "src/lib.rs",
            lines: &[
                LineRecord {
                    slang_start: 3,
                    slang_end: 9,
                    rust_line: 12,
                },
                LineRecord {
                    slang_start: 5,
                    slang_end: 5,
                    rust_line: 14,
                },
            ],
            shape: None,
        };
        write(&record)
    }

    /// The sample record with the shape `shape` given (the macro writes none today, so only a test makes one).
    fn sample_with_shape(shape: &str, write: impl Fn(&KernelRecord<'_>) -> String) -> String {
        sample_written_by(|record| {
            write(&KernelRecord {
                shape: Some(shape),
                ..*record
            })
        })
    }

    #[test]
    fn the_kernel_record_matches_the_golden() {
        assert_golden(
            &sample_json(),
            "sample.kernel.json",
            &include_str!("../../tests/golden/sample.kernel.json").replace("\r\n", "\n"),
        );
    }

    #[test]
    fn the_par_for_record_matches_the_golden() {
        let actual = sample_written_by(|record| record.json_without_source());
        assert_golden(
            &actual,
            "sample.par_for.kernel.json",
            &include_str!("../../tests/golden/sample.par_for.kernel.json").replace("\r\n", "\n"),
        );
        // It is the schema 1 record without its `source`, with the schema raised: nothing else differs.
        let schema_1 = sample_json();
        let source_start = schema_1.find("  \"source\": {").expect("a source block");
        let source_end = source_start
            + schema_1[source_start..].find("  },\n").expect("its end")
            + "  },\n".len();
        let derived = format!("{}{}", &schema_1[..source_start], &schema_1[source_end..])
            .replace("\"schema\": 1", "\"schema\": 2");
        assert_eq!(actual, derived);
    }

    /// Schema 3 (mx.k, D89) is written only for a kernel whose shape is not `grid`: schema 1 or 2's grammar plus a
    /// `shape` key, between `schema` and `source` in key order. No kernel the macro expands today has one, so the
    /// record is made by hand here.
    #[test]
    fn a_record_with_a_shape_is_schema_3_and_a_grid_record_is_unchanged() {
        let schema_1 = sample_json();
        let schema_2 = sample_written_by(|record| record.json_without_source());
        assert!(!schema_1.contains("shape") && !schema_2.contains("shape"));
        // A `#[gpu]` kernel of another shape keeps its `source`.
        let module = sample_with_shape("pixel", |record| record.json());
        assert_eq!(
            module,
            schema_1.replace(
                "  \"schema\": 1,\n",
                "  \"schema\": 3,\n  \"shape\": \"pixel\",\n"
            )
        );
        // A `par_for!` kernel of another shape has none, as in schema 2.
        let site = sample_with_shape("pixel", |record| record.json_without_source());
        assert_eq!(
            site,
            schema_2.replace(
                "  \"schema\": 2,\n",
                "  \"schema\": 3,\n  \"shape\": \"pixel\",\n"
            )
        );
        assert!(!site.contains("\"source\""));
    }

    #[test]
    fn the_call_site_record_matches_the_golden() {
        let lines = [
            LineRecord {
                slang_start: 3,
                slang_end: 9,
                rust_line: 12,
            },
            LineRecord {
                slang_start: 5,
                slang_end: 5,
                rust_line: 14,
            },
        ];
        let actual = SiteRecord {
            key: "0123456789abcdef",
            file: "src/lib.rs",
            lines: &lines,
        }
        .json();
        assert_golden(
            &actual,
            "sample.site.json",
            &include_str!("../../tests/golden/sample.site.json").replace("\r\n", "\n"),
        );
    }

    #[test]
    fn the_site_names_in_the_readers_test_data_are_the_names_the_writer_makes() {
        // `cargo-gust` reads these names from its own test data; if the hash or the name format changes here, the
        // reader's data must change with it. The file is read as text because the crates share no code.
        let reader = include_str!("../../../cargo-gust/src/testdata.rs");
        for (file, line) in [("src/lib.rs", 12), ("tests/other.rs", 40)] {
            let name = site_file_name("022c52d675787198", file, line);
            assert!(
                reader.contains(&format!("\"{name}\"")),
                "{name} ({file}:{line}) is not in testdata.rs"
            );
        }
        // FNV-1a 64 over the file, a zero byte and the decimal line, written out once by hand for a short input.
        let by_hand = [b"a".as_slice(), &[0], b"1"]
            .concat()
            .iter()
            .fold(0xcbf29ce484222325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
            });
        assert_eq!(site_hash("a", 1), format!("{by_hand:016x}"));
        assert_ne!(site_hash("a", 1), site_hash("a", 2));
        assert_ne!(site_hash("a", 1), site_hash("b", 1));
    }

    /// Set to `1` (as `cargo xtask bless` does for tests named `golden`) to rewrite a golden of
    /// `crates/gust-macros/tests/golden` with the actual text. A bless still fails the test, so it cannot pass
    /// silently; review the diff, then rerun without the variable.
    const BLESS_VARIABLE: &str = "BLESS_GOLDEN";

    /// Compares `actual` with the reviewed golden `tests/golden/<name>` (`expected`, already with `\n` ends).
    pub(in crate::expand) fn assert_golden(actual: &str, name: &str, expected: &str) {
        if actual == expected {
            return;
        }
        if std::env::var_os(BLESS_VARIABLE).is_some_and(|value| value == "1") {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/golden")
                .join(name);
            // Keep the line endings the working tree already uses, so a bless diffs only real changes.
            let crlf = fs::read_to_string(&path).is_ok_and(|old| old.contains("\r\n"));
            let text = if crlf {
                actual.replace('\n', "\r\n")
            } else {
                actual.to_owned()
            };
            fs::write(&path, text)
                .unwrap_or_else(|error| panic!("could not write {name}: {error}"));
            panic!("golden blessed; review the diff and rerun without BLESS_GOLDEN ({name})");
        }
        assert_eq!(actual, expected, "{name}");
    }

    #[test]
    fn the_record_is_byte_identical_on_every_run_and_has_sorted_keys() {
        let first = sample_json();
        assert_eq!(first, sample_json());
        assert!(first.ends_with("}\n"));
        assert!(!first.contains('\r'));
        // At every nesting level the keys of an object are in order: check the top-level ones by position.
        let positions: Vec<usize> = [
            "\"entry_point\"",
            "\"key\"",
            "\"module\"",
            "\"name\": \"run\"",
            "\"parameters\"",
            "\"schema\"",
            "\"source\"",
            "\"workgroup_size\"",
        ]
        .iter()
        .map(|key| {
            first
                .find(key)
                .unwrap_or_else(|| panic!("{key} in {first}"))
        })
        .collect();
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "{first}"
        );
    }

    #[test]
    fn json_strings_escape_quotes_backslashes_and_control_characters_only() {
        let mut out = String::new();
        write_string(&mut out, "a\"b\\c\nd\re\tf\u{1}g\u{fc}\u{1f600}");
        assert_eq!(out, "\"a\\\"b\\\\c\\nd\\re\\tf\\u0001g\u{fc}\u{1f600}\"");
    }

    #[test]
    fn records_are_written_with_the_clean_text_as_exact_bytes() {
        let scratch = Scratch::new("write");
        let directory = scratch.0.join("records").join("demo");
        let slang = "uint a;\r\nuint b;\r\n\u{fc}";
        write_records(&directory, "0123456789abcdef", slang, &sample_json()).unwrap();
        assert_eq!(
            fs::read(directory.join("0123456789abcdef.slang")).unwrap(),
            slang.as_bytes()
        );
        assert_eq!(
            fs::read_to_string(directory.join("0123456789abcdef.kernel.json")).unwrap(),
            sample_json()
        );
        // A second run replaces the files with identical bytes.
        write_records(&directory, "0123456789abcdef", slang, &sample_json()).unwrap();
        assert_eq!(
            fs::read_to_string(directory.join("0123456789abcdef.kernel.json")).unwrap(),
            sample_json()
        );
    }

    #[test]
    fn an_existing_record_with_other_bytes_is_kept_and_refused_naming_both_kernels() {
        let scratch = Scratch::new("collision");
        let directory = scratch.0.join("records").join("demo");
        let key = "0123456789abcdef";
        let first = sample_json();
        write_records(&directory, key, "uint a;\n", &first).unwrap();
        // The same key with other Slang text: the keys collided.
        let second = first.replace("\"name\": \"run\"", "\"name\": \"walk\"");
        assert_ne!(first, second);
        let error = write_records(&directory, key, "uint b;\n", &second).unwrap_err();
        assert!(
            error.contains("`demo::walk` and `demo::run`")
                && error.contains(key)
                && error.contains("different Slang text"),
            "{error}"
        );
        // The same entry point and text under another name: they would share the files.
        let error = write_records(&directory, key, "uint a;\n", &second).unwrap_err();
        assert!(
            error.contains("`demo::walk` and `demo::run`")
                && error.contains("same entry point and Slang text"),
            "{error}"
        );
        // The first kernel's files are untouched by both refusals, and no temporary file is left.
        assert_eq!(
            fs::read(directory.join(format!("{key}.slang"))).unwrap(),
            b"uint a;\n"
        );
        assert_eq!(
            fs::read_to_string(directory.join(format!("{key}.kernel.json"))).unwrap(),
            first
        );
        let mut names: Vec<String> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [format!("{key}.kernel.json"), format!("{key}.slang")]
        );
        // Writing the first kernel again is fine, and so is a record whose names cannot be read.
        write_records(&directory, key, "uint a;\n", &first).unwrap();
        let error = write_records(&directory, key, "uint c;\n", "{}").unwrap_err();
        assert!(
            error.contains("a kernel whose name could not be read")
                && error.contains("`demo::run`"),
            "{error}"
        );
    }

    #[test]
    fn two_threads_writing_one_record_never_see_a_half_written_file() {
        let scratch = Scratch::new("parallel");
        let directory = scratch.0.join("records");
        let text = "uint x;\n".repeat(20_000);
        let json = sample_json();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..20 {
                        write_records(&directory, "0123456789abcdef", &text, &json).unwrap();
                    }
                });
            }
        });
    }

    #[test]
    fn a_write_failure_names_the_path() {
        let scratch = Scratch::new("write-failure");
        // A file where the directory should be.
        let blocker = scratch.0.join("blocked");
        fs::write(&blocker, "x").unwrap();
        let error =
            write_records(&blocker.join("demo"), "0123456789abcdef", "s", "{}").unwrap_err();
        assert!(error.contains("blocked"), "{error}");
        assert!(error.contains("could not create"), "{error}");
        // A directory where a record file should be.
        let directory = scratch.0.join("records");
        fs::create_dir_all(directory.join("0123456789abcdef.slang")).unwrap();
        let error = write_records(&directory, "0123456789abcdef", "s", "{}").unwrap_err();
        assert!(error.contains("0123456789abcdef.slang"), "{error}");
        assert!(error.contains("could not write"), "{error}");
    }

    #[test]
    fn a_record_with_cpu_twin_text_is_refused_before_anything_is_written() {
        let scratch = Scratch::new("twin");
        let directory = scratch.0.join("records");
        for poisoned in TWIN_TEXT {
            let text = format!("// {poisoned}\n");
            let error = write_records(&directory, "0123456789abcdef", &text, "{}").unwrap_err();
            assert!(error.contains("CPU twin text"), "{error}");
            let error = write_records(&directory, "0123456789abcdef", "ok", &text).unwrap_err();
            assert!(error.contains("CPU twin text"), "{error}");
        }
        assert!(!directory.exists(), "nothing may be written");
        // A kernel that merely has `cpu` in its name is not a twin.
        write_records(
            &directory,
            "0123456789abcdef",
            "// cpu_scan\n",
            "{\"name\": \"cpu_scan\"}",
        )
        .unwrap();
    }

    #[test]
    fn half_pair_messages_name_the_missing_file_and_the_one_to_delete() {
        let message = incomplete_message(
            "gust-artifacts/0123456789abcdef.wgsl.reflection.json",
            "gust-artifacts/0123456789abcdef.wgsl",
        );
        assert!(message.contains("`gust-artifacts/0123456789abcdef.wgsl` is missing"));
        assert!(message.contains("Delete the file that is left, `gust-artifacts/0123456789abcdef.wgsl.reflection.json`, then run `cargo gust build`"));
    }

    #[test]
    fn user_identifiers_named_like_the_twin_are_not_twin_text() {
        let scratch = Scratch::new("twin-names");
        let directory = scratch.0.join("records");
        // A local, a helper and a struct may be called `gust_dual` or `CpuDevice`; the generated forms still match.
        write_records(
            &directory,
            "0123456789abcdef",
            "struct CpuDevice { uint a; };\nuint gust_dual(uint x) { return x; }\n",
            "{\"rust_type\": \"StructuredBuffer<CpuDevice>\", \"name\": \"gust_dual\"}",
        )
        .unwrap();
        for generated in [
            "#[cfg(any(test, gust_dual))]",
            "# [cfg (any (test , gust_dual))]",
            "impl Run for :: gust :: cpu :: CpuDevice",
            "::gust::cpu::CpuDevice",
        ] {
            assert!(twin_text(generated).is_some(), "{generated}");
        }
    }
}

use std::{collections::HashMap, path::Path};

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};

use crate::{
    diagnostics::{self, Code},
    slang::ParameterFlavor,
};

mod artifact;
use artifact::{Environment, Lookup};

#[derive(Clone, Copy, Debug)]
pub struct KernelOptions {
    pub workgroup_size: [u32; 3],
}

/// What the expansion knows about the build profile (D44, R1-9). The macro cannot infer a profile (the build script of
/// `gust` owns the effective one and its release inference), so it asks one question only: is `GUST_PROFILE` exactly
/// `cpu`? The answer is an argument, never read from the process inside the expander, so expansion tests run in
/// parallel threads without touching the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// `GUST_PROFILE=cpu`: the production CPU target (D21). The CPU impl is ungated, no artifact is looked up or
    /// embedded, and nothing of the artifact directory is tracked.
    Cpu,
    /// Every other value, and an unset variable: the CPU impl is a twin behind `cfg(any(test, gust_dual))` and
    /// artifacts are embedded as before.
    Other,
}

impl Profile {
    /// The profile for the value of `GUST_PROFILE` (`None` when unset). Only the exact text `cpu` is `Cpu`; the build
    /// script rejects every unknown value with the real error, so the macro never validates.
    pub fn from_value(value: Option<&str>) -> Self {
        if value == Some("cpu") {
            Self::Cpu
        } else {
            Self::Other
        }
    }
}

/// Whether a kernel gets a CPU twin (roadmap M3): the `cpu` option of `#[kernel]`. It is kept beside
/// [`KernelOptions`], not inside it, because only the expander reads it; the Slang emitter never does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CpuTwin {
    /// The default: the kernel runs on `gust::cpu::CpuDevice` under `cfg(any(test, gust_dual))`, or in every build
    /// under `GUST_PROFILE=cpu`.
    #[default]
    Emitted,
    /// `cpu = "unsupported"`: no twin is emitted.
    Unsupported,
}

/// [`expand_with`] for a caller that has no profile: [`Profile::Other`].
#[cfg(test)]
pub fn expand(module: syn::ItemMod) -> syn::Result<TokenStream> {
    expand_with(module, Profile::Other)
}

/// Expands a module for the build profile `profile`, reading the crate directory and the export directory from the
/// process environment.
pub fn expand_with(module: syn::ItemMod, profile: Profile) -> syn::Result<TokenStream> {
    expand_in(module, &Environment::from_process(), profile)
}

/// [`expand`] with the crate directory, the export directory and the package name given, for the pipeline tests that
/// assert the codes of the build-directory errors (E0711 to E0713) without touching the process environment.
#[cfg(test)]
pub(crate) fn expand_with_directories(
    module: syn::ItemMod,
    manifest_dir: Option<&Path>,
    export_dir: Option<&Path>,
    package: Option<&str>,
) -> syn::Result<TokenStream> {
    let environment = Environment {
        manifest_dir: manifest_dir.map(Path::to_path_buf),
        export_dir: export_dir.map(Path::to_path_buf),
        package: package.map(str::to_owned),
        reported_file: None,
    };
    expand_in(module, &environment, Profile::Other)
}

/// [`expand_with`] with the environment given, so that tests choose the crate directory and the export directory
/// without touching the process environment.
///
/// Under [`Profile::Cpu`] the artifact directory is not read at all (GPU bytes do not belong in a CPU binary, and a
/// half pair there is no concern of a CPU build), so every kernel is `Lookup::Absent` and the manifest is absent;
/// the per-kernel `require_artifact` item is still emitted, and the build script's profile makes it a no-op.
fn expand_in(
    mut module: syn::ItemMod,
    environment: &Environment,
    profile: Profile,
) -> syn::Result<TokenStream> {
    crate::validate::validate_module(&module)?;
    let original_module = module.clone();
    let module_name = module.ident.to_string();
    let manifest_dir = environment.manifest_dir.as_deref();
    let manifest_present = profile != Profile::Cpu && artifact::manifest_present(manifest_dir);
    let record_directory = environment.record_directory();
    let export_mode = environment.export_dir.is_some();
    // The kernel whose line carries the module's one `require_manifest` check: the first with a complete pair.
    let mut manifest_span: Option<proc_macro2::Span> = None;
    let structs: HashMap<String, &syn::ItemStruct> = original_module
        .content
        .iter()
        .flat_map(|(_, items)| items)
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some((item.ident.to_string(), item)),
            _ => None,
        })
        .collect();
    let (_, items) = module.content.as_mut().expect("validated inline module");
    let mut generated = Vec::new();
    let mut descriptor_paths = Vec::new();

    for item in items.iter_mut() {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        item.attrs.push(syn::parse_quote!(#[repr(C)]));
        item.attrs.push(syn::parse_quote!(#[derive(Clone, Copy)]));
        generated.push(gpu_pod_impl(item)?);
    }

    for item in items.iter_mut() {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let Some((options, cpu)) = kernel_attribute(function)? else {
            continue;
        };
        let original_name = function.sig.ident.clone();
        let original_name_text = original_name.to_string();
        let entry_point = format!("gpu_{}_{}", module_name, original_name_text);
        let typecheck_name = format_ident!("__gpu_typecheck_{}", original_name);
        let visibility = function.vis.clone();
        let parameter_descriptors = parameter_descriptors(&function.sig.inputs)?;
        let original_kernel = find_function(&original_module, &original_name_text);
        let emitted = crate::slang::emit_kernel(&original_module, original_kernel, options)?;
        let reported = reported_file(original_kernel, environment);
        let source_origin = source_origin(reported.as_deref(), manifest_dir, &emitted.lines);
        let slang_source = emitted.source;
        // The key hashes the clean text, so it is computed here: the Slang text is final, and the CPU twin (below)
        // never reaches it.
        let key = artifact::content_key(&entry_point, &slang_source);
        let lookup = match artifact::lookup(manifest_dir, &key) {
            // A CPU build looks at no artifact: nothing is embedded and nothing is tracked.
            _ if profile == Profile::Cpu => Lookup::Absent,
            // Export mode is how `cargo gust build` writes the missing half, and it expands this crate to do so: a
            // half pair must not stop that expansion, so it counts as absent there (and only there).
            Lookup::Incomplete { .. } if export_mode => Lookup::Absent,
            other => other,
        };
        let artifacts = artifact::artifact_tokens(&key, &lookup);
        let strict_artifact = artifact_check_items(&lookup, original_name.span());
        if lookup == Lookup::Present && manifest_span.is_none() {
            manifest_span = Some(original_name.span());
        }
        let export_error = match &record_directory {
            Ok(None) => TokenStream::new(),
            Ok(Some(directory)) => {
                let parameters = parameter_records(&function.sig.inputs, &structs)?;
                let file = origin_file(reported.as_deref(), manifest_dir, &emitted.lines);
                let record = artifact::KernelRecord {
                    key: &key,
                    module: &module_name,
                    name: &original_name_text,
                    entry_point: &entry_point,
                    workgroup_size: options.workgroup_size,
                    parameters: &parameters,
                    file: file.as_deref().unwrap_or_default(),
                    lines: if file.is_some() { &emitted.lines } else { &[] },
                };
                match artifact::write_records(directory, &key, &slang_source, &record.json()) {
                    Ok(()) => TokenStream::new(),
                    Err(message) => artifact::error_item(
                        Code::ExportWriteFailed,
                        original_name.span(),
                        &message,
                    ),
                }
            }
            Err(message) => {
                artifact::error_item(Code::ExportPackageName, original_name.span(), message)
            }
        };
        let [x, y, z] = options.workgroup_size;
        let (handle_doc, typed_calls) = typed_call_items(&original_name, &function.sig.inputs)?;
        let element_assertions = element_layout_assertions(&original_name, &function.sig.inputs)?;
        let cpu_twin = cpu_twin_items(
            &original_name,
            &typecheck_name,
            &function.sig.inputs,
            cpu,
            profile,
        )?;

        function
            .attrs
            .retain(|attribute| !attribute.path().is_ident("kernel"));
        function.sig.ident = typecheck_name;
        function.vis = syn::Visibility::Inherited;

        descriptor_paths.push(original_name.clone());
        generated.push(quote! {
            #[doc = #handle_doc]
            #[allow(non_camel_case_types)]
            #visibility struct #original_name;

            impl #original_name {
                pub const DESCRIPTOR: ::gust::KernelDescriptor =
                    <Self as ::gust::Kernel>::DESCRIPTOR;
            }

            impl ::gust::Kernel for #original_name {
                const DESCRIPTOR: ::gust::KernelDescriptor = ::gust::KernelDescriptor {
                    module: #module_name,
                    name: stringify!(#original_name),
                    entry_point: #entry_point,
                    workgroup_size: [#x, #y, #z],
                    parameters: &[#(#parameter_descriptors),*],
                    slang_source: #slang_source,
                    source: #source_origin,
                    artifacts: #artifacts,
                };
            }

            #typed_calls

            #strict_artifact

            #export_error

            #cpu_twin

            #element_assertions
        });
    }

    items.insert(0, syn::parse_quote! { use ::gust::prelude::*; });
    let tracked_environment = tracked_environment();
    let manifest_include = artifact::manifest_tokens(manifest_present);
    // One check per module, not per kernel: the manifest is a property of the crate, so N kernels with a pair and no
    // manifest are one cause and one error, spanned at the first kernel of them.
    let manifest_check = manifest_span
        .map(|span| require_manifest_item(manifest_present, span))
        .unwrap_or_default();
    let module_ident = &module.ident;
    let module_vis = &module.vis;
    let module_attrs = &module.attrs;
    let content = &module.content.as_ref().expect("inline module").1;
    Ok(quote! {
        #(#module_attrs)*
        #[allow(dead_code)]
        #[allow(unexpected_cfgs)]
        #module_vis mod #module_ident {
            #(#content)*
            #(#generated)*

            #tracked_environment

            #manifest_include

            #manifest_check

            pub const MODULE_DESCRIPTOR: ::gust::ModuleDescriptor =
                ::gust::ModuleDescriptor {
                    name: #module_name,
                    kernels: &[#(#descriptor_paths::DESCRIPTOR),*],
                };
        }
    })
}

/// The profile checks of one kernel, all spanned at the kernel so that a strict build fails at the kernel's own line in
/// the user's file, not inside `gust`:
/// - no artifact: `require_artifact(false)`, which a strict profile (D44) turns into a compile error;
/// - a complete pair: `require_artifact(true)`. The module as a whole then also gets one
///   `require_manifest(<the manifest exists>)` (see `expand_in`), which a strict profile turns into the "run
///   `cargo gust init` and `cargo gust build`" error (R4.3). A kernel without an artifact reports only the artifact,
///   so that one cause is one error; its fix, `cargo gust build`, writes the manifest too;
/// - half a pair (outside export mode, where it is absent): a coded `compile_error!` naming the missing file and the
///   one to delete, and nothing else, so that it is never a silent `&[]`.
///
/// Under `dev` the profile checks do nothing. They are emitted in every expansion.
fn artifact_check_items(lookup: &Lookup, span: proc_macro2::Span) -> TokenStream {
    match lookup {
        Lookup::Absent => require_artifact_item(false, span),
        Lookup::Present => require_artifact_item(true, span),
        Lookup::Incomplete { left, missing } => artifact::error_item(
            Code::HalfArtifactPair,
            span,
            &artifact::incomplete_message(left, missing),
        ),
    }
}

/// `gust::profile::require_artifact(present)` in a `const _`. Under `dev` it does nothing.
fn require_artifact_item(present: bool, span: proc_macro2::Span) -> TokenStream {
    quote_spanned! {span=>
        const _: () = ::gust::profile::require_artifact(#present);
    }
}

/// `gust::profile::require_manifest(present)` in a `const _`. Under `dev` it does nothing.
fn require_manifest_item(present: bool, span: proc_macro2::Span) -> TokenStream {
    quote_spanned! {span=>
        const _: () = ::gust::profile::require_manifest(#present);
    }
}

/// The parameters of a kernel as `kernel.json` records them: the facts [`parameter_descriptors`] turns into tokens,
/// plus the element layout of each storage parameter.
fn parameter_records(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
    structs: &HashMap<String, &syn::ItemStruct>,
) -> syn::Result<Vec<artifact::ParameterRecord>> {
    Ok(parameter_facts(inputs)?
        .into_iter()
        .map(|fact| {
            let (kind, access) = match fact.flavor {
                ParameterFlavor::Invocation => ("invocation", "not_applicable"),
                ParameterFlavor::StorageRead => ("storage", "read_only"),
                ParameterFlavor::StorageReadWrite => ("storage", "read_write"),
                ParameterFlavor::Uniform => ("uniform", "read_only"),
                ParameterFlavor::Value => ("value", "not_applicable"),
            };
            let layout = fact
                .element
                .as_ref()
                .and_then(|element| artifact::element_layout(element, structs));
            artifact::ParameterRecord {
                name: fact.name,
                rust_type: fact.rust_type,
                kind,
                access,
                binding: fact.binding,
                layout,
            }
        })
        .collect())
}

/// Reads the two variables that decide what a kernel crate's expansion contains, so that rustc records them as
/// dependencies of the crate (`option_env!` is tracked) and Cargo re-expands the `#[gpu]` module when either one
/// changes: `GUST_PROFILE` (the profile, D44) and `GUST_EXPORT_DIR` (export mode, D46). The values are not used.
fn tracked_environment() -> TokenStream {
    quote! {
        const _: ::core::option::Option<&::core::primitive::str> = ::core::option_env!("GUST_PROFILE");
        const _: ::core::option::Option<&::core::primitive::str> = ::core::option_env!("GUST_EXPORT_DIR");
    }
}

/// The `source:` field of a kernel descriptor: the Rust file the kernel was
/// written in plus the line table recorded by the emitter. `file` is the kernel's
/// span file passed through [`recorded_path`], so it is never absolute (the
/// descriptor is compiled into public binaries and a path would leak the build
/// machine's layout) and uses `/` only (a later `#line` directive would embed it
/// in a Slang string literal, where `\` is an escape). When the compiler gave no
/// real file (a span from a synthesized token stream, as in unit tests), or
/// [`recorded_path`] could not produce a safe one, the origin is unknown and the
/// table is dropped, because lines without a file cannot be reported.
fn source_origin(
    reported: Option<&str>,
    manifest_dir: Option<&Path>,
    lines: &[crate::slang::LineRecord],
) -> TokenStream {
    let Some(reported) = reported else {
        return quote!(::gust::SourceOrigin::UNKNOWN);
    };
    origin_tokens(reported, manifest_dir.and_then(Path::to_str), lines)
}

/// The file rustc reports for `kernel`, or the one the [`Environment`] gives in its place; `None` when there is no
/// real file (a span from a synthesized token stream).
fn reported_file(kernel: &syn::ItemFn, environment: &Environment) -> Option<String> {
    let reported = environment
        .reported_file
        .clone()
        .unwrap_or_else(|| kernel.sig.ident.span().file());
    (!reported.is_empty() && !reported.starts_with('<')).then_some(reported)
}

/// The recorded source file of a kernel when its origin is known, which is when [`source_origin`] emits a table: the
/// export record holds the same file and the same line table as the descriptor. The crate directory comes from the
/// [`Environment`], like everything else the expansion reads.
fn origin_file(
    reported: Option<&str>,
    manifest_dir: Option<&Path>,
    lines: &[crate::slang::LineRecord],
) -> Option<String> {
    known_origin_file(reported?, manifest_dir.and_then(Path::to_str), lines)
}

/// The file of an origin: [`recorded_path`], unless that is empty or there are no lines to map.
fn known_origin_file(
    reported: &str,
    manifest_dir: Option<&str>,
    lines: &[crate::slang::LineRecord],
) -> Option<String> {
    let file = recorded_path(reported, manifest_dir);
    (!file.is_empty() && !lines.is_empty()).then_some(file)
}

/// [`source_origin`] once the span file and the manifest dir are known.
fn origin_tokens(
    reported: &str,
    manifest_dir: Option<&str>,
    lines: &[crate::slang::LineRecord],
) -> TokenStream {
    let Some(file) = known_origin_file(reported, manifest_dir, lines) else {
        return quote!(::gust::SourceOrigin::UNKNOWN);
    };
    let lines = lines.iter().map(|line| {
        let (slang_start, slang_end, rust_line) =
            (line.slang_start, line.slang_end, line.rust_line);
        quote!(::gust::LineSpan { slang_start: #slang_start, slang_end: #slang_end, rust_line: #rust_line })
    });
    quote!(::gust::SourceOrigin { file: #file, lines: &[#(#lines),*] })
}

/// The path stored in a descriptor, or `""` when no safe path can be derived. The
/// result is a whitelist, not a best effort: it is accepted only if it is relative
/// and every `/`-separated component is valid (see [`is_valid_record`]: non-empty,
/// not `.` or `..`, only ASCII letters and digits, `_ . + -` and Unicode
/// Alphabetic characters), so it can neither leak a build machine's layout nor
/// break out of a Slang string literal.
///
/// The whitelist is lexical sanitisation, not a proof of anonymity. A relative
/// path that passes it is kept exactly as rustc reported it: if rustc's path prefix
/// remapping already turned a location into something like
/// `home/<name>/shared/kernel.rs`, that is what is recorded, because nothing here
/// can tell a remapped prefix from a real directory. Both paths are first
/// normalised lexically (see
/// [`NormalizedPath`]: repeated separators collapsed, `.` dropped, `..` resolved),
/// then:
/// - a relative path is kept as its normalised components, unless it climbs out
///   of its root (starts with `..`), which stores the file name only;
/// - an absolute path whose components lie under `manifest_dir` (the expanding
///   crate's `CARGO_MANIFEST_DIR`) is stored relative to it;
/// - any other absolute path (Unix root, drive, UNC or `\\?\` verbatim), including
///   one that leaves the manifest dir by way of `..`, is stored as its file
///   name only.
///
/// A result that fails the whitelist is replaced by the file name, and by `""` if
/// that fails it too; [`origin_tokens`] turns `""` into an unknown origin.
fn recorded_path(reported: &str, manifest_dir: Option<&str>) -> String {
    let path = NormalizedPath::new(reported);
    let record = contained_record(&path, manifest_dir);
    if is_valid_record(&record) {
        return record;
    }
    let name = file_name(path.parts.iter().map(String::as_str));
    if is_valid_record(&name) {
        name
    } else {
        String::new()
    }
}

/// [`recorded_path`] before the whitelist check.
fn contained_record(path: &NormalizedPath, manifest_dir: Option<&str>) -> String {
    if path.root.is_empty() {
        return if path.parts.first().is_some_and(|part| part == "..") {
            file_name(path.parts.iter().map(String::as_str))
        } else {
            path.parts.join("/")
        };
    }
    if let Some(manifest_dir) = manifest_dir.filter(|_| path.root != OPAQUE_ROOT) {
        let dir = NormalizedPath::new(manifest_dir);
        // Windows paths compare case-insensitively; a false match on another
        // platform could only shorten a path that is already under the root.
        let windows = dir.root != "/";
        let same = |a: &str, b: &str| {
            if windows {
                a.eq_ignore_ascii_case(b)
            } else {
                a == b
            }
        };
        if !dir.root.is_empty()
            && dir.root != OPAQUE_ROOT
            && same(&path.root, &dir.root)
            && path.parts.len() > dir.parts.len()
            && dir.parts.iter().zip(&path.parts).all(|(a, b)| same(a, b))
        {
            return path.parts[dir.parts.len()..].join("/");
        }
    }
    file_name(path.parts.iter().map(String::as_str))
}

/// What may be stored: a non-empty relative path whose every `/`-separated
/// component is valid. This also excludes a drive (`:`), a backslash and control
/// characters, which are not in the component alphabet.
fn is_valid_record(record: &str) -> bool {
    !record.is_empty() && record.split('/').all(is_valid_component)
}

/// A non-empty name of ASCII letters and digits, `_`, `.`, `+`, `-` and non-ASCII
/// characters with the Unicode Alphabetic property (`char::is_alphabetic`); never
/// `.` or `..` (nor any other name made of dots only, which Windows does not keep
/// as a distinct name). Alphabetic includes combining marks such as U+0345; a
/// decomposed accent such as U+0301 is not Alphabetic and is rejected.
fn is_valid_component(part: &str) -> bool {
    !part.is_empty()
        && !part.chars().all(|c| c == '.')
        && part.chars().all(|c| {
            c.is_ascii_alphanumeric()
                || matches!(c, '_' | '.' | '+' | '-')
                || (!c.is_ascii() && c.is_alphabetic())
        })
}

/// The last component that is not `..`.
fn file_name<'a>(parts: impl Iterator<Item = &'a str>) -> String {
    parts
        .filter(|part| *part != "..")
        .last()
        .unwrap_or_default()
        .to_owned()
}

/// The `root` of a `\\?\` verbatim path that is neither `\\?\X:\` nor
/// `\\?\UNC\`: GLOBALROOT, `Volume{GUID}`, device and drive-relative forms. It is
/// absolute and opaque, so it never counts as lying under a manifest dir.
const OPAQUE_ROOT: &str = "//?/";

/// A path split into a root and lexically normalised components. `root` is `""`
/// (relative), `/` (Unix), `X:/` (drive), `X:` (drive-relative), `//host/share`
/// (UNC) or [`OPAQUE_ROOT`]. Of the `\\?\` verbatim forms only `\\?\X:\...` and
/// `\\?\UNC\host\share\...` (any case of `UNC`) are recognised, as the drive and
/// UNC paths they name; every other one is opaque. Separators are `/` or `\`. In
/// `parts` repeated separators are collapsed, `.` is dropped and `..` pops its
/// parent; a `..` with no parent is dropped under a root and kept at the head of a
/// relative path.
struct NormalizedPath {
    root: String,
    parts: Vec<String>,
}

impl NormalizedPath {
    fn new(path: &str) -> Self {
        let path = path.replace('\\', "/");
        let mut opaque = false;
        let path = if let Some(rest) = path.strip_prefix("//?/") {
            let bytes = rest.as_bytes();
            if rest
                .get(..4)
                .is_some_and(|p| p.eq_ignore_ascii_case("UNC/"))
            {
                format!("//{}", &rest[4..])
            } else if bytes.len() >= 3
                && bytes[0].is_ascii_alphabetic()
                && bytes[1] == b':'
                && bytes[2] == b'/'
            {
                rest.to_owned()
            } else {
                opaque = true;
                rest.to_owned()
            }
        } else {
            path
        };
        let bytes = path.as_bytes();
        let (root, rest) = if opaque {
            (OPAQUE_ROOT.to_owned(), path.clone())
        } else if path.starts_with("//") {
            // The host and share are the root, not components that `..` can pop.
            let names: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
            let split = names.len().min(2);
            (
                format!("//{}", names[..split].join("/")),
                names[split..].join("/"),
            )
        } else if path.starts_with('/') {
            ("/".to_owned(), path.clone())
        } else if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            let rooted = path[2..].starts_with('/');
            let separator = if rooted { "/" } else { "" };
            (format!("{}{separator}", &path[..2]), path[2..].to_owned())
        } else {
            (String::new(), path.clone())
        };
        let mut parts: Vec<String> = Vec::new();
        for part in rest.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    if parts.last().is_some_and(|last| last != "..") {
                        parts.pop();
                    } else if root.is_empty() {
                        parts.push("..".to_owned());
                    }
                }
                _ => parts.push(part.to_owned()),
            }
        }
        NormalizedPath { root, parts }
    }
}

fn find_function<'a>(module: &'a syn::ItemMod, name: &str) -> &'a syn::ItemFn {
    module
        .content
        .as_ref()
        .expect("inline module")
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .expect("kernel exists in original module")
}

/// The options the Slang emitter needs. Every caller other than the expander wants only these; the `cpu`
/// option is parsed (and rejected when it is wrong) by [`kernel_attribute`], which this wraps.
pub fn kernel_options(function: &syn::ItemFn) -> syn::Result<Option<KernelOptions>> {
    Ok(kernel_attribute(function)?.map(|(options, _)| options))
}

/// All options of a `#[kernel]` attribute: `workgroup_size(x, y, z)` and `cpu = "unsupported"`, each at
/// most once. `None` when the function has no `#[kernel]`.
pub fn kernel_attribute(function: &syn::ItemFn) -> syn::Result<Option<(KernelOptions, CpuTwin)>> {
    if function
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("kernel"))
        .count()
        > 1
    {
        return Err(diagnostics::error(
            Code::DuplicateKernelAttribute,
            &function.sig,
            "duplicate #[kernel] attribute",
        ));
    }
    let Some(attribute) = function
        .attrs
        .iter()
        .find(|attribute| attribute.path().is_ident("kernel"))
    else {
        return Ok(None);
    };
    if matches!(attribute.meta, syn::Meta::Path(_)) {
        return Ok(Some((
            KernelOptions {
                workgroup_size: [64, 1, 1],
            },
            CpuTwin::Emitted,
        )));
    }

    let mut result = None;
    let mut cpu = None;
    // The options below report their own coded errors. Failures `syn` words itself (a
    // missing comma, a literal that is not an integer) get the malformed-attribute code.
    diagnostics::parse_nested_meta(attribute, Code::MalformedKernelAttribute, |meta| {
        if meta.path.is_ident("workgroup_size") {
            if result.is_some() {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::DuplicateWorkgroupSize,
                    "duplicate workgroup_size option",
                ));
            }
            let content;
            syn::parenthesized!(content in meta.input);
            let x: syn::LitInt = content.parse()?;
            content.parse::<syn::Token![,]>()?;
            let y: syn::LitInt = content.parse()?;
            content.parse::<syn::Token![,]>()?;
            let z: syn::LitInt = content.parse()?;
            if !content.is_empty() {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::WorkgroupDimensionCount,
                    "expected exactly three workgroup dimensions",
                ));
            }
            let dimensions = [x.base10_parse()?, y.base10_parse()?, z.base10_parse()?];
            if dimensions.contains(&0) {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::WorkgroupDimensionZero,
                    "workgroup dimensions must be nonzero",
                ));
            }
            result = Some(dimensions);
            Ok(())
        } else if meta.path.is_ident("cpu") {
            if cpu.is_some() {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::DuplicateCpuOption,
                    "duplicate cpu option",
                ));
            }
            let value: syn::LitStr = meta.value()?.parse()?;
            if value.value() != "unsupported" {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::UnknownCpuValue,
                    "unknown `cpu` value; the only value is `\"unsupported\"`",
                ));
            }
            cpu = Some(CpuTwin::Unsupported);
            Ok(())
        } else {
            Err(diagnostics::meta_error(
                &meta,
                Code::UnknownKernelOption,
                "expected `workgroup_size(x, y, z)` or `cpu = \"unsupported\"`",
            ))
        }
    })?;
    Ok(Some((
        KernelOptions {
            workgroup_size: result.unwrap_or([64, 1, 1]),
        },
        cpu.unwrap_or_default(),
    )))
}

/// What the expander knows about one kernel parameter. The descriptor tokens ([`parameter_descriptors`]) and the
/// export record ([`parameter_records`]) are both made from it, so they cannot disagree about a binding.
struct ParameterFacts {
    name: String,
    rust_type: String,
    flavor: ParameterFlavor,
    /// The binding in descriptor set 0 of a storage or uniform parameter, numbered in declaration order.
    binding: Option<u32>,
    /// The element type of a storage parameter.
    element: Option<syn::Type>,
}

fn parameter_facts(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<Vec<ParameterFacts>> {
    let mut next_binding = 0_u32;
    inputs
        .iter()
        .map(|input| {
            let syn::FnArg::Typed(input) = input else {
                return Err(diagnostics::error(
                    Code::SelfParameter,
                    input,
                    "GPU functions cannot use self",
                ));
            };
            let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
                return Err(diagnostics::error(
                    Code::ParameterNeedsIdentifier,
                    &input.pat,
                    "GPU parameters need identifier names",
                ));
            };
            let flavor = crate::slang::parameter_flavor(&input.ty);
            let is_resource = matches!(
                flavor,
                ParameterFlavor::StorageRead
                    | ParameterFlavor::StorageReadWrite
                    | ParameterFlavor::Uniform
            );
            let binding = is_resource.then(|| {
                let binding = next_binding;
                next_binding += 1;
                binding
            });
            let element = if matches!(
                flavor,
                ParameterFlavor::StorageRead | ParameterFlavor::StorageReadWrite
            ) {
                Some(crate::slang::wrapper_element_type(&input.ty)?.clone())
            } else {
                None
            };
            Ok(ParameterFacts {
                name: crate::slang::emit_ident(&pattern.ident),
                rust_type: input.ty.to_token_stream().to_string(),
                flavor,
                binding,
                element,
            })
        })
        .collect()
}

fn parameter_descriptors(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<Vec<TokenStream>> {
    parameter_facts(inputs)?
        .into_iter()
        .map(|fact| {
            let ParameterFacts {
                name,
                rust_type,
                flavor,
                binding,
                element,
            } = fact;
            let (kind, access) = match flavor {
                ParameterFlavor::Invocation => (
                    quote!(::gust::ParameterKind::Invocation),
                    quote!(::gust::Access::NotApplicable),
                ),
                ParameterFlavor::StorageRead => (
                    quote!(::gust::ParameterKind::Storage),
                    quote!(::gust::Access::ReadOnly),
                ),
                ParameterFlavor::StorageReadWrite => (
                    quote!(::gust::ParameterKind::Storage),
                    quote!(::gust::Access::ReadWrite),
                ),
                ParameterFlavor::Uniform => (
                    quote!(::gust::ParameterKind::Uniform),
                    quote!(::gust::Access::ReadOnly),
                ),
                ParameterFlavor::Value => (
                    quote!(::gust::ParameterKind::Value),
                    quote!(::gust::Access::NotApplicable),
                ),
            };
            let binding = match binding {
                Some(binding) => {
                    quote!(Some(::gust::ResourceBinding { group: 0, binding: #binding }))
                }
                None => quote!(None),
            };
            let resource_layout = match element {
                Some(element) => quote!(Some(::gust::ResourceLayout::storage_v1(
                    <#element as ::gust::GpuPod>::LAYOUT,
                ))),
                None => quote!(None),
            };
            Ok(quote! {
                ::gust::ParameterDescriptor {
                    name: #name,
                    rust_type: #rust_type,
                    kind: #kind,
                    access: #access,
                    resource_layout: #resource_layout,
                    binding: #binding,
                }
            })
        })
        .collect()
}

/// One storage parameter of a kernel, as the typed call needs it.
struct StorageArgument {
    name: syn::Ident,
    element: syn::Type,
    access: TokenStream,
}

/// The typed host API of one kernel (roadmap M2): `call`, `submit` and `dispatch` on the handle, plus a
/// non-generic constant assertion that the generated argument count equals the descriptor's number of
/// storage parameters (so `cargo check` catches a mismatch, not only monomorphisation).
///
/// Each storage parameter becomes one `impl Into<Arg<'r, Ri>>` argument, in declaration order, under its
/// own name. `Ri` must be a `gust::Resource` of the parameter's element type and of the device of the
/// first argument; every `Ri` carries `'r` (S0 decision A: `Arg<'r, Ri>` needs `Ri: 'r`, and rustc does not
/// infer that for a function signature). Generated value names carry the reserved `__gust_` prefix (the
/// validator rejects it in user code) and the generic parameters carry `__Gust`, so a user parameter named
/// `element_count` or `device`, or an element type named `R0` or `D`, cannot collide with them.
///
/// A kernel with a `Uniform` parameter gets none of this: the runtime rejects non-storage resources
/// (`gust_wgpu::Error::UnsupportedParameter`), so a typed call could never run. The handle and its
/// descriptor are unchanged. A plain value parameter never reaches this function: the emitter rejects it
/// at expansion (GUST-E0801), so only uniform parameters take that path.
///
/// Returns the doc text for the kernel's handle struct (users read it on `kernel_name`; it states both
/// rules above for the user) and the items.
fn typed_call_items(
    kernel: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<(String, TokenStream)> {
    let mut arguments = Vec::new();
    for input in inputs {
        // `parameter_descriptors` has already rejected a receiver or a non-identifier pattern.
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
            continue;
        };
        let access = match crate::slang::parameter_flavor(&input.ty) {
            ParameterFlavor::Invocation => continue,
            ParameterFlavor::StorageRead => quote!(::gust::Access::ReadOnly),
            ParameterFlavor::StorageReadWrite => quote!(::gust::Access::ReadWrite),
            // `Value` is defensive: the emitter rejects a plain value parameter (GUST-E0801) before
            // this runs, so in practice only a uniform parameter gets here.
            ParameterFlavor::Uniform | ParameterFlavor::Value => {
                return Ok((
                    format!(
                        "Handle of the GUST kernel `{kernel}`. It has a uniform parameter, so it has no typed \
                         `call`, `submit` or `dispatch` (a call such as `{kernel}::call(..)` is error E0599); \
                         the typed host API covers storage parameters only."
                    ),
                    TokenStream::new(),
                ));
            }
        };
        arguments.push(StorageArgument {
            name: pattern.ident.clone(),
            element: crate::slang::wrapper_element_type(&input.ty)?.clone(),
            access,
        });
    }
    let total = arguments.len();
    let count = proc_macro2::Literal::usize_unsuffixed(total);
    let count_message = format!(
        "`{kernel}`: the generated typed call takes a different number of arguments than its descriptor has storage parameters"
    );
    let names: Vec<String> = arguments.iter().map(|a| a.name.to_string()).collect();
    let order = if names.is_empty() {
        "no storage arguments".to_owned()
    } else {
        format!(
            "then the storage arguments in declaration order: {}",
            names.join(", ")
        )
    };
    let turbofish = if arguments.is_empty() {
        " The kernel has no storage arguments to infer the device from, so name it: `call::<Device>(..)`."
    } else {
        ""
    };
    let call_doc = format!(
        "Typed call of `{kernel}`: `__gust_element_count`, {order}. Each storage argument is a reference to a \
         buffer of the kernel's element type, optionally wrapped in `gust::Independent` to let its length \
         differ from the element count.{turbofish}"
    );
    let handle_doc = format!(
        "Handle of the GUST kernel `{kernel}`, with the typed host API `call`, `submit` and `dispatch` \
         (storage parameters only: a kernel with a uniform parameter has none of them).{turbofish} \
         Under `cfg(test)` of the crate that defines the kernel, or `--cfg gust_dual`, the kernel also runs on \
         `gust::cpu::CpuDevice` (its CPU twin), unless `#[kernel(cpu = \"unsupported\")]`. Under \
         `GUST_PROFILE=cpu` it runs on `gust::cpu::CpuDevice` in every build of the crate."
    );
    let submit_doc = format!(
        "Submit `{kernel}`: `__gust_device`, `__gust_element_count`, {order}. Returns a job to wait on."
    );
    let dispatch_doc =
        format!("Run `{kernel}` and wait: `__gust_device`, `__gust_element_count`, {order}.");

    let assertion = quote! {
        const _: () = ::core::assert!(
            <#kernel as ::gust::Kernel>::DESCRIPTOR.storage_parameter_count() == #count,
            #count_message
        );
    };

    if arguments.is_empty() {
        return Ok((
            handle_doc,
            quote! {
                #assertion

                impl #kernel {
                    #[doc = #call_doc]
                    pub fn call<'r, __GustD: ::gust::TypedDevice>(
                        __gust_element_count: ::core::primitive::u32,
                    ) -> ::gust::Call<'r, __GustD, Self, 0> {
                        ::gust::Call::new(__gust_element_count, [])
                    }

                    #[doc = #submit_doc]
                    #[allow(clippy::type_complexity)]
                    pub fn submit<'d, __GustD: ::gust::Run<Self, 0>>(
                        __gust_device: &'d __GustD,
                        __gust_element_count: ::core::primitive::u32,
                    ) -> ::core::result::Result<
                        <__GustD as ::gust::Device>::Job<'d>,
                        <__GustD as ::gust::Device>::Error,
                    > {
                        <__GustD as ::gust::Run<Self, 0>>::submit_call(__gust_device, &Self::call(__gust_element_count))
                    }

                    #[doc = #dispatch_doc]
                    pub fn dispatch<__GustD: ::gust::Run<Self, 0>>(
                        __gust_device: &__GustD,
                        __gust_element_count: ::core::primitive::u32,
                    ) -> ::core::result::Result<(), <__GustD as ::gust::Device>::Error> {
                        <__GustD as ::gust::Run<Self, 0>>::dispatch_call(__gust_device, &Self::call(__gust_element_count))
                    }
                }
            },
        ));
    }

    let resources: Vec<syn::Ident> = (0..total)
        .map(|index| format_ident!("__GustR{}", index))
        .collect();
    let first = &resources[0];
    let device = quote!(<#first as ::gust::Resource>::Device);
    let parameters: Vec<TokenStream> = arguments
        .iter()
        .zip(&resources)
        .map(|(argument, resource)| {
            let name = &argument.name;
            quote!(#name: impl ::core::convert::Into<::gust::Arg<'r, #resource>>)
        })
        .collect();
    let bounds: Vec<TokenStream> = arguments
        .iter()
        .zip(&resources)
        .enumerate()
        .map(|(index, (argument, resource))| {
            let element = &argument.element;
            // Spanned on the parameter, so rustc's "required by this bound" note points at the
            // kernel parameter the bound belongs to, not at `#[gpu]`.
            let span = argument.name.span();
            if index == 0 {
                quote_spanned!(span=> #resource: ::gust::Resource<Element = #element> + 'r)
            } else {
                quote_spanned!(span=> #resource: ::gust::Resource<Element = #element, Device = #device> + 'r)
            }
        })
        .collect();
    let forwarded: Vec<&syn::Ident> = arguments.iter().map(|argument| &argument.name).collect();
    let bindings = arguments.iter().map(|argument| {
        let (name, access) = (&argument.name, &argument.access);
        quote!(::core::convert::Into::into(#name).bind(#access))
    });

    Ok((
        handle_doc,
        quote! {
            #assertion

            impl #kernel {
                #[doc = #call_doc]
                #[allow(clippy::too_many_arguments)]
                pub fn call<'r, #(#resources),*>(
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> ::gust::Call<'r, #device, Self, #count>
                where
                    #(#bounds,)*
                {
                    ::gust::Call::new(__gust_element_count, [#(#bindings),*])
                }

                #[doc = #submit_doc]
                #[allow(clippy::too_many_arguments, clippy::type_complexity)]
                pub fn submit<'d, 'r, #(#resources),*>(
                    __gust_device: &'d #device,
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> ::core::result::Result<
                    <#device as ::gust::Device>::Job<'d>,
                    <#device as ::gust::Device>::Error,
                >
                where
                    #(#bounds,)*
                    #device: ::gust::Run<Self, #count>,
                {
                    <#device as ::gust::Run<Self, #count>>::submit_call(
                        __gust_device,
                        &Self::call(__gust_element_count, #(#forwarded),*),
                    )
                }

                #[doc = #dispatch_doc]
                #[allow(clippy::too_many_arguments)]
                pub fn dispatch<'r, #(#resources),*>(
                    __gust_device: &#device,
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> ::core::result::Result<(), <#device as ::gust::Device>::Error>
                where
                    #(#bounds,)*
                    #device: ::gust::Run<Self, #count>,
                {
                    <#device as ::gust::Run<Self, #count>>::dispatch_call(
                        __gust_device,
                        &Self::call(__gust_element_count, #(#forwarded),*),
                    )
                }
            }
        },
    ))
}

/// The CPU twin of one kernel (roadmap M3): `impl Run<Kernel, N> for ::gust::cpu::CpuDevice` under
/// `cfg(any(test, gust_dual))`, or ungated under [`Profile::Cpu`] (the production CPU target, D21: the same impl with
/// no `cfg` copy, so `cpu` plus `cfg(test)` or `--cfg gust_dual` still has exactly one). It calls the shadow body (the
/// renamed typecheck function) as written, once per invocation,
/// and reaches the bindings only through `::gust::cpu::{validate, read, write, invocations}`: no body
/// copy, no rewrite, no IR and no downcast in the emitted code. `gust/src/cpu.rs` holds a hand-written impl of this
/// exact shape in its tests, which is the specification.
///
/// Nothing is emitted for `cpu = "unsupported"` and for a kernel with a uniform parameter, which has no typed call
/// to run (see [`typed_call_items`]). Locals take the user's parameter names (the validator reserves `__gust_`, so
/// they cannot collide with `__gust_call` and `__gust_id`).
fn cpu_twin_items(
    kernel: &syn::Ident,
    typecheck: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
    cpu: CpuTwin,
    profile: Profile,
) -> syn::Result<TokenStream> {
    if cpu == CpuTwin::Unsupported {
        return Ok(TokenStream::new());
    }
    let mut names = Vec::new();
    let mut acquisitions = Vec::new();
    let mut arguments = Vec::new();
    for input in inputs {
        // `parameter_descriptors` has already rejected a receiver or a non-identifier pattern.
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
            continue;
        };
        let name = &pattern.ident;
        let label = name.to_string();
        match crate::slang::parameter_flavor(&input.ty) {
            ParameterFlavor::Invocation => {
                let ty = &input.ty;
                arguments.push(quote! {
                    <#ty as ::gust::cpu::FromGlobalId>::from_global_id(__gust_id)
                });
            }
            ParameterFlavor::StorageRead => {
                let element = crate::slang::wrapper_element_type(&input.ty)?;
                names.push(name.clone());
                acquisitions.push(quote! {
                    let #name = ::gust::cpu::read::<#element>(#name, #label)?;
                });
                arguments.push(quote!(::gust::Storage::new(&#name)));
            }
            ParameterFlavor::StorageReadWrite => {
                let element = crate::slang::wrapper_element_type(&input.ty)?;
                names.push(name.clone());
                acquisitions.push(quote! {
                    let mut #name = ::gust::cpu::write::<#element>(#name, #label)?;
                });
                arguments.push(quote!(::gust::StorageMut::new(&mut #name)));
            }
            ParameterFlavor::Uniform | ParameterFlavor::Value => return Ok(TokenStream::new()),
        }
    }
    let count = proc_macro2::Literal::usize_unsuffixed(names.len());
    let gate = match profile {
        Profile::Cpu => TokenStream::new(),
        Profile::Other => quote!(#[cfg(any(test, gust_dual))]),
    };
    Ok(quote! {
        #gate
        impl ::gust::Run<#kernel, #count> for ::gust::cpu::CpuDevice {
            fn submit_call(
                &self,
                __gust_call: &::gust::Call<'_, Self, #kernel, #count>,
            ) -> ::core::result::Result<::gust::cpu::CpuJob, ::gust::cpu::Error> {
                ::gust::cpu::validate(self, __gust_call)?;
                let [#(#names),*] = *__gust_call.bindings();
                #(#acquisitions)*
                for __gust_id in
                    ::gust::cpu::invocations(__gust_call.kernel(), __gust_call.element_count())
                {
                    #typecheck(#(#arguments),*);
                }
                ::core::result::Result::Ok(::gust::cpu::CpuJob::complete())
            }
        }
    })
}

/// A type as written, for a message: rustc's token spacing (`super :: Outside`, `Vec < u32 >`) without the spaces
/// around `::`, `<`, `>` and `,`. Other spaces (`dyn Trait`) are kept.
fn type_text(ty: &syn::Type) -> String {
    ty.to_token_stream()
        .to_string()
        .replace(" :: ", "::")
        .replace(":: ", "::")
        .replace(" < ", "<")
        .replace("< ", "<")
        .replace(" >", ">")
        .replace(" ,", ",")
}

/// One `const _` per storage parameter of `kernel`: the element layout recorded in the kernel's descriptor (the
/// element's `GpuPod::LAYOUT`) has the size and alignment of the Rust type and is a StorageV1 layout whose stride
/// is the Rust size, so a hand-written `GpuPod` impl that lies is a compile error here. It covers everything the
/// runtime checks (`create_typed_buffer`, dispatch validation) cover. The assertion reads the
/// descriptor rather than naming `GpuPod` again, so an element that is no `GpuPod` still reports the one
/// E0277 the descriptor already gives (the `gpupod-missing` fixture expects exactly one). A kernel with a
/// uniform parameter is covered too (it has no typed call, but its storage buffers are still created from these
/// elements). The message is a static string (const panics cannot format) that names the element, the parameter
/// and the kernel.
fn element_layout_assertions(
    kernel: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<TokenStream> {
    let mut assertions = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
            continue;
        };
        if !matches!(
            crate::slang::parameter_flavor(&input.ty),
            ParameterFlavor::StorageRead | ParameterFlavor::StorageReadWrite
        ) {
            continue;
        }
        let element = crate::slang::wrapper_element_type(&input.ty)?;
        // `assert!` reads its message as a format string.
        let text = type_text(element).replace('{', "{{").replace('}', "}}");
        let parameter = &pattern.ident;
        let message = format!(
            "GUST StorageV1: element `{text}` layout does not match its Rust representation \
             (parameter `{parameter}` of kernel `{kernel}`)"
        );
        // `parameter_descriptors` emits one descriptor per input, in order, so `index` is this parameter's.
        let index = proc_macro2::Literal::usize_unsuffixed(index);
        assertions.push(quote! {
            const _: () = ::core::assert!(
                match <#kernel as ::gust::Kernel>::DESCRIPTOR.parameters[#index].resource_layout {
                    ::core::option::Option::Some(__gust_layout) => {
                        __gust_layout.element.size as usize == ::core::mem::size_of::<#element>()
                            && __gust_layout.element.alignment as usize
                                == ::core::mem::align_of::<#element>()
                            && __gust_layout.element_stride as usize
                                == ::core::mem::size_of::<#element>()
                            && __gust_layout.element.is_storage_v1()
                    }
                    ::core::option::Option::None => false,
                },
                #message
            );
        });
    }
    Ok(quote!(#(#assertions)*))
}

fn gpu_pod_impl(item: &syn::ItemStruct) -> syn::Result<TokenStream> {
    let ident = &item.ident;
    let syn::Fields::Named(fields) = &item.fields else {
        return Err(diagnostics::error(
            Code::PodStructNamedFields,
            item,
            "GPU POD structs require named fields",
        ));
    };
    let field_layouts = fields.named.iter().map(|field| {
        let field_ident = field.ident.as_ref().expect("named field");
        let field_name = crate::slang::emit_ident(field_ident);
        let ty = &field.ty;
        quote! {
            ::gust::StructFieldLayout {
                name: #field_name,
                offset: ::core::mem::offset_of!(#ident, #field_ident) as u32,
                ty: <#ty as ::gust::GpuPod>::LAYOUT,
            }
        }
    });
    let struct_name = ident.to_string();
    // One assertion per field, in declaration order: the field sits at the sum of the sizes of the fields
    // before it. Const panics cannot format, so the message is a static string that names the field.
    let types: Vec<&syn::Type> = fields.named.iter().map(|field| &field.ty).collect();
    let offset_assertions = fields.named.iter().enumerate().map(|(index, field)| {
        let field_ident = field.ident.as_ref().expect("named field");
        let earlier = &types[..index];
        let message = format!(
            "GUST StorageV1: field `{field_ident}` of `{struct_name}` is not at its padding-free offset"
        );
        quote! {
            ::core::assert!(
                ::core::mem::offset_of!(#ident, #field_ident)
                    == 0 #( + ::core::mem::size_of::<#earlier>() )*,
                #message
            );
        }
    });
    let padding_message = format!(
        "GUST StorageV1: `{struct_name}` has tail padding (its fields do not add up to its size)"
    );
    let alignment_message = format!("GUST StorageV1: `{struct_name}` must have alignment 4");
    let storage_message = format!("GUST StorageV1: `{struct_name}` layout is not StorageV1");
    // One block, after the impl whose `LAYOUT` it reads (const items are order-independent), so one cause is one
    // E0080 and the asserts run in this order: the field that is off its offset is named first, then the total,
    // then the alignment (an alignment other than 4 also fails the last one), then the whole layout.
    Ok(quote! {
        unsafe impl ::gust::GpuPod for #ident {
            const LAYOUT: ::gust::TypeLayout = ::gust::TypeLayout::structure(
                stringify!(#ident),
                ::core::mem::size_of::<Self>() as u32,
                ::core::mem::align_of::<Self>() as u32,
                &[#(#field_layouts),*],
            );
        }

        const _: () = {
            ::core::assert!(::core::mem::size_of::<#ident>() > 0);
            ::core::assert!(
                ::core::mem::size_of::<#ident>() <= ::core::primitive::u32::MAX as ::core::primitive::usize
            );
            #(#offset_assertions)*
            ::core::assert!(
                ::core::mem::size_of::<#ident>()
                    == 0 #( + ::core::mem::size_of::<#types>() )*,
                #padding_message
            );
            ::core::assert!(::core::mem::align_of::<#ident>() == 4, #alignment_message);
            ::core::assert!(
                <#ident as ::gust::GpuPod>::LAYOUT.is_storage_v1(),
                #storage_message
            );
        };
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;

    /// Not reachable through `#[gpu]`: the validator rejects a struct without named fields
    /// (E0104) before the host side is generated. Listed in the guard's `UNREACHABLE`.
    #[test]
    fn pod_struct_without_named_fields_is_coded_when_validation_is_skipped() {
        let item: syn::ItemStruct = syn::parse_quote! { struct P(u32, u32); };
        crate::diagnostics::testing::assert_code(
            &gpu_pod_impl(&item).unwrap_err(),
            Code::PodStructNamedFields,
        );
    }

    #[test]
    fn a_structs_asserts_share_one_const_after_its_impl_in_a_fixed_order() {
        let item: syn::ItemStruct = syn::parse_quote! { struct Holder { a: uint, w: Wide8 } };
        let text = gpu_pod_impl(&item).unwrap().to_string();
        // One block: one cause is one E0080.
        assert_eq!(text.matches("const _").count(), 1, "{text}");
        let at = |needle: &str| {
            text.find(needle)
                .unwrap_or_else(|| panic!("no `{needle}` in {text}"))
        };
        let order = [
            ":: gust :: GpuPod for Holder",
            "const _",
            "field `a` of `Holder` is not at its padding-free offset",
            "field `w` of `Holder` is not at its padding-free offset",
            "`Holder` has tail padding",
            "`Holder` must have alignment 4",
            "`Holder` layout is not StorageV1",
        ];
        let positions: Vec<usize> = order.iter().map(|needle| at(needle)).collect();
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "{positions:?} for {order:?} in {text}"
        );
        // The unnamed size-sum assert carries a message too, so no bare `assertion failed` can reach a user.
        assert!(
            text.contains("== 0 + :: core :: mem :: size_of :: < uint > () + :: core :: mem :: size_of :: < Wide8 > () , \"GUST StorageV1: `Holder` has tail padding"),
            "{text}"
        );
        assert!(!text.contains("u32 :: MAX as usize ) ;"), "{text}");
    }

    #[test]
    fn element_text_keeps_the_spaces_that_mean_something() {
        let text = |ty: syn::Type| type_text(&ty);
        assert_eq!(text(syn::parse_quote!(super::Outside)), "super::Outside");
        assert_eq!(text(syn::parse_quote!(Vec<Vec<u32>>)), "Vec<Vec<u32>>");
        assert_eq!(
            text(syn::parse_quote!(Box<dyn core::fmt::Debug>)),
            "Box<dyn core::fmt::Debug>"
        );
    }

    #[test]
    fn descriptor_contains_slang_not_custom_ir_or_spirv() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod layout {
                #[kernel]
                pub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {
                    let i = id.x;
                }
            }
        };
        let output = expand(module).unwrap().to_string();
        assert!(output.contains("slang_source"));
        assert!(!output.contains("spirv_words"));
        assert!(!output.contains("fn cpu"));
    }

    #[test]
    fn descriptor_origin_is_unknown_when_the_compiler_gave_no_real_file() {
        // Outside a compiler invocation spans carry no real path, so the table is
        // dropped: lines without a file cannot be reported.
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(
            "mod layout {\n#[kernel]\npub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n}",
        )
        .unwrap();
        let output = expand(module).unwrap().to_string();
        assert!(output.contains("source : :: gust :: SourceOrigin :: UNKNOWN"));
        assert!(!output.contains("LineSpan"));
    }

    /// The lines on which the identifiers called `name` of `tokens` start, depth first.
    fn ident_lines(tokens: TokenStream, name: &str, lines: &mut Vec<usize>) {
        for tree in tokens {
            match tree {
                proc_macro2::TokenTree::Ident(ident) if ident == name => {
                    lines.push(ident.span().start().line);
                }
                proc_macro2::TokenTree::Group(group) => ident_lines(group.stream(), name, lines),
                _ => {}
            }
        }
    }

    #[test]
    fn every_kernel_checks_its_artifact_spanned_at_the_kernel() {
        // `first` is on line 3, `second` on line 7.
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(
            "mod layout {\n#[kernel]\npub fn first(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n#[kernel]\npub fn second(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n}",
        )
        .unwrap();
        let output = expand(module).unwrap();
        let text = output.to_string();
        // No `gust-artifacts/` exists in this crate, so the argument is `false`; one check per kernel, none per module.
        assert_eq!(
            text.matches("const _ : () = :: gust :: profile :: require_artifact (false) ;")
                .count(),
            2,
            "{text}"
        );
        assert_eq!(text.matches("require_artifact").count(), 2, "{text}");
        // Each check carries the line of its own kernel, which is where a strict build reports the error.
        let mut lines = Vec::new();
        ident_lines(output, "require_artifact", &mut lines);
        assert_eq!(lines, [3, 7], "{text}");
    }

    #[test]
    fn the_checks_use_the_value_the_artifacts_field_reports() {
        let artifacts = artifact::artifact_tokens("0123456789abcdef", &Lookup::Absent);
        assert_eq!(artifacts.to_string(), "& []");
        let item = require_artifact_item(true, proc_macro2::Span::call_site()).to_string();
        assert_eq!(
            item,
            "const _ : () = :: gust :: profile :: require_artifact (true) ;"
        );
        let item = require_manifest_item(false, proc_macro2::Span::call_site()).to_string();
        assert_eq!(
            item,
            "const _ : () = :: gust :: profile :: require_manifest (false) ;"
        );
    }

    // ---- artifacts and export mode (m4.c) ----

    use artifact::tests::Scratch;

    const KERNEL: &str = "mod layout {\n#[kernel]\npub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n}";

    /// An environment for a crate named `demo-crate` at `manifest` and, optionally, with export mode on.
    fn environment(manifest: Option<&Path>, export: Option<&Path>) -> Environment {
        Environment {
            manifest_dir: manifest.map(Path::to_path_buf),
            export_dir: export.map(Path::to_path_buf),
            package: Some("demo-crate".to_owned()),
            reported_file: None,
        }
    }

    fn expanded(source: &str, environment: &Environment) -> TokenStream {
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
        expand_in(module, environment, Profile::Other).unwrap()
    }

    /// Expands `source` in export mode once and returns its key and clean Slang text, as the records say them.
    fn exported(source: &str) -> (String, String) {
        let scratch = Scratch::new("exported");
        let records = scratch.0.join("records");
        expanded(source, &environment(None, Some(&records)));
        let directory = records.join("demo-crate");
        let mut keys: Vec<String> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter_map(|name| name.strip_suffix(".slang").map(str::to_owned))
            .collect();
        assert_eq!(keys.len(), 1, "{keys:?}");
        let key = keys.remove(0);
        let slang = fs::read_to_string(directory.join(format!("{key}.slang"))).unwrap();
        (key, slang)
    }

    /// A crate directory holding the artifact pair of `KERNEL` and, if asked, the manifest.
    fn crate_with_pair(name: &str, manifest: bool) -> (Scratch, String) {
        let (key, _) = exported(KERNEL);
        let scratch = Scratch::new(name);
        scratch.artifact(&format!("{key}.wgsl"), "wgsl bytes");
        scratch.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
        if manifest {
            scratch.artifact("manifest.json", "{}");
        }
        (scratch, key)
    }

    #[test]
    fn the_exported_key_is_the_content_key_of_the_entry_point_and_the_clean_text() {
        let (key, slang) = exported(KERNEL);
        assert_eq!(key, artifact::content_key("gpu_layout_inspect", &slang));
        assert!(
            !slang.contains("#line"),
            "the key hashes the clean text: {slang}"
        );
    }

    #[test]
    fn a_complete_pair_is_embedded_and_required() {
        let (scratch, key) = crate_with_pair("pair-present", true);
        let text = expanded(KERNEL, &environment(Some(&scratch.0), None)).to_string();
        assert!(
            text.contains(
                "artifacts : & [:: gust :: Artifact { format : :: gust :: Format :: Wgsl"
            ),
            "{text}"
        );
        assert!(text.contains(&format!("key : \"{key}\"")), "{text}");
        assert!(text.contains(&format!(
            "bytes : :: core :: include_bytes ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/gust-artifacts/{key}.wgsl\"))"
        )), "{text}");
        assert!(text.contains(&format!(
            "reflection : :: core :: include_str ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/gust-artifacts/{key}.wgsl.reflection.json\"))"
        )), "{text}");
        assert!(text.contains("require_artifact (true)"), "{text}");
        assert!(!text.contains("require_artifact (false)"), "{text}");
        assert!(text.contains("require_manifest (true)"), "{text}");
        assert_eq!(
            text.matches("manifest.json").count(),
            1,
            "one include: {text}"
        );
    }

    #[test]
    fn without_files_the_slice_is_empty_and_the_artifact_is_required() {
        let scratch = Scratch::new("pair-absent");
        scratch.artifact("0000000000000000.wgsl", "another kernel's file");
        let text = expanded(KERNEL, &environment(Some(&scratch.0), None)).to_string();
        assert!(text.contains("artifacts : & []"), "{text}");
        assert!(text.contains("require_artifact (false)"), "{text}");
        assert!(!text.contains("require_artifact (true)"), "{text}");
        // One cause, one error under a strict profile: the artifact alone is reported.
        assert!(!text.contains("require_manifest"), "{text}");
        assert!(!text.contains("include_bytes"), "{text}");
        assert!(!text.contains("compile_error"), "{text}");
    }

    #[test]
    fn half_a_pair_is_a_compile_error_naming_the_missing_file_never_a_silent_empty_slice() {
        let (key, _) = exported(KERNEL);
        for remove in [format!("{key}.wgsl.reflection.json"), format!("{key}.wgsl")] {
            let missing = &remove;
            let half = Scratch::new(&format!("half {remove}"));
            half.artifact(&format!("{key}.wgsl"), "x");
            half.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
            fs::remove_file(half.0.join("gust-artifacts").join(&remove)).unwrap();
            let output = expanded(KERNEL, &environment(Some(&half.0), None));
            let text = output.to_string();
            assert!(text.contains("compile_error !"), "{text}");
            assert!(text.contains("[GUST-E0711]"), "{text}");
            assert!(
                text.contains(&format!("`gust-artifacts/{missing}` is missing")),
                "{text}"
            );
            // The message names the file to delete (the one that is left) and the command that then works.
            let left = if remove.ends_with(".json") {
                format!("{key}.wgsl")
            } else {
                format!("{key}.wgsl.reflection.json")
            };
            assert!(
                text.contains(&format!(
                    "Delete the file that is left, `gust-artifacts/{left}`, then run `cargo gust build`"
                )),
                "{text}"
            );
            // Not a silent `&[]`: the error is the whole story, with no profile check that could add a second one.
            assert!(!text.contains("require_artifact"), "{text}");
            assert!(!text.contains("include_bytes"), "{text}");
            // The error sits at the kernel's identifier (line 3 of KERNEL).
            let mut lines = Vec::new();
            ident_lines(output, "compile_error", &mut lines);
            assert_eq!(lines, [3], "{text}");
        }
    }

    /// `cargo gust build` expands the kernel crate in export mode to learn what to compile; a half pair on disk must
    /// not stop that expansion, or the build tool could never repair the half it finds.
    #[test]
    fn in_export_mode_half_a_pair_is_absent_so_that_the_build_tool_can_repair_it() {
        let (key, _) = exported(KERNEL);
        for remove in [format!("{key}.wgsl.reflection.json"), format!("{key}.wgsl")] {
            let half = Scratch::new(&format!("half-export {remove}"));
            half.artifact(&format!("{key}.wgsl"), "x");
            half.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
            fs::remove_file(half.0.join("gust-artifacts").join(&remove)).unwrap();
            let records = half.0.join("target-records");
            let text = expanded(KERNEL, &environment(Some(&half.0), Some(&records))).to_string();
            assert!(!text.contains("compile_error"), "{text}");
            assert!(text.contains("artifacts : & []"), "{text}");
            assert!(text.contains("require_artifact (false)"), "{text}");
            assert!(!text.contains("include_bytes"), "{text}");
            // The records were still written: that is what the build tool reads.
            assert!(
                records
                    .join("demo-crate")
                    .join(format!("{key}.kernel.json"))
                    .is_file()
            );
            // The same crate without export mode keeps the hard error.
            let strict = expanded(KERNEL, &environment(Some(&half.0), None)).to_string();
            assert!(strict.contains("[GUST-E0711]"), "{strict}");
        }
    }

    /// The manifest check is a property of the crate, so a module with several kernels that have a pair gets it once.
    #[test]
    fn require_manifest_is_emitted_once_per_module_not_once_per_kernel() {
        let two = "mod two {\n#[kernel]\npub fn a(id: SV_DispatchThreadID, v: StructuredBuffer<f32>) { let i = id.x; }\n#[kernel]\npub fn b(id: SV_DispatchThreadID, mut v: RWStructuredBuffer<f32>) { v[id.x] = 1.0; }\n}";
        let scratch = Scratch::new("manifest-once");
        let records = scratch.0.join("records");
        expanded(two, &environment(None, Some(&records)));
        let directory = records.join("demo-crate");
        let mut keys = 0;
        for entry in fs::read_dir(&directory).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            if let Some(key) = name.strip_suffix(".slang") {
                scratch.artifact(&format!("{key}.wgsl"), "wgsl");
                scratch.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
                keys += 1;
            }
        }
        assert_eq!(keys, 2);
        for manifest in [false, true] {
            if manifest {
                scratch.artifact("manifest.json", "{}");
            }
            let output = expanded(two, &environment(Some(&scratch.0), None));
            let text = output.to_string();
            assert_eq!(text.matches("require_artifact (true)").count(), 2, "{text}");
            assert_eq!(
                text.matches(&format!("require_manifest ({manifest})"))
                    .count(),
                1,
                "{text}"
            );
            assert_eq!(text.matches("require_manifest").count(), 1, "{text}");
            // The one check sits at the first kernel with a pair (`a`, line 3 of the module).
            let mut lines = Vec::new();
            ident_lines(output, "require_manifest", &mut lines);
            assert_eq!(lines, [3], "{text}");
        }
    }

    #[test]
    fn a_strict_profile_needs_the_manifest_only_beside_a_present_pair() {
        let (without, _) = crate_with_pair("manifest-absent", false);
        let text = expanded(KERNEL, &environment(Some(&without.0), None)).to_string();
        assert!(text.contains("require_artifact (true)"), "{text}");
        assert!(text.contains("require_manifest (false)"), "{text}");
        assert!(!text.contains("manifest.json"), "{text}");
        let mut lines = Vec::new();
        ident_lines(
            expanded(KERNEL, &environment(Some(&without.0), None)),
            "require_manifest",
            &mut lines,
        );
        assert_eq!(lines, [3]);
    }

    #[test]
    fn the_manifest_is_included_once_per_module_even_without_kernels() {
        let scratch = Scratch::new("manifest-modules");
        scratch.artifact("manifest.json", "{}");
        for (source, kernels) in [(KERNEL, 1), ("mod empty {}", 0)] {
            let text = expanded(source, &environment(Some(&scratch.0), None)).to_string();
            assert_eq!(
                text.matches("\"/gust-artifacts/manifest.json\"").count(),
                1,
                "{text}"
            );
            assert_eq!(text.matches("require_artifact").count(), kernels, "{text}");
        }
        let two = "mod two {\n#[kernel]\npub fn a(id: SV_DispatchThreadID, v: StructuredBuffer<f32>) { let i = id.x; }\n#[kernel]\npub fn b(id: SV_DispatchThreadID, v: StructuredBuffer<f32>) { let i = id.x; }\n}";
        let text = expanded(two, &environment(Some(&scratch.0), None)).to_string();
        assert_eq!(text.matches("manifest.json").count(), 1, "{text}");
    }

    #[test]
    fn export_mode_writes_both_records_under_the_package_and_matches_the_golden() {
        let scratch = Scratch::new("export-golden");
        let records = scratch.0.join("records");
        let source = "mod demo {\nstruct Pair { a: uint, b: float }\n#[kernel(workgroup_size(32, 2, 1))]\npub fn run(id: SV_DispatchThreadID, pairs: StructuredBuffer<Pair>, mut out: RWStructuredBuffer<float>) {\nout[id.x] = pairs[id.x].b;\n}\n}";
        expanded(source, &environment(None, Some(&records)));
        let directory = records.join("demo-crate");
        let mut names: Vec<String> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names.len(), 2, "{names:?}");
        let key = names[0].strip_suffix(".kernel.json").unwrap().to_owned();
        assert_eq!(
            names,
            [format!("{key}.kernel.json"), format!("{key}.slang")]
        );
        let slang = fs::read(directory.join(format!("{key}.slang"))).unwrap();
        assert_eq!(
            key,
            artifact::content_key("gpu_demo_run", std::str::from_utf8(&slang).unwrap())
        );
        let json = fs::read_to_string(directory.join(format!("{key}.kernel.json"))).unwrap();
        // The key follows the emitter's output, which has its own goldens; the rest of the record is pinned here.
        assert_eq!(json.matches(&key).count(), 1, "{json}");
        artifact::tests::assert_golden(
            &json.replace(&key, "<key>"),
            "demo-run.kernel.json",
            &include_str!("../../tests/golden/demo-run.kernel.json").replace("\r\n", "\n"),
        );
        // A second expansion rewrites identical bytes.
        expanded(source, &environment(None, Some(&records)));
        assert_eq!(
            fs::read(directory.join(format!("{key}.kernel.json"))).unwrap(),
            json.as_bytes()
        );
        assert_eq!(
            fs::read(directory.join(format!("{key}.slang"))).unwrap(),
            slang
        );
    }

    #[test]
    fn export_mode_keeps_the_strict_assert_for_what_is_present() {
        let (scratch, _) = crate_with_pair("export-present", true);
        let records = scratch.0.join("target-records");
        let text = expanded(KERNEL, &environment(Some(&scratch.0), Some(&records))).to_string();
        assert!(text.contains("require_artifact (true)"), "{text}");
        assert!(
            text.contains("artifacts : & [:: gust :: Artifact"),
            "{text}"
        );
        let empty = Scratch::new("export-absent");
        let text = expanded(KERNEL, &environment(Some(&empty.0), Some(&records))).to_string();
        assert!(text.contains("require_artifact (false)"), "{text}");
        assert!(!text.contains("require_artifact (true)"), "{text}");
        assert!(text.contains("artifacts : & []"), "{text}");
    }

    #[test]
    fn export_mode_without_a_package_name_is_a_compile_error() {
        let scratch = Scratch::new("export-no-package");
        let mut environment = environment(None, Some(&scratch.0));
        environment.package = None;
        let text = expanded(KERNEL, &environment).to_string();
        assert!(text.contains("compile_error !"), "{text}");
        assert!(text.contains("[GUST-E0713]"), "{text}");
        assert!(text.contains("CARGO_PKG_NAME"), "{text}");
        assert!(fs::read_dir(&scratch.0).unwrap().next().is_none());
    }

    #[test]
    fn an_export_write_failure_is_a_compile_error_naming_the_path() {
        let scratch = Scratch::new("export-failure");
        let blocker = scratch.0.join("blocked");
        fs::write(&blocker, "a file where the export directory should be").unwrap();
        let output = expanded(KERNEL, &environment(None, Some(&blocker)));
        let text = output.to_string();
        assert!(text.contains("compile_error !"), "{text}");
        assert!(text.contains("[GUST-E0712]"), "{text}");
        assert!(text.contains("blocked"), "{text}");
        assert!(text.contains("could not create"), "{text}");
        let mut lines = Vec::new();
        ident_lines(output, "compile_error", &mut lines);
        assert_eq!(lines, [3], "{text}");
    }

    #[test]
    fn no_export_without_the_variable() {
        let scratch = Scratch::new("no-export");
        let text = expanded(KERNEL, &environment(Some(&scratch.0), None)).to_string();
        assert!(!text.contains("compile_error"), "{text}");
        assert!(fs::read_dir(&scratch.0).unwrap().next().is_none());
    }

    /// The sibling of `descriptor_contains_slang_not_custom_ir_or_spirv`'s `fn cpu` check: the records of a kernel
    /// that does get a CPU twin hold none of the twin's text, though the expansion itself does.
    #[test]
    fn the_records_of_a_kernel_with_a_twin_hold_no_twin_text() {
        let scratch = Scratch::new("twin-free");
        let records = scratch.0.join("records");
        let expansion = expanded(KERNEL, &environment(None, Some(&records))).to_string();
        for marker in ["gust_dual", "CpuDevice", "__gpu_typecheck_"] {
            assert!(
                expansion.contains(marker),
                "the expansion must carry the twin, or this test proves nothing: {marker}"
            );
        }
        let directory = records.join("demo-crate");
        let mut checked = 0;
        for entry in fs::read_dir(&directory).unwrap() {
            let record = fs::read_to_string(entry.unwrap().path()).unwrap();
            for marker in artifact::TWIN_TEXT {
                assert!(!record.contains(marker), "{marker} in {record}");
            }
            assert!(!record.contains("cpu"), "no `cpu` at all in {record}");
            checked += 1;
        }
        assert_eq!(checked, 2);
    }

    /// A user identifier is not twin text: a struct called `CpuDevice` and a function called `gust_dual` export like
    /// any other names. The guard matches the forms the generated twin takes (see `TWIN_TEXT`), and reserving the two
    /// names would reject programs that are valid today.
    #[test]
    fn user_names_like_the_twin_export_without_a_bug_report() {
        let source = "mod names {\nstruct CpuDevice { a: uint }\nfn gust_dual(x: uint) -> uint { x + 1u32 }\n#[kernel]\npub fn run(id: SV_DispatchThreadID, v: StructuredBuffer<CpuDevice>, mut out: RWStructuredBuffer<uint>) {\nlet gust_dual_value = gust_dual(v[id.x].a);\nout[id.x] = gust_dual_value;\n}\n}";
        let scratch = Scratch::new("twin-names");
        let records = scratch.0.join("records");
        let text = expanded(source, &environment(None, Some(&records))).to_string();
        assert!(!text.contains("compile_error"), "{text}");
        let directory = records.join("demo-crate");
        let mut slang = String::new();
        let mut json = String::new();
        for entry in fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            let text = fs::read_to_string(&path).unwrap();
            if path
                .extension()
                .is_some_and(|extension| extension == "slang")
            {
                slang = text;
            } else {
                json = text;
            }
        }
        assert!(slang.contains("gust_dual"), "{slang}");
        assert!(slang.contains("CpuDevice"), "{slang}");
        assert!(json.contains("StructuredBuffer < CpuDevice >"), "{json}");
    }

    /// The source table of the record is the descriptor's: with a real file and a line table, both are non-empty and
    /// equal. (The fallback spans of a unit test have no file, so the file is injected through the `Environment`.)
    #[test]
    fn export_records_hold_the_source_file_and_the_line_table_the_descriptor_holds() {
        let scratch = Scratch::new("origin");
        let records = scratch.0.join("records");
        let mut environment = environment(Some(&scratch.0), Some(&records));
        environment.reported_file = Some(format!(
            "{}/src/kernels.rs",
            scratch.0.display().to_string().replace('\\', "/")
        ));
        let text = expanded(KERNEL, &environment).to_string();
        // The descriptor: the file relative to the crate, and a non-empty table.
        assert!(
            text.contains("source : :: gust :: SourceOrigin { file : \"src/kernels.rs\" , lines : & [:: gust :: LineSpan"),
            "{text}"
        );
        let directory = records.join("demo-crate");
        let record = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.to_string_lossy().ends_with(".kernel.json"))
            .unwrap();
        let json = fs::read_to_string(record).unwrap();
        assert!(json.contains("\"file\": \"src/kernels.rs\""), "{json}");
        assert!(!json.contains("\"lines\": []"), "{json}");
        // The same table, entry for entry: every `[start, end, rust_line]` triple of the descriptor is in the record.
        let mut triples = 0;
        for part in text.split("slang_start : ").skip(1) {
            let start = part.split(',').next().unwrap();
            let end = part
                .split("slang_end : ")
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap();
            let line = part
                .split("rust_line : ")
                .nth(1)
                .unwrap()
                .split(['}', ','])
                .next()
                .unwrap();
            let number = |text: &str| -> String {
                text.trim()
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect()
            };
            let triple = format!("[{}, {}, {}]", number(start), number(end), number(line));
            assert!(json.contains(&triple), "{triple} not in {json}");
            triples += 1;
        }
        assert!(triples > 0, "{text}");
        // Without a real file there is no table in either place.
        let anonymous = Scratch::new("origin-none");
        let records = anonymous.0.join("records");
        let text = expanded(KERNEL, &environment_without_file(&records)).to_string();
        assert!(text.contains("SourceOrigin :: UNKNOWN"), "{text}");
    }

    fn environment_without_file(records: &Path) -> Environment {
        environment(None, Some(records))
    }

    #[test]
    fn the_record_file_is_the_one_the_descriptor_origin_names() {
        let lines = [crate::slang::LineRecord {
            slang_start: 3,
            slang_end: 4,
            rust_line: 12,
        }];
        assert_eq!(
            known_origin_file("/repo/src/lib.rs", REPO, &lines).as_deref(),
            Some("src/lib.rs")
        );
        assert_eq!(known_origin_file("/repo/src/lib.rs", REPO, &[]), None);
        assert_eq!(known_origin_file("src/my kernel.rs", REPO, &lines), None);
    }

    #[test]
    fn every_expansion_tracks_the_profile_and_the_export_directory() {
        let kernel = "mod layout {\n#[kernel]\npub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n}";
        for source in [kernel, "mod empty {}"] {
            let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
            let text = expand(module).unwrap().to_string();
            for variable in ["GUST_PROFILE", "GUST_EXPORT_DIR"] {
                let item = format!(
                    "const _ : :: core :: option :: Option < & :: core :: primitive :: str > = :: core :: option_env ! (\"{variable}\") ;"
                );
                assert_eq!(text.matches(&item).count(), 1, "{variable} in {text}");
            }
        }
    }

    const UNIX_ROOT: Option<&str> = Some("/srv/build/gust/crates/demo");
    const DRIVE_ROOT: Option<&str> = Some(r"C:\build_gust\crates\demo");
    const REPO: Option<&str> = Some("/repo");

    #[test]
    fn recorded_path_keeps_relative_paths_and_normalises_separators() {
        assert_eq!(
            recorded_path("crates/demo/src/lib.rs", UNIX_ROOT),
            "crates/demo/src/lib.rs"
        );
        assert_eq!(
            recorded_path(r"crates\demo\src\lib.rs", DRIVE_ROOT),
            "crates/demo/src/lib.rs"
        );
        assert_eq!(recorded_path("src/lib.rs", None), "src/lib.rs");
    }

    #[test]
    fn recorded_path_makes_absolute_paths_under_the_manifest_dir_relative() {
        assert_eq!(
            recorded_path("/srv/build/gust/crates/demo/src/lib.rs", UNIX_ROOT),
            "src/lib.rs"
        );
        assert_eq!(
            recorded_path(r"C:\build_gust\crates\demo\src\kernels\a.rs", DRIVE_ROOT),
            "src/kernels/a.rs"
        );
        // Drive letter case and a verbatim prefix do not defeat the match, and a
        // trailing separator on the manifest dir is harmless.
        assert_eq!(
            recorded_path(
                r"\\?\c:\build_gust\crates\demo\src\lib.rs",
                Some(r"C:\build_gust\crates\demo\")
            ),
            "src/lib.rs"
        );
        assert_eq!(
            recorded_path(
                r"\\?\UNC\build\share\crates\demo\src\lib.rs",
                Some(r"\\build\share\crates\demo")
            ),
            "src/lib.rs"
        );
    }

    #[test]
    fn recorded_path_keeps_only_the_file_name_of_absolute_paths_outside_the_manifest_dir() {
        for (reported, manifest_dir) in [
            ("/opt/other/crate/src/lib.rs", UNIX_ROOT),
            // A sibling that merely shares the manifest dir as a string prefix.
            ("/srv/build/gust/crates/demo-two/src/lib.rs", UNIX_ROOT),
            (r"D:\elsewhere\src\lib.rs", DRIVE_ROOT),
            (r"\\server\share\src\lib.rs", DRIVE_ROOT),
            (r"\\?\C:\Windows\src\lib.rs", DRIVE_ROOT),
            (r"\\?\UNC\server\share\src\lib.rs", DRIVE_ROOT),
            ("/opt/other/crate/src/lib.rs", None),
            (r"C:\build_gust\crates\demo\src\lib.rs", None),
            // The manifest dir itself is not a file inside it.
            ("/srv/build/gust/crates/demo", UNIX_ROOT),
        ] {
            let stored = recorded_path(reported, manifest_dir);
            let expected = reported.rsplit(['/', '\\']).next().unwrap().to_owned();
            assert_eq!(stored, expected, "{reported}");
            assert!(
                !stored.contains('/') && !stored.contains('\\'),
                "{reported}"
            );
        }
    }

    #[test]
    fn recorded_path_collapses_separators_and_dot_segments_before_the_containment_check() {
        // Cases the manifest dir `/repo` and the reported path must agree on once
        // both are normalised: a doubled separator used to leave `/src/lib.rs`.
        for (reported, manifest_dir, expected) in [
            ("/repo//src/lib.rs", REPO, "src/lib.rs"),
            ("/repo/./src/lib.rs", REPO, "src/lib.rs"),
            ("/repo/src/../src/lib.rs", REPO, "src/lib.rs"),
            ("/repo/../repo/src/lib.rs", REPO, "src/lib.rs"),
            ("/../repo/src/lib.rs", REPO, "src/lib.rs"),
            ("/repo/src/.//lib.rs", Some("/repo//"), "src/lib.rs"),
            (r"C:\repo\src\lib.rs", Some(r"C:\repo"), "src/lib.rs"),
            (r"C:\repo\\src\.\lib.rs", Some(r"C:\repo"), "src/lib.rs"),
            (r"c:\REPO\src\lib.rs", Some(r"C:\repo"), "src/lib.rs"),
            (
                r"C:\repo\src\..\src\lib.rs",
                Some(r"C:\repo\."),
                "src/lib.rs",
            ),
            // Relative paths lose their `.` and repeated separators, and resolve `..`.
            ("./src/lib.rs", None, "src/lib.rs"),
            ("src//a/./lib.rs", None, "src/a/lib.rs"),
            ("src/a/../lib.rs", None, "src/lib.rs"),
        ] {
            assert_eq!(
                recorded_path(reported, manifest_dir),
                expected,
                "{reported}"
            );
            // The whitelist never has to replace these.
            assert!(
                is_valid_record(&contained(reported, manifest_dir)),
                "{reported}"
            );
        }
    }

    #[test]
    fn recorded_path_never_follows_dot_dot_out_of_the_manifest_dir_or_a_relative_root() {
        for (reported, manifest_dir, expected) in [
            // Leaves the manifest dir by way of `..` into a sibling tree.
            ("/repo/../home/alice/kernel.rs", REPO, "kernel.rs"),
            ("/repo/src/../../x.rs", REPO, "x.rs"),
            ("/repo/..", REPO, ""),
            (r"C:\repo\..\other\lib.rs", Some(r"C:\repo"), "lib.rs"),
            // A relative path that leaves its root.
            ("../other/src/lib.rs", REPO, "lib.rs"),
            ("../other/src/lib.rs", None, "lib.rs"),
            ("src/../../lib.rs", None, "lib.rs"),
            (r"..\..\lib.rs", None, "lib.rs"),
            ("..", None, ""),
        ] {
            assert_eq!(
                recorded_path(reported, manifest_dir),
                expected,
                "{reported}"
            );
            let stored = recorded_path(reported, manifest_dir);
            assert!(stored.is_empty() || is_valid_record(&stored), "{reported}");
        }
    }

    #[test]
    fn recorded_path_whitelist_replaces_a_relative_result_that_looks_absolute() {
        // `c:` as the first component of a relative path is a drive on Windows.
        assert_eq!(contained("./c:/x.rs", None), "c:/x.rs");
        assert!(!is_valid_record("c:/x.rs"));
        assert_eq!(recorded_path("./c:/x.rs", None), "x.rs");
    }

    #[test]
    fn recorded_path_rejects_drive_like_components_even_in_the_file_name() {
        // A colon inside a component can never be stored, so when the file name
        // has one too there is nothing safe left and the origin becomes unknown.
        for (reported, manifest_dir) in [
            ("/repo/C:..", REPO),
            ("/repo/C:D:x.rs", REPO),
            ("/repo/C:x.rs", REPO),
            ("C:D:x.rs", None),
            ("a/C:x.rs", None),
            (r"\\?\C:", None),
            (r"\\?\C:lib.rs", None),
            ("C:", None),
        ] {
            assert_eq!(recorded_path(reported, manifest_dir), "", "{reported}");
        }
    }

    #[test]
    fn recorded_path_treats_unrecognised_verbatim_forms_as_opaque_roots() {
        // Only `\\?\X:\` and `\\?\UNC\` name a drive or UNC path. GLOBALROOT,
        // Volume{GUID} and device forms used to lose the prefix and be kept as a
        // relative path, leaking `Users/someone`.
        for (reported, manifest_dir, expected) in [
            (
                r"\\?\GLOBALROOT\Device\HarddiskVolume1\Users\someone\kernel.rs",
                None,
                "kernel.rs",
            ),
            (
                r"\\?\GLOBALROOT\Device\HarddiskVolume1\Users\someone\kernel.rs",
                Some(r"\\?\GLOBALROOT\Device\HarddiskVolume1\Users\someone"),
                "kernel.rs",
            ),
            (
                r"\\?\Volume{12345678-1234-1234-1234-123456789abc}\src\lib.rs",
                Some(r"C:\build_gust\crates\demo"),
                "lib.rs",
            ),
            (r"\\?\pipe\kernel.rs", UNIX_ROOT, "kernel.rs"),
            (r"//?/GLOBALROOT/x/kernel.rs", None, "kernel.rs"),
            // The recognised forms still resolve against the manifest dir.
            (
                r"\\?\unc\build\share\crates\demo\src\lib.rs",
                Some(r"\\build\share\crates\demo"),
                "src/lib.rs",
            ),
            (
                r"\\?\C:\build_gust\crates\demo\src\lib.rs",
                DRIVE_ROOT,
                "src/lib.rs",
            ),
        ] {
            assert_eq!(
                recorded_path(reported, manifest_dir),
                expected,
                "{reported}"
            );
        }
    }

    #[test]
    fn recorded_path_has_no_origin_for_empty_and_dot_only_inputs() {
        for reported in [
            "", ".", "..", "./", "../..", "/", r"\", "//", "./.", "src/..",
        ] {
            for manifest_dir in [None, REPO, DRIVE_ROOT] {
                assert_eq!(recorded_path(reported, manifest_dir), "", "{reported:?}");
            }
        }
    }

    #[test]
    fn recorded_path_keeps_unicode_letters_and_rejects_everything_else() {
        for (reported, expected) in [
            ("src/ünï/kernel_é.rs", "src/ünï/kernel_é.rs"),
            ("日本語/ファイル.rs", "日本語/ファイル.rs"),
            ("src/a.b/lib-1+2.rs", "src/a.b/lib-1+2.rs"),
            // A component outside the alphabet falls back to the file name.
            ("src/my kernels/lib.rs", "lib.rs"),
            ("src/.../lib.rs", "lib.rs"),
            ("src/a\u{ff1a}b/lib.rs", "lib.rs"),
            ("src/a\u{202e}b/lib.rs", "lib.rs"),
            ("src/a\nb/lib.rs", "lib.rs"),
            ("src/a\"b/lib.rs", "lib.rs"),
            ("src/a?b/lib.rs", "lib.rs"),
            // ... and when the file name is outside it too there is no origin.
            ("src/my kernel.rs", ""),
            ("src/kernel\u{0}.rs", ""),
            ("src/k\u{ff1a}.rs", ""),
            ("src/\u{1f600}.rs", ""),
            ("/opt/other/\"quoted\".rs", ""),
        ] {
            assert_eq!(recorded_path(reported, None), expected, "{reported:?}");
        }
    }

    #[test]
    fn recorded_path_unicode_boundary_follows_the_alphabetic_property() {
        // U+0345 COMBINING GREEK YPOGEGRAMMENI is a combining mark that is Alphabetic.
        assert!('\u{345}'.is_alphabetic());
        // U+0301 COMBINING ACUTE ACCENT (the decomposed form of an accent) is not.
        assert!(!'\u{301}'.is_alphabetic());
        for (reported, expected) in [
            // An Alphabetic combining mark survives, inside a component and on its own.
            ("src/a\u{345}b/lib.rs", "src/a\u{345}b/lib.rs"),
            ("src/\u{345}/lib.rs", "src/\u{345}/lib.rs"),
            // A decomposed accent falls back to the file name.
            ("src/e\u{301}/lib.rs", "lib.rs"),
            ("src/\u{301}/lib.rs", "lib.rs"),
            // ... and to no origin when the file name carries it.
            ("src/e\u{301}.rs", ""),
            // The precomposed letter is Alphabetic and survives.
            ("src/\u{e9}/lib.rs", "src/\u{e9}/lib.rs"),
        ] {
            assert_eq!(recorded_path(reported, None), expected, "{reported:?}");
        }
    }

    #[test]
    fn recorded_path_keeps_a_remapped_relative_prefix_as_reported() {
        // rustc `--remap-path-prefix` can leave a relative path naming a home
        // directory; the lexical whitelist cannot tell and keeps it.
        assert_eq!(
            recorded_path("home/alice/shared/kernel.rs", Some("/repo")),
            "home/alice/shared/kernel.rs"
        );
    }

    #[test]
    fn recorded_path_handles_genuinely_mixed_separators() {
        for (reported, manifest_dir, expected) in [
            (r"src\a/b\lib.rs", None, "src/a/b/lib.rs"),
            (r"C:/repo\src/lib.rs", Some(r"C:\repo"), "src/lib.rs"),
            (r"/repo\src\..\lib.rs", REPO, "lib.rs"),
            (r"\repo/src\lib.rs", REPO, "src/lib.rs"),
            (
                r"\\build/share\crates/demo\src/lib.rs",
                Some(r"\\build\share\crates\demo"),
                "src/lib.rs",
            ),
            (r"//?/C:\repo/src\lib.rs", Some(r"C:\repo"), "src/lib.rs"),
            (r"src\/\a//b\\lib.rs", None, "src/a/b/lib.rs"),
        ] {
            assert_eq!(
                recorded_path(reported, manifest_dir),
                expected,
                "{reported}"
            );
        }
    }

    #[test]
    fn recorded_path_is_always_empty_or_a_safe_relative_path() {
        for reported in [
            "/a/b.rs",
            r"C:\a\b.rs",
            "C:b.rs",
            r"\\h\s\b.rs",
            r"\\?\C:\a\b.rs",
            r"\\?\UNC\h\s\b.rs",
            "relative/b.rs",
            "../a/b.rs",
            r"a\..\..\b.rs",
            "/a/../../b.rs",
            "/a//b/./../c.rs",
            r"C:\a\..\..\b.rs",
            "C:..\\b.rs",
            r"\\h\s\..\..\b.rs",
            "/repo/C:..",
            "/repo/C:D:x.rs",
            r"\\?\GLOBALROOT\Device\HarddiskVolume1\Users\someone\kernel.rs",
            r"\\?\Volume{12345678-1234-1234-1234-123456789abc}\src\lib.rs",
            r"\\.\pipe\kernel.rs",
            "",
            ".",
            "..",
            "src/ünï/kernel_é.rs",
            "src/a b/c\td.rs",
            "src/a\u{0}b.rs",
            "a/\u{ff1a}/b.rs",
        ] {
            for manifest_dir in [
                None,
                UNIX_ROOT,
                DRIVE_ROOT,
                REPO,
                Some("/a"),
                Some(r"C:\a"),
                Some(r"\\?\GLOBALROOT\Device\HarddiskVolume1"),
                Some(""),
            ] {
                let stored = recorded_path(reported, manifest_dir);
                let context = format!("{reported:?} in {manifest_dir:?} -> {stored:?}");
                assert!(!stored.starts_with('/'), "{context}");
                assert!(!stored.contains('\\'), "{context}");
                assert!(!stored.contains(':'), "{context}");
                assert!(!stored.contains('?'), "{context}");
                assert!(!stored.split('/').any(|part| part == ".."), "{context}");
                assert!(
                    stored.chars().all(|c| c.is_ascii_alphanumeric()
                        || "_.+-/".contains(c)
                        || (!c.is_ascii() && c.is_alphabetic())),
                    "{context}"
                );
                assert!(stored.is_empty() || is_valid_record(&stored), "{context}");
            }
        }
    }

    #[test]
    fn an_empty_recorded_path_drops_the_whole_line_table() {
        use crate::slang::LineRecord;
        let lines = [LineRecord {
            slang_start: 3,
            slang_end: 4,
            rust_line: 12,
        }];
        let unknown = "SourceOrigin :: UNKNOWN";
        // No safe file: no origin, never `LineSpan`s with an empty file.
        for reported in [".", "..", r"\\?\C:", "/repo/C:..", "src/my kernel.rs"] {
            let output = origin_tokens(reported, REPO, &lines).to_string();
            assert!(output.ends_with(unknown), "{reported}: {output}");
            assert!(!output.contains("LineSpan"), "{reported}: {output}");
        }
        // A safe file keeps its table.
        let output = origin_tokens("/repo/src/lib.rs", REPO, &lines).to_string();
        assert!(output.contains("\"src/lib.rs\""), "{output}");
        assert!(output.contains("rust_line : 12u32"), "{output}");
        // ... and no lines is still unknown.
        let output = origin_tokens("/repo/src/lib.rs", REPO, &[]).to_string();
        assert!(output.ends_with(unknown), "{output}");
    }

    // ---- the build profile (m5.e1, R1-9): injected, never read from the process ----

    /// Expands `source` for `profile` in `environment`, in process: no test here sets or removes a variable.
    fn expanded_for(source: &str, environment: &Environment, profile: Profile) -> TokenStream {
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
        expand_in(module, environment, profile).unwrap()
    }

    #[test]
    fn only_the_exact_text_cpu_is_the_cpu_profile() {
        assert_eq!(Profile::from_value(Some("cpu")), Profile::Cpu);
        for other in [
            None,
            Some("dev"),
            Some("native"),
            Some("web"),
            Some("CPU"),
            Some("Cpu"),
            Some(" cpu"),
            Some("cpu "),
            Some("cpu+simd"),
            Some(""),
            Some("bogus"),
        ] {
            assert_eq!(Profile::from_value(other), Profile::Other, "{other:?}");
        }
    }

    #[test]
    fn under_cpu_a_complete_pair_is_neither_embedded_nor_tracked() {
        let (scratch, _) = crate_with_pair("cpu-pair-present", true);
        let environment = environment(Some(&scratch.0), None);
        // The control: the same crate directory under every other profile embeds the pair and tracks the manifest.
        let other = expanded_for(KERNEL, &environment, Profile::Other).to_string();
        assert!(other.contains("include_bytes"), "{other}");
        assert!(other.contains("manifest.json"), "{other}");
        let cpu = expanded_for(KERNEL, &environment, Profile::Cpu).to_string();
        assert!(cpu.contains("artifacts : & []"), "{cpu}");
        for needle in [
            "include_bytes",
            "include_str",
            "manifest.json",
            "gust-artifacts",
            "require_manifest",
        ] {
            assert!(!cpu.contains(needle), "`{needle}` in {cpu}");
        }
        // The per-kernel check is still emitted; the build script's profile makes it a no-op under `cpu`.
        assert_eq!(cpu.matches("require_artifact (false)").count(), 1, "{cpu}");
    }

    #[test]
    fn under_cpu_half_a_pair_is_no_error_because_no_artifact_is_looked_at() {
        let (key, _) = exported(KERNEL);
        let half = Scratch::new("cpu-half-pair");
        half.artifact(&format!("{key}.wgsl"), "x");
        let environment = environment(Some(&half.0), None);
        let other = expanded_for(KERNEL, &environment, Profile::Other).to_string();
        assert!(other.contains("[GUST-E0711]"), "{other}");
        let cpu = expanded_for(KERNEL, &environment, Profile::Cpu).to_string();
        assert!(!cpu.contains("compile_error"), "{cpu}");
        assert!(!cpu.contains("E0711"), "{cpu}");
    }

    #[test]
    fn every_expansion_of_every_profile_emits_the_profile_tracking_item() {
        let item = "const _ : :: core :: option :: Option < & :: core :: primitive :: str > = :: core :: option_env ! (\"GUST_PROFILE\") ;";
        for profile in [Profile::Cpu, Profile::Other] {
            for source in [KERNEL, "mod empty {}"] {
                let text = expanded_for(source, &Environment::default(), profile).to_string();
                assert_eq!(text.matches(item).count(), 1, "{profile:?}: {text}");
            }
        }
    }

    /// D44's rebuild tracking has two mechanisms, and S5.4 measured (probe/s54/s54.log) that each alone suffices to
    /// re-expand a kernel crate after `GUST_PROFILE` changes: the `option_env!("GUST_PROFILE")` item every expansion
    /// emits (rustc tracks it), and `rerun-if-env-changed=GUST_PROFILE` in the build script of `gust` (the cascade
    /// rebuilds every dependent). With both gone an expansion goes stale: a `cpu` build keeps the `dev` expansion. The
    /// macro reads the variable with `std::env::var`, which rustc does not track. This guard fails when both are gone;
    /// each has its own test besides.
    #[test]
    fn a_profile_change_is_tracked_by_at_least_one_mechanism() {
        let emitted = [Profile::Cpu, Profile::Other].iter().all(|profile| {
            expanded_for(KERNEL, &Environment::default(), *profile)
                .to_string()
                .contains("option_env ! (\"GUST_PROFILE\")")
        });
        // Only the part before the unit tests: the tests of the script name the directive too.
        let script = include_str!("../../../gust/build.rs");
        let production = script.split("#[cfg(test)]").next().unwrap();
        assert!(
            production.len() < script.len(),
            "the script has no test module to cut off"
        );
        let rerun = production.contains("\"cargo:rerun-if-env-changed=GUST_PROFILE\"");
        assert!(
            emitted || rerun,
            "neither `option_env!(\"GUST_PROFILE\")` in the expansion nor `rerun-if-env-changed=GUST_PROFILE` in \
             crates/gust/build.rs remains: a profile change would leave stale expansions (S5.4, V3)"
        );
    }

    /// [`contained_record`] from the reported string.
    fn contained(reported: &str, manifest_dir: Option<&str>) -> String {
        contained_record(&NormalizedPath::new(reported), manifest_dir)
    }
}

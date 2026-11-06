//! The crate's declarations in its manifest: `[package.metadata.gust]` (M8, D138, plan section 2.1).
//!
//! `#[gpu]` (a module or a map function) and `par_for!` read `CARGO_MANIFEST_DIR/Cargo.toml` once per expansion.
//! Three keys are declared:
//!
//! ```toml
//! [package.metadata.gust]
//! profiles = ["native", "native+subgroups", "web", "cpu"]   # the build profiles the crate declares (default list below)
//! limits = "portable"                                        # the compile-time floor (m8.f)
//!
//! [package.metadata.gust.routes]                             # routing per function tag (m8.r1)
//! physics = { targets = ["gpu", "cpu"], default = "gpu" }
//! "physics.broadphase" = {}
//! ```
//!
//! The reader is the `toml` crate's span-keeping parser (`toml::de::DeTable`, no serde): `cargo publish` rewrites a
//! manifest, so the same table may arrive as inline tables, dotted keys or headers
//! (`[package.metadata.gust.routes.physics]`) with literal strings, and every form reads the same. Inside the `gust`
//! tables the reader accepts bare or quoted keys, strings, arrays of strings and tables; anything else there is
//! GUST-E0753, reported with `Cargo.toml:LINE:COLUMN` taken from the spans. The rest of the manifest is `toml`'s
//! business: it must be valid TOML (cargo has read it already), and nothing else of it is looked at.
//!
//! What an E0753 can point at: the error is raised at the module's name, at the function's name or at the call (a proc
//! macro has no span in `Cargo.toml`), so rustc shows that identifier, and the file, line and column of the offending
//! key or value are in the message. Every `#[gpu]` module, every `#[gpu] fn` (an element-wise map) and every `par_for!`
//! call of the crate reports it once, because each goes through `expand_validated`. The path in the message is always
//! the bare file name. A crate with none of the three never has its manifest read.
//!
//! An absent key and a present one differ: [`Declarations::profiles`] is `None` when the crate wrote no list (the
//! default list, `["native", "web", "cpu"]`, is applied by the checks that need it), and an explicit list, even one
//! equal to the default, is stricter (plan section 2.1, Q4). An explicit list is never empty (an empty list declares
//! no build, so there would be nothing for the exhaustiveness check to hold), never repeats an entry, and names no
//! `dev` profile: a dev build is checked as `native` with the same capabilities (D138), so the list says
//! `native...`. Each of these is GUST-E0753 at the offending entry (at the key for the empty list).

use std::{fs, path::Path};

use toml::{
    Spanned,
    de::{DeString, DeTable, DeValue},
};

use crate::expand::{Profile, ProfileTarget};

/// The manifest's file name as every message names it: never a path.
pub(crate) const FILE: &str = "Cargo.toml";

/// The keys of `[package.metadata.gust]`.
const KEYS: [&str; 3] = ["profiles", "limits", "routes"];

/// The one value of `limits` this release knows.
const PORTABLE: &str = "portable";

/// The two targets a route may name.
const ROUTE_TARGETS: [&str; 2] = ["gpu", "cpu"];

/// What the crate declared in `[package.metadata.gust]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Declarations {
    /// `profiles`: the build profiles the crate declares, as written; `None` when the key is absent.
    pub profiles: Option<Vec<String>>,
    /// `limits`; `None` when the key is absent.
    pub limits: Option<Limits>,
    /// `routes`, in the order of the tags' names. Read for syntax only; routing is m8.r1's.
    pub routes: Vec<Route>,
}

/// The value of `limits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Limits {
    /// `limits = "portable"`: every body is checked against the WebGPU defaults (m8.f).
    Portable,
}

/// One entry of `routes`: a function tag and what the crate said of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Route {
    /// The key as written (`physics`, or `physics.broadphase` when quoted). Its syntax is judged by E0748 (m8.r1).
    pub tag: String,
    /// `targets`, `gpu` and `cpu`; `None` when absent.
    pub targets: Option<Vec<String>>,
    /// `default`, `gpu` or `cpu`; `None` when absent.
    pub default: Option<String>,
}

/// A malformed manifest: where, and what. [`ManifestError`]'s text is the message of GUST-E0753.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ManifestError {
    /// 1-based line in `Cargo.toml`; 0 when the error has no place (an unreadable file).
    pub line: usize,
    /// 1-based column in characters; 0 with `line`.
    pub column: usize,
    /// What is wrong, without the place.
    pub message: String,
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(formatter, "{FILE}: {}", self.message)
        } else {
            write!(
                formatter,
                "{FILE}:{}:{}: {}",
                self.line, self.column, self.message
            )
        }
    }
}

/// Reads the declarations of the crate whose manifest directory is `manifest_dir`. `Ok(None)` when there is no
/// directory or no `Cargo.toml` in it (a unit test's scratch crate; a real build always has one), so the caller
/// emits no tracking for it.
pub(crate) fn read_dir(manifest_dir: Option<&Path>) -> Result<Option<Declarations>, ManifestError> {
    let Some(directory) = manifest_dir else {
        return Ok(None);
    };
    let path = directory.join(FILE);
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).map_err(|error| ManifestError {
        line: 0,
        column: 0,
        message: format!("the file cannot be read ({})", error.kind()),
    })?;
    read(&text).map(Some)
}

/// The declarations in the manifest text `text`.
pub(crate) fn read(text: &str) -> Result<Declarations, ManifestError> {
    let document = DeTable::parse(text).map_err(|error| {
        place(
            text,
            error.span().map(|span| span.start),
            format!("not valid TOML: {}", error.message()),
        )
    })?;
    let root = document.get_ref();
    let Some(gust) = descend(root, &["package", "metadata", "gust"]) else {
        return Ok(Declarations::default());
    };
    let table = as_table(text, gust, "[package.metadata.gust]")?;
    let mut declarations = Declarations::default();
    for (key, value) in table.iter() {
        match name_of(key) {
            "profiles" => {
                declarations.profiles = Some(profiles(text, key.span().start, value)?);
            }
            "limits" => declarations.limits = Some(limits(text, value)?),
            "routes" => declarations.routes = routes(text, value)?,
            other => {
                return Err(place(
                    text,
                    Some(key.span().start),
                    format!(
                        "unknown key `{other}` in [package.metadata.gust]; the keys are {}",
                        KEYS.join(", ")
                    ),
                ));
            }
        }
    }
    Ok(declarations)
}

/// The tokens that make cargo track the manifest: any edit of `Cargo.toml` re-expands the module (the declarations
/// change what a module expands to, and cargo does not track a manifest that no `include_str!` names; plan section 1,
/// S0 (g)). The path is the crate's directory read by the compiler, never a literal.
pub(crate) fn tracking() -> proc_macro2::TokenStream {
    quote::quote! {
        const _: &::core::primitive::str =
            ::core::include_str!(::core::concat!(::core::env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
    }
}

/// `table["a"]["b"]..` of the document, through tables only.
fn descend<'a, 'i>(table: &'a DeTable<'i>, path: &[&str]) -> Option<&'a Spanned<DeValue<'i>>> {
    let (first, rest) = path.split_first()?;
    let (_, value) = find(table, first)?;
    if rest.is_empty() {
        return Some(value);
    }
    descend(value.get_ref().as_table()?, rest)
}

/// The entry named `name` of `table`.
fn find<'a, 'i>(
    table: &'a DeTable<'i>,
    name: &str,
) -> Option<(&'a Spanned<DeString<'i>>, &'a Spanned<DeValue<'i>>)> {
    table.iter().find(|(key, _)| name_of(key) == name)
}

/// The text of a key.
fn name_of<'a>(key: &'a Spanned<DeString<'_>>) -> &'a str {
    key.get_ref()
}

/// The error at the byte offset `offset` of `text`.
fn place(text: &str, offset: Option<usize>, message: String) -> ManifestError {
    let Some(offset) = offset else {
        return ManifestError {
            line: 0,
            column: 0,
            message,
        };
    };
    let offset = offset.min(text.len());
    // A span can end inside a multi-byte character only if the parser were wrong; stay on a boundary anyway.
    let mut at = offset;
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    let before = &text[..at];
    let line = before.matches('\n').count() + 1;
    let column = before
        .rsplit('\n')
        .next()
        .map_or(0, |last| last.chars().count())
        + 1;
    ManifestError {
        line,
        column,
        message,
    }
}

/// The error at `value` for a value of the wrong kind.
fn wrong_kind(text: &str, value: &Spanned<DeValue<'_>>, wanted: &str, what: &str) -> ManifestError {
    place(
        text,
        Some(value.span().start),
        format!(
            "{what} must be {wanted}, found {}",
            value.get_ref().type_str()
        ),
    )
}

fn as_table<'a, 'i>(
    text: &str,
    value: &'a Spanned<DeValue<'i>>,
    what: &str,
) -> Result<&'a DeTable<'i>, ManifestError> {
    value
        .get_ref()
        .as_table()
        .ok_or_else(|| wrong_kind(text, value, "a table", what))
}

/// An array of strings, each with its own place.
fn strings<'a>(
    text: &str,
    value: &'a Spanned<DeValue<'_>>,
    what: &str,
) -> Result<Vec<(&'a str, usize)>, ManifestError> {
    let array = value
        .get_ref()
        .as_array()
        .ok_or_else(|| wrong_kind(text, value, "an array of strings", what))?;
    array
        .iter()
        .map(|item| {
            item.get_ref()
                .as_str()
                .map(|string| (string, item.span().start))
                .ok_or_else(|| wrong_kind(text, item, "a string", &format!("each entry of {what}")))
        })
        .collect()
}

/// `profiles`: a non-empty array of distinct strings, each a profile `GUST_PROFILE` accepts, none of them a `dev`
/// profile (D138: a dev build is checked as `native` with the same capabilities, so the list names `native...`).
/// `key` is the offset of the key, where an empty list is reported; the other refusals are at the entry.
fn profiles(
    text: &str,
    key: usize,
    value: &Spanned<DeValue<'_>>,
) -> Result<Vec<String>, ManifestError> {
    let entries = strings(text, value, "profiles")?;
    if entries.is_empty() {
        return Err(place(
            text,
            Some(key),
            "an empty list declares no build; omit the key for the default".to_owned(),
        ));
    }
    let mut declared: Vec<String> = Vec::new();
    for (name, offset) in entries {
        let Some(parsed) = Profile::parse(name) else {
            return Err(place(
                text,
                Some(offset),
                format!(
                    "`{name}` is not a build profile; write <target>[+<capability>] with the target native, web or cpu and the capability subgroups (web and cpu take none)"
                ),
            ));
        };
        if parsed.target == Some(ProfileTarget::Dev) {
            return Err(place(
                text,
                Some(offset),
                "a dev build is checked as `native` with the same capabilities; list `native...`"
                    .to_owned(),
            ));
        }
        if declared.iter().any(|earlier| earlier == name) {
            return Err(place(
                text,
                Some(offset),
                format!("`{name}` is listed twice in profiles"),
            ));
        }
        declared.push(name.to_owned());
    }
    Ok(declared)
}

/// `limits`: the string `"portable"`.
fn limits(text: &str, value: &Spanned<DeValue<'_>>) -> Result<Limits, ManifestError> {
    match value.get_ref().as_str() {
        Some(PORTABLE) => Ok(Limits::Portable),
        Some(other) => Err(place(
            text,
            Some(value.span().start),
            format!("limits is \"{other}\", but the one value is \"{PORTABLE}\""),
        )),
        None => Err(wrong_kind(text, value, "the string \"portable\"", "limits")),
    }
}

/// `routes`: a table of tags; each tag is a table of `targets` (an array of `gpu` and `cpu`) and `default`.
fn routes(text: &str, value: &Spanned<DeValue<'_>>) -> Result<Vec<Route>, ManifestError> {
    let table = as_table(text, value, "routes")?;
    let mut routes = Vec::new();
    for (tag, entry) in table.iter() {
        let tag_name = name_of(tag);
        let fields = entry.get_ref().as_table().ok_or_else(|| {
            wrong_kind(
                text,
                entry,
                "a table such as { targets = [\"gpu\", \"cpu\"], default = \"gpu\" }",
                &format!("the route `{tag_name}`"),
            )
        })?;
        let mut route = Route {
            tag: tag_name.to_owned(),
            targets: None,
            default: None,
        };
        for (key, field) in fields.iter() {
            match name_of(key) {
                "targets" => {
                    let names = strings(text, field, "targets")?;
                    for (name, offset) in &names {
                        route_target(text, name, *offset)?;
                    }
                    route.targets =
                        Some(names.into_iter().map(|(name, _)| name.to_owned()).collect());
                }
                "default" => {
                    let name = field
                        .get_ref()
                        .as_str()
                        .ok_or_else(|| wrong_kind(text, field, "\"gpu\" or \"cpu\"", "default"))?;
                    route_target(text, name, field.span().start)?;
                    route.default = Some(name.to_owned());
                }
                other if field.get_ref().is_table() => {
                    return Err(place(
                        text,
                        Some(key.span().start),
                        format!(
                            "`{tag_name}.{other}` is a table inside the route `{tag_name}`; a dotted tag is written as a quoted key, \"{tag_name}.{other}\""
                        ),
                    ));
                }
                other => {
                    return Err(place(
                        text,
                        Some(key.span().start),
                        format!(
                            "unknown key `{other}` in the route `{tag_name}`; the keys are targets and default"
                        ),
                    ));
                }
            }
        }
        routes.push(route);
    }
    Ok(routes)
}

fn route_target(text: &str, name: &str, offset: usize) -> Result<(), ManifestError> {
    if ROUTE_TARGETS.contains(&name) {
        Ok(())
    } else {
        Err(place(
            text,
            Some(offset),
            format!("`{name}` is not a route target; the targets are gpu and cpu"),
        ))
    }
}

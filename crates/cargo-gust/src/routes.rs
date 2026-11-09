//! The routing driver (M8, slice m8.r2; D65 to D67, D137, D144).
//!
//! An application chooses the target of each route tag in a `gust.toml` beside its manifest:
//!
//! ```toml
//! [routes]
//! "kernel-lib::physics" = "cpu"                                # <package>::<tag> = "gpu" | "cpu"
//! "kernel-lib::physics.broadphase" = { native = "gpu", web = "cpu" }   # keyed by the build profile
//! ```
//!
//! `cargo gust build`, `check`, `run` and `test` read it when it exists beside the selected package (without it the
//! driver does nothing, which is what plain `cargo` does: the macro then resolves the tags from the source defaults).
//! The tags themselves are declared in the `[package.metadata.gust.routes]` tables of the packages of the build, which
//! `cargo metadata` reports, inline or in the header form `cargo package` writes. The driver validates the file
//! against those declarations, resolves every declared tag the way the `#[gpu]` macro does, sets one environment
//! variable per tag (`GUST_ROUTE__<package>__<tag>`, spelled by [`gust::routes::variable_name`]) on the cargo it
//! starts, and writes the resolved table to `gust-artifacts/routes.json`.
//!
//! The resolution, nearest first (D66, D137 as amended): the `cpu` profile sends every tag to the CPU (an entry that
//! says `gpu` is an error before cargo builds, and the tag's `targets` are not consulted); else the nearest entry of
//! the tag or one of its declared ancestors that has a decision for the build's profile; else the nearest declared
//! `default`; else the GPU. A target-keyed entry uses the key of the profile (`dev`, `native`, `web`); a `dev` build
//! uses its `dev` key, else the `native` one (D138: a dev build is checked as `native`). The result must lie in the
//! tag's `targets` (both targets when the key is absent). A route that names a capability profile is refused: it is out
//! of M8.
//!
//! What the macro knows and this driver cannot: the per-kernel set (a kernel declared `cpu = "unsupported"` under a tag
//! with no `targets` allows only the GPU), so the macro's own E0749 stays the backstop; and the true profile of a
//! `run` or `test` whose profile cargo infers (a custom cargo profile, `build.target` or `CARGO_BUILD_TARGET`): the
//! driver reads `GUST_PROFILE`, else `--release`, `--profile release` and `--target wasm32-*`, as `gust`'s build script
//! does for those cases. It gives the profile it read to cargo as `GUST_ROUTED_PROFILE`, and the `#[gpu]` expansion of a
//! routed module fails the compilation of that crate when `gust` was built for another profile (m8.z).
//!
//! The `gust.toml` reader is a small subset of TOML (a `[routes]` table, quoted keys, strings and one-line inline
//! tables), written here so that the tool keeps depending on nothing but `gust`; anything else is an error that
//! names the line.

use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use gust::routes::{VARIABLE_PREFIX, is_valid_tag, variable_name};

use crate::{
    collect::{Package, Workspace, path_key},
    commands::ARTIFACTS_DIR,
    doctor,
    json::{self, Json},
};

/// The routing file beside the application's manifest.
pub(crate) const CONFIG_FILE: &str = "gust.toml";

/// The resolved table in the artifact directory of the selected package.
pub(crate) const ROUTES_FILE: &str = "routes.json";

/// The two route targets.
const TARGETS: [&str; 2] = ["gpu", "cpu"];

/// The variable the driver sets, beside the route variables, to the profile it resolved them for (m8.z). The `#[gpu]`
/// expansion of a module with a route tag compares it with the profile `gust` was built for and fails the compilation
/// of that crate when they differ, because the driver reads its command line and `GUST_PROFILE` only, and the build
/// also sees `build.target`, `CARGO_BUILD_TARGET` and custom profiles. The name is the macro's
/// (`ROUTED_PROFILE_VARIABLE` in `crates/gust-macros/src/expand/mod.rs`; the macro cannot depend on this tool, so a
/// test reads the source); it does not start with `GUST_ROUTE__`, so it is neither a route variable nor reported by
/// [`RoutePlan::notes`].
pub(crate) const ROUTED_PROFILE_VARIABLE: &str = "GUST_ROUTED_PROFILE";

/// Removes [`ROUTED_PROFILE_VARIABLE`] from a command that applies no routes: only a driver run that applies a plan sets
/// it, so one inherited from an outer `cargo gust run` or `test` (a program that starts cargo again) is stale, and the
/// guard would refuse a build of another profile.
pub(crate) fn without_routed_profile(command: &mut Command) {
    command.env_remove(ROUTED_PROFILE_VARIABLE);
}

/// The profiles a target-keyed entry may name.
const PLATFORM_KEYS: [&str; 3] = ["dev", "native", "web"];

// ---------------------------------------------------------------------------------------------------------------
// gust.toml

/// What an entry says: one target for every profile, or one per profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Value {
    /// `"cpu"`.
    Target(String),
    /// `{ native = "gpu", web = "cpu" }`, in source order.
    ByProfile(Vec<(String, String)>),
}

/// One line of the `[routes]` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    /// 1-based line of the key.
    pub(crate) line: usize,
    /// The key as written: `<package>::<tag>`.
    pub(crate) key: String,
    pub(crate) value: Value,
}

impl Entry {
    /// The target this entry decides for a build of `profile`, if it decides one.
    fn decision(&self, profile: &str) -> Option<&str> {
        match &self.value {
            Value::Target(target) => Some(target),
            Value::ByProfile(pairs) => {
                let find = |key: &str| {
                    pairs
                        .iter()
                        .find(|(name, _)| name == key)
                        .map(|(_, target)| target.as_str())
                };
                match profile {
                    "dev" => find("dev").or_else(|| find("native")),
                    "cpu" => None,
                    other => find(other),
                }
            }
        }
    }

    /// Every target the entry names, for the checks that do not depend on the profile.
    fn targets(&self) -> Vec<&str> {
        match &self.value {
            Value::Target(target) => vec![target.as_str()],
            Value::ByProfile(pairs) => pairs.iter().map(|(_, target)| target.as_str()).collect(),
        }
    }
}

/// A read `gust.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Config {
    pub(crate) entries: Vec<Entry>,
}

/// A position in one line of text.
struct Cursor<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, at: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.text[self.at..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let next = self.peek()?;
        self.at += next.len_utf8();
        Some(next)
    }

    fn skip_spaces(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.at += 1;
        }
    }

    /// Whether only spaces and a comment are left.
    fn at_end(&mut self) -> bool {
        self.skip_spaces();
        matches!(self.peek(), None | Some('#'))
    }

    /// A basic (`"..."`, no escapes) or literal (`'...'`) string.
    fn string(&mut self) -> Result<String, String> {
        let quote = self.bump().unwrap_or('"');
        let rest = self.rest();
        let Some(end) = rest.find(quote) else {
            return Err("a string is not closed on its line".to_owned());
        };
        let text = &rest[..end];
        if quote == '"' && text.contains('\\') {
            return Err(
                "an escape in a string is not supported in gust.toml; write the text between single quotes"
                    .to_owned(),
            );
        }
        self.at += end + 1;
        Ok(text.to_owned())
    }

    /// A bare key of an inline table (letters, digits, `_`, `-` and `+`, so that `native+subgroups` reaches its own
    /// error), or a quoted one.
    fn inline_key(&mut self) -> Result<String, String> {
        match self.peek() {
            Some('"' | '\'') => self.string(),
            Some(c) if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '+') => {
                let start = self.at;
                while self
                    .peek()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '+'))
                {
                    self.bump();
                }
                Ok(self.text[start..self.at].to_owned())
            }
            _ => Err(format!(
                "expected a profile name such as native or web, found `{}`",
                self.rest().trim_end()
            )),
        }
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some('"' | '\'') => self.string().map(Value::Target),
            Some('{') => {
                self.bump();
                let mut pairs: Vec<(String, String)> = Vec::new();
                loop {
                    self.skip_spaces();
                    if self.peek() == Some('}') {
                        self.bump();
                        break;
                    }
                    let key = self.inline_key()?;
                    self.skip_spaces();
                    if self.bump() != Some('=') {
                        return Err(format!("expected `=` after `{key}`"));
                    }
                    self.skip_spaces();
                    if !matches!(self.peek(), Some('"' | '\'')) {
                        return Err(format!(
                            "the value of `{key}` must be a string, \"gpu\" or \"cpu\""
                        ));
                    }
                    let target = self.string()?;
                    if pairs.iter().any(|(name, _)| *name == key) {
                        return Err(format!("`{key}` is set twice in this table"));
                    }
                    pairs.push((key, target));
                    self.skip_spaces();
                    match self.peek() {
                        Some(',') => {
                            self.bump();
                        }
                        Some('}') => {}
                        _ => return Err("expected `,` or `}` in the table".to_owned()),
                    }
                }
                if pairs.is_empty() {
                    return Err(
                        "an empty table decides nothing; write { native = \"gpu\", web = \"cpu\" }"
                            .to_owned(),
                    );
                }
                Ok(Value::ByProfile(pairs))
            }
            _ => Err(format!(
                "expected \"gpu\", \"cpu\" or a table such as {{ native = \"gpu\", web = \"cpu\" }}, found `{}`",
                self.rest().trim_end()
            )),
        }
    }
}

/// Reads `gust.toml`. `file` names it in the messages (`file:line: ...`).
pub(crate) fn parse_config(text: &str, file: &str) -> Result<Config, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut entries: Vec<Entry> = Vec::new();
    let mut in_routes = false;
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let fail = |message: String| format!("{file}:{line}: {message}");
        let mut cursor = Cursor::new(raw);
        if cursor.at_end() {
            continue;
        }
        if cursor.peek() == Some('[') {
            let rest = cursor.rest();
            let Some(close) = rest.find(']') else {
                return Err(fail("a table header is not closed".to_owned()));
            };
            let name = rest[1..close].trim();
            if rest.starts_with("[[") {
                return Err(fail(
                    "an array of tables is not supported; the only table of gust.toml is [routes]"
                        .to_owned(),
                ));
            }
            if name != "routes" {
                return Err(fail(format!(
                    "unknown table `{name}`; the only table of gust.toml is [routes]"
                )));
            }
            cursor.at += close + 1;
            if !cursor.at_end() {
                return Err(fail("unexpected text after the table header".to_owned()));
            }
            if in_routes {
                return Err(fail("the [routes] table appears twice".to_owned()));
            }
            in_routes = true;
            continue;
        }
        if !in_routes {
            return Err(fail(
                "an entry belongs under the [routes] header; write [routes] first".to_owned(),
            ));
        }
        let key = match cursor.peek() {
            Some('"' | '\'') => cursor.string().map_err(&fail)?,
            _ => {
                let bare: String = cursor
                    .rest()
                    .chars()
                    .take_while(|c| !c.is_whitespace() && *c != '=')
                    .collect();
                return Err(fail(format!(
                    "the key `{bare}` must be quoted, as \"<package>::<tag>\""
                )));
            }
        };
        cursor.skip_spaces();
        if cursor.bump() != Some('=') {
            return Err(fail(format!("expected `=` after \"{key}\"")));
        }
        cursor.skip_spaces();
        let value = cursor.value().map_err(&fail)?;
        if !cursor.at_end() {
            return Err(fail(format!(
                "unexpected `{}` after the value",
                cursor.rest().trim_end()
            )));
        }
        if let Some(first) = entries.iter().find(|entry| entry.key == key) {
            return Err(fail(format!(
                "tag `{key}`: this entry repeats the one on line {}",
                first.line
            )));
        }
        entries.push(Entry { line, key, value });
    }
    Ok(Config { entries })
}

// ---------------------------------------------------------------------------------------------------------------
// The declarations: cargo metadata

/// One tag a package declares in `[package.metadata.gust.routes]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Declared {
    pub(crate) tag: String,
    /// `targets`; `None` when the key is absent.
    pub(crate) targets: Option<Vec<String>>,
    /// `default`; `None` when the key is absent.
    pub(crate) default: Option<String>,
    /// 1-based line in the package's `Cargo.toml`, found by searching its text (`cargo metadata` keeps no spans).
    pub(crate) line: Option<usize>,
}

/// A package of the build that declares tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Declaring {
    /// The Cargo package name, as `cargo metadata` reports it.
    pub(crate) name: String,
    pub(crate) manifest: PathBuf,
    /// Sorted by tag.
    pub(crate) tags: Vec<Declared>,
}

impl Declaring {
    /// `Cargo.toml:LINE` with the manifest's path, or the path alone when the line is unknown.
    fn place(&self, declared: &Declared) -> String {
        match declared.line {
            Some(line) => format!("{}:{line}", self.manifest.display()),
            None => self.manifest.display().to_string(),
        }
    }
}

/// The packages of the dependency graph of the selected package (itself included) and what they declare.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Graph {
    /// Every package name of the graph, sorted.
    pub(crate) packages: Vec<String>,
    /// The ones that declare at least one tag, sorted by name then manifest.
    pub(crate) declaring: Vec<Declaring>,
}

/// The line of `Cargo.toml` that declares `tag` in a `routes` table: a `tag = ...` or `"tag" = ...` key, or a
/// `[package.metadata.gust.routes.tag]` header. A text search, best effort: `cargo metadata` gives no spans.
pub(crate) fn declaration_line(manifest: &str, tag: &str) -> Option<usize> {
    let quoted = format!("\"{tag}\"");
    let single = format!("'{tag}'");
    let is_key = |trimmed: &str| {
        [tag, quoted.as_str(), single.as_str()].iter().any(|key| {
            trimmed
                .strip_prefix(key)
                .is_some_and(|rest| rest.trim_start().starts_with('='))
        })
    };
    let is_header = |trimmed: &str| {
        let inner = trimmed.trim_start_matches('[').trim_end_matches(']').trim();
        inner.contains("gust.routes.")
            && [tag, quoted.as_str(), single.as_str()]
                .iter()
                .any(|key| inner.ends_with(&format!("gust.routes.{key}")))
    };
    let mut in_routes = false;
    for (index, line) in manifest.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            if is_header(trimmed) {
                return Some(index + 1);
            }
            in_routes = trimmed.trim_matches(['[', ']']).trim() == "package.metadata.gust.routes";
            continue;
        }
        if in_routes && is_key(trimmed) {
            return Some(index + 1);
        }
    }
    None
}

/// Reads the output of `cargo metadata --format-version 1` (with the dependencies): the closure of the package whose
/// manifest is `selected` over `resolve.nodes`, and the tags each package of it declares. `manifest_text` reads a
/// manifest for the line of a declaration.
pub(crate) fn read_graph(
    metadata: &str,
    selected: &Path,
    manifest_text: &dyn Fn(&Path) -> Option<String>,
) -> Result<Graph, String> {
    let root = json::parse(metadata).map_err(|error| format!("cargo metadata: {error}"))?;
    let packages = root
        .get("packages")
        .and_then(Json::as_array)
        .ok_or("cargo metadata: no `packages` array")?;
    let nodes = root
        .get("resolve")
        .and_then(|resolve| resolve.get("nodes"))
        .and_then(Json::as_array)
        .ok_or("cargo metadata: no `resolve.nodes`; the dependency graph is missing")?;
    let id_of = |package: &Json| package.get("id").and_then(Json::as_str).map(str::to_owned);
    let wanted = path_key(selected);
    let start = packages
        .iter()
        .find(|package| {
            package
                .get("manifest_path")
                .and_then(Json::as_str)
                .is_some_and(|path| path_key(Path::new(path)) == wanted)
        })
        .and_then(id_of)
        .ok_or_else(|| {
            "cargo metadata: the selected package is not in the dependency graph".to_owned()
        })?;
    let mut edges: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in nodes {
        let Some(id) = node.get("id").and_then(Json::as_str) else {
            continue;
        };
        let dependencies = node
            .get("dependencies")
            .and_then(Json::as_array)
            .map(|list| list.iter().filter_map(Json::as_str).collect())
            .unwrap_or_default();
        edges.insert(id, dependencies);
    }
    let mut seen: Vec<&str> = vec![start.as_str()];
    let mut next = 0;
    while let Some(&id) = seen.get(next) {
        next += 1;
        for dependency in edges.get(id).into_iter().flatten() {
            if !seen.contains(dependency) {
                seen.push(dependency);
            }
        }
    }
    let mut graph = Graph::default();
    for package in packages {
        let Some(id) = id_of(package) else { continue };
        if !seen.contains(&id.as_str()) {
            continue;
        }
        let name = package
            .get("name")
            .and_then(Json::as_str)
            .ok_or("cargo metadata: a package has no `name`")?;
        let manifest = PathBuf::from(
            package
                .get("manifest_path")
                .and_then(Json::as_str)
                .ok_or_else(|| {
                    format!("cargo metadata: package `{name}` has no `manifest_path`")
                })?,
        );
        graph.packages.push(name.to_owned());
        let routes = package
            .get("metadata")
            .and_then(|metadata| metadata.get("gust"))
            .and_then(|gust| gust.get("routes"));
        let Some(routes) = routes.filter(|routes| !routes.is_null()) else {
            continue;
        };
        let text = manifest_text(&manifest);
        let mut declaring = Declaring {
            name: name.to_owned(),
            manifest,
            tags: Vec::new(),
        };
        let Json::Object(members) = routes else {
            return Err(format!(
                "{}: `[package.metadata.gust.routes]` of package `{name}` must be a table of tags",
                declaring.manifest.display()
            ));
        };
        for (tag, value) in members {
            let line = text.as_deref().and_then(|text| declaration_line(text, tag));
            let mut declared = Declared {
                tag: tag.clone(),
                targets: None,
                default: None,
                line,
            };
            let place = declaring.place(&declared);
            let Json::Object(fields) = value else {
                return Err(format!(
                    "{place}: tag `{name}::{tag}`: a route is a table such as {{ targets = [\"gpu\", \"cpu\"], default = \"gpu\" }}"
                ));
            };
            for (key, field) in fields {
                match key.as_str() {
                    "targets" => {
                        let items = field.as_array().ok_or_else(|| {
                            format!("{place}: tag `{name}::{tag}`: `targets` must be an array of strings")
                        })?;
                        let mut names = Vec::new();
                        for item in items {
                            names.push(
                                item.as_str()
                                    .ok_or_else(|| {
                                        format!("{place}: tag `{name}::{tag}`: `targets` must be an array of strings")
                                    })?
                                    .to_owned(),
                            );
                        }
                        declared.targets = Some(names);
                    }
                    "default" => {
                        declared.default = Some(
                            field
                                .as_str()
                                .ok_or_else(|| {
                                    format!("{place}: tag `{name}::{tag}`: `default` must be \"gpu\" or \"cpu\"")
                                })?
                                .to_owned(),
                        );
                    }
                    other if matches!(field, Json::Object(_)) => {
                        return Err(format!(
                            "{place}: tag `{name}::{tag}`: `{tag}.{other}` is a table inside the route `{tag}`; a dotted tag is written as a quoted key, \"{tag}.{other}\""
                        ));
                    }
                    other => {
                        return Err(format!(
                            "{place}: tag `{name}::{tag}`: unknown key `{other}` in the route `{tag}`; the keys are targets and default"
                        ));
                    }
                }
            }
            declaring.tags.push(declared);
        }
        declaring.tags.sort_by(|a, b| a.tag.cmp(&b.tag));
        if !declaring.tags.is_empty() {
            graph.declaring.push(declaring);
        }
    }
    graph.packages.sort();
    graph.packages.dedup();
    graph.declaring.sort_by(|a, b| {
        (a.name.as_str(), path_key(&a.manifest)).cmp(&(b.name.as_str(), path_key(&b.manifest)))
    });
    Ok(graph)
}

// ---------------------------------------------------------------------------------------------------------------
// Resolution

/// What decided a tag's target, in the order of the resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Level {
    /// `GUST_PROFILE=cpu`.
    Profile,
    /// A `gust.toml` entry of the tag or of an ancestor.
    Entry,
    /// The `default` of the tag or of an ancestor in its `Cargo.toml`.
    Default,
    /// Nothing said anything: the GPU.
    Fallback,
}

impl Level {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Profile => "profile",
            Self::Entry => "entry",
            Self::Default => "default",
            Self::Fallback => "fallback",
        }
    }
}

/// One resolved tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) package: String,
    pub(crate) tag: String,
    pub(crate) variable: String,
    /// `gpu` or `cpu`.
    pub(crate) target: &'static str,
    pub(crate) level: Level,
    /// The `<package>::<tag>` whose entry or default decided it; `None` at the profile and fallback levels.
    pub(crate) from: Option<String>,
}

/// `a.b.c`, `a.b`, `a`.
fn chain(tag: &str) -> Vec<&str> {
    let mut names = vec![tag];
    let mut name = tag;
    while let Some(dot) = name.rfind('.') {
        name = &name[..dot];
        names.push(name);
    }
    names
}

fn target_name(name: &str) -> Option<&'static str> {
    TARGETS.iter().copied().find(|target| *target == name)
}

/// The declared tags by `(package, tag)`, each with the package that declares it.
type DeclaredMap<'a> = BTreeMap<(String, String), (&'a Declaring, &'a Declared)>;

/// The tags of the graph by `(package, tag)`, checked: well-named, valid `targets` and `default`, and no two that
/// spell one variable. Every declaring package of the build is checked, also one whose tags `gust.toml` never names and
/// no module uses (the macro reads only the tags it expands, so it is lazier): a malformed declaration in any
/// dependency stops a routed command, with its `Cargo.toml` and line. Two versions of one package share their
/// variables, so they are accepted (and count once) only when their whole `routes` tables are equal ([`first_difference`]):
/// an unset variable is resolved by each macro from its own version's chain of ancestor defaults, which one value
/// cannot serve when the chains differ. The spelling is not injective across package names
/// (`gust::routes::variable_name`), which is the other collision.
fn declared_tags(graph: &Graph) -> Result<DeclaredMap<'_>, String> {
    let mut tags: DeclaredMap<'_> = BTreeMap::new();
    let mut variables: BTreeMap<String, String> = BTreeMap::new();
    for (index, package) in graph.declaring.iter().enumerate() {
        if let Some(first) = graph.declaring[..index]
            .iter()
            .find(|earlier| earlier.name == package.name)
        {
            match first_difference(first, package) {
                None => continue,
                Some(tag) => {
                    return Err(format!(
                        "two versions of `{}` ({} and {}) declare `{tag}` differently; one variable, `{}`, routes both; unify the versions",
                        package.name,
                        first.manifest.display(),
                        package.manifest.display(),
                        variable_name(&package.name, &tag)
                    ));
                }
            }
        }
        for declared in &package.tags {
            let place = package.place(declared);
            let full = format!("{}::{}", package.name, declared.tag);
            if !is_valid_tag(&declared.tag) {
                return Err(format!(
                    "{place}: tag `{full}` is ill-named: a tag is segments of [a-z0-9_]+ joined by single `.`, with no `__` and no segment that begins or ends with `_`"
                ));
            }
            if let Some(targets) = &declared.targets {
                if targets.is_empty() {
                    return Err(format!(
                        "{place}: tag `{full}`: `targets` is empty, so no build can route it; list gpu, cpu or both"
                    ));
                }
                if let Some(bad) = targets.iter().find(|name| target_name(name).is_none()) {
                    return Err(format!(
                        "{place}: tag `{full}`: `{bad}` is not a route target; the targets are gpu and cpu"
                    ));
                }
            }
            if let Some(default) = &declared.default
                && target_name(default).is_none()
            {
                return Err(format!(
                    "{place}: tag `{full}`: the default `{default}` is not a route target; the targets are gpu and cpu"
                ));
            }
            let variable = variable_name(&package.name, &declared.tag);
            let folded = variable.to_ascii_uppercase();
            let description = format!("`{full}` ({place})");
            if let Some(first) = variables.get(&folded) {
                return Err(format!(
                    "{place}: tag `{full}` and {first} spell one variable, `{variable}` (compared without case); rename the package or the tag"
                ));
            }
            variables.insert(folded, description);
            tags.insert(
                (package.name.clone(), declared.tag.clone()),
                (package, declared),
            );
        }
    }
    Ok(tags)
}

/// The first tag (in tag order) on which the `routes` tables of two packages differ: declared by one only, or with other
/// `targets` (compared as sets, so the order of the list does not matter) or another `default`.
fn first_difference(first: &Declaring, second: &Declaring) -> Option<String> {
    /// What the comparison looks at: `targets` as a sorted set, and `default`.
    type Shape = (Option<Vec<String>>, Option<String>);
    fn shape(package: &Declaring, tag: &str) -> Option<Shape> {
        let declared = package.tags.iter().find(|declared| declared.tag == tag)?;
        let targets = declared.targets.as_ref().map(|list| {
            let mut names = list.clone();
            names.sort_unstable();
            names.dedup();
            names
        });
        Some((targets, declared.default.clone()))
    }
    let mut tags: Vec<&str> = first
        .tags
        .iter()
        .chain(&second.tags)
        .map(|declared| declared.tag.as_str())
        .collect();
    tags.sort_unstable();
    tags.dedup();
    tags.into_iter()
        .find(|tag| shape(first, tag) != shape(second, tag))
        .map(str::to_owned)
}

/// The set of targets a declared tag allows.
fn allowed(declared: &Declared) -> Vec<&str> {
    match &declared.targets {
        Some(targets) => targets.iter().map(String::as_str).collect(),
        None => TARGETS.to_vec(),
    }
}

fn set_words(declared: &Declared) -> String {
    let names: Vec<String> = allowed(declared)
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect();
    format!("targets = [{}]", names.join(", "))
}

/// Checks the shape of every entry against the declarations and returns them by `(package, tag)`.
fn checked_entries<'a>(
    graph: &Graph,
    declared: &DeclaredMap<'_>,
    config: &'a Config,
    file: &str,
) -> Result<BTreeMap<(String, String), &'a Entry>, String> {
    let mut entries = BTreeMap::new();
    for entry in &config.entries {
        let fail =
            |message: String| format!("{file}:{}: tag `{}`: {message}", entry.line, entry.key);
        let Some((package, tag)) = entry
            .key
            .split_once("::")
            .filter(|(package, tag)| !package.is_empty() && !tag.is_empty())
        else {
            return Err(fail(
                "the key must be `<package>::<tag>`, for example \"kernel-lib::physics\""
                    .to_owned(),
            ));
        };
        if !declared.contains_key(&(package.to_owned(), tag.to_owned())) {
            let own: Vec<String> = declared
                .keys()
                .filter(|(name, _)| name == package)
                .map(|(_, tag)| format!("`{tag}`"))
                .collect();
            return Err(fail(
                if !graph.packages.iter().any(|name| name == package) {
                    format!(
                        "unknown tag: `{package}` is not a package of this build's dependency graph"
                    )
                } else if own.is_empty() {
                    format!(
                        "unknown tag: `{package}` declares no route tags in [package.metadata.gust.routes]"
                    )
                } else if !is_valid_tag(tag) {
                    format!(
                        "unknown tag: `{tag}` is ill-named, and `{package}` declares {}",
                        own.join(", ")
                    )
                } else {
                    format!("unknown tag: `{package}` declares {}", own.join(", "))
                },
            ));
        }
        let pairs: Vec<(Option<&str>, &str)> = match &entry.value {
            Value::Target(target) => vec![(None, target.as_str())],
            Value::ByProfile(pairs) => pairs
                .iter()
                .map(|(key, target)| (Some(key.as_str()), target.as_str()))
                .collect(),
        };
        for (key, target) in pairs {
            if let Some(key) = key {
                if key.contains('+') {
                    return Err(fail(format!(
                        "`{key}` names a capability profile; a route that names one is not supported in this release (key the entry by native, web or dev)"
                    )));
                }
                if key == "cpu" {
                    return Err(fail(
                        "a `cpu` key has nothing to decide: the cpu profile routes every tag to the CPU; remove it"
                            .to_owned(),
                    ));
                }
                if !PLATFORM_KEYS.contains(&key) {
                    return Err(fail(format!(
                        "`{key}` is not a profile key; the keys are dev, native and web"
                    )));
                }
            }
            if target_name(target).is_none() {
                return Err(fail(if target.contains(['+', ':']) {
                    format!(
                        "`{target}` names a capability profile; a route that names one is not supported in this release (the targets are gpu and cpu)"
                    )
                } else {
                    format!("`{target}` is not a route target; the targets are gpu and cpu")
                }));
            }
        }
        entries.insert((package.to_owned(), tag.to_owned()), entry);
    }
    Ok(entries)
}

/// Resolves every declared tag of the graph for a build of `profile` (`dev`, `native`, `web` or `cpu`).
///
/// `file` is the path of `gust.toml` as the messages print it. Rows are sorted by package and tag.
pub(crate) fn resolve(
    graph: &Graph,
    config: &Config,
    file: &str,
    profile: &str,
) -> Result<Vec<Row>, String> {
    let declared = declared_tags(graph)?;
    let entries = checked_entries(graph, &declared, config, file)?;
    let cpu = profile == "cpu";
    let mut rows = Vec::new();
    for ((package, tag), (owner_of_tag, info)) in &declared {
        let full = format!("{package}::{tag}");
        let variable = variable_name(package, tag);
        // The nearest entry with a decision for this build.
        let decision = chain(tag).into_iter().find_map(|name| {
            entries
                .get(&(package.clone(), name.to_owned()))
                .and_then(|entry| entry.decision(profile).map(|target| (*entry, target)))
        });
        if cpu {
            if let Some((entry, "gpu")) = decision {
                return Err(format!(
                    "{file}:{}: tag `{full}`: GUST_PROFILE=cpu builds the CPU target only, but the entry `{}` routes it to `gpu`{}; remove the entry or route it to `cpu`",
                    entry.line,
                    entry.key,
                    if entry.key == full {
                        String::new()
                    } else {
                        " by inheritance".to_owned()
                    }
                ));
            }
            rows.push(Row {
                package: package.clone(),
                tag: tag.clone(),
                variable,
                target: "cpu",
                level: Level::Profile,
                from: None,
            });
            continue;
        }
        // An entry of this very tag is checked for every profile, not only this one. An entry inherited from an ancestor
        // is checked for this build's profile only, so `"solver" = { web = "cpu" }` passes under `dev` and is refused under
        // `web` for a child that allows only the GPU (a conflict shows when the profile that causes it builds).
        if let Some(own) = entries.get(&(package.clone(), tag.clone())) {
            for target in own.targets() {
                if !allowed(info).contains(&target) {
                    return Err(format!(
                        "{file}:{}: tag `{full}`: routed to `{target}`, but the tag allows only {} (declared in its Cargo.toml)",
                        own.line,
                        set_words(info)
                    ));
                }
            }
        }
        if let Some((entry, target)) = decision {
            if !allowed(info).contains(&target) {
                return Err(format!(
                    "{file}:{}: tag `{full}`: the entry `{}` routes it to `{target}` by inheritance, but the tag allows only {}; give `{tag}` an entry of its own or widen its targets",
                    entry.line,
                    entry.key,
                    set_words(info)
                ));
            }
            rows.push(Row {
                package: package.clone(),
                tag: tag.clone(),
                variable,
                target: target_name(target).unwrap_or("gpu"),
                level: Level::Entry,
                from: Some(entry.key.clone()),
            });
            continue;
        }
        // The source default of the tag, else of its nearest ancestor that has one.
        let source = chain(tag).into_iter().find_map(|name| {
            declared
                .get(&(package.clone(), name.to_owned()))
                .and_then(|(owner, owned)| {
                    owned
                        .default
                        .as_deref()
                        .map(|default| (*owner, *owned, default))
                })
        });
        if let Some((owner, owned, default)) = source {
            if !allowed(info).contains(&default) {
                return Err(format!(
                    "{}: tag `{full}`: the default `{default}` of `{package}::{}` routes it to `{default}`, but the tag allows only {}",
                    owner.place(owned),
                    owned.tag,
                    set_words(info)
                ));
            }
            rows.push(Row {
                package: package.clone(),
                tag: tag.clone(),
                variable,
                target: target_name(default).unwrap_or("gpu"),
                level: Level::Default,
                from: Some(format!("{package}::{}", owned.tag)),
            });
            continue;
        }
        if !allowed(info).contains(&"gpu") {
            return Err(format!(
                "{}: tag `{full}`: no entry and no default routes it, which leaves the GPU, but the tag allows only {}; add a `default` to the declaration, or an entry for it in {file}",
                owner_of_tag.place(info),
                set_words(info)
            ));
        }
        rows.push(Row {
            package: package.clone(),
            tag: tag.clone(),
            variable,
            target: "gpu",
            level: Level::Fallback,
            from: None,
        });
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------------------------------------------
// The plan

/// A resolved routing: the variables to set and the table to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutePlan {
    /// The build profile the table was resolved for.
    pub(crate) profile: String,
    pub(crate) rows: Vec<Row>,
    /// `<package>/gust-artifacts`, where `routes.json` goes.
    artifacts: PathBuf,
}

impl RoutePlan {
    pub(crate) fn new(profile: &str, rows: Vec<Row>, artifacts: PathBuf) -> Self {
        Self {
            profile: profile.to_owned(),
            rows,
            artifacts,
        }
    }

    /// `(variable, target)` for every declared tag.
    pub(crate) fn variables(&self) -> Vec<(String, String)> {
        self.rows
            .iter()
            .map(|row| (row.variable.clone(), row.target.to_owned()))
            .collect()
    }

    /// Sets the route variables on a cargo command, and [`ROUTED_PROFILE_VARIABLE`] to the profile they were resolved for
    /// (a preset value is overridden, as the route variables are). Every command that builds goes through here.
    pub(crate) fn apply(&self, command: &mut Command) {
        for (name, value) in self.variables() {
            command.env(name, value);
        }
        command.env(ROUTED_PROFILE_VARIABLE, &self.profile);
    }

    /// `routes.json`: tag, target and deciding level, in the order of the rows, so that a rerun gives the same bytes.
    pub(crate) fn json(&self) -> String {
        let routes = self
            .rows
            .iter()
            .map(|row| {
                Json::object(vec![
                    ("package", Json::text(&row.package)),
                    ("tag", Json::text(&row.tag)),
                    ("variable", Json::text(&row.variable)),
                    ("target", Json::text(row.target)),
                    ("level", Json::text(row.level.name())),
                    ("from", row.from.as_deref().map_or(Json::Null, Json::text)),
                ])
            })
            .collect();
        Json::object(vec![
            ("schema", Json::number(1)),
            ("profile", Json::text(&self.profile)),
            ("routes", Json::Array(routes)),
        ])
        .to_pretty()
    }

    /// Writes `gust-artifacts/routes.json` unless it already holds these bytes.
    pub(crate) fn write(&self) -> Result<(), String> {
        let path = self.artifacts.join(ROUTES_FILE);
        let text = self.json();
        if fs::read(&path).is_ok_and(|bytes| bytes == text.as_bytes()) {
            return Ok(());
        }
        fs::create_dir_all(&self.artifacts)
            .and_then(|()| fs::write(&path, text))
            .map_err(|error| format!("cannot write {}: {error}", path.display()))
    }

    /// Notes on `GUST_ROUTE__*` variables the environment already holds (`preset`: name and value): the driver sets
    /// every declared tag, so a preset that differs is overridden, and one that matches no declared tag (a typo) has no
    /// effect. Names compare without case on Windows only, where environment names are case-insensitive; elsewhere
    /// `gust_route__x` is a variable of its own that no macro reads.
    pub(crate) fn notes(&self, preset: &[(String, String)]) -> Vec<String> {
        let same = |a: &str, b: &str| {
            if cfg!(windows) {
                a.eq_ignore_ascii_case(b)
            } else {
                a == b
            }
        };
        let mut notes = Vec::new();
        for (name, value) in preset {
            let prefix = name.get(..VARIABLE_PREFIX.len());
            if !prefix.is_some_and(|prefix| same(prefix, VARIABLE_PREFIX)) {
                continue;
            }
            match self.rows.iter().find(|row| same(&row.variable, name)) {
                Some(row) if row.target != value => notes.push(format!(
                    "gust: note: {name}={value} is set in the environment; {CONFIG_FILE} resolves it to `{}`, which is what the build gets",
                    row.target
                )),
                Some(_) => {}
                None => notes.push(format!(
                    "gust: note: {name} is set in the environment but matches no route tag that the crates of this build declare; the driver does not use it"
                )),
            }
        }
        notes
    }

    /// One line per tag for the terminal.
    pub(crate) fn summary(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|row| {
                let by = row
                    .from
                    .as_deref()
                    .map_or(String::new(), |from| format!(" `{from}`"));
                format!(
                    "gust: route {}::{} -> {} ({}{by}, profile {})",
                    row.package,
                    row.tag,
                    row.target,
                    row.level.name(),
                    self.profile
                )
            })
            .collect()
    }
}

/// The arguments of a `run` or `test` that change which packages the build uses or whether `Cargo.lock` may change:
/// the feature selection, `--locked`, `--frozen`, `--offline` and `--config` (a `--config` can change the registry, the
/// source replacement or `net.offline`; `-Z` flags are not forwarded). `cargo metadata` gets the same ones, so its graph
/// holds the optional dependencies that `--features` turns on, and `--locked` is not defeated by a metadata call that
/// rewrites a stale lock. Words after `--` belong to the program.
pub(crate) fn metadata_flags(arguments: &[String]) -> Vec<String> {
    let mut flags = Vec::new();
    let mut index = 0;
    while let Some(argument) = arguments.get(index) {
        index += 1;
        match argument.as_str() {
            "--" => break,
            "--all-features" | "--no-default-features" | "--locked" | "--frozen" | "--offline" => {
                flags.push(argument.clone());
            }
            "--features" | "-F" | "--config" => {
                flags.push(argument.clone());
                if let Some(value) = arguments.get(index) {
                    flags.push(value.clone());
                    index += 1;
                }
            }
            other if other.starts_with("--features=") || other.starts_with("--config=") => {
                flags.push(argument.clone());
            }
            other if other.starts_with("-F") && other.len() > 2 => flags.push(argument.clone()),
            _ => {}
        }
    }
    flags
}

/// `cargo metadata --format-version 1 --manifest-path <path>` plus the flags of [`metadata_flags`] for `arguments`,
/// with the dependencies (the declarations of the packages the build uses are in them).
pub(crate) fn metadata_command(
    cargo: &OsString,
    manifest_path: &Path,
    arguments: &[String],
) -> Command {
    let mut command = Command::new(cargo);
    command
        .args(["metadata", "--format-version", "1", "--manifest-path"])
        .arg(manifest_path)
        .args(metadata_flags(arguments));
    command
}

/// Whether a `gust.toml` is beside the manifest of `package`.
pub(crate) fn has_config(package: &Package) -> bool {
    package.directory().join(CONFIG_FILE).is_file()
}

/// The package names given to `--exclude` (`--exclude X` and `--exclude=X`, before any `--`). Names are taken as
/// written: a glob pattern excludes nothing here, which can only make the check below stricter.
pub(crate) fn excluded(arguments: &[String]) -> Vec<String> {
    let mut names = Vec::new();
    let mut index = 0;
    while let Some(argument) = arguments.get(index) {
        index += 1;
        if argument == "--" {
            break;
        }
        if argument == "--exclude" {
            if let Some(value) = arguments.get(index) {
                names.push(value.clone());
                index += 1;
            }
        } else if let Some(value) = argument.strip_prefix("--exclude=") {
            names.push(value.to_owned());
        }
    }
    names
}

/// The packages cargo builds at the workspace root when neither `-p` nor `--workspace` is given: the default members.
/// `None` when `manifest` is not the root manifest, or when the default selection is just the root package itself (the
/// ordinary case, which the driver routes as one package).
pub(crate) fn default_selection<'a>(
    workspace: &'a Workspace,
    manifest: &Path,
) -> Option<&'a [Package]> {
    let defaults = workspace.default_members.as_deref()?;
    let at_root = manifest
        .parent()
        .is_some_and(|directory| path_key(directory) == path_key(&workspace.root));
    if !at_root {
        return None;
    }
    match defaults {
        [only] if path_key(&only.manifest_path) == path_key(manifest) => None,
        _ => Some(defaults),
    }
}

/// `run` and `test` with `--workspace` or `--all` (every member, less the `--exclude`d ones), or at a workspace root
/// whose default selection is several packages ([`default_selection`]), build several packages, but the driver routes
/// one package at a time: a member with a `gust.toml` of its own would be built without its routes, and a test that
/// only checks numbers passes on the wrong path. Refused when any selected member has a `gust.toml`, naming each; when
/// none has one the arguments pass through unchanged.
pub(crate) fn refuse_workspace(members: &[Package], excluded: &[String]) -> Result<(), String> {
    let routed: Vec<String> = members
        .iter()
        .filter(|member| !excluded.contains(&member.name) && has_config(member))
        .map(|member| {
            format!(
                "`{}` ({})",
                member.name,
                member.directory().join(CONFIG_FILE).display()
            )
        })
        .collect();
    if routed.is_empty() {
        return Ok(());
    }
    Err(format!(
        "`--workspace`, `--all` and the default members of a workspace root build several packages, but the driver routes one package at a time and would build {} without its routes; name one package with -p and run the packages one by one, or `--exclude` the members that have a {CONFIG_FILE}",
        routed.join(", ")
    ))
}

/// Removes the `routes.json` of `package`: no `gust.toml` applies, so the table describes routes the build no longer
/// has. A file that is not there is fine.
pub(crate) fn forget(package: &Package) -> Result<(), String> {
    let path = package.directory().join(ARTIFACTS_DIR).join(ROUTES_FILE);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot remove {}: {error}", path.display())),
    }
}

/// The plan for the selected package: `None` when no `gust.toml` is beside its manifest. `profile` is the build's;
/// `arguments` are the user's cargo arguments (see [`metadata_flags`]; `build` and `check` have none).
pub(crate) fn plan(
    cargo: &OsString,
    package: &Package,
    profile: &str,
    arguments: &[String],
) -> Result<Option<RoutePlan>, String> {
    let config_path = package.directory().join(CONFIG_FILE);
    if !config_path.is_file() {
        return Ok(None);
    }
    let file = config_path.display().to_string();
    let text =
        fs::read_to_string(&config_path).map_err(|error| format!("cannot read {file}: {error}"))?;
    let config = parse_config(&text, &file)?;
    let output = metadata_command(cargo, &package.manifest_path, arguments)
        .output()
        .map_err(|error| format!("cannot run `cargo metadata`: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo metadata` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let graph = read_graph(
        &String::from_utf8_lossy(&output.stdout),
        &package.manifest_path,
        &|path| fs::read_to_string(path).ok(),
    )?;
    let rows = resolve(&graph, &config, &file, profile)?;
    Ok(Some(RoutePlan::new(
        profile,
        rows,
        package.directory().join(ARTIFACTS_DIR),
    )))
}

// ---------------------------------------------------------------------------------------------------------------
// run and test

/// The build profile (its target) of a `cargo run` or `cargo test` with these arguments: `GUST_PROFILE` when set,
/// else the one `gust`'s build script infers: `web` for a release build on `wasm32`, `native` for any other release
/// build, `dev` otherwise. Only `--release`, `-r`, `--profile release` and `--target wasm32-*` are looked at.
pub(crate) fn build_profile(
    gust_profile: Option<&str>,
    arguments: &[String],
) -> Result<&'static str, String> {
    let mut release = false;
    let mut wasm = false;
    let mut index = 0;
    while let Some(argument) = arguments.get(index) {
        index += 1;
        if argument == "--" {
            break;
        }
        match argument.as_str() {
            "--release" | "-r" | "--profile=release" => release = true,
            "--profile" if arguments.get(index).is_some_and(|value| value == "release") => {
                release = true;
            }
            "--target" => {
                wasm |= arguments
                    .get(index)
                    .is_some_and(|value| value.starts_with("wasm32"));
            }
            other => {
                wasm |= other
                    .strip_prefix("--target=")
                    .is_some_and(|value| value.starts_with("wasm32"));
            }
        }
    }
    doctor::effective_profile(gust_profile, release, wasm).map(|(profile, _)| profile)
}

/// `cargo <verb> <arguments>` with the route variables of `plan` set. The arguments are the user's, unchanged.
pub(crate) fn cargo_command(
    cargo: &OsString,
    verb: &str,
    arguments: &[String],
    plan: Option<&RoutePlan>,
) -> Command {
    let mut command = Command::new(cargo);
    command.arg(verb).args(arguments);
    if let Some(plan) = plan {
        plan.apply(&mut command);
    } else {
        without_routed_profile(&mut command);
    }
    command
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const FILE: &str = "gust.toml";

    // ---- gust.toml ----

    #[test]
    fn the_forms_of_an_entry_are_read_with_their_lines() {
        let text = "# routes\n\n[routes]  # the table\n\"kernel-lib::physics\" = \"cpu\"\n'kernel-lib::fusion' = 'gpu' # literal\n\"kernel-lib::physics.broadphase\" = { native = \"gpu\", web = 'cpu' }\n";
        let config = parse_config(text, FILE).unwrap();
        assert_eq!(config.entries.len(), 3);
        assert_eq!(config.entries[0].line, 4);
        assert_eq!(config.entries[0].key, "kernel-lib::physics");
        assert_eq!(config.entries[0].value, Value::Target("cpu".to_owned()));
        assert_eq!(config.entries[1].line, 5);
        assert_eq!(config.entries[1].value, Value::Target("gpu".to_owned()));
        assert_eq!(config.entries[2].line, 6);
        assert_eq!(
            config.entries[2].value,
            Value::ByProfile(vec![
                ("native".to_owned(), "gpu".to_owned()),
                ("web".to_owned(), "cpu".to_owned()),
            ])
        );
    }

    #[test]
    fn crlf_a_bom_and_an_empty_file_read() {
        let config = parse_config("\u{feff}[routes]\r\n\"a::b\" = \"cpu\"\r\n", FILE).unwrap();
        assert_eq!(config.entries.len(), 1);
        assert_eq!(config.entries[0].line, 2);
        assert_eq!(parse_config("", FILE).unwrap(), Config::default());
        assert_eq!(parse_config("[routes]\n", FILE).unwrap(), Config::default());
    }

    #[test]
    fn a_syntax_error_names_the_file_and_the_line() {
        let cases: &[(&str, usize, &str)] = &[
            ("[routes]\nphysics = \"cpu\"\n", 2, "must be quoted"),
            ("[routes]\n\"a::b\" \"cpu\"\n", 2, "expected `=`"),
            ("[routes]\n\"a::b\" = cpu\n", 2, "found `cpu`"),
            ("[routes]\n\"a::b\" = \"cpu\n", 2, "not closed"),
            ("[routes]\n\"a::b\" = \"cpu\" x\n", 2, "after the value"),
            ("[routes]\n\"a::b\" = {}\n", 2, "empty table"),
            (
                "[routes]\n\"a::b\" = { native = gpu }\n",
                2,
                "must be a string",
            ),
            (
                "[routes]\n\"a::b\" = { native = \"gpu\" web = \"cpu\" }\n",
                2,
                "expected `,` or `}`",
            ),
            (
                "[routes]\n\"a::b\" = { native = \"gpu\", native = \"cpu\" }\n",
                2,
                "set twice",
            ),
            ("[routes]\n\"a::b\" = \"c\\u0070u\"\n", 2, "escape"),
            (
                "\"a::b\" = \"cpu\"\n",
                1,
                "belongs under the [routes] header",
            ),
            ("[routes]\n[other]\n", 2, "unknown table `other`"),
            ("[routes]\n[[routes]]\n", 2, "array of tables"),
            ("[routes]\n[routes]\n", 2, "appears twice"),
            (
                "[routes]\n\"a::b\" = \"cpu\"\n\"a::b\" = \"gpu\"\n",
                3,
                "repeats the one on line 2",
            ),
            ("[routes\n", 1, "not closed"),
        ];
        for (text, line, phrase) in cases {
            let error = parse_config(text, FILE).unwrap_err();
            assert!(
                error.starts_with(&format!("{FILE}:{line}: ")) && error.contains(phrase),
                "{text:?}: {error}"
            );
        }
    }

    // ---- the declarations ----

    /// A trimmed `cargo metadata` document: `app` depends on `kernel-lib`, which declares tags in the inline form;
    /// `other` is a workspace member that `app` does not use and declares a tag too.
    fn metadata(routes: &str) -> String {
        format!(
            r#"{{"packages":[
{{"id":"app 0.1.0","name":"app","manifest_path":"/w/app/Cargo.toml","metadata":null}},
{{"id":"kernel-lib 0.1.0","name":"kernel-lib","manifest_path":"/w/kernel-lib/Cargo.toml","metadata":{{"gust":{{"routes":{routes}}}}}}},
{{"id":"other 0.1.0","name":"other","manifest_path":"/w/other/Cargo.toml","metadata":{{"gust":{{"routes":{{"x":{{}}}}}}}}}}
],"resolve":{{"nodes":[
{{"id":"app 0.1.0","dependencies":["kernel-lib 0.1.0"]}},
{{"id":"kernel-lib 0.1.0","dependencies":[]}},
{{"id":"other 0.1.0","dependencies":[]}}
]}}}}"#
        )
    }

    const ROUTES: &str = r#"{"physics":{"targets":["gpu","cpu"],"default":"gpu"},"physics.broadphase":{},"fusion":{"default":"cpu"},"render":{"targets":["gpu"]}}"#;

    fn graph_of(routes: &str) -> Graph {
        read_graph(&metadata(routes), Path::new("/w/app/Cargo.toml"), &|_| None).unwrap()
    }

    #[test]
    fn the_graph_is_the_closure_of_the_selected_package() {
        let graph = graph_of(ROUTES);
        assert_eq!(graph.packages, ["app", "kernel-lib"]);
        assert_eq!(
            graph.declaring.len(),
            1,
            "`other` is not a dependency of `app`"
        );
        let kernel_lib = &graph.declaring[0];
        assert_eq!(kernel_lib.name, "kernel-lib");
        let tags: Vec<&str> = kernel_lib.tags.iter().map(|t| t.tag.as_str()).collect();
        assert_eq!(tags, ["fusion", "physics", "physics.broadphase", "render"]);
        let physics = &kernel_lib.tags[1];
        assert_eq!(
            physics.targets,
            Some(vec!["gpu".to_owned(), "cpu".to_owned()])
        );
        assert_eq!(physics.default.as_deref(), Some("gpu"));
        assert_eq!(kernel_lib.tags[0].targets, None);
    }

    #[test]
    fn a_malformed_declaration_is_an_error_with_its_tag() {
        for (routes, phrase) in [
            (r#"{"a":"cpu"}"#, "a route is a table"),
            (r#"{"a":{"targets":"gpu"}}"#, "array of strings"),
            (r#"{"a":{"targets":[1]}}"#, "array of strings"),
            (r#"{"a":{"default":1}}"#, "must be \"gpu\" or \"cpu\""),
            (r#"{"a":{"wat":1}}"#, "unknown key `wat`"),
            // An unquoted dotted key arrives as a nested table; the macro refuses it with the same words.
            (
                r#"{"a":{"b":{}}}"#,
                "a dotted tag is written as a quoted key",
            ),
        ] {
            let result = read_graph(&metadata(routes), Path::new("/w/app/Cargo.toml"), &|_| None);
            let error = result.unwrap_err();
            assert!(
                error.contains("`kernel-lib::a`") && error.contains(phrase),
                "{error}"
            );
        }
    }

    #[test]
    fn a_declaration_line_is_found_in_every_manifest_form() {
        let inline = "[package]\nname = \"k\"\n\n[package.metadata.gust.routes]\nphysics = { targets = [\"gpu\"] }\n\"physics.broadphase\" = {}\n";
        assert_eq!(declaration_line(inline, "physics"), Some(5));
        assert_eq!(declaration_line(inline, "physics.broadphase"), Some(6));
        assert_eq!(declaration_line(inline, "missing"), None);
        let header = "[package]\nname = \"k\"\n\n[package.metadata.gust.routes.physics]\ntargets = [\n    \"gpu\",\n]\n\n[package.metadata.gust.routes.\"physics.broadphase\"]\n";
        assert_eq!(declaration_line(header, "physics"), Some(4));
        assert_eq!(declaration_line(header, "physics.broadphase"), Some(9));
        // A key of another table with the same name is not a declaration.
        let other =
            "[package.metadata.docs]\nphysics = 1\n[package.metadata.gust.routes]\nphysics = {}\n";
        assert_eq!(declaration_line(other, "physics"), Some(4));
    }

    #[test]
    fn the_line_of_a_declaration_reaches_the_message() {
        let text = "[package.metadata.gust.routes]\nrender = { targets = [\"gpu\"] }\n";
        let graph = read_graph(&metadata(ROUTES), Path::new("/w/app/Cargo.toml"), &|_| {
            Some(text.to_owned())
        })
        .unwrap();
        let render = graph.declaring[0]
            .tags
            .iter()
            .find(|t| t.tag == "render")
            .unwrap();
        assert_eq!(render.line, Some(2));
    }

    // ---- the resolution ----

    fn resolved(toml: &str, profile: &str) -> Result<Vec<Row>, String> {
        let graph = graph_of(ROUTES);
        let config = parse_config(toml, FILE)?;
        resolve(&graph, &config, FILE, profile)
    }

    /// `tag target level` for each row, in row order.
    fn table(rows: &[Row]) -> Vec<String> {
        rows.iter()
            .map(|row| format!("{} {} {}", row.tag, row.target, row.level.name()))
            .collect()
    }

    #[test]
    fn with_no_entry_the_defaults_and_then_the_gpu_decide() {
        let rows = resolved("[routes]\n", "dev").unwrap();
        assert_eq!(
            table(&rows),
            [
                "fusion cpu default",
                "physics gpu default",
                "physics.broadphase gpu default",
                "render gpu fallback"
            ]
        );
        assert_eq!(rows[0].from.as_deref(), Some("kernel-lib::fusion"));
        assert_eq!(
            rows[2].from.as_deref(),
            Some("kernel-lib::physics"),
            "the ancestor's default"
        );
        assert_eq!(rows[3].from, None);
        assert_eq!(rows[0].variable, "GUST_ROUTE__kernel_lib__fusion");
    }

    #[test]
    fn the_nearest_entry_wins_and_children_inherit() {
        let rows = resolved("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n", "dev").unwrap();
        assert_eq!(
            table(&rows),
            [
                "fusion cpu default",
                "physics cpu entry",
                "physics.broadphase cpu entry",
                "render gpu fallback"
            ]
        );
        assert_eq!(rows[2].from.as_deref(), Some("kernel-lib::physics"));
        let rows = resolved(
            "[routes]\n\"kernel-lib::physics\" = \"cpu\"\n\"kernel-lib::physics.broadphase\" = \"gpu\"\n\"kernel-lib::fusion\" = \"gpu\"\n",
            "dev",
        )
        .unwrap();
        assert_eq!(
            table(&rows),
            [
                "fusion gpu entry",
                "physics cpu entry",
                "physics.broadphase gpu entry",
                "render gpu fallback"
            ]
        );
    }

    #[test]
    fn a_target_keyed_entry_follows_the_profile_and_dev_falls_back_to_native() {
        let toml = "[routes]\n\"kernel-lib::physics\" = { native = \"cpu\", web = \"gpu\" }\n";
        for (profile, wanted) in [("dev", "cpu"), ("native", "cpu"), ("web", "gpu")] {
            let rows = resolved(toml, profile).unwrap();
            assert_eq!(rows[1].target, wanted, "{profile}");
            assert_eq!(rows[1].level, Level::Entry);
        }
        // `dev` has its own key, which beats `native`.
        let toml = "[routes]\n\"kernel-lib::physics\" = { dev = \"gpu\", native = \"cpu\" }\n";
        assert_eq!(resolved(toml, "dev").unwrap()[1].target, "gpu");
        assert_eq!(resolved(toml, "native").unwrap()[1].target, "cpu");
        // A table with no key for the profile decides nothing there: the default decides.
        let toml = "[routes]\n\"kernel-lib::physics\" = { web = \"cpu\" }\n";
        let rows = resolved(toml, "native").unwrap();
        assert_eq!((rows[1].target, rows[1].level), ("gpu", Level::Default));
        // `web = "cpu"` is accepted (S0 ended in A).
        let rows = resolved(toml, "web").unwrap();
        assert_eq!((rows[1].target, rows[1].level), ("cpu", Level::Entry));
    }

    #[test]
    fn the_cpu_profile_sends_every_tag_to_the_cpu_and_refuses_a_gpu_entry() {
        let rows = resolved("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n", "cpu").unwrap();
        assert!(
            rows.iter()
                .all(|row| row.target == "cpu" && row.level == Level::Profile)
        );
        // Source defaults of `gpu` yield, and so does a tag's `targets` (render allows only the GPU).
        assert_eq!(rows[3].tag, "render");
        let error = resolved("[routes]\n\n\"kernel-lib::physics\" = \"gpu\"\n", "cpu").unwrap_err();
        assert!(
            error.starts_with("gust.toml:3: tag `kernel-lib::physics`: GUST_PROFILE=cpu")
                && !error.contains("inheritance"),
            "{error}"
        );
        // The child inherits the gpu entry: named at the entry's line, with the child's tag.
        let error = resolved("[routes]\n\"kernel-lib::physics\" = \"gpu\"\n", "cpu").unwrap_err();
        assert!(
            error.contains("gust.toml:2:") && error.contains("tag `kernel-lib::physics`"),
            "{error}"
        );
        // A table that decides nothing for `cpu` is no error there.
        resolved(
            "[routes]\n\"kernel-lib::physics\" = { native = \"gpu\" }\n",
            "cpu",
        )
        .unwrap();
    }

    #[test]
    fn the_errors_carry_file_line_and_tag() {
        let cases: &[(&str, &str, usize, &str, &str)] = &[
            // text, profile, line, tag, phrase
            (
                "[routes]\n\"kernel-lib::nope\" = \"cpu\"\n",
                "dev",
                2,
                "kernel-lib::nope",
                "unknown tag: `kernel-lib` declares `fusion`, `physics`, `physics.broadphase`, `render`",
            ),
            (
                "[routes]\n\n\"ghost::physics\" = \"cpu\"\n",
                "dev",
                3,
                "ghost::physics",
                "not a package of this build's dependency graph",
            ),
            (
                "[routes]\n\"app::physics\" = \"cpu\"\n",
                "dev",
                2,
                "app::physics",
                "declares no route tags",
            ),
            (
                "[routes]\n\"other::x\" = \"cpu\"\n",
                "dev",
                2,
                "other::x",
                "not a package of this build's dependency graph",
            ),
            (
                "[routes]\n\"physics\" = \"cpu\"\n",
                "dev",
                2,
                "physics",
                "the key must be `<package>::<tag>`",
            ),
            (
                "[routes]\n\"kernel-lib::render\" = \"cpu\"\n",
                "dev",
                2,
                "kernel-lib::render",
                "allows only targets = [\"gpu\"]",
            ),
            (
                "[routes]\n\"kernel-lib::render\" = { web = \"cpu\" }\n",
                "dev",
                2,
                "kernel-lib::render",
                "routed to `cpu`",
            ),
            (
                "[routes]\n\"kernel-lib::physics\" = \"turbo\"\n",
                "dev",
                2,
                "kernel-lib::physics",
                "`turbo` is not a route target",
            ),
            (
                "[routes]\n\"kernel-lib::physics\" = \"gpu+subgroups\"\n",
                "dev",
                2,
                "kernel-lib::physics",
                "capability profile",
            ),
            (
                "[routes]\n\"kernel-lib::physics\" = { \"native+subgroups\" = \"gpu\" }\n",
                "dev",
                2,
                "kernel-lib::physics",
                "capability profile",
            ),
            (
                "[routes]\n\"kernel-lib::physics\" = { cpu = \"cpu\" }\n",
                "dev",
                2,
                "kernel-lib::physics",
                "`cpu` key has nothing to decide",
            ),
            (
                "[routes]\n\"kernel-lib::physics\" = { ios = \"cpu\" }\n",
                "dev",
                2,
                "kernel-lib::physics",
                "not a profile key",
            ),
        ];
        for (toml, profile, line, tag, phrase) in cases {
            let error = resolved(toml, profile).unwrap_err();
            assert!(
                error.starts_with(&format!("{FILE}:{line}: tag `{tag}`: "))
                    && error.contains(phrase),
                "{toml:?}: {error}"
            );
        }
    }

    #[test]
    fn an_inherited_conflict_names_the_entry_and_the_child() {
        let routes = r#"{"solver":{"targets":["gpu","cpu"]},"solver.pinned":{"targets":["gpu"]}}"#;
        let graph = graph_of(routes);
        let config = parse_config("[routes]\n\n\"kernel-lib::solver\" = \"cpu\"\n", FILE).unwrap();
        let error = resolve(&graph, &config, FILE, "dev").unwrap_err();
        assert!(
            error.starts_with("gust.toml:3: tag `kernel-lib::solver.pinned`: the entry `kernel-lib::solver` routes it to `cpu` by inheritance")
                && error.contains("targets = [\"gpu\"]"),
            "{error}"
        );
        // Its own entry settles it.
        let config = parse_config(
            "[routes]\n\"kernel-lib::solver\" = \"cpu\"\n\"kernel-lib::solver.pinned\" = \"gpu\"\n",
            FILE,
        )
        .unwrap();
        let rows = resolve(&graph, &config, FILE, "dev").unwrap();
        assert_eq!(
            table(&rows),
            ["solver cpu entry", "solver.pinned gpu entry"]
        );
        // Under cpu the tag's set is not consulted.
        let config = parse_config("[routes]\n\"kernel-lib::solver\" = \"cpu\"\n", FILE).unwrap();
        resolve(&graph, &config, FILE, "cpu").unwrap();
    }

    #[test]
    fn a_default_or_a_fallback_outside_the_set_is_an_error_here_as_in_the_macro() {
        let routes = r#"{"a":{"targets":["cpu"]},"b":{"targets":["gpu"],"default":"cpu"}}"#;
        let graph = graph_of(routes);
        let config = Config::default();
        let error = resolve(&graph, &config, FILE, "dev").unwrap_err();
        assert!(
            error.contains("tag `kernel-lib::a`") && error.contains("leaves the GPU"),
            "{error}"
        );
        let routes = r#"{"b":{"targets":["gpu"],"default":"cpu"}}"#;
        let error = resolve(&graph_of(routes), &config, FILE, "dev").unwrap_err();
        assert!(
            error.contains("tag `kernel-lib::b`") && error.contains("the default `cpu`"),
            "{error}"
        );
    }

    #[test]
    fn a_bad_declaration_is_refused_before_anything_is_resolved() {
        for (routes, phrase) in [
            (r#"{"A":{}}"#, "ill-named"),
            (r#"{"a__b":{}}"#, "ill-named"),
            (r#"{"a_.b":{}}"#, "ill-named"),
            (r#"{"a":{"targets":[]}}"#, "`targets` is empty"),
            (
                r#"{"a":{"targets":["tpu"]}}"#,
                "`tpu` is not a route target",
            ),
            (
                r#"{"a":{"default":"native"}}"#,
                "default `native` is not a route target",
            ),
        ] {
            let error = resolve(&graph_of(routes), &Config::default(), FILE, "dev").unwrap_err();
            assert!(
                error.contains("tag `kernel-lib::") && error.contains(phrase),
                "{routes}: {error}"
            );
        }
    }

    #[test]
    fn two_pairs_that_spell_one_variable_are_refused_naming_both() {
        // `a__b`/`c`, `a`/`b.c` and `a--b`/`c` all spell GUST_ROUTE__a__b__c.
        let metadata = r#"{"packages":[
{"id":"app","name":"app","manifest_path":"/w/app/Cargo.toml","metadata":null},
{"id":"p1","name":"a__b","manifest_path":"/w/p1/Cargo.toml","metadata":{"gust":{"routes":{"c":{}}}}},
{"id":"p2","name":"a","manifest_path":"/w/p2/Cargo.toml","metadata":{"gust":{"routes":{"b.c":{}}}}}
],"resolve":{"nodes":[{"id":"app","dependencies":["p1","p2"]},{"id":"p1","dependencies":[]},{"id":"p2","dependencies":[]}]}}"#;
        let graph = read_graph(metadata, Path::new("/w/app/Cargo.toml"), &|_| None).unwrap();
        let error = resolve(&graph, &Config::default(), FILE, "dev").unwrap_err();
        assert!(
            error.contains("`a::b.c`")
                && error.contains("`a__b::c`")
                && error.contains("GUST_ROUTE__a__b__c")
                && error.contains("/w/p1/Cargo.toml")
                && error.contains("/w/p2/Cargo.toml"),
            "{error}"
        );
        // Package names that differ only in case spell one Windows variable.
        let metadata = metadata
            .replace("\"a__b\"", "\"Kern\"")
            .replace("\"name\":\"a\"", "\"name\":\"kern\"");
        let metadata = metadata
            .replace("{\"c\":{}}", "{\"t\":{}}")
            .replace("{\"b.c\":{}}", "{\"t\":{}}");
        let graph = read_graph(&metadata, Path::new("/w/app/Cargo.toml"), &|_| None).unwrap();
        let error = resolve(&graph, &Config::default(), FILE, "dev").unwrap_err();
        assert!(error.contains("without case"), "{error}");
    }

    #[test]
    fn the_resolution_does_not_depend_on_the_order_of_the_entries() {
        let one = "[routes]\n\"kernel-lib::physics\" = \"cpu\"\n\"kernel-lib::fusion\" = \"gpu\"\n";
        let two = "[routes]\n\"kernel-lib::fusion\" = \"gpu\"\n\"kernel-lib::physics\" = \"cpu\"\n";
        assert_eq!(resolved(one, "dev").unwrap(), resolved(two, "dev").unwrap());
    }

    /// `app` depends on two versions of `kernel-lib`, which declare `physics` as `first` and `second`.
    fn two_versions(first: &str, second: &str) -> Graph {
        two_versions_of(
            &format!("{{\"physics\":{first}}}"),
            &format!("{{\"physics\":{second}}}"),
        )
    }

    /// `app` depends on two versions of `kernel-lib` whose whole `routes` tables are `first` and `second` (JSON objects).
    fn two_versions_of(first: &str, second: &str) -> Graph {
        let metadata = format!(
            r#"{{"packages":[
{{"id":"app","name":"app","manifest_path":"/w/app/Cargo.toml","metadata":null}},
{{"id":"k1","name":"kernel-lib","manifest_path":"/w/k1/Cargo.toml","metadata":{{"gust":{{"routes":{first}}}}}}},
{{"id":"k2","name":"kernel-lib","manifest_path":"/w/k2/Cargo.toml","metadata":{{"gust":{{"routes":{second}}}}}}}
],"resolve":{{"nodes":[{{"id":"app","dependencies":["k1","k2"]}},{{"id":"k1","dependencies":[]}},{{"id":"k2","dependencies":[]}}]}}}}"#
        );
        read_graph(&metadata, Path::new("/w/app/Cargo.toml"), &|_| None).unwrap()
    }

    #[test]
    fn two_versions_that_declare_a_tag_alike_share_one_variable_and_resolve_once() {
        let alike = r#"{"targets":["gpu","cpu"],"default":"gpu"}"#;
        let graph = two_versions(alike, alike);
        assert_eq!(graph.declaring.len(), 2);
        let config = parse_config("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n", FILE).unwrap();
        let rows = resolve(&graph, &config, FILE, "dev").unwrap();
        assert_eq!(table(&rows), ["physics cpu entry"]);
        assert_eq!(rows[0].variable, "GUST_ROUTE__kernel_lib__physics");
        // Without an entry too: the pair counts once.
        assert_eq!(
            table(&resolve(&graph, &Config::default(), FILE, "dev").unwrap()),
            ["physics gpu default"]
        );
    }

    #[test]
    fn two_versions_that_declare_a_tag_differently_are_refused_with_advice_that_fits() {
        for (first, second) in [
            (r#"{"default":"gpu"}"#, r#"{"default":"cpu"}"#),
            (r#"{"targets":["gpu"]}"#, r#"{}"#),
        ] {
            let graph = two_versions(first, second);
            let error = resolve(&graph, &Config::default(), FILE, "dev").unwrap_err();
            assert!(
                error.contains("two versions of `kernel-lib`")
                    && error.contains("/w/k1/Cargo.toml")
                    && error.contains("/w/k2/Cargo.toml")
                    && error.contains("declare `physics` differently")
                    && error.contains("GUST_ROUTE__kernel_lib__physics")
                    && error.contains("unify the versions")
                    && !error.contains("rename"),
                "{error}"
            );
        }
    }

    #[test]
    fn two_versions_are_compared_as_whole_tables_with_targets_as_sets() {
        // The order of `targets` does not matter.
        let graph = two_versions(
            r#"{"targets":["gpu","cpu"],"default":"gpu"}"#,
            r#"{"targets":["cpu","gpu"],"default":"gpu"}"#,
        );
        let rows = resolve(&graph, &Config::default(), FILE, "dev").unwrap();
        assert_eq!(table(&rows), ["physics gpu default"]);
        // Only an ancestor's default differs: the child `physics.broadphase` is declared alike by both, but the macro of
        // each version resolves it from its own chain, so one variable cannot serve both.
        let graph = two_versions_of(
            r#"{"physics":{"default":"cpu"},"physics.broadphase":{}}"#,
            r#"{"physics":{},"physics.broadphase":{}}"#,
        );
        let error = resolve(&graph, &Config::default(), FILE, "dev").unwrap_err();
        assert!(
            error.contains("two versions of `kernel-lib`")
                && error.contains("/w/k1/Cargo.toml")
                && error.contains("/w/k2/Cargo.toml")
                && error.contains("declare `physics` differently")
                && error.contains("unify the versions"),
            "{error}"
        );
        // A tag that only one version declares is a difference too.
        let graph = two_versions_of(
            r#"{"physics":{},"physics.broadphase":{}}"#,
            r#"{"physics":{}}"#,
        );
        let error = resolve(&graph, &Config::default(), FILE, "dev").unwrap_err();
        assert!(
            error.contains("declare `physics.broadphase` differently"),
            "{error}"
        );
        // Equal tables: resolved once.
        let graph = two_versions_of(
            r#"{"physics":{"default":"cpu"},"physics.broadphase":{}}"#,
            r#"{"physics.broadphase":{},"physics":{"default":"cpu"}}"#,
        );
        let rows = resolve(&graph, &Config::default(), FILE, "dev").unwrap();
        assert_eq!(
            table(&rows),
            ["physics cpu default", "physics.broadphase cpu default"]
        );
    }

    #[test]
    fn a_tag_with_no_gpu_form_and_no_default_names_its_declaration() {
        let graph = graph_of(r#"{"a":{"targets":["cpu"]}}"#);
        let error = resolve(&graph, &Config::default(), FILE, "dev").unwrap_err();
        assert!(
            error.starts_with("/w/kernel-lib/Cargo.toml: tag `kernel-lib::a`:")
                && error.contains("add a `default` to the declaration")
                && error.contains("an entry for it in gust.toml"),
            "{error}"
        );
    }

    #[test]
    fn cargo_metadata_gets_the_feature_and_lock_flags_of_the_command() {
        let cargo = OsString::from("cargo");
        let manifest = Path::new("/w/app/Cargo.toml");
        let args_of = |line: &str| -> Vec<String> {
            let command = metadata_command(&cargo, manifest, &words(line));
            command
                .get_args()
                .map(|a| a.to_string_lossy().into_owned())
                .collect()
        };
        let base = [
            "metadata",
            "--format-version",
            "1",
            "--manifest-path",
            "/w/app/Cargo.toml",
        ];
        assert_eq!(args_of("-p app --release"), base);
        let mut wanted = base.map(str::to_owned).to_vec();
        wanted.extend(words(
            "--features phys,more --all-features --no-default-features --locked --frozen --offline",
        ));
        assert_eq!(
            args_of(
                "--features phys,more -p app --all-features --no-default-features --locked --frozen --offline"
            ),
            wanted
        );
        // The spellings of a feature list, and nothing after `--`.
        assert_eq!(args_of("--features=a")[5..], ["--features=a"]);
        assert_eq!(args_of("-F a,b")[5..], ["-F", "a,b"]);
        assert_eq!(args_of("-Fa")[5..], ["-Fa"]);
        // `--config` in both spellings (its value is one word), and `-Z` is left out.
        assert_eq!(
            args_of("--config net.offline=true --config=a.b=c -Z unstable-options")[5..],
            ["--config", "net.offline=true", "--config=a.b=c"]
        );
        assert_eq!(args_of("-p app -- --locked --features x"), base);
        assert_eq!(args_of("--release --target wasm32-unknown-unknown"), base);
    }

    #[test]
    fn a_workspace_run_is_refused_when_any_selected_member_has_a_gust_toml() {
        // A virtual root `w` with members `app` (routed) and `tools` (no gust.toml).
        let scratch = crate::store::tests::Scratch::new("routes-workspace");
        let member = |name: &str, routed: bool| {
            let directory = scratch.0.join(name);
            fs::create_dir_all(&directory).unwrap();
            if routed {
                fs::write(directory.join(CONFIG_FILE), "[routes]\n").unwrap();
            }
            Package {
                name: name.to_owned(),
                manifest_path: directory.join("Cargo.toml"),
            }
        };
        let members = [member("app", true), member("tools", false)];
        let error = refuse_workspace(&members, &[]).unwrap_err();
        assert!(
            error.contains("`--workspace`, `--all`")
                && error.contains("`app`")
                && error.contains("gust.toml")
                && !error.contains("`tools`")
                && error.contains("-p")
                && error.contains("`--exclude`"),
            "{error}"
        );
        // Excluding the routed member leaves nothing to route: the arguments pass through.
        assert_eq!(refuse_workspace(&members, &["app".to_owned()]), Ok(()));
        // No member has one.
        assert_eq!(refuse_workspace(&members[1..], &[]), Ok(()));
    }

    /// `cargo metadata --no-deps` for a workspace at `root` whose packages are `members` (`name`, directory relative to
    /// the root; "" is the root package) and whose default members are `defaults` (names).
    fn workspace_at(
        root: &str,
        members: &[(&str, &str)],
        defaults: &[&str],
    ) -> crate::collect::Workspace {
        let path = |directory: &str| {
            if directory.is_empty() {
                format!("{root}/Cargo.toml")
            } else {
                format!("{root}/{directory}/Cargo.toml")
            }
        };
        let packages: Vec<String> = members
            .iter()
            .map(|(name, directory)| {
                format!(
                    r#"{{"id":"{name} 0.0.0","name":"{name}","manifest_path":"{}"}}"#,
                    path(directory)
                )
            })
            .collect();
        let ids: Vec<String> = defaults
            .iter()
            .map(|name| format!(r#""{name} 0.0.0""#))
            .collect();
        let text = format!(
            r#"{{"packages":[{}],"workspace_default_members":[{}],"workspace_root":"{root}","target_directory":"{root}/target"}}"#,
            packages.join(","),
            ids.join(",")
        );
        crate::collect::parse_metadata(&text).unwrap()
    }

    #[test]
    fn the_default_selection_of_a_workspace_root_is_checked_like_a_workspace_run() {
        let scratch = crate::store::tests::Scratch::new("routes-default-selection");
        let root = scratch.0.to_string_lossy().replace('\\', "/");
        let manifest = |directory: &str| {
            if directory.is_empty() {
                scratch.0.join("Cargo.toml")
            } else {
                scratch.0.join(directory).join("Cargo.toml")
            }
        };
        // A virtual root with two members and no gust.toml: cargo builds both, the arguments pass through.
        let workspace = workspace_at(&root, &[("a", "a"), ("b", "b")], &["a", "b"]);
        let selected = default_selection(&workspace, &manifest(""))
            .expect("a virtual root selects its defaults");
        assert_eq!(selected.len(), 2);
        assert_eq!(refuse_workspace(selected, &[]), Ok(()));
        // One routed member: refused, and named.
        fs::create_dir_all(scratch.0.join("a")).unwrap();
        fs::write(scratch.0.join("a").join(CONFIG_FILE), "[routes]\n").unwrap();
        let error = refuse_workspace(selected, &[]).unwrap_err();
        assert!(error.contains("`a`") && !error.contains("`b`"), "{error}");
        // A root package with `default-members` that names another member: that member's file is checked too.
        fs::create_dir_all(scratch.0.join("other")).unwrap();
        fs::write(scratch.0.join("other").join(CONFIG_FILE), "[routes]\n").unwrap();
        let workspace = workspace_at(
            &root,
            &[("root", ""), ("other", "other")],
            &["root", "other"],
        );
        let selected = default_selection(&workspace, &manifest(""))
            .expect("default-members select two packages");
        let error = refuse_workspace(selected, &[]).unwrap_err();
        assert!(
            error.contains("`other`") && !error.contains("`root`"),
            "{error}"
        );
        // The ordinary cases are the driver's own: the root package alone, a member's manifest, and an unknown selection.
        let workspace = workspace_at(&root, &[("root", ""), ("other", "other")], &["root"]);
        assert!(default_selection(&workspace, &manifest("")).is_none());
        assert!(default_selection(&workspace, &manifest("other")).is_none());
        let mut unknown = workspace.clone();
        unknown.default_members = None;
        assert!(default_selection(&unknown, &manifest("")).is_none());
    }

    #[test]
    fn the_excluded_names_are_read_in_both_spellings_and_stop_at_the_double_dash() {
        assert_eq!(
            excluded(&words(
                "--workspace --exclude a --exclude=b -p c -- --exclude d"
            )),
            ["a", "b"]
        );
        assert!(excluded(&words("--workspace --release")).is_empty());
    }

    #[test]
    fn a_preset_route_variable_that_differs_or_matches_no_tag_gets_a_note() {
        let plan = plan_of("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n");
        let preset = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
            pairs
                .iter()
                .map(|(n, v)| ((*n).to_owned(), (*v).to_owned()))
                .collect()
        };
        // Equal to the resolved value, or not a route variable: silent.
        assert!(
            plan.notes(&preset(&[
                ("GUST_ROUTE__kernel_lib__physics", "cpu"),
                ("PATH", "x")
            ]))
            .is_empty()
        );
        let notes = plan.notes(&preset(&[("GUST_ROUTE__kernel_lib__physics", "gpu")]));
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(
            notes[0].contains("GUST_ROUTE__kernel_lib__physics=gpu")
                && notes[0].contains("resolves it to `cpu`"),
            "{notes:?}"
        );
        // Names compare without case on Windows only. There the lowercase prefix is a route variable that equals the
        // resolved value (silent); elsewhere it is not a route variable at all (silent too).
        let lowercase = plan.notes(&preset(&[("gust_route__KERNEL_LIB__physics", "cpu")]));
        assert!(lowercase.is_empty(), "{lowercase:?}");
        let notes = plan.notes(&preset(&[("GUST_ROUTE__kernel_lib__phyiscs", "cpu")]));
        assert!(
            notes.len() == 1 && notes[0].contains("matches no route tag"),
            "{notes:?}"
        );
    }

    #[test]
    fn forgetting_the_table_removes_the_file_and_tolerates_its_absence() {
        let scratch = crate::store::tests::Scratch::new("routes-forget");
        let artifacts = scratch.0.join(ARTIFACTS_DIR);
        fs::create_dir_all(&artifacts).unwrap();
        fs::write(artifacts.join(ROUTES_FILE), "{}").unwrap();
        fs::write(artifacts.join("manifest.json"), "{}").unwrap();
        let package = Package {
            name: "app".to_owned(),
            manifest_path: scratch.0.join("Cargo.toml"),
        };
        forget(&package).unwrap();
        assert!(!artifacts.join(ROUTES_FILE).exists());
        assert!(
            artifacts.join("manifest.json").is_file(),
            "only the table goes"
        );
        forget(&package).unwrap();
        assert!(!has_config(&package));
        fs::write(scratch.0.join(CONFIG_FILE), "[routes]\n").unwrap();
        assert!(has_config(&package));
    }

    // ---- the plan ----

    fn plan_of(toml: &str) -> RoutePlan {
        RoutePlan::new(
            "dev",
            resolved(toml, "dev").unwrap(),
            PathBuf::from("/w/app/gust-artifacts"),
        )
    }

    #[test]
    fn routes_json_is_deterministic_and_complete() {
        let plan = plan_of("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n");
        let text = plan.json();
        assert_eq!(text, plan.json());
        assert_eq!(
            text,
            plan_of("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n").json(),
            "a rerun gives the same bytes"
        );
        let expected = "{\n  \"profile\": \"dev\",\n  \"routes\": [\n    {\n      \"from\": \"kernel-lib::fusion\",\n      \"level\": \"default\",\n      \"package\": \"kernel-lib\",\n      \"tag\": \"fusion\",\n      \"target\": \"cpu\",\n      \"variable\": \"GUST_ROUTE__kernel_lib__fusion\"\n    },\n";
        assert!(text.starts_with(expected), "{text}");
        assert!(text.contains("\"from\": null"), "{text}");
        assert!(text.ends_with("  \"schema\": 1\n}\n"), "{text}");
        assert!(
            !text.contains('\\') && !text.contains("/w/"),
            "no path in the table: {text}"
        );
    }

    #[test]
    fn writing_routes_json_creates_the_directory_and_leaves_equal_bytes_alone() {
        let scratch = crate::store::tests::Scratch::new("routes");
        let plan = RoutePlan::new(
            "dev",
            resolved("[routes]\n", "dev").unwrap(),
            scratch.0.join(ARTIFACTS_DIR),
        );
        plan.write().unwrap();
        let path = scratch.0.join(ARTIFACTS_DIR).join(ROUTES_FILE);
        assert_eq!(fs::read_to_string(&path).unwrap(), plan.json());
        let before = fs::metadata(&path).unwrap().modified().unwrap();
        plan.write().unwrap();
        assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), before);
    }

    /// The route variables of a command, as `(name, value)`.
    fn route_envs(command: &Command) -> BTreeSet<(String, String)> {
        command
            .get_envs()
            .filter_map(|(name, value)| {
                let name = name.to_string_lossy().into_owned();
                name.starts_with("GUST_ROUTE__").then(|| {
                    (
                        name,
                        value
                            .map(|v| v.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                    )
                })
            })
            .collect()
    }

    #[test]
    fn run_and_test_commands_carry_the_route_variables_of_the_plan() {
        // The `build` and `check` side (the runner of the collect step) is tested in commands.rs, `routed_runner`.
        let plan = plan_of("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n");
        let wanted: BTreeSet<(String, String)> = plan.variables().into_iter().collect();
        assert_eq!(wanted.len(), 4);
        assert!(wanted.contains(&(
            "GUST_ROUTE__kernel_lib__physics".to_owned(),
            "cpu".to_owned()
        )));
        let cargo = OsString::from("cargo");
        let arguments = [
            "-p".to_owned(),
            "app".to_owned(),
            "--".to_owned(),
            "x".to_owned(),
        ];
        let run = cargo_command(&cargo, "run", &arguments, Some(&plan));
        let test = cargo_command(&cargo, "test", &arguments, Some(&plan));
        assert_eq!(route_envs(&run), wanted);
        assert_eq!(route_envs(&test), wanted);
        // The arguments reach cargo as given, after the verb.
        let args: Vec<String> = run
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, ["run", "-p", "app", "--", "x"]);
        // Without a plan nothing is set.
        assert!(route_envs(&cargo_command(&cargo, "run", &arguments, None)).is_empty());
    }

    /// The value a command sets for `name` (`None`: it sets nothing for it).
    fn env_value(command: &Command, name: &str) -> Option<String> {
        command
            .get_envs()
            .find(|(key, _)| key.to_string_lossy() == name)
            .and_then(|(_, value)| value)
            .map(|value| value.to_string_lossy().into_owned())
    }

    /// m8.z (S1 of the M8 sanity checks): a command with routes also carries the profile they were resolved for, which
    /// the `#[gpu]` expansion of a routed module compares with the profile `gust` was built for. A command without
    /// routes carries nothing, and removes a stale inherited value (the variable is the driver's own), so plain
    /// behaviour (D67) is unchanged, and the variable is not one of the route variables.
    #[test]
    fn commands_with_routes_export_the_profile_the_routes_were_resolved_for() {
        // The name is the one the macro reads; the macro cannot depend on this tool, so the source is read.
        assert_eq!(ROUTED_PROFILE_VARIABLE, "GUST_ROUTED_PROFILE");
        let macro_source =
            include_str!("../../gust-macros/src/expand/mod.rs").replace("\r\n", "\n");
        assert!(
            macro_source.contains(&format!(
                "pub(crate) const ROUTED_PROFILE_VARIABLE: &str = \"{ROUTED_PROFILE_VARIABLE}\";"
            )),
            "crates/gust-macros/src/expand/mod.rs reads another variable"
        );
        let cargo = OsString::from("cargo");
        let arguments = ["-p".to_owned(), "app".to_owned()];
        let rows = plan_of("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n").rows;
        for profile in ["dev", "native", "web", "cpu"] {
            let plan = RoutePlan::new(
                profile,
                rows.clone(),
                PathBuf::from("/w/app/gust-artifacts"),
            );
            for verb in ["run", "test"] {
                let command = cargo_command(&cargo, verb, &arguments, Some(&plan));
                assert_eq!(
                    env_value(&command, ROUTED_PROFILE_VARIABLE).as_deref(),
                    Some(profile),
                    "{verb} {profile}"
                );
                // Not a route variable: the set of route variables is what it was.
                assert_eq!(route_envs(&command).len(), plan.rows.len(), "{verb}");
            }
            // The collect step of build and check goes through `apply` too.
            let mut collect = Command::new("cargo");
            plan.apply(&mut collect);
            assert_eq!(
                env_value(&collect, ROUTED_PROFILE_VARIABLE).as_deref(),
                Some(profile)
            );
        }
        let plain = cargo_command(&cargo, "run", &arguments, None);
        assert_eq!(env_value(&plain, ROUTED_PROFILE_VARIABLE), None);
        // Nothing is set, and a value the driver inherited (from an outer `cargo gust run` whose program starts cargo
        // again) is removed, so that a build of another profile is not refused for it.
        assert!(
            plain.get_envs().any(
                |(key, value)| key.to_string_lossy() == ROUTED_PROFILE_VARIABLE && value.is_none()
            ),
            "the stale variable must be removed"
        );
        let mut collect = Command::new("cargo");
        without_routed_profile(&mut collect);
        assert_eq!(env_value(&collect, ROUTED_PROFILE_VARIABLE), None);
        // A preset value of the driver's own environment is overridden, as the route variables are, and the notes about
        // presets do not mention it.
        let plan = plan_of("[routes]\n");
        let preset = vec![(ROUTED_PROFILE_VARIABLE.to_owned(), "native".to_owned())];
        assert!(plan.notes(&preset).is_empty());
    }

    // ---- the profile of run and test ----

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn the_profile_of_run_and_test_is_the_one_the_build_script_infers() {
        let cases: &[(Option<&str>, &str, &str)] = &[
            (None, "-p app", "dev"),
            (None, "-p app --release", "native"),
            (None, "-r", "native"),
            (None, "--profile release", "native"),
            (None, "--profile=release", "native"),
            (None, "--profile dev", "dev"),
            (None, "--release --target wasm32-unknown-unknown", "web"),
            (None, "--release --target=wasm32-unknown-unknown", "web"),
            (None, "--target wasm32-unknown-unknown", "dev"),
            (None, "--release -- --release", "native"),
            (None, "-- --release", "dev"),
            (Some("cpu"), "--release", "cpu"),
            (Some("dev+subgroups"), "--release", "dev"),
            (Some("web"), "", "web"),
        ];
        for (env, arguments, wanted) in cases {
            assert_eq!(
                build_profile(*env, &words(arguments)).unwrap(),
                *wanted,
                "{env:?} {arguments}"
            );
        }
        assert!(
            build_profile(Some("turbo"), &[])
                .unwrap_err()
                .contains("unknown GUST_PROFILE")
        );
        assert!(build_profile(Some(""), &[]).unwrap_err().contains("empty"));
    }

    #[test]
    fn the_summary_names_each_tag_its_target_and_what_decided() {
        let plan = plan_of("[routes]\n\"kernel-lib::physics\" = \"cpu\"\n");
        let lines = plan.summary();
        assert_eq!(
            lines[1],
            "gust: route kernel-lib::physics -> cpu (entry `kernel-lib::physics`, profile dev)"
        );
        assert_eq!(
            lines[3],
            "gust: route kernel-lib::render -> gpu (fallback, profile dev)"
        );
    }
}

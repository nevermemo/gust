//! Build-time routing in the macro (m8.r1, D65 to D67, D137): the tag, the resolver, the three expansions and the
//! codes GUST-E0744 (tag sets), GUST-E0748 and GUST-E0749.
//!
//! The route variables are read from a table here, never from the process, so no test sets an environment variable.

use std::{
    fs,
    path::{Path, PathBuf},
};

use proc_macro2::{Span, TokenStream};

use crate::{
    diagnostics::{
        Code,
        testing::{assert_code, parse_str},
    },
    expand::{
        Environment, Profile, expand_in_routed, expand_map_in_routed,
        routing::{self, Level, Request, Tag, Target},
    },
    manifest,
};

// ---------------------------------------------------------------------------------------------------------------
// The copy of `gust::routes`

/// `(package, tag, variable)`: the golden spelling. `gust-compute-macros` keeps a copy of `variable_name` and
/// asserts this table verbatim (m8.r1); change both together.
const GOLDEN: &[(&str, &str, &str)] = &[
    ("kernel-lib", "add", "GUST_ROUTE__kernel_lib__add"),
    ("kernels", "physics", "GUST_ROUTE__kernels__physics"),
    (
        "a-b-c",
        "physics.broadphase.pairs",
        "GUST_ROUTE__a_b_c__physics__broadphase__pairs",
    ),
    ("x", "t_1", "GUST_ROUTE__x__t_1"),
    (
        "gust-compute",
        "physics.broadphase",
        "GUST_ROUTE__gust_compute__physics__broadphase",
    ),
];

#[test]
fn the_variable_spells_the_package_and_the_tag() {
    for (package, tag, variable) in GOLDEN {
        assert_eq!(
            routing::variable_name(package, tag),
            *variable,
            "{package} {tag}"
        );
    }
}

/// The rows of the `GOLDEN` table in the Rust `source`, trimmed, from the line of `const GOLDEN` to the closing `];`.
fn golden_rows(source: &str) -> Vec<String> {
    let mut rows = Vec::new();
    let mut inside = false;
    for line in source.lines().map(str::trim) {
        if line.starts_with("const GOLDEN") {
            inside = true;
        } else if inside && line == "];" {
            return rows;
        } else if inside {
            rows.push(line.to_owned());
        }
    }
    panic!("no complete GOLDEN table");
}

/// The token stream of the function `name` in `source`, from its signature line to the closing brace at the
/// signature's indentation, as text: the lexer drops comments and layout (rustfmt wraps the copy at a deeper indent
/// than the original) and keeps every token, string and char literals included, so `"__"` and `"_ _"` differ.
/// `pub fn` and `pub(crate) fn` are read as one spelling.
fn function_text(source: &str, name: &str) -> String {
    let mut lines = Vec::new();
    let mut indent = None;
    for line in source.lines() {
        let trimmed = line.trim_start();
        let here = line.len() - trimmed.len();
        if indent.is_none() {
            let signature = trimmed
                .strip_prefix("pub(crate) fn ")
                .or_else(|| trimmed.strip_prefix("pub fn "));
            if signature.is_some_and(|rest| rest.starts_with(&format!("{name}("))) {
                indent = Some(here);
                lines.push(format!("pub fn {}", signature.unwrap()));
            }
        } else {
            lines.push(trimmed.to_owned());
            if trimmed == "}" && Some(here) == indent {
                return tokens(&lines.join("\n"));
            }
        }
    }
    panic!("no complete function {name}");
}

/// `source` lexed and printed back: the same text for the same tokens, whatever the layout and comments.
fn tokens(source: &str) -> String {
    source
        .parse::<proc_macro2::TokenStream>()
        .unwrap_or_else(|error| panic!("not Rust tokens: {error}\n{source}"))
        .to_string()
}

#[test]
fn the_comparison_ignores_layout_and_comments_but_not_a_character_in_a_literal() {
    // N1: a literal's characters are tokens. The old comparison removed every whitespace character, so a `' '` or a
    // `"_ _"` crept into a copy would not have been seen.
    let base = "fn f(s: &mut String) { s.push_str(\"__\"); s.push('x'); }";
    let relaid = "fn f(s: &mut String) {\n    // a note\n    s.push_str(\n        \"__\"\n    );\n    s.push('x');\n}";
    assert_eq!(tokens(base), tokens(relaid));
    for changed in [
        base.replace("\"__\"", "\"_ _\""),
        base.replace("'x'", "' '"),
        base.replace("\"__\"", "\"_\""),
        base.replace("push_str", "push_string"),
    ] {
        assert_ne!(tokens(base), tokens(&changed), "{changed}");
    }
}

#[test]
fn the_golden_table_and_the_functions_are_the_ones_in_gust_row_for_row() {
    // The macro cannot call `gust::routes` (the dependency runs the other way), so the two spellings are compared as
    // text: a row added to one table and not the other, or a changed line in a function, fails here (ruling 4).
    let theirs = include_str!("../../../gust/src/routes.rs");
    let mine = include_str!("routes.rs");
    let rows = golden_rows(theirs);
    assert!(rows.len() >= 10, "{rows:?}");
    assert_eq!(golden_rows(mine), rows);
    let routing_source = include_str!("../expand/mod.rs");
    for name in ["variable_name", "is_valid_tag"] {
        let copy = function_text(routing_source, name);
        assert!(copy.len() > 100, "{copy}");
        assert_eq!(copy, function_text(theirs, name), "{name}");
    }
}

#[test]
fn the_prefix_is_the_one_of_gust() {
    assert_eq!(routing::VARIABLE_PREFIX, "GUST_ROUTE__");
}

#[test]
fn the_collisions_of_gust_are_the_same_here() {
    // `a_.b` and `a._b` spell one variable, and so do `a__b` and `a.b`: a valid tag has neither shape.
    assert_eq!(
        routing::variable_name("c", "a_.b"),
        routing::variable_name("c", "a._b")
    );
    assert_eq!(
        routing::variable_name("c", "a__b"),
        routing::variable_name("c", "a.b")
    );
    for tag in ["a_.b", "a._b", "_a", "a_", "a__b"] {
        assert!(!routing::is_valid_tag(tag), "{tag}");
    }
    assert!(routing::is_valid_tag("a.b"));
    let one = routing::variable_name("a__b", "c");
    assert_eq!(routing::variable_name("a", "b.c"), one);
    assert_eq!(routing::variable_name("a--b", "c"), one);
}

/// Tags the rules accept.
const VALID: &[&str] = &[
    "a",
    "a.b",
    "a_b",
    "a.b.c",
    "a_b.c",
    "a.b_c",
    "ab",
    "a1",
    "a1.b2",
    "physics.broadphase",
    "physics_broadphase",
    "t_1",
];

/// Tags the rules refuse.
const INVALID: &[&str] = &[
    "", ".", "a.", ".a", "a..b", "A", "a-b", "a b", "a.B", "a/b", "\u{e9}", "a._", "_", "a_.b",
    "a._b", "_a", "a_", "a__b", "a.b__", "a.__b",
];

#[test]
fn the_tag_rules() {
    for tag in VALID {
        assert!(routing::is_valid_tag(tag), "{tag}");
        assert_eq!(routing::tag_fault(tag), None, "{tag}");
    }
    for tag in INVALID {
        assert!(!routing::is_valid_tag(tag), "{tag:?}");
    }
}

#[test]
fn a_fault_is_named_exactly_when_the_tag_is_ill_named() {
    for tag in VALID.iter().chain(INVALID) {
        assert_eq!(
            routing::tag_fault(tag).is_none(),
            routing::is_valid_tag(tag),
            "{tag:?}"
        );
    }
}

#[test]
fn valid_tags_map_to_distinct_variables() {
    for (index, left) in VALID.iter().enumerate() {
        for right in &VALID[index + 1..] {
            assert_ne!(
                routing::variable_name("c", left),
                routing::variable_name("c", right),
                "{left} {right}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The attribute

fn parse(arguments: &str) -> syn::Result<Tag> {
    routing::parse_attribute(arguments.parse().unwrap())
}

#[test]
fn the_attribute_is_route_equals_a_string() {
    for arguments in [
        "route = \"physics\"",
        "route=\"physics\"",
        "route = \"physics\",",
    ] {
        assert_eq!(parse(arguments).unwrap().text, "physics", "{arguments}");
    }
    assert_eq!(
        parse("route = \"physics.broadphase\"").unwrap().text,
        "physics.broadphase"
    );
    // The literal's text is read as it is: an ill-named tag is the resolver's error, not the attribute's.
    assert_eq!(parse("route = \"Not A Tag\"").unwrap().text, "Not A Tag");
}

#[test]
fn any_other_argument_is_e0701() {
    for arguments in [
        "x",
        "route",
        "route =",
        "route = 5",
        "route = \"a\", \"b\"",
        "route = \"a\" extra",
        "name = \"a\"",
        "workgroup(4)",
        "route(\"a\")",
    ] {
        let error = parse(arguments).unwrap_err();
        assert_code(&error, Code::GpuArguments);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The resolver

fn tag(text: &str) -> Tag {
    Tag {
        text: text.to_owned(),
        span: Span::call_site(),
    }
}

fn environment(package: Option<&str>) -> Environment {
    Environment {
        package: package.map(str::to_owned),
        ..Environment::default()
    }
}

const PACKAGE: &str = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n";

/// The declarations of a manifest that has `routes` as its `[package.metadata.gust.routes]` table body.
fn declarations(routes: &str) -> manifest::Declarations {
    manifest::read(&format!(
        "{PACKAGE}\n[package.metadata.gust.routes]\n{routes}\n"
    ))
    .unwrap()
}

/// Resolves `route` against `routes` under `profile`, with the process variables `variables`.
fn resolve(
    routes: &str,
    route: &str,
    profile: Profile,
    variables: &[(&str, &str)],
) -> syn::Result<Option<routing::Resolved>> {
    let lookup = |name: &str| {
        variables
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, value)| (*value).to_owned())
    };
    let request = Request {
        tag: Some(tag(route)),
        variables: &lookup,
    };
    routing::resolve(
        &request,
        &environment(Some("demo")),
        &declarations(routes),
        profile,
    )
}

/// The target and the level `route` resolves to.
fn decided(
    routes: &str,
    route: &str,
    profile: Profile,
    variables: &[(&str, &str)],
) -> (Target, Level) {
    let resolved = resolve(routes, route, profile, variables)
        .unwrap()
        .expect("a tag was given");
    (resolved.target, resolved.level)
}

const PHYSICS: &str = "GUST_ROUTE__demo__physics";

#[test]
fn no_tag_is_no_route_and_reads_nothing() {
    let touched = std::cell::Cell::new(false);
    let lookup = |_: &str| {
        touched.set(true);
        None
    };
    let request = Request {
        tag: None,
        variables: &lookup,
    };
    let resolved = routing::resolve(
        &request,
        &environment(Some("demo")),
        &declarations("physics = {}"),
        Profile::Cpu,
    )
    .unwrap();
    assert!(resolved.is_none());
    assert!(!touched.get());
    assert!(Request::none().tag.is_none());
}

#[test]
fn nothing_said_is_the_gpu() {
    assert_eq!(
        decided("physics = {}", "physics", Profile::Other, &[]),
        (Target::Gpu, Level::Nothing)
    );
}

#[test]
fn the_source_default_is_the_next_level() {
    for (default, target) in [("cpu", Target::Cpu), ("gpu", Target::Gpu)] {
        assert_eq!(
            decided(
                &format!("physics = {{ default = \"{default}\" }}"),
                "physics",
                Profile::Other,
                &[]
            ),
            (target, Level::Default("physics".to_owned())),
            "{default}"
        );
    }
}

#[test]
fn a_tag_without_a_default_takes_the_nearest_ancestors() {
    let routes = "physics = { default = \"cpu\" }\n\"physics.broadphase\" = {}\n\"physics.broadphase.pairs\" = {}";
    assert_eq!(
        decided(routes, "physics.broadphase.pairs", Profile::Other, &[]),
        (Target::Cpu, Level::Default("physics".to_owned()))
    );
    // The nearest ancestor with a default wins, not the farthest.
    let routes = "physics = { default = \"cpu\" }\n\"physics.broadphase\" = { default = \"gpu\" }\n\"physics.broadphase.pairs\" = {}";
    assert_eq!(
        decided(routes, "physics.broadphase.pairs", Profile::Other, &[]),
        (Target::Gpu, Level::Default("physics.broadphase".to_owned()))
    );
    // The tag's own default beats its ancestors'.
    let routes = "physics = { default = \"gpu\" }\n\"physics.broadphase\" = { default = \"cpu\" }";
    assert_eq!(
        decided(routes, "physics.broadphase", Profile::Other, &[]),
        (Target::Cpu, Level::Default("physics.broadphase".to_owned()))
    );
    // An ancestor that is not declared does not matter: only declared defaults are read.
    let routes = "\"physics.broadphase\" = { default = \"cpu\" }";
    assert_eq!(
        decided(routes, "physics.broadphase", Profile::Other, &[]),
        (Target::Cpu, Level::Default("physics.broadphase".to_owned()))
    );
}

#[test]
fn the_variable_beats_every_default() {
    let routes = "physics = { default = \"cpu\" }";
    assert_eq!(
        decided(routes, "physics", Profile::Other, &[(PHYSICS, "gpu")]),
        (Target::Gpu, Level::Variable(PHYSICS.to_owned()))
    );
    let routes = "physics = { default = \"gpu\" }";
    assert_eq!(
        decided(routes, "physics", Profile::Other, &[(PHYSICS, "cpu")]),
        (Target::Cpu, Level::Variable(PHYSICS.to_owned()))
    );
}

#[test]
fn the_variable_is_the_one_of_the_package_and_the_tag() {
    let routes = "\"physics.broadphase\" = {}";
    let variable = "GUST_ROUTE__demo__physics__broadphase";
    assert_eq!(
        decided(
            routes,
            "physics.broadphase",
            Profile::Other,
            &[(variable, "cpu")]
        ),
        (Target::Cpu, Level::Variable(variable.to_owned()))
    );
    // Another tag's, another package's and the parent tag's variable say nothing about this one.
    for other in [
        "GUST_ROUTE__demo__physics",
        "GUST_ROUTE__other__physics__broadphase",
        "GUST_ROUTE__demo__physics__broadphase__pairs",
    ] {
        assert_eq!(
            decided(
                routes,
                "physics.broadphase",
                Profile::Other,
                &[(other, "cpu")]
            ),
            (Target::Gpu, Level::Nothing),
            "{other}"
        );
    }
}

#[test]
fn the_cpu_profile_is_the_first_level() {
    // It beats the source default (which yields: a crate with the usual `default = "gpu"` still builds for the CPU),
    // agrees with a `cpu` variable and needs no variable.
    for variables in [&[][..], &[(PHYSICS, "cpu")][..]] {
        assert_eq!(
            decided(
                "physics = { default = \"gpu\" }",
                "physics",
                Profile::Cpu,
                variables
            ),
            (Target::Cpu, Level::Profile)
        );
    }
}

#[test]
fn a_variable_that_says_gpu_under_the_cpu_profile_is_e0749() {
    let error = resolve("physics = {}", "physics", Profile::Cpu, &[(PHYSICS, "gpu")]).unwrap_err();
    assert_code(&error, Code::RouteTarget);
    let message = error.to_string();
    assert!(
        message.contains(PHYSICS) && message.contains("GUST_PROFILE=cpu"),
        "{message}"
    );
}

#[test]
fn a_variable_that_is_not_a_target_is_e0749() {
    for value in ["GPU", "native", "web", " cpu", "cpu ", "0"] {
        let error = resolve(
            "physics = {}",
            "physics",
            Profile::Other,
            &[(PHYSICS, value)],
        )
        .unwrap_err();
        assert_code(&error, Code::RouteTarget);
        let message = error.to_string();
        assert!(
            message.contains(PHYSICS) && message.contains("gpu and cpu"),
            "{message}"
        );
    }
    // Under the CPU profile the value is read too: the variable is wrong whatever the profile.
    let error = resolve("physics = {}", "physics", Profile::Cpu, &[(PHYSICS, "web")]).unwrap_err();
    assert_code(&error, Code::RouteTarget);
}

#[test]
fn an_empty_variable_is_unset() {
    // The process lookup drops an empty value; the resolver never sees it.
    assert_eq!(
        routing::process_variable("GUST_ROUTE__demo__never_set_by_a_test"),
        None
    );
    assert_eq!(
        decided("physics = {}", "physics", Profile::Other, &[]),
        (Target::Gpu, Level::Nothing)
    );
}

#[test]
fn an_ill_named_tag_is_e0748_at_the_literal() {
    for text in INVALID {
        let error = resolve("physics = {}", text, Profile::Other, &[]).unwrap_err();
        assert_code(&error, Code::RouteTag);
        let message = error.to_string();
        assert!(message.contains("ill-named"), "{text:?}: {message}");
    }
}

#[test]
fn a_tag_that_is_not_declared_is_e0748_and_names_the_ones_that_are() {
    let error = resolve(
        "physics = {}\n\"physics.broadphase\" = {}",
        "render",
        Profile::Other,
        &[],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTag);
    let message = error.to_string();
    assert!(
        message.contains("`render` is not declared")
            && message.contains("`physics`")
            && message.contains("`physics.broadphase`")
            && message.contains("only by the crate that declares it"),
        "{message}"
    );
    // The parent of a declared tag is not declared by that.
    let error = resolve(
        "\"physics.broadphase\" = {}",
        "physics",
        Profile::Other,
        &[],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTag);
    // A crate that declares no routes says so.
    let error = routing::resolve(
        &Request {
            tag: Some(tag("physics")),
            variables: &|_| None,
        },
        &environment(Some("demo")),
        &manifest::Declarations::default(),
        Profile::Other,
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTag);
    assert!(error.to_string().contains("declares none"), "{error}");
}

#[test]
fn a_declared_key_that_is_ill_named_is_e0748_when_a_module_is_routed() {
    let error = resolve(
        "physics = {}\n\"Bad.Tag\" = {}",
        "physics",
        Profile::Other,
        &[],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTag);
    assert!(error.to_string().contains("`Bad.Tag` declared"), "{error}");
}

// ---------------------------------------------------------------------------------------------------------------
// The expansions

/// A scratch crate directory below the temporary directory, with `manifest` as its `Cargo.toml`. Removed when dropped.
struct Crate(PathBuf);

impl Crate {
    fn new(name: &str, routes: &str) -> Self {
        Self::with_manifest(
            name,
            Some(&format!(
                "{PACKAGE}\n[package.metadata.gust.routes]\n{routes}\n"
            )),
        )
    }

    fn with_manifest(name: &str, manifest: Option<&str>) -> Self {
        let directory =
            std::env::temp_dir().join(format!("gust-m8r1-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        if let Some(text) = manifest {
            fs::write(directory.join("Cargo.toml"), text).unwrap();
        }
        Self(directory)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Crate {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const VECTOR_ADD: &str = "mod vector_add { \
    #[kernel] \
    pub fn add(id: SV_DispatchThreadID, a: StructuredBuffer<float>, b: StructuredBuffer<float>, \
               mut out: RWStructuredBuffer<float>) { \
        let i = id.x; \
        if i < out.len() { out[i] = a[i] + b[i]; } \
    } \
}";

const GPU_ONLY: &str = "mod gpu_only { \
    #[kernel(cpu = \"unsupported\")] \
    pub fn add(id: SV_DispatchThreadID, a: StructuredBuffer<float>, b: StructuredBuffer<float>, \
               mut out: RWStructuredBuffer<float>) { \
        let i = id.x; \
        if i < out.len() { out[i] = a[i] + b[i]; } \
    } \
}";

/// `source` expanded as a module of the crate in `directory`, routed by `route`, under `profile`, with `variables`.
fn expand(
    source: &str,
    directory: &Path,
    profile: Profile,
    route: Option<&str>,
    variables: &[(&str, &str)],
) -> syn::Result<TokenStream> {
    let lookup = |name: &str| {
        variables
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, value)| (*value).to_owned())
    };
    let request = Request {
        tag: route.map(tag),
        variables: &lookup,
    };
    let environment = Environment {
        manifest_dir: Some(directory.to_path_buf()),
        export_dir: None,
        package: Some("demo".to_owned()),
        reported_file: None,
    };
    expand_in_routed(parse_str(source).unwrap(), &environment, profile, &request)
}

fn text(
    source: &str,
    directory: &Path,
    profile: Profile,
    route: Option<&str>,
    variables: &[(&str, &str)],
) -> String {
    expand(source, directory, profile, route, variables)
        .unwrap()
        .to_string()
}

fn squeezed(text: &str) -> String {
    text.split_whitespace().collect()
}

/// The `option_env!` item the expansion emits for the variable `name`.
fn tracked(name: &str) -> String {
    format!(
        "const _ : :: core :: option :: Option < & :: core :: primitive :: str > = :: core :: option_env ! (\"{name}\") ;"
    )
}

/// The variable `cargo gust` sets to the profile it resolved the routes for (m8.z).
const ROUTED: &str = "GUST_ROUTED_PROFILE";

/// `routed` without the `option_env!` items of the route variable and of `GUST_ROUTED_PROFILE`.
fn without_tracking(routed: &str) -> String {
    routed
        .replace(&tracked(PHYSICS), "")
        .replace(&tracked(ROUTED), "")
}

const GPU_MARKER: &str = "impl :: gust :: target :: GpuKernel for add";
const GATE: &str = "# [cfg (any (test , gust_dual))]";
const UNGATED_IMPL: &str = "impl :: gust :: Run < add , 3 > for :: gust :: cpu :: CpuDevice";
const MANIFEST_ITEM: &str = ":: core :: include_str ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/Cargo.toml\")) ;";

#[test]
fn an_untagged_module_is_what_it_was_and_reads_no_route() {
    let scratch = Crate::new("untagged", "physics = { default = \"cpu\" }");
    let plain = text(VECTOR_ADD, scratch.path(), Profile::Other, None, &[]);
    assert!(
        plain.contains("target : :: gust :: target :: Target :: Gpu"),
        "{plain}"
    );
    assert!(plain.contains(GPU_MARKER), "{plain}");
    assert!(plain.contains("require_artifact (false)"), "{plain}");
    assert_eq!(plain.matches(GATE).count(), 1, "{plain}");
    assert!(!plain.contains("GUST_ROUTE__"), "{plain}");
    // The variables and the declared default say nothing about a module that carries no tag.
    let with_variables = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        None,
        &[(PHYSICS, "cpu")],
    );
    assert_eq!(plain, with_variables);
}

#[test]
fn an_untagged_module_under_the_cpu_profile_is_a_cpu_target() {
    // D137's first level: the `cpu` profile sends every item to the CPU, tagged or not, so the descriptor says `Cpu`
    // and there is no `GpuKernel` impl (review r1, S1).
    let scratch = Crate::new("untagged-cpu", "physics = {}");
    let cpu = text(VECTOR_ADD, scratch.path(), Profile::Cpu, None, &[]);
    assert!(
        cpu.contains("target : :: gust :: target :: Target :: Cpu"),
        "{cpu}"
    );
    assert!(!cpu.contains("Target :: Gpu"), "{cpu}");
    assert!(!cpu.contains("GpuKernel"), "{cpu}");
    assert_eq!(cpu.matches(UNGATED_IMPL).count(), 1, "{cpu}");
    assert_eq!(cpu.matches(GATE).count(), 0, "{cpu}");
    // The no-op artifact check of the `cpu` profile stays as it was before routing; only a CPU route drops it.
    assert_eq!(cpu.matches("require_artifact (false)").count(), 1, "{cpu}");
}

#[test]
fn a_gpu_routed_module_is_the_untagged_one_plus_the_tracked_variable() {
    let scratch = Crate::new("gpu-routed", "physics = {}");
    let plain = text(VECTOR_ADD, scratch.path(), Profile::Other, None, &[]);
    for variables in [&[][..], &[(PHYSICS, "gpu")][..]] {
        let routed = text(
            VECTOR_ADD,
            scratch.path(),
            Profile::Other,
            Some("physics"),
            variables,
        );
        assert_eq!(routed.matches(&tracked(PHYSICS)).count(), 1, "{routed}");
        assert_eq!(
            squeezed(&without_tracking(&routed)),
            squeezed(&plain),
            "{variables:?}"
        );
        assert!(routed.contains("Target :: Gpu") && routed.contains(GPU_MARKER));
    }
}

#[test]
fn a_cpu_routed_module_is_a_cpu_body_in_a_gpu_build() {
    let scratch = Crate::new("cpu-routed", "physics = {}");
    let routed = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "cpu")],
    );
    // The target and the marker: `HeadlessDevice` cannot take this kernel (ruling 5).
    assert!(
        routed.contains("target : :: gust :: target :: Target :: Cpu"),
        "{routed}"
    );
    assert!(!routed.contains("Target :: Gpu"), "{routed}");
    assert!(!routed.contains("GpuKernel"), "{routed}");
    // The impl is ungated, so `CpuDevice` runs it in a production build.
    assert_eq!(routed.matches(UNGATED_IMPL).count(), 1, "{routed}");
    assert_eq!(routed.matches(GATE).count(), 0, "{routed}");
    assert!(!routed.contains("any (test"), "{routed}");
    // No artifact is looked up, embedded or required, and no manifest check is made: a strict profile has nothing to
    // complain of.
    assert!(!routed.contains("require_artifact"), "{routed}");
    assert!(!routed.contains("require_manifest"), "{routed}");
    assert!(!routed.contains("include_bytes"), "{routed}");
    assert!(routed.contains("artifacts : & []"), "{routed}");
    // The variable is tracked, once.
    assert_eq!(routed.matches(&tracked(PHYSICS)).count(), 1, "{routed}");
}

#[test]
fn a_cpu_routed_module_is_the_cpu_profile_expansion_plus_the_tracked_variable() {
    // Since S1 the `cpu` profile gives an untagged module the CPU target and no marker too, so a CPU-routed module in
    // a GPU build is that expansion, token for token, except for the tracked variable and the no-op artifact check
    // that only a CPU route drops.
    let scratch = Crate::new("cpu-equal", "physics = {}");
    let cpu_profile = text(VECTOR_ADD, scratch.path(), Profile::Cpu, None, &[]);
    let routed = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "cpu")],
    );
    let expected = cpu_profile.replace(
        "const _ : () = :: gust :: profile :: require_artifact (false) ;",
        "",
    );
    assert_eq!(squeezed(&without_tracking(&routed)), squeezed(&expected));
}

#[test]
fn a_tag_under_the_cpu_profile_is_a_cpu_target_with_no_variable() {
    let scratch = Crate::new("tag-cpu-profile", "physics = { default = \"gpu\" }");
    let routed = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Cpu,
        Some("physics"),
        &[],
    );
    assert!(routed.contains("Target :: Cpu"), "{routed}");
    assert!(!routed.contains("GpuKernel"), "{routed}");
    assert_eq!(routed.matches(UNGATED_IMPL).count(), 1, "{routed}");
}

#[test]
fn the_source_default_routes_a_build_that_sets_nothing() {
    let scratch = Crate::new("default-cpu", "physics = { default = \"cpu\" }");
    let routed = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[],
    );
    assert!(
        routed.contains("Target :: Cpu") && !routed.contains("GpuKernel"),
        "{routed}"
    );
    // The variable still wins over it.
    let flipped = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "gpu")],
    );
    assert!(
        flipped.contains("Target :: Gpu") && flipped.contains(GPU_MARKER),
        "{flipped}"
    );
}

#[test]
fn the_manifest_is_tracked_once_whatever_the_route() {
    let scratch = Crate::new("tracking", "physics = {}");
    for route in [None, Some("physics")] {
        let routed = text(VECTOR_ADD, scratch.path(), Profile::Other, route, &[]);
        assert_eq!(routed.matches(MANIFEST_ITEM).count(), 1, "{routed}");
        let directory = scratch.path().to_string_lossy().replace('\\', "/");
        assert!(!routed.replace('\\', "/").contains(&directory), "{routed}");
    }
}

#[test]
fn a_dotted_tag_tracks_the_variable_with_double_underscores() {
    let scratch = Crate::new("dotted", "\"physics.broadphase\" = {}");
    let routed = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics.broadphase"),
        &[],
    );
    assert_eq!(
        routed
            .matches(&tracked("GUST_ROUTE__demo__physics__broadphase"))
            .count(),
        1,
        "{routed}"
    );
}

#[test]
fn a_tag_the_crate_does_not_declare_is_e0748_in_the_expansion() {
    let scratch = Crate::new("undeclared", "physics = {}");
    let error = expand(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("render"),
        &[],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTag);
    // No manifest at all: nothing is declared.
    let none = Crate::with_manifest("no-manifest", None);
    let error = expand(
        VECTOR_ADD,
        none.path(),
        Profile::Other,
        Some("physics"),
        &[],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTag);
    // No directory (a unit test's scratch crate) declares nothing either.
    let environment = Environment::default();
    let request = Request {
        tag: Some(tag("physics")),
        variables: &|_| None,
    };
    let error = expand_in_routed(
        parse_str(VECTOR_ADD).unwrap(),
        &environment,
        Profile::Other,
        &request,
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTag);
}

#[test]
fn a_tag_is_resolved_for_the_module_once_and_all_its_kernels_share_it() {
    let scratch = Crate::new("two-kernels", "physics = {}");
    let source = "mod two { \
        #[kernel] pub fn first(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) { out[id.x] = 1.0; } \
        #[kernel] pub fn second(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) { out[id.x] = 2.0; } \
    }";
    let routed = text(
        source,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "cpu")],
    );
    assert_eq!(routed.matches("Target :: Cpu").count(), 2, "{routed}");
    assert_eq!(routed.matches("GpuKernel").count(), 0, "{routed}");
    assert_eq!(routed.matches(&tracked(PHYSICS)).count(), 1, "{routed}");
}

// ---------------------------------------------------------------------------------------------------------------
// The tag's set

#[test]
fn a_target_outside_the_explicit_set_is_e0749_at_the_kernel() {
    let scratch = Crate::new("set-gpu", "physics = { targets = [\"gpu\"] }");
    let error = expand(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "cpu")],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTarget);
    let message = error.to_string();
    assert!(
        message.contains("kernel `add` is routed to `cpu` by the variable")
            && message.contains("`targets = [\"gpu\"]`"),
        "{message}"
    );
    // The same set with the GPU route is fine, and so is a set that lists both.
    expand(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[],
    )
    .unwrap();
    let both = Crate::new("set-both", "physics = { targets = [\"gpu\", \"cpu\"] }");
    for value in ["gpu", "cpu"] {
        expand(
            VECTOR_ADD,
            both.path(),
            Profile::Other,
            Some("physics"),
            &[(PHYSICS, value)],
        )
        .unwrap();
    }
}

#[test]
fn a_cpu_only_set_refuses_the_gpu_that_nothing_routed() {
    let scratch = Crate::new("set-cpu", "physics = { targets = [\"cpu\"] }");
    let error = expand(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTarget);
    let message = error.to_string();
    assert!(
        message.contains("because no variable and no default routes it"),
        "{message}"
    );
    // Routed to the CPU by the variable, or by the profile, it builds.
    expand(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "cpu")],
    )
    .unwrap();
    expand(
        VECTOR_ADD,
        scratch.path(),
        Profile::Cpu,
        Some("physics"),
        &[],
    )
    .unwrap();
    let defaulted = Crate::new(
        "set-cpu-default",
        "physics = { targets = [\"cpu\"], default = \"cpu\" }",
    );
    expand(
        VECTOR_ADD,
        defaulted.path(),
        Profile::Other,
        Some("physics"),
        &[],
    )
    .unwrap();
}

#[test]
fn the_message_names_what_decided_the_target() {
    let by_default = Crate::new(
        "decided-default",
        "physics = { targets = [\"gpu\"], default = \"cpu\" }",
    );
    let error = expand(
        VECTOR_ADD,
        by_default.path(),
        Profile::Other,
        Some("physics"),
        &[],
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("by the `default` of `physics` in Cargo.toml"),
        "{error}"
    );
    let ancestor = Crate::new(
        "decided-ancestor",
        "physics = { default = \"cpu\" }\n\"physics.broadphase\" = { targets = [\"gpu\"] }",
    );
    let error = expand(
        VECTOR_ADD,
        ancestor.path(),
        Profile::Other,
        Some("physics.broadphase"),
        &[],
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("by the `default` of its ancestor `physics` in Cargo.toml"),
        "{error}"
    );
}

#[test]
fn without_targets_the_set_is_both_for_a_kernel_with_a_cpu_form() {
    let scratch = Crate::new("no-targets", "physics = {}");
    for value in ["gpu", "cpu"] {
        expand(
            VECTOR_ADD,
            scratch.path(),
            Profile::Other,
            Some("physics"),
            &[(PHYSICS, value)],
        )
        .unwrap();
    }
}

#[test]
fn without_targets_the_set_of_a_gpu_only_kernel_is_the_gpu() {
    let scratch = Crate::new("gpu-only", "physics = {}");
    // The GPU route builds.
    let gpu = text(
        GPU_ONLY,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[],
    );
    assert!(gpu.contains("Target :: Gpu"), "{gpu}");
    // The CPU route is outside the set.
    let error = expand(
        GPU_ONLY,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "cpu")],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTarget);
    assert!(
        error
            .to_string()
            .contains("just `gpu` (the kernel has no CPU form"),
        "{error}"
    );
    // The default routes it the same way.
    let defaulted = Crate::new("gpu-only-default", "physics = { default = \"cpu\" }");
    let error = expand(
        GPU_ONLY,
        defaulted.path(),
        Profile::Other,
        Some("physics"),
        &[],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTarget);
}

#[test]
fn a_set_that_lists_cpu_for_a_kernel_with_no_cpu_form_is_e0744_in_every_build() {
    let scratch = Crate::new("e0744", "physics = { targets = [\"gpu\", \"cpu\"] }");
    for (profile, variables) in [
        (Profile::Other, &[][..]),
        (Profile::Other, &[(PHYSICS, "cpu")][..]),
        (Profile::Other, &[(PHYSICS, "gpu")][..]),
        (Profile::Cpu, &[][..]),
    ] {
        let error = expand(
            GPU_ONLY,
            scratch.path(),
            profile,
            Some("physics"),
            variables,
        )
        .unwrap_err();
        assert_code(&error, Code::ProfileNotCovered);
        let message = error.to_string();
        assert!(
            message.contains("kernel `add` has no CPU form")
                && message.contains("`physics`")
                && message.contains("drop `cpu` from the set"),
            "{message}"
        );
    }
    // A set that does not list `cpu` is fine for the kernel (GPU-only is what it says).
    let gpu_set = Crate::new("e0744-gpu-set", "physics = { targets = [\"gpu\"] }");
    expand(
        GPU_ONLY,
        gpu_set.path(),
        Profile::Other,
        Some("physics"),
        &[],
    )
    .unwrap();
}

#[test]
fn the_cpu_profile_decides_over_the_set_as_it_does_for_an_untagged_gpu_only_kernel() {
    // `GUST_PROFILE=cpu` builds the whole crate for the CPU; a kernel that has no CPU form has no impl there, tagged
    // or not, and the set is not consulted (the profile is the first level of D66).
    let scratch = Crate::new("profile-over-set", "physics = { targets = [\"gpu\"] }");
    let routed = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Cpu,
        Some("physics"),
        &[],
    );
    assert!(routed.contains("Target :: Cpu"), "{routed}");
    let gpu_only = text(GPU_ONLY, scratch.path(), Profile::Cpu, Some("physics"), &[]);
    assert!(!gpu_only.contains("impl :: gust :: Run"), "{gpu_only}");
}

// ---------------------------------------------------------------------------------------------------------------
// The portable floor still holds a CPU-routed kernel (ruling 3)

const TOO_WIDE: &str = "mod wide { \
    #[kernel(workgroup_size(512, 1, 1))] \
    pub fn add(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) { out[id.x] = 1.0; } \
}";

#[test]
fn a_cpu_routed_kernel_over_the_portable_floor_is_still_refused() {
    let scratch = Crate::with_manifest(
        "floor",
        Some(&format!(
            "{PACKAGE}\n[package.metadata.gust]\nlimits = \"portable\"\n\n[package.metadata.gust.routes]\nphysics = {{}}\n"
        )),
    );
    // Routed to the GPU, to the CPU by a variable, by the profile: the floor is checked first, and the same.
    for (profile, variables) in [
        (Profile::Other, &[][..]),
        (Profile::Other, &[(PHYSICS, "cpu")][..]),
        (Profile::Cpu, &[][..]),
    ] {
        let error = expand(
            TOO_WIDE,
            scratch.path(),
            profile,
            Some("physics"),
            variables,
        )
        .unwrap_err();
        assert_code(&error, Code::PortableFloor);
    }
    // The same kernel without the declaration is fine, routed or not: the floor is the crate's choice.
    let plain = Crate::new("floor-undeclared", "physics = {}");
    expand(
        TOO_WIDE,
        plain.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "cpu")],
    )
    .unwrap();
}

// ---------------------------------------------------------------------------------------------------------------
// A map function

const MAP: &str = "pub fn doubled(x: f32) -> f32 { x * 2.0 }";

fn map_text(
    directory: &Path,
    profile: Profile,
    route: Option<&str>,
    variables: &[(&str, &str)],
) -> syn::Result<String> {
    let lookup = |name: &str| {
        variables
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, value)| (*value).to_owned())
    };
    let request = Request {
        tag: route.map(tag),
        variables: &lookup,
    };
    let environment = Environment {
        manifest_dir: Some(directory.to_path_buf()),
        export_dir: None,
        package: Some("demo".to_owned()),
        reported_file: None,
    };
    expand_map_in_routed(parse_str(MAP).unwrap(), &environment, profile, &request)
        .map(|tokens| tokens.to_string())
}

#[test]
fn a_gpu_fn_follows_its_route_in_the_kernel_and_in_the_host_functions() {
    let scratch = Crate::new("map", "physics = {}");
    let gpu = map_text(scratch.path(), Profile::Other, Some("physics"), &[]).unwrap();
    assert!(
        gpu.contains("Target :: Gpu") && gpu.contains("GpuKernel"),
        "{gpu}"
    );
    assert!(gpu.contains("map_gpu"), "{gpu}");
    assert_eq!(gpu.matches(&tracked(PHYSICS)).count(), 1, "{gpu}");
    let cpu = map_text(
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "cpu")],
    )
    .unwrap();
    assert!(cpu.contains("Target :: Cpu"), "{cpu}");
    assert!(!cpu.contains("GpuKernel"), "{cpu}");
    // The CPU family: no GPU API in the host functions, the `cpu` chunk body instead.
    assert!(
        !cpu.contains("map_gpu") && !cpu.contains("gust_wgpu"),
        "{cpu}"
    );
    assert!(cpu.contains("__gust_map_cpu_chunk"), "{cpu}");
    assert!(!cpu.contains("require_artifact"), "{cpu}");
    // The untagged function is what it was.
    let plain = map_text(scratch.path(), Profile::Other, None, &[(PHYSICS, "cpu")]).unwrap();
    assert!(
        plain.contains("Target :: Gpu") && plain.contains("map_gpu"),
        "{plain}"
    );
    assert!(!plain.contains("GUST_ROUTE__"), "{plain}");
}

#[test]
fn a_gpu_fn_reports_a_malformed_manifest_at_its_name_and_a_bad_route_at_the_tag() {
    let broken = Crate::with_manifest(
        "map-broken",
        Some(&format!(
            "{PACKAGE}\n[package.metadata.gust]\nlimits = \"device\"\n"
        )),
    );
    let error = map_text(broken.path(), Profile::Other, Some("physics"), &[]).unwrap_err();
    assert_code(&error, Code::GustMetadata);
    let scratch = Crate::new("map-bad", "physics = {}");
    let error = map_text(scratch.path(), Profile::Other, Some("render"), &[]).unwrap_err();
    assert_code(&error, Code::RouteTag);
    let error = map_text(
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(PHYSICS, "web")],
    )
    .unwrap_err();
    assert_code(&error, Code::RouteTarget);
}

// ---------------------------------------------------------------------------------------------------------------
// The entry point

fn output_text(attribute: &str, item: &str) -> String {
    crate::gpu_impl_routed(
        attribute.parse().unwrap(),
        item.parse().unwrap(),
        Profile::Other,
        &|_| None,
    )
    .to_string()
}

#[test]
fn a_route_argument_goes_to_the_resolver_and_anything_else_stays_e0701() {
    // The crate under test is `gust-compute-macros` itself, which declares no routes.
    let output = output_text("route = \"physics\"", VECTOR_ADD);
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert!(output.contains("[GUST-E0748]"), "{output}");
    assert!(
        output.contains("mod vector_add"),
        "the module follows the error: {output}"
    );
    let output = output_text("route = \"Not A Tag\"", VECTOR_ADD);
    assert!(
        output.contains("[GUST-E0748]") && output.contains("ill-named"),
        "{output}"
    );
    for attribute in ["x", "route = 5", "route = \"a\", \"b\"", "workgroup(4)"] {
        let output = output_text(attribute, VECTOR_ADD);
        assert!(output.contains("[GUST-E0701]"), "{attribute}: {output}");
        assert!(output.contains("mod vector_add"), "{attribute}: {output}");
    }
    let output = output_text("", VECTOR_ADD);
    assert!(!output.contains("compile_error"), "{output}");
}

#[test]
fn a_route_on_a_function_is_resolved_too() {
    let output = output_text("route = \"physics\"", MAP);
    assert!(output.contains("[GUST-E0748]"), "{output}");
    assert!(
        output.contains("pub fn doubled"),
        "the function follows the error: {output}"
    );
}

#[test]
fn a_route_on_another_item_is_still_not_a_module() {
    let output = output_text("route = \"physics\"", "struct S;");
    assert!(output.contains("[GUST-E0702]"), "{output}");
}

// ---------------------------------------------------------------------------------------------------------------
// The guard against a build the driver did not resolve the routes for (m8.z)

/// The `if` the guard emits for a build of `gust` with the profile `built`.
fn guard_branch(built: &str) -> String {
    format!(":: gust :: profile :: is_profile (\"{built}\")")
}

#[test]
fn a_tagged_module_without_a_routed_profile_tracks_the_variable_and_checks_nothing() {
    // Plain cargo (D67): the variable is read so that rustc tracks it, and nothing is compared.
    let scratch = Crate::new("guard-unset", "physics = {}");
    let routed = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[],
    );
    assert_eq!(routed.matches(&tracked(ROUTED)).count(), 1, "{routed}");
    assert!(!routed.contains("is_profile"), "{routed}");
    assert!(!routed.contains("panic"), "{routed}");
}

#[test]
fn a_routed_profile_makes_one_check_per_other_profile_naming_both() {
    let scratch = Crate::new("guard-set", "physics = {}");
    for routed_profile in ["dev", "native", "web", "cpu"] {
        let routed = text(
            VECTOR_ADD,
            scratch.path(),
            Profile::Other,
            Some("physics"),
            &[(ROUTED, routed_profile)],
        );
        assert_eq!(routed.matches(&tracked(ROUTED)).count(), 1, "{routed}");
        assert_eq!(routed.matches("is_profile").count(), 3, "{routed}");
        assert!(!routed.contains(&guard_branch(routed_profile)), "{routed}");
        for built in ["dev", "native", "web", "cpu"] {
            if built == routed_profile {
                continue;
            }
            assert!(routed.contains(&guard_branch(built)), "{built}: {routed}");
            let message = routing::routed_message(routed_profile, built);
            assert!(routed.contains(&message), "{routed_profile} {built}");
            assert!(
                message.contains(&format!("for the profile `{routed_profile}`"))
                    && message.contains(&format!("of gust is `{built}`")),
                "{message}"
            );
        }
    }
}

#[test]
fn the_message_of_the_guard_names_the_causes_and_the_ways_out() {
    let message = routing::routed_message("dev", "web");
    for part in [
        "(GUST_ROUTED_PROFILE)",
        "`build.target` in .cargo/config.toml",
        "CARGO_BUILD_TARGET",
        "a custom profile inherits `release`",
        "set only in the `[env]` table of the config",
        "built for the host as a build dependency",
        "set GUST_PROFILE=web",
        "set GUST_PROFILE to the profile you mean in the environment",
        "unset it if you are running plain cargo",
    ] {
        assert!(message.contains(part), "{part}: {message}");
    }
    // It is the literal of a `panic!`, which reads `{` and `}` as a format.
    assert!(
        !message.contains('{') && !message.contains('}'),
        "{message}"
    );
}

/// The tokens of `message` as a string literal, the way the expansion's text spells it.
fn literal(message: &str) -> String {
    proc_macro2::Literal::string(message).to_string()
}

#[test]
fn a_routed_profile_that_is_no_profile_is_a_compile_error_naming_the_variable_the_value_and_the_profiles()
 {
    // m9.0 (m8.z re-check N2 and N3). Before, "bogus" made four checks that each named a profile it was not, a value
    // with a `{` was a format-string error of the `panic!`, and an empty value was no check at all. Now every value that
    // is not a profile is one error that names the variable, the value and the profiles.
    let scratch = Crate::new("guard-bogus", "physics = {}");
    for (value, shown) in [
        ("bogus", "\"bogus\""),
        ("", "\"\""),
        ("DEV", "\"DEV\""),
        ("dev ", "\"dev \""),
        ("dev+subgroups", "\"dev+subgroups\""),
        ("de{v}", "\"de{v}\""),
        ("{0}", "\"{0}\""),
        ("}{", "\"}{\""),
        ("a\"b", "\"a\\\"b\""),
    ] {
        let routed = text(
            VECTOR_ADD,
            scratch.path(),
            Profile::Other,
            Some("physics"),
            &[(ROUTED, value)],
        );
        let message = routing::unknown_routed_profile_message(value);
        assert!(
            message.contains("GUST_ROUTED_PROFILE")
                && message.contains(&format!("is {shown},"))
                && message.contains("the valid profiles are dev, native, web, cpu."),
            "{value:?}: {message}"
        );
        // One `compile_error!` that carries the message, no check against the build and no `panic!` (a `{` in the value
        // would be a format string there), and the variable is still tracked.
        assert_eq!(
            routed.matches("compile_error").count(),
            1,
            "{value:?}: {routed}"
        );
        assert!(routed.contains(&literal(&message)), "{value:?}: {routed}");
        assert!(!routed.contains("is_profile"), "{value:?}: {routed}");
        assert!(!routed.contains("panic"), "{value:?}: {routed}");
        assert_eq!(
            routed.matches(&tracked(ROUTED)).count(),
            1,
            "{value:?}: {routed}"
        );
        // The rest of the module is still expanded: the guard is one error beside it, not a replacement for it.
        assert!(routed.contains(GPU_MARKER), "{value:?}: {routed}");
    }
    // The four profiles are not errors, and their message is unchanged.
    for value in ["dev", "native", "web", "cpu"] {
        let routed = text(
            VECTOR_ADD,
            scratch.path(),
            Profile::Other,
            Some("physics"),
            &[(ROUTED, value)],
        );
        assert!(!routed.contains("compile_error"), "{value}: {routed}");
    }
}

#[test]
fn the_process_keeps_an_empty_routed_profile_and_no_other_empty_variable() {
    use std::ffi::OsString;

    let value = |text: &str| Some(OsString::from(text));
    // Unset is unset for both: plain cargo, nothing is checked (D67).
    assert_eq!(routing::variable_value(ROUTED, None), None);
    assert_eq!(routing::variable_value(PHYSICS, None), None);
    // Empty: a route variable reads it as unset (as it always did), the routed profile reads it as a value to refuse.
    assert_eq!(routing::variable_value(PHYSICS, value("")), None);
    assert_eq!(
        routing::variable_value(ROUTED, value("")),
        Some(String::new())
    );
    assert_eq!(
        routing::variable_value(ROUTED, value("web")),
        Some("web".to_owned())
    );
    assert_eq!(
        routing::variable_value(PHYSICS, value("cpu")),
        Some("cpu".to_owned())
    );
    // Text that is not UTF-8 is unset for a route variable and, lossily read, a value the guard refuses for the routed
    // profile. (`OsString` from wide units is the portable way to make one.)
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStringExt as _;
        let lone_surrogate = OsString::from_wide(&[0xD800]);
        assert_eq!(
            routing::variable_value(PHYSICS, Some(lone_surrogate.clone())),
            None
        );
        let routed = routing::variable_value(ROUTED, Some(lone_surrogate)).unwrap();
        assert!(!routed.is_empty() && !["dev", "native", "web", "cpu"].contains(&routed.as_str()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt as _;
        let invalid = OsString::from_vec(vec![0xFF]);
        assert_eq!(
            routing::variable_value(PHYSICS, Some(invalid.clone())),
            None
        );
        let routed = routing::variable_value(ROUTED, Some(invalid)).unwrap();
        assert!(!routed.is_empty() && !["dev", "native", "web", "cpu"].contains(&routed.as_str()));
    }
}

#[test]
fn two_routed_modules_in_one_crate_each_carry_their_own_check() {
    // m9.0 (m8.z re-check N3). Each module's expansion holds its own `option_env!` and its own three `is_profile` checks,
    // spanned at its own tag; the items are `const _`, so two of them in one crate do not collide. That the two checks
    // both fire in rustc is the probe in the report of the slice (one crate, two modules, `GUST_ROUTED_PROFILE=web` under
    // a `dev` build: two errors, one per tag literal).
    let scratch = Crate::new("guard-two", "physics = {}\nfusion = {}");
    let second = VECTOR_ADD
        .replace("vector_add", "vector_two")
        .replace("fn add(", "fn two(");
    assert_ne!(second, VECTOR_ADD);
    let first = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        Some("physics"),
        &[(ROUTED, "web")],
    );
    let other = text(
        &second,
        scratch.path(),
        Profile::Other,
        Some("fusion"),
        &[(ROUTED, "web")],
    );
    for (name, expansion) in [("vector_add", &first), ("vector_two", &other)] {
        assert!(expansion.contains(&format!("mod {name}")), "{expansion}");
        assert_eq!(
            expansion.matches(&tracked(ROUTED)).count(),
            1,
            "{expansion}"
        );
        assert_eq!(expansion.matches("is_profile").count(), 3, "{expansion}");
        for built in ["dev", "native", "cpu"] {
            assert!(
                expansion.contains(&guard_branch(built)),
                "{built}: {expansion}"
            );
            assert!(
                expansion.contains(&routing::routed_message("web", built)),
                "{built}: {expansion}"
            );
        }
        assert!(!expansion.contains(&guard_branch("web")), "{expansion}");
    }
}

#[test]
fn an_untagged_module_reads_no_routed_profile() {
    // Only a kernel whose target comes from a route tag can be misrouted, so an untagged module is what it was.
    let scratch = Crate::new("guard-untagged", "physics = {}");
    let plain = text(VECTOR_ADD, scratch.path(), Profile::Other, None, &[]);
    let with_variable = text(
        VECTOR_ADD,
        scratch.path(),
        Profile::Other,
        None,
        &[(ROUTED, "native")],
    );
    assert_eq!(plain, with_variable);
    assert!(!plain.contains(ROUTED), "{plain}");
}

#[test]
fn the_guard_does_not_change_the_target_a_tag_resolves_to() {
    // The guard reads the profile; it never decides one.
    let scratch = Crate::new("guard-target", "physics = { default = \"cpu\" }");
    for variables in [&[][..], &[(ROUTED, "dev")][..], &[(ROUTED, "native")][..]] {
        let routed = text(
            VECTOR_ADD,
            scratch.path(),
            Profile::Other,
            Some("physics"),
            variables,
        );
        assert!(routed.contains("Target :: Cpu"), "{variables:?}: {routed}");
    }
}

//! Build-time routing (M8, D65 to D67, D137): the target a `#[gpu_module]` module is built for.
//!
//! `#[gpu_module(route = "physics")]` on a module (or on a `#[gpu_fn] fn`) names a tag the crate declares in
//! `[package.metadata.gust.routes]` of its own `Cargo.toml`. The module's target is resolved once per expansion,
//! nearest first:
//!
//! 1. `GUST_PROFILE=cpu` is the CPU (a variable that says `gpu` then is GUST-E0749; a source default of `gpu` yields);
//! 2. the variable `GUST_ROUTE__<package>__<tag>`, which `cargo gust` sets (m8.r2) and anyone may set;
//! 3. the tag's own `default` in the manifest, then each ancestor's (`physics.broadphase`, then `physics`);
//! 4. the GPU.
//!
//! The result is checked against the tag's set per kernel ([`Resolved::check_kernel`]): the explicit `targets`, else
//! `gpu` and `cpu` for a kernel with a CPU form and `gpu` alone for one without. A module that carries no tag is never
//! looked at here, and its expansion is what it was before routing.
//!
//! The macro cannot depend on `gust` (the dependency runs the other way), so [`variable_name`] and [`is_valid_tag`]
//! are token-for-token copies of `gust::routes::variable_name` and `gust::routes::is_valid_tag` (the `content_key`
//! precedent in `artifact.rs`; `tests/routes.rs` compares the lexed token streams, so layout and comments may differ); `tests/routes.rs` asserts the golden `(package, tag, variable)` table of
//! `crates/gust/src/routes.rs`, pasted verbatim, so the two spellings cannot drift unnoticed.

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream};

use super::{Environment, Profile};
use crate::{
    diagnostics::{self, Code},
    manifest::Declarations,
};

/// The prefix of every route variable.
pub(crate) const VARIABLE_PREFIX: &str = "GUST_ROUTE__";

/// The variable that chooses the target of `tag` in the Cargo package `package`: `GUST_ROUTE__<package>__<tag>`,
/// with `-` in the package name spelled `_` and every `.` of the tag spelled `__`.
///
/// `package` is the Cargo package name as `cargo metadata` reports it, with its hyphens (`gust-compute`), and not the
/// library name (`gust`).
///
/// The mapping is injective on the tag half only for valid tags (see [`is_valid_tag`]): the tags `a_.b` and `a._b`
/// would both map to `a___b`, and `a__b` collides with `a.b`, which is why a valid tag has no `__` and no segment
/// that begins or ends with `_`. It is not injective across packages: Cargo allows package names with `__` or `--`,
/// names that differ only in `-` against `_`, and (on Windows, where environment names ignore case) names that differ
/// only in case. So `variable_name("a__b", "c")`, `variable_name("a--b", "c")` and `variable_name("a", "b.c")` are one
/// variable, and so are `a-b` and `a_b`. Two declared (package, tag) pairs that produce one variable name are an error
/// of the driver that sets the variables, not something this function detects. The function does not check its
/// arguments; the macro and the driver check the tag first.
pub(crate) fn variable_name(package: &str, tag: &str) -> String {
    let mut name = String::with_capacity(VARIABLE_PREFIX.len() + package.len() + 2 + tag.len() + 2);
    name.push_str(VARIABLE_PREFIX);
    name.extend(package.chars().map(|c| if c == '-' { '_' } else { c }));
    name.push_str("__");
    for c in tag.chars() {
        if c == '.' {
            name.push_str("__");
        } else {
            name.push(c);
        }
    }
    name
}

/// Whether `tag` is a well-formed route tag: segments of `[a-z0-9_]+` joined by single `.`, with no `__` anywhere
/// (the variable's separator), no segment that begins or ends with `_`, and no leading, trailing or doubled `.`.
pub(crate) fn is_valid_tag(tag: &str) -> bool {
    !tag.is_empty()
        && tag.split('.').all(|segment| {
            !segment.is_empty()
                && !segment.starts_with('_')
                && !segment.ends_with('_')
                && !segment.contains("__")
                && segment
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        })
}

/// Why `tag` is not a well-formed tag, in the words of the message; `None` exactly when [`is_valid_tag`] holds (a
/// test walks both over the same tags). The macro's own addition to the copy above.
pub(crate) fn tag_fault(tag: &str) -> Option<&'static str> {
    if tag.is_empty() {
        return Some("it is empty");
    }
    for segment in tag.split('.') {
        if segment.is_empty() {
            return Some("it has an empty segment (a leading or a trailing `.`, or `..`)");
        }
        if segment.contains("__") {
            return Some("a segment holds `__`, which spells the `.` of the route variable");
        }
        if segment.starts_with('_') || segment.ends_with('_') {
            return Some(
                "a segment begins or ends with `_` (`a_.b` and `a._b` would name one route variable)",
            );
        }
        if !segment
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            return Some("a segment holds a character outside `[a-z0-9_]`");
        }
    }
    None
}

/// A target a route chooses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// A GPU device runs the kernel.
    Gpu,
    /// The sequential `CpuDevice` runs the kernel (D51).
    Cpu,
}

impl Target {
    /// The name in a route, a variable and a manifest.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }

    /// The target named `name`, or `None` for any other text.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "gpu" => Some(Self::Gpu),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// The tag `#[gpu_module(route = "..")]` named, and where.
#[derive(Clone, Debug)]
pub(crate) struct Tag {
    /// The text of the literal.
    pub text: String,
    /// The span of the literal, where the errors about the tag are reported.
    pub span: Span,
}

/// What `#[gpu_module(..)]` or `#[gpu_fn(..)]` says: the route tag and the facade path, each of them optional.
pub(crate) struct Attribute {
    /// `route = "<tag>"`.
    pub tag: Option<Tag>,
    /// `crate = <path>` (audit S6, D48): where `gust` is, for a crate that has no `::gust` of its own.
    pub krate: Option<syn::Path>,
}

/// The arguments `#[gpu_module]` takes: `route = "<tag>"` and `crate = <path>`, in either order, each at most once,
/// with an optional trailing comma.
struct Arguments(Attribute);

const ARGUMENTS_TEXT: &str = "`#[gpu_module]` and `#[gpu_fn]` take `route = \"<tag>\"` and `crate = <path>` (for example `crate = ::inner::gust`), \
                                  each at most once, separated by commas";

impl Parse for Arguments {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut attribute = Attribute {
            tag: None,
            krate: None,
        };
        while !input.is_empty() {
            if input.peek(syn::Token![crate]) {
                let keyword: syn::Token![crate] = input.parse()?;
                if attribute.krate.is_some() {
                    return Err(diagnostics::error(
                        Code::GpuArguments,
                        keyword,
                        "`#[gpu_module]` and `#[gpu_fn]` take `crate = <path>` at most once",
                    ));
                }
                input.parse::<syn::Token![=]>()?;
                attribute.krate = Some(crate::forward::parse_facade(input, Code::GpuArguments)?);
            } else {
                let name: syn::Ident = input.parse()?;
                if name != "route" {
                    return Err(diagnostics::error_at(
                        Code::GpuArguments,
                        name.span(),
                        format!(
                            "{ARGUMENTS_TEXT}, in the current Rust-to-Slang subset; `{name}` is not one of them"
                        ),
                    ));
                }
                if attribute.tag.is_some() {
                    return Err(diagnostics::error(
                        Code::GpuArguments,
                        name,
                        "`#[gpu_module]` and `#[gpu_fn]` take `route = \"<tag>\"` at most once",
                    ));
                }
                input.parse::<syn::Token![=]>()?;
                let literal: syn::LitStr = input.parse()?;
                attribute.tag = Some(Tag {
                    text: literal.value(),
                    span: literal.span(),
                });
            }
            if input.is_empty() {
                break;
            }
            if !input.peek(syn::Token![,]) {
                return Err(diagnostics::error_at(
                    Code::GpuArguments,
                    input.span(),
                    format!("{ARGUMENTS_TEXT}; an argument is followed by a comma or nothing"),
                ));
            }
            input.parse::<syn::Token![,]>()?;
        }
        Ok(Self(attribute))
    }
}

/// The arguments of `#[gpu_module(..)]`; GUST-E0701 for anything but `route = "<tag>"` and `crate = <path>`.
pub(crate) fn parse_attribute(arguments: TokenStream) -> syn::Result<Attribute> {
    let Arguments(attribute) = diagnostics::parse2(Code::GpuArguments, arguments)?;
    Ok(attribute)
}

/// What reaches the expansion of a route: the tag the attribute named, if any, and where the route variables are
/// read from (the process for the macro, a table for a unit test, so that tests never set an environment variable).
pub(crate) struct Request<'a> {
    /// The tag; `None` for a module or function with no `route` (nothing here is looked at then).
    pub tag: Option<Tag>,
    /// The value of the variable called `name`, `None` when it is unset or empty (but for `GUST_ROUTED_PROFILE`, whose
    /// empty value is a value: the guard refuses it).
    pub variables: &'a dyn Fn(&str) -> Option<String>,
}

fn no_variables(_name: &str) -> Option<String> {
    None
}

impl Request<'static> {
    /// No tag.
    pub(crate) fn none() -> Self {
        Self {
            tag: None,
            variables: &no_variables,
        }
    }
}

/// The value of the process environment variable `name`, `None` when it is unset, empty or not text. The macro
/// reads each route variable once per expansion, and emits the `option_env!` that makes rustc track the read.
///
/// `GUST_ROUTED_PROFILE` is the exception (m9.0): unset means plain cargo and nothing is checked, but a variable set
/// to nothing, or to text that is not UTF-8 (read lossily), is a value the guard has to refuse, so it is kept.
pub(crate) fn process_variable(name: &str) -> Option<String> {
    variable_value(name, std::env::var_os(name))
}

/// What [`process_variable`] makes of the value the process has for `name` (a function of the value, so that a unit
/// test reaches the empty and the non-UTF-8 cases without setting an environment variable).
pub(crate) fn variable_value(name: &str, value: Option<std::ffi::OsString>) -> Option<String> {
    let value = value?;
    if name == ROUTED_PROFILE_VARIABLE {
        return Some(value.to_string_lossy().into_owned());
    }
    value.into_string().ok().filter(|value| !value.is_empty())
}

/// The variable `cargo gust` sets to the target part of the profile it resolved the routes for (m8.z).
pub(crate) const ROUTED_PROFILE_VARIABLE: &str = "GUST_ROUTED_PROFILE";

/// The profiles a build can have, as `gust::profile::NAME` spells them.
const ROUTED_PROFILES: [&str; 4] = ["dev", "native", "web", "cpu"];

/// The text of the compile error of a routed module built for `built` while the driver resolved its routes for
/// `routed`. It holds no `{` or `}`: it is the literal of a `panic!`.
pub(crate) fn routed_message(routed: &str, built: &str) -> String {
    format!(
        "gust: `cargo gust` resolved the routes of this build for the profile `{routed}` ({ROUTED_PROFILE_VARIABLE}), but \
             this build of gust is `{built}`: gust infers its profile from GUST_PROFILE, cargo's PROFILE and the target \
             architecture, and the driver reads only its own command line and environment, so they differ when the target \
             comes from `build.target` in .cargo/config.toml or from CARGO_BUILD_TARGET, when a custom profile inherits \
             `release`, when GUST_PROFILE is set only in the `[env]` table of the config, or when this crate is built for the \
             host as a build dependency (a wasm32 release build builds it as `native`; set GUST_PROFILE=web in the \
             environment then, so that both copies infer `web`). Pass `--release` and `--target` to `cargo gust` itself, or \
             set GUST_PROFILE to the profile you mean in the environment, so that both agree. ({ROUTED_PROFILE_VARIABLE} is \
             set by `cargo gust` only; unset it if you are running plain cargo.)"
    )
}

/// The text of the compile error of a routed module when `GUST_ROUTED_PROFILE` holds `value`, which is none of the
/// profiles (`ROUTED_PROFILES`; an empty value included), so there is nothing to compare the build with (m9.0). It is
/// the literal of a `compile_error!`, not a format, and the value is written with its escapes (`{value:?}`).
pub(crate) fn unknown_routed_profile_message(value: &str) -> String {
    format!(
        "gust: {ROUTED_PROFILE_VARIABLE} is {value:?}, which is not a build profile; the valid profiles are {}. \
             `cargo gust` sets this variable to the profile it resolved the routes for, so it is never empty; unset it if you \
             are running plain cargo, or set it to the profile you mean.",
        ROUTED_PROFILES.join(", ")
    )
}

/// What decided a module's target, for the words of an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    /// `GUST_PROFILE=cpu`.
    Profile,
    /// The variable of this name.
    Variable(String),
    /// The `default` of this tag in `Cargo.toml`: the module's own tag or an ancestor of it.
    Default(String),
    /// Nothing said anything, so the GPU.
    Nothing,
}

/// A module's route, resolved.
#[derive(Clone, Debug)]
pub(crate) struct Resolved {
    tag: Tag,
    /// The variable of the tag, `None` when the crate's package name is unknown (a unit test's scratch crate).
    variable: Option<String>,
    /// The target the module is built for.
    pub(crate) target: Target,
    /// What decided it.
    pub(crate) level: Level,
    /// The tag's `targets` as written; `None` when the key is absent.
    targets: Option<Vec<String>>,
}

fn error(code: Code, span: Span, message: String) -> syn::Error {
    diagnostics::error_at(code, span, message)
}

/// The source default of `tag`: its own `default`, else the nearest ancestor's (`a.b.c`, then `a.b`, then `a`),
/// with the tag that carries it.
fn source_default(declarations: &Declarations, tag: &str) -> Option<(Target, String)> {
    let mut name = tag;
    loop {
        let said = declarations
            .routes
            .iter()
            .find(|route| route.tag == name)
            .and_then(|route| route.default.as_deref())
            .and_then(Target::from_name);
        if let Some(target) = said {
            return Some((target, name.to_owned()));
        }
        name = &name[..name.rfind('.')?];
    }
}

/// Resolves the route of a module whose crate declared `declarations`, built under `profile`. `Ok(None)` when the
/// request carries no tag. The errors are spanned at the tag's literal.
pub(crate) fn resolve(
    request: &Request<'_>,
    environment: &Environment,
    declarations: &Declarations,
    profile: Profile,
) -> syn::Result<Option<Resolved>> {
    let Some(tag) = &request.tag else {
        return Ok(None);
    };
    let span = tag.span;
    // `tag_fault` words what `is_valid_tag` refuses; the fallback is for a rule one has and the other has not yet
    // learned, which a test over both rules forbids.
    let fault = |tag: &str| tag_fault(tag).unwrap_or("it breaks the rules of a tag");
    if !is_valid_tag(&tag.text) {
        return Err(error(
            Code::RouteTag,
            span,
            format!(
                "the route tag `{}` is ill-named: {}; a tag is segments of `[a-z0-9_]+` joined by single `.`, with no `__`",
                tag.text,
                fault(&tag.text)
            ),
        ));
    }
    // The declared keys are held to the same rules: a tag the driver could not spell as a variable is as wrong in
    // the manifest as in the attribute.
    if let Some(declared) = declarations
        .routes
        .iter()
        .find(|route| !is_valid_tag(&route.tag))
    {
        return Err(error(
            Code::RouteTag,
            span,
            format!(
                "the route tag `{}` declared in `[package.metadata.gust.routes]` of Cargo.toml is ill-named: {}",
                declared.tag,
                fault(&declared.tag)
            ),
        ));
    }
    let Some(route) = declarations
        .routes
        .iter()
        .find(|route| route.tag == tag.text)
    else {
        let declared: Vec<String> = declarations
            .routes
            .iter()
            .map(|route| format!("`{}`", route.tag))
            .collect();
        let declared = if declared.is_empty() {
            "it declares none".to_owned()
        } else {
            format!("it declares {}", declared.join(", "))
        };
        return Err(error(
            Code::RouteTag,
            span,
            format!(
                "the route tag `{}` is not declared in `[package.metadata.gust.routes]` of this crate's Cargo.toml ({declared}); a tag is used only by the crate that declares it, so declare it there, for example `\"{}\" = {{}}`",
                tag.text, tag.text
            ),
        ));
    };
    let variable = environment
        .package
        .as_deref()
        .map(|package| variable_name(package, &tag.text));
    let from_variable = match variable
        .as_deref()
        .and_then(|name| (request.variables)(name).map(|value| (name, value)))
    {
        Some((name, value)) => match Target::from_name(&value) {
            Some(target) => Some((target, name.to_owned())),
            None => {
                return Err(error(
                    Code::RouteTarget,
                    span,
                    format!(
                        "`{name}` is `{value}`, which is not a route target; the targets are gpu and cpu"
                    ),
                ));
            }
        },
        None => None,
    };
    let (target, level) = if profile.is_cpu() {
        // A route that names the GPU cannot be honoured by the build that holds no GPU form (D65). A source
        // default of `gpu` is not a choice made for this build: it yields, or no crate with the usual
        // `default = "gpu"` could be built for the CPU.
        if let Some((Target::Gpu, name)) = &from_variable {
            return Err(error(
                Code::RouteTarget,
                span,
                format!(
                    "the route tag `{}` is routed to `gpu` by `{name}`, but GUST_PROFILE=cpu builds the CPU target only; unset the variable or route the tag to `cpu`",
                    tag.text
                ),
            ));
        }
        (Target::Cpu, Level::Profile)
    } else if let Some((target, name)) = from_variable {
        (target, Level::Variable(name))
    } else if let Some((target, owner)) = source_default(declarations, &tag.text) {
        (target, Level::Default(owner))
    } else {
        (Target::Gpu, Level::Nothing)
    };
    Ok(Some(Resolved {
        tag: tag.clone(),
        variable,
        target,
        level,
        targets: route.targets.clone(),
    }))
}

impl Resolved {
    /// The profile the module is expanded under: the CPU target's for a module routed to the CPU, else `profile`.
    pub(crate) fn effective(&self, profile: Profile) -> Profile {
        if self.target == Target::Cpu {
            Profile::Cpu
        } else {
            profile
        }
    }

    /// The `option_env!` of the route variable, which makes rustc track the read; nothing when the variable has no
    /// name.
    pub(crate) fn tracking(&self) -> TokenStream {
        let Some(name) = &self.variable else {
            return TokenStream::new();
        };
        quote! {
            const _: ::core::option::Option<&::core::primitive::str> = ::core::option_env!(#name);
        }
    }

    /// The guard of a routed module against a build the driver did not resolve its routes for (m8.z).
    ///
    /// `cargo gust` resolves the routes for the profile it sees (its command line and `GUST_PROFILE`) and gives that
    /// profile to cargo as `GUST_ROUTED_PROFILE`. Cargo can build another one (`build.target` or `CARGO_BUILD_TARGET`
    /// in its config, a custom profile that inherits `release`), and then the routes are for a build this is not.
    ///
    /// The expansion reads the variable with `option_env!`, so rustc tracks it for the routed crate alone, as it does
    /// the route variables: a switch between plain cargo and the driver then recompiles the routed crates and the
    /// application, and not `gust` (D67's measured scope). When the variable is set the expansion also checks it
    /// against the profile `gust` was built for (`gust::profile::is_profile`, a hidden item). The check is a const
    /// evaluation spanned at the route's literal, so a mismatch is a compile error there. Unset, plain cargo, nothing
    /// is checked (D67). Set to anything but a profile (empty included, which is not unset), it is a compile error of
    /// its own that names the variable, the value and the profiles (m9.0).
    ///
    /// The macro cannot see the profile `gust` was built for, only `gust`'s constant can, so there is one `if` per
    /// profile that is not the routed one, each with a message that names both.
    pub(crate) fn guard(&self, routed: Option<&str>) -> TokenStream {
        let span = self.tag.span;
        let tracking = quote_spanned! { span =>
            const _: ::core::option::Option<&::core::primitive::str> =
                ::core::option_env!("GUST_ROUTED_PROFILE");
        };
        let Some(routed) = routed else {
            return tracking;
        };
        // A value that is no profile (a typo, an empty variable, text with a `{`) cannot be compared with the build:
        // it is its own compile error, and no `panic!` carries it (m9.0).
        if !ROUTED_PROFILES.contains(&routed) {
            let message = unknown_routed_profile_message(routed);
            return quote_spanned! { span =>
                #tracking
                ::core::compile_error!(#message);
            };
        }
        let checks = ROUTED_PROFILES
            .iter()
            .filter(|built| **built != routed)
            .map(|built| {
                let message = routed_message(routed, built);
                quote_spanned! { span =>
                    if ::gust::profile::is_profile(#built) {
                        ::core::panic!(#message);
                    }
                }
            });
        quote_spanned! { span =>
            #tracking
            const _: () = {
                #(#checks)*
            };
        }
    }

    /// How the target was decided, for a message.
    fn how(&self) -> String {
        match &self.level {
            Level::Profile => "by GUST_PROFILE=cpu".to_owned(),
            Level::Variable(name) => format!("by the variable `{name}`"),
            Level::Default(owner) if *owner == self.tag.text => {
                format!("by the `default` of `{owner}` in Cargo.toml")
            }
            Level::Default(owner) => {
                format!("by the `default` of its ancestor `{owner}` in Cargo.toml")
            }
            Level::Nothing => {
                "because no variable and no default routes it, which leaves the GPU".to_owned()
            }
        }
    }

    /// The tag's set against one kernel of the module: GUST-E0744 when the explicit `targets` names `cpu` and the
    /// kernel has no CPU form (in every build, the CPU one included), then GUST-E0749 when the target the module is
    /// built for is outside the set. The set is the explicit `targets`, else `gpu` and `cpu` for a kernel with a CPU
    /// form and `gpu` alone for one without. Under `GUST_PROFILE=cpu` the set is not consulted: the profile decides,
    /// and a kernel with no CPU form is that build's concern as it always was (it has no impl).
    pub(crate) fn check_kernel(&self, kernel: &syn::Ident, has_cpu_form: bool) -> syn::Result<()> {
        let tag = &self.tag.text;
        let names_cpu = self
            .targets
            .as_ref()
            .is_some_and(|set| set.iter().any(|name| name == "cpu"));
        if names_cpu && !has_cpu_form {
            return Err(diagnostics::error(
                Code::ProfileNotCovered,
                kernel,
                format!(
                    "kernel `{kernel}` has no CPU form, but the route tag `{tag}` lists `cpu` in `targets` in `[package.metadata.gust.routes]`: give the kernel a CPU form (remove `cpu = \"unsupported\"`; a kernel with workgroup memory or a barrier has none, GUST-E0717), or drop `cpu` from the set"
                ),
            ));
        }
        if self.level == Level::Profile {
            return Ok(());
        }
        let allowed = match &self.targets {
            Some(set) => set.iter().any(|name| name == self.target.name()),
            None => self.target == Target::Gpu || has_cpu_form,
        };
        if allowed {
            return Ok(());
        }
        let set = match &self.targets {
                Some(set) => format!(
                    "`targets = [{}]`",
                    set.iter()
                        .map(|name| format!("\"{name}\""))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                None => "just `gpu` (the kernel has no CPU form, `cpu = \"unsupported\"`, and the tag lists no `targets`)"
                    .to_owned(),
            };
        Err(diagnostics::error(
            Code::RouteTarget,
            kernel,
            format!(
                "kernel `{kernel}` is routed to `{}` {}, but the route tag `{tag}` allows {set}: route the tag to a target in the set, or change the set",
                self.target.name(),
                self.how()
            ),
        ))
    }
}

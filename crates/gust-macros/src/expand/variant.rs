//! Variants of a kernel (M8, D139): `#[variant(of = kernel, requires(..))]` functions beside the `#[kernel]` they
//! replace on a device that has the capabilities.
//!
//! The `#[kernel]` function is the kernel and its portable body (or, with `#[kernel(requires(..))]`, a body that needs
//! capabilities itself). [`lower`] turns every variant into a `#[kernel]` function, before validation, so that the
//! validator and the Slang emitter see plain kernels and every body is validated and lowered in every build. The
//! lowered attribute has the kernel's `workgroup_size` and `shape`, the body's own `requires(..)` and a `cpu`: the
//! kernel's own `cpu` for the portable body (the body whose set is empty, which is the one the twin is built from, so
//! the validator judges it by that option), `cpu = "unsupported"` for every body that requires a capability (it never
//! has a twin). The kernel's own body is lowered the same way. [`VariantPlan`] remembers which body
//! belongs to which kernel. The expansion builds every body (and writes the record of each in export mode), selects
//! one per kernel for the profile ([`VariantPlan::select`]) and gives the kernel's descriptor that body's Slang and
//! artifact; the handle and the typed calls are the kernel's own, and the CPU twin is the portable body's (the body
//! whose set is empty, [`KernelBodies::portable`]).

use std::collections::HashMap;

use proc_macro2::Span;
use quote::ToTokens;

use super::{Capability, CapabilitySet, CpuTwin, Profile};
use crate::diagnostics::{self, Code};

/// The profiles a crate declares when its manifest has no `profiles` key (M8, D138).
const DEFAULT_PROFILES: [&str; 3] = ["native", "web", "cpu"];

/// The kernels of a module that have a variant or a body with `requires(..)`; every other kernel is one portable
/// body and is not in the plan.
#[derive(Clone, Debug, Default)]
pub(crate) struct VariantPlan {
    kernels: Vec<KernelBodies>,
}

/// The bodies of one kernel.
#[derive(Clone, Debug)]
pub(crate) struct KernelBodies {
    /// The kernel's name.
    pub kernel: String,
    /// The span of the kernel's name, for the errors of the plan.
    pub span: Span,
    /// What the kernel's own body requires (`#[kernel(requires(..))]`; empty for a portable body).
    pub own: CapabilitySet,
    /// The kernel's `cpu` option as written. It belongs to the portable body only (the empty-set body, whose twin
    /// the kernel gets); every other body is lowered with `cpu = "unsupported"` and so is never the twin's.
    pub cpu: CpuTwin,
    /// The variants, in source order.
    pub variants: Vec<VariantBody>,
}

/// One `#[variant]` function.
#[derive(Clone, Debug)]
pub(crate) struct VariantBody {
    /// The function's name.
    pub name: String,
    /// The capabilities it requires.
    pub requires: CapabilitySet,
}

/// What the plan says of one function of the module.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Body<'a> {
    /// The variant function's name; `""` for the kernel's own body.
    pub variant: &'a str,
    /// The capabilities the body requires.
    pub requires: CapabilitySet,
}

/// The outcome of choosing among the bodies of a kernel for a set of capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Choice {
    /// The body at this index (0 is the kernel's own, then the variants in source order).
    Body(usize),
    /// No body requires only capabilities the set has.
    NoneFits,
    /// Two bodies fit and neither requires more than the other.
    Tie(usize, usize),
}

impl KernelBodies {
    /// The requirement of each body: the kernel's own first, then the variants in source order.
    pub(crate) fn sets(&self) -> Vec<CapabilitySet> {
        let mut sets = vec![self.own];
        sets.extend(self.variants.iter().map(|variant| variant.requires));
        sets
    }

    /// The name of the function of body `index` (see [`Choice::Body`]).
    pub(crate) fn function(&self, index: usize) -> &str {
        match index.checked_sub(1).and_then(|at| self.variants.get(at)) {
            Some(variant) => &variant.name,
            None => &self.kernel,
        }
    }

    /// The portable body: the one whose set is empty, the kernel's own or its empty-set variant (at most one,
    /// GUST-E0743). The CPU twin and the `cpu` form are this body's. `None` when every body requires a capability.
    pub(crate) fn portable(&self) -> Option<usize> {
        self.sets().iter().position(|set| set.is_empty())
    }

    /// The body to build on a device with `available`: the one with the most required capabilities among those
    /// whose requirements are all available (the largest subset). Two bodies at the same size are a tie.
    pub(crate) fn choose(&self, available: CapabilitySet) -> Choice {
        let sets = self.sets();
        let fits: Vec<usize> = (0..sets.len())
            .filter(|&index| sets[index].is_subset_of(available))
            .collect();
        let most = fits.iter().map(|&index| sets[index].len()).max();
        let best: Vec<usize> = fits
            .into_iter()
            .filter(|&index| Some(sets[index].len()) == most)
            .collect();
        match best[..] {
            [] => Choice::NoneFits,
            [index] => Choice::Body(index),
            [first, second, ..] => Choice::Tie(first, second),
        }
    }

    /// The first set of available capabilities, out of every subset of `capability_count` capabilities, for which
    /// two bodies tie, with the two bodies. A tie is an error in every build, whatever the profile.
    pub(crate) fn first_tie(&self, capability_count: u32) -> Option<(usize, usize, CapabilitySet)> {
        (0..1_u32 << capability_count).find_map(|mask| {
            let available = CapabilitySet::from_bits(u8::try_from(mask).ok()?);
            match self.choose(available) {
                Choice::Tie(first, second) => Some((first, second, available)),
                _ => None,
            }
        })
    }

    /// The bodies as a sentence part: "`a` requires {x}, `b` requires {}".
    fn describe(&self) -> String {
        let mut parts = vec![format!("`{}` requires {}", self.kernel, set_text(self.own))];
        parts.extend(
            self.variants.iter().map(|variant| {
                format!("`{}` requires {}", variant.name, set_text(variant.requires))
            }),
        );
        parts.join(", ")
    }
}

impl VariantPlan {
    /// The kernel `name` of the plan.
    pub(crate) fn kernel(&self, name: &str) -> Option<&KernelBodies> {
        self.kernels.iter().find(|kernel| kernel.kernel == name)
    }

    /// The kernel that function `function` is a variant of; `None` for any other function.
    pub(crate) fn owner_of(&self, function: &str) -> Option<&KernelBodies> {
        self.kernels.iter().find(|kernel| {
            kernel
                .variants
                .iter()
                .any(|variant| variant.name == function)
        })
    }

    /// The body that function `function` is, when the plan has it (a variant, or a kernel with a plan).
    pub(crate) fn body_of(&self, function: &str) -> Option<Body<'_>> {
        if let Some(kernel) = self.owner_of(function) {
            let variant = kernel
                .variants
                .iter()
                .find(|variant| variant.name == function)?;
            return Some(Body {
                variant: &variant.name,
                requires: variant.requires,
            });
        }
        let kernel = self.kernel(function)?;
        Some(Body {
            variant: "",
            requires: kernel.own,
        })
    }

    /// The function the profile builds for each kernel of the plan (kernel name to function name). Under the CPU
    /// target no GPU body is built and the portable body is named (the kernel's own when it has none). A kernel
    /// with no body the profile can use is GUST-E0750, except in export mode (`export_mode`): `cargo gust build`
    /// builds under a fixed profile and writes every body, so there the kernel's own body stands for the descriptor
    /// of a build that is never shipped.
    pub(crate) fn select(
        &self,
        profile: Profile,
        export_mode: bool,
    ) -> syn::Result<HashMap<String, String>> {
        let mut selected = HashMap::new();
        for kernel in &self.kernels {
            let index = if profile.is_cpu() {
                kernel.portable().unwrap_or(0)
            } else {
                match kernel.choose(profile.capabilities) {
                    Choice::Body(index) => index,
                    Choice::NoneFits if export_mode => 0,
                    Choice::NoneFits => {
                        return Err(diagnostics::error_at(
                            Code::NoBodySelectable,
                            kernel.span,
                            format!(
                                "kernel `{}` has no body that the build profile {} can use: {}; build with a \
                                     profile that provides them, or add a variant that requires less",
                                kernel.kernel,
                                profile_text(profile),
                                kernel.describe()
                            ),
                        ));
                    }
                    Choice::Tie(first, second) => {
                        return Err(tie_error(
                            kernel,
                            first,
                            second,
                            profile.capabilities,
                            kernel.span,
                        ));
                    }
                }
            };
            selected.insert(kernel.kernel.clone(), kernel.function(index).to_owned());
        }
        Ok(selected)
    }
}

/// The text of a set in a message: `{subgroups}` or `{}`.
fn set_text(set: CapabilitySet) -> String {
    format!("{{{}}}", set.names().join(", "))
}

/// The text of a profile in a message: `native+subgroups`, `dev`, or what an unset variable is.
fn profile_text(profile: Profile) -> String {
    let Some(target) = profile.target else {
        return "(`GUST_PROFILE` unset or unknown; no capability)".to_owned();
    };
    let mut text = format!("`{}", target.text());
    for name in profile.capabilities.names() {
        text.push('+');
        text.push_str(name);
    }
    text.push('`');
    text
}

fn tie_error(
    kernel: &KernelBodies,
    first: usize,
    second: usize,
    available: CapabilitySet,
    span: Span,
) -> syn::Error {
    let sets = kernel.sets();
    diagnostics::error_at(
        Code::VariantAmbiguous,
        span,
        format!(
            "kernel `{}` has two bodies that a device with {} cannot tell apart: `{}` requires {} and `{}` requires \
                 {}; give them sets of different sizes, or one a superset of the other",
            kernel.kernel,
            set_text(available),
            kernel.function(first),
            set_text(sets[first]),
            kernel.function(second),
            set_text(sets[second]),
        ),
    )
}

/// The profiles a crate declares, for the exhaustiveness rule of D23 (the owner's `match` rule): every declared
/// platform has an arm.
#[derive(Clone, Debug)]
pub(crate) struct Declared {
    /// Each entry as written, with the profile it parses to.
    entries: Vec<(String, Profile)>,
    /// Whether the manifest wrote the list (`profiles = [..]`); an absent key is the default list, which is not the
    /// same as a present list that equals it (D138).
    explicit: bool,
}

impl Declared {
    /// The declared profiles: `profiles` as the manifest wrote it, or the default list when it did not.
    pub(crate) fn new(profiles: Option<&[String]>) -> Self {
        let explicit = profiles.is_some();
        let entries = match profiles {
            Some(list) => list.iter().map(String::as_str).collect::<Vec<_>>(),
            None => DEFAULT_PROFILES.to_vec(),
        }
        .into_iter()
        // The manifest reader refused every entry the grammar refuses (GUST-E0753), so none is dropped here.
        .filter_map(|entry| Profile::parse(entry).map(|profile| (entry.to_owned(), profile)))
        .collect();
        Self { entries, explicit }
    }

    /// Whether an explicit list names the CPU target. The default list does: a kernel with `cpu = "unsupported"`
    /// meets that entry with the arm it wrote (Q4); only a list the crate wrote can be contradicted.
    pub(crate) fn names_cpu_explicitly(&self) -> bool {
        self.explicit && self.entries.iter().any(|(_, profile)| profile.is_cpu())
    }

    /// Every kernel of the plan has a body for every declared GPU profile (GUST-E0744). A dev build is checked as
    /// `native` with the same capabilities, so the list has no `dev` entry to check.
    pub(crate) fn check(&self, plan: &VariantPlan) -> syn::Result<()> {
        for (entry, profile) in &self.entries {
            if profile.is_cpu() {
                continue;
            }
            // An absent `profiles` key declares the default list; say so, to a crate that declared nothing.
            let origin = if self.explicit {
                ""
            } else {
                " (the default list; the manifest has no `profiles` key)"
            };
            for kernel in &plan.kernels {
                if kernel.choose(profile.capabilities) == Choice::NoneFits {
                    return Err(diagnostics::error_at(
                        Code::ProfileNotCovered,
                        kernel.span,
                        format!(
                            "kernel `{}` has no body for the declared profile `{entry}`{origin}: {}; declare a \
                                 profile with the capabilities in `[package.metadata.gust] profiles`, or add a \
                                 variant that requires less",
                            kernel.kernel,
                            kernel.describe()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The kernel `name` has a CPU form when the explicit list names `cpu` (GUST-E0744 otherwise). `bodies` are the
    /// kernel's bodies when it has a plan. The CPU form is the portable body's (D140), so a kernel every body of
    /// which requires a capability has none, whatever its `cpu` option says.
    pub(crate) fn check_cpu_form(
        &self,
        name: &syn::Ident,
        cpu: CpuTwin,
        bodies: Option<&KernelBodies>,
    ) -> syn::Result<()> {
        if !self.names_cpu_explicitly() {
            return Ok(());
        }
        if let Some(bodies) = bodies.filter(|bodies| bodies.portable().is_none()) {
            return Err(diagnostics::error(
                Code::ProfileNotCovered,
                name,
                format!(
                    "kernel `{name}` has no CPU form, but `[package.metadata.gust] profiles` declares `cpu`: \
                         {}, and the CPU form is the body that requires nothing; add a variant with `requires()`, or \
                         drop `cpu` from `profiles`",
                    bodies.describe()
                ),
            ));
        }
        if cpu == CpuTwin::Unsupported {
            return Err(diagnostics::error(
                Code::ProfileNotCovered,
                name,
                format!(
                    "kernel `{name}` has no CPU form (`cpu = \"unsupported\"`) but `[package.metadata.gust] \
                         profiles` declares `cpu`: remove `cpu = \"unsupported\"` (a kernel with workgroup memory or a \
                         barrier has no CPU form, GUST-E0717), or drop `cpu` from `profiles`"
                ),
            ));
        }
        Ok(())
    }
}

fn is_variant(attribute: &syn::Attribute) -> bool {
    attribute.path().is_ident("variant")
}

fn is_kernel(attribute: &syn::Attribute) -> bool {
    attribute.path().is_ident("kernel")
}

/// The capabilities a name stands for: a capability, or an alias for a set (`vulkan` is `{subgroups}`, `webgpu` is
/// `{}`), as the manifest and the attribute spell them.
fn resolve(name: &syn::Ident) -> syn::Result<CapabilitySet> {
    let text = name.to_string();
    if let Some(capability) = Capability::from_name(&text) {
        return Ok(CapabilitySet::EMPTY.with(capability));
    }
    match text.as_str() {
        "vulkan" => Ok(CapabilitySet::EMPTY.with(Capability::Subgroups)),
        "webgpu" => Ok(CapabilitySet::EMPTY),
        _ => Err(diagnostics::error(
            Code::UnknownCapability,
            name,
            format!(
                "unknown capability `{text}` in `requires(..)`; the capabilities are {}, the aliases `vulkan` \
                     (`subgroups`) and `webgpu` (none)",
                Capability::ALL
                    .iter()
                    .map(|capability| format!("`{}`", capability.name()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )),
    }
}

/// `requires(a, b)`: the union of the sets the names stand for.
fn requires_from(input: syn::parse::ParseStream<'_>) -> syn::Result<CapabilitySet> {
    let content;
    syn::parenthesized!(content in input);
    let names =
        syn::punctuated::Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated(&content)?;
    let mut set = CapabilitySet::EMPTY;
    for name in &names {
        set = set.union(resolve(name)?);
    }
    Ok(set)
}

/// What `#[variant(..)]` says: the kernel it replaces and the capabilities it requires.
fn parse_variant(attribute: &syn::Attribute) -> syn::Result<(syn::Ident, CapabilitySet)> {
    let mut of: Option<syn::Ident> = None;
    let mut requires: Option<CapabilitySet> = None;
    diagnostics::parse_nested_meta(attribute, Code::VariantAttribute, |meta| {
        if meta.path.is_ident("of") {
            if of.is_some() {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::VariantAttribute,
                    "duplicate `of` option",
                ));
            }
            of = Some(meta.value()?.parse()?);
            Ok(())
        } else if meta.path.is_ident("requires") {
            if requires.is_some() {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::VariantAttribute,
                    "duplicate `requires` option",
                ));
            }
            requires = Some(requires_from(meta.input)?);
            Ok(())
        } else {
            Err(diagnostics::meta_error(
                &meta,
                Code::VariantAttribute,
                "#[variant] takes `of = <kernel>` and an optional `requires(..)`; the workgroup size, the shape and \
                     the CPU form are the kernel's",
            ))
        }
    })?;
    let Some(of) = of else {
        return Err(diagnostics::error(
            Code::VariantAttribute,
            attribute,
            "#[variant] needs the kernel it replaces: `#[variant(of = kernel, requires(subgroups))]`",
        ));
    };
    Ok((of, requires.unwrap_or(CapabilitySet::EMPTY)))
}

/// What `lower` reads of a `#[kernel(..)]` attribute.
struct KernelParts {
    /// The `requires(..)`: what the kernel's own body needs.
    requires: CapabilitySet,
    /// The options a variant of the kernel shares (`workgroup_size(..)` and `shape = ..`), as written.
    layout: Vec<proc_macro2::TokenStream>,
    /// The `cpu = "unsupported"` the kernel wrote, as written (none when it wrote none or a faulty one).
    cpu_tokens: Option<proc_macro2::TokenStream>,
    /// What `cpu_tokens` says.
    cpu: CpuTwin,
    /// Whether the attribute is made of options this module re-spells (`workgroup_size`, `shape`, a valid `cpu`
    /// and `requires`). Any other option, or a walk that failed, is left for `kernel_attribute_facts` to report on
    /// the attribute as written, so the attribute is not rewritten.
    rewritable: bool,
}

/// The `requires(..)`, the `cpu` and the layout options of a `#[kernel(..)]` attribute. Every other option is the
/// business of `kernel_attribute_facts`, which reports its errors; only an unknown capability and a second
/// `requires(..)` (GUST-E0740, as on `#[variant]`) are reported here.
fn kernel_parts(attribute: &syn::Attribute) -> syn::Result<KernelParts> {
    let mut parts = KernelParts {
        requires: CapabilitySet::EMPTY,
        layout: Vec::new(),
        cpu_tokens: None,
        cpu: CpuTwin::Emitted,
        rewritable: true,
    };
    if matches!(attribute.meta, syn::Meta::Path(_)) {
        return Ok(parts);
    }
    let mut seen_requires = false;
    let mut first_error: Option<syn::Error> = None;
    // A failure of the walk itself (an option that does not parse) is left for `kernel_attribute_facts`.
    let walk = diagnostics::parse_nested_meta(attribute, Code::MalformedKernelAttribute, |meta| {
        if meta.path.is_ident("requires") {
            let found = requires_from(meta.input);
            if seen_requires && first_error.is_none() {
                first_error = Some(diagnostics::meta_error(
                    &meta,
                    Code::VariantAttribute,
                    "duplicate `requires` option",
                ));
            }
            seen_requires = true;
            match found {
                Ok(found) => parts.requires = parts.requires.union(found),
                Err(error)
                    if first_error.is_none() && is_coded(&error, Code::UnknownCapability) =>
                {
                    first_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        } else if meta.path.is_ident("workgroup_size") {
            let path = meta.path.clone();
            let content;
            syn::parenthesized!(content in meta.input);
            let inner: proc_macro2::TokenStream = content.parse()?;
            parts.layout.push(quote::quote!(#path(#inner)));
        } else if meta.path.is_ident("shape") {
            let path = meta.path.clone();
            let value: syn::Ident = meta.value()?.parse()?;
            parts.layout.push(quote::quote!(#path = #value));
        } else if meta.path.is_ident("cpu") {
            let path = meta.path.clone();
            let value: syn::LitStr = meta.value()?.parse()?;
            if value.value() == "unsupported" && parts.cpu_tokens.is_none() {
                parts.cpu_tokens = Some(quote::quote!(#path = #value));
                parts.cpu = CpuTwin::Unsupported;
            } else {
                // A duplicate or an unknown value: `kernel_attribute_facts` reports it.
                parts.rewritable = false;
            }
        } else {
            parts.rewritable = false;
            if meta.input.peek(syn::token::Paren) {
                let content;
                syn::parenthesized!(content in meta.input);
                content.parse::<proc_macro2::TokenStream>()?;
            } else if meta.input.peek(syn::Token![=]) {
                meta.value()?.parse::<syn::Expr>()?;
            }
        }
        Ok(())
    });
    if walk.is_err() {
        parts.rewritable = false;
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(parts),
    }
}

/// The attribute a body is lowered to: the kernel's `workgroup_size` and `shape`, the body's own `requires(..)`
/// and its `cpu`. The `cpu` option belongs to the portable body, the one whose set is empty (it is the body the
/// kernel's twin is built from, so the validator judges it with the kernel's `cpu`, GUST-E0717): that body gets
/// the kernel's `cpu` as written, or none. Every body with a non-empty set gets `cpu = "unsupported"`: it never
/// has a twin (D139). The kernel's `requires` is not a variant's: the validator judges a body by its own
/// attribute, and m8.s reads the capabilities a body may use from it.
fn lowered_attribute(
    layout: &[proc_macro2::TokenStream],
    requires: CapabilitySet,
    kernel_cpu: Option<&proc_macro2::TokenStream>,
    span: Span,
) -> syn::Attribute {
    let mut options: Vec<proc_macro2::TokenStream> = layout.to_vec();
    if !requires.is_empty() {
        options.push(quote::quote_spanned!(span=> cpu = "unsupported"));
    } else if let Some(cpu) = kernel_cpu {
        options.push(cpu.clone());
    }
    if !requires.is_empty() {
        let names = requires
            .names()
            .into_iter()
            .map(|name| syn::Ident::new(name, span));
        options.push(quote::quote_spanned!(span=> requires(#(#names),*)));
    }
    syn::parse_quote_spanned!(span=> #[kernel(#(#options),*)])
}

fn is_coded(error: &syn::Error, code: Code) -> bool {
    error.to_string().starts_with(&format!("[{}]", code.text()))
}

/// Every attribute list of an item that is not a function, and of the items inside an `impl`, a `trait` or an
/// `extern` block (their functions included), for the check that `#[variant]` stands on a function of the module
/// only, and for the recovery that drops a stray one.
fn foreign_attribute_lists(item: &mut syn::Item) -> Vec<&mut Vec<syn::Attribute>> {
    match item {
        syn::Item::Const(item) => vec![&mut item.attrs],
        syn::Item::Enum(item) => vec![&mut item.attrs],
        syn::Item::ExternCrate(item) => vec![&mut item.attrs],
        syn::Item::ForeignMod(item) => {
            let mut lists = vec![&mut item.attrs];
            for inner in &mut item.items {
                match inner {
                    syn::ForeignItem::Fn(inner) => lists.push(&mut inner.attrs),
                    syn::ForeignItem::Static(inner) => lists.push(&mut inner.attrs),
                    syn::ForeignItem::Type(inner) => lists.push(&mut inner.attrs),
                    syn::ForeignItem::Macro(inner) => lists.push(&mut inner.attrs),
                    _ => {}
                }
            }
            lists
        }
        syn::Item::Impl(item) => {
            let mut lists = vec![&mut item.attrs];
            for inner in &mut item.items {
                match inner {
                    syn::ImplItem::Const(inner) => lists.push(&mut inner.attrs),
                    syn::ImplItem::Fn(inner) => lists.push(&mut inner.attrs),
                    syn::ImplItem::Type(inner) => lists.push(&mut inner.attrs),
                    syn::ImplItem::Macro(inner) => lists.push(&mut inner.attrs),
                    _ => {}
                }
            }
            lists
        }
        syn::Item::Macro(item) => vec![&mut item.attrs],
        syn::Item::Mod(item) => vec![&mut item.attrs],
        syn::Item::Static(item) => vec![&mut item.attrs],
        syn::Item::Struct(item) => vec![&mut item.attrs],
        syn::Item::Trait(item) => {
            let mut lists = vec![&mut item.attrs];
            for inner in &mut item.items {
                match inner {
                    syn::TraitItem::Const(inner) => lists.push(&mut inner.attrs),
                    syn::TraitItem::Fn(inner) => lists.push(&mut inner.attrs),
                    syn::TraitItem::Type(inner) => lists.push(&mut inner.attrs),
                    syn::TraitItem::Macro(inner) => lists.push(&mut inner.attrs),
                    _ => {}
                }
            }
            lists
        }
        syn::Item::TraitAlias(item) => vec![&mut item.attrs],
        syn::Item::Type(item) => vec![&mut item.attrs],
        syn::Item::Union(item) => vec![&mut item.attrs],
        syn::Item::Use(item) => vec![&mut item.attrs],
        // A function is the business of `lower`; `Verbatim` has no attributes we can read.
        _ => Vec::new(),
    }
}

/// The inputs and the output of a signature as text: what "token-identical" compares.
fn signature_text(signature: &syn::Signature) -> (String, String, String) {
    (
        signature.generics.to_token_stream().to_string(),
        signature.inputs.to_token_stream().to_string(),
        signature.output.to_token_stream().to_string(),
    )
}

/// The function at `index`, which the scan found to be one.
fn function_at(items: &[syn::Item], index: usize) -> &syn::ItemFn {
    match &items[index] {
        syn::Item::Fn(function) => function,
        _ => unreachable!("an index of a function"),
    }
}

/// The kernel named `name`.
fn kernel_named<'a>(
    items: &[syn::Item],
    kernels: &'a [KernelFound],
    name: &syn::Ident,
) -> Option<&'a KernelFound> {
    kernels
        .iter()
        .find(|kernel| function_at(items, kernel.index).sig.ident == *name)
}

struct VariantFound {
    index: usize,
    of: syn::Ident,
    requires: CapabilitySet,
    position: usize,
}

struct KernelFound {
    index: usize,
    own: CapabilitySet,
    /// The options a variant of the kernel shares.
    layout: Vec<proc_macro2::TokenStream>,
    /// The kernel's `cpu = "unsupported"` as written, and what it says.
    cpu_tokens: Option<proc_macro2::TokenStream>,
    cpu: CpuTwin,
    /// Where the `#[kernel]` attribute stands among the function's, and whether it may be rewritten (see
    /// `KernelParts::rewritable`).
    position: usize,
    rewritable: bool,
}

/// Lowers the variants of `module` to kernels and returns the plan: the check of the attributes (GUST-E0740 to
/// GUST-E0743) and the rewrite described at [`variant`](self). A module without `#[variant]` and without
/// `requires(..)` comes back unchanged with an empty plan.
pub(crate) fn lower(mut module: syn::ItemMod) -> syn::Result<(syn::ItemMod, VariantPlan)> {
    let Some((_, items)) = module.content.as_mut() else {
        return Ok((module, VariantPlan::default()));
    };
    for item in items.iter_mut() {
        for list in foreign_attribute_lists(item) {
            if let Some(attribute) = list.iter().find(|a| is_variant(a)) {
                return Err(diagnostics::error(
                    Code::VariantAttribute,
                    attribute,
                    "#[variant] goes on a function of a `#[gpu_module]` module, beside the `#[kernel]` it replaces",
                ));
            }
        }
    }
    let mut variants: Vec<VariantFound> = Vec::new();
    let mut kernels: Vec<KernelFound> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let found: Vec<(usize, &syn::Attribute)> = function
            .attrs
            .iter()
            .enumerate()
            .filter(|(_, attribute)| is_variant(attribute))
            .collect();
        let kernel_attribute = function
            .attrs
            .iter()
            .enumerate()
            .find(|(_, a)| is_kernel(a));
        match (found.as_slice(), kernel_attribute) {
            ([], None) => {}
            ([], Some((position, attribute))) => {
                let parts = kernel_parts(attribute)?;
                kernels.push(KernelFound {
                    index,
                    own: parts.requires,
                    layout: parts.layout,
                    cpu_tokens: parts.cpu_tokens,
                    cpu: parts.cpu,
                    position,
                    rewritable: parts.rewritable,
                });
            }
            ([(position, attribute)], None) => {
                let (of, requires) = parse_variant(attribute)?;
                variants.push(VariantFound {
                    index,
                    of,
                    requires,
                    position: *position,
                });
            }
            ([(_, attribute)], Some(_)) => {
                return Err(diagnostics::error(
                    Code::VariantAttribute,
                    attribute,
                    format!(
                        "`{}` is both a #[kernel] and a #[variant]: a function is the kernel or one of its variants",
                        function.sig.ident
                    ),
                ));
            }
            ([_, (_, second), ..], _) => {
                return Err(diagnostics::error(
                    Code::VariantAttribute,
                    second,
                    "a function has one #[variant] attribute",
                ));
            }
        }
    }
    // Each variant names a kernel of the module and has its signature.
    for variant in &variants {
        let body = function_at(items, variant.index);
        let Some(kernel) = kernel_named(items, &kernels, &variant.of) else {
            let names_a_variant = variants
                .iter()
                .any(|other| function_at(items, other.index).sig.ident == variant.of);
            let what = if names_a_variant {
                "a variant, not a kernel"
            } else {
                "no #[kernel] function of this module"
            };
            return Err(diagnostics::error(
                Code::VariantTarget,
                &variant.of,
                format!(
                    "#[variant(of = {})] on `{}` names {what}",
                    variant.of, body.sig.ident
                ),
            ));
        };
        let kernel = function_at(items, kernel.index);
        if signature_text(&kernel.sig) != signature_text(&body.sig) {
            return Err(diagnostics::error(
                Code::VariantAttribute,
                &body.sig,
                format!(
                    "variant `{}` does not have the signature of kernel `{}`: the parameters and the return type \
                         are written the same, token for token, so that one call serves every body",
                    body.sig.ident, kernel.sig.ident
                ),
            ));
        }
    }

    // The plan: the kernels that have a variant or a body of their own that requires something.
    let mut plan = VariantPlan::default();
    for kernel in &kernels {
        let function = function_at(items, kernel.index);
        let bodies = KernelBodies {
            kernel: function.sig.ident.to_string(),
            span: function.sig.ident.span(),
            own: kernel.own,
            cpu: kernel.cpu,
            variants: variants
                .iter()
                .filter(|variant| variant.of == function.sig.ident)
                .map(|variant| VariantBody {
                    name: function_at(items, variant.index).sig.ident.to_string(),
                    requires: variant.requires,
                })
                .collect(),
        };
        if !bodies.variants.is_empty() || !bodies.own.is_empty() {
            plan.kernels.push(bodies);
        }
    }
    // A tie is an error in every build: the profile of the build is not what decides it.
    let capabilities = u32::try_from(Capability::ALL.len()).unwrap_or(8).min(8);
    for bodies in &plan.kernels {
        if let Some((first, second, available)) = bodies.first_tie(capabilities) {
            // At the later of the two bodies, which is the one the author added last.
            let later = bodies.function(first.max(second));
            let span = variants
                .iter()
                .map(|variant| &function_at(items, variant.index).sig.ident)
                .find(|name| *name == later)
                .map_or(bodies.span, syn::Ident::span);
            return Err(tie_error(bodies, first, second, available, span));
        }
    }

    // The rewrite: a variant is a function with the kernel's `workgroup_size` and `shape`, its own `requires(..)`
    // and a `cpu`: the kernel's when its set is empty (the portable body, whose twin the kernel gets), otherwise
    // `cpu = "unsupported"`. The kernel's own body is rewritten the same way when its set is not empty.
    let mut rewrites: Vec<(usize, usize, syn::Attribute)> = variants
        .iter()
        .filter_map(|variant| {
            let kernel = kernel_named(items, &kernels, &variant.of)?;
            let written = &function_at(items, variant.index).attrs[variant.position];
            let span = written.pound_token.spans[0];
            Some((
                variant.index,
                variant.position,
                lowered_attribute(
                    &kernel.layout,
                    variant.requires,
                    kernel.cpu_tokens.as_ref(),
                    span,
                ),
            ))
        })
        .collect();
    for kernel in &kernels {
        if kernel.own.is_empty() || !kernel.rewritable {
            continue;
        }
        let written = &function_at(items, kernel.index).attrs[kernel.position];
        let span = written.pound_token.spans[0];
        rewrites.push((
            kernel.index,
            kernel.position,
            lowered_attribute(&kernel.layout, kernel.own, None, span),
        ));
    }
    for (index, position, attribute) in rewrites {
        if let syn::Item::Fn(function) = &mut items[index] {
            function.attrs[position] = attribute;
        }
    }
    Ok((module, plan))
}

/// The error for `#[variant]` on a `#[gpu_fn]` function (an element-wise map has one body), or `None` without one.
pub(crate) fn misplaced_on_function(function: &syn::ItemFn) -> Option<syn::Error> {
    let attribute = function.attrs.iter().find(|a| is_variant(a))?;
    Some(diagnostics::error(
        Code::VariantAttribute,
        attribute,
        "#[variant] goes on a function of a `#[gpu_module]` module; a `#[gpu_fn]` function is one body",
    ))
}

/// `function` without its `#[variant]` attributes: the form that goes back beside an error.
pub(crate) fn without_variant(mut function: syn::ItemFn) -> syn::ItemFn {
    function.attrs.retain(|attribute| !is_variant(attribute));
    function
}

/// A rejected module as the user wrote it, made safe to hand back beside the error: a `#[variant]` function
/// becomes a `#[kernel]` (error recovery then renames it as it does a kernel), and `#[variant]` anywhere else is
/// dropped, because an unknown attribute would be a second error.
pub(crate) fn recoverable(mut module: syn::ItemMod) -> syn::ItemMod {
    let Some((_, items)) = module.content.as_mut() else {
        return module;
    };
    for item in items {
        if let syn::Item::Fn(function) = item {
            for attribute in &mut function.attrs {
                if is_variant(attribute) {
                    *attribute = syn::parse_quote!(#[kernel]);
                }
            }
            continue;
        }
        for list in foreign_attribute_lists(item) {
            list.retain(|a| !is_variant(a));
        }
    }
    module
}

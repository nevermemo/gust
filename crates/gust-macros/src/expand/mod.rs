//! The expansion of `#[gpu_module]` modules, `#[gpu_fn]` functions and their kernels into Rust items.
//!
//! The entry points are [`expand_in`] and [`expand_in_routed`] (with the `expand` family beside them); the other files
//! hold the expansion by concern, and this one declares them and re-exports what the rest of the crate names.
//!
//! - `build_profile`: the build profile (target, capabilities, CPU twin)
//! - `module`: the expansion of one validated module
//! - `routing` and `variant`: build-time routing and kernel variants
//! - `kernel_attributes` and `parameters`: the `#[kernel]` attribute and the kernel's parameters
//! - `typed_call`, `cpu_twin` and `pod`: the typed host call, the CPU twin and the plain-data records
//! - `capture`: the typed `capture` of a kernel (copy a call's inputs, replay chosen invocations on the twin)
//! - `artifact_items` and `origin`: artifact checks and source origins
//! - `artifact`, `map` and `pad`: existing expansions of artifacts, element-wise maps and `#[pad]` structs
//! - `tests`: the unit tests, by concern

use std::{collections::HashMap, path::Path};

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::ext::IdentExt as _;

use crate::{
    diagnostics::{self, Code},
    profile::{GRID, GRID_SPLIT, PIXEL, ReflectionKind, Shape, StageProfile, TwinHarness},
    slang::{ParameterFlavor, parameter_flavor_in},
    validate::classify::{self, AttributeFacts, ShapePin},
};

mod artifact;
mod artifact_items;
mod build_profile;
mod capture;
mod cpu_twin;
mod kernel_attributes;
mod module;
mod origin;
mod parameters;
pub(crate) mod pod;
pub(crate) mod routing;
#[cfg(test)]
mod tests;
mod typed_call;
pub(crate) mod variant;

use artifact::Lookup;
#[cfg(test)]
pub(crate) use artifact::TWIN_TEXT;
pub(crate) use artifact::{Environment, content_key};
use artifact_items::*;
pub use build_profile::*;
use cpu_twin::*;
pub(crate) use kernel_attributes::*;
use module::*;
use origin::*;
#[cfg(test)]
pub(crate) use parameters::parameter_bindings;
use parameters::*;
use pod::*;
use typed_call::*;

pub(crate) mod pad;

mod map;
#[cfg(test)]
pub(crate) use map::expand_map_in_routed;
pub(crate) use map::{ElementReject, element_types, expand_map_routed};
#[cfg(test)]
pub(crate) use map::{expand_module_in_routed, map_kernel, map_module_kernels};

/// What names a kernel's export record to the tool that reads it: a `#[gpu_module]` module or `#[gpu_fn]` function is
/// one place in the source, so its record says where (schema 1, `source`); a `par_for!` call is a body that can stand
/// at any number of places, so its record says nothing of them (schema 2) and each place writes its own sidecar (R1.1
/// M5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Identity {
    /// A `#[gpu_module]` module or `#[gpu_fn]` function: schema 1, the record holds the kernel's `source`.
    Module,
    /// A `par_for!` call whose first line is `call_line`: schema 2 and a sidecar `<key>.<site16>.site.json`.
    Site { call_line: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct KernelOptions {
    pub workgroup_size: [u32; 3],
}

/// [`expand_with`] for a caller that has no profile: [`Profile::Other`].
#[cfg(test)]
pub fn expand(module: syn::ItemMod) -> syn::Result<TokenStream> {
    expand_with(module, Profile::Other)
}

/// Expands a module for the build profile `profile`, reading the crate directory and the export directory from the
/// process environment.
#[cfg(test)]
pub fn expand_with(module: syn::ItemMod, profile: Profile) -> syn::Result<TokenStream> {
    expand_in(module, &Environment::from_process(), profile)
}

/// [`expand_with`] for a module that may carry a route (`#[gpu_module(route = "tag")]`, M8, D65 to D67).
pub(crate) fn expand_with_routed(
    module: syn::ItemMod,
    profile: Profile,
    request: &routing::Request<'_>,
) -> syn::Result<TokenStream> {
    expand_in_routed(module, &Environment::from_process(), profile, request)
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

/// [`expand_with_directories`] for a `#[gpu_fn]` function (an element-wise map): the same environment, so that the
/// tests of the manifest can drive the map path, which reads the crate's manifest too (GUST-E0753 at the function's
/// name).
#[cfg(test)]
pub(crate) fn expand_map_with_directories(
    function: syn::ItemFn,
    manifest_dir: Option<&Path>,
) -> syn::Result<TokenStream> {
    let environment = Environment {
        manifest_dir: manifest_dir.map(Path::to_path_buf),
        export_dir: None,
        package: Some("demo".to_owned()),
        reported_file: None,
    };
    map::expand_map_in(function, &environment, Profile::Other)
}

/// [`expand_with`] with the environment given, so that tests choose the crate directory and the export directory
/// without touching the process environment.
///
/// Under [`Profile::Cpu`] the artifact directory is not read at all (GPU bytes do not belong in a CPU binary, and a
/// half pair there is no concern of a CPU build), so every kernel is `Lookup::Absent` and the manifest is absent;
/// the per-kernel `require_artifact` item is still emitted, and the build script's profile makes it a no-op.
#[cfg(test)]
pub(crate) fn expand_in(
    module: syn::ItemMod,
    environment: &Environment,
    profile: Profile,
) -> syn::Result<TokenStream> {
    expand_in_routed(module, environment, profile, &routing::Request::none())
}

/// [`expand_in`] with the route the attribute named. The portable floor (m8.f) is checked here, before the route is
/// resolved and whatever it says: a kernel routed to the CPU is held to the GPU floor too, because a route is a
/// per-consumer, per-build choice (a variable flips it) and the source must stay valid for the GPU (m8.r1).
pub(crate) fn expand_in_routed(
    module: syn::ItemMod,
    environment: &Environment,
    profile: Profile,
    request: &routing::Request<'_>,
) -> syn::Result<TokenStream> {
    let (module, plan, constructors) = prepare_user_module(module, environment)?;
    expand_validated_with(
        module,
        environment,
        profile,
        &GRID,
        &constructors,
        Identity::Module,
        &plan,
        request,
    )
}

/// The passes a user's module goes through before it is expanded, in their order: the variants become kernels first
/// (M8, D139), so that every body is validated and lowered like any kernel; the `#[pad]` structs are padded next
/// (m14.pad, D239), so that the validator, the layout assertions, the `GpuPod` impls and the Slang emitter all read
/// the padded struct; then the declared limits (before E0718, m8.f) and the validation. This is the one place a user's
/// module is prepared: [`expand_in_routed`] and the module form of `#[gpu_fn]` (`map.rs`, which adds its kernels
/// afterwards) both call it. The free `#[gpu_fn]` function and `par_for!` paths expand a module of their own making,
/// which holds no user struct, through `expand_validated`.
pub(crate) fn prepare_user_module(
    module: syn::ItemMod,
    environment: &Environment,
) -> syn::Result<(syn::ItemMod, variant::VariantPlan, TokenStream)> {
    let (module, plan) = variant::lower(module)?;
    let (module, constructors) = pad::apply(module, &GRID)?;
    crate::validate::limits::check_declared(&module, environment.manifest_dir.as_deref())?;
    crate::validate::validate_module_in(&GRID, &module)?;
    Ok((module, plan, constructors))
}

/// [`expand_in`] for a module that is already validated: the module's items are expanded as they stand, and `extra`
/// is emitted inside the generated module after the kernels' items. The map generator (`map.rs`) validates the user's
/// function as a helper of a module, then expands the module it synthesises around it (whose generated kernel uses
/// the reserved `__gust_` names the validator refuses in user code) and adds the host functions as `extra`. A module
/// built this way has no variants: every kernel is one portable body.
pub(crate) fn expand_validated(
    module: syn::ItemMod,
    environment: &Environment,
    profile: Profile,
    stage: &dyn StageProfile,
    extra: &TokenStream,
    identity: Identity,
) -> syn::Result<TokenStream> {
    expand_validated_routed(
        module,
        environment,
        profile,
        stage,
        extra,
        identity,
        &routing::Request::none(),
    )
}

/// [`expand_validated`] for a module that may carry a route: the map generator's `#[gpu_fn(route = "tag")] fn`.
pub(crate) fn expand_validated_routed(
    module: syn::ItemMod,
    environment: &Environment,
    profile: Profile,
    stage: &dyn StageProfile,
    extra: &TokenStream,
    identity: Identity,
    request: &routing::Request<'_>,
) -> syn::Result<TokenStream> {
    expand_validated_with(
        module,
        environment,
        profile,
        stage,
        extra,
        identity,
        &variant::VariantPlan::default(),
        request,
    )
}

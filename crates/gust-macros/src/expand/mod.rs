use std::{collections::HashMap, path::Path};

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::ext::IdentExt as _;

use crate::{
    diagnostics::{self, Code},
    profile::{GRID, PIXEL, ReflectionKind, Shape, StageProfile, TwinHarness},
    slang::{ParameterFlavor, parameter_flavor_in},
    validate::classify::{self, AttributeFacts, ShapePin},
};

mod artifact;
use artifact::Lookup;
#[cfg(test)]
pub(crate) use artifact::TWIN_TEXT;
pub(crate) use artifact::{Environment, content_key};

mod map;
#[cfg(test)]
pub(crate) use map::expand_map_in_routed;
#[cfg(test)]
pub(crate) use map::map_kernel;
pub(crate) use map::{ElementReject, element_types, expand_map_routed};

/// What names a kernel's export record to the tool that reads it: a `#[gpu]` module or function is one place in the
/// source, so its record says where (schema 1, `source`); a `par_for!` call is a body that can stand at any number of
/// places, so its record says nothing of them (schema 2) and each place writes its own sidecar (R1.1 M5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Identity {
    /// A `#[gpu]` module or function: schema 1, the record holds the kernel's `source`.
    Module,
    /// A `par_for!` call whose first line is `call_line`: schema 2 and a sidecar `<key>.<site16>.site.json`.
    Site { call_line: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct KernelOptions {
    pub workgroup_size: [u32; 3],
}

/// What the expansion knows about the build profile (D44, R1-9, M8 D138): the target `GUST_PROFILE` names and the
/// capabilities after it. The macro cannot infer a profile (the build script of `gust` owns the effective one and its
/// release inference), so `target` is `None` when the variable is unset or its value is one the build script refuses.
/// The profile is an argument, never read from the process inside the expander, so expansion tests run in parallel
/// threads without touching the environment.
///
/// `Profile::Cpu` and `Profile::Other` are the two values the expander has always branched on, kept as constants:
/// `Cpu` is exactly `GUST_PROFILE=cpu`, `Other` is an unset or refused variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    /// The target, `None` when the variable is unset (the build script infers it) or refused.
    pub target: Option<ProfileTarget>,
    /// The capabilities after the target.
    pub capabilities: CapabilitySet,
}

/// The target of a [`Profile`]: the four names `GUST_PROFILE` starts with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileTarget {
    /// `dev`: no artifact is required and the run-time bridge is allowed.
    Dev,
    /// `native`: strict, for the native GPU runtime.
    Native,
    /// `web`: strict, for the browser.
    Web,
    /// `cpu`: the production CPU target (D21).
    Cpu,
}

impl ProfileTarget {
    /// The target's name in `GUST_PROFILE` and in a manifest.
    const fn text(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Native => "native",
            Self::Web => "web",
            Self::Cpu => "cpu",
        }
    }
}

/// A capability of a device (M8). This is the macro's copy of `gust::capability::Capability`, which it cannot depend
/// on; the names are asserted equal by the `GRAMMAR` table of both crates' tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// Subgroup (wave) operations.
    Subgroups,
}

impl Capability {
    /// Every capability this release knows, in the order of `gust::capability::Capability::ALL`.
    pub const ALL: &'static [Capability] = &[Capability::Subgroups];

    /// The path of the capability in the expansion: `::gust::capability::Capability::Subgroups`.
    fn tokens(self) -> TokenStream {
        match self {
            Self::Subgroups => quote!(::gust::capability::Capability::Subgroups),
        }
    }

    /// The capability named `name` in a profile (`subgroups`); `f16` and `atomics64` are reserved and unknown here.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "subgroups" => Some(Self::Subgroups),
            _ => None,
        }
    }

    /// The name of the capability in a profile and in a manifest (`subgroups`): the inverse of [`Self::from_name`].
    pub const fn name(self) -> &'static str {
        match self {
            Self::Subgroups => "subgroups",
        }
    }

    const fn bit(self) -> u8 {
        match self {
            Self::Subgroups => 1,
        }
    }
}

/// A set of [`Capability`], small enough that [`Profile`] stays `Copy`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapabilitySet(u8);

impl CapabilitySet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// Whether `capability` is in the set.
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & capability.bit() != 0
    }

    /// The set with `capability` added.
    #[must_use]
    pub const fn with(self, capability: Capability) -> Self {
        Self(self.0 | capability.bit())
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set whose members are the bits of `bits`, one per capability in the order of [`Capability::ALL`] (bit 0 is
    /// `Capability::ALL[0]`). The tests of the selection use bits no capability has yet, to make sets of several.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// The set of the members of both.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every member of the set is in `other`.
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    /// The number of members.
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// The names of the members, in the order of [`Capability::ALL`].
    pub fn names(self) -> Vec<&'static str> {
        Capability::ALL
            .iter()
            .filter(|capability| self.contains(**capability))
            .map(|capability| capability.name())
            .collect()
    }

    /// The paths of the members in the expansion, in the order of [`Capability::ALL`].
    fn tokens(self) -> Vec<TokenStream> {
        Capability::ALL
            .iter()
            .filter(|capability| self.contains(**capability))
            .map(|capability| capability.tokens())
            .collect()
    }
}

#[allow(non_upper_case_globals)]
impl Profile {
    /// `GUST_PROFILE=cpu`: the production CPU target (D21). The CPU impl is ungated, no artifact is looked up or
    /// embedded, and nothing of the artifact directory is tracked.
    pub const Cpu: Self = Self {
        target: Some(ProfileTarget::Cpu),
        capabilities: CapabilitySet::EMPTY,
    };

    /// An unset variable, or a value the build script refuses: the CPU impl is a twin behind
    /// `cfg(any(test, gust_dual))` and artifacts are embedded as before.
    pub const Other: Self = Self {
        target: None,
        capabilities: CapabilitySet::EMPTY,
    };
}

impl Profile {
    /// The profile for the value of `GUST_PROFILE` (`None` when unset). A value the grammar refuses is [`Profile::Other`]:
    /// the build script rejects it with the real error, so the macro never reports it.
    pub fn from_value(value: Option<&str>) -> Self {
        value.and_then(Self::parse).unwrap_or(Self::Other)
    }

    /// The grammar of `GUST_PROFILE`: `<target>(+<capability>)*`, exact and lower-case, with no trimming. `None` for
    /// what the build script refuses: an unknown target, an empty, unknown, reserved (`f16`, `atomics64`) or repeated
    /// capability, any capability on `cpu`, and `subgroups` on `web`. The build script (`crates/gust/build.rs`) holds
    /// the mirror of this function with the real messages; `GRAMMAR` in the tests of both is the same table.
    pub fn parse(value: &str) -> Option<Self> {
        let mut pieces = value.split('+');
        let target = match pieces.next()? {
            "dev" => ProfileTarget::Dev,
            "native" => ProfileTarget::Native,
            "web" => ProfileTarget::Web,
            "cpu" => ProfileTarget::Cpu,
            _ => return None,
        };
        let mut capabilities = CapabilitySet::EMPTY;
        for name in pieces {
            let capability = Capability::from_name(name)?;
            if capabilities.contains(capability) {
                return None;
            }
            capabilities = capabilities.with(capability);
        }
        let refused = match target {
            ProfileTarget::Cpu => !capabilities.is_empty(),
            ProfileTarget::Web => capabilities.contains(Capability::Subgroups),
            ProfileTarget::Dev | ProfileTarget::Native => false,
        };
        (!refused).then_some(Self {
            target: Some(target),
            capabilities,
        })
    }

    /// Whether this is the production CPU target (`GUST_PROFILE=cpu`).
    pub fn is_cpu(self) -> bool {
        self == Self::Cpu
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
#[cfg(test)]
pub fn expand_with(module: syn::ItemMod, profile: Profile) -> syn::Result<TokenStream> {
    expand_in(module, &Environment::from_process(), profile)
}

/// [`expand_with`] for a module that may carry a route (`#[gpu(route = "tag")]`, M8, D65 to D67).
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

/// [`expand_with_directories`] for a `#[gpu]` function (an element-wise map): the same environment, so that the tests
/// of the manifest can drive the map path, which reads the crate's manifest too (GUST-E0753 at the function's name).
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
    // The variants become kernels first (M8, D139), so that every body is validated and lowered like any kernel.
    let (module, plan) = variant::lower(module)?;
    crate::validate::limits::check_declared(&module, environment.manifest_dir.as_deref())?; // before E0718 (m8.f)
    crate::validate::validate_module_in(&GRID, &module)?;
    expand_validated_with(
        module,
        environment,
        profile,
        &GRID,
        &TokenStream::new(),
        Identity::Module,
        &plan,
        request,
    )
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

/// [`expand_validated`] for a module that may carry a route: the map generator's `#[gpu(route = "tag")] fn`.
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

/// One body of a kernel as the expansion built it (M8, D139): the pieces of its descriptor that differ from body to
/// body. Every body is built in every build; the descriptor of the kernel takes the selected one's.
struct BuiltBody {
    /// The kernel the body belongs to.
    kernel: String,
    /// The function's name.
    function: String,
    /// Whether the build selected this body.
    selected: bool,
    /// The variant function's name, `""` for the kernel's own body.
    variant: String,
    requires: CapabilitySet,
    entry_point: String,
    slang_source: String,
    source_origin: TokenStream,
    artifacts: TokenStream,
    strict_artifact: TokenStream,
    export_error: TokenStream,
    /// The renamed function (`__gpu_typecheck_*`), for the proof that a variant has its kernel's signature.
    typecheck: syn::Ident,
}

/// What a kernel's expansion keeps for the end of the walk, when every body of it has been built.
struct PendingKernel {
    name: syn::Ident,
    typecheck: syn::Ident,
    visibility: syn::Visibility,
    handle_doc: String,
    module_name: String,
    shape_path: TokenStream,
    compatible_shapes: Vec<TokenStream>,
    workgroup_size: [u32; 3],
    parameter_descriptors: Vec<TokenStream>,
    typed_calls: TokenStream,
    cpu_twin: TokenStream,
    element_assertions: TokenStream,
}

/// [`expand_validated`] for a module whose variants [`variant::lower`] has lowered, with the `plan` it made, and the
/// route the module carries (`request`, M8): a module routed to the CPU is expanded as the CPU target is, for its own
/// kernels only.
#[allow(clippy::too_many_arguments)]
fn expand_validated_with(
    mut module: syn::ItemMod,
    environment: &Environment,
    profile: Profile,
    stage: &dyn StageProfile,
    extra: &TokenStream,
    identity: Identity,
    plan: &variant::VariantPlan,
    request: &routing::Request<'_>,
) -> syn::Result<TokenStream> {
    let original_module = module.clone();
    let module_name = module.ident.to_string();
    let manifest_dir = environment.manifest_dir.as_deref();
    // The crate's declarations (`[package.metadata.gust]`): read in every build, a CPU build included, because the
    // declared profiles bind it too (m8.b1). A malformed table is GUST-E0753 at the module; no `Cargo.toml` (a unit
    // test's scratch crate) declares nothing and tracks nothing.
    let declarations = crate::manifest::read_dir(manifest_dir).map_err(|error| {
        diagnostics::error_at(Code::GustMetadata, module.ident.span(), error.to_string())
    })?;
    let manifest_tracking = declarations
        .as_ref()
        .map(|_| crate::manifest::tracking())
        .unwrap_or_default();
    let declarations = declarations.unwrap_or_default();
    // The module's route, resolved nearest first (D66): the profile, the tag's variable, the tag's default and its
    // ancestors', then the GPU. A module routed to the CPU is built as the CPU target is for everything below that
    // depends on the profile: the ungated impl, no artifact, no manifest check, the portable body (D137). The
    // portable floor was checked before this, whatever the route says (`expand_in_routed`).
    let route = routing::resolve(request, environment, &declarations, profile)?;
    let profile = route
        .as_ref()
        .map_or(profile, |route| route.effective(profile));
    // A tag that resolved to `cpu` (a CPU route): the module needs no artifact and no manifest check.
    let routed_cpu = route
        .as_ref()
        .is_some_and(|route| route.target == routing::Target::Cpu);
    // The module's kernels share one target: the CPU's when the effective profile is `cpu`, that is when the build is
    // the `cpu` profile (every item, tagged or not: D137's first level) or the module's tag resolved to `cpu`; else
    // the GPU's. An untagged module under `cpu` keeps its no-op `require_artifact(false)`, as it had before routing.
    let cpu_target = profile == Profile::Cpu;
    // Every declared GPU profile has a body of every kernel (E0744), and the profile of this build has one (E0750).
    let declared = variant::Declared::new(declarations.profiles.as_deref());
    declared.check(plan)?;
    let export_mode = environment.export_dir.is_some();
    // Export mode (`cargo gust build`) builds under a fixed profile and writes every body, so a kernel no body of which
    // that profile can use is not an error there: the build is never shipped (M8, D139).
    let selected = plan.select(profile, export_mode)?;
    let manifest_present = profile != Profile::Cpu && artifact::manifest_present(manifest_dir);
    let record_directory = environment.record_directory();
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
    // The bodies of the module's kernels, every one built, and the kernels themselves, finished once all bodies are.
    let mut bodies: Vec<BuiltBody> = Vec::new();
    let mut pending: Vec<PendingKernel> = Vec::new();

    // The structs a constant buffer reaches are `#[repr(C, align(16))]` and carry the `Std140V1` assertions (D57); the
    // validator guarantees no struct is also a storage element.
    let original_items = &original_module
        .content
        .as_ref()
        .expect("validated inline module")
        .1;
    let uniform_structs = crate::uniform::reach(stage, original_items)?.uniform;
    let structs_by_name = crate::uniform::module_structs(original_items);
    for item in items.iter_mut() {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        let name = item.ident.unraw().to_string();
        if uniform_structs.contains(&name) {
            item.attrs.push(syn::parse_quote!(#[repr(C, align(16))]));
        } else {
            item.attrs.push(syn::parse_quote!(#[repr(C)]));
        }
        item.attrs.push(syn::parse_quote!(#[derive(Clone, Copy)]));
        let rules = if uniform_structs.contains(&name) {
            PodRules::Std140(crate::uniform::std140_struct(
                structs_by_name[&name],
                &structs_by_name,
                0,
            ))
        } else {
            PodRules::Storage
        };
        generated.push(gpu_pod_impl(item, &rules)?);
    }

    for item in items.iter_mut() {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let Some((options, cpu, attribute_facts)) = kernel_attribute_facts(stage, function)? else {
            continue;
        };
        let original_name = function.sig.ident.clone();
        let original_name_text = original_name.to_string();
        // A variant is built like a kernel and belongs to the kernel it names (M8, D139); `selected` says which body of
        // a kernel with several this build uses. A function the plan does not know is one portable body.
        let owner = plan.owner_of(&original_name_text);
        let body_facts = plan.body_of(&original_name_text);
        let kernel_name =
            owner.map_or_else(|| original_name_text.clone(), |owner| owner.kernel.clone());
        // The kernel's `cpu` option belongs to its portable body, which is lowered with it; the own body of a kernel
        // that requires a capability is lowered with `cpu = "unsupported"`, so a kernel of the plan reads its `cpu`
        // from the bodies, not from its own attribute (M8, D139).
        let cpu = plan
            .kernel(&original_name_text)
            .map_or(cpu, |bodies| bodies.cpu);
        let is_selected = selected
            .get(&kernel_name)
            .is_none_or(|chosen| *chosen == original_name_text);
        if owner.is_none() {
            declared.check_cpu_form(&original_name, cpu, plan.kernel(&original_name_text))?;
            if let Some(route) = &route {
                // The CPU form is the portable body's, as `declared.check_cpu_form` has it.
                let has_cpu_form = cpu != CpuTwin::Unsupported
                    && plan
                        .kernel(&original_name_text)
                        .is_none_or(|bodies| bodies.portable().is_some());
                route.check_kernel(&original_name, has_cpu_form)?;
            }
        }
        let entry_point = format!("gpu_{}_{}", module_name, original_name_text);
        let typecheck_name = format_ident!("__gpu_typecheck_{}", original_name);
        let visibility = function.vis.clone();
        let original_kernel = find_function(&original_module, &original_name_text);
        // The shapes the kernel can take (D63, plan 5.1) and the check of the pin (5.3). The descriptor lists the
        // result; the shape the kernel is built as is its pin's: a kernel pinned `pixel` that the classifier accepts is
        // built with the `pixel` profile (slice mx.p1a), every other kernel with the module's.
        let helpers = crate::validate::helper_names(&original_module);
        let classification = classify::classify(stage, &helpers, original_kernel, &attribute_facts);
        let mut stage: &dyn StageProfile = stage;
        if let Some(pin) = &attribute_facts.shape {
            classify::check_pin(original_kernel, &classification, pin)?;
            if pin.shape != classify::Shape::from_profile(stage.shape()) {
                stage = profile_of(pin.shape);
            }
        }
        let parameter_descriptors = parameter_descriptors(stage, &function.sig.inputs)?;
        let compatible_shapes: Vec<TokenStream> = classification
            .shapes
            .iter()
            .map(|shape| shape.tokens())
            .collect();
        // The workgroup size the descriptor and the record carry: what the reflection of the shape expects (D17). A
        // fragment has none; `[1, 1, 1]` stands for it, since the twin and the runtime refuse a size of 0.
        let workgroup_size = match stage.reflection() {
            ReflectionKind::Compute => options.workgroup_size,
            ReflectionKind::Fragment => stage.options().default_workgroup,
        };
        let emitted = crate::slang::emit_kernel_in(
            stage,
            &original_module,
            original_kernel,
            KernelOptions { workgroup_size },
        )?;
        // The shape the profile gives the kernel: the descriptor's `shape`, and the export record's when it is not
        // `grid` (mx.k).
        let shape = stage.shape();
        let shape_path = shape_tokens(shape);
        let reported = reported_file(original_kernel, environment);
        let source_origin = source_origin(reported.as_deref(), manifest_dir, &emitted.lines);
        let slang_source = emitted.source;
        // The key hashes the clean text, so it is computed here: the Slang text is final, and the CPU twin (below)
        // never reaches it.
        let key = artifact::content_key(&entry_point, &slang_source);
        let lookup = match artifact::lookup(manifest_dir, &key) {
            // A CPU build looks at no artifact: nothing is embedded and nothing is tracked. Nor does a build look at
            // the artifact of a body it does not use: only the selected body is in the crate (M8, D139).
            _ if profile == Profile::Cpu || !is_selected => Lookup::Absent,
            // Export mode is how `cargo gust build` writes the missing half, and it expands this crate to do so: a
            // half pair must not stop that expansion, so it counts as absent there (and only there).
            Lookup::Incomplete { .. } if export_mode => Lookup::Absent,
            other => other,
        };
        let artifacts = artifact::artifact_tokens(&key, &lookup);
        // A kernel routed to the CPU has no artifact to require, in a strict profile (`native`, `web`) too: it is a CPU
        // body in a GPU build (D137).
        let strict_artifact = if is_selected && !routed_cpu {
            artifact_check_items(&lookup, original_name.span())
        } else {
            TokenStream::new()
        };
        if lookup == Lookup::Present && manifest_span.is_none() {
            manifest_span = Some(original_name.span());
        }
        let export_error = match &record_directory {
            Ok(None) => TokenStream::new(),
            Ok(Some(directory)) => {
                let parameters =
                    parameter_records(stage, &function.sig.inputs, &structs, &structs_by_name)?;
                let file = origin_file(reported.as_deref(), manifest_dir, &emitted.lines);
                // Schema 4 for a body that needs a capability or is a variant: the record names the kernel it
                // belongs to and, by `variant`, which of its bodies it is (M8, D139).
                let requires_names = body_facts.map_or_else(Vec::new, |body| body.requires.names());
                let variant_record = body_facts
                    .filter(|body| !body.requires.is_empty() || !body.variant.is_empty())
                    .map(|body| artifact::VariantRecord {
                        requires: &requires_names,
                        variant: body.variant,
                    });
                let record = artifact::KernelRecord {
                    key: &key,
                    module: &module_name,
                    name: &kernel_name,
                    entry_point: &entry_point,
                    workgroup_size,
                    parameters: &parameters,
                    file: file.as_deref().unwrap_or_default(),
                    lines: if file.is_some() { &emitted.lines } else { &[] },
                    shape: record_shape(shape),
                    variant: variant_record,
                };
                let written = match identity {
                    Identity::Module => {
                        artifact::write_records(directory, &key, &slang_source, &record.json())
                    }
                    // A `par_for!` call: the record carries no origin (equal bodies write equal bytes) and the
                    // call's own file and line table go to its sidecar (R1.1 M5).
                    Identity::Site { call_line } => {
                        let site = artifact::SiteRecord {
                            key: &key,
                            file: record.file,
                            lines: record.lines,
                        };
                        artifact::write_site_records(
                            directory,
                            &key,
                            &slang_source,
                            &record.json_without_source(),
                            &artifact::site_file_name(&key, record.file, call_line),
                            &site.json(),
                        )
                    }
                };
                match written {
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
        bodies.push(BuiltBody {
            kernel: kernel_name,
            function: original_name_text.clone(),
            selected: is_selected,
            variant: body_facts.map_or_else(String::new, |body| body.variant.to_owned()),
            requires: body_facts.map_or(CapabilitySet::EMPTY, |body| body.requires),
            entry_point,
            slang_source,
            source_origin,
            artifacts,
            strict_artifact,
            export_error,
            typecheck: typecheck_name.clone(),
        });
        if owner.is_some() {
            // A variant has no handle, descriptor, typed call or CPU twin of its own: they are its kernel's.
            retire_kernel(function, typecheck_name);
            continue;
        }
        let (handle_doc, typed_calls) =
            typed_call_items(stage, &original_name, &function.sig.inputs)?;
        let element_assertions =
            element_layout_assertions(stage, &original_name, &function.sig.inputs)?;
        // The twin and the `cpu` form are the portable body's, whichever body the build selects (M8, D139): the body
        // whose set is empty, which is the kernel's own or its empty-set variant (GUST-E0743 makes it unique). A kernel
        // none of whose bodies is portable has no CPU form and no twin.
        let portable = match plan.kernel(&original_name_text) {
            None => Some(original_name_text.clone()),
            Some(bodies) => bodies
                .portable()
                .map(|index| bodies.function(index).to_owned()),
        };
        let cpu_twin = match portable {
            Some(portable) => {
                let portable = syn::Ident::new(&portable, original_name.span());
                cpu_twin_items(
                    stage,
                    &original_name,
                    &format_ident!("__gpu_typecheck_{}", portable),
                    &function.sig.inputs,
                    workgroup_size,
                    cpu,
                    profile,
                )?
            }
            None => TokenStream::new(),
        };
        retire_kernel(function, typecheck_name.clone());
        pending.push(PendingKernel {
            name: original_name,
            typecheck: typecheck_name,
            visibility,
            handle_doc,
            module_name: module_name.clone(),
            shape_path,
            compatible_shapes,
            workgroup_size,
            parameter_descriptors,
            typed_calls,
            cpu_twin,
            element_assertions,
        });
    }

    // Every body is built; each kernel now takes the body this build selected. A kernel built for the GPU carries the
    // marker the GPU runtime bounds its typed entry points on; one routed to the CPU does not, so `HeadlessDevice`
    // refuses it at the call site (D137).
    let target_path = if cpu_target {
        quote!(::gust::target::Target::Cpu)
    } else {
        quote!(::gust::target::Target::Gpu)
    };
    for kernel in pending {
        let PendingKernel {
            name,
            typecheck,
            visibility,
            handle_doc,
            module_name,
            shape_path,
            compatible_shapes,
            workgroup_size: [x, y, z],
            parameter_descriptors,
            typed_calls,
            cpu_twin,
            element_assertions,
        } = kernel;
        let kernel_text = name.to_string();
        let of_kernel = || bodies.iter().filter(|body| body.kernel == kernel_text);
        let body = of_kernel()
            .find(|body| body.selected)
            .expect("each kernel has a selected body");
        let BuiltBody {
            variant,
            entry_point,
            slang_source,
            source_origin,
            artifacts,
            strict_artifact,
            ..
        } = body;
        let requires = body.requires.tokens();
        // What the plan chose is the one thing the descriptor says of the others; their errors are theirs.
        let export_errors: Vec<&TokenStream> = of_kernel().map(|body| &body.export_error).collect();
        // One signature for every body: the token rule of `variant::lower` says it, and rustc proves it here, by
        // making the two functions the elements of one array, which is a type error unless their fn-item types coerce
        // to one fn pointer type (a mismatch is E0308 at the variant).
        let signature_proofs: Vec<TokenStream> = of_kernel()
            .filter(|other| other.function != kernel_text)
            .map(|other| {
                let other = &other.typecheck;
                quote_spanned! {other.span()=>
                    const _: () = {
                        let _ = [#typecheck, #other];
                    };
                }
            })
            .collect();

        descriptor_paths.push(name.clone());
        let gpu_marker = if cpu_target {
            TokenStream::new()
        } else {
            quote!(impl ::gust::target::GpuKernel for #name {})
        };
        generated.push(quote! {
            #[doc = #handle_doc]
            #[allow(non_camel_case_types)]
            #visibility struct #name;

            impl #name {
                /// The descriptor of this kernel.
                pub const DESCRIPTOR: ::gust::KernelDescriptor =
                    <Self as ::gust::Kernel>::DESCRIPTOR;
            }

            impl ::gust::Kernel for #name {
                // `KernelDescriptor` is `#[non_exhaustive]`, so the expansion cannot write a literal; the hidden
                // macro takes the same named fields and builds the descriptor (see its documentation).
                const DESCRIPTOR: ::gust::KernelDescriptor = ::gust::__gust_descriptor! {
                    module: #module_name,
                    name: stringify!(#name),
                    entry_point: #entry_point,
                    shape: #shape_path,
                    compatible_shapes: &[#(#compatible_shapes),*],
                    workgroup_size: [#x, #y, #z],
                    parameters: &[#(#parameter_descriptors),*],
                    slang_source: #slang_source,
                    source: #source_origin,
                    artifacts: #artifacts,
                    // The selected body: `""` and no capability for the portable one. The target is the module's
                    // route (m8.r1): the CPU for a CPU-routed module and for the `cpu` profile, else the GPU.
                    variant: #variant,
                    requires: &[#(#requires),*],
                    target: #target_path,
                };
            }

            #gpu_marker

            #typed_calls

            #strict_artifact

            #(#export_errors)*

            #(#signature_proofs)*

            #cpu_twin

            #element_assertions
        });
    }

    items.insert(0, syn::parse_quote! { use ::gust::prelude::*; });
    // The route variable of a routed module is read the way `GUST_PROFILE` is, so that cargo re-expands the module when
    // it changes; an untagged module reads none (its expansion text is what it was before routing).
    // It also reads `GUST_ROUTED_PROFILE` and, when `cargo gust` set it, checks it against the profile `gust` was built
    // for (`Resolved::guard`).
    let tracked_environment = {
        let base = tracked_environment();
        let routes = route
            .as_ref()
            .map(routing::Resolved::tracking)
            .unwrap_or_default();
        let routed = (request.variables)(routing::ROUTED_PROFILE_VARIABLE);
        let guard = route
            .as_ref()
            .map(|route| route.guard(routed.as_deref()))
            .unwrap_or_default();
        quote!(#base #routes #guard)
    };
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

            #extra

            #tracked_environment

            #manifest_tracking

            #manifest_include

            #manifest_check

            /// The descriptors of the kernels of this module.
            pub const MODULE_DESCRIPTOR: ::gust::ModuleDescriptor =
                ::gust::ModuleDescriptor {
                    name: #module_name,
                    kernels: &[#(#descriptor_paths::DESCRIPTOR),*],
                };
        }
    })
}

/// Build-time routing (M8, D65 to D67, D137): the target a `#[gpu]` module is built for.
///
/// `#[gpu(route = "physics")]` on a module (or on a `#[gpu] fn`) names a tag the crate declares in
/// `[package.metadata.gust.routes]` of its own `Cargo.toml`. The module's target is resolved once per expansion,
/// nearest first:
///
/// 1. `GUST_PROFILE=cpu` is the CPU (a variable that says `gpu` then is GUST-E0749; a source default of `gpu` yields);
/// 2. the variable `GUST_ROUTE__<package>__<tag>`, which `cargo gust` sets (m8.r2) and anyone may set;
/// 3. the tag's own `default` in the manifest, then each ancestor's (`physics.broadphase`, then `physics`);
/// 4. the GPU.
///
/// The result is checked against the tag's set per kernel ([`Resolved::check_kernel`]): the explicit `targets`, else
/// `gpu` and `cpu` for a kernel with a CPU form and `gpu` alone for one without. A module that carries no tag is never
/// looked at here, and its expansion is what it was before routing.
///
/// The macro cannot depend on `gust` (the dependency runs the other way), so [`variable_name`] and [`is_valid_tag`]
/// are token-for-token copies of `gust::routes::variable_name` and `gust::routes::is_valid_tag` (the `content_key`
/// precedent in `artifact.rs`; `tests/routes.rs` compares the lexed token streams, so layout and comments may differ); `tests/routes.rs` asserts the golden `(package, tag, variable)` table of
/// `crates/gust/src/routes.rs`, pasted verbatim, so the two spellings cannot drift unnoticed.
pub(crate) mod routing {
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
        let mut name =
            String::with_capacity(VARIABLE_PREFIX.len() + package.len() + 2 + tag.len() + 2);
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

    /// The tag `#[gpu(route = "..")]` named, and where.
    #[derive(Clone, Debug)]
    pub(crate) struct Tag {
        /// The text of the literal.
        pub text: String,
        /// The span of the literal, where the errors about the tag are reported.
        pub span: Span,
    }

    /// The one argument `#[gpu]` takes: `route = "<tag>"`, with an optional trailing comma.
    struct Argument(syn::LitStr);

    impl Parse for Argument {
        fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
            let name: syn::Ident = input.parse()?;
            if name != "route" {
                return Err(diagnostics::error_at(
                    Code::GpuArguments,
                    name.span(),
                    format!(
                        "#[gpu] takes one argument, `route = \"<tag>\"`, in the current Rust-to-Slang subset; `{name}` is not it"
                    ),
                ));
            }
            input.parse::<syn::Token![=]>()?;
            let literal: syn::LitStr = input.parse()?;
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            }
            if !input.is_empty() {
                return Err(diagnostics::error_at(
                    Code::GpuArguments,
                    input.span(),
                    "#[gpu] takes one argument, `route = \"<tag>\"`, and nothing after it",
                ));
            }
            Ok(Self(literal))
        }
    }

    /// The tag in the arguments of `#[gpu(..)]`; GUST-E0701 for anything but `route = "<tag>"`.
    pub(crate) fn parse_attribute(arguments: TokenStream) -> syn::Result<Tag> {
        let Argument(literal) = diagnostics::parse2(Code::GpuArguments, arguments)?;
        Ok(Tag {
            text: literal.value(),
            span: literal.span(),
        })
    }

    /// What reaches the expansion of a route: the tag the attribute named, if any, and where the route variables are
    /// read from (the process for the macro, a table for a unit test, so that tests never set an environment variable).
    pub(crate) struct Request<'a> {
        /// The tag; `None` for a module or function with no `route` (nothing here is looked at then).
        pub tag: Option<Tag>,
        /// The value of the variable called `name`, `None` when it is unset or empty.
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
    pub(crate) fn process_variable(name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|value| !value.is_empty())
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
        /// is checked (D67).
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
        pub(crate) fn check_kernel(
            &self,
            kernel: &syn::Ident,
            has_cpu_form: bool,
        ) -> syn::Result<()> {
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
}

/// Variants of a kernel (M8, D139): `#[variant(of = kernel, requires(..))]` functions beside the `#[kernel]` they
/// replace on a device that has the capabilities.
///
/// The `#[kernel]` function is the kernel and its portable body (or, with `#[kernel(requires(..))]`, a body that needs
/// capabilities itself). [`lower`] turns every variant into a `#[kernel]` function, before validation, so that the
/// validator and the Slang emitter see plain kernels and every body is validated and lowered in every build. The
/// lowered attribute has the kernel's `workgroup_size` and `shape`, the body's own `requires(..)` and a `cpu`: the
/// kernel's own `cpu` for the portable body (the body whose set is empty, which is the one the twin is built from, so
/// the validator judges it by that option), `cpu = "unsupported"` for every body that requires a capability (it never
/// has a twin). The kernel's own body is lowered the same way. [`VariantPlan`] remembers which body
/// belongs to which kernel. The expansion builds every body (and writes the record of each in export mode), selects
/// one per kernel for the profile ([`VariantPlan::select`]) and gives the kernel's descriptor that body's Slang and
/// artifact; the handle and the typed calls are the kernel's own, and the CPU twin is the portable body's (the body
/// whose set is empty, [`KernelBodies::portable`]).
pub(crate) mod variant {
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
        pub(crate) fn first_tie(
            &self,
            capability_count: u32,
        ) -> Option<(usize, usize, CapabilitySet)> {
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
            parts.extend(self.variants.iter().map(|variant| {
                format!("`{}` requires {}", variant.name, set_text(variant.requires))
            }));
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
        let walk =
            diagnostics::parse_nested_meta(attribute, Code::MalformedKernelAttribute, |meta| {
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
                            if first_error.is_none()
                                && is_coded(&error, Code::UnknownCapability) =>
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
                        "#[variant] goes on a function of a `#[gpu]` module, beside the `#[kernel]` it replaces",
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

    /// The error for `#[variant]` on a `#[gpu]` function (an element-wise map has one body), or `None` without one.
    pub(crate) fn misplaced_on_function(function: &syn::ItemFn) -> Option<syn::Error> {
        let attribute = function.attrs.iter().find(|a| is_variant(a))?;
        Some(diagnostics::error(
            Code::VariantAttribute,
            attribute,
            "#[variant] goes on a function of a `#[gpu]` module; a `#[gpu]` function is one body",
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
}

/// The kernel function as the module keeps it once it is expanded: a private type-checking shadow, so that a callable
/// host function is never offered, and no `#[kernel]` for rustc to ask about.
fn retire_kernel(function: &mut syn::ItemFn, typecheck: syn::Ident) {
    function
        .attrs
        .retain(|attribute| !attribute.path().is_ident("kernel"));
    function.sig.ident = typecheck;
    function.vis = syn::Visibility::Inherited;
}

/// The profile that builds a kernel pinned to `shape`.
fn profile_of(shape: classify::Shape) -> &'static dyn StageProfile {
    match shape {
        classify::Shape::Grid => &GRID,
        classify::Shape::Pixel => &PIXEL,
    }
}

/// The path of `gust::Shape` for the descriptor of a kernel of `shape`.
fn shape_tokens(shape: Shape) -> TokenStream {
    match shape {
        Shape::Grid => quote!(::gust::Shape::Grid),
        Shape::Pixel => quote!(::gust::Shape::Pixel),
    }
}

/// The shape's name in the export record, or `None` for `grid`: only a kernel of another shape has a `shape` key and
/// so a schema 3 record (D89); a `grid` kernel's record stays what it was, byte for byte.
fn record_shape(shape: Shape) -> Option<&'static str> {
    match shape {
        Shape::Grid => None,
        Shape::Pixel => Some("pixel"),
    }
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
/// plus the element layout of each storage parameter (`structs`, keyed as the source spells the names) and of each
/// uniform parameter (`structs_by_unraw_name`, keyed as [`crate::uniform::module_structs`] keys them).
fn parameter_records(
    stage: &dyn StageProfile,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
    structs: &HashMap<String, &syn::ItemStruct>,
    structs_by_unraw_name: &HashMap<String, &syn::ItemStruct>,
) -> syn::Result<Vec<artifact::ParameterRecord>> {
    Ok(parameter_facts(stage, inputs)?
        .into_iter()
        .map(|fact| {
            let (kind, access) = match fact.flavor {
                ParameterFlavor::Invocation => ("invocation", "not_applicable"),
                ParameterFlavor::StorageRead => ("storage", "read_only"),
                ParameterFlavor::StorageReadWrite => ("storage", "read_write"),
                ParameterFlavor::Uniform => ("uniform", "read_only"),
                ParameterFlavor::Value => ("value", "not_applicable"),
                ParameterFlavor::SampledTexture => ("texture", "read_only"),
                ParameterFlavor::StorageTexture | ParameterFlavor::WriteTexture => {
                    ("texture", "read_write")
                }
                ParameterFlavor::Sampler => ("sampler", "not_applicable"),
                ParameterFlavor::LocalId
                | ParameterFlavor::GroupId
                | ParameterFlavor::Workgroup => {
                    unreachable!("`parameter_facts` leaves out what is not in the descriptor")
                }
            };
            // The numbers are the ones the `const` assertions of the expansion check against the real types (D80): the
            // storage layout is the running sum of the field sizes, the uniform layout is `uniform::std140_struct`.
            // A texture's layout is its texel type and the texture facts (D91).
            let layout = match (&fact.element, &fact.uniform_element, &fact.texture) {
                (Some(element), _, _) => artifact::storage_resource(element, structs),
                (None, Some(element), _) => {
                    artifact::uniform_resource(element, structs_by_unraw_name)
                }
                (None, None, Some(texture)) => artifact::texture_resource(
                    &texture.record_element,
                    artifact::TextureRecord {
                        format: texture.format.map(|format| format.name),
                        access: texture.access.record_name(),
                        class: texture.scalar.name(),
                    },
                ),
                (None, None, None) => None,
            };
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
/// option is parsed (and rejected when it is wrong) by `kernel_attribute`, which this wraps. For the `grid` profile and
/// for tests only: the call sites of the macro pass their profile to `kernel_attribute_in`.
#[cfg(test)]
pub fn kernel_options(function: &syn::ItemFn) -> syn::Result<Option<KernelOptions>> {
    Ok(kernel_attribute(function)?.map(|(options, _)| options))
}

/// All options of a `#[kernel]` attribute: `workgroup_size(x, y, z)` and `cpu = "unsupported"`, each at
/// most once. `None` when the function has no `#[kernel]`. For the `grid` profile and for tests only.
#[cfg(test)]
pub fn kernel_attribute(function: &syn::ItemFn) -> syn::Result<Option<(KernelOptions, CpuTwin)>> {
    kernel_attribute_in(&GRID, function)
}

/// `kernel_attribute` with the options the profile `stage` accepts (D63): which of them exist, the workgroup size of
/// a kernel that gives none, and the text of the error for any other option.
pub(crate) fn kernel_attribute_in(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
) -> syn::Result<Option<(KernelOptions, CpuTwin)>> {
    Ok(kernel_attribute_facts(stage, function)?.map(|(options, cpu, _)| (options, cpu)))
}

/// [`kernel_attribute_in`] with the facts the shape classifier reads (slice mx.c): the `shape = ..` pin, and whether
/// the kernel set a workgroup size itself. The `shape` option is the dialect's, not a profile's: it is parsed under
/// every profile, and `cpu` is refused as a shape (a target is not a shape, D51).
pub(crate) fn kernel_attribute_facts(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
) -> syn::Result<Option<(KernelOptions, CpuTwin, AttributeFacts)>> {
    let rules = stage.options();
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
                workgroup_size: rules.default_workgroup,
            },
            CpuTwin::Emitted,
            AttributeFacts::default(),
        )));
    }

    let mut result = None;
    let mut cpu = None;
    let mut pin: Option<ShapePin> = None;
    let mut size_tokens: Option<TokenStream> = None;
    // The options below report their own coded errors. Failures `syn` words itself (a
    // missing comma, a literal that is not an integer) get the malformed-attribute code.
    diagnostics::parse_nested_meta(attribute, Code::MalformedKernelAttribute, |meta| {
        if rules.workgroup_size && meta.path.is_ident("workgroup_size") {
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
            size_tokens = Some(quote!(workgroup_size(#x, #y, #z)));
            Ok(())
        } else if meta.path.is_ident("shape") {
            if pin.is_some() {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::ShapeOption,
                    "duplicate `shape` option",
                ));
            }
            let path = meta.path.clone();
            let Ok(input) = meta.value() else {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::ShapeOption,
                    "the `shape` option needs a value: `shape = grid` or `shape = pixel`",
                ));
            };
            let Ok(value) = input.parse::<syn::Ident>() else {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::ShapeOption,
                    "the value of `shape` is a bare name: `shape = grid` or `shape = pixel`",
                ));
            };
            let shape = match value.to_string().as_str() {
                "grid" => classify::Shape::Grid,
                "pixel" => classify::Shape::Pixel,
                "cpu" => {
                    return Err(diagnostics::error(
                        Code::ShapeOption,
                        &value,
                        "`cpu` is a target, not a shape: build with `GUST_PROFILE=cpu` to run a kernel on the CPU, \
                         and pin `shape = grid` or `shape = pixel` only",
                    ));
                }
                other => {
                    return Err(diagnostics::error(
                        Code::ShapeUnknown,
                        &value,
                        format!("unknown shape `{other}`; the shapes are `grid` and `pixel`"),
                    ));
                }
            };
            pin = Some(ShapePin {
                shape,
                tokens: quote!(#path = #value),
            });
            Ok(())
        } else if rules.cpu && meta.path.is_ident("cpu") {
            if cpu.is_some() {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::DuplicateCpuOption,
                    "duplicate cpu option",
                ));
            }
            let value: syn::LitStr = meta.value()?.parse()?;
            if value.value() == "lockstep" {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::UnknownCpuValue,
                    "`cpu = \"lockstep\"` is not a CPU strategy: GUST defines none for barrier kernels (D118); the only \
                     value is `\"unsupported\"`",
                ));
            }
            if value.value() != "unsupported" {
                return Err(diagnostics::meta_error(
                    &meta,
                    Code::UnknownCpuValue,
                    "unknown `cpu` value; the only value is `\"unsupported\"`",
                ));
            }
            cpu = Some(CpuTwin::Unsupported);
            Ok(())
        } else if meta.path.is_ident("requires") {
            // The capabilities the kernel's own body needs (M8, D139). `variant::lower` reads them (and reports an
            // unknown one, GUST-E0741) before this runs; here the option is only accepted.
            let content;
            syn::parenthesized!(content in meta.input);
            content.parse::<TokenStream>()?;
            Ok(())
        } else {
            Err(diagnostics::meta_error(
                &meta,
                Code::UnknownKernelOption,
                rules.expected,
            ))
        }
    })?;
    Ok(Some((
        KernelOptions {
            workgroup_size: result.unwrap_or(rules.default_workgroup),
        },
        cpu.unwrap_or_default(),
        AttributeFacts {
            shape: pin,
            workgroup_size: size_tokens,
        },
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
    /// The struct type of a uniform parameter.
    uniform_element: Option<syn::Type>,
    /// The facts of a texture parameter.
    texture: Option<TextureParameter>,
}

/// What the descriptor and the export record say about a texture parameter (mx.t3, D64, D91).
struct TextureParameter {
    access: crate::slang::texture::Access,
    format: Option<&'static crate::slang::texture::FormatInfo>,
    /// The Rust type of one texel as the descriptor names it: the user's type for a sampled texture, the format's
    /// texel for a storage one.
    element: syn::Type,
    /// The same type by its bare name, which is what the record's layout writer reads.
    record_element: syn::Type,
    scalar: crate::slang::texture::Scalar,
}

fn parameter_facts(
    stage: &dyn StageProfile,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<Vec<ParameterFacts>> {
    let rules = stage.binding_rules();
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
            let flavor = parameter_flavor_in(stage, &input.ty);
            // The ids derived from the dispatch-thread id and the workgroup memory are no parameter of the entry
            // point: neither the descriptor nor the export record lists them.
            if !flavor.in_descriptor() {
                return Ok(None);
            }
            let binding = rules.gets_slot(flavor).then(|| {
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
            let uniform_element = if flavor == ParameterFlavor::Uniform {
                Some(crate::slang::wrapper_element_type(&input.ty)?.clone())
            } else {
                None
            };
            let texture = if flavor.is_texture() {
                let spec = crate::slang::texture::texture_spec(flavor, &input.ty)
                    .map_err(|message| diagnostics::error(Code::TextureType, &input.ty, message))?;
                let (element, record_element) = match (spec.sampled_texel, spec.format) {
                    (Some(texel), _) => (texel.clone(), crate::slang::texture::leaf_type(texel)),
                    (None, Some(format)) => (format.texel_type(), format.texel_leaf_type()),
                    (None, None) => unreachable!("a texture has a texel type or a format"),
                };
                Some(TextureParameter {
                    access: spec.access,
                    format: spec.format,
                    element,
                    record_element,
                    scalar: spec.scalar,
                })
            } else {
                None
            };
            Ok(Some(ParameterFacts {
                name: crate::slang::emit_ident(&pattern.ident),
                rust_type: input.ty.to_token_stream().to_string(),
                flavor,
                binding,
                element,
                uniform_element,
                texture,
            }))
        })
        .filter_map(Result::transpose)
        .collect()
}

/// The name and descriptor binding of each parameter the descriptor lists, for the test that the emitter numbers the
/// same slots.
#[cfg(test)]
pub(crate) fn parameter_bindings(
    stage: &dyn StageProfile,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<Vec<(String, Option<u32>)>> {
    Ok(parameter_facts(stage, inputs)?
        .into_iter()
        .map(|fact| (fact.name, fact.binding))
        .collect())
}

fn parameter_descriptors(
    stage: &dyn StageProfile,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<Vec<TokenStream>> {
    parameter_facts(stage, inputs)?
        .into_iter()
        .map(|fact| {
            let ParameterFacts {
                name,
                rust_type,
                flavor,
                binding,
                element,
                uniform_element,
                texture,
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
                // A sampled texture is read-only; a storage or write-only one is bound read-write (D64).
                ParameterFlavor::SampledTexture => (
                    quote!(::gust::ParameterKind::Texture),
                    quote!(::gust::Access::ReadOnly),
                ),
                ParameterFlavor::StorageTexture | ParameterFlavor::WriteTexture => (
                    quote!(::gust::ParameterKind::Texture),
                    quote!(::gust::Access::ReadWrite),
                ),
                ParameterFlavor::Sampler => (
                    quote!(::gust::ParameterKind::Sampler),
                    quote!(::gust::Access::NotApplicable),
                ),
                ParameterFlavor::LocalId
                | ParameterFlavor::GroupId
                | ParameterFlavor::Workgroup => {
                    unreachable!("`parameter_facts` leaves out what is not in the descriptor")
                }
            };
            let binding = match binding {
                Some(binding) => {
                    quote!(Some(::gust::ResourceBinding { group: 0, binding: #binding }))
                }
                None => quote!(None),
            };
            let resource_layout = match (element, uniform_element) {
                // A texture: its texel type is the element, and the facts the shader witnesses (D91) follow. The
                // element is named once, so a texel type that is no `GpuPod` still reports one E0277.
                (None, None) if texture.is_some() => {
                    let texture = texture.expect("guarded by the match arm");
                    let element = &texture.element;
                    let format = match texture.format {
                        Some(format) => {
                            let tokens = format.format_tokens();
                            quote!(::core::option::Option::Some(#tokens))
                        }
                        None => quote!(::core::option::Option::None),
                    };
                    let access = texture.access.tokens();
                    let class = texture.scalar.class_tokens();
                    quote!(Some(::gust::ResourceLayout::texture_v1(
                        <#element as ::gust::GpuPod>::LAYOUT,
                        ::gust::TextureLayout {
                            format: #format,
                            dimension: ::gust::TextureDimension::D2,
                            access: #access,
                            class: #class,
                        }
                    )))
                }
                // StorageV1 when the element is scalars and padding-free structs of them, StorageV2 when it holds
                // vectors or matrices. The element is named once, so a type that is no `GpuPod` still reports one
                // E0277.
                (Some(element), _) => quote!(Some({
                    const __GUST_ELEMENT: ::gust::TypeLayout = <#element as ::gust::GpuPod>::LAYOUT;
                    if __GUST_ELEMENT.is_storage_v1() {
                        ::gust::ResourceLayout::storage_v1(__GUST_ELEMENT)
                    } else {
                        ::gust::ResourceLayout::storage_v2(__GUST_ELEMENT)
                    }
                })),
                // A uniform is `Std140V1` always. Whether the type really is one is asserted next to the descriptor
                // (`element_layout_assertions`), with the same single E0277 for a type that is no `GpuPod`.
                (None, Some(element)) => quote!(Some(::gust::ResourceLayout::std140_v1(
                    <#element as ::gust::GpuPod>::LAYOUT
                ))),
                (None, None) => quote!(None),
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

/// An argument of a typed call in declaration order: the index into the resources or into the samplers.
#[derive(Clone, Copy)]
enum CallParameter {
    Resource(usize),
    Sampler(usize),
}

/// One storage, uniform or texture parameter of a kernel, as the typed call needs it.
struct ResourceArgument {
    name: syn::Ident,
    element: syn::Type,
    access: TokenStream,
    /// `::gust::kind::Storage`, `Uniform` or `Texture`: the `Kind` the bound of the generated call names.
    kind: TokenStream,
    /// Whether the parameter is a texture (its `kind` is `::gust::kind::Texture`).
    texture: bool,
}

/// The typed host API of one kernel (roadmap M2): `call`, `submit` and `dispatch` on the handle, plus a
/// non-generic constant assertion that the generated argument count equals the descriptor's number of
/// storage plus uniform parameters (so `cargo check` catches a mismatch, not only monomorphisation).
///
/// Each storage or uniform parameter becomes one `impl Into<Arg<'r, Ri>>` argument, in declaration order, under its
/// own name. `Ri` must be a `gust::Resource` of the parameter's element type and of the device of the
/// first argument, and of the parameter's kind: `Kind = ::gust::kind::Storage` for a buffer and `Kind =
/// ::gust::kind::Uniform` for a uniform (D57), so a `GpuUniform` where a buffer is expected, or the other way round,
/// is a compile error and not a refusal at submit. Every `Ri` carries `'r` (S0 decision A: `Arg<'r, Ri>` needs
/// `Ri: 'r`, and rustc does not infer that for a function signature). Generated value names carry the reserved
/// `__gust_` prefix (the validator rejects it in user code) and the generic parameters carry `__Gust`, so a user
/// parameter named `element_count` or `device`, or an element type named `R0` or `D`, cannot collide with them.
///
/// A plain value parameter never reaches this function: the emitter rejects it at expansion (GUST-E0801).
///
/// Returns the doc text for the kernel's handle struct (users read it on `kernel_name`; it states the rules above
/// for the user) and the items.
fn typed_call_items(
    stage: &dyn StageProfile,
    kernel: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<(String, TokenStream)> {
    let mut arguments = Vec::new();
    // The sampler parameters, and the declaration order of the call's arguments (a resource or a sampler).
    let mut samplers: Vec<syn::Ident> = Vec::new();
    let mut call_order: Vec<CallParameter> = Vec::new();
    for input in inputs {
        // `parameter_descriptors` has already rejected a receiver or a non-identifier pattern.
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
            continue;
        };
        let flavor = parameter_flavor_in(stage, &input.ty);
        // A texture is a resource of kind `Texture` whose element is its texel type; a sampler is not a resource at
        // all but a `SamplerArg` of the same device (D64).
        let mut texel = None;
        let (access, kind) = match flavor {
            ParameterFlavor::Sampler => {
                call_order.push(CallParameter::Sampler(samplers.len()));
                samplers.push(pattern.ident.clone());
                continue;
            }
            ParameterFlavor::SampledTexture
            | ParameterFlavor::StorageTexture
            | ParameterFlavor::WriteTexture => {
                let spec = crate::slang::texture::texture_spec(flavor, &input.ty)
                    .map_err(|message| diagnostics::error(Code::TextureType, &input.ty, message))?;
                texel = Some(spec.element_type());
                let access = if spec.access == crate::slang::texture::Access::Sampled {
                    quote!(::gust::Access::ReadOnly)
                } else {
                    quote!(::gust::Access::ReadWrite)
                };
                (access, quote!(::gust::kind::Texture))
            }
            // The derived ids and the workgroup memory are not arguments of a call: no buffer stands for them.
            ParameterFlavor::Invocation
            | ParameterFlavor::LocalId
            | ParameterFlavor::GroupId
            | ParameterFlavor::Workgroup => continue,
            ParameterFlavor::StorageRead => (
                quote!(::gust::Access::ReadOnly),
                quote!(::gust::kind::Storage),
            ),
            ParameterFlavor::StorageReadWrite => (
                quote!(::gust::Access::ReadWrite),
                quote!(::gust::kind::Storage),
            ),
            // A uniform is read-only for the kernel and has the kind `Uniform`: a `GpuUniform` is the only resource
            // that fits its bound, and a `GpuBuffer` the only one that fits a storage bound.
            ParameterFlavor::Uniform => (
                quote!(::gust::Access::ReadOnly),
                quote!(::gust::kind::Uniform),
            ),
            // `Value` is defensive: the emitter rejects a plain value parameter (GUST-E0801) before this runs.
            ParameterFlavor::Value => {
                return Ok((
                    format!(
                        "Handle of the GUST kernel `{kernel}`. It has a plain value parameter, which no Slang compute \
                         entry point accepts, so it has no typed `call`, `submit` or `dispatch`."
                    ),
                    TokenStream::new(),
                ));
            }
        };
        let element = match texel {
            Some(texel) => texel,
            None => crate::slang::wrapper_element_type(&input.ty)?.clone(),
        };
        call_order.push(CallParameter::Resource(arguments.len()));
        arguments.push(ResourceArgument {
            name: pattern.ident.clone(),
            element,
            access,
            kind,
            texture: flavor.is_texture(),
        });
    }
    let total = arguments.len();
    let count = proc_macro2::Literal::usize_unsuffixed(total);
    let sampler_total = samplers.len();
    let sampler_count = proc_macro2::Literal::usize_unsuffixed(sampler_total);
    let count_message = format!(
        "`{kernel}`: the generated typed call takes a different number of arguments than its descriptor has storage, \
         uniform and texture parameters, or a different number of samplers than its sampler parameters"
    );
    let names: Vec<String> = call_order
        .iter()
        .map(|parameter| match *parameter {
            CallParameter::Resource(index) => arguments[index].name.to_string(),
            CallParameter::Sampler(index) => samplers[index].to_string(),
        })
        .collect();
    // The wording of a kernel without textures is the one it had before they existed.
    let textured = !samplers.is_empty() || arguments.iter().any(|argument| argument.texture);
    let order = if names.is_empty() {
        "no resource arguments".to_owned()
    } else if textured {
        format!(
            "then the storage, uniform, texture and sampler arguments in declaration order: {}",
            names.join(", ")
        )
    } else {
        format!(
            "then the storage and uniform arguments in declaration order: {}",
            names.join(", ")
        )
    };
    let turbofish = if arguments.is_empty() {
        " The kernel has no resource arguments to infer the device from, so name it: `call::<Device>(..)`."
    } else {
        ""
    };
    let call_doc = format!(
        "Typed call of `{kernel}`: `__gust_element_count`, {order}. A storage argument is a reference to a \
         buffer of the kernel's element type, optionally wrapped in `gust::Independent` to let its length \
         differ from the element count; a uniform argument is a reference to a uniform of the kernel's struct \
         (it has no length).{turbofish}"
    );
    let handle_doc = format!(
        "Handle of the GUST kernel `{kernel}`, with the typed host API `call`, `submit` and `dispatch` \
         (one argument per storage buffer or uniform parameter, in parameter order).{turbofish} \
         Under `cfg(test)` of the crate that defines the kernel, or `--cfg gust_dual`, the kernel also runs on \
         `gust::cpu::CpuDevice` (its CPU twin), unless `#[kernel(cpu = \"unsupported\")]`. Under \
         `GUST_PROFILE=cpu` it runs on `gust::cpu::CpuDevice` in every build of the crate."
    );
    let submit_doc = format!(
        "Submit `{kernel}`: `__gust_device`, `__gust_element_count`, {order}. Returns a job to wait on."
    );
    let dispatch_doc = format!(
        "Run `{kernel}` and wait (on `wasm32`, a device whose job cannot block returns its wait error after \
         submitting): `__gust_device`, `__gust_element_count`, {order}."
    );
    // The multi-dimensional forms (slice m7.d): the same three functions over a `gust::Extent` in place of the element
    // count. The one-dimensional forms are `Extent { x: n, y: 1, z: 1 }`.
    let extent_note = "The extent is `gust::Extent { x, y, z }`: each axis is rounded up to whole workgroups, a shared \
         buffer holds exactly `x * y * z` elements, an axis of 0 runs nothing, and the rounded launch is at most \
         2^32 invocations on every backend.";
    let call_extent_doc = format!(
        "Typed call of `{kernel}` over an extent: `__gust_extent`, {order}. Like `call`, with a `gust::Extent` \
         in place of the element count. {extent_note}{turbofish}"
    );
    let submit_extent_doc = format!(
        "Submit `{kernel}` over an extent: `__gust_device`, `__gust_extent`, {order}. Returns a job to wait on. \
         {extent_note}"
    );
    let dispatch_extent_doc = format!(
        "Run `{kernel}` over an extent and wait (on `wasm32`, a device whose job cannot block returns its wait \
         error after submitting): `__gust_device`, `__gust_extent`, {order}. {extent_note}"
    );

    let assertion = if textured {
        quote! {
            const _: () = ::core::assert!(
                <#kernel as ::gust::Kernel>::DESCRIPTOR.storage_parameter_count()
                    + <#kernel as ::gust::Kernel>::DESCRIPTOR.uniform_parameter_count()
                    + <#kernel as ::gust::Kernel>::DESCRIPTOR.texture_parameter_count()
                    == #count
                    && <#kernel as ::gust::Kernel>::DESCRIPTOR.sampler_parameter_count()
                        == #sampler_count,
                #count_message
            );
        }
    } else {
        quote! {
            const _: () = ::core::assert!(
                <#kernel as ::gust::Kernel>::DESCRIPTOR.storage_parameter_count()
                    + <#kernel as ::gust::Kernel>::DESCRIPTOR.uniform_parameter_count()
                    == #count,
                #count_message
            );
        }
    };

    // A sampler is used through a texture's `sample`, so a kernel with a sampler and no resource has nothing to take
    // the device from: it gets no typed call (its descriptor, record and twin are unaffected).
    if arguments.is_empty() && !samplers.is_empty() {
        return Ok((
            format!(
                "Handle of the GUST kernel `{kernel}`. It has a sampler parameter but no buffer or texture, so it has \
                 no typed `call`, `submit` or `dispatch`."
            ),
            TokenStream::new(),
        ));
    }
    if arguments.is_empty() {
        // The forms over a `gust::Extent`: a shape whose invocations are not a grid has the one-dimensional forms only.
        let extent_items = if stage.binding_rules().extent_calls {
            quote! {
                #[doc = #call_extent_doc]
                pub fn call_extent<'r, __GustD: ::gust::TypedDevice>(
                    __gust_extent: ::gust::Extent,
                ) -> ::gust::Call<'r, __GustD, Self, 0> {
                    ::gust::Call::new_extent(__gust_extent, [])
                }

                #[doc = #submit_extent_doc]
                #[allow(clippy::type_complexity)]
                pub fn submit_extent<'d, __GustD: ::gust::Run<Self, 0>>(
                    __gust_device: &'d __GustD,
                    __gust_extent: ::gust::Extent,
                ) -> ::core::result::Result<
                    <__GustD as ::gust::Device>::Job<'d>,
                    <__GustD as ::gust::Device>::Error,
                > {
                    <__GustD as ::gust::Run<Self, 0>>::submit_call(__gust_device, &Self::call_extent(__gust_extent))
                }

                #[doc = #dispatch_extent_doc]
                pub fn dispatch_extent<__GustD: ::gust::Run<Self, 0>>(
                    __gust_device: &__GustD,
                    __gust_extent: ::gust::Extent,
                ) -> ::core::result::Result<(), <__GustD as ::gust::Device>::Error> {
                    <__GustD as ::gust::Run<Self, 0>>::dispatch_call(
                        __gust_device,
                        &Self::call_extent(__gust_extent),
                    )
                }
            }
        } else {
            TokenStream::new()
        };
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

                    #extent_items
                }
            },
        ));
    }

    let resources: Vec<syn::Ident> = (0..total)
        .map(|index| format_ident!("__GustR{}", index))
        .collect();
    let sampler_types: Vec<syn::Ident> = (0..sampler_total)
        .map(|index| format_ident!("__GustS{}", index))
        .collect();
    let first = &resources[0];
    let device = quote!(<#first as ::gust::Resource>::Device);
    // In declaration order: a resource is any `Into<Arg>` of its type, a sampler a reference to its type.
    let parameters: Vec<TokenStream> = call_order
        .iter()
        .map(|parameter| match *parameter {
            CallParameter::Resource(index) => {
                let (name, resource) = (&arguments[index].name, &resources[index]);
                quote!(#name: impl ::core::convert::Into<::gust::Arg<'r, #resource>>)
            }
            CallParameter::Sampler(index) => {
                let (name, sampler) = (&samplers[index], &sampler_types[index]);
                quote!(#name: &'r #sampler)
            }
        })
        .collect();
    let mut bounds: Vec<TokenStream> = arguments
        .iter()
        .zip(&resources)
        .enumerate()
        .map(|(index, (argument, resource))| {
            let (element, kind) = (&argument.element, &argument.kind);
            // Spanned on the parameter, so rustc's "required by this bound" note points at the
            // kernel parameter the bound belongs to, not at `#[gpu]`.
            let span = argument.name.span();
            if index == 0 {
                quote_spanned!(span=> #resource: ::gust::Resource<Element = #element, Kind = #kind> + 'r)
            } else {
                quote_spanned!(span=> #resource: ::gust::Resource<Element = #element, Device = #device, Kind = #kind> + 'r)
            }
        })
        .collect();
    // A sampler is an argument of the same device as the resources, and that device must have samplers (D64).
    for (name, sampler) in samplers.iter().zip(&sampler_types) {
        let span = name.span();
        bounds.push(quote_spanned!(span=> #sampler: ::gust::SamplerArg<Device = #device> + 'r));
    }
    if !samplers.is_empty() {
        bounds.push(quote!(#device: ::gust::SamplerDevice));
    }
    let forwarded: Vec<&syn::Ident> = call_order
        .iter()
        .map(|parameter| match *parameter {
            CallParameter::Resource(index) => &arguments[index].name,
            CallParameter::Sampler(index) => &samplers[index],
        })
        .collect();
    let bindings: Vec<TokenStream> = arguments
        .iter()
        .map(|argument| {
            let (name, access) = (&argument.name, &argument.access);
            quote!(::core::convert::Into::into(#name).bind(#access))
        })
        .collect();
    let sampler_bindings: Vec<TokenStream> = samplers
        .iter()
        .map(|name| quote!(::gust::SamplerArg::bind(#name)))
        .collect();
    // The call type, its constructors and the `Run` bound: `Call<.., N>` for a kernel without samplers, and the
    // `SampledCall` of its `S` samplers for one with them.
    let (call_type, new_call, new_extent_call, run_bound) = if samplers.is_empty() {
        (
            quote!(::gust::Call<'r, #device, Self, #count>),
            quote!(::gust::Call::new(__gust_element_count, [#(#bindings),*])),
            quote!(::gust::Call::new_extent(__gust_extent, [#(#bindings),*])),
            quote!(#device: ::gust::Run<Self, #count>),
        )
    } else {
        (
            quote!(::gust::SampledCall<'r, #device, Self, #count, #sampler_count>),
            quote!(::gust::Call::new_with_samplers(
                __gust_element_count,
                [#(#bindings),*],
                [#(#sampler_bindings),*]
            )),
            quote!(::gust::Call::new_extent_with_samplers(
                __gust_extent,
                [#(#bindings),*],
                [#(#sampler_bindings),*]
            )),
            quote!(#device: ::gust::Run<
                Self,
                #count,
                [<#device as ::gust::SamplerDevice>::SamplerBinding<'r>; #sampler_count]
            >),
        )
    };
    let run_trait = if samplers.is_empty() {
        quote!(::gust::Run<Self, #count>)
    } else {
        quote!(::gust::Run<
            Self,
            #count,
            [<#device as ::gust::SamplerDevice>::SamplerBinding<'r>; #sampler_count]
        >)
    };
    let generics: Vec<&syn::Ident> = resources.iter().chain(&sampler_types).collect();
    // The forms over a `gust::Extent`: a shape whose invocations are not a grid has the one-dimensional forms only.
    let extent_items = if stage.binding_rules().extent_calls {
        quote! {
            #[doc = #call_extent_doc]
            #[allow(clippy::too_many_arguments)]
            pub fn call_extent<'r, #(#generics),*>(
                __gust_extent: ::gust::Extent,
                #(#parameters),*
            ) -> #call_type
            where
                #(#bounds,)*
            {
                #new_extent_call
            }

            #[doc = #submit_extent_doc]
            #[allow(clippy::too_many_arguments, clippy::type_complexity)]
            pub fn submit_extent<'d, 'r, #(#generics),*>(
                __gust_device: &'d #device,
                __gust_extent: ::gust::Extent,
                #(#parameters),*
            ) -> ::core::result::Result<
                <#device as ::gust::Device>::Job<'d>,
                <#device as ::gust::Device>::Error,
            >
            where
                #(#bounds,)*
                #run_bound,
            {
                <#device as #run_trait>::submit_call(
                    __gust_device,
                    &Self::call_extent(__gust_extent, #(#forwarded),*),
                )
            }

            #[doc = #dispatch_extent_doc]
            #[allow(clippy::too_many_arguments)]
            pub fn dispatch_extent<'r, #(#generics),*>(
                __gust_device: &#device,
                __gust_extent: ::gust::Extent,
                #(#parameters),*
            ) -> ::core::result::Result<(), <#device as ::gust::Device>::Error>
            where
                #(#bounds,)*
                #run_bound,
            {
                <#device as #run_trait>::dispatch_call(
                    __gust_device,
                    &Self::call_extent(__gust_extent, #(#forwarded),*),
                )
            }
        }
    } else {
        TokenStream::new()
    };

    Ok((
        handle_doc,
        quote! {
            #assertion

            impl #kernel {
                #[doc = #call_doc]
                #[allow(clippy::too_many_arguments)]
                pub fn call<'r, #(#generics),*>(
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> #call_type
                where
                    #(#bounds,)*
                {
                    #new_call
                }

                #[doc = #submit_doc]
                #[allow(clippy::too_many_arguments, clippy::type_complexity)]
                pub fn submit<'d, 'r, #(#generics),*>(
                    __gust_device: &'d #device,
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> ::core::result::Result<
                    <#device as ::gust::Device>::Job<'d>,
                    <#device as ::gust::Device>::Error,
                >
                where
                    #(#bounds,)*
                    #run_bound,
                {
                    <#device as #run_trait>::submit_call(
                        __gust_device,
                        &Self::call(__gust_element_count, #(#forwarded),*),
                    )
                }

                #[doc = #dispatch_doc]
                #[allow(clippy::too_many_arguments)]
                pub fn dispatch<'r, #(#generics),*>(
                    __gust_device: &#device,
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> ::core::result::Result<(), <#device as ::gust::Device>::Error>
                where
                    #(#bounds,)*
                    #run_bound,
                {
                    <#device as #run_trait>::dispatch_call(
                        __gust_device,
                        &Self::call(__gust_element_count, #(#forwarded),*),
                    )
                }

                #extent_items
            }
        },
    ))
}

/// The CPU twin of one kernel (roadmap M3): `impl Run<Kernel, N> for ::gust::cpu::CpuDevice` under
/// `cfg(any(test, gust_dual))`, or ungated under [`Profile::Cpu`] (the production CPU target, D21: the same impl with
/// no `cfg` copy, so `cpu` plus `cfg(test)` or `--cfg gust_dual` still has exactly one). It calls the shadow body (the
/// renamed typecheck function) as written, once per invocation, and reaches the bindings only through
/// `::gust::cpu::{validate, read, write, invocations}`: no body copy, no rewrite, no IR and no downcast in the
/// emitted code. `gust/src/cpu.rs` holds a hand-written impl of this exact shape in its tests, which is the
/// specification.
///
/// Nothing is emitted for `cpu = "unsupported"`. A uniform parameter is read by copy (`::gust::cpu::read_uniform`) and
/// passed as `::gust::Uniform(copy)` (D57). Locals take the user's parameter names (the validator reserves `__gust_`,
/// so they cannot collide with `__gust_call` and `__gust_id`).
fn cpu_twin_items(
    stage: &dyn StageProfile,
    kernel: &syn::Ident,
    typecheck: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
    workgroup_size: [u32; 3],
    cpu: CpuTwin,
    profile: Profile,
) -> syn::Result<TokenStream> {
    if cpu == CpuTwin::Unsupported {
        return Ok(TokenStream::new());
    }
    let mut names = Vec::new();
    let mut sampler_names: Vec<syn::Ident> = Vec::new();
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
        match parameter_flavor_in(stage, &input.ty) {
            ParameterFlavor::Invocation => {
                let ty = &input.ty;
                arguments.push(quote! {
                    <#ty as ::gust::cpu::FromGlobalId>::from_global_id(__gust_id)
                });
            }
            // The local and the group id of the invocation, as the GPU derives them from the dispatch-thread id and
            // the workgroup size (slice m7.g1). A kernel with workgroup memory has no twin, so this serves a kernel
            // that only reads the ids.
            ParameterFlavor::LocalId | ParameterFlavor::GroupId => {
                let [x, y, z] = workgroup_size;
                let is_group = parameter_flavor_in(stage, &input.ty) == ParameterFlavor::GroupId;
                // An axis of size 1 is spelled out (`id % 1` is 0, `id / 1` is `id`): clippy denies `modulo_one`
                // in the generated twin, and the default size is `64, 1, 1`.
                let axis = |component: TokenStream, size: u32| {
                    if size == 1 {
                        if is_group { component } else { quote!(0u32) }
                    } else if is_group {
                        quote!(#component / #size)
                    } else {
                        quote!(#component % #size)
                    }
                };
                let (ax, ay, az) = (
                    axis(quote!(__gust_id.x), x),
                    axis(quote!(__gust_id.y), y),
                    axis(quote!(__gust_id.z), z),
                );
                arguments.push(quote! {
                    ::gust::UVec3 {
                        x: #ax,
                        y: #ay,
                        z: #az,
                    }
                });
            }
            // Validation requires `cpu = "unsupported"` of a kernel with workgroup memory, so this is not reached.
            ParameterFlavor::Workgroup => return Ok(TokenStream::new()),
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
            ParameterFlavor::Uniform => {
                // The uniform is read once, by copy, and each invocation gets `Uniform(copy)`: the shadow body cannot
                // write through it, like the GPU's read-only constant buffer. `Uniform` has `Deref` only, and the
                // validator refuses `mut` on the parameter and `.0` on it (GUST-E0321), the two ways around that.
                let element = crate::slang::wrapper_element_type(&input.ty)?;
                names.push(name.clone());
                acquisitions.push(quote! {
                    let #name = ::gust::cpu::read_uniform::<#element>(#name, #label)?;
                });
                arguments.push(quote!(::gust::Uniform(#name)));
            }
            // A texture is acquired by its texel type and viewed through the kernel-side type over the same texels
            // (`gust/src/cpu/texture.rs` has the hand-written impl of this shape, which is the specification).
            ParameterFlavor::SampledTexture => {
                let spec = crate::slang::texture::texture_spec(
                    parameter_flavor_in(stage, &input.ty),
                    &input.ty,
                )
                .map_err(|message| diagnostics::error(Code::TextureType, &input.ty, message))?;
                let texel = spec.element_type();
                names.push(name.clone());
                acquisitions.push(quote! {
                    let #name = ::gust::cpu::texture::read::<#texel>(#name, #label)?;
                });
                arguments
                    .push(quote!(::gust::Texture2D::new(&#name.texels, #name.width, #name.height)));
            }
            ParameterFlavor::StorageTexture | ParameterFlavor::WriteTexture => {
                let flavor = parameter_flavor_in(stage, &input.ty);
                let spec = crate::slang::texture::texture_spec(flavor, &input.ty)
                    .map_err(|message| diagnostics::error(Code::TextureType, &input.ty, message))?;
                let texel = spec.element_type();
                let marker = spec
                    .format
                    .expect("a storage texture has a format")
                    .marker_tokens();
                let view = if flavor == ParameterFlavor::StorageTexture {
                    quote!(::gust::StorageTexture2D)
                } else {
                    quote!(::gust::WriteTexture2D)
                };
                names.push(name.clone());
                acquisitions.push(quote! {
                    let mut #name = ::gust::cpu::texture::write::<#texel>(#name, #label)?;
                });
                arguments.push(quote!(
                    #view::<#marker>::new(&mut #name.texels, #name.width, #name.height)
                ));
            }
            ParameterFlavor::Sampler => {
                sampler_names.push(name.clone());
                arguments.push(quote!(::gust::cpu::texture::sampler(#name)));
            }
            ParameterFlavor::Value => return Ok(TokenStream::new()),
        }
    }
    let count = proc_macro2::Literal::usize_unsuffixed(names.len());
    // A kernel with samplers is `Run<K, N, [CpuSamplerBinding; S]>`, and reads the sampler bindings of its call.
    let sampler_count = proc_macro2::Literal::usize_unsuffixed(sampler_names.len());
    let (run_trait, call_type, sampler_binding) = if sampler_names.is_empty() {
        (
            quote!(::gust::Run<#kernel, #count>),
            quote!(::gust::Call<'_, Self, #kernel, #count>),
            TokenStream::new(),
        )
    } else {
        (
            quote!(::gust::Run<
                #kernel,
                #count,
                [::gust::cpu::texture::CpuSamplerBinding<'_>; #sampler_count]
            >),
            quote!(::gust::Call<
                '_,
                Self,
                #kernel,
                #count,
                [::gust::cpu::texture::CpuSamplerBinding<'_>; #sampler_count]
            >),
            quote!(let [#(#sampler_names),*] = *__gust_call.samplers();),
        )
    };
    let gate = if profile.is_cpu() {
        TokenStream::new()
    } else {
        quote!(#[cfg(any(test, gust_dual))])
    };
    // How the twin visits the invocations of the call: the shape's harness.
    let invocations = match stage.twin() {
        TwinHarness::Linear => {
            quote!(::gust::cpu::invocations(
                __gust_call.kernel(),
                __gust_call.extent()
            ))
        }
    };
    Ok(quote! {
        #gate
        impl #run_trait for ::gust::cpu::CpuDevice {
            fn submit_call(
                &self,
                __gust_call: &#call_type,
            ) -> ::core::result::Result<::gust::cpu::CpuJob, ::gust::cpu::Error> {
                ::gust::cpu::validate(self, __gust_call)?;
                let [#(#names),*] = *__gust_call.bindings();
                #sampler_binding
                #(#acquisitions)*
                for __gust_id in #invocations {
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
/// element's `GpuPod::LAYOUT`) has the size and alignment of the Rust type and is a StorageV2 layout (every StorageV1
/// layout is one) whose stride
/// is the Rust size, so a hand-written `GpuPod` impl that lies is a compile error here. It covers everything the
/// runtime checks (`create_typed_buffer`, dispatch validation) cover. The assertion reads the
/// descriptor rather than naming `GpuPod` again, so an element that is no `GpuPod` still reports the one
/// E0277 the descriptor already gives (the `gpupod-missing` fixture expects exactly one). A uniform parameter gets the
/// same assertion for `Std140V1` (its rules, the Rust size and alignment of the struct, and
/// `TypeLayout::is_std140_v1`). The message is a static string (const panics cannot format) that names the element,
/// the parameter and the kernel.
fn element_layout_assertions(
    stage: &dyn StageProfile,
    kernel: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<TokenStream> {
    let mut assertions = Vec::new();
    // The position in the descriptor: the derived ids and the workgroup memory take none.
    let mut position = 0_usize;
    for input in inputs {
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
            continue;
        };
        let flavor = parameter_flavor_in(stage, &input.ty);
        if !flavor.in_descriptor() {
            continue;
        }
        let index = position;
        position += 1;
        if !matches!(
            flavor,
            ParameterFlavor::StorageRead
                | ParameterFlavor::StorageReadWrite
                | ParameterFlavor::Uniform
        ) {
            continue;
        }
        let element = crate::slang::wrapper_element_type(&input.ty)?;
        // `assert!` reads its message as a format string.
        let text = type_text(element).replace('{', "{{").replace('}', "}}");
        let parameter = &pattern.ident;
        // `parameter_descriptors` emits one descriptor per input, in order, so `index` is this parameter's.
        let index = proc_macro2::Literal::usize_unsuffixed(index);
        if flavor == ParameterFlavor::Uniform {
            let message = format!(
                "GUST Std140V1: uniform element `{text}` layout does not match its Rust representation \
                 (parameter `{parameter}` of kernel `{kernel}`)"
            );
            assertions.push(quote! {
                const _: () = ::core::assert!(
                    match <#kernel as ::gust::Kernel>::DESCRIPTOR.parameters[#index].resource_layout {
                        ::core::option::Option::Some(__gust_layout) => {
                            ::core::matches!(
                                __gust_layout.rules,
                                ::gust::ShaderLayoutRules::Std140V1
                            ) && __gust_layout.element.size as usize
                                == ::core::mem::size_of::<#element>()
                                && __gust_layout.element.alignment as usize
                                    == ::core::mem::align_of::<#element>()
                                && __gust_layout.element_stride as usize
                                    == ::core::mem::size_of::<#element>()
                                && __gust_layout.element.is_std140_v1()
                        }
                        ::core::option::Option::None => false,
                    },
                    #message
                );
            });
            continue;
        }
        let message = format!(
            "GUST StorageV1: element `{text}` layout does not match its Rust representation \
             (parameter `{parameter}` of kernel `{kernel}`)"
        );
        assertions.push(quote! {
            const _: () = ::core::assert!(
                match <#kernel as ::gust::Kernel>::DESCRIPTOR.parameters[#index].resource_layout {
                    ::core::option::Option::Some(__gust_layout) => {
                        __gust_layout.element.size as usize == ::core::mem::size_of::<#element>()
                            && __gust_layout.element.alignment as usize
                                == ::core::mem::align_of::<#element>()
                            && __gust_layout.element_stride as usize
                                == ::core::mem::size_of::<#element>()
                            && __gust_layout.element.is_storage_v2()
                    }
                    ::core::option::Option::None => false,
                },
                #message
            );
        });
    }
    Ok(quote!(#(#assertions)*))
}

/// The layout rules the `const` assertions of a struct's `GpuPod` impl check.
#[derive(Clone, Debug)]
enum PodRules {
    /// `StorageV1` or `StorageV2`: `#[repr(C)]`, the largest field alignment, and no padding.
    Storage,
    /// `Std140V1` (the struct is reachable from a uniform parameter): `#[repr(C, align(16))]`, alignment and size
    /// rounded up to 16, and still no padding. The model is the macro's own constant-buffer layout of the struct (the
    /// validator rejected every module where it has a gap or a tail), `None` when a field type is one the macro cannot
    /// see into.
    Std140(Option<crate::uniform::Std140Struct>),
}

fn gpu_pod_impl(item: &syn::ItemStruct, rules: &PodRules) -> syn::Result<TokenStream> {
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
    let model = match rules {
        PodRules::Std140(Some(model)) => Some(model),
        _ => None,
    };
    let offset_assertions = fields.named.iter().enumerate().map(|(index, field)| {
        let field_ident = field.ident.as_ref().expect("named field");
        let earlier = &types[..index];
        if let Some(model) = model {
            // The constant-buffer offset the macro computed, named in the message (const panics cannot format).
            let offset = model.offsets[index] as usize;
            let message = format!(
                "GUST Std140V1: field `{field_ident}` of `{struct_name}` is not at byte {offset} of the \
                 constant-buffer layout (it is at another offset in Rust)"
            );
            return quote! {
                ::core::assert!(
                    ::core::mem::offset_of!(#ident, #field_ident) == #offset
                        && ::core::mem::offset_of!(#ident, #field_ident)
                            == 0 #( + ::core::mem::size_of::<#earlier>() )*,
                    #message
                );
            };
        }
        let message = match rules {
            PodRules::Storage => format!(
                // The wording keeps the name `StorageV1` because tests/ui pins it; the rule is StorageV2's (the offset
                // of a field is the sum of the sizes before it, which a vector's alignment can break).
                "GUST StorageV1: field `{field_ident}` of `{struct_name}` is not at its padding-free offset"
            ),
            PodRules::Std140(_) => format!(
                "GUST Std140V1: field `{field_ident}` of `{struct_name}` is not at its padding-free offset"
            ),
        };
        quote! {
            ::core::assert!(
                ::core::mem::offset_of!(#ident, #field_ident)
                    == 0 #( + ::core::mem::size_of::<#earlier>() )*,
                #message
            );
        }
    });
    let (padding_message, alignment_message, storage_message) = match rules {
        PodRules::Storage => (
            format!(
                "GUST StorageV2: `{struct_name}` has tail padding (its fields do not add up to its size)"
            ),
            format!(
                "GUST StorageV2: `{struct_name}` must have the largest alignment of its fields"
            ),
            format!("GUST StorageV2: `{struct_name}` layout is not StorageV2"),
        ),
        PodRules::Std140(_) => (
            format!(
                "GUST Std140V1: `{struct_name}` has padding (its fields do not add up to its size, which a \
                 constant buffer rounds up to 16): add explicit fields"
            ),
            format!(
                "GUST Std140V1: `{struct_name}` must have alignment 16, or the largest alignment of its fields if that is \
                 more"
            ),
            format!("GUST Std140V1: `{struct_name}` layout is not Std140V1"),
        ),
    };
    // The alignment the struct must have: the largest field alignment, and for a constant buffer at least 16.
    let floor: usize = match rules {
        PodRules::Storage => 1,
        PodRules::Std140(_) => 16,
    };
    let layout_check = match rules {
        PodRules::Storage => quote!(is_storage_v2()),
        PodRules::Std140(_) => quote!(is_std140_v1()),
    };
    let model_size = model.map(|model| {
        let size = model.size as usize;
        let message = format!(
            "GUST Std140V1: `{struct_name}` is not {size} bytes, the size of its constant-buffer layout"
        );
        quote! {
            ::core::assert!(::core::mem::size_of::<#ident>() == #size, #message);
        }
    });
    // One block, after the impl whose `LAYOUT` it reads (const items are order-independent), so one cause is one
    // E0080 and the asserts run in this order: the field that is off its offset is named first, then the total,
    // then the alignment (a lying alignment also fails the last one), then the whole layout. Vectors and matrices make
    // padding possible (`{ a: f32, v: Vec2 }`), and the validator (E0208) rejects what it can see; these asserts are
    // the check for what it cannot (a field type the macro cannot see into).
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
            #model_size
            ::core::assert!(
                ::core::mem::align_of::<#ident>()
                    == { let mut __gust_alignment = #floor; #(
                        if ::core::mem::align_of::<#types>() > __gust_alignment {
                            __gust_alignment = ::core::mem::align_of::<#types>();
                        }
                    )* __gust_alignment },
                #alignment_message
            );
            ::core::assert!(
                <#ident as ::gust::GpuPod>::LAYOUT.#layout_check,
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
            &gpu_pod_impl(&item, &PodRules::Storage).unwrap_err(),
            Code::PodStructNamedFields,
        );
    }

    #[test]
    fn a_structs_asserts_share_one_const_after_its_impl_in_a_fixed_order() {
        let item: syn::ItemStruct = syn::parse_quote! { struct Holder { a: uint, w: Wide8 } };
        let text = gpu_pod_impl(&item, &PodRules::Storage).unwrap().to_string();
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
            "`Holder` must have the largest alignment of its fields",
            "`Holder` layout is not StorageV2",
        ];
        let positions: Vec<usize> = order.iter().map(|needle| at(needle)).collect();
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "{positions:?} for {order:?} in {text}"
        );
        // The unnamed size-sum assert carries a message too, so no bare `assertion failed` can reach a user.
        assert!(
            text.contains("== 0 + :: core :: mem :: size_of :: < uint > () + :: core :: mem :: size_of :: < Wide8 > () , \"GUST StorageV2: `Holder` has tail padding"),
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
        assert!(Profile::from_value(Some("cpu")).is_cpu());
        // The valid targets keep their name (M8: the profile carries a target and capabilities now); they are not
        // `Cpu`.
        for (value, target) in [
            ("dev", ProfileTarget::Dev),
            ("native", ProfileTarget::Native),
            ("web", ProfileTarget::Web),
        ] {
            let profile = Profile::from_value(Some(value));
            assert_eq!(profile.target, Some(target), "{value}");
            assert_ne!(profile, Profile::Cpu, "{value}");
            assert!(!profile.is_cpu(), "{value}");
        }
        // An unset variable and every value the build script refuses is `Other`, `cpu+simd` among them.
        for other in [
            None,
            Some("CPU"),
            Some("Cpu"),
            Some(" cpu"),
            Some("cpu "),
            Some("cpu+simd"),
            Some("cpu+subgroups"),
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

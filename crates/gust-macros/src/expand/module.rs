//! The expansion of one validated module: its kernels, bodies and splits.

use super::*;

/// One body of a kernel as the expansion built it (M8, D139): the pieces of its descriptor that differ from body to
/// body. Every body is built in every build; the descriptor of the kernel takes the selected one's.
pub(super) struct BuiltBody {
    /// The kernel the body belongs to.
    pub(super) kernel: String,
    /// The function's name.
    pub(super) function: String,
    /// Whether the build selected this body.
    pub(super) selected: bool,
    /// The variant function's name, `""` for the kernel's own body.
    pub(super) variant: String,
    pub(super) requires: CapabilitySet,
    pub(super) entry_point: String,
    pub(super) slang_source: String,
    pub(super) source_origin: TokenStream,
    pub(super) artifacts: TokenStream,
    pub(super) strict_artifact: TokenStream,
    pub(super) export_error: TokenStream,
    /// The split variant of the body (m11.g4a): `Some` for every body of a `grid` kernel built for the GPU; only the
    /// selected body's is in the descriptor.
    pub(super) split: Option<SplitBody>,
    /// The renamed function (`__gpu_typecheck_*`), for the proof that a variant has its kernel's signature.
    pub(super) typecheck: syn::Ident,
}

/// The split variant of a body: the kernel's Slang with the hidden launch-base uniform (m11.g4a).
pub(super) struct SplitBody {
    pub(super) source: String,
    /// The key of the variant's artifacts: [`artifact::content_key`] of the body's entry point and the variant's text.
    pub(super) key: String,
    /// What `gust-artifacts/` holds under that key, as this build sees it (m11.psplit-a, D209).
    pub(super) lookup: Lookup,
    /// The line table of the variant's own text, for the export record.
    pub(super) lines: Vec<crate::slang::LineRecord>,
    /// The `SourceOrigin` of the variant's own text, whose lines are one or two further down than the base's.
    pub(super) origin: TokenStream,
    /// The binding of the hidden uniform: the first one after the kernel's own.
    pub(super) binding: u32,
}

/// What a kernel's expansion keeps for the end of the walk, when every body of it has been built.
pub(super) struct PendingKernel {
    pub(super) name: syn::Ident,
    pub(super) typecheck: syn::Ident,
    pub(super) visibility: syn::Visibility,
    pub(super) handle_doc: String,
    pub(super) module_name: String,
    pub(super) shape_path: TokenStream,
    pub(super) compatible_shapes: Vec<TokenStream>,
    pub(super) workgroup_size: [u32; 3],
    pub(super) parameter_descriptors: Vec<TokenStream>,
    pub(super) typed_calls: TokenStream,
    pub(super) cpu_twin: TokenStream,
    pub(super) element_assertions: TokenStream,
}

/// [`expand_validated`] for a module whose variants [`variant::lower`] has lowered, with the `plan` it made, and the
/// route the module carries (`request`, M8): a module routed to the CPU is expanded as the CPU target is, for its own
/// kernels only.
#[allow(clippy::too_many_arguments)]
pub(super) fn expand_validated_with(
    mut module: syn::ItemMod,
    environment: &Environment,
    profile: Profile,
    stage: &dyn StageProfile,
    extra: &TokenStream,
    identity: Identity,
    plan: &variant::VariantPlan,
    request: &routing::Request<'_>,
) -> syn::Result<TokenStream> {
    // What the classifier and the emitter read: the module with every trait path call written as the method call it
    // lowers to (slice m9.g1). The twin and the typecheck item keep the tokens as the user wrote them.
    let original_module =
        crate::validate::traits::desugar(&module)?.unwrap_or_else(|| module.clone());
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
            syn::Item::Struct(item) => Some((item.ident.unraw().to_string(), item)),
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

    // A field-less enum is `#[repr(u32)]` unless it says `repr(i32)`, and `Clone` and `Copy`; the user derives the
    // comparison (m9.e1, D157). There is no `GpuPod`: an enum is never stored (GUST-E0211).
    for item in items.iter_mut() {
        let syn::Item::Enum(item) = item else {
            continue;
        };
        if !item
            .attrs
            .iter()
            .any(|attribute| attribute.path().is_ident("repr"))
        {
            item.attrs.push(syn::parse_quote!(#[repr(u32)]));
        }
        item.attrs.push(syn::parse_quote!(#[derive(Clone, Copy)]));
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
        // The artifact pair of a key as this build sees it: a CPU build looks at no artifact: nothing is embedded and
        // nothing is tracked. Nor does a build look at the artifact of a body it does not use: only the selected body is
        // in the crate (M8, D139). Export mode is how `cargo gust build` writes the missing half, and it expands this
        // crate to do so: a half pair must not stop that expansion, so it counts as absent there (and only there).
        let find_artifact = |key: &str| match artifact::lookup(manifest_dir, key) {
            _ if profile == Profile::Cpu || !is_selected => Lookup::Absent,
            Lookup::Incomplete { .. } if export_mode => Lookup::Absent,
            other => other,
        };
        // The split variant of a body of a `grid` kernel built for the GPU (m11.g4a): the same Slang with one hidden
        // uniform. In export mode every such body has one, selected or not, so that its record is written and a later
        // build that selects another body finds the split artifact of that one (m11.psplit-a, D209); outside export
        // mode only the selected body emits one. Only the selected body's is embedded. It has a key of its own, from
        // its own text, and the pair under that key is its artifact.
        let split = if !cpu_target && shape == Shape::Grid && (is_selected || export_mode) {
            let facts = parameter_facts(stage, &function.sig.inputs)?;
            let binding = u32::try_from(facts.iter().filter(|fact| fact.binding.is_some()).count())
                .expect("a kernel has fewer than 2^32 parameters");
            let variant = crate::slang::emit_kernel_in(
                &GRID_SPLIT,
                &original_module,
                original_kernel,
                KernelOptions { workgroup_size },
            )?;
            // The binding the descriptor names is the one the emitter wrote: a mismatch would be a variant the
            // runtime's reflection check refuses at run time, so it is a build error here (an internal one: no
            // user input can make the two differ).
            let declared = format!(
                "[[vk::binding({binding}, 0)]] ConstantBuffer<__GustLaunchBase> __gust_launch_base;"
            );
            assert!(
                variant.source.contains(&declared),
                "internal error: the split variant's binding differs from the descriptor's"
            );
            let key = artifact::content_key(&entry_point, &variant.source);
            let lookup = find_artifact(&key);
            Some(SplitBody {
                origin: self::source_origin(reported.as_deref(), manifest_dir, &variant.lines),
                key,
                lookup,
                lines: variant.lines,
                source: variant.source,
                binding,
            })
        } else {
            None
        };
        // The key hashes the clean text, so it is computed here: the Slang text is final, and the CPU twin (below)
        // never reaches it.
        let key = artifact::content_key(&entry_point, &slang_source);
        let lookup = find_artifact(&key);
        let artifacts = artifact::artifact_tokens(&key, &lookup);
        // A kernel routed to the CPU has no artifact to require, in a strict profile (`native`, `web`) too: it is a CPU
        // body in a GPU build (D137).
        let mut strict_artifact = if is_selected && !routed_cpu {
            artifact_check_items(&lookup, original_name.span())
        } else {
            TokenStream::new()
        };
        // The split's artifact is never required: without it the launch keeps its `DeviceLimit` refusal (D208). A half
        // pair is the same error as a body's, and in export mode it counted as absent above.
        if is_selected
            && !routed_cpu
            && let Some(split) = &split
            && matches!(split.lookup, Lookup::Incomplete { .. })
        {
            strict_artifact.extend(artifact_check_items(&split.lookup, original_name.span()));
        }
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
                    split_of: None,
                };
                let write = |record: &artifact::KernelRecord<'_>, slang: &str| match identity {
                    Identity::Module => {
                        artifact::write_records(directory, record.key, slang, &record.json())
                    }
                    // A `par_for!` call: the record carries no origin (equal bodies write equal bytes) and the
                    // call's own file and line table go to its sidecar (R1.1 M5).
                    Identity::Site { call_line } => {
                        let site = artifact::SiteRecord {
                            key: record.key,
                            file: record.file,
                            lines: record.lines,
                        };
                        artifact::write_site_records(
                            directory,
                            record.key,
                            slang,
                            &record.json_without_source(),
                            &artifact::site_file_name(record.key, record.file, call_line),
                            &site.json(),
                        )
                    }
                };
                let mut written = write(&record, &slang_source);
                // The split of the body is a record of its own (schema 5, m11.psplit-a): the body's parameters and the
                // hidden uniform, the split's key and text, the body's `requires` and `variant`.
                if let (Ok(()), Some(split)) = (&written, &split) {
                    let mut split_parameters = parameters.clone();
                    split_parameters.push(artifact::launch_base_parameter(split.binding));
                    let split_file = origin_file(reported.as_deref(), manifest_dir, &split.lines);
                    let split_record = artifact::KernelRecord {
                        key: &split.key,
                        parameters: &split_parameters,
                        file: split_file.as_deref().unwrap_or_default(),
                        lines: if split_file.is_some() {
                            &split.lines
                        } else {
                            &[]
                        },
                        // A schema 5 record always names the body's `requires` and `variant`: none, for a plain body.
                        variant: Some(variant_record.unwrap_or(artifact::VariantRecord {
                            requires: &[],
                            variant: "",
                        })),
                        split_of: Some(&key),
                        ..record
                    };
                    written = write(&split_record, &split.source);
                }
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
            split,
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
            // No twin, but a call can still be captured: `capture` says so (`NoTwin`).
            None => capture::no_twin_items(stage, &original_name, &function.sig.inputs, profile)?,
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
        // The kernel's target as a type, for `gust::routed::RoutedDevice` (m9.h1): the side the marker above implies.
        let routed_side = if cpu_target {
            quote!(::gust::routed::OnCpu)
        } else {
            quote!(::gust::routed::OnGpu)
        };
        // The split variant's descriptor (m11.g4a): an ordinary descriptor of its own, whose parameters are the
        // kernel's and the hidden launch-base uniform. It is a constant of its own so that the kernel's descriptor can
        // borrow it for `'static`; a kernel without a variant says `None`.
        // The kernel's parameters are written once, as a constant of their own, and both descriptors take them from it:
        // what a parameter's type has to satisfy (`GpuPod`) is then checked, and reported, once.
        let split_ident = format_ident!("__GUST_SPLIT_{}", name);
        let params_ident = format_ident!("__GUST_PARAMS_{}", name);
        let parameter_count = parameter_descriptors.len();
        let parameter_indices: Vec<syn::Index> =
            (0..parameter_count).map(syn::Index::from).collect();
        let params_item = quote! {
            #[doc(hidden)]
            #[allow(non_upper_case_globals)]
            const #params_ident: [::gust::ParameterDescriptor; #parameter_count] =
                [#(#parameter_descriptors),*];
        };
        let (split_item, split_field) = match &body.split {
            Some(split) => {
                let SplitBody {
                    source,
                    origin,
                    binding,
                    key: split_key,
                    lookup: split_lookup,
                    ..
                } = split;
                // The variant's own artifact pair when the crate holds it, else none: a host compiles the variant at
                // run time, or refuses the launch (D208, D209).
                let split_artifacts = artifact::artifact_tokens(split_key, split_lookup);
                let item = quote! {
                    #[doc(hidden)]
                    #[allow(non_upper_case_globals)]
                    const #split_ident: ::gust::KernelDescriptor = ::gust::__gust_descriptor! {
                        module: #module_name,
                        name: stringify!(#name),
                        entry_point: #entry_point,
                        shape: #shape_path,
                        compatible_shapes: &[#(#compatible_shapes),*],
                        workgroup_size: [#x, #y, #z],
                        parameters: &[
                            #(#params_ident[#parameter_indices],)*
                            ::gust::KernelDescriptor::__gust_launch_base_parameter(#binding)
                        ],
                        slang_source: #source,
                        source: #origin,
                        artifacts: #split_artifacts,
                        variant: #variant,
                        requires: &[#(#requires),*],
                        target: #target_path,
                        split: ::core::option::Option::None,
                    };
                };
                (item, quote!(::core::option::Option::Some(&#split_ident)))
            }
            None => (TokenStream::new(), quote!(::core::option::Option::None)),
        };
        generated.push(quote! {
            #params_item

            #split_item

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
                    parameters: &#params_ident,
                    slang_source: #slang_source,
                    source: #source_origin,
                    artifacts: #artifacts,
                    // The selected body: `""` and no capability for the portable one. The target is the module's
                    // route (m8.r1): the CPU for a CPU-routed module and for the `cpu` profile, else the GPU.
                    variant: #variant,
                    requires: &[#(#requires),*],
                    target: #target_path,
                    split: #split_field,
                };
            }

            #gpu_marker

            impl ::gust::routed::Routed for #name {
                type Side = #routed_side;
            }

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
                ::gust::ModuleDescriptor::new(
                    #module_name,
                    &[#(#descriptor_paths::DESCRIPTOR),*],
                );
        }
    })
}

/// The kernel function as the module keeps it once it is expanded: a private type-checking shadow, so that a callable
/// host function is never offered, and no `#[kernel]` for rustc to ask about.
pub(super) fn retire_kernel(function: &mut syn::ItemFn, typecheck: syn::Ident) {
    function
        .attrs
        .retain(|attribute| !attribute.path().is_ident("kernel"));
    function.sig.ident = typecheck;
    function.vis = syn::Visibility::Inherited;
}

/// The profile that builds a kernel pinned to `shape`.
pub(super) fn profile_of(shape: classify::Shape) -> &'static dyn StageProfile {
    match shape {
        classify::Shape::Grid => &GRID,
        classify::Shape::Pixel => &PIXEL,
    }
}

/// The path of `gust::Shape` for the descriptor of a kernel of `shape`.
pub(super) fn shape_tokens(shape: Shape) -> TokenStream {
    match shape {
        Shape::Grid => quote!(::gust::Shape::Grid),
        Shape::Pixel => quote!(::gust::Shape::Pixel),
    }
}

/// The shape's name in the export record, or `None` for `grid`: only a kernel of another shape has a `shape` key and
/// so a schema 3 record (D89); a `grid` kernel's record stays what it was, byte for byte.
pub(super) fn record_shape(shape: Shape) -> Option<&'static str> {
    match shape {
        Shape::Grid => None,
        Shape::Pixel => Some("pixel"),
    }
}

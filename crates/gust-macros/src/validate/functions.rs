//! Function items: the kernel and helper checks and the workgroup rules.

use super::*;

pub(super) fn validate_function(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
    function_names: &HashSet<String>,
    result_helpers: &HashSet<String>,
    methods: &MethodTable,
    owner: Option<&Owner<'_>>,
    closure: bool,
) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    // The word the messages use for a function that is not a kernel: a closure is judged as a helper (slice m9.c).
    let noun = if closure { "closure" } else { "helper" };
    // What the profile makes of each parameter (a receiver is none; `validate_function` rejects it later).
    let flavor_of = |argument: &syn::FnArg| match argument {
        syn::FnArg::Typed(argument) => Some(parameter_flavor_in(stage, &argument.ty)),
        syn::FnArg::Receiver(_) => None,
    };
    let is_kernel = crate::expand::kernel_attribute_in(stage, function)?.is_some();
    // The names of the subgroup operations are the dialect's: a call of one is that operation (the emitter and the
    // classifier read the name), so a function of the module cannot take the name (m8.s, D140).
    if barrier::subgroup_arity(&function.sig.ident.unraw().to_string()).is_some() {
        return Err(diagnostics::error(
            Code::SubgroupNotAllowed,
            &function.sig.ident,
            format!(
                "`{}` is the name of a subgroup operation; a function of a GPU module cannot take it, because a call \
                 of the name is that operation: choose another name",
                function.sig.ident
            ),
        ));
    }
    if !is_kernel
        && let Some(argument) = function
            .sig
            .inputs
            .iter()
            .find(|argument| flavor_of(argument) == Some(ParameterFlavor::Workgroup))
    {
        return Err(diagnostics::error(
            Code::WorkgroupParameterOnHelper,
            argument,
            format!(
                "`Workgroup<[T; N]>` memory belongs to a kernel: a {noun} cannot take it as a parameter; declare the \
                 array on the kernel and pass the {noun} the values it needs"
            ),
        ));
    }
    let invocations = function
        .sig
        .inputs
        .iter()
        .filter(|argument| flavor_of(argument) == Some(ParameterFlavor::Invocation))
        .count();
    if is_kernel {
        if invocations != 1 {
            return Err(diagnostics::error(
                Code::KernelDispatchParameter,
                &function.sig,
                stage.binding_rules().missing_dispatch_validate,
            ));
        }
        if !matches!(&function.sig.output, syn::ReturnType::Default)
            && !matches!(&function.sig.output, syn::ReturnType::Type(_, ty) if matches!(ty.as_ref(), syn::Type::Tuple(tuple) if tuple.elems.is_empty()))
        {
            return Err(diagnostics::error(
                Code::KernelReturnsUnit,
                &function.sig.output,
                "GPU kernels must return unit",
            ));
        }
    } else if function.sig.inputs.iter().any(|argument| {
        matches!(
            flavor_of(argument),
            Some(
                ParameterFlavor::StorageRead
                    | ParameterFlavor::StorageReadWrite
                    | ParameterFlavor::Uniform
                    | ParameterFlavor::SampledTexture
                    | ParameterFlavor::StorageTexture
                    | ParameterFlavor::WriteTexture
                    | ParameterFlavor::Sampler
            )
        )
    }) {
        return Err(diagnostics::error(
            Code::HelperResourceParameter,
            &function.sig,
            if closure {
                "a closure takes value parameters only (a buffer, uniform, texture or sampler is a name of the kernel, \
                 which a closure cannot capture); pass the values it needs"
            } else {
                "GPU helper functions currently accept value parameters only; access resources in the kernel"
            },
        ));
    }
    if is_kernel {
        // The constructs the shape refuses, judged from what the kernel declares (the profile never reads the body).
        let mut checks = BodyChecks {
            workgroup_memory: function
                .sig
                .inputs
                .iter()
                .any(|argument| flavor_of(argument) == Some(ParameterFlavor::Workgroup)),
            barrier: barrier::has_barrier(function),
            atomics: !atomic_buffer_names(function).is_empty(),
            forbidden: Vec::new(),
        };
        stage.validate_body(&mut checks);
        if let Some(sentence) = checks.forbidden.into_iter().next() {
            // A construct the kernel's shape does not take (GUST-E0729); the profile supplies the sentence.
            return Err(diagnostics::error(
                Code::ProfileRefusal,
                &function.sig,
                sentence,
            ));
        }
    }
    if function.sig.asyncness.is_some() {
        return Err(diagnostics::error(
            Code::AsyncFunction,
            &function.sig,
            "async is CPU-only in GPU modules",
        ));
    }
    if function.sig.unsafety.is_some() {
        return Err(diagnostics::error(
            Code::UnsafeFunction,
            &function.sig,
            "unsafe is not part of the Rust-to-Slang subset",
        ));
    }
    // A generic helper or inherent method was judged by `generics::check_signature` before the call (slice m9.g2); a
    // generic kernel or closure is not accepted.
    if function.sig.abi.is_some()
        || ((is_kernel || closure) && !function.sig.generics.params.is_empty())
    {
        return Err(diagnostics::error(
            Code::ExternOrGenericFunction,
            &function.sig,
            "extern GPU functions and generic kernels are not supported: a generic is a helper function or an inherent method with trait bounds, and kernels are concrete",
        ));
    }

    let mut visitor = RestrictedVisitor::new(function_names.clone(), result_helpers.clone());
    visitor.methods = methods.clone();
    if let Some(owner) = owner {
        visitor.self_struct = Some(owner.name.to_owned());
        visitor.owner_fields = owner.fields.clone();
    }
    visitor.atomic_buffers = atomic_buffer_names(function);
    visitor.array_lens = crate::slang::array_lengths(&function.sig.inputs, &function.block);
    // An array constant cannot be shadowed (a binding of its name is GUST-E0127), so its entry never meets a local's.
    for (name, length) in &methods.const_arrays {
        visitor.array_lens.insert(name.clone(), Some(*length));
    }
    for argument in &function.sig.inputs {
        if let syn::FnArg::Typed(argument) = argument
            && let syn::Pat::Ident(name) = argument.pat.as_ref()
        {
            if parameter_flavor_in(stage, &argument.ty) == ParameterFlavor::Value {
                visitor.value_params.insert(name.ident.unraw().to_string());
                if let Some(enum_name) = enum_type_name(&argument.ty, &methods.enums) {
                    visitor
                        .enum_values
                        .insert(name.ident.unraw().to_string(), enum_name);
                }
                if is_tuple_type(&argument.ty) {
                    visitor.tuple_values.insert(name.ident.unraw().to_string());
                }
            }
            match parameter_flavor_in(stage, &argument.ty) {
                ParameterFlavor::Invocation => {
                    visitor.invocations.insert(name.ident.unraw().to_string());
                }
                ParameterFlavor::LocalId | ParameterFlavor::GroupId => {
                    visitor.derived_ids.insert(name.ident.unraw().to_string());
                }
                ParameterFlavor::Workgroup => {
                    visitor.derived_ids.insert(name.ident.unraw().to_string());
                    if let syn::Type::Path(path) = argument.ty.as_ref() {
                        visitor
                            .workgroup_types
                            .insert(std::ptr::from_ref(path) as usize);
                    }
                    if let Err(message) = workgroup_array(&argument.ty) {
                        visitor.reject(Code::WorkgroupShape, &argument.ty, &message);
                    }
                }
                ParameterFlavor::StorageRead => {
                    visitor.resources.insert(name.ident.unraw().to_string());
                }
                ParameterFlavor::Uniform => {
                    visitor.uniforms.insert(name.ident.unraw().to_string());
                    // A constant buffer is read-only: a `mut` binding would let the twin's copy be written.
                    if name.mutability.is_some() {
                        visitor.reject(
                            Code::UniformWholeValue,
                            name,
                            "a uniform parameter is read-only: declare it without `mut`",
                        );
                    }
                }
                ParameterFlavor::StorageReadWrite => {
                    visitor.resources.insert(name.ident.unraw().to_string());
                    if let Some(element) = rw_buffer_element_type(&argument.ty) {
                        visitor
                            .read_write_buffers
                            .insert(name.ident.unraw().to_string(), element);
                    }
                }
                flavor @ (ParameterFlavor::SampledTexture
                | ParameterFlavor::StorageTexture
                | ParameterFlavor::WriteTexture) => {
                    let name = name.ident.unraw().to_string();
                    match texture::texture_spec(flavor, &argument.ty) {
                        Ok(spec) => {
                            visitor.textures.insert(
                                name,
                                TextureFacts {
                                    access: spec.access,
                                    filterable: spec.filterable(),
                                },
                            );
                        }
                        Err(message) => {
                            visitor.textures.insert(
                                name,
                                TextureFacts {
                                    access: texture::Access::Sampled,
                                    filterable: false,
                                },
                            );
                            visitor.reject(Code::TextureType, &argument.ty, &message);
                        }
                    }
                    if let syn::Type::Path(path) = argument.ty.as_ref() {
                        visitor
                            .texture_types
                            .insert(std::ptr::from_ref(path) as usize);
                    }
                }
                ParameterFlavor::Sampler => {
                    visitor.samplers.insert(name.ident.unraw().to_string());
                    if !texture::sampler_has_no_arguments(&argument.ty) {
                        visitor.reject(
                            Code::TextureType,
                            &argument.ty,
                            "`Sampler` takes no type arguments",
                        );
                    }
                    if let syn::Type::Path(path) = argument.ty.as_ref() {
                        visitor
                            .texture_types
                            .insert(std::ptr::from_ref(path) as usize);
                    }
                }
                ParameterFlavor::Value if matches!(argument.ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some()) =>
                {
                    visitor.result_values.insert(name.ident.unraw().to_string());
                }
                _ => {}
            }
        }
    }
    if !matches!(function.sig.output, syn::ReturnType::Default)
        && let Some(syn::Stmt::Expr(syn::Expr::Match(tail), None)) = function.block.stmts.last()
    {
        visitor.body_tail_match = Some(std::ptr::from_ref(tail) as usize);
    }
    visitor.visit_item_fn(function);
    visitor.errors.finish()?;
    // The closures bound in the body, once nothing else in it was found (slice m9.c).
    validate_closures(
        stage,
        function,
        function_names,
        result_helpers,
        methods,
        owner,
    )?;
    check_workgroup_rules(stage, function, is_kernel, closure)
}

/// The workgroup rules of a function the generic validation accepted (see `barrier`): the barrier positions (E0414),
/// no `return` (E0415) and no `break` or `continue` in a barrier loop (E0416), then the CPU strategy (E0717) and the
/// memory limit (E0718) of a kernel that uses workgroup memory or a barrier.
pub(super) fn check_workgroup_rules(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
    is_kernel: bool,
    closure: bool,
) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    let has_memory = function.sig.inputs.iter().any(|argument| {
        matches!(argument, syn::FnArg::Typed(argument)
            if parameter_flavor_in(stage, &argument.ty) == ParameterFlavor::Workgroup)
    });
    // A kernel that calls a subgroup operation is a barrier kernel for the early exits and the jumps (E0415, E0416): an
    // invocation that skipped the call would leave the others of its subgroup with a wrong value.
    let uses_subgroups = is_kernel && barrier::has_subgroup_call(function);
    let barrier_kernel =
        is_kernel && (has_memory || barrier::has_barrier(function) || uses_subgroups);
    barrier::check(
        function,
        is_kernel,
        barrier_kernel,
        barrier::requires_subgroups(function),
        closure,
    )?;
    // Full subgroups only (E0751): the invocation count is a multiple of 64, which every subgroup size of the
    // certified adapters (4, 8, 32, 64) divides. `Error::SubgroupSize` is the run-time check of the same rule.
    if uses_subgroups
        && let Some((options, _)) = crate::expand::kernel_attribute_in(stage, function)?
    {
        let [x, y, z] = options.workgroup_size;
        // In `u128`: three `u32` dimensions multiply to under 2^96, whereas `workgroup_size` has no upper bound and the
        // product in `u64` overflowed (and panicked the macro) for dimensions near `u32::MAX`.
        let invocations = u128::from(x) * u128::from(y) * u128::from(z);
        if !invocations.is_multiple_of(64) {
            return Err(diagnostics::error(
                Code::SubgroupInvocations,
                &function.sig,
                format!(
                    "kernel `{}` calls subgroup operations, so its workgroup runs whole subgroups only: the invocation \
                     count must be a multiple of 64, and `workgroup_size({x}, {y}, {z})` has {invocations}",
                    function.sig.ident
                ),
            ));
        }
    }
    if !(has_memory || barrier::has_barrier(function)) {
        return Ok(());
    }
    if !barrier_kernel {
        return Ok(());
    }
    let bytes = barrier::workgroup_bytes(function);
    if bytes > barrier::WORKGROUP_MEMORY_LIMIT {
        return Err(diagnostics::error(
            Code::WorkgroupMemoryLimit,
            &function.sig,
            format!(
                "the `Workgroup` arrays of this kernel take {bytes} bytes, over the limit of {} bytes (16 KiB, the \
                 WebGPU default): a scalar or `bool` element takes 4 bytes, a 2-wide vector 8 and a 3-wide or \
                 4-wide one 16; shorten the arrays",
                barrier::WORKGROUP_MEMORY_LIMIT
            ),
        ));
    }
    let twin = crate::expand::kernel_attribute_in(stage, function)?.map(|(_, twin)| twin);
    if !matches!(twin, Some(crate::expand::CpuTwin::Unsupported)) {
        let construct = if has_memory {
            "`Workgroup<[T; N]>` memory"
        } else {
            "`workgroup_barrier()`"
        };
        return Err(diagnostics::error(
            Code::BarrierKernelCpu,
            &function.sig,
            format!(
                "kernel `{}` uses {construct}, which has no CPU twin yet: say so with `cpu = \"unsupported\"`, as in \
                 `#[kernel(workgroup_size(256, 1, 1), cpu = \"unsupported\")]`",
                function.sig.ident
            ),
        ));
    }
    Ok(())
}

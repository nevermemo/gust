//! The `#[kernel]` attribute of a function: its options and facts.

use super::*;

pub(super) fn find_function<'a>(module: &'a syn::ItemMod, name: &str) -> &'a syn::ItemFn {
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

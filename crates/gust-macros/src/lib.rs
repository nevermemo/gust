use proc_macro::TokenStream;

mod diagnostics;
mod expand;
mod math_types;
mod recover;
mod slang;
mod validate;

use diagnostics::Code;

#[cfg(test)]
#[path = "tests/regression.rs"]
mod regression_tests;

/// Marks an inline Rust module as GUST GPU code.
///
/// The macro validates a deliberately small Rust subset, translates each
/// kernel directly to Slang, and generates host-side handles and descriptors.
#[proc_macro_attribute]
pub fn gpu(attribute: TokenStream, item: TokenStream) -> TokenStream {
    // The one place that reads the build profile (D44, R1-9): everything below takes it as an argument, so tests
    // never set an environment variable. `gust`'s build script validates the value; the macro only asks whether it
    // is exactly `cpu`. The expansion also emits `option_env!("GUST_PROFILE")`, which is what makes rustc track this
    // read: `std::env::var` alone is not tracked.
    let value = std::env::var("GUST_PROFILE").ok();
    let profile = expand::Profile::from_value(value.as_deref());
    gpu_impl_with(attribute.into(), item.into(), profile).into()
}

/// [`gpu_impl_with`] for a caller that has no profile: the `Other` path, which keeps the CPU twin behind
/// `cfg(any(test, gust_dual))`.
#[cfg(test)]
pub(crate) fn gpu_impl(
    attribute: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    gpu_impl_with(attribute, item, expand::Profile::Other)
}

/// [`gpu`] on `proc_macro2` tokens, so it can be tested outside a proc-macro context.
///
/// Every failure returns the `compile_error!` followed by the user's own code (see
/// [`recover`]), so rustc and editors still resolve the module's items and the one real
/// diagnostic is not buried in follow-on "cannot find" errors. The attribute-argument
/// error is reported before a parse error, as before.
pub(crate) fn gpu_impl_with(
    attribute: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
    profile: expand::Profile,
) -> proc_macro2::TokenStream {
    // `syn` words the parse failure itself; the code is added in front.
    let parsed = diagnostics::parse2::<syn::ItemMod>(Code::ItemNotModule, item.clone());
    if !attribute.is_empty() {
        let error = diagnostics::error_at(
            Code::GpuArguments,
            proc_macro2::Span::call_site(),
            "#[gpu] does not accept arguments in the current Rust-to-Slang subset",
        );
        return recover::beside_error(error, &item, parsed.ok());
    }
    let module = match parsed {
        Ok(module) => module,
        Err(error) => return recover::beside_error(error, &item, None),
    };
    match expand::expand_with(module.clone(), profile) {
        Ok(tokens) => tokens,
        Err(error) => recover::beside_error(error, &item, Some(module)),
    }
}

//! The procedural macros of GUST: `#[gpu]` and `par_for!`.
//!
//! Use them through `gust` (`gust::gpu`, `gust::par_for!`), which re-exports them and owns the types they emit; this
//! crate is not meant to be depended on directly. `#[gpu]` validates a small Rust subset and translates it directly
//! from the `syn` syntax tree to Slang; every rejection carries a stable diagnostic code (`GUST-E....`, see
//! `docs/DIAGNOSTICS.md`).
//!
//! # Generated items and `missing_docs`
//!
//! The macros emit public items (the handle and descriptor of each kernel, the typed calls, the module of a map
//! function). Each is documented or `#[doc(hidden)]`, so a crate with `#![deny(missing_docs)]`
//! compiles over a `#[gpu]` module or function.
//!
//! # Stability
//!
//! The crate is pre-1.0 and follows SemVer for `0.x`: a breaking change bumps the minor version, a patch release only
//! adds or fixes. The four crates of a release (`gust`, this crate, the wgpu backend and `cargo-gust`) share one
//! version and depend on each other with exact (`=`) pins, because the macros emit items that only the same release of
//! `gust` has. What a minor bump may change: the Slang that `#[gpu]` emits, the diagnostics' wording (their codes
//! are stable, see `docs/DIAGNOSTICS.md`), and the accepted subset of Rust, which may grow.
//!
//! # Support surface
//!
//! `gust_mul_impls!` generates the operator impls of `gust::math` and is invoked only by `gust`. It is
//! `#[doc(hidden)]`; it has no stability promise and a patch release may change it. So have the `__gust_*` names that
//! the expansions emit and the `descriptor;` form of `par_for!` (D103).
#![warn(missing_docs)]

use proc_macro::TokenStream;

mod diagnostics;
mod expand;
mod manifest;
mod math_types;
mod mul_impls;
mod names;
mod par_for;
mod profile;
mod recover;
mod slang;
mod uniform;
mod validate;

use diagnostics::Code;

#[cfg(test)]
#[path = "tests/regression.rs"]
mod regression_tests;

#[cfg(test)]
#[path = "tests/map.rs"]
mod map_tests;

#[cfg(test)]
#[path = "tests/workgroup.rs"]
mod workgroup_tests;

#[cfg(test)]
#[path = "tests/grid.rs"]
mod grid_tests;

#[cfg(test)]
#[path = "tests/par_for.rs"]
mod par_for_tests;

#[cfg(test)]
#[path = "tests/profile.rs"]
mod profile_tests;

#[cfg(test)]
#[path = "tests/manifest.rs"]
mod manifest_tests;

#[cfg(test)]
#[path = "tests/variant.rs"]
mod variant_tests;

#[cfg(test)]
#[path = "tests/subgroup.rs"]
mod subgroup_tests;

#[cfg(test)]
#[path = "tests/routes.rs"]
mod routes_tests;

/// Marks an inline Rust module as GUST GPU code, or a free scalar function as an element-wise map (D60).
///
/// On a module, the macro validates a deliberately small Rust subset, translates each
/// kernel directly to Slang, and generates host-side handles and descriptors. On a
/// function (`#[gpu] pub fn saxpy(a: f32, x: f32, y: f32) -> f32 { .. }`) the function stays
/// as written and the macro adds a module of the same name with a generated grid-stride
/// kernel and the calls `map`, `map_on`, `map_gpu`, `map_gpu_on`, `map_async` and
/// `map_on_async` (under `GUST_PROFILE=cpu`: `map`, `map_on` and `map_on_async`).
#[proc_macro_attribute]
pub fn gpu(attribute: TokenStream, item: TokenStream) -> TokenStream {
    // The one place that reads the build profile (D44, R1-9): everything below takes it as an argument, so tests
    // never set an environment variable. `gust`'s build script validates the value; the macro only asks whether it
    // is exactly `cpu`. The expansion also emits `option_env!("GUST_PROFILE")`, which is what makes rustc track this
    // read: `std::env::var` alone is not tracked.
    let value = std::env::var("GUST_PROFILE").ok();
    let profile = expand::Profile::from_value(value.as_deref());
    gpu_impl_routed(
        attribute.into(),
        item.into(),
        profile,
        &expand::routing::process_variable,
    )
    .into()
}

/// A parallel loop written at the call site (slice m7.p, plan section 3.1):
///
/// ```text
/// gust::par_for!(i in 0..n, [a: &[f32], b: &[f32], out: &mut [f32]] {
///     out[i] = a[i] * b[i];
/// });
/// ```
///
/// The body is the dialect, run once per index of the ranges as a kernel; the bracketed list names the buffers
/// (`&[T]`, `&mut [T]`, `&GpuVec<T>`, `&mut GpuVec<T>`) and scalars (`u32`, `i32`, `f32`, `bool`) it uses, each taken
/// from the variable of that name at the call. The call returns `Result<(), gust::map::Error>`.
///
/// Two optional clauses come first, in this order and each ending in `;`: `workgroup_size(x, y, z);` overrides the
/// workgroup (the defaults are 256 for one variable, 16 x 16 for two and 8 x 8 x 4 for three), and `on(device);` names
/// the device to run on (a `gust_wgpu::HeadlessDevice`, or a `gust::cpu::CpuDevice` under `GUST_PROFILE=cpu`); without
/// it the call opens the default device. The iterations run in no defined order and in parallel, so a body whose
/// iteration depends on another is a bug in the body that nothing diagnoses (docs/SEMANTICS.md).
///
/// ```text
/// gust::par_for!(workgroup_size(8, 8, 1); on(&device); (x, y) in (0..w, 0..h), [out: &mut [u32], w: u32] {
///     out[y * w + x] = y * 1000u32 + x;
/// });
/// ```
#[proc_macro]
pub fn par_for(input: TokenStream) -> TokenStream {
    // The one place that reads the build profile, as for `#[gpu]` (D44, R1-9).
    let value = std::env::var("GUST_PROFILE").ok();
    let profile = expand::Profile::from_value(value.as_deref());
    par_for::par_for(input.into(), profile).into()
}

/// The `Mul` and `MulAssign` impls of `gust::math`, generated from the product table that also drives the Slang
/// overloads `__gust_mul` (slice m6.m2). Only `crates/gust/src/math.rs` invokes it, once and with no argument.
#[doc(hidden)]
#[proc_macro]
pub fn gust_mul_impls(input: TokenStream) -> TokenStream {
    if !input.is_empty() {
        return quote::quote!(compile_error!("gust_mul_impls! takes no arguments");).into();
    }
    mul_impls::mul_impls().into()
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
///
/// No route variable is read: a test that wants one calls [`gpu_impl_routed`] with a table.
#[cfg(test)]
pub(crate) fn gpu_impl_with(
    attribute: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
    profile: expand::Profile,
) -> proc_macro2::TokenStream {
    gpu_impl_routed(attribute, item, profile, &|_| None)
}

/// [`gpu_impl_with`] with the place the route variables are read from (the process for the macro, M8, D66). The one
/// argument `#[gpu]` takes is `route = "<tag>"`; anything else is GUST-E0701.
pub(crate) fn gpu_impl_routed(
    attribute: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
    profile: expand::Profile,
    variables: &dyn Fn(&str) -> Option<String>,
) -> proc_macro2::TokenStream {
    // `syn` words the parse failure itself; the code is added in front.
    let parsed = diagnostics::parse2::<syn::Item>(Code::ItemNotModule, item.clone());
    let tag = if attribute.is_empty() {
        None
    } else {
        match expand::routing::parse_attribute(attribute) {
            Ok(tag) => Some(tag),
            Err(error) => {
                let module = match parsed {
                    Ok(syn::Item::Mod(module)) => Some(without_nested_gpu(module)),
                    _ => None,
                };
                return recover::beside_error(error, &item, module);
            }
        }
    };
    let request = expand::routing::Request { tag, variables };
    match parsed {
        Ok(syn::Item::Mod(module)) => {
            match expand::expand_with_routed(module.clone(), profile, &request) {
                Ok(tokens) => tokens,
                Err(error) => recover::beside_error(error, &item, Some(without_nested_gpu(module))),
            }
        }
        // A free function: the element-wise map calls of D60 (m7.a1). On error the function follows the error as
        // written.
        // A `#[variant]` on it is GUST-E0740: the function is one body. It goes back without the attribute, which
        // would be an unknown attribute, and so a second error, to rustc.
        Ok(syn::Item::Fn(function)) => {
            if let Some(error) = expand::variant::misplaced_on_function(&function) {
                let function = expand::variant::without_variant(function);
                return recover::beside_error(
                    error,
                    &quote::ToTokens::to_token_stream(&function),
                    None,
                );
            }
            match expand::expand_map_routed(function, profile, &request) {
                Ok(tokens) => tokens,
                Err(error) => recover::beside_error(error, &item, None),
            }
        }
        Ok(other) => {
            let error = diagnostics::error(
                Code::ItemNotModule,
                &other,
                "#[gpu] goes on an inline module (`#[gpu] mod name { .. }`) or on a free scalar function \
                 (`#[gpu] fn name(..) -> T { .. }`)",
            );
            recover::beside_error(error, &item, None)
        }
        Err(error) => recover::beside_error(error, &item, None),
    }
}

/// A rejected module goes back beside the error as written, except that a `#[gpu]` function inside it (GUST-E0716)
/// loses the attribute: left on, it would expand by itself in the user's crate and add errors that only follow from
/// the first. A `#[variant]` function goes back as a `#[kernel]` and `#[variant]` anywhere else is dropped (M8, D139):
/// the recovery of `recover.rs` knows the one attribute, and an unknown attribute would be a second error.
fn without_nested_gpu(mut module: syn::ItemMod) -> syn::ItemMod {
    if let Some((_, items)) = &mut module.content {
        for item in items {
            if let syn::Item::Fn(function) = item {
                function
                    .attrs
                    .retain(|attribute| !validate::is_gpu_attribute(attribute));
            }
        }
    }
    expand::variant::recoverable(module)
}

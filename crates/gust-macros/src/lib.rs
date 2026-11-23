//! The procedural macros of GUST: `#[gpu_module]`, `#[gpu_fn]` and `par_for!`.
//!
//! Use them through `gust` (`gust::gpu_module`, `gust::gpu_fn`, `gust::par_for!`), which re-exports them and owns the
//! types they emit; this crate is not meant to be depended on directly. `#[gpu_module]` and `#[gpu_fn]` validate a
//! small Rust subset and translate it directly from the `syn` syntax tree to Slang; every rejection carries a stable
//! diagnostic code (`GUST-E....`, see `docs/DIAGNOSTICS.md`).
//!
//! # Generated items and `missing_docs`
//!
//! The macros emit public items (the handle and descriptor of each kernel, the typed calls, the module of a map
//! function). Each is documented or `#[doc(hidden)]`, so a crate with `#![deny(missing_docs)]`
//! compiles over a `#[gpu_module]` module or `#[gpu_fn]` function.
//!
//! # Stability
//!
//! The crate is pre-1.0 and follows SemVer for `0.x`: a breaking change bumps the minor version, a patch release only
//! adds or fixes. The four crates of a release (`gust`, this crate, the wgpu backend and `cargo-gust`) share one
//! version and depend on each other with exact (`=`) pins, because the macros emit items that only the same release of
//! `gust` has. What a minor bump may change: the Slang that `#[gpu_module]` emits, the diagnostics' wording (their
//! codes are stable, see `docs/DIAGNOSTICS.md`), and the accepted subset of Rust, which may grow.
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
mod forward;
mod manifest;
mod math_types;
mod mul_impls;
mod names;
mod par_for;
mod profile;
mod recover;
mod slang;
mod task;
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

#[cfg(test)]
#[path = "tests/forward.rs"]
mod forward_tests;

#[cfg(test)]
#[path = "tests/consts.rs"]
mod const_tests;

#[cfg(test)]
#[path = "tests/value_if.rs"]
mod value_if_tests;

#[cfg(test)]
#[path = "tests/value_match_untyped.rs"]
mod value_match_untyped_tests;

#[cfg(test)]
#[path = "tests/pad.rs"]
mod pad_tests;

#[cfg(test)]
#[path = "tests/capture.rs"]
mod capture_tests;
#[path = "tests/task.rs"]
mod task_tests;

/// Marks an inline Rust module as GUST GPU code.
///
/// The macro validates a deliberately small Rust subset, translates each kernel directly to Slang, and generates
/// host-side handles and descriptors. A free scalar function takes [`gpu_fn`] instead (D237); `#[gpu_module]` on
/// anything but an inline module is GUST-E0702.
///
/// Two optional arguments, in either order and each at most once (anything else is GUST-E0701): `route = "<tag>"`
/// (build-time routing) and `crate = <path>`. The expansion names the facade as `::gust`, which only a crate with a
/// direct dependency on `gust-compute` named `gust` has; a crate that reaches it through another crate's
/// `pub use gust;` says where with `#[gpu_module(crate = ::inner::gust)]` (the path starts with `::` or `crate::`). A
/// final pass then rewrites each `::gust::` of the output that begins a path to that path: not one after a path
/// segment (as in `::inner::gust::`) or in a qualified path (`<T as X>::gust::`); one after a keyword, `->`, a
/// lifetime or the end of a list of generics (`impl<T> ::gust::Tr`) is rewritten, and so is one after a `>` with no
/// `<` before it in the same group. A `::gust` right after any other `>` is left as written, which includes a
/// comparison when an earlier `<` (an operator too, in this or an earlier statement of the block) is still unmatched,
/// as in `a < b && d > ::gust::K`: rustc then reports E0433 for it, and writing `d > (::gust::K)` avoids that.
/// Spacing is ignored, so `S<T> ::gust::x` and `S<T>::gust::x` are alike. Tokens you wrote yourself inside the module
/// that spell one are rewritten too. Without the argument nothing is rewritten.
#[proc_macro_attribute]
pub fn gpu_module(attribute: TokenStream, item: TokenStream) -> TokenStream {
    gpu_entry(Form::Module, attribute, item)
}

/// Marks a free scalar function as an element-wise map (D60).
///
/// The function stays as written (`#[gpu_fn] pub fn saxpy(a: f32, x: f32, y: f32) -> f32 { .. }`) and the macro adds a
/// module of the same name with a generated grid-stride kernel and the calls `map`, `map_on`, `map_gpu`,
/// `map_gpu_on`, `map_async` and `map_on_async` (under `GUST_PROFILE=cpu`: `map`, `map_on` and `map_on_async`). A
/// module takes [`gpu_module`] instead (D237); `#[gpu_fn]` on anything but a free function is GUST-E0702.
///
/// It takes the same two optional arguments as [`gpu_module`], `route = "<tag>"` and `crate = <path>`, with the same
/// rules (`#[gpu_fn(crate = ::inner::gust)]`). The expansion names `::gust_wgpu` literally, so it needs a direct
/// dependency on `gust-compute-wgpu` (the module form does not).
#[proc_macro_attribute]
pub fn gpu_fn(attribute: TokenStream, item: TokenStream) -> TokenStream {
    gpu_entry(Form::Function, attribute, item)
}

/// The shared body of the two attribute macros.
fn gpu_entry(form: Form, attribute: TokenStream, item: TokenStream) -> TokenStream {
    // The one place that reads the build profile (D44, R1-9): everything below takes it as an argument, so tests
    // never set an environment variable. `gust`'s build script validates the value; the macro only asks whether it
    // is exactly `cpu`. The expansion also emits `option_env!("GUST_PROFILE")`, which is what makes rustc track this
    // read: `std::env::var` alone is not tracked.
    let value = std::env::var("GUST_PROFILE").ok();
    let profile = expand::Profile::from_value(value.as_deref());
    gpu_impl_routed(
        form,
        attribute.into(),
        item.into(),
        profile,
        &expand::routing::process_variable,
    )
    .into()
}

/// Which attribute macro was written: the item it is on has to match (D237).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    /// `#[gpu_module]`: an inline module.
    Module,
    /// `#[gpu_fn]`: a free scalar function, the element-wise map.
    Function,
}

impl Form {
    /// The form that fits `item`, for tests that pass an item without saying which attribute is on it.
    #[cfg(test)]
    fn of(item: &proc_macro2::TokenStream) -> Form {
        match diagnostics::parse2::<syn::Item>(Code::ItemNotModule, item.clone()) {
            Ok(syn::Item::Fn(_)) => Form::Function,
            _ => Form::Module,
        }
    }
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
/// A third clause, `crate(<path>);`, goes with the others (before the loop, each at most
/// once) and names where `gust` is for a crate that reaches it through another crate (`crate(::inner::gust);`, as
/// `#[gpu_module(crate = ..)]` does); the path starts with `::` or `crate::`. It rewrites each `::gust::` of the
/// expansion that begins a path (the rule of `#[gpu_module(crate = ..)]`), in your body too. The expansion names
/// `::gust_wgpu` literally, so a call still needs a direct dependency
/// on `gust-compute-wgpu`. The clause is the grammar's own: `crate` followed by `(` cannot begin a loop pattern.
///
/// ```text
/// gust::par_for!(workgroup_size(8, 8, 1); on(&device); (x, y) in (0..w, 0..h), [out: &mut [u32], w: u32] {
///     out[y * w + x] = y * 1000u32 + x;
/// });
/// ```
#[proc_macro]
pub fn par_for(input: TokenStream) -> TokenStream {
    // The one place that reads the build profile, as for `#[gpu_module]` (D44, R1-9).
    let value = std::env::var("GUST_PROFILE").ok();
    let profile = expand::Profile::from_value(value.as_deref());
    par_for::par_for(input.into(), profile).into()
}

/// A stage written at the call site, run as a free task (plan m14.s2, slice s2a1):
///
/// ```text
/// let doubled = gust::gpu_task!(n, [a: &[f32]], |i| -> f32 { a[i] * 2.0 }).dispatch()?;      // 1-D
/// let grid = gust::gpu_task!((w, h), [k: u32], |x, y| -> u32 { y * 100u32 + x + k }).dispatch()?;  // 2-D, row-major
/// gust::gpu_task!(&mut data, [data: &mut [f32]], |i| { data[i] = data[i] * 0.5; }).dispatch()?;   // in place
/// ```
///
/// The first part is the shape: an extent (anything `TryInto<u32>`) for a 1-D stage, a `(u32, u32)` pair for a 2-D one,
/// or `&mut target` for an in-place stage (1-D, the length of its target). The bracketed list names every capture with
/// its type, each taken from the variable of that name at the call: a host slice `&[T]` (a buffer element type of
/// `par_for!`: `f32`, `i32`, `u32`, `Vec2`, `Vec4`, `IVec2`, `IVec4`, `UVec2`, `UVec4`, `Mat2`, `Mat4`) or a scalar
/// (`u32`, `i32`, `f32`, `bool`); it is left out when there are none. The closure has an explicit result type, the
/// element type of the result buffer (the closure's last expression is written to it at the invocation's own index,
/// row-major for a 2-D stage); an in-place stage has none, and writes its own target (`target[i]`, read only at its own
/// index). Captures are read-only.
///
/// The options of `par_for!` come first, each ending in `;`: `workgroup_size(x, y, z);`, `on(device);` (the device to
/// run on; without it the call opens the default device) and `crate(<path>);`.
///
/// The value is a `gust::task::Task`: `.submit()` starts the work and returns the asynchronous read-back on every
/// target, `.dispatch()` also waits for it (native targets). The body is the dialect, validated and lowered through the
/// `par_for!` generator, so the kernel has the artifacts, records, routing and CPU twin of any kernel. The diagnostics
/// of the call itself are GUST-E1001, E1002, E1003, E1007 and E1008 (`docs/DIAGNOSTICS.md`).
#[proc_macro]
pub fn gpu_task(input: TokenStream) -> TokenStream {
    // The one place that reads the build profile, as for `par_for!`.
    let value = std::env::var("GUST_PROFILE").ok();
    let profile = expand::Profile::from_value(value.as_deref());
    task::gpu_task(input.into(), profile).into()
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

/// [`gpu_entry`] on `proc_macro2` tokens, so it can be tested outside a proc-macro context. The form is the one that
/// fits `item`; a test of the refusal of the wrong attribute calls [`gpu_impl_routed`] with the form it wants.
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
    gpu_impl_routed(Form::of(&item), attribute, item, profile, &|_| None)
}

/// [`gpu_impl_with`] with the place the route variables are read from (the process for the macro, M8, D66). The
/// arguments `#[gpu_module]` and `#[gpu_fn]` take are `route = "<tag>"` and `crate = <path>` (audit S6); anything else
/// is GUST-E0701. With a `crate` the whole output, errors and the user's own code beside them included, goes through
/// [`forward::forward_crate`].
pub(crate) fn gpu_impl_routed(
    form: Form,
    attribute: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
    profile: expand::Profile,
    variables: &dyn Fn(&str) -> Option<String>,
) -> proc_macro2::TokenStream {
    // `syn` words the parse failure itself; the code is added in front.
    let parsed = diagnostics::parse2::<syn::Item>(Code::ItemNotModule, item.clone());
    let (tag, krate) = if attribute.is_empty() {
        (None, None)
    } else {
        match expand::routing::parse_attribute(attribute.clone()) {
            Ok(attribute) => (attribute.tag, attribute.krate),
            Err(error) => {
                let module = match parsed {
                    Ok(syn::Item::Mod(module)) => Some(without_nested_gpu(module)),
                    _ => None,
                };
                // The arguments as a whole are refused, but a well-formed `crate = <path>` among them still says
                // where the facade is: the recovered code (its prelude import and `GpuPod` stand-ins name
                // `::gust::`) is forwarded to it as the expansion would be.
                let recovered = recover::beside_error(error, &item, module);
                return match recover::facade_in_arguments(&attribute) {
                    Some(path) => forward::forward_crate(recovered, &path),
                    None => recovered,
                };
            }
        }
    };
    let request = expand::routing::Request { tag, variables };
    let expansion = expand_item(form, parsed, &item, profile, &request);
    match krate {
        Some(path) => forward::forward_crate(expansion, &path),
        None => expansion,
    }
}

/// The expansion of the item the attribute of `form` is on, before the facade path is forwarded. An item of the other
/// kind is GUST-E0702, and the message names the attribute that fits it (D237).
fn expand_item(
    form: Form,
    parsed: syn::Result<syn::Item>,
    item: &proc_macro2::TokenStream,
    profile: expand::Profile,
    request: &expand::routing::Request<'_>,
) -> proc_macro2::TokenStream {
    match parsed {
        Ok(syn::Item::Mod(module)) if form == Form::Module => {
            match expand::expand_with_routed(module.clone(), profile, request) {
                Ok(tokens) => tokens,
                Err(error) => recover::beside_error(error, item, Some(without_nested_gpu(module))),
            }
        }
        // A free function: the element-wise map calls of D60 (m7.a1). On error the function follows the error as
        // written.
        // A `#[variant]` on it is GUST-E0740: the function is one body. It goes back without the attribute, which
        // would be an unknown attribute, and so a second error, to rustc.
        Ok(syn::Item::Fn(function)) if form == Form::Function => {
            if let Some(error) = expand::variant::misplaced_on_function(&function) {
                let function = expand::variant::without_variant(function);
                return recover::beside_error(
                    error,
                    &quote::ToTokens::to_token_stream(&function),
                    None,
                );
            }
            match expand::expand_map_routed(function, profile, request) {
                Ok(tokens) => tokens,
                Err(error) => recover::beside_error(error, item, None),
            }
        }
        // The other attribute's item: a module under `#[gpu_fn]` goes back as written, with the nested attributes
        // dropped as for any rejected module. A function under `#[gpu_module]` goes back without `#[kernel]` and
        // `#[variant]`: they are unknown attributes outside a module, and so a second error, to rustc.
        Ok(syn::Item::Fn(mut function)) => {
            let error =
                diagnostics::error(Code::ItemNotModule, &function, wrong_item_message(form));
            function.attrs.retain(|attribute| {
                !attribute.path().is_ident("kernel") && !attribute.path().is_ident("variant")
            });
            recover::beside_error(error, &quote::ToTokens::to_token_stream(&function), None)
        }
        Ok(syn::Item::Mod(module)) => {
            let error = diagnostics::error(Code::ItemNotModule, &module, wrong_item_message(form));
            recover::beside_error(error, item, Some(without_nested_gpu(module)))
        }
        Ok(other) => {
            let error = diagnostics::error(Code::ItemNotModule, &other, wrong_item_message(form));
            recover::beside_error(error, item, None)
        }
        Err(error) => recover::beside_error(error, item, None),
    }
}

/// The text of GUST-E0702 for an item the attribute of `form` does not take: what it takes, and which attribute takes
/// the other kind.
fn wrong_item_message(form: Form) -> &'static str {
    match form {
        Form::Module => {
            "`#[gpu_module]` goes on an inline module (`#[gpu_module] mod name { .. }`); a free scalar function takes \
             `#[gpu_fn]` (`#[gpu_fn] fn name(..) -> T { .. }`)"
        }
        Form::Function => {
            "`#[gpu_fn]` goes on a free scalar function (`#[gpu_fn] fn name(..) -> T { .. }`); an inline module takes \
             `#[gpu_module]` (`#[gpu_module] mod name { .. }`)"
        }
    }
}

/// A rejected module goes back beside the error as written, except that a `#[gpu_fn]` function inside it (GUST-E0716)
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
    // A `#[pad]` struct goes back padded, with its `new`, when the module has no misuse of the attribute (m14.pad):
    // host code names both, and a missing one would be a second error. Where the attribute is misused the module goes
    // back as written and `recover.rs` drops the attribute.
    expand::pad::recover(expand::variant::recoverable(module), &profile::GRID)
}

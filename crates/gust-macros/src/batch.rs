//! `gust::gpu_batch!` (plan m14.s2, slice s2a2): a sealed batch of stages, written as one block.
//!
//! ```text
//! let (mut batch, h) = gust::gpu_batch!("exposure", on(&device); in_flight(1); |b| {
//!     let size = b.param::<(u32, u32)>((16, 8));
//!     let pixels = b.input::<Vec4>(&size);
//!     let gain = b.param::<f32>(1.0);
//!     let lum = gust::gpu_task!(&size, |x, y| -> f32 { pixels[(x, y)].x * gain.get() });
//!     Handles { size, pixels, gain, lum }
//! })?;
//! ```
//!
//! The macro reads the block as written: every `let` of `b.input::<T>(..)`, `b.param::<T>(..)` or `gpu_task!(.., |..|
//! -> T { .. })` gives a name its type (a stage's type is its closure's `-> T`, an in-place stage's the target's). It
//! rewrites each inner `gpu_task!` to a stage of the batch: the hidden clause `batch(<builder>, "<name>");` and a
//! capture list with the handles and parameters its body uses, each with the type the block gave it, so the list holds
//! only what the block does not declare. The body is lowered by [`crate::task`] (a 2-D index `h[(x, y)]` over the
//! handle's width, `p.get()` as the scalar); this file does the reading of the block.
//!
//! The diagnostics are GUST-E1005 (a form the macro cannot track or this release does not have: `in_flight` outside 1 to 4,
//! a stage inside a closure or a function, a host slice, a name of a type it cannot see) and GUST-E1006 (a body shape a
//! batch stage does not take), the latter raised by [`crate::task`].

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    Token,
    ext::IdentExt as _,
    parse::{Parse, ParseStream},
    visit::Visit,
    visit_mut::VisitMut,
};

use crate::{
    diagnostics::{self, Code},
    par_for::RawBinding,
    task::{batch_form, type_argument},
};

/// The macro input as written.
struct Input {
    name: syn::LitStr,
    device: Option<syn::Expr>,
    krate: Option<syn::Path>,
    in_flight: u32,
    builder: syn::Ident,
    stmts: Vec<syn::Stmt>,
}

/// The most frames in flight a batch takes: `in_flight(4)`.
const MAX_IN_FLIGHT: u32 = 4;

impl Parse for Input {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: syn::LitStr = input.parse()?;
        input.parse::<Token![,]>()?;
        let (mut device, mut krate, mut in_flight) = (None, None, None);
        while let Some((keyword, _)) = input.cursor().ident() {
            if !input.peek2(syn::token::Paren) {
                break;
            }
            match keyword.to_string().as_str() {
                "on" => {
                    input.call(syn::Ident::parse_any)?;
                    let content;
                    syn::parenthesized!(content in input);
                    if device.is_some() {
                        return Err(batch_form(&keyword, "`on(..)` given twice"));
                    }
                    device = Some(content.parse::<syn::Expr>()?);
                    input.parse::<Token![;]>()?;
                }
                "in_flight" => {
                    input.call(syn::Ident::parse_any)?;
                    let content;
                    syn::parenthesized!(content in input);
                    let count = content.parse::<syn::LitInt>().map_err(|_| {
                        batch_form(&keyword, "`in_flight(..)` takes a number: `in_flight(1)`")
                    })?;
                    input.parse::<Token![;]>()?;
                    if in_flight.is_some() {
                        return Err(batch_form(&keyword, "`in_flight(..)` given twice"));
                    }
                    match count.base10_parse::<u32>() {
                        Ok(count) if (1..=MAX_IN_FLIGHT).contains(&count) => {
                            in_flight = Some(count);
                        }
                        _ => {
                            return Err(batch_form(
                                &count,
                                format!(
                                    "`in_flight(..)` is from 1 to {MAX_IN_FLIGHT} frames: `in_flight(2)` keeps \
                                     two in flight"
                                ),
                            ));
                        }
                    }
                }
                "crate" => {
                    input.call(syn::Ident::parse_any)?;
                    let content;
                    syn::parenthesized!(content in input);
                    krate = Some(content.parse::<syn::Path>()?);
                    input.parse::<Token![;]>()?;
                }
                _ => break,
            }
        }
        let closure = match input.parse::<syn::Expr>()? {
            syn::Expr::Closure(closure) => closure,
            other => {
                return Err(batch_form(
                    other,
                    "the last part of a `gpu_batch!` is its build block: `|b| { .. }`",
                ));
            }
        };
        let _ = input.parse::<Option<Token![,]>>()?;
        if !input.is_empty() {
            return Err(batch_form(
                input.cursor().token_stream(),
                "`gpu_batch!` ends with its build block",
            ));
        }
        if closure.inputs.len() != 1 {
            return Err(batch_form(
                &closure.inputs,
                "the build block takes the builder: `|b| { .. }`",
            ));
        }
        let builder = match &closure.inputs[0] {
            syn::Pat::Ident(ident) if ident.subpat.is_none() => ident.ident.clone(),
            syn::Pat::Type(typed) => match &*typed.pat {
                syn::Pat::Ident(ident) if ident.subpat.is_none() => ident.ident.clone(),
                other => {
                    return Err(batch_form(
                        other,
                        "the builder is a plain name: `|b| { .. }`",
                    ));
                }
            },
            other => {
                return Err(batch_form(
                    other,
                    "the builder is a plain name: `|b| { .. }`",
                ));
            }
        };
        let stmts = match *closure.body {
            syn::Expr::Block(block) if block.attrs.is_empty() && block.label.is_none() => {
                block.block.stmts
            }
            other => vec![syn::Stmt::Expr(other, None)],
        };
        Ok(Self {
            name,
            device,
            krate,
            in_flight: in_flight.unwrap_or(1),
            builder,
            stmts,
        })
    }
}

/// What a name of the block is, as far as the macro can tell.
enum Kind {
    /// A buffer handle with this element type (`input`, or a stage's result).
    Handle(syn::Type),
    /// A parameter with this type.
    Param(syn::Type),
    /// Anything else.
    Opaque,
}

/// What a `let` of the block is initialised with, as far as the rewrite is concerned.
enum Init {
    Task,
    Builder,
    Other,
    None,
}

/// The inner `gpu_task!` as written.
struct StageInput {
    /// The keyword options, as written (each with its `;`).
    options: TokenStream,
    krate: bool,
    shape: syn::Expr,
    list: Vec<RawBinding>,
    closure: syn::ExprClosure,
}

impl Parse for StageInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut options = TokenStream::new();
        let mut krate = false;
        while let Some((keyword, _)) = input.cursor().ident() {
            if !input.peek2(syn::token::Paren) {
                break;
            }
            let text = keyword.to_string();
            if !matches!(
                text.as_str(),
                "workgroup_size" | "on" | "crate" | "descriptor"
            ) {
                break;
            }
            let ident = input.call(syn::Ident::parse_any)?;
            let group: proc_macro2::Group = input.parse()?;
            let semicolon: Token![;] = input.parse()?;
            krate |= text == "crate";
            options.extend(quote!(#ident #group #semicolon));
        }
        let shape: syn::Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let mut list = Vec::new();
        if input.peek(syn::token::Bracket) {
            let content;
            syn::bracketed!(content in input);
            list = content
                .parse_terminated(RawBinding::parse, Token![,])?
                .into_iter()
                .collect();
            input.parse::<Token![,]>()?;
        }
        let closure = match input.parse::<syn::Expr>()? {
            syn::Expr::Closure(closure) => closure,
            other => {
                return Err(diagnostics::error(
                    Code::TaskForm,
                    other,
                    "the last part of a `gpu_task!` is its closure: `|i| -> T { .. }`, `|x, y| -> T { .. }`, or \
                     `|i| { .. }` for an in-place stage",
                ));
            }
        };
        let _ = input.parse::<Option<Token![,]>>()?;
        if !input.is_empty() {
            return Err(diagnostics::error(
                Code::TaskForm,
                input.cursor().token_stream(),
                "`gpu_task!` ends with its closure",
            ));
        }
        Ok(Self {
            options,
            krate,
            shape,
            list,
            closure,
        })
    }
}

/// The names a body mentions as a plain path (a handle, a parameter, a scalar, a closure parameter).
#[derive(Default)]
struct Used {
    names: Vec<syn::Ident>,
}

impl<'ast> Visit<'ast> for Used {
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none()
            && let Some(ident) = path.path.get_ident()
            && !self.names.contains(ident)
        {
            self.names.push(ident.clone());
        }
        syn::visit::visit_expr_path(self, path);
    }
}

fn names_in(expression: &syn::Expr) -> Vec<syn::Ident> {
    let mut used = Used::default();
    used.visit_expr(expression);
    used.names
}

/// The block's `let`s and the stages, rewritten in order.
struct Rewriter {
    builder: syn::Ident,
    krate: Option<syn::Path>,
    /// Every `let` of the block so far; a later one of the same name hides an earlier.
    tracked: Vec<(syn::Ident, Kind)>,
    /// The `async` blocks and loops the walk is inside: what they hold is not recorded once per batch.
    unrecorded: Vec<&'static str>,
    /// Closures and functions the walk is inside.
    depth: usize,
    error: Option<syn::Error>,
}

fn is_task(path: &syn::Path) -> bool {
    path.segments
        .last()
        .is_some_and(|segment| segment.ident == "gpu_task")
}

fn simple_name(pat: &syn::Pat) -> Option<(syn::Ident, Option<syn::Type>)> {
    match pat {
        syn::Pat::Ident(ident) if ident.subpat.is_none() => Some((ident.ident.clone(), None)),
        syn::Pat::Type(typed) => {
            let (ident, _) = simple_name(&typed.pat)?;
            Some((ident, Some((*typed.ty).clone())))
        }
        _ => None,
    }
}

impl Rewriter {
    fn fail(&mut self, error: syn::Error) {
        match &mut self.error {
            Some(first) => first.combine(error),
            None => self.error = Some(error),
        }
    }

    fn lookup(&self, name: &syn::Ident) -> Option<&Kind> {
        self.tracked
            .iter()
            .rev()
            .find(|(tracked, _)| tracked == name)
            .map(|(_, kind)| kind)
    }

    fn is_builder_call(&self, call: &syn::ExprMethodCall, method: &str) -> bool {
        call.method == method
            && matches!(&*call.receiver, syn::Expr::Path(path) if path.path.is_ident(&self.builder))
    }

    /// `b.input::<T>(shape)` is `b.input::<T, _>(shape)`: the shape's type is the rest of the call's generics.
    fn complete_input(&self, call: &mut syn::ExprMethodCall) {
        if !self.is_builder_call(call, "input") {
            return;
        }
        if let Some(turbofish) = &mut call.turbofish
            && turbofish.args.len() == 1
        {
            turbofish
                .args
                .push(syn::GenericArgument::Type(syn::parse_quote!(_)));
        }
    }

    /// Rewrite an inner `gpu_task!`; the element type of its result when known.
    fn stage(&mut self, mac: &mut syn::Macro, name: Option<&str>) -> Option<syn::Type> {
        if self.depth > 0 {
            self.fail(batch_form(
                &mac.path,
                "a stage of a batch is written directly in the build block: this `gpu_task!` is inside a closure or a \
                 function, where the macro cannot tell which handles it captures",
            ));
            return None;
        }
        if let Some(what) = self.unrecorded.last() {
            self.fail(diagnostics::error(
                Code::BatchBody,
                &mac.path,
                format!(
                    "a stage of a batch is recorded once, when the block runs: this `gpu_task!` is inside {what}, where it \
                     would be recorded never or once per turn. Write each stage directly in the build block"
                ),
            ));
            return None;
        }
        let parsed = match diagnostics::parse2::<StageInput>(Code::TaskForm, mac.tokens.clone()) {
            Ok(parsed) => parsed,
            Err(error) => {
                self.fail(error);
                return None;
            }
        };
        let StageInput {
            options,
            krate,
            shape,
            list,
            closure,
        } = parsed;
        let parameters: Vec<syn::Ident> = closure
            .inputs
            .iter()
            .filter_map(|pat| simple_name(pat).map(|(ident, _)| ident))
            .collect();

        // The in-place target, when the block declares it.
        let target = match &shape {
            syn::Expr::Reference(reference) if reference.mutability.is_some() => {
                match &*reference.expr {
                    syn::Expr::Path(path) => path.path.get_ident().cloned(),
                    _ => None,
                }
            }
            _ => None,
        };
        let declared_target = target
            .as_ref()
            .filter(|name| matches!(self.lookup(name), Some(Kind::Handle(_))))
            .cloned();

        let mut entries: Vec<TokenStream> = Vec::new();
        let written: Vec<syn::Ident> = list.iter().map(|entry| entry.name.clone()).collect();
        for entry in &list {
            if declared_target.as_ref() == Some(&entry.name) {
                continue;
            }
            let (name, ty) = (&entry.name, &entry.ty);
            entries.push(quote!(#name: #ty));
        }
        if let Some(name) = &declared_target {
            let Some(Kind::Handle(element)) = self.lookup(name) else {
                unreachable!("the target was found as a handle")
            };
            entries.push(quote!(#name: &mut Stage<#element>));
        }
        let mut failures = Vec::new();
        for used in names_in(&closure.body) {
            if parameters.contains(&used)
                || written.contains(&used)
                || declared_target.as_ref() == Some(&used)
            {
                continue;
            }
            match self.lookup(&used) {
                Some(Kind::Handle(element)) => entries.push(quote!(#used: Stage<#element>)),
                Some(Kind::Param(element)) => entries.push(quote!(#used: Param<#element>)),
                Some(Kind::Opaque) => failures.push(batch_form(
                    &used,
                    format!(
                        "`{}` is declared in the block with a type the macro cannot see: a stage captures the \
                         handles and parameters of the block (`b.input::<T>(..)`, `b.param::<T>(..)`, a stage's \
                         `-> T`), and a scalar written in the capture list",
                        used.unraw()
                    ),
                )),
                None => {}
            }
        }
        for error in failures {
            self.fail(error);
        }

        let builder = &self.builder;
        let label = name.unwrap_or("");
        let krate_option = match (&self.krate, krate) {
            (Some(path), false) => quote!(crate(#path);),
            _ => TokenStream::new(),
        };
        let bracket = if entries.is_empty() {
            TokenStream::new()
        } else {
            quote!([#(#entries),*],)
        };
        mac.tokens = quote! {
            batch(#builder, #label); #options #krate_option #shape, #bracket #closure
        };

        match &closure.output {
            syn::ReturnType::Type(_, ty) => Some((**ty).clone()),
            syn::ReturnType::Default => match self.lookup(declared_target.as_ref()?) {
                Some(Kind::Handle(element)) => Some(element.clone()),
                _ => None,
            },
        }
    }
}

impl VisitMut for Rewriter {
    fn visit_local_mut(&mut self, local: &mut syn::Local) {
        let named = simple_name(&local.pat);
        let annotation = named.as_ref().and_then(|(_, ty)| ty.clone());
        let label = named.as_ref().map(|(ident, _)| ident.unraw().to_string());
        let mut kind = Kind::Opaque;
        let mode = match local.init.as_ref().map(|init| &*init.expr) {
            Some(syn::Expr::Macro(mac)) if is_task(&mac.mac.path) => Init::Task,
            Some(syn::Expr::MethodCall(call))
                if self.is_builder_call(call, "input") || self.is_builder_call(call, "param") =>
            {
                Init::Builder
            }
            Some(_) => Init::Other,
            None => Init::None,
        };
        match mode {
            Init::Task => {
                let Some(syn::Expr::Macro(mac)) = local.init.as_mut().map(|init| &mut *init.expr)
                else {
                    unreachable!("the init is a macro call")
                };
                if let Some(element) = self.stage(&mut mac.mac, label.as_deref()) {
                    kind = Kind::Handle(element);
                }
            }
            Init::Builder => {
                let Some(syn::Expr::MethodCall(call)) =
                    local.init.as_mut().map(|init| &mut *init.expr)
                else {
                    unreachable!("the init is a method call")
                };
                let is_input = call.method == "input";
                self.complete_input(call);
                let turbofish = call
                    .turbofish
                    .as_ref()
                    .and_then(|arguments| arguments.args.first())
                    .and_then(|argument| match argument {
                        syn::GenericArgument::Type(ty) => Some(ty.clone()),
                        _ => None,
                    });
                let element = turbofish.or_else(|| {
                    annotation.as_ref().and_then(|ty| {
                        type_argument(ty, if is_input { "Stage" } else { "Param" }).cloned()
                    })
                });
                if let Some(element) = element {
                    kind = if is_input {
                        Kind::Handle(element)
                    } else {
                        Kind::Param(element)
                    };
                }
                // The arguments are the block's own code; the call is named by its `let`.
                for argument in &mut call.args {
                    self.visit_expr_mut(argument);
                }
                if let Some(label) = &label {
                    let builder = &self.builder;
                    let init = local.init.as_mut().expect("an init");
                    let original = (*init.expr).clone();
                    *init.expr = syn::parse_quote!({
                        #builder.next_name(#label);
                        #original
                    });
                }
            }
            Init::Other => syn::visit_mut::visit_local_mut(self, local),
            Init::None => {}
        }
        if let Some((ident, _)) = named {
            self.tracked.push((ident, kind));
        } else {
            // A pattern the macro does not read: every name in it is a block-declared name of a type it cannot see.
            let mut bound = Vec::new();
            collect_bindings(&local.pat, &mut bound);
            for ident in bound {
                self.tracked.push((ident, Kind::Opaque));
            }
        }
    }

    fn visit_macro_mut(&mut self, mac: &mut syn::Macro) {
        if is_task(&mac.path) {
            let _ = self.stage(mac, None);
        }
    }

    fn visit_expr_method_call_mut(&mut self, call: &mut syn::ExprMethodCall) {
        self.complete_input(call);
        syn::visit_mut::visit_expr_method_call_mut(self, call);
    }

    fn visit_expr_closure_mut(&mut self, closure: &mut syn::ExprClosure) {
        self.depth += 1;
        syn::visit_mut::visit_expr_closure_mut(self, closure);
        self.depth -= 1;
    }

    fn visit_item_fn_mut(&mut self, function: &mut syn::ItemFn) {
        self.depth += 1;
        syn::visit_mut::visit_item_fn_mut(self, function);
        self.depth -= 1;
    }

    fn visit_expr_async_mut(&mut self, block: &mut syn::ExprAsync) {
        self.unrecorded.push("an `async` block");
        syn::visit_mut::visit_expr_async_mut(self, block);
        self.unrecorded.pop();
    }

    fn visit_expr_for_loop_mut(&mut self, expr: &mut syn::ExprForLoop) {
        self.unrecorded.push("a loop");
        syn::visit_mut::visit_expr_for_loop_mut(self, expr);
        self.unrecorded.pop();
    }

    fn visit_expr_while_mut(&mut self, expr: &mut syn::ExprWhile) {
        self.unrecorded.push("a loop");
        syn::visit_mut::visit_expr_while_mut(self, expr);
        self.unrecorded.pop();
    }

    fn visit_expr_loop_mut(&mut self, expr: &mut syn::ExprLoop) {
        self.unrecorded.push("a loop");
        syn::visit_mut::visit_expr_loop_mut(self, expr);
        self.unrecorded.pop();
    }

    fn visit_impl_item_fn_mut(&mut self, function: &mut syn::ImplItemFn) {
        self.depth += 1;
        syn::visit_mut::visit_impl_item_fn_mut(self, function);
        self.depth -= 1;
    }
}

fn collect_bindings(pat: &syn::Pat, out: &mut Vec<syn::Ident>) {
    match pat {
        syn::Pat::Ident(ident) => out.push(ident.ident.clone()),
        syn::Pat::Tuple(tuple) => tuple
            .elems
            .iter()
            .for_each(|pat| collect_bindings(pat, out)),
        syn::Pat::Type(typed) => collect_bindings(&typed.pat, out),
        syn::Pat::Reference(reference) => collect_bindings(&reference.pat, out),
        syn::Pat::Paren(paren) => collect_bindings(&paren.pat, out),
        syn::Pat::Slice(slice) => slice
            .elems
            .iter()
            .for_each(|pat| collect_bindings(pat, out)),
        syn::Pat::TupleStruct(tuple) => tuple
            .elems
            .iter()
            .for_each(|pat| collect_bindings(pat, out)),
        syn::Pat::Struct(structure) => structure
            .fields
            .iter()
            .for_each(|field| collect_bindings(&field.pat, out)),
        _ => {}
    }
}

/// `gpu_batch!` for the process.
pub(crate) fn gpu_batch(input: TokenStream) -> TokenStream {
    let mut krate = None;
    let mut device = None;
    match expand(input, &mut krate, &mut device) {
        Ok(tokens) => match krate {
            Some(path) => crate::forward::forward_crate(tokens, &path),
            None => tokens,
        },
        Err(error) => {
            // The errors, then a result of the shape the call has, so that one mistake is one error. With the device
            // read, its type is pinned too, so that `batch.begin()` after the call adds no "type annotations needed".
            let errors = error.to_compile_error();
            let tokens = match device {
                Some(device) => quote!({
                    #errors
                    ::gust::batch::failed(&(#device))
                }),
                None => quote!({
                    #errors
                    let __gust_result: ::core::result::Result<
                        (::gust::batch::Batch<_>, _),
                        ::gust::batch::Error,
                    > = ::core::unreachable!();
                    __gust_result
                }),
            };
            match krate {
                Some(path) => crate::forward::forward_crate(tokens, &path),
                None => tokens,
            }
        }
    }
}

fn expand(
    input: TokenStream,
    krate: &mut Option<syn::Path>,
    device_seen: &mut Option<TokenStream>,
) -> syn::Result<TokenStream> {
    let parsed = diagnostics::parse2::<Input>(Code::BatchForm, input)?;
    let Input {
        name,
        device,
        krate: written,
        in_flight,
        builder,
        mut stmts,
    } = parsed;
    krate.clone_from(&written);
    let Some(device) = device else {
        return Err(diagnostics::error_at(
            Code::BatchForm,
            Span::call_site(),
            "a batch is built for one device: `gpu_batch!(\"name\", on(&device); |b| { .. })`",
        ));
    };
    *device_seen = Some(quote!(#device));

    // The tail is the block's value: the handles the frames use.
    let mut tail = match stmts.last() {
        Some(syn::Stmt::Expr(_, None)) => match stmts.pop() {
            Some(syn::Stmt::Expr(expression, None)) => Some(expression),
            _ => unreachable!("the last statement was an expression"),
        },
        // A macro call that ends the block parses as a statement; without its `;` it is the block's value.
        Some(syn::Stmt::Macro(call)) if call.semi_token.is_none() => match stmts.pop() {
            Some(syn::Stmt::Macro(call)) => Some(syn::Expr::Macro(syn::ExprMacro {
                attrs: call.attrs,
                mac: call.mac,
            })),
            _ => unreachable!("the last statement was a macro call"),
        },
        _ => None,
    };

    let mut rewriter = Rewriter {
        builder: builder.clone(),
        krate: written,
        tracked: Vec::new(),
        unrecorded: Vec::new(),
        depth: 0,
        error: None,
    };
    for statement in &mut stmts {
        rewriter.visit_stmt_mut(statement);
    }
    if let Some(tail) = &mut tail {
        rewriter.visit_expr_mut(tail);
    }
    if let Some(error) = rewriter.error.take() {
        return Err(error);
    }
    let exposed: Vec<syn::Ident> = match &tail {
        Some(tail) => names_in(tail)
            .into_iter()
            .filter(|ident| matches!(rewriter.lookup(ident), Some(Kind::Handle(_))))
            .collect(),
        None => Vec::new(),
    };
    let tail = match tail {
        Some(tail) => quote!(#tail),
        None => TokenStream::new(),
    };
    Ok(quote! {
        ::gust::batch::build(
            #name,
            &(#device),
            #in_flight,
            |#builder: &::gust::batch::Builder<_>| {
                #(#stmts)*
                #(#builder.expose(&#exposed);)*
                #tail
            },
        )
    })
}

//! Closure bindings, the hoisted helper functions they become, and the scopes that name them.

use syn::{ext::IdentExt as _, visit::Visit};

/// `let f = |..| ..;` (slice m9.c): the binding and its closure. The pattern may carry `mut` or a type, which the
/// validator refuses (GUST-E0330); a `let` with another pattern or with an `else` is no closure binding.
pub(crate) fn closure_binding(local: &syn::Local) -> Option<(&syn::PatIdent, &syn::ExprClosure)> {
    let pattern = match &local.pat {
        syn::Pat::Ident(pattern) => pattern,
        syn::Pat::Type(typed) => match typed.pat.as_ref() {
            syn::Pat::Ident(pattern) => pattern,
            _ => return None,
        },
        _ => return None,
    };
    let initial = local.init.as_ref()?;
    if initial.diverge.is_some() {
        return None;
    }
    match initial.expr.as_ref() {
        syn::Expr::Closure(closure) => Some((pattern, closure)),
        _ => None,
    }
}

/// A closure binding of a function body: the `let`, its pattern and its closure.
pub(crate) type ClosureSite<'a> = (&'a syn::Local, &'a syn::PatIdent, &'a syn::ExprClosure);

/// The closure bindings of `block` in source order, without looking inside a closure (a nested closure is
/// GUST-E0332, and a closure body is validated and emitted as a function of its own).
pub(crate) fn closure_sites(block: &syn::Block) -> Vec<ClosureSite<'_>> {
    struct Sites<'a>(Vec<ClosureSite<'a>>);
    impl<'ast> Visit<'ast> for Sites<'ast> {
        fn visit_local(&mut self, local: &'ast syn::Local) {
            if let Some((pattern, closure)) = closure_binding(local) {
                self.0.push((local, pattern, closure));
                return;
            }
            syn::visit::visit_local(self, local);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut sites = Sites(Vec::new());
    sites.visit_block(block);
    sites.0
}

/// The closure as a plain function called `name`, when it has the accepted shape (every parameter `name: Type`, an
/// explicit result type, a block body): `fn name(x: f32) -> f32 { .. }` with the tokens, and so the spans, of the
/// closure. The validator judges the body as the body of a helper, and the emitter hoists it as one (slice m9.c).
pub(crate) fn closure_function(
    name: &syn::Ident,
    closure: &syn::ExprClosure,
) -> Option<syn::ItemFn> {
    let mut inputs = syn::punctuated::Punctuated::<syn::FnArg, syn::Token![,]>::new();
    for input in &closure.inputs {
        let syn::Pat::Type(typed) = input else {
            return None;
        };
        if !matches!(typed.pat.as_ref(), syn::Pat::Ident(_)) {
            return None;
        }
        inputs.push(syn::FnArg::Typed(typed.clone()));
    }
    let output = match &closure.output {
        output @ syn::ReturnType::Type(..) => output.clone(),
        syn::ReturnType::Default => return None,
    };
    let syn::Expr::Block(body) = closure.body.as_ref() else {
        return None;
    };
    let block = &body.block;
    Some(syn::parse_quote! { fn #name(#inputs) #output #block })
}

thread_local! {
    /// The Slang name of every closure of the module being emitted, by the address of its `syn::ExprClosure`:
    /// `__gust_cl_<owner>_<n>`, `n` counting the closures of the module in source order (slice m9.c). The number is
    /// unique, so a helper `S_m` and a method `S::m` cannot give two closures one name.
    pub(super) static CLOSURE_NAMES: std::cell::RefCell<std::collections::HashMap<usize, String>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The closures callable from the function being emitted: the name of the binding and the Slang name.
    static CLOSURE_SCOPE: std::cell::RefCell<std::collections::HashMap<String, String>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The address that identifies a closure in [`CLOSURE_NAMES`].
pub(super) fn closure_key(closure: &syn::ExprClosure) -> usize {
    std::ptr::from_ref(closure) as usize
}

/// The Slang names of the closures of `items`, numbered in source order over the functions, then the methods of each
/// `impl` block as they come (the order of the items).
pub(super) fn closure_names(items: &[syn::Item]) -> std::collections::HashMap<usize, String> {
    let mut names = std::collections::HashMap::new();
    let mut number = 0_usize;
    let mut take = |owner: &str, block: &syn::Block| {
        for (_, _, closure) in closure_sites(block) {
            names.insert(closure_key(closure), format!("__gust_cl_{owner}_{number}"));
            number += 1;
        }
    };
    for item in items {
        match item {
            syn::Item::Fn(function) => {
                take(&function.sig.ident.unraw().to_string(), &function.block)
            }
            syn::Item::Impl(block) => {
                let owner = crate::validate::impl_self_name(block).unwrap_or_default();
                for inner in &block.items {
                    if let syn::ImplItem::Fn(method) = inner {
                        take(
                            &format!("{owner}_{}", method.sig.ident.unraw()),
                            &method.block,
                        );
                    }
                }
            }
            _ => {}
        }
    }
    names
}

/// Sets [`CLOSURE_NAMES`] for a scope and restores the previous map when dropped.
pub(super) struct ClosureNamesScope(std::collections::HashMap<usize, String>);

impl ClosureNamesScope {
    pub(super) fn enter(next: std::collections::HashMap<usize, String>) -> Self {
        Self(CLOSURE_NAMES.with(|cell| cell.replace(next)))
    }
}

impl Drop for ClosureNamesScope {
    fn drop(&mut self) {
        CLOSURE_NAMES.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Makes the closures bound in `block` callable by their binding names, for a scope.
pub(super) struct ClosureScope(std::collections::HashMap<String, String>);

impl ClosureScope {
    pub(super) fn enter(block: &syn::Block) -> Self {
        let scope = CLOSURE_NAMES.with(|names| {
            let names = names.borrow();
            closure_sites(block)
                .into_iter()
                .filter_map(|(_, pattern, closure)| {
                    Some((
                        pattern.ident.unraw().to_string(),
                        names.get(&closure_key(closure))?.clone(),
                    ))
                })
                .collect::<std::collections::HashMap<_, _>>()
        });
        Self(CLOSURE_SCOPE.with(|cell| cell.replace(scope)))
    }
}

impl Drop for ClosureScope {
    fn drop(&mut self) {
        CLOSURE_SCOPE.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// The Slang name `call` calls, when it calls a closure of the function being emitted.
pub(super) fn closure_call_target(call: &syn::ExprCall) -> Option<String> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let name = path.path.get_ident()?.unraw().to_string();
    CLOSURE_SCOPE.with(|cell| cell.borrow().get(&name).cloned())
}

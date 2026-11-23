//! Closures: capture scans, closure checks and the atomic-buffer scan of a function.

use super::*;

/// The identifiers a closure body uses that nothing inside it binds (slice m9.c): its captures, in first-use order.
///
/// A walk with a stack of scopes. The parameters bind in the outermost one; a `let` binds after its initializer, a
/// `for` variable, an `if let` and a `match` arm bind inside their body, and a block ends its scope. A single identifier
/// in expression position (a callee, a receiver, a field shorthand `P { x }`) that is not bound is free, and free is a
/// capture, except as the callee of a call, where a helper of the module and a name of the dialect (`Some`, `Ok`, `Err`, the
/// barrier, the subgroup and atomic operations, judged by their own rules) are not captures, and `None` as a value. A
/// local that shadows a helper name is a capture when it is used as a value. A module holds functions, structs, enums and `impl`s only, so
/// every other free identifier names a local, a parameter, `self` or a closure of the surrounding function. Paths of two
/// segments (`Vec2::new`, `E::A`, `S::f`, `Mat4::IDENTITY`) and the paths of struct literals and types are items.
pub(super) struct CaptureScan<'a> {
    pub(super) scopes: Vec<HashSet<String>>,
    pub(super) helpers: &'a HashSet<String>,
    pub(super) free: Vec<String>,
}

impl CaptureScan<'_> {
    pub(super) fn bind(&mut self, pattern: &syn::Pat) {
        struct Names(Vec<String>);
        impl<'p> Visit<'p> for Names {
            fn visit_pat_ident(&mut self, pattern: &'p syn::PatIdent) {
                self.0.push(pattern.ident.unraw().to_string());
                visit::visit_pat_ident(self, pattern);
            }
        }
        let mut names = Names(Vec::new());
        names.visit_pat(pattern);
        if let Some(scope) = self.scopes.last_mut() {
            scope.extend(names.0);
        }
    }

    /// A use of the single identifier `name`: a callee when it is the function of a call. A helper and a name of the dialect
    /// are exempt only as callees (a local may shadow a helper name, and then the value use is a capture); `None` is the one
    /// dialect name that is a value, and no `let` can bind it.
    pub(super) fn mention(&mut self, name: String, callee: bool) {
        let exempt = if callee {
            is_dialect_callee(&name)
                || self.helpers.contains(&name)
                || self.helpers.contains(&format!("r#{name}"))
        } else {
            name == "None"
        } || self.helpers.contains(&format!("const:{name}")); // a module `const` is global, not a capture (m11.g3)
        if exempt
            || self.scopes.iter().any(|scope| scope.contains(&name))
            || self.free.contains(&name)
        {
            return;
        }
        self.free.push(name);
    }
}

impl<'ast> Visit<'ast> for CaptureScan<'_> {
    fn visit_type(&mut self, _: &'ast syn::Type) {}

    fn visit_expr_macro(&mut self, _: &'ast syn::ExprMacro) {}

    fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}

    fn visit_block(&mut self, block: &'ast syn::Block) {
        self.scopes.push(HashSet::new());
        visit::visit_block(self, block);
        self.scopes.pop();
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(initial) = &local.init {
            self.visit_expr(&initial.expr);
            if let Some((_, diverge)) = &initial.diverge {
                self.visit_expr(diverge);
            }
        }
        self.bind(&local.pat);
    }

    fn visit_expr_for_loop(&mut self, expression: &'ast syn::ExprForLoop) {
        self.visit_expr(&expression.expr);
        self.scopes.push(HashSet::new());
        self.bind(&expression.pat);
        self.visit_block(&expression.body);
        self.scopes.pop();
    }

    fn visit_expr_if(&mut self, expression: &'ast syn::ExprIf) {
        let syn::Expr::Let(binding) = expression.cond.as_ref() else {
            visit::visit_expr_if(self, expression);
            return;
        };
        self.visit_expr(&binding.expr);
        self.scopes.push(HashSet::new());
        self.bind(&binding.pat);
        self.visit_block(&expression.then_branch);
        self.scopes.pop();
        if let Some((_, alternative)) = &expression.else_branch {
            self.visit_expr(alternative);
        }
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        self.visit_expr(&expression.expr);
        for arm in &expression.arms {
            self.scopes.push(HashSet::new());
            self.bind(&arm.pat);
            if let Some((_, guard)) = &arm.guard {
                self.visit_expr(guard);
            }
            self.visit_expr(&arm.body);
            self.scopes.pop();
        }
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        match call.func.as_ref() {
            syn::Expr::Path(path) if path.qself.is_none() => {
                if let Some(name) = path.path.get_ident() {
                    self.mention(name.unraw().to_string(), true);
                }
            }
            other => self.visit_expr(other),
        }
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none()
            && let Some(name) = path.path.get_ident()
        {
            self.mention(name.unraw().to_string(), false);
        }
    }
}

/// The captures of `closure`: the identifiers its body uses that it does not bind (see [`CaptureScan`]).
pub(super) fn closure_captures(
    closure: &syn::ExprClosure,
    helpers: &HashSet<String>,
) -> Vec<String> {
    let mut scan = CaptureScan {
        scopes: vec![HashSet::new()],
        helpers,
        free: Vec::new(),
    };
    for input in &closure.inputs {
        scan.bind(input);
    }
    scan.visit_expr(&closure.body);
    scan.free
}

/// The closures bound in the body of `function` (slice m9.c). Each is judged on its own and reported once, at its `let`:
/// the form (GUST-E0314, `move` E0329), the binding (E0330), the signature (E0331), a closure inside it (E0332), the
/// name (E0901, E0328), the captures (E0329); then the body is validated as the body of a helper whose parameters are
/// the closure's (D154: the helper rule covers a closure body, so no closure holds a barrier, a subgroup operation,
/// workgroup memory, an atomic or a resource, and the call graph treats its calls as calls of the function that holds it).
pub(super) fn validate_closures(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
    function_names: &HashSet<String>,
    result_helpers: &HashSet<String>,
    methods: &MethodTable,
    owner: Option<&Owner<'_>>,
) -> syn::Result<()> {
    let sites = crate::slang::closure_sites(&function.block);
    if sites.is_empty() {
        return Ok(());
    }
    // How many times each name is bound in the function (parameters, locals, loop variables, patterns, the bindings and
    // the parameters of the closures): a closure name that is bound twice shadows or is shadowed, and the validator
    // would have to track which meaning a call has.
    struct Bindings(HashMap<String, usize>);
    impl<'ast> Visit<'ast> for Bindings {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            *self.0.entry(pattern.ident.unraw().to_string()).or_default() += 1;
            visit::visit_pat_ident(self, pattern);
        }
    }
    let mut bindings = Bindings(HashMap::new());
    bindings.visit_item_fn(function);
    let closure_names: HashSet<String> = sites
        .iter()
        .map(|(_, pattern, _)| pattern.ident.unraw().to_string())
        .collect();
    // The owner of a closure body has no fields to clash with: the hoisted function is a free function.
    let no_fields = HashSet::new();
    let free_owner = owner.map(|owner| Owner {
        name: owner.name,
        fields: &no_fields,
    });

    let mut errors = diagnostics::Collector::default();
    for (local, pattern, closure) in sites {
        let result = validate_closure(
            stage,
            local,
            pattern,
            closure,
            &bindings.0,
            &closure_names,
            function_names,
            result_helpers,
            methods,
            free_owner.as_ref(),
        );
        if let Err(error) = result {
            errors.push(error);
        }
    }
    errors.finish()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_closure(
    stage: &dyn StageProfile,
    local: &syn::Local,
    pattern: &syn::PatIdent,
    closure: &syn::ExprClosure,
    bindings: &HashMap<String, usize>,
    closure_names: &HashSet<String>,
    function_names: &HashSet<String>,
    result_helpers: &HashSet<String>,
    methods: &MethodTable,
    owner: Option<&Owner<'_>>,
) -> syn::Result<()> {
    let name = pattern.ident.unraw().to_string();
    // The form: what has no lowering at all.
    if closure.asyncness.is_some()
        || closure.constness.is_some()
        || closure.movability.is_some()
        || closure.lifetimes.is_some()
        || !closure.attrs.is_empty()
    {
        return Err(diagnostics::error(
            Code::Closure,
            local,
            "this closure form has no lowering: write `let f = |x: f32| -> f32 { .. };` without `async`, `static`, \
             `const`, a `for<..>` binder or an attribute",
        ));
    }
    if closure.capture.is_some() {
        return Err(diagnostics::error(
            Code::ClosureCapture,
            local,
            format!(
                "`move` makes the closure `{name}` own what it captures; a closure here captures nothing, so it takes \
                 everything it uses as a parameter: remove `move`"
            ),
        ));
    }
    // The binding: immutable, no annotation, a plain name.
    if pattern.mutability.is_some() {
        return Err(diagnostics::error(
            Code::ClosureValue,
            local,
            format!(
                "a closure binding is immutable: write `let {name} = |..| ..;`, not `let mut {name}`"
            ),
        ));
    }
    if matches!(local.pat, syn::Pat::Type(_))
        || pattern.by_ref.is_some()
        || pattern.subpat.is_some()
    {
        return Err(diagnostics::error(
            Code::ClosureValue,
            local,
            format!(
                "the closure `{name}` takes no type annotation on its binding: its parameters and result carry the types"
            ),
        ));
    }
    // The signature: typed parameters, an explicit non-unit result, a block body.
    for input in &closure.inputs {
        let typed = matches!(input, syn::Pat::Type(typed)
            if matches!(typed.pat.as_ref(), syn::Pat::Ident(parameter)
                if parameter.by_ref.is_none() && parameter.subpat.is_none()));
        if !typed {
            return Err(diagnostics::error(
                Code::ClosureSignature,
                input,
                "every parameter of a closure is `name: Type` (`|x: f32, y: f32| -> f32 { .. }`): the macro sees no types, \
                 so none is inferred",
            ));
        }
    }
    match &closure.output {
        syn::ReturnType::Default => {
            return Err(diagnostics::error(
                Code::ClosureSignature,
                closure,
                format!(
                    "the closure `{name}` needs an explicit result type: `|x: f32| -> f32 {{ .. }}`"
                ),
            ));
        }
        syn::ReturnType::Type(_, ty) if matches!(peel_type(ty), syn::Type::Tuple(tuple) if tuple.elems.is_empty()) =>
        {
            return Err(diagnostics::error(
                Code::ClosureSignature,
                ty,
                "a closure gives a value, and a closure cannot change anything outside itself: a unit result has no use",
            ));
        }
        syn::ReturnType::Type(..) => {}
    }
    if !matches!(closure.body.as_ref(), syn::Expr::Block(body) if body.label.is_none()) {
        return Err(diagnostics::error(
            Code::ClosureSignature,
            &closure.body,
            "the body of a closure is a block: `|x: f32| -> f32 { x * 2.0f32 }`",
        ));
    }
    // A closure inside the body.
    struct Nested<'a>(Option<&'a syn::ExprClosure>);
    impl<'a> Visit<'a> for Nested<'a> {
        fn visit_expr_closure(&mut self, inner: &'a syn::ExprClosure) {
            self.0.get_or_insert(inner);
        }
    }
    let mut nested = Nested(None);
    nested.visit_expr(&closure.body);
    if let Some(inner) = nested.0 {
        return Err(diagnostics::error(
            Code::NestedClosure,
            inner,
            format!(
                "a closure inside the closure `{name}`: each closure is a function of its own, and a closure cannot hold \
                 another; bind both in the surrounding function"
            ),
        ));
    }
    // The name.
    let raw = pattern.ident.unraw().to_string();
    if raw.starts_with("__gust_") || raw.starts_with("__gpu_") || raw.starts_with("__Gust") {
        return Err(diagnostics::error(
            Code::ReservedPrefix,
            &pattern.ident,
            "the __gust_, __gpu_ and __Gust prefixes are reserved for generated shader symbols",
        ));
    }
    let clash = if function_names.contains(&name) || methods.by_name.contains_key(&name) {
        Some("a helper function or a method of the module")
    } else if is_dialect_callee(&name) {
        Some("a name the dialect gives a meaning to")
    } else if bindings.get(&name).copied().unwrap_or(0) > 1 {
        Some(
            "another binding of the function (a parameter, a local, a loop variable, a pattern or another closure)",
        )
    } else {
        None
    };
    if let Some(what) = clash {
        return Err(diagnostics::error(
            Code::MethodNameClash,
            &pattern.ident,
            format!(
                "the closure `{name}` is named like {what}: a closure name is unique in its function, so that a call \
                 `{name}(..)` has one meaning; rename the closure"
            ),
        ));
    }
    // The captures.
    let captures = closure_captures(closure, function_names);
    if !captures.is_empty() {
        let earlier: Vec<&String> = captures
            .iter()
            .filter(|capture| closure_names.contains(*capture))
            .collect();
        let listed = captures
            .iter()
            .map(|capture| format!("`{capture}`"))
            .collect::<Vec<_>>()
            .join(", ");
        let note = if earlier.is_empty() {
            String::new()
        } else {
            format!(
                " ({} {} a closure, and a closure cannot call a closure)",
                earlier
                    .iter()
                    .map(|capture| format!("`{capture}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
                if earlier.len() == 1 { "is" } else { "are" }
            )
        };
        return Err(diagnostics::error(
            Code::ClosureCapture,
            local,
            format!(
                "the closure `{name}` uses {listed}, which belong to the surrounding function{note}: a closure captures \
                 nothing, so pass every value it needs as a parameter"
            ),
        ));
    }
    // The body, as the body of a helper (D154).
    let function = crate::slang::closure_function(&pattern.ident, closure)
        .expect("the form, the signature and the body were checked above");
    validate_function(
        stage,
        &function,
        function_names,
        result_helpers,
        methods,
        owner,
        true,
    )
}

/// The first call of a closure of `closures` inside `expression` (a `for` bound), in source order (slice m9.c).
pub(super) fn first_closure_call<'a>(
    closures: &HashSet<String>,
    expression: &'a syn::Expr,
) -> Option<&'a syn::ExprCall> {
    struct Scan<'m, 'a> {
        closures: &'m HashSet<String>,
        found: Option<&'a syn::ExprCall>,
    }
    impl<'a> Visit<'a> for Scan<'_, 'a> {
        fn visit_expr_call(&mut self, call: &'a syn::ExprCall) {
            if self.found.is_some() {
                return;
            }
            if let syn::Expr::Path(path) = call.func.as_ref()
                && path.qself.is_none()
                && path
                    .path
                    .get_ident()
                    .is_some_and(|name| self.closures.contains(&name.unraw().to_string()))
            {
                self.found = Some(call);
                return;
            }
            visit::visit_expr_call(self, call);
        }
    }
    let mut scan = Scan {
        closures,
        found: None,
    };
    scan.visit_expr(expression);
    scan.found
}

/// The first call of a `&mut self` method of the module inside `expression` (a `for` bound), in source order. The macro
/// sees no types, so a name that some `&mut self` method has counts, as in the call graph (D162).
pub(super) fn first_mutating_call<'a>(
    methods: &MethodTable,
    expression: &'a syn::Expr,
) -> Option<&'a syn::ExprMethodCall> {
    struct Scan<'m, 'a> {
        methods: &'m MethodTable,
        found: Option<&'a syn::ExprMethodCall>,
    }
    impl<'a> Visit<'a> for Scan<'_, 'a> {
        fn visit_expr_method_call(&mut self, call: &'a syn::ExprMethodCall) {
            if self.found.is_some() {
                return;
            }
            if self
                .methods
                .by_name
                .get(&call.method.unraw().to_string())
                .is_some_and(|facts| facts.ref_mut)
            {
                self.found = Some(call);
                return;
            }
            visit::visit_expr_method_call(self, call);
        }
    }
    let mut scan = Scan {
        methods,
        found: None,
    };
    scan.visit_expr(expression);
    scan.found
}

pub(super) fn atomic_buffer_names(function: &syn::ItemFn) -> HashSet<String> {
    let mut usage = AtomicBufferUsage(HashSet::new());
    usage.visit_item_fn(function);
    usage.0
}

pub(super) struct AtomicBufferUsage(HashSet<String>);

impl<'ast> Visit<'ast> for AtomicBufferUsage {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let Some(_) = (match call.func.as_ref() {
            syn::Expr::Path(path) => atomic_operation_name(path),
            _ => None,
        }) else {
            visit::visit_expr_call(self, call);
            return;
        };
        if let Some(syn::Expr::Reference(reference)) = call.args.first()
            && let syn::Expr::Index(index) = reference.expr.as_ref()
            && let syn::Expr::Path(path) = index.expr.as_ref()
            && let Some(name) = path.path.get_ident()
        {
            self.0.insert(name.unraw().to_string());
        }
        visit::visit_expr_call(self, call);
    }
}

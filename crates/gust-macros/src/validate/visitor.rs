//! The restricted-body visitor: its state, constructor and the `match` and tuple checks.

use super::*;

/// What the visitor needs to know about a texture parameter.
#[derive(Clone, Copy)]
pub(super) struct TextureFacts {
    pub(super) access: texture::Access,
    /// Whether `sample` exists: a sampled texture of float texels.
    pub(super) filterable: bool,
}

pub(super) struct RestrictedVisitor {
    pub(super) function_names: HashSet<String>,
    pub(super) result_helpers: HashSet<String>,
    pub(super) result_values: HashSet<String>,
    pub(super) resources: HashSet<String>,
    pub(super) invocations: HashSet<String>,
    /// `LocalId`, `GroupId` and `Workgroup` parameters: locals cannot shadow them, and `.global_id()` is not theirs.
    pub(super) derived_ids: HashSet<String>,
    /// The `Workgroup<..>` parameter types of the kernel, by address: the only place the type may stand (E0210).
    pub(super) workgroup_types: HashSet<usize>,
    /// Uniform parameters. They are read field by field only (E0321), and are not in `resources` because `.len()` of
    /// one is E0306.
    pub(super) uniforms: HashSet<String>,
    /// Texture parameters (mx.t3). They are used as the receiver of a texture method only (GUST-E0736).
    pub(super) textures: HashMap<String, TextureFacts>,
    /// `Sampler` parameters: the first argument of `sample` and nothing else.
    pub(super) samplers: HashSet<String>,
    /// The texture and sampler parameter types of the kernel, by address: the only place the types may stand.
    pub(super) texture_types: HashSet<usize>,
    /// Read-write buffer parameters and their element type, for atomic receivers.
    pub(super) read_write_buffers: HashMap<String, String>,
    /// Buffer parameters used by an atomic operation in this function.
    pub(super) atomic_buffers: HashSet<String>,
    /// Set while inside a type position where `Option` has no proven layout.
    pub(super) option_forbidden: Option<&'static str>,
    /// Set while inside a type position where `Result` has no proven layout.
    pub(super) result_forbidden: Option<&'static str>,
    /// Number of enclosing `for` loops; `break`/`continue` need at least one.
    pub(super) loop_depth: usize,
    /// The module's method names, each with whether some impl declares it with `&mut self` or `self` (m9.a1).
    pub(super) methods: MethodTable,
    /// The struct whose `impl` holds the function being visited: `Self` is its name there (slice m9.a2).
    pub(super) self_struct: Option<String>,
    /// The fields of that struct: a helper of the same name would be captured by the field inside the Slang struct.
    pub(super) owner_fields: HashSet<String>,
    /// The value parameters of the function (not its receiver): a `&mut self` method is not called on one of them.
    pub(super) value_params: HashSet<String>,
    /// The `loop_depth` at the start of each enclosing enum `match` arm, innermost last: a `break` or `continue` at that
    /// depth would bind to the Slang `switch`, not to the Rust loop (GUST-E0417).
    pub(super) switch_depths: Vec<usize>,
    /// The `match` that ends the body of a function with a result type, by address (m9.e2 review S1): when its arms
    /// give values it is a value in tail position, GUST-E0419.
    pub(super) body_tail_match: Option<usize>,
    /// The parameters and locals whose type is an enum of the module, with the enum's name (by name, as `value_params`).
    pub(super) enum_values: HashMap<String, String>,
    /// The parameters and locals that hold a tuple, by name (slice m9.t; scoped like `enum_values`): a `match` on one
    /// and a comparison of one have no lowering.
    pub(super) tuple_values: HashSet<String>,
    /// The names of the function that hold an array and its length, `None` for a name bound otherwise or with another
    /// length (slice m9.r1; `crate::slang::array_lengths`, which the emitter reads as well).
    pub(super) array_lens: HashMap<String, Option<u32>>,
    /// How many `[..]` index operands enclose the node being visited, counting only the operators that carry a `usize`
    /// value on (see `visit_expr`): `as usize` and an array `.len()` stand there and nowhere else (GUST-E0334).
    pub(super) index_depth: usize,
    /// The `a.len()` calls of arrays that are the operand of a cast to a 32-bit scalar (`a.len() as u32`), by address.
    pub(super) len_casts: HashSet<usize>,
    /// The closures bound by a `let` in scope (slice m9.c), by binding name: `f(..)` calls one, anything else that names
    /// one is GUST-E0330. The names are unique in a function and apart from every other binding (GUST-E0328), so the
    /// set needs no shadowing rules; it is restored at the end of a block.
    pub(super) closures: HashSet<String>,
    pub(super) errors: diagnostics::Collector,
}

impl RestrictedVisitor {
    pub(super) fn new(function_names: HashSet<String>, result_helpers: HashSet<String>) -> Self {
        Self {
            function_names,
            result_helpers,
            result_values: HashSet::new(),
            resources: HashSet::new(),
            invocations: HashSet::new(),
            derived_ids: HashSet::new(),
            workgroup_types: HashSet::new(),
            uniforms: HashSet::new(),
            textures: HashMap::new(),
            samplers: HashSet::new(),
            texture_types: HashSet::new(),
            read_write_buffers: HashMap::new(),
            atomic_buffers: HashSet::new(),
            option_forbidden: None,
            result_forbidden: None,
            loop_depth: 0,
            methods: MethodTable::default(),
            self_struct: None,
            owner_fields: HashSet::new(),
            value_params: HashSet::new(),
            switch_depths: Vec::new(),
            body_tail_match: None,
            enum_values: HashMap::new(),
            tuple_values: HashSet::new(),
            array_lens: HashMap::new(),
            index_depth: 0,
            len_casts: HashSet::new(),
            closures: HashSet::new(),
            errors: diagnostics::Collector::default(),
        }
    }

    /// Whether `expression` is a tuple, when the macro can tell without types: a tuple expression, a parameter or local
    /// that holds one, or a call of a helper or method that returns one (a method by its name alone).
    pub(super) fn is_tuple_expr(&self, expression: &syn::Expr) -> bool {
        match peel_expr(expression) {
            syn::Expr::Tuple(tuple) => !tuple.elems.is_empty(),
            syn::Expr::Path(path) if path.qself.is_none() => path
                .path
                .get_ident()
                .is_some_and(|name| self.tuple_values.contains(&name.unraw().to_string())),
            syn::Expr::Call(call) => matches!(
                call.func.as_ref(),
                syn::Expr::Path(path) if path.qself.is_none()
                    && path.path.segments.last().is_some_and(|last| {
                        self.methods.tuple_returns.contains(&last.ident.unraw().to_string())
                    })
            ),
            syn::Expr::MethodCall(call) => self
                .methods
                .tuple_returns
                .contains(&call.method.unraw().to_string()),
            _ => false,
        }
    }

    /// `let (a, b) = value;` and `let (a, b): (A, B) = value;` (slice m9.t): 2 to 8 plain identifiers, nothing nested.
    pub(super) fn visit_tuple_local<'ast>(
        &mut self,
        local: &'ast syn::Local,
        tuple: &'ast syn::PatTuple,
        annotation: Option<&'ast syn::Type>,
    ) {
        use syn::visit::Visit;
        if tuple.elems.len() < 2 || tuple.elems.len() > MAX_TUPLE_ELEMENTS {
            self.reject(
                Code::TupleShape,
                tuple,
                &format!(
                    "a tuple pattern has 2 to {MAX_TUPLE_ELEMENTS} elements, this one has {}",
                    tuple.elems.len()
                ),
            );
            return;
        }
        for element in &tuple.elems {
            match element {
                syn::Pat::Ident(ident) if ident.by_ref.is_some() || ident.subpat.is_some() => {
                    self.reject(
                        Code::BindingMode,
                        element,
                        "a tuple element binds by value: no `ref` and no `@`",
                    );
                    return;
                }
                syn::Pat::Ident(_) => {}
                _ => {
                    self.reject(
                        Code::LetNeedsIdentifier,
                        element,
                        "a tuple destructuring binds one plain identifier per element: no nested pattern, `_` or `..`; \
                         bind the element to a name and ignore it",
                    );
                    return;
                }
            }
        }
        // The names and the attributes go through the per-binding checks of a plain `let` (GUST-E0901 on a reserved
        // prefix, which would let `__gust_r_new` capture a renamed local; review m9.t M1).
        for attribute in &local.attrs {
            self.visit_attribute(attribute);
        }
        for element in &tuple.elems {
            self.visit_pat(element);
        }
        if let Some(ty) = annotation {
            let same_length = matches!(
                peel_type(ty),
                syn::Type::Tuple(types) if types.elems.len() == tuple.elems.len()
            );
            if !same_length {
                self.reject(
                    Code::TupleShape,
                    ty,
                    "a tuple destructuring is annotated with a tuple type of the same length",
                );
                return;
            }
        }
        if local
            .init
            .as_ref()
            .is_some_and(|initial| initial.diverge.is_some())
        {
            self.reject(
                Code::LetElse,
                local,
                "let-else requires control-flow lowering and is not supported yet",
            );
            return;
        }
        let element_types: Vec<Option<&syn::Type>> = match annotation.map(peel_type) {
            Some(syn::Type::Tuple(types)) => types.elems.iter().map(Some).collect(),
            _ => vec![None; tuple.elems.len()],
        };
        // The initializer is evaluated before the names exist (`let (a, b) = f(a);` reads the outer `a`).
        if let Some(ty) = annotation {
            self.visit_type(ty);
        }
        if let Some(initial) = &local.init {
            if matches!(peel_expr(&initial.expr), syn::Expr::Match(_)) {
                self.reject(
                    Code::ValueMatchForm,
                    &initial.expr,
                    "a tuple destructuring takes no `match` initializer: bind the `match` to a typed tuple `let` first, then destructure it",
                );
                return;
            }
            self.visit_expr(&initial.expr);
        }
        for (element, ty) in tuple.elems.iter().zip(element_types) {
            let syn::Pat::Ident(ident) = element else {
                continue;
            };
            let name = ident.ident.unraw().to_string();
            if self.is_parameter_name(&name) {
                self.reject(
                    Code::ShadowedParameter,
                    element,
                    "GPU locals cannot shadow resource or dispatch-thread parameters",
                );
                return;
            }
            let key = ident.ident.unraw().to_string();
            self.value_params.remove(&key);
            self.result_values.remove(&ident.ident.unraw().to_string());
            // What is not visible stays unknown: an enum held in a tuple is not tracked (documented limitation).
            match ty.and_then(|ty| enum_type_name(ty, &self.methods.enums)) {
                Some(enum_name) => {
                    self.enum_values.insert(key.clone(), enum_name);
                }
                None => {
                    self.enum_values.remove(&key);
                }
            }
            if ty.is_some_and(is_tuple_type) {
                self.tuple_values.insert(key);
            } else {
                self.tuple_values.remove(&key);
            }
        }
    }

    /// The enum of the module that `expression` is a value of, when the macro can tell without types: a variant path, a
    /// parameter or local declared with the enum, or a call of a helper that returns one.
    pub(super) fn enum_of(&self, expression: &syn::Expr) -> Option<String> {
        match peel_expr(expression) {
            syn::Expr::Path(path) if path.qself.is_none() => {
                if let Some(name) = path.path.get_ident() {
                    return self.enum_values.get(&name.unraw().to_string()).cloned();
                }
                let (enum_name, _) = self.variant_path(&path.path)?;
                Some(enum_name)
            }
            syn::Expr::Call(call) => {
                let syn::Expr::Path(path) = call.func.as_ref() else {
                    return None;
                };
                let name = path.path.get_ident()?.unraw().to_string();
                self.methods.enum_returns.get(&name).cloned()
            }
            _ => None,
        }
    }

    /// `(enum, variant)` when `path` is `E::V` with `E` an enum of the module and `V` one of its variants.
    pub(super) fn variant_path(&self, path: &syn::Path) -> Option<(String, String)> {
        if path.leading_colon.is_some() || path.segments.len() != 2 {
            return None;
        }
        let first = &path.segments[0];
        let last = &path.segments[1];
        if !first.arguments.is_none() || !last.arguments.is_none() {
            return None;
        }
        let enum_name = first.ident.unraw().to_string();
        let variant = last.ident.unraw().to_string();
        self.methods
            .enums
            .get(&enum_name)?
            .variants
            .contains(&variant)
            .then_some((enum_name, variant))
    }

    /// Whether `tokens` name an enum of the module anywhere.
    pub(super) fn mentions_enum(&self, tokens: impl quote::ToTokens) -> bool {
        fn scan(stream: proc_macro2::TokenStream, enums: &HashMap<String, EnumFacts>) -> bool {
            stream.into_iter().any(|tree| match tree {
                proc_macro2::TokenTree::Ident(ident) => {
                    enums.contains_key(&ident.unraw().to_string())
                }
                proc_macro2::TokenTree::Group(group) => scan(group.stream(), enums),
                _ => false,
            })
        }
        !self.methods.enums.is_empty() && scan(tokens.to_token_stream(), &self.methods.enums)
    }

    /// Judges a `match` that lowers to a Slang `switch` (slice m9.e1, D157, and e2 for integers and the value form) and
    /// visits its parts: on an enum (`E::A | E::B`, `_`) or on an integer (`0`, `1 | 2`, `-1i32`, `_`), as a statement or
    /// as the initializer of a `let`.
    /// Refuses a `match` on a tuple (GUST-E0515; tuple patterns are not part of the dialect). Returns whether it did.
    pub(super) fn reject_tuple_scrutinee(&mut self, expression: &syn::ExprMatch) -> bool {
        if !self.is_tuple_expr(&expression.expr) {
            return false;
        }
        self.reject(
            Code::MatchScrutinee,
            &expression.expr,
            "a tuple is not a `match` scrutinee (there are no tuple patterns): destructure it with `let (a, b) = t;` and match on a part",
        );
        true
    }

    pub(super) fn check_switch_match(&mut self, expression: &syn::ExprMatch, form: MatchForm) {
        if self.reject_tuple_scrutinee(expression) {
            return;
        }
        let integer = expression
            .arms
            .iter()
            .any(|arm| has_literal_alternative(&arm.pat));
        let on = if integer {
            // An integer scrutinee is any expression: `switch (e)` evaluates it once, and the patterns decide the form.
            SwitchOn::Integer
        } else {
            let scrutinee_ok = matches!(
                peel_expr(&expression.expr),
                syn::Expr::Path(_) | syn::Expr::Call(_) | syn::Expr::MethodCall(_)
            );
            // A method call's enum is not tracked; the first arm names it.
            let first_path = expression
                .arms
                .iter()
                .find_map(|arm| first_variant_path(&arm.pat));
            let named_by_arm = first_path.and_then(|path| {
                let owner = path.segments.first()?.ident.unraw().to_string();
                self.methods.enums.contains_key(&owner).then_some(owner)
            });
            let Some(enum_name) = scrutinee_ok
                .then(|| self.enum_of(&expression.expr).or(named_by_arm))
                .flatten()
            else {
                // A path that names no enum of the module (`u32::MAX`, `Other::A`) is a constant used as a pattern.
                let constant = first_path.filter(|path| {
                    path.segments.first().is_none_or(|first| {
                        !self
                            .methods
                            .enums
                            .contains_key(&first.ident.unraw().to_string())
                    })
                });
                if let Some(path) = constant {
                    self.reject(
                        Code::MatchPattern,
                        path,
                        "a constant is not a pattern: an integer `match` takes literals (`0`, `1u32`, `-1i32`) and `_`, an enum `match` takes `E::A` of an enum of the module",
                    );
                    return;
                }
                self.reject(
                    Code::MatchScrutinee,
                    &expression.expr,
                    "a `match` on an enum takes a local, a parameter, a call or a variant as its scrutinee: bind a larger expression to a local first (`let m = ..; match m { .. }`); a `match` on a `u32` or `i32` takes any expression but needs an integer literal arm (`0`, `1 | 2`) before its `_`",
                );
                return;
            };
            SwitchOn::Enum(enum_name)
        };
        let mut seen: HashSet<String> = HashSet::new();
        let mut seen_numbers: HashSet<i128> = HashSet::new();
        let mut wildcard = false;
        for arm in &expression.arms {
            if let Some(attribute) = arm.attrs.first() {
                self.reject(
                    Code::MatchArmAttribute,
                    attribute,
                    "attributes on a `match` arm are not supported",
                );
                return;
            }
            if let Some((_, guard)) = &arm.guard {
                self.reject(
                    Code::EnumMatchGuard,
                    guard,
                    "a `match` arm takes no `if` guard: the Slang `switch` has no guard, and a guard would make the arm's coverage depend on a run-time value; test the condition inside the arm",
                );
                return;
            }
            let mut alternatives: Vec<&syn::Pat> = Vec::new();
            flatten_or(&arm.pat, &mut alternatives);
            let lone = alternatives.len() == 1;
            for pattern in alternatives {
                match (pattern, &on) {
                    (syn::Pat::Wild(_), _) if lone => {
                        if wildcard {
                            self.reject(Code::MatchPattern, pattern, "a `match` has one `_` arm");
                            return;
                        }
                        wildcard = true;
                    }
                    (syn::Pat::Path(path), SwitchOn::Enum(enum_name)) if path.qself.is_none() => {
                        match self.variant_path(&path.path) {
                            Some((owner, variant)) if &owner == enum_name => {
                                if !seen.insert(variant.clone()) || wildcard {
                                    self.reject(
                                        Code::MatchPattern,
                                        pattern,
                                        &format!("`{owner}::{variant}` is matched twice, or after `_`: the second arm can never run"),
                                    );
                                    return;
                                }
                            }
                            _ => {
                                self.reject(
                                    Code::MatchPattern,
                                    pattern,
                                    &format!("a `match` arm on `{enum_name}` names one of its variants as `{enum_name}::Variant`"),
                                );
                                return;
                            }
                        }
                    }
                    (syn::Pat::Lit(literal), SwitchOn::Integer) => match integer_pattern(literal) {
                        Ok((number, _)) => {
                            if !seen_numbers.insert(number) || wildcard {
                                self.reject(
                                    Code::MatchPattern,
                                    pattern,
                                    &format!("`{number}` is matched twice, or after `_`: the second arm can never run"),
                                );
                                return;
                            }
                        }
                        Err(reason) => {
                            self.reject(Code::MatchPattern, pattern, reason);
                            return;
                        }
                    },
                    (syn::Pat::Range(_), _) => {
                        self.reject(
                            Code::MatchPattern,
                            pattern,
                            "a range is not a pattern of a `match` here: list the integer literals (`0 | 1 | 2`) or test the range with `if`",
                        );
                        return;
                    }
                    (_, SwitchOn::Enum(_)) => {
                        self.reject(
                            Code::MatchPattern,
                            pattern,
                            "an enum `match` arm is `E::A`, `E::A | E::B` or `_`: bindings, literals, tuples and ranges have no lowering (an `_` joined by `|` is refused as well)",
                        );
                        return;
                    }
                    (_, SwitchOn::Integer) => {
                        self.reject(
                            Code::MatchPattern,
                            pattern,
                            "an integer `match` arm is a literal (`0`, `1 | 2`, `-1i32`) or `_`: bindings, constants, enum variants (an enum `match` does not mix with literals), tuples and ranges have no lowering (an `_` joined by `|` is refused as well)",
                        );
                        return;
                    }
                }
            }
            if form == MatchForm::Statement && !matches!(arm.body.as_ref(), syn::Expr::Block(_)) {
                self.reject(
                    Code::MatchArmBlock,
                    &arm.body,
                    "each arm of a statement `match` is a block: `E::A => { .. }`; an arm that gives a value belongs to `let x: T = match e { .. };` with a typed `let`",
                );
                return;
            }
        }
        if matches!(on, SwitchOn::Integer) && !wildcard {
            self.reject(
                Code::MatchPattern,
                expression,
                "an integer `match` ends in a `_` arm: no list of literals covers every `u32` or `i32`",
            );
            return;
        }
        self.visit_expr(&expression.expr);
        for arm in &expression.arms {
            self.switch_depths.push(self.loop_depth);
            match form {
                MatchForm::Statement => {
                    let syn::Expr::Block(body) = arm.body.as_ref() else {
                        unreachable!("a statement `match` arm was checked to be a block")
                    };
                    if let Some(tail) = value_tail(&body.block) {
                        self.switch_depths.pop();
                        self.reject(
                            Code::MatchAsValue,
                            tail,
                            "the arms of a statement `match` end in a statement; a tail expression makes the `match` a value, which is `let x: T = match e { .. };` with a typed `let`: bind it there, or end the arm's last statement with `;`",
                        );
                        return;
                    }
                    self.visit_block(&body.block);
                }
                MatchForm::Value => self.visit_value_arm(&arm.body),
            }
            self.switch_depths.pop();
        }
    }
}

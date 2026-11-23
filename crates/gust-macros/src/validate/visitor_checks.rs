//! The restricted-body visitor: the helper checks for places, textures, jumps, methods and atomics.

use super::*;

impl RestrictedVisitor {
    /// An arm of a value `match`: an expression, a block that ends in an expression (or in a `return`), or a nested
    /// value `match`. The value is assigned to the `let` it initializes.
    pub(super) fn visit_value_arm(&mut self, body: &syn::Expr) {
        match peel_expr(body) {
            syn::Expr::Block(block) if block.label.is_none() => {
                self.visit_value_block(&block.block);
            }
            syn::Expr::Match(inner) if is_switch_match(inner) => {
                self.check_switch_match(inner, MatchForm::Value);
            }
            other => self.visit_expr(other),
        }
    }

    /// The statements of a block whose last statement is its value; the block is a scope of its own.
    pub(super) fn visit_value_block(&mut self, block: &syn::Block) {
        let outer = self.enum_values.clone();
        let outer_tuples = self.tuple_values.clone();
        let (head, tail) = match block.stmts.split_last() {
            Some((syn::Stmt::Expr(tail, None), head)) => (head, Some(tail)),
            Some((syn::Stmt::Expr(syn::Expr::Return(_), Some(_)), _)) => {
                (block.stmts.as_slice(), None)
            }
            _ => {
                self.reject(
                    Code::MatchArmBlock,
                    block,
                    "each arm of a value `match` gives a value: an expression, or a block whose last line is an expression without `;` (or a `return`)",
                );
                return;
            }
        };
        for statement in head {
            self.visit_stmt(statement);
        }
        if let Some(tail) = tail {
            self.visit_value_arm(tail);
        }
        self.enum_values = outer;
        self.tuple_values = outer_tuples;
    }

    /// Whether `name` is a kernel parameter: a local cannot shadow one.
    pub(super) fn is_parameter_name(&self, name: &str) -> bool {
        self.resources.contains(name)
            || self.invocations.contains(name)
            || self.derived_ids.contains(name)
            || self.uniforms.contains(name)
            || self.textures.contains_key(name)
            || self.samplers.contains(name)
    }

    /// Whether `expression` is a bare use of a uniform parameter, possibly in parentheses.
    pub(super) fn is_uniform_parameter(&self, expression: &syn::Expr) -> bool {
        match peel_expr(expression) {
            syn::Expr::Path(path) => {
                path.qself.is_none()
                    && path
                        .path
                        .get_ident()
                        .is_some_and(|name| self.uniforms.contains(&name.unraw().to_string()))
            }
            _ => false,
        }
    }

    /// The name of a bare parameter use: `img` or `(img)`.
    pub(super) fn direct_name(expression: &syn::Expr) -> Option<String> {
        match peel_expr(expression) {
            syn::Expr::Path(path) if path.qself.is_none() => {
                path.path.get_ident().map(|ident| ident.unraw().to_string())
            }
            _ => None,
        }
    }

    /// A method call whose receiver is a texture or a sampler parameter (slice mx.t3, D64): `load`, `store`,
    /// `sample`, `width` and `height` on the textures that have them. Returns whether the receiver was one; the call
    /// is then checked here and its arguments visited, and the receiver is not visited (a texture is used as a
    /// receiver only, GUST-E0736).
    pub(super) fn check_texture_call(&mut self, call: &syn::ExprMethodCall) -> bool {
        let Some(receiver) = Self::direct_name(&call.receiver) else {
            return false;
        };
        if self.samplers.contains(&receiver) {
            self.reject(
                Code::TextureUse,
                call,
                &format!(
                    "`{receiver}` is a `Sampler`: it has no methods; pass it as the first argument of a texture's \
                     `.sample({receiver}, u, v)`"
                ),
            );
            return true;
        }
        let Some(facts) = self.textures.get(&receiver).copied() else {
            return false;
        };
        let method = call.method.unraw().to_string();
        if call.turbofish.is_some() {
            self.reject(
                Code::BuiltinTypeArguments,
                call,
                "GPU built-ins do not accept type arguments",
            );
            return true;
        }
        let Some(kind) = texture::Method::from_name(&method) else {
            self.reject(
                Code::TextureMethod,
                call,
                &format!(
                    "a texture has the methods `.load(x, y)`, `.store(x, y, v)`, `.sample(sampler, u, v)`, \
                     `.width()` and `.height()`, not `.{method}(..)`"
                ),
            );
            return true;
        };
        if !kind.exists_on(facts.access) {
            let reason = match (kind, facts.access) {
                (texture::Method::Load, _) => {
                    "a write-only texture cannot be read: use `StorageTexture2D` (three formats) or bind the texture \
                     as a sampled `Texture2D` in a second kernel"
                }
                (texture::Method::Store, _) => {
                    "a sampled texture is read-only: use `StorageTexture2D` or `WriteTexture2D` to write"
                }
                (_, texture::Access::ReadWrite) => {
                    "`.sample(..)` filters a sampled `Texture2D`; a storage texture is read with `.load(x, y)`"
                }
                _ => {
                    "`.sample(..)` filters a sampled `Texture2D`; a write-only texture has `.store(x, y, v)`, \
                     `.width()` and `.height()` only"
                }
            };
            self.reject(
                Code::TextureMethod,
                call,
                &format!("`{receiver}` has no `.{method}(..)`: {reason}"),
            );
            return true;
        }
        if call.args.len() != kind.arity() {
            self.reject(
                Code::TextureCoordinate,
                call,
                &format!(
                    "`.{method}(..)` takes {} argument(s): {}",
                    kind.arity(),
                    match kind {
                        texture::Method::Load => "the texel address `x, y` (both `u32`)",
                        texture::Method::Store =>
                            "the texel address `x, y` (both `u32`) and the value",
                        texture::Method::Sample =>
                            "a sampler and the coordinates `u, v` (both `f32`)",
                        _ => "none",
                    }
                ),
            );
            return true;
        }
        let mut skipped = 0;
        match kind {
            texture::Method::Sample => {
                if !facts.filterable {
                    self.reject(
                        Code::SampleInteger,
                        call,
                        &format!(
                            "`{receiver}` has integer texels, which are never filtered: read it with \
                             `.load(x, y)`, or sample a texture of `f32`, `Vec2` or `Vec4` texels"
                        ),
                    );
                    return true;
                }
                let sampler = Self::direct_name(&call.args[0]);
                if !sampler.is_some_and(|name| self.samplers.contains(&name)) {
                    self.reject(
                        Code::SamplerArgument,
                        &call.args[0],
                        "the first argument of `.sample(..)` is a `Sampler` parameter of the kernel, passed by name",
                    );
                }
                skipped = 1;
                for coordinate in call.args.iter().skip(1) {
                    if let Some(problem) = coordinate_problem(coordinate) {
                        self.reject(
                            Code::TextureCoordinate,
                            coordinate,
                            &format!("a sample coordinate is an `f32`, and this is {problem}"),
                        );
                    }
                }
            }
            texture::Method::Load | texture::Method::Store => {
                for coordinate in call.args.iter().take(2) {
                    if let Some(problem) = address_problem(coordinate) {
                        self.reject(
                            Code::TextureCoordinate,
                            coordinate,
                            &format!("a texel address is a `u32`, and this is {problem}"),
                        );
                    }
                }
            }
            texture::Method::Width | texture::Method::Height => {}
        }
        for argument in call.args.iter().skip(skipped) {
            self.visit_expr(argument);
        }
        true
    }

    /// The identifier a place expression is rooted in: `out` for `out`, `out[i]`, `out[i].a.b` and parenthesised forms.
    pub(super) fn place_root(expression: &syn::Expr) -> Option<String> {
        match peel_expr(expression) {
            syn::Expr::Path(path) if path.qself.is_none() => {
                path.path.get_ident().map(|ident| ident.unraw().to_string())
            }
            syn::Expr::Field(field) => Self::place_root(&field.base),
            syn::Expr::Index(index) => Self::place_root(&index.expr),
            _ => None,
        }
    }

    /// A call `recv.name(args)` of a method of the module (slice m9.a1). A `&self` method copies its receiver, so any
    /// place may be one. A method that takes `&mut self` or `self` would write through a storage, uniform, workgroup
    /// or texture place, a shared write that the kernel statements do not show, so it is refused there (GUST-E0333,
    /// D154); the validator decides from the root identifier of the receiver, no type is needed.
    ///
    /// Slice m9.a2: a `&mut self` method is called on a local place only, a `let mut` binding or a field or index of one
    /// (or on `self` inside a `&mut self` method). A receiver that is no place (the result of a call, a literal) has no
    /// Slang lvalue to mutate, and a value parameter is an `in` parameter there (slangc warns that the change is lost),
    /// so both are GUST-E0333 as well: copy the value to a `let mut` local first.
    /// GUST-E0421 for the place of an assignment: Rust evaluates the value before the place, and Slang evaluates a call
    /// inside the place first (review m9.a2 M1), so a `&mut self` call there would run in the other order.
    pub(super) fn reject_mutating_in_place(&mut self, place: &syn::Expr) -> bool {
        if let Some(call) = first_mutating_call(&self.methods, place) {
            self.reject(
                Code::ForBoundSideEffect,
                call,
                &format!(
                    "the place of an assignment calls `.{}(..)`, which takes `&mut self`: Rust evaluates the assigned value before the place and Slang evaluates the call in the place first, so the order of the effects would differ; call the method before the statement and use the value (`let i = c.next(); out[i] = value;`)",
                    call.method.unraw()
                ),
            );
            return true;
        }
        false
    }

    /// GUST-E0421 for an indexed place of a local whose index reads something the value of the assignment changes
    /// (slice m9.r1): Rust evaluates the value of `a[s.n as usize] = s.bump()` first and the index after it, and Slang
    /// computes the index before the value (measured on slangc 2026.21 for `=` and `op=`, on a local array), so the
    /// element written would differ. The roots are the receivers of the `&mut self` calls in the value; the index
    /// "reads" a root when it names it. Every indexed place is covered, a storage or workgroup element included (fix
    /// round 1 of m9.r1): slangc 2026.21 reads the index of `out[c.n] = c.next()` first, where 2026.20 called first,
    /// so a rule that relied on its order would not be robust.
    pub(super) fn reject_stale_index(&mut self, place: &syn::Expr, value: &syn::Expr) -> bool {
        struct Roots<'m> {
            methods: &'m MethodTable,
            roots: Vec<(String, String)>,
        }
        impl<'a> Visit<'a> for Roots<'_> {
            fn visit_expr_method_call(&mut self, call: &'a syn::ExprMethodCall) {
                if self
                    .methods
                    .by_name
                    .get(&call.method.unraw().to_string())
                    .is_some_and(|facts| facts.ref_mut)
                    && let Some(root) = RestrictedVisitor::place_root(&call.receiver)
                {
                    self.roots.push((root, call.method.unraw().to_string()));
                }
                visit::visit_expr_method_call(self, call);
            }
        }
        struct Names(HashSet<String>);
        impl<'a> Visit<'a> for Names {
            fn visit_expr_path(&mut self, path: &'a syn::ExprPath) {
                if let Some(name) = path.path.get_ident() {
                    self.0.insert(name.unraw().to_string());
                }
            }
        }
        struct Indexes<'a>(Vec<&'a syn::Expr>);
        impl<'a> Visit<'a> for Indexes<'a> {
            fn visit_expr_index(&mut self, index: &'a syn::ExprIndex) {
                self.0.push(&index.index);
                visit::visit_expr_index(self, index);
            }
        }
        if Self::place_root(place).is_none() {
            return false;
        }
        let mut roots = Roots {
            methods: &self.methods,
            roots: Vec::new(),
        };
        roots.visit_expr(value);
        if roots.roots.is_empty() {
            return false;
        }
        let roots = roots.roots;
        let mut indexes = Indexes(Vec::new());
        indexes.visit_expr(place);
        for index in indexes.0 {
            let mut names = Names(HashSet::new());
            names.visit_expr(index);
            if let Some((root, method)) = roots.iter().find(|(root, _)| names.0.contains(root)) {
                // A keyword name keeps its `r#` in the text, so the suggestion is valid Rust (Opus round 3 N4).
                let (root, method) = (crate::names::spelling(root), crate::names::spelling(method));
                self.reject(
                    Code::ForBoundSideEffect,
                    index,
                    &format!(
                        "the index of the assigned place reads `{root}`, and the value calls `.{method}(..)` on it with `&mut self`: Rust evaluates the value first and the index after it, Slang computes the index first, so another element than Rust's would be written; call the method first (`let v = {root}.{method}(..); a[i] = v;`)"
                    ),
                );
                return true;
            }
        }
        false
    }

    pub(super) fn check_module_method_call(
        &mut self,
        call: &syn::ExprMethodCall,
        facts: MethodFacts,
    ) {
        let method = call.method.unraw().to_string();
        if call.turbofish.is_some() {
            self.reject(
                Code::TurbofishOnGenericCall,
                call,
                &format!(
                    "type arguments are inferred from the arguments of a call: write `.{method}(a)`, not `.{method}::<X>(a)`"
                ),
            );
            return;
        }
        let root = Self::place_root(&call.receiver);
        if facts.mutating()
            && let Some(root) = &root
            && (self.resources.contains(root)
                || self.uniforms.contains(root)
                || self.derived_ids.contains(root))
        {
            self.reject(
                Code::MutatingMethodOnSharedPlace,
                call,
                &format!(
                    "`.{method}(..)` takes `&mut self` or `self`, and `{root}` is a storage buffer, a uniform, a workgroup array or an invocation id of the kernel: a method would write through it where no statement shows the write; copy the element to a local, call the method on the local and store the result (`out[i] = value;`)"
                ),
            );
            return;
        }
        if facts.ref_mut {
            let problem = match &root {
                None => Some("the receiver is not a place (it is the result of a call, a literal or a block)".to_owned()),
                Some(root) if self.value_params.contains(root) => Some(format!(
                    "`{root}` is a parameter, which Slang passes as a read-only value (`in`); the dialect asks for an explicit local copy before a mutating call"
                )),
                Some(_) => None,
            };
            if let Some(problem) = problem {
                self.reject(
                    Code::MutatingMethodOnSharedPlace,
                    call,
                    &format!(
                        "`.{method}(..)` takes `&mut self` and is called on a `let mut` local only: {problem}; bind the value with `let mut copy = ..;`, call the method on `copy` and use `copy`"
                    ),
                );
                return;
            }
        }
        self.visit_expr(&call.receiver);
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    pub(super) fn reject(&mut self, code: Code, node: impl quote::ToTokens, message: &str) {
        self.errors.reject(code, node, message);
    }

    pub(super) fn is_known_result(&self, expression: &syn::Expr) -> bool {
        match peel_expr(expression) {
            syn::Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|name| self.result_values.contains(&name.unraw().to_string())),
            syn::Expr::Call(call) => match call.func.as_ref() {
                syn::Expr::Path(path) => path
                    .path
                    .get_ident()
                    .is_some_and(|name| self.result_helpers.contains(&name.unraw().to_string())),
                _ => false,
            },
            _ => false,
        }
    }

    /// Shared checks for `break` and `continue` in statement position.
    pub(super) fn check_jump(
        &mut self,
        node: impl quote::ToTokens,
        label: Option<&syn::Lifetime>,
        name: &str,
    ) {
        if let Some(label) = label {
            self.reject(
                Code::LoopLabel,
                label,
                "loop labels are not supported in the Rust-to-Slang subset; restructure with a flag or a helper return",
            );
            return;
        }
        if self.loop_depth == 0 {
            self.reject(
                Code::JumpOutsideLoop,
                node,
                &format!("`{name}` outside a loop has no meaning"),
            );
        } else if self.switch_depths.last() == Some(&self.loop_depth) {
            let reason = if name == "break" {
                "would end the Slang `switch`, not the loop around it"
            } else {
                "is refused for symmetry with `break` (Slang's `continue` inside a `switch` does continue the loop)"
            };
            self.reject(
                Code::JumpInMatchArm,
                node,
                &format!(
                    "`{name}` in an arm of an enum `match` {reason}; set a flag the loop tests, or move the `match` into a helper that returns"
                ),
            );
        }
    }

    /// Validates an atomic operation call on a read-write buffer element.
    /// Accepts `atomic_add`, `atomic_min`, `atomic_max`, `atomic_exchange`,
    /// and `atomic_compare_exchange` on `u32`/`uint` and `i32`/`int` elements.
    /// `atomic_min` and `atomic_max` are unsigned-only (Slang/WGSL constraint).
    pub(super) fn validate_atomic(&mut self, op_name: &str, call: &syn::ExprCall) {
        let expected_args = match op_name {
            "atomic_compare_exchange" => 3,
            _ => 2,
        };
        if call.args.len() != expected_args {
            self.reject(
                Code::AtomicArgumentCount,
                call,
                &format!("`{op_name}` takes exactly {expected_args} arguments"),
            );
            return;
        }
        let syn::Expr::Reference(ref reference) = call.args[0] else {
            self.reject(
                Code::AtomicNotReference,
                &call.args[0],
                &format!("`{op_name}` requires a mutable reference to a buffer element:`&mut buffer[index]`"),
            );
            return;
        };
        if reference.mutability.is_none() {
            self.reject(
                Code::AtomicReferenceNotMutable,
                &call.args[0],
                &format!("`{op_name}` requires a mutable reference:`&mut buffer[index]`"),
            );
            return;
        }
        let syn::Expr::Index(index) = reference.expr.as_ref() else {
            self.reject(
                Code::AtomicOperandNotElement,
                &reference.expr,
                &format!("`{op_name}` requires a read-write buffer element: `&mut buffer[index]`"),
            );
            return;
        };
        let syn::Expr::Path(buffer_path) = index.expr.as_ref() else {
            self.reject(
                Code::AtomicReceiverNotBuffer,
                &index.expr,
                &format!("`{op_name}` requires a direct buffer receiver: `&mut buffer[index]`"),
            );
            return;
        };
        let buffer_name = buffer_path
            .path
            .get_ident()
            .map(|ident| ident.unraw().to_string());
        let element = match buffer_name
            .as_ref()
            .and_then(|name| self.read_write_buffers.get(name))
        {
            Some(element) => element,
            None => {
                self.reject(
                    Code::AtomicBufferParameter,
                    &index.expr,
                    &format!("`{op_name}` requires a read-write buffer parameter: `RWStructuredBuffer<u32>` or `RWStructuredBuffer<i32>`"),
                );
                return;
            }
        };
        let is_unsigned = element == "u32" || element == "uint";
        let is_signed = element == "i32" || element == "int";
        if !is_unsigned && !is_signed {
            self.reject(
                Code::AtomicElementType,
                &index.expr,
                &format!("`{op_name}` requires a `u32`/`uint` or `i32`/`int` element; other element types are not supported for atomics"),
            );
            return;
        }
        if (op_name == "atomic_min" || op_name == "atomic_max") && !is_unsigned {
            self.reject(
                Code::AtomicSignedMinMax,
                &index.expr,
                &format!("`{op_name}` requires an unsigned element (`u32`/`uint`); signed atomics for min/max are not supported"),
            );
            return;
        }
        self.visit_expr(&index.index);
        for arg in call.args.iter().skip(1) {
            self.visit_expr(arg);
        }
    }
}

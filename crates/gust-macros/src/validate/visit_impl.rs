//! The `Visit` implementation of the restricted-body visitor: one method per syntax node it checks.

use super::*;

impl<'ast> Visit<'ast> for RestrictedVisitor {
    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        if pattern.by_ref.is_some() || pattern.subpat.is_some() {
            self.reject(
                Code::BindingMode,
                pattern,
                "reference and @ binding modes are not supported in GPU code",
            );
            return;
        }
        visit::visit_pat_ident(self, pattern);
    }

    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        if let Some(message) = attribute_refusal(attribute) {
            self.reject(Code::UnsupportedAttribute, attribute, message);
        }
    }

    fn visit_ident(&mut self, ident: &'ast syn::Ident) {
        // The unrawed text, because emission strips a raw prefix: `r#__gust_r_new` would otherwise print the name that
        // renaming gives `new`.
        let name = ident.unraw().to_string();
        if name.starts_with("__gust_") || name.starts_with("__gpu_") || name.starts_with("__Gust") {
            self.reject(
                Code::ReservedPrefix,
                ident,
                "the __gust_, __gpu_ and __Gust prefixes are reserved for generated shader symbols",
            );
        }
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        // `Self` is the name of the struct of the enclosing `impl` (slice m9.a2): alone, as a type or a struct literal.
        // `Self::f(..)` is judged by `visit_expr_call`; any other path through `Self` has no lowering.
        if path
            .segments
            .first()
            .is_some_and(|segment| segment.ident == "Self")
            && (self.self_struct.is_none() || path.segments.len() != 1)
        {
            let message = if self.self_struct.is_none() {
                "`Self` is the type of an `impl` block: outside one, write the name of the struct"
            } else {
                "`Self` is accepted as a type, in a struct literal and in the call `Self::f(..)` of an associated function: other paths through it have no lowering"
            };
            self.reject(Code::UnsupportedModuleItem, path, message);
            return;
        }
        for segment in &path.segments {
            if BANNED_NAMES.contains(&segment.ident.unraw().to_string().as_str()) {
                self.reject(
                    Code::CpuOnlyName,
                    path,
                    "CPU-only library/type is not available inside a #[gpu_module] module",
                );
                return;
            }
        }
        if is_qualified_math_path(path) {
            self.reject(Code::QualifiedMathPath, path, QUALIFIED_MATH_PATH);
            return;
        }
        visit::visit_path(self, path);
    }

    fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
        use crate::slang::{ParameterFlavor, parameter_flavor};

        // `Workgroup<[T; N]>` stands only as a kernel parameter type, and `validate_function` checked its shape.
        if ty.qself.is_none()
            && ty
                .path
                .segments
                .last()
                .is_some_and(|segment| crate::names::ident_is(&segment.ident, "Workgroup"))
        {
            if !self
                .workgroup_types
                .contains(&(std::ptr::from_ref(ty) as usize))
            {
                self.reject(
                    Code::WorkgroupShape,
                    ty,
                    "`Workgroup<[T; N]>` is the type of a kernel parameter only: not a local, a field, a helper \
                     parameter or a result",
                );
            }
            return;
        }
        // A texture or sampler type stands as a kernel parameter type only (`validate_function` registered those).
        if ty.qself.is_none()
            && parameter_flavor(&syn::Type::Path(ty.clone())).is_texture_or_sampler()
        {
            if !self
                .texture_types
                .contains(&(std::ptr::from_ref(ty) as usize))
            {
                self.reject(
                    Code::TextureUse,
                    ty,
                    "a texture or sampler type is the type of a kernel parameter only: not a local, a field, a helper \
                     parameter or a result",
                );
            }
            return;
        }
        if (option_payload(ty).is_some() || result_payload(ty).is_some()) && self.mentions_enum(ty)
        {
            self.reject(
                Code::EnumInResource,
                ty,
                "an enum is not the payload of an `Option` or a `Result`: it has no `GpuPod` layout and no proven `Option` lowering; return the enum from a helper, or its integer",
            );
            return;
        }
        if let Some(payload) = option_payload(ty) {
            if let Some(context) = self.option_forbidden {
                self.reject(
                    Code::OptionLayout,
                    ty,
                    &format!(
                        "`Option` is not supported in {context}: it has no proven storage layout; keep Option values in locals and helper functions"
                    ),
                );
                return;
            }
            let supported = payload.is_some_and(|payload| match payload {
                syn::Type::Path(inner) => {
                    inner.path.get_ident().is_some()
                        && option_payload(inner).is_none()
                        && parameter_flavor(payload) == ParameterFlavor::Value
                }
                _ => false,
            });
            if !supported {
                self.reject(
                    Code::OptionPayload,
                    ty,
                    "`Option` payload must be a supported scalar or a module struct; nested Options and resources are not lowered",
                );
                return;
            }
            visit::visit_type_path(self, ty);
            return;
        }
        if let Some(payloads) = result_payload(ty) {
            if let Some(context) = self.result_forbidden {
                self.reject(
                    Code::ResultLayout,
                    ty,
                    &format!(
                        "`Result` is not supported in {context}: it has no proven storage layout; keep Result values in locals and helper functions"
                    ),
                );
                return;
            }
            let supported = payloads.is_some_and(|(ok, err)| {
                ok.to_token_stream().to_string() == err.to_token_stream().to_string()
                    && matches!(ok, syn::Type::Path(inner) if inner.path.get_ident().is_some()
                        && option_payload(inner).is_none()
                        && result_payload(inner).is_none()
                        && parameter_flavor(ok) == ParameterFlavor::Value)
            });
            if !supported {
                self.reject(
                    Code::ResultShape,
                    ty,
                    "`Result` is supported only as `Result<T, T>` with a supported scalar or module-struct payload; mixed, nested, and resource payloads are not lowered",
                );
                return;
            }
            visit::visit_type_path(self, ty);
            return;
        }
        if self.option_forbidden.is_some()
            && let Some(ident) = ty.path.get_ident()
            && let Some(math) = math_types::math_type(&ident.unraw().to_string())
            && !math.buffer
        {
            self.reject(
                Code::ThreeWideInBuffer,
                ty,
                &format!(
                    "`{}` cannot be a struct field or a resource element: 3-wide types have stride 16 in a buffer and no Rust layout that matches in every position; use the 4-wide type or scalars in buffers (`{}` stays fine as a local, a helper parameter or a helper result)",
                    math.name, math.name
                ),
            );
            return;
        }
        if let Some(ident) = ty.path.get_ident() {
            let name = ident.unraw().to_string();
            if name == "usize" {
                self.reject(
                    Code::UsizeOutsideIndex,
                    ty,
                    "`usize` is the type of an array index in Rust and has no GPU type: it appears only as `i as usize` inside the brackets of an index (`a[i as usize]`); keep the value in a `u32`",
                );
                return;
            }
            if UNSUPPORTED_PRIMITIVES.contains(&name.as_str()) {
                self.reject(
                    Code::UnsupportedPrimitive,
                    ty,
                    &format!(
                        "`{name}` is not a supported GPU type; use the 32-bit scalars `f32`/`float`, `i32`/`int`, `u32`/`uint`, or `bool`"
                    ),
                );
                return;
            }
        }
        let is_resource = matches!(
            parameter_flavor(&syn::Type::Path(ty.clone())),
            ParameterFlavor::StorageRead
                | ParameterFlavor::StorageReadWrite
                | ParameterFlavor::Uniform
        );
        if is_resource {
            let previous = self.option_forbidden.replace("resource element types");
            let previous_result = self.result_forbidden.replace("resource element types");
            visit::visit_type_path(self, ty);
            self.option_forbidden = previous;
            self.result_forbidden = previous_result;
            return;
        }
        visit::visit_type_path(self, ty);
    }

    fn visit_expr_cast(&mut self, cast: &'ast syn::ExprCast) {
        // `i as usize` is the index of an array (slice m9.r1) and nothing else: the dialect has no `usize` value.
        if matches!(peel_type(&cast.ty), syn::Type::Path(path)
            if path.qself.is_none() && crate::names::path_is(&path.path, "usize"))
        {
            if self.index_depth == 0 {
                self.reject(
                    Code::UsizeOutsideIndex,
                    cast,
                    "`as usize` is accepted inside the brackets of an array index only (`a[i as usize]`): the dialect has no `usize` value, so keep the number in a `u32`",
                );
                return;
            }
            if self.enum_of(&cast.expr).is_some() {
                self.reject(
                    Code::EnumShape,
                    cast,
                    "an enum casts to its repr type only: write `e as u32 as usize` in an index",
                );
                return;
            }
            self.visit_expr(&cast.expr);
            return;
        }
        // `a.len() as u32`: the one place the `usize` length of an array may stand outside an index.
        if let syn::Expr::MethodCall(call) = peel_expr(&cast.expr)
            && crate::names::ident_is(&call.method, "len")
        {
            self.len_casts.insert(std::ptr::from_ref(call) as usize);
        }
        if let Some(enum_name) = self.enum_of(&cast.expr) {
            let repr = self.methods.enums[&enum_name].repr_name();
            let target = match cast.ty.as_ref() {
                syn::Type::Path(path) => {
                    path.path.get_ident().map(|ident| ident.unraw().to_string())
                }
                _ => None,
            };
            let matches_repr = target.as_deref().is_some_and(|target| {
                target == repr
                    || (repr == "u32" && target == "uint")
                    || (repr == "i32" && target == "int")
            });
            if !matches_repr {
                self.reject(
                    Code::EnumShape,
                    &cast.ty,
                    &format!(
                        "an enum casts to its repr type only: `{enum_name}` is `#[repr({repr})]`, so write `as {repr}`"
                    ),
                );
                return;
            }
            self.visit_expr(&cast.expr);
            return;
        }
        let supported = match cast.ty.as_ref() {
            syn::Type::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| CAST_TARGETS.contains(&ident.unraw().to_string().as_str())),
            _ => false,
        };
        if !supported {
            self.reject(
                Code::UnsupportedCastTarget,
                &cast.ty,
                "casts are supported only to the 32-bit scalar types `f32`/`float`, `i32`/`int`, and `u32`/`uint`",
            );
            return;
        }
        visit::visit_expr_cast(self, cast);
    }

    fn visit_expr_const(&mut self, expression: &'ast syn::ExprConst) {
        self.reject(
            Code::ConstBlock,
            expression,
            "const blocks are not in the Rust-to-Slang subset; write the value directly",
        );
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if matches!(peel_expr(&call.func), syn::Expr::Closure(_)) {
            self.reject(
                Code::ClosureValue,
                call,
                "a closure is accepted only as the initializer of a `let` and is then called as `f(a)`: bind it first (`let f = |x: f32| -> f32 { x };` and `f(1.0f32)`); it cannot be called where it is written",
            );
            return;
        }
        // `f(a, b)` where `f` is a closure bound by an earlier `let` (slice m9.c): the arguments are ordinary expressions,
        // evaluated left to right like a helper's. A closure that is named like a dialect callee or a helper is refused
        // by the closure pass (GUST-E0328), so the name decides here.
        if let syn::Expr::Path(path) = call.func.as_ref()
            && path.qself.is_none()
            && let Some(ident) = path.path.get_ident()
            && self.closures.contains(&ident.unraw().to_string())
        {
            for argument in &call.args {
                self.visit_expr(argument);
            }
            return;
        }
        if let syn::Expr::Path(path) = call.func.as_ref() {
            let contains_banned_name =
                path.path.segments.iter().any(|segment| {
                    BANNED_NAMES.contains(&segment.ident.unraw().to_string().as_str())
                });
            if contains_banned_name {
                self.reject(
                    Code::CpuOnlyName,
                    path,
                    "CPU-only library/type is not available inside a #[gpu_module] module",
                );
                return;
            }
            if is_qualified_math_path(&path.path) {
                self.reject(Code::QualifiedMathPath, path, QUALIFIED_MATH_PATH);
                return;
            }
            if crate::names::path_is(&path.path, "Some") {
                if call.args.len() != 1 {
                    self.reject(
                        Code::SomeArguments,
                        call,
                        "`Some` takes exactly one argument",
                    );
                    return;
                }
                if self.is_tuple_expr(&call.args[0]) {
                    self.reject(
                        Code::TupleShape,
                        &call.args[0],
                        "a tuple is not the payload of a `Some`: keep the optional value in a struct with named fields",
                    );
                    return;
                }
                // The callee is the prelude constructor, not a bare path value.
                self.visit_expr(&call.args[0]);
                return;
            }
            if crate::names::path_is(&path.path, "Ok") || crate::names::path_is(&path.path, "Err") {
                if call.args.len() != 1 {
                    self.reject(
                        Code::OkErrArguments,
                        call,
                        "`Ok` and `Err` take exactly one argument",
                    );
                    return;
                }
                if self.is_tuple_expr(&call.args[0]) {
                    self.reject(
                        Code::TupleShape,
                        &call.args[0],
                        "a tuple is not the payload of an `Ok` or `Err`: return a struct with named fields",
                    );
                    return;
                }
                self.visit_expr(&call.args[0]);
                return;
            }
            if is_barrier_call(call) {
                // The statement form is taken in `visit_stmt`; a call that reaches here is part of an expression.
                self.reject(
                    Code::BarrierForm,
                    call,
                    "`workgroup_barrier()` is a statement of its own (`workgroup_barrier();`), not a value or part of \
                     an expression",
                );
                return;
            }
            if let Some(operation) = barrier::subgroup_operation(call) {
                // The position and the capability are judged by `barrier::check`, once the generic visitor passed.
                let arity = barrier::subgroup_arity(operation).unwrap_or(0);
                if call.args.len() != arity {
                    self.reject(
                        Code::BuiltinArguments,
                        call,
                        &format!("`{operation}` takes exactly {arity} argument(s) here",),
                    );
                    return;
                }
                for argument in &call.args {
                    self.visit_expr(argument);
                }
                return;
            }
            if let Some(op) = atomic_operation_name(path) {
                self.reject(
                    Code::AtomicAsValue,
                    call,
                    &if op == "atomic_add" {
                        "`atomic_add` gives its old value only as the whole initializer of a `let` (`let old = \
                         atomic_add(&mut c[i], 1u32);`); anywhere else it is supported only as a statement"
                            .to_owned()
                    } else {
                        format!("`{op}` is supported only as a statement, not as a value")
                    },
                );
                return;
            }
            if let Some((math, member)) = math_types::associated_item(&path.path)
                && path.qself.is_none()
                && member != "IDENTITY"
            {
                let expected = match member {
                    "splat" => 1,
                    _ => math_types::width(math),
                };
                if call.args.len() != expected {
                    self.reject(
                        Code::BuiltinArguments,
                        call,
                        &format!(
                            "`{}::{member}` takes exactly {expected} argument(s) here",
                            math.name
                        ),
                    );
                    return;
                }
                // The callee is the prelude constructor, not a bare path value.
                for argument in &call.args {
                    self.visit_expr(argument);
                }
                return;
            }
        }
        // `S::f(..)` and `Self::f(..)`: an associated function of an `impl` of the module (slice m9.a2). A function with
        // a receiver is called with the receiver written before the dot, so `S::method(x)` is refused.
        if let syn::Expr::Path(path) = call.func.as_ref()
            && path.qself.is_none()
            && path.path.leading_colon.is_none()
            && path.path.segments.len() == 2
            && path
                .path
                .segments
                .iter()
                .all(|segment| segment.arguments.is_none())
        {
            let first = path.path.segments[0].ident.unraw().to_string();
            let member = path.path.segments[1].ident.unraw().to_string();
            let owner = if first == "Self" {
                self.self_struct.clone()
            } else {
                Some(first)
            };
            if let Some(has_receiver) = owner.and_then(|owner| {
                self.methods
                    .members
                    .get(&format!("{owner}::{member}"))
                    .copied()
            }) {
                if has_receiver {
                    self.reject(
                        Code::UnknownCall,
                        call,
                        &format!(
                            "`{member}` takes a receiver: call it as `value.{member}(..)`, the path form is for associated functions without one"
                        ),
                    );
                    return;
                }
                for argument in &call.args {
                    self.visit_expr(argument);
                }
                return;
            }
        }
        // `f::<X>(a)` on a function of the module (slice m9.g2): type arguments are inferred from the arguments only.
        if let syn::Expr::Path(path) = call.func.as_ref()
            && path.qself.is_none()
            && path.path.leading_colon.is_none()
            && path.path.segments.len() == 1
            && !path.path.segments[0].arguments.is_none()
            && self
                .function_names
                .contains(&path.path.segments[0].ident.unraw().to_string())
        {
            self.reject(
                Code::TurbofishOnGenericCall,
                path,
                "type arguments are inferred from the arguments of a call: write `f(a)`, not `f::<X>(a)` (a generic function of the dialect has parameters whose types fix its type parameters)",
            );
            return;
        }
        let allowed = match call.func.as_ref() {
            syn::Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| self.function_names.contains(&ident.unraw().to_string())),
            _ => false,
        };
        if allowed
            && let syn::Expr::Path(path) = call.func.as_ref()
            && let Some(ident) = path.path.get_ident()
            && self.owner_fields.contains(&ident.unraw().to_string())
        {
            self.reject(
                Code::MethodNameClash,
                call,
                &format!(
                    "`{ident}` is a helper function and also a field of `{}`: inside the Slang struct the field would capture the call; rename the helper or the field",
                    self.self_struct.as_deref().unwrap_or_default()
                ),
            );
            return;
        }
        if !allowed {
            self.reject(
                Code::UnknownCall,
                call,
                "only calls to helper functions declared in the same GPU module are allowed, and `S::f(..)` or `Self::f(..)` for an associated function of a struct of the module (the path form `self::S::f(..)` is not accepted: write `S::f(..)`)",
            );
            return;
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_index(&mut self, index: &'ast syn::ExprIndex) {
        if let syn::Expr::Path(path) = index.expr.as_ref()
            && let Some(name) = path.path.get_ident()
            && self.atomic_buffers.contains(&name.unraw().to_string())
        {
            self.reject(
                Code::AtomicDirectAccess,
                index,
                "atomic buffer elements cannot be read or written directly; use an atomic operation",
            );
            return;
        }
        // The index operand is where `as usize` and an array `.len()` stand (Rust indexes an array with a `usize`; the
        // dialect has no `usize` value elsewhere, GUST-E0334). The base is visited outside it.
        for attribute in &index.attrs {
            self.visit_attribute(attribute);
        }
        self.visit_expr(&index.expr);
        self.index_depth += 1;
        self.visit_expr(&index.index);
        self.index_depth -= 1;
    }

    /// Only the operators that carry a `usize` value on keep the index operand open: a call, a method call (but the
    /// `.len()` of an array), a field or a block starts a new value, so `a[f(x as usize)]` is E0334.
    fn visit_expr(&mut self, expression: &'ast syn::Expr) {
        if self.index_depth > 0
            && !matches!(
                expression,
                syn::Expr::Paren(_)
                    | syn::Expr::Group(_)
                    | syn::Expr::Binary(_)
                    | syn::Expr::Unary(_)
                    | syn::Expr::Cast(_)
                    | syn::Expr::Lit(_)
                    | syn::Expr::Path(_)
                    | syn::Expr::Index(_)
            )
            && !matches!(expression, syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "len"))
        {
            let outer = std::mem::replace(&mut self.index_depth, 0);
            visit::visit_expr(self, expression);
            self.index_depth = outer;
            return;
        }
        visit::visit_expr(self, expression);
    }

    fn visit_type_array(&mut self, ty: &'ast syn::TypeArray) {
        if crate::slang::array_length_literal(&ty.len).is_none() {
            self.reject(
                Code::ArrayLength,
                &ty.len,
                "an array length is an integer literal greater than zero (`[f32; 4]`): not `0`, a constant or an expression",
            );
            return;
        }
        if let Some(problem) = array_element_problem(&ty.elem) {
            self.reject(Code::ArrayElement, &ty.elem, problem);
            return;
        }
        visit::visit_type_array(self, ty);
    }

    fn visit_type_slice(&mut self, ty: &'ast syn::TypeSlice) {
        self.reject(
            Code::HelperResourceParameter,
            ty,
            "slices are not supported: pass a fixed-size array `[T; N]` by value, or the values it holds",
        );
    }

    fn visit_expr_array(&mut self, array: &'ast syn::ExprArray) {
        if array.elems.is_empty() {
            self.reject(
                Code::ArrayLength,
                array,
                "an array has at least one element: `[]` has the length zero",
            );
            return;
        }
        if let Some(nested) = array.elems.iter().find(|element| {
            matches!(
                peel_expr(element),
                syn::Expr::Array(_) | syn::Expr::Repeat(_)
            )
        }) {
            self.reject(
                Code::ArrayElement,
                nested,
                "an array element cannot be an array: nested arrays are not supported, use one flat array and index `i * N + j`",
            );
            return;
        }
        visit::visit_expr_array(self, array);
    }

    fn visit_expr_repeat(&mut self, repeat: &'ast syn::ExprRepeat) {
        if crate::slang::array_length_literal(&repeat.len).is_none() {
            self.reject(
                Code::ArrayLength,
                &repeat.len,
                "the length of `[e; N]` is an integer literal greater than zero: not `0`, a constant or an expression",
            );
            return;
        }
        if matches!(
            peel_expr(&repeat.expr),
            syn::Expr::Array(_) | syn::Expr::Repeat(_)
        ) {
            self.reject(
                Code::ArrayElement,
                &repeat.expr,
                "an array element cannot be an array: nested arrays are not supported, use one flat array and index `i * N + j`",
            );
            return;
        }
        // The length is not an expression of the kernel: only the repeated operand is visited.
        self.visit_expr(&repeat.expr);
    }

    fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
        if self.reject_mutating_in_place(&assign.left) {
            return;
        }
        if self.reject_stale_index(&assign.left, &assign.right) {
            return;
        }
        visit::visit_expr_assign(self, assign);
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        // A constant shift amount of the bit width (32, every integer of the dialect is 32 bits) or more: rustc refuses
        // it (deny-by-default `arithmetic_overflow`), and naga rejects the WGSL, so the macro refuses it. A run-time
        // amount stays and masks on the GPU (KM-04).
        if matches!(
            binary.op,
            syn::BinOp::Shl(_)
                | syn::BinOp::Shr(_)
                | syn::BinOp::ShlAssign(_)
                | syn::BinOp::ShrAssign(_)
        ) && let Some(amount) = constant_shift_amount(&binary.right)
            && !(0..32).contains(&amount)
        {
            let operator = match binary.op {
                syn::BinOp::Shl(_) | syn::BinOp::ShlAssign(_) => "<<",
                _ => ">>",
            };
            let shown = if amount == i128::MAX {
                "larger than any integer".to_string()
            } else {
                amount.to_string()
            };
            self.reject(
                Code::ShiftAmountOutOfRange,
                &binary.right,
                &format!(
                    "the constant shift amount {shown} is outside 0..32, the width of a 32-bit integer: rustc refuses it (`arithmetic_overflow`) and WGSL refuses it (a constant shift amount must be below the width); use an amount in 0..32, or write the mask yourself (`x {operator} (amount & 31)`) if wrapping is wanted"
                ),
            );
            return;
        }
        // Comparing tuples is a scope choice of slice m9.t, not a limit of Slang (review m9.t S1: slangc lowers `==` and
        // the orderings of a `Tuple` with Rust's semantics, NaN included): a comparison of a visible tuple is refused
        // until a later slice admits it with GPU cases. A tuple behind an unseen name is judged by slangc.
        if matches!(
            binary.op,
            syn::BinOp::Eq(_)
                | syn::BinOp::Ne(_)
                | syn::BinOp::Lt(_)
                | syn::BinOp::Le(_)
                | syn::BinOp::Gt(_)
                | syn::BinOp::Ge(_)
        ) && (self.is_tuple_expr(&binary.left) || self.is_tuple_expr(&binary.right))
        {
            self.reject(
                Code::TupleShape,
                binary,
                "comparisons on tuples are not supported yet: compare the elements (`a.0 == b.0 && a.1 == b.1`)",
            );
            return;
        }
        // The place of an assignment or compound assignment, and the value of `*=` in a module that names a matrix
        // (lowered to `place = __gust_mul(place, value)`), evaluate a `&mut self` call in another order than Rust
        // (GUST-E0421, review m9.a2 M1). A field place with no index, `c.n += c.bump()`, matches Rust and stays (an indexed place does not: see `reject_stale_index`).
        let is_assign_op = matches!(
            binary.op,
            syn::BinOp::AddAssign(_)
                | syn::BinOp::SubAssign(_)
                | syn::BinOp::MulAssign(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
                | syn::BinOp::BitXorAssign(_)
                | syn::BinOp::BitAndAssign(_)
                | syn::BinOp::BitOrAssign(_)
                | syn::BinOp::ShlAssign(_)
                | syn::BinOp::ShrAssign(_)
        );
        if is_assign_op && self.reject_mutating_in_place(&binary.left) {
            return;
        }
        if is_assign_op && self.reject_stale_index(&binary.left, &binary.right) {
            return;
        }
        // `/=`, `%=` and `^=` lower to the `inout` helpers `__gust_div_assign`, `__gust_rem_assign` and
        // `__gust_xor_assign` (no other operator does), and slangc computes the index of an indexed place before the
        // value: `out[c.n] /= c.next()` would change another element than Rust (re-check m9.a2 R1). A place with no
        // index (`e.n /= e.bump()`) has no address to compute early and stays accepted.
        if matches!(
            binary.op,
            syn::BinOp::DivAssign(_) | syn::BinOp::RemAssign(_) | syn::BinOp::BitXorAssign(_)
        ) && place_has_index(&binary.left)
            && let Some(call) = first_mutating_call(&self.methods, &binary.right)
        {
            self.reject(
                Code::ForBoundSideEffect,
                call,
                &format!(
                    "the value of `/=`, `%=` or `^=` on an indexed place calls `.{}(..)`, which takes `&mut self`: these operators lower to an `inout` helper and Slang computes the index before the call, so another element than Rust's would change; call the method first (`let v = c.next(); out[i] /= v;`)",
                    call.method.unraw()
                ),
            );
            return;
        }
        if self.methods.matrix
            && matches!(binary.op, syn::BinOp::MulAssign(_))
            && let Some(call) = first_mutating_call(&self.methods, &binary.right)
        {
            self.reject(
                Code::ForBoundSideEffect,
                call,
                &format!(
                    "the value of `*=` calls `.{}(..)`, which takes `&mut self`: in a module that names a matrix `*=` is lowered to `place = __gust_mul(place, value)`, which reads the place before the call; write `let v = ..; x = x * v;`",
                    call.method.unraw()
                ),
            );
            return;
        }
        let is_div_or_rem = matches!(
            binary.op,
            syn::BinOp::Div(_)
                | syn::BinOp::Rem(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
        );
        if is_div_or_rem && is_literal_zero(&binary.right) {
            self.reject(
                Code::DivisionByLiteralZero,
                binary,
                "integer division or modulo by a literal zero has no portable GPU result; guard the divisor before dividing",
            );
            return;
        }
        visit::visit_expr_binary(self, binary);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if self.check_texture_call(call) {
            return;
        }
        let method = call.method.unraw().to_string();
        let in_dialect = matches!(
            method.as_str(),
            "global_id"
                | "len"
                | "is_some"
                | "is_none"
                | "is_ok"
                | "is_err"
                | "unwrap_or"
                | "unwrap"
                | "expect"
        ) || wrapping_arity(&method).is_some()
            || scalar_math_arity(&method).is_some()
            || math_types::is_swizzle(&method)
            || math_types::method_arity(&method).is_some();
        if !in_dialect
            && let Some(&facts) = self.methods.by_name.get(&call.method.unraw().to_string())
        {
            self.check_module_method_call(call, facts);
            return;
        }
        // A type argument on a method outside the dialect is that method's problem (E0305, E0325), not a built-in's.
        if call.turbofish.is_some() && in_dialect {
            self.reject(
                Code::BuiltinTypeArguments,
                call,
                "GPU built-ins do not accept type arguments",
            );
            return;
        }
        match method.as_str() {
            "is_some" | "is_none" | "is_ok" | "is_err" if call.args.is_empty() => {
                visit::visit_expr_method_call(self, call);
                return;
            }
            "unwrap_or" if call.args.len() == 1 => {
                visit::visit_expr_method_call(self, call);
                return;
            }
            "is_some" | "is_none" | "is_ok" | "is_err" => {
                self.reject(
                    Code::PredicateArguments,
                    call,
                    "`is_some`, `is_none`, `is_ok`, and `is_err` take no arguments",
                );
                return;
            }
            "unwrap_or" => {
                self.reject(
                    Code::UnwrapOrArguments,
                    call,
                    "`unwrap_or` takes exactly one argument",
                );
                return;
            }
            "unwrap" | "expect" => {
                self.reject(
                    Code::UnwrapPanics,
                    call,
                    "`Option::unwrap`/`expect` panics on `None` in Rust and GPU code cannot panic; use `unwrap_or` or `if let Some(x) = ..`",
                );
                return;
            }
            _ => {}
        }
        let math_arity = if math_types::is_swizzle(&method) {
            Some(0)
        } else {
            math_types::method_arity(&method)
        };
        if let Some(arity) = math_arity {
            if call.args.len() != arity {
                self.reject(
                    Code::BuiltinArguments,
                    call,
                    &format!("`.{method}()` takes exactly {arity} argument(s)"),
                );
                return;
            }
            visit::visit_expr_method_call(self, call);
            return;
        }
        if let Some(arity) = scalar_math_arity(&method) {
            if call.args.len() != arity {
                self.reject(
                    Code::BuiltinArguments,
                    call,
                    &format!("`.{method}()` takes exactly {arity} argument(s)"),
                );
                return;
            }
            // `min`, `max` and `clamp` are the same Slang intrinsic on every scalar type, so the operand types decide
            // the result. A literal without a suffix is `int` in Slang and whatever rustc infers in the twin, so for a
            // `u32` receiver the two differ (the trap of GUST-E0326): ask for the suffix.
            if matches!(method.as_str(), "min" | "max" | "clamp" | "rem_euclid")
                && let Some(literal) = std::iter::once(call.receiver.as_ref())
                    .chain(call.args.iter())
                    .find_map(unsuffixed_literal_only)
            {
                self.reject(
                    Code::UnsuffixedLiteralLet,
                    literal,
                    &format!(
                        "an integer literal without a suffix as an operand of `.{method}(..)` has no type of its own: \
                         Slang types it `int` while rustc infers the type from the other operand, so the two can \
                         differ; add a suffix (`x.{method}(3u32)`)"
                    ),
                );
                return;
            }
            // Slice m11.math3: `b.pow(e)` is `__gust_pow(b, uint(e))` and the base alone picks the `u32` or `i32`
            // overload, so a base literal without a suffix would be `int` in Slang and whatever rustc infers in the
            // twin: the same trap, asked for the same suffix. The exponent is always a `u32` in Rust.
            if method == "pow"
                && let Some(literal) = unsuffixed_literal_only(&call.receiver)
            {
                self.reject(
                    Code::UnsuffixedLiteralLet,
                    literal,
                    "an integer literal without a suffix as the base of `.pow(..)` has no type of its own: Slang \
                     types it `int` while rustc infers the type from the later uses, so the two can differ; add a \
                     suffix (`2u32.pow(e)`)",
                );
                return;
            }
            visit::visit_expr_method_call(self, call);
            return;
        }
        if let Some(arity) = wrapping_arity(&method) {
            if call.args.len() != arity {
                self.reject(
                    Code::BuiltinArguments,
                    call,
                    &format!("`.{method}()` takes exactly {arity} argument(s)"),
                );
                return;
            }
            visit::visit_expr_method_call(self, call);
            return;
        }
        let compatibility_builtin = matches!(method.as_str(), "global_id" | "len");
        if !call.args.is_empty() {
            if compatibility_builtin {
                self.reject(
                    Code::BuiltinArguments,
                    call,
                    &format!("`.{method}()` takes no arguments"),
                );
            } else {
                self.reject(
                    Code::MethodNotInDialect,
                    call,
                    &format!(
                        "the method `.{method}(..)` is not in the dialect; {ACCEPTED_METHODS}"
                    ),
                );
            }
            return;
        }
        if !compatibility_builtin {
            self.reject(
                Code::UnsupportedMethod,
                call,
                &format!("`.{method}()` has no lowering; {ACCEPTED_METHODS}"),
            );
            return;
        }
        let receiver = match call.receiver.as_ref() {
            syn::Expr::Path(path) => path.path.get_ident().map(|ident| ident.unraw().to_string()),
            _ => None,
        };
        // The `.len()` of an array is its `N` (slice m9.r1): a `usize` in Rust, so it stands inside an index or directly
        // under a cast to a 32-bit scalar.
        // Slice m9.r2a: the receiver may also be an array field (`s.items`, `els[i].items`, judged by the field name) or
        // an element of a buffer of arrays (`out[i]`).
        if method == "len"
            && let Some(length) = crate::slang::array_place_length(
                &call.receiver,
                &self.array_lens,
                &self.methods.array_fields,
            )
        {
            let name = call
                .receiver
                .to_token_stream()
                .to_string()
                .replace(" [", "[");
            // The emitter writes the constant `N` and drops the receiver, so a receiver that runs code (a call or a
            // method call, anywhere in its index or base) would run on the CPU twin and not on the GPU: refuse it
            // (fix round 1 of m9.r2a, M1). E0306 because the receiver is not one the length can be read from.
            if receiver.is_none() && len_receiver_runs_code(&call.receiver) {
                self.reject(
                    Code::LenReceiver,
                    call,
                    &format!(
                        "`{name}.len()` is the constant length of the array, so the call inside `{name}` would not run on the GPU: bind the element to a local first (`let e = ..; e.len()`)"
                    ),
                );
                return;
            }
            if receiver.is_none() {
                self.visit_expr(&call.receiver);
            }
            if length.is_none() {
                self.reject(
                    Code::LenReceiver,
                    call,
                    &format!(
                        "`{name}` is bound more than once in this function with another length or as something else (for a field: two structs give the field name another length, or one gives it no array), so `.len()` cannot say which length it means: give each array its own name"
                    ),
                );
            } else if self.index_depth == 0
                && !self
                    .len_casts
                    .contains(&(std::ptr::from_ref(call) as usize))
            {
                self.reject(
                    Code::UsizeOutsideIndex,
                    call,
                    &format!(
                        "`{name}.len()` is a `usize` in Rust, which the dialect has no value for: use it inside an index (`a[a.len() - 1]`) or cast it (`{name}.len() as u32`)"
                    ),
                );
            }
            return;
        }
        let valid = receiver.as_ref().is_some_and(|name| {
            if method == "len" {
                self.resources.contains(name)
            } else {
                self.invocations.contains(name)
            }
        });
        if !valid {
            if method == "len" {
                self.reject(
                    Code::LenReceiver,
                    call,
                    "`.len()` requires a direct storage parameter receiver, or an array local or parameter with a written type (`let g: [f32; 4] = ..;`)",
                );
            } else {
                self.reject(
                    Code::GlobalIdReceiver,
                    call,
                    "`.global_id()` requires a direct dispatch-thread parameter receiver",
                );
            }
            return;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        // `params.scale` and `params.view.mul_vec4(p)` read one field of a uniform: the base is not a whole-value use.
        if self.is_uniform_parameter(&field.base) {
            // `u.0` is the wrapper's own field: it reaches the whole struct (and the twin could write through it).
            if matches!(field.member, syn::Member::Unnamed(_)) {
                self.reject(
                    Code::UniformWholeValue,
                    field,
                    "a uniform parameter is read one field at a time by name (`params.scale`); `.0` reaches the \
                     whole struct",
                );
            }
            return;
        }
        // `t.0` .. `t.7`: a tuple has at most eight elements (a tuple struct is refused as GUST-E0104, so an unnamed
        // field is a tuple element).
        if let syn::Member::Unnamed(index) = &field.member
            && index.index as usize >= MAX_TUPLE_ELEMENTS
        {
            self.reject(
                Code::TupleShape,
                field,
                &format!(
                    "a tuple has at most {MAX_TUPLE_ELEMENTS} elements, so `.{}` names none",
                    index.index
                ),
            );
            return;
        }
        visit::visit_expr_field(self, field);
    }

    fn visit_type_tuple(&mut self, ty: &'ast syn::TypeTuple) {
        // `()` is the unit type: the result of a function without a value.
        if ty.elems.is_empty() {
            return;
        }
        if ty.elems.len() < 2 || ty.elems.len() > MAX_TUPLE_ELEMENTS {
            self.reject(
                Code::TupleShape,
                ty,
                &format!(
                    "a tuple type has 2 to {MAX_TUPLE_ELEMENTS} elements, this one has {}: group the values in a struct with named fields",
                    ty.elems.len()
                ),
            );
            return;
        }
        for element in &ty.elems {
            if crate::slang::parameter_flavor(peel_type(element))
                != crate::slang::ParameterFlavor::Value
            {
                self.reject(
                    Code::TupleInResource,
                    element,
                    "a tuple element must be a value type: a buffer, uniform, workgroup array, texture, sampler or invocation id is a resource of the kernel, not a value to carry in a tuple",
                );
                return;
            }
            if let Some(problem) = tuple_element_problem(element) {
                self.reject(Code::TupleShape, element, problem);
                return;
            }
        }
        visit::visit_type_tuple(self, ty);
    }

    fn visit_expr_tuple(&mut self, tuple: &'ast syn::ExprTuple) {
        if tuple.elems.is_empty() {
            visit::visit_expr_tuple(self, tuple);
            return;
        }
        if tuple.elems.len() < 2 || tuple.elems.len() > MAX_TUPLE_ELEMENTS {
            self.reject(
                Code::TupleShape,
                tuple,
                &format!(
                    "a tuple has 2 to {MAX_TUPLE_ELEMENTS} elements, this one has {}: group the values in a struct with named fields",
                    tuple.elems.len()
                ),
            );
            return;
        }
        // A resource parameter by name is no value either (the values of a buffer element are read through it).
        if let Some(resource) = tuple.elems.iter().find(|element| {
            matches!(peel_expr(element), syn::Expr::Path(path)
            if path.qself.is_none()
                && path.path.get_ident().is_some_and(|name| {
                    let name = name.unraw().to_string();
                    self.resources.contains(&name)
                        || self.uniforms.contains(&name)
                        || self.textures.contains_key(&name)
                        || self.samplers.contains(&name)
                }))
        }) {
            self.reject(
                Code::TupleInResource,
                resource,
                "a tuple element must be a value type: a buffer, uniform, texture or sampler parameter is a resource of the kernel, not a value to carry in a tuple",
            );
            return;
        }
        // Slang types `makeTuple(1, x)` with an `int` element where rustc may infer `u32` from the use of the tuple.
        if let Some(literal) = tuple.elems.iter().find_map(unsuffixed_literal_only) {
            self.reject(
                Code::UnsuffixedLiteralLet,
                literal,
                "an integer literal without a suffix in a tuple gives the element no type of its own: Slang types it `int` while rustc infers the type from the later uses, so the two can differ; add a suffix (`1u32`, `-1i32`)",
            );
            return;
        }
        visit::visit_expr_tuple(self, tuple);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        // A texture or a sampler is the receiver of a texture method (or the sampler argument of `sample`) and
        // nothing else; those uses never reach this function (`check_texture_call`).
        if path.qself.is_none()
            && let Some(name) = path.path.get_ident().map(|ident| ident.unraw().to_string())
            && (self.textures.contains_key(&name) || self.samplers.contains(&name))
        {
            let uses = if self.samplers.contains(&name) {
                format!(
                    "`{name}` is a sampler parameter: it is used as `texture.sample({name}, u, v)` only"
                )
            } else {
                format!(
                    "`{name}` is a texture parameter: it is used as `{name}.load(x, y)`, \
                     `{name}.store(x, y, v)`, `{name}.sample(sampler, u, v)`, `{name}.width()` or \
                     `{name}.height()` only"
                )
            };
            self.reject(
                Code::TextureUse,
                path,
                &format!("{uses}; it cannot be copied, passed or returned"),
            );
            return;
        }
        // Any other use of a uniform parameter by name copies or passes the whole struct: slangc crashes on a whole
        // struct copy out of a constant buffer on WGSL, and the twin would accept what the GPU cannot compile.
        if path.qself.is_none()
            && path
                .path
                .get_ident()
                .is_some_and(|name| self.uniforms.contains(&name.unraw().to_string()))
        {
            self.reject(
                Code::UniformWholeValue,
                path,
                "a uniform parameter is read one field at a time (`params.scale`, `params.view.mul_vec4(p)`); \
                 copying or passing the whole struct (`*params`, `let p = params;`) is not supported",
            );
            return;
        }
        // A closure is called and is nothing else (slice m9.c); a call reaches `visit_expr_call`, not this function.
        if path.qself.is_none()
            && let Some(name) = path.path.get_ident().map(|name| name.unraw().to_string())
            && self.closures.contains(&name)
        {
            self.reject(
                Code::ClosureValue,
                path,
                &format!(
                    "`{name}` is a closure: it can only be called, `{name}(a, b)`; it cannot be passed, returned, stored, \
                     copied or compared"
                ),
            );
            return;
        }
        // `E::V` names a variant of an enum of the module (slice m9.e1).
        if path.qself.is_none() && self.variant_path(&path.path).is_some() {
            return;
        }
        // A bare path value must be a single identifier (a local variable or
        // parameter). Multi-segment paths are associated constants or foreign
        // items (e.g. `f32::INFINITY`, `u32::MAX`), which have no Slang lowering.
        // Callee paths are handled by `visit_expr_call`, which only descends for
        // single-identifier helpers, so this fires only for bare values.
        if path.qself.is_none()
            && let Some((_, "IDENTITY")) = math_types::associated_item(&path.path)
        {
            return;
        }
        if path.qself.is_none() && math_constant(&path.path).is_some() {
            return;
        }
        if path.qself.is_none() && crate::names::path_is(&path.path, "workgroup_barrier") {
            self.reject(
                Code::BarrierForm,
                path,
                "`workgroup_barrier` is called as a statement (`workgroup_barrier();`), not used as a value",
            );
        } else if is_qualified_math_path(&path.path) {
            self.reject(Code::QualifiedMathPath, path, QUALIFIED_MATH_PATH);
        } else if path.path.get_ident().is_none() {
            self.reject(
                Code::ForeignPath,
                path,
                "associated constants and foreign paths are not in the Rust-to-Slang subset; reference local variables, parameters, or same-module helpers",
            );
        } else if crate::names::path_is(&path.path, "Some")
            || crate::names::path_is(&path.path, "Ok")
            || crate::names::path_is(&path.path, "Err")
        {
            self.reject(
                Code::VariantAsValue,
                path,
                "`Some`, `Ok`, and `Err` must be called with one argument",
            );
        }
    }

    fn visit_expr_if(&mut self, expression: &'ast syn::ExprIf) {
        let syn::Expr::Let(binding) = expression.cond.as_ref() else {
            visit::visit_expr_if(self, expression);
            return;
        };
        let name = option_pattern(&binding.pat).or_else(|| result_pattern(&binding.pat));
        let Some(name) = name else {
            self.reject(
                Code::IfLetPattern,
                &binding.pat,
                "`if let` supports only the `Some(identifier)` or `Ok(identifier)` pattern",
            );
            return;
        };
        let text = name.ident.unraw().to_string();
        if self.is_parameter_name(&text) {
            self.reject(
                Code::ShadowedParameter,
                name,
                "GPU locals cannot shadow resource or dispatch-thread parameters",
            );
            return;
        }
        // `if let Some(mut v) = Some(7)` binds a local of the scrutinee's payload type, which is as untyped as the
        // initializer of a `let` (GUST-E0326).
        if let Some(literal) = unsuffixed_literal_only(&binding.expr) {
            self.reject(
                Code::UnsuffixedLiteralLet,
                literal,
                "an integer literal without a suffix or a type annotation gives the bound value no type of its own: \
                 Slang types it `int` while rustc infers the type from the later uses, so the two can differ; add a \
                 suffix (`Some(7u32)`)",
            );
            return;
        }
        self.visit_pat_ident(name);
        self.visit_expr(&binding.expr);
        // The bound name is not an enum value inside the block (it may shadow one), and the outer meaning returns after.
        let outer = self.enum_values.clone();
        let outer_tuples = self.tuple_values.clone();
        self.enum_values.remove(&text);
        self.tuple_values.remove(&text);
        self.visit_block(&expression.then_branch);
        self.enum_values = outer;
        self.tuple_values = outer_tuples;
        if let Some((_, alternative)) = &expression.else_branch {
            self.visit_expr(alternative);
        }
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        // Reached for a `match` anywhere but a statement or the initializer of a `let` (`visit_local`): an argument, an
        // operand, an assigned value, a `return`, a bound.
        if self.reject_tuple_scrutinee(expression) {
            return;
        }
        if is_switch_match(expression) {
            self.reject(
                Code::ValueMatchForm,
                expression,
                "a `match` that gives a value is accepted only as the initializer of a typed `let` (or of an untyped `let` whose arms show the type): `let x: u32 = match e { E::A => 1u32, _ => 2u32 };`; bind it to a typed local first, then use the local",
            );
            return;
        }
        self.reject(
            Code::MatchAsValue,
            expression,
            "a `match` on a `Result` is supported only as a statement, not as a value",
        );
    }

    fn visit_expr_let(&mut self, expression: &'ast syn::ExprLet) {
        // Reached only outside the direct `if let` condition position.
        self.reject(
            Code::IfLetPosition,
            expression,
            "`if let` is supported only as `if let Some(x) = value { .. }` or `if let Ok(x) = value { .. }`; let chains and other let positions are not",
        );
    }

    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        if expression.rest.is_some() {
            self.reject(
                Code::StructUpdateSyntax,
                expression,
                "struct update syntax (`..base`) is not supported; list every field explicitly",
            );
            return;
        }
        visit::visit_expr_struct(self, expression);
    }

    fn visit_expr_try(&mut self, expression: &'ast syn::ExprTry) {
        self.reject(
            Code::TryOperator,
            expression,
            "the `?` operator is not in the Rust-to-Slang subset",
        );
    }

    fn visit_expr_unsafe(&mut self, expression: &'ast syn::ExprUnsafe) {
        self.reject(
            Code::UnsafeBlock,
            expression,
            "unsafe blocks are not in the Rust-to-Slang subset",
        );
    }

    fn visit_expr_macro(&mut self, expression: &'ast syn::ExprMacro) {
        self.reject(
            Code::MacroInModule,
            expression,
            "macros cannot run inside a Rust-to-Slang module",
        );
    }

    fn visit_expr_closure(&mut self, expression: &'ast syn::ExprClosure) {
        // Reached for a closure that is not the initializer of a plain `let` (that one is `visit_local`'s): an argument,
        // a result, a field, an operand, a parenthesized initializer (slice m9.c).
        self.reject(
            Code::ClosureValue,
            expression,
            "a closure is accepted only as the initializer of a `let` (`let f = |x: f32| -> f32 { x * 2.0f32 };`) \
             and is then called as `f(a)`: it cannot be passed, returned, stored or written where a value is expected",
        );
    }

    fn visit_expr_loop(&mut self, expression: &'ast syn::ExprLoop) {
        self.reject(
            Code::LoopNotSupported,
            expression,
            "`loop` is not supported: its trip count is not fixed at entry, so a GPU thread could never be proven to terminate; use `for i in start..end`",
        );
    }

    fn visit_expr_for_loop(&mut self, expression: &'ast syn::ExprForLoop) {
        if let Some(label) = &expression.label {
            self.reject(
                Code::LoopLabel,
                label,
                "loop labels are not supported in the Rust-to-Slang subset; restructure with a flag or a helper return",
            );
            return;
        }
        let variable = match loop_variable(&expression.pat) {
            Ok(variable) => variable,
            Err((code, message)) => {
                self.reject(code, &expression.pat, message);
                return;
            }
        };
        if let Some(variable) = variable {
            let name = variable.ident.unraw().to_string();
            if self.is_parameter_name(&name) {
                self.reject(
                    Code::ShadowedParameter,
                    variable,
                    "GPU locals cannot shadow resource or dispatch-thread parameters",
                );
                return;
            }
            self.visit_pat_ident(variable);
        }
        let (start, end) = match range_bounds(&expression.expr) {
            Ok(bounds) => bounds,
            Err((code, message)) => {
                self.reject(code, &expression.expr, message);
                return;
            }
        };
        for bound in [start, end] {
            if is_unsuffixed_integer_literal(bound) {
                self.reject(
                    Code::UnsuffixedLoopBound,
                    bound,
                    "integer-literal loop bounds need an explicit `u32`/`i32` suffix so the Slang counter type matches rustc's inference",
                );
                return;
            }
        }
        // The emitted loop evaluates the end bound before the start bound (spike M9.s0 P9, measured on both targets),
        // where Rust evaluates the start first: a bound that changes a value is refused, not reordered (GUST-E0421).
        for bound in [start, end] {
            if let Some(call) = first_mutating_call(&self.methods, bound) {
                self.reject(
                    Code::ForBoundSideEffect,
                    call,
                    &format!(
                        "the bound of a `for` range calls `.{}(..)`, which takes `&mut self`: the Slang loop evaluates its end bound before its start bound, so the order of the effects would differ from Rust's; call the method before the loop and use the value (`let n = c.next(); for i in 0u32..n {{ .. }}`)",
                        call.method.unraw()
                    ),
                );
                return;
            }
            if let Some(call) = first_closure_call(&self.closures, bound) {
                self.reject(
                    Code::ForBoundSideEffect,
                    call,
                    &format!(
                        "the bound of a `for` range calls the closure `{}`: the Slang loop evaluates its end bound before its start bound, and the dialect keeps the bounds of a loop to plain values; compute the value before the loop (`let n = {}(..); for i in 0u32..n {{ .. }}`)",
                        call.func.to_token_stream(),
                        call.func.to_token_stream()
                    ),
                );
                return;
            }
        }
        self.visit_expr(start);
        self.visit_expr(end);
        self.loop_depth += 1;
        let outer = self.enum_values.clone();
        let outer_tuples = self.tuple_values.clone();
        if let Some(variable) = variable {
            self.enum_values.remove(&variable.ident.unraw().to_string());
            self.tuple_values
                .remove(&variable.ident.unraw().to_string());
        }
        self.visit_block(&expression.body);
        self.enum_values = outer;
        self.tuple_values = outer_tuples;
        self.loop_depth -= 1;
    }

    fn visit_block(&mut self, block: &'ast syn::Block) {
        // An enum local lives to the end of its block: a `let` inside it that shadows an outer name is forgotten after,
        // and a shadow of a value parameter ends with its block too (the parameter is a by-value `in` again).
        let outer = self.enum_values.clone();
        let outer_tuples = self.tuple_values.clone();
        let outer_closures = self.closures.clone();
        let outer_value_params = self.value_params.clone();
        visit::visit_block(self, block);
        self.enum_values = outer;
        self.tuple_values = outer_tuples;
        self.closures = outer_closures;
        self.value_params = outer_value_params;
    }

    fn visit_expr_while(&mut self, expression: &'ast syn::ExprWhile) {
        self.reject(
            Code::WhileNotSupported,
            expression,
            "`while` loops are not supported: the trip count is not fixed at entry, so a GPU thread could never be proven to terminate; use `for i in start..end` with a `break`",
        );
    }

    fn visit_expr_range(&mut self, expression: &'ast syn::ExprRange) {
        // `visit_expr_for_loop` reads its bounds directly, so this is reached only
        // for a range used as a value.
        self.reject(
            Code::RangeAsValue,
            expression,
            "ranges are supported only as the iterable of a `for` loop",
        );
    }

    fn visit_stmt(&mut self, statement: &'ast syn::Stmt) {
        match statement {
            syn::Stmt::Expr(syn::Expr::Match(expression), None)
                if is_switch_match(expression)
                    && self.body_tail_match == Some(std::ptr::from_ref(expression) as usize)
                    && expression.arms.iter().any(|arm| match arm.body.as_ref() {
                        syn::Expr::Block(body) => value_tail(&body.block).is_some(),
                        _ => true,
                    }) =>
            {
                self.reject(
                    Code::ValueMatchForm,
                    expression,
                    "a `match` that gives a value is accepted only as the initializer of a typed `let`, not as the tail of a function (an untyped `let` whose arms show the type takes it too): `let x: u32 = match e { E::A => 1u32, _ => 2u32 }; x`; bind it to a typed local first, then return the local",
                );
            }
            syn::Stmt::Expr(syn::Expr::Match(expression), _) if is_switch_match(expression) => {
                self.check_switch_match(expression, MatchForm::Statement);
            }
            syn::Stmt::Expr(syn::Expr::Match(expression), semicolon) => {
                if self.reject_tuple_scrutinee(expression) {
                    return;
                }
                let (ok_name, _, err_name, _) = match result_match_arms(expression) {
                    Ok(arms) => arms,
                    Err(error) => {
                        self.errors.push(error);
                        return;
                    }
                };
                if semicolon.is_none() {
                    self.reject(
                        Code::MatchNeedsSemicolon,
                        expression,
                        "Result match is supported only as a statement; terminate it with `;`",
                    );
                    return;
                }
                if !self.is_known_result(&expression.expr) {
                    self.reject(
                        Code::MatchScrutinee,
                        &expression.expr,
                        "statement match is supported only for a known `Result<T, T>` local, parameter, or same-module helper call",
                    );
                    return;
                }
                for name in [ok_name, err_name] {
                    let text = name.ident.unraw().to_string();
                    if self.is_parameter_name(&text) {
                        self.reject(
                            Code::ShadowedParameter,
                            name,
                            "GPU locals cannot shadow resource or dispatch-thread parameters",
                        );
                        return;
                    }
                    self.visit_pat_ident(name);
                }
                self.visit_expr(&expression.expr);
                for arm in &expression.arms {
                    let syn::Expr::Block(body) = arm.body.as_ref() else {
                        unreachable!("result_match_arms checked block bodies");
                    };
                    self.visit_block(&body.block);
                }
            }
            syn::Stmt::Expr(syn::Expr::Call(call), semicolon) if is_barrier_call(call) => {
                if !call.args.is_empty() || semicolon.is_none() {
                    self.reject(
                        Code::BarrierForm,
                        call,
                        "`workgroup_barrier()` takes no arguments and is a statement ending in `;`",
                    );
                }
            }
            syn::Stmt::Expr(syn::Expr::Call(call), _) => {
                if let syn::Expr::Path(path) = call.func.as_ref()
                    && let Some(op) = atomic_operation_name(path)
                {
                    self.validate_atomic(op, call);
                    return;
                }
                visit::visit_stmt(self, statement);
            }
            syn::Stmt::Expr(syn::Expr::Break(jump), _) => {
                if let Some(value) = &jump.expr {
                    self.reject(
                        Code::BreakWithValue,
                        value,
                        "`break` with a value is not supported",
                    );
                    return;
                }
                self.check_jump(jump, jump.label.as_ref(), "break");
            }
            syn::Stmt::Expr(syn::Expr::Continue(jump), _) => {
                self.check_jump(jump, jump.label.as_ref(), "continue");
            }
            _ => visit::visit_stmt(self, statement),
        }
    }

    fn visit_expr_break(&mut self, expression: &'ast syn::ExprBreak) {
        // Statement-position jumps are handled in `visit_stmt`.
        self.reject(
            Code::JumpAsValue,
            expression,
            "`break` and `continue` are supported only as statements, not as values",
        );
    }

    fn visit_expr_continue(&mut self, expression: &'ast syn::ExprContinue) {
        self.reject(
            Code::JumpAsValue,
            expression,
            "`break` and `continue` are supported only as statements, not as values",
        );
    }

    fn visit_expr_await(&mut self, expression: &'ast syn::ExprAwait) {
        self.reject(
            Code::Await,
            expression,
            "await is CPU-only and cannot appear in GPU code",
        );
    }

    fn visit_expr_unary(&mut self, expression: &'ast syn::ExprUnary) {
        // `*x` dereferences a reference, and no value of the dialect is one (`&self` is lowered to `this`, which is not
        // a pointer), so Slang would be handed `*this`: refused here with the code of an unsupported expression.
        let before = self.errors.count();
        visit::visit_expr_unary(self, expression);
        // The operand is judged first (`*u` on a whole uniform is GUST-E0321), and the dereference only when nothing
        // in the operand was refused, so one expression gives one report.
        let after = self.errors.count();
        if matches!(expression.op, syn::UnOp::Deref(_)) && before == after {
            self.reject(
                Code::UnsupportedExpression,
                expression,
                "the dereference `*` is not part of the Rust-to-Slang value model: use the value (`self.field`, a local copy) instead of `*self`",
            );
        }
    }

    // The `Self` type of a shorthand receiver is not written in the source: the receiver is judged by its own rules.
    fn visit_receiver(&mut self, _receiver: &'ast syn::Receiver) {}

    fn visit_expr_reference(&mut self, expression: &'ast syn::ExprReference) {
        self.reject(
            Code::Reference,
            expression,
            "references are not part of the Rust-to-Slang value model",
        );
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some((tuple, annotation)) = crate::slang::tuple_pattern(&local.pat) {
            self.visit_tuple_local(local, tuple, annotation);
            return;
        }
        if let Ok((pattern, _)) = crate::slang::local_binding(&local.pat) {
            let name = pattern.ident.unraw().to_string();
            if self.is_parameter_name(&name) {
                self.reject(
                    Code::ShadowedParameter,
                    local,
                    "GPU locals cannot shadow resource or dispatch-thread parameters",
                );
                return;
            }
            // A local of the name of a value parameter is a binding of its own: `let mut p = p;` is how a `&mut self`
            // method gets a place to work on (the rule is by name; `visit_block` restores the set when the block of the shadow ends).
            self.value_params.remove(&pattern.ident.unraw().to_string());
        }
        if crate::slang::local_binding(&local.pat).is_err() {
            self.reject(
                Code::LetNeedsIdentifier,
                local,
                "GPU let bindings require a single identifier",
            );
            return;
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
        // `let f = |x: T| -> R { .. };` (slice m9.c): the name is bound and the body is not visited here. The closure
        // pass of `validate_function` judges the form, the names, the captures and the body as the body of a helper,
        // once this visit found nothing else.
        if let Some((pattern, closure)) = crate::slang::closure_binding(local) {
            let key = pattern.ident.unraw().to_string();
            if let syn::ReturnType::Type(_, ty) = &closure.output {
                if let Some(name) = enum_type_name(ty, &self.methods.enums) {
                    self.methods.enum_returns.insert(key.clone(), name);
                }
                if is_tuple_type(ty) {
                    self.methods.tuple_returns.insert(key.clone());
                }
                if matches!(ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some()) {
                    self.result_helpers.insert(key.clone());
                }
            }
            self.enum_values.remove(&key);
            self.tuple_values.remove(&key);
            self.closures.insert(key);
            return;
        }
        if matches!(local.pat, syn::Pat::Ident(_))
            && let Some(literal) = local
                .init
                .as_ref()
                .and_then(|initial| unsuffixed_literal_only(&initial.expr))
        {
            self.reject(
                Code::UnsuffixedLiteralLet,
                literal,
                "an integer literal without a suffix or a type annotation gives the local no type of its own: Slang \
                 types it `int` while rustc infers the type from the later uses, so the two can differ; add a suffix \
                 (`0u32`, `0i32`) or a type annotation (`let x: u32 = 0;`)",
            );
            return;
        }
        if let Some(value_match) = local
            .init
            .as_ref()
            .and_then(|initial| switch_match_of(&initial.expr))
        {
            // The value form (slices m9.e2 and m13.matchu): the initializer of a `let`, and of nothing else. Whether
            // an untyped `let` has a type its arms show is decided where the type is inferred, when Slang is emitted.
            self.visit_pat(&local.pat);
            self.check_switch_match(value_match, MatchForm::Value);
        } else if let Some(call) = local
            .init
            .as_ref()
            .and_then(|initial| fetch_add_call(&initial.expr))
        {
            // The old value of `atomic_add` (slice m11.fetch): the whole initializer of a `let`, and nothing else. In
            // any other position the value stays refused (`GUST-E0606`), so no order-sensitive form is reachable.
            self.visit_pat(&local.pat);
            self.validate_atomic("atomic_add", call);
        } else {
            visit::visit_local(self, local);
        }
        // The enum a local holds (by name; a rebinding to anything else forgets it).
        if let Ok((binding, _)) = crate::slang::local_binding(&local.pat) {
            let annotated = match &local.pat {
                syn::Pat::Type(typed) => enum_type_name(&typed.ty, &self.methods.enums),
                _ => None,
            };
            let held = annotated.or_else(|| {
                local
                    .init
                    .as_ref()
                    .and_then(|initial| self.enum_of(&initial.expr))
            });
            let key = binding.ident.unraw().to_string();
            match held {
                Some(enum_name) => {
                    self.enum_values.insert(key.clone(), enum_name);
                }
                None => {
                    self.enum_values.remove(&key);
                }
            }
            // The tuple a local holds (by name, like the enum): an annotation or a visible tuple initializer.
            let tuple_held = matches!(&local.pat, syn::Pat::Type(typed) if is_tuple_type(&typed.ty))
                || local
                    .init
                    .as_ref()
                    .is_some_and(|initial| self.is_tuple_expr(&initial.expr));
            if tuple_held {
                self.tuple_values.insert(key);
            } else {
                self.tuple_values.remove(&key);
            }
        }
        let (pattern, annotated_result) = match &local.pat {
            syn::Pat::Ident(pattern) => (pattern, false),
            syn::Pat::Type(typed) => {
                let syn::Pat::Ident(pattern) = typed.pat.as_ref() else {
                    return;
                };
                let result = matches!(typed.ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some());
                (pattern, result)
            }
            _ => return,
        };
        let initialized_result = local
            .init
            .as_ref()
            .is_some_and(|initial| self.is_known_result(&initial.expr));
        if annotated_result || initialized_result {
            self.result_values.insert(pattern.ident.unraw().to_string());
        }
    }
}

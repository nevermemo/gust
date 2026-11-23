//! Predicates over single expressions: addresses, coordinates, atomics, literals, shifts and places.

use super::*;

/// Why `expression` is visibly not a `u32` texel address (GUST-E0732), or `None` when it may be one: the macro sees no
/// types, so it refuses what is plainly another type and leaves the rest to rustc, which checks the shadow body.
pub(super) fn address_problem(expression: &syn::Expr) -> Option<&'static str> {
    match peel_expr(expression) {
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Float(_) => Some("a float literal"),
            syn::Lit::Int(int) if !matches!(int.suffix(), "" | "u32") => {
                Some("an integer literal that is not a `u32`")
            }
            _ => None,
        },
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            Some("a negated value (a texel address cannot be negative)")
        }
        syn::Expr::Cast(cast) => match cast.ty.as_ref() {
            syn::Type::Path(path)
                if crate::names::path_is(&path.path, "u32")
                    || crate::names::path_is(&path.path, "uint") =>
            {
                None
            }
            _ => Some("a cast to a type other than `u32`"),
        },
        _ => None,
    }
}

/// Why `expression` is visibly not an `f32` sample coordinate (GUST-E0732), or `None`: the counterpart of
/// [`address_problem`].
pub(super) fn coordinate_problem(expression: &syn::Expr) -> Option<&'static str> {
    match peel_expr(expression) {
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Int(_) => {
                Some("an integer literal (write `0.5`, or `1.0` for a whole number)")
            }
            syn::Lit::Float(float) if !matches!(float.suffix(), "" | "f32") => {
                Some("a float literal that is not an `f32`")
            }
            _ => None,
        },
        syn::Expr::Cast(cast) => match cast.ty.as_ref() {
            syn::Type::Path(path)
                if crate::names::path_is(&path.path, "f32")
                    || crate::names::path_is(&path.path, "float") =>
            {
                None
            }
            _ => Some("a cast to a type other than `f32`"),
        },
        _ => None,
    }
}

/// The atomic operation name if the path is a recognized atomic intrinsic.
pub(super) fn atomic_operation_name(path: &syn::ExprPath) -> Option<&'static str> {
    match path.path.get_ident()?.unraw().to_string().as_str() {
        "atomic_add" => Some("atomic_add"),
        "atomic_min" => Some("atomic_min"),
        "atomic_max" => Some("atomic_max"),
        "atomic_exchange" => Some("atomic_exchange"),
        "atomic_compare_exchange" => Some("atomic_compare_exchange"),
        _ => None,
    }
}

/// The call, when `expression` is exactly an `atomic_add(..)` call: the one atomic whose old value a `let` may bind
/// (slice m11.fetch). Parentheses and groups around it are not looked through, so the form stays one spelling.
pub(super) fn fetch_add_call(expression: &syn::Expr) -> Option<&syn::ExprCall> {
    let syn::Expr::Call(call) = expression else {
        return None;
    };
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    (atomic_operation_name(path) == Some("atomic_add")).then_some(call)
}

/// Whether an expression is an integer literal with value zero, including a
/// unary negation, parentheses and a macro-expanded group around one (the spellings the emitter keeps as a plain
/// operator, so a zero divisor never reaches slangc). Float zeros are excluded: floating-point division by
/// zero is defined (it yields infinity/NaN), so only integer div/rem by zero is
/// a portable-semantic hazard.
pub(crate) fn is_literal_zero(expression: &syn::Expr) -> bool {
    match peel_expr(expression) {
        syn::Expr::Lit(literal) => matches!(
            &literal.lit,
            syn::Lit::Int(value) if value.base10_digits().bytes().all(|digit| digit == b'0')
        ),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            is_literal_zero(&unary.expr)
        }
        _ => false,
    }
}

/// The value of a constant shift amount: an integer literal (any suffix or none), through parentheses, groups, unary
/// `-` and `as` casts to `u32`/`uint` or `i32`/`int` (the cast wraps as in Rust), or `None` for anything else. A literal
/// too large for `i128` saturates, so it is refused. No arithmetic is folded: a sum of literals, a local bound to a
/// literal or a float literal cast passes the macro (a known gap): Slang folds it to a constant, naga rejects the
/// WGSL, and rustc refuses it when building (`arithmetic_overflow`, not under `cargo check`). Only an amount Slang
/// cannot fold (from a buffer, a parameter, a loop variable) is run-time, and only that kind masks on the GPU (KM-04).
pub(super) fn constant_shift_amount(expression: &syn::Expr) -> Option<i128> {
    match peel_expr(expression) {
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Int(value) => Some(value.base10_parse::<i128>().unwrap_or(i128::MAX)),
            _ => None,
        },
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            constant_shift_amount(&unary.expr).map(|value| value.saturating_neg())
        }
        syn::Expr::Cast(cast) => {
            let value = constant_shift_amount(&cast.expr)?;
            let syn::Type::Path(target) = peel_type(&cast.ty) else {
                return None;
            };
            if crate::names::path_is(&target.path, "u32")
                || crate::names::path_is(&target.path, "uint")
            {
                Some(i128::from(value as u32))
            } else if crate::names::path_is(&target.path, "i32")
                || crate::names::path_is(&target.path, "int")
            {
                Some(i128::from(value as i32))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A loop bound that is a bare integer literal (through parentheses and unary
/// minus) without a type suffix. Slang would type it `int` while rustc may infer
/// `u32` from the other bound, so the counter types would silently diverge.
pub(super) fn is_unsuffixed_integer_literal(expression: &syn::Expr) -> bool {
    match peel_expr(expression) {
        syn::Expr::Lit(literal) => {
            matches!(&literal.lit, syn::Lit::Int(value) if value.suffix().is_empty())
        }
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            is_unsuffixed_integer_literal(&unary.expr)
        }
        _ => false,
    }
}

/// The first unsuffixed integer literal of an initializer whose type nothing in the initializer itself fixes, or `None`
/// when some operand is typed (a suffix, a cast, a name, a call, a field, an index) or the expression is something
/// else. Covered: a bare literal, through parentheses, groups, unary `-` and `!`; arithmetic and bitwise binary
/// operators with a literal-only operand on each side; a shift with a literal-only left operand (the result takes the
/// left operand's type, whatever the right one is); `Some(..)`, `Ok(..)` and `Err(..)` of a literal-only payload;
/// `X.unwrap_or(Y)` with both literal-only (`Some(7).unwrap_or(0)`). Also applied to an `if let` scrutinee.
/// Comparisons and `&&`/`||` give `bool`, so they are not covered. Slang types such an initializer `int`, while rustc
/// types the local from its later uses (`u32` for `d /= y` with `y: u32`), so the two would silently diverge (D86).
pub(super) fn unsuffixed_literal_only(expression: &syn::Expr) -> Option<&syn::Expr> {
    match expression {
        syn::Expr::Lit(literal) => {
            matches!(&literal.lit, syn::Lit::Int(value) if value.suffix().is_empty())
                .then_some(expression)
        }
        syn::Expr::Paren(paren) => unsuffixed_literal_only(&paren.expr),
        syn::Expr::Group(group) => unsuffixed_literal_only(&group.expr),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_) | syn::UnOp::Not(_)) => {
            unsuffixed_literal_only(&unary.expr)
        }
        syn::Expr::Binary(binary)
            if matches!(binary.op, syn::BinOp::Shl(_) | syn::BinOp::Shr(_)) =>
        {
            unsuffixed_literal_only(&binary.left)
        }
        syn::Expr::Binary(binary)
            if matches!(
                binary.op,
                syn::BinOp::Add(_)
                    | syn::BinOp::Sub(_)
                    | syn::BinOp::Mul(_)
                    | syn::BinOp::Div(_)
                    | syn::BinOp::Rem(_)
                    | syn::BinOp::BitAnd(_)
                    | syn::BinOp::BitOr(_)
                    | syn::BinOp::BitXor(_)
            ) =>
        {
            let left = unsuffixed_literal_only(&binary.left)?;
            unsuffixed_literal_only(&binary.right)?;
            Some(left)
        }
        syn::Expr::Call(call)
            if call.args.len() == 1
                && matches!(
                    call.func.as_ref(),
                    syn::Expr::Path(path)
                        if crate::names::path_is(&path.path, "Some") || crate::names::path_is(&path.path, "Ok") || crate::names::path_is(&path.path, "Err")
                ) =>
        {
            unsuffixed_literal_only(&call.args[0])
        }
        // `Some(7).unwrap_or(0)`: the receiver is an `Option<{integer}>`, which rustc accepts, and the default is
        // the same unconstrained integer. (On a bare integer receiver rustc itself refuses the method call.)
        syn::Expr::MethodCall(call)
            if crate::names::ident_is(&call.method, "unwrap_or") && call.args.len() == 1 =>
        {
            let receiver = unsuffixed_literal_only(&call.receiver)?;
            unsuffixed_literal_only(&call.args[0])?;
            Some(receiver)
        }
        _ => None,
    }
}

/// Whether `place` holds an index expression anywhere (`a[i]`, `a[i].f`, `a.f[i]`).
pub(super) fn place_has_index(place: &syn::Expr) -> bool {
    struct Scan(bool);
    impl<'a> Visit<'a> for Scan {
        fn visit_expr_index(&mut self, _: &'a syn::ExprIndex) {
            self.0 = true;
        }
    }
    let mut scan = Scan(false);
    scan.visit_expr(place);
    scan.0
}

/// The first use of `self` as a value in `block`: any `self` that is not the base of a field access (`self.n`) or the
/// receiver of a method call (`self.bump()`).
pub(super) fn bare_self_use(block: &syn::Block) -> Option<&syn::ExprPath> {
    struct Scan<'a> {
        found: Option<&'a syn::ExprPath>,
    }
    fn is_self(expression: &syn::Expr) -> bool {
        match expression {
            syn::Expr::Paren(inner) => is_self(&inner.expr),
            syn::Expr::Path(path) => {
                path.qself.is_none() && crate::names::path_is(&path.path, "self")
            }
            _ => false,
        }
    }
    impl<'a> Visit<'a> for Scan<'a> {
        fn visit_expr_field(&mut self, field: &'a syn::ExprField) {
            if !is_self(&field.base) {
                visit::visit_expr_field(self, field);
            }
        }
        fn visit_expr_method_call(&mut self, call: &'a syn::ExprMethodCall) {
            if !is_self(&call.receiver) {
                self.visit_expr(&call.receiver);
            }
            for argument in &call.args {
                self.visit_expr(argument);
            }
        }
        fn visit_expr_path(&mut self, path: &'a syn::ExprPath) {
            if self.found.is_none()
                && path.qself.is_none()
                && crate::names::path_is(&path.path, "self")
            {
                self.found = Some(path);
            }
        }
    }
    let mut scan = Scan { found: None };
    scan.visit_block(block);
    scan.found
}

/// Whether evaluating `expr` can run code (a call, a method call, or any form this check does not know): the receiver
/// of an array `.len()` that does so is refused, because the emitter replaces `.len()` by a constant (M1, fix round 1 of
/// m9.r2a). Names, literals, fields, indexes and plain arithmetic do not.
pub(super) fn len_receiver_runs_code(expr: &syn::Expr) -> bool {
    match peel_expr(expr) {
        syn::Expr::Path(_) | syn::Expr::Lit(_) => false,
        syn::Expr::Field(field) => len_receiver_runs_code(&field.base),
        syn::Expr::Index(index) => {
            len_receiver_runs_code(&index.expr) || len_receiver_runs_code(&index.index)
        }
        syn::Expr::Cast(cast) => len_receiver_runs_code(&cast.expr),
        syn::Expr::Unary(unary) => len_receiver_runs_code(&unary.expr),
        syn::Expr::Binary(binary) => {
            len_receiver_runs_code(&binary.left) || len_receiver_runs_code(&binary.right)
        }
        _ => true,
    }
}

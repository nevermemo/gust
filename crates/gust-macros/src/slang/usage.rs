//! What a module or function uses, found by one walk, so each helper is defined only where called.

use syn::visit::Visit;

use super::calls::{is_atomic_call, is_result_constructor, is_some_constructor};
use super::operators::division_helper;
use super::types::emit_ident;

/// What a module (or one function of it) uses, found by one walk: each helper of the emitted Slang is defined only
/// where something calls it, and the buffers of atomic calls take `Atomic<T>` element types. The module is scanned for
/// everything but the subgroup operations and the atomic buffers, which only the kernel is scanned for (a helper
/// function holds no subgroup operation, GUST-E0745).
#[derive(Default)]
pub(super) struct Usage {
    /// A unary `!`: Slang overloads Rust's boolean and bitwise `!` (`__gust_not`).
    pub(super) not: bool,
    /// A `^` or `^=` (`__gust_xor`).
    pub(super) xor: bool,
    /// A `.fract()`.
    pub(super) fract: bool,
    /// Any Option feature; both generic helpers are then emitted.
    pub(super) option: bool,
    /// Any Result feature.
    pub(super) result: bool,
    /// The division helpers: `div` (for `/` and `/=`) and `rem` (for `%` and `%=`, and for `a.rem_euclid(b)`, which is
    /// built on `__gust_rem`, slice m11.math2).
    pub(super) div: bool,
    pub(super) rem: bool,
    /// The helpers of `round`, `signum` and `rem_euclid` (slice m11.math2), `hypot` and the integer `pow` (m11.math3).
    pub(super) round: bool,
    pub(super) signum: bool,
    pub(super) rem_euclid: bool,
    pub(super) hypot: bool,
    pub(super) pow: bool,
    /// The subgroup operations called, for the helpers [`subgroup_helper_text`] defines.
    pub(super) subgroups: std::collections::BTreeSet<&'static str>,
    /// The buffers referenced by atomic calls.
    pub(super) atomic_buffers: std::collections::HashSet<String>,
}

impl Usage {
    pub(super) fn of_module(module: &syn::ItemMod) -> Self {
        let mut usage = Self::default();
        usage.visit_item_mod(module);
        usage
    }

    pub(super) fn of_function(function: &syn::ItemFn) -> Self {
        let mut usage = Self::default();
        usage.visit_item_fn(function);
        usage
    }
}

impl<'ast> Visit<'ast> for Usage {
    fn visit_expr_unary(&mut self, expression: &'ast syn::ExprUnary) {
        self.not |= matches!(expression.op, syn::UnOp::Not(_));
        syn::visit::visit_expr_unary(self, expression);
    }

    fn visit_expr_binary(&mut self, expression: &'ast syn::ExprBinary) {
        self.xor |= matches!(
            expression.op,
            syn::BinOp::BitXor(_) | syn::BinOp::BitXorAssign(_)
        );
        match division_helper(expression) {
            Some(("div", _)) => self.div = true,
            Some(_) => self.rem = true,
            None => {}
        }
        syn::visit::visit_expr_binary(self, expression);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.option |= is_some_constructor(call);
        self.result |= is_result_constructor(call);
        if let Some(name) = crate::validate::subgroup_operation(call) {
            self.subgroups.insert(name);
        }
        if is_atomic_call(call)
            && let syn::Expr::Reference(reference) = &call.args[0]
            && let syn::Expr::Index(index) = reference.expr.as_ref()
            && let syn::Expr::Path(path) = index.expr.as_ref()
            && let Some(segment) = path.path.segments.first()
        {
            self.atomic_buffers.insert(emit_ident(&segment.ident));
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let name = crate::names::name_of(&call.method);
        let arity = call.args.len();
        self.round |= name == "round" && arity == 0;
        self.signum |= name == "signum" && arity == 0;
        self.rem_euclid |= name == "rem_euclid" && arity == 1;
        self.rem |= name == "rem_euclid" && arity == 1;
        self.hypot |= name == "hypot" && arity == 1;
        self.pow |= name == "pow" && arity == 1;
        self.fract |= name == "fract" && arity == 0;
        self.option |= matches!(name.as_str(), "is_some" | "is_none");
        self.result |= matches!(name.as_str(), "is_ok" | "is_err");
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
        if let Some(segment) = ty.path.segments.last() {
            self.option |= crate::names::ident_is(&segment.ident, "Option");
            self.result |= crate::names::ident_is(&segment.ident, "Result");
        }
        syn::visit::visit_type_path(self, ty);
    }

    fn visit_expr_let(&mut self, expression: &'ast syn::ExprLet) {
        self.option |= crate::validate::option_pattern(&expression.pat).is_some();
        self.result |= crate::validate::result_pattern(&expression.pat).is_some();
        syn::visit::visit_expr_let(self, expression);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.option |= crate::names::path_is(&path.path, "None");
        syn::visit::visit_expr_path(self, path);
    }
}

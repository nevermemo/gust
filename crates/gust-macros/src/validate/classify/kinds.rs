//! The ledger of the constructs a kernel body can hold, and what the classifier does with each (D94).
//!
//! D94 promises that anything the classifier does not recognise is `grid`-only. The rules in the parent module look
//! for the shape-dependent constructs they know, so on their own they would give a construct they have never heard of
//! `pixel` by default. This ledger turns that around: it lists every kind of expression, statement and pattern, every
//! built-in method and every callee form that the validator and the emitter accept in a kernel body today, with the
//! classifier's decision for each. [`scan`] walks a body and names every node whose kind is not listed, and the
//! classifier makes that a reason for `grid` (`Why::Unrecognised`). A kind the validator starts to accept is therefore
//! `grid`-only until someone adds it here, and the tests of this module (`tests::the_ledger_*`) fail at the same moment
//! because they probe the validator and the emitter with one program per kind of syn (and per callee form) and compare
//! the answer with this list.
//!
//! The probes test one form per kind. A kind that is already listed (`Call`, `Reference`, `Path`, `Unary`) and that the
//! validator later accepts in a new position is therefore not caught by them; it is still classified by the kind's own
//! rule, which is written for the kind, not for the position.
//!
//! The validator cannot share the list: it is a deny-list over `syn::visit`'s defaults (a kind is accepted because no
//! `visit_*` method rejects it), and the emitter lowers by `match` with a refusing fallback, so neither holds a list of
//! accepted kinds. The probes in the tests are the tie.
//!
//! The helper names a call may name come from the validator itself ([`super::super::helper_names`], the set its
//! `visit_expr_call` checks a callee against), so a callee is a "helper" exactly when the validator would accept it as
//! one.

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::visit::{self, Visit};

use super::Why;
use crate::math_types;
use crate::validate::{atomic_operation_name, is_barrier_call};

/// The part of the language an [`Entry`] is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Class {
    /// A `syn::Expr` variant.
    Expression,
    /// A `syn::Stmt` variant.
    Statement,
    /// A `syn::Pat` variant.
    Pattern,
    /// A built-in method of the dialect, by name.
    Method,
    /// A call of a method of an `impl` block of the module, by the marker `.name` in `helpers` (slice m9.a1).
    ModuleMethod,
    /// The form of the callee of a call (`Expr::Call`): a helper, a built-in by name, or a math-type constructor.
    Callee,
}

/// The classifier's decision for one kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Handling {
    /// The kind computes a value or moves control inside one invocation, and no shape depends on it.
    Neutral,
    /// A rule of the classifier looks at it; the text says which.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the rule is documentation the tests print")
    )]
    Inspected(&'static str),
    /// The kind rules `pixel` out.
    Excluded(Why),
}

/// One kind and the classifier's decision for it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Entry {
    pub(super) class: Class,
    pub(super) kind: &'static str,
    pub(super) handling: Handling,
}

const fn entry(class: Class, kind: &'static str, handling: Handling) -> Entry {
    Entry {
        class,
        kind,
        handling,
    }
}

const fn expression(kind: &'static str, handling: Handling) -> Entry {
    entry(Class::Expression, kind, handling)
}

const fn method(kind: &'static str, handling: Handling) -> Entry {
    entry(Class::Method, kind, handling)
}

const fn callee(kind: &'static str, handling: Handling) -> Entry {
    entry(Class::Callee, kind, handling)
}

const fn module_method(kind: &'static str, handling: Handling) -> Entry {
    entry(Class::ModuleMethod, kind, handling)
}

use Handling::{Excluded, Inspected, Neutral};

/// Every kind a kernel body can hold that the validator and the emitter both accept.
///
/// `Excluded` entries are the kinds that `pixel` cannot take; the rest are kinds the classifier knows. The kinds of
/// `syn` that are not here are refused by the validator or the emitter today, and each has a probe in the tests.
pub(super) const LEDGER: &[Entry] = &[
    // ---- expressions ----
    expression("Array", Neutral),
    expression(
        "Assign",
        Inspected("OutputScan and BodyScan: the target of the write"),
    ),
    expression(
        "Binary",
        Inspected("OutputScan and BodyScan: compound assignment targets"),
    ),
    expression("Block", Neutral),
    expression("Break", Neutral),
    expression(
        "Call",
        Inspected("BodyScan: atomics and barriers; this ledger's callee entries"),
    ),
    expression("Cast", Neutral),
    expression("Continue", Neutral),
    expression("Field", Neutral),
    expression(
        "ForLoop",
        Inspected("Definite: a loop never counts as a write"),
    ),
    expression("Group", Neutral),
    expression(
        "If",
        Inspected("Definite: both arms, or a transparent guard"),
    ),
    expression("Index", Inspected("OutputScan: a read of an output")),
    expression("Let", Neutral),
    expression("Lit", Neutral),
    expression("Match", Inspected("Definite: every arm")),
    expression("MethodCall", Inspected("this ledger's method entries")),
    expression("Paren", Neutral),
    expression(
        "Path",
        Inspected("OutputScan: any other mention of an output"),
    ),
    expression("Range", Neutral),
    expression(
        "Reference",
        Inspected("BodyScan: the argument of an atomic, which is `Why::Atomic`"),
    ),
    expression("Return", Excluded(Why::Return)),
    expression("Struct", Neutral),
    expression("Unary", Neutral),
    // ---- statements ----
    entry(
        Class::Statement,
        "Local",
        Inspected("BodyScan: the own-index alias and the bindings"),
    ),
    entry(Class::Statement, "Expr", Neutral),
    // ---- patterns ----
    entry(
        Class::Pattern,
        "Ident",
        Inspected("BodyScan: every binding of a name"),
    ),
    entry(Class::Pattern, "Type", Neutral),
    entry(Class::Pattern, "TupleStruct", Neutral),
    entry(Class::Pattern, "Wild", Neutral),
    // ---- built-in methods ----
    method(
        "global_id",
        Inspected("Own: `id.global_id().x` is the own index"),
    ),
    method("len", Inspected("OutputScan: `out.len()` is allowed")),
    method("is_some", Neutral),
    method("is_none", Neutral),
    method("is_ok", Neutral),
    method("is_err", Neutral),
    method("unwrap_or", Neutral),
    method("dot", Neutral),
    method("mul_vec2", Neutral),
    method("mul_vec3", Neutral),
    method("mul_vec4", Neutral),
    method("mul_mat2", Neutral),
    method("mul_mat3", Neutral),
    method("mul_mat4", Neutral),
    method("transpose", Neutral),
    method("wrapping_add", Neutral),
    method("wrapping_sub", Neutral),
    method("wrapping_mul", Neutral),
    method("wrapping_neg", Neutral),
    // A swizzle is two to four of `x`, `y`, `z`, `w` (`math_types::is_swizzle`).
    method("swizzle", Neutral),
    // A texture's methods. A storage texture is excluded as a parameter; a sampled one is read.
    method("load", Neutral),
    method(
        "store",
        Inspected("the parameter: a storage texture is `Why::StorageTexture`"),
    ),
    method("sample", Neutral),
    method("width", Neutral),
    method("height", Neutral),
    // ---- a method of the module: `recv.name(args)` where an `impl` block declares `name` (D133, slice m9.a1) ----
    // The method has value parameters and a value result and holds no barrier, subgroup operation, resource or call (the
    // validator refuses each, so the classifier scans only the kernel and relies on those refusals, as for a helper); the
    // receiver is an ordinary place expression the scan walks. Mutation through a shared receiver is GUST-E0333.
    module_method("method", Neutral),
    // ---- callees: the forms of `visit_expr_call` (validate/mod.rs) that accept a call ----
    // A free function declared in the module, which is not a `#[kernel]`. A helper takes value parameters only
    // (`Code::HelperResourceParameter`) and holds no barrier (`Code::BarrierPosition`), so it reaches no resource, no
    // barrier and no atomic; the classifier scans only the kernel and relies on both refusals (D94, pinned by tests).
    callee("helper", Neutral),
    callee("Some", Neutral),
    callee("Ok", Neutral),
    callee("Err", Neutral),
    callee(
        "workgroup_barrier",
        Inspected("BodyScan: a barrier is `Why::Barrier`"),
    ),
    callee(
        "atomic_add",
        Inspected("BodyScan: an atomic is `Why::Atomic`"),
    ),
    callee(
        "atomic_min",
        Inspected("BodyScan: an atomic is `Why::Atomic`"),
    ),
    callee(
        "atomic_max",
        Inspected("BodyScan: an atomic is `Why::Atomic`"),
    ),
    callee(
        "atomic_exchange",
        Inspected("BodyScan: an atomic is `Why::Atomic`"),
    ),
    callee(
        "atomic_compare_exchange",
        Inspected("BodyScan: an atomic is `Why::Atomic`"),
    ),
    // `Vec3::new(..)`, `Vec3::splat(..)`, `Mat4::from_cols(..)`, `Mat4::from_rows(..)`: values, no resource.
    callee("math::new", Neutral),
    callee("math::splat", Neutral),
    callee("math::from_cols", Neutral),
    callee("math::from_rows", Neutral),
];

/// The ledger's entry for `kind` of `class`.
pub(super) fn lookup(class: Class, kind: &str) -> Option<&'static Entry> {
    LEDGER
        .iter()
        .find(|entry| entry.class == class && entry.kind == kind)
}

/// The name of the `syn::Expr` variant, for the variants the ledger lists; `None` for any other, which includes every
/// variant a later `syn` adds.
pub(super) fn expression_kind(expression: &syn::Expr) -> Option<&'static str> {
    Some(match expression {
        syn::Expr::Array(_) => "Array",
        syn::Expr::Assign(_) => "Assign",
        syn::Expr::Binary(_) => "Binary",
        syn::Expr::Block(_) => "Block",
        syn::Expr::Break(_) => "Break",
        syn::Expr::Call(_) => "Call",
        syn::Expr::Cast(_) => "Cast",
        syn::Expr::Continue(_) => "Continue",
        syn::Expr::Field(_) => "Field",
        syn::Expr::ForLoop(_) => "ForLoop",
        syn::Expr::Group(_) => "Group",
        syn::Expr::If(_) => "If",
        syn::Expr::Index(_) => "Index",
        syn::Expr::Let(_) => "Let",
        syn::Expr::Lit(_) => "Lit",
        syn::Expr::Match(_) => "Match",
        syn::Expr::MethodCall(_) => "MethodCall",
        syn::Expr::Paren(_) => "Paren",
        syn::Expr::Path(_) => "Path",
        syn::Expr::Range(_) => "Range",
        syn::Expr::Reference(_) => "Reference",
        syn::Expr::Return(_) => "Return",
        syn::Expr::Struct(_) => "Struct",
        syn::Expr::Unary(_) => "Unary",
        _ => return None,
    })
}

/// The name of the `syn::Stmt` variant the ledger lists; an item or a macro in a body is `None`.
pub(super) fn statement_kind(statement: &syn::Stmt) -> Option<&'static str> {
    match statement {
        syn::Stmt::Local(_) => Some("Local"),
        syn::Stmt::Expr(..) => Some("Expr"),
        syn::Stmt::Item(_) | syn::Stmt::Macro(_) => None,
    }
}

/// The name of the `syn::Pat` variant, for the variants the ledger lists.
pub(super) fn pattern_kind(pattern: &syn::Pat) -> Option<&'static str> {
    Some(match pattern {
        syn::Pat::Ident(_) => "Ident",
        syn::Pat::Type(_) => "Type",
        syn::Pat::TupleStruct(_) => "TupleStruct",
        syn::Pat::Wild(_) => "Wild",
        _ => return None,
    })
}

/// The ledger's name of a method: itself, or `swizzle` for two to four of `x`, `y`, `z`, `w`.
pub(super) fn method_kind(name: &str) -> &str {
    if crate::math_types::is_swizzle(name) {
        "swizzle"
    } else {
        name
    }
}

/// The ledger's name of the callee of `call`, or `None` for a form the validator's `visit_expr_call` does not accept
/// (anything but a plain path, a qualified path, a path to a name that is neither a built-in nor in `helpers`).
///
/// A built-in name wins over a helper of the same name, as in the validator (`Some`, `Ok`, `Err`, the barrier, an
/// atomic, then a helper); a math-type constructor has two segments, so it cannot be mistaken for either.
pub(super) fn callee_kind(call: &syn::ExprCall, helpers: &HashSet<String>) -> Option<String> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    if let Some((_, member)) = math_types::associated_item(&path.path)
        && member != "IDENTITY"
    {
        return Some(format!("math::{member}"));
    }
    let name = path.path.get_ident()?.to_string();
    if matches!(name.as_str(), "Some" | "Ok" | "Err") {
        return Some(name);
    }
    if is_barrier_call(call) {
        return Some("workgroup_barrier".to_owned());
    }
    if let Some(operation) = atomic_operation_name(path) {
        return Some(operation.to_owned());
    }
    helpers.contains(&name).then(|| "helper".to_owned())
}

/// What the ledger makes of the body of a kernel: the nodes whose kind rules `pixel` out and the nodes whose kind is not
/// listed at all, in source order. `helpers` are the module's helper functions (`validate::helper_names`).
pub(super) fn scan(block: &syn::Block, helpers: &HashSet<String>) -> Vec<(Why, TokenStream)> {
    let mut scan = KindScan {
        helpers,
        findings: Vec::new(),
    };
    scan.visit_block(block);
    scan.findings
}

struct KindScan<'a> {
    helpers: &'a HashSet<String>,
    findings: Vec<(Why, TokenStream)>,
}

impl KindScan<'_> {
    /// Records `node` when its kind is excluded or not listed; `true` when the walk may go on inside it.
    fn check(&mut self, class: Class, kind: Option<&str>, node: impl ToTokens) -> bool {
        match kind
            .and_then(|kind| lookup(class, kind))
            .map(|e| e.handling)
        {
            Some(Excluded(why)) => {
                self.findings.push((why, node.to_token_stream()));
                true
            }
            Some(_) => true,
            None => {
                self.findings
                    .push((Why::Unrecognised, node.to_token_stream()));
                false
            }
        }
    }
}

impl<'ast> Visit<'ast> for KindScan<'_> {
    fn visit_expr(&mut self, expression: &'ast syn::Expr) {
        if self.check(Class::Expression, expression_kind(expression), expression) {
            visit::visit_expr(self, expression);
        }
    }

    fn visit_stmt(&mut self, statement: &'ast syn::Stmt) {
        if self.check(Class::Statement, statement_kind(statement), statement) {
            visit::visit_stmt(self, statement);
        }
    }

    fn visit_pat(&mut self, pattern: &'ast syn::Pat) {
        if self.check(Class::Pattern, pattern_kind(pattern), pattern) {
            visit::visit_pat(self, pattern);
        }
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        // A subgroup operation (M8) is found by its name in `BodyScan` and is `Why::Subgroup`; it is not a callee entry
        // of the ledger, which lists the other forms of call one by one. Its arguments are walked like any call's.
        if crate::validate::subgroup_operation(call).is_some() {
            visit::visit_expr_call(self, call);
            return;
        }
        let kind = callee_kind(call, self.helpers);
        if self.check(Class::Callee, kind.as_deref(), call) {
            visit::visit_expr_call(self, call);
        }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let name = call.method.to_string();
        // A name of the module's own methods never is a dialect name (GUST-E0328), so the marker decides first.
        let (class, kind) = if self.helpers.contains(&format!(".{name}")) {
            (Class::ModuleMethod, "method")
        } else {
            (Class::Method, method_kind(&name))
        };
        if self.check(class, Some(kind), call) {
            visit::visit_expr_method_call(self, call);
        }
    }
}

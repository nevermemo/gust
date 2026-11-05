//! The ledger of the constructs a kernel body can hold, and what the classifier does with each (D94).
//!
//! D94 promises that anything the classifier does not recognise is `grid`-only. The rules in the parent module look
//! for the shape-dependent constructs they know, so on their own they would give a construct they have never heard of
//! `pixel` by default. This ledger turns that around: it lists every kind of expression, statement and pattern, and
//! every built-in method, that the validator and the emitter accept in a kernel body today, with the classifier's
//! decision for each. [`scan`] walks a body and names every node whose kind is not listed, and the
//! classifier makes that a reason for `grid` (`Why::Unrecognised`). A kind the validator starts to accept is therefore
//! `grid`-only until someone adds it here, and the tests of this module (`tests::the_ledger_*`) fail at the same moment
//! because they probe the validator and the emitter with one program per kind of syn and compare the answer with
//! this list.
//!
//! The validator cannot share the list: it is a deny-list over `syn::visit`'s defaults (a kind is accepted because no
//! `visit_*` method rejects it), and the emitter lowers by `match` with a refusing fallback, so neither holds a list of
//! accepted kinds. The probes in the tests are the tie.

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::visit::{self, Visit};

use super::Why;

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
    expression("Call", Inspected("BodyScan: atomics and barriers")),
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

/// What the ledger makes of the body of a kernel: the nodes whose kind rules `pixel` out and the nodes whose kind is not
/// listed at all, in source order.
pub(super) fn scan(block: &syn::Block) -> Vec<(Why, TokenStream)> {
    let mut scan = KindScan::default();
    scan.visit_block(block);
    scan.findings
}

#[derive(Default)]
struct KindScan {
    findings: Vec<(Why, TokenStream)>,
}

impl KindScan {
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

impl<'ast> Visit<'ast> for KindScan {
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

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let name = call.method.to_string();
        if self.check(Class::Method, Some(method_kind(&name)), call) {
            visit::visit_expr_method_call(self, call);
        }
    }
}

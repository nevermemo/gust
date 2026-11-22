//! Operator predicates, the division helpers' selection and the `__gust_keep` rule for `<`.

use super::guards;
use crate::names::peel_expr;

/// Whether the place of a `*=` can be written twice: an identifier, a field path, or an index whose expression is
/// built from literals, names, fields, indexing, arithmetic and `.global_id()`/`.len()`. Anything else (a call, an
/// `if`, a block) may have an effect, and Rust evaluates the place of `place *= value` once while
/// `place = __gust_mul(place, value)` evaluates it twice. A `Group` is the invisible parentheses of a `macro_rules!`
/// `$p:expr`.
pub(crate) fn is_pure_place(place: &syn::Expr) -> bool {
    match peel_expr(place) {
        syn::Expr::Path(_) | syn::Expr::Lit(_) => true,
        syn::Expr::Field(field) => is_pure_place(&field.base),
        syn::Expr::Index(index) => is_pure_place(&index.expr) && is_pure_place(&index.index),
        syn::Expr::Unary(unary) => is_pure_place(&unary.expr),
        syn::Expr::Cast(cast) => is_pure_place(&cast.expr),
        syn::Expr::Binary(binary) => {
            is_pure_place(&binary.left)
                && is_pure_place(&binary.right)
                && !is_assignment_operator(&binary.op)
        }
        syn::Expr::MethodCall(call) => {
            let name = crate::names::name_of(&call.method);
            let builtin = call.args.is_empty() && matches!(name.as_str(), "global_id" | "len");
            // The wrapping methods are arithmetic on their operands, like `+` and `-` above.
            let wrapping = crate::validate::wrapping_arity(&name) == Some(call.args.len())
                && call.args.iter().all(is_pure_place);
            // The scalar math methods (slice m11.math) are pure functions of their operands as well.
            let scalar_math = crate::validate::scalar_math_arity(&name) == Some(call.args.len())
                && call.args.iter().all(is_pure_place);
            (builtin || wrapping || scalar_math) && is_pure_place(&call.receiver)
        }
        _ => false,
    }
}

fn is_assignment_operator(op: &syn::BinOp) -> bool {
    matches!(
        op,
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
    )
}

/// Whether an expression is a `<` comparison, through parentheses and the right operand of `&&` and `||`.
///
/// WGSL's template-list discovery (which naga's lexer follows) starts a candidate at a `<` that follows an
/// identifier and ends it at any later `>` on the same nesting depth, whatever follows the `>`: `>=` and `>>` close
/// one too. A `)` or `]` drops the candidates opened inside its brackets; `=`, `:`, `;` and `{` drop all of them;
/// `&&` and `||` drop only the candidates opened before them, so a `<` after a `&&` stays pending. `<=` and `<<` do
/// not open one. A comparison that is an argument sits at the depth of the list, and slangc keeps no parentheses
/// around it. slangc does parenthesise shifts and turns `&&` and `||` into branches today, so `g(a < b, c >> 1)` and
/// `g(c && a < b, d > e)` come out safe, but by slangc's choice and not by the rule, so the right operand of `&&`
/// and `||` counts as well. Only `<` opens a candidate, so only `<` needs the helper.
fn is_less_than(expression: &syn::Expr) -> bool {
    match peel_expr(expression) {
        syn::Expr::Binary(binary) => match binary.op {
            syn::BinOp::Lt(_) => true,
            syn::BinOp::And(_) | syn::BinOp::Or(_) => is_less_than(&binary.right),
            _ => false,
        },
        _ => false,
    }
}

/// Whether the element at `index` of a comma-separated list of `len` elements goes through `__gust_keep`.
///
/// A candidate opened in the last element ends at the `)` that closes the list, so only an element followed by
/// another one can meet a later `>`. The rule is the same for call arguments, array elements and the two operands
/// of `^`, which is a call to `__gust_xor`.
pub(super) fn needs_keep(expression: &syn::Expr, index: usize, len: usize) -> bool {
    index + 1 < len && is_less_than(expression)
}

/// Whether an expression is a literal operand, possibly negated or parenthesised (`2`, `-1`, `(2.0)`).
pub(super) fn is_literal_operand(expression: &syn::Expr) -> bool {
    match peel_expr(expression) {
        syn::Expr::Lit(_) => true,
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            is_literal_operand(&unary.expr)
        }
        _ => false,
    }
}

/// The helper a `/`, `%`, `/=` or `%=` goes through: its name (`div` or `rem`) and whether it is the `_assign` form.
/// `None` for a literal divisor (the validator rejects a literal zero, so the plain operator is safe) and for every
/// other operator.
pub(super) fn division_helper(binary: &syn::ExprBinary) -> Option<(&'static str, bool)> {
    let helper = match binary.op {
        syn::BinOp::Div(_) => ("div", false),
        syn::BinOp::Rem(_) => ("rem", false),
        syn::BinOp::DivAssign(_) => ("div", true),
        syn::BinOp::RemAssign(_) => ("rem", true),
        _ => return None,
    };
    // With the guards of the SPIR-V route on (D172), `%` and `%=` take the helper for a literal divisor too: slangc's own
    // `%` becomes `OpSRem`, which is poison for negative operands on Vulkan (naga writes `a - d * (a / d)`).
    (!is_literal_operand(&binary.right) || (guards::enabled() && helper.0 == "rem"))
        .then_some(helper)
}

/// Whether `expression` is an integer literal, through parentheses.
pub(super) fn is_integer_literal(expression: &syn::Expr) -> bool {
    match peel_expr(expression) {
        syn::Expr::Lit(literal) => matches!(literal.lit, syn::Lit::Int(_)),
        _ => false,
    }
}

/// The literal operand of a unary minus, if it is an integer literal.
pub(super) fn negated_integer_literal(unary: &syn::ExprUnary) -> Option<&syn::LitInt> {
    match (&unary.op, unary.expr.as_ref()) {
        (syn::UnOp::Neg(_), syn::Expr::Lit(literal)) => match &literal.lit {
            syn::Lit::Int(value) => Some(value),
            _ => None,
        },
        _ => None,
    }
}

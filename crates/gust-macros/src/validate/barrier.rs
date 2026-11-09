//! Workgroup memory and the workgroup barrier (slice m7.g1; plan section 4.2).
//!
//! A kernel that declares `Workgroup<[T; N]>` memory or calls `workgroup_barrier()` is a barrier kernel. The barrier
//! is a control barrier: every invocation of the workgroup has to reach it the same number of times, or the result
//! is undefined. The dialect cannot prove that for arbitrary control flow, so it admits one shape and rejects the
//! rest at the barrier (E0414), at an early `return` (E0415) and at a `break` or `continue` that leaves a loop with a
//! barrier in it (E0416):
//!
//! - a barrier is a statement of the kernel body, or a statement of a top-level `for` loop whose range is two
//!   integer literals (every invocation runs the same trip count);
//! - the kernel has no `return`, and no `break` or `continue` in a loop that holds a barrier;
//! - a helper (or a map function) never holds a barrier, because it may be called under a condition.
//!
//! The generic visitor runs first. These rules run only when it found nothing, so a construct that is rejected for its
//! own sake (`while`, a loop label, `?`) is reported once, by its own code.
//!
//! The subgroup operations (M8, m8.s, D140) take the same discipline: a subgroup operation combines the values of the
//! invocations of a subgroup, so every one of them has to reach it, and the kernel that holds one is a barrier kernel
//! for E0415 and E0416. A call is accepted only in the body of a kernel whose attribute says `requires(subgroups)`
//! (E0745: not in a portable body, not in a helper) and only where `approve_value` finds it (E0747).

use std::collections::HashSet;

use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};

use crate::diagnostics::{self, Code};
use crate::math_types;

/// The most workgroup memory a kernel may declare, in bytes. The WebGPU default limit is 16384.
pub(crate) const WORKGROUP_MEMORY_LIMIT: u64 = 16 * 1024;

/// Whether `call` is `workgroup_barrier(..)`, whatever its arguments.
pub(crate) fn is_barrier_call(call: &syn::ExprCall) -> bool {
    match call.func.as_ref() {
        syn::Expr::Path(path) => path.qself.is_none() && path.path.is_ident("workgroup_barrier"),
        _ => false,
    }
}

/// The call of a statement that is exactly `workgroup_barrier();`.
fn barrier_statement(statement: &syn::Stmt) -> Option<&syn::ExprCall> {
    match statement {
        syn::Stmt::Expr(syn::Expr::Call(call), Some(_)) if is_barrier_call(call) => Some(call),
        _ => None,
    }
}

/// The subgroup operations the dialect has and the number of arguments each takes: the cells S1 and S3 executed
/// (M8, D140), the last four derived in the dialect (`elect`, `count`, `inclusive_add`, `inclusive_mul`).
pub(crate) const SUBGROUP_OPERATIONS: &[(&str, usize)] = &[
    ("subgroup_add", 1),
    ("subgroup_mul", 1),
    ("subgroup_min", 1),
    ("subgroup_max", 1),
    ("subgroup_and", 1),
    ("subgroup_or", 1),
    ("subgroup_xor", 1),
    ("subgroup_all", 1),
    ("subgroup_any", 1),
    ("subgroup_ballot", 1),
    ("subgroup_broadcast_first", 1),
    ("subgroup_broadcast", 2),
    ("subgroup_shuffle", 2),
    ("subgroup_exclusive_add", 1),
    ("subgroup_exclusive_mul", 1),
    ("subgroup_size", 0),
    ("subgroup_lane", 0),
    ("subgroup_elect", 0),
    ("subgroup_count", 1),
    ("subgroup_inclusive_add", 1),
    ("subgroup_inclusive_mul", 1),
];

/// The number of arguments of the subgroup operation called `name`, or `None` when `name` is none.
pub(crate) fn subgroup_arity(name: &str) -> Option<usize> {
    SUBGROUP_OPERATIONS
        .iter()
        .find(|(operation, _)| *operation == name)
        .map(|(_, arity)| *arity)
}

/// The name of the subgroup operation `call` calls: a bare path to one of [`SUBGROUP_OPERATIONS`], whatever its
/// arguments. A module function cannot take the name (GUST-E0745), so the name decides.
pub(crate) fn subgroup_operation(call: &syn::ExprCall) -> Option<&'static str> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let ident = path.path.get_ident()?.to_string();
    SUBGROUP_OPERATIONS
        .iter()
        .find(|(operation, _)| *operation == ident)
        .map(|(operation, _)| *operation)
}

/// Whether `function` calls a subgroup operation anywhere.
pub(crate) fn has_subgroup_call(function: &syn::ItemFn) -> bool {
    let mut finder = Finder::default();
    finder.visit_item_fn(function);
    finder.subgroup
}

/// Whether the attribute of the kernel `function` says `requires(subgroups)` (or the alias `vulkan`, which is the
/// same set). `variant::lower` has spelt every body's attribute with its own set by the time the validator runs; a
/// portable body has none, and so has a function that is no kernel.
pub(crate) fn requires_subgroups(function: &syn::ItemFn) -> bool {
    function
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("kernel"))
        .any(|attribute| {
            let Ok(metas) = crate::diagnostics::parse_args_with(
                attribute,
                crate::diagnostics::Code::SubgroupNotAllowed,
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) else {
                return false;
            };
            metas.iter().any(|meta| {
                let syn::Meta::List(requires) = meta else {
                    return false;
                };
                // The names are plain identifiers: look at the tokens, no parse is needed.
                requires.path.is_ident("requires")
                    && requires.tokens.clone().into_iter().any(|token| {
                        matches!(&token, proc_macro2::TokenTree::Ident(name)
                            if name == "subgroups" || name == "vulkan")
                    })
            })
        })
}

/// The element type and the length of a `Workgroup<[T; N]>` parameter type, or why the type is not that.
pub(crate) fn workgroup_array(ty: &syn::Type) -> Result<(&syn::Type, u32), String> {
    let syn::Type::Path(path) = ty else {
        return Err("a workgroup parameter is `Workgroup<[T; N]>`".to_owned());
    };
    let segment = path
        .path
        .segments
        .last()
        .filter(|segment| segment.ident == "Workgroup")
        .ok_or_else(|| "a workgroup parameter is `Workgroup<[T; N]>`".to_owned())?;
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Err("`Workgroup` needs an array type argument: `Workgroup<[T; N]>`".to_owned());
    };
    let Some(syn::GenericArgument::Type(syn::Type::Array(array))) = arguments.args.first() else {
        return Err("`Workgroup` holds an array: `Workgroup<[T; N]>`".to_owned());
    };
    if arguments.args.len() != 1 {
        return Err("`Workgroup` takes one type argument: `Workgroup<[T; N]>`".to_owned());
    }
    let length = match &array.len {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(length),
            ..
        }) => length
            .base10_parse::<u32>()
            .ok()
            .filter(|length| *length > 0),
        _ => None,
    };
    let Some(length) = length else {
        return Err("the length of a `Workgroup` array is a nonzero integer literal".to_owned());
    };
    if element_bytes(&array.elem).is_none() {
        return Err(
            "the element of a `Workgroup` array is `f32`, `i32`, `u32`, `bool` or a vector type (`Vec2`, `Vec3`, \
             `Vec4`, `IVec2`, `IVec4`, `UVec2`, `UVec3`, `UVec4`)"
                .to_owned(),
        );
    }
    Ok((&array.elem, length))
}

/// The bytes one element of workgroup memory takes: 4 for a scalar or a `bool`, 8 for a 2-wide vector and 16 for a
/// 3-wide or 4-wide one. `None` for any other type.
fn element_bytes(element: &syn::Type) -> Option<u64> {
    let syn::Type::Path(path) = element else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let name = path.path.get_ident()?.unraw().to_string();
    if matches!(
        name.as_str(),
        "f32" | "float" | "i32" | "int" | "u32" | "uint" | "bool"
    ) {
        return Some(4);
    }
    math_types::math_type(&name)
        .filter(|math| !math.matrix)
        .map(|math| u64::from(if math.size <= 8 { math.size } else { 16 }))
}

/// The bytes of workgroup memory the well-formed `Workgroup` parameters of `function` declare together.
pub(crate) fn workgroup_bytes(function: &syn::ItemFn) -> u64 {
    function
        .sig
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            syn::FnArg::Typed(argument) => workgroup_array(&argument.ty).ok(),
            syn::FnArg::Receiver(_) => None,
        })
        .map(|(element, length)| element_bytes(element).unwrap_or(0) * u64::from(length))
        .sum()
}

/// Whether `function` calls `workgroup_barrier()` anywhere.
pub(crate) fn has_barrier(function: &syn::ItemFn) -> bool {
    let mut finder = Finder::default();
    finder.visit_item_fn(function);
    finder.barrier
}

/// What a walk found: a `workgroup_barrier()` call, a subgroup operation.
#[derive(Default)]
struct Finder {
    barrier: bool,
    subgroup: bool,
}

impl Finder {
    /// Whether the walk found a construct that every invocation has to reach.
    fn found(&self) -> bool {
        self.barrier || self.subgroup
    }
}

impl<'ast> Visit<'ast> for Finder {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if is_barrier_call(call) {
            self.barrier = true;
        }
        if subgroup_operation(call).is_some() {
            self.subgroup = true;
        }
        visit::visit_expr_call(self, call);
    }
}

/// Approves the subgroup operations that `expression` evaluates unconditionally, as the value of a statement of a
/// position the walk allows: through parentheses, groups, casts, unary and binary operators other than `&&` and `||`
/// (which skip their right operand), and the arguments of a subgroup operation. An index, a method call, a field,
/// a call of a helper, an `if` or a `match` end the walk: a subgroup operation under them stays unapproved.
fn approve_value(expression: &syn::Expr, approved: &mut HashSet<usize>) {
    match expression {
        syn::Expr::Paren(paren) => approve_value(&paren.expr, approved),
        syn::Expr::Group(group) => approve_value(&group.expr, approved),
        syn::Expr::Cast(cast) => approve_value(&cast.expr, approved),
        syn::Expr::Unary(unary) => approve_value(&unary.expr, approved),
        syn::Expr::Binary(binary)
            if !matches!(binary.op, syn::BinOp::And(_) | syn::BinOp::Or(_))
                && !is_compound_assignment(&binary.op) =>
        {
            approve_value(&binary.left, approved);
            approve_value(&binary.right, approved);
        }
        syn::Expr::Call(call) if subgroup_operation(call).is_some() => {
            approved.insert(std::ptr::from_ref(call) as usize);
            for argument in &call.args {
                approve_value(argument, approved);
            }
        }
        _ => {}
    }
}

fn is_compound_assignment(op: &syn::BinOp) -> bool {
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

/// Approves the subgroup operations of one statement of an allowed position: the value of a `let`, and the right-hand
/// side of an assignment or a compound assignment.
fn approve_statement(statement: &syn::Stmt, approved: &mut HashSet<usize>) {
    match statement {
        syn::Stmt::Local(local) => {
            if let Some(initial) = &local.init
                && initial.diverge.is_none()
            {
                approve_value(&initial.expr, approved);
            }
        }
        syn::Stmt::Expr(syn::Expr::Assign(assign), _) => approve_value(&assign.right, approved),
        syn::Stmt::Expr(syn::Expr::Binary(binary), _) if is_compound_assignment(&binary.op) => {
            approve_value(&binary.right, approved);
        }
        _ => {}
    }
}

/// A `for` loop over `a..b` with two integer literals.
fn literal_range_loop(statement: &syn::Stmt) -> Option<&syn::ExprForLoop> {
    let syn::Stmt::Expr(syn::Expr::ForLoop(expression), _) = statement else {
        return None;
    };
    let syn::Expr::Range(range) = expression.expr.as_ref() else {
        return None;
    };
    let literal = |bound: &Option<Box<syn::Expr>>| {
        matches!(
            bound.as_deref(),
            Some(syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(_),
                ..
            }))
        )
    };
    (matches!(range.limits, syn::RangeLimits::HalfOpen(_))
        && literal(&range.start)
        && literal(&range.end))
    .then_some(expression)
}

/// Checks the barrier rules of a function whose generic validation passed: E0414, E0415 and E0416.
///
/// `is_barrier_kernel` is true for a kernel with a `Workgroup` parameter or a barrier call; a helper is never one,
/// and every barrier in it is E0414.
///
/// `subgroups_allowed` is true for a kernel whose attribute says `requires(subgroups)` ([`requires_subgroups`]); a
/// subgroup operation anywhere else is E0745, and one at a position [`approve_statement`] does not reach is E0747.
pub(crate) fn check(
    function: &syn::ItemFn,
    is_kernel: bool,
    is_barrier_kernel: bool,
    subgroups_allowed: bool,
) -> syn::Result<()> {
    let mut approved = HashSet::new();
    let mut approved_subgroups = HashSet::new();
    if is_kernel {
        for statement in &function.block.stmts {
            if let Some(call) = barrier_statement(statement) {
                approved.insert(std::ptr::from_ref(call) as usize);
            } else if let Some(expression) = literal_range_loop(statement) {
                for inner in &expression.body.stmts {
                    if let Some(call) = barrier_statement(inner) {
                        approved.insert(std::ptr::from_ref(call) as usize);
                    }
                    approve_statement(inner, &mut approved_subgroups);
                }
            } else {
                approve_statement(statement, &mut approved_subgroups);
            }
        }
    }
    let mut pass = Pass {
        approved,
        approved_subgroups,
        subgroups_allowed: is_kernel && subgroups_allowed,
        subgroup_note: if is_kernel && has_subgroup_call(function) {
            " (this kernel calls a subgroup operation, which every invocation of the subgroup has to reach)"
        } else {
            ""
        },
        is_kernel,
        forbid_return: is_barrier_kernel,
        loops: Vec::new(),
        error: None,
    };
    pass.visit_item_fn(function);
    pass.error.map_or(Ok(()), Err)
}

struct Pass {
    approved: HashSet<usize>,
    /// The subgroup operations at a position the kernel may use them at.
    approved_subgroups: HashSet<usize>,
    /// Whether this is the body of a kernel with `requires(subgroups)`.
    subgroups_allowed: bool,
    /// Non-empty when the kernel holds a subgroup operation: the messages about an early exit lead with the subgroup
    /// case, and this is appended to the messages about a jump.
    subgroup_note: &'static str,
    is_kernel: bool,
    forbid_return: bool,
    /// For each enclosing `for` loop, whether it holds a barrier.
    loops: Vec<bool>,
    error: Option<syn::Error>,
}

impl Pass {
    fn reject(&mut self, code: Code, node: impl quote::ToTokens, message: &str) {
        let error = diagnostics::error(code, node, message);
        if let Some(existing) = &mut self.error {
            existing.combine(error);
        } else {
            self.error = Some(error);
        }
    }

    fn jump(&mut self, node: impl quote::ToTokens, name: &str, label: Option<&syn::Lifetime>) {
        // A labelled jump can leave an outer loop that holds a barrier, whatever the innermost loop holds. Loop labels
        // are GUST-E0403 in every kernel today; this keeps rule 2 true if a later slice admits them.
        if self.forbid_return && label.is_some() {
            self.reject(
                Code::JumpInBarrierLoop,
                node,
                &format!(
                    "a labelled `{name}` in a kernel with workgroup memory or a barrier can skip the barriers of \
                     an outer loop that the other invocations wait at; run the loops to their ends and guard the \
                     work with `if`{}",
                    self.subgroup_note
                ),
            );
        } else if self.loops.last() == Some(&true) {
            self.reject(
                Code::JumpInBarrierLoop,
                node,
                &format!(
                    "`{name}` in a loop that holds a `workgroup_barrier()` lets some invocations skip a barrier \
                     the others wait at; run the loop to its end and guard the work inside it with `if`{}",
                    self.subgroup_note
                ),
            );
        }
    }
}

impl<'ast> Visit<'ast> for Pass {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if is_barrier_call(call) && !self.approved.contains(&(std::ptr::from_ref(call) as usize)) {
            let message = if self.is_kernel {
                "`workgroup_barrier()` is allowed only as a statement of the kernel body, or of a top-level \
                 `for a..b` loop with two literal bounds: every invocation of the workgroup has to reach it the \
                 same number of times"
            } else {
                "`workgroup_barrier()` is allowed only in a kernel: a helper or a method may be called under a condition, so \
                 not every invocation would reach the barrier; call the barrier from the kernel body"
            };
            self.reject(Code::BarrierPosition, call, message);
        }
        if let Some(name) = subgroup_operation(call) {
            if !self.subgroups_allowed {
                let message = if self.is_kernel {
                    format!(
                        "`{name}` is a subgroup operation, which only a body that says `requires(subgroups)` may \
                         use: put it in a `#[variant(of = .., requires(subgroups))]` body, or give the kernel \
                         `requires(subgroups)` when it has no portable body"
                    )
                } else {
                    format!(
                        "`{name}` is a subgroup operation, which only the body of a kernel with \
                         `requires(subgroups)` may call: a helper may be called under a condition, so not every \
                         invocation of the subgroup would reach it; call it from the kernel body"
                    )
                };
                self.reject(Code::SubgroupNotAllowed, call, &message);
            } else if !self
                .approved_subgroups
                .contains(&(std::ptr::from_ref(call) as usize))
            {
                self.reject(
                    Code::SubgroupPosition,
                    call,
                    &format!(
                        "`{name}` is allowed only as the whole value of a `let` or an assignment, or as an operand of \
                         arithmetic or a comparison there (not under `&&` or `||`), at the top level of the kernel \
                         body or of a top-level `for a..b` loop with two literal bounds: every invocation of the \
                         subgroup has to reach it, and the value of an `if`, an index or a method call may not"
                    ),
                );
            }
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_return(&mut self, expression: &'ast syn::ExprReturn) {
        if self.forbid_return {
            self.reject(
                Code::ReturnInBarrierKernel,
                expression,
                &if self.subgroup_note.is_empty() {
                    "a kernel with workgroup memory or a barrier has no `return`: an invocation that returns early \
                     never reaches the barriers the others wait at; guard the rest of the work with `if`"
                        .to_owned()
                } else {
                    "a kernel that calls a subgroup operation has no `return` (nor has one with workgroup memory or \
                     a barrier): an invocation that returns early never reaches the operation the others of the \
                     subgroup combine at; guard the rest of the work with `if`"
                        .to_owned()
                },
            );
        }
        visit::visit_expr_return(self, expression);
    }

    fn visit_expr_for_loop(&mut self, expression: &'ast syn::ExprForLoop) {
        let mut finder = Finder::default();
        finder.visit_block(&expression.body);
        self.visit_expr(&expression.expr);
        self.loops.push(finder.found());
        self.visit_block(&expression.body);
        self.loops.pop();
    }

    fn visit_expr_break(&mut self, expression: &'ast syn::ExprBreak) {
        self.jump(expression, "break", expression.label.as_ref());
        visit::visit_expr_break(self, expression);
    }

    fn visit_expr_continue(&mut self, expression: &'ast syn::ExprContinue) {
        self.jump(expression, "continue", expression.label.as_ref());
    }

    fn visit_expr_try(&mut self, expression: &'ast syn::ExprTry) {
        // `?` is an early exit like `return`. It is GUST-E0311 in every function today; this keeps rule 3 true if a
        // later slice admits it.
        if self.forbid_return {
            self.reject(
                Code::ReturnInBarrierKernel,
                expression,
                &if self.subgroup_note.is_empty() {
                    "a kernel with workgroup memory or a barrier has no early exit: `?` can leave the kernel before \
                     the barriers the other invocations wait at; guard the rest of the work with `if`"
                        .to_owned()
                } else {
                    "a kernel that calls a subgroup operation has no early exit (nor has one with workgroup memory \
                     or a barrier): `?` can leave the kernel before the operation the other invocations of the \
                     subgroup combine at; guard the rest of the work with `if`"
                        .to_owned()
                },
            );
        }
        visit::visit_expr_try(self, expression);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::testing::parse_str;

    fn kernel(body: &str) -> syn::ItemFn {
        parse_str(&format!("fn k() {{ {body} }}")).unwrap()
    }

    fn message(function: &syn::ItemFn, barrier_kernel: bool) -> Option<String> {
        check(function, true, barrier_kernel, false)
            .err()
            .map(|error| error.to_string())
    }

    /// The pass is the default for what the generic visitor does not reject: it is called directly, as if a later
    /// slice admitted `?` and loop labels.
    #[test]
    fn the_pass_rejects_try_in_a_barrier_kernel_by_itself() {
        let function = kernel("let v = half(1u32)?; workgroup_barrier();");
        let text = message(&function, true).expect("`?` in a barrier kernel");
        assert!(text.contains("early exit"), "{text}");
        assert!(message(&function, false).is_none());
    }

    #[test]
    fn the_pass_rejects_a_labelled_break_or_continue_in_a_barrier_kernel_by_itself() {
        for jump in ["break 'r;", "continue 'r;"] {
            let function = kernel(&format!(
                "'r: for a in 0u32..4u32 {{ workgroup_barrier(); for b in 0u32..2u32 {{ {jump} }} }}"
            ));
            let text = message(&function, true).expect(jump);
            assert!(text.contains("labelled"), "{jump}: {text}");
        }
    }

    #[test]
    fn an_unlabelled_jump_in_a_loop_without_a_barrier_stays_allowed() {
        let function =
            kernel("workgroup_barrier(); for a in 0u32..4u32 { if a == 2u32 { break; } }");
        assert!(message(&function, true).is_none());
    }
}

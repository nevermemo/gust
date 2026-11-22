//! Module-level array constants (slice m12.carr): `const NAME: [T; N] = [e0, e1, ..];` with `T` one of the four constant
//! scalars and `N` a literal from 1 to 256. The elements are the expressions the scalar form accepts
//! (`check_const_expr`); the item is read one element at a time (`NAME[i]`, `NAME.len()`) and never as a whole.

use std::collections::HashMap;

use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};

use super::{ConstScope, check_const_expr, const_form, const_scalar};
use crate::diagnostics::{self, Code};
use crate::names::peel_expr;

/// The most elements an array constant has.
const MAX_LENGTH: u32 = 256;

/// The array constants of the module with their lengths, by name: every `const` whose type is `[T; N]` with a literal
/// `N` (the item itself is judged by [`validate`]).
pub(crate) fn array_constants(items: &[syn::Item]) -> HashMap<String, u32> {
    items
        .iter()
        .filter_map(|item| {
            let syn::Item::Const(item) = item else {
                return None;
            };
            let syn::Type::Array(array) = item.ty.as_ref() else {
                return None;
            };
            let length = crate::slang::array_length_literal(&array.len)?;
            Some((item.ident.unraw().to_string(), length))
        })
        .collect()
}

/// Checks the type and the initializer of an array constant. The name and the attributes were judged by the caller.
pub(super) fn validate(
    item: &syn::ItemConst,
    array: &syn::TypeArray,
    scope: &ConstScope,
) -> syn::Result<()> {
    let Some(length) = crate::slang::array_length_literal(&array.len) else {
        return Err(diagnostics::error(
            Code::ArrayLength,
            &array.len,
            "an array length is an integer literal greater than zero (`[u32; 4]`): not `0`, a constant or an expression",
        ));
    };
    if length > MAX_LENGTH {
        return Err(const_form(
            &array.len,
            &format!(
                "an array `const` has at most {MAX_LENGTH} elements: split the table, or pass it as a storage buffer"
            ),
        ));
    }
    let Some(scalar) = const_scalar(&array.elem) else {
        let message = if matches!(array.elem.as_ref(), syn::Type::Array(_)) {
            "an array `const` is not nested: write one array of `u32`, `i32`, `f32` or `bool` and compute the row and column in the index"
        } else {
            "an array `const` holds `u32`, `i32`, `f32` or `bool` elements: a vector, a struct, a tuple, a reference and `usize` have no constant lowering here"
        };
        return Err(const_form(&array.elem, message));
    };
    match peel_expr(&item.expr) {
        syn::Expr::Array(list) => {
            if list.elems.len() != length as usize {
                return Err(const_form(
                    &item.expr,
                    &format!(
                        "the array `const` has {length} elements by its type and {} in its initializer",
                        list.elems.len()
                    ),
                ));
            }
            for element in &list.elems {
                check_const_expr(element, scalar, scope, false)?;
            }
            Ok(())
        }
        syn::Expr::Repeat(repeat) => {
            if crate::slang::array_length_literal(&repeat.len) != Some(length) {
                return Err(const_form(
                    &repeat.len,
                    &format!(
                        "the repeat length of an array `const` is the literal {length} of its type"
                    ),
                ));
            }
            check_const_expr(&repeat.expr, scalar, scope, false)
        }
        _ => Err(const_form(
            &item.expr,
            "an array `const` is initialized with an array literal `[a, b, c]` or a repeat `[e; N]` of the same length",
        )),
    }
}

/// Where an array constant may stand in a function, a method or a closure: as the base of an index read (`NAME[i]`) and
/// as the receiver of `.len()`. Anything else is a whole-array use or a store, GUST-E0127.
pub(super) fn check_uses(items: &[syn::Item], arrays: &HashMap<String, u32>) -> syn::Result<()> {
    if arrays.is_empty() {
        return Ok(());
    }
    let mut scan = Uses {
        arrays,
        found: None,
    };
    for item in items {
        if !matches!(item, syn::Item::Const(_)) {
            scan.visit_item(item);
        }
    }
    scan.found.map_or(Ok(()), Err)
}

struct Uses<'a> {
    arrays: &'a HashMap<String, u32>,
    found: Option<syn::Error>,
}

impl Uses<'_> {
    /// The array constant `expr` names, when it is the bare name of one.
    fn named(&self, expr: &syn::Expr) -> Option<String> {
        let syn::Expr::Path(path) = peel_expr(expr) else {
            return None;
        };
        let name = path.qself.is_none().then(|| path.path.get_ident())??;
        let name = name.unraw().to_string();
        self.arrays.contains_key(&name).then_some(name)
    }

    /// The array constant a place is an element of (`NAME[i]`, `NAME[i].f`), also on the left of a destructuring
    /// assignment (`(NAME[0], x) = ..`, `[NAME[0], x] = ..`, `S { a: NAME[0] } = ..`, `P(NAME[0], x) = ..`).
    fn root(&self, place: &syn::Expr) -> Option<String> {
        match peel_expr(place) {
            syn::Expr::Index(index) => self.named(&index.expr).or_else(|| self.root(&index.expr)),
            syn::Expr::Field(field) => self.root(&field.base),
            syn::Expr::Tuple(tuple) => tuple.elems.iter().find_map(|element| self.root(element)),
            syn::Expr::Array(array) => array.elems.iter().find_map(|element| self.root(element)),
            syn::Expr::Struct(literal) => literal
                .fields
                .iter()
                .find_map(|field| self.root(&field.expr)),
            syn::Expr::Call(call) if matches!(peel_expr(&call.func), syn::Expr::Path(_)) => {
                call.args.iter().find_map(|argument| self.root(argument))
            }
            _ => None,
        }
    }

    fn store(&mut self, place: &syn::Expr, target: &impl quote::ToTokens) {
        if self.found.is_none()
            && let Some(name) = self.root(place)
        {
            self.found = Some(const_form(
                target,
                &format!(
                    "`{name}` is an array constant and is read only: write the changed table to a local array (`let mut t: [..; N] = ..;`) and store into that"
                ),
            ));
        }
    }
}

impl<'ast> Visit<'ast> for Uses<'_> {
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if self.found.is_none()
            && path.qself.is_none()
            && let Some(ident) = path.path.get_ident()
            && self.arrays.contains_key(&ident.unraw().to_string())
        {
            self.found = Some(const_form(
                path,
                &format!(
                    "`{ident}` is an array constant and is read one element at a time (`{ident}[i]`, `{ident}.len()`): a whole-array use (a `let`, an argument, a returned value, a comparison) has no lowering; index it, or copy the elements into a local array"
                ),
            ));
        }
    }

    fn visit_expr_index(&mut self, index: &'ast syn::ExprIndex) {
        if self.named(&index.expr).is_some() {
            self.visit_expr(&index.index);
        } else {
            visit::visit_expr_index(self, index);
        }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "len" && call.args.is_empty() && self.named(&call.receiver).is_some() {
            return;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
        self.store(&assign.left, assign);
        visit::visit_expr_assign(self, assign);
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        use syn::BinOp as B;
        if matches!(
            binary.op,
            B::AddAssign(_)
                | B::SubAssign(_)
                | B::MulAssign(_)
                | B::DivAssign(_)
                | B::RemAssign(_)
                | B::BitXorAssign(_)
                | B::BitAndAssign(_)
                | B::BitOrAssign(_)
                | B::ShlAssign(_)
                | B::ShrAssign(_)
        ) {
            self.store(&binary.left, binary);
        }
        visit::visit_expr_binary(self, binary);
    }

    fn visit_expr_reference(&mut self, reference: &'ast syn::ExprReference) {
        if reference.mutability.is_some() {
            self.store(&reference.expr, reference);
        }
        visit::visit_expr_reference(self, reference);
    }
}

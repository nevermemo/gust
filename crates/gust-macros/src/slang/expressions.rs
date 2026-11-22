//! Expressions and literals.

use quote::ToTokens;

use super::arrays::{ARRAY_LITERAL_OK, array_place_length_of, comparison_array_operand};
use super::calls::{
    associated_call_path, atomic_op_name, atomic_slang_method, emit_arguments, emit_list,
    emit_math_constructor, emit_math_method, emit_scalar_math, emit_subgroup_call, identity_matrix,
    is_atomic_call, is_math_method, is_ok_constructor, is_result_constructor, is_some_constructor,
    math_constructor,
};
use super::closures::closure_call_target;
use super::control::emit_variant_path;
use super::lowering::{lowering, self_struct};
use super::operators::{
    division_helper, is_literal_operand, is_pure_place, negated_integer_literal,
};
use super::types::{emit_ident, emit_type, is_usize_type};
use super::{collisions, guards, texture};
use crate::{
    diagnostics::{self, Code},
    math_types,
    names::peel_expr,
};

pub(super) fn emit_expression(expression: &syn::Expr) -> syn::Result<String> {
    Ok(match expression {
        syn::Expr::Array(array) => {
            // Slang takes `{ .. }` where the type is known: a typed `let`, an assignment, a call argument or a returned
            // value (`emit_value_expression` says so); anywhere else it is no array.
            if !ARRAY_LITERAL_OK.with(|flag| flag.replace(false)) {
                return Err(diagnostics::error(
                    Code::UnsupportedExpression,
                    array,
                    "an array literal is the initializer of a typed `let`, the right side of an assignment, a call argument or a returned value: bind it to a typed local first",
                ));
            }
            format!("{{ {} }}", emit_list(array.elems.iter())?.join(", "))
        }
        syn::Expr::Assign(assign) => format!(
            "{} = {}",
            emit_expression(&assign.left)?,
            emit_value_expression(&assign.right)?
        ),
        syn::Expr::Repeat(repeat) => {
            return Err(diagnostics::error(
                Code::UnsupportedExpression,
                repeat,
                "`[e; N]` is the whole initializer of a typed `let`, the right side of an assignment statement or a returned value, so that `e` is evaluated once before the array is built: bind it first (`let a: [T; N] = [e; N];`)",
            ));
        }
        syn::Expr::Binary(binary)
            if matches!(
                binary.op,
                syn::BinOp::Eq(_)
                    | syn::BinOp::Ne(_)
                    | syn::BinOp::Lt(_)
                    | syn::BinOp::Le(_)
                    | syn::BinOp::Gt(_)
                    | syn::BinOp::Ge(_)
            ) && comparison_array_operand(peel_expr(&binary.left))
                && comparison_array_operand(peel_expr(&binary.right)) =>
        {
            return Err(diagnostics::error(
                Code::UnsupportedExpression,
                binary,
                "arrays are not compared with `==`, `!=`, `<`, `<=`, `>` or `>=`: compare the elements",
            ));
        }
        // `i as usize` in an index is lowered like `i as u32` (fix round 1 of m9.r1): the index gets the 32-bit
        // unsigned value of the source, so a `bool` source and a signed `%` index as in Rust, and the remaining
        // difference is the 32/64-bit width class of D39. The validator took the cast only in an index (GUST-E0334).
        syn::Expr::Cast(cast) if is_usize_type(&cast.ty) => {
            format!("uint({})", emit_expression(&cast.expr)?)
        }
        syn::Expr::Binary(binary) if matches!(binary.op, syn::BinOp::BitXor(_)) => format!(
            "__gust_xor({})",
            emit_list([binary.left.as_ref(), binary.right.as_ref()].into_iter())?.join(", ")
        ),
        syn::Expr::Binary(binary) if matches!(binary.op, syn::BinOp::BitXorAssign(_)) => format!(
            "__gust_xor_assign({}, {})",
            emit_expression(&binary.left)?,
            emit_expression(&binary.right)?
        ),
        // `/`, `%`, `/=` and `%=` with a divisor that is not a literal go through `__gust_div` and `__gust_rem`.
        syn::Expr::Binary(binary) if division_helper(binary).is_some() => {
            let (name, assign) = division_helper(binary).expect("guarded by the match arm");
            format!(
                "__gust_{name}{}({}, {})",
                if assign { "_assign" } else { "" },
                emit_expression(&binary.left)?,
                emit_expression(&binary.right)?
            )
        }
        // In a module that names a matrix, `*` and `*=` go through the overloads of `__gust_mul`.
        // A product in a future `const` or array length must not take this arm: Slang cannot use a `__gust_mul(..)`
        // call as a size, so the slice that adds constants lowers a product there with the plain `*`.
        syn::Expr::Binary(binary) if lowering().mul && matches!(binary.op, syn::BinOp::Mul(_)) => {
            format!(
                "__gust_mul({}, {})",
                emit_expression(&binary.left)?,
                emit_expression(&binary.right)?
            )
        }
        syn::Expr::Binary(binary)
            if lowering().mul && matches!(binary.op, syn::BinOp::MulAssign(_)) =>
        {
            if !is_pure_place(&binary.left) {
                return Err(diagnostics::error(
                    Code::MulAssignPlace,
                    binary,
                    "`*=` on a place that is more than names, fields, literals and index arithmetic (a call, an `if` \
                     or a block in the place): Rust evaluates the place once and the lowering writes it twice; write \
                     `x = x * y` instead",
                ));
            }
            let place = emit_expression(&binary.left)?;
            format!(
                "({place} = __gust_mul({place}, {}))",
                emit_expression(&binary.right)?
            )
        }
        syn::Expr::Binary(binary) => format!(
            "({} {} {})",
            emit_expression(&binary.left)?,
            binary.op.to_token_stream(),
            emit_expression(&binary.right)?
        ),
        syn::Expr::Call(call) if is_some_constructor(call) => {
            format!("__gust_some({})", emit_expression(&call.args[0])?)
        }
        syn::Expr::Call(call) if is_result_constructor(call) => {
            let helper = if is_ok_constructor(call) {
                "__gust_ok"
            } else {
                "__gust_err"
            };
            format!("{helper}({})", emit_expression(&call.args[0])?)
        }
        syn::Expr::Call(call) if is_atomic_call(call) => {
            let op_name =
                atomic_op_name(call).expect("is_atomic_call guarantees a recognized name");
            let method =
                atomic_slang_method(op_name).expect("recognized atomic op has a Slang method");
            let receiver = call
                .args
                .first()
                .expect("atomic call has at least one argument");
            let syn::Expr::Reference(reference) = receiver else {
                return Err(diagnostics::error(
                    Code::AtomicEmitterReference,
                    receiver,
                    format!("`{op_name}` requires a mutable reference to a buffer element"),
                ));
            };
            let syn::Expr::Index(index) = reference.expr.as_ref() else {
                return Err(diagnostics::error(
                    Code::AtomicEmitterElement,
                    &reference.expr,
                    format!("`{op_name}` requires a buffer element"),
                ));
            };
            let args: Vec<String> = call
                .args
                .iter()
                .skip(1)
                .map(emit_expression)
                .collect::<syn::Result<_>>()?;
            format!(
                "{}[{}].{}({})",
                emit_expression(&index.expr)?,
                emit_expression(&index.index)?,
                method,
                args.join(", ")
            )
        }
        syn::Expr::Call(call) if math_constructor(call).is_some() => emit_math_constructor(call)?,
        // The validator admits the barrier as a statement with no arguments only.
        syn::Expr::Call(call) if crate::validate::is_barrier_call(call) => {
            "GroupMemoryBarrierWithGroupSync()".to_owned()
        }
        // The validator admits a subgroup operation in the body of a `requires(subgroups)` kernel only (GUST-E0745).
        syn::Expr::Call(call) if crate::validate::subgroup_operation(call).is_some() => {
            emit_subgroup_call(call)?
        }
        // An associated function, `S::f(..)` or `Self::f(..)` (slice m9.a2): Slang calls a `static` member as `S.f(..)`.
        // Both names are renamed like their declarations. The validator admits only paths of two plain segments here.
        syn::Expr::Call(call) if associated_call_path(call).is_some() => {
            let (owner, member) = associated_call_path(call).expect("guarded by the match arm");
            let owner = if owner == "Self" {
                collisions::emit_name(&self_struct().ok_or_else(|| {
                    diagnostics::error(
                        Code::UnknownCall,
                        call,
                        "`Self` outside an `impl` block has no lowering",
                    )
                })?)
                .into_owned()
            } else {
                collisions::emit_name(&owner).into_owned()
            };
            format!(
                "{owner}.{}({})",
                collisions::emit_name(&member),
                emit_arguments(call.args.iter())?.join(", ")
            )
        }
        // A closure of the function being emitted (slice m9.c): the hoisted function, arguments left to right as for a helper.
        syn::Expr::Call(call) if closure_call_target(call).is_some() => format!(
            "{}({})",
            closure_call_target(call).expect("guarded by the match arm"),
            emit_arguments(call.args.iter())?.join(", ")
        ),
        syn::Expr::Call(call) => format!(
            "{}({})",
            emit_expression(&call.func)?,
            emit_arguments(call.args.iter())?.join(", ")
        ),
        syn::Expr::Cast(cast) => {
            format!("{}({})", emit_type(&cast.ty)?, emit_expression(&cast.expr)?)
        }
        syn::Expr::Field(field) => format!(
            "{}.{}",
            emit_expression(&field.base)?,
            match &field.member {
                syn::Member::Named(ident) => emit_ident(ident),
                // A tuple element: Slang's `Tuple` names its elements `_0`, `_1`, ...
                syn::Member::Unnamed(index) => format!("_{}", index.index),
            }
        ),
        syn::Expr::Tuple(tuple) if !tuple.elems.is_empty() => {
            format!("makeTuple({})", emit_list(tuple.elems.iter())?.join(", "))
        }
        syn::Expr::Group(group) => emit_expression(&group.expr)?,
        syn::Expr::Index(index) => {
            // The SPIR-V route's guard (D172): naga's `Restrict` policy for arrays, as explicit code. A non-literal
            // index of an array whose length the tables name is clamped to `min(uint(i), N - 1)`. A storage buffer
            // has no length here and is not clamped (D174b).
            let position = emit_expression(&index.index)?;
            let position = match array_place_length_of(&index.expr) {
                Some(length)
                    if guards::enabled() && length > 0 && !is_literal_operand(&index.index) =>
                {
                    format!("min(uint({position}), {}u)", length - 1)
                }
                _ => position,
            };
            format!("{}[{}]", emit_expression(&index.expr)?, position)
        }
        syn::Expr::Lit(literal) => emit_literal(&literal.lit)?,
        syn::Expr::MethodCall(call) if is_math_method(call) => emit_math_method(call)?,
        // The scalar math methods (slice m11.math); the validator has checked the argument count.
        syn::Expr::MethodCall(call)
            if crate::validate::scalar_math_arity(&crate::names::name_of(&call.method))
                .is_some_and(|arity| arity == call.args.len()) =>
        {
            emit_scalar_math(call)?
        }
        // The wrapping methods are the plain operators, even in a module that names a matrix (where `*` is
        // `__gust_mul`): WGSL integer arithmetic wraps modulo 2^32 (section 6.2.3). The macro sees no types, so a
        // negation is `0 - a`, which slangc types as the operand's own type and lowers without a unary minus
        // (WGSL has no negation of a `u32`).
        syn::Expr::MethodCall(call)
            if crate::validate::wrapping_arity(&crate::names::name_of(&call.method))
                .is_some_and(|arity| arity == call.args.len()) =>
        {
            let receiver = emit_expression(&call.receiver)?;
            match crate::names::name_of(&call.method).as_str() {
                "wrapping_neg" => format!("(0 - {receiver})"),
                name => {
                    let operator = match name {
                        "wrapping_add" => '+',
                        "wrapping_sub" => '-',
                        "wrapping_mul" => '*',
                        other => unreachable!(
                            "`{other}` is accepted by `wrapping_arity` but has no emitter arm"
                        ),
                    };
                    format!(
                        "({receiver} {operator} {})",
                        emit_expression(&call.args[0])?
                    )
                }
            }
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "global_id") => {
            emit_expression(&call.receiver)?
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "len") => {
            // The length of an array is its `N` (slice m9.r1); the length of a storage parameter is a constant of the
            // dispatch. An array field (`s.items.len()`) and an array element of a buffer (`out[i].len()`) are judged by
            // the tables of the module and of the kernel (slice m9.r2a).
            if let Some(length) = array_place_length_of(&call.receiver) {
                return Ok(length.to_string());
            }
            let receiver = emit_expression(&call.receiver)?;
            format!("__gpu_len_{receiver}")
        }
        // `load`, `store`, `sample`, `width` and `height` of a texture parameter (slice mx.t3); the validator has
        // refused them on anything else.
        syn::Expr::MethodCall(call) if texture::is_texture_method(call) => {
            texture::emit_method(call)?
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "is_some") => {
            format!("{}.hasValue", emit_expression(&call.receiver)?)
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "is_none") => {
            format!("(!{}.hasValue)", emit_expression(&call.receiver)?)
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "is_ok") => {
            format!("{}.isOk", emit_expression(&call.receiver)?)
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "is_err") => {
            format!("(!{}.isOk)", emit_expression(&call.receiver)?)
        }
        syn::Expr::MethodCall(call)
            if crate::names::ident_is(&call.method, "unwrap_or") && call.args.len() == 1 =>
        {
            format!(
                "__gust_unwrap_or({}, {})",
                emit_expression(&call.receiver)?,
                emit_expression(&call.args[0])?
            )
        }
        // Every other method call is a method of the module: the validator refuses a name that is neither a dialect method
        // nor one (GUST-E0305, E0325), and a method cannot take a dialect name (GUST-E0328). Slang resolves the receiver's
        // type, so the call is the member call as written (D156).
        syn::Expr::MethodCall(call) => format!(
            "{}.{}({})",
            emit_expression(&call.receiver)?,
            emit_ident(&call.method),
            emit_arguments(call.args.iter())?.join(", ")
        ),
        syn::Expr::Paren(paren) => format!("({})", emit_expression(&paren.expr)?),
        syn::Expr::Path(path) if crate::names::path_is(&path.path, "None") => "none".to_owned(),
        // `std::f32::consts::{PI, TAU, E}` (slice m11.math): the shortest decimal that parses back to the same `f32`.
        syn::Expr::Path(path)
            if path.qself.is_none() && crate::validate::math_constant(&path.path).is_some() =>
        {
            let value =
                crate::validate::math_constant(&path.path).expect("guarded by the match arm");
            format!("float({value:?})")
        }
        // The receiver of a method is `this` in Slang (slice m9.a1).
        syn::Expr::Path(path)
            if path.qself.is_none() && crate::names::path_is(&path.path, "self") =>
        {
            "this".to_owned()
        }
        syn::Expr::Path(path)
            if path.qself.is_none()
                && matches!(
                    math_types::associated_item(&path.path),
                    Some((_, "IDENTITY"))
                ) =>
        {
            let (math, _) =
                math_types::associated_item(&path.path).expect("guarded by the match arm");
            identity_matrix(math)
        }
        // A single-segment path is a value name (a local, a parameter or a helper), so it is renamed like its
        // declaration. A longer path names a type or module first and is emitted as written.
        syn::Expr::Path(path) if path.path.get_ident().is_some() && path.qself.is_none() => {
            emit_ident(path.path.get_ident().expect("guarded by the match arm"))
        }
        // `E::V`, a variant of an enum of the module (the validator took no other two-segment path value).
        syn::Expr::Path(path)
            if path.qself.is_none()
                && path.path.leading_colon.is_none()
                && path.path.segments.len() == 2 =>
        {
            emit_variant_path(&path.path)
        }
        syn::Expr::Path(path) => path.path.to_token_stream().to_string().replace(" :: ", "."),
        syn::Expr::Return(expression) => match &expression.expr {
            Some(value) => format!("return {}", emit_value_expression(value)?),
            None => "return".to_owned(),
        },
        syn::Expr::Struct(value) => {
            return Err(diagnostics::error(
                Code::StructLiteralPosition,
                value,
                "GPU struct literals are supported only as a `let` initializer or the right side of an assignment; bind the struct to a local first, or copy a buffer element and update its fields",
            ));
        }
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Not(_)) => {
            format!("__gust_not({})", emit_expression(&unary.expr)?)
        }
        syn::Expr::Unary(unary) if negated_integer_literal(unary).is_some() => {
            let value = negated_integer_literal(unary).expect("guarded by the match arm");
            format!("(-{})", integer_literal(value, true)?)
        }
        // A literal operand is a scalar whatever the module names, so it keeps the plain `-` (the integer literal above
        // has its own arm); every other operand of a unary `-` is `__gust_neg(a)` in a module that names a matrix.
        syn::Expr::Unary(unary)
            if lowering().neg
                && matches!(unary.op, syn::UnOp::Neg(_))
                && !matches!(unary.expr.as_ref(), syn::Expr::Lit(_)) =>
        {
            format!("__gust_neg({})", emit_expression(&unary.expr)?)
        }
        syn::Expr::Unary(unary) => format!(
            "({}{})",
            unary.op.to_token_stream(),
            emit_expression(&unary.expr)?
        ),
        syn::Expr::If(branch) => {
            return Err(diagnostics::error(
                Code::UnsupportedExpression,
                branch,
                "an `if` gives a value only as the whole initializer of a `let` (`let x: u32 = if c { 1u32 } else { 2u32 };`), with an `else`: bind it to a local first, then use the local",
            ));
        }
        other => {
            return Err(diagnostics::error(
                Code::UnsupportedExpression,
                other,
                "this Rust expression is not in the Rust-to-Slang subset yet",
            ));
        }
    })
}

/// The Slang spelling of an integer literal; `negated` says a unary minus applies to it directly.
///
/// The dialect has no 64-bit integers, but slangc types a bare decimal above `i32::MAX` as `int64_t`, which WGSL
/// and SPIR-V validation refuse once it meets a 32-bit operand. In Rust such an unsuffixed literal can only be
/// `u32` (an `i32` one is a compile error), so it is spelled `uint(..)`, exactly as `u32`-suffixed ones are. Bare
/// digits stay for everything else: values up to `i32::MAX`, a negated literal (`-2147483648` is `i32::MIN` and
/// slangc types it right), and a value above `u32::MAX`, which no 32-bit type holds and which stays loud (slangc
/// keeps it 64-bit and shader creation refuses it) instead of being truncated by a `uint(..)`. The M6 vector
/// operands call this too.
pub(super) fn integer_literal(value: &syn::LitInt, negated: bool) -> syn::Result<String> {
    let digits = value.base10_digits();
    match value.suffix() {
        "" => {
            let above_i32 = digits
                .parse::<u64>()
                .is_ok_and(|v| v > i32::MAX as u64 && v <= u64::from(u32::MAX));
            Ok(if above_i32 && !negated {
                format!("uint({digits})")
            } else {
                digits.to_owned()
            })
        }
        "u32" => Ok(format!("uint({digits})")),
        "i32" => Ok(format!("int({digits})")),
        "f32" => Ok(format!("float({digits}.0)")),
        _ => Err(diagnostics::error(
            Code::UnsupportedLiteral,
            value,
            "GPU literals support booleans and 32-bit numbers (i32, u32, f32) only",
        )),
    }
}

pub(super) fn emit_literal(literal: &syn::Lit) -> syn::Result<String> {
    let unsupported = || {
        diagnostics::error(
            Code::UnsupportedLiteral,
            literal,
            "GPU literals support booleans and 32-bit numbers (i32, u32, f32) only",
        )
    };
    match literal {
        syn::Lit::Bool(value) => Ok(value.value.to_string()),
        syn::Lit::Int(value) => integer_literal(value, false),
        syn::Lit::Float(value) => match value.suffix() {
            "" => Ok(value.base10_digits().to_owned()),
            "f32" => Ok(format!("float({})", value.base10_digits())),
            _ => Err(unsupported()),
        },
        _ => Err(unsupported()),
    }
}

/// [`emit_expression`] for a position where an array literal has a lowering: the flag lives only for the one expression
/// when it is a literal, so it cannot let a literal nested somewhere else through.
pub(super) fn emit_value_expression(expression: &syn::Expr) -> syn::Result<String> {
    ARRAY_LITERAL_OK.with(|flag| {
        flag.set(matches!(peel_expr(expression), syn::Expr::Array(_)));
    });
    // A parenthesized literal is emitted without the parentheses: `({ .. })` is not an initializer list to slangc.
    let inner = peel_expr(expression);
    let text = emit_expression(if matches!(inner, syn::Expr::Array(_)) {
        inner
    } else {
        expression
    });
    ARRAY_LITERAL_OK.with(|flag| flag.set(false));
    text
}

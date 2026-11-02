use std::collections::{HashMap, HashSet};

use quote::ToTokens;
use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};

use crate::diagnostics::{self, Code};
use crate::math_types;

const BANNED_NAMES: &[&str] = &[
    "std",
    "core",
    "alloc",
    "String",
    "Vec",
    "Box",
    "Rc",
    "Arc",
    "HashMap",
    "HashSet",
    "println",
    "print",
    "eprintln",
    "dbg",
    "panic",
    "thread",
    "spawn",
    "File",
    "TcpStream",
];

/// Numeric conversions with a proven Slang lowering and GPU-verified semantics.
const CAST_TARGETS: &[&str] = &["f32", "float", "i32", "int", "u32", "uint"];

/// Rust primitives outside the four-byte scalar subset. They are not shadowed by
/// the prelude, so rustc accepts them, and `emit_type` would pass the Rust
/// spelling through to Slang unchanged.
const UNSUPPORTED_PRIMITIVES: &[&str] = &[
    "i8", "i16", "i64", "i128", "isize", "u8", "u16", "u64", "u128", "usize", "f16", "f64", "f128",
    "char", "str",
];

/// The `Option<T>` shape this dialect lowers to Slang `Optional<T>`: a single
/// segment with exactly one type argument.
fn option_payload(ty: &syn::TypePath) -> Option<Option<&syn::Type>> {
    if ty.qself.is_some() || ty.path.segments.len() != 1 {
        return None;
    }
    let segment = &ty.path.segments[0];
    if segment.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Some(None);
    };
    let mut types = arguments.args.iter().filter_map(|argument| match argument {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    match (types.next(), types.next(), arguments.args.len()) {
        (Some(payload), None, 1) => Some(Some(payload)),
        _ => Some(None),
    }
}

/// The deterministic `Result<T, T>` subset lowered to `__GustResult<T>`.
fn result_payload(ty: &syn::TypePath) -> Option<Option<(&syn::Type, &syn::Type)>> {
    if ty.qself.is_some() || ty.path.segments.len() != 1 {
        return None;
    }
    let segment = &ty.path.segments[0];
    if segment.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Some(None);
    };
    let types = arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect::<Vec<_>>();
    match types.as_slice() {
        [ok, err] if arguments.args.len() == 2 => Some(Some((*ok, *err))),
        _ => Some(None),
    }
}

/// The element type of an `RWStructuredBuffer<T>` parameter, when the type is a
/// single-segment path with exactly one type argument.
fn rw_buffer_element_type(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    let segment = &type_path.path.segments[0];
    if segment.ident != "RWStructuredBuffer" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    let syn::GenericArgument::Type(element) = arguments.args.first()? else {
        return None;
    };
    let syn::Type::Path(element_path) = element else {
        return None;
    };
    element_path.path.get_ident().map(ToString::to_string)
}

/// `Some(identifier)` in an `if let`, the only pattern with a direct Slang lowering.
pub fn option_pattern(pattern: &syn::Pat) -> Option<&syn::PatIdent> {
    let syn::Pat::TupleStruct(tuple) = pattern else {
        return None;
    };
    if !tuple.path.is_ident("Some") || tuple.elems.len() != 1 {
        return None;
    }
    match &tuple.elems[0] {
        syn::Pat::Ident(ident) => Some(ident),
        _ => None,
    }
}

pub fn result_pattern(pattern: &syn::Pat) -> Option<&syn::PatIdent> {
    let syn::Pat::TupleStruct(tuple) = pattern else {
        return None;
    };
    if !tuple.path.is_ident("Ok") || tuple.elems.len() != 1 {
        return None;
    }
    match &tuple.elems[0] {
        syn::Pat::Ident(ident) => Some(ident),
        _ => None,
    }
}

fn err_pattern(pattern: &syn::Pat) -> Option<&syn::PatIdent> {
    let syn::Pat::TupleStruct(tuple) = pattern else {
        return None;
    };
    if !tuple.path.is_ident("Err") || tuple.elems.len() != 1 {
        return None;
    }
    match &tuple.elems[0] {
        syn::Pat::Ident(ident) => Some(ident),
        _ => None,
    }
}

pub fn result_match_arms(
    expression: &syn::ExprMatch,
) -> syn::Result<(&syn::PatIdent, &syn::Block, &syn::PatIdent, &syn::Block)> {
    let mut ok = None;
    let mut err = None;
    for arm in &expression.arms {
        if !arm.attrs.is_empty() {
            return Err(diagnostics::error(
                Code::MatchArmAttribute,
                &arm.attrs[0],
                "match arm attributes are not supported",
            ));
        }
        if arm.guard.is_some() {
            return Err(diagnostics::error(
                Code::MatchGuard,
                arm,
                "match guards are not supported in the Rust-to-Slang subset",
            ));
        }
        let syn::Expr::Block(body) = arm.body.as_ref() else {
            return Err(diagnostics::error(
                Code::MatchArmBlock,
                &arm.body,
                "Result matches require block arms",
            ));
        };
        if let Some(binding) = result_pattern(&arm.pat) {
            if binding.mutability.is_some() || binding.by_ref.is_some() || binding.subpat.is_some()
            {
                return Err(diagnostics::error(
                    Code::MatchBinding,
                    binding,
                    "Result match bindings must be plain identifiers; `mut`, `ref`, and @ bindings are not supported",
                ));
            }
            if ok.replace((binding, &body.block)).is_some() {
                return Err(diagnostics::error(
                    Code::MatchArmCount,
                    &arm.pat,
                    "Result matches require exactly one `Ok` arm and one `Err` arm",
                ));
            }
        } else if let Some(binding) = err_pattern(&arm.pat) {
            if binding.mutability.is_some() || binding.by_ref.is_some() || binding.subpat.is_some()
            {
                return Err(diagnostics::error(
                    Code::MatchBinding,
                    binding,
                    "Result match bindings must be plain identifiers; `mut`, `ref`, and @ bindings are not supported",
                ));
            }
            if err.replace((binding, &body.block)).is_some() {
                return Err(diagnostics::error(
                    Code::MatchArmCount,
                    &arm.pat,
                    "Result matches require exactly one `Ok` arm and one `Err` arm",
                ));
            }
        } else {
            return Err(diagnostics::error(
                Code::MatchPattern,
                &arm.pat,
                "Result matches support only `Ok(identifier)` and `Err(identifier)` patterns",
            ));
        }
    }
    match (ok, err) {
        (Some((ok_name, ok_body)), Some((err_name, err_body))) => {
            Ok((ok_name, ok_body, err_name, err_body))
        }
        _ => Err(diagnostics::error(
            Code::MatchArmCount,
            expression,
            "Result matches require exactly one `Ok` arm and one `Err` arm",
        )),
    }
}

/// The atomic operation name if the path is a recognized atomic intrinsic.
fn atomic_operation_name(path: &syn::ExprPath) -> Option<&'static str> {
    match path.path.get_ident()?.to_string().as_str() {
        "atomic_add" => Some("atomic_add"),
        "atomic_min" => Some("atomic_min"),
        "atomic_max" => Some("atomic_max"),
        "atomic_exchange" => Some("atomic_exchange"),
        "atomic_compare_exchange" => Some("atomic_compare_exchange"),
        _ => None,
    }
}

/// Whether an expression is an integer literal with value zero, including a
/// unary negation of one. Float zeros are excluded: floating-point division by
/// zero is defined (it yields infinity/NaN), so only integer div/rem by zero is
/// a portable-semantic hazard.
fn is_literal_zero(expression: &syn::Expr) -> bool {
    match expression {
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

/// A loop bound that is a bare integer literal (through parentheses and unary
/// minus) without a type suffix. Slang would type it `int` while rustc may infer
/// `u32` from the other bound, so the counter types would silently diverge.
fn is_unsuffixed_integer_literal(expression: &syn::Expr) -> bool {
    match expression {
        syn::Expr::Lit(literal) => {
            matches!(&literal.lit, syn::Lit::Int(value) if value.suffix().is_empty())
        }
        syn::Expr::Paren(paren) => is_unsuffixed_integer_literal(&paren.expr),
        syn::Expr::Group(group) => is_unsuffixed_integer_literal(&group.expr),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            is_unsuffixed_integer_literal(&unary.expr)
        }
        _ => false,
    }
}

/// A rejection reason without a span: the code and message the caller attaches to
/// the node it was checking (see [`diagnostics::error`]).
pub type Reason = (Code, &'static str);

/// The loop-variable binding of a `for` loop: `_` or a plain immutable identifier.
pub fn loop_variable(pattern: &syn::Pat) -> Result<Option<&syn::PatIdent>, Reason> {
    match pattern {
        syn::Pat::Wild(_) => Ok(None),
        syn::Pat::Ident(ident) if ident.mutability.is_some() => Err((
            Code::MutableLoopVariable,
            "the loop variable is immutable in the Rust-to-Slang subset: Slang's counter would diverge from Rust's fresh per-iteration binding; copy it into a `let mut` local",
        )),
        syn::Pat::Ident(ident) if ident.by_ref.is_some() || ident.subpat.is_some() => Err((
            Code::BindingMode,
            "reference and @ binding modes are not supported in GPU code",
        )),
        syn::Pat::Ident(ident) => Ok(Some(ident)),
        _ => Err((
            Code::LoopVariablePattern,
            "the `for` loop variable must be a single identifier or `_`",
        )),
    }
}

/// The `start..end` bounds of a `for` loop, or the diagnostic explaining why the
/// iterable has no bounded Slang lowering.
pub fn range_bounds(iterable: &syn::Expr) -> Result<(&syn::Expr, &syn::Expr), Reason> {
    let syn::Expr::Range(range) = iterable else {
        return Err((
            Code::ForIterable,
            "`for` iterates only over an exclusive integer range `start..end`; buffers, iterators, and adaptors are not iterable in the Rust-to-Slang subset",
        ));
    };
    if matches!(range.limits, syn::RangeLimits::Closed(_)) {
        return Err((
            Code::InclusiveRange,
            "inclusive ranges (`..=`) are not supported: when `end` is the type maximum the Slang counter never exceeds it and the loop would not terminate; use `start..end + 1`",
        ));
    }
    match (range.start.as_deref(), range.end.as_deref()) {
        (Some(start), Some(end)) => Ok((start, end)),
        _ => Err((
            Code::OpenRange,
            "`for` ranges need both bounds (`start..end`) so the trip count is fixed at loop entry",
        )),
    }
}

pub fn validate_module(module: &syn::ItemMod) -> syn::Result<()> {
    let Some((_, items)) = &module.content else {
        return Err(diagnostics::error(
            Code::InlineModuleRequired,
            module,
            "#[gpu] requires an inline module; use `#[gpu] mod name { ... }`",
        ));
    };

    let function_names: HashSet<String> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if !function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .collect();

    let result_helpers = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if matches!(&function.sig.output, syn::ReturnType::Type(_, ty) if matches!(ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some())) =>
            {
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .collect();

    for item in items {
        match item {
            syn::Item::Fn(function) => {
                validate_function(function, &function_names, &result_helpers)?
            }
            syn::Item::Struct(item) => validate_struct(item)?,
            _ => {
                return Err(diagnostics::error(
                    Code::UnsupportedModuleItem,
                    item,
                    "GPU modules may currently contain only functions and plain structs",
                ));
            }
        }
    }
    let structs: HashMap<String, &syn::ItemStruct> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some((item.ident.unraw().to_string(), item)),
            _ => None,
        })
        .collect();
    // The shape of the uniform parameters and of the structs they reach comes before the padding checks: a `Mat2` in
    // a uniform struct is E0205 whatever else is wrong with its layout.
    crate::uniform::check_parameters(items)?;
    for item in items {
        if let syn::Item::Struct(item) = item {
            check_padding(item, &structs)?;
        }
    }
    crate::uniform::check_padding(items)?;
    Ok(())
}

/// The size and alignment of a field type the macro can compute from names alone: a scalar, a vector, a matrix or a
/// struct of the module whose fields are all such types. `None` when any part is unknown (a type the macro cannot see
/// into), so the `const` assertions the expansion emits stay the check for it.
fn field_layout(
    ty: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<(u32, u32)> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.leading_colon.is_some() {
        return None;
    }
    let ident = path.path.get_ident()?;
    let name = ident.unraw().to_string();
    if let Some(layout) = math_types::known_layout(&name) {
        return Some(layout);
    }
    // A struct cannot contain itself, but the guard keeps a malformed module from recursing.
    if depth > 16 {
        return None;
    }
    struct_layout(structs.get(&name)?, structs, depth + 1)
        .map(|(size, alignment, _)| (size, alignment))
}

/// A struct's `#[repr(C)]` size, alignment and the offsets of its fields, when every field has a known layout.
fn struct_layout(
    item: &syn::ItemStruct,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<(u32, u32, Vec<u32>)> {
    let syn::Fields::Named(fields) = &item.fields else {
        return None;
    };
    let mut offset = 0_u32;
    let mut alignment = 1_u32;
    let mut offsets = Vec::new();
    for field in &fields.named {
        let (size, field_alignment) = field_layout(&field.ty, structs, depth)?;
        offset = offset.checked_next_multiple_of(field_alignment)?;
        offsets.push(offset);
        offset = offset.checked_add(size)?;
        alignment = alignment.max(field_alignment);
    }
    Some((
        offset.checked_next_multiple_of(alignment)?,
        alignment,
        offsets,
    ))
}

/// E0208: a struct whose layout has implicit padding is not a `GpuPod` type (`pod_slice_as_bytes` would read the
/// padding bytes, which are uninitialised). The model covers scalars, vectors, matrices and the module's structs;
/// anything else is left to the expansion's `const` assertions.
fn check_padding(
    item: &syn::ItemStruct,
    structs: &HashMap<String, &syn::ItemStruct>,
) -> syn::Result<()> {
    let syn::Fields::Named(fields) = &item.fields else {
        return Ok(());
    };
    let Some((size, _, offsets)) = struct_layout(item, structs, 0) else {
        return Ok(());
    };
    let mut end = 0_u32;
    for (field, offset) in fields.named.iter().zip(offsets) {
        let Some((field_size, _)) = field_layout(&field.ty, structs, 0) else {
            return Ok(());
        };
        let name = field
            .ident
            .as_ref()
            .map_or_else(String::new, ToString::to_string);
        if offset != end {
            return Err(diagnostics::error(
                Code::ImplicitPadding,
                field,
                format!(
                    "field `{name}` of `{}` would sit at offset {offset}, {} bytes after the previous field ends at {end}: GPU structs may not contain implicit padding; order the fields by decreasing alignment, or fill the gap with an explicit field",
                    item.ident,
                    offset - end,
                ),
            ));
        }
        end = offset + field_size;
    }
    if size != end {
        return Err(diagnostics::error(
            Code::ImplicitPadding,
            &item.ident,
            format!(
                "`{}` ends at byte {end} but its size is {size}: GPU structs may not contain implicit tail padding; add explicit fields up to a multiple of the largest field alignment",
                item.ident,
            ),
        ));
    }
    Ok(())
}

fn validate_struct(item: &syn::ItemStruct) -> syn::Result<()> {
    let struct_name = item.ident.unraw().to_string();
    if math_types::is_reserved_type_name(&struct_name) {
        return Err(diagnostics::error(
            Code::ReservedTypeName,
            &item.ident,
            format!(
                "`{struct_name}` is a type the dialect maps to a Slang type itself, so a struct cannot be named that; choose another name"
            ),
        ));
    }
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(diagnostics::error(
            Code::GenericStruct,
            &item.generics,
            "generic GPU structs are not supported yet",
        ));
    }
    let syn::Fields::Named(fields) = &item.fields else {
        return Err(diagnostics::error(
            Code::StructNeedsNamedFields,
            item,
            "GPU structs must have named fields",
        ));
    };
    if fields.named.is_empty() {
        return Err(diagnostics::error(
            Code::EmptyStruct,
            item,
            "GPU structs must contain at least one field",
        ));
    }
    if let Some(attribute) = item
        .attrs
        .iter()
        .find(|attribute| attribute.path().is_ident("repr"))
    {
        return Err(diagnostics::error(
            Code::ManualRepr,
            attribute,
            "GPU structs receive repr(C) automatically",
        ));
    }
    for attribute in item
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("derive"))
    {
        let derives = diagnostics::parse_args_with(
            attribute,
            Code::MalformedDerive,
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        )?;
        if derives.iter().any(|path| {
            path.segments
                .last()
                .is_some_and(|segment| segment.ident == "Clone" || segment.ident == "Copy")
        }) {
            return Err(diagnostics::error(
                Code::ManualCloneCopy,
                attribute,
                "GPU structs receive Clone and Copy automatically",
            ));
        }
    }
    let mut visitor = RestrictedVisitor::new(HashSet::new(), HashSet::new());
    visitor.option_forbidden = Some("GPU struct fields");
    visitor.result_forbidden = Some("GPU struct fields");
    visitor.visit_item_struct(item);
    visitor.finish()
}

fn validate_function(
    function: &syn::ItemFn,
    function_names: &HashSet<String>,
    result_helpers: &HashSet<String>,
) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor};

    let is_kernel = crate::expand::kernel_options(function)?.is_some();
    let invocations = function.sig.inputs.iter().filter(|argument| matches!(argument,
        syn::FnArg::Typed(argument) if parameter_flavor(&argument.ty) == ParameterFlavor::Invocation)).count();
    if is_kernel {
        if invocations != 1 {
            return Err(diagnostics::error(
                Code::KernelDispatchParameter,
                &function.sig,
                "a GPU kernel requires exactly one dispatch-thread parameter",
            ));
        }
        if !matches!(&function.sig.output, syn::ReturnType::Default)
            && !matches!(&function.sig.output, syn::ReturnType::Type(_, ty) if matches!(ty.as_ref(), syn::Type::Tuple(tuple) if tuple.elems.is_empty())) {
            return Err(diagnostics::error(
                Code::KernelReturnsUnit,
                &function.sig.output,
                "GPU kernels must return unit",
            ));
        }
    } else if function.sig.inputs.iter().any(|argument| matches!(argument,
        syn::FnArg::Typed(argument) if matches!(parameter_flavor(&argument.ty), ParameterFlavor::StorageRead | ParameterFlavor::StorageReadWrite | ParameterFlavor::Uniform))) {
        return Err(diagnostics::error(
            Code::HelperResourceParameter,
            &function.sig,
            "GPU helper functions currently accept value parameters only; access resources in the kernel",
        ));
    }
    if function.sig.asyncness.is_some() {
        return Err(diagnostics::error(
            Code::AsyncFunction,
            &function.sig,
            "async is CPU-only in GPU modules",
        ));
    }
    if function.sig.unsafety.is_some() {
        return Err(diagnostics::error(
            Code::UnsafeFunction,
            &function.sig,
            "unsafe is not part of the Rust-to-Slang subset",
        ));
    }
    if function.sig.abi.is_some() || !function.sig.generics.params.is_empty() {
        return Err(diagnostics::error(
            Code::ExternOrGenericFunction,
            &function.sig,
            "extern and generic GPU functions are not supported yet",
        ));
    }

    let mut visitor = RestrictedVisitor::new(function_names.clone(), result_helpers.clone());
    visitor.atomic_buffers = atomic_buffer_names(function);
    for argument in &function.sig.inputs {
        if let syn::FnArg::Typed(argument) = argument
            && let syn::Pat::Ident(name) = argument.pat.as_ref()
        {
            match parameter_flavor(&argument.ty) {
                ParameterFlavor::Invocation => {
                    visitor.invocations.insert(name.ident.to_string());
                }
                ParameterFlavor::StorageRead => {
                    visitor.resources.insert(name.ident.to_string());
                }
                ParameterFlavor::Uniform => {
                    visitor.uniforms.insert(name.ident.to_string());
                }
                ParameterFlavor::StorageReadWrite => {
                    visitor.resources.insert(name.ident.to_string());
                    if let Some(element) = rw_buffer_element_type(&argument.ty) {
                        visitor
                            .read_write_buffers
                            .insert(name.ident.to_string(), element);
                    }
                }
                ParameterFlavor::Value if matches!(argument.ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some()) =>
                {
                    visitor.result_values.insert(name.ident.to_string());
                }
                _ => {}
            }
        }
    }
    visitor.visit_item_fn(function);
    visitor.finish()
}

fn atomic_buffer_names(function: &syn::ItemFn) -> HashSet<String> {
    let mut usage = AtomicBufferUsage(HashSet::new());
    usage.visit_item_fn(function);
    usage.0
}

struct AtomicBufferUsage(HashSet<String>);

impl<'ast> Visit<'ast> for AtomicBufferUsage {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let Some(_) = (match call.func.as_ref() {
            syn::Expr::Path(path) => atomic_operation_name(path),
            _ => None,
        }) else {
            visit::visit_expr_call(self, call);
            return;
        };
        if let Some(syn::Expr::Reference(reference)) = call.args.first()
            && let syn::Expr::Index(index) = reference.expr.as_ref()
            && let syn::Expr::Path(path) = index.expr.as_ref()
            && let Some(name) = path.path.get_ident()
        {
            self.0.insert(name.to_string());
        }
        visit::visit_expr_call(self, call);
    }
}

const QUALIFIED_MATH_PATH: &str = "math types are the prelude's: write `Vec4`, `Mat4` and their constructors without a module path (`Vec4::new`, `Mat4::IDENTITY`)";

/// Whether `path` reaches a math type through a module path (`super::Mat4`, `gust::math::Vec4::new`, `::Mat4`) instead
/// of by the prelude name. `Vec4::new` and `Mat4::IDENTITY` are the prelude form and are not qualified.
fn is_qualified_math_path(path: &syn::Path) -> bool {
    let names_math_type = path
        .segments
        .iter()
        .any(|segment| math_types::math_type(&segment.ident.unraw().to_string()).is_some());
    if !names_math_type {
        return false;
    }
    if path.leading_colon.is_some() {
        return true;
    }
    if path.segments.len() == 1 {
        return false;
    }
    // Two segments whose first is the math type are the prelude form, whatever the second is (a member the type does
    // not have is the caller's error, not a qualification).
    !(path.segments.len() == 2
        && math_types::math_type(&path.segments[0].ident.unraw().to_string()).is_some())
}

struct RestrictedVisitor {
    function_names: HashSet<String>,
    result_helpers: HashSet<String>,
    result_values: HashSet<String>,
    resources: HashSet<String>,
    invocations: HashSet<String>,
    /// Uniform parameters. They are read field by field only (E0321), and are not in `resources` because `.len()` of
    /// one is E0306.
    uniforms: HashSet<String>,
    /// Read-write buffer parameters and their element type, for atomic receivers.
    read_write_buffers: HashMap<String, String>,
    /// Buffer parameters used by an atomic operation in this function.
    atomic_buffers: HashSet<String>,
    /// Set while inside a type position where `Option` has no proven layout.
    option_forbidden: Option<&'static str>,
    /// Set while inside a type position where `Result` has no proven layout.
    result_forbidden: Option<&'static str>,
    /// Number of enclosing `for` loops; `break`/`continue` need at least one.
    loop_depth: usize,
    error: Option<syn::Error>,
}

impl RestrictedVisitor {
    fn new(function_names: HashSet<String>, result_helpers: HashSet<String>) -> Self {
        Self {
            function_names,
            result_helpers,
            result_values: HashSet::new(),
            resources: HashSet::new(),
            invocations: HashSet::new(),
            uniforms: HashSet::new(),
            read_write_buffers: HashMap::new(),
            atomic_buffers: HashSet::new(),
            option_forbidden: None,
            result_forbidden: None,
            loop_depth: 0,
            error: None,
        }
    }

    /// Whether `name` is a kernel parameter: a local cannot shadow one.
    fn is_parameter_name(&self, name: &str) -> bool {
        self.resources.contains(name)
            || self.invocations.contains(name)
            || self.uniforms.contains(name)
    }

    /// Whether `expression` is a bare use of a uniform parameter, possibly in parentheses.
    fn is_uniform_parameter(&self, expression: &syn::Expr) -> bool {
        match expression {
            syn::Expr::Path(path) => {
                path.qself.is_none()
                    && path
                        .path
                        .get_ident()
                        .is_some_and(|name| self.uniforms.contains(&name.to_string()))
            }
            syn::Expr::Paren(paren) => self.is_uniform_parameter(&paren.expr),
            syn::Expr::Group(group) => self.is_uniform_parameter(&group.expr),
            _ => false,
        }
    }

    fn reject(&mut self, code: Code, node: impl quote::ToTokens, message: &str) {
        self.push(diagnostics::error(code, node, message));
    }

    /// Adds `error` after the ones already collected.
    fn push(&mut self, error: syn::Error) {
        if let Some(existing) = &mut self.error {
            existing.combine(error);
        } else {
            self.error = Some(error);
        }
    }

    fn finish(self) -> syn::Result<()> {
        self.error.map_or(Ok(()), Err)
    }

    fn is_known_result(&self, expression: &syn::Expr) -> bool {
        match expression {
            syn::Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|name| self.result_values.contains(&name.to_string())),
            syn::Expr::Call(call) => match call.func.as_ref() {
                syn::Expr::Path(path) => path
                    .path
                    .get_ident()
                    .is_some_and(|name| self.result_helpers.contains(&name.to_string())),
                _ => false,
            },
            syn::Expr::Paren(paren) => self.is_known_result(&paren.expr),
            syn::Expr::Group(group) => self.is_known_result(&group.expr),
            _ => false,
        }
    }

    /// Shared checks for `break` and `continue` in statement position.
    fn check_jump(
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
        }
    }

    /// Validates an atomic operation call on a read-write buffer element.
    /// Accepts `atomic_add`, `atomic_min`, `atomic_max`, `atomic_exchange`,
    /// and `atomic_compare_exchange` on `u32`/`uint` and `i32`/`int` elements.
    /// `atomic_min` and `atomic_max` are unsigned-only (Slang/WGSL constraint).
    fn validate_atomic(&mut self, op_name: &str, call: &syn::ExprCall) {
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
        let buffer_name = buffer_path.path.get_ident().map(ToString::to_string);
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
        if !["doc", "allow", "warn", "deny", "forbid", "derive", "kernel"]
            .iter()
            .any(|name| attribute.path().is_ident(name))
        {
            self.reject(
                Code::UnsupportedAttribute,
                attribute,
                "this attribute has no proven Rust-to-Slang meaning and cannot be dropped",
            );
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
        for segment in &path.segments {
            if BANNED_NAMES.contains(&segment.ident.to_string().as_str()) {
                self.reject(
                    Code::CpuOnlyName,
                    path,
                    "CPU-only library/type is not available inside a #[gpu] module",
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
            let name = ident.to_string();
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
        let supported = match cast.ty.as_ref() {
            syn::Type::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| CAST_TARGETS.contains(&ident.to_string().as_str())),
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
        if let syn::Expr::Path(path) = call.func.as_ref() {
            let contains_banned_name = path
                .path
                .segments
                .iter()
                .any(|segment| BANNED_NAMES.contains(&segment.ident.to_string().as_str()));
            if contains_banned_name {
                self.reject(
                    Code::CpuOnlyName,
                    path,
                    "CPU-only library/type is not available inside a #[gpu] module",
                );
                return;
            }
            if is_qualified_math_path(&path.path) {
                self.reject(Code::QualifiedMathPath, path, QUALIFIED_MATH_PATH);
                return;
            }
            if path.path.is_ident("Some") {
                if call.args.len() != 1 {
                    self.reject(
                        Code::SomeArguments,
                        call,
                        "`Some` takes exactly one argument",
                    );
                    return;
                }
                // The callee is the prelude constructor, not a bare path value.
                self.visit_expr(&call.args[0]);
                return;
            }
            if path.path.is_ident("Ok") || path.path.is_ident("Err") {
                if call.args.len() != 1 {
                    self.reject(
                        Code::OkErrArguments,
                        call,
                        "`Ok` and `Err` take exactly one argument",
                    );
                    return;
                }
                self.visit_expr(&call.args[0]);
                return;
            }
            if let Some(op) = atomic_operation_name(path) {
                self.reject(
                    Code::AtomicAsValue,
                    call,
                    &format!("`{op}` is supported only as a statement, not as a value"),
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
        let allowed = match call.func.as_ref() {
            syn::Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| self.function_names.contains(&ident.to_string())),
            _ => false,
        };
        if !allowed {
            self.reject(
                Code::UnknownCall,
                call,
                "only calls to helper functions declared in the same GPU module are allowed",
            );
            return;
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_index(&mut self, index: &'ast syn::ExprIndex) {
        if let syn::Expr::Path(path) = index.expr.as_ref()
            && let Some(name) = path.path.get_ident()
            && self.atomic_buffers.contains(&name.to_string())
        {
            self.reject(
                Code::AtomicDirectAccess,
                index,
                "atomic buffer elements cannot be read or written directly; use an atomic operation",
            );
            return;
        }
        visit::visit_expr_index(self, index);
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        let is_div_or_rem = matches!(binary.op, syn::BinOp::Div(_) | syn::BinOp::Rem(_));
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
        if call.turbofish.is_some() {
            self.reject(
                Code::BuiltinTypeArguments,
                call,
                "GPU built-ins do not accept type arguments",
            );
            return;
        }
        let method = call.method.to_string();
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
        if !call.args.is_empty() {
            self.reject(
                Code::BuiltinArguments,
                call,
                "GPU built-ins do not accept arguments or type arguments",
            );
            return;
        }
        if !matches!(method.as_str(), "global_id" | "len") {
            self.reject(
                Code::UnsupportedMethod,
                call,
                "only the compatibility built-ins `.global_id()` and `.len()`, the Option/Result methods `.is_some()`, `.is_none()`, `.is_ok()`, `.is_err()`, `.unwrap_or(default)`, and the vector and matrix methods (swizzles such as `.xy()`, `.dot(v)`, `.mul_vec4(v)`, `.mul_mat4(m)`, `.transpose()`) are supported",
            );
            return;
        }
        let receiver = match call.receiver.as_ref() {
            syn::Expr::Path(path) => path.path.get_ident().map(ToString::to_string),
            _ => None,
        };
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
                    "`.len()` requires a direct storage parameter receiver",
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
            return;
        }
        visit::visit_expr_field(self, field);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        // Any other use of a uniform parameter by name copies or passes the whole struct: slangc crashes on a whole
        // struct copy out of a constant buffer on WGSL, and the twin would accept what the GPU cannot compile.
        if path.qself.is_none()
            && path
                .path
                .get_ident()
                .is_some_and(|name| self.uniforms.contains(&name.to_string()))
        {
            self.reject(
                Code::UniformWholeValue,
                path,
                "a uniform parameter is read one field at a time (`params.scale`, `params.view.mul_vec4(p)`); \
                 copying or passing the whole struct (`*params`, `let p = params;`) is not supported",
            );
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
        if is_qualified_math_path(&path.path) {
            self.reject(Code::QualifiedMathPath, path, QUALIFIED_MATH_PATH);
        } else if path.path.get_ident().is_none() {
            self.reject(
                Code::ForeignPath,
                path,
                "associated constants and foreign paths are not in the Rust-to-Slang subset; reference local variables, parameters, or same-module helpers",
            );
        } else if path.path.is_ident("Some")
            || path.path.is_ident("Ok")
            || path.path.is_ident("Err")
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
        let text = name.ident.to_string();
        if self.is_parameter_name(&text) {
            self.reject(
                Code::ShadowedParameter,
                name,
                "GPU locals cannot shadow resource or dispatch-thread parameters",
            );
            return;
        }
        self.visit_pat_ident(name);
        self.visit_expr(&binding.expr);
        self.visit_block(&expression.then_branch);
        if let Some((_, alternative)) = &expression.else_branch {
            self.visit_expr(alternative);
        }
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        self.reject(
            Code::MatchAsValue,
            expression,
            "Result match is supported only as a statement, not as a value",
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
        self.reject(
            Code::Closure,
            expression,
            "closures are not supported in the Rust-to-Slang subset",
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
            let name = variable.ident.to_string();
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
        self.visit_expr(start);
        self.visit_expr(end);
        self.loop_depth += 1;
        self.visit_block(&expression.body);
        self.loop_depth -= 1;
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
            syn::Stmt::Expr(syn::Expr::Match(expression), semicolon) => {
                let (ok_name, _, err_name, _) = match result_match_arms(expression) {
                    Ok(arms) => arms,
                    Err(error) => {
                        self.push(error);
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
                    let text = name.ident.to_string();
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

    fn visit_expr_reference(&mut self, expression: &'ast syn::ExprReference) {
        self.reject(
            Code::Reference,
            expression,
            "references are not part of the Rust-to-Slang value model",
        );
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Ok((pattern, _)) = crate::slang::local_binding(&local.pat) {
            let name = pattern.ident.to_string();
            if self.is_parameter_name(&name) {
                self.reject(
                    Code::ShadowedParameter,
                    local,
                    "GPU locals cannot shadow resource or dispatch-thread parameters",
                );
                return;
            }
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
        visit::visit_local(self, local);
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
            self.result_values.insert(pattern.ident.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::testing::{assert_code, assert_codes, codes_of_result};

    use super::*;

    #[test]
    fn rejects_cpu_allocation() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[kernel]
                pub fn bad(ctx: Invocation) {
                    let values = Vec::new();
                }
            }
        };
        let error = validate_module(&module).expect_err("Vec must be rejected");
        assert_code(&error, Code::CpuOnlyName);
    }

    #[test]
    fn independent_causes_report_every_code_in_source_order() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[kernel]
                pub fn bad(ctx: Invocation) {
                    loop { }
                    while true { }
                    let values = Vec::new();
                }
            }
        };
        let error = validate_module(&module).expect_err("three causes must be rejected");
        assert_codes(
            &error,
            &[
                Code::LoopNotSupported,
                Code::WhileNotSupported,
                Code::CpuOnlyName,
            ],
        );
    }

    #[test]
    fn an_accepted_module_reports_no_code() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod good {
                #[kernel]
                pub fn add(ctx: Invocation, mut out: StorageMut<f32>) {
                    let i = ctx.global_id().x;
                    if i < out.len() { out[i] = 1.0; }
                }
            }
        };
        assert!(codes_of_result(&validate_module(&module)).is_empty());
    }

    #[test]
    fn a_module_without_inline_content_and_stray_items_are_coded() {
        let outline: syn::ItemMod = syn::parse_quote! { mod outline; };
        assert_code(
            &validate_module(&outline).expect_err("outline module"),
            Code::InlineModuleRequired,
        );
        let stray: syn::ItemMod = syn::parse_quote! { mod bad { const LIMIT: u32 = 1; } };
        assert_code(
            &validate_module(&stray).expect_err("const item"),
            Code::UnsupportedModuleItem,
        );
    }

    #[test]
    fn signature_restrictions_are_coded() {
        for (module, code) in [
            (
                syn::parse_quote! { mod bad { async fn f() {} } },
                Code::AsyncFunction,
            ),
            (
                syn::parse_quote! { mod bad { unsafe fn f() {} } },
                Code::UnsafeFunction,
            ),
            (
                syn::parse_quote! { mod bad { fn f<T>(x: T) {} } },
                Code::ExternOrGenericFunction,
            ),
            (
                syn::parse_quote! { mod bad { extern "C" fn f() {} } },
                Code::ExternOrGenericFunction,
            ),
            (
                syn::parse_quote! { mod bad { struct Pair(uint, uint); } },
                Code::StructNeedsNamedFields,
            ),
        ] {
            let module: syn::ItemMod = module;
            assert_code(&validate_module(&module).expect_err("rejected"), code);
        }
    }

    #[test]
    fn a_derive_that_does_not_parse_is_wrapped_with_a_code() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[derive = "Debug"]
                struct Params { value: f32 }
            }
        };
        // `syn` words this failure itself; the code is added in front of it.
        let error = validate_module(&module).expect_err("malformed derive");
        assert_code(&error, Code::MalformedDerive);
    }

    #[test]
    fn accepts_vector_add_subset() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod good {
                #[kernel]
                pub fn add(ctx: Invocation, a: Storage<f32>, b: Storage<f32>, mut out: StorageMut<f32>) {
                    let i = ctx.global_id().x;
                    if i < out.len() {
                        out[i] = a[i] + b[i];
                    }
                }
            }
        };
        validate_module(&module).expect("vector add should validate");
    }

    #[test]
    fn rejects_empty_or_manually_represented_gpu_structs() {
        let empty: syn::ItemMod = syn::parse_quote! {
            mod bad {
                struct Empty {}
            }
        };
        assert_code(
            &validate_module(&empty).expect_err("empty GPU struct must fail"),
            Code::EmptyStruct,
        );

        let represented: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[repr(C)]
                struct Params { value: f32 }
            }
        };
        assert_code(
            &validate_module(&represented).expect_err("manual representation must fail"),
            Code::ManualRepr,
        );

        let copied: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[derive(Debug, Copy)]
                struct Params { value: f32 }
            }
        };
        assert_code(
            &validate_module(&copied).expect_err("manual Copy derive must fail"),
            Code::ManualCloneCopy,
        );
    }

    #[test]
    fn accepts_non_abi_derives_on_gpu_structs() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod good {
                #[derive(Debug, PartialEq)]
                struct Params { scale: f32, bias: f32 }
            }
        };
        validate_module(&module).expect("non-ABI derives should remain available");
    }
}

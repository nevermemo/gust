//! Module-level `const` items: their types, literals, operators, patterns and names.

use super::*;

/// The refusal of a module-level `const` that is outside the accepted form (slice m11.g3, D205).
pub(super) fn const_form(spanned: &impl quote::ToTokens, message: &str) -> syn::Error {
    diagnostics::error(Code::ConstForm, spanned, message)
}

/// Checks one module-level `const` (slice m11.g3, D205; slice m11.cexpr, D210): `const NAME: T = <expression>;`, `pub`
/// or private, with `T` one of `u32`, `i32`, `f32`, `bool` (or their dialect aliases `uint`, `int`, `float`) and the
/// initializer an expression of literals of that type and of constants of that type declared above (see
/// `check_const_expr`). Nothing is evaluated: the Slang text holds the expression the Rust item holds, operator for
/// operator. `struct_fields` is the field names of each struct of the module; `const_names` every module `const`
/// name; `declared` the constants checked so far, in source order, with their types (this item joins it when it
/// passes).
pub(super) fn validate_const(
    item: &syn::ItemConst,
    struct_fields: &HashMap<String, HashSet<String>>,
    const_names: &HashSet<String>,
    array_consts: &HashMap<String, u32>,
    declared: &mut HashMap<String, ConstScalar>,
) -> syn::Result<()> {
    // Documentation and lint levels only, as on an `impl`: `#[cfg(..)] const` would let the Slang and the twin disagree
    // about which constants exist.
    for attribute in &item.attrs {
        if !["doc", "allow", "warn", "deny", "forbid"]
            .iter()
            .any(|name| attribute.path().is_ident(name))
        {
            return Err(diagnostics::error(
                Code::UnsupportedAttribute,
                attribute,
                "only documentation and lint attributes (`doc`, `allow`, `warn`, `deny`, `forbid`) are accepted on a module `const`: any other has no proven Rust-to-Slang meaning and cannot be dropped",
            ));
        }
    }
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(const_form(
            &item.generics,
            "a module `const` is not generic: write `const NAME: u32 = 4096;`",
        ));
    }
    let name = item.ident.unraw().to_string();
    if name == "_" {
        return Err(const_form(
            &item.ident,
            "an unnamed `const _` has no name to read: give the constant a name",
        ));
    }
    if name.starts_with("__gust_") || name.starts_with("__gpu_") || name.starts_with("__Gust") {
        return Err(diagnostics::error(
            Code::ReservedPrefix,
            &item.ident,
            "the __gust_, __gpu_ and __Gust prefixes are reserved for generated shader symbols",
        ));
    }
    // Inside a Slang struct an unqualified name is a field before it is a global: a method reading the constant would
    // read the field of the same name instead.
    if struct_fields.values().any(|fields| fields.contains(&name)) {
        return Err(const_form(
            &item.ident,
            &format!(
                "`{name}` is also the name of a struct field of this module: in a method the Slang struct would read the field where Rust reads the constant; rename one of them"
            ),
        ));
    }
    let scope = ConstScope {
        names: const_names,
        arrays: array_consts,
        declared,
        own: &name,
    };
    // An array of the four scalars (slice m12.carr). It joins no scope: a constant expression never names it.
    if let syn::Type::Array(array) = item.ty.as_ref() {
        return const_arrays::validate(item, array, &scope);
    }
    let Some(scalar) = const_scalar(&item.ty) else {
        return Err(const_form(
            &item.ty,
            "a module `const` has one of the types `u32`, `i32`, `f32` or `bool`, or is an array `[T; N]` of one of them: a vector, a struct, a reference and `usize` have no constant lowering here; pass such a value as a uniform or a parameter",
        ));
    };
    check_const_expr(&item.expr, scalar, &scope, false)?;
    declared.insert(name, scalar);
    Ok(())
}

/// The type of a module `const` (slice m11.g3): one of the four scalars, by name or dialect alias.
pub(crate) fn const_scalar(ty: &syn::Type) -> Option<ConstScalar> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let segment = &path.path.segments[0];
    match (
        segment.arguments.is_empty(),
        segment.ident.unraw().to_string().as_str(),
    ) {
        (true, "u32" | "uint") => Some(ConstScalar::U32),
        (true, "i32" | "int") => Some(ConstScalar::I32),
        (true, "f32" | "float") => Some(ConstScalar::F32),
        (true, "bool") => Some(ConstScalar::Bool),
        _ => None,
    }
}

/// What a `const` initializer may name (slice m11.cexpr): every constant of the module, the ones declared above the
/// item (with their types), and the item's own name.
pub(super) struct ConstScope<'a> {
    pub(super) names: &'a HashSet<String>,
    /// The array constants of the module (slice m12.carr): a constant expression names none of them.
    pub(super) arrays: &'a HashMap<String, u32>,
    pub(super) declared: &'a HashMap<String, ConstScalar>,
    pub(super) own: &'a str,
}

/// The Rust spelling of a constant type, for messages.
pub(super) fn const_type_name(scalar: ConstScalar) -> &'static str {
    match scalar {
        ConstScalar::U32 => "u32",
        ConstScalar::I32 => "i32",
        ConstScalar::F32 => "f32",
        ConstScalar::Bool => "bool",
    }
}

/// "a" or "an" before the spelling of a constant type (`an i32`, `an f32`, `a u32`, `a bool`), for messages.
pub(super) fn const_article(scalar: ConstScalar) -> &'static str {
    match scalar {
        ConstScalar::I32 | ConstScalar::F32 => "an",
        ConstScalar::U32 | ConstScalar::Bool => "a",
    }
}

/// Whether the literal is a value of `scalar` (`negated`: a unary minus applies to it directly, as in `-2147483648`).
pub(super) fn const_literal_fits(scalar: ConstScalar, literal: &syn::Lit, negated: bool) -> bool {
    match (scalar, literal) {
        (ConstScalar::U32, syn::Lit::Int(value)) => {
            !negated
                && matches!(value.suffix(), "" | "u32")
                && value
                    .base10_parse::<u64>()
                    .is_ok_and(|v| v <= u64::from(u32::MAX))
        }
        (ConstScalar::I32, syn::Lit::Int(value)) => {
            matches!(value.suffix(), "" | "i32")
                && value
                    .base10_parse::<u64>()
                    .is_ok_and(|v| v <= i32::MAX as u64 + u64::from(negated))
        }
        (ConstScalar::F32, syn::Lit::Float(value)) => matches!(value.suffix(), "" | "f32"),
        (ConstScalar::F32, syn::Lit::Int(value)) => value.suffix() == "f32",
        (ConstScalar::Bool, syn::Lit::Bool(_)) => !negated,
        _ => false,
    }
}

/// The operators each constant type accepts, as the Slang operator that spells them (`None`: refused).
pub(crate) fn const_binary_operator(
    scalar: ConstScalar,
    operator: &syn::BinOp,
) -> Option<&'static str> {
    use syn::BinOp as B;
    let integer = matches!(scalar, ConstScalar::U32 | ConstScalar::I32);
    match operator {
        B::Add(_) if scalar != ConstScalar::Bool => Some("+"),
        B::Sub(_) if scalar != ConstScalar::Bool => Some("-"),
        B::Mul(_) if scalar != ConstScalar::Bool => Some("*"),
        B::Div(_) if scalar != ConstScalar::Bool => Some("/"),
        B::Rem(_) if integer => Some("%"),
        B::BitAnd(_) if integer => Some("&"),
        B::BitOr(_) if integer => Some("|"),
        B::BitXor(_) if integer => Some("^"),
        B::Shl(_) if integer => Some("<<"),
        B::Shr(_) if integer => Some(">>"),
        B::And(_) if scalar == ConstScalar::Bool => Some("&&"),
        B::Or(_) if scalar == ConstScalar::Bool => Some("||"),
        _ => None,
    }
}

/// The accepted initializer of a module `const` of type `scalar` (slice m11.cexpr, D210): literals of that type,
/// constants of that type declared above, parentheses, and the operators `const_binary_operator` lists plus unary `-`
/// (`i32`, `f32`) and `!` (`u32`, `i32`, `bool`); the shift amount of `<<` and `>>` is a `u32` literal or a `u32`
/// constant declared above. `negated`: a unary minus applies to this expression directly.
pub(super) fn check_const_expr(
    expr: &syn::Expr,
    scalar: ConstScalar,
    scope: &ConstScope,
    negated: bool,
) -> syn::Result<()> {
    let type_name = const_type_name(scalar);
    match expr {
        syn::Expr::Paren(inner) => check_const_expr(&inner.expr, scalar, scope, false),
        syn::Expr::Group(inner) => check_const_expr(&inner.expr, scalar, scope, negated),
        syn::Expr::Lit(literal) => {
            if const_literal_fits(scalar, &literal.lit, negated) {
                Ok(())
            } else {
                Err(const_form(
                    expr,
                    &format!(
                        "a literal in {} `{type_name}` module `const` is a literal of that type (`4096u32`, `-1i32`, `0.5f32`, `true`; `-` is for `i32` and `f32`; every operand of the expression has the type of the item): write the literal with the type of the item",
                        const_article(scalar)
                    ),
                ))
            }
        }
        syn::Expr::Path(path) => {
            let single = path
                .qself
                .is_none()
                .then(|| path.path.get_ident())
                .flatten();
            let Some(ident) = single else {
                return Err(const_form(
                    expr,
                    "a path such as `u32::MAX` or `std::f32::consts::PI` is not evaluated in a module `const`: write the literal, or name a `const` of this module declared above",
                ));
            };
            let name = ident.unraw().to_string();
            if name == scope.own {
                return Err(const_form(
                    expr,
                    &format!(
                        "`{name}` refers to itself, which is a forward reference: a module `const` reads only constants declared above it"
                    ),
                ));
            }
            if scope.arrays.contains_key(&name) {
                return Err(const_form(
                    expr,
                    &format!(
                        "`{name}` is an array constant: a module `const` expression names scalar constants only; write the literal"
                    ),
                ));
            }
            match scope.declared.get(&name) {
                Some(&declared) if declared == scalar => Ok(()),
                Some(&declared) => Err(const_form(
                    expr,
                    &format!(
                        "`{name}` is {} `{}` constant and this item is `{type_name}`: every operand of a module `const` expression has the type of the item, and there is no `as`; declare a constant of the right type",
                        const_article(declared),
                        const_type_name(declared)
                    ),
                )),
                None if scope.names.contains(&name) => Err(const_form(
                    expr,
                    &format!(
                        "`{name}` is declared below this item: a module `const` reads only constants declared above it; declare it above"
                    ),
                )),
                None => Err(const_form(
                    expr,
                    &format!(
                        "`{name}` is not a `const` of this module: a module `const` expression names only constants of this module declared above it; write the literal"
                    ),
                )),
            }
        }
        syn::Expr::Unary(unary) => match (&unary.op, scalar) {
            (syn::UnOp::Neg(_), ConstScalar::I32 | ConstScalar::F32) => {
                check_const_expr(&unary.expr, scalar, scope, true)
            }
            (syn::UnOp::Not(_), ConstScalar::U32 | ConstScalar::I32 | ConstScalar::Bool) => {
                check_const_expr(&unary.expr, scalar, scope, false)
            }
            _ => Err(const_form(
                expr,
                &format!(
                    "this unary operator is not accepted in a `{type_name}` module `const`: `-` is for `i32` and `f32` and `!` for `u32`, `i32` and `bool`"
                ),
            )),
        },
        syn::Expr::Binary(binary) => {
            use syn::BinOp as B;
            if matches!(
                binary.op,
                B::Eq(_) | B::Ne(_) | B::Lt(_) | B::Le(_) | B::Gt(_) | B::Ge(_)
            ) {
                return Err(const_form(
                    expr,
                    "a comparison is not accepted in a module `const`: write the `bool` literal, or compute the comparison where it is used",
                ));
            }
            if const_binary_operator(scalar, &binary.op).is_none() {
                return Err(const_form(
                    expr,
                    &format!(
                        "this operator is not accepted in a `{type_name}` module `const`: `u32` and `i32` take `+ - * / % & | ^ << >>`, `f32` takes `+ - * /`, `bool` takes `&&` and `||`"
                    ),
                ));
            }
            check_const_expr(&binary.left, scalar, scope, false)?;
            if matches!(binary.op, B::Shl(_) | B::Shr(_)) {
                check_const_shift_amount(&binary.right, scope)
            } else {
                check_const_expr(&binary.right, scalar, scope, false)
            }
        }
        syn::Expr::Cast(_) => Err(const_form(
            expr,
            "an `as` cast is not accepted in a module `const`: write the literal or constant with the type of the item (`4096u32`, `2.0f32`)",
        )),
        syn::Expr::Call(_) | syn::Expr::MethodCall(_) | syn::Expr::Macro(_) => Err(const_form(
            expr,
            "a call (a function, a method such as `x.pow(2)`, a macro) is not accepted in a module `const`: write the literal, or an expression of literals and constants declared above",
        )),
        syn::Expr::Index(_) => Err(const_form(
            expr,
            "an element of an array constant is not read in a module `const` initializer: write the literal",
        )),
        syn::Expr::Block(_)
        | syn::Expr::If(_)
        | syn::Expr::Match(_)
        | syn::Expr::Tuple(_)
        | syn::Expr::Array(_) => Err(const_form(
            expr,
            "a block, `if`, `match`, tuple or array is not accepted in a module `const`: write an expression of literals and constants declared above",
        )),
        _ => Err(const_form(
            expr,
            "a module `const` is initialized with an expression of literals of its type and constants of its type declared above, with parentheses and the operators `+ - * / % & | ^ << >> ! && ||` (`-` for `i32` and `f32`): write the literal",
        )),
    }
}

/// The shift amount of a constant `<<` or `>>`: a `u32` literal or a `u32` constant declared above.
pub(super) fn check_const_shift_amount(expr: &syn::Expr, scope: &ConstScope) -> syn::Result<()> {
    match expr {
        syn::Expr::Paren(inner) => check_const_shift_amount(&inner.expr, scope),
        syn::Expr::Group(inner) => check_const_shift_amount(&inner.expr, scope),
        syn::Expr::Lit(literal) if const_literal_fits(ConstScalar::U32, &literal.lit, false) => {
            Ok(())
        }
        syn::Expr::Path(_) => check_const_expr(expr, ConstScalar::U32, scope, false),
        _ => Err(const_form(
            expr,
            "the shift amount in a module `const` is a `u32` literal or a `u32` constant declared above",
        )),
    }
}

/// The scalar types a module `const` may have.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstScalar {
    U32,
    I32,
    F32,
    Bool,
}

/// A pattern that is the bare name of a module `const` (slice m11.g3): rustc compares the scrutinee with the constant,
/// where the Slang lowering of an identifier pattern binds a new name, so the two would differ silently. Every pattern
/// of every function, method and closure is looked at (a `let`, a parameter and a closure parameter of that name are
/// errors of rustc as well).
pub(super) fn check_const_patterns(items: &[syn::Item]) -> syn::Result<()> {
    let names: HashSet<String> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Const(item) => Some(item.ident.unraw().to_string()),
            _ => None,
        })
        .collect();
    if names.is_empty() {
        return Ok(());
    }
    struct Scan<'a> {
        names: &'a HashSet<String>,
        found: Option<syn::Error>,
    }
    impl<'ast> Visit<'ast> for Scan<'_> {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            let name = pattern.ident.unraw().to_string();
            if self.found.is_none() && self.names.contains(&name) {
                self.found = Some(const_form(
                    pattern,
                    &format!(
                        "`{name}` is a module `const`: as a pattern rustc compares with it, but the Slang lowering of a name pattern binds a new name; match a literal, or compare with `==` in an `if`"
                    ),
                ));
            }
            visit::visit_pat_ident(self, pattern);
        }
    }
    let mut scan = Scan {
        names: &names,
        found: None,
    };
    for item in items {
        if !matches!(item, syn::Item::Const(_)) {
            scan.visit_item(item);
        }
    }
    scan.found.map_or(Ok(()), Err)
}

/// A module `const` whose name is also a name of another namespace or a name the prelude gives (slice m11.g3, review): a
/// struct, enum, trait or type parameter of that name reaches slangc as a clash, a method of that name may be read in
/// place of the constant inside its Slang struct, and a prelude name (`None`, `Some`, `Vec2`, `uint`) means the prelude.
pub(super) fn check_const_names(items: &[syn::Item]) -> syn::Result<()> {
    let consts: Vec<&syn::ItemConst> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Const(item) => Some(item),
            _ => None,
        })
        .collect();
    if consts.is_empty() {
        return Ok(());
    }
    // The type-namespace and method names of the module, with what each is.
    struct Names(Vec<(String, &'static str)>);
    impl<'ast> Visit<'ast> for Names {
        fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
            self.0.push((item.ident.unraw().to_string(), "a struct"));
            visit::visit_item_struct(self, item);
        }
        fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
            self.0.push((item.ident.unraw().to_string(), "an enum"));
            visit::visit_item_enum(self, item);
        }
        fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
            self.0.push((item.ident.unraw().to_string(), "a trait"));
            visit::visit_item_trait(self, item);
        }
        fn visit_type_param(&mut self, item: &'ast syn::TypeParam) {
            self.0
                .push((item.ident.unraw().to_string(), "a type parameter"));
            visit::visit_type_param(self, item);
        }
        fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
            self.0
                .push((item.sig.ident.unraw().to_string(), "a method"));
            visit::visit_impl_item_fn(self, item);
        }
        fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
            self.0
                .push((item.sig.ident.unraw().to_string(), "a trait method"));
            visit::visit_trait_item_fn(self, item);
        }
    }
    let mut names = Names(Vec::new());
    for item in items {
        names.visit_item(item);
    }
    for item in consts {
        let name = item.ident.unraw().to_string();
        if matches!(
            name.as_str(),
            "None" | "Some" | "Ok" | "Err" | "u32" | "i32" | "f32"
        ) || math_types::is_reserved_type_name(&name)
        {
            return Err(const_form(
                &item.ident,
                &format!(
                    "`{name}` is a name of the Rust prelude or of GUST's math and scalar prelude: name the constant otherwise"
                ),
            ));
        }
        if let Some((_, what)) = names.0.iter().find(|(other, _)| *other == name) {
            return Err(const_form(
                &item.ident,
                &format!(
                    "`{name}` is also the name of {what} of this module: one name must not mean two things (in Slang the two would clash); rename one of them"
                ),
            ));
        }
    }
    Ok(())
}

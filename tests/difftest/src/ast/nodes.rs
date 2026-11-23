//! The nodes of the AST: operators, expressions, user items, and the places a call or a set names.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// Arithmetic negation, `i32` and `f32` only (Rust has no `-` on `u32`).
    Neg,
    /// `!`: bitwise on integers, logical on `bool`.
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    /// The right operand is always `u32`.
    Shl,
    /// The right operand is always `u32`; arithmetic on `i32`.
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Short-circuit, as in Rust.
    And,
    Or,
}

impl BinOp {
    pub const ALL: [BinOp; 18] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Le,
        BinOp::Gt,
        BinOp::Ge,
        BinOp::And,
        BinOp::Or,
    ];

    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }

    /// Rust's binary operator precedence (higher binds tighter); used by the re-parser of the emitted text.
    pub fn precedence(self) -> u8 {
        match self {
            BinOp::Mul | BinOp::Div | BinOp::Rem => 10,
            BinOp::Add | BinOp::Sub => 9,
            BinOp::Shl | BinOp::Shr => 8,
            BinOp::BitAnd => 7,
            BinOp::BitXor => 6,
            BinOp::BitOr => 5,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 4,
            BinOp::And => 3,
            BinOp::Or => 2,
        }
    }

    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    pub fn is_shift(self) -> bool {
        matches!(self, BinOp::Shl | BinOp::Shr)
    }

    pub fn is_logical(self) -> bool {
        matches!(self, BinOp::And | BinOp::Or)
    }

    /// The type of `l op r` when the left operand has type `operand`.
    pub fn result_ty(self, operand: Ty) -> Ty {
        if self.is_comparison() || self.is_logical() {
            Ty::Bool
        } else {
            operand
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// A literal: the 32-bit pattern. `i32::MIN` has no literal form (see `Expr::i32_min`). An `f32` literal is a
    /// finite value that is not a negative zero.
    Lit(Ty, u32),
    /// A local variable (a `let` binding or a loop variable).
    Var(u32, Ty),
    /// The invocation index `i` (`id.x`), `u32`.
    Gid,
    /// `<input buffer>.len()`, `u32`.
    Len(usize),
    /// `<input buffer>[index]`; `index` is `u32`.
    Input {
        buf: usize,
        ty: Ty,
        index: Box<Expr>,
    },
    /// `ty` is the operand and the result type.
    Un { op: UnOp, ty: Ty, e: Box<Expr> },
    /// `ty` is the type of the left operand; the result type follows `BinOp::result_ty`.
    Bin {
        op: BinOp,
        ty: Ty,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    /// `e as to` among `u32`, `i32` and `f32`. Between `u32` and `i32` the bits are unchanged; to and from `f32` the
    /// value is converted (the interpreter's rules for the cases it accepts are in `interp.rs`).
    Cast { to: Ty, e: Box<Expr> },
    /// `Vec4::new(a, b, c, d)` (`ty` a vector, `args` its scalars in order), or for a matrix `Mat4::from_cols(c0, ..)`
    /// (`rows` false: the columns, each a `Vec` of the matrix size) or `Mat4::from_rows(r0, ..)` (`rows` true).
    Make { ty: Ty, rows: bool, args: Vec<Expr> },
    /// `Vec4::splat(e)`, `e` a scalar of the lane type.
    Splat { ty: Ty, e: Box<Expr> },
    /// `e.x`, `e.y`, ...: one component of a vector (a matrix has no component access).
    Comp { e: Box<Expr>, index: u8 },
    /// `e.xy()`, `e.wzyx()`: the named components as a vector of the word's length (2 or 4 letters).
    Swizzle { e: Box<Expr>, lanes: Vec<u8> },
    /// `l.dot(r)`: two vectors of one type, a scalar of the lane type.
    Dot { l: Box<Expr>, r: Box<Expr> },
    /// `e.transpose()`.
    Transpose { e: Box<Expr> },
    /// `l op r` where at least one operand is a vector or a matrix; `ty` is the result type (`agg_bin_ty`).
    AggBin {
        op: BinOp,
        ty: Ty,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    /// `-e` on a float vector, a signed integer vector or a matrix.
    AggNeg { e: Box<Expr> },
    /// A read of a field place (`v3.f1`, `self.f0`, `v3.f2.f0`) or, with an empty path from `self`, the receiver by
    /// value. `ty` is the type of the place. (Format 8.)
    Read { place: Place, ty: Ty },
    /// A call of `Methods::fns[f]`: `recv.m(args)` (a method), `S::m(args)` (an associated function) or `m(args)` (a
    /// helper). `ty` is the return type, which a value call always has. (Format 8.)
    Call {
        f: usize,
        /// The struct of an associated function (`S0::m3(..)`), copied from the table so the text needs no table.
        owner: Option<u8>,
        recv: Option<Place>,
        args: Vec<Expr>,
        ty: Ty,
    },
    /// `S0 { f1: b, f0: a }`: every field once, in written order, evaluated in that order. `order[w]` is the
    /// declaration index of the `w`th written field (a permutation; the identity is declaration order). (Format 8.)
    StructLit {
        ty: Ty,
        fields: Vec<Expr>,
        order: Vec<u8>,
    },
    /// `E0::V1`: variant `v` of enum `e`. (Format 9.)
    Variant { e: u8, v: u8 },
    /// `x as u32`: an enum value as its repr type (`to` is `u32` or `i32`, the enum's repr; Rust allows no other
    /// target here, E0122). (Format 9.)
    EnumAs { to: Ty, e: Box<Expr> },
    /// `(a, b, c)`: a tuple literal of type `ty`; the elements are evaluated left to right, each completely before the
    /// next. (Format 10.)
    Tuple { ty: Ty, elems: Vec<Expr> },
    /// `f().1`: one element of a tuple that is no place (a call result). A `t.1` on a local is a `Read`. (Format 10.)
    TupleField { e: Box<Expr>, index: u8 },
    /// `[a, b, c]`: an array literal of type `ty`; the elements are evaluated left to right. (Format 11.)
    ArrayLit { ty: Ty, elems: Vec<Expr> },
    /// `[e; N]`: `e` is evaluated once and copied `N` times. (Format 11.)
    Repeat { ty: Ty, e: Box<Expr> },
    /// `a.len()` of an array place, a `u32`; the dialect accepts it only inside an index or under `as u32`. (Format 11.)
    ArrLen { place: Place },
    /// `cl3(a, b)`: a call of the closure bound by `Stmt::LetClosure` with `var` 3. The arguments run left to right,
    /// each completely, then the body runs once; `ty` is the closure's result type. (Format 12.)
    ClosureCall { var: u32, args: Vec<Expr>, ty: Ty },
    /// `Tr0::t0m1(&s, a)`, `<S1 as Tr0>::t0m1(&s, a)` or `Tr0::t0m1(s, a)`: a call of a trait-impl method written as a path
    /// call (format 13). `call` is the `Expr::Call` of the method, whose receiver is the first argument; the semantics are
    /// those of `s.t0m1(a)`.
    PathCall { form: PathForm, call: Box<Expr> },
    /// `a.rem_euclid(b)`, `a.mul_add(b, c)`, `x.floor()`, ...: a method of the scalar math set (format 14, slice
    /// m11.dt2). `args[0]` is the receiver, the rest are the arguments in written order; `ty` is the result type.
    Math { f: MathFn, ty: Ty, args: Vec<Expr> },
    /// `K0`: the module `const` item `consts[k]` of the program, a `u32` or `i32` (format 14, slice m11.dt2).
    Const { k: u8, ty: Ty },
}

/// The scalar math methods that format 14 generates (slice m11.dt2). The receiver is the first argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MathFn {
    /// `a.rem_euclid(b)` on `u32` and `i32`.
    RemEuclid,
    /// `a.mul_add(b, c)` on `f32`.
    MulAdd,
    /// `x.abs()`, `x.floor()`, `x.ceil()`, `x.trunc()` on `f32`.
    Abs,
    Floor,
    Ceil,
    Trunc,
    /// `a.min(b)`, `a.max(b)` on `f32`.
    Min,
    Max,
    /// `x.clamp(lo, hi)` on `f32`.
    Clamp,
}

impl MathFn {
    pub const ALL: [MathFn; 9] = [
        MathFn::RemEuclid,
        MathFn::MulAdd,
        MathFn::Abs,
        MathFn::Floor,
        MathFn::Ceil,
        MathFn::Trunc,
        MathFn::Min,
        MathFn::Max,
        MathFn::Clamp,
    ];

    /// The method name as written.
    pub fn name(self) -> &'static str {
        match self {
            MathFn::RemEuclid => "rem_euclid",
            MathFn::MulAdd => "mul_add",
            MathFn::Abs => "abs",
            MathFn::Floor => "floor",
            MathFn::Ceil => "ceil",
            MathFn::Trunc => "trunc",
            MathFn::Min => "min",
            MathFn::Max => "max",
            MathFn::Clamp => "clamp",
        }
    }

    /// The number of operands, the receiver included.
    pub fn arity(self) -> usize {
        match self {
            MathFn::Abs | MathFn::Floor | MathFn::Ceil | MathFn::Trunc => 1,
            MathFn::RemEuclid | MathFn::Min | MathFn::Max => 2,
            MathFn::MulAdd | MathFn::Clamp => 3,
        }
    }

    /// Whether the method is on `f32` (`rem_euclid` is on the integers).
    pub fn is_float(self) -> bool {
        self != MathFn::RemEuclid
    }

    /// The counted name of a use, the same string in `operator_counts` and in the interpreter's coverage.
    pub fn counted_name(self, ty: Ty) -> String {
        format!(".{} {}", self.name(), ty.name())
    }
}

/// How a path call names the trait (format 13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathForm {
    /// `Tr0::t0m1(&s, a)`.
    Trait,
    /// `<S1 as Tr0>::t0m1(&s, a)`.
    Qualified,
}

/// A non-capturing closure (format 12): `let cl3 = |v4: u32, mut v5: [u32; 3]| -> u32 { body; tail };`. Every
/// parameter is typed, the result type is written and the body is a block; the binding is immutable. `params` holds
/// the variable, its type and whether it is `mut`. The body sees only its parameters (no capture, E0329), calls
/// helpers and methods but no closure (E0332), and holds no `return`, buffer, `i` or `self`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureDef {
    pub var: u32,
    pub params: Vec<(u32, Ty, bool)>,
    pub ret: Ty,
    pub body: Vec<Stmt>,
    pub tail: Expr,
}

/// A field-less enum of the module (format 9): its repr and its variants `V0`, `V1`, ... A variant's value is a 32-bit
/// pattern (an `i32` repr stores negative values in two's complement); `explicit` says the text writes `= value`, the
/// others count up from the variant before (the first one from 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumDef {
    pub repr: Ty,
    pub variants: Vec<(u32, bool)>,
}

impl EnumDef {
    /// The value of variant `v` as a number of the repr type.
    pub fn value(&self, v: u8) -> i64 {
        let bits = self.variants[usize::from(v)].0;
        if self.repr == Ty::I32 {
            i64::from(bits as i32)
        } else {
            i64::from(bits)
        }
    }
}

/// A pattern of a `match` arm: one or more variants (`E0::V1 | E0::V3`), one or more integer literals (`0 | 3u32`;
/// format 10: the 32-bit pattern and whether the suffix is written), or `_`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pat {
    Variants(Vec<u8>),
    Lits(Vec<(u32, bool)>),
    Wild,
}

/// One arm of a `match`: the pattern and the block it runs. In the value form (`Stmt::LetMatch`, `Stmt::TailMatch`)
/// the block ends in the arm's value: a `Stmt::Yield`, a `Stmt::Return` or a `Stmt::TailMatch`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arm {
    pub pat: Pat,
    pub body: Vec<Stmt>,
}

/// A module struct: its fields in order, each a `u32`, an `i32` (not `bool`: a GPU struct is a `GpuPod` type) or an
/// earlier struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDef {
    pub fields: Vec<Ty>,
}

/// How a function receives its receiver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recv {
    /// A helper or an associated function (no receiver).
    None,
    /// `&self`.
    Shared,
    /// `&mut self` (`[mutating]` in Slang).
    Mut,
    /// `self` by value (never `mut self`: E0116).
    Value,
}

/// What a trait method returns (format 13, slice m9.d2): nothing, a scalar, or `Self` (the receiver's type).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TRet {
    Unit,
    Ty(Ty),
    SelfTy,
}

/// One method of a module trait: a receiver (`&self`, `&mut self` or `self`), scalar parameters, no body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraitMethod {
    pub recv: Recv,
    pub params: Vec<Ty>,
    pub ret: TRet,
}

/// A module trait `Tr<n>` (format 13): methods `t<n>m<k>`, each implemented by exactly two types of the module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraitDef {
    pub methods: Vec<TraitMethod>,
}

/// A function that is the impl of trait method `m` of trait `tr` for the struct or enum `ty` (format 13). A struct impl
/// keeps `owner = Some(j)`, so every rule about the methods of a struct holds for it; an enum impl has no owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Imp {
    pub tr: u8,
    pub m: u8,
    pub ty: Ty,
}

/// One type parameter of a generic function: its trait bounds, whether `Copy` is among them, and whether the bounds are
/// written in a `where` clause (format 13).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenParam {
    pub bounds: Vec<u8>,
    pub copy: bool,
    pub where_clause: bool,
}

/// A helper, a method or an associated function. Parameters are immutable value parameters (a scalar or a struct). The
/// body is a block of the kernel's statements; `tail` is the value (absent for a unit `&mut self` method).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnDef {
    /// The struct whose `impl` holds the function (a method or an associated function); `None` for a helper.
    pub owner: Option<u8>,
    pub recv: Recv,
    pub params: Vec<(u32, Ty)>,
    pub ret: Option<Ty>,
    pub body: Vec<Stmt>,
    pub tail: Option<Expr>,
    /// The trait method this function implements (format 13); `None` for every other function.
    pub imp: Option<Imp>,
    /// The type parameters of a generic helper or method (format 13); empty for every other function.
    pub gens: Vec<GenParam>,
}

/// The struct and function tables of a method program (format 8). Function `k` calls only functions below `k`, so the
/// call graph is acyclic by construction; every name is unique (`S<k>`, `f<k>` fields, `m<k>` functions).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Methods {
    pub structs: Vec<StructDef>,
    pub fns: Vec<FnDef>,
    /// The enums of an enum program (format 9); empty for every other space.
    pub enums: Vec<EnumDef>,
    /// The traits of a trait program (format 13); empty for every other space.
    pub traits: Vec<TraitDef>,
    /// The module `const` items of a math program (format 14); empty for every other space. Item `k` is `K<k>`, and its
    /// initializer holds only literals and the items below it.
    pub consts: Vec<ConstDef>,
}

/// A module `const K<k>: ty = init;` (format 14, slice m11.dt2, D210): `ty` is `u32` or `i32`, and `init` is an
/// expression of literals, earlier constants, the unary `-` and `!` and the binary integer operators (no cast, no
/// call).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstDef {
    pub ty: Ty,
    pub init: Expr,
}

impl Methods {
    pub fn is_empty(&self) -> bool {
        self.structs.is_empty()
            && self.fns.is_empty()
            && self.enums.is_empty()
            && self.consts.is_empty()
    }

    /// The function that implements trait method `m` of trait `tr` for `ty`, if there is one.
    pub fn impl_fn(&self, tr: u8, m: u8, ty: Ty) -> Option<usize> {
        self.fns
            .iter()
            .position(|def| def.imp == Some(Imp { tr, m, ty }))
    }

    /// The types that implement trait `tr` (the types of the impls of its first method), in table order.
    pub fn implementors(&self, tr: u8) -> Vec<Ty> {
        self.fns
            .iter()
            .filter_map(|def| {
                def.imp
                    .filter(|imp| imp.tr == tr && imp.m == 0)
                    .map(|imp| imp.ty)
            })
            .collect()
    }

    /// Whether `ty` implements trait `tr`.
    pub fn implements(&self, ty: Ty, tr: u8) -> bool {
        self.implementors(tr).contains(&ty)
    }

    /// Whether any function of the program is a trait impl or generic (a trait program, format 13).
    pub fn has_traits(&self) -> bool {
        !self.traits.is_empty()
    }

    /// The number of scalar components of a value of struct `j`, nested structs flattened.
    pub fn flat_len(&self, j: u8) -> usize {
        self.structs[usize::from(j)]
            .fields
            .iter()
            .map(|ty| self.ty_len(*ty))
            .sum()
    }

    /// The flat length of a scalar (1) or of a struct.
    pub fn ty_len(&self, ty: Ty) -> usize {
        if let Ty::Array(a) = ty {
            return usize::from(a.n) * self.ty_len(a.e.ty());
        }
        match ty {
            Ty::Struct(j) => self.flat_len(j),
            Ty::Tuple(t) => t.elems().iter().map(|e| self.ty_len(e.ty())).sum(),
            Ty::Vec(..) | Ty::Mat(_) => ty.lanes(),
            _ => 1,
        }
    }

    /// The offset of element `k` of tuple type `t` in the flat representation.
    pub fn tuple_offset(&self, t: TupleTy, k: u8) -> usize {
        t.elems()[..usize::from(k)]
            .iter()
            .map(|e| self.ty_len(e.ty()))
            .sum()
    }

    /// The offset of field `k` of struct `j` in the flat representation.
    pub fn field_offset(&self, j: u8, k: u8) -> usize {
        self.structs[usize::from(j)].fields[..usize::from(k)]
            .iter()
            .map(|ty| self.ty_len(*ty))
            .sum()
    }

    /// The type reached from a value of type `ty` by following `path`, and its offset in the flat representation.
    pub fn walk(&self, mut ty: Ty, path: &[u8]) -> (Ty, usize) {
        let mut offset = 0;
        for &k in path {
            match ty {
                Ty::Struct(j) => {
                    offset += self.field_offset(j, k);
                    ty = self.structs[usize::from(j)].fields[usize::from(k)];
                }
                Ty::Tuple(t) => {
                    offset += self.tuple_offset(t, k);
                    ty = t.elems()[usize::from(k)].ty();
                }
                other => panic!("a field of {}", other.name()),
            }
        }
        (ty, offset)
    }
}

/// The root of a place: a local variable, or the receiver `self`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Root {
    Var(u32),
    SelfV,
}

/// A local (or `self`) followed by an optional index and a path of field indices: `v3`, `v3.f1`, `self.f2.f0`,
/// `v3[k]`, `v3[k].f1` (format 11: the index applies to an array root, the path follows it). A place with an index
/// holds an expression, which may have side effects: the interpreter locates it once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub root: Root,
    /// The type of the root (a struct, an array, or a scalar local with an empty path), so that a place is self-describing.
    pub root_ty: Ty,
    pub path: Vec<u8>,
    /// `root[index as usize]` when the root is an array: `index` is the `u32` (or `i32`) expression before the cast.
    pub index: Option<Box<Expr>>,
}

impl Place {
    pub fn var(var: u32, ty: Ty) -> Place {
        Place {
            root: Root::Var(var),
            root_ty: ty,
            path: Vec::new(),
            index: None,
        }
    }

    pub fn this(ty: Ty) -> Place {
        Place {
            root: Root::SelfV,
            root_ty: ty,
            path: Vec::new(),
            index: None,
        }
    }

    pub fn field(&self, k: u8) -> Place {
        let mut path = self.path.clone();
        path.push(k);
        Place {
            root: self.root,
            root_ty: self.root_ty,
            path,
            index: self.index.clone(),
        }
    }
}

//! The AST of the generated kernels: integers, booleans, `f32` (from DIFFTEST_FORMAT 2) and, for the vector programs
//! (DIFFTEST_FORMAT 6, slice m6.d), the `gust::math` vectors and matrices.
//!
//! Every scalar value is a 32-bit pattern (`u32`); `bool` is 0 or 1. Nodes that the interpreter and the emitter need to
//! type carry their operand type, so `Expr::ty` is O(1). `Program::check` verifies the invariants the generator
//! promises, and the unit tests run it over every generated program.
//!
//! The vector types never reach a buffer: inputs and the output are scalar, so the result file, the classifier and the
//! shrinker see scalars only. A vector exists as a local or an intermediate, is built from scalars (`Vec4::new`,
//! `splat`, `Mat4::from_cols`) and is read back through a component, a `dot`, or a swizzle that is read in turn. The
//! aggregate operators have their own nodes (`AggBin`, `AggNeg`, `Dot`, ...), so every pattern that matches `Bin` or
//! `Un` still sees scalars only.

use std::collections::{BTreeMap, BTreeSet};

/// The element type of a vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lane {
    U32,
    I32,
    F32,
}

impl Lane {
    /// The scalar type of the elements.
    pub fn ty(self) -> Ty {
        match self {
            Lane::U32 => Ty::U32,
            Lane::I32 => Ty::I32,
            Lane::F32 => Ty::F32,
        }
    }
}

/// The widths the program space uses: `gust::math` has 2- and 4-wide vectors and matrices with a buffer layout; the
/// 3-wide types are locals only and are not generated.
pub const WIDTHS: [u8; 2] = [2, 4];

/// The types of the program space. There is no `bool` buffer (`bool` has no `GpuPod`), and no vector or matrix buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Ty {
    U32,
    I32,
    Bool,
    /// `f32` (R1-4 of the M5 plan: the exact-interval subset). A value is the 32-bit IEEE pattern.
    F32,
    /// `Vec2`/`Vec4`, `IVec2`/`IVec4`, `UVec2`/`UVec4`: the element type and the width (2 or 4).
    Vec(Lane, u8),
    /// `Mat2`/`Mat4`: square, `f32`, column-major.
    Mat(u8),
    /// A struct of the module (format 8, slice m9.d1a): `S0`, `S1`, `S2`, an index into `Methods::structs`. Never a
    /// buffer element, a vector lane or a result type.
    Struct(u8),
    /// A field-less enum of the module (format 9, slice m9.d1b): `E0`, `E1`, an index into `Methods::enums`. A value
    /// lives in a local, a parameter or a result of a function; it is never a field, a buffer element, a vector lane
    /// or a result type, and it comes only from a variant path, a variable or a call.
    Enum(u8),
    /// A tuple of two to four elements (format 10, slice m9.d1c). It lives in a local, a parameter or a result; it is
    /// never a field, a buffer element or a result type of the kernel. The type describes itself (no table), so `Ty`
    /// stays `Copy` and the emitter needs no context to spell it.
    Tuple(TupleTy),
}

/// An element type of a tuple: a scalar, a module struct or a module enum (format 10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Elem {
    U32,
    I32,
    Bool,
    Struct(u8),
    Enum(u8),
}

impl Elem {
    pub fn ty(self) -> Ty {
        match self {
            Elem::U32 => Ty::U32,
            Elem::I32 => Ty::I32,
            Elem::Bool => Ty::Bool,
            Elem::Struct(j) => Ty::Struct(j),
            Elem::Enum(e) => Ty::Enum(e),
        }
    }

    /// The element kind of a type, if a tuple may hold it.
    pub fn of(ty: Ty) -> Option<Elem> {
        Some(match ty {
            Ty::U32 => Elem::U32,
            Ty::I32 => Elem::I32,
            Ty::Bool => Elem::Bool,
            Ty::Struct(j) => Elem::Struct(j),
            Ty::Enum(e) => Elem::Enum(e),
            _ => return None,
        })
    }
}

/// The most elements of a generated tuple, and the most scalar components its flat value holds (an `Agg`).
pub const MAX_TUPLE: usize = 4;
pub const MAX_TUPLE_FLAT: usize = 16;

/// A tuple type: `n` elements in `e[..n]` (the rest is padding, `Elem::U32`, so derived equality is type equality).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TupleTy {
    pub n: u8,
    pub e: [Elem; MAX_TUPLE],
}

impl TupleTy {
    pub fn new(elems: &[Elem]) -> TupleTy {
        assert!(
            (2..=MAX_TUPLE).contains(&elems.len()),
            "{} elements",
            elems.len()
        );
        let mut e = [Elem::U32; MAX_TUPLE];
        e[..elems.len()].copy_from_slice(elems);
        TupleTy {
            n: elems.len() as u8,
            e,
        }
    }

    pub fn elems(&self) -> &[Elem] {
        &self.e[..usize::from(self.n)]
    }

    /// The Rust spelling: `(u32, S0, E1)`.
    pub fn text(&self) -> String {
        let parts: Vec<String> = self.elems().iter().map(|e| e.ty().text()).collect();
        format!("({})", parts.join(", "))
    }
}

impl Ty {
    /// The Rust spelling of any type, a tuple included (`name` is a `&'static str` and cannot spell one).
    pub fn text(self) -> String {
        match self {
            Ty::Tuple(t) => t.text(),
            other => other.name().to_owned(),
        }
    }

    /// A tuple.
    pub fn is_tuple(self) -> bool {
        matches!(self, Ty::Tuple(_))
    }

    /// The Rust spelling of a scalar, vector, matrix, struct or enum; a tuple has `text`.
    pub fn name(self) -> &'static str {
        match self {
            Ty::U32 => "u32",
            Ty::I32 => "i32",
            Ty::Bool => "bool",
            Ty::F32 => "f32",
            Ty::Vec(Lane::F32, 2) => "Vec2",
            Ty::Vec(Lane::F32, 4) => "Vec4",
            Ty::Vec(Lane::I32, 2) => "IVec2",
            Ty::Vec(Lane::I32, 4) => "IVec4",
            Ty::Vec(Lane::U32, 2) => "UVec2",
            Ty::Vec(Lane::U32, 4) => "UVec4",
            Ty::Mat(2) => "Mat2",
            Ty::Mat(4) => "Mat4",
            Ty::Struct(0) => "S0",
            Ty::Struct(1) => "S1",
            Ty::Struct(2) => "S2",
            Ty::Enum(0) => "E0",
            Ty::Enum(1) => "E1",
            Ty::Tuple(_) => "(tuple)",
            Ty::Vec(..) | Ty::Mat(_) | Ty::Struct(_) | Ty::Enum(_) => {
                panic!("a width, struct or enum outside the program space: {self:?}")
            }
        }
    }

    /// The dialect spelling used for buffer element types (scalars only: no vector or matrix buffer is generated).
    pub fn dialect_name(self) -> &'static str {
        match self {
            Ty::U32 => "uint",
            Ty::I32 => "int",
            Ty::Bool => "bool",
            Ty::F32 => "float",
            Ty::Vec(..) | Ty::Mat(_) | Ty::Struct(_) | Ty::Enum(_) | Ty::Tuple(_) => {
                panic!("{} is not a buffer element type", self.name())
            }
        }
    }

    /// The spelling of a struct field, a parameter or a return type: the dialect name of a scalar, `S<k>` of a struct,
    /// `E<k>` of an enum. (A tuple is spelled by `text`.)
    pub fn decl_name(self) -> &'static str {
        match self {
            Ty::Struct(_) | Ty::Enum(_) => self.name(),
            other => other.dialect_name(),
        }
    }

    /// `decl_name` for any type, a tuple included.
    pub fn decl_text(self) -> String {
        match self {
            Ty::Tuple(t) => t.text(),
            other => other.decl_name().to_owned(),
        }
    }

    /// A struct of the module.
    pub fn is_struct(self) -> bool {
        matches!(self, Ty::Struct(_))
    }

    /// An enum of the module.
    pub fn is_enum(self) -> bool {
        matches!(self, Ty::Enum(_))
    }

    pub fn is_int(self) -> bool {
        matches!(self, Ty::U32 | Ty::I32)
    }

    /// A vector or a matrix.
    pub fn is_aggregate(self) -> bool {
        matches!(self, Ty::Vec(..) | Ty::Mat(_))
    }

    /// The type of one component: the type itself for a scalar, the lane type for a vector, `f32` for a matrix.
    pub fn scalar(self) -> Ty {
        match self {
            Ty::Vec(lane, _) => lane.ty(),
            Ty::Mat(_) => Ty::F32,
            other => other,
        }
    }

    /// The number of components: 1 for a scalar, `n` for a vector, `n * n` for a matrix.
    pub fn lanes(self) -> usize {
        match self {
            Ty::Vec(_, n) => usize::from(n),
            Ty::Mat(n) => usize::from(n) * usize::from(n),
            _ => 1,
        }
    }
}

/// The result type of the aggregate operator `l op r`, if `gust::math` has the operator for those operand types (the
/// product table, the `Div` impls and the matrix `Add`/`Sub` of `crates/gust/src/math.rs`).
pub fn agg_bin_ty(op: BinOp, l: Ty, r: Ty) -> Option<Ty> {
    let arithmetic = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div);
    match (l, r) {
        (Ty::Vec(a, n), Ty::Vec(b, m)) if a == b && n == m && arithmetic => Some(l),
        (Ty::Vec(a, _), s) if s == a.ty() && matches!(op, BinOp::Mul | BinOp::Div) => Some(l),
        (s, Ty::Vec(a, _)) if s == a.ty() && op == BinOp::Mul => Some(r),
        (Ty::Mat(n), Ty::Mat(m))
            if n == m && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) =>
        {
            Some(l)
        }
        (Ty::Mat(_), Ty::F32) if op == BinOp::Mul => Some(l),
        (Ty::F32, Ty::Mat(_)) if op == BinOp::Mul => Some(r),
        (Ty::Mat(n), Ty::Vec(Lane::F32, m)) if n == m && op == BinOp::Mul => Some(r),
        _ => None,
    }
}

/// The letters of the component fields and swizzle methods, by component index.
pub const COMPONENT_LETTERS: [char; 4] = ['x', 'y', 'z', 'w'];

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
}

/// The struct and function tables of a method program (format 8). Function `k` calls only functions below `k`, so the
/// call graph is acyclic by construction; every name is unique (`S<k>`, `f<k>` fields, `m<k>` functions).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Methods {
    pub structs: Vec<StructDef>,
    pub fns: Vec<FnDef>,
    /// The enums of an enum program (format 9); empty for every other space.
    pub enums: Vec<EnumDef>,
}

impl Methods {
    pub fn is_empty(&self) -> bool {
        self.structs.is_empty() && self.fns.is_empty() && self.enums.is_empty()
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
        match ty {
            Ty::Struct(j) => self.flat_len(j),
            Ty::Tuple(t) => t.elems().iter().map(|e| self.ty_len(e.ty())).sum(),
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

/// A local (or `self`) followed by a path of field indices: `v3`, `v3.f1`, `self.f2.f0`. A place has no side effects
/// and no index, so its evaluation order is not observable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub root: Root,
    /// The type of the root (a struct, or a scalar local with an empty path), so that a place is self-describing.
    pub root_ty: Ty,
    pub path: Vec<u8>,
}

impl Place {
    pub fn var(var: u32, ty: Ty) -> Place {
        Place {
            root: Root::Var(var),
            root_ty: ty,
            path: Vec::new(),
        }
    }

    pub fn this(ty: Ty) -> Place {
        Place {
            root: Root::SelfV,
            root_ty: ty,
            path: Vec::new(),
        }
    }

    pub fn field(&self, k: u8) -> Place {
        let mut path = self.path.clone();
        path.push(k);
        Place {
            root: self.root,
            root_ty: self.root_ty,
            path,
        }
    }
}

/// The counted name of a call (`call mut`, ...), the same string in `operator_counts` and in the interpreter's coverage.
pub fn call_name(recv: Recv, owner: Option<u8>) -> &'static str {
    match (recv, owner) {
        (Recv::Mut, _) => "call &mut self",
        (Recv::Shared, _) => "call &self",
        (Recv::Value, _) => "call self",
        (Recv::None, Some(_)) => "call assoc",
        (Recv::None, None) => "call helper",
    }
}

/// The counted name of a `Stmt::Set`: `set =`, `set +=`, ...
pub fn set_name(op: Option<BinOp>) -> String {
    match op {
        None => "set =".to_owned(),
        Some(op) => format!("set {}=", op.symbol()),
    }
}

/// Names of the order-sensitive shapes (round r1 of m9.d1a). The generator biases toward them, `operator_counts` counts
/// them statically and the interpreter counts them when it runs the node, under the same names.
pub const SHAPE_STMT_CALL: &str = "shape stmt &mut self call";
pub const SHAPE_EXPR_CALL: &str = "shape expr &mut self call";
pub const SHAPE_LOGIC_CALL: &str = "shape &mut self call right of &&/||";
pub const SHAPE_READ_AND_CALL: &str = "shape read and &mut self call on one root";
pub const SHAPE_VALUE_CALL_INNER: &str =
    "shape by-value call with &mut self call on its root in its arguments";

/// Whether `e` is a `&mut self` call, and on which root. An element of such a call's tuple result (`c.pair().1`) is one
/// too: the call runs before the element is read.
fn mut_call_root(methods: &Methods, e: &Expr) -> Option<Root> {
    match e {
        Expr::Call {
            f,
            recv: Some(place),
            ..
        } if methods.fns[*f].recv == Recv::Mut => Some(place.root),
        Expr::TupleField { e, .. } => mut_call_root(methods, e),
        _ => None,
    }
}

/// Whether `e` or a node below it is a `&mut self` call on `root`.
fn contains_mut_call_on(methods: &Methods, e: &Expr, root: Root) -> bool {
    let mut found = false;
    walk_expr(e, &mut |n| found |= mut_call_root(methods, n) == Some(root));
    found
}

/// `shape same-root set <op>=`: an assignment to a place whose value contains a `&mut self` call on the place's root.
pub fn set_shape(
    methods: &Methods,
    place: &Place,
    op: Option<BinOp>,
    value: &Expr,
) -> Option<String> {
    contains_mut_call_on(methods, value, place.root)
        .then(|| format!("shape same-root {}", set_name(op)))
}

/// The shape of a binary: a `&mut self` call as the right operand of `&&`/`||`, or a read of a place and a `&mut self`
/// call on its root as the two operands.
pub fn bin_shape(methods: &Methods, op: BinOp, l: &Expr, r: &Expr) -> Option<&'static str> {
    if matches!(op, BinOp::And | BinOp::Or) && mut_call_root(methods, r).is_some() {
        return Some(SHAPE_LOGIC_CALL);
    }
    // `==` or `!=` of two enum values that are calls mutating one root: the left call must run first.
    if matches!(op, BinOp::Eq | BinOp::Ne) && l.ty().is_enum() {
        let (mut left, mut right) = (Vec::new(), Vec::new());
        mutating_roots(methods, l, &mut left);
        mutating_roots(methods, r, &mut right);
        return left
            .iter()
            .any(|root| right.contains(root))
            .then_some(SHAPE_ENUM_EQ);
    }
    let read_root = |e: &Expr| match e {
        Expr::Read { place, .. } => Some(place.root),
        _ => None,
    };
    let same = |a: Option<Root>, b: Option<Root>| a.is_some() && a == b;
    (same(read_root(l), mut_call_root(methods, r)) || same(mut_call_root(methods, l), read_root(r)))
        .then_some(SHAPE_READ_AND_CALL)
}

/// The shape of a call: a by-value call whose arguments contain a `&mut self` call on the receiver's root.
pub fn call_shape(
    methods: &Methods,
    f: usize,
    recv: &Option<Place>,
    args: &[Expr],
) -> Option<&'static str> {
    let place = recv.as_ref()?;
    (methods.fns[f].recv == Recv::Value
        && args
            .iter()
            .any(|arg| contains_mut_call_on(methods, arg, place.root)))
    .then_some(SHAPE_VALUE_CALL_INNER)
}

/// Names of the constructs of the enum space (format 9, slice m9.d1b), counted statically by `operator_counts` and when
/// executed by the interpreter, under the same names. The `shape` ones are the order-sensitive forms the generator is
/// biased toward.
pub const MATCH: &str = "match";
pub const MATCH_WILD: &str = "match with _";
pub const MATCH_DEFAULT: &str = "match without _ (the default rule)";
pub const MATCH_GROUPED: &str = "match grouped arm";
pub const MATCH_NESTED: &str = "match nested in an arm";
pub const MATCH_IN_FOR: &str = "match in a for body";
pub const MATCH_ARM_RETURN: &str = "match arm ends in return";
pub const MATCH_ARM_CALL: &str = "match arm calls";
pub const MATCH_ARM_SET: &str = "match arm assigns";
pub const SHAPE_MATCH_SCRUTINEE: &str =
    "shape match scrutinee &mut self call on a root an arm uses";
pub const SHAPE_ENUM_EQ: &str = "shape enum ==/!= of two calls on one root";
pub const SHAPE_ENUM_ARG: &str = "shape enum argument built from a &mut self call";
pub const SHAPE_ENUM_ARGS_ROOT: &str = "shape enum arguments built from calls on one root";

/// Every shape and `match` construct name that `check-difftest` prints a row for, so a name that is never written still
/// shows as a row of zeros (`coverage_json` adds these to the operator rows).
pub const KNOWN_SHAPE_NAMES: [&str; 42] = [
    SHAPE_STMT_CALL,
    SHAPE_EXPR_CALL,
    SHAPE_LOGIC_CALL,
    SHAPE_READ_AND_CALL,
    SHAPE_VALUE_CALL_INNER,
    SHAPE_MATCH_SCRUTINEE,
    SHAPE_ENUM_EQ,
    SHAPE_ENUM_ARG,
    SHAPE_ENUM_ARGS_ROOT,
    MATCH,
    MATCH_WILD,
    MATCH_DEFAULT,
    MATCH_GROUPED,
    MATCH_NESTED,
    MATCH_IN_FOR,
    MATCH_ARM_RETURN,
    MATCH_ARM_CALL,
    MATCH_ARM_SET,
    "return",
    "enum as u32",
    "enum as i32",
    MATCH_VALUE,
    MATCH_VALUE_NESTED,
    MATCH_VALUE_BLOCK,
    MATCH_VALUE_RETURN,
    MATCH_INT,
    MATCH_INT_GROUP,
    MATCH_ARM_RETURN_RUN,
    TUPLE_LIT,
    TUPLE_FIELD,
    TUPLE_READ,
    TUPLE_SET,
    TUPLE_SET_OP,
    TUPLE_DESTRUCT,
    TUPLE_DESTRUCT_TYPED,
    TUPLE_ARG,
    SHAPE_TUPLE_LIT_ROOT,
    SHAPE_TUPLE_SET_CALL,
    SHAPE_TUPLE_FIELD_CALL,
    SHAPE_DESTRUCT_CALL,
    SHAPE_LT_ELEMENT,
    SHAPE_VMATCH_SCRUTINEE,
];

/// Names of the constructs of format 10 (slice m9.d1c), counted statically by `operator_counts` and when executed by
/// the interpreter, under the same names. The `shape` ones are the order-sensitive forms the generator is biased toward.
pub const MATCH_VALUE: &str = "match value (typed let)";
pub const MATCH_VALUE_NESTED: &str = "match value arm is a nested value match";
pub const MATCH_VALUE_BLOCK: &str = "match value arm block with statements";
pub const MATCH_VALUE_RETURN: &str = "match value arm ends in return";
pub const MATCH_INT: &str = "match integer patterns";
pub const MATCH_INT_GROUP: &str = "match integer group arm";
/// Counted only when executed: a `return` ran inside an arm of a `match` (the nit of the m9.d1b re-check, R1).
pub const MATCH_ARM_RETURN_RUN: &str = "match arm return executed";
pub const TUPLE_LIT: &str = "tuple literal";
pub const TUPLE_FIELD: &str = "tuple field of a call";
pub const TUPLE_READ: &str = "tuple element read";
pub const TUPLE_SET: &str = "tuple element set";
pub const TUPLE_SET_OP: &str = "tuple element compound set";
pub const TUPLE_DESTRUCT: &str = "tuple destructuring";
pub const TUPLE_DESTRUCT_TYPED: &str = "tuple destructuring annotated";
pub const TUPLE_ARG: &str = "tuple argument";
pub const SHAPE_TUPLE_LIT_ROOT: &str =
    "shape tuple literal with a &mut self call and another element using its root";
pub const SHAPE_TUPLE_SET_CALL: &str = "shape tuple element set with a &mut self call in the value";
pub const SHAPE_TUPLE_FIELD_CALL: &str = "shape f().k of a &mut self call";
pub const SHAPE_DESTRUCT_CALL: &str = "shape destructuring of a &mut self call";
pub const SHAPE_LT_ELEMENT: &str = "shape < before the last tuple element or argument";
pub const SHAPE_VMATCH_SCRUTINEE: &str =
    "shape value match scrutinee &mut self call on a root an arm uses";

/// The counted name of an enum value read as its repr type.
pub fn enum_as_name(to: Ty) -> String {
    format!("enum as {}", to.name())
}

/// Whether a place steps through a tuple (its root is a tuple local or parameter).
pub fn place_in_tuple(place: &Place) -> bool {
    place.root_ty.is_tuple()
}

/// Whether `e` reads or calls on `root` anywhere: a variable, a field read, a receiver.
fn uses_root(e: &Expr, root: Root) -> bool {
    let mut found = false;
    walk_expr(e, &mut |n| {
        found |= match n {
            Expr::Var(v, _) => root == Root::Var(*v),
            Expr::Read { place, .. } => place.root == root,
            Expr::Call {
                recv: Some(place), ..
            } => place.root == root,
            _ => false,
        };
    });
    found
}

/// `shape tuple literal with a &mut self call and another element using its root`: the elements run left to right, so
/// what an element sees of the root depends on the calls of the elements before it.
pub fn tuple_lit_shape(methods: &Methods, elems: &[Expr]) -> Option<&'static str> {
    let shared = elems.iter().enumerate().any(|(k, element)| {
        let mut roots = Vec::new();
        mutating_roots(methods, element, &mut roots);
        roots.iter().any(|&root| {
            elems
                .iter()
                .enumerate()
                .any(|(l, other)| l != k && uses_root(other, root))
        })
    });
    shared.then_some(SHAPE_TUPLE_LIT_ROOT)
}

/// `shape < before the last tuple element or argument`: a `<` comparison that is not the last element of a tuple literal
/// or the last argument of a call (the shape that made the macro emit `__gust_keep`, fixed by m9.f1).
pub fn lt_before_last(items: &[Expr]) -> Option<&'static str> {
    (items.len() > 1
        && items[..items.len() - 1]
            .iter()
            .any(|item| matches!(item, Expr::Bin { op: BinOp::Lt, .. })))
    .then_some(SHAPE_LT_ELEMENT)
}

/// The names a tuple literal counts under: itself, and the shapes it is.
pub fn tuple_lit_names(methods: &Methods, elems: &[Expr]) -> Vec<&'static str> {
    let mut names = vec![TUPLE_LIT];
    names.extend(tuple_lit_shape(methods, elems));
    names.extend(lt_before_last(elems));
    names
}

/// The names the arguments of a call count under: a tuple argument, and a `<` before the last argument.
pub fn call_arg_names(args: &[Expr]) -> Vec<&'static str> {
    let mut names = Vec::new();
    if args.iter().any(|arg| arg.ty().is_tuple()) {
        names.push(TUPLE_ARG);
    }
    names.extend(lt_before_last(args));
    names
}

/// The names `f().k` counts under: itself, and the shape when `f` is a `&mut self` call.
pub fn tuple_field_names(methods: &Methods, e: &Expr) -> Vec<&'static str> {
    let mut names = vec![TUPLE_FIELD];
    if mut_call_root(methods, e).is_some() {
        names.push(SHAPE_TUPLE_FIELD_CALL);
    }
    names
}

/// The names an assignment counts under besides `set_name`: only for a place inside a tuple.
pub fn tuple_set_names(
    methods: &Methods,
    place: &Place,
    op: Option<BinOp>,
    value: &Expr,
) -> Vec<&'static str> {
    if !place.root_ty.is_tuple() {
        return Vec::new();
    }
    let mut names = vec![if op.is_some() {
        TUPLE_SET_OP
    } else {
        TUPLE_SET
    }];
    let mut roots = Vec::new();
    mutating_roots(methods, value, &mut roots);
    if !roots.is_empty() {
        names.push(SHAPE_TUPLE_SET_CALL);
    }
    names
}

/// The names a destructuring counts under: itself, the annotation, and the shape of a `&mut self` call as its source.
pub fn destruct_names(methods: &Methods, annotate: bool, init: &Expr) -> Vec<&'static str> {
    let mut names = vec![TUPLE_DESTRUCT];
    if annotate {
        names.push(TUPLE_DESTRUCT_TYPED);
    }
    if mut_call_root(methods, init).is_some() {
        names.push(SHAPE_DESTRUCT_CALL);
    }
    names
}

/// The names a value `match` counts under (`tail`: it is the value of an arm of another one): itself, an integer
/// scrutinee, a grouped integer arm, an arm with statements, an arm that returns, and the scrutinee shape.
pub fn value_match_names(
    methods: &Methods,
    scrut: &Expr,
    arms: &[Arm],
    tail: bool,
) -> Vec<&'static str> {
    let mut names = vec![if tail {
        MATCH_VALUE_NESTED
    } else {
        MATCH_VALUE
    }];
    if !scrut.ty().is_enum() {
        names.push(MATCH_INT);
    }
    if arms
        .iter()
        .any(|arm| matches!(&arm.pat, Pat::Lits(lits) if lits.len() > 1))
    {
        names.push(MATCH_INT_GROUP);
    }
    if arms.iter().any(|arm| arm.body.len() > 1) {
        names.push(MATCH_VALUE_BLOCK);
    }
    if arms
        .iter()
        .any(|arm| matches!(arm.body.last(), Some(Stmt::Return { .. })))
    {
        names.push(MATCH_VALUE_RETURN);
    }
    if match_shape(methods, scrut, arms).is_some() {
        names.push(SHAPE_VMATCH_SCRUTINEE);
    }
    names
}

/// Whether a statement list refers to `root` anywhere: a variable, a field read, a place written, a receiver.
pub fn stmts_use_root(stmts: &[Stmt], root: Root) -> bool {
    let mut found = false;
    let mut look = |e: &Expr| {
        found |= match e {
            Expr::Var(v, _) => root == Root::Var(*v),
            Expr::Read { place, .. } => place.root == root,
            Expr::Call {
                recv: Some(place), ..
            } => place.root == root,
            _ => false,
        };
    };
    walk_stmts(stmts, &mut look);
    fn block(stmts: &[Stmt], root: Root) -> bool {
        stmts.iter().any(|stmt| match stmt {
            Stmt::Set { place, .. } => place.root == root,
            Stmt::Assign { var, .. } | Stmt::AssignComp { var, .. } => Root::Var(*var) == root,
            Stmt::Call {
                recv: Some(place), ..
            } => place.root == root,
            Stmt::If { then, els, .. } => block(then, root) || block(els, root),
            Stmt::For { body, .. } => block(body, root),
            Stmt::Match { arms, .. } => arms.iter().any(|arm| block(&arm.body, root)),
            _ => false,
        })
    }
    found || block(stmts, root)
}

/// The shape of a statement `match`: its scrutinee holds a `&mut self` call on a root that an arm uses.
pub fn match_shape(methods: &Methods, scrut: &Expr, arms: &[Arm]) -> Option<&'static str> {
    let mut roots = Vec::new();
    mutating_roots(methods, scrut, &mut roots);
    roots
        .iter()
        .any(|&root| arms.iter().any(|arm| stmts_use_root(&arm.body, root)))
        .then_some(SHAPE_MATCH_SCRUTINEE)
}

/// The shapes of a call with enum arguments: an argument that is a `&mut self` call (a pure helper call does not count:
/// nothing depends on its order), and two arguments that are calls (or hold calls) mutating one root.
pub fn enum_call_shapes(methods: &Methods, args: &[Expr]) -> Vec<&'static str> {
    let mut shapes = Vec::new();
    let built: Vec<&Expr> = args
        .iter()
        .filter(|arg| arg.ty().is_enum() && matches!(arg, Expr::Call { .. }))
        .collect();
    let mut per_arg: Vec<Vec<Root>> = Vec::new();
    for arg in &built {
        let mut roots = Vec::new();
        mutating_roots(methods, arg, &mut roots);
        roots.dedup();
        per_arg.push(roots);
    }
    if per_arg.iter().any(|roots| !roots.is_empty()) {
        shapes.push(SHAPE_ENUM_ARG);
    }
    let shared = per_arg.iter().enumerate().any(|(k, roots)| {
        roots
            .iter()
            .any(|root| per_arg[k + 1..].iter().any(|other| other.contains(root)))
    });
    if shared {
        shapes.push(SHAPE_ENUM_ARGS_ROOT);
    }
    shapes
}

/// The names a statement `match` counts under: itself, its kind of ending, a grouped arm, an arm that returns, calls or
/// assigns, and where it sits (`in_for`: inside a loop body of its function, `nested`: inside an arm of a `match`).
pub fn match_names(arms: &[Arm], in_for: bool, nested: bool) -> Vec<&'static str> {
    let mut names = vec![MATCH];
    names.push(if arms.iter().any(|arm| arm.pat == Pat::Wild) {
        MATCH_WILD
    } else {
        MATCH_DEFAULT
    });
    if arms
        .iter()
        .any(|arm| matches!(&arm.pat, Pat::Variants(v) if v.len() > 1))
    {
        names.push(MATCH_GROUPED);
    }
    if arms
        .iter()
        .any(|arm| matches!(arm.body.last(), Some(Stmt::Return { .. })))
    {
        names.push(MATCH_ARM_RETURN);
    }
    let calls = |arm: &Arm| {
        let mut found = arm.body.iter().any(|s| matches!(s, Stmt::Call { .. }));
        walk_stmts(&arm.body, &mut |e| found |= matches!(e, Expr::Call { .. }));
        found
    };
    if arms.iter().any(calls) {
        names.push(MATCH_ARM_CALL);
    }
    if arms.iter().any(|arm| {
        arm.body
            .iter()
            .any(|s| matches!(s, Stmt::Set { .. } | Stmt::Assign { .. }))
    }) {
        names.push(MATCH_ARM_SET);
    }
    if in_for {
        names.push(MATCH_IN_FOR);
    }
    if nested {
        names.push(MATCH_NESTED);
    }
    names
}

/// The counted name of a struct literal.
pub const STRUCT_LIT: &str = "struct literal";

/// The most structs a method program declares, and the most scalars a struct holds once nested structs are flattened.
pub const MAX_STRUCTS: usize = 3;
pub const MAX_FLAT: usize = 8;
/// The most enums an enum program declares, and the most variants of one.
pub const MAX_ENUMS: usize = 2;
pub const MAX_VARIANTS: usize = 5;

impl Expr {
    pub fn u32(value: u32) -> Expr {
        Expr::Lit(Ty::U32, value)
    }

    /// An `i32` literal. `i32::MIN` has no literal in the emitted dialect, so it is refused here; use `i32_min`.
    pub fn i32(value: i32) -> Expr {
        assert!(value != i32::MIN, "i32::MIN has no literal form");
        Expr::Lit(Ty::I32, value as u32)
    }

    /// An `f32` literal. A negative zero and a non-finite value have no literal here.
    pub fn f32(value: f32) -> Expr {
        assert!(value.is_finite(), "non-finite f32 literal");
        assert!(
            value.to_bits() != (-0.0f32).to_bits(),
            "negative zero has no literal form"
        );
        Expr::Lit(Ty::F32, value.to_bits())
    }

    pub fn bool(value: bool) -> Expr {
        Expr::Lit(Ty::Bool, u32::from(value))
    }

    /// `i32::MIN` as `-2147483647i32 - 1i32` (the exact value, no overflow), the only spelling the emitter uses.
    pub fn i32_min() -> Expr {
        Expr::bin(BinOp::Sub, Expr::i32(i32::MIN + 1), Expr::i32(1))
    }

    pub fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        let ty = l.ty();
        Expr::Bin {
            op,
            ty,
            l: Box::new(l),
            r: Box::new(r),
        }
    }

    /// `-e`. A negated literal is folded into the literal (the emitted text is the same, and the re-parser reads
    /// `-5i32` as one literal), so the AST never holds `Un(Neg, Lit)`.
    pub fn negated(e: Expr) -> Expr {
        match e {
            Expr::Lit(Ty::I32, bits) if bits as i32 != i32::MIN => {
                return Expr::i32(-(bits as i32));
            }
            // `-0.0` stays an operation: the literal form of a negative zero does not exist.
            Expr::Lit(Ty::F32, bits) if f32::from_bits(bits) != 0.0 => {
                return Expr::f32(-f32::from_bits(bits));
            }
            _ => {}
        }
        Expr::Un {
            op: UnOp::Neg,
            ty: e.ty(),
            e: Box::new(e),
        }
    }

    pub fn inverted(e: Expr) -> Expr {
        let ty = e.ty();
        Expr::Un {
            op: UnOp::Not,
            ty,
            e: Box::new(e),
        }
    }

    pub fn cast(to: Ty, e: Expr) -> Expr {
        Expr::Cast { to, e: Box::new(e) }
    }

    pub fn input(buf: usize, ty: Ty, index: Expr) -> Expr {
        Expr::Input {
            buf,
            ty,
            index: Box::new(index),
        }
    }

    pub fn ty(&self) -> Ty {
        match self {
            Expr::Lit(ty, _) | Expr::Var(_, ty) | Expr::Input { ty, .. } | Expr::Un { ty, .. } => {
                *ty
            }
            Expr::Read { ty, .. }
            | Expr::Call { ty, .. }
            | Expr::StructLit { ty, .. }
            | Expr::Tuple { ty, .. } => *ty,
            Expr::TupleField { e, index } => match e.ty() {
                Ty::Tuple(t) => t.elems()[usize::from(*index)].ty(),
                other => panic!("a tuple field of {}", other.name()),
            },
            Expr::Variant { e, .. } => Ty::Enum(*e),
            Expr::EnumAs { to, .. } => *to,
            Expr::Gid | Expr::Len(_) => Ty::U32,
            Expr::Bin { op, ty, .. } => op.result_ty(*ty),
            Expr::Cast { to, .. } => *to,
            Expr::Make { ty, .. } | Expr::Splat { ty, .. } | Expr::AggBin { ty, .. } => *ty,
            Expr::Comp { e, .. } | Expr::Dot { l: e, .. } => e.ty().scalar(),
            Expr::Swizzle { e, lanes } => Ty::Vec(
                match e.ty() {
                    Ty::Vec(lane, _) => lane,
                    other => panic!("a swizzle of {}", other.name()),
                },
                lanes.len() as u8,
            ),
            Expr::Transpose { e } | Expr::AggNeg { e } => e.ty(),
        }
    }

    /// `Vec4::new(..)` or `Vec4::splat(..)`-style construction of a vector from its scalars.
    pub fn vec_new(ty: Ty, args: Vec<Expr>) -> Expr {
        Expr::Make {
            ty,
            rows: false,
            args,
        }
    }

    /// `l op r` on aggregates; panics if `gust::math` has no such operator (use `agg_bin_ty` to ask).
    pub fn agg_bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        let ty = agg_bin_ty(op, l.ty(), r.ty()).unwrap_or_else(|| {
            panic!(
                "no `{}` on {} and {}",
                op.symbol(),
                l.ty().name(),
                r.ty().name()
            )
        });
        Expr::AggBin {
            op,
            ty,
            l: Box::new(l),
            r: Box::new(r),
        }
    }

    pub fn comp(e: Expr, index: u8) -> Expr {
        Expr::Comp {
            e: Box::new(e),
            index,
        }
    }

    pub fn swizzle(e: Expr, lanes: Vec<u8>) -> Expr {
        Expr::Swizzle {
            e: Box::new(e),
            lanes,
        }
    }

    pub fn dot(l: Expr, r: Expr) -> Expr {
        Expr::Dot {
            l: Box::new(l),
            r: Box::new(r),
        }
    }

    /// Calls `f` on each direct child, in the order every traversal of the program uses (the order of
    /// `visit_exprs`, which the reducer's node numbering relies on).
    pub fn for_each_child(&self, f: &mut dyn FnMut(&Expr)) {
        match self {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) | Expr::Variant { .. } => {}
            Expr::Input { index, .. } => f(index),
            Expr::Un { e, .. }
            | Expr::Cast { e, .. }
            | Expr::EnumAs { e, .. }
            | Expr::Splat { e, .. }
            | Expr::Comp { e, .. }
            | Expr::Swizzle { e, .. }
            | Expr::Transpose { e }
            | Expr::AggNeg { e }
            | Expr::TupleField { e, .. } => f(e),
            Expr::Bin { l, r, .. } | Expr::Dot { l, r } | Expr::AggBin { l, r, .. } => {
                f(l);
                f(r);
            }
            Expr::Make { args, .. } => args.iter().for_each(f),
            Expr::Read { .. } => {}
            Expr::Call { args, .. } => args.iter().for_each(f),
            Expr::StructLit { fields, .. } => fields.iter().for_each(f),
            Expr::Tuple { elems, .. } => elems.iter().for_each(f),
        }
    }

    /// [`Expr::for_each_child`] with mutable access.
    pub fn for_each_child_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        match self {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) | Expr::Variant { .. } => {}
            Expr::Input { index, .. } => f(index),
            Expr::Un { e, .. }
            | Expr::Cast { e, .. }
            | Expr::EnumAs { e, .. }
            | Expr::Splat { e, .. }
            | Expr::Comp { e, .. }
            | Expr::Swizzle { e, .. }
            | Expr::Transpose { e }
            | Expr::AggNeg { e }
            | Expr::TupleField { e, .. } => f(e),
            Expr::Bin { l, r, .. } | Expr::Dot { l, r } | Expr::AggBin { l, r, .. } => {
                f(l);
                f(r);
            }
            Expr::Make { args, .. } => args.iter_mut().for_each(f),
            Expr::Read { .. } => {}
            Expr::Call { args, .. } => args.iter_mut().for_each(f),
            Expr::StructLit { fields, .. } => fields.iter_mut().for_each(f),
            Expr::Tuple { elems, .. } => elems.iter_mut().for_each(f),
        }
    }

    /// The number of nodes.
    pub fn size(&self) -> usize {
        let mut total = 1;
        self.for_each_child(&mut |child| total += child.size());
        total
    }

    pub fn reads_input(&self) -> bool {
        if matches!(self, Expr::Input { .. }) {
            return true;
        }
        let mut found = false;
        self.for_each_child(&mut |child| found |= child.reads_input());
        found
    }

    /// Whether the expression reads an `f32` input buffer directly: an `f32` `Input` node that no cast sits above.
    /// A read behind a cast (`(in1[i] * 0u32) as f32`) can fold to a constant, so it does not count.
    pub fn reads_f32_input_directly(&self) -> bool {
        match self {
            Expr::Input { ty, .. } => *ty == Ty::F32,
            Expr::Cast { .. } => false,
            _ => {
                let mut found = false;
                self.for_each_child(&mut |child| found |= child.reads_f32_input_directly());
                found
            }
        }
    }

    /// The counted name of a vector or matrix operator (`new Vec4`, `* Mat4 Vec4`, `.xy Vec4`, ...), the same string in
    /// `operator_counts` and in the interpreter's coverage; `None` for every other node.
    pub fn aggregate_name(&self) -> Option<String> {
        Some(match self {
            Expr::Make {
                ty: ty @ Ty::Mat(_),
                rows,
                ..
            } => format!(
                "{} {}",
                if *rows { "from_rows" } else { "from_cols" },
                ty.name()
            ),
            Expr::Make { ty, .. } => format!("new {}", ty.name()),
            Expr::Splat { ty, .. } => format!("splat {}", ty.name()),
            Expr::Comp { e, .. } => format!("comp {}", e.ty().name()),
            Expr::Swizzle { e, lanes } => format!("swizzle {}->{}", e.ty().name(), lanes.len()),
            Expr::Dot { l, .. } => format!("dot {}", l.ty().name()),
            Expr::Transpose { e } => format!("transpose {}", e.ty().name()),
            Expr::AggBin { op, l, r, .. } => {
                format!("{} {} {}", op.symbol(), l.ty().name(), r.ty().name())
            }
            Expr::AggNeg { e } => format!("-{}", e.ty().name()),
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    Let {
        var: u32,
        ty: Ty,
        mutable: bool,
        init: Expr,
    },
    Assign {
        var: u32,
        value: Expr,
    },
    /// `v.x = value;`: one component of a mutable vector local (`index` 0 is `x`).
    AssignComp {
        var: u32,
        index: u8,
        value: Expr,
    },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        els: Vec<Stmt>,
    },
    /// `for var in 0u32..trips`, `trips` a literal in `1..=MAX_TRIPS`.
    For {
        var: u32,
        trips: u32,
        body: Vec<Stmt>,
    },
    /// `place = value;` (`op` None) or `place op= value;` on a field place or a local, `ty` the type of the place
    /// (an integer type when `op` is set; `bool` has no compound assignment here). Rust evaluates `value` first, then
    /// (for `op=` on primitives) reads the place. (Format 8.)
    Set {
        place: Place,
        ty: Ty,
        op: Option<BinOp>,
        value: Expr,
    },
    /// `recv.m(args);` or `m(args);` with the value, if any, dropped. (Format 8.)
    Call {
        f: usize,
        owner: Option<u8>,
        recv: Option<Place>,
        args: Vec<Expr>,
    },
    /// `match scrut { E0::V0 | E0::V2 => { .. } E0::V1 => { .. } _ => { .. } }` on an enum: the scrutinee is evaluated
    /// once, the first arm whose pattern names the value runs. Without `_` every variant is named once and the last arm
    /// is the `default` of the Slang `switch`. (Format 9.)
    Match {
        scrut: Expr,
        arms: Vec<Arm>,
    },
    /// `return value;` in a helper, the last statement of its block. (Format 9.)
    Return {
        value: Expr,
    },
    /// `let (a, mut b): (u32, S0) = init;`: destructure a tuple into plain names (`vars`: the variable and whether it is
    /// `mut`), the annotation written when `annotate` (always when an element is an enum). `ty` is the tuple type. (Format 10.)
    LetTuple {
        vars: Vec<(u32, bool)>,
        ty: Ty,
        annotate: bool,
        init: Expr,
    },
    /// `let v: ty = match scrut { pat => value, .. };`: the value form of `match`, only as a typed `let` initializer. Each
    /// arm body is a block of statements ending in a `Yield`, a `Return` (helpers) or a `TailMatch`. (Format 10.)
    LetMatch {
        var: u32,
        ty: Ty,
        mutable: bool,
        scrut: Expr,
        arms: Vec<Arm>,
    },
    /// The value of an arm of a value `match`: the last statement of the arm's block, written as the bare expression. (Format 10.)
    Yield {
        value: Expr,
    },
    /// A nested value `match` as the value of an arm: the last statement of the arm's block. (Format 10.)
    TailMatch {
        scrut: Expr,
        arms: Vec<Arm>,
    },
}

/// The largest loop trip count the generator writes.
pub const MAX_TRIPS: u32 = 8;

/// One read-only input buffer; `data` holds one 32-bit pattern per element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputBuf {
    pub ty: Ty,
    pub data: Vec<u32>,
}

/// A generated kernel: `if i < out.len() { <body>; out[i] = <result>; }` over `count` elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub seed: u64,
    pub attempt: u64,
    pub count: u32,
    pub inputs: Vec<InputBuf>,
    pub body: Vec<Stmt>,
    pub result: Expr,
    /// The structs and functions of a method program (format 8); empty for every other space.
    pub methods: Methods,
}

impl Program {
    /// The element type of the output buffer: the result type, with `bool` written as `u32` 0 or 1.
    pub fn out_ty(&self) -> Ty {
        match self.result.ty() {
            Ty::Bool => Ty::U32,
            ty => ty,
        }
    }

    /// Whether any value of the program is an `f32`: an input or the result (a local always descends from one of
    /// them, or from a cast of an integer, which `visit_exprs` sees).
    pub fn has_float(&self) -> bool {
        let mut found = self.inputs.iter().any(|input| input.ty == Ty::F32);
        visit_exprs(self, &mut |e| found |= e.ty().scalar() == Ty::F32);
        found
    }

    /// Whether any expression is a vector or a matrix operation (the vector programs of `Space::Vector`).
    pub fn has_aggregate(&self) -> bool {
        let mut found = false;
        visit_exprs(self, &mut |e| found |= e.ty().is_aggregate());
        found
    }

    /// Whether the kernel itself calls a method or an associated function, in an expression or as a statement (a
    /// method program whose kernel calls nothing, or only free helpers, tests nothing new).
    pub fn has_call(&self) -> bool {
        fn block(methods: &Methods, stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Call { f, .. } => is_method_or_assoc(methods, *f),
                Stmt::If { then, els, .. } => block(methods, then) || block(methods, els),
                Stmt::For { body, .. } => block(methods, body),
                Stmt::Match { arms, .. }
                | Stmt::LetMatch { arms, .. }
                | Stmt::TailMatch { arms, .. } => arms.iter().any(|arm| block(methods, &arm.body)),
                _ => false,
            })
        }
        fn is_method_or_assoc(methods: &Methods, f: usize) -> bool {
            methods.fns[f].recv != Recv::None || methods.fns[f].owner.is_some()
        }
        let methods = &self.methods;
        let mut found = block(methods, &self.body);
        let mut look = |e: &Expr| {
            found |= matches!(e, Expr::Call { f, .. } if is_method_or_assoc(methods, *f));
        };
        walk_stmts(&self.body, &mut look);
        walk_expr(&self.result, &mut look);
        found
    }

    /// Whether the kernel or a function holds a statement `match` (an enum program without one tests nothing new).
    pub fn has_match(&self) -> bool {
        fn block(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Match { .. } => true,
                Stmt::If { then, els, .. } => block(then) || block(els),
                Stmt::For { body, .. } => block(body),
                _ => false,
            })
        }
        block(&self.body) || self.methods.fns.iter().any(|def| block(&def.body))
    }

    /// Whether the program holds a value `match` or a tuple (a value program without either tests nothing new).
    pub fn has_value_form(&self) -> bool {
        fn block(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::LetMatch { .. } | Stmt::LetTuple { .. } => true,
                Stmt::If { then, els, .. } => block(then) || block(els),
                Stmt::For { body, .. } => block(body),
                Stmt::Match { arms, .. } => arms.iter().any(|arm| block(&arm.body)),
                _ => false,
            })
        }
        let mut tuple = false;
        visit_exprs(self, &mut |e| {
            tuple |= matches!(e, Expr::Tuple { .. } | Expr::TupleField { .. }) || e.ty().is_tuple();
        });
        tuple || block(&self.body) || self.methods.fns.iter().any(|def| block(&def.body))
    }

    /// Whether any expression reads an input buffer (a program that reads none is a constant, not a test).
    pub fn reads_input(&self) -> bool {
        fn block(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Let { init, .. } | Stmt::LetTuple { init, .. } => init.reads_input(),
                Stmt::Assign { value, .. }
                | Stmt::AssignComp { value, .. }
                | Stmt::Set { value, .. }
                | Stmt::Yield { value }
                | Stmt::Return { value } => value.reads_input(),
                Stmt::Call { args, .. } => args.iter().any(Expr::reads_input),
                Stmt::If { cond, then, els } => cond.reads_input() || block(then) || block(els),
                Stmt::For { body, .. } => block(body),
                Stmt::Match { scrut, arms }
                | Stmt::LetMatch { scrut, arms, .. }
                | Stmt::TailMatch { scrut, arms } => {
                    scrut.reads_input() || arms.iter().any(|arm| block(&arm.body))
                }
            })
        }
        block(&self.body) || self.result.reads_input()
    }

    /// Verifies the typing and scoping invariants. `Err` names the first violation.
    pub fn check(&self) -> Result<(), String> {
        if self.count == 0 {
            return Err("count is zero".into());
        }
        if self.inputs.is_empty() || self.inputs.len() > 3 {
            return Err(format!("{} input buffers", self.inputs.len()));
        }
        for (k, input) in self.inputs.iter().enumerate() {
            if input.ty == Ty::Bool {
                return Err(format!("input {k} is a bool buffer"));
            }
            if input.ty.is_aggregate() || input.ty.is_struct() || input.ty.is_enum() {
                return Err(format!(
                    "input {k} is a vector, matrix, struct or enum buffer"
                ));
            }
            if input.data.len() != self.count as usize {
                return Err(format!("input {k} has {} elements", input.data.len()));
            }
        }
        if self.result.ty().is_aggregate()
            || self.result.ty().is_struct()
            || self.result.ty().is_enum()
        {
            return Err(
                "the result is a vector, a matrix, a struct or an enum (the output buffer is scalar)"
                    .into(),
            );
        }
        self.methods.check_tables()?;
        let mut cx = Check {
            inputs: self.inputs.iter().map(|input| input.ty).collect(),
            methods: &self.methods,
            scope: Vec::new(),
            seen: BTreeSet::new(),
            in_fn: None,
        };
        // Functions first (their variables are numbered in the program's one id space), each in a scope of its own.
        for (k, def) in self.methods.fns.iter().enumerate() {
            cx.function(k, def)?;
        }
        cx.in_fn = None;
        cx.scope.clear();
        // The body and the result share the guarded block's scope.
        for stmt in &self.body {
            cx.stmt(stmt)?;
        }
        cx.expr(&self.result)
    }
}

impl Methods {
    /// A tuple type of a parameter, a result or a local: 2 to 4 elements that exist, at most `MAX_TUPLE_FLAT` scalars.
    pub fn check_tuple(&self, t: TupleTy) -> Result<(), String> {
        if !(2..=MAX_TUPLE).contains(&usize::from(t.n)) {
            return Err(format!("a tuple of {} elements", t.n));
        }
        for e in t.elems() {
            match e {
                Elem::Struct(j) if usize::from(*j) >= self.structs.len() => {
                    return Err(format!("a tuple element of unknown S{j}"));
                }
                Elem::Enum(k) if usize::from(*k) >= self.enums.len() => {
                    return Err(format!("a tuple element of unknown E{k}"));
                }
                _ => {}
            }
        }
        if self.ty_len(Ty::Tuple(t)) > MAX_TUPLE_FLAT {
            return Err("a tuple of more than 16 scalars".into());
        }
        Ok(())
    }

    /// The structural rules of the tables: struct fields refer to earlier structs, sizes are bounded, every function has
    /// a consistent receiver and return type, and calls only go downwards (checked with the bodies).
    pub fn check_tables(&self) -> Result<(), String> {
        if self.structs.len() > MAX_STRUCTS {
            return Err(format!("{} structs", self.structs.len()));
        }
        if self.enums.len() > MAX_ENUMS {
            return Err(format!("{} enums", self.enums.len()));
        }
        for (k, def) in self.enums.iter().enumerate() {
            if !matches!(def.repr, Ty::U32 | Ty::I32) {
                return Err(format!("E{k} has the repr {}", def.repr.name()));
            }
            if !(2..=MAX_VARIANTS).contains(&def.variants.len()) {
                return Err(format!("E{k} has {} variants", def.variants.len()));
            }
            let mut seen = BTreeSet::new();
            for (v, &(bits, explicit)) in def.variants.iter().enumerate() {
                let value = def.value(v as u8);
                if !explicit {
                    // A variant without `= value` counts up from the one before (the first is 0).
                    let expected = if v == 0 {
                        0
                    } else {
                        def.value(v as u8 - 1) + 1
                    };
                    if value != expected {
                        return Err(format!(
                            "E{k}::V{v} is {value}, counting up gives {expected}"
                        ));
                    }
                }
                if !seen.insert(bits) {
                    return Err(format!("E{k} has the value {value} twice"));
                }
            }
        }
        for (j, def) in self.structs.iter().enumerate() {
            if def.fields.is_empty() || def.fields.len() > 3 {
                return Err(format!("struct S{j} has {} fields", def.fields.len()));
            }
            for ty in &def.fields {
                match ty {
                    // A GPU struct is a `GpuPod` type: its fields are `u32`, `i32` (no `bool`) or an earlier struct.
                    Ty::U32 | Ty::I32 => {}
                    Ty::Struct(inner) if usize::from(*inner) < j => {}
                    other => return Err(format!("S{j} has a field of {}", other.name())),
                }
            }
            if self.flat_len(j as u8) > MAX_FLAT {
                return Err(format!("S{j} is {} scalars", self.flat_len(j as u8)));
            }
        }
        for (k, def) in self.fns.iter().enumerate() {
            if let Some(owner) = def.owner
                && usize::from(owner) >= self.structs.len()
            {
                return Err(format!("m{k} is a function of an unknown struct"));
            }
            if def.recv != Recv::None && def.owner.is_none() {
                return Err(format!("m{k} has a receiver and no struct"));
            }
            for (_, ty) in &def.params {
                match ty {
                    Ty::U32 | Ty::I32 | Ty::Bool => {}
                    Ty::Struct(j) if usize::from(*j) < self.structs.len() => {}
                    Ty::Enum(e) if usize::from(*e) < self.enums.len() => {}
                    Ty::Tuple(t) => self.check_tuple(*t)?,
                    other => return Err(format!("m{k} has a parameter of {}", other.name())),
                }
            }
            if let Some(Ty::Enum(e)) = def.ret
                && usize::from(e) >= self.enums.len()
            {
                return Err(format!("m{k} returns an unknown enum"));
            }
            if let Some(Ty::Tuple(t)) = def.ret {
                self.check_tuple(t)?;
            }
            match (def.recv, def.owner, def.ret) {
                (Recv::None, Some(owner), ret) if ret != Some(Ty::Struct(owner)) => {
                    return Err(format!("associated m{k} does not return its struct"));
                }
                (
                    Recv::None,
                    None,
                    Some(Ty::U32 | Ty::I32 | Ty::Bool | Ty::Enum(_) | Ty::Tuple(_)),
                ) => {}
                (Recv::None, None, _) => return Err(format!("helper m{k} has no scalar result")),
                (
                    Recv::Shared | Recv::Value,
                    _,
                    Some(Ty::U32 | Ty::I32 | Ty::Bool | Ty::Enum(_) | Ty::Tuple(_)),
                ) => {}
                (Recv::Shared | Recv::Value, _, _) => {
                    return Err(format!("method m{k} has no scalar result"));
                }
                (
                    Recv::Mut,
                    _,
                    None | Some(Ty::U32 | Ty::I32 | Ty::Bool | Ty::Enum(_) | Ty::Tuple(_)),
                ) => {}
                (Recv::Mut, _, _) => return Err(format!("method m{k} returns a struct")),
                _ => {}
            }
            if def.ret.is_some() != def.tail.is_some() {
                return Err(format!("m{k}: the tail and the result type disagree"));
            }
        }
        Ok(())
    }
}

struct Check<'a> {
    inputs: Vec<Ty>,
    methods: &'a Methods,
    /// (variable, type, mutable), innermost last.
    scope: Vec<(u32, Ty, bool)>,
    seen: BTreeSet<u32>,
    /// The index of the function being checked (`None` in the kernel). A function sees no buffer, no `i`.
    in_fn: Option<usize>,
}

/// A `return` is the last statement of its block (the statements after one would be dead code).
fn returns_last(stmts: &[Stmt]) -> Result<(), String> {
    match stmts.split_last() {
        Some((_, rest)) if rest.iter().any(|s| matches!(s, Stmt::Return { .. })) => {
            Err("a `return` that is not the last statement of its block".into())
        }
        _ => Ok(()),
    }
}

/// The roots of the `&mut self` calls inside `expr` (a call's receiver, and the calls in its arguments).
fn mutating_roots(methods: &Methods, expr: &Expr, out: &mut Vec<Root>) {
    if let Expr::Call {
        f,
        recv: Some(place),
        ..
    } = expr
        && methods.fns[*f].recv == Recv::Mut
    {
        out.push(place.root);
    }
    expr.for_each_child(&mut |child| mutating_roots(methods, child, out));
}

impl Check<'_> {
    /// The type of a place, or why it is not one.
    fn place_ty(&self, place: &Place) -> Result<Ty, String> {
        let root = match place.root {
            Root::Var(var) => match self.lookup(var) {
                Some((ty, _)) => ty,
                None => return Err(format!("place rooted at unknown v{var}")),
            },
            Root::SelfV => match self.in_fn.and_then(|k| self.methods.fns[k].owner) {
                Some(owner) if self.self_recv() != Recv::None => Ty::Struct(owner),
                _ => return Err("`self` outside a method".into()),
            },
        };
        if root != place.root_ty {
            return Err("a place with the wrong root type".into());
        }
        let mut ty = root;
        for &k in &place.path {
            ty = match ty {
                Ty::Struct(j) => *self.methods.structs[usize::from(j)]
                    .fields
                    .get(usize::from(k))
                    .ok_or_else(|| format!("S{j} has no field {k}"))?,
                Ty::Tuple(t) if k < t.n => t.elems()[usize::from(k)].ty(),
                other => return Err(format!("a field of {}", other.name())),
            };
        }
        Ok(ty)
    }

    fn self_recv(&self) -> Recv {
        self.in_fn.map_or(Recv::None, |k| self.methods.fns[k].recv)
    }

    /// Whether the place may be written: a `let mut` local, or `self` of a `&mut self` method.
    fn writable(&self, place: &Place) -> bool {
        match place.root {
            Root::Var(var) => self.lookup(var).is_some_and(|(_, mutable)| mutable),
            Root::SelfV => self.self_recv() == Recv::Mut,
        }
    }

    /// A call's shape: the callee exists below the caller, the arguments match, the receiver is a place of the
    /// owner's type that may be borrowed as the callee needs, and no `&mut self` call on the receiver's root sits in
    /// the arguments of a call that borrows it (rustc's E0499 and E0502).
    fn call(
        &mut self,
        f: usize,
        owner: Option<u8>,
        recv: &Option<Place>,
        args: &[Expr],
    ) -> Result<Option<Ty>, String> {
        let methods = self.methods;
        let Some(def) = methods.fns.get(f) else {
            return Err(format!("call of unknown m{f}"));
        };
        if def.recv == Recv::None && owner != def.owner {
            return Err(format!("m{f} is called with the wrong path"));
        }
        if self.in_fn.is_some_and(|caller| f >= caller) {
            return Err(format!(
                "m{f} is called from a function that is not above it"
            ));
        }
        if args.len() != def.params.len() {
            return Err(format!("m{f} takes {} arguments", def.params.len()));
        }
        for (arg, (_, ty)) in args.iter().zip(&def.params) {
            self.expr(arg)?;
            if arg.ty() != *ty {
                return Err(format!("argument of m{f} has the wrong type"));
            }
        }
        match (def.recv, recv) {
            (Recv::None, None) => {}
            (Recv::None, Some(_)) | (_, None) => {
                return Err(format!("m{f}: receiver and signature disagree"));
            }
            (kind, Some(place)) => {
                if self.place_ty(place)? != Ty::Struct(def.owner.unwrap_or(u8::MAX)) {
                    return Err(format!("receiver of m{f} has the wrong type"));
                }
                if kind == Recv::Mut && !self.writable(place) {
                    return Err(format!(
                        "`&mut self` call m{f} on a place that is not writable"
                    ));
                }
                if kind != Recv::Value {
                    let mut roots = Vec::new();
                    for arg in args {
                        mutating_roots(methods, arg, &mut roots);
                    }
                    if roots.contains(&place.root) {
                        return Err(format!(
                            "an argument of m{f} calls a `&mut self` method on its receiver's root"
                        ));
                    }
                }
            }
        }
        Ok(def.ret)
    }

    /// One function in a scope of its own: its parameters, its body and its tail.
    fn function(&mut self, k: usize, def: &FnDef) -> Result<(), String> {
        self.in_fn = Some(k);
        self.scope.clear();
        for &(var, ty) in &def.params {
            self.declare(var, ty, false)?;
        }
        returns_last(&def.body)?;
        for stmt in &def.body {
            self.stmt(stmt)?;
        }
        match (&def.tail, def.ret) {
            (Some(tail), Some(ty)) => {
                self.expr(tail)?;
                if tail.ty() != ty {
                    return Err(format!("the tail of m{k} has the wrong type"));
                }
            }
            (None, None) => {}
            _ => return Err(format!("m{k}: the tail and the result type disagree")),
        }
        // A bare `self` as a value is only a `self` method's: in `&mut self` it binds the reference (E0116) and in
        // `&self` it has the type `&Self`.
        if def.recv != Recv::Value {
            let mut bare = false;
            walk_fn_exprs(def, &mut |e| {
                bare |= matches!(e, Expr::Read { place, .. } if place.root == Root::SelfV && place.path.is_empty());
            });
            if bare {
                return Err(format!("m{k} uses `self` as a value"));
            }
        }
        Ok(())
    }

    fn declare(&mut self, var: u32, ty: Ty, mutable: bool) -> Result<(), String> {
        if !self.seen.insert(var) {
            return Err(format!("variable v{var} declared twice"));
        }
        self.scope.push((var, ty, mutable));
        Ok(())
    }

    fn lookup(&self, var: u32) -> Option<(Ty, bool)> {
        self.scope
            .iter()
            .rev()
            .find(|(id, ..)| *id == var)
            .map(|&(_, ty, mutable)| (ty, mutable))
    }

    fn block(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        let mark = self.scope.len();
        returns_last(stmts)?;
        for stmt in stmts {
            self.stmt(stmt)?;
        }
        self.scope.truncate(mark);
        Ok(())
    }

    /// A statement `match` on an enum: the scrutinee is a local, a parameter, a call or a variant path (E0515), the
    /// patterns name each variant at most once (E0521), `_` is last and stands for at least one variant, and without
    /// `_` every variant is named, so Rust accepts the match and the last arm is the `default` of the Slang `switch`.
    fn match_stmt(&mut self, scrut: &Expr, arms: &[Arm], value: Option<Ty>) -> Result<(), String> {
        self.expr(scrut)?;
        let e = match scrut.ty() {
            Ty::Enum(e) => e,
            Ty::U32 | Ty::I32 if value.is_some() => return self.int_match(scrut.ty(), arms, value),
            other => return Err(format!("a match on {}", other.name())),
        };
        if !matches!(
            scrut,
            Expr::Var(..) | Expr::Call { .. } | Expr::Variant { .. }
        ) {
            return Err("a match scrutinee is a local, a call or a variant".into());
        }
        let Some(def) = self.methods.enums.get(usize::from(e)) else {
            return Err(format!("a match on unknown E{e}"));
        };
        let mut named = BTreeSet::new();
        let mut wild = false;
        let mut named_arms = 0;
        for arm in arms {
            if wild {
                return Err("an arm after `_`".into());
            }
            match &arm.pat {
                Pat::Wild => wild = true,
                Pat::Lits(_) => return Err("an integer pattern in an enum match".into()),
                Pat::Variants(variants) => {
                    named_arms += 1;
                    if variants.is_empty() {
                        return Err("an arm with no variant".into());
                    }
                    for &v in variants {
                        if usize::from(v) >= def.variants.len() {
                            return Err(format!("E{e} has no variant V{v}"));
                        }
                        if !named.insert(v) {
                            return Err(format!("E{e}::V{v} is named twice"));
                        }
                    }
                }
            }
            self.arm_body(&arm.body, value)?;
        }
        if named_arms == 0 {
            return Err("a match with no variant arm".into());
        }
        match (wild, named.len() == def.variants.len()) {
            (false, false) => Err("a match without `_` that does not name every variant".into()),
            (true, true) => Err("a `_` arm that no variant reaches".into()),
            _ => Ok(()),
        }
    }

    /// The body of a `match` arm: a block, and in the value form (`value` the type of the `let`) a block that ends in
    /// the value of the arm: a `Yield` of that type, a `Return` (helpers) or a nested value `match`.
    fn arm_body(&mut self, body: &[Stmt], value: Option<Ty>) -> Result<(), String> {
        let Some(ty) = value else {
            return self.block(body);
        };
        let mark = self.scope.len();
        returns_last(body)?;
        let Some((last, rest)) = body.split_last() else {
            return Err("an empty value arm".into());
        };
        for stmt in rest {
            self.stmt(stmt)?;
        }
        match last {
            Stmt::Yield { value } => {
                self.expr(value)?;
                if value.ty() != ty {
                    return Err(format!(
                        "a value arm yields {} for {}",
                        value.ty().name(),
                        ty.name()
                    ));
                }
            }
            Stmt::Return { .. } => self.stmt(last)?,
            Stmt::TailMatch { scrut, arms } => self.match_stmt(scrut, arms, Some(ty))?,
            _ => return Err("a value arm that does not end in its value".into()),
        }
        self.scope.truncate(mark);
        Ok(())
    }

    /// A value `match` on an integer: distinct `u32`/`i32` literal patterns (E0521 for a value twice), the literals of
    /// one arm joined by `|`, and a final `_` (an integer `match` is not exhaustive without it).
    fn int_match(&mut self, scrut_ty: Ty, arms: &[Arm], value: Option<Ty>) -> Result<(), String> {
        let mut seen = BTreeSet::new();
        let mut wild = false;
        let mut literal_arms = 0;
        for arm in arms {
            if wild {
                return Err("an arm after `_`".into());
            }
            match &arm.pat {
                Pat::Wild => wild = true,
                Pat::Variants(_) => return Err("a variant pattern in an integer match".into()),
                Pat::Lits(lits) => {
                    literal_arms += 1;
                    if lits.is_empty() {
                        return Err("an arm with no literal".into());
                    }
                    for &(bits, _) in lits {
                        if scrut_ty == Ty::I32 && bits == i32::MIN as u32 {
                            return Err("an i32::MIN pattern".into());
                        }
                        if !seen.insert(bits) {
                            return Err(format!("the pattern {bits} is written twice"));
                        }
                    }
                }
            }
            self.arm_body(&arm.body, value)?;
        }
        if literal_arms == 0 || !wild {
            return Err("an integer match needs a literal arm and a final `_`".into());
        }
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let {
                var,
                ty,
                mutable,
                init,
            } => {
                self.expr(init)?;
                if init.ty() != *ty {
                    return Err(format!(
                        "let v{var}: {} initialised with {}",
                        ty.name(),
                        init.ty().name()
                    ));
                }
                self.declare(*var, *ty, *mutable)
            }
            Stmt::Assign { var, value } => {
                self.expr(value)?;
                match self.lookup(*var) {
                    Some((ty, true)) if ty == value.ty() => Ok(()),
                    Some((_, true)) => Err(format!("assignment to v{var} has the wrong type")),
                    Some((_, false)) => Err(format!("assignment to immutable v{var}")),
                    None => Err(format!("assignment to unknown v{var}")),
                }
            }
            Stmt::AssignComp { var, index, value } => {
                self.expr(value)?;
                match self.lookup(*var) {
                    Some((Ty::Vec(lane, width), true)) => {
                        if *index >= width {
                            Err(format!("component {index} of a {width}-wide v{var}"))
                        } else if value.ty() != lane.ty() {
                            Err(format!("component assignment to v{var} has the wrong type"))
                        } else {
                            Ok(())
                        }
                    }
                    Some((_, true)) => Err(format!("component assignment to a non-vector v{var}")),
                    Some((_, false)) => Err(format!("component assignment to immutable v{var}")),
                    None => Err(format!("component assignment to unknown v{var}")),
                }
            }
            Stmt::If { cond, then, els } => {
                self.expr(cond)?;
                if cond.ty() != Ty::Bool {
                    return Err("`if` condition is not bool".into());
                }
                self.block(then)?;
                self.block(els)
            }
            Stmt::For { var, trips, body } => {
                if !(1..=MAX_TRIPS).contains(trips) {
                    return Err(format!("loop trip count {trips}"));
                }
                let mark = self.scope.len();
                self.declare(*var, Ty::U32, false)?;
                self.block(body)?;
                self.scope.truncate(mark);
                Ok(())
            }
            Stmt::Set {
                place,
                ty,
                op,
                value,
            } => {
                self.expr(value)?;
                if self.place_ty(place)? != *ty {
                    return Err("assignment to a place of another type".into());
                }
                if !self.writable(place) {
                    return Err("assignment to a place that is not writable".into());
                }
                match op {
                    None if value.ty() == *ty => Ok(()),
                    None => Err("assignment of the wrong type".into()),
                    Some(op) => {
                        let right = if op.is_shift() { Ty::U32 } else { *ty };
                        let allowed = matches!(
                            op,
                            BinOp::Add
                                | BinOp::Sub
                                | BinOp::Mul
                                | BinOp::Div
                                | BinOp::Rem
                                | BinOp::BitAnd
                                | BinOp::BitOr
                                | BinOp::BitXor
                                | BinOp::Shl
                                | BinOp::Shr
                        );
                        if !allowed || !ty.is_int() || value.ty() != right {
                            return Err(format!("compound `{}=` on {}", op.symbol(), ty.name()));
                        }
                        Ok(())
                    }
                }
            }
            Stmt::Call {
                f,
                owner,
                recv,
                args,
            } => self.call(*f, *owner, recv, args).map(|_| ()),
            Stmt::Match { scrut, arms } => self.match_stmt(scrut, arms, None),
            Stmt::Yield { .. } | Stmt::TailMatch { .. } => {
                Err("a `Yield` or `TailMatch` that does not end a value arm".into())
            }
            Stmt::LetMatch {
                var,
                ty,
                mutable,
                scrut,
                arms,
            } => {
                if !matches!(ty, Ty::U32 | Ty::I32 | Ty::Bool | Ty::Enum(_)) {
                    return Err(format!("a value match of type {}", ty.name()));
                }
                self.match_stmt(scrut, arms, Some(*ty))?;
                self.declare(*var, *ty, *mutable)
            }
            Stmt::LetTuple {
                vars,
                ty,
                annotate,
                init,
            } => {
                self.expr(init)?;
                let Ty::Tuple(t) = ty else {
                    return Err("a destructuring of a non-tuple".into());
                };
                self.methods.check_tuple(*t)?;
                if init.ty() != *ty || vars.len() != usize::from(t.n) {
                    return Err("a destructuring that does not fit its tuple".into());
                }
                if !annotate && t.elems().iter().any(|e| matches!(e, Elem::Enum(_))) {
                    return Err("an unannotated destructuring of an enum".into());
                }
                for (&(var, mutable), elem) in vars.iter().zip(t.elems()) {
                    self.declare(var, elem.ty(), mutable)?;
                }
                Ok(())
            }
            Stmt::Return { value } => {
                self.expr(value)?;
                let helper = self.in_fn.map(|k| &self.methods.fns[k]);
                match helper {
                    Some(def)
                        if def.recv == Recv::None
                            && def.owner.is_none()
                            && def.ret == Some(value.ty()) =>
                    {
                        Ok(())
                    }
                    _ => Err("a `return` outside a helper, or of the wrong type".into()),
                }
            }
        }
    }

    fn expr(&mut self, expr: &Expr) -> Result<(), String> {
        match expr {
            Expr::Lit(Ty::Bool, bits) if *bits > 1 => Err("bool literal out of range".into()),
            Expr::Lit(Ty::I32, bits) if *bits == i32::MIN as u32 => Err("i32::MIN literal".into()),
            Expr::Lit(Ty::F32, bits) if !f32::from_bits(*bits).is_finite() => {
                Err("non-finite f32 literal".into())
            }
            Expr::Lit(Ty::F32, bits) if *bits == (-0.0f32).to_bits() => {
                Err("negative zero literal".into())
            }
            Expr::Lit(ty, _) if ty.is_aggregate() || ty.is_struct() || ty.is_enum() => {
                Err("a vector, matrix, struct or enum literal".into())
            }
            Expr::Variant { e, v } => match self.methods.enums.get(usize::from(*e)) {
                Some(def) if usize::from(*v) < def.variants.len() => Ok(()),
                _ => Err(format!("E{e}::V{v} does not exist")),
            },
            Expr::EnumAs { to, e } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Enum(k)
                        if self.methods.enums.get(usize::from(k)).map(|def| def.repr)
                            == Some(*to) =>
                    {
                        Ok(())
                    }
                    other => Err(format!("{} as {}", other.name(), to.name())),
                }
            }
            Expr::Gid if self.in_fn.is_some() => Err("`i` inside a function".into()),
            Expr::Lit(..) | Expr::Gid => Ok(()),
            Expr::Var(var, ty) => match self.lookup(*var) {
                Some((declared, _)) if declared == *ty => Ok(()),
                Some(_) => Err(format!("v{var} used at the wrong type")),
                None => Err(format!("v{var} used out of scope")),
            },
            Expr::Len(_) | Expr::Input { .. } if self.in_fn.is_some() => {
                Err("a buffer inside a function".into())
            }
            Expr::Len(buf) => {
                if *buf < self.inputs.len() {
                    Ok(())
                } else {
                    Err(format!("len of unknown buffer {buf}"))
                }
            }
            Expr::Input { buf, ty, index } => {
                self.expr(index)?;
                if ty.is_aggregate()
                    || ty.is_struct()
                    || ty.is_enum()
                    || self.inputs.get(*buf) != Some(ty)
                {
                    return Err(format!("input {buf} read at the wrong type"));
                }
                if index.ty() != Ty::U32 {
                    return Err("buffer index is not u32".into());
                }
                Ok(())
            }
            Expr::Un { op, ty, e } => {
                self.expr(e)?;
                if e.ty() != *ty {
                    return Err("unary operand type mismatch".into());
                }
                match op {
                    UnOp::Neg if !matches!(*ty, Ty::I32 | Ty::F32) => {
                        Err("negation of a non-i32, non-f32".into())
                    }
                    UnOp::Neg if matches!(**e, Expr::Lit(Ty::I32, _)) => {
                        Err("negated literal not folded".into())
                    }
                    UnOp::Neg if matches!(**e, Expr::Lit(Ty::F32, bits) if f32::from_bits(bits) != 0.0) => {
                        Err("negated literal not folded".into())
                    }
                    UnOp::Not if *ty == Ty::F32 => Err("`!` on f32".into()),
                    UnOp::Not if ty.is_aggregate() || ty.is_struct() || ty.is_enum() => {
                        Err("`!` on a vector, matrix, struct or enum".into())
                    }
                    UnOp::Neg if ty.is_struct() => Err("negation of a struct".into()),
                    _ => Ok(()),
                }
            }
            Expr::Bin { op, ty, l, r } => {
                self.expr(l)?;
                self.expr(r)?;
                if l.ty() != *ty {
                    return Err("binary left operand type mismatch".into());
                }
                if ty.is_aggregate() {
                    return Err(format!("`{}` on {} is an `AggBin`", op.symbol(), ty.name()));
                }
                if ty.is_struct() {
                    return Err(format!("`{}` on a struct", op.symbol()));
                }
                if ty.is_enum() {
                    // `==` and `!=` only: the enum derives `PartialEq` and nothing else.
                    return if matches!(op, BinOp::Eq | BinOp::Ne) && r.ty() == *ty {
                        Ok(())
                    } else {
                        Err(format!("`{}` on an enum", op.symbol()))
                    };
                }
                let (need_int, right) = if op.is_shift() {
                    (true, Ty::U32)
                } else if op.is_logical() {
                    (false, Ty::Bool)
                } else if op.is_comparison()
                    || (*ty == Ty::F32
                        && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div))
                {
                    (false, *ty)
                } else {
                    (true, *ty)
                };
                if op.is_logical() && *ty != Ty::Bool {
                    return Err("logical operator on a non-bool".into());
                }
                if op.is_comparison() && !matches!(op, BinOp::Eq | BinOp::Ne) && *ty == Ty::Bool {
                    return Err(format!("`{}` on bool", op.symbol()));
                }
                if need_int && !ty.is_int() {
                    return Err(format!("`{}` on {}", op.symbol(), ty.name()));
                }
                if r.ty() != right {
                    return Err(format!("`{}` right operand type", op.symbol()));
                }
                Ok(())
            }
            Expr::Cast { to, e } => {
                self.expr(e)?;
                let castable = |ty: Ty| {
                    ty != Ty::Bool && !ty.is_aggregate() && !ty.is_struct() && !ty.is_enum()
                };
                if !castable(*to) || !castable(e.ty()) || e.ty() == *to {
                    return Err("cast outside u32, i32 and f32".into());
                }
                Ok(())
            }
            Expr::Make { ty, rows, args } => {
                for arg in args {
                    self.expr(arg)?;
                }
                let want = match ty {
                    Ty::Vec(lane, width) if WIDTHS.contains(width) && !rows => {
                        (usize::from(*width), lane.ty())
                    }
                    Ty::Mat(n) if WIDTHS.contains(n) => (usize::from(*n), Ty::Vec(Lane::F32, *n)),
                    _ => return Err(format!("`{}` cannot be built from components", ty.name())),
                };
                if args.len() != want.0 || args.iter().any(|arg| arg.ty() != want.1) {
                    return Err(format!(
                        "`{}` needs {} arguments of type {}",
                        ty.name(),
                        want.0,
                        want.1.name()
                    ));
                }
                // SEMANTICS section 4: a compile-time constant `Mat2` crashes naga's SPIR-V front end. A matrix whose
                // constructor reads no `f32` input buffer directly (a read behind a cast can fold to a constant) is a
                // candidate for constant folding, so the generator never writes one.
                if *ty == Ty::Mat(2) && !args.iter().any(Expr::reads_f32_input_directly) {
                    return Err(
                        "a `Mat2` with no direct `f32` input read could be a compile-time constant"
                            .into(),
                    );
                }
                Ok(())
            }
            Expr::Splat { ty, e } => {
                self.expr(e)?;
                match ty {
                    Ty::Vec(lane, width) if WIDTHS.contains(width) && e.ty() == lane.ty() => Ok(()),
                    _ => Err(format!("`{}::splat` of {}", ty.name(), e.ty().name())),
                }
            }
            Expr::Comp { e, index } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Vec(_, width) if *index < width => Ok(()),
                    other => Err(format!("component {index} of {}", other.name())),
                }
            }
            Expr::Swizzle { e, lanes } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Vec(_, width)
                        if WIDTHS.contains(&(lanes.len() as u8))
                            && lanes.iter().all(|&k| k < width) =>
                    {
                        Ok(())
                    }
                    other => Err(format!(
                        "a {}-letter swizzle of {}",
                        lanes.len(),
                        other.name()
                    )),
                }
            }
            Expr::Dot { l, r } => {
                self.expr(l)?;
                self.expr(r)?;
                match (l.ty(), r.ty()) {
                    (a @ Ty::Vec(..), b) if a == b => Ok(()),
                    (a, b) => Err(format!("dot of {} and {}", a.name(), b.name())),
                }
            }
            Expr::Transpose { e } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Mat(_) => Ok(()),
                    other => Err(format!("transpose of {}", other.name())),
                }
            }
            Expr::AggBin { op, ty, l, r } => {
                self.expr(l)?;
                self.expr(r)?;
                if agg_bin_ty(*op, l.ty(), r.ty()) != Some(*ty) {
                    return Err(format!(
                        "`{}` on {} and {} is not {}",
                        op.symbol(),
                        l.ty().name(),
                        r.ty().name(),
                        ty.name()
                    ));
                }
                Ok(())
            }
            Expr::AggNeg { e } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Vec(Lane::F32 | Lane::I32, _) | Ty::Mat(_) => Ok(()),
                    other => Err(format!("negation of {}", other.name())),
                }
            }
            Expr::Read { place, ty } => {
                if self.place_ty(place)? != *ty {
                    return Err("a field read at the wrong type".into());
                }
                match place.root {
                    Root::Var(_) if place.path.is_empty() => {
                        Err("a bare local is a `Var`, not a `Read`".into())
                    }
                    Root::SelfV if place.path.is_empty() && self.self_recv() != Recv::Value => {
                        Err("`self` as a value outside a `self` method".into())
                    }
                    _ => Ok(()),
                }
            }
            Expr::Call {
                f,
                owner,
                recv,
                args,
                ty,
            } => match self.call(*f, *owner, recv, args)? {
                Some(ret) if ret == *ty => Ok(()),
                _ => Err(format!("a call of m{f} has the wrong type")),
            },
            Expr::Tuple { ty, elems } => {
                let Ty::Tuple(t) = ty else {
                    return Err("a tuple literal of a non-tuple".into());
                };
                self.methods.check_tuple(*t)?;
                if elems.len() != usize::from(t.n) {
                    return Err("a tuple literal of the wrong length".into());
                }
                for (elem, want) in elems.iter().zip(t.elems()) {
                    // E0810: a struct literal is only a `let` initializer or the right side of an assignment.
                    if matches!(elem, Expr::StructLit { .. }) {
                        return Err("a struct literal as a tuple element".into());
                    }
                    self.expr(elem)?;
                    if elem.ty() != want.ty() {
                        return Err("a tuple element of the wrong type".into());
                    }
                }
                Ok(())
            }
            Expr::TupleField { e, index } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Tuple(t) if *index < t.n && matches!(**e, Expr::Call { .. }) => Ok(()),
                    other => Err(format!("element {index} of {} (not a call)", other.name())),
                }
            }
            Expr::StructLit { ty, fields, order } => {
                let Ty::Struct(j) = ty else {
                    return Err("a struct literal of a non-struct".into());
                };
                let Some(def) = self.methods.structs.get(usize::from(*j)) else {
                    return Err(format!("a literal of unknown S{j}"));
                };
                if fields.len() != def.fields.len() {
                    return Err(format!("S{j} literal has {} fields", fields.len()));
                }
                let mut seen = order.clone();
                seen.sort_unstable();
                if order.len() != fields.len()
                    || seen.iter().enumerate().any(|(k, &d)| usize::from(d) != k)
                {
                    return Err(format!("S{j} literal order is not a permutation"));
                }
                for (field, &d) in fields.iter().zip(order) {
                    let want = &def.fields[usize::from(d)];
                    self.expr(field)?;
                    if field.ty() != *want {
                        return Err(format!("a field of the S{j} literal has the wrong type"));
                    }
                }
                Ok(())
            }
        }
    }
}

/// Calls `f` on `e` and every node below it (pre-order).
pub fn walk_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    e.for_each_child(&mut |child| walk_expr(child, f));
}

/// Calls `f` on every expression node of a statement list (pre-order): both branches of every `if`, loop bodies.
pub fn walk_stmts(stmts: &[Stmt], f: &mut dyn FnMut(&Expr)) {
    for stmt in stmts {
        match stmt {
            Stmt::Let { init, .. } | Stmt::LetTuple { init, .. } => walk_expr(init, f),
            Stmt::Assign { value, .. }
            | Stmt::AssignComp { value, .. }
            | Stmt::Set { value, .. }
            | Stmt::Yield { value }
            | Stmt::Return { value } => walk_expr(value, f),
            Stmt::Call { args, .. } => args.iter().for_each(|arg| walk_expr(arg, f)),
            Stmt::If { cond, then, els } => {
                walk_expr(cond, f);
                walk_stmts(then, f);
                walk_stmts(els, f);
            }
            Stmt::For { body, .. } => walk_stmts(body, f),
            Stmt::Match { scrut, arms }
            | Stmt::LetMatch { scrut, arms, .. }
            | Stmt::TailMatch { scrut, arms } => {
                walk_expr(scrut, f);
                for arm in arms {
                    walk_stmts(&arm.body, f);
                }
            }
        }
    }
}

/// [`walk_stmts`] over a function: its body, then its tail.
pub fn walk_fn_exprs(def: &FnDef, f: &mut dyn FnMut(&Expr)) {
    walk_stmts(&def.body, f);
    if let Some(tail) = &def.tail {
        walk_expr(tail, f);
    }
}

/// Calls `f` on every expression node of `program` (pre-order): the body's statements, both branches of every `if`,
/// loop bodies and the result, dead code included, then the body and tail of every function in table order. (The
/// reducer's node numbering is this order.)
pub fn visit_exprs(program: &Program, f: &mut dyn FnMut(&Expr)) {
    walk_stmts(&program.body, f);
    walk_expr(&program.result, f);
    for def in &program.methods.fns {
        walk_fn_exprs(def, f);
    }
}

/// The number of terms of the maximal `+`/`-` chain that starts at `expr` (1 if `expr` is not an `f32` addition or
/// subtraction, or the same on float vectors or matrices, which add lane by lane). This is the "written terms" count of
/// R1-4; a chain continued through a variable is not counted.
pub fn chain_terms(expr: &Expr) -> usize {
    match expr {
        Expr::Bin {
            op: BinOp::Add | BinOp::Sub,
            ty: Ty::F32,
            l,
            r,
        } => chain_terms(l) + chain_terms(r),
        Expr::AggBin {
            op: BinOp::Add | BinOp::Sub,
            ty,
            l,
            r,
        } if ty.scalar() == Ty::F32 => chain_terms(l) + chain_terms(r),
        _ => 1,
    }
}

/// The counted name of `AssignComp` statements.
pub const ASSIGN_COMP: &str = "assign comp";

/// How many times each operator is used in `program` (for coverage counts in tests and reports).
pub fn operator_counts(program: &Program) -> BTreeMap<String, usize> {
    fn expr(methods: &Methods, e: &Expr, counts: &mut BTreeMap<String, usize>) {
        let name = match e {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) | Expr::Input { .. } => None,
            Expr::Read { place, .. } => place.root_ty.is_tuple().then(|| TUPLE_READ.to_owned()),
            Expr::Tuple { .. } | Expr::TupleField { .. } => None,
            Expr::Un { op, ty, .. } => Some(match op {
                UnOp::Neg if *ty == Ty::F32 => "-f32".to_owned(),
                UnOp::Neg => "-".to_owned(),
                UnOp::Not => format!("!{}", ty.name()),
            }),
            Expr::Bin { op, ty, .. } => Some(format!("{} {}", op.symbol(), ty.name())),
            Expr::Cast { to, .. } => Some(format!("as {}", to.name())),
            Expr::Call { f, .. } => {
                let def = &methods.fns[*f];
                Some(call_name(def.recv, def.owner).to_owned())
            }
            Expr::StructLit { .. } => Some(STRUCT_LIT.to_owned()),
            Expr::EnumAs { to, .. } => Some(enum_as_name(*to)),
            Expr::Variant { .. } => None,
            other => other.aggregate_name(),
        };
        if let Some(name) = name {
            *counts.entry(name).or_default() += 1;
        }
        let shape = match e {
            Expr::Bin { op, l, r, .. } => bin_shape(methods, *op, l, r),
            Expr::Call { f, recv, args, .. } => {
                if methods.fns[*f].recv == Recv::Mut {
                    *counts.entry(SHAPE_EXPR_CALL.to_owned()).or_default() += 1;
                }
                call_shape(methods, *f, recv, args)
            }
            _ => None,
        };
        if let Some(shape) = shape {
            *counts.entry(shape.to_owned()).or_default() += 1;
        }
        if let Expr::Call { args, .. } = e {
            for shape in enum_call_shapes(methods, args) {
                *counts.entry(shape.to_owned()).or_default() += 1;
            }
            for name in call_arg_names(args) {
                *counts.entry(name.to_owned()).or_default() += 1;
            }
        }
        match e {
            Expr::Tuple { elems, .. } => {
                for name in tuple_lit_names(methods, elems) {
                    *counts.entry(name.to_owned()).or_default() += 1;
                }
            }
            Expr::TupleField { e: inner, .. } => {
                for name in tuple_field_names(methods, inner) {
                    *counts.entry(name.to_owned()).or_default() += 1;
                }
            }
            _ => {}
        }
        e.for_each_child(&mut |child| expr(methods, child, counts));
    }
    fn block(
        methods: &Methods,
        stmts: &[Stmt],
        in_for: bool,
        nested: bool,
        counts: &mut BTreeMap<String, usize>,
    ) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { init, .. } => expr(methods, init, counts),
                Stmt::LetTuple { annotate, init, .. } => {
                    for name in destruct_names(methods, *annotate, init) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    expr(methods, init, counts);
                }
                Stmt::Yield { value } => expr(methods, value, counts),
                Stmt::LetMatch { scrut, arms, .. } | Stmt::TailMatch { scrut, arms } => {
                    let tail = matches!(stmt, Stmt::TailMatch { .. });
                    for name in value_match_names(methods, scrut, arms, tail) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    expr(methods, scrut, counts);
                    for arm in arms {
                        block(methods, &arm.body, in_for, true, counts);
                    }
                }
                Stmt::Assign { value, .. } => expr(methods, value, counts),
                Stmt::AssignComp { value, .. } => {
                    *counts.entry(ASSIGN_COMP.to_owned()).or_default() += 1;
                    expr(methods, value, counts);
                }
                Stmt::Set {
                    place, op, value, ..
                } => {
                    *counts.entry(set_name(*op)).or_default() += 1;
                    if let Some(shape) = set_shape(methods, place, *op, value) {
                        *counts.entry(shape).or_default() += 1;
                    }
                    for name in tuple_set_names(methods, place, *op, value) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    expr(methods, value, counts);
                }
                Stmt::Call { f, recv, args, .. } => {
                    let def = &methods.fns[*f];
                    if def.recv == Recv::Mut {
                        *counts.entry(SHAPE_STMT_CALL.to_owned()).or_default() += 1;
                    }
                    if let Some(shape) = call_shape(methods, *f, recv, args) {
                        *counts.entry(shape.to_owned()).or_default() += 1;
                    }
                    *counts
                        .entry(call_name(def.recv, def.owner).to_owned())
                        .or_default() += 1;
                    for shape in enum_call_shapes(methods, args) {
                        *counts.entry(shape.to_owned()).or_default() += 1;
                    }
                    for name in call_arg_names(args) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    args.iter().for_each(|arg| expr(methods, arg, counts));
                }
                Stmt::If { cond, then, els } => {
                    expr(methods, cond, counts);
                    block(methods, then, in_for, nested, counts);
                    block(methods, els, in_for, nested, counts);
                }
                Stmt::For { body, .. } => {
                    *counts.entry("for".to_owned()).or_default() += 1;
                    block(methods, body, true, nested, counts);
                }
                Stmt::Match { scrut, arms } => {
                    for name in match_names(arms, in_for, nested) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    if let Some(shape) = match_shape(methods, scrut, arms) {
                        *counts.entry(shape.to_owned()).or_default() += 1;
                    }
                    expr(methods, scrut, counts);
                    for arm in arms {
                        block(methods, &arm.body, in_for, true, counts);
                    }
                }
                Stmt::Return { value } => {
                    *counts.entry("return".to_owned()).or_default() += 1;
                    expr(methods, value, counts);
                }
            }
        }
    }
    let mut counts = BTreeMap::new();
    let methods = &program.methods;
    block(methods, &program.body, false, false, &mut counts);
    expr(methods, &program.result, &mut counts);
    for def in &methods.fns {
        block(methods, &def.body, false, false, &mut counts);
        if let Some(tail) = &def.tail {
            expr(methods, tail, &mut counts);
        }
    }
    counts
}

/// Every type a value can have, for the tables below.
fn value_types() -> Vec<Ty> {
    let mut types = vec![Ty::U32, Ty::I32, Ty::F32];
    for lane in [Lane::U32, Lane::I32, Lane::F32] {
        for n in WIDTHS {
            types.push(Ty::Vec(lane, n));
        }
    }
    types.extend(WIDTHS.map(Ty::Mat));
    types
}

/// The operator names (as `operator_counts` writes them) of every row the generator can write: each `agg_bin_ty` row
/// with the four arithmetic operators, each negation (`i32` and `f32` vectors, matrices) and each swizzle shape
/// (a vector of width 2 or 4 read as a vector of width 2 or 4).
pub fn writable_aggregate_forms() -> BTreeSet<String> {
    let mut forms = BTreeSet::new();
    let types = value_types();
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div] {
        for &l in &types {
            for &r in &types {
                if agg_bin_ty(op, l, r).is_some() {
                    forms.insert(format!("{} {} {}", op.symbol(), l.name(), r.name()));
                }
            }
        }
    }
    for &ty in &types {
        match ty {
            Ty::Vec(Lane::I32 | Lane::F32, _) | Ty::Mat(_) => {
                forms.insert(format!("-{}", ty.name()));
            }
            _ => {}
        }
        if let Ty::Vec(..) = ty {
            for width in WIDTHS {
                forms.insert(format!("swizzle {}->{}", ty.name(), width));
            }
        }
    }
    forms
}

//! The types of the AST: lanes, vector, matrix, array and tuple types, and the aggregate operator typing.

use super::*;

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
    /// A fixed-size array `[T; N]` (format 11, slice m9.d1d). It lives in a local, a parameter or a result; it is never a
    /// field, a buffer element, a tuple element or the result of the kernel. Like a tuple it describes itself.
    Array(ArrTy),
    /// A type parameter of the generic function being drawn or checked: `T` (0) or `U` (1) (format 13, slice m9.d2). It
    /// lives in a parameter, a `let x: T`, a result of a generic function and the receiver of a call through a bound.
    /// Its value is a struct (flat components) or an enum of the instantiation; the interpreter binds it per call.
    Param(u8),
}

/// An element type of a tuple or an array: a scalar, a module struct or a module enum (format 10), and (arrays only,
/// format 11) a vector or a matrix. The generator draws no vector or matrix element (see DIFFTEST.md); the interpreter,
/// the emitter and the checker handle them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Elem {
    U32,
    I32,
    Bool,
    Struct(u8),
    Enum(u8),
    Vec(Lane, u8),
    Mat(u8),
}

impl Elem {
    pub fn ty(self) -> Ty {
        match self {
            Elem::U32 => Ty::U32,
            Elem::I32 => Ty::I32,
            Elem::Bool => Ty::Bool,
            Elem::Struct(j) => Ty::Struct(j),
            Elem::Enum(e) => Ty::Enum(e),
            Elem::Vec(lane, n) => Ty::Vec(lane, n),
            Elem::Mat(n) => Ty::Mat(n),
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

    /// The element kind of a type, if an array may hold it (a tuple's kinds, and vectors and matrices).
    pub fn of_array(ty: Ty) -> Option<Elem> {
        match ty {
            Ty::Vec(lane, n) => Some(Elem::Vec(lane, n)),
            Ty::Mat(n) => Some(Elem::Mat(n)),
            other => Elem::of(other),
        }
    }
}

/// The most elements of a generated array, and the most scalar components its flat value holds (an `Agg`).
pub const MAX_ARRAY: usize = 5;
pub const MAX_ARRAY_FLAT: usize = 16;

/// An array type: `n` elements of `e`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArrTy {
    pub n: u8,
    pub e: Elem,
}

impl ArrTy {
    pub fn new(e: Elem, n: u8) -> ArrTy {
        assert!(n >= 1, "an array of no elements");
        ArrTy { n, e }
    }

    /// The Rust spelling: `[u32; 3]`.
    pub fn text(&self) -> String {
        format!("[{}; {}]", self.e.ty().text(), self.n)
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
            Ty::Array(a) => a.text(),
            other => other.name().to_owned(),
        }
    }

    /// A tuple.
    pub fn is_tuple(self) -> bool {
        matches!(self, Ty::Tuple(_))
    }

    /// An array.
    pub fn is_array(self) -> bool {
        matches!(self, Ty::Array(_))
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
            Ty::Array(_) => "(array)",
            Ty::Param(0) => "T",
            Ty::Param(1) => "U",
            Ty::Vec(..) | Ty::Mat(_) | Ty::Struct(_) | Ty::Enum(_) | Ty::Param(_) => {
                panic!("a width, struct, enum or parameter outside the program space: {self:?}")
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
            Ty::Vec(..)
            | Ty::Mat(_)
            | Ty::Struct(_)
            | Ty::Enum(_)
            | Ty::Tuple(_)
            | Ty::Array(_)
            | Ty::Param(_) => {
                panic!("{} is not a buffer element type", self.name())
            }
        }
    }

    /// The spelling of a struct field, a parameter or a return type: the dialect name of a scalar, `S<k>` of a struct,
    /// `E<k>` of an enum. (A tuple is spelled by `text`.)
    pub fn decl_name(self) -> &'static str {
        match self {
            Ty::Struct(_) | Ty::Enum(_) | Ty::Param(_) => self.name(),
            other => other.dialect_name(),
        }
    }

    /// `decl_name` for any type, a tuple included.
    pub fn decl_text(self) -> String {
        match self {
            Ty::Tuple(t) => t.text(),
            Ty::Array(a) => format!("[{}; {}]", a.e.ty().decl_text(), a.n),
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

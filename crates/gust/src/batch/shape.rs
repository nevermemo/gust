//! The shapes and the values of a batch: the rank markers [`D1`] and [`D2`], what can be a shape ([`IntoShape`]), and the
//! parameter types ([`ParamType`]).

use core::marker::PhantomData;

use super::Param;

mod sealed {
    pub trait Sealed {}
}

/// The rank of a stage: 1-D ([`D1`]) or 2-D ([`D2`]). Sealed.
pub trait Rank: sealed::Sealed + 'static {}

/// The rank of a 1-D stage (a flat view, indexed `v[i]`).
#[derive(Clone, Copy, Debug)]
pub enum D1 {}

/// The rank of a 2-D stage (a row-major view, indexed `v[(x, y)]`).
#[derive(Clone, Copy, Debug)]
pub enum D2 {}

impl sealed::Sealed for D1 {}
impl sealed::Sealed for D2 {}

impl Rank for D1 {}

impl Rank for D2 {}

/// A value a [`Param`] holds, packed for the batch's table. Macro support.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Value {
    /// A `u32`.
    U32(u32),
    /// An `i32`.
    I32(i32),
    /// An `f32`.
    F32(f32),
    /// A `bool`.
    Bool(bool),
    /// A shape: width and height.
    Pair(u32, u32),
}

/// The types a [`Param`] can hold: `u32`, `i32`, `f32` and `bool` (read in a stage's body as `p.get()`), and `(u32, u32)`
/// (a shape: it can size a stage, and a body cannot read it).
pub trait ParamType: Copy + 'static {
    /// Pack the value for the batch's table. Macro support.
    #[doc(hidden)]
    fn pack(self) -> Value;

    /// Unpack a value that [`pack`](Self::pack) made. Macro support.
    #[doc(hidden)]
    fn unpack(value: Value) -> Self;
}

macro_rules! param_type {
    ($ty:ty, $variant:ident) => {
        impl ParamType for $ty {
            fn pack(self) -> Value {
                Value::$variant(self)
            }

            fn unpack(value: Value) -> Self {
                match value {
                    Value::$variant(inner) => inner,
                    other => unreachable!("a {} parameter holds {other:?}", stringify!($ty)),
                }
            }
        }
    };
}

param_type!(u32, U32);
param_type!(i32, I32);
param_type!(f32, F32);
param_type!(bool, Bool);

impl ParamType for (u32, u32) {
    fn pack(self) -> Value {
        Value::Pair(self.0, self.1)
    }

    fn unpack(value: Value) -> Self {
        match value {
            Value::Pair(width, height) => (width, height),
            other => unreachable!("a shape parameter holds {other:?}"),
        }
    }
}

/// Where a shape comes from. Macro support.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum ShapeSource {
    /// Written at the call: the extent along each axis (1 for an unused axis).
    Fixed([u32; 2]),
    /// The current value of a parameter of the batch with this identity and number.
    Param {
        /// The batch the parameter belongs to.
        batch: u64,
        /// The parameter's number in that batch.
        id: usize,
    },
}

/// What can give a stage or an input its shape: a `u32` (1-D), a `(u32, u32)` (2-D), or a reference to a [`Param`] of
/// either type. `Rank` is the rank the shape has; a 2-D stage takes a 2-D shape only.
pub trait IntoShape {
    /// The rank of this shape.
    type Rank: Rank;

    /// Where the extents come from. Macro support.
    #[doc(hidden)]
    fn source(&self) -> ShapeSource;
}

impl IntoShape for u32 {
    type Rank = D1;

    fn source(&self) -> ShapeSource {
        ShapeSource::Fixed([*self, 1])
    }
}

impl IntoShape for (u32, u32) {
    type Rank = D2;

    fn source(&self) -> ShapeSource {
        ShapeSource::Fixed([self.0, self.1])
    }
}

impl IntoShape for &Param<u32> {
    type Rank = D1;

    fn source(&self) -> ShapeSource {
        ShapeSource::Param {
            batch: self.batch(),
            id: self.id(),
        }
    }
}

impl IntoShape for &Param<(u32, u32)> {
    type Rank = D2;

    fn source(&self) -> ShapeSource {
        ShapeSource::Param {
            batch: self.batch(),
            id: self.id(),
        }
    }
}

/// The values of a batch's parameters as a frame sees them. Macro support.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Values(pub(crate) Vec<Value>);

impl Values {
    /// The value of the parameter numbered `id`.
    ///
    /// # Panics
    ///
    /// If `id` is not a parameter of the batch that made these values, or holds another type (the macro's own
    /// numbers and types are always right).
    #[must_use]
    pub fn get<T: ParamType>(&self, id: usize) -> T {
        T::unpack(self.0[id])
    }
}

/// A marker that keeps a handle's type parameters without owning them.
pub(crate) type Marker<T> = PhantomData<fn() -> T>;

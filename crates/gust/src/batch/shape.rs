//! The shapes and the values of a batch: the rank markers [`D1`] and [`D2`], what can be a shape ([`IntoShape`]), the
//! parameter types ([`ParamType`]) and the values derived from them ([`Derived`]).

use core::any::Any;
use core::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use super::{BatchDevice, Param, Stage};
use crate::{GpuEnum, GpuPod};

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

/// The types a [`Param`] can hold: `u32`, `i32`, `f32` and `bool` (read in a stage's body as `p.get()`), an enum with
/// `#[derive(GpuEnum)]` (read the same way, usually in a `match`), and `(u32, u32)` (a shape: it can size a stage, and a
/// body cannot read it).
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

/// An enum with `#[derive(GpuEnum)]` is a parameter: it is held as its `u32` word (D270), `set` takes the enum, and a
/// stage reads it as `p.get()` in a `match`.
impl<E: GpuEnum> ParamType for E {
    fn pack(self) -> Value {
        Value::U32(self.to_word())
    }

    fn unpack(value: Value) -> Self {
        match value {
            Value::U32(word) => {
                E::from_word(word).expect("an enum parameter holds a word that `pack` made")
            }
            other => unreachable!("an enum parameter holds {other:?}"),
        }
    }
}

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

/// How a function turns the value of a parameter into a derived value. Macro support.
#[doc(hidden)]
#[derive(Clone)]
pub struct Derive(Rc<dyn Fn(Value) -> Value>);

impl Derive {
    pub(crate) fn apply(&self, value: Value) -> Value {
        (self.0)(value)
    }

    pub(crate) fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl core::fmt::Debug for Derive {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Derive")
    }
}

/// Where a shape comes from. Macro support.
#[doc(hidden)]
#[derive(Clone, Debug)]
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
    /// A value computed on the CPU from a parameter, at each frame ([`Derived`]).
    Derived {
        /// The batch the parameter belongs to.
        batch: u64,
        /// The parameter's number in that batch.
        source: usize,
        /// The function that makes the value from the parameter's.
        derive: Derive,
    },
    /// The shape of a buffer of the batch (the extent of an in-place stage), whatever makes it.
    Buffer {
        /// The batch the buffer belongs to.
        batch: u64,
        /// The buffer's number in that batch.
        id: usize,
    },
}

/// Where the extents of a buffer come from once the batch is sealed: written at the call, or the current value of a
/// parameter (a derived value is a parameter of its own, made again whenever the one it comes from is set).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rule {
    Fixed([u32; 2]),
    Param(usize),
}

impl Rule {
    /// The extents the rule gives for these parameter values.
    pub(crate) fn dims(self, values: &[Value]) -> [u32; 2] {
        match self {
            Self::Fixed(dims) => dims,
            Self::Param(id) => match values[id] {
                Value::U32(n) => [n, 1],
                Value::Pair(width, height) => [width, height],
                other => unreachable!("a shape parameter holds {other:?}"),
            },
        }
    }
}

/// A value computed on the CPU from a parameter of the batch, at each frame: `p.count()` and `p.map(|v| -> T { .. })`.
/// A `u32` or a `(u32, u32)` of it can size a stage or an input (`gpu_task!(&grid, ..)`), and a `set` of the parameter it
/// comes from resizes what depends on it (see [`FrameRun::set`](super::FrameRun::set)). No stage or GPU pass computes it.
pub struct Derived<T> {
    batch: u64,
    source: usize,
    derive: Derive,
    marker: Marker<T>,
}

impl<T> Clone for Derived<T> {
    fn clone(&self) -> Self {
        Self {
            batch: self.batch,
            source: self.source,
            derive: self.derive.clone(),
            marker: PhantomData,
        }
    }
}

impl<T> core::fmt::Debug for Derived<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Derived")
            .field("batch", &self.batch)
            .field("source", &self.source)
            .finish()
    }
}

impl<T: ParamType> Param<T> {
    /// A value computed from this parameter's value on the CPU (before anything is encoded). `f` is a plain function of
    /// the value: it must give the same result for the same value, and it runs when the batch is sealed and at each
    /// `set` of this parameter. A shape it gives sizes a stage or an input like a parameter does.
    ///
    /// ```text
    /// let size = b.param::<(u32, u32)>((16, 8));
    /// let grid = size.map(|(w, h)| -> (u32, u32) { (w / 4, h / 4) });
    /// ```
    #[must_use]
    pub fn map<U: ParamType>(&self, f: impl Fn(T) -> U + 'static) -> Derived<U> {
        Derived {
            batch: self.batch(),
            source: self.id(),
            derive: Derive(Rc::new(move |value| f(T::unpack(value)).pack())),
            marker: PhantomData,
        }
    }
}

impl Param<(u32, u32)> {
    /// The number of elements of this shape (`width * height`, `u32::MAX` if that does not fit in a `u32`), as a 1-D
    /// shape that follows the parameter.
    #[must_use]
    pub fn count(&self) -> Derived<u32> {
        self.map(|(width, height)| width.saturating_mul(height))
    }
}

/// What can give a stage or an input its shape: a `u32` (1-D), a `(u32, u32)` (2-D), a reference to a [`Param`] or a
/// [`Derived`] of either type, or a reference to a handle of the batch (the shape of that buffer, for an in-place
/// stage). `Rank` is the rank the shape has; a 2-D stage takes a 2-D shape only.
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

impl IntoShape for &Derived<u32> {
    type Rank = D1;

    fn source(&self) -> ShapeSource {
        ShapeSource::Derived {
            batch: self.batch,
            source: self.source,
            derive: self.derive.clone(),
        }
    }
}

impl IntoShape for &Derived<(u32, u32)> {
    type Rank = D2;

    fn source(&self) -> ShapeSource {
        ShapeSource::Derived {
            batch: self.batch,
            source: self.source,
            derive: self.derive.clone(),
        }
    }
}

impl<T, R: Rank> IntoShape for &Stage<T, R> {
    type Rank = R;

    fn source(&self) -> ShapeSource {
        ShapeSource::Buffer {
            batch: self.batch,
            id: self.id,
        }
    }
}

/// The values of a batch's parameters and the extents of its buffers as a frame sees them. Macro support.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Values {
    pub(crate) params: Vec<Value>,
    pub(crate) dims: Vec<[u32; 2]>,
}

impl Values {
    /// The value of the parameter numbered `id`.
    ///
    /// # Panics
    ///
    /// If `id` is not a parameter of the batch that made these values, or holds another type (the macro's own
    /// numbers and types are always right).
    #[must_use]
    pub fn get<T: ParamType>(&self, id: usize) -> T {
        T::unpack(self.params[id])
    }

    /// The extents (width, height) of the buffer numbered `id`.
    ///
    /// # Panics
    ///
    /// If `id` is not a buffer of the batch that made these values.
    #[must_use]
    pub fn dims(&self, id: usize) -> [u32; 2] {
        self.dims[id]
    }
}

/// The buffers of a batch as a resized frame will have them, for a stage that is prepared again over them. Macro
/// support.
#[doc(hidden)]
pub struct Layout<'a> {
    pub(crate) buffers: &'a [Arc<dyn Any>],
    pub(crate) dims: &'a [[u32; 2]],
}

impl Layout<'_> {
    /// The extents (width, height) of the buffer numbered `id`.
    ///
    /// # Panics
    ///
    /// If `id` is not a buffer of the batch.
    #[must_use]
    pub fn dims(&self, id: usize) -> [u32; 2] {
        self.dims[id]
    }

    /// The buffer numbered `id`, as the stage sealed it.
    ///
    /// # Panics
    ///
    /// If `id` is not a buffer of the batch or `T` is not its element (the macro's own numbers and types are always
    /// right).
    #[must_use]
    pub fn buffer<D: BatchDevice, T: GpuPod>(&self, id: usize) -> Arc<D::Buffer<T>>
    where
        D::Buffer<T>: 'static,
    {
        self.buffers[id]
            .downcast_ref::<Arc<D::Buffer<T>>>()
            .expect("a stage's buffer is of the type it was sealed with")
            .clone()
    }
}

/// A marker that keeps a handle's type parameters without owning them.
pub(crate) type Marker<T> = PhantomData<fn() -> T>;

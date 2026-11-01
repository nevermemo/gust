use core::ops::{Deref, Index, IndexMut};

/// Slang-shaped scalar spellings for Rust-side parsing and type checking.
#[allow(non_camel_case_types)]
pub type float = f32;
#[allow(non_camel_case_types)]
pub type int = i32;
#[allow(non_camel_case_types)]
pub type uint = u32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UVec3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Rust shadow of Slang's `SV_DispatchThreadID` semantic value.
#[allow(non_camel_case_types)]
pub type SV_DispatchThreadID = UVec3;

/// GPU invocation metadata. The CPU twin (`gust::cpu`) constructs one per invocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Invocation {
    global_id: UVec3,
}

impl Invocation {
    pub const fn new(global_id: UVec3) -> Self {
        Self { global_id }
    }

    pub const fn global_id(self) -> UVec3 {
        self.global_id
    }
}

/// Rust shadow type for Slang's read-only structured buffer.
#[derive(Clone, Copy, Debug)]
pub struct Storage<'a, T> {
    values: &'a [T],
}

/// Slang-shaped spelling for a read-only structured buffer.
pub type StructuredBuffer<'a, T> = Storage<'a, T>;

impl<'a, T> Storage<'a, T> {
    pub const fn new(values: &'a [T]) -> Self {
        Self { values }
    }

    pub fn len(&self) -> u32 {
        u32::try_from(self.values.len()).expect("GPU buffers are limited to u32::MAX elements")
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl<T> Index<u32> for Storage<'_, T> {
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        &self.values[index as usize]
    }
}

/// Rust shadow type for Slang's read-write structured buffer.
#[derive(Debug)]
pub struct StorageMut<'a, T> {
    values: &'a mut [T],
}

/// Slang-shaped spelling for a read-write structured buffer.
pub type RWStructuredBuffer<'a, T> = StorageMut<'a, T>;

impl<'a, T> StorageMut<'a, T> {
    pub fn new(values: &'a mut [T]) -> Self {
        Self { values }
    }

    pub fn len(&self) -> u32 {
        u32::try_from(self.values.len()).expect("GPU buffers are limited to u32::MAX elements")
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl<T> Index<u32> for StorageMut<'_, T> {
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        &self.values[index as usize]
    }
}

impl<T> IndexMut<u32> for StorageMut<'_, T> {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self.values[index as usize]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct Uniform<T>(pub T);

impl<T> Deref for Uniform<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

mod atomic_scalar {
    /// The element types the atomic shadow functions accept: `u32` and `i32`. The trait is public only so that the
    /// bound of the public functions below is nameable; it sits in a private module, so nothing else can use it, and
    /// it is not part of the API.
    pub trait AtomicScalar: Copy {
        /// `target = target + operand` with wrap-around, as the Slang `Atomic<T>::add` method does (WGSL `atomicAdd`, 17.8.3.1).
        fn add(target: &mut Self, operand: Self);
        fn min(target: &mut Self, operand: Self);
        fn max(target: &mut Self, operand: Self);
        fn exchange(target: &mut Self, operand: Self);
        /// Stores `value` when `*target == compare` and returns whether it did.
        fn compare_exchange(target: &mut Self, compare: Self, value: Self) -> bool;
    }

    macro_rules! atomic_scalar {
        ($($ty:ty),*) => {$(
            impl AtomicScalar for $ty {
                fn add(target: &mut Self, operand: Self) {
                    *target = target.wrapping_add(operand);
                }
                fn min(target: &mut Self, operand: Self) {
                    *target = (*target).min(operand);
                }
                fn max(target: &mut Self, operand: Self) {
                    *target = (*target).max(operand);
                }
                fn exchange(target: &mut Self, operand: Self) {
                    *target = operand;
                }
                fn compare_exchange(target: &mut Self, compare: Self, value: Self) -> bool {
                    if *target == compare {
                        *target = value;
                        true
                    } else {
                        false
                    }
                }
            }
        )*};
    }

    atomic_scalar!(u32, i32);
}

use atomic_scalar::AtomicScalar;

/// Shadow of the Slang `Atomic<T>` methods (`add`, `min`, `max`, `exchange`, `compareExchange`; WGSL `atomic*`),
/// executed only by CPU twins.
///
/// A twin runs one invocation at a time (`gust::cpu`), so a plain read-modify-write is atomic. On the GPU the
/// same call lowers to the matching `Atomic<T>` method on a buffer element. The operation returns the old value in Rust; the GPU
/// form is statement-only, so the old value is not observable in a kernel. Only `u32` and `i32` elements are
/// accepted (a float receiver is a rustc error here, as the validator already makes it a macro error), and the
/// validator also rejects `i32` for min and max. Add wraps modulo 2^32 on both targets.
pub fn atomic_add<T: AtomicScalar>(target: &mut T, operand: T) -> T {
    let old = *target;
    T::add(target, operand);
    old
}

/// `target = min(target, operand)`; see [`atomic_add`]. Returns the old value.
pub fn atomic_min<T: AtomicScalar>(target: &mut T, operand: T) -> T {
    let old = *target;
    T::min(target, operand);
    old
}

/// `target = max(target, operand)`; see [`atomic_add`]. Returns the old value.
pub fn atomic_max<T: AtomicScalar>(target: &mut T, operand: T) -> T {
    let old = *target;
    T::max(target, operand);
    old
}

/// `target = operand`; see [`atomic_add`]. Returns the old value.
pub fn atomic_exchange<T: AtomicScalar>(target: &mut T, operand: T) -> T {
    let old = *target;
    T::exchange(target, operand);
    old
}

/// Stores `value` when `*target == compare` and returns whether it did; see [`atomic_add`].
///
/// The GPU form is WGSL's `atomicCompareExchangeWeak`, which may fail spuriously (17.8.5); the twin never does.
pub fn atomic_compare_exchange<T: AtomicScalar>(target: &mut T, compare: T, value: T) -> bool {
    T::compare_exchange(target, compare, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_add_wraps_at_the_type_limits_and_returns_the_old_value() {
        let mut unsigned = u32::MAX;
        assert_eq!(atomic_add(&mut unsigned, 1), u32::MAX);
        assert_eq!(unsigned, 0);
        let mut unsigned = 5u32;
        assert_eq!(atomic_add(&mut unsigned, u32::MAX), 5);
        assert_eq!(unsigned, 4);
        let mut signed = i32::MAX;
        assert_eq!(atomic_add(&mut signed, 1), i32::MAX);
        assert_eq!(signed, i32::MIN);
        let mut signed = -7i32;
        assert_eq!(atomic_add(&mut signed, 3), -7);
        assert_eq!(signed, -4);
    }

    #[test]
    fn atomic_min_and_max_keep_the_extreme_and_return_the_old_value() {
        let mut cell = 1000u32;
        assert_eq!(atomic_min(&mut cell, 3), 1000);
        assert_eq!(cell, 3);
        assert_eq!(atomic_min(&mut cell, 9), 3);
        assert_eq!(cell, 3);
        assert_eq!(atomic_max(&mut cell, 9), 3);
        assert_eq!(cell, 9);
        assert_eq!(atomic_max(&mut cell, 2), 9);
        assert_eq!(cell, 9);
        // Unsigned order: `u32::MAX` is the largest value, not -1.
        assert_eq!(atomic_max(&mut cell, u32::MAX), 9);
        assert_eq!(cell, u32::MAX);
    }

    #[test]
    fn atomic_exchange_stores_and_returns_the_old_value() {
        let mut cell = 9999u32;
        assert_eq!(atomic_exchange(&mut cell, 4), 9999);
        assert_eq!(cell, 4);
        let mut signed = -7i32;
        assert_eq!(atomic_exchange(&mut signed, -5), -7);
        assert_eq!(signed, -5);
    }

    #[test]
    fn atomic_compare_exchange_swaps_only_when_equal() {
        let mut cell = 7u32;
        assert!(atomic_compare_exchange(&mut cell, 7, 11));
        assert_eq!(cell, 11);
        assert!(!atomic_compare_exchange(&mut cell, 7, 13));
        assert_eq!(cell, 11);
        let mut signed = -5i32;
        assert!(atomic_compare_exchange(&mut signed, -5, 11));
        assert_eq!(signed, 11);
        assert!(!atomic_compare_exchange(&mut signed, -5, 1));
        assert_eq!(signed, 11);
    }
}

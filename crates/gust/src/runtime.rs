use core::ops::{Deref, Index, IndexMut};

/// Slang-shaped scalar spellings for Rust-side parsing and type checking.
#[allow(non_camel_case_types)]
pub type float = f32;
/// Slang-shaped spelling of `i32`.
#[allow(non_camel_case_types)]
pub type int = i32;
/// Slang-shaped spelling of `u32`.
#[allow(non_camel_case_types)]
pub type uint = u32;

/// A three-component unsigned vector: the Rust shadow of the `uint3` that carries an invocation id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UVec3 {
    /// The first component.
    pub x: u32,
    /// The second component.
    pub y: u32,
    /// The third component.
    pub z: u32,
}

/// Rust shadow of Slang's `SV_DispatchThreadID` semantic value.
#[allow(non_camel_case_types)]
pub type SV_DispatchThreadID = UVec3;

/// The invocation's index inside its workgroup (slice m7.g1, plan 4.1): `id % workgroup_size` per dimension, where `id`
/// is the dispatch-thread id. The Slang lowering derives it from the dispatch-thread parameter and the kernel's
/// `workgroup_size`; the entry point has no second semantic parameter. `SV_GroupThreadID` is the Slang spelling.
pub type LocalId = UVec3;

/// The index of the invocation's workgroup: `id / workgroup_size` per dimension (see [`LocalId`]). `SV_GroupID` is the
/// Slang spelling.
pub type GroupId = UVec3;

/// Rust shadow of Slang's `SV_GroupThreadID` semantic value; the same as [`LocalId`].
#[allow(non_camel_case_types)]
pub type SV_GroupThreadID = UVec3;

/// Rust shadow of Slang's `SV_GroupID` semantic value; the same as [`GroupId`].
#[allow(non_camel_case_types)]
pub type SV_GroupID = UVec3;

/// Rust shadow of workgroup-shared memory (slice m7.g1, plan 4.1): the kernel parameter `Workgroup<[T; N]>` is a
/// module-scope `groupshared T name[N];` in Slang, shared by the invocations of one workgroup, and is not a resource:
/// it has no binding, no descriptor entry and no argument of the typed call.
///
/// Its contents start unspecified on the GPU, a read before the same workgroup writes the element is outside the
/// domain, and so is a race (`docs/SEMANTICS.md` section 5). A kernel with a `Workgroup` parameter or a
/// [`workgroup_barrier`] must carry `#[kernel(cpu = "unsupported")]` (GUST-E0717): the sequential CPU twin cannot
/// model it. This type exists so that the shadow body type-checks as ordinary Rust; indexing is by `u32`, as for
/// buffers.
#[derive(Clone, Copy, Debug)]
pub struct Workgroup<A> {
    values: A,
}

impl<A> Workgroup<A> {
    /// Wraps `values` as the contents of the workgroup memory.
    pub const fn new(values: A) -> Self {
        Self { values }
    }
}

impl<T, const N: usize> Index<u32> for Workgroup<[T; N]> {
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        &self.values[index as usize]
    }
}

impl<T, const N: usize> IndexMut<u32> for Workgroup<[T; N]> {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self.values[index as usize]
    }
}

/// Rust shadow of the workgroup barrier (Slang `GroupMemoryBarrierWithGroupSync()`, WGSL `workgroupBarrier()`, SPIR-V
/// `OpControlBarrier`): statement-only in kernels. Every invocation of the workgroup must reach it, which the validator
/// enforces by position (GUST-E0414 to E0416). It does nothing here: a kernel that calls it has no CPU twin.
pub fn workgroup_barrier() {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
}

/// The scalar types a subgroup operation takes: `u32`, `i32` and `f32` (M8, D140). Sealed.
pub trait SubgroupScalar: Copy + sealed::Sealed {
    /// The identity of the sum: what the first lane of an exclusive sum gets.
    const ADD_IDENTITY: Self;
    /// The identity of the product: what the first lane of an exclusive product gets.
    const MUL_IDENTITY: Self;
}

impl SubgroupScalar for u32 {
    const ADD_IDENTITY: Self = 0;
    const MUL_IDENTITY: Self = 1;
}

impl SubgroupScalar for i32 {
    const ADD_IDENTITY: Self = 0;
    const MUL_IDENTITY: Self = 1;
}

impl SubgroupScalar for f32 {
    const ADD_IDENTITY: Self = 0.0;
    const MUL_IDENTITY: Self = 1.0;
}

/// The integer types the bitwise subgroup operations take: `u32` and `i32` (M8, D140). Sealed.
pub trait SubgroupInteger: SubgroupScalar {}

impl SubgroupInteger for u32 {}
impl SubgroupInteger for i32 {}

// Rust shadows of the subgroup operations (M8, D140). A subgroup is the group of invocations the GPU runs in lockstep;
// the operations combine a value across the active invocations of the calling invocation's subgroup. Each shadow has
// the meaning of a subgroup of one invocation, so that a body type-checks as ordinary Rust and the CPU form of a
// kernel never runs one: a body that uses them is accepted only under `requires(subgroups)` (GUST-E0745), has no
// twin, and the portable body is the kernel's CPU form. The lane count is a run-time value, `subgroup_size()`.

/// The sum of `value` over the invocations of the subgroup (Slang `WaveActiveSum`, WGSL `subgroupAdd`). Integers wrap;
/// an `f32` sum has no defined order. The shadow is the sum over one invocation.
pub fn subgroup_add<T: SubgroupScalar>(value: T) -> T {
    value
}

/// The product of `value` over the invocations of the subgroup (`WaveActiveProduct`, `subgroupMul`).
pub fn subgroup_mul<T: SubgroupScalar>(value: T) -> T {
    value
}

/// The smallest `value` of the subgroup (`WaveActiveMin`, `subgroupMin`).
pub fn subgroup_min<T: SubgroupScalar>(value: T) -> T {
    value
}

/// The largest `value` of the subgroup (`WaveActiveMax`, `subgroupMax`).
pub fn subgroup_max<T: SubgroupScalar>(value: T) -> T {
    value
}

/// The bitwise AND of `value` over the subgroup (`WaveActiveBitAnd`, `subgroupAnd`).
pub fn subgroup_and<T: SubgroupInteger>(value: T) -> T {
    value
}

/// The bitwise OR of `value` over the subgroup (`WaveActiveBitOr`, `subgroupOr`).
pub fn subgroup_or<T: SubgroupInteger>(value: T) -> T {
    value
}

/// The bitwise XOR of `value` over the subgroup (`WaveActiveBitXor`, `subgroupXor`).
pub fn subgroup_xor<T: SubgroupInteger>(value: T) -> T {
    value
}

/// Whether `value` is true in every invocation of the subgroup (`WaveActiveAllTrue`, `subgroupAll`).
pub fn subgroup_all(value: bool) -> bool {
    value
}

/// Whether `value` is true in any invocation of the subgroup (`WaveActiveAnyTrue`, `subgroupAny`).
pub fn subgroup_any(value: bool) -> bool {
    value
}

/// A bit mask of the invocations of the subgroup whose `value` is true (`WaveActiveBallot`, `subgroupBallot`): bit
/// `k` of the 128 bits, word `k / 32` bit `k % 32`, is invocation `k`'s. A subgroup is at most 128 invocations wide.
pub fn subgroup_ballot(value: bool) -> crate::math::UVec4 {
    crate::math::UVec4::new(u32::from(value), 0, 0, 0)
}

/// The `value` of the first active invocation of the subgroup (`WaveReadLaneFirst`, `subgroupBroadcastFirst`). Ran on
/// the RTX 5090 and on lavapipe for `i32` and `f32`, which spike S3 did not cover (`u32` only); not run on the Radeon
/// 890M (`docs/SEMANTICS.md`, "Forms beyond S3's cells").
pub fn subgroup_broadcast_first<T: SubgroupScalar>(value: T) -> T {
    value
}

/// The `value` of invocation `lane` of the subgroup, for a `lane` that is the same in every invocation. A literal
/// `lane` is `WaveBroadcastLaneAt` (`subgroupBroadcast`); any other expression is `WaveReadLaneAt`, which is
/// `subgroupShuffle` (WGSL's broadcast takes a constant). The shadow is the subgroup of one invocation. S3 covered
/// `u32`. Beyond it, all four forms ran on the RTX 5090 and on lavapipe: `f32` and `i32` with a literal lane, and `i32`
/// and `f32` with a computed lane (`i32` with a literal lane and `f32` with a computed lane are the two cells m8.z
/// added to `tests/subgroup.rs`); none ran on the Radeon 890M (`docs/SEMANTICS.md`, "Forms beyond S3's cells").
pub fn subgroup_broadcast<T: SubgroupScalar>(value: T, lane: u32) -> T {
    let _ = lane;
    value
}

/// The `value` of invocation `lane` of the subgroup, where `lane` may differ per invocation (`WaveShuffle`,
/// `subgroupShuffle`). A computed `lane` lowers to `WaveShuffle`, where S3's cell was `WaveReadLaneAt`; that form,
/// and `i32` and `f32` values, ran on the RTX 5090 and on lavapipe, not on the Radeon 890M (`docs/SEMANTICS.md`,
/// "Forms beyond S3's cells").
pub fn subgroup_shuffle<T: SubgroupScalar>(value: T, lane: u32) -> T {
    let _ = lane;
    value
}

/// The sum of `value` over the invocations of the subgroup below this one (`WavePrefixSum`,
/// `subgroupExclusiveAdd`); the first invocation gets 0.
pub fn subgroup_exclusive_add<T: SubgroupScalar>(value: T) -> T {
    let _ = value;
    T::ADD_IDENTITY
}

/// The product of `value` over the invocations of the subgroup below this one (`WavePrefixProduct`,
/// `subgroupExclusiveMul`); the first invocation gets 1.
pub fn subgroup_exclusive_mul<T: SubgroupScalar>(value: T) -> T {
    let _ = value;
    T::MUL_IDENTITY
}

/// The sum of `value` over this invocation and the ones below it: the exclusive sum plus `value`, a derived
/// operation.
pub fn subgroup_inclusive_add<T: SubgroupScalar>(value: T) -> T {
    value
}

/// The product of `value` over this invocation and the ones below it: the exclusive product times `value`, a derived
/// operation.
pub fn subgroup_inclusive_mul<T: SubgroupScalar>(value: T) -> T {
    value
}

/// The number of invocations of the subgroup whose `value` is true: the popcount of the ballot, a derived operation.
pub fn subgroup_count(value: bool) -> u32 {
    u32::from(value)
}

/// Whether this is the first active invocation of the subgroup, a derived operation (`subgroup_broadcast_first` of
/// the lane equals the lane; the native `subgroupElect` is not in naga 30.0.1).
pub fn subgroup_elect() -> bool {
    true
}

/// The number of invocations of the subgroup (`WaveGetLaneCount`): 32, 64 or another power of two the adapter
/// reports, fixed for a dispatch and not for a device (SEMANTICS section 5). The shadow subgroup has one.
pub fn subgroup_size() -> u32 {
    1
}

/// This invocation's index in its subgroup, `0..subgroup_size()` (`WaveGetLaneIndex`). The shadow subgroup has lane 0.
pub fn subgroup_lane() -> u32 {
    0
}

/// GPU invocation metadata. The CPU twin (`gust::cpu`) constructs one per invocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Invocation {
    global_id: UVec3,
}

impl Invocation {
    /// The metadata of the invocation whose global id is `global_id`.
    pub const fn new(global_id: UVec3) -> Self {
        Self { global_id }
    }

    /// The invocation's global id, the dispatch-thread id.
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
    /// Wraps `values` as the read-only buffer that a kernel parameter has in the shadow body.
    pub const fn new(values: &'a [T]) -> Self {
        Self { values }
    }

    /// The number of elements. Panics if the slice holds more than `u32::MAX`, which a GPU buffer cannot.
    pub fn len(&self) -> u32 {
        u32::try_from(self.values.len()).expect("GPU buffers are limited to u32::MAX elements")
    }

    /// Whether the buffer holds no element.
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
    /// Wraps `values` as the read-write buffer that a kernel parameter has in the shadow body.
    pub fn new(values: &'a mut [T]) -> Self {
        Self { values }
    }

    /// The number of elements. Panics if the slice holds more than `u32::MAX`, which a GPU buffer cannot.
    pub fn len(&self) -> u32 {
        u32::try_from(self.values.len()).expect("GPU buffers are limited to u32::MAX elements")
    }

    /// Whether the buffer holds no element.
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

/// The read-only view of a uniform (constant buffer) that a kernel's `Uniform<T>` parameter has: `T` is a struct of the
/// `#[gpu]` module, read one field at a time (`params.scale`). `Deref` gives that on the CPU twin, which passes
/// `Uniform(copy)` of the bound value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct Uniform<T>(pub T);

/// Slang-shaped spelling for a uniform: `ConstantBuffer<T>` is `Uniform<T>`.
pub type ConstantBuffer<T> = Uniform<T>;

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
/// The GPU form is WGSL's `atomicCompareExchangeWeak`, which may fail spuriously (17.8.5); the twin never does. The
/// contract stays weak (D47). Whether the Slang of the current pin retries after such a failure is recorded in
/// `docs/SEMANTICS.md`, section 6.
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
    fn workgroup_memory_is_indexed_by_u32_and_written_through_index_mut() {
        let mut scratch = Workgroup::new([0.0f32; 4]);
        scratch[2u32] = 5.5;
        scratch[3u32] = scratch[2u32] + 1.0;
        assert_eq!(scratch[0u32], 0.0);
        assert_eq!(scratch[2u32], 5.5);
        assert_eq!(scratch[3u32], 6.5);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn workgroup_memory_panics_past_its_length_like_an_array() {
        let scratch = Workgroup::new([1u32; 4]);
        let _ = scratch[4u32];
    }

    #[test]
    fn workgroup_barrier_does_nothing_on_the_host() {
        workgroup_barrier();
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

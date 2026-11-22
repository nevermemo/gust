#[gust::gpu_module]
mod atomics {
    // T10: atomic operations on RWStructuredBuffer elements.
    //
    // `workgroup_size(1, 1, 1)` makes the dispatch produce exactly `element_count`
    // threads (one per workgroup), so a 257-element dispatch yields exactly 257
    // contending operations rather than a workgroup-rounded count.

    /// All threads increment the same shared counter. Final value = thread count.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run(
        _id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<uint>,
    ) {
        atomic_add(&mut counter[0], 1u32);
    }

    /// Each thread attempts to lower the counter to its thread ID.
    /// Final value = minimum thread ID (0).
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_min(
        id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<uint>,
    ) {
        atomic_min(&mut counter[0], id.x);
    }

    /// Each thread attempts to raise the counter to its thread ID.
    /// Final value = maximum thread ID (256 for 257 threads).
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_max(
        id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<uint>,
    ) {
        atomic_max(&mut counter[0], id.x);
    }

    /// Each thread exchanges the counter with its thread ID.
    /// Final value is non-deterministic (last writer wins), range [0, 256].
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_exchange(
        id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<uint>,
    ) {
        atomic_exchange(&mut counter[0], id.x);
    }

    /// Each thread attempts CAS: if counter == 0, set to thread ID.
    /// Exactly one thread succeeds; final value is that thread's ID.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_cas(
        id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<uint>,
    ) {
        atomic_compare_exchange(&mut counter[0], 999u32, id.x + 1u32);
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_cas_success(
        _id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<uint>,
    ) {
        atomic_compare_exchange(&mut counter[0], 7u32, 11u32);
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_cas_failure(
        _id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<uint>,
    ) {
        atomic_compare_exchange(&mut counter[0], 0u32, 11u32);
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_signed_add(
        _id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<int>,
    ) {
        atomic_add(&mut counter[0], 3i32);
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_signed_exchange(
        _id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<int>,
    ) {
        atomic_exchange(&mut counter[0], -5i32);
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run_signed_cas(
        _id: SV_DispatchThreadID,
        mut counter: RWStructuredBuffer<int>,
    ) {
        atomic_compare_exchange(&mut counter[0], -5i32, 11i32);
    }
}

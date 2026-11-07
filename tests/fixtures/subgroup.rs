#[gust::gpu]
mod subgroup {
    // Subgroup operations (M8, m8.s, D140). Every kernel here is a pair: the kernel is the portable body, which every
    // profile has, and `<name>_subgroups` is the body a build with the `subgroups` capability gets instead
    // (`GUST_PROFILE=dev+subgroups`). The subgroup operations are accepted in the second only, so under `dev` the
    // portable body is the kernel and the subgroup body is still validated and lowered but not selected.
    //
    // The discipline the validator enforces (GUST-E0745, E0747, E0751, and E0415/E0416 as for a barrier): a subgroup
    // operation is the whole value of a `let` (or an assignment) at the top level of the body, never under an `if`, an
    // index or a `&&`; the body has no `return`; the invocation count is a multiple of 64. A bounds guard is a statement
    // `if` around the loads and the stores, so that every invocation of a subgroup reaches every operation: an
    // invocation past the end of the data holds 0 and still takes part.
    //
    // The subgroup size is a run-time value (32 on the RTX 5090, 64 on the Radeon 890M, 8 on lavapipe), so the kernels
    // that exercise the operations write the size and the lane they ran with beside the results, and a test rebuilds
    // the subgroups from them. Every result word of `lanes_*` and `votes` is at `id.x * 16 + word`.

    // The contract of `partial_sums` is the sum of `out`: the sum of the data, whatever the subgroup size. The portable
    // body copies the data, the subgroup body writes each subgroup's total from its elected lane and 0 from the others.
    // It is the same-signature three-way kernel of the plan: portable on the GPU, portable on the CPU (the twin and the
    // `cpu` target), subgroup on the GPU.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn partial_sums(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        if id.x < data.len() {
            out[id.x] = data[id.x];
        }
    }

    #[variant(of = partial_sums, requires(subgroups))]
    pub fn partial_sums_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let total = subgroup_add(x);
        let first = subgroup_elect();
        if id.x < data.len() {
            if first {
                out[id.x] = total;
            } else {
                out[id.x] = 0u32;
            }
        }
    }

    // The contract of `count_if` is the sum of `out`: the number of elements over 10. The portable body writes 1 for an
    // element over 10 and 0 for the others, the subgroup body writes the count of each subgroup (the popcount of the
    // ballot) from its elected lane.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn count_if(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        if id.x < data.len() {
            if data[id.x] > 10u32 {
                out[id.x] = 1u32;
            } else {
                out[id.x] = 0u32;
            }
        }
    }

    #[variant(of = count_if, requires(subgroups))]
    pub fn count_if_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let over = x > 10u32;
        let count = subgroup_count(over);
        let first = subgroup_elect();
        if id.x < data.len() {
            if first {
                out[id.x] = count;
            } else {
                out[id.x] = 0u32;
            }
        }
    }

    // Words 0 to 15 of an invocation, `u32`: sum, product, min, max, and, or, xor, exclusive sum, exclusive product,
    // inclusive sum, inclusive product, the value of the first lane, the value of lane 2 (a literal lane: a broadcast),
    // the value of the neighbour `lane ^ 1` (a shuffle), then the subgroup size and the lane. The portable body writes
    // zeros.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn lanes_u32(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        if id.x < data.len() {
            for word in 0u32..16u32 {
                out[id.x * 16u32 + word] = 0u32;
            }
        }
    }

    #[variant(of = lanes_u32, requires(subgroups))]
    pub fn lanes_u32_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let lane = subgroup_lane();
        let size = subgroup_size();
        let sum = subgroup_add(x);
        let product = subgroup_mul(x);
        let least = subgroup_min(x);
        let most = subgroup_max(x);
        let bits_and = subgroup_and(x);
        let bits_or = subgroup_or(x);
        let bits_xor = subgroup_xor(x);
        let before_sum = subgroup_exclusive_add(x);
        let before_product = subgroup_exclusive_mul(x);
        let through_sum = subgroup_inclusive_add(x);
        let through_product = subgroup_inclusive_mul(x);
        let first = subgroup_broadcast_first(x);
        let second_lane = subgroup_broadcast(x, 2u32);
        let neighbour = subgroup_shuffle(x, lane ^ 1u32);
        if id.x < data.len() {
            let base = id.x * 16u32;
            out[base] = sum;
            out[base + 1u32] = product;
            out[base + 2u32] = least;
            out[base + 3u32] = most;
            out[base + 4u32] = bits_and;
            out[base + 5u32] = bits_or;
            out[base + 6u32] = bits_xor;
            out[base + 7u32] = before_sum;
            out[base + 8u32] = before_product;
            out[base + 9u32] = through_sum;
            out[base + 10u32] = through_product;
            out[base + 11u32] = first;
            out[base + 12u32] = second_lane;
            out[base + 13u32] = neighbour;
            out[base + 14u32] = size;
            out[base + 15u32] = lane;
        }
    }

    // The same words over `i32` (the bitwise reductions go through `u32`, S3), except words 12 and 13: the value of the
    // lane `size / 2` (a broadcast of a lane that is not a literal, which lowers to a shuffle) and of lane
    // `(lane * 5 + 3) % size` (a shuffle by a computed lane). The size and the lane are written as `i32`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn lanes_i32(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<i32>,
        mut out: RWStructuredBuffer<i32>,
    ) {
        if id.x < data.len() {
            for word in 0u32..16u32 {
                out[id.x * 16u32 + word] = 0i32;
            }
        }
    }

    #[variant(of = lanes_i32, requires(subgroups))]
    pub fn lanes_i32_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<i32>,
        mut out: RWStructuredBuffer<i32>,
    ) {
        let mut x = 0i32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let lane = subgroup_lane();
        let size = subgroup_size();
        let sum = subgroup_add(x);
        let product = subgroup_mul(x);
        let least = subgroup_min(x);
        let most = subgroup_max(x);
        let bits_and = subgroup_and(x);
        let bits_or = subgroup_or(x);
        let bits_xor = subgroup_xor(x);
        let before_sum = subgroup_exclusive_add(x);
        let before_product = subgroup_exclusive_mul(x);
        let through_sum = subgroup_inclusive_add(x);
        let through_product = subgroup_inclusive_mul(x);
        let first = subgroup_broadcast_first(x);
        let middle = subgroup_broadcast(x, size / 2u32);
        let scattered = subgroup_shuffle(x, (lane * 5u32 + 3u32) % size);
        if id.x < data.len() {
            let base = id.x * 16u32;
            out[base] = sum;
            out[base + 1u32] = product;
            out[base + 2u32] = least;
            out[base + 3u32] = most;
            out[base + 4u32] = bits_and;
            out[base + 5u32] = bits_or;
            out[base + 6u32] = bits_xor;
            out[base + 7u32] = before_sum;
            out[base + 8u32] = before_product;
            out[base + 9u32] = through_sum;
            out[base + 10u32] = through_product;
            out[base + 11u32] = first;
            out[base + 12u32] = middle;
            out[base + 13u32] = scattered;
            out[base + 14u32] = size as i32;
            out[base + 15u32] = lane as i32;
        }
    }

    // The `f32` words: 0 sum, 1 product, 2 min, 3 max, 4 and 5 the exclusive sum and product, 6 and 7 the inclusive sum
    // and product, 8 the first lane's value, 9 lane 2's value, 10 the neighbour `lane ^ 2`, 11 the size and 12 the lane
    // (both as `f32`); words 13 to 15 are not written. The portable body writes zeros.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn lanes_f32(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        if id.x < data.len() {
            for word in 0u32..16u32 {
                out[id.x * 16u32 + word] = 0.0f32;
            }
        }
    }

    #[variant(of = lanes_f32, requires(subgroups))]
    pub fn lanes_f32_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let mut x = 0.0f32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let lane = subgroup_lane();
        let size = subgroup_size();
        let sum = subgroup_add(x);
        let product = subgroup_mul(x);
        let least = subgroup_min(x);
        let most = subgroup_max(x);
        let before_sum = subgroup_exclusive_add(x);
        let before_product = subgroup_exclusive_mul(x);
        let through_sum = subgroup_inclusive_add(x);
        let through_product = subgroup_inclusive_mul(x);
        let first = subgroup_broadcast_first(x);
        let second_lane = subgroup_broadcast(x, 2u32);
        let neighbour = subgroup_shuffle(x, lane ^ 2u32);
        if id.x < data.len() {
            let base = id.x * 16u32;
            out[base] = sum;
            out[base + 1u32] = product;
            out[base + 2u32] = least;
            out[base + 3u32] = most;
            out[base + 4u32] = before_sum;
            out[base + 5u32] = before_product;
            out[base + 6u32] = through_sum;
            out[base + 7u32] = through_product;
            out[base + 8u32] = first;
            out[base + 9u32] = second_lane;
            out[base + 10u32] = neighbour;
            out[base + 11u32] = size as f32;
            out[base + 12u32] = lane as f32;
        }
    }

    // The votes over the predicate `data[i] > 10`: word 0 `all`, 1 `any`, 2 to 5 the four words of the ballot, 6 the
    // count, 7 `elect`, 8 the size, 9 the lane. The portable body writes zeros.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn votes(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        if id.x < data.len() {
            for word in 0u32..16u32 {
                out[id.x * 16u32 + word] = 0u32;
            }
        }
    }

    #[variant(of = votes, requires(subgroups))]
    pub fn votes_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let over = x > 10u32;
        let every = subgroup_all(over);
        let some = subgroup_any(over);
        let ballot = subgroup_ballot(over);
        let count = subgroup_count(over);
        let first = subgroup_elect();
        let size = subgroup_size();
        let lane = subgroup_lane();
        if id.x < data.len() {
            let base = id.x * 16u32;
            out[base] = 0u32;
            if every {
                out[base] = 1u32;
            }
            out[base + 1u32] = 0u32;
            if some {
                out[base + 1u32] = 1u32;
            }
            out[base + 2u32] = ballot.x;
            out[base + 3u32] = ballot.y;
            out[base + 4u32] = ballot.z;
            out[base + 5u32] = ballot.w;
            out[base + 6u32] = count;
            out[base + 7u32] = 0u32;
            if first {
                out[base + 7u32] = 1u32;
            }
            out[base + 8u32] = size;
            out[base + 9u32] = lane;
        }
    }
}

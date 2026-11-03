#[gust::gpu]
mod scan {
    // m7.r2: the kernels of `gust_algo::scan` (plan section 5.3), `u32` only, plain `+` (inside the domain while no
    // inclusive prefix, the total included, overflows).
    //
    // The schedule cuts the input into chunks of 256 elements, one invocation per chunk, and recurses on the chunk
    // totals until at most 256 values are left:
    //
    //   scan_chunks_u32   level 0: invocation p scans chunk p of `input` into `out` (the inclusive prefix inside the
    //                     chunk) and writes the chunk total to `sums[p]`;
    //   scan_sums_u32     levels 1 and up: the same over the previous level's totals, in place;
    //   scan_top_u32      one invocation scans the last (at most 256) totals in place: now they are real prefixes;
    //   add_back_sums_u32 back down, one level per dispatch: `data[j] += next[j / 256 - 1]` for j >= 256;
    //   add_back_inclusive_u32 / add_back_exclusive_u32
    //                     the last step on `out`: invocation p adds the carry `sums[p - 1]` to chunk p. The exclusive
    //                     form also subtracts the element itself (`input`), which turns the inclusive prefix into
    //                     the exclusive one inside this same dispatch, with no extra pass.
    //
    // Every write is to the invocation's own element or its own chunk, so there is no race and no atomic. The add-back
    // over `out` is per chunk, not per element, so its invocation count stays at ceil(n / 256), inside the 1D ceiling
    // of the dispatch (an element per invocation would not be for the longest inputs).

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn scan_chunks_u32(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
        mut sums: RWStructuredBuffer<uint>,
    ) {
        let p = id.x;
        if p < sums.len() {
            let n = input.len();
            let first = p * 256u32;
            let mut acc = 0u32;
            for k in 0u32..256u32 {
                let index = first + k;
                if index >= n {
                    break;
                }
                acc += input[index];
                out[index] = acc;
            }
            sums[p] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn scan_sums_u32(
        id: SV_DispatchThreadID,
        mut data: RWStructuredBuffer<uint>,
        mut sums: RWStructuredBuffer<uint>,
    ) {
        let p = id.x;
        if p < sums.len() {
            let n = data.len();
            let first = p * 256u32;
            let mut acc = 0u32;
            for k in 0u32..256u32 {
                let index = first + k;
                if index >= n {
                    break;
                }
                acc += data[index];
                data[index] = acc;
            }
            sums[p] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn scan_top_u32(id: SV_DispatchThreadID, mut data: RWStructuredBuffer<uint>) {
        if id.x == 0u32 {
            let n = data.len();
            let mut acc = 0u32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                acc += data[index];
                data[index] = acc;
            }
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn add_back_sums_u32(
        id: SV_DispatchThreadID,
        next: StructuredBuffer<uint>,
        mut data: RWStructuredBuffer<uint>,
    ) {
        let j = id.x;
        if j < data.len() && j >= 256u32 {
            data[j] += next[j / 256u32 - 1u32];
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn add_back_inclusive_u32(
        id: SV_DispatchThreadID,
        sums: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let p = id.x;
        if p < sums.len() && p > 0u32 {
            let carry = sums[p - 1u32];
            let n = out.len();
            let first = p * 256u32;
            for k in 0u32..256u32 {
                let index = first + k;
                if index >= n {
                    break;
                }
                out[index] += carry;
            }
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn add_back_exclusive_u32(
        id: SV_DispatchThreadID,
        sums: StructuredBuffer<uint>,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let p = id.x;
        if p < sums.len() {
            let mut carry = 0u32;
            if p > 0u32 {
                carry = sums[p - 1u32];
            }
            let n = out.len();
            let first = p * 256u32;
            for k in 0u32..256u32 {
                let index = first + k;
                if index >= n {
                    break;
                }
                out[index] = out[index] + carry - input[index];
            }
        }
    }
}

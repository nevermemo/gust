#[gust::gpu_module]
mod loops {
    pub struct Window {
        pub lo: uint,
        pub hi: uint,
    }

    // Closed form: n * (n - 1) / 2 for the host reference.
    #[allow(clippy::assign_op_pattern)]
    fn sum_below(n: uint) -> uint {
        let mut total = 0u32;
        for i in 0u32..n {
            total = total + i;
        }
        total
    }

    // Signed bounds; zero is skipped with `continue`. Host: -3+-2+-1+1+2 = -3 for (-3, 3).
    #[allow(clippy::assign_op_pattern)]
    fn signed_sum_skip_zero(lo: int, hi: int) -> int {
        let mut total = 0i32;
        for k in lo..hi {
            if k == 0i32 {
                continue;
            }
            total = total + k;
        }
        total
    }

    // `break` leaves the loop; `return` leaves the helper from inside the body.
    fn first_multiple_at_least(factor: uint, minimum: uint) -> uint {
        for m in 1u32..1000u32 {
            let candidate = m * factor;
            if candidate >= minimum {
                return candidate;
            }
        }
        0u32
    }

    // `mut_range_bound` is allowed on purpose: the kernel proves the trip count is
    // fixed when the range is built, on the GPU as in Rust.
    #[allow(
        clippy::assign_op_pattern,
        clippy::explicit_counter_loop,
        clippy::mut_range_bound
    )]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        windows: StructuredBuffer<Window>,
        mut sums: RWStructuredBuffer<uint>,
        mut counts: RWStructuredBuffer<uint>,
        mut nested: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < input.len() {
            let x = input[i];
            // The end bound is evaluated once: shrinking `limit` in the body does not
            // shorten the loop (Rust `Range` semantics).
            let mut limit = x % 7u32 + 1u32;
            let mut trips = 0u32;
            for _ in 0u32..limit {
                limit = limit - 1u32;
                trips = trips + 1u32;
            }
            counts[i] = trips;

            // Window over the input buffer, guarded by `.len()` on every read.
            let w = windows[i];
            let mut acc = 0u32;
            for j in w.lo..w.hi {
                if j < input.len() {
                    acc = acc + input[j];
                } else {
                    break;
                }
            }
            sums[i] = acc + sum_below(x % 10u32) + first_multiple_at_least(3u32, x % 20u32);

            // Nested loops with break in the inner loop; outer continues.
            let mut grid = 0u32;
            for a in 0u32..4u32 {
                for b in a..4u32 {
                    if a * b > 6u32 {
                        break;
                    }
                    grid = grid + a * b + 1u32;
                }
            }
            let neg_sum = signed_sum_skip_zero(-3i32, x as int % 5i32);
            nested[i] = grid * 100u32 + (neg_sum + 10i32) as uint;
        }
    }
}

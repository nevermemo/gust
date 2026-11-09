// Two helpers that call each other (m9.a2, GUST-E0120): the cycle is reported at the call that closes it.

#[gust::gpu]
mod block {
    fn even(n: u32) -> bool {
        if n == 0u32 {
            return true;
        }
        odd(n - 1u32)
    }

    fn odd(n: u32) -> bool {
        if n == 0u32 {
            return false;
        }
        even(n - 1u32)
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if even(id.x) {
            out[id.x] = 1u32;
        }
    }
}

fn main() {}

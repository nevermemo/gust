// A type parameter with no trait of the module as a bound (m9.g2 round 2, GUST-E0125): `Copy` alone does not count. Such a
// `T` could be a number, a tuple or an array, and an unsuffixed literal would fix its Slang type differently from Rust's.

#[gust::gpu]
mod block {
    fn keep<T: Copy>(t: T) -> T {
        t
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = (keep(1) << 31u32) >> 31u32;
    }
}

fn main() {}

// Control case for arrays (m9.r1): literal, repeat, dynamic and literal index, index assignment, `.len()`, an array
// parameter and result of a helper. This crate must compile.

#[gust::gpu]
mod block {
    fn twice(a: [u32; 3]) -> [u32; 3] {
        [a[0] * 2u32, a[1] * 2u32, a[2] * 2u32]
    }


    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let mut a: [u32; 3] = [1u32, 2u32, 3u32];
        let z: [u32; 3] = [0u32; 3];
        a[(id.x % 3u32) as usize] += 5u32;
        let b = twice(a);
        out[id.x] = b[0] + b[2] + z[1] + (a.len() as u32);
    }
}

fn main() {}

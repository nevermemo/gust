#[gust::gpu]
mod sentinel {
    // Slice m9.sen (D168, D169): the NVIDIA array defect, as `#[gpu]` kernels. The shapes follow the investigations'
    // shrunk programs (arrbug/shrunk.slang, arrbug2/min.rs), so GUST's own emission is what reaches the driver.

    // D168's struct: its integer field holds 0, a value the compiler can prove unchanged.
    pub struct S0 {
        pub f0: u32,
        pub f1: i32,
    }

    // D168, N = 3: an array of one value, `v[k] -= 0` at a dynamic index, then element 0 read.
    // Rust: every out[i] is -3 - a[0].
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn d168_n3(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<i32>,
        mut out: RWStructuredBuffer<i32>,
    ) {
        let i = id.x;
        if i < out.len() {
            let s = S0 {
                f0: 15u32,
                f1: 0i32,
            };
            let rep = -3i32 - a[0u32];
            let mut v: [i32; 3] = [rep; 3];
            v[(i % 3u32) as usize] -= s.f1;
            out[i] = v[0u32 as usize];
        }
    }

    // D168, N = 5: the same program with five elements.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn d168_n5(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<i32>,
        mut out: RWStructuredBuffer<i32>,
    ) {
        let i = id.x;
        if i < out.len() {
            let s = S0 {
                f0: 15u32,
                f1: 0i32,
            };
            let rep = -3i32 - a[0u32];
            let mut v: [i32; 5] = [rep; 5];
            v[(i % 5u32) as usize] -= s.f1;
            out[i] = v[0u32 as usize];
        }
    }

    // D169, length 4: a dynamic store, a copy, then `^= 0` at a dynamic index of the copy.
    // Rust: out[i] is 2097152 where i % 4 == 3, else 2048.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn d169_len4(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let i = id.x;
        if i < out.len() {
            let x: u32 = 2048u32;
            let mut v46: [u32; 4] = [x, x, 1073742547u32, x];
            v46[(i % (v46.len() as u32)) as usize] <<= 10u32;
            let mut v48: [u32; 4] = v46;
            v48[(i % 4u32) as usize] ^= 0u32;
            out[i] = v48[3u32 as usize];
        }
    }
}

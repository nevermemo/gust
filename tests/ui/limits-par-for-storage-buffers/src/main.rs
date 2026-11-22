// A `par_for!` call in a crate with `limits = "portable"` is checked against the portable floor too (m11.floor): its
// kernel binds one storage buffer per slice binding, so these 9 are over the floor of 8 and GUST-E0746 points at the
// call. expected.txt pins the code and the line of the call.

fn fill(
    b0: &[u32],
    b1: &[u32],
    b2: &[u32],
    b3: &[u32],
    b4: &[u32],
    b5: &[u32],
    b6: &[u32],
    b7: &[u32],
    out: &mut [u32],
) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(i in 0..n, [b0: &[u32], b1: &[u32], b2: &[u32], b3: &[u32], b4: &[u32], b5: &[u32], b6: &[u32], b7: &[u32], out: &mut [u32]] {
        out[i] = b0[i] + b7[i];
    })
}

fn main() {}

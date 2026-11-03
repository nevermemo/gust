// A map function takes scalars, vectors and matrices by value and returns one: `f64` is none of them, so this is
// GUST-E0714 (docs/DIAGNOSTICS.md). The error points at the parameter's type; the function follows as written.

#[gust::gpu]
pub fn halve(x: f64) -> f32 {
    x as f32 * 0.5
}

fn main() {}

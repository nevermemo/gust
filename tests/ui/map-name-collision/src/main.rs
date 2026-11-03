// The module `#[gpu]` generates for a function holds `map`, `map_on`, `map_gpu`, .. and the function's own name, so a
// function called `map` collides with its own module's `map`: GUST-E0715. The error points at the name.

#[gust::gpu]
pub fn map(x: f32) -> f32 {
    x + 1.0
}

fn main() {}

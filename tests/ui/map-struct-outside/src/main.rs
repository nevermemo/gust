// A map function takes a struct only when the `#[gpu_module]` module it is written in declares it: a struct outside the
// module is GUST-E0714, because the macro cannot see its fields (docs/DIAGNOSTICS.md). The error points at the type.

#[derive(Clone, Copy)]
pub struct Outside {
    pub value: u32,
}

#[gust::gpu_module]
mod shaders {
    #[gust::gpu_fn]
    pub fn read(item: super::Outside) -> u32 {
        item.value
    }
}

fn main() {}

// `#[pad]` on a struct only a storage buffer reaches, GUST-E0219 (a storage gap is GUST-E0208, fixed by ordering).

#[gust::gpu]
mod block {
    #[pad]
    pub struct Item {
        pub a: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, items: StructuredBuffer<Item>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = items[id.x].a;
    }
}

fn main() {}

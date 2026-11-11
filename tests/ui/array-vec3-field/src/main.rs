// A stored array of a three-wide element: the stride (16) differs from the size (12), GUST-E0214.

#[gust::gpu]
mod block {
    pub struct Row {
        pub corners: [Vec3; 4],
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut rows: RWStructuredBuffer<Row>) {
        rows[id.x].corners[0] = Vec3::new(1.0f32, 2.0f32, 3.0f32);
    }
}

fn main() {}

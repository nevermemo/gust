#[gust::gpu]
pub mod pre_arrays {
    pub struct Bag {
        pub items: [u32; 4],
        pub count: u32,
    }

    pub struct Pt {
        pub x: u32,
        pub y: f32,
    }

    pub struct Poly {
        pub pts: [Pt; 2],
        pub tip: [Vec2; 2],
        pub n: u32,
        pub pad: u32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn element(id: SV_DispatchThreadID, mut quads: RWStructuredBuffer<[u32; 4]>) {
        let i = id.x;
        if i < quads.len() {
            let q = quads[i];
            quads[i][0] = q[3] + 100u32;
            quads[i][3] = q[0] + q[1] + q[2];
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn field(id: SV_DispatchThreadID, mut bags: RWStructuredBuffer<Bag>) {
        let i = id.x;
        if i < bags.len() {
            bags[i].items[1] += 7u32;
            bags[i].count = bags[i].items.len() as u32 + bags[i].items[0];
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn shapes(
        id: SV_DispatchThreadID,
        mut pairs: RWStructuredBuffer<[Vec2; 2]>,
        mut polys: RWStructuredBuffer<Poly>,
    ) {
        let i = id.x;
        if i < pairs.len() {
            let p = pairs[i];
            pairs[i][0] = p[1];
            pairs[i][1] = p[0];
            polys[i].pts[1].x = polys[i].pts[0].x + 1u32;
            polys[i].tip[0] = polys[i].tip[1];
            polys[i].n = polys[i].pts[1].x;
        }
    }
}

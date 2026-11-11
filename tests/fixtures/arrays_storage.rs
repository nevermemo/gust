#[gust::gpu]
mod arrays_storage {
    // Slice m9.r2a (D159): fixed-size arrays in structs and in storage. A struct field `a: [T; N]` and a buffer element
    // `RWStructuredBuffer<[T; N]>`, with `N` in the GpuPod set (1 to 32 and the powers of two from 64 to 4096) and `T` a
    // scalar, a 2- or 4-wide vector, `Mat2`, `Mat4` or a struct of the module. The Slang is `T a[N];` in the struct and
    // `RWStructuredBuffer<T[N]>`; the places are `els[i].a[j]` and `cells[i][j]`, whole arrays copy in and out.

    pub struct Bag {
        pub items: [u32; 4],
        pub gains: [f32; 2],
        pub count: u32,
        pub total: u32,
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

    // A local struct with an array field: it counts its calls and changes its own array.
    pub struct Ring {
        pub n: u32,
        pub ring: [u32; 4],
    }

    impl Ring {
        pub fn bump(&mut self) -> u32 {
            self.n += 1u32;
            self.ring[1] += 10u32;
            self.ring[1]
        }
    }

    pub struct Orders {
        pub a: u32,
        pub b: u32,
        pub c: u32,
        pub c_n: u32,
        pub d: u32,
        pub e: u32,
        pub f: u32,
        pub g: [u32; 4],
    }

    fn sum2(a: u32, b: u32) -> u32 {
        a * 1000u32 + b
    }

    // Whole arrays in and out of buffers, with `T[N]` as the buffer element.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn roundtrip(
        id: SV_DispatchThreadID,
        mut quads: RWStructuredBuffer<[u32; 4]>,
        mut floats: RWStructuredBuffer<[f32; 8]>,
        rows: StructuredBuffer<[u32; 4]>,
        mut picks: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < quads.len() {
            let q = quads[i];
            quads[i][0] = q[0] + 11u32;
            quads[i][3] = q[0] + q[1] + q[2] + q[3] + 1u32;
            for j in 0u32..8u32 {
                floats[i][j as usize] = floats[i][j as usize] * 2.0f32 + 1.0f32;
            }
            picks[i] = rows[i][(i % 4u32) as usize];
        }
    }

    // Array fields of storage elements.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn fields(
        id: SV_DispatchThreadID,
        mut bags: RWStructuredBuffer<Bag>,
        mut polys: RWStructuredBuffer<Poly>,
    ) {
        let i = id.x;
        if i < bags.len() {
            let t: [u32; 4] = bags[i].items;
            bags[i].items[0] += 8u32;
            bags[i].items[2] += 99u32;
            bags[i].items[3] = t[0];
            bags[i].gains[0] *= 2.0f32;
            bags[i].gains[1] = bags[i].gains[1] * 2.0f32 + 0.5f32;
            bags[i].count = bags[i].items.len() as u32;
            bags[i].total = t[0] + t[1] + t[2] + t[3];

            polys[i].pts[1].x += 7u32;
            polys[i].pts[0].y = polys[i].pts[0].y + polys[i].pts[1].y;
            let tip = polys[i].tip;
            polys[i].tip = [tip[1], tip[0]];
            polys[i].n = polys[i].n + polys[i].pts.len() as u32;
            polys[i].pad = polys[i].tip.len() as u32;
        }
    }

    // The wide elements: `Vec4`, `Mat4` and `Mat2`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn wide(
        id: SV_DispatchThreadID,
        mut vecs: RWStructuredBuffer<[Vec4; 2]>,
        mut mats: RWStructuredBuffer<[Mat4; 2]>,
        mut smalls: RWStructuredBuffer<[Mat2; 2]>,
    ) {
        let i = id.x;
        if i < vecs.len() {
            let t = vecs[i];
            vecs[i] = [t[1], t[0]];
            vecs[i][1].w = t[0].x + 10.0f32;
            let m = mats[i];
            mats[i] = [m[1], m[0]];
            let s = smalls[i];
            smalls[i][0] = s[1];
            smalls[i][1] = s[0];
        }
    }

    // Evaluation order on array places, in Rust's (every row gets the same values).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn orders(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Orders>) {
        let i = id.x;
        if i < out.len() {
            let mut ma = Ring { n: 0u32, ring: [1u32, 2u32, 3u32, 4u32] };
            ma.ring[1] += ma.bump();
            out[i].a = ma.ring[1] * 10u32 + ma.n;

            let mut mb = Ring { n: 0u32, ring: [1u32, 2u32, 3u32, 4u32] };
            mb.ring[1] = mb.ring[1] + mb.bump();
            out[i].b = mb.ring[1];

            let mut mc = Ring { n: 0u32, ring: [1u32, 2u32, 3u32, 4u32] };
            mc.ring = [mc.bump(), mc.bump(), mc.bump(), mc.bump()];
            out[i].c = mc.ring[0] + mc.ring[1] + mc.ring[2] + mc.ring[3];
            out[i].c_n = mc.n;

            let mut md = Ring { n: 0u32, ring: [1u32, 2u32, 3u32, 4u32] };
            let t: [u32; 4] = [md.ring[0], md.bump(), md.ring[1], md.bump()];
            out[i].d = t[0] * 1000000u32 + t[1] * 10000u32 + t[2] * 100u32 + t[3];

            let mut me = Ring { n: 0u32, ring: [1u32, 2u32, 3u32, 4u32] };
            out[i].e = sum2(me.ring[1], me.bump());

            let mut mf = Ring { n: 0u32, ring: [1u32, 2u32, 3u32, 4u32] };
            out[i].f = sum2(mf.bump(), mf.ring[1]);

            let mut mg = Ring { n: 0u32, ring: [1u32, 2u32, 3u32, 4u32] };
            out[i].g[2] = mg.bump();
        }
    }
}

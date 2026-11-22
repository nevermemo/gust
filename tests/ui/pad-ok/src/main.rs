// `#[pad]` (m14.pad, D239): GUST inserts the padding of the constant-buffer layout and generates `new`. A flat struct
// (a tail), a struct with a nested struct (a gap and a tail) and a struct with a gap before a vector. This crate must
// compile.

#[gust::gpu]
mod block {
    #[pad]
    pub struct Tail {
        pub a: f32,
    }

    #[pad]
    pub struct Flat {
        pub a: f32,
        pub v: Vec2,
    }

    pub struct Inner {
        pub x: f32,
        pub y: f32,
        pub z: f32,
        pub w: f32,
    }

    #[pad]
    pub struct Nested {
        pub a: f32,
        pub inner: Inner,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        t: Uniform<Tail>,
        f: Uniform<Flat>,
        n: Uniform<Nested>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        out[id.x] = t.a + f.a + f.v.x + n.a + n.inner.w;
    }
}

fn main() {
    let tail = block::Tail::new(1.0);
    let flat = block::Flat::new(1.0, gust::math::Vec2::new(2.0, 3.0));
    let inner = block::Inner {
        x: 1.0,
        y: 2.0,
        z: 3.0,
        w: 4.0,
    };
    let nested = block::Nested::new(1.0, inner);
    std::hint::black_box((tail, flat, nested));
}

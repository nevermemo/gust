// Explicit type arguments on a call of a generic helper (m9.g2, GUST-E0123): type arguments are inferred only.

#[gust::gpu]
mod block {
    pub trait Tag {
        fn tag(&self) -> u32;
    }

    pub struct Sq {
        pub side: u32,
    }

    impl Tag for Sq {
        fn tag(&self) -> u32 {
            self.side
        }
    }

    fn pick<T: Tag>(t: T) -> u32 {
        t.tag()
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let s = Sq { side: id.x };
        out[id.x] = pick::<Sq>(s);
    }
}

fn main() {}

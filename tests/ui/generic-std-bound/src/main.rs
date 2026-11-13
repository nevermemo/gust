// A std trait as a bound (m9.g2, GUST-E0125): only the traits of the module and `Copy` are bounds.

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

    fn largest<T: Tag + PartialOrd + core::ops::Add<Output = T>>(a: T, b: T) -> T {
        a
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}

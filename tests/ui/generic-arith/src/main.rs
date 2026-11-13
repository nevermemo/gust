// Arithmetic on a type parameter (m9.g2): a bound of the module has no operators, so rustc refuses the `+` on the CPU
// twin and the typecheck item; the error is rustc's and no Slang is built from it.

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

    fn twice<T: Tag + Copy>(t: T) -> T {
        t + t
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let s = Sq { side: id.x };
        out[id.x] = twice(s).tag();
    }
}

fn main() {}

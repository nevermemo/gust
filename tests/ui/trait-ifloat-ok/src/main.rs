// A user trait named like a Slang stdlib interface (m9.g1): `IFloat` is renamed in the generated Slang, so the crate
// compiles. This crate must compile.

#[gust::gpu]
mod halves {
    pub trait IFloat {
        fn half(&self) -> u32;
    }

    pub struct Num {
        pub v: u32,
    }

    impl IFloat for Num {
        fn half(&self) -> u32 {
            self.v / 2u32
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let n = Num { v: id.x };
        out[id.x] = n.half();
    }
}

fn main() {}

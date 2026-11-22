// A struct whose `Clone` and `Copy` the user wrote by hand: the validator refuses the trait impls (GUST-E0102,
// a trait that is not the module's own) and the refused module is handed back with them, so the recovery must not add
// a derive or an impl of its own for those traits (E0119). It still gives the struct its `GpuPod` stand-in, so the
// upload below adds no error. expected.txt pins the one error.

fn upload<T: gust::GpuPod>(_value: &T) {}

#[gust::gpu_module]
mod shaping {
    pub struct Params {
        pub low: u32,
        pub high: u32,
    }

    impl Clone for Params {
        fn clone(&self) -> Self {
            Params { low: self.low, high: self.high }
        }
    }

    impl Copy for Params {}

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {
    upload(&shaping::Params { low: 1, high: 5 });
}

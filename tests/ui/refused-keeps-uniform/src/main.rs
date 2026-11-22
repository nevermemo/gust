// A module the macro refuses still gets the items its expansion would have written, so the only error is the
// macro's own. Here `(a..=b).contains(&x)` is GUST-E0325 (a method that is not in the dialect); the host code that
// uploads the module's `Params` as a uniform (any call that needs `GpuPod`) must not add a rustc error saying `Params`
// is not a GPU POD type. expected.txt pins the one error.

fn upload<T: gust::GpuPod>(_value: &T) {}

#[gust::gpu]
mod shaping {
    pub struct Params {
        pub low: u32,
        pub high: u32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, params: Uniform<Params>, mut out: RWStructuredBuffer<u32>) {
        let i = id.x;
        if i < out.len() {
            if (params.low..=params.high).contains(&i) {
                out[i] = 1;
            }
        }
    }
}

fn main() {
    upload(&shaping::Params { low: 1, high: 5 });
}

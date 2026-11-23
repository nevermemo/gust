// Control case: the buffer element is a struct declared inside the module, which the macro
// makes a `GpuPod`. This must compile, so the diagnostic attribute cannot break valid code.

#[gust::gpu_module]
mod kernels {
    struct Item {
        value: u32,
    }

    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Item>) {
        let i = id.x;
        if i < out.len() {
            out[i] = Item { value: i };
        }
    }
}

fn main() {
    let descriptor = <kernels::run as gust::Kernel>::DESCRIPTOR;
    assert_eq!(descriptor.name, "run");
}

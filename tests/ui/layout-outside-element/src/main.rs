// A buffer element declared outside the `#[gust::gpu_module]` module whose hand-written `GpuPod` says it
// is one four-byte scalar while the Rust type is eight bytes. The descriptor records the claimed
// layout, so the const element assertion names the kernel parameter in one `E0080` at the
// attribute.

#[derive(Clone, Copy)]
struct Short {
    a: u32,
    b: u32,
}

// SAFETY (for the purpose of this fixture): deliberately wrong, that is what the test is about.
unsafe impl gust::GpuPod for Short {
    const LAYOUT: gust::TypeLayout = gust::TypeLayout::scalar("Short", gust::ScalarKind::U32);
}

#[gust::gpu_module]
mod kernels {
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<super::Short>) {
        let i = id.x;
        if i < out.len() {}
    }
}

fn main() {
    let short = Short { a: 0, b: 0 };
    let _ = (short.a, short.b);
}

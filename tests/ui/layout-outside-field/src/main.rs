// A struct field whose type is a hand-written `GpuPod` declared outside the `#[gust::gpu_module]` module,
// with alignment 8. The macro accepts the path and the generated impl is padding-free in Rust's
// terms, but `w` would sit at offset 8 after `a`, not at 4. The const layout assertions name the
// field in one `E0080` at the attribute (const panics carry no span of their own).

#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct Wide8 {
    value: u32,
}

// SAFETY (for the purpose of this fixture): deliberately wrong, that is what the test is about.
unsafe impl gust::GpuPod for Wide8 {
    const LAYOUT: gust::TypeLayout = gust::TypeLayout::scalar("Wide8", gust::ScalarKind::U32);
}

#[gust::gpu_module]
mod layouts {
    pub struct Holder {
        pub a: uint,
        pub w: super::Wide8,
    }
}

fn main() {
    let _ = Wide8 { value: 0 }.value;
}

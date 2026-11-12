// `Self` inside a tuple result of a trait method (m9.g1): rustc needs `Self: Sized` for it, and a trait of this dialect
// has no supertrait and no where clause to say so, so the form is rustc's E0277 on the module as written. The macro does
// not see it first; this fixture pins that the refusal is rustc's and not a slangc message.

#[gust::gpu]
mod block {
    pub trait Pack {
        fn split(self) -> (Self, u32);
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}

// Recursion through a generic helper and the impl of its bound (m9.g2, GUST-E0120): a call `t.m()` is an edge to every
// method named `m`, so every instantiation is covered.

#[gust::gpu]
mod block {
    pub trait Depth {
        fn depth(&self) -> u32;
    }

    pub struct Node {
        pub n: u32,
    }

    impl Depth for Node {
        fn depth(&self) -> u32 {
            let m = Node { n: self.n };
            deepen(m)
        }
    }

    fn deepen<T: Depth>(t: T) -> u32 {
        t.depth()
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}

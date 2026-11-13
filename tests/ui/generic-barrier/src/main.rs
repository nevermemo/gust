// A barrier reached only through the impl of a bound (m9.g2, D154, GUST-E0414): every impl is validated as a helper, so
// the barrier is refused where it is written, whichever instantiation reaches it.

#[gust::gpu]
mod block {
    pub trait Sync {
        fn sync(&self) -> u32;
    }

    pub struct Node {
        pub n: u32,
    }

    impl Sync for Node {
        fn sync(&self) -> u32 {
            workgroup_barrier();
            1u32
        }
    }

    fn through<T: Sync>(t: T) -> u32 {
        t.sync()
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let n = Node { n: id.x };
        out[id.x] = through(n);
    }
}

fn main() {}

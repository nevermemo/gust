// Control case for generics (m9.g2): a generic helper with two bounds and a `where` clause, a generic inherent method,
// `-> T`, a generic helper that calls another, a unit enum as the instantiation. This crate must compile.

#[gust::gpu]
mod shapes {
    pub trait Shape {
        fn area(&self) -> u32;
        fn doubled(self) -> Self;
    }

    pub trait Named {
        fn ident(&self) -> u32;
    }

    pub struct Sq {
        pub side: u32,
    }

    #[derive(PartialEq, Eq)]
    pub enum Kind {
        Small,
        Big,
    }

    impl Shape for Sq {
        fn area(&self) -> u32 {
            self.side * self.side
        }

        fn doubled(self) -> Sq {
            let s = Sq { side: self.side * 2u32 };
            s
        }
    }

    impl Named for Sq {
        fn ident(&self) -> u32 {
            self.side
        }
    }

    impl Named for Kind {
        fn ident(&self) -> u32 {
            match self {
                Kind::Small => {
                    return 1u32;
                }
                Kind::Big => {
                    return 2u32;
                }
            }
        }
    }

    pub struct Holder {
        pub k: u32,
    }

    impl Holder {
        pub fn fits<T: Shape>(&self, other: T) -> u32 {
            other.area() + self.k
        }
    }

    fn describe<T: Shape + Named>(t: T) -> u32 {
        t.area() * 10u32 + t.ident()
    }

    fn label<T>(t: T) -> u32
    where
        T: Named,
    {
        t.ident() + 1u32
    }

    fn quad<T: Shape>(t: T) -> T {
        t.doubled().doubled()
    }

    fn both<T: Shape + Named + Copy>(t: T) -> u32 {
        describe(t) + label(t)
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let s = Sq { side: id.x };
        let h = Holder { k: 1u32 };
        out[id.x] = both(s) + quad(s).area() + label(Kind::Big) + h.fits(s);
    }
}

fn main() {}

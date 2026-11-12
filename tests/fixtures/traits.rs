#[gust::gpu]
mod traits {
    // Slice m9.g1 (D170): traits as Slang interfaces, static dispatch only. A method has a receiver and no body.
    pub trait Shape {
        fn area(&self) -> u32;
        fn doubled(self) -> Self;
        fn grow(&mut self, by: u32);
    }

    pub trait Tagged {
        fn tag(&self) -> u32;
    }

    pub trait Plus {
        fn plus(self, k: u32) -> u32;
    }

    pub struct Rect {
        pub w: u32,
        pub h: u32,
    }

    pub struct Square {
        pub side: u32,
    }

    // A unit enum implements a trait too.
    #[derive(PartialEq, Eq)]
    pub enum Kind {
        Small,
        Big,
    }

    // Records the order of its method calls.
    pub struct Counter {
        pub n: u32,
    }

    pub struct Results {
        pub rect_area: u32,
        pub square_area: u32,
        pub rect_doubled: u32,
        pub square_doubled: u32,
        pub grown: u32,
        pub grown_square: u32,
        pub kind_small: u32,
        pub kind_big: u32,
        pub two_traits: u32,
        pub helper_call: u32,
        pub closure_call: u32,
        pub path_ref: u32,
        pub path_value: u32,
        pub path_qualified: u32,
        pub path_mut: u32,
        pub order_args: u32,
        pub order_self: u32,
        pub order_self_path: u32,
        pub method_helper: u32,
    }

    impl Rect {
        pub fn perimeter(&self) -> u32 {
            2u32 * (self.w + self.h)
        }
    }

    impl Shape for Rect {
        fn area(&self) -> u32 {
            self.w * self.h
        }

        #[allow(clippy::let_and_return)]
        fn doubled(self) -> Rect {
            let r = Rect {
                w: self.w * 2u32,
                h: self.h * 2u32,
            };
            r
        }

        fn grow(&mut self, by: u32) {
            self.w += by;
            self.h += by;
        }
    }

    impl Shape for Square {
        fn area(&self) -> u32 {
            self.side * self.side
        }

        #[allow(clippy::let_and_return)]
        fn doubled(self) -> Square {
            let r = Square {
                side: self.side * 2u32,
            };
            r
        }

        fn grow(&mut self, by: u32) {
            self.side += by;
        }
    }

    // A trait method that calls a helper and an inherent method.
    impl Tagged for Rect {
        fn tag(&self) -> u32 {
            bias(self.perimeter())
        }
    }

    impl Tagged for Square {
        fn tag(&self) -> u32 {
            bias(self.side)
        }
    }

    impl Tagged for Kind {
        #[allow(clippy::needless_return)]
        fn tag(&self) -> u32 {
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

    impl Counter {
        pub fn bump(&mut self) -> u32 {
            self.n += 1u32;
            self.n
        }
    }

    impl Plus for Counter {
        fn plus(self, k: u32) -> u32 {
            self.n * 100u32 + k
        }
    }

    fn bias(v: u32) -> u32 {
        v + 1000u32
    }

    // A trait call in a helper.
    fn area_of(r: Rect) -> u32 {
        r.area() + 1u32
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<u32>,
        mut results: RWStructuredBuffer<Results>,
    ) {
        let i = id.x;
        if i < input.len() {
            let x = input[i];
            let r = Rect { w: x, h: 3u32 };
            let s = Square { side: x + 1u32 };
            let rect_area = r.area();
            let square_area = s.area();
            let rect_doubled = r.doubled().area();
            let square_doubled = s.doubled().area();
            let mut g = Rect { w: x, h: 1u32 };
            g.grow(2u32);
            let mut gs = Square { side: x };
            gs.grow(5u32);

            let two_traits = r.area() + r.tag();
            let helper_call = area_of(r);
            let through = |q: Rect| -> u32 { q.area() + q.tag() };
            let cr = Rect { w: x, h: 2u32 };
            let closure_call = through(cr);

            // Each spelling of a path call is the method call on its first argument.
            let path_ref = Shape::area(&r);
            let pv = Rect { w: 1u32, h: x };
            let path_value = Shape::doubled(pv).area();
            let path_qualified = <Square as Shape>::area(&s);
            let mut pm = Rect { w: x, h: x };
            Shape::grow(&mut pm, 1u32);
            let path_mut = pm.area();

            // Arguments run left to right, and a by-value `self` is copied before the arguments run.
            let mut c = Counter { n: 0u32 };
            let order_args = pair(c.bump(), c.bump());
            let mut d = Counter { n: x };
            let order_self = d.plus(d.bump());
            let mut e = Counter { n: x };
            let order_self_path = Plus::plus(e, e.bump());

            results[i] = Results {
                rect_area,
                square_area,
                rect_doubled,
                square_doubled,
                grown: g.area(),
                grown_square: gs.area(),
                kind_small: Kind::Small.tag(),
                kind_big: Kind::Big.tag(),
                two_traits,
                helper_call,
                closure_call,
                path_ref,
                path_value,
                path_qualified,
                path_mut,
                order_args,
                order_self,
                order_self_path,
                method_helper: r.tag(),
            };
        }
    }

    fn pair(a: u32, b: u32) -> u32 {
        a * 10u32 + b
    }
}

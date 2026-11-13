#[gust::gpu]
mod generics {
    // Slice m9.g2 (D170): generic helpers and methods with trait bounds, static specialisation only. A type parameter is
    // bounded by the traits of the module (and `Copy`, which is dropped), is inferred from the arguments, and is typed
    // where it is written (a parameter, a result, `let x: T`).
    pub trait Shape {
        fn area(&self) -> u32;
        fn doubled(self) -> Self;
        fn grow(&mut self, by: u32);
    }

    pub trait Named {
        fn ident(&self) -> u32;
    }

    // Two arguments, to see the order they run in through a trait method on a `T`.
    pub trait Mix {
        fn mix(&self, a: u32, b: u32) -> u32;
    }

    pub trait Plus {
        fn plus(self, k: u32) -> u32;
    }

    pub trait Score {
        fn score(&self) -> u32;
    }

    pub struct Rect {
        pub w: u32,
        pub h: u32,
    }

    pub struct Square {
        pub side: u32,
    }

    // A struct with an array field is an instantiation too (spike m9.g0 P12).
    pub struct Bag {
        pub items: [u32; 4],
    }

    // A unit enum is one, with the trait it implements.
    #[derive(PartialEq, Eq)]
    pub enum Kind {
        Small,
        Big,
    }

    // Records the order of its method calls.
    pub struct Counter {
        pub n: u32,
    }

    pub struct Holder {
        pub k: u32,
    }

    pub struct Results {
        pub sum_rect: u32,
        pub sum_square: u32,
        pub desc_rect: u32,
        pub desc_square: u32,
        pub desc_bag: u32,
        pub label_small: u32,
        pub label_big: u32,
        pub label_rect: u32,
        pub quad_rect: u32,
        pub quad_square: u32,
        pub quad_bag: u32,
        pub grown_rect: u32,
        pub grown_square: u32,
        pub grown_bag: u32,
        pub chain_rect: u32,
        pub chain_square: u32,
        pub method_fits: u32,
        pub method_fits_bag: u32,
        pub method_weigh: u32,
        pub order_args: u32,
        pub order_self: u32,
        pub order_inner: u32,
        pub order_method: u32,
        pub via_closure: u32,
        pub via_trait: u32,
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

    impl Shape for Bag {
        fn area(&self) -> u32 {
            self.items[0] + self.items[1] + self.items[2] + self.items[3]
        }

        #[allow(clippy::let_and_return)]
        fn doubled(self) -> Bag {
            let b = Bag {
                items: [
                    self.items[0] * 2u32,
                    self.items[1] * 2u32,
                    self.items[2] * 2u32,
                    self.items[3] * 2u32,
                ],
            };
            b
        }

        fn grow(&mut self, by: u32) {
            self.items[0] += by;
            self.items[3] += by;
        }
    }

    impl Named for Rect {
        fn ident(&self) -> u32 {
            self.w * 100u32 + self.h
        }
    }

    impl Named for Square {
        fn ident(&self) -> u32 {
            7000u32 + self.side
        }
    }

    impl Named for Bag {
        fn ident(&self) -> u32 {
            self.items[0] * 10u32 + self.items[3]
        }
    }

    impl Named for Kind {
        #[allow(clippy::needless_return)]
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

    impl Mix for Square {
        fn mix(&self, a: u32, b: u32) -> u32 {
            self.side * 1000u32 + a * 10u32 + b
        }
    }

    impl Plus for Counter {
        fn plus(self, k: u32) -> u32 {
            self.n * 100u32 + k
        }
    }

    // A trait method that calls generic helpers, with a struct and with the enum.
    impl Score for Square {
        fn score(&self) -> u32 {
            let a = Square { side: self.side };
            let b = Square { side: 1u32 };
            sum_two(a, b) + label(Kind::Big)
        }
    }

    impl Counter {
        pub fn bump(&mut self) -> u32 {
            self.n += 1u32;
            self.n
        }

        // A generic `&mut self` method whose arguments of a trait call on the `T` mutate the receiver.
        pub fn run_with<T: Mix>(&mut self, t: T) -> u32 {
            t.mix(self.bump(), self.bump())
        }
    }

    impl Holder {
        // Generic inherent methods: one type parameter bounded by `Shape`, one by `Mix`.
        pub fn fits<T: Shape>(&self, other: T) -> u32 {
            other.area() * 10u32 + self.k
        }

        pub fn weigh<T: Mix>(&self, t: T, a: u32, b: u32) -> u32 {
            t.mix(a, b) + self.k
        }
    }

    // One helper, two instantiations or more.
    fn sum_two<T: Shape>(a: T, b: T) -> u32 {
        a.area() + b.area()
    }

    // Two bounds.
    fn describe<T: Shape + Named>(t: T) -> u32 {
        t.area() * 10u32 + t.ident()
    }

    // A `where` clause, and a unit enum as the instantiation.
    fn label<T>(t: T) -> u32
    where
        T: Named,
    {
        t.ident() + 1u32
    }

    // `T` returned.
    fn quad<T: Shape>(t: T) -> T {
        t.doubled().doubled()
    }

    // `&mut self` through `T`, on a copy bound by `let mut`.
    fn grow_twice<T: Shape>(t: T, by: u32) -> T {
        let mut c: T = t;
        c.grow(by);
        c.grow(by + 1u32);
        c
    }

    // A generic helper calls other generic helpers; `Copy` is dropped from the Slang bounds.
    fn chain<T: Shape + Named + Copy>(t: T) -> u32 {
        describe(t) + sum_two(t, t) + label(t)
    }

    fn gmix<T: Mix>(t: T, a: u32, b: u32) -> u32 {
        t.mix(a, b)
    }

    fn gplus<T: Plus>(t: T, k: u32) -> u32 {
        t.plus(k)
    }

    // Side effects in the arguments of a trait call on a `T`.
    fn gmix_local<T: Mix>(t: T) -> u32 {
        let mut c = Counter { n: 0u32 };
        t.mix(c.bump(), c.bump())
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
            let bag = Bag {
                items: [x, 1u32, 2u32, 3u32],
            };

            let other = Rect { w: 1u32, h: x };
            let sum_rect = sum_two(r, other);
            let two = Square { side: 2u32 };
            let sum_square = sum_two(s, two);
            let desc_rect = describe(r);
            let desc_square = describe(s);
            let desc_bag = describe(bag);
            let label_rect = label(r);
            let quad_rect = quad(r).area();
            let quad_square = quad(s).area();
            let quad_bag = quad(bag).area();
            let grown_rect = grow_twice(r, 2u32).area();
            let grown_square = grow_twice(s, 2u32).area();
            let grown_bag = grow_twice(bag, 2u32).area();
            let chain_rect = chain(r);
            let chain_square = chain(s);

            let h = Holder { k: 5u32 };
            let method_fits = h.fits(r);
            let method_fits_bag = h.fits(bag);
            // A generic method with side effects in its arguments: they run left to right.
            let mut c2 = Counter { n: x };
            let method_weigh = h.weigh(s, c2.bump(), c2.bump());

            // Arguments run left to right; a `T` passed by value is copied before a later argument mutates its source.
            let mut c = Counter { n: x };
            let order_args = gmix(s, c.bump(), c.bump());
            let mut d = Counter { n: x };
            let order_self = gplus(d, d.bump());
            let order_inner = gmix_local(s);
            let mut e = Counter { n: x };
            let order_method = e.run_with(s);

            // A generic call from a closure body.
            let through = |q: Rect| -> u32 { sum_two(q, q) + label(q) };
            let cr = Rect { w: x, h: 2u32 };
            let via_closure = through(cr);

            results[i] = Results {
                sum_rect,
                sum_square,
                desc_rect,
                desc_square,
                desc_bag,
                label_small: label(Kind::Small),
                label_big: label(Kind::Big),
                label_rect,
                quad_rect,
                quad_square,
                quad_bag,
                grown_rect,
                grown_square,
                grown_bag,
                chain_rect,
                chain_square,
                method_fits,
                method_fits_bag,
                method_weigh,
                order_args,
                order_self,
                order_inner,
                order_method,
                via_closure,
                via_trait: s.score(),
            };
        }
    }
}

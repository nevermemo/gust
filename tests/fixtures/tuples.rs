#[gust::gpu]
mod tuples {
    // Slice m9.t (D158): tuple values `(a, b)` and types `(A, B)` of two to eight elements, as locals, parameters, helper
    // and method results and method arguments; `t.0`; `let (x, y) = f();` and `let (x, y): (A, B) = ..;` (one
    // temporary, then one local per element); element assignment. The Slang is `Tuple<A, B>`, `makeTuple(a, b)`, `._0`.

    // A struct element of a tuple.
    pub struct Pt {
        pub a: u32,
        pub b: f32,
    }

    // An enum element of a tuple.
    #[derive(PartialEq, Eq)]
    pub enum Mode {
        Low = 1,
        Mid,
        High,
    }

    // Records the order of its method calls: `tick` counts, `pair` and `shift` take and give tuples.
    pub struct Trace {
        pub n: u32,
    }

    impl Trace {
        pub fn tick(&mut self) -> u32 {
            self.n += 1u32;
            self.n
        }

        // A tuple result of a `&self` method.
        pub fn pair(&self) -> (u32, u32) {
            (self.n, self.n + 1u32)
        }

        // A tuple argument of a `&mut self` method.
        pub fn shift(&mut self, t: (u32, u32)) -> u32 {
            self.n += t.0;
            self.n * 100u32 + t.1
        }
    }

    // One output buffer: the portable limit is eight storage buffers per shader stage.
    pub struct Results {
        pub ret_a: u32,
        pub ret_b: f32,
        pub second: f32,
        pub typed: u32,
        pub typed_f: f32,
        pub param: f32,
        pub strukt: u32,
        pub enum_cast: u32,
        pub eight: u32,
        pub eight_last: u32,
        pub assign_a: u32,
        pub assign_b: f32,
        pub nested: u32,
        pub nested_chain: u32,
        pub literal_bind: u32,
        pub value_match: u32,
        pub mixed: u32,
        pub vec_el: f32,
        pub order_a: u32,
        pub order_b: u32,
        pub order_c: u32,
        pub order_d: u32,
        pub order_e: u32,
        pub order_f: u32,
        pub order_g: u32,
        pub order_g_n: u32,
        pub order_h: u32,
        pub order_i: u32,
        pub order_n: u32,
    }

    fn classify(v: u32) -> (u32, f32) {
        (v * 2u32 + 1u32, v as f32 * 0.5f32)
    }

    // A tuple parameter.
    fn weigh(t: (u32, f32), k: f32) -> f32 {
        t.0 as f32 * k + t.1
    }

    fn with_pt(v: u32) -> (Pt, u32) {
        let p = Pt {
            a: v + 1u32,
            b: (v + 1u32) as f32,
        };
        (p, v + 1u32)
    }

    fn mode_from(v: u32) -> Mode {
        if v == 0u32 {
            return Mode::Low;
        }
        if v == 1u32 {
            return Mode::Mid;
        }
        Mode::High
    }

    fn mode_pair(v: u32) -> (Mode, u32) {
        (mode_from(v), v)
    }

    // Eight elements: the most there are.
    fn eight(v: u32) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
        (
            v,
            v + 1u32,
            v + 2u32,
            v + 3u32,
            v + 4u32,
            v + 5u32,
            v + 6u32,
            v + 7u32,
        )
    }

    // A tuple of tuples.
    fn nest(v: u32) -> ((u32, u32), u32) {
        ((v, v + 1u32), v + 2u32)
    }

    fn mix(v: u32) -> (f32, bool, i32) {
        (v as f32, v > 1u32, -(v as i32))
    }

    fn vecs(v: u32) -> (Vec2, u32) {
        (Vec2::new(v as f32, 1.5f32), v)
    }

    fn use_pair(t: (u32, u32)) -> u32 {
        t.0 * 10u32 + t.1
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<u32>,
        mut results: RWStructuredBuffer<Results>,
    ) {
        let i = id.x;
        if i < input.len() {
            let v = input[i];

            // A destructured call result.
            let (a, b) = classify(v);

            // `t.1` on a local.
            let t = classify(v);

            // An annotated destructuring.
            let (tc, td): (u32, f32) = classify(v + 1u32);

            // A struct element, then an annotated enum element.
            let (p, pn) = with_pt(v);
            let (m, w): (Mode, u32) = mode_pair(v);

            // Eight elements, destructured and indexed.
            let (e0, e1, e2, e3, e4, e5, e6, e7) = eight(v);
            let big = eight(v);

            // Element assignment and compound assignment.
            let mut u = classify(v);
            u.0 = u.0 * 2u32 + 99u32;
            u.1 += 1.0f32;

            // A tuple of tuples, destructured and chained.
            let (inner, z) = nest(v);
            let q = nest(v);

            // A tuple literal as the initializer.
            let (x, y) = (v + 1u32, v * 2u32);

            // A tuple as the result of a value `match`.
            let vm: (u32, u32) = match v {
                0 => (1u32, 2u32),
                1 | 2 => (3u32, 4u32),
                _ => (5u32, 6u32),
            };

            // bool and i32 elements.
            let (f, flag, s) = mix(v);
            let mut mixed = (f as u32) + ((s + 4i32) as u32);
            if flag {
                mixed += 100u32;
            }

            // A vector element.
            let vt = vecs(v);

            // Evaluation order, in Rust's: the elements left to right, each fully before the next. `c` starts at 5.
            let mut c = Trace { n: 5u32 };
            let o1 = (c.n, c.tick());
            let o2 = (c.tick(), c.n);
            let o3 = (c, c.tick());
            let o4 = (c.tick(), c);
            let (ox, oy) = (c.tick(), c.tick());
            let mut d = Trace { n: 0u32 };
            let o6 = use_pair((d.tick(), d.tick()));
            let mut sh = Trace { n: 10u32 };
            let mut e = Trace { n: 0u32 };
            let o7 = sh.shift((e.tick(), e.tick()));
            let o8 = c.pair().1 + c.tick();
            let o9 = c.tick() + c.pair().1;

            results[i] = Results {
                ret_a: a,
                ret_b: b,
                second: t.1 + 0.5f32,
                typed: tc + 2u32,
                typed_f: td,
                param: weigh(classify(v), 1.0f32),
                strukt: p.a + pn + (p.b as u32),
                enum_cast: (m as u32) * 10u32 + w,
                eight: e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7,
                eight_last: big.7,
                assign_a: u.0,
                assign_b: u.1,
                nested: inner.0 + inner.1 + z,
                nested_chain: q.0.1 + q.1,
                literal_bind: x * 10u32 + y,
                value_match: vm.0 * 10u32 + vm.1,
                mixed,
                vec_el: vt.0.x + vt.0.y,
                order_a: o1.0 * 100u32 + o1.1,
                order_b: o2.0 * 100u32 + o2.1,
                order_c: o3.0.n * 100u32 + o3.1,
                order_d: o4.0 * 100u32 + o4.1.n,
                order_e: ox * 100u32 + oy,
                order_f: o6,
                order_g: o7,
                order_g_n: sh.n,
                order_h: o8,
                order_i: o9,
                order_n: c.n,
            };
        }
    }
}

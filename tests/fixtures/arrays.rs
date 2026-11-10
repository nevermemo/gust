#[gust::gpu]
mod arrays {
    // Slice m9.r1 (D159): fixed-size arrays `[T; N]` as values: `N` a literal above zero, `T` a scalar, vector, matrix or
    // struct or enum of the module. Literals `[a, b]` and repeats `[e; N]`; indexing with `i as usize` (the only place
    // `usize` is accepted) and `a[2]`; index assignment; arrays as helper and method parameters and results; `.len()`.
    // The Slang is `T[N] a = { .. }`, `a[i]`, and a repeat is one temporary (`__gust_rep_<k>`) and then `{ t, t, .. }`.

    // A struct element of an array.
    pub struct Pt {
        pub a: u32,
        pub b: f32,
    }

    // An enum element of an array.
    #[derive(PartialEq, Eq)]
    pub enum Mode {
        Low = 1,
        Mid,
        High,
    }

    // Records the order of its method calls: `bump` counts and gives the new value.
    pub struct Counter {
        pub n: u32,
    }

    impl Counter {
        pub fn bump(&mut self) -> u32 {
            self.n += 1u32;
            self.n
        }

        pub fn get(&self) -> u32 {
            self.n
        }

        // An array parameter and an array result of methods.
        pub fn spread(&mut self, base: [u32; 3]) -> [u32; 3] {
            self.n += 1u32;
            [base[0] + self.n, base[1] + self.n, base[2] + self.n]
        }
    }

    // One output buffer: the portable limit is eight storage buffers per shader stage.
    pub struct Results {
        pub lit_sum: u32,
        pub rep_ends: u32,
        pub bump_once: u32,
        pub bump_n: u32,
        pub dyn_rw: u32,
        pub compound: u32,
        pub helper: f32,
        pub copy_check: f32,
        pub len_n: u32,
        pub struct_el: u32,
        pub fill17: u32,
        pub fill17_len: u32,
        pub assign_rep: u32,
        pub assign_fill: u32,
        pub tail_rep: u32,
        pub tail_lit: u32,
        pub enum_el: u32,
        pub vec_el: f32,
        pub method_arr: u32,
        pub method_n: u32,
        pub shadow: u32,
        pub order_a: u32,
        pub order_g: u32,
        pub order_f: u32,
        pub order_h: u32,
        pub order_i: u32,
        pub order_i_other: u32,
        pub order_j: u32,
        pub order_b: u32,
        pub order_pure: u32,
        pub order_pure_compound: u32,
    }

    fn dbl4(x: [f32; 4]) -> [f32; 4] {
        let mut r: [f32; 4] = x;
        for i in 0u32..4u32 {
            r[i as usize] = x[i as usize] * 2.0f32;
        }
        r
    }

    fn sum4(x: [f32; 4]) -> f32 {
        x[0] + x[1] + x[2] + x[3]
    }

    // The parameter is a copy: the caller's array does not change.
    fn bump_all(mut x: [f32; 4]) -> f32 {
        x[0] += 100.0f32;
        x[0]
    }

    // `.len()` of a parameter, inside an index.
    fn last(x: [f32; 4]) -> f32 {
        x[x.len() - 1]
    }

    // A repeat as the value of a function, and a literal as the value of another.
    fn repeated(v: u32) -> [u32; 3] {
        [v; 3]
    }

    fn make3(v: u32) -> [u32; 3] {
        [v, v + 1u32, v + 2u32]
    }

    fn pair_code(a: [u32; 2]) -> u32 {
        a[0] * 10u32 + a[1]
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

            // A literal, and a repeat read at its ends.
            let a: [u32; 4] = [v, v + 1u32, v + 2u32, v + 3u32];
            let r: [u32; 4] = [v + 1u32; 4];

            // A repeat evaluates its operand once: `bump` runs once, so the four elements are equal.
            let mut s = Counter { n: v };
            let b: [u32; 4] = [s.bump(); 4];

            // A dynamic index, read and written.
            let mut d: [u32; 5] = [0u32; 5];
            d[(v % 5u32) as usize] = 10u32 + v;
            d[((v + 1u32) % 5u32) as usize] = 7u32;

            // A compound index assignment, with a literal and a dynamic index.
            let mut c: [u32; 3] = [1u32, 2u32, 3u32];
            c[1] += v;
            c[(v % 3u32) as usize] += 10u32;

            // A helper with an array parameter and result; the argument is a copy.
            let f: [f32; 4] = [v as f32, 1.0f32, 2.0f32, 3.0f32];
            let g = dbl4(f);

            // `.len()` of a local and of a storage buffer side by side.
            let l: [u32; 6] = [1u32, 2u32, 3u32, 4u32, 5u32, 6u32];

            // A struct element.
            let p0 = Pt { a: v, b: 1.0f32 };
            let p1 = Pt { a: v + 1u32, b: 2.0f32 };
            let mut ps: [Pt; 2] = [p0, p1];
            ps[1].a += 5u32;
            ps[0].b = 2.5f32;

            // A 17-element repeat, which is filled by a loop, and its use.
            let big: [u32; 17] = [v + 3u32; 17];
            let mut total = 0u32;
            for k in 0u32..17u32 {
                total += big[k as usize];
            }

            // A repeat assigned to a local, short and long (the old value is read first, which the unused-assignment lint wants).
            let mut w: [u32; 4] = [1u32; 4];
            let w_first = w[0];
            w = [v; 4];
            let mut w2: [u32; 17] = [1u32; 17];
            let w2_first = w2[16];
            w2 = [v; 17];
            let mut w2_total = 0u32;
            for k in 0u32..17u32 {
                w2_total += w2[k as usize];
            }

            // An enum element and a vector element.
            let m: [Mode; 2] = [Mode::Low, Mode::High];
            let vs: [Vec2; 2] = [Vec2::new(v as f32, 1.0f32), Vec2::new(2.0f32, 3.0f32)];

            // A method with an array parameter and result, on a counter that starts at v.
            let mut sp = Counter { n: v };
            let spread = sp.spread([10u32, 20u32, 30u32]);

            // A rebinding that reads the array it replaces.
            let h: [u32; 2] = [v, v + 1u32];
            let mut shadow = 0u32;
            if v < 10u32 {
                let h: [u32; 2] = [h[1], h[0]];
                shadow = h[0] * 10u32 + h[1];
            }

            // Evaluation order, in Rust's. A literal of calls on one root, left to right.
            let mut s1 = Counter { n: 0u32 };
            let oa: [u32; 3] = [s1.bump(), s1.bump(), s1.bump()];
            // An array argument built from calls.
            let mut s2 = Counter { n: 0u32 };
            let og = pair_code([s2.bump(), s2.bump()]);
            // Two reads of one table, each with an index that calls.
            let tbl: [u32; 4] = [10u32, 20u32, 30u32, 40u32];
            let mut s3 = Counter { n: 0u32 };
            let of = tbl[s3.bump() as usize] * 100u32 + tbl[s3.bump() as usize];
            // An index on a call result.
            let mut s4 = Counter { n: 0u32 };
            let oh = make3(s4.bump())[1];
            // A method on an element, with an index that calls another counter.
            let mut s5 = Counter { n: 0u32 };
            let c0 = Counter { n: 100u32 };
            let c1 = Counter { n: 200u32 };
            let c2 = Counter { n: 300u32 };
            let mut cs: [Counter; 3] = [c0, c1, c2];
            let oi = cs[s5.bump() as usize].bump();
            // An element assigned from its neighbours.
            let mut j: [u32; 3] = [1u32, 2u32, 3u32];
            j[1] = j[0] + j[2];
            // The value first, then the index (Rust's order for `a[s.n] = s.bump()`), written as two statements.
            let mut s6 = Counter { n: 0u32 };
            let mut e6: [u32; 4] = [0u32; 4];
            let val = s6.bump();
            e6[s6.n as usize] = val;
            // A plain index and a calling value.
            let mut s7 = Counter { n: 4u32 };
            let k7: u32 = 2u32;
            let mut e7: [u32; 4] = [0u32; 4];
            e7[k7 as usize] = s7.bump();
            let first = e7[2];
            e7[k7 as usize] += s7.bump();

            results[i] = Results {
                lit_sum: a[0] + a[1] + a[2] + a[3],
                rep_ends: r[3] + r[0],
                bump_once: b[0] + b[1] + b[2] + b[3],
                bump_n: s.n,
                dyn_rw: d[v as usize] * 100u32 + d[((v + 1u32) % 5u32) as usize],
                compound: c[0] * 10000u32 + c[1] * 100u32 + c[2],
                helper: sum4(g) + f[0],
                copy_check: bump_all(f) + f[0] + last(f),
                len_n: (l.len() as u32) * 100u32 + l[l.len() - 1],
                struct_el: ps[0].a + ps[1].a + (ps[0].b as u32),
                fill17: total,
                fill17_len: big.len() as u32,
                assign_rep: w[0] + w[3] + w_first - 1u32,
                assign_fill: w2_total + w2_first - 1u32,
                tail_rep: repeated(v)[2],
                tail_lit: make3(v)[0] + make3(v)[2],
                enum_el: (m[1] as u32) * 10u32 + (m[0] as u32),
                vec_el: vs[0].x + vs[1].y,
                method_arr: spread[0] + spread[1] + spread[2],
                method_n: sp.n,
                shadow,
                order_a: oa[0] * 100u32 + oa[1] * 10u32 + oa[2],
                order_g: og,
                order_f: of,
                order_h: oh,
                order_i: oi,
                order_i_other: cs[0].n + cs[2].n,
                order_j: j[0] * 100u32 + j[1] * 10u32 + j[2],
                order_b: e6[1] * 10u32 + e6[0],
                order_pure: first * 100u32 + e7[0],
                order_pure_compound: e7[2],
            };
        }
    }
}

#[gust::gpu]
mod closures {
    pub struct Counter {
        pub n: u32,
        pub step: u32,
    }

    #[derive(PartialEq, Eq)]
    pub enum Mode {
        A,
        B,
    }

    // One output buffer: the portable limit is eight storage buffers per shader stage.
    pub struct Results {
        pub added: u32,
        pub poly: u32,
        pub scaled: u32,
        pub via_helper: u32,
        pub via_method: u32,
        pub looped: u32,
        pub float_kind: f32,
        pub int_kind: i32,
        pub bool_kind: u32,
        pub vector_kind: f32,
        pub matrix_kind: f32,
        pub struct_kind: u32,
        pub enum_kind: u32,
        pub tuple_kind: u32,
        pub array_kind: u32,
        pub order_two_ticks: u32,
        pub order_read_then_tick: u32,
        pub order_tick_then_read: u32,
        pub order_nested: u32,
        pub placed: u32,
    }

    impl Counter {
        // A `&mut self` method that gives a value, to see the order the arguments of a closure call run in.
        pub fn tick(&mut self) -> u32 {
            self.n += self.step;
            self.n
        }

        // A closure in a method body: `self` is not captured, it is passed.
        pub fn scaled(&self, k: u32) -> u32 {
            let mul = |a: u32, b: u32| -> u32 { a * b };
            mul(self.n, k)
        }
    }

    fn shift(v: u32) -> u32 {
        v + 100u32
    }

    // A closure in a helper body, called twice.
    fn poly(x: u32) -> u32 {
        let sq = |a: u32| -> u32 { a * a };
        sq(x) + sq(x + 1u32)
    }

    #[kernel(workgroup_size(64, 1, 1))]
    #[allow(clippy::redundant_closure_call)]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<u32>,
        mut results: RWStructuredBuffer<Results>,
        mut scratch: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < input.len() {
            let x = input[i];
            // A closure in the kernel body.
            let add = |a: u32, b: u32| -> u32 { a + b };
            let added = add(x, 1u32);
            // A closure that calls a helper, and one that calls a method.
            let through_helper = |v: u32| -> u32 { shift(v) * 2u32 };
            let through_method = |c: Counter, k: u32| -> u32 { c.scaled(k) };
            let base = Counter { n: x, step: 1u32 };
            let scaled = base.scaled(3u32);
            let via_helper = through_helper(x);
            let via_method = through_method(base, 5u32);
            let mut looped = 0u32;
            for k in 0u32..4u32 {
                looped += add(k, x);
            }
            // One closure for each kind of parameter.
            let float_closure = |a: f32, b: f32| -> f32 { a * b + 0.5f32 };
            let int_closure = |a: i32, b: i32| -> i32 { a - b };
            let bool_closure = |a: bool, b: bool| -> u32 {
                if a && !b { 1u32 } else { 0u32 }
            };
            let vector_closure =
                |v: Vec2, w: Vec4| -> f32 { v.x + v.y + w.z + w.w };
            let matrix_closure = |m: Mat2, v: Vec2| -> f32 { m.mul_vec2(v).x };
            let struct_closure = |c: Counter| -> u32 { c.n + c.step };
            let mode_closure = |m: Mode| -> u32 { if m == Mode::A { 10u32 } else { 20u32 } };
            let tuple_closure = |t: (u32, u32)| -> u32 { t.0 * 10u32 + t.1 };
            let array_closure = |a: [u32; 3]| -> u32 { a[0] + a[1] + a[2] };
            let float_kind = float_closure(x as f32, 0.25f32);
            let int_kind = int_closure(x as i32, 7i32);
            let bool_kind = bool_closure(x > 3u32, x > 6u32);
            let vector_kind = vector_closure(
                Vec2::new(x as f32, 1.0f32),
                Vec4::new(0.0f32, 0.0f32, 2.0f32, 3.0f32),
            );
            let matrix = Mat2::from_cols(Vec2::new(2.0f32, 0.0f32), Vec2::new(0.0f32, 3.0f32));
            let matrix_kind = matrix_closure(matrix, Vec2::new(x as f32, 1.0f32));
            let counter = Counter { n: x, step: 3u32 };
            let struct_kind = struct_closure(counter);
            let mut mode = Mode::A;
            if x > 3u32 {
                mode = Mode::B;
            }
            let enum_kind = mode_closure(mode);
            let tuple_kind = tuple_closure((x, 1u32));
            let array_kind = array_closure([x, 1u32, 2u32]);
            // The order the arguments of a closure call run in is the order of Rust: left to right.
            let pair = |a: u32, b: u32| -> u32 { a * 1000u32 + b };
            let mut c1 = Counter { n: 0u32, step: 1u32 };
            let order_two_ticks = pair(c1.tick(), c1.tick());
            let mut c2 = Counter { n: 0u32, step: 1u32 };
            let order_read_then_tick = pair(c2.n, c2.tick());
            let mut c3 = Counter { n: 0u32, step: 1u32 };
            let order_tick_then_read = pair(c3.tick(), c3.n);
            let mut c4 = Counter { n: 0u32, step: 1u32 };
            let order_nested = pair(pair(c4.tick(), c4.tick()), c4.tick());
            // Closures in the index and in the value of one assignment.
            let slot = |a: u32| -> u32 { a };
            let odd = |a: u32| -> u32 { a * 2u32 + 1u32 };
            scratch[slot(i)] = odd(i);
            results[i] = Results {
                added,
                poly: poly(x),
                scaled,
                via_helper,
                via_method,
                looped,
                float_kind,
                int_kind,
                bool_kind,
                vector_kind,
                matrix_kind,
                struct_kind,
                enum_kind,
                tuple_kind,
                array_kind,
                order_two_ticks,
                order_read_then_tick,
                order_tick_then_read,
                order_nested,
                placed: scratch[i],
            };
        }
    }
}

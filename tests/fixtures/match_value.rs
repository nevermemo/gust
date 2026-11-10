#[gust::gpu]
mod match_value {
    // Slice m9.e2 (D157): the value form of `match` (`let x: T = match e { .. };`) on an enum and on `u32` and `i32`
    // scrutinees with integer-literal patterns, in both forms. The Slang is `T x;` and a `switch` that assigns `x`.
    #[derive(PartialEq, Eq)]
    pub enum Mode {
        Off,
        Low = 3,
        Mid,
        High,
        Max = 10,
    }

    // A struct-typed result of a value `match`.
    pub struct Pair {
        pub a: u32,
        pub b: f32,
    }

    // Records the order of its method calls: `mark` appends a digit to `n`, `tick` counts.
    pub struct Trace {
        pub n: u32,
    }

    impl Trace {
        pub fn mark(&mut self, tag: u32) -> u32 {
            self.n = self.n * 10u32 + tag;
            tag
        }

        pub fn tick(&mut self) -> u32 {
            self.n += 1u32;
            self.n
        }
    }

    // One output buffer: the portable limit is eight storage buffers per shader stage.
    pub struct Results {
        pub enum_value: u32,
        pub rank: u32,
        pub bucket: u32,
        pub wide: u32,
        pub sgn: i32,
        pub stmt_int: u32,
        pub nested: u32,
        pub pair_a: u32,
        pub pair_b: f32,
        pub early: u32,
        pub shadow: u32,
        pub scrut: u32,
        pub scrut_n: u32,
        pub arm_after: u32,
        pub arm_before: u32,
        pub arm_n: u32,
        pub same: u32,
        pub same_n: u32,
        pub taken: u32,
        pub taken_n: u32,
    }

    fn mode_from(v: u32) -> Mode {
        if v == 0u32 {
            return Mode::Off;
        }
        if v == 1u32 {
            return Mode::Low;
        }
        if v == 2u32 {
            return Mode::Mid;
        }
        if v == 3u32 {
            return Mode::High;
        }
        Mode::Max
    }

    // Arm expressions, a grouped arm whose block ends in an expression, and `_`.
    fn enum_value(m: Mode) -> u32 {
        let w: u32 = match m {
            Mode::Off => 1u32,
            Mode::Low | Mode::Mid => {
                let t = m as u32;
                t * 10u32
            }
            _ => 7u32,
        };
        w + 1u32
    }

    // Every variant is named and there is no `_`: the last arm is the `default`, so the assignment is total.
    fn rank(m: Mode) -> u32 {
        let r: u32 = match m {
            Mode::Off => 10u32,
            Mode::Low => 11u32,
            Mode::Mid => 12u32,
            Mode::High => 13u32,
            Mode::Max => 14u32,
        };
        r + 100u32
    }

    // A `u32` scrutinee: unsuffixed and suffixed literals, a grouped arm.
    fn bucket(v: u32) -> u32 {
        let b: u32 = match v {
            0 => 100u32,
            1 | 2 => 200u32,
            3u32 => 300u32,
            _ => 400u32,
        };
        b + 1u32
    }

    // Literals above `i32::MAX`, and a scrutinee that is an expression.
    fn wide(v: u32) -> u32 {
        let w: u32 = match v * 1000000000u32 {
            0 => 1u32,
            1000000000 => 2u32,
            3000000000 => 3u32,
            4000000000u32 => 4u32,
            _ => 5u32,
        };
        w + 1u32
    }

    // An `i32` scrutinee with negative literals, suffixed and not.
    fn sgn(s: i32) -> i32 {
        let r: i32 = match s {
            -2 => -20i32,
            -1i32 => -10i32,
            0 => 0i32,
            1 | 2 => 10i32,
            _ => 99i32,
        };
        r - 1i32
    }

    // The statement form with integer literals.
    fn stmt_int(v: u32) -> u32 {
        let mut out = 5u32;
        match v {
            0 => {
                out += 1u32;
            }
            1 | 2 => {
                out += 20u32;
            }
            _ => {
                out += 300u32;
            }
        }
        out
    }

    // A value `match` as an arm (`Off`), a value `match` in a `let` of an arm (`Low | Mid`) and as the tail of an arm
    // block (`_`): all assign the one result.
    fn nested(m: Mode, v: u32) -> u32 {
        let n: u32 = match m {
            Mode::Off => match v {
                0 => 1u32,
                _ => 2u32,
            },
            Mode::Low | Mode::Mid => {
                let inner: u32 = match v {
                    0 | 1 => 10u32,
                    _ => 20u32,
                };
                inner + 1u32
            }
            _ => {
                let k = v + 1u32;
                match k {
                    1 => 100u32,
                    _ => 200u32,
                }
            }
        };
        n + 1u32
    }

    // An arm that returns: the other arms give the value.
    fn early(v: u32) -> u32 {
        let r: u32 = match v {
            0 => {
                return 99u32;
            }
            1 => 5u32,
            _ => 6u32,
        };
        r + 1u32
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
            let m = mode_from(v);

            // A struct-typed result: a struct literal as an arm, and as the tail of an arm block.
            let p: Pair = match mode_from(v) {
                Mode::Off => Pair { a: 1u32, b: 0.5f32 },
                Mode::Low => {
                    let t = v + 2u32;
                    Pair { a: t, b: 1.5f32 }
                }
                _ => Pair {
                    a: v * 2u32,
                    b: 2.5f32,
                },
            };

            // The new local has the name of the outer one that the `match` reads, in a block of its own.
            let k = v;
            let mut shadow = 0u32;
            if v < 100u32 {
                let k: u32 = match k {
                    0 => k + 10u32,
                    _ => k + 20u32,
                };
                shadow = k + 1u32;
            }

            // The scrutinee is evaluated once: the arm sees the count after one `tick`, not after one per label.
            let mut t = Trace { n: 0u32 };
            let scrut: u32 = match t.tick() {
                1 | 3 | 5 => t.tick() * 10u32,
                _ => 0u32,
            };

            // An arm reads a local that the scrutinee read and the arm's own call changes: the left operand first.
            let mut c = Trace { n: 5u32 };
            let arm_after: u32 = match c.n {
                5 => c.n + c.tick(),
                _ => 0u32,
            };
            let arm_before: u32 = match c.n {
                6 => c.tick() + c.n,
                _ => 0u32,
            };

            // The scrutinee and the arm call the same mutating method.
            let mut d = Trace { n: 0u32 };
            let same: u32 = match d.tick() {
                1 => d.tick() * 100u32,
                _ => 0u32,
            };

            // Only the arm that is taken runs its call.
            let mut e = Trace { n: 0u32 };
            let taken: u32 = match mode_from(v) {
                Mode::Off => e.mark(1u32) + 100u32,
                Mode::Low => e.mark(2u32) + 100u32,
                _ => e.mark(3u32) + 100u32,
            };

            results[i] = Results {
                enum_value: enum_value(m),
                rank: rank(mode_from(v)),
                bucket: bucket(v),
                wide: wide(v),
                sgn: sgn((v as i32) - 2i32),
                stmt_int: stmt_int(v),
                nested: nested(mode_from(v), (v + 1u32) % 2u32),
                pair_a: p.a + 1u32,
                pair_b: p.b,
                early: early(v),
                shadow,
                scrut,
                scrut_n: t.n,
                arm_after,
                arm_before,
                arm_n: c.n,
                same,
                same_n: d.n,
                taken,
                taken_n: e.n,
            };
        }
    }
}

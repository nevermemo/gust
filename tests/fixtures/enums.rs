#[gust::gpu]
mod enums {
    // Slice m9.e1 (D157): field-less enums and the statement `match` on them. `Clone`, `Copy` and `#[repr(u32)]` are
    // added by the macro; the user derives the comparison. The variants after an explicit discriminant count up.
    #[derive(PartialEq, Eq)]
    pub enum Mode {
        Off,
        Low = 3,
        Mid,
        High,
        Max = 10,
    }

    // A signed repr, a negative discriminant and a variant name (`Mid`) that another enum has too.
    #[derive(PartialEq, Eq)]
    #[repr(i32)]
    pub enum Sign {
        Neg = -2,
        Mid,
        Pos = 5,
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
        pub code: u32,
        pub grade: u32,
        pub bumped: u32,
        pub eq: u32,
        pub ne: u32,
        pub sign: i32,
        pub weight: u32,
        pub scrut: u32,
        pub eq_order: u32,
        pub ne_order: u32,
        pub arg_order: u32,
        pub built: u32,
        pub built_trace: u32,
    }

    // An enum returned from a helper.
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

    fn sign_from(v: u32) -> Sign {
        let k = v % 3u32;
        if k == 0u32 {
            return Sign::Neg;
        }
        if k == 1u32 {
            return Sign::Mid;
        }
        Sign::Pos
    }

    // A statement `match` with a grouped arm and `_`; the arms fall out of the `switch`.
    fn classify(m: Mode) -> u32 {
        let mut out = 5u32;
        match m {
            Mode::Off => {
                out += 5u32;
            }
            Mode::Low | Mode::Mid => {
                out += 15u32;
            }
            _ => {
                out += 25u32;
            }
        }
        out
    }

    // Every arm returns, and `_` takes the rest: a `return` in an arm of a helper is legal.
    #[allow(clippy::needless_return)]
    fn bump_mode(m: Mode) -> Mode {
        match m {
            Mode::Off => {
                return Mode::Low;
            }
            Mode::Low | Mode::Mid => {
                return Mode::High;
            }
            _ => {
                return Mode::Off;
            }
        }
    }

    // Every variant is named and there is no `_`: the last arm is the `default` of the Slang `switch`.
    #[allow(clippy::needless_return)]
    fn weight(s: Sign) -> u32 {
        match s {
            Sign::Neg => {
                return 1u32;
            }
            Sign::Mid => {
                return 2u32;
            }
            Sign::Pos => {
                return 3u32;
            }
        }
    }

    // An enum argument built from a call: the arguments are evaluated left to right.
    fn take(a: Mode, b: Mode) -> u32 {
        (a as u32) * 100u32 + (b as u32)
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
            let s = sign_from(v);
            let mut eq = 0u32;
            if m == Mode::Mid {
                eq = 1u32;
            }
            let mut ne = 0u32;
            if m != Mode::Off {
                ne = 1u32;
            }

            // The scrutinee is evaluated once: the arm sees the count after one `tick`, not after one per label.
            let mut t = Trace { n: 0u32 };
            let mut scrut = 0u32;
            match mode_from(t.tick()) {
                Mode::Off => {
                    scrut += 100u32;
                }
                Mode::Low | Mode::Mid => {
                    scrut += t.tick();
                }
                _ => {
                    scrut += 200u32;
                }
            }

            // `==` and `!=` on two calls: the left call runs first.
            let mut a = Trace { n: 0u32 };
            let same = mode_from(a.mark(1u32)) == mode_from(a.mark(2u32));
            let mut eq_order = a.n;
            if same {
                eq_order += 1000000u32;
            }
            let mut b = Trace { n: 0u32 };
            let different = mode_from(b.mark(3u32)) != mode_from(b.mark(4u32));
            let mut ne_order = b.n;
            if !different {
                ne_order += 1000000u32;
            }

            // Two enum arguments, each built from a call.
            let mut u = Trace { n: 0u32 };
            let arg_order = take(mode_from(u.tick()), mode_from(u.tick()));
            let mut w = Trace { n: 0u32 };
            let built = take(mode_from(w.mark(1u32)), mode_from(w.mark(2u32)));

            results[i] = Results {
                code: m as u32,
                grade: classify(m),
                bumped: bump_mode(m) as u32,
                eq,
                ne,
                sign: s as i32,
                weight: weight(s),
                scrut,
                eq_order,
                ne_order,
                arg_order,
                built,
                built_trace: w.n,
            };
        }
    }
}

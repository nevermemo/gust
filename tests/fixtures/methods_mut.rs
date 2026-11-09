#[gust::gpu]
mod methods_mut {
    pub struct Counter {
        pub n: uint,
        pub step: uint,
    }

    pub struct Acc {
        pub total: uint,
        pub count: uint,
    }

    // One output buffer: the portable limit is eight storage buffers per shader stage.
    pub struct Results {
        pub bumped: uint,
        pub consumed: uint,
        pub scaled: uint,
        pub mean: uint,
        pub looped: uint,
        pub helper: uint,
    }

    impl Counter {
        // An associated function (no receiver): `static` in Slang. `Self` is `Counter`.
        #[allow(clippy::let_and_return)]
        pub fn new(start: uint, step: uint) -> Self {
            let counter = Self {
                n: start,
                step,
            };
            counter
        }

        // `&mut self`: `[mutating]` in Slang, callable on a local.
        #[allow(clippy::assign_op_pattern)]
        pub fn bump(&mut self) {
            self.n = self.n + self.step;
        }

        // A method that calls a method.
        pub fn bump_twice(&mut self) {
            self.bump();
            self.bump();
        }

        // `self` by value.
        pub fn value(self) -> uint {
            self.n
        }

        // A method that calls a helper.
        pub fn scaled(&self, factor: uint) -> uint {
            scale_by(self.n, factor)
        }
    }

    // A second `impl` block of the same struct, with an associated function that calls an associated function.
    impl Counter {
        pub fn unit() -> Counter {
            Self::new(0u32, 1u32)
        }
    }

    impl Acc {
        #[allow(clippy::assign_op_pattern)]
        pub fn add(&mut self, amount: uint) {
            self.total = self.total + amount;
            self.count = self.count + 1u32;
        }

        pub fn mean(&self) -> uint {
            if self.count == 0u32 {
                return 0u32;
            }
            self.total / self.count
        }
    }

    fn scale_by(value: uint, factor: uint) -> uint {
        value * factor
    }

    // A helper that calls methods.
    fn drive(start: uint) -> uint {
        let mut counter = Counter::new(start, 3u32);
        counter.bump_twice();
        counter.value()
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut results: RWStructuredBuffer<Results>,
    ) {
        let i = id.x;
        if i < input.len() {
            let mut counter = Counter::new(input[i], 2u32);
            counter.bump();
            let bumped = counter.n;
            counter.bump_twice();
            let scaled = counter.scaled(3u32);
            let consumed = counter.value();
            let mut acc = Acc {
                total: 0u32,
                count: 0u32,
            };
            acc.add(input[i]);
            acc.add(10u32);
            let mut unit = Counter::unit();
            for k in 0u32..4u32 {
                unit.bump();
                acc.add(k);
            }
            let mean = acc.mean();
            let looped = unit.value();
            let helper = drive(input[i]);
            results[i] = Results {
                bumped,
                consumed,
                scaled,
                mean,
                looped,
                helper,
            };
        }
    }
}

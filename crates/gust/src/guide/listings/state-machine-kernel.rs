#[gpu_module]
mod scanner {
    /// What a symbol is, as far as the scanner cares.
    #[derive(PartialEq, Eq)]
    pub enum Class {
        Space,
        Digit,
        Letter,
        Other,
    }

    /// Where a machine is: outside a token or inside one.
    #[derive(PartialEq, Eq)]
    pub enum State {
        Out,
        In,
    }

    /// One symbol of the stream, as the ASCII code the host stored.
    pub struct Ascii {
        pub code: u32,
    }

    /// A trait of the module: static dispatch, so every call names the method of the concrete type.
    pub trait Classify {
        fn class(&self) -> Class;
    }

    impl Classify for Ascii {
        // The dialect has no `RangeInclusive::contains`, so the comparisons are written out.
        #[allow(clippy::manual_range_contains)]
        fn class(&self) -> Class {
            let c = self.code;
            if c == 32u32 {
                return Class::Space;
            }
            if c >= 48u32 && c <= 57u32 {
                return Class::Digit;
            }
            if (c >= 65u32 && c <= 90u32) || (c >= 97u32 && c <= 122u32) {
                return Class::Letter;
            }
            Class::Other
        }
    }

    /// Which classes belong to a token. Two sets of rules, so the generic `feed` below has two instances.
    pub trait Rules {
        fn accepts(&self, class: Class) -> bool;
    }

    // A struct needs a field (an empty one is GUST-E0105), so the two rule types carry an unused `tag`.
    pub struct Words {
        pub tag: u32,
    }

    pub struct Numbers {
        pub tag: u32,
    }

    impl Rules for Words {
        fn accepts(&self, class: Class) -> bool {
            class == Class::Letter
        }
    }

    impl Rules for Numbers {
        fn accepts(&self, class: Class) -> bool {
            class == Class::Digit
        }
    }

    /// The result of one record: how many tokens of each kind and the longest of each.
    pub struct Summary {
        pub words: u32,
        pub longest_word: u32,
        pub numbers: u32,
        pub longest_number: u32,
    }

    /// A finite-state machine over the symbols of one record. An enum is not a field of a struct (GUST-E0211), so the
    /// machine keeps its state as the integer `State as u32` and `feed` turns it back into a `State`.
    pub struct Machine {
        pub inside: u32,
        pub tokens: u32,
        pub run: u32,
        pub longest: u32,
    }

    /// The transition: the next state and whether a token starts here (a tuple result, taken apart by `feed`).
    fn transition(state: State, accepted: bool) -> (State, u32) {
        let mut next = State::Out;
        let mut started = 0u32;
        match state {
            State::Out => {
                if accepted {
                    next = State::In;
                    started = 1u32;
                }
            }
            State::In => {
                if accepted {
                    next = State::In;
                }
            }
        }
        (next, started)
    }

    impl Machine {
        /// Feeds one symbol: a generic method, bounded by the module's traits.
        pub fn feed<R: Rules, S: Classify>(&mut self, rules: R, symbol: S) {
            let accepted = rules.accepts(symbol.class());
            let mut state = State::Out;
            if self.inside == 1u32 {
                state = State::In;
            }
            let (next, started) = transition(state, accepted);
            self.tokens += started;
            if accepted {
                if started == 1u32 {
                    self.run = 1u32;
                } else {
                    self.run += 1u32;
                }
                if self.run > self.longest {
                    self.longest = self.run;
                }
            }
            self.inside = next as u32;
        }

        pub fn totals(&self) -> (u32, u32) {
            (self.tokens, self.longest)
        }
    }

    /// One invocation scans one record of 16 symbols. A token cut by the end of a record ends there: every record
    /// starts with both machines outside a token.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scan(
        id: SV_DispatchThreadID,
        text: StructuredBuffer<uint>,
        mut summary: RWStructuredBuffer<Summary>,
    ) {
        let r = id.x;
        if r < text.len() / 16u32 && r < summary.len() {
            let mut words = Machine {
                inside: 0u32,
                tokens: 0u32,
                run: 0u32,
                longest: 0u32,
            };
            let mut numbers = Machine {
                inside: 0u32,
                tokens: 0u32,
                run: 0u32,
                longest: 0u32,
            };
            for k in 0u32..16u32 {
                let code = text[r * 16u32 + k];
                let symbol = Ascii { code };
                let word_rules = Words { tag: 0u32 };
                let number_rules = Numbers { tag: 0u32 };
                words.feed(word_rules, symbol);
                numbers.feed(number_rules, symbol);
            }
            let (w, wl) = words.totals();
            let (n, nl) = numbers.totals();
            summary[r] = Summary {
                words: w,
                longest_word: wl,
                numbers: n,
                longest_number: nl,
            };
        }
    }
}

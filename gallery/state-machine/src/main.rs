// ANCHOR: imports
use gust::{Independent, gpu};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
// ANCHOR_END: imports

// ANCHOR: kernel
#[gpu]
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
// ANCHOR_END: kernel

// ANCHOR: host
/// Symbols per record (the literal 16 of the kernel).
const RECORD: usize = 16;

/// The stream of `records` records: words, numbers, spaces and a few other symbols, from a small generator, so the
/// same text is produced on every run. Tokens run over the end of a record on purpose.
fn stream(records: usize) -> Vec<u8> {
    let mut seed = 0x2545_f491_u32;
    let mut text = Vec::with_capacity(records * RECORD + 8);
    while text.len() < records * RECORD {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let length = 1 + usize::try_from((seed >> 24) % 7).expect("below 7");
        let kind = (seed >> 16) % 8;
        for step in 0..length {
            let pick = (usize::try_from(seed >> (step % 16)).expect("a u32") + step) % 26;
            let letter = u8::try_from(pick).expect("below 26");
            text.push(match kind {
                0..=3 => b'a' + letter,
                4 | 5 => b'0' + letter % 10,
                6 => b' ',
                _ => b"-_./,;"[pick % 6],
            });
        }
        text.push(b' ');
    }
    text.truncate(records * RECORD);
    text
}

/// What one record holds, as the kernel reports it.
#[derive(Debug, PartialEq, Eq)]
struct Counts {
    words: u32,
    longest_word: u32,
    numbers: u32,
    longest_number: u32,
}

/// The independent host reference: it splits the record at every symbol outside the token set instead of running a
/// state machine, and never goes through the kernel (D26, D42).
fn host_counts(record: &[u8]) -> Counts {
    let tokens = |keep: fn(&u8) -> bool| -> (u32, u32) {
        let lengths: Vec<usize> = record
            .split(|b| !keep(b))
            .map(<[u8]>::len)
            .filter(|&length| length > 0)
            .collect();
        let to_u32 = |n: usize| u32::try_from(n).expect("a record has 16 symbols");
        (
            to_u32(lengths.len()),
            to_u32(lengths.iter().copied().max().unwrap_or(0)),
        )
    };
    let (words, longest_word) = tokens(u8::is_ascii_alphabetic);
    let (numbers, longest_number) = tokens(u8::is_ascii_digit);
    Counts {
        words,
        longest_word,
        numbers,
        longest_number,
    }
}

/// The symbols as the kernel's buffer holds them: one `u32` per symbol.
fn symbols(text: &[u8]) -> Vec<u32> {
    text.iter().map(|&b| u32::from(b)).collect()
}

/// An empty result buffer's contents: one zeroed summary per record.
fn empty_summaries(records: usize) -> Vec<scanner::Summary> {
    (0..records)
        .map(|_| scanner::Summary {
            words: 0,
            longest_word: 0,
            numbers: 0,
            longest_number: 0,
        })
        .collect()
}

/// A kernel result in the host reference's type, so the two compare with `assert_eq!`.
fn counts_of(summary: &scanner::Summary) -> Counts {
    Counts {
        words: summary.words,
        longest_word: summary.longest_word,
        numbers: summary.numbers,
        longest_number: summary.longest_number,
    }
}
// ANCHOR_END: host

// ANCHOR: main
fn main() {
    let records = 1000;
    let text = stream(records);
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let gpu_text = device.create_typed_buffer("text", &symbols(&text), GpuBufferAccess::ReadOnly);
    let summary = device.create_typed_buffer(
        "summary",
        &empty_summaries(records),
        GpuBufferAccess::ReadWrite,
    );
    // One invocation per record; the text is 16 times longer than that, which `Independent` says is meant.
    let invocations = u32::try_from(records).expect("fewer than 2^32 records");
    scanner::scan::dispatch(&device, invocations, Independent(&gpu_text), &summary)
        .expect("dispatch");
    let got = device.read_typed_buffer(&summary).expect("read back");
    let mut total = (0u32, 0u32, 0u32, 0u32);
    for (i, record) in text.chunks(RECORD).enumerate() {
        let want = host_counts(record);
        assert_eq!(
            counts_of(&got[i]),
            want,
            "record {i}: {:?}",
            String::from_utf8_lossy(record)
        );
        total = (
            total.0 + want.words,
            total.1.max(want.longest_word),
            total.2 + want.numbers,
            total.3.max(want.longest_number),
        );
    }
    println!(
        "state-machine: {records} records of {RECORD} symbols, {} words (longest {}), {} numbers (longest {})",
        total.0, total.1, total.2, total.3
    );
    println!("adapter: {}", device.adapter_info().name);
}
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::{Independent, cpu::CpuDevice};

    use super::{
        Counts, RECORD, counts_of, empty_summaries, host_counts, scanner, stream, symbols,
    };

    /// The kernel on the CPU twin, one result per record.
    fn on_twin(text: &[u8], invocations: u32) -> Vec<Counts> {
        let device = CpuDevice::new();
        let input = device.buffer(&symbols(text));
        let out = device.buffer(&empty_summaries(text.len() / RECORD));
        scanner::scan::dispatch(&device, invocations, Independent(&input), &out)
            .expect("the twin dispatches");
        out.read().iter().map(counts_of).collect()
    }

    #[test]
    fn the_twin_matches_hand_counted_values() {
        // "ab 12 cd3 4567 x" is 16 symbols: words ab, cd, x (longest 2), numbers 12, 3, 4567 (longest 4).
        let got = on_twin(b"ab 12 cd3 4567 x", 1);
        let want = Counts {
            words: 3,
            longest_word: 2,
            numbers: 3,
            longest_number: 4,
        };
        assert_eq!(got, [want]);
        // A record of one long token is one token, whatever follows it in the stream.
        let got = on_twin(b"abcdefghijklmnop", 1);
        let want = Counts {
            words: 1,
            longest_word: 16,
            numbers: 0,
            longest_number: 0,
        };
        assert_eq!(got, [want]);
    }

    #[test]
    fn the_twin_matches_the_host_reference_exactly() {
        // Sizes below, at and above the workgroup size of 64, ones that are not a multiple of it, and 1000 records.
        for records in [1usize, 63, 64, 65, 1000] {
            let text = stream(records);
            let got = on_twin(&text, u32::try_from(records).expect("small"));
            assert_eq!(got.len(), records);
            for (i, record) in text.chunks(RECORD).enumerate() {
                assert_eq!(
                    got[i],
                    host_counts(record),
                    "records = {records}, record {i}"
                );
            }
        }
    }

    #[test]
    fn the_stream_has_tokens_and_cut_tokens() {
        // A guard on the example: every kind of token occurs, and a token runs across the end of a record.
        let text = stream(1000);
        let counts: Vec<Counts> = text.chunks(RECORD).map(host_counts).collect();
        assert!(counts.iter().any(|c| c.words > 2));
        assert!(counts.iter().any(|c| c.numbers > 1));
        assert!(counts.iter().any(|c| c.longest_word >= 5));
        assert!(
            text.chunks(RECORD)
                .skip(1)
                .any(|r| r[0].is_ascii_alphanumeric())
        );
    }
}
// ANCHOR_END: tests

//! A small deterministic PRNG, std only. The sequence is part of the seed contract: a change to anything in this
//! file alters every generated program and must bump `DIFFTEST_FORMAT`.

/// The SplitMix64 finalizer applied to `x` plus the golden-ratio increment (one step of the generator).
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xorshift64*: a 64-bit xorshift state with a multiplicative output scramble.
#[derive(Clone, Debug)]
pub struct Prng {
    state: u64,
}

impl Prng {
    /// Seeds the generator. The seed goes through SplitMix64 so that nearby seeds give unrelated streams and the
    /// all-zero state (a fixed point of xorshift) cannot occur.
    pub fn new(seed: u64) -> Self {
        let mixed = splitmix64(seed);
        Self {
            state: if mixed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                mixed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A value in `0..bound` (`bound` > 0). The modulo bias is irrelevant for test generation.
    pub fn below(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "below(0)");
        self.next_u32() % bound
    }
}

/// The byte stream of program `attempt` of `seed`: `len` bytes of the PRNG output, little-endian per word. The
/// stream is a pure function of its arguments.
pub fn byte_stream(seed: u64, attempt: u64, len: usize) -> Vec<u8> {
    let mixed = splitmix64(seed) ^ splitmix64(attempt ^ 0xD1B5_4A32_D192_ED03);
    let mut prng = Prng::new(mixed);
    let mut bytes = Vec::with_capacity(len + 8);
    while bytes.len() < len {
        bytes.extend_from_slice(&prng.next_u64().to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_matches_the_published_vector_for_seed_zero() {
        // The reference SplitMix64 generator started at state 0 first emits 0xE220A8397B1DCDAF.
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn xorshift_star_first_outputs_are_pinned() {
        // Reproduced with an independent arbitrary-precision evaluation (perl Math::BigInt) of SplitMix64 for the
        // seed followed by three xorshift64* steps.
        let mut prng = Prng::new(1);
        let first = [prng.next_u64(), prng.next_u64(), prng.next_u64()];
        assert_eq!(
            first,
            [
                0x4B46_A55D_F361_1B9B,
                0xD7E1_F141_0E76_3EF4,
                0x5F14_EC66_975F_9B06
            ]
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_stream_and_a_different_one_does_not() {
        assert_eq!(byte_stream(7, 3, 100), byte_stream(7, 3, 100));
        assert_ne!(byte_stream(7, 3, 100), byte_stream(8, 3, 100));
        assert_ne!(byte_stream(7, 3, 100), byte_stream(7, 4, 100));
        assert_eq!(byte_stream(7, 3, 100).len(), 100);
    }

    #[test]
    fn a_prefix_of_a_stream_is_the_shorter_stream() {
        let long = byte_stream(5, 9, 64);
        assert_eq!(byte_stream(5, 9, 21), long[..21]);
    }
}

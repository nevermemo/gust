//! The seed corpus (`seeds.txt`): which seeds are committed and what each one decodes to.
//!
//! Each line is `seed programs discarded [reason=count ...] hash=<fnv1a64>` where `hash` is the FNV-1a hash of the
//! `src/lib.rs` the emitter writes for the seed's programs, followed by the input data of every program
//! (type tag, length and the 32-bit patterns, little endian), because the data is not part of the emitted text. A
//! generator or emitter change therefore shows up as a changed line; it is either a bug or a deliberate format change that bumps `DIFFTEST_FORMAT` and regenerates the
//! file (`difftest-gen seeds 1 8`).

use crate::ast::Program;
use crate::emit::emit_crate_source;
use crate::generator::{DIFFTEST_FORMAT, generate_batch};

/// The 64-bit FNV-1a hash.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_update(0xCBF2_9CE4_8422_2325, bytes)
}

/// Continues a 64-bit FNV-1a hash over more bytes.
pub fn fnv1a64_update(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The corpus hash of a batch: the emitted `lib.rs`, then every input buffer of every program.
pub fn batch_hash(programs: &[Program]) -> u64 {
    batch_hash_as_format(programs, DIFFTEST_FORMAT)
}

/// [`batch_hash`] with the format stamp of the emitted text read as `format`: the stamp is the one thing a format bump
/// changes in a program that is otherwise the same, so this reproduces the hash an older format wrote for it.
pub fn batch_hash_as_format(programs: &[Program], format: u32) -> u64 {
    let text = emit_crate_source(programs).replacen(
        &format!("(DIFFTEST_FORMAT {DIFFTEST_FORMAT})"),
        &format!("(DIFFTEST_FORMAT {format})"),
        1,
    );
    let mut hash = fnv1a64(text.as_bytes());
    for program in programs {
        for input in &program.inputs {
            hash = fnv1a64_update(hash, &[input.ty.name().len() as u8]);
            hash = fnv1a64_update(hash, input.ty.name().as_bytes());
            hash = fnv1a64_update(hash, &(input.data.len() as u32).to_le_bytes());
            for word in &input.data {
                hash = fnv1a64_update(hash, &word.to_le_bytes());
            }
        }
    }
    hash
}

/// The line of `seeds.txt` for one seed.
pub fn corpus_line(seed: u64) -> Result<String, String> {
    let batch = generate_batch(seed)?;
    let mut line = format!(
        "{seed} {} {}",
        batch.programs.len(),
        batch.discarded_total()
    );
    for (reason, count) in &batch.discarded {
        line.push_str(&format!(" {reason}={count}"));
    }
    let hash = batch_hash(&batch.programs);
    line.push_str(&format!(" hash={hash:016x}"));
    Ok(line)
}

/// The whole file for the seeds `first..=last`, with its two comment lines.
pub fn corpus_text(first: u64, last: u64) -> Result<String, String> {
    let mut text = format!(
        "# DIFFTEST_FORMAT {DIFFTEST_FORMAT}\n# seed programs discarded [reason=count ...] hash=<fnv1a64 of the emitted lib.rs and the input data>\n"
    );
    for seed in first..=last {
        text.push_str(&corpus_line(seed)?);
        text.push('\n');
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_matches_the_published_vectors() {
        // Test vectors of the FNV reference (isthe.com/chongo/tech/comp/fnv).
        assert_eq!(fnv1a64(b""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xAF63_DC4C_8601_EC8C);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_F739_67E8);
    }

    #[test]
    fn the_committed_corpus_is_what_the_generator_decodes_today() {
        let committed = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/seeds.txt"))
            .unwrap()
            .replace("\r\n", "\n");
        let regenerated = corpus_text(1, 8).unwrap();
        assert_eq!(
            committed, regenerated,
            "seeds.txt is stale: a generator or emitter change must bump DIFFTEST_FORMAT and regenerate it \
             (cargo run --bin difftest-gen -- seeds 1 8 > seeds.txt)"
        );
    }

    #[test]
    fn the_hash_covers_the_input_data_as_well_as_the_emitted_text() {
        let mut programs = generate_batch(1).unwrap().programs;
        let before = batch_hash(&programs);
        let text = emit_crate_source(&programs);
        programs[0].inputs[0].data[0] ^= 1;
        assert_eq!(
            text,
            emit_crate_source(&programs),
            "the text does not show the data"
        );
        assert_ne!(before, batch_hash(&programs));
    }

    #[test]
    fn the_committed_seeds_reach_the_min_over_minus_one_discard() {
        // A `-1` divisor is biased in (generator.rs) so that this condition of the domain is exercised by the corpus.
        let text = corpus_text(1, 8).unwrap();
        assert!(text.contains("min-over-minus-one="), "{text}");
    }
}

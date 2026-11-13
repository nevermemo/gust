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
    use crate::generator::{
        ARRAY_ATTEMPT_BASE, ARRAY_END, CLOSURE_ATTEMPT_BASE, ENUM_ATTEMPT_BASE, ENUM_END,
        METHOD_ATTEMPT_BASE, METHOD_END, PROGRAMS_PER_SEED, SCALAR_PROGRAMS_PER_SEED, Space,
        TRAIT_ATTEMPT_BASE, TRAIT_PROGRAMS_PER_SEED, VALUE_ATTEMPT_BASE, VALUE_END,
        VECTOR_ATTEMPT_BASE, VECTOR_END, generate_batch_in,
    };

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

    /// Format 4 (m5.c2) refused the programs whose divisor slangc folds to zero, and only seed 4 lost any; format 5
    /// (m6.dv) takes the refusal back, so every seed holds its format-3 programs again, byte for byte (the hashes of
    /// `seeds.txt` at format 3, read with the stamp as 3), `s4_a26` among them. Format 6 (m6.d) leaves the float space
    /// as it was: this is still `generate_batch_in(Space::Float, ..)`, whose programs are the scalar part of a batch.
    #[test]
    fn the_float_space_still_holds_the_format_3_programs_of_every_seed() {
        let format_3 = [
            (1, 0xf156_6491_8988_01a2_u64),
            (2, 0x2bd1_d67b_b246_8d74),
            (3, 0x89eb_84af_5f29_8ca7),
            (4, 0xdcc8_2882_5b74_22bd),
            (5, 0xb700_977e_985a_cc91),
            (6, 0x5d96_2c93_e516_88ba),
            (7, 0x5673_ba1f_bd44_28a0),
            (8, 0x5fc5_0da8_5029_1f4f),
        ];
        for (seed, hash) in format_3 {
            let batch = generate_batch_in(Space::Float, seed).unwrap();
            let same = batch_hash_as_format(&batch.programs, 3) == hash;
            assert!(same, "seed {seed}");
        }
    }

    /// Format 9 (m9.d1b): the method space and the enum space of seeds 1 to 4 (16 programs each, as `generate_batch_in`
    /// keeps them) decode to these programs. The method space changed in format 9 (the biased shift amount is written
    /// `call and 31`, same-root `and=` and `shr=` are reachable), so its format-8 pin is re-taken here. A change to a
    /// decoder, the emitter's text, or the interpreter's acceptance moves a hash: it is a deliberate format change
    /// (bump `DIFFTEST_FORMAT`, re-pin) or a bug.
    #[test]
    fn the_method_and_enum_spaces_hold_the_format_9_programs_of_the_first_seeds() {
        let hashes = |space: Space, base: u64| -> Vec<(u64, u64)> {
            (1..=4)
                .map(|seed| {
                    let batch = generate_batch_in(space, seed).unwrap();
                    assert!(batch.programs.iter().all(|p| p.attempt >= base));
                    (seed, batch_hash_as_format(&batch.programs, 9))
                })
                .collect()
        };
        let method: Vec<(u64, u64)> = vec![
            (1, 4688680707108088843),
            (2, 12364637530606668533),
            (3, 1814103977557610224),
            (4, 8884440047440317631),
        ];
        let enums: Vec<(u64, u64)> = vec![
            (1, 17115249202000948236),
            (2, 8452528715074681578),
            (3, 12513220889235945946),
            (4, 9138584645308773336),
        ];
        assert_eq!(hashes(Space::Method, METHOD_ATTEMPT_BASE), method, "method");
        assert_eq!(hashes(Space::Enum, ENUM_ATTEMPT_BASE), enums, "enum");
    }

    /// Format 10 (m9.d1c): the value space of seeds 1 to 4 (16 programs each, as `generate_batch_in` keeps them) decodes
    /// to these programs. The float, vector, method and enum spaces are not in this pin: their format 3 and format 9
    /// pins above still hold (the value forms sit behind a flag that only the value space sets).
    #[test]
    fn the_value_space_holds_the_format_10_programs_of_the_first_seeds() {
        let hashes: Vec<(u64, u64)> = (1..=4)
            .map(|seed| {
                let batch = generate_batch_in(Space::Value, seed).unwrap();
                assert!(
                    batch
                        .programs
                        .iter()
                        .all(|p| p.attempt >= VALUE_ATTEMPT_BASE)
                );
                (seed, batch_hash_as_format(&batch.programs, 10))
            })
            .collect();
        let pinned: Vec<(u64, u64)> = vec![
            (1, 4556502341734226730),
            (2, 13035520527677460174),
            (3, 5938680198507040942),
            (4, 762944537114388028),
        ];
        assert_eq!(hashes, pinned, "value");
    }

    /// Format 11 (m9.d1d), re-pinned in format 12 (m9.d1e fix round 2: the D168 guard now covers every array, D169): the
    /// array space of seeds 1 to 4 (16 programs each, as `generate_batch_in` keeps them)
    /// decodes to these programs. The older spaces are untouched: their format 3, 9 and 10 pins above still hold.
    #[test]
    fn the_array_space_holds_the_programs_of_the_first_seeds_with_the_widened_guard() {
        let hashes: Vec<(u64, u64)> = (1..=4)
            .map(|seed| {
                let batch = generate_batch_in(Space::Array, seed).unwrap();
                assert!(
                    batch
                        .programs
                        .iter()
                        .all(|p| p.attempt >= ARRAY_ATTEMPT_BASE)
                );
                (seed, batch_hash_as_format(&batch.programs, 11))
            })
            .collect();
        let pinned: Vec<(u64, u64)> = vec![
            (1, 4623533607603803085),
            (2, 12657800841464584760),
            (3, 16336804947945770575),
            (4, 7162040925401680788),
        ];
        assert_eq!(hashes, pinned, "array");
    }

    /// Format 12 (m9.d1e): the closure space of seeds 1 to 4 (16 programs each, as `generate_batch_in` keeps them)
    /// decodes to these programs. The older spaces are untouched: their pins above still hold, each read with its own
    /// stamp.
    #[test]
    fn the_closure_space_holds_the_format_12_programs_of_the_first_seeds() {
        let hashes: Vec<(u64, u64)> = (1..=4)
            .map(|seed| {
                let batch = generate_batch_in(Space::Closure, seed).unwrap();
                assert!(
                    batch
                        .programs
                        .iter()
                        .all(|p| p.attempt >= CLOSURE_ATTEMPT_BASE && p.has_closure())
                );
                (seed, batch_hash_as_format(&batch.programs, 12))
            })
            .collect();
        let pinned: Vec<(u64, u64)> = vec![
            (1, 10042002379242349144),
            (2, 6698736558324710702),
            (3, 14225375692781081802),
            (4, 2186207238784607583),
        ];
        assert_eq!(hashes, pinned, "closure");
    }

    /// A batch is the first five programs of the float space, four of the vector space, one of the method space, two of
    /// the enum space, two of the value space and two of the array space (format 11): the scalar decoding of every seed is what
    /// format 5 wrote, and so is the vector decoding of format 6; only the cut of the batch is new.
    #[test]
    fn a_batch_is_three_float_four_vector_one_method_one_enum_one_value_two_array_two_closure_and_two_trait_programs()
     {
        for seed in 1..=8 {
            let batch = generate_batch(seed).unwrap();
            let float = generate_batch_in(Space::Float, seed).unwrap();
            let vector = generate_batch_in(Space::Vector, seed).unwrap();
            assert_eq!(batch.programs.len(), PROGRAMS_PER_SEED);
            for k in 0..SCALAR_PROGRAMS_PER_SEED {
                assert_eq!(
                    batch.programs[k], float.programs[k],
                    "seed {seed} program {k}"
                );
                assert!(!batch.programs[k].has_aggregate());
                assert!(batch.programs[k].methods.is_empty());
            }
            for k in SCALAR_PROGRAMS_PER_SEED..VECTOR_END {
                let program = &batch.programs[k];
                assert!(program.attempt >= VECTOR_ATTEMPT_BASE, "seed {seed}");
                assert!(program.attempt < METHOD_ATTEMPT_BASE, "seed {seed}");
                assert!(
                    program.has_aggregate(),
                    "seed {seed} attempt {}",
                    program.attempt
                );
                assert!(program.methods.is_empty());
                assert_eq!(
                    program,
                    &vector.programs[k - SCALAR_PROGRAMS_PER_SEED],
                    "seed {seed} program {k}"
                );
            }
            for program in &batch.programs[VECTOR_END..] {
                assert!(program.attempt >= METHOD_ATTEMPT_BASE, "seed {seed}");
                assert!(
                    program.has_call() && !program.methods.is_empty(),
                    "seed {seed} attempt {}",
                    program.attempt
                );
                assert!(!program.has_float() && !program.has_aggregate());
            }
            for program in &batch.programs[VECTOR_END..METHOD_END] {
                assert!(program.attempt < ENUM_ATTEMPT_BASE, "seed {seed}");
                assert!(program.methods.enums.is_empty() && !program.has_match());
            }
            for program in &batch.programs[METHOD_END..ENUM_END] {
                assert!(program.attempt >= ENUM_ATTEMPT_BASE, "seed {seed}");
                assert!(program.attempt < VALUE_ATTEMPT_BASE, "seed {seed}");
                assert!(program.has_match() && !program.methods.enums.is_empty());
            }
            for program in &batch.programs[ENUM_END..VALUE_END] {
                assert!(program.attempt >= VALUE_ATTEMPT_BASE, "seed {seed}");
                assert!(program.has_value_form() && !program.methods.enums.is_empty());
            }
            for program in &batch.programs[VALUE_END..ARRAY_END] {
                assert!(program.attempt >= ARRAY_ATTEMPT_BASE, "seed {seed}");
                assert!(
                    program.has_array(),
                    "seed {seed} attempt {}",
                    program.attempt
                );
            }
            let last = &batch.programs[PROGRAMS_PER_SEED - TRAIT_PROGRAMS_PER_SEED..];
            assert_eq!(last.len(), 2);
            for program in last {
                assert!(program.attempt >= TRAIT_ATTEMPT_BASE, "seed {seed}");
                assert!(program.methods.has_traits() && program.has_trait_forms());
            }
        }
    }

    /// Format 13 (m9.d2): the trait space of seeds 1 to 4 (16 programs each, as `generate_batch_in` keeps them)
    /// decodes to these programs. The older spaces are untouched: their pins above still hold, each read with its own
    /// stamp.
    #[test]
    fn the_trait_space_holds_the_format_13_programs_of_the_first_seeds() {
        let hashes: Vec<(u64, u64)> = (1..=4)
            .map(|seed| {
                let batch = generate_batch_in(Space::Trait, seed).unwrap();
                assert!(
                    batch
                        .programs
                        .iter()
                        .all(|p| p.attempt >= TRAIT_ATTEMPT_BASE && p.has_trait_forms())
                );
                (seed, batch_hash(&batch.programs))
            })
            .collect();
        let pinned: Vec<(u64, u64)> = vec![
            (1, 9563745800497969671),
            (2, 6395250079408895879),
            (3, 17641521707558961229),
            (4, 16935275922206718759),
        ];
        assert_eq!(hashes, pinned, "trait");
    }
}

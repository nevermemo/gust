//! Subgroup operations on a real adapter, from committed artifacts (M8, slice m8.e; D140, D141).
//!
//! One kernel, `lanes::survey`, written twice: the portable body (every profile has it) writes zeros, and
//! `survey_subgroups` is the body a build with the `subgroups` capability gets. Each invocation of the subgroup body
//! writes four words beside its position in the data: the **lane count** (`subgroup_size()`), its own lane
//! (`subgroup_lane()`), the **ballot popcount** of "my element is over [`THRESHOLD`]" (`subgroup_count`) and the
//! inclusive **prefix sum** of the elements of its subgroup (`subgroup_inclusive_add`).
//!
//! The subgroup size is a property of the pipeline on the adapter (32 on an RTX 5090, 32 to 64 on a Radeon 890M, 8 on
//! lavapipe), so the program does not assume one: [`check`] rebuilds every subgroup from the size and the lane the
//! invocation wrote and compares the other two words with a host model.
//!
//! This is the strict proof of the capability (plan section 2.1): `GUST_PROFILE=native+subgroups cargo run -p
//! subgroup-reduce` builds with no Slang compiler (the crate does not enable `slang-runtime`, which a strict profile
//! refuses) and runs the committed WGSL of `gust-artifacts/`, written by `cargo gust build -p subgroup-reduce`. Under
//! `dev` (or any profile without the capability) the portable body is the kernel, [`run`] says so, and there is
//! nothing to check: the program does not pretend otherwise.

use gust::capability::Capability;
use gust_wgpu::{BufferBinding, DeviceOptions, Error, GpuBufferAccess, HeadlessDevice};

/// The kernels. The subgroup body follows the discipline of D140: every subgroup operation is the whole value of a
/// `let` at the top level of the body, the bounds guard is a statement `if` around the load and the stores (an
/// invocation past the end of the data holds 0 and still takes part), and there is no `return`.
#[gust::gpu]
pub mod lanes {
    // Words of invocation i, at `i * 4`: the subgroup size, the lane, the ballot popcount and the inclusive prefix sum.
    // The portable body writes zeros.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn survey(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        if id.x < data.len() {
            for word in 0u32..4u32 {
                out[id.x * 4u32 + word] = 0u32;
            }
        }
    }

    #[variant(of = survey, requires(subgroups))]
    pub fn survey_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let size = subgroup_size();
        let lane = subgroup_lane();
        let over = x > 10u32;
        let count = subgroup_count(over);
        let through = subgroup_inclusive_add(x);
        if id.x < data.len() {
            let base = id.x * 4u32;
            out[base] = size;
            out[base + 1u32] = lane;
            out[base + 2u32] = count;
            out[base + 3u32] = through;
        }
    }
}

/// Words each invocation writes.
pub const WORDS: usize = 4;

/// The element value above which the ballot counts an invocation (the literal `10u32` of the kernel).
pub const THRESHOLD: u32 = 10;

/// Elements surveyed: not a multiple of 64, so the last workgroup is partly past the data and its invocations hold 0.
pub const ELEMENTS: usize = 300;

/// The data: small positive values, some over [`THRESHOLD`] and some under it.
pub fn input(count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| ((i * 37 + 11) % 29 + 1) as u32)
        .collect()
}

/// Whether the body the build selected is the subgroup one (the build has the `subgroups` capability).
pub fn subgroup_body_selected() -> bool {
    lanes::survey::DESCRIPTOR
        .requires
        .contains(&Capability::Subgroups)
}

/// What [`check`] found, for a run whose results it accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Findings {
    /// The smallest and the largest subgroup size any invocation reported.
    pub sizes: (usize, usize),
    /// Invocations checked (the length of the data).
    pub invocations: usize,
    /// Subgroups that held at least one element, counted by their lane 0.
    pub subgroups: usize,
    /// The sum over subgroups of the ballot popcount, which must equal the host's count of elements over [`THRESHOLD`].
    pub ballot_total: u64,
    /// The sum over subgroups of the last lane's prefix sum, which must equal the sum of the data.
    pub prefix_total: u64,
}

/// Judges the words of a survey of `data` against the host model, or says what is wrong. Subgroups are the runs of
/// `size` consecutive invocations the words describe (`lane = index % size`, checked, not assumed); an invocation past
/// the end of the data contributes 0 and still takes part.
pub fn check(words: &[u32], data: &[u32]) -> Result<Findings, String> {
    if words.len() != data.len() * WORDS {
        return Err(format!(
            "{} words for {} elements, expected {}",
            words.len(),
            data.len(),
            data.len() * WORDS
        ));
    }
    let value = |index: usize| data.get(index).copied().unwrap_or(0);
    let (mut smallest, mut largest) = (usize::MAX, 0);
    let (mut subgroups, mut ballot_total, mut prefix_total) = (0, 0u64, 0u64);
    for index in 0..data.len() {
        let word = &words[index * WORDS..(index + 1) * WORDS];
        let (size, lane) = (word[0] as usize, word[1] as usize);
        if !(2..=128).contains(&size) || !size.is_power_of_two() {
            return Err(format!("invocation {index}: subgroup size {size}"));
        }
        if lane != index % size {
            return Err(format!(
                "invocation {index}: lane {lane} of size {size}, expected lane {}",
                index % size
            ));
        }
        let first = index - lane;
        let over = (first..first + size)
            .filter(|&member| value(member) > THRESHOLD)
            .count() as u32;
        let through = (first..=index).fold(0u32, |sum, member| sum.wrapping_add(value(member)));
        if word[2] != over {
            return Err(format!(
                "invocation {index} (lane {lane} of {size}): ballot popcount {}, expected {over}",
                word[2]
            ));
        }
        if word[3] != through {
            return Err(format!(
                "invocation {index} (lane {lane} of {size}): prefix sum {}, expected {through}",
                word[3]
            ));
        }
        smallest = smallest.min(size);
        largest = largest.max(size);
        if lane == 0 {
            subgroups += 1;
            ballot_total += u64::from(word[2]);
        }
        if lane == size - 1 || index + 1 == data.len() {
            // The last lane of a subgroup holds its sum. A subgroup cut by the end of the data ends at the last
            // element, whose prefix is the sum of the elements that exist (the invocations past the end add 0).
            prefix_total += u64::from(word[3]);
        }
    }
    if data.is_empty() {
        return Err("no data to survey".to_owned());
    }
    let host_over = data.iter().filter(|&&x| x > THRESHOLD).count() as u64;
    if ballot_total != host_over {
        return Err(format!(
            "the ballot popcounts add up to {ballot_total}, the data has {host_over} elements over {THRESHOLD}"
        ));
    }
    let host_sum: u64 = data.iter().map(|&x| u64::from(x)).sum();
    if prefix_total != host_sum {
        return Err(format!(
            "the last prefix sums add up to {prefix_total}, the data sums to {host_sum}"
        ));
    }
    Ok(Findings {
        sizes: (smallest, largest),
        invocations: data.len(),
        subgroups,
        ballot_total,
        prefix_total,
    })
}

/// Runs the selected body of `lanes::survey` over `data` on `device` and returns its words.
pub fn survey(device: &HeadlessDevice, data: &[u32]) -> Result<Vec<u32>, Error> {
    let input = device.create_typed_buffer("survey data", data, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer(
        "survey words",
        &vec![0xDEAD_BEEF_u32; data.len() * WORDS],
        GpuBufferAccess::ReadWrite,
    );
    device.dispatch_buffers(
        &lanes::survey::DESCRIPTOR,
        data.len() as u32,
        &[
            BufferBinding::read_only(&input),
            BufferBinding::read_write(&out).independent_length(),
        ],
    )?;
    device.read_typed_buffer(&out)
}

/// Opens the default device (which has the capabilities this build was compiled for), surveys [`input`] and judges
/// the result. Returns the lines the program prints; the last one starts with `subgroup-reduce: OK body=`.
pub fn run() -> Result<Vec<String>, String> {
    let device = HeadlessDevice::with_options(DeviceOptions::default())
        .map_err(|error| format!("no device: {error}"))?;
    let mut lines = vec![
        format!(
            "subgroup-reduce: adapter {} (subgroup size range {:?})",
            device.adapter_info().name,
            device.subgroup_sizes()
        ),
        format!(
            "subgroup-reduce: build capabilities {:?}",
            gust::capability::enabled()
        ),
    ];
    let data = input(ELEMENTS);
    let words = survey(&device, &data).map_err(|error| format!("dispatch failed: {error}"))?;
    if !subgroup_body_selected() {
        if words.iter().any(|&word| word != 0) {
            return Err("the portable body must write zeros".to_owned());
        }
        lines.push(
            "subgroup-reduce: the portable body ran (this build has no subgroups capability): nothing to check; \
             the strict proof is GUST_PROFILE=native+subgroups"
                .to_owned(),
        );
        lines.push("subgroup-reduce: OK body=portable".to_owned());
        return Ok(lines);
    }
    let found = check(&words, &data)?;
    lines.push(format!(
        "subgroup-reduce: lane count {} to {} over {} invocations in {} subgroups",
        found.sizes.0, found.sizes.1, found.invocations, found.subgroups
    ));
    lines.push(format!(
        "subgroup-reduce: ballot popcount: {} elements over {THRESHOLD}, as the host counts them",
        found.ballot_total
    ));
    lines.push(format!(
        "subgroup-reduce: prefix sum: the subgroups' last prefix sums add up to {}, the sum of the data",
        found.prefix_total
    ));
    lines.push("subgroup-reduce: OK body=subgroups".to_owned());
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The words a correct subgroup body writes for `data` with subgroups of `size`.
    fn model(data: &[u32], size: usize) -> Vec<u32> {
        let value = |index: usize| data.get(index).copied().unwrap_or(0);
        let mut words = Vec::new();
        for index in 0..data.len() {
            let (lane, first) = (index % size, index - index % size);
            let over = (first..first + size)
                .filter(|&member| value(member) > THRESHOLD)
                .count();
            let through: u32 = (first..=index).map(value).sum();
            words.extend([size as u32, lane as u32, over as u32, through]);
        }
        words
    }

    #[test]
    fn the_model_is_accepted_for_every_subgroup_size_of_the_adapters_met() {
        let data = input(ELEMENTS);
        for size in [4usize, 8, 16, 32, 64, 128] {
            let found = check(&model(&data, size), &data).unwrap_or_else(|error| {
                panic!("size {size}: {error}");
            });
            assert_eq!(found.sizes, (size, size));
            assert_eq!(found.invocations, ELEMENTS);
            assert_eq!(found.subgroups, ELEMENTS.div_ceil(size));
            assert_eq!(
                found.ballot_total,
                data.iter().filter(|&&x| x > THRESHOLD).count() as u64
            );
            assert_eq!(
                found.prefix_total,
                data.iter().map(|&x| u64::from(x)).sum::<u64>()
            );
        }
    }

    #[test]
    fn a_wrong_word_is_refused_and_named() {
        let data = input(ELEMENTS);
        let good = model(&data, 32);
        for (word, what) in [
            (0usize, "subgroup size"),
            (1, "lane"),
            (2, "ballot popcount"),
            (3, "prefix sum"),
        ] {
            let mut words = good.clone();
            // Invocation 70: lane 6 of the subgroup 64..96.
            words[70 * WORDS + word] ^= 1;
            let error = check(&words, &data).unwrap_err();
            assert!(error.contains("invocation 70"), "{what}: {error}");
        }
        // A size that is not a power of two, a size of 1, and the zeros of the portable body.
        let mut words = good.clone();
        words[0] = 24;
        assert!(
            check(&words, &data)
                .unwrap_err()
                .contains("subgroup size 24")
        );
        assert!(check(&vec![0; good.len()], &data).is_err());
        // Too few words.
        assert!(
            check(&good[..good.len() - 1], &data)
                .unwrap_err()
                .contains("words for")
        );
        assert!(check(&[], &[]).is_err());
    }

    #[test]
    fn the_input_has_elements_on_both_sides_of_the_threshold() {
        let data = input(ELEMENTS);
        assert_eq!(data.len(), ELEMENTS);
        let over = data.iter().filter(|&&x| x > THRESHOLD).count();
        assert!(over > 0 && over < ELEMENTS, "{over} of {ELEMENTS}");
        assert_ne!(
            ELEMENTS % 64,
            0,
            "the last workgroup must be partly past the data"
        );
    }

    #[test]
    fn the_descriptor_says_which_body_the_build_selected() {
        let selected = subgroup_body_selected();
        assert_eq!(
            selected,
            gust::capability::enabled().contains(&Capability::Subgroups)
        );
        assert_eq!(
            lanes::survey::DESCRIPTOR.variant.is_empty(),
            !selected,
            "the portable body has no variant name"
        );
    }

    /// The whole program on the adapter: under a build with the capability the words are checked, otherwise the
    /// portable zeros.
    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn the_selected_body_runs_on_the_adapter() {
        let lines = run().unwrap();
        let last = lines.last().unwrap();
        let expected = if subgroup_body_selected() {
            "subgroup-reduce: OK body=subgroups"
        } else {
            "subgroup-reduce: OK body=portable"
        };
        assert_eq!(last, expected, "{lines:?}");
    }
}

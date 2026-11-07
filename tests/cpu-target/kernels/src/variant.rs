// Capability variants in the production CPU target (M8, slice m8.s; D139, D140): a kernel with a portable body and a
// `requires(subgroups)` variant. Built under `GUST_PROFILE=cpu` the macro selects the portable body, the CPU twin is
// built from it, and the variant's subgroup operations are validated and lowered but never reach the binary: a build
// for the CPU has no subgroup, and the variant has `cpu = "unsupported"` (D42). The two kernels here are the
// `partial_sums` and `count_if` pairs of `tests/fixtures/subgroup.rs`, so the CPU result is the portable body's, the
// one the GPU gives under `dev`. Under another profile (the `cargo check` of the workspace tools) the module is built
// as any other: `requires` only changes which body a build selects.

#[gust::gpu]
pub mod pairs {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn partial_sums(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        if id.x < data.len() {
            out[id.x] = data[id.x];
        }
    }

    #[variant(of = partial_sums, requires(subgroups))]
    pub fn partial_sums_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let total = subgroup_add(x);
        let first = subgroup_elect();
        if id.x < data.len() {
            if first {
                out[id.x] = total;
            } else {
                out[id.x] = 0u32;
            }
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn count_if(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        if id.x < data.len() {
            if data[id.x] > 10u32 {
                out[id.x] = 1u32;
            } else {
                out[id.x] = 0u32;
            }
        }
    }

    #[variant(of = count_if, requires(subgroups))]
    pub fn count_if_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let over = x > 10u32;
        let count = subgroup_count(over);
        let first = subgroup_elect();
        if id.x < data.len() {
            if first {
                out[id.x] = count;
            } else {
                out[id.x] = 0u32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use gust::cpu::CpuDevice;

    use super::pairs::{count_if, partial_sums};

    /// More than one workgroup (64 wide) and not a multiple of it.
    const ELEMENTS: usize = 1000;

    fn data() -> Vec<u32> {
        (0..ELEMENTS as u32).map(|i| (i * 13 + 5) % 23).collect()
    }

    /// The CPU form of a kernel with a variant is the portable body: `partial_sums` copies the data and `count_if` flags
    /// the elements over 10, so the contract (the sum of the output) holds with the portable meaning, three times in a
    /// row on the same device (no state carries over between launches).
    #[test]
    fn the_cpu_form_of_a_kernel_with_a_subgroup_variant_is_the_portable_body() {
        let device = CpuDevice::new();
        let input = data();
        for run in 0..3 {
            let buffer = device.buffer(&input);
            let copied = device.buffer(&vec![u32::MAX; ELEMENTS]);
            partial_sums::dispatch(&device, ELEMENTS as u32, &buffer, &copied).unwrap();
            assert_eq!(copied.read(), input, "partial_sums, run {run}");

            let flags = device.buffer(&vec![u32::MAX; ELEMENTS]);
            count_if::dispatch(&device, ELEMENTS as u32, &buffer, &flags).unwrap();
            let expected: Vec<u32> = input.iter().map(|v| u32::from(*v > 10)).collect();
            assert_eq!(flags.read(), expected, "count_if, run {run}");
            let counted: u32 = flags.read().iter().sum();
            assert_eq!(
                counted as usize,
                input.iter().filter(|v| **v > 10).count(),
                "run {run}"
            );
        }
    }

    /// A kernel with a variant is a kernel like any other for the CPU: its descriptor under the cpu profile names the
    /// portable body (no variant, nothing required) and the variant's Slang is not in the binary.
    #[cfg(gust_cpu_profile)]
    #[test]
    fn the_descriptor_under_the_cpu_profile_names_the_portable_body() {
        use gust::Kernel as _;
        for descriptor in [
            &partial_sums::DESCRIPTOR,
            &count_if::DESCRIPTOR,
        ] {
            assert_eq!(descriptor.variant, "", "{}", descriptor.name);
            assert!(descriptor.requires.is_empty(), "{}", descriptor.name);
            assert!(!descriptor.slang_source.contains("Wave"), "{}", descriptor.name);
        }
    }
}

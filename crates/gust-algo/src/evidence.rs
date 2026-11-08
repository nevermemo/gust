//! The subgroup form of the reduction's per-block step, as evidence (M8, slice m8.e; D140, D146, Opus 14).
//!
//! The shipped per-block step of the `u32` sum is `sum_u32_block` of [`kernels`](crate::kernels): one workgroup of 256
//! lanes owns 8192 elements, each lane adds up to 32 terms, and a tree in workgroup memory (eight rounds, a barrier
//! each) leaves one value per workgroup. [`block_sum_kernels::block_sum`] is that step again as a pair of bodies: the
//! kernel is the portable body (the same shared-memory tree) and `block_sum_subgroups` is the body a build with the
//! `subgroups` capability gets (`GUST_PROFILE=dev+subgroups`): after the lane chain one `subgroup_add`, the first lane
//! of each subgroup writes the subgroup's total to workgroup memory, one barrier, and every lane adds the at most 32
//! totals (`256 / subgroup_size()`, a loop with a literal bound of 32, so the subgroup sizes 8 to 128 are covered:
//! 32 down to 2 totals) before lane 0 writes the workgroup's value.
//!
//! **Limitations (D39: documented, not chased).** The subgroup body is correct for the subgroup sizes 8 to 128 only: the
//! read loop has a literal bound of 32 totals, so a size below 8 (the WARP adapter reports 4) would add only the first
//! 32 of its totals. [`sum_u32`] therefore refuses a device whose reported minimum subgroup size is below 8 when the
//! build selects the subgroup body, rather than return a wrong sum (the body is the one the measurement timed, so it
//! is not changed). The body also indexes the totals by `lid.x / subgroup_size()`, which assumes that a lane's subgroup
//! is `index / size`, the linear mapping every adapter measured so far has shown (`docs/SEMANTICS.md`); this kernel does
//! not check it at run time. On an adapter that reports a range, the guard looks at the range's minimum, so Metal's
//! fixed 4..64 is refused too, whatever the GPU really does.
//!
//! **This is not a variant of a shipped kernel.** The plan's rule (section 3, row m8.e; Opus 14) lets the subgroup form
//! become the `requires(subgroups)` variant of the shipped kernel only where it is faster at 1e6, 1e7 and 1e8 elements
//! on both adapters, judged by the slowest of three runs of the subgroup form against the fastest of three of v1. It
//! was not (the table is in `docs/development/VALIDATION.md`), so [`Variant`](crate::Variant) has no third member,
//! [`crate::variant::shipped`] is unchanged and the public reductions run the portable v1 kernels. [`sum_u32`] runs
//! the pair for the measurement (`examples/sg_ab.rs`) and for the tests, which judge whichever body the build selects
//! against the `i128` reference and against v1: under `dev` the portable body, under `dev+subgroups` the subgroup one.
//!
//! The pair has no CPU form (`cpu = "unsupported"`, as every kernel with workgroup memory, D101).

use gust::Independent;
use gust_wgpu::{BufferDispatch, GpuVec};

use crate::{Error, limits, schedule::BlockSchedule};

#[gust::gpu]
pub mod block_sum_kernels {
    // The block step of `sum_u32`: lane t adds `block + t + 256 * k` for k in 0..32 (the lane chain of v1) and writes
    // `partials[gid]`, the workgroup's total. The portable body is v1's tree. The subgroup body replaces the tree by a
    // subgroup add and one round of per-subgroup totals; it is a barrier kernel for the validator (no `return`, the
    // subgroup operation at the top level, every lane reaches it), and 256 is a multiple of every subgroup size.
    #[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]
    pub fn block_sum(
        _id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<uint>,
        mut partials: RWStructuredBuffer<uint>,
        mut scratch: Workgroup<[u32; 256]>,
    ) {
        let n = input.len();
        let first = gid.x * 8192u32 + lid.x;
        let mut acc = 0u32;
        for k in 0u32..32u32 {
            let index = first + k * 256u32;
            if index >= n {
                break;
            }
            acc += input[index];
        }
        scratch[lid.x] = acc;
        workgroup_barrier();
        for round in 0u32..8u32 {
            let s = 128u32 >> round;
            if lid.x < s {
                scratch[lid.x] += scratch[lid.x + s];
            }
            workgroup_barrier();
        }
        if lid.x == 0u32 {
            partials[gid.x] = scratch[0u32];
        }
    }

    #[variant(of = block_sum, requires(subgroups))]
    pub fn block_sum_subgroups(
        _id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<uint>,
        mut partials: RWStructuredBuffer<uint>,
        mut scratch: Workgroup<[u32; 256]>,
    ) {
        let n = input.len();
        let first = gid.x * 8192u32 + lid.x;
        let mut acc = 0u32;
        for k in 0u32..32u32 {
            let index = first + k * 256u32;
            if index >= n {
                break;
            }
            acc += input[index];
        }
        let total = subgroup_add(acc);
        let size = subgroup_size();
        let lane = subgroup_lane();
        if lane == 0u32 {
            scratch[lid.x / size] = total;
        }
        workgroup_barrier();
        let groups = 256u32 / size;
        let mut sum = 0u32;
        for j in 0u32..32u32 {
            if j < groups {
                sum += scratch[j];
            }
        }
        if lid.x == 0u32 {
            partials[gid.x] = sum;
        }
    }
}

/// The smallest subgroup size the subgroup body of [`block_sum_kernels`] adds up correctly: 256 lanes make 32 subgroup
/// totals at size 8, and the body reads 32 totals.
pub const MIN_SUBGROUP_SIZE: u32 = 8;

/// The host-side guard of the subgroup body: `requires_subgroups` says that the build selected it, `sizes` is
/// `HeadlessDevice::subgroup_sizes`. A device that reports no range is left to the capability error of the dispatch.
/// A reported minimum below [`MIN_SUBGROUP_SIZE`] is refused as a `DeviceLimit` (the existing variant that reads
/// "needs 8 for `min_subgroup_size`, the limit is 4"), whatever the range's maximum is.
pub(crate) fn check_subgroup_sizes(
    requires_subgroups: bool,
    sizes: Option<(u32, u32)>,
) -> Result<(), Error> {
    match sizes {
        Some((minimum, _)) if requires_subgroups && minimum < MIN_SUBGROUP_SIZE => {
            Err(Error::Device(gust_wgpu::Error::DeviceLimit {
                kernel: "gust_algo::evidence".to_owned(),
                limit: "min_subgroup_size",
                requested: u64::from(MIN_SUBGROUP_SIZE),
                maximum: u64::from(minimum),
            }))
        }
        _ => Ok(()),
    }
}

/// The sum of a `u32` vector by [`block_sum_kernels::block_sum`], the body the build selects: the block passes of
/// [`BlockSchedule`] in one batch (one workgroup per 8192 elements, until one workgroup is left), then the read-back of
/// the one-element result. Exact and in the domain only when no partial sum overflows, as the shipped `sum`. The sum of
/// an empty input is `0`.
///
/// # Errors
///
/// `Error::Device` for a device failure, an input past the storage binding limit or a first dispatch past the
/// workgroup limit (before any buffer is taken), or the capability error of a build whose body needs `subgroups` on a
/// device opened without it. A build that selects the subgroup body also gets `Error::Device(gust_wgpu::Error::DeviceLimit)` (limit
/// `min_subgroup_size`, before any buffer is taken) on a device whose reported minimum subgroup size is below
/// [`MIN_SUBGROUP_SIZE`], where the body would add a wrong sum (see the limitations in the module doc).
pub fn sum_u32(input: &GpuVec<u32>) -> Result<u32, Error> {
    if input.is_empty() {
        return Ok(0);
    }
    let schedule = BlockSchedule::new(input.len()).expect("the input is not empty");
    let device = input.device();
    check_subgroup_sizes(
        !block_sum_kernels::block_sum::DESCRIPTOR.requires.is_empty(),
        device.subgroup_sizes(),
    )?;
    let passes = schedule.passes();
    let workgroups = passes.first().map_or(1, |pass| pass.workgroups);
    limits::check(
        device,
        "gust_algo::evidence",
        workgroups,
        input.len(),
        size_of::<u32>(),
    )?;
    let partials = passes[..passes.len() - 1]
        .iter()
        .map(|pass| GpuVec::<u32>::zeroed_on(device, pass.workgroups))
        .collect::<Result<Vec<_>, _>>()?;
    let out = GpuVec::<u32>::zeroed_on(device, 1)?;
    let mut calls = Vec::with_capacity(passes.len());
    let mut source = input;
    for (index, pass) in passes.iter().enumerate() {
        let destination = if index + 1 == passes.len() {
            &out
        } else {
            &partials[index]
        };
        calls.push(block_sum_kernels::block_sum::call(
            pass.invocations(),
            Independent(source),
            Independent(destination),
        ));
        source = destination;
    }
    let batch: Vec<BufferDispatch<'_>> = calls.iter().map(BufferDispatch::from).collect();
    device.dispatch_batch(&batch)?;
    Ok(out.to_vec()?[0])
}

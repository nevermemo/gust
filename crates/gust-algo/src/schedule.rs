//! The schedules of `reduce` (plan section 5.2, [`Schedule`]) and of `scan` (section 5.3, [`ScanSchedule`]).
//!
//! A reduction: how many block-strided passes an input of `n` elements takes, and
//! how long a chain of additions each lane of each pass evaluates (the numbers the float bound of [`crate::bound`]
//! is built from).
//!
//! A pass over `N` elements launches `W = ceil(N / BLOCK)` workgroups of [`LANES`] lanes and writes `LANES * W`
//! partials, one per lane; a remaining `N <= LANES` is reduced by the terminal kernel, one invocation, into the result.
//! So `n <= 256` is a single dispatch, `n <= 65 536` two, and an input of 2^29 elements (a 2 GiB binding) four.

/// Lanes of a workgroup of the pass kernels.
pub const LANES: u32 = 256;
/// Elements a workgroup owns: lane `t` reads `block + t + LANES * k` for `k` in `0..CHAIN`.
pub const BLOCK: u32 = 65_536;
/// Most terms one lane of a pass adds (and the terminal kernel's limit): `BLOCK / LANES`.
pub const CHAIN: u32 = BLOCK / LANES;

/// One block-strided pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    /// Elements the pass reads (the length of its input).
    pub input_len: u32,
    /// Workgroups of [`LANES`] lanes it launches: `ceil(input_len / BLOCK)`.
    pub workgroups: u32,
}

impl Pass {
    /// The block-strided launch over `input_len` elements that the element-wise kernels of `compact` use: the same
    /// workgroups of [`LANES`] lanes, each lane visiting up to [`CHAIN`] elements of its block, so the invocation count
    /// ([`partials`](Self::partials)) stays inside the 1D workgroup ceiling for every input the binding limit admits.
    pub const fn over(input_len: u32) -> Self {
        Self {
            input_len,
            workgroups: input_len.div_ceil(BLOCK),
        }
    }

    /// Partials it writes, which is its invocation count: one per lane.
    pub const fn partials(self) -> u32 {
        self.workgroups * LANES
    }

    /// The most terms any lane of the pass adds (a lane starts from the identity, so its first term costs no rounding):
    /// lane 0 of workgroup 0 reads `0, LANES, 2 * LANES, ...` below `input_len`, capped at [`CHAIN`].
    pub const fn longest_chain(self) -> u32 {
        let terms = self.input_len.div_ceil(LANES);
        if terms < CHAIN { terms } else { CHAIN }
    }
}

/// The dispatches of one reduction: the block-strided passes, then the terminal kernel over `final_len` values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    passes: Vec<Pass>,
    final_len: u32,
}

impl Schedule {
    /// The schedule for `n` elements, or `None` for `n == 0` (an empty input has no dispatch at all).
    pub fn new(n: u32) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let mut passes = Vec::new();
        let mut remaining = n;
        while remaining > LANES {
            // `remaining <= u32::MAX`, so the workgroup count is at most 2^16 and the partial count at most 2^24.
            let workgroups = u32::try_from(u64::from(remaining).div_ceil(u64::from(BLOCK)))
                .expect("at most 2^16 workgroups");
            let pass = Pass {
                input_len: remaining,
                workgroups,
            };
            remaining = pass.partials();
            passes.push(pass);
        }
        Some(Self {
            passes,
            final_len: remaining,
        })
    }

    /// The block-strided passes, in order.
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }

    /// The values the terminal kernel combines: the last pass's partials, or the input itself when `n <= LANES`.
    pub const fn final_len(&self) -> u32 {
        self.final_len
    }

    /// Dispatches of the batch: the passes and the terminal kernel.
    pub fn dispatches(&self) -> usize {
        self.passes.len() + 1
    }

    /// The longest chain of additions per level, in order: each pass's, then the terminal kernel's (`final_len`).
    pub fn chains(&self) -> Vec<u32> {
        let mut chains: Vec<u32> = self
            .passes
            .iter()
            .map(|pass| pass.longest_chain())
            .collect();
        chains.push(self.final_len);
        chains
    }
}

/// Elements one invocation of a scan kernel owns (the chunk width of plan section 5.3).
pub const CHUNK: u32 = 256;

/// The levels of one scan (plan section 5.3): the input is cut into chunks of [`CHUNK`] elements, one invocation each,
/// and the chunk totals are scanned the same way until at most [`CHUNK`] are left.
///
/// `sums()[j]` is the length of the totals buffer of level `j + 1`: `ceil(n / 256)`, then `ceil(that / 256)`, and so on,
/// the last one at most 256 long. With `k` levels the batch is the level-0 scan, `k - 1` scans of totals, the top scan,
/// `k - 1` add-backs into totals and the final add-back: `2k + 1` dispatches. That is 3 for `n <= 65 536`, 5 for
/// `n <= 2^24` and 7 above it, up to the end of the `u32` range (`2^32 - 1` elements has levels of `2^24`, `2^16` and
/// 256 totals).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanSchedule {
    sums: Vec<u32>,
}

impl ScanSchedule {
    /// The schedule for `n` elements, or `None` for `n == 0` (an empty input has no dispatch at all).
    pub fn new(n: u32) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let mut sums = vec![n.div_ceil(CHUNK)];
        while sums[sums.len() - 1] > CHUNK {
            sums.push(sums[sums.len() - 1].div_ceil(CHUNK));
        }
        Some(Self { sums })
    }

    /// The lengths of the totals buffers, level 1 first. The first is also the invocation count of the level-0 scan and
    /// of the final add-back; the last is what the top scan handles (at most [`CHUNK`]).
    pub fn sums(&self) -> &[u32] {
        &self.sums
    }

    /// Dispatches of the batch: `2k + 1` for `k` levels of totals.
    pub fn dispatches(&self) -> usize {
        2 * self.sums.len() + 1
    }

    /// The workgroups of the widest dispatch (level 0 and the final add-back, one invocation per chunk): the number a
    /// device's per-dimension workgroup limit has to admit.
    pub fn workgroups(&self) -> u32 {
        self.sums[0].div_ceil(LANES)
    }
}

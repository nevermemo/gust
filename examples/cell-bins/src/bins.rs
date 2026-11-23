//! Binning elements by the cell of a `side x side` grid they lie in, rebuilt every step into the same buffers (m11.bins).
//!
//! The bins are a counting sort of the elements by cell, four steps:
//!
//! 1. `clear`: the per-cell counts and the per-cell cursors to zero.
//! 2. `count`: every element adds one to the count of its cell with `atomic_add`.
//! 3. the exclusive scan of the counts (`gust_algo::scan::exclusive_into`) gives `starts`: the first slot of each cell.
//! 4. `scatter`: every element takes the old value of an `atomic_add` on its cell's cursor as its place inside the cell
//!    and writes its own index to `order[starts[cell] + place]`.
//!
//! A cell is `floor(x) + side * floor(y)` for a position `(x, y)` in `[0, side)` on each axis. A position at or past
//! `side - 1` on either axis (any finite value) is put in the last cell of that axis: the value is clamped in `f32`
//! (a comparison and an assignment) before the conversion to an integer. `+inf` is fed to the same clamp by the test
//! cases (`far_case`, in `gpu_matches_the_host_reference`) but is not promised: `docs/SEMANTICS.md` 15.7.2 lets a GPU
//! assume that no infinity is present, so a program that needs it handled should clamp on the host first. A negative
//! or NaN position is the caller's error (a float to unsigned conversion of it is not defined the same way on every
//! device).
//!
//! # Contract
//!
//! After [`Bins::rebuild`] (or `TwinBins::rebuild`) with `n` elements:
//!
//! - `counts[c]` is the number of elements in cell `c`, and `starts[c]` is the sum of the counts before it;
//! - the elements of cell `c` are `order[starts[c] .. starts[c] + counts[c]]`, each element index exactly once;
//! - the slots of `order` from `n` on are not written, and hold whatever they held before.
//!
//! The *set* of elements in each cell is exact. Their *order* inside a cell is unspecified on the GPU: it is the order
//! in which the atomic adds happen, which depends on the schedule and can differ from one run to the next. Compare per
//! cell as sets (sort the cell's slice first), never the raw order. The CPU twin runs the elements in index order, so
//! its order inside a cell is ascending; do not rely on that either.
//!
//! The bins' own buffers (the counts, the cursors, the starts and the order) are made once, by [`Bins::new`], and every
//! rebuild overwrites all of the counts, the cursors and the starts (and the first `n` slots of the order). The scan
//! takes its temporaries (one per level of totals) from the device pool: the first rebuild may miss the pool and
//! allocate them, and a second `exclusive_into` of the same size takes no new device buffer (a test in `gust-algo`
//! asserts that the pool's miss counter does not move, at one size, 70,001 elements, on the Radeon 890M only; nothing
//! in this crate measures it).

#[cfg(any(test, gust_dual))]
use gust::cpu::{CpuBuffer, CpuDevice, CpuUniform};
use gust::{Independent, gpu_module};
use gust_wgpu::{DeviceHandle, GpuUniform, GpuVec, HeadlessDevice};

#[gpu_module]
mod binning {
    /// The grid's side, then three pad fields: a uniform is `Std140V1`, a struct is rounded up to 16 bytes and may
    /// have no padding, so one `u32` is written with three explicit pad fields.
    pub struct Params {
        pub side: u32,
        pub pad0: u32,
        pub pad1: u32,
        pub pad2: u32,
    }

    /// The cell index of the position `(x, y)` on a grid of `side` cells per side: `floor(x) + side * floor(y)`, each
    /// axis clamped to the last cell in `f32`, before the conversion (which is not defined for 2^32 or more, or
    /// infinity). `side` is at least 1 and at most 65535 (its square fits in `u32`, and it is exact in `f32`).
    pub fn cell(x: f32, y: f32, side: u32) -> u32 {
        let last = (side - 1u32) as f32;
        let mut fx = x;
        if fx > last {
            fx = last;
        }
        let mut fy = y;
        if fy > last {
            fy = last;
        }
        (fy as u32) * side + (fx as u32)
    }

    /// Zeroes the counts and the cursors (both have one slot per cell).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn clear(
        id: SV_DispatchThreadID,
        mut counts: RWStructuredBuffer<uint>,
        mut cursor: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < counts.len() {
            counts[i] = 0u32;
            cursor[i] = 0u32;
        }
    }

    /// Adds one to the count of the cell of every element.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn count(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        xs: StructuredBuffer<float>,
        ys: StructuredBuffer<float>,
        mut counts: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < xs.len() {
            atomic_add(&mut counts[cell(xs[i], ys[i], p.side)], 1u32);
        }
    }

    /// Writes every element's index into its cell's slice of `order`: the place inside the slice is the old value of
    /// the atomic add on the cell's cursor, so which element gets which place depends on the schedule.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn scatter(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        xs: StructuredBuffer<float>,
        ys: StructuredBuffer<float>,
        starts: StructuredBuffer<uint>,
        mut cursor: RWStructuredBuffer<uint>,
        mut order: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < xs.len() {
            let c = cell(xs[i], ys[i], p.side);
            let old = atomic_add(&mut cursor[c], 1u32);
            order[starts[c] + old] = i;
        }
    }
}

type Error = Box<dyn std::error::Error>;

/// The number of cells of a `side x side` grid; `side` must be at least 1 and its square must fit in `u32`.
fn grid_cells(side: u32) -> Result<u32, Error> {
    if side == 0 {
        return Err("the grid has no cells (side 0)".into());
    }
    side.checked_mul(side)
        .ok_or_else(|| "the grid has more than u32::MAX cells".into())
}

/// The parameter struct of the kernels for a grid of `side` cells per side.
const fn params(side: u32) -> binning::Params {
    binning::Params {
        side,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// The checks of one rebuild, shared by the device bins and the twin: equal lengths, at most `capacity`.
fn check_elements(xs: usize, ys: usize, capacity: usize) -> Result<(), Error> {
    if xs != ys {
        return Err(format!("{xs} x positions for {ys} y positions").into());
    }
    if xs > capacity {
        return Err(format!("{xs} elements for bins of capacity {capacity}").into());
    }
    Ok(())
}

/// The bins on a device: the buffers are made once and rebuilt in place. See the module documentation for the contract.
pub struct Bins {
    handle: DeviceHandle,
    params: GpuUniform<binning::Params>,
    counts: GpuVec<u32>,
    starts: GpuVec<u32>,
    cursor: GpuVec<u32>,
    order: GpuVec<u32>,
}

impl Bins {
    /// Bins for a `side x side` grid and up to `capacity` elements, with every buffer allocated.
    pub fn new(handle: &DeviceHandle, side: u32, capacity: u32) -> Result<Self, Error> {
        let cells = grid_cells(side)?;
        Ok(Self {
            handle: DeviceHandle::clone(handle),
            params: handle.create_uniform("cell-bins params", &params(side))?,
            counts: GpuVec::zeroed_on(handle, cells)?,
            starts: GpuVec::zeroed_on(handle, cells)?,
            cursor: GpuVec::zeroed_on(handle, cells)?,
            order: GpuVec::zeroed_on(handle, capacity)?,
        })
    }

    /// The most elements one rebuild takes.
    pub fn capacity(&self) -> u32 {
        self.order.len()
    }

    /// Rebuilds the bins from the positions `(xs[i], ys[i])`, which must have the same length, at most
    /// [`capacity`](Self::capacity). Four device steps (clear, count, scan, scatter); the bins' own buffers are reused
    /// (see the module documentation for the scan's temporaries).
    pub fn rebuild(&mut self, xs: &GpuVec<f32>, ys: &GpuVec<f32>) -> Result<(), Error> {
        check_elements(
            xs.len() as usize,
            ys.len() as usize,
            self.capacity() as usize,
        )?;
        let device: &HeadlessDevice = &self.handle;
        binning::clear::dispatch(device, self.counts.len(), &self.counts, &self.cursor)?;
        binning::count::dispatch(
            device,
            xs.len(),
            &self.params,
            xs,
            ys,
            Independent(&self.counts),
        )?;
        gust_algo::scan::exclusive_into(&self.counts, &self.starts)?;
        binning::scatter::dispatch(
            device,
            xs.len(),
            &self.params,
            xs,
            ys,
            Independent(&self.starts),
            Independent(&self.cursor),
            Independent(&self.order),
        )?;
        Ok(())
    }

    /// The bins on the host: one readback of each buffer. A program reads the bins through this and [`Snapshot`].
    pub fn read(&self) -> Result<Snapshot, Error> {
        Ok(Snapshot {
            counts: self.counts.to_vec()?,
            starts: self.starts.to_vec()?,
            order: self.order.to_vec()?,
        })
    }
}

/// The bins read back to the host.
#[derive(Clone)]
pub struct Snapshot {
    /// The number of elements in each cell.
    pub counts: Vec<u32>,
    /// The first slot of each cell in `order`: the exclusive sum of `counts`.
    pub starts: Vec<u32>,
    /// The element indices grouped by cell; slots from the element count on are stale (see the module documentation).
    pub order: Vec<u32>,
}

impl Snapshot {
    /// The elements of `cell`, in the order the build left them (unspecified: see the module documentation).
    pub fn cell(&self, cell: usize) -> &[u32] {
        let from = self.starts[cell] as usize;
        &self.order[from..from + self.counts[cell] as usize]
    }
}

/// The same four steps on the CPU twin (the kernels' own Rust bodies; the scan is a plain host prefix sum: the
/// twin of `gust_algo::scan` is not available outside that crate's own tests).
#[cfg(any(test, gust_dual))]
pub struct TwinBins {
    device: CpuDevice,
    side: u32,
    params: CpuUniform<binning::Params>,
    counts: CpuBuffer<u32>,
    starts: CpuBuffer<u32>,
    cursor: CpuBuffer<u32>,
    order: CpuBuffer<u32>,
}

#[cfg(any(test, gust_dual))]
impl TwinBins {
    /// Bins for a `side x side` grid and up to `capacity` elements; the same checks as [`Bins::new`].
    pub fn new(side: u32, capacity: u32) -> Result<Self, Error> {
        let device = CpuDevice::new();
        let cells = vec![0u32; grid_cells(side)? as usize];
        Ok(Self {
            params: device.uniform(params(side))?,
            counts: device.buffer(&cells),
            starts: device.buffer(&cells),
            cursor: device.buffer(&cells),
            order: device.buffer(&vec![0u32; capacity as usize]),
            device,
            side,
        })
    }

    /// The twin of [`Bins::rebuild`], with the same checks; the element buffers belong to [`device`](Self::device).
    pub fn rebuild(&mut self, xs: &CpuBuffer<f32>, ys: &CpuBuffer<f32>) -> Result<(), Error> {
        check_elements(xs.len(), ys.len(), self.order.len())?;
        let n = xs.len() as u32;
        let cells = self.side * self.side;
        binning::clear::dispatch(&self.device, cells, &self.counts, &self.cursor)?;
        binning::count::dispatch(
            &self.device,
            n,
            &self.params,
            xs,
            ys,
            Independent(&self.counts),
        )?;
        let starts: Vec<u32> = self
            .counts
            .read()
            .iter()
            .scan(0u32, |total, &count| {
                let start = *total;
                *total += count;
                Some(start)
            })
            .collect();
        self.starts.write(&starts)?;
        binning::scatter::dispatch(
            &self.device,
            n,
            &self.params,
            xs,
            ys,
            Independent(&self.starts),
            Independent(&self.cursor),
            Independent(&self.order),
        )?;
        Ok(())
    }

    /// The device the element buffers of [`rebuild`](Self::rebuild) belong to.
    pub const fn device(&self) -> &CpuDevice {
        &self.device
    }

    /// The bins on the host.
    pub fn read(&self) -> Snapshot {
        Snapshot {
            counts: self.counts.read(),
            starts: self.starts.read(),
            order: self.order.read(),
        }
    }
}

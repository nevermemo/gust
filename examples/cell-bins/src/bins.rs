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
//! A cell is `floor(x) + side * floor(y)` for a position `(x, y)` in `[0, side)` on each axis. A position past the last
//! cell on either axis is put in the last cell of that axis; a negative or NaN position is the caller's error (a float
//! to unsigned conversion of it is not defined the same way on every device).
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
//! A rebuild allocates no device buffer: the counts, the cursors, the starts and the order are made once, by
//! [`Bins::new`], and every rebuild overwrites all of the counts, the cursors and the starts (and the first `n` slots
//! of the order). The bins are not safe to rebuild from two threads at once.

#[cfg(any(test, gust_dual))]
use gust::cpu::{CpuBuffer, CpuDevice};
use gust::{Independent, gpu};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice};

#[gpu]
mod binning {
    /// The cell index of the position `(x, y)` on a grid of `side` cells per side: `floor(x) + side * floor(y)`, each
    /// axis clamped to the last cell.
    pub fn cell(x: f32, y: f32, side: u32) -> u32 {
        let mut cx = x as u32;
        if cx >= side {
            cx = side - 1u32;
        }
        let mut cy = y as u32;
        if cy >= side {
            cy = side - 1u32;
        }
        cy * side + cx
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

    /// Adds one to the count of the cell of every element. `params[0]` is the side.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn count(
        id: SV_DispatchThreadID,
        params: StructuredBuffer<uint>, // side; bound with `Independent`
        xs: StructuredBuffer<float>,
        ys: StructuredBuffer<float>,
        mut counts: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < xs.len() {
            atomic_add(&mut counts[cell(xs[i], ys[i], params[0u32])], 1u32);
        }
    }

    /// Writes every element's index into its cell's slice of `order`: the place inside the slice is the old value of
    /// the atomic add on the cell's cursor, so which element gets which place depends on the schedule.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn scatter(
        id: SV_DispatchThreadID,
        params: StructuredBuffer<uint>, // side; bound with `Independent`
        xs: StructuredBuffer<float>,
        ys: StructuredBuffer<float>,
        starts: StructuredBuffer<uint>,
        mut cursor: RWStructuredBuffer<uint>,
        mut order: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < xs.len() {
            let c = cell(xs[i], ys[i], params[0u32]);
            let old = atomic_add(&mut cursor[c], 1u32);
            order[starts[c] + old] = i;
        }
    }
}

/// The bins on a device: the buffers are made once and rebuilt in place. See the module documentation for the contract.
pub struct Bins {
    handle: DeviceHandle,
    params: GpuVec<u32>,
    counts: GpuVec<u32>,
    starts: GpuVec<u32>,
    cursor: GpuVec<u32>,
    order: GpuVec<u32>,
}

impl Bins {
    /// Bins for a `side x side` grid and up to `capacity` elements, with every buffer allocated.
    pub fn new(
        handle: &DeviceHandle,
        side: u32,
        capacity: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cells = side
            .checked_mul(side)
            .ok_or("the grid has more than u32::MAX cells")?;
        Ok(Self {
            handle: DeviceHandle::clone(handle),
            params: GpuVec::from_slice_on(handle, &[side])?,
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
    /// [`capacity`](Self::capacity). Four device steps (clear, count, scan, scatter), no allocation of a device buffer.
    pub fn rebuild(
        &self,
        xs: &GpuVec<f32>,
        ys: &GpuVec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if xs.len() != ys.len() {
            return Err(format!("{} x positions for {} y positions", xs.len(), ys.len()).into());
        }
        if xs.len() > self.capacity() {
            return Err(format!(
                "{} elements for bins of capacity {}",
                xs.len(),
                self.capacity()
            )
            .into());
        }
        let device: &HeadlessDevice = &self.handle;
        binning::clear::dispatch(device, self.counts.len(), &self.counts, &self.cursor)?;
        binning::count::dispatch(
            device,
            xs.len(),
            Independent(&self.params),
            xs,
            ys,
            Independent(&self.counts),
        )?;
        gust_algo::scan::exclusive_into(&self.counts, &self.starts)?;
        binning::scatter::dispatch(
            device,
            xs.len(),
            Independent(&self.params),
            xs,
            ys,
            Independent(&self.starts),
            Independent(&self.cursor),
            Independent(&self.order),
        )?;
        Ok(())
    }

    /// The bins on the host: one readback of each buffer. (A program that reads the bins in a kernel of its own
    /// binds `counts`, `starts` and `order` instead; they are the fields of this struct.)
    pub fn read(&self) -> Result<Snapshot, Box<dyn std::error::Error>> {
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
    pub counts: Vec<u32>,
    pub starts: Vec<u32>,
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
    params: CpuBuffer<u32>,
    counts: CpuBuffer<u32>,
    starts: CpuBuffer<u32>,
    cursor: CpuBuffer<u32>,
    order: CpuBuffer<u32>,
}

#[cfg(any(test, gust_dual))]
impl TwinBins {
    /// Bins for a `side x side` grid and up to `capacity` elements.
    pub fn new(side: u32, capacity: u32) -> Self {
        let device = CpuDevice::new();
        let cells = vec![0u32; (side * side) as usize];
        Self {
            params: device.buffer(&[side]),
            counts: device.buffer(&cells),
            starts: device.buffer(&cells),
            cursor: device.buffer(&cells),
            order: device.buffer(&vec![0u32; capacity as usize]),
            device,
            side,
        }
    }

    /// The twin of [`Bins::rebuild`]; the element buffers belong to [`device`](Self::device).
    pub fn rebuild(
        &self,
        xs: &CpuBuffer<f32>,
        ys: &CpuBuffer<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = xs.len() as u32;
        let cells = self.side * self.side;
        binning::clear::dispatch(&self.device, cells, &self.counts, &self.cursor)?;
        binning::count::dispatch(
            &self.device,
            n,
            Independent(&self.params),
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
            Independent(&self.params),
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

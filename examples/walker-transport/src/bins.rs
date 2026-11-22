//! Bins of elements by the integer cell each one lies in, rebuilt every step into the same buffers.
//!
//! This is the counting sort of `examples/cell-bins` (`src/bins.rs`), copied here and adapted: the elements are cell
//! indices (`u32`, each below the number of cells) instead of `f32` positions, so there is no cell arithmetic and no
//! side. Four device steps:
//!
//! 1. `clear`: the per-cell counts and the per-cell cursors to zero.
//! 2. `count`: every element adds one to the count of its cell with `atomic_add`.
//! 3. the exclusive scan of the counts (`gust_algo::scan::exclusive_into`) gives `starts`: the first slot of each cell.
//! 4. `scatter`: every element takes the old value of an `atomic_add` on its cell's cursor as its place inside the cell
//!    and writes its own index to `order[starts[cell] + place]`.
//!
//! # Contract
//!
//! After [`Bins::rebuild`] (or `TwinBins::rebuild`) with `n` elements, each cell below the number of cells:
//!
//! - `counts[c]` is the number of elements in cell `c`, and `starts[c]` is the sum of the counts before it;
//! - the elements of cell `c` are `order[starts[c] .. starts[c] + counts[c]]`, each element index exactly once;
//! - the slots of `order` from `n` on are not written.
//!
//! The *set* of elements in each cell is exact. Their *order* inside a cell is unspecified on the GPU (the order in
//! which the atomic adds happen), so [`Snapshot::grouped`] sorts each cell's slice before anything is compared. A cell
//! index at or past the number of cells is the caller's error: the kernels do not check it.

#[cfg(any(test, gust_dual))]
use gust::cpu::{CpuBuffer, CpuDevice};
use gust::{Independent, gpu};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice};

#[gpu]
mod binning {
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
        cells: StructuredBuffer<uint>,
        mut counts: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < cells.len() {
            atomic_add(&mut counts[cells[i]], 1u32);
        }
    }

    /// Writes every element's index into its cell's slice of `order`: the place inside the slice is the old value of
    /// the atomic add on the cell's cursor, so which element gets which place depends on the schedule.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn scatter(
        id: SV_DispatchThreadID,
        cells: StructuredBuffer<uint>,
        starts: StructuredBuffer<uint>,
        mut cursor: RWStructuredBuffer<uint>,
        mut order: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < cells.len() {
            let c = cells[i];
            let old = atomic_add(&mut cursor[c], 1u32);
            order[starts[c] + old] = i;
        }
    }
}

type Error = Box<dyn std::error::Error>;

/// The check of one rebuild, shared by the device bins and the twin: at most `capacity` elements.
fn check_elements(elements: usize, capacity: usize) -> Result<(), Error> {
    if elements > capacity {
        return Err(format!("{elements} elements for bins of capacity {capacity}").into());
    }
    Ok(())
}

/// The bins on a device: the buffers are made once and rebuilt in place. See the module documentation for the contract.
pub struct Bins {
    handle: DeviceHandle,
    counts: GpuVec<u32>,
    starts: GpuVec<u32>,
    cursor: GpuVec<u32>,
    order: GpuVec<u32>,
}

impl Bins {
    /// Bins for `cells` cells and up to `capacity` elements, with every buffer allocated.
    pub fn new(handle: &DeviceHandle, cells: u32, capacity: u32) -> Result<Self, Error> {
        Ok(Self {
            handle: DeviceHandle::clone(handle),
            counts: GpuVec::zeroed_on(handle, cells)?,
            starts: GpuVec::zeroed_on(handle, cells)?,
            cursor: GpuVec::zeroed_on(handle, cells)?,
            order: GpuVec::zeroed_on(handle, capacity)?,
        })
    }

    /// The per-cell counts, valid after a rebuild: a program reads them in its own kernels.
    pub const fn counts(&self) -> &GpuVec<u32> {
        &self.counts
    }

    /// Rebuilds the bins from the cell of each element (at most the capacity of them). Four device steps (clear,
    /// count, scan, scatter); the bins' own buffers are reused.
    pub fn rebuild(&mut self, cells: &GpuVec<u32>) -> Result<(), Error> {
        check_elements(cells.len() as usize, self.order.len() as usize)?;
        let device: &HeadlessDevice = &self.handle;
        binning::clear::dispatch(device, self.counts.len(), &self.counts, &self.cursor)?;
        binning::count::dispatch(device, cells.len(), cells, Independent(&self.counts))?;
        gust_algo::scan::exclusive_into(&self.counts, &self.starts)?;
        binning::scatter::dispatch(
            device,
            cells.len(),
            cells,
            Independent(&self.starts),
            Independent(&self.cursor),
            Independent(&self.order),
        )?;
        Ok(())
    }

    /// The bins on the host: one readback of each buffer.
    pub fn read(&self) -> Result<Snapshot, Error> {
        Ok(Snapshot {
            counts: self.counts.to_vec()?,
            starts: self.starts.to_vec()?,
            order: self.order.to_vec()?,
        })
    }
}

/// The bins read back to the host.
pub struct Snapshot {
    /// The number of elements in each cell.
    pub counts: Vec<u32>,
    /// The first slot of each cell in `order`: the exclusive sum of `counts`.
    pub starts: Vec<u32>,
    /// The element indices grouped by cell, in the order the build left them.
    pub order: Vec<u32>,
}

impl Snapshot {
    /// `order` with every cell's slice sorted ascending, after checking that `starts` is the exclusive sum of `counts`
    /// and that the counts fill `order` exactly (the capacity equals the element count).
    pub fn grouped(&self) -> Result<Vec<u32>, String> {
        let mut grouped = self.order.clone();
        let mut total = 0usize;
        for (cell, (&count, &start)) in self.counts.iter().zip(&self.starts).enumerate() {
            if start as usize != total {
                return Err(format!(
                    "bins: starts[{cell}] is {start}, the counts before it sum to {total}"
                ));
            }
            let end = total + count as usize;
            let slice = grouped.get_mut(total..end).ok_or_else(|| {
                format!(
                    "bins: cell {cell} ends at {end} of {} slots",
                    self.order.len()
                )
            })?;
            slice.sort_unstable();
            total = end;
        }
        if total != grouped.len() {
            return Err(format!(
                "bins: the counts sum to {total} for {} slots",
                grouped.len()
            ));
        }
        Ok(grouped)
    }
}

/// The same four steps on the CPU twin (the kernels' own Rust bodies; the scan is a plain host prefix sum: the
/// twin of `gust_algo::scan` is not available outside that crate's own tests).
#[cfg(any(test, gust_dual))]
pub struct TwinBins {
    device: CpuDevice,
    counts: CpuBuffer<u32>,
    starts: CpuBuffer<u32>,
    cursor: CpuBuffer<u32>,
    order: CpuBuffer<u32>,
}

#[cfg(any(test, gust_dual))]
impl TwinBins {
    /// Bins for `cells` cells and up to `capacity` elements.
    pub fn new(cells: u32, capacity: u32) -> Self {
        let device = CpuDevice::new();
        let zeros = vec![0u32; cells as usize];
        Self {
            counts: device.buffer(&zeros),
            starts: device.buffer(&zeros),
            cursor: device.buffer(&zeros),
            order: device.buffer(&vec![0u32; capacity as usize]),
            device,
        }
    }

    /// The per-cell counts, valid after a rebuild.
    pub const fn counts(&self) -> &CpuBuffer<u32> {
        &self.counts
    }

    /// The twin of [`Bins::rebuild`], with the same check; the element buffer belongs to [`device`](Self::device).
    pub fn rebuild(&mut self, cells: &CpuBuffer<u32>) -> Result<(), Error> {
        check_elements(cells.len(), self.order.len())?;
        let n = cells.len() as u32;
        binning::clear::dispatch(
            &self.device,
            self.counts.len() as u32,
            &self.counts,
            &self.cursor,
        )?;
        binning::count::dispatch(&self.device, n, cells, Independent(&self.counts))?;
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
            cells,
            Independent(&self.starts),
            Independent(&self.cursor),
            Independent(&self.order),
        )?;
        Ok(())
    }

    /// The device the buffers of the program belong to.
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

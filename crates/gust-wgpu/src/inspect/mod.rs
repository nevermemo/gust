//! The buffer inspector (g2.insp): a buffer's elements as a table with the field names the element layout already
//! carries, per-field statistics, a before/after snapshot diff, and the same data as JSON.
//!
//! Start with [`HeadlessDevice::inspect`]. The mechanism is one readback of the asked range (`copy_buffer_to_buffer`
//! of those bytes into a staging buffer, then a mapping) and the [`gust::TypeLayout`] of the element, which names every
//! field and its byte offset. Nothing runs unless a method below is called, and a buffer that is not read-write is
//! refused with [`Error::PersistentBufferNotReadable`] and a hint (see [`InspectError`]).
//!
//! A struct's fields become one column each; a nested struct, a vector, a matrix and an array are flattened into
//! columns named `outer.inner`, `pos.x`, `m[column][row]` and `values[3]`. A scalar buffer has the one column `value`.
//!
//! On native the blocking methods (`table`, `stats`, `snapshot`, `to_json`, and `Display`) wait for the readback; on
//! `wasm32` only the `_async` forms exist, over the same mapping as `read_typed_buffer_async`.

use std::{
    fmt,
    ops::{Bound, RangeBounds},
};

mod diff;
mod stats;
mod table;

pub use diff::{Changes, ElementChange, FieldChange, Snapshot};
pub use stats::{FieldStats, InspectStats};
pub use table::{InspectTable, Value};

use table::columns_of;

use crate::{
    Completion, Error, GpuBuffer, GpuBufferAccess, HeadlessDevice, MappingReady, PendingMapping,
};

/// Why an inspection failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum InspectError {
    /// The readback failed. For a buffer that is not read-write this is [`Error::PersistentBufferNotReadable`], and the
    /// message adds how to make the buffer readable.
    Readback(Error),
    /// The asked element range does not lie inside the buffer, or starts after it ends.
    Range {
        /// The first element asked for.
        start: usize,
        /// One past the last element asked for.
        end: usize,
        /// The buffer's element count.
        len: usize,
    },
    /// The element layout has a kind this release cannot decode into columns.
    Unsupported(&'static str),
}

impl fmt::Display for InspectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Readback(error @ Error::PersistentBufferNotReadable) => write!(
                formatter,
                "{error}; hint: create the buffer with `GpuBufferAccess::ReadWrite` to inspect it",
            ),
            Self::Readback(error) => write!(formatter, "{error}"),
            Self::Range { start, end, len } => write!(
                formatter,
                "inspect range {start}..{end} does not fit a buffer of {len} elements",
            ),
            Self::Unsupported(name) => write!(
                formatter,
                "the element type `{name}` has a layout this release cannot show as columns",
            ),
        }
    }
}

impl std::error::Error for InspectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Readback(error) => Some(error),
            _ => None,
        }
    }
}

impl From<Error> for InspectError {
    fn from(error: Error) -> Self {
        Self::Readback(error)
    }
}

/// The view of one buffer that [`HeadlessDevice::inspect`] returns; narrow it with [`range`](Self::range), then ask
/// for a table, statistics, a snapshot or JSON. Printing it (`{}`) prints the table. Nothing is read until then.
pub struct Inspector<'a, T> {
    device: &'a HeadlessDevice,
    buffer: &'a GpuBuffer<T>,
    start: usize,
    end: usize,
}

impl<T> fmt::Debug for Inspector<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Inspector")
            .field("start", &self.start)
            .field("end", &self.end)
            .finish_non_exhaustive()
    }
}

impl HeadlessDevice {
    /// Look at `buffer`'s elements with their field names. See [`Inspector`] and the [`inspect`](crate::inspect)
    /// module.
    ///
    /// ```no_run
    /// # use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
    /// let device = HeadlessDevice::new()?;
    /// let values = device.create_typed_buffer("values", &[1.0f32, 2.0, 3.0], GpuBufferAccess::ReadWrite);
    /// println!("{}", device.inspect(&values).range(1..3));
    /// let stats = device.inspect(&values).stats()?;
    /// assert_eq!(stats.field("value").unwrap().nan_count, Some(0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn inspect<'a, T>(&'a self, buffer: &'a GpuBuffer<T>) -> Inspector<'a, T> {
        Inspector::new(self, buffer)
    }
}

impl<'a, T> Inspector<'a, T> {
    fn new(device: &'a HeadlessDevice, buffer: &'a GpuBuffer<T>) -> Self {
        Self {
            device,
            buffer,
            start: 0,
            end: buffer.len,
        }
    }

    /// Restrict the view to the elements in `range` (`100..110`, `..4`, `..`). A range that does not fit the buffer
    /// is [`InspectError::Range`] when something is read.
    #[must_use]
    pub fn range(mut self, range: impl RangeBounds<usize>) -> Self {
        self.start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start.saturating_add(1),
            Bound::Unbounded => 0,
        };
        self.end = match range.end_bound() {
            Bound::Included(&end) => end.saturating_add(1),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.buffer.len,
        };
        self
    }

    /// Read the range: one row per element. Blocks; not on `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn table(&self) -> Result<InspectTable, InspectError> {
        let pending = self.start_readback()?;
        if pending.is_some() {
            self.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| Error::Poll(error.to_string()))?;
        }
        let bytes = crate::completion::blocking::block_on_mapping(self.finish_readback(pending))?;
        self.decode(&bytes)
    }

    /// `Inspector::table` without blocking: the async readback of `read_typed_buffer_async`, which on native
    /// completes only while something polls the device.
    pub async fn table_async(&self) -> Result<InspectTable, InspectError> {
        let pending = self.start_readback()?;
        let bytes = self.finish_readback(pending).await?;
        self.decode(&bytes)
    }

    /// Per-field minimum, maximum and NaN count over the range. Blocks; not on `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn stats(&self) -> Result<InspectStats, InspectError> {
        self.table().map(|table| InspectStats::of(&table))
    }

    /// `Inspector::stats` without blocking.
    pub async fn stats_async(&self) -> Result<InspectStats, InspectError> {
        self.table_async()
            .await
            .map(|table| InspectStats::of(&table))
    }

    /// Copy the range now, to compare it with the buffer after a dispatch with [`Snapshot::changed`]. Blocks; not on
    /// `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn snapshot(&self) -> Result<Snapshot<'a, T>, InspectError> {
        self.table().map(|table| Snapshot {
            buffer: self.buffer,
            table,
        })
    }

    /// `Inspector::snapshot` without blocking.
    pub async fn snapshot_async(&self) -> Result<Snapshot<'a, T>, InspectError> {
        self.table_async().await.map(|table| Snapshot {
            buffer: self.buffer,
            table,
        })
    }

    /// The range as JSON (see [`InspectTable::to_json`]). Blocks; not on `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_json(&self) -> Result<String, InspectError> {
        self.table().map(|table| table.to_json())
    }

    /// `Inspector::to_json` without blocking.
    pub async fn to_json_async(&self) -> Result<String, InspectError> {
        self.table_async().await.map(|table| table.to_json())
    }

    fn decode(&self, bytes: &[u8]) -> Result<InspectTable, InspectError> {
        InspectTable::decode(
            &self.buffer.layout,
            self.start,
            bytes,
            self.end - self.start,
        )
    }

    /// Check the buffer and the range, then copy the range into a staging buffer and request its mapping. `None`
    /// for an empty range, which needs no GPU work.
    fn start_readback(&self) -> Result<Option<PendingMapping>, InspectError> {
        self.device.validate_buffer_device(self.buffer)?;
        if self.buffer.access != GpuBufferAccess::ReadWrite {
            return Err(Error::PersistentBufferNotReadable.into());
        }
        if self.start > self.end || self.end > self.buffer.len {
            return Err(InspectError::Range {
                start: self.start,
                end: self.end,
                len: self.buffer.len,
            });
        }
        // Validate the layout even for an empty range, so the answer does not depend on the range.
        columns_of(&self.buffer.layout)?;
        if self.start == self.end {
            return Ok(None);
        }
        let stride = u64::from(self.buffer.layout.stride);
        let offset = self.start as u64 * stride;
        let size = (self.end - self.start) as u64 * stride;
        let device = &self.device.device;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST inspect readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GUST inspect readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer.raw, offset, &staging, 0, size);
        self.device.submit_counted([encoder.finish()]);
        let mapped = Completion::new();
        let callback = mapped.clone();
        staging.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = callback.complete(result.map_err(|error| error.to_string()));
        });
        Ok(Some(PendingMapping {
            readback: staging,
            mapped,
        }))
    }

    async fn finish_readback(&self, pending: Option<PendingMapping>) -> Result<Vec<u8>, Error> {
        let Some(PendingMapping { readback, mapped }) = pending else {
            return Ok(Vec::new());
        };
        MappingReady {
            device: &self.device.device,
            mapped,
        }
        .await?;
        let view = readback
            .get_mapped_range(..)
            .map_err(|error| Error::Map(error.to_string()))?;
        let bytes = view.to_vec();
        drop(view);
        readback.unmap();
        Ok(bytes)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T> fmt::Display for Inspector<'_, T> {
    /// The table of the range; a failed readback prints `<inspect failed: ...>` (a formatter cannot carry the error;
    /// call [`Inspector::table`] to get it).
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.table() {
            Ok(table) => fmt::Display::fmt(&table, formatter),
            Err(error) => writeln!(formatter, "<inspect failed: {error}>"),
        }
    }
}

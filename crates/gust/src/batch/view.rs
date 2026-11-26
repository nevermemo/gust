//! A stable view of a batch's buffer, for a caller that reads it on the GPU (decision D261, slice m14.s2v).

use core::sync::atomic::Ordering;
use std::sync::Arc;

use super::{Batch, BatchDevice, Element, Error, Stage};
use crate::GpuPod;

/// A caller's stable view of one buffer of a batch (an input, or the result of a stage, that the build block returned):
/// the device's own buffer ([`MapHost::Buffer`](crate::map::MapHost::Buffer): `GpuBuffer<T>` on the GPU device,
/// `CpuBuffer<T>` on the CPU twin), the element count and extents, and the generation of the batch it was taken at.
/// Made by [`Batch::view`].
///
/// A view keeps its buffer, and that buffer's lease on the device's pool, alive. A `wgpu::BindGroup` over the buffer
/// keeps only the raw GPU buffer alive, not that lease. So **keep the view (and the batch) alive for as long as any
/// bind group built over it**: if both are dropped while the bind group stays, the allocation goes back to the pool,
/// another owner can take it, and the bind group then reads that owner's data. The view is the batch's buffer, not a
/// copy: a frame makes a new one only when it resizes the batch, so the caller keeps one view, and the bind group it
/// built, for as long as [`Batch::generation`] stays what it was. **After a resize** (the generation moved) a view taken
/// before it still holds the old buffer, with its old data and extents, and a bind group built over it is stale: fetch the
/// view again and build the bind group again.
///
/// **What a reader sees.** The buffer is the batch's own (decision D258: one set, frames kept apart by queue order). A
/// reader sees, at its place in queue order, what the last frame recorded before that place left there, whatever
/// `in_flight(n)` is.
///
/// * A pass recorded into the same encoder **after** [`FrameRun::encode`](super::FrameRun::encode) returned reads that
///   frame: its `write`s and the results of all its stages (the final values of an in-place stage).
/// * A pass recorded into that encoder **before** `encode` reads the frame before it: the last frame that reached the
///   queue, inputs included; this frame's `write`s have not run yet.
/// * A pass in a later submission reads the last frame submitted before it (after `submit()` returns, that frame).
/// * A view carries no check that a frame ran (a ticket does). An encoder whose `encode` failed, whose
///   [`Encoded`](super::Encoded) was dropped, or that was reported with a submission that did not carry it must be
///   dropped with every pass recorded in it, as before. A pass in another encoder then reads the batch as D258 leaves
///   it: the values of the last frame that reached the queue.
/// * Compare [`Batch::generation`] after `encode` or `submit` returns and before recording a pass; a view of an older
///   generation is stale: it holds valid memory with old data, never another owner's.
///
/// **Read only.** Bind it as read-only storage. This is a rule, not a check (the binding type is the caller's): a write
/// through a view, or passing [`buffer`](Self::buffer) to a call that writes it, is outside the equality of the two
/// devices (D224) and outside D258, and no diagnostic says so.
///
/// The rule relies on D258's order: submit the frames of a batch in the order they were begun (see
/// [`FrameRun::encode`](super::FrameRun::encode)).
///
/// A view follows its [`Batch`]: fetch it and use it on the thread that built the batch. No `Send` or `Sync` promise
/// is made across platforms.
pub struct View<D: BatchDevice, T: GpuPod> {
    buffer: Arc<D::Buffer<T>>,
    dims: [u32; 2],
    generation: u64,
}

impl<D: BatchDevice, T: GpuPod> Clone for View<D, T> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            dims: self.dims,
            generation: self.generation,
        }
    }
}

impl<D: BatchDevice, T: GpuPod> core::fmt::Debug for View<D, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("View")
            .field("dims", &self.dims)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl<D: BatchDevice, T: GpuPod> View<D, T> {
    /// The device's buffer: `GpuBuffer<T>` on the GPU device (bind it with `GpuBuffer::as_binding`, which covers the
    /// live elements and not the size class), `CpuBuffer<T>` on the CPU twin. Read only (see the type's rule).
    #[must_use]
    pub fn buffer(&self) -> &D::Buffer<T> {
        &self.buffer
    }

    /// The number of live elements (`width * height`), not the size of the allocation. A buffer of no element still has
    /// an allocation of one placeholder element, as GUST's own bindings of an empty buffer do, so the caller reads the
    /// count here (`0`), not from the shader's `arrayLength` (`1`).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.dims[0] as usize * self.dims[1] as usize
    }

    /// Whether there is no element.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The width: the extent along x (the whole length of a 1-D handle).
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.dims[0]
    }

    /// The height: the extent along y (1 for a 1-D handle). A 2-D buffer is row-major.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.dims[1]
    }

    /// The generation of the batch when the view was taken. Stale once [`Batch::generation`] differs.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

impl<D: BatchDevice> Batch<D> {
    /// A stable [`View`] of an input or the result of a stage that the build block returned, for a pass that reads it
    /// on the GPU. It makes no submission and records nothing; the view is the batch's buffer itself.
    ///
    /// A reader sees, at its place in queue order, what the last frame recorded before that place left there, whatever
    /// `in_flight(n)` is: a pass recorded into the same encoder after [`FrameRun::encode`](super::FrameRun::encode)
    /// returned reads that frame (its `write`s and every stage's result), a pass recorded before `encode` reads the
    /// frame before it (this frame's `write`s have not run yet), and a pass in a later submission reads the last frame
    /// submitted before it. A view carries no check that a frame ran: an encoder whose `encode` failed, whose
    /// [`Encoded`](super::Encoded) was dropped, or that was reported with a submission that did not carry it must be
    /// dropped with every pass recorded in it. Compare [`generation`](Self::generation) after `encode` or `submit`
    /// returns and before recording a pass. Bind the buffer as read-only storage; a write through it is outside D224
    /// and D258. Keep the view (and the batch) alive for as long as any bind group built over it: the bind group does
    /// not keep the buffer's pool lease alive. See [`View`] for the full rule.
    ///
    /// Parameters, extent records, and slots are not readable.
    ///
    /// The view holds what the device stores: a handle of a [`GpuPod`] type gives a `View<D, T>`, and a handle of a
    /// `#[derive(GpuEnum)]` enum gives a `View<D, u32>` of its words (the pass reads them as the enum's words; a word
    /// that is no variant is possible only through the raw tier or a device fault, see [`Element`]).
    ///
    /// # Errors
    ///
    /// [`Error::Foreign`] for a handle of another batch, [`Error::NotOutput`] for one the block did not return.
    pub fn view<T: Element, R>(&self, handle: &Stage<T, R>) -> Result<View<D, T::Word>, Error>
    where
        D::Buffer<T::Word>: 'static,
    {
        let (_, buffer, dims) = self.core.output(handle)?;
        Ok(View {
            buffer,
            dims,
            generation: self.generation(),
        })
    }

    /// One counter per batch, constant until a buffer behind a handle is replaced (a resize, or later a slot
    /// replacement): a frame that resizes moves it, by the number of buffers it replaced, before `encode` or `submit`
    /// returns. A caller keeps its views and bind groups, compares this once per frame after `encode` or `submit`
    /// returns, and fetches its views again and rebuilds its bind groups only when it moved. An older view stays valid
    /// memory with the data and extents it had; a bind group built over it is stale after a resize.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.core.generation.load(Ordering::Relaxed)
    }
}

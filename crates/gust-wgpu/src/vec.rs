//! `GpuVec<T>`: a vector that lives on a device (m7.a0).
//!
//! Residency belongs to buffers, not to functions: a `GpuVec` holds its device ([`DeviceHandle`]) and a pooled
//! allocation, hands out a `GpuBuffer` view for typed calls, and returns the allocation to the device's
//! [`BufferPool`](crate::BufferPool) when dropped. The allocation may be larger than the vector; every binding
//! exposes the logical length only.

use std::fmt;

use gust::{Access, GpuPod, LengthRule, Resource};

use crate::{
    BufferBinding, DeviceHandle, Error, GpuBuffer, HeadlessDevice, default_device, pool::Fill,
};

/// A vector of `T` resident on one device, taken from that device's buffer pool and returned to it on drop.
///
/// Create it with [`from_slice`](Self::from_slice) or [`zeroed`](Self::zeroed) (the
/// [default device](crate::default_device)) or their `_on` forms (a device you hold), read it back with
/// [`to_vec`](Self::to_vec) or [`to_vec_async`](Self::to_vec_async), and pass `&gpu_vec` where a typed call takes a
/// buffer: it is a [`Resource`] of [`HeadlessDevice`]. Its length is `u32`, the type of a dispatch's element count.
///
/// All `GpuVec` arguments of one call must share one device (compared by [`HeadlessDevice::id`]);
/// [`check_device`](Self::check_device) is the test, and a mismatch is [`Error::MixedDevices`]. A `GpuVec` is
/// `Send + Sync` on native; on `wasm32` its device handle is an `Rc`, so it is neither.
///
/// The allocation behind a `GpuVec` is a pooled size class, usually larger than `len()` elements. Everything that
/// takes a `GpuVec` (typed calls, [`to_vec`](Self::to_vec)) sees the logical length only. The raw escape hatch
/// ([`as_buffer`](Self::as_buffer) and `GpuBuffer::as_wgpu_buffer`) does not: it is the whole class, with the
/// previous owner's bytes past the logical length, and commands you record in an encoder of your own and submit after
/// the `GpuVec` dropped may run after the pool has cleared or reused the buffer. Submit them before the drop.
pub struct GpuVec<T: GpuPod> {
    buffer: GpuBuffer<T>,
    device: DeviceHandle,
}

impl<T: GpuPod> GpuVec<T> {
    /// Upload `values` to the default device.
    pub fn from_slice(values: &[T]) -> Result<Self, Error> {
        Self::from_slice_on(&default_device()?, values)
    }

    /// Upload `values` to `device`, into a pooled buffer; the logical range is written in full. A slice above
    /// `u32::MAX` elements, or above the device's `max_buffer_size`, is `Error::DeviceLimit`, and an allocation the
    /// adapter cannot serve is `Error::OutOfMemory` (native; the browser applies the pre-check only).
    pub fn from_slice_on(device: &DeviceHandle, values: &[T]) -> Result<Self, Error> {
        check_len(values.len())?;
        let buffer = device.pooled_buffer("GpuVec", values.len(), Fill::Values(values))?;
        Ok(Self {
            buffer,
            device: DeviceHandle::clone(device),
        })
    }

    /// A vector of `len` zeros on the default device.
    pub fn zeroed(len: u32) -> Result<Self, Error> {
        Self::zeroed_on(&default_device()?, len)
    }

    /// A vector of `len` zeros on `device`. A recycled allocation is cleared in a submission of its own before this
    /// returns, so any later read of the vector sees zeros.
    pub fn zeroed_on(device: &DeviceHandle, len: u32) -> Result<Self, Error> {
        let buffer = device.pooled_buffer("GpuVec", len as usize, Fill::Zeroed)?;
        Ok(Self {
            buffer,
            device: DeviceHandle::clone(device),
        })
    }

    /// The number of elements.
    pub fn len(&self) -> u32 {
        u32::try_from(self.buffer.len()).expect("a GpuVec is at most u32::MAX elements long")
    }

    /// Whether there are no elements.
    pub const fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// The device this vector lives on.
    pub const fn device(&self) -> &DeviceHandle {
        &self.device
    }

    /// The id of [`device`](Self::device).
    pub fn device_id(&self) -> u64 {
        self.device.id()
    }

    /// Fail with [`Error::MixedDevices`] unless this vector lives on `device`.
    pub fn check_device(&self, device: &HeadlessDevice) -> Result<(), Error> {
        device.check_same_device(self.device.id())
    }

    /// The vector as the buffer a typed call takes. Its allocation may be larger than `len()` elements, so the
    /// raw buffer of the result is the whole size class: see the type's notes on `as_wgpu_buffer`.
    pub const fn as_buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    /// Copy the elements to the host. Blocks the calling thread until the device has finished the earlier work on
    /// the vector and the copy; native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_vec(&self) -> Result<Vec<T>, Error> {
        self.device.read_pooled(&self.buffer)
    }

    /// Copy the elements to the host without blocking: the copy and the mapping start when the future is first
    /// polled, and each poll polls the device once. On native the future completes only while something polls
    /// the device (see `HeadlessDevice::read_typed_buffer_async`); in the browser a plain `.await` suffices.
    ///
    /// Dropping the future after the mapping was requested is safe: the staging buffer stays with the mapping until
    /// its callback has fired and then returns to the pool; it is never reused while mapped.
    pub async fn to_vec_async(&self) -> Result<Vec<T>, Error> {
        self.device.read_pooled_async(&self.buffer).await
    }
}

/// A vector of more elements than a dispatch can count is a `DeviceLimit` error, like any other too-large request.
fn check_len(len: usize) -> Result<(), Error> {
    if u32::try_from(len).is_err() {
        return Err(Error::DeviceLimit {
            kernel: "GpuVec".to_owned(),
            limit: "element count (u32)",
            requested: len as u64,
            maximum: u64::from(u32::MAX),
        });
    }
    Ok(())
}

impl<T: GpuPod> Resource for GpuVec<T> {
    type Element = T;
    type Device = HeadlessDevice;
    type Kind = gust::kind::Storage;

    fn bind<'resources>(
        &'resources self,
        access: Access,
        length: LengthRule,
    ) -> BufferBinding<'resources> {
        self.buffer.bind(access, length)
    }
}

impl<T: GpuPod> fmt::Debug for GpuVec<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuVec")
            .field("len", &self.buffer.len())
            .field("device", &self.device.id())
            .finish()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    //! The tail of a pooled allocation: a unit test, because reading the raw bytes back needs the wgpu device, which
    //! `HeadlessDevice` keeps private.

    use std::sync::Arc;

    use gust::{
        Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
        ResourceLayout, SourceOrigin,
    };

    use super::GpuVec;
    use crate::{BufferBinding, HeadlessDevice};

    const fn storage(name: &'static str, access: Access, binding: u32) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "uint",
            kind: ParameterKind::Storage,
            access,
            resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
            binding: Some(ResourceBinding { group: 0, binding }),
        }
    }

    /// `add_one(src: RO, dst: RW)`: `dst[i] = src[i] + 1` for `i < dst.len()` as the kernel sees `dst`.
    const ADD_ONE: KernelDescriptor = KernelDescriptor {
        module: "vec_tail_tests",
        name: "add_one",
        entry_point: "vec_tail_add_one",
        shape: gust::Shape::Grid,
        compatible_shapes: &[gust::Shape::Grid],
        workgroup_size: [64, 1, 1],
        parameters: &[
            storage("src", Access::ReadOnly, 0),
            storage("dst", Access::ReadWrite, 1),
        ],
        slang_source: r#"
[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void vec_tail_add_one(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n) { dst[id.x] = src[id.x] + 1u; }
}
"#,
        source: SourceOrigin::UNKNOWN,
        artifacts: &[],
    };

    /// Every byte of `buffer`, read back through a staging buffer of the test's own.
    fn raw_words(device: &HeadlessDevice, buffer: &wgpu::Buffer) -> Vec<u32> {
        let size = buffer.size();
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw tail readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        device.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result.map_err(|error| error.to_string()));
        });
        device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device finishes");
        // Another test thread's poll may have collected this callback: wait for the callback, not for the poll.
        crate::completion::blocking::recv_mapping(&receiver).expect("the staging buffer maps");
        let view = staging
            .get_mapped_range(..)
            .expect("the mapped range of the staging buffer");
        crate::pod_vec_from_bytes(&view).expect("whole words")
    }

    /// m7.a0 review SF3: a pooled allocation larger than its vector keeps the previous owner's bytes past the logical
    /// end (by design; the raw buffer shows them, as documented), and a dispatch over the vector never writes into
    /// that tail, not even with the invocations of the last workgroup (17 000 elements run as 17 024 invocations).
    /// Binding the whole class instead of the logical length lets those invocations write, and this test fails.
    #[test]
    fn a_dispatch_leaves_the_tail_of_a_recycled_allocation_alone() {
        const SENTINEL: u32 = 0xDEAD_BEEF;
        const LOGICAL: usize = 17_000;
        const CLASS_WORDS: usize = 32_768;
        let device = Arc::new(HeadlessDevice::new().expect("Vulkan adapter required"));

        // The previous owner fills the 131 072-byte class with a sentinel and goes away.
        let previous = GpuVec::from_slice_on(&device, &vec![SENTINEL; CLASS_WORDS]).unwrap();
        assert_eq!(previous.as_buffer().as_wgpu_buffer().size(), 131_072);
        drop(previous);

        // The next owner gets the same allocation. `from_slice` writes its logical range only.
        let hits = device.buffer_pool_stats().hits;
        let values = (0..LOGICAL as u32).collect::<Vec<_>>();
        let dst = GpuVec::from_slice_on(&device, &values).unwrap();
        assert_eq!(
            device.buffer_pool_stats().hits,
            hits + 1,
            "the class must be recycled"
        );
        let src = GpuVec::from_slice_on(&device, &values).unwrap();
        device
            .dispatch_buffers(
                &ADD_ONE,
                LOGICAL as u32,
                &[
                    BufferBinding::read_only(src.as_buffer()),
                    BufferBinding::read_write(dst.as_buffer()),
                ],
            )
            .unwrap();

        let words = raw_words(&device, dst.as_buffer().as_wgpu_buffer());
        assert_eq!(words.len(), CLASS_WORDS);
        let expected = values.iter().map(|value| value + 1).collect::<Vec<_>>();
        assert_eq!(&words[..LOGICAL], expected.as_slice(), "the logical range");
        assert!(
            words[LOGICAL..].iter().all(|word| *word == SENTINEL),
            "the tail past the logical length was written by the dispatch"
        );
    }
}

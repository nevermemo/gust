//! `GpuVec` and the buffer pool (m7.a0): logical lengths on pooled allocations, clearing on a zeroed hand-out,
//! idle-bytes eviction, mapping lifetimes, and device identity.
//!
//! The kernels are hand-written descriptors (Slang source included) built for one property each. Expectations come
//! from host loops and from the pool's own counters.

#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use std::{
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Waker},
    thread,
    time::{Duration, Instant},
};

use gust::{
    Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout, SourceOrigin,
};
use gust_wgpu::{BufferBinding, DeviceHandle, Error, GpuVec, HeadlessDevice};

fn device() -> DeviceHandle {
    Arc::new(HeadlessDevice::new().expect("Vulkan adapter required"))
}

/// Drive a future that completes only while the device is polled (each poll of it polls the device once).
fn drive<T>(future: impl Future<Output = T>) -> T {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    let started = Instant::now();
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the readback never completed"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

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

/// `probe_len(src: RO, out: RW)`: invocation 0 writes the length the kernel sees for `src` and `src`'s last element.
const PROBE_LEN: KernelDescriptor = KernelDescriptor {
    module: "gpu_vec_tests",
    name: "probe_len",
    entry_point: "gpu_vec_probe_len",
    shape: gust::Shape::Grid,
    compatible_shapes: &[gust::Shape::Grid],
    workgroup_size: [1, 1, 1],
    parameters: &[
        storage("src", Access::ReadOnly, 0),
        storage("out", Access::ReadWrite, 1),
    ],
    slang_source: r#"
[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> out;
[shader("compute")] [numthreads(1, 1, 1)]
void gpu_vec_probe_len(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; src.GetDimensions(n, stride);
    uint m; uint stride_out; out.GetDimensions(m, stride_out);
    out[0] = n;
    out[1] = src[n - 1u];
    out[2] = m;
}
"#,
    source: SourceOrigin::UNKNOWN,
    artifacts: &[],
};

/// `add_one(src: RO, dst: RW)`: `dst[i] = src[i] + 1` for `i < dst.len()`.
const ADD_ONE: KernelDescriptor = KernelDescriptor {
    module: "gpu_vec_tests",
    name: "add_one",
    entry_point: "gpu_vec_add_one",
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
void gpu_vec_add_one(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n) { dst[id.x] = src[id.x] + 1u; }
}
"#,
    source: SourceOrigin::UNKNOWN,
    artifacts: &[],
};

fn ramp(len: u32) -> Vec<u32> {
    (0..len)
        .map(|index| index.wrapping_mul(2_654_435_761))
        .collect()
}

#[test]
fn a_gpu_vec_is_send_and_sync_on_native() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GpuVec<u32>>();
    assert_send_sync::<GpuVec<f32>>();
    assert_send_sync::<HeadlessDevice>();
    assert_send_sync::<gust_wgpu::BufferPool>();
}

#[test]
fn a_gpu_vec_round_trips_and_knows_its_device() {
    let device = device();
    let values = ramp(1000);
    let vec = GpuVec::from_slice_on(&device, &values).unwrap();
    assert_eq!(vec.len(), 1000);
    assert!(!vec.is_empty());
    assert_eq!(vec.device_id(), device.id());
    assert!(Arc::ptr_eq(vec.device(), &device));
    assert_eq!(vec.to_vec().unwrap(), values);
    assert_eq!(drive(vec.to_vec_async()).unwrap(), values);
    assert_eq!(vec.as_buffer().len(), 1000);
    assert!(format!("{vec:?}").contains("1000"));

    let floats = GpuVec::from_slice_on(&device, &[1.5_f32, -2.0, 0.25]).unwrap();
    assert_eq!(floats.to_vec().unwrap(), [1.5, -2.0, 0.25]);
}

#[test]
fn an_empty_gpu_vec_is_allowed() {
    let device = device();
    let empty = GpuVec::<u32>::from_slice_on(&device, &[]).unwrap();
    assert_eq!((empty.len(), empty.is_empty()), (0, true));
    assert!(empty.to_vec().unwrap().is_empty());
    assert!(drive(empty.to_vec_async()).unwrap().is_empty());
    let zeroed = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    assert!(zeroed.to_vec().unwrap().is_empty());
}

/// The kernel sees the logical length of a vector on a larger pooled allocation, not the class size.
#[test]
fn a_binding_exposes_the_logical_length_not_the_pooled_size() {
    let device = device();
    // 17 000 u32 are 68 000 bytes: above 64 KiB, so the allocation is the 131 072-byte class (32 768 elements).
    let values = ramp(17_000);
    let src = GpuVec::from_slice_on(&device, &values).unwrap();
    assert_eq!(src.as_buffer().as_wgpu_buffer().size(), 131_072);
    let out = GpuVec::<u32>::zeroed_on(&device, 3).unwrap();
    device
        .dispatch_buffers(
            &PROBE_LEN,
            1,
            &[
                BufferBinding::read_only(src.as_buffer()).independent_length(),
                BufferBinding::read_write(out.as_buffer()).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(out.to_vec().unwrap(), [17_000, values[16_999], 3]);
}

/// A dispatch over a pooled vector writes exactly its logical length and the typed length checks use it.
#[test]
fn a_dispatch_over_pooled_vectors_uses_the_logical_length() {
    let device = device();
    let values = ramp(17_000);
    let src = GpuVec::from_slice_on(&device, &values).unwrap();
    let dst = GpuVec::<u32>::zeroed_on(&device, 17_000).unwrap();
    assert_eq!(dst.as_buffer().as_wgpu_buffer().size(), 131_072);
    device
        .dispatch_buffers(
            &ADD_ONE,
            17_000,
            &[
                BufferBinding::read_only(src.as_buffer()),
                BufferBinding::read_write(dst.as_buffer()),
            ],
        )
        .unwrap();
    let expected = values
        .iter()
        .map(|value| value.wrapping_add(1))
        .collect::<Vec<_>>();
    assert_eq!(dst.to_vec().unwrap(), expected);
    // An element count other than the logical length is refused, not read past the end.
    match device.dispatch_buffers(
        &ADD_ONE,
        32_768,
        &[
            BufferBinding::read_only(src.as_buffer()),
            BufferBinding::read_write(dst.as_buffer()),
        ],
    ) {
        Err(Error::BufferLength {
            expected, actual, ..
        }) => {
            assert_eq!((expected, actual), (32_768, 17_000));
        }
        other => panic!("expected BufferLength, got {other:?}"),
    }
}

/// A buffer uploaded, dropped and handed out again as zeroed reads as zeros.
#[test]
fn a_recycled_buffer_handed_out_as_zeroed_reads_zeros() {
    let device = device();
    for (uploaded, zeroed) in [
        (5_000_u32, 5_000_u32),
        (20_000, 17_000),
        (20_000, 20_000),
        (300, 300),
    ] {
        let first =
            GpuVec::from_slice_on(&device, &vec![0xDEAD_BEEF_u32; uploaded as usize]).unwrap();
        let class = first.as_buffer().as_wgpu_buffer().size();
        drop(first);
        let hits = device.buffer_pool_stats().hits;
        let again = GpuVec::<u32>::zeroed_on(&device, zeroed).unwrap();
        assert_eq!(
            device.buffer_pool_stats().hits,
            hits + 1,
            "the allocation of {uploaded} elements ({class} bytes) must be recycled for {zeroed}, \
             or this test proves nothing"
        );
        assert_eq!(again.to_vec().unwrap(), vec![0_u32; zeroed as usize]);
        assert!(again.as_buffer().as_wgpu_buffer().size() >= u64::from(zeroed) * 4);
    }
}

#[test]
fn recycled_buffers_are_reused_by_class_and_counted() {
    let device = device();
    let pool = device.buffer_pool().clone();
    let before = pool.stats();
    let first = GpuVec::from_slice_on(&device, &ramp(70_000)).unwrap(); // 280 000 bytes: 524 288-byte class
    assert_eq!(first.as_buffer().as_wgpu_buffer().size(), 524_288);
    let stats = pool.stats();
    assert_eq!((stats.misses, stats.hits), (before.misses + 1, before.hits));
    assert_eq!(stats.outstanding, before.outstanding + 1);
    drop(first);
    let stats = pool.stats();
    assert_eq!(stats.outstanding, before.outstanding);
    assert_eq!(stats.returned, before.returned + 1);
    assert_eq!(stats.free_bytes, before.free_bytes + 524_288);
    // Another length of the same class takes it back.
    let second = GpuVec::from_slice_on(&device, &ramp(100_000)).unwrap(); // 400 000 bytes: same class
    let stats = pool.stats();
    assert_eq!(stats.hits, before.hits + 1);
    assert_eq!(stats.free_bytes, before.free_bytes);
    // Below 64 KiB the class is the exact size: 1000 u32 do not share a buffer with 1001.
    drop(second);
    let exact = GpuVec::from_slice_on(&device, &ramp(1000)).unwrap();
    assert_eq!(exact.as_buffer().as_wgpu_buffer().size(), 4000);
    drop(exact);
    let hits = pool.stats().hits;
    let other = GpuVec::from_slice_on(&device, &ramp(1001)).unwrap();
    assert_eq!(
        pool.stats().hits,
        hits,
        "a different exact size is a different class"
    );
    assert_eq!(other.as_buffer().as_wgpu_buffer().size(), 4004);
}

#[test]
fn the_pool_drops_the_least_recently_returned_buffer_over_its_cap() {
    // A fresh device so the idle lists hold only what this test returns.
    let device = device();
    let pool = device.buffer_pool().clone();
    let a = GpuVec::<u32>::zeroed_on(&device, 20_000).unwrap(); // 131 072-byte class
    let b = GpuVec::<u32>::zeroed_on(&device, 40_000).unwrap(); // 262 144
    let c = GpuVec::<u32>::zeroed_on(&device, 100_000).unwrap(); // 524 288
    let sizes =
        [a.as_buffer(), b.as_buffer(), c.as_buffer()].map(|buffer| buffer.as_wgpu_buffer().size());
    assert_eq!(sizes, [131_072, 262_144, 524_288]);
    pool.set_max_bytes(sizes[1] + sizes[2]);
    drop(a);
    drop(b);
    assert_eq!(pool.stats().evicted, 0);
    assert_eq!(pool.stats().free_bytes, sizes[0] + sizes[1]);
    drop(c);
    // 917 504 idle bytes against a cap of 786 432: the oldest return (`a`) goes.
    let stats = pool.stats();
    assert_eq!(
        (stats.evicted, stats.free_bytes, stats.free_buffers),
        (1, sizes[1] + sizes[2], 2)
    );
    let hits = stats.hits;
    let a_again = GpuVec::<u32>::zeroed_on(&device, 20_000).unwrap();
    assert_eq!(
        pool.stats().hits,
        hits,
        "the evicted class is allocated again"
    );
    let b_again = GpuVec::<u32>::zeroed_on(&device, 40_000).unwrap();
    let c_again = GpuVec::<u32>::zeroed_on(&device, 100_000).unwrap();
    assert_eq!(
        pool.stats().hits,
        hits + 2,
        "the newer classes were still idle"
    );
    assert_eq!(pool.stats().free_buffers, 0);
    // A buffer larger than the cap is not kept at all.
    pool.set_max_bytes(1000);
    drop((a_again, b_again, c_again));
    let stats = pool.stats();
    assert_eq!((stats.free_bytes, stats.free_buffers), (0, 0));
    assert_eq!(stats.max_bytes, 1000);
    pool.clear();
    assert_eq!(pool.stats().free_buffers, 0);
}

type Readback<'a> = std::pin::Pin<Box<dyn Future<Output = Result<Vec<u32>, Error>> + 'a>>;

/// Start a readback of `vec` and return it while its mapping is still pending: the copy is submitted and the mapping
/// requested, but the first poll has not seen the callback. A readback that is already complete at its first poll is
/// finished and dropped, and the next attempt is made. A device that finishes every copy before the first poll (a
/// software one can) never gives a pending readback: the result is `None` and the caller skips what needs one.
fn start_pending(vec: &GpuVec<u32>) -> Option<Readback<'_>> {
    let mut context = Context::from_waker(Waker::noop());
    for _ in 0..200 {
        let mut readback: Readback<'_> = Box::pin(vec.to_vec_async());
        if readback.as_mut().poll(&mut context).is_pending() {
            return Some(readback);
        }
    }
    eprintln!(
        "every readback finished at its first poll here: the pending-mapping checks are skipped"
    );
    None
}

/// A readback whose future is dropped after the mapping was requested leaves the staging buffer with the mapping's
/// callback, which puts it back; a pending staging buffer is never handed to another readback.
#[test]
fn a_dropped_readback_future_leaves_its_staging_buffer_with_the_mapping() {
    let device = device();
    let pool = device.buffer_pool().clone();
    let values = ramp(1 << 20);
    let source = GpuVec::from_slice_on(&device, &values).unwrap();
    let flush = GpuVec::from_slice_on(&device, &ramp(100)).unwrap();

    // Start a readback (copy submitted, mapping requested) and drop the future without waiting for it.
    let Some(pending) = start_pending(&source) else {
        return;
    };
    let stats = pool.stats();
    assert_eq!(
        stats.outstanding, 3,
        "the two vectors and the pending staging buffer"
    );
    drop(pending);
    // Another blocking readback polls the device with `Wait`, which runs the first mapping's callback.
    assert_eq!(flush.to_vec().unwrap(), ramp(100));
    let stats = pool.stats();
    assert_eq!(
        stats.discarded_mapped, 0,
        "nothing was dropped while mapped"
    );
    assert_eq!(
        stats.outstanding, 2,
        "the source and the flush vector; all staging buffers are back"
    );
    assert_eq!(
        stats.free_buffers, 2,
        "the dropped readback's and the flush's staging buffers are idle: {stats:?}"
    );

    // A readback in flight keeps its staging buffer: a second one cannot be given the same buffer.
    let Some(first) = start_pending(&source) else {
        return;
    };
    let during = pool.stats();
    assert_eq!(during.outstanding, 3);
    assert_eq!(
        during.free_buffers, 1,
        "the pending readback holds the only staging buffer of its class: {during:?}"
    );
    assert_eq!(source.to_vec().unwrap(), values);
    let after = pool.stats();
    assert_eq!(
        after.outstanding, 3,
        "the two vectors and the first readback's staging buffer"
    );
    assert_eq!(
        (after.hits, after.misses),
        (during.hits, during.misses + 1),
        "the second readback allocated its own staging buffer: the first one is still pending"
    );
    assert_eq!(after.free_buffers, 2);
    assert_eq!(drive(first).unwrap(), values);
    assert_eq!(pool.stats().outstanding, 2);
    assert_eq!(pool.stats().discarded_mapped, 0);
}

#[test]
fn resident_arguments_must_share_one_device() {
    let first = device();
    let second = device();
    assert_ne!(first.id(), second.id());
    let vec = GpuVec::from_slice_on(&first, &[1_u32, 2, 3]).unwrap();
    vec.check_device(&first).expect("its own device");
    match vec.check_device(&second) {
        Err(Error::MixedDevices {
            first: asked,
            second: owner,
        }) => {
            assert_eq!((asked, owner), (second.id(), first.id()));
        }
        other => panic!("expected MixedDevices, got {other:?}"),
    }
    first.check_same_device(first.id()).unwrap();
    assert!(matches!(
        first.check_same_device(second.id()),
        Err(Error::MixedDevices { .. })
    ));
    // A dispatch on the other device refuses the buffer as before (`ForeignPersistentBuffer`).
    let out = GpuVec::<u32>::zeroed_on(&second, 3).unwrap();
    match second.dispatch_buffers(
        &ADD_ONE,
        3,
        &[
            BufferBinding::read_only(vec.as_buffer()),
            BufferBinding::read_write(out.as_buffer()),
        ],
    ) {
        Err(Error::ForeignPersistentBuffer) => {}
        other => panic!("expected ForeignPersistentBuffer, got {other:?}"),
    }
    // Reading a vector back through the other device's buffer functions is refused too.
    assert!(matches!(
        second.read_typed_buffer(vec.as_buffer()),
        Err(Error::ForeignPersistentBuffer)
    ));
}

#[test]
fn the_default_device_serves_from_slice_and_zeroed() {
    let values = ramp(4096);
    let vec = GpuVec::from_slice(&values).unwrap();
    assert_eq!(vec.to_vec().unwrap(), values);
    let zeros = GpuVec::<u32>::zeroed(4096).unwrap();
    assert_eq!(zeros.to_vec().unwrap(), vec![0; 4096]);
    assert_eq!(zeros.device_id(), vec.device_id());
    assert!(Arc::ptr_eq(
        zeros.device(),
        &gust_wgpu::default_device().unwrap()
    ));
}

/// `MapHost` for the GPU device: upload, a zeroed output, readback, through the pool.
#[test]
fn the_gpu_device_is_a_map_host() {
    use gust::map::MapHost;
    let device = device();
    let uploaded = MapHost::upload(&*device, &[3_u32, 1, 4, 1, 5]).unwrap();
    assert_eq!(uploaded.len(), 5);
    assert_eq!(
        MapHost::readback(&*device, &uploaded).unwrap(),
        [3, 1, 4, 1, 5]
    );
    let output = MapHost::zeroed::<f32>(&*device, 7).unwrap();
    assert_eq!(MapHost::readback(&*device, &output).unwrap(), [0.0; 7]);
    // The asynchronous readback is reachable through the trait alone, as the web profile's map call needs it.
    async fn round_trip<D: gust::map::MapHost>(
        device: &D,
        values: &[u32],
    ) -> Result<Vec<u32>, D::Error> {
        let buffer = device.upload(values)?;
        device.readback_async(&buffer).await
    }
    assert_eq!(drive(round_trip(&*device, &[9, 8, 7])).unwrap(), [9, 8, 7]);
    // A map error carries the backend error and downcasts to it.
    let error = gust::map::Error::device(Error::PersistentBufferNotReadable);
    assert!(matches!(
        error.downcast_ref::<Error>(),
        Some(Error::PersistentBufferNotReadable)
    ));
}

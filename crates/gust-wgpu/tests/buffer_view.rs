//! `GpuBuffer::view_as`: one buffer, two element types (slice m14.bzview).
//!
//! A view is a borrowed `Copy` lens over the same bytes. The kernels are macro-generated; expectations come from host
//! loops over the words the kernels must have stored, never from running a shader shadow body. The shape under test
//! is a store of whole `UVec4` elements into a `u32` buffer that later stages of the same frame read word by word,
//! and the other way round, in one raw-tier batch (`dispatch_batch`), at a launch the host splits, on a pooled
//! `GpuVec`, and in a staged graph.
//!
//! No in-tree `GpuPod` reaches the stride rule of `view_as` (see
//! `no_pod_type_of_the_tree_trips_the_stride_rule`), and a `GpuBuffer` of a type that does cannot be made
//! (`create_typed_buffer` asserts the storage layout), so the rule is pinned by the library's unit tests on the
//! layouts and, here, through the target side with a hand-written element.

#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

mod common;

use common::{device_handle, device_reported as device};

use gust::{
    GpuPod, Independent, ScalarKind, TypeLayout, TypeLayoutKind,
    math::{IVec2, IVec4, Mat2, Mat4, UVec2, UVec4, Vec2, Vec4},
};
use gust_wgpu::{
    BufferBinding, BufferDispatch, BufferViewRule, Error, GpuBuffer, GpuBufferAccess, GpuVec,
    HeadlessDevice, StagedGraph,
};

#[gust::gpu_module]
mod views {
    /// A scatter, as Breeze's store is: invocation `i` writes the quad `j = (7919 i) mod len`, a permutation of the
    /// quads (7919 is prime and no length here is a multiple of it). `quads[j]` holds the words `12 j + 1, 12 j + 4,
    /// 12 j + 7, 12 j + 10`: word `w` of the buffer is `3 w + 1`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn store_quads(id: SV_DispatchThreadID, mut quads: RWStructuredBuffer<UVec4>) {
        let i = id.x;
        if i < quads.len() {
            let j = (i * 7919u32) % quads.len();
            let base = j * 12u32;
            quads[j] = UVec4::new(base + 1u32, base + 4u32, base + 7u32, base + 10u32);
        }
    }

    /// `out[0]` is the length the kernel sees for `quads`.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn view_len(
        id: SV_DispatchThreadID,
        quads: StructuredBuffer<UVec4>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        if id.x == 0u32 {
            out[0] = quads.len();
        }
    }

    /// `out[i] = words[i]`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn read_words(
        id: SV_DispatchThreadID,
        words: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = words[i];
        }
    }

    /// `sums[i]` is the sum of the four words `4 i` to `4 i + 3`; `words` has its own length (four times `sums`'s).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sum_words(
        id: SV_DispatchThreadID,
        words: StructuredBuffer<u32>,
        mut sums: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < sums.len() && i * 4u32 + 3u32 < words.len() {
            let at = i * 4u32;
            sums[i] = words[at] + words[at + 1u32] + words[at + 2u32] + words[at + 3u32];
        }
    }

    /// `words[i] = 3 i + 1`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn store_words(id: SV_DispatchThreadID, mut words: RWStructuredBuffer<u32>) {
        let i = id.x;
        if i < words.len() {
            words[i] = i * 3u32 + 1u32;
        }
    }

    /// `out[4 i + k]` is lane `k` of `quads[i]`; `out` has its own length (four times `quads`'s).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn read_quads(
        id: SV_DispatchThreadID,
        quads: StructuredBuffer<UVec4>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < quads.len() && i * 4u32 + 3u32 < out.len() {
            let quad = quads[i];
            out[i * 4u32] = quad.x;
            out[i * 4u32 + 1u32] = quad.y;
            out[i * 4u32 + 2u32] = quad.z;
            out[i * 4u32 + 3u32] = quad.w;
        }
    }

    /// Reads `src` as words and writes `quads` as quads: bound to one buffer and its view, it is an aliased call.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn mix(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut quads: RWStructuredBuffer<UVec4>,
    ) {
        let i = id.x;
        if i < quads.len() && i * 4u32 + 3u32 < src.len() {
            let at = i * 4u32;
            quads[i] = UVec4::new(src[at], src[at + 1u32], src[at + 2u32], src[at + 3u32]);
        }
    }

    /// `to[i] = from[i]`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn copy_quads(
        id: SV_DispatchThreadID,
        from: StructuredBuffer<UVec4>,
        mut to: RWStructuredBuffer<UVec4>,
    ) {
        let i = id.x;
        if i < to.len() {
            to[i] = from[i];
        }
    }

    /// `out[i] = words[4 i + 2] + quads[i].z`: both are the same word when one buffer is bound twice, read-only.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn both(
        id: SV_DispatchThreadID,
        words: StructuredBuffer<u32>,
        quads: StructuredBuffer<UVec4>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < out.len() && i * 4u32 + 2u32 < words.len() {
            out[i] = words[i * 4u32 + 2u32] + quads[i].z;
        }
    }
}

/// Quads, and so four times as many words, in the shape tests.
const QUADS: u32 = 1000;

/// What `store_quads` and `store_words` leave in word `w`.
const fn word(w: u32) -> u32 {
    3 * w + 1
}

fn expected_words(count: u32) -> Vec<u32> {
    (0..count).map(word).collect()
}

fn expected_sums(quads: u32) -> Vec<u32> {
    (0..quads)
        .map(|quad| (0..4).map(|lane| word(4 * quad + lane)).sum())
        .collect()
}

fn zeros(device: &HeadlessDevice, name: &str, len: u32) -> GpuBuffer<u32> {
    device.create_typed_buffer(name, &vec![0u32; len as usize], GpuBufferAccess::ReadWrite)
}

fn assert_all_equal(actual: &[u32], expected: &[u32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    if let Some(at) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        panic!(
            "{what}: first difference at {at}: got {}, expected {}",
            actual[at], expected[at]
        );
    }
}

/// Breeze's shape, forward, in one batch: whole quads stored through the view, then the same bytes read as words by
/// two other stages of the frame (one with the same extent as the words, one over the quads with an independent
/// length). The view in the first binding list is a temporary: the binding borrows the source, not the view.
fn forward_batch(device: &HeadlessDevice) {
    let slab = zeros(device, "slab", 4 * QUADS);
    let copy = zeros(device, "copy", 4 * QUADS);
    let sums = zeros(device, "sums", QUADS);

    let store = [BufferBinding::read_write_view(
        &slab.view_as::<UVec4>().unwrap(),
    )];
    let read = [
        BufferBinding::read_only(&slab),
        BufferBinding::read_write(&copy),
    ];
    let sum = [
        BufferBinding::read_only(&slab).independent_length(),
        BufferBinding::read_write(&sums),
    ];
    device
        .dispatch_batch(&[
            BufferDispatch::new(&views::store_quads::DESCRIPTOR, QUADS, &store),
            BufferDispatch::new(&views::read_words::DESCRIPTOR, 4 * QUADS, &read),
            BufferDispatch::new(&views::sum_words::DESCRIPTOR, QUADS, &sum),
        ])
        .unwrap();

    let words = expected_words(4 * QUADS);
    assert_all_equal(&device.read_typed_buffer(&slab).unwrap(), &words, "slab");
    assert_all_equal(&device.read_typed_buffer(&copy).unwrap(), &words, "copy");
    assert_all_equal(
        &device.read_typed_buffer(&sums).unwrap(),
        &expected_sums(QUADS),
        "sums",
    );
}

#[test]
fn quads_stored_through_a_view_are_read_word_by_word_in_the_same_batch() {
    let device = device();
    forward_batch(&device);
}

/// The other direction, in one batch: words stored through the source, then read back as quads by the view.
fn reverse_batch(device: &HeadlessDevice) {
    let slab = zeros(device, "slab", 4 * QUADS);
    let out = zeros(device, "out", 4 * QUADS);
    let quads = slab.view_as::<UVec4>().unwrap();
    assert_eq!(quads.len(), QUADS as usize);

    let store = [BufferBinding::read_write(&slab)];
    let read = [
        BufferBinding::read_only_view(&quads),
        BufferBinding::read_write(&out).independent_length(),
    ];
    device
        .dispatch_batch(&[
            BufferDispatch::new(&views::store_words::DESCRIPTOR, 4 * QUADS, &store),
            BufferDispatch::new(&views::read_quads::DESCRIPTOR, QUADS, &read),
        ])
        .unwrap();

    let words = expected_words(4 * QUADS);
    assert_all_equal(&device.read_typed_buffer(&slab).unwrap(), &words, "slab");
    assert_all_equal(&device.read_typed_buffer(&out).unwrap(), &words, "out");
}

#[test]
fn words_stored_through_the_source_are_read_as_quads_through_the_view_in_the_same_batch() {
    reverse_batch(&device());
}

/// `device` limited to four workgroups per dimension: 1000 quads are 16 workgroups of 64, so every launch of the
/// shape tests is cut by the host into pieces of its split variant.
fn lowered(device: HeadlessDevice) -> HeadlessDevice {
    let mut limits = device.limits().clone();
    limits.max_compute_workgroups_per_dimension = 4;
    device
        .with_limits(limits)
        .expect("lowering limits is allowed")
}

/// The reverse direction at a split launch: two kernels, each compiled once and only as its split variant.
#[test]
fn the_reverse_store_at_a_split_launch_equals_the_unsplit_one() {
    let lowered = lowered(device());
    reverse_batch(&lowered);
    let stats = lowered.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses), (2, 2), "only the variants");
}

/// The same forward batch where the host splits every launch: 1000 quads are 16 workgroups of 64 and the limit is
/// four, so each of the three dispatches runs as several pieces of its split variant, and only the variants compile.
#[test]
fn the_same_store_at_a_split_launch_equals_the_unsplit_one() {
    let lowered = lowered(device());
    forward_batch(&lowered);
    let stats = lowered.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses),
        (3, 3),
        "three kernels, each compiled once, and only as its split variant"
    );
}

/// The store alone through the split variant, with the typed call and a view (no batch).
#[test]
fn a_typed_call_on_a_view_runs_the_split_variant_too() {
    let lowered = lowered(device());
    let slab = zeros(&lowered, "slab", 4 * QUADS);
    views::store_quads::dispatch(&lowered, QUADS, &slab.view_as::<UVec4>().unwrap()).unwrap();
    assert_all_equal(
        &lowered.read_typed_buffer(&slab).unwrap(),
        &expected_words(4 * QUADS),
        "slab",
    );
    let stats = lowered.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses), (1, 1), "only the variant");
}

/// Typed calls: a view is a `Resource` like a buffer, on a shared and on an independent length.
#[test]
fn a_view_is_a_typed_resource_with_both_length_rules() {
    let device = device();
    let slab = zeros(&device, "slab", 4 * QUADS);
    let sums = zeros(&device, "sums", QUADS);
    let quads = slab.view_as::<UVec4>().unwrap();
    // `Copy`: the store goes through the copy, and the original is still a view to read through afterwards.
    let copy = quads;
    views::store_quads::dispatch(&device, QUADS, &copy).unwrap();
    let again = zeros(&device, "again", 4 * QUADS);
    views::read_quads::dispatch(&device, QUADS, &quads, Independent(&again)).unwrap();
    assert_all_equal(
        &device.read_typed_buffer(&again).unwrap(),
        &expected_words(4 * QUADS),
        "again",
    );
    views::sum_words::dispatch(&device, QUADS, Independent(&slab), &sums).unwrap();
    assert_all_equal(
        &device.read_typed_buffer(&sums).unwrap(),
        &expected_sums(QUADS),
        "sums",
    );
}

#[test]
fn a_buffer_of_quads_viewed_as_words_has_four_times_the_length() {
    let device = device();
    let quads: Vec<UVec4> = (0..10)
        .map(|i| UVec4::new(i, i + 1, i + 2, i + 3))
        .collect();
    let buffer = device.create_typed_buffer("quads", &quads, GpuBufferAccess::ReadWrite);
    let words = buffer.view_as::<u32>().unwrap();
    assert_eq!((words.len(), words.layout().stride), (40, 4));
    let out = zeros(&device, "out", 40);
    views::read_words::dispatch(&device, 40, &words, &out).unwrap();
    let expected: Vec<u32> = (0..10).flat_map(|i| [i, i + 1, i + 2, i + 3]).collect();
    assert_all_equal(&device.read_typed_buffer(&out).unwrap(), &expected, "out");
    assert!(format!("{words:?}").contains("len: 40"));
}

/// A pooled `GpuVec` is viewed over its logical length, not the size class.
#[test]
fn a_gpu_vec_is_viewed_over_its_logical_length_not_its_size_class() {
    let device = device_handle();
    // 17 000 words are 68 000 bytes: above 64 KiB, so the allocation is the 131 072-byte class.
    let words = 17_000u32;
    let slab = GpuVec::<u32>::zeroed_on(&device, words).unwrap();
    assert_eq!(slab.as_buffer().as_wgpu_buffer().size(), 131_072);
    let quads = slab.view_as::<UVec4>().unwrap();
    assert_eq!(quads.len(), 4_250, "not 8 192, the class size in quads");
    // The kernel's own `len()` of the view is the logical length, not the class: the binding's size is the logical bytes.
    let seen = zeros(&device, "seen", 1);
    views::view_len::dispatch(&*device, 1, Independent(&quads), &seen).unwrap();
    assert_eq!(device.read_typed_buffer(&seen).unwrap(), [4_250]);
    views::store_quads::dispatch(&*device, 4_250, &quads).unwrap();
    assert_all_equal(&slab.to_vec().unwrap(), &expected_words(words), "slab");

    // One word more is a tail of four bytes, on the logical length.
    let odd = GpuVec::<u32>::zeroed_on(&device, words + 1).unwrap();
    match odd.view_as::<UVec4>().unwrap_err() {
        Error::BufferViewSize {
            buffer_bytes,
            element,
            stride,
            rule,
        } => {
            assert_eq!((buffer_bytes, element, stride), (68_004, "UVec4", 16));
            assert_eq!(rule, BufferViewRule::Tail { tail: 4 });
        }
        other => panic!("expected BufferViewSize, got {other:?}"),
    }
}

#[test]
fn a_tail_is_refused_with_the_bytes_that_would_be_left_over() {
    let device = device();
    let nine = zeros(&device, "nine", 9);
    let error = nine.view_as::<UVec4>().unwrap_err();
    match &error {
        Error::BufferViewSize {
            buffer_bytes,
            element,
            stride,
            rule,
        } => {
            assert_eq!((*buffer_bytes, *element, *stride), (36, "UVec4", 16));
            assert_eq!(*rule, BufferViewRule::Tail { tail: 4 });
        }
        other => panic!("expected BufferViewSize, got {other:?}"),
    }
    assert_eq!(
        error.to_string(),
        "a buffer of 36 bytes is not a whole number of `UVec4` elements (16 bytes each): 4 bytes would be left over, and a view does not round down"
    );
    // Eight words are exactly two quads, and a `UVec2` view of nine words is the tail of one word.
    assert_eq!(
        zeros(&device, "eight", 8).view_as::<UVec4>().unwrap().len(),
        2
    );
    assert!(matches!(
        nine.view_as::<UVec2>().unwrap_err(),
        Error::BufferViewSize {
            buffer_bytes: 36,
            rule: BufferViewRule::Tail { tail: 4 },
            ..
        }
    ));
    assert_eq!(nine.view_as::<f32>().unwrap().len(), 9);
}

/// A hand-written element whose layout stride (16) differs from its Rust size (12), as the target of a view.
#[derive(Clone, Copy)]
#[repr(C)]
struct Padded([u32; 3]);

// SAFETY: only `LAYOUT` is read, never a buffer of this type: the layout is deliberately wrong, to reach the rule.
unsafe impl GpuPod for Padded {
    const LAYOUT: TypeLayout = TypeLayout {
        name: "Padded",
        size: 12,
        alignment: 4,
        stride: 16,
        kind: TypeLayoutKind::Scalar(ScalarKind::U32),
    };
}

#[test]
fn a_type_whose_stride_differs_from_its_rust_size_is_refused_as_the_stride_rule() {
    let device = device();
    // 12 words are 48 bytes, a whole number of 16-byte strides: only the stride rule can refuse this.
    let twelve = zeros(&device, "twelve", 12);
    let error = twelve.view_as::<Padded>().unwrap_err();
    match &error {
        Error::BufferViewSize {
            buffer_bytes,
            element,
            stride,
            rule,
        } => {
            assert_eq!((*buffer_bytes, *element, *stride), (48, "Padded", 16));
            assert_eq!(*rule, BufferViewRule::Stride { rust_size: 12 });
        }
        other => panic!("expected BufferViewSize, got {other:?}"),
    }
    assert_eq!(
        error.to_string(),
        "`Padded` has a layout stride of 16 bytes but a Rust size of 12: a view needs the same bytes in Rust and on the GPU"
    );
}

/// The claim in the module notes: every layout this tree defines has its stride equal to its Rust size.
#[test]
fn no_pod_type_of_the_tree_trips_the_stride_rule() {
    fn check<T: GpuPod>() {
        let layout = T::LAYOUT;
        assert_eq!(
            layout.stride as usize,
            size_of::<T>(),
            "{} has stride {} but Rust size {}",
            layout.name,
            layout.stride,
            size_of::<T>()
        );
    }
    check::<u32>();
    check::<i32>();
    check::<f32>();
    check::<Vec2>();
    check::<Vec4>();
    check::<IVec2>();
    check::<IVec4>();
    check::<UVec2>();
    check::<UVec4>();
    check::<Mat2>();
    check::<Mat4>();
    check::<[f32; 4]>();
    check::<[UVec4; 3]>();
}

#[test]
fn an_empty_buffer_gives_an_empty_view_that_a_dispatch_with_work_refuses_by_length() {
    let device = device();
    let empty = device.create_typed_buffer::<u32>("empty", &[], GpuBufferAccess::ReadWrite);
    let quads = empty.view_as::<UVec4>().unwrap();
    assert_eq!((quads.len(), quads.is_empty()), (0, true));
    assert_eq!(quads.layout().stride, 16);
    let before = device.gust_submissions();

    // The shared rule: the view's own length (0) against the extent (4), before wgpu sees a binding.
    let bindings = [BufferBinding::read_write_view(&quads)];
    match device
        .dispatch_batch(&[BufferDispatch::new(
            &views::store_quads::DESCRIPTOR,
            4,
            &bindings,
        )])
        .unwrap_err()
    {
        Error::BufferLength {
            parameter,
            expected,
            actual,
        } => assert_eq!((parameter.as_str(), expected, actual), ("quads", 4, 0)),
        other => panic!("expected BufferLength, got {other:?}"),
    }

    // The independent rule: refused too, the GPU would see a placeholder element.
    let independent = [BufferBinding::read_write_view(&quads).independent_length()];
    match device
        .dispatch_batch(&[BufferDispatch::new(
            &views::store_quads::DESCRIPTOR,
            4,
            &independent,
        )])
        .unwrap_err()
    {
        Error::IndependentLengthEmpty { parameter } => assert_eq!(parameter, "quads"),
        other => panic!("expected IndependentLengthEmpty, got {other:?}"),
    }
    // The typed call gives the same errors.
    assert!(matches!(
        views::store_quads::dispatch(&device, 4, &quads).unwrap_err(),
        Error::BufferLength { .. }
    ));
    assert!(matches!(
        views::store_quads::dispatch(&device, 4, Independent(&quads)).unwrap_err(),
        Error::IndependentLengthEmpty { .. }
    ));
    assert_eq!(device.gust_submissions(), before, "nothing was submitted");

    // No work, no binding: an empty extent is accepted and skipped.
    device
        .dispatch_batch(&[BufferDispatch::new(
            &views::store_quads::DESCRIPTOR,
            0,
            &bindings,
        )])
        .unwrap();
    assert_eq!(
        device.gust_submissions(),
        before,
        "an empty extent submits nothing"
    );
}

#[test]
fn a_view_never_widens_the_access_of_its_source() {
    let device = device();
    let words = expected_words(8);
    let source = device.create_typed_buffer("source", &words, GpuBufferAccess::ReadOnly);
    let quads = source.view_as::<UVec4>().unwrap();
    assert_eq!(quads.access(), GpuBufferAccess::ReadOnly);

    // Bound read-write it is refused, as the source would be.
    let bindings = [BufferBinding::read_write_view(&quads)];
    match device
        .dispatch_batch(&[BufferDispatch::new(
            &views::store_quads::DESCRIPTOR,
            2,
            &bindings,
        )])
        .unwrap_err()
    {
        Error::PersistentBufferAccess {
            parameter,
            buffer,
            binding,
        } => {
            assert_eq!(parameter, "quads");
            assert_eq!(buffer, GpuBufferAccess::ReadOnly);
            assert_eq!(binding, gust::Access::ReadWrite);
        }
        other => panic!("expected PersistentBufferAccess, got {other:?}"),
    }

    // Read-only it works: the quads of the words.
    let out = zeros(&device, "out", 8);
    let read = [
        BufferBinding::read_only_view(&quads),
        BufferBinding::read_write(&out).independent_length(),
    ];
    device
        .dispatch_batch(&[BufferDispatch::new(
            &views::read_quads::DESCRIPTOR,
            2,
            &read,
        )])
        .unwrap();
    assert_all_equal(&device.read_typed_buffer(&out).unwrap(), &words, "out");
}

/// To the aliasing rule a view and its source are one buffer.
#[test]
fn a_view_and_its_source_are_one_buffer_to_the_aliasing_rule() {
    let device = device();
    let slab =
        device.create_typed_buffer("slab", &expected_words(4 * 64), GpuBufferAccess::ReadWrite);
    let quads = slab.view_as::<UVec4>().unwrap();

    // One call binding the source read-only and its view read-write.
    let aliased = [
        BufferBinding::read_only(&slab).independent_length(),
        BufferBinding::read_write_view(&quads),
    ];
    match device
        .dispatch_batch(&[BufferDispatch::new(&views::mix::DESCRIPTOR, 64, &aliased)])
        .unwrap_err()
    {
        Error::AliasedBinding {
            kernel,
            first,
            second,
        } => {
            assert!(kernel.ends_with("mix"), "{kernel}");
            assert_eq!((first.as_str(), second.as_str()), ("src", "quads"));
        }
        other => panic!("expected AliasedBinding, got {other:?}"),
    }
    // The same through the typed call.
    assert!(matches!(
        views::mix::dispatch(&device, 64, Independent(&slab), &quads).unwrap_err(),
        Error::AliasedBinding { .. }
    ));

    // Both read-only is accepted: the source and its view in one call read the same bytes.
    let out = zeros(&device, "out", 64);
    let readers = [
        BufferBinding::read_only(&slab).independent_length(),
        BufferBinding::read_only_view(&quads),
        BufferBinding::read_write(&out),
    ];
    device
        .dispatch_batch(&[BufferDispatch::new(&views::both::DESCRIPTOR, 64, &readers)])
        .unwrap();
    let expected: Vec<u32> = (0..64).map(|quad| 2 * word(4 * quad + 2)).collect();
    assert_all_equal(&device.read_typed_buffer(&out).unwrap(), &expected, "out");

    // Two views of one buffer, one of them written, are one buffer as well.
    let other = slab.view_as::<UVec4>().unwrap();
    let two_views = [
        BufferBinding::read_only_view(&other),
        BufferBinding::read_write_view(&quads),
    ];
    match device
        .dispatch_batch(&[BufferDispatch::new(
            &views::copy_quads::DESCRIPTOR,
            64,
            &two_views,
        )])
        .unwrap_err()
    {
        Error::AliasedBinding { first, second, .. } => {
            assert_eq!((first.as_str(), second.as_str()), ("from", "to"));
        }
        other => panic!("expected AliasedBinding, got {other:?}"),
    }
}

/// A staged graph orders a writer through a view and a reader of the source by an edge, as for one element type.
#[test]
fn a_staged_graph_orders_a_writer_through_a_view_and_a_reader_of_the_source() {
    let device = device();
    let slab = zeros(&device, "slab", 4 * QUADS);
    let out = zeros(&device, "out", 4 * QUADS);
    let quads = slab.view_as::<UVec4>().unwrap();
    let store = views::store_quads::call(QUADS, &quads);
    let read = views::read_words::call(4 * QUADS, &slab, &out);

    // Without the edge the reader of the source is a missing dependency on the writer through the view.
    let mut missing = StagedGraph::new();
    let writer = missing.dispatch((&store).into(), &[]);
    let reader = missing.dispatch((&read).into(), &[]);
    assert_eq!((writer.index(), reader.index()), (0, 1));
    match device
        .execute_graph(&missing)
        .err()
        .expect("the reader needs an edge")
    {
        Error::GraphMissingDependency { node, producer } => assert_eq!((node, producer), (1, 0)),
        other => panic!("expected GraphMissingDependency, got {other:?}"),
    }
    assert_all_equal(
        &device.read_typed_buffer(&out).unwrap(),
        &vec![0; 4 * QUADS as usize],
        "nothing ran",
    );

    // Derived edges find it by themselves, and the graph runs.
    let mut graph = StagedGraph::new();
    let writer = graph.run(&store);
    let reader = graph.run(&read);
    assert_eq!(graph.dependency_indices(writer), Vec::<usize>::new());
    assert_eq!(graph.dependency_indices(reader), [0]);
    device.execute_graph(&graph).unwrap();
    let words = expected_words(4 * QUADS);
    assert_all_equal(&device.read_typed_buffer(&out).unwrap(), &words, "out");
    assert_all_equal(&device.read_typed_buffer(&slab).unwrap(), &words, "slab");

    // The declared edge does as well: from zeros again, so the result is this run's own.
    let zero = vec![0; 4 * QUADS as usize];
    device.write_typed_buffer(&slab, &zero).unwrap();
    device.write_typed_buffer(&out, &zero).unwrap();
    let mut declared = StagedGraph::new();
    let writer = declared.dispatch((&store).into(), &[]);
    declared.dispatch((&read).into(), &[writer]);
    device.execute_graph(&declared).unwrap();
    assert_all_equal(
        &device.read_typed_buffer(&out).unwrap(),
        &words,
        "declared out",
    );
    assert_all_equal(
        &device.read_typed_buffer(&slab).unwrap(),
        &words,
        "declared slab",
    );
}

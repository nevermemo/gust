//! The split variant of a `grid` kernel on a GPU (slice m11.g4a, D201, D205).
//!
//! Every `grid` kernel the `#[gpu]` macro builds for the GPU also carries a second descriptor,
//! `K::DESCRIPTOR.split`: the same Slang with one hidden uniform at the next free binding, whose `x`, `y` and `z` are
//! added to the dispatch-thread id before anything reads it. This slice proves that variant by running it on a GPU
//! with a split the TEST does by hand: one `BufferDispatch` per piece, each with its own base uniform, in one batch. The
//! host does not split yet (slice m11.g4b). Each result is compared with an unsplit launch of the kernel's own
//! descriptor on a device with the normal limits, with an independent host reference, and with the CPU twin (a second
//! assertion, never in place of the GPU's).
//!
//! The kernels count every invocation with `atomic_add(&mut hits[i], 1)` and write the group id and the local id they
//! see, so an element that runs twice, never or with another id is visible.

#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use gust::{
    Access, Extent, GpuPod, Independent, KernelDescriptor, LengthRule, ParameterKind, Resource,
    ScalarKind, StructFieldLayout, TypeLayout, cpu::CpuDevice,
};
use gust_wgpu::{BufferBinding, BufferDispatch, GpuBuffer, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod split_kernels {
    // `n` and three explicit padding fields: 16 bytes, no implicit padding.
    pub struct Count {
        pub n: u32,
        pub pad0: u32,
        pub pad1: u32,
        pub pad2: u32,
    }

    pub struct Size2 {
        pub w: u32,
        pub h: u32,
        pub pad0: u32,
        pub pad1: u32,
    }

    /// One dimension, workgroup 8: counts, and records the group and the local id it sees.
    #[kernel(workgroup_size(8, 1, 1))]
    pub fn line(
        id: SV_DispatchThreadID,
        group: GroupId,
        local: LocalId,
        p: Uniform<Count>,
        mut hits: RWStructuredBuffer<u32>,
        mut groups: RWStructuredBuffer<u32>,
        mut locals: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < p.n {
            atomic_add(&mut hits[i], 1u32);
            groups[i] = group.x;
            locals[i] = local.x;
        }
    }

    /// Two dimensions, workgroup 4 x 2.
    #[kernel(workgroup_size(4, 2, 1))]
    pub fn plane(
        id: SV_DispatchThreadID,
        group: GroupId,
        local: LocalId,
        p: Uniform<Size2>,
        mut hits: RWStructuredBuffer<u32>,
        mut groups: RWStructuredBuffer<u32>,
        mut locals: RWStructuredBuffer<u32>,
    ) {
        if id.x < p.w && id.y < p.h {
            let i = id.y * p.w + id.x;
            atomic_add(&mut hits[i], 1u32);
            groups[i] = group.y * 4096u32 + group.x;
            locals[i] = local.y * 16u32 + local.x;
        }
    }

    /// The real workgroup size of the budget kit: 256. `tags[i]` is `group * 256 + local`, which is `i` itself.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn wide(
        id: SV_DispatchThreadID,
        group: GroupId,
        local: LocalId,
        p: Uniform<Count>,
        mut hits: RWStructuredBuffer<u32>,
        mut tags: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < p.n {
            atomic_add(&mut hits[i], 1u32);
            tags[i] = group.x * 256u32 + local.x;
        }
    }
}

use split_kernels::{Count, Size2};

/// What a kernel wrote: the three outputs, element by element.
#[derive(Debug, PartialEq, Eq)]
struct Outputs {
    hits: Vec<u32>,
    groups: Vec<u32>,
    locals: Vec<u32>,
}

const UNWRITTEN: u32 = u32::MAX;
/// The hit count of an element past the extent: the guard keeps the kernel from touching it, so it stays this value.
const SENTINEL: u32 = 7;

/// The hidden uniform of a variant as the host writes it: `x`, `y`, `z` and padding, 16 bytes. Written by hand here,
/// from the descriptor's own words, and not taken from the library: the binding is checked against the layout the
/// reflection of the compiled variant reports.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct LaunchBase {
    x: u32,
    y: u32,
    z: u32,
    pad: u32,
}

// SAFETY: four `u32` fields at offsets 0, 4, 8 and 12 under `repr(C)`, no padding bytes; `align(16)` is the
// alignment of a constant-buffer struct.
unsafe impl GpuPod for LaunchBase {
    const LAYOUT: TypeLayout = {
        const FIELD: TypeLayout = TypeLayout::scalar("u32", ScalarKind::U32);
        const FIELDS: &[StructFieldLayout] = &[
            StructFieldLayout {
                name: "x",
                offset: 0,
                ty: FIELD,
            },
            StructFieldLayout {
                name: "y",
                offset: 4,
                ty: FIELD,
            },
            StructFieldLayout {
                name: "z",
                offset: 8,
                ty: FIELD,
            },
            StructFieldLayout {
                name: "pad",
                offset: 12,
                ty: FIELD,
            },
        ];
        TypeLayout::structure("__GustLaunchBase", 16, 16, FIELDS)
    };
}

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// `device` with every compute-workgroup dimension limited to `workgroups` (`tests/limits.rs` does the same).
fn lowered(device: HeadlessDevice, workgroups: u32) -> HeadlessDevice {
    let mut limits = device.limits().clone();
    limits.max_compute_workgroups_per_dimension = workgroups;
    device
        .with_limits(limits)
        .expect("lowering limits is allowed")
}

fn outputs(
    device: &HeadlessDevice,
    hits: &GpuBuffer<u32>,
    groups: &GpuBuffer<u32>,
    locals: &GpuBuffer<u32>,
) -> Outputs {
    Outputs {
        hits: device.read_typed_buffer(hits).unwrap(),
        groups: device.read_typed_buffer(groups).unwrap(),
        locals: device.read_typed_buffer(locals).unwrap(),
    }
}

/// The three output buffers of `len` elements: `hits` starts as zeros up to `used` and `SENTINEL` after, the others as
/// `UNWRITTEN`.
fn fresh(
    device: &HeadlessDevice,
    used: usize,
    len: usize,
) -> (GpuBuffer<u32>, GpuBuffer<u32>, GpuBuffer<u32>) {
    let hits: Vec<u32> = (0..len)
        .map(|i| if i < used { 0 } else { SENTINEL })
        .collect();
    (
        device.create_typed_buffer("hits", &hits, GpuBufferAccess::ReadWrite),
        device.create_typed_buffer("groups", &vec![UNWRITTEN; len], GpuBufferAccess::ReadWrite),
        device.create_typed_buffer("locals", &vec![UNWRITTEN; len], GpuBufferAccess::ReadWrite),
    )
}

/// The same on the CPU twin.
fn twin_fresh(
    cpu: &CpuDevice,
    used: usize,
    len: usize,
) -> (
    gust::cpu::CpuBuffer<u32>,
    gust::cpu::CpuBuffer<u32>,
    gust::cpu::CpuBuffer<u32>,
) {
    let hits: Vec<u32> = (0..len)
        .map(|i| if i < used { 0 } else { SENTINEL })
        .collect();
    (
        cpu.buffer(&hits),
        cpu.buffer(&vec![UNWRITTEN; len]),
        cpu.buffer(&vec![UNWRITTEN; len]),
    )
}

/// An unsplit launch of `line` over `n` elements through its own descriptor, on `device`.
fn line_unsplit(device: &HeadlessDevice, n: u32, len: usize) -> Outputs {
    let count = device
        .create_uniform(
            "count",
            &Count {
                n,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        )
        .unwrap();
    let (hits, groups, locals) = fresh(device, n as usize, len);
    split_kernels::line::dispatch_extent(
        device,
        Extent::linear(n),
        &count,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    )
    .unwrap();
    outputs(device, &hits, &groups, &locals)
}

/// `line` over `n` elements on the CPU twin.
fn line_twin(n: u32, len: usize) -> Outputs {
    let cpu = CpuDevice::new();
    let count = cpu
        .uniform(Count {
            n,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        })
        .unwrap();
    let (hits, groups, locals) = twin_fresh(&cpu, n as usize, len);
    split_kernels::line::dispatch_extent(
        &cpu,
        Extent::linear(n),
        &count,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    )
    .unwrap();
    Outputs {
        hits: hits.read(),
        groups: groups.read(),
        locals: locals.read(),
    }
}

/// `line` over `n` elements, split by hand into dispatches of the given sizes in workgroups (x only): piece `k` runs
/// the variant over its own invocations with the base of its first one in the hidden uniform. One batch, one pass.
fn line_by_hand(device: &HeadlessDevice, n: u32, len: usize, pieces: &[u32]) -> Outputs {
    const WORKGROUP: u32 = 8;
    let variant = split_kernels::line::DESCRIPTOR
        .split
        .expect("a grid kernel built for the GPU has a split variant");
    let count = device
        .create_uniform(
            "count",
            &Count {
                n,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        )
        .unwrap();
    let (hits, groups, locals) = fresh(device, n as usize, len);
    let mut first = 0u32;
    let mut bases = Vec::new();
    for workgroups in pieces {
        let base = LaunchBase {
            x: first * WORKGROUP,
            y: 0,
            z: 0,
            pad: 0,
        };
        bases.push((device.create_uniform("base", &base).unwrap(), *workgroups));
        first += workgroups;
    }
    assert!(
        first * WORKGROUP >= n,
        "the pieces cover every element of the extent"
    );
    let all: Vec<[BufferBinding<'_>; 5]> = bases
        .iter()
        .map(|(base, _)| {
            [
                count.bind(Access::ReadOnly, LengthRule::Independent),
                hits.bind(Access::ReadWrite, LengthRule::Independent),
                groups.bind(Access::ReadWrite, LengthRule::Independent),
                locals.bind(Access::ReadWrite, LengthRule::Independent),
                base.bind(Access::ReadOnly, LengthRule::Independent),
            ]
        })
        .collect();
    let dispatches: Vec<BufferDispatch<'_>> = bases
        .iter()
        .zip(&all)
        .map(|((_, workgroups), bindings)| {
            BufferDispatch::new_extent(variant, Extent::linear(workgroups * WORKGROUP), bindings)
        })
        .collect();
    device.dispatch_batch(&dispatches).unwrap();
    outputs(device, &hits, &groups, &locals)
}

/// What every element of `line` must hold: counted once, the group `i / 8` and the local id `i % 8`; past `n` untouched.
fn line_expected(n: u32, len: usize) -> Outputs {
    let used = |i: usize| i < n as usize;
    Outputs {
        hits: (0..len)
            .map(|i| if used(i) { 1 } else { SENTINEL })
            .collect(),
        groups: (0..len)
            .map(|i| if used(i) { i as u32 / 8 } else { UNWRITTEN })
            .collect(),
        locals: (0..len)
            .map(|i| if used(i) { i as u32 % 8 } else { UNWRITTEN })
            .collect(),
    }
}

#[test]
fn the_variant_is_an_ordinary_descriptor_with_one_more_uniform() {
    for (base, extra_binding) in [
        (&split_kernels::line::DESCRIPTOR, 4),
        (&split_kernels::plane::DESCRIPTOR, 4),
        (&split_kernels::wide::DESCRIPTOR, 3),
    ] {
        let variant = base.split.expect("a variant");
        assert!(variant.split.is_none(), "a variant has no variant");
        assert!(variant.artifacts.is_empty());
        assert_eq!(
            (
                variant.module,
                variant.name,
                variant.entry_point,
                variant.workgroup_size,
                variant.shape,
                variant.target
            ),
            (
                base.module,
                base.name,
                base.entry_point,
                base.workgroup_size,
                base.shape,
                base.target
            )
        );
        assert_ne!(variant.slang_source, base.slang_source);
        assert!(
            !base.slang_source.contains("launch"),
            "the base is as it was"
        );
        assert!(
            variant
                .slang_source
                .contains(&format!("[[vk::binding({extra_binding}, 0)]] ConstantBuffer<__GustLaunchBase> __gust_launch_base;")),
            "{}",
            variant.slang_source
        );
        let (own, hidden) = variant.parameters.split_at(base.parameters.len());
        assert_eq!(own, base.parameters);
        assert_eq!(hidden.len(), 1);
        assert_eq!(hidden[0].name, "__gust_launch_base");
        assert_eq!(hidden[0].kind, ParameterKind::Uniform);
        assert_eq!(hidden[0].access, Access::ReadOnly);
        assert_eq!(
            hidden[0].binding.map(|b| (b.group, b.binding)),
            Some((0, extra_binding))
        );
        let layout = hidden[0].resource_layout.expect("a layout").element;
        assert_eq!((layout.size, layout.alignment), (16, 16));
        assert_eq!(layout, <LaunchBase as GpuPod>::LAYOUT);
    }
}

/// (a) A device lowered to four workgroups per dimension: 100 elements are 13 workgroups, run as 4 + 4 + 4 + 1.
#[test]
fn a_hand_split_launch_past_the_limit_runs_every_element_once_with_its_own_ids() {
    let n = 100;
    let len = 128;
    let lowered = lowered(device(), 4);
    eprintln!(
        "split (a) adapter: {:?}, max workgroups per dimension {}",
        lowered.adapter_info(),
        lowered.limits().max_compute_workgroups_per_dimension
    );
    let split = line_by_hand(&lowered, n, len, &[4, 4, 4, 1]);
    let expected = line_expected(n, len);
    assert_eq!(split, expected, "the hand split against the host reference");
    assert!(split.hits[..n as usize].iter().all(|hit| *hit == 1));
    assert!(split.hits[n as usize..].iter().all(|hit| *hit == SENTINEL));
    // An unsplit launch of the kernel's own descriptor on a device with the normal limits, and the CPU twin.
    assert_eq!(line_unsplit(&device(), n, len), expected, "unsplit");
    assert_eq!(line_twin(n, len), expected, "the CPU twin");
    // On the lowered device the plain launch is refused today, which is why the test had to split by hand.
    let count = lowered
        .create_uniform(
            "count",
            &Count {
                n,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        )
        .unwrap();
    let (hits, groups, locals) = fresh(&lowered, n as usize, len);
    let refused = split_kernels::line::dispatch_extent(
        &lowered,
        Extent::linear(n),
        &count,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    )
    .expect_err("13 workgroups are past a limit of 4");
    assert!(
        format!("{refused:?}").contains("DeviceLimit"),
        "{refused:?}"
    );
}

/// (b) Base 0, one piece: the variant equals the kernel itself.
#[test]
fn a_variant_with_base_zero_is_the_kernel() {
    let n = 20;
    let len = 32;
    let device = device();
    let variant = line_by_hand(&device, n, len, &[3]);
    assert_eq!(variant, line_unsplit(&device, n, len));
    assert_eq!(variant, line_expected(n, len));
}

/// A launch of `plane` over `w x h` through its own descriptor on `device`.
fn plane_unsplit(device: &HeadlessDevice, w: u32, h: u32, len: usize) -> Outputs {
    let size = device
        .create_uniform(
            "size",
            &Size2 {
                w,
                h,
                pad0: 0,
                pad1: 0,
            },
        )
        .unwrap();
    let (hits, groups, locals) = fresh(device, (w * h) as usize, len);
    split_kernels::plane::dispatch_extent(
        device,
        Extent::new(w, h, 1),
        &size,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    )
    .unwrap();
    outputs(device, &hits, &groups, &locals)
}

/// `plane` on the CPU twin.
fn plane_twin(w: u32, h: u32, len: usize) -> Outputs {
    let cpu = CpuDevice::new();
    let size = cpu
        .uniform(Size2 {
            w,
            h,
            pad0: 0,
            pad1: 0,
        })
        .unwrap();
    let (hits, groups, locals) = twin_fresh(&cpu, (w * h) as usize, len);
    split_kernels::plane::dispatch_extent(
        &cpu,
        Extent::new(w, h, 1),
        &size,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    )
    .unwrap();
    Outputs {
        hits: hits.read(),
        groups: groups.read(),
        locals: locals.read(),
    }
}

/// (c) The two-dimensional kernel with bases on x and y: 21 x 9 elements are 6 x 5 workgroups of 4 x 2, run as
/// 2 x 2 pieces of (4 + 2) x (4 + 1) workgroups on a device limited to four per dimension, x innermost.
#[test]
fn a_hand_split_two_dimensional_launch_has_bases_on_x_and_y() {
    let (w, h) = (21u32, 9u32);
    let len = 256;
    let lowered = lowered(device(), 4);
    eprintln!("split (c) adapter: {:?}", lowered.adapter_info());
    let variant = split_kernels::plane::DESCRIPTOR.split.unwrap();
    let size = lowered
        .create_uniform(
            "size",
            &Size2 {
                w,
                h,
                pad0: 0,
                pad1: 0,
            },
        )
        .unwrap();
    let (hits, groups, locals) = fresh(&lowered, (w * h) as usize, len);
    // (first workgroup x, workgroups x, first workgroup y, workgroups y), x innermost.
    let pieces = [
        (0u32, 4u32, 0u32, 4u32),
        (4, 2, 0, 4),
        (0, 4, 4, 1),
        (4, 2, 4, 1),
    ];
    let bases: Vec<_> = pieces
        .iter()
        .map(|(x, _, y, _)| {
            lowered
                .create_uniform(
                    "base",
                    &LaunchBase {
                        x: x * 4,
                        y: y * 2,
                        z: 0,
                        pad: 0,
                    },
                )
                .unwrap()
        })
        .collect();
    let all: Vec<[BufferBinding<'_>; 5]> = bases
        .iter()
        .map(|base| {
            [
                size.bind(Access::ReadOnly, LengthRule::Independent),
                hits.bind(Access::ReadWrite, LengthRule::Independent),
                groups.bind(Access::ReadWrite, LengthRule::Independent),
                locals.bind(Access::ReadWrite, LengthRule::Independent),
                base.bind(Access::ReadOnly, LengthRule::Independent),
            ]
        })
        .collect();
    let dispatches: Vec<BufferDispatch<'_>> = pieces
        .iter()
        .zip(&all)
        .map(|((_, nx, _, ny), bindings)| {
            BufferDispatch::new_extent(variant, Extent::new(nx * 4, ny * 2, 1), bindings)
        })
        .collect();
    lowered.dispatch_batch(&dispatches).unwrap();
    let split = outputs(&lowered, &hits, &groups, &locals);

    // The independent host reference: the group and the local id of each cell of a 4 x 2 tiling.
    let used = (w * h) as usize;
    let mut expected = Outputs {
        hits: (0..len)
            .map(|i| if i < used { 1 } else { SENTINEL })
            .collect(),
        groups: vec![UNWRITTEN; len],
        locals: vec![UNWRITTEN; len],
    };
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            expected.groups[i] = (y / 2) * 4096 + x / 4;
            expected.locals[i] = (y % 2) * 16 + x % 4;
        }
    }
    assert_eq!(split, expected, "the hand split against the host reference");
    assert_eq!(
        plane_unsplit(&device(), w, h, len),
        expected,
        "unsplit, normal limits"
    );
    assert_eq!(plane_twin(w, h, len), expected, "the CPU twin");
}

/// (d) The real size on a device with its own limits: workgroup 256, 65,536 workgroups (2^24 elements) are one more
/// than the limit of 65,535, so they run as 65,535 + 1.
#[test]
fn a_hand_split_launch_past_the_real_limit_runs_every_element_once() {
    let device = device();
    let limit = device.limits().max_compute_workgroups_per_dimension;
    eprintln!(
        "split (d) adapter: {:?}, max workgroups per dimension {limit}",
        device.adapter_info()
    );
    assert_eq!(limit, 65_535, "the portable limit the test is written for");
    let workgroups = limit + 1;
    let n = workgroups * 256;
    let variant = split_kernels::wide::DESCRIPTOR.split.unwrap();
    let count = device
        .create_uniform(
            "count",
            &Count {
                n,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        )
        .unwrap();
    let hits =
        device.create_typed_buffer("hits", &vec![0u32; n as usize], GpuBufferAccess::ReadWrite);
    let tags = device.create_typed_buffer(
        "tags",
        &vec![UNWRITTEN; n as usize],
        GpuBufferAccess::ReadWrite,
    );
    let first = device
        .create_uniform(
            "base0",
            &LaunchBase {
                x: 0,
                y: 0,
                z: 0,
                pad: 0,
            },
        )
        .unwrap();
    let second = device
        .create_uniform(
            "base1",
            &LaunchBase {
                x: limit * 256,
                y: 0,
                z: 0,
                pad: 0,
            },
        )
        .unwrap();
    fn bindings<'a>(
        count: &'a gust_wgpu::GpuUniform<Count>,
        hits: &'a GpuBuffer<u32>,
        tags: &'a GpuBuffer<u32>,
        base: &'a gust_wgpu::GpuUniform<LaunchBase>,
    ) -> [BufferBinding<'a>; 4] {
        [
            count.bind(Access::ReadOnly, LengthRule::Independent),
            hits.bind(Access::ReadWrite, LengthRule::Independent),
            tags.bind(Access::ReadWrite, LengthRule::Independent),
            base.bind(Access::ReadOnly, LengthRule::Independent),
        ]
    }
    let first_bindings = bindings(&count, &hits, &tags, &first);
    let second_bindings = bindings(&count, &hits, &tags, &second);
    device
        .dispatch_batch(&[
            BufferDispatch::new_extent(variant, Extent::linear(limit * 256), &first_bindings),
            BufferDispatch::new_extent(variant, Extent::linear(256), &second_bindings),
        ])
        .unwrap();
    let hits = device.read_typed_buffer(&hits).unwrap();
    let tags = device.read_typed_buffer(&tags).unwrap();
    assert_eq!(hits.len(), n as usize);
    let wrong_hits = hits.iter().filter(|hit| **hit != 1).count();
    let wrong_tags = tags
        .iter()
        .enumerate()
        .filter(|(i, tag)| **tag != *i as u32)
        .count();
    assert_eq!(
        (wrong_hits, wrong_tags),
        (0, 0),
        "elements not counted exactly once, and elements whose group * 256 + local differs from their index"
    );
}

/// (e) An unsplit launch compiles one pipeline, the kernel's own; the variant is a second one, made only when it runs.
#[test]
fn only_the_variant_that_runs_is_compiled() {
    let device = device();
    assert_eq!(device.pipeline_cache_stats().entries, 0);
    let unsplit = line_unsplit(&device, 20, 32);
    assert_eq!(unsplit, line_expected(20, 32));
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits),
        (1, 1, 0),
        "an unsplit launch compiles the base kernel and nothing else"
    );
    let by_hand = line_by_hand(&device, 20, 32, &[3]);
    assert_eq!(by_hand, unsplit);
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses),
        (2, 2),
        "the variant is its own pipeline"
    );
    // The same launch again reuses both.
    let _ = line_unsplit(&device, 20, 32);
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses), (2, 2));
}

#[test]
fn a_descriptor_built_by_hand_has_no_variant() {
    let by_hand = KernelDescriptor::new("m", "k", "gpu_m_k", [8, 1, 1], &[], "// slang");
    assert!(by_hand.split.is_none());
}

//! The split variant of a `grid` kernel on a GPU (slice m11.g4a, D201, D205).
//!
//! Every `grid` kernel the `#[gpu]` macro builds for the GPU also carries a second descriptor,
//! `K::DESCRIPTOR.split`: the same Slang with one hidden uniform at the next free binding, whose `x`, `y` and `z` are
//! added to the dispatch-thread id before anything reads it. This slice proves that variant by running it on a GPU
//! with a split the TEST does by hand: one `BufferDispatch` per piece, each with its own base uniform, in one batch.
//! Slice m11.g4b adds the host's own split (the tests from `the_host_split_` on): a launch past a workgroup limit is cut
//! by the host and must equal the hand split. Each result is compared with an unsplit launch of the kernel's own
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

    pub struct Size3 {
        pub w: u32,
        pub h: u32,
        pub d: u32,
        pub pad0: u32,
    }

    /// Three dimensions, workgroup 4 x 4 x 4: the z axis of a launch has never run on a GPU before slice m11.g4b.
    #[kernel(workgroup_size(4, 4, 4))]
    pub fn cube(
        id: SV_DispatchThreadID,
        group: GroupId,
        local: LocalId,
        p: Uniform<Size3>,
        mut hits: RWStructuredBuffer<u32>,
        mut groups: RWStructuredBuffer<u32>,
        mut locals: RWStructuredBuffer<u32>,
    ) {
        if id.x < p.w && id.y < p.h && id.z < p.d {
            let i = (id.z * p.h + id.y) * p.w + id.x;
            atomic_add(&mut hits[i], 1u32);
            groups[i] = group.z * 10000u32 + group.y * 100u32 + group.x;
            locals[i] = local.z * 16u32 + local.y * 4u32 + local.x;
        }
    }

    /// Unguarded on purpose: `p` holds the rounded extent, every invocation bumps `total[0]` and its own cell.
    #[kernel(workgroup_size(4, 4, 4))]
    pub fn tally(
        id: SV_DispatchThreadID,
        p: Uniform<Size3>,
        mut hits: RWStructuredBuffer<u32>,
        mut total: RWStructuredBuffer<u32>,
    ) {
        atomic_add(&mut total[0], 1u32);
        let i = (id.z * p.h + id.y) * p.w + id.x;
        atomic_add(&mut hits[i], 1u32);
    }

    /// `out[i] = i + 1`: the first call of a chain.
    #[kernel(workgroup_size(8, 1, 1))]
    pub fn stamp(id: SV_DispatchThreadID, p: Uniform<Count>, mut out: RWStructuredBuffer<u32>) {
        let i = id.x;
        if i < p.n {
            out[i] = i + 1u32;
        }
    }

    /// `out[i] = src[i] * 2`: the second call of a chain, which reads what `stamp` wrote.
    #[kernel(workgroup_size(8, 1, 1))]
    pub fn double(
        id: SV_DispatchThreadID,
        p: Uniform<Count>,
        src: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < p.n {
            out[i] = src[i] * 2u32;
        }
    }

    /// Workgroup 64: 5,000,000 elements are 78,125 workgroups, past the portable limit of 65,535.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn narrow(
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
            tags[i] = group.x * 64u32 + local.x;
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

use split_kernels::{Count, Size2, Size3};

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
    // Since slice m11.g4b the plain launch on the lowered device is cut by the host and gives the same answer.
    assert_eq!(line_unsplit(&lowered, n, len), expected, "the host's split");
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

// ---- Slice m11.g4b: the host splits. ----

/// The same expected outputs for a `w x h x d` launch of `cube`: every cell counted once, the group id packed as
/// `gz * 10000 + gy * 100 + gx` and the local id as `lz * 16 + ly * 4 + lx`.
fn cube_expected(w: u32, h: u32, d: u32) -> Outputs {
    let len = (w * h * d) as usize;
    let mut expected = Outputs {
        hits: vec![1; len],
        groups: vec![UNWRITTEN; len],
        locals: vec![UNWRITTEN; len],
    };
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let i = ((z * h + y) * w + x) as usize;
                expected.groups[i] = (z / 4) * 10_000 + (y / 4) * 100 + x / 4;
                expected.locals[i] = (z % 4) * 16 + (y % 4) * 4 + x % 4;
            }
        }
    }
    expected
}

fn size3(w: u32, h: u32, d: u32) -> Size3 {
    Size3 { w, h, d, pad0: 0 }
}

/// `cube` over `w x h x d` through its own descriptor on `device`: on a lowered device the host splits it.
fn cube_auto(device: &HeadlessDevice, w: u32, h: u32, d: u32) -> Outputs {
    let size = device.create_uniform("size", &size3(w, h, d)).unwrap();
    let len = (w * h * d) as usize;
    let (hits, groups, locals) = fresh(device, len, len);
    split_kernels::cube::dispatch_extent(
        device,
        Extent::new(w, h, d),
        &size,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    )
    .unwrap();
    outputs(device, &hits, &groups, &locals)
}

/// `cube` on the CPU twin.
fn cube_twin(w: u32, h: u32, d: u32) -> Outputs {
    let cpu = CpuDevice::new();
    let size = cpu.uniform(size3(w, h, d)).unwrap();
    let len = (w * h * d) as usize;
    let (hits, groups, locals) = twin_fresh(&cpu, len, len);
    split_kernels::cube::dispatch_extent(
        &cpu,
        Extent::new(w, h, d),
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

/// The pieces of one axis: `(first workgroup, workgroups)` for `count` workgroups at most `limit` at a time.
fn axis_pieces(count: u32, limit: u32) -> Vec<(u32, u32)> {
    (0..count.div_ceil(limit))
        .map(|k| (k * limit, limit.min(count - k * limit)))
        .collect()
}

/// `cube` split by the TEST, as g4a did: one dispatch of the variant per piece (x innermost, then y, then z), each with
/// its own base uniform, in one batch.
fn cube_by_hand(device: &HeadlessDevice, w: u32, h: u32, d: u32, limit: u32) -> Outputs {
    let variant = split_kernels::cube::DESCRIPTOR.split.unwrap();
    let size = device.create_uniform("size", &size3(w, h, d)).unwrap();
    let len = (w * h * d) as usize;
    let (hits, groups, locals) = fresh(device, len, len);
    let mut pieces = Vec::new();
    for (fz, nz) in axis_pieces(d.div_ceil(4), limit) {
        for (fy, ny) in axis_pieces(h.div_ceil(4), limit) {
            for (fx, nx) in axis_pieces(w.div_ceil(4), limit) {
                let base = LaunchBase {
                    x: fx * 4,
                    y: fy * 4,
                    z: fz * 4,
                    pad: 0,
                };
                pieces.push((
                    device.create_uniform("base", &base).unwrap(),
                    Extent::new(nx * 4, ny * 4, nz * 4),
                ));
            }
        }
    }
    let all: Vec<[BufferBinding<'_>; 5]> = pieces
        .iter()
        .map(|(base, _)| {
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
        .map(|((_, extent), bindings)| BufferDispatch::new_extent(variant, *extent, bindings))
        .collect();
    device.dispatch_batch(&dispatches).unwrap();
    outputs(device, &hits, &groups, &locals)
}

fn assert_cold(device: &HeadlessDevice) {
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits),
        (0, 0, 0),
        "a refusal must come before compilation and the cache lookup"
    );
}

/// (f) The host's split equals the hand split, the host reference and the CPU twin on x alone, y alone, z alone and all
/// three at once. Limit 4, workgroup 4 x 4 x 4, extent 20 x 20 x 20: 5 workgroups per axis, 2 pieces per axis, 8 pieces.
/// The z axis of a launch has never run on a GPU before this test.
#[test]
fn the_host_split_equals_the_hand_split_on_x_y_z_and_all_three() {
    // The last case is not a multiple of the workgroup on any axis, z included: 6 x 5 x 5 workgroups.
    for (w, h, d) in [
        (20, 4, 4),
        (4, 20, 4),
        (4, 4, 20),
        (20, 20, 20),
        (21, 18, 19),
    ] {
        let lowered_device = lowered(device(), 4);
        eprintln!(
            "split (f) {w}x{h}x{d} adapter: {:?}",
            lowered_device.adapter_info()
        );
        let expected = cube_expected(w, h, d);
        let auto = cube_auto(&lowered_device, w, h, d);
        let stats = lowered_device.pipeline_cache_stats();
        assert_eq!(
            (stats.entries, stats.misses),
            (1, 1),
            "a split launch compiles only the variant"
        );
        assert_eq!(auto, expected, "the host split against the host reference");
        assert_eq!(
            cube_by_hand(&lowered_device, w, h, d, 4),
            auto,
            "the host split against the hand split"
        );
        assert_eq!(cube_twin(w, h, d), expected, "the CPU twin");
        let plain = device();
        assert_eq!(
            cube_auto(&plain, w, h, d),
            expected,
            "unsplit, normal limits"
        );
        let stats = plain.pipeline_cache_stats();
        assert_eq!(
            (stats.entries, stats.misses),
            (1, 1),
            "an unsplit launch compiles only the base"
        );
    }
}

/// (n) No bounds check anywhere: every invocation of the rounded extent bumps a global counter and its own cell, and
/// the cells are exactly the rounded volume. A host that dispatches one workgroup too many (a piece of the full limit
/// where fewer are left) raises the counter above the rounded volume, which no guarded kernel could show.
#[test]
fn a_split_launch_runs_exactly_the_rounded_extent_and_no_invocation_more() {
    for (w, h, d) in [(21u32, 18u32, 19u32), (30, 3, 3), (3, 30, 3), (3, 3, 30)] {
        let lowered = lowered(device(), 4);
        let (rw, rh, rd) = (w.div_ceil(4) * 4, h.div_ceil(4) * 4, d.div_ceil(4) * 4);
        let cells = (rw * rh * rd) as usize;
        let size = lowered.create_uniform("size", &size3(rw, rh, rd)).unwrap();
        let hits =
            lowered.create_typed_buffer("hits", &vec![0u32; cells], GpuBufferAccess::ReadWrite);
        let total = lowered.create_typed_buffer("total", &[0u32], GpuBufferAccess::ReadWrite);
        split_kernels::tally::dispatch_extent(
            &lowered,
            Extent::new(w, h, d),
            &size,
            Independent(&hits),
            Independent(&total),
        )
        .unwrap();
        assert_eq!(
            lowered.read_typed_buffer(&total).unwrap(),
            [cells as u32],
            "{w}x{h}x{d}: invocations run"
        );
        let hits = lowered.read_typed_buffer(&hits).unwrap();
        assert!(
            hits.iter().all(|hit| *hit == 1),
            "{w}x{h}x{d}: every cell once"
        );
    }
}

/// (g) The two-dimensional kernel through the host split on a lowered device equals the twin (the by-hand version of
/// the same launch is test (c)).
#[test]
fn the_host_split_of_a_two_dimensional_launch_equals_the_twin() {
    let (w, h, len) = (21, 9, 256);
    let split = plane_unsplit(&lowered(device(), 4), w, h, len);
    assert_eq!(split, plane_twin(w, h, len));
}

/// (h) Two split calls in one batch: the second reads what the first wrote, so the pieces of the first must have
/// finished (batch order is unchanged).
#[test]
fn a_batch_of_two_split_calls_runs_in_order() {
    let lowered = lowered(device(), 4);
    let (n, len) = (100u32, 104usize);
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
    let mid = lowered.create_typed_buffer("mid", &vec![0u32; len], GpuBufferAccess::ReadWrite);
    let out = lowered.create_typed_buffer("out", &vec![0u32; len], GpuBufferAccess::ReadWrite);
    let first = split_kernels::stamp::call_extent(Extent::linear(n), &count, Independent(&mid));
    let second = split_kernels::double::call_extent(
        Extent::linear(n),
        &count,
        Independent(&mid),
        Independent(&out),
    );
    lowered
        .dispatch_batch(&[(&first).into(), (&second).into()])
        .unwrap();
    let expected: Vec<u32> = (0..len as u32)
        .map(|i| if i < n { (i + 1) * 2 } else { 0 })
        .collect();
    assert_eq!(lowered.read_typed_buffer(&out).unwrap(), expected);
    let stamped: Vec<u32> = (0..len as u32)
        .map(|i| if i < n { i + 1 } else { 0 })
        .collect();
    assert_eq!(lowered.read_typed_buffer(&mid).unwrap(), stamped);
}

/// (i) The typed `submit_extent` path: one job, waited for.
#[test]
fn a_typed_submit_of_a_split_launch_runs_every_element_once() {
    let lowered = lowered(device(), 4);
    let (n, len) = (100u32, 128usize);
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
    let job = split_kernels::line::submit_extent(
        &lowered,
        Extent::linear(n),
        &count,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    )
    .unwrap();
    job.wait().unwrap();
    assert_eq!(
        outputs(&lowered, &hits, &groups, &locals),
        line_expected(n, len)
    );
}

/// The `line` buffers of a launch that is refused, and the error it answers with.
fn refused_line(device: &HeadlessDevice, extent: Extent) -> gust_wgpu::Error {
    let count = device
        .create_uniform(
            "count",
            &Count {
                n: 8,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        )
        .unwrap();
    let (hits, groups, locals) = fresh(device, 8, 8);
    split_kernels::line::dispatch_extent(
        device,
        extent,
        &count,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    )
    .expect_err("the launch is refused")
}

fn limit_of(error: &gust_wgpu::Error) -> (&'static str, u64, u64) {
    match error {
        gust_wgpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        } => (limit, *requested, *maximum),
        other => panic!("expected a DeviceLimit, got {other:?}"),
    }
}

/// (j) More pieces than `MAX_SPLIT_DISPATCHES`: a limit of one workgroup per dimension and 4097 workgroups.
#[test]
fn a_launch_of_more_than_the_most_pieces_is_refused_before_compiling() {
    let lowered = lowered(device(), 1);
    let refused = refused_line(&lowered, Extent::linear(8 * 4097));
    assert_eq!(
        limit_of(&refused),
        (
            "split dispatches",
            4097,
            gust_wgpu::MAX_SPLIT_DISPATCHES as u64
        )
    );
    assert_cold(&lowered);
    // 4096 pieces is the most that runs.
    let ok = lowered_ok_edge();
    assert_eq!(ok, line_expected(8 * 4096, 8 * 4096));
}

/// `line` over 4096 workgroups on a device limited to one: 4096 pieces.
fn lowered_ok_edge() -> Outputs {
    let lowered = lowered(device(), 1);
    line_unsplit(&lowered, 8 * 4096, 8 * 4096)
}

/// (k) A device whose uniform count is lowered to the kernel's own cannot take the variant's extra one: the launch
/// keeps the plain refusal, before anything is compiled.
#[test]
fn a_variant_over_the_uniform_limit_keeps_the_plain_refusal() {
    let device = device();
    let mut limits = device.limits().clone();
    limits.max_compute_workgroups_per_dimension = 4;
    limits.max_uniform_buffers_per_shader_stage = 1;
    let lowered = device.with_limits(limits).unwrap();
    let refused = refused_line(&lowered, Extent::linear(100));
    assert_eq!(
        limit_of(&refused),
        ("max_compute_workgroups_per_dimension", 13, 4)
    );
    assert_cold(&lowered);
}

/// (l) The invocation cap comes before the pieces: past 2^32 the launch is refused as `global_invocation_id`, on the
/// GPU and on the twin, before anything is compiled.
#[test]
fn an_extent_over_the_invocation_cap_is_refused_as_global_invocation_id_on_both() {
    let lowered = lowered(device(), 4);
    let extent = Extent::new(u32::MAX, 2, 1);
    let wanted = ("global_invocation_id", 1u64 << 33, 1u64 << 32);
    assert_eq!(limit_of(&refused_line(&lowered, extent)), wanted);
    assert_cold(&lowered);
    let cpu = CpuDevice::new();
    let count = cpu
        .uniform(Count {
            n: 8,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        })
        .unwrap();
    let (hits, groups, locals) = twin_fresh(&cpu, 8, 8);
    let call = split_kernels::line::call_extent(
        extent,
        &count,
        Independent(&hits),
        Independent(&groups),
        Independent(&locals),
    );
    match gust::cpu::validate(&cpu, &call) {
        Err(gust::cpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) => assert_eq!((limit, requested, maximum), wanted),
        other => panic!("expected the cap on the twin, got {other:?}"),
    }
}

/// (m) The real size on the device's own limits: every element of a launch past 65,535 workgroups runs exactly once,
/// with the group and local id of an unsplit launch (`tags[i] == i`).
#[test]
fn the_host_split_at_real_size_runs_every_element_once() {
    let device = device();
    assert_eq!(device.limits().max_compute_workgroups_per_dimension, 65_535);
    eprintln!("split (m) adapter: {:?}", device.adapter_info());
    for (name, n) in [("narrow", 5_000_000u32), ("wide", (1 << 24) + 1000)] {
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
        let started = std::time::Instant::now();
        if name == "narrow" {
            split_kernels::narrow::dispatch_extent(
                &device,
                Extent::linear(n),
                &count,
                Independent(&hits),
                Independent(&tags),
            )
            .unwrap();
        } else {
            split_kernels::wide::dispatch_extent(
                &device,
                Extent::linear(n),
                &count,
                Independent(&hits),
                Independent(&tags),
            )
            .unwrap();
        }
        let hits = device.read_typed_buffer(&hits).unwrap();
        let tags = device.read_typed_buffer(&tags).unwrap();
        let wrong_hits = hits.iter().filter(|hit| **hit != 1).count();
        let wrong_tags = tags
            .iter()
            .enumerate()
            .filter(|(i, tag)| **tag != *i as u32)
            .count();
        eprintln!("split (m) {name} n={n}: {:?}", started.elapsed());
        assert_eq!((wrong_hits, wrong_tags), (0, 0), "{name}, n = {n}");
    }
}

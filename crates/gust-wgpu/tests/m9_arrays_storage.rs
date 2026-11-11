// The fixture writes `x = x + f()` on purpose (an order case), which clippy would turn into `x += f()`.
#![allow(clippy::assign_op_pattern)]

include!("../../../tests/fixtures/arrays_storage.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust::math::{Mat2, Mat4, Vec2, Vec4};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the fixture's rules (not computed by a function that mirrors the kernel).
// Element i of every buffer is built from v = i % 5, so each table repeats every five rows.
//
// roundtrip: quads host [v, v + 1, v + 2, v + 3] -> [v + 11, v + 1, v + 2, 4 v + 7] (element 3 is the sum 4 v + 6, plus one).
//   floats host v + j / 2 -> 2 v + j + 1. rows host 10 i + j, picks[i] = rows[i][i % 4] = 10 i + i % 4.
// fields: bags host items [v, v + 1, v + 2, v + 3], gains [1.5, 2.5] -> items [v + 8, v + 1, v + 101, v], gains [3.0, 5.5],
//   count 4 (`.len()` of the field), total 4 v + 6. polys host pts [(v, 1.0), (v + 1, 2.0)], tip [(v, 0.5), (1.0, 2.0)],
//   n = v -> pts [(v, 3.0), (v + 8, 2.0)], tip swapped, n = v + 2, pad = 2.
// wide: vecs host [(v, 1, 2, 3), (4, 5, 6, 7)] -> [(4, 5, 6, 7), (v, 1, 2, v + 10)]; mats and smalls swap their pair.
// orders (Rust's order, measured with rustc): a = 241 (the call first, then the element: 12 + 12 = 24, n = 1), b = 14
//   (the left operand is read before the call), c = 12 + 22 + 32 + 42 = 108 with n = 4, d = [1, 12, 12, 22] weighed
//   1000000 / 10000 / 100 / 1 = 1121222, e = sum2(2, 12) = 2012, f = sum2(12, 12) = 12012, g = [9, 9, 12, 9].
const QUAD_0: [u32; 5] = [11, 12, 13, 14, 15];
const QUAD_3: [u32; 5] = [7, 11, 15, 19, 23];
const ORDER_A: u32 = 241;
const ORDER_B: u32 = 14;
const ORDER_C: u32 = 108;
const ORDER_C_N: u32 = 4;
const ORDER_D: u32 = 1_121_222;
const ORDER_E: u32 = 2012;
const ORDER_F: u32 = 12012;
const ORDER_G: [u32; 4] = [9, 9, 12, 9];

fn diag(a: f32, b: f32, c: f32, d: f32) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(a, 0.0, 0.0, 0.0),
        Vec4::new(0.0, b, 0.0, 0.0),
        Vec4::new(0.0, 0.0, c, 0.0),
        Vec4::new(0.0, 0.0, 0.0, d),
    )
}

fn small(a: f32, b: f32, c: f32, d: f32) -> Mat2 {
    Mat2::from_cols(Vec2::new(a, b), Vec2::new(c, d))
}

fn show<T: std::fmt::Debug>(value: &T) -> String {
    format!("{value:?}")
}

#[test]
fn arrays_in_structs_and_storage_run_on_the_gpu_and_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("arrays_storage adapter: {:?}", device.adapter_info());
    for count in [1usize, 63, 64, 65, 257] {
        let cpu = CpuDevice::new();
        let v_of = |i: usize| (i % 5) as u32;

        // roundtrip
        let quads: Vec<[u32; 4]> = (0..count)
            .map(|i| {
                let v = v_of(i);
                [v, v + 1, v + 2, v + 3]
            })
            .collect();
        let floats: Vec<[f32; 8]> = (0..count)
            .map(|i| {
                let v = v_of(i) as f32;
                std::array::from_fn(|j| v + j as f32 / 2.0)
            })
            .collect();
        let rows: Vec<[u32; 4]> = (0..count)
            .map(|i| std::array::from_fn(|j| 10 * i as u32 + j as u32))
            .collect();
        let picks = vec![u32::MAX; count];
        let g_quads = device.create_typed_buffer("quads", &quads, GpuBufferAccess::ReadWrite);
        let g_floats = device.create_typed_buffer("floats", &floats, GpuBufferAccess::ReadWrite);
        let g_rows = device.create_typed_buffer("rows", &rows, GpuBufferAccess::ReadOnly);
        let g_picks = device.create_typed_buffer("picks", &picks, GpuBufferAccess::ReadWrite);
        device
            .dispatch_call(&arrays_storage::roundtrip::call(
                count as u32,
                &g_quads,
                &g_floats,
                &g_rows,
                &g_picks,
            ))
            .unwrap();
        let t_quads = cpu.buffer(&quads);
        let t_floats = cpu.buffer(&floats);
        let t_rows = cpu.buffer(&rows);
        let t_picks = cpu.buffer(&picks);
        arrays_storage::roundtrip::dispatch(
            &cpu,
            count as u32,
            &t_quads,
            &t_floats,
            &t_rows,
            &t_picks,
        )
        .unwrap();
        for (name, quads, floats, picks) in [
            (
                "gpu",
                device.read_typed_buffer(&g_quads).unwrap(),
                device.read_typed_buffer(&g_floats).unwrap(),
                device.read_typed_buffer(&g_picks).unwrap(),
            ),
            ("twin", t_quads.read(), t_floats.read(), t_picks.read()),
        ] {
            assert_eq!(quads.len(), count);
            for i in 0..count {
                let note = format!("{name} roundtrip [{i}], count {count}");
                let (k, v) = (i % 5, v_of(i));
                assert_eq!(
                    quads[i],
                    [QUAD_0[k], v + 1, v + 2, QUAD_3[k]],
                    "quads {note}"
                );
                let want: [f32; 8] = std::array::from_fn(|j| 2.0 * v as f32 + j as f32 + 1.0);
                assert_eq!(floats[i], want, "floats {note}");
                assert_eq!(picks[i], 10 * i as u32 + (i % 4) as u32, "picks {note}");
            }
        }

        // fields
        let bags: Vec<arrays_storage::Bag> = (0..count)
            .map(|i| {
                let v = v_of(i);
                arrays_storage::Bag {
                    items: [v, v + 1, v + 2, v + 3],
                    gains: [1.5, 2.5],
                    count: u32::MAX,
                    total: u32::MAX,
                }
            })
            .collect();
        let polys: Vec<arrays_storage::Poly> = (0..count)
            .map(|i| {
                let v = v_of(i);
                arrays_storage::Poly {
                    pts: [
                        arrays_storage::Pt { x: v, y: 1.0 },
                        arrays_storage::Pt { x: v + 1, y: 2.0 },
                    ],
                    tip: [Vec2::new(v as f32, 0.5), Vec2::new(1.0, 2.0)],
                    n: v,
                    pad: u32::MAX,
                }
            })
            .collect();
        let g_bags = device.create_typed_buffer("bags", &bags, GpuBufferAccess::ReadWrite);
        let g_polys = device.create_typed_buffer("polys", &polys, GpuBufferAccess::ReadWrite);
        device
            .dispatch_call(&arrays_storage::fields::call(
                count as u32,
                &g_bags,
                &g_polys,
            ))
            .unwrap();
        let t_bags = cpu.buffer(&bags);
        let t_polys = cpu.buffer(&polys);
        arrays_storage::fields::dispatch(&cpu, count as u32, &t_bags, &t_polys).unwrap();
        for (name, bags, polys) in [
            (
                "gpu",
                device.read_typed_buffer(&g_bags).unwrap(),
                device.read_typed_buffer(&g_polys).unwrap(),
            ),
            ("twin", t_bags.read(), t_polys.read()),
        ] {
            for i in 0..count {
                let note = format!("{name} fields [{i}], count {count}");
                let v = v_of(i);
                assert_eq!(bags[i].items, [v + 8, v + 1, v + 101, v], "items {note}");
                assert_eq!(bags[i].gains, [3.0, 5.5], "gains {note}");
                assert_eq!(bags[i].count, 4, "count {note}");
                assert_eq!(bags[i].total, 4 * v + 6, "total {note}");
                let p = &polys[i];
                assert_eq!((p.pts[0].x, p.pts[0].y), (v, 3.0), "pts0 {note}");
                assert_eq!((p.pts[1].x, p.pts[1].y), (v + 8, 2.0), "pts1 {note}");
                assert_eq!(
                    show(&p.tip),
                    show(&[Vec2::new(1.0, 2.0), Vec2::new(v as f32, 0.5)]),
                    "tip {note}"
                );
                assert_eq!((p.n, p.pad), (v + 2, 2), "n and pad {note}");
            }
        }

        // wide
        let vecs: Vec<[Vec4; 2]> = (0..count)
            .map(|i| {
                [
                    Vec4::new(v_of(i) as f32, 1.0, 2.0, 3.0),
                    Vec4::new(4.0, 5.0, 6.0, 7.0),
                ]
            })
            .collect();
        let mats: Vec<[Mat4; 2]> = (0..count)
            .map(|i| {
                [
                    diag(v_of(i) as f32 + 1.0, 2.0, 3.0, 4.0),
                    diag(5.0, 6.0, 7.0, 8.0),
                ]
            })
            .collect();
        let smalls: Vec<[Mat2; 2]> = (0..count)
            .map(|i| {
                [
                    small(v_of(i) as f32 + 1.0, 2.0, 3.0, 4.0),
                    small(5.0, 6.0, 7.0, 8.0),
                ]
            })
            .collect();
        let g_vecs = device.create_typed_buffer("vecs", &vecs, GpuBufferAccess::ReadWrite);
        let g_mats = device.create_typed_buffer("mats", &mats, GpuBufferAccess::ReadWrite);
        let g_smalls = device.create_typed_buffer("smalls", &smalls, GpuBufferAccess::ReadWrite);
        device
            .dispatch_call(&arrays_storage::wide::call(
                count as u32,
                &g_vecs,
                &g_mats,
                &g_smalls,
            ))
            .unwrap();
        let t_vecs = cpu.buffer(&vecs);
        let t_mats = cpu.buffer(&mats);
        let t_smalls = cpu.buffer(&smalls);
        arrays_storage::wide::dispatch(&cpu, count as u32, &t_vecs, &t_mats, &t_smalls).unwrap();
        for (name, vecs, mats, smalls) in [
            (
                "gpu",
                device.read_typed_buffer(&g_vecs).unwrap(),
                device.read_typed_buffer(&g_mats).unwrap(),
                device.read_typed_buffer(&g_smalls).unwrap(),
            ),
            ("twin", t_vecs.read(), t_mats.read(), t_smalls.read()),
        ] {
            for i in 0..count {
                let note = format!("{name} wide [{i}], count {count}");
                let v = v_of(i) as f32;
                assert_eq!(
                    show(&vecs[i]),
                    show(&[
                        Vec4::new(4.0, 5.0, 6.0, 7.0),
                        Vec4::new(v, 1.0, 2.0, v + 10.0)
                    ]),
                    "vecs {note}"
                );
                assert_eq!(
                    show(&mats[i]),
                    show(&[diag(5.0, 6.0, 7.0, 8.0), diag(v + 1.0, 2.0, 3.0, 4.0)]),
                    "mats {note}"
                );
                assert_eq!(
                    show(&smalls[i]),
                    show(&[small(5.0, 6.0, 7.0, 8.0), small(v + 1.0, 2.0, 3.0, 4.0)]),
                    "smalls {note}"
                );
            }
        }

        // orders
        let blank = vec![
            arrays_storage::Orders {
                a: u32::MAX,
                b: u32::MAX,
                c: u32::MAX,
                c_n: u32::MAX,
                d: u32::MAX,
                e: u32::MAX,
                f: u32::MAX,
                g: [9; 4],
            };
            count
        ];
        let g_out = device.create_typed_buffer("orders", &blank, GpuBufferAccess::ReadWrite);
        device
            .dispatch_call(&arrays_storage::orders::call(count as u32, &g_out))
            .unwrap();
        let t_out = cpu.buffer(&blank);
        arrays_storage::orders::dispatch(&cpu, count as u32, &t_out).unwrap();
        for (name, rows) in [
            ("gpu", device.read_typed_buffer(&g_out).unwrap()),
            ("twin", t_out.read()),
        ] {
            assert_eq!(rows.len(), count);
            for (i, row) in rows.iter().enumerate() {
                let note = format!("{name} orders [{i}], count {count}");
                assert_eq!(row.a, ORDER_A, "a {note}");
                assert_eq!(row.b, ORDER_B, "b {note}");
                assert_eq!(row.c, ORDER_C, "c {note}");
                assert_eq!(row.c_n, ORDER_C_N, "c_n {note}");
                assert_eq!(row.d, ORDER_D, "d {note}");
                assert_eq!(row.e, ORDER_E, "e {note}");
                assert_eq!(row.f, ORDER_F, "f {note}");
                assert_eq!(row.g, ORDER_G, "g {note}");
            }
        }
    }
}

#[test]
fn the_arrays_storage_fixture_compiles_to_wgsl_with_stored_arrays() {
    let wgsl = gust::slang::compile_wgsl(&arrays_storage::roundtrip::DESCRIPTOR).unwrap();
    assert!(wgsl.contains("array<u32, i32(4)>"), "{wgsl}");
    let wgsl = gust::slang::compile_wgsl(&arrays_storage::fields::DESCRIPTOR).unwrap();
    assert!(wgsl.contains("array<u32, i32(4)>"), "{wgsl}");
}

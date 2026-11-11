include!("../../../tests/fixtures/arrays.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the fixture's rules (not computed by a function that mirrors the kernel).
//
// Element i of `input` is i % 5, so each table below repeats every five rows: v = 0, 1, 2, 3, 4.
//   lit_sum     = `[v, v + 1, v + 2, v + 3]` summed: 4v + 6                         : 6, 10, 14, 18, 22
//   rep_ends    = `[v + 1; 4]` read at 3 and 0: 2 (v + 1)                           : 2, 4, 6, 8, 10
//   bump_once   = `[s.bump(); 4]` with s.n = v: the operand runs once and gives v + 1, four times: 4 (v + 1) : 4, 8, 12, 16, 20
//   bump_n      = s.n afterwards: v + 1 (the operand ran once)                      : 1, 2, 3, 4, 5
//   dyn_rw      = `d = [0; 5]`, d[v] = 10 + v, d[v + 1 mod 5] = 7: d[v] * 100 + 7   : 1007, 1107, 1207, 1307, 1407
//   compound    = `c = [1, 2, 3]`, `c[1] += v`, `c[v mod 3] += 10`, as c0 * 10000 + c1 * 100 + c2:
//                 v = 0: [11, 2, 3]; 1: [1, 13, 3]; 2: [1, 4, 13]; 3: [11, 5, 3]; 4: [1, 16, 3]
//                                                                                    : 110203, 11303, 10413, 110503, 11603
//   helper      = f = [v, 1, 2, 3], g = dbl4(f) = [2v, 2, 4, 6]: sum4(g) + f[0] = 2v + 12 + v : 12, 15, 18, 21, 24
//   copy_check  = bump_all(f) is f[0] + 100 on its own copy; f[0] stays v; last(f) = 3: (v + 100) + v + 3 : 103, 105, 107, 109, 111
//   len_n       = `l = [1, .., 6]`: l.len() * 100 + l[5] = 600 + 6                  : 606
//   struct_el   = `[Pt; 2]` with a = v and v + 1, `ps[1].a += 5`, `ps[0].b = 2.5`: v + (v + 6) + 2 : 8, 10, 12, 14, 16
//   fill17      = `[v + 3; 17]` summed: 17 (v + 3)                                  : 51, 68, 85, 102, 119
//   fill17_len  = 17
//   assign_rep  = `w = [v; 4]` assigned: w[0] + w[3] = 2v                           : 0, 2, 4, 6, 8
//   assign_fill = `w2 = [v; 17]` assigned, summed: 17v                              : 0, 17, 34, 51, 68
//   tail_rep    = `repeated(v)[2]`, a returned `[v; 3]`: v                          : 0, 1, 2, 3, 4
//   tail_lit    = `make3(v)[0] + make3(v)[2]` = v + (v + 2)                         : 2, 4, 6, 8, 10
//   enum_el     = `[Mode::Low, Mode::High]`: (High = 3) * 10 + (Low = 1)            : 31
//   vec_el      = `[Vec2; 2]`: vs[0].x + vs[1].y = v + 3                            : 3, 4, 5, 6, 7
//   method_arr  = `sp.spread([10, 20, 30])` with sp.n = v: n becomes v + 1, each element gains it: 60 + 3 (v + 1) : 63, 66, 69, 72, 75
//   method_n    = sp.n = v + 1                                                      : 1, 2, 3, 4, 5
//   shadow      = inner `h = [h[1], h[0]]` = [v + 1, v]: (v + 1) * 10 + v = 11v + 10 : 10, 21, 32, 43, 54
// The evaluation-order cases do not depend on v (the order of Rust, written out by hand; each `Counter` starts at 0
// and `bump` adds one and gives the new value):
//   order_a = `[s1.bump(), s1.bump(), s1.bump()]` is [1, 2, 3]: 1 * 100 + 2 * 10 + 3 = 123
//   order_g = `pair_code([s2.bump(), s2.bump()])` is [1, 2]: 1 * 10 + 2 = 12
//   order_f = `tbl[s3.bump()] * 100 + tbl[s3.bump()]` with tbl = [10, 20, 30, 40]: tbl[1] = 20, tbl[2] = 30: 2030
//   order_h = `make3(s4.bump())[1]` = make3(1) = [1, 2, 3], element 1 is 2
//   order_i = `cs[s5.bump()].bump()` with cs[1].n = 200: the index is 1, cs[1].bump() gives 201; order_i_other is
//             cs[0].n + cs[2].n = 100 + 300 = 400 (only cs[1] changed)
//   order_j = `j[1] = j[0] + j[2]` on [1, 2, 3] gives [1, 4, 3]: 143
//   order_b = the value first, then the index (two statements): val = 1, `e6[s6.n] = val` writes e6[1] = 1: 1 * 10 + e6[0] = 10
//   order_pure = `e7[k7] = s7.bump()` with s7.n = 4 (k7 = 2): e7[2] = 5, first = 5; 5 * 100 + e7[0] = 500
//   order_pure_compound = `e7[k7] += s7.bump()`: bump gives 6, e7[2] = 5 + 6 = 11
//   order_i_mid = cs[1].n after `cs[s5.bump()].bump()`: the write-back reached the element, 200 + 1 = 201
// Index casts (fix round 1), with tbl = [10, 20, 30, 40]:
//   idx_signed = `tbl[(sv as usize) % 4]` with sv = v + 1 (an `i32`): index 1, 2, 3, 0, 1 : 20, 30, 40, 10, 20
//   idx_bool   = `tbl[(v > 2) as usize]`: index 0, 0, 0, 1, 1                              : 10, 10, 10, 20, 20
// Compound operators on the elements of `[50, 3, 100, 77, 9]` with d = v + 1: `-= d`, `<<= 1`, `/= d`, `%= d`, `^= d`,
// weighed as e0 * 1000000 + e1 * 100000 + e2 * 100 + e3 * 10 + e4 (the same on a local and on a parameter):
//   v = 0 (d = 1): [49, 6, 100, 0, 8]   : 49610008
//   v = 1 (d = 2): [48, 6, 50, 1, 11]   : 48605021
//   v = 2 (d = 3): [47, 6, 33, 2, 10]   : 47603330
//   v = 3 (d = 4): [46, 6, 25, 1, 13]   : 46602523
//   v = 4 (d = 5): [45, 6, 20, 2, 12]   : 45602032
const LIT_SUM: [u32; 5] = [6, 10, 14, 18, 22];
const REP_ENDS: [u32; 5] = [2, 4, 6, 8, 10];
const BUMP_ONCE: [u32; 5] = [4, 8, 12, 16, 20];
const BUMP_N: [u32; 5] = [1, 2, 3, 4, 5];
const DYN_RW: [u32; 5] = [1007, 1107, 1207, 1307, 1407];
const COMPOUND: [u32; 5] = [110203, 11303, 10413, 110503, 11603];
const HELPER: [f32; 5] = [12.0, 15.0, 18.0, 21.0, 24.0];
const COPY_CHECK: [f32; 5] = [103.0, 105.0, 107.0, 109.0, 111.0];
const LEN_N: u32 = 606;
const STRUCT_EL: [u32; 5] = [8, 10, 12, 14, 16];
const FILL17: [u32; 5] = [51, 68, 85, 102, 119];
const FILL17_LEN: u32 = 17;
const ASSIGN_REP: [u32; 5] = [0, 2, 4, 6, 8];
const ASSIGN_FILL: [u32; 5] = [0, 17, 34, 51, 68];
const TAIL_REP: [u32; 5] = [0, 1, 2, 3, 4];
const TAIL_LIT: [u32; 5] = [2, 4, 6, 8, 10];
const ENUM_EL: u32 = 31;
const VEC_EL: [f32; 5] = [3.0, 4.0, 5.0, 6.0, 7.0];
const METHOD_ARR: [u32; 5] = [63, 66, 69, 72, 75];
const METHOD_N: [u32; 5] = [1, 2, 3, 4, 5];
const SHADOW: [u32; 5] = [10, 21, 32, 43, 54];
const ORDER_A: u32 = 123;
const ORDER_G: u32 = 12;
const ORDER_F: u32 = 2030;
const ORDER_H: u32 = 2;
const ORDER_I: u32 = 201;
const ORDER_I_OTHER: u32 = 400;
const ORDER_J: u32 = 143;
const ORDER_B: u32 = 10;
const ORDER_PURE: u32 = 500;
const ORDER_PURE_COMPOUND: u32 = 11;
const ORDER_I_MID: u32 = 201;
const IDX_SIGNED: [u32; 5] = [20, 30, 40, 10, 20];
const IDX_BOOL: [u32; 5] = [10, 10, 10, 20, 20];
const OPS: [u32; 5] = [49610008, 48605021, 47603330, 46602523, 45602032];

#[test]
fn arrays_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("arrays adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| (i % 5) as u32).collect();
        let input_buf =
            device.create_typed_buffer("arrays input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            arrays::Results {
                lit_sum: u32::MAX,
                rep_ends: u32::MAX,
                bump_once: u32::MAX,
                bump_n: u32::MAX,
                dyn_rw: u32::MAX,
                compound: u32::MAX,
                helper: -1.0,
                copy_check: -1.0,
                len_n: u32::MAX,
                struct_el: u32::MAX,
                fill17: u32::MAX,
                fill17_len: u32::MAX,
                assign_rep: u32::MAX,
                assign_fill: u32::MAX,
                tail_rep: u32::MAX,
                tail_lit: u32::MAX,
                enum_el: u32::MAX,
                vec_el: -1.0,
                method_arr: u32::MAX,
                method_n: u32::MAX,
                shadow: u32::MAX,
                order_a: u32::MAX,
                order_g: u32::MAX,
                order_f: u32::MAX,
                order_h: u32::MAX,
                order_i: u32::MAX,
                order_i_other: u32::MAX,
                order_j: u32::MAX,
                order_b: u32::MAX,
                order_pure: u32::MAX,
                order_pure_compound: u32::MAX,
                order_i_mid: u32::MAX,
                idx_signed: u32::MAX,
                idx_bool: u32::MAX,
                ops_local: u32::MAX,
                ops_param: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("arrays results", &blank, GpuBufferAccess::ReadWrite);
        let call = arrays::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        arrays::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for (i, row) in got.iter().enumerate() {
                let note = format!("{name} [{i}], count {count}");
                let k = i % 5;
                assert_eq!(row.lit_sum, LIT_SUM[k], "lit_sum {note}");
                assert_eq!(row.rep_ends, REP_ENDS[k], "rep_ends {note}");
                assert_eq!(row.bump_once, BUMP_ONCE[k], "bump_once {note}");
                assert_eq!(row.bump_n, BUMP_N[k], "bump_n {note}");
                assert_eq!(row.dyn_rw, DYN_RW[k], "dyn_rw {note}");
                assert_eq!(row.compound, COMPOUND[k], "compound {note}");
                assert_eq!(row.helper, HELPER[k], "helper {note}");
                assert_eq!(row.copy_check, COPY_CHECK[k], "copy_check {note}");
                assert_eq!(row.len_n, LEN_N, "len_n {note}");
                assert_eq!(row.struct_el, STRUCT_EL[k], "struct_el {note}");
                assert_eq!(row.fill17, FILL17[k], "fill17 {note}");
                assert_eq!(row.fill17_len, FILL17_LEN, "fill17_len {note}");
                assert_eq!(row.assign_rep, ASSIGN_REP[k], "assign_rep {note}");
                assert_eq!(row.assign_fill, ASSIGN_FILL[k], "assign_fill {note}");
                assert_eq!(row.tail_rep, TAIL_REP[k], "tail_rep {note}");
                assert_eq!(row.tail_lit, TAIL_LIT[k], "tail_lit {note}");
                assert_eq!(row.enum_el, ENUM_EL, "enum_el {note}");
                assert_eq!(row.vec_el, VEC_EL[k], "vec_el {note}");
                assert_eq!(row.method_arr, METHOD_ARR[k], "method_arr {note}");
                assert_eq!(row.method_n, METHOD_N[k], "method_n {note}");
                assert_eq!(row.shadow, SHADOW[k], "shadow {note}");
                assert_eq!(row.order_a, ORDER_A, "order_a {note}");
                assert_eq!(row.order_g, ORDER_G, "order_g {note}");
                assert_eq!(row.order_f, ORDER_F, "order_f {note}");
                assert_eq!(row.order_h, ORDER_H, "order_h {note}");
                assert_eq!(row.order_i, ORDER_I, "order_i {note}");
                assert_eq!(row.order_i_other, ORDER_I_OTHER, "order_i_other {note}");
                assert_eq!(row.order_j, ORDER_J, "order_j {note}");
                assert_eq!(row.order_b, ORDER_B, "order_b {note}");
                assert_eq!(row.order_pure, ORDER_PURE, "order_pure {note}");
                assert_eq!(
                    row.order_pure_compound, ORDER_PURE_COMPOUND,
                    "order_pure_compound {note}"
                );
                assert_eq!(row.order_i_mid, ORDER_I_MID, "order_i_mid {note}");
                assert_eq!(row.idx_signed, IDX_SIGNED[k], "idx_signed {note}");
                assert_eq!(row.idx_bool, IDX_BOOL[k], "idx_bool {note}");
                assert_eq!(row.ops_local, OPS[k], "ops_local {note}");
                assert_eq!(row.ops_param, OPS[k], "ops_param {note}");
            }
        }
    }
}

#[test]
fn the_arrays_fixture_compiles_to_wgsl_with_its_arrays() {
    let descriptor = &arrays::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
    // A Slang array local is a WGSL `array<T, i32(N)>` value (a returned array is an out-pointer); the 17-element repeat is among them.
    assert!(wgsl.contains("array<u32, i32(17)>"), "{wgsl}");
}

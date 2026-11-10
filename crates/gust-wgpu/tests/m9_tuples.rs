include!("../../../tests/fixtures/tuples.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the fixture's rules (not computed by a function that mirrors the kernel).
//
// Element i of `input` is i % 5, so each table below repeats every five rows: v = 0, 1, 2, 3, 4.
// `classify(v)` is (2v + 1, v / 2): (1, 0), (3, 0.5), (5, 1), (7, 1.5), (9, 2).
//   ret_a        = the first element of `let (a, b) = classify(v)`            : 1, 3, 5, 7, 9
//   ret_b        = the second                                                 : 0, 0.5, 1, 1.5, 2
//   second       = `classify(v).1` read as `t.1`, plus 0.5                     : 0.5, 1, 1.5, 2, 2.5
//   typed        = `let (tc, td): (u32, f32) = classify(v + 1)`; tc = 2v + 3, plus 2 : 5, 7, 9, 11, 13
//   typed_f      = td = (v + 1) / 2                                           : 0.5, 1, 1.5, 2, 2.5
//   param        = weigh((2v + 1, v / 2), 1.0) = (2v + 1) * 1 + v / 2         : 1, 3.5, 6, 8.5, 11
//   strukt       = `with_pt(v)` is (Pt { a: v + 1, b: v + 1 }, v + 1): a + n + b as u32 = 3 (v + 1) : 3, 6, 9, 12, 15
//   enum_cast    = (Low = 1, Mid = 2, High = 3 for v = 0, 1, else) * 10 + v    : 10, 21, 32, 33, 34
//   eight        = v + (v + 1) + .. + (v + 7) = 8v + 28                         : 28, 36, 44, 52, 60
//   eight_last   = element 7 = v + 7                                          : 7, 8, 9, 10, 11
//   assign_a     = `u.0 = u.0 * 2 + 99` with u.0 = 2v + 1 = 4v + 101            : 101, 105, 109, 113, 117
//   assign_b     = `u.1 += 1.0` with u.1 = v / 2                              : 1, 1.5, 2, 2.5, 3
//   nested       = `((v, v + 1), v + 2)` destructured: 3v + 3                 : 3, 6, 9, 12, 15
//   nested_chain = `q.0.1 + q.1` = (v + 1) + (v + 2) = 2v + 3                 : 3, 5, 7, 9, 11
//   literal_bind = `let (x, y) = (v + 1, v * 2)`: x * 10 + y = 12v + 10        : 10, 22, 34, 46, 58
//   value_match  = a tuple from a value `match` (0 -> (1, 2), 1 or 2 -> (3, 4), `_` -> (5, 6)): .0 * 10 + .1 : 12, 34, 34, 56, 56
//   mixed        = f as u32 + (s + 4) as u32 with f = v, s = -v is 4; plus 100 when v > 1 : 4, 4, 104, 104, 104
//   vec_el       = a Vec2 element (v, 1.5): x + y                             : 1.5, 2.5, 3.5, 4.5, 5.5
// The evaluation-order cases do not depend on v (the order of Rust, written out by hand). `c.n` starts at 5, `tick`
// adds one and gives the new value:
//   order_a = `(c.n, c.tick())`: 5 is read, then tick gives 6: 5 * 100 + 6 = 506; c.n = 6
//   order_b = `(c.tick(), c.n)`: tick gives 7, then c.n is read (7): 707; c.n = 7
//   order_c = `(c, c.tick())`: c is copied first (n = 7), then tick gives 8: 7 * 100 + 8 = 708; c.n = 8
//   order_d = `(c.tick(), c)`: tick gives 9, then c is copied (n = 9): 9 * 100 + 9 = 909; c.n = 9
//   order_e = `let (ox, oy) = (c.tick(), c.tick())`: 10 then 11: 10 * 100 + 11 = 1011; c.n = 11
//   order_f = `use_pair((d.tick(), d.tick()))` with d.n = 0: (1, 2): 1 * 10 + 2 = 12
//   order_g / order_g_n = `sh.shift((e.tick(), e.tick()))` with sh.n = 10, e.n = 0: the argument is (1, 2), shift adds
//                 1 (sh.n = 11) and gives 11 * 100 + 2 = 1102
//   order_h = `c.pair().1 + c.tick()` with c.n = 11: pair is (11, 12), so .1 = 12, then tick gives 12: 24; c.n = 12
//   order_i = `c.tick() + c.pair().1`: tick gives 13, then pair is (13, 14): 13 + 14 = 27; c.n = 13
//   order_n = 13
// Element places against `&mut self` calls: Rust evaluates the value of `place op= value` before a primitive place.
// With g.n = v, tg = (1, 10), th = (Trace { n: v }, 0), tj = (Trace { n: v + 3 }, 0), pc.n = v:
//   place_a    = `tg.0 += g.tick()`: tick gives v + 1, so tg.0 = 1 + (v + 1) = v + 2        : 2, 3, 4, 5, 6
//   place_b    = `tg.1 = tg.0 + g.tick()`: tick gives v + 2: (v + 2) + (v + 2) = 2v + 4     : 4, 6, 8, 10, 12
//   place_add  = `th.0.n += th.0.tick()`: tick makes n = v + 1 and gives v + 1: n = 2v + 2    : 2, 4, 6, 8, 10
//   place_div  = `tj.0.n /= tj.0.tick()`: tick makes n = v + 4 and gives v + 4: n = 1        : 1, 1, 1, 1, 1
//   place_pair = `pc.n + pc.tick_pair().1`: pc.n is read (v), then tick_pair is (v + 1, v + 11): 2v + 11 : 11, 13, 15, 17, 19
const RET_A: [u32; 5] = [1, 3, 5, 7, 9];
const RET_B: [f32; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];
const SECOND: [f32; 5] = [0.5, 1.0, 1.5, 2.0, 2.5];
const TYPED: [u32; 5] = [5, 7, 9, 11, 13];
const TYPED_F: [f32; 5] = [0.5, 1.0, 1.5, 2.0, 2.5];
const PARAM: [f32; 5] = [1.0, 3.5, 6.0, 8.5, 11.0];
const STRUKT: [u32; 5] = [3, 6, 9, 12, 15];
const ENUM_CAST: [u32; 5] = [10, 21, 32, 33, 34];
const EIGHT: [u32; 5] = [28, 36, 44, 52, 60];
const EIGHT_LAST: [u32; 5] = [7, 8, 9, 10, 11];
const ASSIGN_A: [u32; 5] = [101, 105, 109, 113, 117];
const ASSIGN_B: [f32; 5] = [1.0, 1.5, 2.0, 2.5, 3.0];
const NESTED: [u32; 5] = [3, 6, 9, 12, 15];
const NESTED_CHAIN: [u32; 5] = [3, 5, 7, 9, 11];
const LITERAL_BIND: [u32; 5] = [10, 22, 34, 46, 58];
const VALUE_MATCH: [u32; 5] = [12, 34, 34, 56, 56];
const MIXED: [u32; 5] = [4, 4, 104, 104, 104];
const VEC_EL: [f32; 5] = [1.5, 2.5, 3.5, 4.5, 5.5];
const ORDER_A: u32 = 506;
const ORDER_B: u32 = 707;
const ORDER_C: u32 = 708;
const ORDER_D: u32 = 909;
const ORDER_E: u32 = 1011;
const ORDER_F: u32 = 12;
const ORDER_G: u32 = 1102;
const ORDER_G_N: u32 = 11;
const ORDER_H: u32 = 24;
const ORDER_I: u32 = 27;
const ORDER_N: u32 = 13;
const PLACE_A: [u32; 5] = [2, 3, 4, 5, 6];
const PLACE_B: [u32; 5] = [4, 6, 8, 10, 12];
const PLACE_ADD: [u32; 5] = [2, 4, 6, 8, 10];
const PLACE_DIV: [u32; 5] = [1, 1, 1, 1, 1];
const PLACE_PAIR: [u32; 5] = [11, 13, 15, 17, 19];

#[test]
fn tuples_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("tuples adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| (i % 5) as u32).collect();
        let input_buf =
            device.create_typed_buffer("tuples input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            tuples::Results {
                ret_a: u32::MAX,
                ret_b: -1.0,
                second: -1.0,
                typed: u32::MAX,
                typed_f: -1.0,
                param: -1.0,
                strukt: u32::MAX,
                enum_cast: u32::MAX,
                eight: u32::MAX,
                eight_last: u32::MAX,
                assign_a: u32::MAX,
                assign_b: -1.0,
                nested: u32::MAX,
                nested_chain: u32::MAX,
                literal_bind: u32::MAX,
                value_match: u32::MAX,
                mixed: u32::MAX,
                vec_el: -1.0,
                order_a: u32::MAX,
                order_b: u32::MAX,
                order_c: u32::MAX,
                order_d: u32::MAX,
                order_e: u32::MAX,
                order_f: u32::MAX,
                order_g: u32::MAX,
                order_g_n: u32::MAX,
                order_h: u32::MAX,
                order_i: u32::MAX,
                order_n: u32::MAX,
                place_a: u32::MAX,
                place_b: u32::MAX,
                place_add: u32::MAX,
                place_div: u32::MAX,
                place_pair: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("tuples results", &blank, GpuBufferAccess::ReadWrite);
        let call = tuples::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        tuples::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for (i, row) in got.iter().enumerate() {
                let note = format!("{name} [{i}], count {count}");
                let k = i % 5;
                assert_eq!(row.ret_a, RET_A[k], "ret_a {note}");
                assert_eq!(row.ret_b, RET_B[k], "ret_b {note}");
                assert_eq!(row.second, SECOND[k], "second {note}");
                assert_eq!(row.typed, TYPED[k], "typed {note}");
                assert_eq!(row.typed_f, TYPED_F[k], "typed_f {note}");
                assert_eq!(row.param, PARAM[k], "param {note}");
                assert_eq!(row.strukt, STRUKT[k], "strukt {note}");
                assert_eq!(row.enum_cast, ENUM_CAST[k], "enum_cast {note}");
                assert_eq!(row.eight, EIGHT[k], "eight {note}");
                assert_eq!(row.eight_last, EIGHT_LAST[k], "eight_last {note}");
                assert_eq!(row.assign_a, ASSIGN_A[k], "assign_a {note}");
                assert_eq!(row.assign_b, ASSIGN_B[k], "assign_b {note}");
                assert_eq!(row.nested, NESTED[k], "nested {note}");
                assert_eq!(row.nested_chain, NESTED_CHAIN[k], "nested_chain {note}");
                assert_eq!(row.literal_bind, LITERAL_BIND[k], "literal_bind {note}");
                assert_eq!(row.value_match, VALUE_MATCH[k], "value_match {note}");
                assert_eq!(row.mixed, MIXED[k], "mixed {note}");
                assert_eq!(row.vec_el, VEC_EL[k], "vec_el {note}");
                assert_eq!(row.order_a, ORDER_A, "order_a {note}");
                assert_eq!(row.order_b, ORDER_B, "order_b {note}");
                assert_eq!(row.order_c, ORDER_C, "order_c {note}");
                assert_eq!(row.order_d, ORDER_D, "order_d {note}");
                assert_eq!(row.order_e, ORDER_E, "order_e {note}");
                assert_eq!(row.order_f, ORDER_F, "order_f {note}");
                assert_eq!(row.order_g, ORDER_G, "order_g {note}");
                assert_eq!(row.order_g_n, ORDER_G_N, "order_g_n {note}");
                assert_eq!(row.order_h, ORDER_H, "order_h {note}");
                assert_eq!(row.order_i, ORDER_I, "order_i {note}");
                assert_eq!(row.order_n, ORDER_N, "order_n {note}");
                assert_eq!(row.place_a, PLACE_A[k], "place_a {note}");
                assert_eq!(row.place_b, PLACE_B[k], "place_b {note}");
                assert_eq!(row.place_add, PLACE_ADD[k], "place_add {note}");
                assert_eq!(row.place_div, PLACE_DIV[k], "place_div {note}");
                assert_eq!(row.place_pair, PLACE_PAIR[k], "place_pair {note}");
            }
        }
    }
}

#[test]
fn the_tuples_fixture_compiles_to_wgsl_with_its_tuple_values() {
    let descriptor = &tuples::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
    // The tuple elements are members `_0` and `_1` of a generated struct (`makeTuple(a, b)` and `t._0` in the Slang).
    assert!(wgsl.contains("_0") && wgsl.contains("_1"), "{wgsl}");
}

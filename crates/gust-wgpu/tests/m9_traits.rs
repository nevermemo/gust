include!("../../../tests/fixtures/traits.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the rules of the fixture (not computed by a function that mirrors the kernel).
//
// Element i of `input` is x = (i % 5) * 2, so the five columns below repeat: x = 0, 2, 4, 6, 8.
// r = Rect { w: x, h: 3 }, s = Square { side: x + 1 }; `bias(v)` is v + 1000; Rect::tag = bias(perimeter), perimeter =
// 2 * (w + h); Square::tag = bias(side); Kind::Small.tag = 1 and Kind::Big.tag = 2.
//   rect_area       = 3x                                  : 0, 6, 12, 18, 24
//   square_area     = (x+1)^2                             : 1, 9, 25, 49, 81
//   rect_doubled    = (2x * 6) = 12x    (self -> Self)    : 0, 24, 48, 72, 96
//   square_doubled  = (2(x+1))^2                          : 4, 36, 100, 196, 324
//   grown           = (x+2) * 3         (&mut self, +2)   : 6, 12, 18, 24, 30
//   grown_square    = (x+5)^2           (&mut self, +5)   : 25, 49, 81, 121, 169
//   kind_small = 1, kind_big = 2        (a unit enum implements the trait)
//   two_traits      = 3x + (1000 + 2(x+3)) = 1006 + 5x    : 1006, 1016, 1026, 1036, 1046
//   helper_call     = 3x + 1            (trait call in a helper)                  : 1, 7, 13, 19, 25
//   closure_call    = 2x + (1000 + 2(x+2)) = 1004 + 4x    (trait calls in a closure): 1004, 1012, 1020, 1028, 1036
//   path_ref        = 3x                (T::m(&r))                                : 0, 6, 12, 18, 24
//   path_value      = 4x                (T::doubled(pv) with pv = {1, x}, area 2 * 2x) : 0, 8, 16, 24, 32
//   path_qualified  = (x+1)^2           (<Square as Shape>::area(&s))             : 1, 9, 25, 49, 81
//   path_mut        = (x+1)^2           (Shape::grow(&mut pm, 1), pm = {x, x})    : 1, 9, 25, 49, 81
//   method_helper   = 1006 + 2x         (a trait method that calls a helper and an inherent method) : 1006, 1010, 1014, 1018, 1022
// Order (Rust: left to right, and a by-value `self` is copied before the arguments run):
//   order_args      = s.mix(c.bump(), c.bump()), mix = side * 1000 + a * 10 + b, side = x + 1, a = 1, b = 2:
//                     (x+1) * 1000 + 12 : 1012, 3012, 5012, 7012, 9012 (a trait method with two arguments, left to right)
//   order_args_path = Mix::mix(&s, c2.bump(), c2.bump()): the same call  : 1012, 3012, 5012, 7012, 9012
//   order_self      = d.plus(d.bump()) with d.n = x: self copied first (n = x), the argument is x + 1,
//                     plus = n * 100 + k = 100x + x + 1 = 101x + 1 : 1, 203, 405, 607, 809
//   order_self_path = Plus::plus(e, e.bump()): the same call          : 1, 203, 405, 607, 809
// method_score = r.score() = 2 * area + tag = 6x + 1006 + 2x = 1006 + 8x (an inherent method that calls trait methods): 1006, 1022, 1038, 1054, 1070
// pk = Square { side: x + 2 }:
//   pack_sum   = pk.sum2([x, 1]) = (x+2) + x + 1 = 2x + 3             (an array parameter)       : 3, 7, 11, 15, 19
//   pack_both  = pk.both() = (x+2, 2(x+2)) as .0 * 100 + .1 = 102(x+2) (a tuple result)   : 204, 408, 612, 816, 1020
const RECT_AREA: [u32; 5] = [0, 6, 12, 18, 24];
const SQUARE_AREA: [u32; 5] = [1, 9, 25, 49, 81];
const RECT_DOUBLED: [u32; 5] = [0, 24, 48, 72, 96];
const SQUARE_DOUBLED: [u32; 5] = [4, 36, 100, 196, 324];
const GROWN: [u32; 5] = [6, 12, 18, 24, 30];
const GROWN_SQUARE: [u32; 5] = [25, 49, 81, 121, 169];
const TWO_TRAITS: [u32; 5] = [1006, 1016, 1026, 1036, 1046];
const HELPER_CALL: [u32; 5] = [1, 7, 13, 19, 25];
const CLOSURE_CALL: [u32; 5] = [1004, 1012, 1020, 1028, 1036];
const PATH_VALUE: [u32; 5] = [0, 8, 16, 24, 32];
const METHOD_HELPER: [u32; 5] = [1006, 1010, 1014, 1018, 1022];
const ORDER_ARGS: [u32; 5] = [1012, 3012, 5012, 7012, 9012];
const METHOD_SCORE: [u32; 5] = [1006, 1022, 1038, 1054, 1070];
const PACK_SUM: [u32; 5] = [3, 7, 11, 15, 19];
const PACK_BOTH: [u32; 5] = [204, 408, 612, 816, 1020];
const ORDER_SELF: [u32; 5] = [1, 203, 405, 607, 809];

#[test]
#[allow(clippy::needless_range_loop)]
fn traits_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("traits adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| (i % 5) as u32 * 2).collect();
        let input_buf =
            device.create_typed_buffer("traits input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            traits::Results {
                rect_area: u32::MAX,
                square_area: u32::MAX,
                rect_doubled: u32::MAX,
                square_doubled: u32::MAX,
                grown: u32::MAX,
                grown_square: u32::MAX,
                kind_small: u32::MAX,
                kind_big: u32::MAX,
                two_traits: u32::MAX,
                helper_call: u32::MAX,
                closure_call: u32::MAX,
                path_ref: u32::MAX,
                path_value: u32::MAX,
                path_qualified: u32::MAX,
                path_mut: u32::MAX,
                order_args: u32::MAX,
                order_args_path: u32::MAX,
                order_self: u32::MAX,
                order_self_path: u32::MAX,
                method_helper: u32::MAX,
                method_score: u32::MAX,
                pack_sum: u32::MAX,
                pack_both: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("traits results", &blank, GpuBufferAccess::ReadWrite);
        let call = traits::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        traits::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        assert_eq!(twin.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for i in 0..count {
                let k = i % 5;
                let note = format!("{name} [{i}], count {count}");
                assert_eq!(got[i].rect_area, RECT_AREA[k], "rect_area {note}");
                assert_eq!(got[i].square_area, SQUARE_AREA[k], "square_area {note}");
                assert_eq!(got[i].rect_doubled, RECT_DOUBLED[k], "rect_doubled {note}");
                assert_eq!(
                    got[i].square_doubled, SQUARE_DOUBLED[k],
                    "square_doubled {note}"
                );
                assert_eq!(got[i].grown, GROWN[k], "grown {note}");
                assert_eq!(got[i].grown_square, GROWN_SQUARE[k], "grown_square {note}");
                assert_eq!(got[i].kind_small, 1, "kind_small {note}");
                assert_eq!(got[i].kind_big, 2, "kind_big {note}");
                assert_eq!(got[i].two_traits, TWO_TRAITS[k], "two_traits {note}");
                assert_eq!(got[i].helper_call, HELPER_CALL[k], "helper_call {note}");
                assert_eq!(got[i].closure_call, CLOSURE_CALL[k], "closure_call {note}");
                assert_eq!(got[i].path_ref, RECT_AREA[k], "path_ref {note}");
                assert_eq!(got[i].path_value, PATH_VALUE[k], "path_value {note}");
                assert_eq!(
                    got[i].path_qualified, SQUARE_AREA[k],
                    "path_qualified {note}"
                );
                assert_eq!(got[i].path_mut, SQUARE_AREA[k], "path_mut {note}");
                assert_eq!(got[i].order_args, ORDER_ARGS[k], "order_args {note}");
                assert_eq!(
                    got[i].order_args_path, ORDER_ARGS[k],
                    "order_args_path {note}"
                );
                assert_eq!(got[i].method_score, METHOD_SCORE[k], "method_score {note}");
                assert_eq!(got[i].pack_sum, PACK_SUM[k], "pack_sum {note}");
                assert_eq!(got[i].pack_both, PACK_BOTH[k], "pack_both {note}");
                assert_eq!(got[i].order_self, ORDER_SELF[k], "order_self {note}");
                assert_eq!(
                    got[i].order_self_path, ORDER_SELF[k],
                    "order_self_path {note}"
                );
                assert_eq!(
                    got[i].method_helper, METHOD_HELPER[k],
                    "method_helper {note}"
                );
            }
        }
    }
}

#[test]
fn the_trait_fixture_compiles_to_wgsl() {
    let descriptor = &traits::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
}

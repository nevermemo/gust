include!("../../../tests/fixtures/generics.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the rules of the fixture (not computed by a function that mirrors the kernel).
//
// Element i of `input` is x = (i % 5) * 2, so the five columns repeat: x = 0, 2, 4, 6, 8.
// r = Rect { w: x, h: 3 }, s = Square { side: x + 1 }, bag = Bag { items: [x, 1, 2, 3] }.
// areas: Rect 3x, Square (x+1)^2, Bag x + 6. ident: Rect 100w + h, Square 7000 + side, Bag 10 * items[0] + items[3],
// Kind::Small 1, Kind::Big 2. mix(a, b) = side * 1000 + 10a + b. plus(k) = n * 100 + k.
//   sum_rect     = area(r) + area(Rect { 1, x }) = 4x                         : 0, 8, 16, 24, 32
//   sum_square   = (x+1)^2 + 2^2                                               : 5, 13, 29, 53, 85
//   desc_rect    = 10 * 3x + (100x + 3) = 130x + 3                             : 3, 263, 523, 783, 1043
//   desc_square  = 10 (x+1)^2 + 7000 + (x+1)                                   : 7011, 7093, 7255, 7497, 7819
//   desc_bag     = 10 (x+6) + (10x + 3) = 20x + 63                             : 63, 103, 143, 183, 223
//   label_small  = 1 + 1 = 2, label_big = 2 + 1 = 3 (a `where` clause, a unit enum)
//   label_rect   = 100x + 3 + 1                                                : 4, 204, 404, 604, 804
//   quad_rect    = area after two doublings = (4x) * 12 = 48x                  : 0, 96, 192, 288, 384
//   quad_square  = (4 (x+1))^2                                                 : 16, 144, 400, 784, 1296
//   quad_bag     = 4 (x+6)                                                     : 24, 32, 40, 48, 56
//   grown_rect   = grow(2) then grow(3): (x+5) * 8                             : 40, 56, 72, 88, 104
//   grown_square = (x+1+5)^2                                                   : 36, 64, 100, 144, 196
//   grown_bag    = items[0] + 5, items[3] + 5: (x+5) + 1 + 2 + 8 = x + 16      : 16, 18, 20, 22, 24
//   chain_rect   = describe + sum_two(r, r) + label = (130x+3) + 6x + (100x+4) : 7, 479, 951, 1423, 1895
//   chain_square = desc_square + 2 (x+1)^2 + (7000 + (x+1) + 1)                : 14015, 14115, 14311, 14603, 14991
//   method_fits  = 10 * 3x + 5                                                 : 5, 65, 125, 185, 245
//   method_fits_bag = 10 (x+6) + 5                                             : 65, 85, 105, 125, 145
//   method_weigh = h.weigh(s, c2.bump(), c2.bump()), c2.n = x: a = x+1, b = x+2, k = 5: (x+1) * 1000 + 10 (x+1) + (x+2) + 5 = 1011x + 1017 : 1017, 3039, 5061, 7083, 9105 (swapped arguments would give 1011x + 1026)
// Order (Rust: left to right, and a by-value T is copied before a later argument mutates its source):
//   order_args   = gmix(s, c.bump(), c.bump()), c.n = x: a = x+1, b = x+2     : 1012, 3034, 5056, 7078, 9100
//   order_self   = gplus(d, d.bump()): self copied first (n = x), k = x+1: 101x + 1 : 1, 203, 405, 607, 809
//   order_inner  = gmix_local(s): bump, bump on a counter at 0: a = 1, b = 2   : 1012, 3012, 5012, 7012, 9012
//   order_method = e.run_with(s), e.n = x: a = x+1, b = x+2                    : 1012, 3034, 5056, 7078, 9100
//   via_closure  = sum_two(q, q) + label(q), q = Rect { x, 2 } = 4x + 100x + 3 : 3, 211, 419, 627, 835
//   via_trait    = s.score() = (x+1)^2 + 1 + 3                                 : 5, 13, 29, 53, 85
const SUM_RECT: [u32; 5] = [0, 8, 16, 24, 32];
const SUM_SQUARE: [u32; 5] = [5, 13, 29, 53, 85];
const DESC_RECT: [u32; 5] = [3, 263, 523, 783, 1043];
const DESC_SQUARE: [u32; 5] = [7011, 7093, 7255, 7497, 7819];
const DESC_BAG: [u32; 5] = [63, 103, 143, 183, 223];
const LABEL_RECT: [u32; 5] = [4, 204, 404, 604, 804];
const QUAD_RECT: [u32; 5] = [0, 96, 192, 288, 384];
const QUAD_SQUARE: [u32; 5] = [16, 144, 400, 784, 1296];
const QUAD_BAG: [u32; 5] = [24, 32, 40, 48, 56];
const GROWN_RECT: [u32; 5] = [40, 56, 72, 88, 104];
const GROWN_SQUARE: [u32; 5] = [36, 64, 100, 144, 196];
const GROWN_BAG: [u32; 5] = [16, 18, 20, 22, 24];
const CHAIN_RECT: [u32; 5] = [7, 479, 951, 1423, 1895];
const CHAIN_SQUARE: [u32; 5] = [14015, 14115, 14311, 14603, 14991];
const METHOD_FITS: [u32; 5] = [5, 65, 125, 185, 245];
const METHOD_FITS_BAG: [u32; 5] = [65, 85, 105, 125, 145];
const METHOD_WEIGH: [u32; 5] = [1017, 3039, 5061, 7083, 9105];
const ORDER_ARGS: [u32; 5] = [1012, 3034, 5056, 7078, 9100];
const ORDER_SELF: [u32; 5] = [1, 203, 405, 607, 809];
const ORDER_INNER: [u32; 5] = [1012, 3012, 5012, 7012, 9012];
const VIA_CLOSURE: [u32; 5] = [3, 211, 419, 627, 835];

#[test]
#[allow(clippy::needless_range_loop)]
fn generics_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("generics adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| (i % 5) as u32 * 2).collect();
        let input_buf =
            device.create_typed_buffer("generics input", &input, GpuBufferAccess::ReadOnly);
        let m = u32::MAX;
        let blank = vec![
            generics::Results {
                sum_rect: m,
                sum_square: m,
                desc_rect: m,
                desc_square: m,
                desc_bag: m,
                label_small: m,
                label_big: m,
                label_rect: m,
                quad_rect: m,
                quad_square: m,
                quad_bag: m,
                grown_rect: m,
                grown_square: m,
                grown_bag: m,
                chain_rect: m,
                chain_square: m,
                method_fits: m,
                method_fits_bag: m,
                method_weigh: m,
                order_args: m,
                order_self: m,
                order_inner: m,
                order_method: m,
                via_closure: m,
                via_trait: m,
            };
            count
        ];
        let results =
            device.create_typed_buffer("generics results", &blank, GpuBufferAccess::ReadWrite);
        let call = generics::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        generics::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        assert_eq!(twin.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for i in 0..count {
                let k = i % 5;
                let n = format!("{name} [{i}], count {count}");
                let g = &got[i];
                assert_eq!(g.sum_rect, SUM_RECT[k], "sum_rect {n}");
                assert_eq!(g.sum_square, SUM_SQUARE[k], "sum_square {n}");
                assert_eq!(g.desc_rect, DESC_RECT[k], "desc_rect {n}");
                assert_eq!(g.desc_square, DESC_SQUARE[k], "desc_square {n}");
                assert_eq!(g.desc_bag, DESC_BAG[k], "desc_bag {n}");
                assert_eq!(g.label_small, 2, "label_small {n}");
                assert_eq!(g.label_big, 3, "label_big {n}");
                assert_eq!(g.label_rect, LABEL_RECT[k], "label_rect {n}");
                assert_eq!(g.quad_rect, QUAD_RECT[k], "quad_rect {n}");
                assert_eq!(g.quad_square, QUAD_SQUARE[k], "quad_square {n}");
                assert_eq!(g.quad_bag, QUAD_BAG[k], "quad_bag {n}");
                assert_eq!(g.grown_rect, GROWN_RECT[k], "grown_rect {n}");
                assert_eq!(g.grown_square, GROWN_SQUARE[k], "grown_square {n}");
                assert_eq!(g.grown_bag, GROWN_BAG[k], "grown_bag {n}");
                assert_eq!(g.chain_rect, CHAIN_RECT[k], "chain_rect {n}");
                assert_eq!(g.chain_square, CHAIN_SQUARE[k], "chain_square {n}");
                assert_eq!(g.method_fits, METHOD_FITS[k], "method_fits {n}");
                assert_eq!(g.method_fits_bag, METHOD_FITS_BAG[k], "method_fits_bag {n}");
                assert_eq!(g.method_weigh, METHOD_WEIGH[k], "method_weigh {n}");
                assert_eq!(g.order_args, ORDER_ARGS[k], "order_args {n}");
                assert_eq!(g.order_self, ORDER_SELF[k], "order_self {n}");
                assert_eq!(g.order_inner, ORDER_INNER[k], "order_inner {n}");
                assert_eq!(g.order_method, ORDER_ARGS[k], "order_method {n}");
                assert_eq!(g.via_closure, VIA_CLOSURE[k], "via_closure {n}");
                assert_eq!(g.via_trait, SUM_SQUARE[k], "via_trait {n}");
            }
        }
    }
}

#[test]
fn the_generics_fixture_compiles_to_wgsl() {
    let descriptor = &generics::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
}

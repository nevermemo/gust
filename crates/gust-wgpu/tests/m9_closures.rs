include!("../../../tests/fixtures/closures.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the rules of the fixture (not computed by a function that mirrors the kernel).
//
// Element i of `input` is x = (i % 5) * 2, so the five columns below repeat: x = 0, 2, 4, 6, 8.
//   added       = x + 1                 (a closure in the kernel body)                              : 1, 3, 5, 7, 9
//   poly        = x*x + (x+1)*(x+1)     (a closure in a helper, called twice)                       : 1, 13, 41, 85, 145
//   scaled      = x * 3                 (a closure in a method body: `mul(self.n, k)`, n = x, k = 3): 0, 6, 12, 18, 24
//   via_helper  = (x + 100) * 2         (a closure that calls the helper `shift`)                   : 200, 204, 208, 212, 216
//   via_method  = x * 5                 (a closure that takes a struct and calls its method, k = 5) : 0, 10, 20, 30, 40
//   looped      = 6 + 4x                (a closure called in a loop: k = 0..3 adds 6, and 4 times x): 6, 14, 22, 30, 38
//   float_kind  = x * 0.25 + 0.5        (an f32 closure; all values are exact in binary)            : 0.5, 1.0, 1.5, 2.0, 2.5
//   int_kind    = x - 7                 (an i32 closure)                                            : -7, -5, -3, -1, 1
//   bool_kind   = 1 when x > 3 and not x > 6 (two bool parameters)                                  : 0, 0, 1, 1, 0
//   vector_kind = x + 1 + 2 + 3         (a Vec2 and a Vec4 parameter: v.x + v.y + w.z + w.w)        : 6, 8, 10, 12, 14
//   matrix_kind = 2x                    (a Mat2 with columns (2,0) and (0,3) times (x,1), x part)   : 0, 4, 8, 12, 16
//   struct_kind = x + 3                 (a struct parameter: n = x, step = 3)                       : 3, 5, 7, 9, 11
//   enum_kind   = 10 for A, 20 for B    (the mode is B when x > 3)                                  : 10, 10, 20, 20, 20
//   tuple_kind  = x * 10 + 1            (a (u32, u32) parameter with (x, 1))                        : 1, 21, 41, 61, 81
//   array_kind  = x + 1 + 2             (a [u32; 3] parameter with [x, 1, 2])                       : 3, 5, 7, 9, 11
// The order the arguments of a closure call run in is the order of Rust, left to right, for a counter that starts at
// 0 with step 1 and `pair(a, b) = a * 1000 + b`:
//   order_two_ticks      = pair(tick, tick)              = 1 * 1000 + 2        = 1002
//   order_read_then_tick = pair(n, tick)                 = 0 * 1000 + 1        = 1
//   order_tick_then_read = pair(tick, n)                 = 1 * 1000 + 1        = 1001
//   order_nested         = pair(pair(tick, tick), tick)  = 1002 * 1000 + 3     = 1002003
// placed = 2 * i + 1 (a closure in the index and one in the value of `scratch[slot(i)] = odd(i)`; i is the element).
// A parameter is a copy (hand-written): a closure `|mut a: [u32; 3]| { a[0] = 9; a[0] }` called with the caller's `[x, 1, 2]`
// gives 9 and leaves the caller's a[0] = x (array_written 9, array_after x); the same for a `Counter` with `c.n = 77`
// (struct_written 77, struct_after x).
const ADDED: [u32; 5] = [1, 3, 5, 7, 9];
const POLY: [u32; 5] = [1, 13, 41, 85, 145];
const SCALED: [u32; 5] = [0, 6, 12, 18, 24];
const VIA_HELPER: [u32; 5] = [200, 204, 208, 212, 216];
const VIA_METHOD: [u32; 5] = [0, 10, 20, 30, 40];
const LOOPED: [u32; 5] = [6, 14, 22, 30, 38];
const FLOAT_KIND: [f32; 5] = [0.5, 1.0, 1.5, 2.0, 2.5];
const INT_KIND: [i32; 5] = [-7, -5, -3, -1, 1];
const BOOL_KIND: [u32; 5] = [0, 0, 1, 1, 0];
const VECTOR_KIND: [f32; 5] = [6.0, 8.0, 10.0, 12.0, 14.0];
const MATRIX_KIND: [f32; 5] = [0.0, 4.0, 8.0, 12.0, 16.0];
const STRUCT_KIND: [u32; 5] = [3, 5, 7, 9, 11];
const ENUM_KIND: [u32; 5] = [10, 10, 20, 20, 20];
const TUPLE_KIND: [u32; 5] = [1, 21, 41, 61, 81];
const ARRAY_KIND: [u32; 5] = [3, 5, 7, 9, 11];
const ORDER_TWO_TICKS: u32 = 1002;
const ORDER_READ_THEN_TICK: u32 = 1;
const ORDER_TICK_THEN_READ: u32 = 1001;
const ORDER_NESTED: u32 = 1_002_003;

#[test]
#[allow(clippy::needless_range_loop)]
fn closures_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("closures adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| (i % 5) as u32 * 2).collect();
        let input_buf =
            device.create_typed_buffer("closures input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            closures::Results {
                added: u32::MAX,
                poly: u32::MAX,
                scaled: u32::MAX,
                via_helper: u32::MAX,
                via_method: u32::MAX,
                looped: u32::MAX,
                float_kind: f32::MAX,
                int_kind: i32::MAX,
                bool_kind: u32::MAX,
                vector_kind: f32::MAX,
                matrix_kind: f32::MAX,
                struct_kind: u32::MAX,
                enum_kind: u32::MAX,
                tuple_kind: u32::MAX,
                array_kind: u32::MAX,
                order_two_ticks: u32::MAX,
                order_read_then_tick: u32::MAX,
                order_tick_then_read: u32::MAX,
                order_nested: u32::MAX,
                placed: u32::MAX,
                array_written: u32::MAX,
                array_after: u32::MAX,
                struct_written: u32::MAX,
                struct_after: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("closures results", &blank, GpuBufferAccess::ReadWrite);
        let scratch_blank = vec![u32::MAX; count];
        let scratch = device.create_typed_buffer(
            "closures scratch",
            &scratch_blank,
            GpuBufferAccess::ReadWrite,
        );
        let call = closures::run::call(count as u32, &input_buf, &results, &scratch);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        let twin_scratch = cpu.buffer(&scratch_blank);
        closures::run::dispatch(
            &cpu,
            count as u32,
            &twin_input,
            &twin_results,
            &twin_scratch,
        )
        .unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        assert_eq!(twin.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for i in 0..count {
                let k = i % 5;
                let note = format!("{name} [{i}], count {count}");
                assert_eq!(got[i].added, ADDED[k], "added {note}");
                assert_eq!(got[i].poly, POLY[k], "poly {note}");
                assert_eq!(got[i].scaled, SCALED[k], "scaled {note}");
                assert_eq!(got[i].via_helper, VIA_HELPER[k], "via_helper {note}");
                assert_eq!(got[i].via_method, VIA_METHOD[k], "via_method {note}");
                assert_eq!(got[i].looped, LOOPED[k], "looped {note}");
                assert_eq!(got[i].float_kind, FLOAT_KIND[k], "float_kind {note}");
                assert_eq!(got[i].int_kind, INT_KIND[k], "int_kind {note}");
                assert_eq!(got[i].bool_kind, BOOL_KIND[k], "bool_kind {note}");
                assert_eq!(got[i].vector_kind, VECTOR_KIND[k], "vector_kind {note}");
                assert_eq!(got[i].matrix_kind, MATRIX_KIND[k], "matrix_kind {note}");
                assert_eq!(got[i].struct_kind, STRUCT_KIND[k], "struct_kind {note}");
                assert_eq!(got[i].enum_kind, ENUM_KIND[k], "enum_kind {note}");
                assert_eq!(got[i].tuple_kind, TUPLE_KIND[k], "tuple_kind {note}");
                assert_eq!(got[i].array_kind, ARRAY_KIND[k], "array_kind {note}");
                assert_eq!(
                    got[i].order_two_ticks, ORDER_TWO_TICKS,
                    "order_two_ticks {note}"
                );
                assert_eq!(
                    got[i].order_read_then_tick, ORDER_READ_THEN_TICK,
                    "order_read_then_tick {note}"
                );
                assert_eq!(
                    got[i].order_tick_then_read, ORDER_TICK_THEN_READ,
                    "order_tick_then_read {note}"
                );
                assert_eq!(got[i].order_nested, ORDER_NESTED, "order_nested {note}");
                assert_eq!(got[i].placed, 2 * i as u32 + 1, "placed {note}");
                assert_eq!(got[i].array_written, 9, "array_written {note}");
                assert_eq!(got[i].array_after, input[i], "array_after {note}");
                assert_eq!(got[i].struct_written, 77, "struct_written {note}");
                assert_eq!(got[i].struct_after, input[i], "struct_after {note}");
            }
        }
    }
}

#[test]
fn the_closure_fixture_compiles_to_wgsl_with_its_hoisted_functions() {
    let descriptor = &closures::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
    // Slang renames the hoisted free function `__gust_cl_<owner>_<n>` in WGSL (measured on slangc 2026.21): one
    // function per closure, whose parameters are the closure's, called directly. The owner of a method closure is
    // `<Struct>_<method>`, and `n` counts the closures of the module in source order.
    for owner in ["Counter_scaled_0", "poly_1", "run_2", "run_16", "run_18"] {
        assert!(
            wgsl.contains(&format!("gust_cl_{owner}")),
            "no hoisted function for {owner} in the WGSL\n{wgsl}"
        );
    }
}

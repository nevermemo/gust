include!("../../../tests/fixtures/enums.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the fixture's rules (not computed by a function that mirrors the kernel).
//
// Element i of `input` is i % 5, so each table below repeats every five rows: v = 0, 1, 2, 3, 4.
// `mode_from(v)` is Off, Low, Mid, High, Max for v = 0..4, and `Mode` is { Off, Low = 3, Mid, High, Max = 10 }: the
// variants after `Low = 3` count up (Mid 4, High 5), and Max is explicit.
//   code      = mode as u32                                              : 0, 3, 4, 5, 10
//   grade     = classify: 5 + (5 for Off, 15 for Low or Mid, 25 for the rest) : 10, 20, 20, 30, 30
//   bumped    = bump_mode: Off -> Low, Low or Mid -> High, `_` -> Off, as u32  : 3, 5, 5, 0, 0
//   eq        = 1 when the mode is Mid                                   : 0, 0, 1, 0, 0
//   ne        = 1 when the mode is not Off                               : 0, 1, 1, 1, 1
// `sign_from(v)` picks by v % 3: 0 Neg, 1 Mid, 2 Pos, and `Sign` (repr i32) is { Neg = -2, Mid, Pos = 5 }, so Mid is -1.
//   sign      = sign_from(v) as i32                                      : -2, -1, 5, -2, -1
//   weight    = Neg 1, Mid 2, Pos 3                                      : 1, 2, 3, 1, 2
// The evaluation-order cases do not depend on v:
//   scrut     = the scrutinee `mode_from(t.tick())` runs once (n = 1, Low); the arm `Low | Mid` ticks again: n = 2    : 2
//   eq_order  = `mode_from(a.mark(1)) == mode_from(a.mark(2))` appends the digits 1 then 2 to a.n (left call first) : 12
//   ne_order  = the same with 3 and 4 and `!=`                                                                     : 34
//   arg_order = take(mode_from(u.tick()), mode_from(u.tick())): the first tick is 1 (Low, 3), the second 2 (Mid, 4),
//               and take is 3 * 100 + 4                                                                            : 304
//   built     = take(mode_from(w.mark(2)), mode_from(w.mark(3))): Mid (4) then High (5), 4 * 100 + 5; swapped it is 504  : 405
//   built_trace = w.n after those two calls (the digits 2 then 3)                                                  : 23
// `Sign` derives `PartialOrd` and orders by discriminant (-2, -1, 5):
//   less      = 1 when sign_from(v) < Sign::Mid, that is for Neg only                                              : 1, 0, 0, 1, 0
// The method cases: q.mode_of(v) is mode_from(v); add_mode adds the mode's discriminant to q.n.
//   method_code = mode_of(v) as u32, the same as code                                                              : 0, 3, 4, 5, 10
//   method_sum  = add_mode(Low) gives 3, then add_mode(mode) gives 3 + code, and the sum is 6 + code               : 6, 9, 10, 11, 16
const CODE: [u32; 5] = [0, 3, 4, 5, 10];
const GRADE: [u32; 5] = [10, 20, 20, 30, 30];
const BUMPED: [u32; 5] = [3, 5, 5, 0, 0];
const EQ: [u32; 5] = [0, 0, 1, 0, 0];
const NE: [u32; 5] = [0, 1, 1, 1, 1];
const SIGN: [i32; 5] = [-2, -1, 5, -2, -1];
const WEIGHT: [u32; 5] = [1, 2, 3, 1, 2];
const SCRUT: u32 = 2;
const EQ_ORDER: u32 = 12;
const NE_ORDER: u32 = 34;
const ARG_ORDER: u32 = 304;
const BUILT: u32 = 405;
const BUILT_TRACE: u32 = 23;
const LESS: [u32; 5] = [1, 0, 0, 1, 0];
const METHOD_SUM: [u32; 5] = [6, 9, 10, 11, 16];

#[test]
fn enums_and_matches_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("enums adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| (i % 5) as u32).collect();
        let input_buf =
            device.create_typed_buffer("enums input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            enums::Results {
                code: u32::MAX,
                grade: u32::MAX,
                bumped: u32::MAX,
                eq: u32::MAX,
                ne: u32::MAX,
                sign: i32::MIN,
                weight: u32::MAX,
                scrut: u32::MAX,
                eq_order: u32::MAX,
                ne_order: u32::MAX,
                arg_order: u32::MAX,
                built: u32::MAX,
                built_trace: u32::MAX,
                less: u32::MAX,
                method_code: u32::MAX,
                method_sum: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("enums results", &blank, GpuBufferAccess::ReadWrite);
        let call = enums::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        enums::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for i in 0..count {
                let note = format!("{name} [{i}], count {count}");
                assert_eq!(got[i].code, CODE[i % 5], "code {note}");
                assert_eq!(got[i].grade, GRADE[i % 5], "grade {note}");
                assert_eq!(got[i].bumped, BUMPED[i % 5], "bumped {note}");
                assert_eq!(got[i].eq, EQ[i % 5], "eq {note}");
                assert_eq!(got[i].ne, NE[i % 5], "ne {note}");
                assert_eq!(got[i].sign, SIGN[i % 5], "sign {note}");
                assert_eq!(got[i].weight, WEIGHT[i % 5], "weight {note}");
                assert_eq!(got[i].scrut, SCRUT, "scrut {note}");
                assert_eq!(got[i].eq_order, EQ_ORDER, "eq_order {note}");
                assert_eq!(got[i].ne_order, NE_ORDER, "ne_order {note}");
                assert_eq!(got[i].arg_order, ARG_ORDER, "arg_order {note}");
                assert_eq!(got[i].built, BUILT, "built {note}");
                assert_eq!(got[i].built_trace, BUILT_TRACE, "built_trace {note}");
                assert_eq!(got[i].less, LESS[i % 5], "less {note}");
                assert_eq!(got[i].method_code, CODE[i % 5], "method_code {note}");
                assert_eq!(got[i].method_sum, METHOD_SUM[i % 5], "method_sum {note}");
            }
        }
    }
}

#[test]
fn the_enum_fixture_compiles_to_wgsl_with_its_switches() {
    let descriptor = &enums::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
    // Measured shape: `classify`, `bump_mode`, `weight` and the kernel's `match` are four `switch` statements, each
    // ending in `default`; the discriminants (0, 3, 4 and the signed -2, -1) are the case values.
    assert_eq!(wgsl.matches("switch(").count(), 4, "{wgsl}");
    assert_eq!(wgsl.matches("default :").count(), 4, "{wgsl}");
    for label in [
        "case u32(0):",
        "case u32(3), u32(4):",
        "case i32(-2):",
        "case i32(-1):",
    ] {
        assert!(wgsl.contains(label), "{label}\n{wgsl}");
    }
}

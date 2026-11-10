include!("../../../tests/fixtures/match_value.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the fixture's rules (not computed by a function that mirrors the kernel).
//
// Element i of `input` is i % 5, so each table below repeats every five rows: v = 0, 1, 2, 3, 4.
// `mode_from(v)` is Off, Low, Mid, High, Max for v = 0..4; `Mode` is { Off, Low = 3, Mid, High, Max = 10 }, so Low is 3
// and Mid 4.
//   enum_value = (Off 1, Low or Mid: the discriminant * 10, `_` 7) + 1       : 2, 31, 41, 8, 8
//   rank       = (Off 10, Low 11, Mid 12, High 13, Max 14) + 100             : 110, 111, 112, 113, 114
//   bucket     = (0 -> 100, 1 or 2 -> 200, 3 -> 300, `_` -> 400) + 1         : 101, 201, 201, 301, 401
//   wide       = the match is on v * 10^9 (0, 1e9, 2e9, 3e9, 4e9; all below 2^32): 0 -> 1, 1e9 -> 2, 3e9 -> 3,
//                4e9 -> 4, `_` -> 5, plus 1                                    : 2, 3, 6, 4, 5
//   sgn        = on s = v - 2 (-2, -1, 0, 1, 2): -2 -> -20, -1 -> -10, 0 -> 0, 1 or 2 -> 10, minus 1 : -21, -11, -1, 9, 9
//   stmt_int   = 5 plus (0 -> 1, 1 or 2 -> 20, `_` -> 300)                   : 6, 25, 25, 305, 305
//   nested     = called with the mode and (v + 1) % 2 (1, 0, 1, 0, 1):
//                Off with 1 -> 2; Low with 0 -> inner 10, +1 = 11; Mid with 1 -> inner 10, +1 = 11;
//                High with 0 -> k = 1 -> 100; Max with 1 -> k = 2 -> 200; plus 1 : 3, 12, 12, 101, 201
//   pair_a     = p.a + 1 with p = Off (1, 0.5), Low (v + 2 = 3, 1.5), `_` (v * 2, 2.5)  : 2, 4, 5, 7, 9
//   pair_b     = p.b                                                         : 0.5, 1.5, 2.5, 2.5, 2.5
//   early      = 0 returns 99, 1 -> 5 + 1, `_` -> 6 + 1                      : 99, 6, 7, 7, 7
//   shadow     = the new `k` is v + 10 for 0 and v + 20 otherwise, plus 1    : 11, 22, 23, 24, 25
// The evaluation-order cases do not depend on v (the order of Rust, written out by hand):
//   scrut / scrut_n = `t.tick()` runs once (n = 1, matches 1 | 3 | 5), the arm ticks again (n = 2): 2 * 10 = 20, n = 2
//   arm_after / arm_before / arm_n = c.n starts at 5. `match c.n` takes 5, the arm is `c.n + c.tick()`: the left
//                operand is read first (5), tick gives 6, so 11. Then `match c.n` (6), the arm is `c.tick() + c.n`:
//                tick gives 7 and the right operand is read after it (7): 14, and c.n is 7.
//   same / same_n = `match d.tick()` (1) and the arm ticks again (2): 2 * 100 = 200, n = 2
//   taken / taken_n = only the taken arm calls `mark`: Off mark(1) -> 101, n 1; Low mark(2) -> 102, n 2; others
//                mark(3) -> 103, n 3                                         : taken 101, 102, 103, 103, 103
//   two = `g.tick() * 10 + g.tick()` with g.n = 0: the left call first (1), then the right (2): 1 * 10 + 2 = 12
//   duo_a / duo_b = `Duo { a: h.tick(), b: h.tick() }` with h.n = 0: the fields in the order written, a = 1, b = 2
const ENUM_VALUE: [u32; 5] = [2, 31, 41, 8, 8];
const RANK: [u32; 5] = [110, 111, 112, 113, 114];
const BUCKET: [u32; 5] = [101, 201, 201, 301, 401];
const WIDE: [u32; 5] = [2, 3, 6, 4, 5];
const SGN: [i32; 5] = [-21, -11, -1, 9, 9];
const STMT_INT: [u32; 5] = [6, 25, 25, 305, 305];
const NESTED: [u32; 5] = [3, 12, 12, 101, 201];
const PAIR_A: [u32; 5] = [2, 4, 5, 7, 9];
const PAIR_B: [f32; 5] = [0.5, 1.5, 2.5, 2.5, 2.5];
const EARLY: [u32; 5] = [99, 6, 7, 7, 7];
const SHADOW: [u32; 5] = [11, 22, 23, 24, 25];
const SCRUT: u32 = 20;
const SCRUT_N: u32 = 2;
const ARM_AFTER: u32 = 11;
const ARM_BEFORE: u32 = 14;
const ARM_N: u32 = 7;
const SAME: u32 = 200;
const SAME_N: u32 = 2;
const TAKEN: [u32; 5] = [101, 102, 103, 103, 103];
const TAKEN_N: [u32; 5] = [1, 2, 3, 3, 3];
const TWO: u32 = 12;
const DUO_A: u32 = 1;
const DUO_B: u32 = 2;

#[test]
fn value_matches_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("match_value adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| (i % 5) as u32).collect();
        let input_buf =
            device.create_typed_buffer("match_value input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            match_value::Results {
                enum_value: u32::MAX,
                rank: u32::MAX,
                bucket: u32::MAX,
                wide: u32::MAX,
                sgn: i32::MIN,
                stmt_int: u32::MAX,
                nested: u32::MAX,
                pair_a: u32::MAX,
                pair_b: -1.0,
                early: u32::MAX,
                shadow: u32::MAX,
                scrut: u32::MAX,
                scrut_n: u32::MAX,
                arm_after: u32::MAX,
                arm_before: u32::MAX,
                arm_n: u32::MAX,
                same: u32::MAX,
                same_n: u32::MAX,
                taken: u32::MAX,
                taken_n: u32::MAX,
                two: u32::MAX,
                duo_a: u32::MAX,
                duo_b: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("match_value results", &blank, GpuBufferAccess::ReadWrite);
        let call = match_value::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        match_value::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for (i, row) in got.iter().enumerate() {
                let note = format!("{name} [{i}], count {count}");
                let k = i % 5;
                assert_eq!(row.enum_value, ENUM_VALUE[k], "enum_value {note}");
                assert_eq!(row.rank, RANK[k], "rank {note}");
                assert_eq!(row.bucket, BUCKET[k], "bucket {note}");
                assert_eq!(row.wide, WIDE[k], "wide {note}");
                assert_eq!(row.sgn, SGN[k], "sgn {note}");
                assert_eq!(row.stmt_int, STMT_INT[k], "stmt_int {note}");
                assert_eq!(row.nested, NESTED[k], "nested {note}");
                assert_eq!(row.pair_a, PAIR_A[k], "pair_a {note}");
                assert_eq!(row.pair_b, PAIR_B[k], "pair_b {note}");
                assert_eq!(row.early, EARLY[k], "early {note}");
                assert_eq!(row.shadow, SHADOW[k], "shadow {note}");
                assert_eq!(row.scrut, SCRUT, "scrut {note}");
                assert_eq!(row.scrut_n, SCRUT_N, "scrut_n {note}");
                assert_eq!(row.arm_after, ARM_AFTER, "arm_after {note}");
                assert_eq!(row.arm_before, ARM_BEFORE, "arm_before {note}");
                assert_eq!(row.arm_n, ARM_N, "arm_n {note}");
                assert_eq!(row.same, SAME, "same {note}");
                assert_eq!(row.same_n, SAME_N, "same_n {note}");
                assert_eq!(row.taken, TAKEN[k], "taken {note}");
                assert_eq!(row.taken_n, TAKEN_N[k], "taken_n {note}");
                assert_eq!(row.two, TWO, "two {note}");
                assert_eq!(row.duo_a, DUO_A, "duo_a {note}");
                assert_eq!(row.duo_b, DUO_B, "duo_b {note}");
            }
        }
    }
}

#[test]
fn the_match_value_fixture_compiles_to_wgsl_with_its_switches() {
    let descriptor = &match_value::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
    // Counted from the fixture: one `switch` in each of enum_value, rank, bucket, wide, sgn, stmt_int, early, four in
    // nested (the outer, an arm, a `let` in an arm, the tail of an arm), nine in the kernel (p, shadow, scrut,
    // arm_after, arm_before, same, taken, two, duo).
    assert_eq!(wgsl.matches("switch(").count(), 20, "{wgsl}");
    for label in [
        "case u32(3000000000):",
        "case u32(4000000000):",
        "case i32(-2):",
        "case i32(-1):",
    ] {
        assert!(wgsl.contains(label), "{label}\n{wgsl}");
    }
}

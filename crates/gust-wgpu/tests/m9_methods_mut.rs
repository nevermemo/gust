include!("../../../tests/fixtures/methods_mut.rs");

use gust::Run;
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

// Expected values, written by hand from the fixture's rules (not computed by a function that mirrors the kernel).
//
// Element i of `input` is (i % 5) * 2, so the five rows below repeat: input = 0, 2, 4, 6, 8.
//   bumped   = input + 2            (Counter::new(input, 2), then one `bump`)          : 2, 4, 6, 8, 10
//   consumed = input + 6            (`bump_twice` adds 4 more; `value(self)` reads n)  : 6, 8, 10, 12, 14
//   scaled   = (input + 6) * 3      (a `&self` method that calls the helper `scale_by`): 18, 24, 30, 36, 42
//   mean     = (input + 16) / 6     (adds input, 10, then k = 0..3, which sum to 6, so total = input + 16 and
//                                    count = 6; integer division)                       : 2, 3, 3, 3, 4
//   looped   = 4                    (`Counter::unit()` has n 0 and step 1; four `bump`s in a loop)
//   helper   = input + 10           (the helper `drive` makes a counter with step 5 and calls `bump_twice`): 10, 12, 14, 16, 18
//   kept     = input + 2            (`let kept = counter;` after one `bump`, then two more: the copy does not follow): 2, 4, 6, 8, 10
//   after    = input + 6            (`counter.n` read after the by-value `value()` and `shifted()` calls: unchanged)  : 6, 8, 10, 12, 14
//   shifted  = input + 106          (`shifted(self)` adds 100 to its own copy)                                     : 106, 108, 110, 112, 114
//   nested   = input + 8            (`wrap.inner.bump()` twice on a field of a local, step 4)                      : 8, 10, 12, 14, 16
const BUMPED: [u32; 5] = [2, 4, 6, 8, 10];
const CONSUMED: [u32; 5] = [6, 8, 10, 12, 14];
const SCALED: [u32; 5] = [18, 24, 30, 36, 42];
const MEAN: [u32; 5] = [2, 3, 3, 3, 4];
const LOOPED: u32 = 4;
const HELPER: [u32; 5] = [10, 12, 14, 16, 18];
const KEPT: [u32; 5] = [2, 4, 6, 8, 10];
const AFTER: [u32; 5] = [6, 8, 10, 12, 14];
const SHIFTED: [u32; 5] = [106, 108, 110, 112, 114];
const NESTED: [u32; 5] = [8, 10, 12, 14, 16];

#[test]
fn mutating_methods_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("methods_mut adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| (i % 5) as u32 * 2).collect();
        let input_buf =
            device.create_typed_buffer("methods_mut input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            methods_mut::Results {
                bumped: u32::MAX,
                consumed: u32::MAX,
                scaled: u32::MAX,
                mean: u32::MAX,
                looped: u32::MAX,
                helper: u32::MAX,
                kept: u32::MAX,
                after: u32::MAX,
                shifted: u32::MAX,
                nested: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("methods_mut results", &blank, GpuBufferAccess::ReadWrite);
        let call = methods_mut::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        methods_mut::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for i in 0..count {
                let note = format!("{name} [{i}], count {count}");
                assert_eq!(got[i].bumped, BUMPED[i % 5], "bumped {note}");
                assert_eq!(got[i].consumed, CONSUMED[i % 5], "consumed {note}");
                assert_eq!(got[i].scaled, SCALED[i % 5], "scaled {note}");
                assert_eq!(got[i].mean, MEAN[i % 5], "mean {note}");
                assert_eq!(got[i].looped, LOOPED, "looped {note}");
                assert_eq!(got[i].helper, HELPER[i % 5], "helper {note}");
                assert_eq!(got[i].kept, KEPT[i % 5], "kept {note}");
                assert_eq!(got[i].after, AFTER[i % 5], "after {note}");
                assert_eq!(got[i].shifted, SHIFTED[i % 5], "shifted {note}");
                assert_eq!(got[i].nested, NESTED[i % 5], "nested {note}");
            }
        }
    }
}

#[test]
fn the_mutating_method_fixture_compiles_to_wgsl_with_its_members() {
    let descriptor = &methods_mut::run::DESCRIPTOR;
    let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
    // Slang writes a member `m` of the struct `S` as the function `S_m_<n>` (read from the real WGSL). The `new` of
    // the fixture is renamed by the emitter (`__gust_r_new`) at its declaration and at every call alike.
    for name in [
        "Counter_gust_r_new",
        "Counter_bump",
        "Counter_bump_twice",
        "Counter_value",
        "Counter_shifted",
        "Counter_scaled",
        "Counter_unit",
        "Acc_add",
        "Acc_mean",
    ] {
        assert!(
            wgsl.contains(&format!("fn {name}_")),
            "no function `{name}` in the WGSL\n{wgsl}"
        );
    }
    // `[mutating]` takes the receiver by pointer; a method on `self` by value takes it as a value. Matched on the
    // line of the function, without Slang's numbered suffixes.
    let line_of = |prefix: &str| {
        wgsl.lines()
            .find(|line| {
                line.strip_prefix(prefix)
                    .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
            })
            .unwrap_or_else(|| panic!("no `{prefix}` in the WGSL\n{wgsl}"))
    };
    assert!(
        line_of("fn Counter_bump_").contains("ptr<function, Counter_0>"),
        "{wgsl}"
    );
    assert!(!line_of("fn Counter_value_").contains("ptr<"), "{wgsl}");
}

// m9.f1: a `<` comparison that is not the last element of a list goes through the `__gust_keep` helper (WGSL reads
// `a < b, c > d` as a template list). The helper's definition is emitted by a scan of the module, which must see
// every list the emitter builds: a method call (`&self` and `&mut self`), an associated function (`S::f`,
// `Self::f`), a plain call and the operands of `^` (array values are not in the dialect before m9 r1).
//
// Hand-written values: input element i is i % 8, so a = 0..7 repeating, with x = (a < 3) and y = (a < 5).
//   a      : 0   1   2   3   4   5   6   7
//   x, y   : TT  TT  TT  FT  FT  FF  FF  FF
//   score  : x adds 1, y adds 2          : 3, 3, 3, 2, 2, 0, 0, 0   (by_ref, assoc, via_self, free)
//   by_mut : x adds 1, y adds 10 to hits : 11, 11, 11, 10, 10, 0, 0, 0
//   xor    : x ^ y as 0 or 1             : 0, 0, 0, 1, 1, 0, 0, 0
const KEEP_SCORE: [u32; 8] = [3, 3, 3, 2, 2, 0, 0, 0];
const KEEP_HITS: [u32; 8] = [11, 11, 11, 10, 10, 0, 0, 0];
const KEEP_XOR: [u32; 8] = [0, 0, 0, 1, 1, 0, 0, 0];

#[gust::gpu]
mod keep_calls {
    pub struct Probe {
        pub hits: uint,
    }

    pub struct Results {
        pub by_ref: uint,
        pub by_mut: uint,
        pub assoc: uint,
        pub via_self: uint,
        pub free: uint,
        pub xor: uint,
    }

    impl Probe {
        pub fn score(&self, x: bool, y: bool) -> uint {
            let mut r = 0u32;
            if x {
                r += 1u32;
            }
            if y {
                r += 2u32;
            }
            r
        }

        pub fn note(&mut self, x: bool, y: bool) {
            if x {
                self.hits += 1u32;
            }
            if y {
                self.hits += 10u32;
            }
        }

        pub fn pair(x: bool, y: bool) -> uint {
            let mut r = 0u32;
            if x {
                r += 1u32;
            }
            if y {
                r += 2u32;
            }
            r
        }

        pub fn via_self(&self, a: uint) -> uint {
            Self::pair(a < 3u32, 5u32 > a)
        }
    }

    fn free(x: bool, y: bool) -> uint {
        let mut r = 0u32;
        if x {
            r += 1u32;
        }
        if y {
            r += 2u32;
        }
        r
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut results: RWStructuredBuffer<Results>,
    ) {
        let i = id.x;
        if i < input.len() {
            let a = input[i];
            let mut probe = Probe { hits: 0u32 };
            let by_ref = probe.score(a < 3u32, 5u32 > a);
            probe.note(a < 3u32, 5u32 > a);
            let assoc = Probe::pair(a < 3u32, 5u32 > a);
            let via_self = probe.via_self(a);
            let free_score = free(a < 3u32, 5u32 > a);
            let mut xor = 0u32;
            if (a < 3u32) ^ (5u32 > a) {
                xor = 1u32;
            }
            results[i] = Results {
                by_ref,
                by_mut: probe.hits,
                assoc,
                via_self,
                free: free_score,
                xor,
            };
        }
    }
}

#[test]
fn keep_helper_lists_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    for count in [1, 8, 65] {
        let input: Vec<u32> = (0..count).map(|i| (i % 8) as u32).collect();
        let input_buf = device.create_typed_buffer("keep input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            keep_calls::Results {
                by_ref: u32::MAX,
                by_mut: u32::MAX,
                assoc: u32::MAX,
                via_self: u32::MAX,
                free: u32::MAX,
                xor: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("keep results", &blank, GpuBufferAccess::ReadWrite);
        let call = keep_calls::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        keep_calls::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for (i, row) in got.iter().enumerate() {
                let note = format!("{name} [{i}], count {count}");
                let a = i % 8;
                assert_eq!(row.by_ref, KEEP_SCORE[a], "by_ref {note}");
                assert_eq!(row.by_mut, KEEP_HITS[a], "by_mut {note}");
                assert_eq!(row.assoc, KEEP_SCORE[a], "assoc {note}");
                assert_eq!(row.via_self, KEEP_SCORE[a], "via_self {note}");
                assert_eq!(row.free, KEEP_SCORE[a], "free {note}");
                assert_eq!(row.xor, KEEP_XOR[a], "xor {note}");
            }
        }
    }
}

#[test]
fn the_keep_helper_module_compiles_to_wgsl() {
    let wgsl = gust::slang::compile_wgsl(&keep_calls::run::DESCRIPTOR).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
}

#[test]
fn the_keep_helper_module_compiles_to_spirv() {
    let words = gust::slang::compile_spirv(&keep_calls::run::DESCRIPTOR).unwrap();
    gust::spirv::validate_structure(&words).unwrap();
}

// The module above also has plain and associated calls with a wrapped argument, so the scan finds the helper through
// them. This module has no other wrapped list than the two method calls, so only the scan of method calls can define
// `__gust_keep` here (the seed 31 shape of the difftest, `v.m2(a[i] < b, c)`). Values as above: by_ref = KEEP_SCORE,
// by_mut = KEEP_HITS.
#[gust::gpu]
mod keep_methods {
    pub struct Probe {
        pub hits: uint,
    }

    pub struct Results {
        pub by_ref: uint,
        pub by_mut: uint,
    }

    impl Probe {
        pub fn score(&self, x: bool, y: bool) -> uint {
            let mut r = 0u32;
            if x {
                r += 1u32;
            }
            if y {
                r += 2u32;
            }
            r
        }

        pub fn note(&mut self, x: bool, y: bool) {
            if x {
                self.hits += 1u32;
            }
            if y {
                self.hits += 10u32;
            }
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut results: RWStructuredBuffer<Results>,
    ) {
        let i = id.x;
        if i < input.len() {
            let a = input[i];
            let mut probe = Probe { hits: 0u32 };
            let by_ref = probe.score(a < 3u32, 5u32 > a);
            probe.note(a < 3u32, 5u32 > a);
            results[i] = Results {
                by_ref,
                by_mut: probe.hits,
            };
        }
    }
}

#[test]
fn keep_helper_method_calls_execute_on_gpu_and_on_the_cpu_twin() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    for count in [1, 8, 65] {
        let input: Vec<u32> = (0..count).map(|i| (i % 8) as u32).collect();
        let input_buf =
            device.create_typed_buffer("keep m input", &input, GpuBufferAccess::ReadOnly);
        let blank = vec![
            keep_methods::Results {
                by_ref: u32::MAX,
                by_mut: u32::MAX,
            };
            count
        ];
        let results =
            device.create_typed_buffer("keep m results", &blank, GpuBufferAccess::ReadWrite);
        let call = keep_methods::run::call(count as u32, &input_buf, &results);
        device.dispatch_call(&call).unwrap();
        let actual = device.read_typed_buffer(&results).unwrap();

        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_results = cpu.buffer(&blank);
        keep_methods::run::dispatch(&cpu, count as u32, &twin_input, &twin_results).unwrap();
        let twin = twin_results.read();

        assert_eq!(actual.len(), count);
        for (name, got) in [("gpu", &actual), ("twin", &twin)] {
            for (i, row) in got.iter().enumerate() {
                let note = format!("{name} [{i}], count {count}");
                assert_eq!(row.by_ref, KEEP_SCORE[i % 8], "by_ref {note}");
                assert_eq!(row.by_mut, KEEP_HITS[i % 8], "by_mut {note}");
            }
        }
    }
}

#[test]
fn the_keep_method_module_compiles_to_wgsl() {
    let wgsl = gust::slang::compile_wgsl(&keep_methods::run::DESCRIPTOR).unwrap();
    assert!(wgsl.contains("compute"), "{wgsl}");
}

#[test]
fn the_keep_method_module_compiles_to_spirv() {
    let words = gust::slang::compile_spirv(&keep_methods::run::DESCRIPTOR).unwrap();
    gust::spirv::validate_structure(&words).unwrap();
}

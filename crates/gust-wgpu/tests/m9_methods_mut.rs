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
//   helper   = input + 6            (the helper `drive` makes a counter with step 3 and calls `bump_twice`): 6, 8, 10, 12, 14
const BUMPED: [u32; 5] = [2, 4, 6, 8, 10];
const CONSUMED: [u32; 5] = [6, 8, 10, 12, 14];
const SCALED: [u32; 5] = [18, 24, 30, 36, 42];
const MEAN: [u32; 5] = [2, 3, 3, 3, 4];
const LOOPED: u32 = 4;
const HELPER: [u32; 5] = [6, 8, 10, 12, 14];

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
    // `[mutating]` takes the receiver by pointer; a method on `self` by value takes it as a value.
    assert!(
        wgsl.contains("fn Counter_bump_0( this_0 : ptr<function, Counter_0>)"),
        "{wgsl}"
    );
    assert!(
        wgsl.contains("fn Counter_value_0( this_3 : Counter_0)"),
        "{wgsl}"
    );
}

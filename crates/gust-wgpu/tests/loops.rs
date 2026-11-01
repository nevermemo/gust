include!("../../../tests/fixtures/loops.rs");

use gust::cpu::CpuDevice;
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

/// Independent host reference for the `loops` fixture. Closed forms are used where
/// they exist so the reference does not mirror the kernel's loop structure.
struct Expected {
    count: u32,
    sum: u32,
    nested: u32,
}

fn reference(x: u32, window: (u32, u32), input: &[u32]) -> Expected {
    // `for _ in 0..limit` with `limit` shrinking inside the body: Rust fixes the
    // trip count when the range is built, so it equals the initial limit.
    let count = x % 7 + 1;

    let (lo, hi) = window;
    let window_sum: u32 = (lo..hi.min(input.len() as u32))
        .map(|j| input[j as usize])
        .sum();
    let n = x % 10;
    let sum_below = n * n.saturating_sub(1) / 2;
    let minimum = x % 20;
    let first_multiple = 3 * minimum.div_ceil(3).max(1);
    let sum = window_sum + sum_below + first_multiple;

    // grid is input-independent: pairs (a, b) with a <= b < 4 and a*b <= 6 taken in
    // order until the first product above 6 in each row: 4 + 9 + 12 = 25.
    let grid = 25;
    let hi = (x as i32) % 5;
    let neg_sum: i32 = (-3..hi).filter(|k| *k != 0).sum();
    let nested = grid * 100 + (neg_sum + 10) as u32;

    Expected { count, sum, nested }
}

#[test]
fn bounded_loops_execute_on_gpu() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("loops adapter: {:?}", device.adapter_info());
    for count in [1, 63, 64, 65, 257] {
        let input: Vec<u32> = (0..count).map(|i| ((i * 37 + 11) % 200) as u32).collect();
        // The last window always runs past the buffer so the `else { break; }` path
        // is taken at every count; small counts add more overruns.
        let windows: Vec<loops::Window> = (0..count)
            .map(|i| {
                let lo = (i % 5) as u32;
                let hi = if i + 1 == count {
                    count as u32 + 3
                } else {
                    lo + input[i] % 9
                };
                loops::Window { lo, hi }
            })
            .collect();
        let input_buf =
            device.create_typed_buffer("loops input", &input, GpuBufferAccess::ReadOnly);
        let windows_buf =
            device.create_typed_buffer("loops windows", &windows, GpuBufferAccess::ReadOnly);
        let sums = device.create_typed_buffer(
            "loops sums",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        let counts = device.create_typed_buffer(
            "loops counts",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        let nested = device.create_typed_buffer(
            "loops nested",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );

        device
            .dispatch_buffers(
                &loops::run::DESCRIPTOR,
                count as u32,
                &[
                    BufferBinding::read_only(&input_buf),
                    BufferBinding::read_only(&windows_buf),
                    BufferBinding::read_write(&sums),
                    BufferBinding::read_write(&counts),
                    BufferBinding::read_write(&nested),
                ],
            )
            .unwrap();

        let actual_sums = device.read_typed_buffer(&sums).unwrap();
        let actual_counts = device.read_typed_buffer(&counts).unwrap();
        let actual_nested = device.read_typed_buffer(&nested).unwrap();

        // The CPU twin: the same typed call with the device swapped (macro-emitted under `cfg(test)`).
        let cpu = CpuDevice::new();
        let twin_input = cpu.buffer(&input);
        let twin_windows = cpu.buffer(&windows);
        let twin_sums = cpu.buffer(&vec![0u32; count]);
        let twin_counts = cpu.buffer(&vec![0u32; count]);
        let twin_nested = cpu.buffer(&vec![0u32; count]);
        loops::run::dispatch(
            &cpu,
            count as u32,
            &twin_input,
            &twin_windows,
            &twin_sums,
            &twin_counts,
            &twin_nested,
        )
        .unwrap();
        let twin_sums = twin_sums.read();
        let twin_counts = twin_counts.read();
        let twin_nested = twin_nested.read();
        assert_eq!(twin_sums, actual_sums, "twin sums, count {count}");
        assert_eq!(twin_counts, actual_counts, "twin counts, count {count}");
        assert_eq!(twin_nested, actual_nested, "twin nested, count {count}");

        let mut break_paths = 0;
        for i in 0..count {
            let expected = reference(input[i], (windows[i].lo, windows[i].hi), &input);
            if windows[i].hi > count as u32 {
                break_paths += 1;
            }
            assert_eq!(
                actual_counts[i], expected.count,
                "counts[{i}] for input {}",
                input[i]
            );
            assert_eq!(
                actual_sums[i], expected.sum,
                "sums[{i}] for input {}",
                input[i]
            );
            assert_eq!(
                actual_nested[i], expected.nested,
                "nested[{i}] for input {}",
                input[i]
            );
            assert_eq!(
                twin_counts[i], expected.count,
                "twin counts[{i}] for input {}",
                input[i]
            );
            assert_eq!(
                twin_sums[i], expected.sum,
                "twin sums[{i}] for input {}",
                input[i]
            );
            assert_eq!(
                twin_nested[i], expected.nested,
                "twin nested[{i}] for input {}",
                input[i]
            );
        }
        assert!(
            break_paths > 0,
            "every count must exercise the window break path"
        );
    }
}

#[test]
fn loops_compile_to_wgsl() {
    let descriptor = &loops::run::DESCRIPTOR;
    assert!(
        gust::slang::compile_wgsl(descriptor)
            .unwrap()
            .contains("@compute")
    );
}

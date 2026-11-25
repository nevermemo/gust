//! The library ops written in a build block (m14.s2f2c, D262): `gust::sum`, `sum_wrapping`, `min`, `max`, `reduce`,
//! `count`, `scan`, `scan_exclusive` and `compact`, each the whole init of a `let`, on the CPU device and on the GPU.
//!
//! The macro form is held to the builder API (`b.op(&[..], gust_algo::batch::Sum::of(&values))`, which the macro lowers
//! to) and to host references, over three frames that write different inputs. Two stages read the results: one reads a
//! sum at index zero (`total[0]`, the capture of the plan's 2.2 example) and one gathers through the indices of a
//! compaction. The inputs are small integers, and floats that are multiples of 2^-6 below 1, so every sum, prefix and
//! shifted value is exact in `u32` and `f32` whatever the order of the additions. The GPU tests are `#[ignore]`d like
//! the example's other GPU tests: they run in full verification, on the adapter the environment selects, and print its
//! name.
//!
//! Past its count, the `indices` of a compaction are unspecified (D259), so the indices, and the gather through them,
//! are compared up to the count.

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, Error, Stage, StageInfo};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_algo::batch::{Compact, Count, Max, Min, Reduce, Scan, Sum, SumWrapping};
use gust_algo::reduce::Op;
use gust_wgpu::HeadlessDevice;

/// A length that is not a power of two, and one that takes two passes of every reduction.
const SIZES: [u32; 2] = [1000, 100_003];

/// The handles of the batch: the inputs and every output.
struct Handles {
    values: Stage<u32, D1>,
    floats: Stage<f32, D1>,
    flags: Stage<u32, D1>,
    total: Stage<u32, D1>,
    wrapping: Stage<u32, D1>,
    least: Stage<u32, D1>,
    most: Stage<u32, D1>,
    reduced: Stage<u32, D1>,
    ftotal: Stage<f32, D1>,
    count: Stage<u32, D1>,
    inclusive: Stage<u32, D1>,
    exclusive: Stage<u32, D1>,
    idx: Stage<u32, D1>,
    n_set: Stage<u32, D1>,
    offset: Stage<u32, D1>,
    gathered: Stage<u32, D1>,
    shifted: Stage<f32, D1>,
}

/// The batch written with the macro forms.
macro_rules! define_macro_build {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>, n: u32) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("macro-ops", on(device); in_flight(1); |b| {
                let values = b.input::<u32>(n);
                let floats = b.input::<f32>(n);
                let flags = b.input::<u32>(n);
                let total = gust::sum(&values);
                let wrapping = gust::sum_wrapping(&values);
                let least = gust::min(&values);
                let most = gust::max(&values);
                let reduced = gust::reduce(&values, Op::Max);
                let ftotal = gust::sum(&floats);
                let count = gust::count(&flags);
                let inclusive = gust::scan(&values);
                let exclusive = gust::scan_exclusive(&values);
                let (idx, n_set) = gust::compact(&flags);
                let offset = gpu_task!(n, |i| -> u32 { values[i] + total[0] });
                let gathered = gpu_task!(n, |i| -> u32 { values[idx[i]] });
                let shifted = gpu_task!(n, |i| -> f32 { floats[i] + ftotal[0] });
                Handles {
                    values, floats, flags, total, wrapping, least, most, reduced, ftotal, count,
                    inclusive, exclusive, idx, n_set, offset, gathered, shifted,
                }
            })
        }
    };
}

/// The same batch written with the builder API the macro lowers to.
macro_rules! define_builder_build {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>, n: u32) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("builder-ops", on(device); in_flight(1); |b| {
                let values = b.input::<u32>(n);
                let floats = b.input::<f32>(n);
                let flags = b.input::<u32>(n);
                let total = b.op(&["total"], Sum::of(&values));
                let wrapping = b.op(&["wrapping"], SumWrapping::of(&values));
                let least = b.op(&["least"], Min::of(&values));
                let most = b.op(&["most"], Max::of(&values));
                let reduced = b.op(&["reduced"], Reduce::of(Op::Max, &values));
                let ftotal = b.op(&["ftotal"], Sum::of(&floats));
                let count = b.op(&["count"], Count::of(&flags));
                let inclusive = b.op(&["inclusive"], Scan::of(&values));
                let exclusive = b.op(&["exclusive"], Scan::exclusive(&values));
                let (idx, n_set) = b.op(&["idx", "n_set"], Compact::of(&flags));
                let offset = gpu_task!(n, [total: Stage<u32>], |i| -> u32 { values[i] + total[0] });
                let gathered = gpu_task!(n, [idx: Stage<u32>], |i| -> u32 { values[idx[i]] });
                let shifted = gpu_task!(n, [ftotal: Stage<f32>], |i| -> f32 { floats[i] + ftotal[0] });
                for handle in [&total, &wrapping, &least, &most, &reduced, &count, &inclusive, &exclusive, &idx, &n_set] {
                    b.expose(handle);
                }
                b.expose(&ftotal);
                Handles {
                    values, floats, flags, total, wrapping, least, most, reduced, ftotal, count,
                    inclusive, exclusive, idx, n_set, offset, gathered, shifted,
                }
            })
        }
    };
}

define_macro_build!(macro_cpu, CpuDevice);
define_macro_build!(macro_gpu, HeadlessDevice);
define_builder_build!(builder_cpu, CpuDevice);
define_builder_build!(builder_gpu, HeadlessDevice);

/// What a frame writes: an input that is `None` keeps the value of the frame before.
#[derive(Clone)]
struct Frame {
    values: Option<Vec<u32>>,
    floats: Option<Vec<f32>>,
    flags: Option<Vec<u32>>,
}

/// The inputs a frame leaves behind.
#[derive(Clone, Default)]
struct Inputs {
    values: Vec<u32>,
    floats: Vec<f32>,
    flags: Vec<u32>,
}

fn values(n: u32, seed: u32) -> Vec<u32> {
    (0..n).map(|i| (i * 37 + seed * 101) % 4093).collect()
}

fn floats(n: u32, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 53 + seed * 17) % 64) as f32 / 64.0)
        .collect()
}

fn flags(n: u32, seed: u32) -> Vec<u32> {
    (0..n)
        .map(|i| {
            if (i * 7 + seed).is_multiple_of(3) {
                1 + i % 5
            } else {
                0
            }
        })
        .collect()
}

/// Three frames: every input, then the values and the flags (the floats hold), then every input.
fn frames(n: u32) -> [Frame; 3] {
    [
        Frame {
            values: Some(values(n, 1)),
            floats: Some(floats(n, 1)),
            flags: Some(flags(n, 1)),
        },
        Frame {
            values: Some(values(n, 2)),
            floats: None,
            flags: Some(flags(n, 2)),
        },
        Frame {
            values: Some(values(n, 3)),
            floats: Some(floats(n, 3)),
            flags: Some(flags(n, 3)),
        },
    ]
}

/// What one frame gave, or what the host says it must give. `idx` and `gathered` hold the first `n_set` entries.
#[derive(Debug, PartialEq)]
struct Out {
    total: u32,
    wrapping: u32,
    least: u32,
    most: u32,
    reduced: u32,
    ftotal: f32,
    count: u32,
    inclusive: Vec<u32>,
    exclusive: Vec<u32>,
    idx: Vec<u32>,
    n_set: u32,
    offset: Vec<u32>,
    gathered: Vec<u32>,
    shifted: Vec<f32>,
}

/// The host's answer for `inputs`: independent of the ops, and exact (see the module comment).
fn reference(inputs: &Inputs) -> Out {
    let total: u32 = inputs.values.iter().sum();
    let set: Vec<u32> = (0..inputs.flags.len() as u32)
        .filter(|&i| inputs.flags[i as usize] != 0)
        .collect();
    let mut running = 0;
    let (inclusive, exclusive): (Vec<u32>, Vec<u32>) = inputs
        .values
        .iter()
        .map(|&value| {
            let before = running;
            running += value;
            (running, before)
        })
        .unzip();
    let ftotal: f32 = inputs.floats.iter().sum();
    Out {
        total,
        wrapping: total,
        least: *inputs.values.iter().min().unwrap(),
        most: *inputs.values.iter().max().unwrap(),
        reduced: *inputs.values.iter().max().unwrap(),
        ftotal,
        count: set.len() as u32,
        inclusive,
        exclusive,
        n_set: set.len() as u32,
        offset: inputs.values.iter().map(|value| value + total).collect(),
        gathered: set.iter().map(|&i| inputs.values[i as usize]).collect(),
        shifted: inputs.floats.iter().map(|value| value + ftotal).collect(),
        idx: set,
    }
}

fn one<T: Copy>(values: Result<Vec<T>, Error>) -> T {
    let values = values.expect("the ticket resolves");
    assert_eq!(values.len(), 1);
    values[0]
}

/// Run one frame: write what it writes, reserve a ticket of every output, submit, read.
fn run<D: BatchDevice>(batch: &mut Batch<D>, h: &Handles, frame: &Frame) -> Out
where
    D::Buffer<u32>: 'static,
    D::Buffer<f32>: 'static,
{
    let mut run = block_on(batch.begin()).unwrap();
    if let Some(values) = &frame.values {
        run.write(&h.values, values).unwrap();
    }
    if let Some(floats) = &frame.floats {
        run.write(&h.floats, floats).unwrap();
    }
    if let Some(flags) = &frame.flags {
        run.write(&h.flags, flags).unwrap();
    }
    let total = run.ticket(&h.total).unwrap();
    let wrapping = run.ticket(&h.wrapping).unwrap();
    let least = run.ticket(&h.least).unwrap();
    let most = run.ticket(&h.most).unwrap();
    let reduced = run.ticket(&h.reduced).unwrap();
    let ftotal = run.ticket(&h.ftotal).unwrap();
    let count = run.ticket(&h.count).unwrap();
    let inclusive = run.ticket(&h.inclusive).unwrap();
    let exclusive = run.ticket(&h.exclusive).unwrap();
    let idx = run.ticket(&h.idx).unwrap();
    let n_set = run.ticket(&h.n_set).unwrap();
    let offset = run.ticket(&h.offset).unwrap();
    let gathered = run.ticket(&h.gathered).unwrap();
    let shifted = run.ticket(&h.shifted).unwrap();
    let _submitted = run.submit().unwrap();
    let n_set = one(block_on(n_set));
    let mut idx = block_on(idx).unwrap();
    let mut gathered = block_on(gathered).unwrap();
    idx.truncate(n_set as usize);
    gathered.truncate(n_set as usize);
    Out {
        total: one(block_on(total)),
        wrapping: one(block_on(wrapping)),
        least: one(block_on(least)),
        most: one(block_on(most)),
        reduced: one(block_on(reduced)),
        ftotal: one(block_on(ftotal)),
        count: one(block_on(count)),
        inclusive: block_on(inclusive).unwrap(),
        exclusive: block_on(exclusive).unwrap(),
        idx,
        n_set,
        offset: block_on(offset).unwrap(),
        gathered,
        shifted: block_on(shifted).unwrap(),
    }
}

/// The stage with this name, which every op names by its `let` (the first output of a compaction, `idx`).
fn stage<'a>(stages: &'a [StageInfo], name: &str) -> &'a StageInfo {
    stages
        .iter()
        .find(|stage| stage.name == name)
        .unwrap_or_else(|| panic!("no stage `{name}` in {stages:?}"))
}

/// The macro batch and the builder batch, frame by frame: equal to each other and to the host.
fn check<D: BatchDevice>(
    mut macro_batch: (Batch<D>, Handles),
    mut builder_batch: (Batch<D>, Handles),
    n: u32,
) where
    D::Buffer<u32>: 'static,
    D::Buffer<f32>: 'static,
{
    // The names the `let`s gave, and the length each result has.
    let stages = macro_batch.0.stages();
    for (name, length) in [
        ("total", 1),
        ("ftotal", 1),
        ("count", 1),
        ("inclusive", n),
        ("exclusive", n),
        ("idx", n),
    ] {
        assert_eq!(stage(&stages, name).dims, [length, 1], "{name}");
    }
    assert_eq!(stage(&stages, "idx").reads, ["flags"]);
    assert_eq!(stage(&stages, "total").reads, ["values"]);
    let mut inputs = Inputs::default();
    for (round, frame) in frames(n).iter().enumerate() {
        if let Some(values) = &frame.values {
            inputs.values.clone_from(values);
        }
        if let Some(floats) = &frame.floats {
            inputs.floats.clone_from(floats);
        }
        if let Some(flags) = &frame.flags {
            inputs.flags.clone_from(flags);
        }
        let expected = reference(&inputs);
        let from_macro = run(&mut macro_batch.0, &macro_batch.1, frame);
        let from_builder = run(&mut builder_batch.0, &builder_batch.1, frame);
        assert_eq!(
            from_macro, from_builder,
            "frame {round}, n = {n}: macro and builder API differ"
        );
        assert_eq!(
            from_macro, expected,
            "frame {round}, n = {n}: macro and host differ"
        );
        // The capture of the plan's 2.2 example: `total[0]` read by a stage, exact.
        assert_eq!(
            from_macro.offset[0],
            inputs.values[0] + expected.total,
            "total[0]"
        );
    }
}

#[test]
fn twin_macro_ops_equal_the_builder_api_and_the_host() {
    for n in SIZES {
        let device = Arc::new(CpuDevice::new());
        check(
            macro_cpu(&device, n).unwrap(),
            builder_cpu(&device, n).unwrap(),
            n,
        );
    }
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_macro_ops_equal_the_builder_api_and_the_host() {
    let device = Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }));
    println!("adapter: {}", device.adapter_info().name);
    for n in SIZES {
        check(
            macro_gpu(&device, n).unwrap(),
            builder_gpu(&device, n).unwrap(),
            n,
        );
    }
}

// The element-wise map calls of `#[gust::gpu_fn]` on a function (M7, D60) in the production CPU target: `map` and
// `map_on` run on `gust::cpu::CpuDevice` with no GPU API, in parallel chunks of the output (slice m7.c,
// `GUST_CPU_THREADS`), and the CPU-only chunk body `__gust_map_cpu_chunk` runs one range of elements. Built under
// `GUST_PROFILE=cpu` only (see `build.rs`).

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use gust::{
    cpu::{CpuDevice, parallel::ELEMENTS_PER_THREAD},
    map::{Broadcast, Error},
    math::{Mat4, Vec4},
};

#[gust::gpu_fn]
pub fn scale_add(a: f32, x: f32, y: f32) -> f32 {
    a * x + y
}

#[gust::gpu_fn]
pub fn pick(flag: bool, x: u32, y: u32) -> u32 {
    if flag { x } else { y }
}

#[gust::gpu_fn]
pub fn both(a: u32, b: u32, c: u32) -> u32 {
    a % b + a / c
}

#[gust::gpu_fn]
pub fn is_multiple(a: u32, b: u32) -> bool {
    a % b == 0
}

/// More than one workgroup (256 wide) and not a multiple of it.
const ELEMENTS: usize = 1000;

/// Long enough for several threads at the default bound of the parallel map (`ELEMENTS_PER_THREAD` elements per thread)
/// and not divisible by the thread count: four threads' worth of elements and a remainder.
const MANY: usize = 4 * ELEMENTS_PER_THREAD + 7;

/// Runs the map calls and compares them with host references. Small integers, so every float result is exact.
pub fn run() -> Result<(), String> {
    let x: Vec<f32> = (0..ELEMENTS).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..ELEMENTS).map(|i| (3 * i % 17) as f32 - 8.0).collect();
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| 2.0 * x + y).collect();
    let flags: Vec<bool> = (0..ELEMENTS).map(|i| i % 3 == 0).collect();
    let left: Vec<u32> = (0..ELEMENTS as u32).collect();

    let mapped = scale_add::map(Broadcast(2.0), &x, &y).map_err(|error| format!("scale_add::map: {error}"))?;
    if mapped != expected {
        return Err("scale_add::map differs from the host reference".to_owned());
    }
    let device = CpuDevice::new();
    let on = scale_add::map_on(&device, &x, Broadcast(1.0), &y).map_err(|error| format!("scale_add::map_on: {error}"))?;
    if on != x.iter().zip(&y).map(|(a, y)| a + y).collect::<Vec<_>>() {
        return Err("scale_add::map_on differs from the host reference".to_owned());
    }
    let picked = pick::map(&flags, &left, Broadcast(7)).map_err(|error| format!("pick::map: {error}"))?;
    let picked_expected: Vec<u32> = flags.iter().zip(&left).map(|(flag, x)| if *flag { *x } else { 7 }).collect();
    if picked != picked_expected {
        return Err("pick::map differs from the host reference".to_owned());
    }
    // The chunk body: elements 300 to 700, the broadcast scalar as a slice of one.
    let mut chunk = vec![0.0f32; 400];
    let current = Cell::new(0);
    scale_add::__gust_map_cpu_chunk(300, 700, &[2.0], &x, &y, &mut chunk, &current);
    if chunk != expected[300..700] {
        return Err("scale_add::__gust_map_cpu_chunk differs from the host reference".to_owned());
    }
    if current.get() != 699 {
        return Err(format!("the chunk body left element {} as the current one, not 699", current.get()));
    }
    // A mismatched length is an error before any element is computed.
    match scale_add::map(Broadcast(2.0), &x, &y[..10]) {
        Err(Error::LengthMismatch {
            parameter: "y",
            expected: ELEMENTS,
            found: 10,
        }) => {}
        other => return Err(format!("scale_add::map with a short argument returned {other:?}")),
    }
    parallel()?;
    structs()?;
    wide_structs()?;
    println!("CPU target: map calls match (scale_add, pick, {ELEMENTS} elements)");
    Ok(())
}

/// The parallel path (a long input, several chunks on a machine with the threads), the `bool` result, the lowest
/// failing index and a clean call after a panic, each against a host reference.
fn parallel() -> Result<(), String> {
    let x: Vec<f32> = (0..MANY).map(|i| (i % 1021) as f32).collect();
    let y: Vec<f32> = (0..MANY).map(|i| (5 * i % 29) as f32 - 14.0).collect();
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| 3.0 * x + y).collect();
    let mapped = scale_add::map(Broadcast(3.0), &x, &y).map_err(|error| format!("parallel scale_add::map: {error}"))?;
    if mapped != expected {
        return Err("the parallel scale_add::map differs from the host reference".to_owned());
    }
    let numbers: Vec<u32> = (0..MANY as u32).map(|i| i.wrapping_mul(2_654_435_761) >> 8).collect();
    let divisors: Vec<u32> = (0..MANY as u32).map(|i| i % 7 + 1).collect();
    let multiples = is_multiple::map(&numbers, &divisors)
        .map_err(|error| format!("parallel is_multiple::map: {error}"))?;
    if !multiples.iter().zip(numbers.iter().zip(&divisors)).all(|(got, (a, b))| *got == (a % b == 0)) {
        return Err("the parallel is_multiple::map differs from the host reference".to_owned());
    }
    panics_report_the_lowest_index(&numbers, &divisors)?;
    Ok(())
}

/// Two different panics in one call, `a % b` with `b == 0` ("remainder") and `a / c` with `c == 0` ("divide"), placed
/// so that they land in different chunks when there are several threads: the lower index must be the one that comes
/// out, whichever of the two it is, then nothing is returned and the next call is clean. The first panic is in the
/// second chunk and the other in the last (the same chunk when the machine has fewer than three threads, where the
/// order within a chunk is the check). The hook is silenced so that the expected panics do not clutter the output.
fn panics_report_the_lowest_index(numbers: &[u32], divisors: &[u32]) -> Result<(), String> {
    let threads = gust::cpu::parallel::map_threads(MANY as u32).map_err(|error| error.to_string())?;
    let size = MANY.div_ceil(threads);
    let low = (size + 5).min(MANY - 10);
    let high = MANY - 3;
    let clean: Vec<u32> = (0..MANY as u32).map(|i| i % 5 + 1).collect();

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut texts = Vec::new();
    for (remainder_at, divide_at, expected) in [(low, high, "remainder"), (high, low, "divide by zero")] {
        let mut remainders = divisors.to_vec();
        remainders[remainder_at] = 0;
        let mut divides = clean.clone();
        divides[divide_at] = 0;
        let panicked = catch_unwind(AssertUnwindSafe(|| both::map(numbers, &remainders, &divides)));
        texts.push((expected, panicked.map_err(|payload| panic_text(&*payload))));
    }
    std::panic::set_hook(hook);

    for (expected, outcome) in texts {
        match outcome {
            Err(text) if text.contains(expected) => {}
            Err(text) => {
                return Err(format!("both::map reported {text:?} where the lowest index panics with {expected:?}"));
            }
            Ok(_) => return Err("both::map with zero divisors returned a value".to_owned()),
        }
    }
    let results = both::map(numbers, divisors, &clean).map_err(|error| format!("both::map after a panic: {error}"))?;
    let all_match = results
        .iter()
        .zip(numbers.iter().zip(divisors.iter().zip(&clean)))
        .all(|(got, (a, (b, c)))| *got == a % b + a / c);
    if !all_match {
        return Err("both::map after a panic differs from the host reference".to_owned());
    }
    Ok(())
}

/// The message of a panic payload (`&str` for a constant message, `String` for a formatted one).
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

// m14.mapw1: a map over a struct of the module that declares it. Under `GUST_PROFILE=cpu` the calls are `map`, `map_on`
// and the chunk body, and no `::gust_wgpu` is named.
#[gust::gpu_module]
pub mod records {
    pub struct Item {
        pub weight: f32,
        pub id: u32,
    }

    #[gust::gpu_fn]
    pub fn bump(item: Item, k: f32) -> Item {
        let mut bumped = item;
        bumped.weight = item.weight * k;
        bumped.id = item.id + 1u32;
        bumped
    }

    #[gust::gpu_fn]
    pub fn total(item: Item) -> f32 {
        item.weight + item.id as f32
    }
}

/// The struct map in the production CPU target: `map`, `map_on`, the chunk body and a parallel run (a long input),
/// each against a host reference. Small integers, so every float result is exact.
fn structs() -> Result<(), String> {
    use records::Item;
    let make = |count: usize| -> Vec<Item> {
        (0..count)
            .map(|i| Item {
                weight: (i % 11) as f32 - 5.0,
                id: i as u32 % 97,
            })
            .collect()
    };
    let reference = |item: &Item| (item.weight * 2.0, item.id + 1);
    for count in [ELEMENTS, MANY] {
        let items = make(count);
        let expected: Vec<(f32, u32)> = items.iter().map(reference).collect();
        let bumped = records::bump::map(&items, Broadcast(2.0)).map_err(|error| format!("records::bump::map: {error}"))?;
        if bumped.iter().map(|item| (item.weight, item.id)).collect::<Vec<_>>() != expected {
            return Err(format!("records::bump::map differs from the host reference ({count} elements)"));
        }
        let on = records::bump::map_on(&CpuDevice::new(), &items, Broadcast(2.0))
            .map_err(|error| format!("records::bump::map_on: {error}"))?;
        if on.iter().map(|item| (item.weight, item.id)).collect::<Vec<_>>() != expected {
            return Err(format!("records::bump::map_on differs from the host reference ({count} elements)"));
        }
        let totals = records::total::map(&items).map_err(|error| format!("records::total::map: {error}"))?;
        if !totals.iter().zip(&items).all(|(got, item)| *got == item.weight + item.id as f32) {
            return Err(format!("records::total::map differs from the host reference ({count} elements)"));
        }
    }
    // The chunk body: elements 300 to 700, the broadcast scalar as a slice of one.
    let items = make(ELEMENTS);
    let mut chunk = make(400);
    let current = Cell::new(0);
    records::bump::__gust_map_cpu_chunk(300, 700, &items, &[2.0], &mut chunk, &current);
    if chunk.iter().map(|item| (item.weight, item.id)).collect::<Vec<_>>()
        != items[300..700].iter().map(reference).collect::<Vec<_>>()
    {
        return Err("records::bump::__gust_map_cpu_chunk differs from the host reference".to_owned());
    }
    Ok(())
}

// m14.mapw2: a struct with a `Vec4` and a `Mat4` field, and one that holds it, through the same calls.
#[gust::gpu_module]
pub mod wide {
    pub struct Frame {
        pub id: u32,
        pub fill_a: [u32; 3],
        pub dir: Vec4,
        pub tag: u32,
        pub fill_b: [u32; 3],
        pub basis: Mat4,
    }

    pub struct Slot {
        pub flag: u32,
        pub fill_a: [u32; 3],
        pub frame: Frame,
        pub weight: f32,
        pub fill_b: [u32; 3],
    }

    #[gust::gpu_fn]
    pub fn slot_step(slot: Slot, k: f32) -> Slot {
        let mut stepped = slot;
        stepped.flag = slot.flag + 1u32;
        stepped.fill_a = slot.fill_b;
        stepped.frame.fill_a = slot.frame.fill_b;
        stepped.frame.fill_b = slot.frame.fill_a;
        stepped.fill_b = slot.fill_a;
        stepped.frame.id = slot.frame.tag;
        stepped.frame.dir = slot.frame.basis.mul_vec4(slot.frame.dir) * k;
        stepped.frame.tag = slot.frame.id;
        stepped.frame.basis = slot.frame.basis.transpose();
        stepped.weight = slot.weight * k;
        stepped
    }
}

/// Every field of a slot, the matrix as the product with each unit vector (it has no public fields).
type SlotFields = (u32, u32, [f32; 4], u32, [[f32; 4]; 4], f32, [[u32; 3]; 4]);

fn slot_fields(slot: &wide::Slot) -> SlotFields {
    let dir = slot.frame.dir;
    let columns = [
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    ]
    .map(|unit| {
        let column = slot.frame.basis.mul_vec4(unit);
        [column.x, column.y, column.z, column.w]
    });
    (
        slot.flag,
        slot.frame.id,
        [dir.x, dir.y, dir.z, dir.w],
        slot.frame.tag,
        columns,
        slot.weight,
        [slot.fill_a, slot.fill_b, slot.frame.fill_a, slot.frame.fill_b],
    )
}

/// `wide::slot_step` through `map`, `map_on` and the chunk body, against a host reference.
fn wide_structs() -> Result<(), String> {
    use wide::{Frame, Slot};
    let value = |i: usize, salt: usize| ((i * 7 + salt * 3) % 11) as f32 - 5.0;
    let vec4 = |i: usize, salt: usize| Vec4::new(value(i, salt), value(i, salt + 1), value(i, salt + 2), value(i, salt + 3));
    let make = |count: usize| -> Vec<Slot> {
        (0..count)
            .map(|i| Slot {
                flag: i as u32 % 50,
                fill_a: [i as u32, 1, 2],
                frame: Frame {
                    id: i as u32 % 97,
                    fill_a: [3, i as u32, 4],
                    dir: vec4(i, 0),
                    tag: i as u32 % 31 + 5,
                    fill_b: [5, 6, i as u32],
                    basis: Mat4::from_cols(vec4(i, 10), vec4(i, 20), vec4(i, 30), vec4(i, 40)),
                },
                weight: value(i, 50),
                fill_b: [i as u32 + 7, 8, 9],
            })
            .collect()
    };
    let reference = |slot: &Slot| {
        slot_fields(&Slot {
            flag: slot.flag + 1,
            fill_a: slot.fill_b,
            frame: Frame {
                id: slot.frame.tag,
                fill_a: slot.frame.fill_b,
                dir: slot.frame.basis.mul_vec4(slot.frame.dir) * 2.0,
                tag: slot.frame.id,
                fill_b: slot.frame.fill_a,
                basis: slot.frame.basis.transpose(),
            },
            weight: slot.weight * 2.0,
            fill_b: slot.fill_a,
        })
    };
    for count in [ELEMENTS, MANY] {
        let slots = make(count);
        let expected: Vec<SlotFields> = slots.iter().map(reference).collect();
        let stepped = wide::slot_step::map(&slots, Broadcast(2.0)).map_err(|error| format!("wide::slot_step::map: {error}"))?;
        if stepped.iter().map(slot_fields).collect::<Vec<_>>() != expected {
            return Err(format!("wide::slot_step::map differs from the host reference ({count} elements)"));
        }
        let on = wide::slot_step::map_on(&CpuDevice::new(), &slots, Broadcast(2.0))
            .map_err(|error| format!("wide::slot_step::map_on: {error}"))?;
        if on.iter().map(slot_fields).collect::<Vec<_>>() != expected {
            return Err(format!("wide::slot_step::map_on differs from the host reference ({count} elements)"));
        }
    }
    // The chunk body: elements 300 to 700, the broadcast scalar as a slice of one.
    let slots = make(ELEMENTS);
    let mut chunk = make(400);
    let current = Cell::new(0);
    wide::slot_step::__gust_map_cpu_chunk(300, 700, &slots, &[2.0], &mut chunk, &current);
    if chunk.iter().map(slot_fields).collect::<Vec<_>>() != slots[300..700].iter().map(reference).collect::<Vec<_>>() {
        return Err("wide::slot_step::__gust_map_cpu_chunk differs from the host reference".to_owned());
    }
    Ok(())
}

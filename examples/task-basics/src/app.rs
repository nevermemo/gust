//! `gpu_task!` as a free task: four small stages written at the call site, run with `submit` and with `dispatch`.
//!
//! - **scale** (1-D): `a[i] * k + b`, a host slice and two scalars; the result is a fresh `Vec<f32>`.
//! - **grid** (2-D): `bias[x] + y * 100`, a `(WIDTH, HEIGHT)` launch whose result is row-major, `y * WIDTH + x`.
//! - **halve** (in place): `data[i] = data[i] * 0.5 + 1.0` on a `&mut [f32]`; the slice holds the answer once the
//!   task is awaited.
//! - **weigh** (1-D): `weights[i % 4] + a[i]`, a second host-slice capture read at a computed index.
//!
//! Every value is exactly representable in `f32` and every operation on it is exact, so the GPU result equals the host
//! reference bit for bit, whatever the rounding mode of the device.
//!
//! The stages run through [`How`]: `Dispatch` blocks until the device is done (`Task::dispatch`), `Submit` starts the
//! work with `Task::submit` and polls the returned future with a no-op waker, counting the polls (a poll polls the
//! device once and never waits). The host reference is written plainly below, not from the closures. The same
//! macro-generated functions run on the CPU twin (`CpuDevice`, test builds) and on the GPU (`HeadlessDevice`).

#[cfg(any(test, gust_dual))]
use gust::cpu::CpuDevice;
use gust::gpu_task;
use gust::map::Error;
use gust::task::Pending;
use gust_wgpu::{DeviceHandle, HeadlessDevice, default_device};
use std::task::{Context, Poll, Waker};

/// Elements of the 1-D stages.
pub const N: u32 = 1000;
/// Columns and rows of the 2-D stage.
pub const WIDTH: u32 = 37;
pub const HEIGHT: u32 = 19;
/// Elements of the in-place stage.
pub const LEN: usize = 777;

/// How a run waits for a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum How {
    /// `Task::dispatch`: start the work and block until the read-back is done.
    Dispatch,
    /// `Task::submit`, then poll the future with a no-op waker until it is ready.
    Submit,
}

/// The inputs of one run.
#[derive(Clone, Debug)]
pub struct Inputs {
    pub a: Vec<f32>,
    pub bias: Vec<u32>,
    pub weights: Vec<f32>,
    pub data: Vec<f32>,
}

/// The results of one run.
#[derive(Clone, Debug, PartialEq)]
pub struct Outputs {
    pub scaled: Vec<f32>,
    pub grid: Vec<u32>,
    pub halved: Vec<f32>,
    pub weighed: Vec<f32>,
}

/// The inputs: small integers and quarters, so every product and sum below is exact in `f32`.
pub fn inputs() -> Inputs {
    Inputs {
        a: (0..N).map(|i| i as f32).collect(),
        bias: (0..WIDTH).map(|x| x * 3).collect(),
        weights: vec![0.25, 0.5, 1.0, 2.0],
        data: (0..LEN).map(|i| (i % 50) as f32).collect(),
    }
}

/// The host reference, written from the rules of the stages.
pub fn reference(inputs: &Inputs) -> Outputs {
    let scaled = inputs.a.iter().map(|a| a * 3.0 + 0.25).collect();
    let mut grid = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            grid.push(inputs.bias[x as usize] + y * 100);
        }
    }
    let halved = inputs.data.iter().map(|d| d * 0.5 + 1.0).collect();
    let weighed = inputs
        .a
        .iter()
        .enumerate()
        .map(|(i, a)| inputs.weights[i % 4] + a)
        .collect();
    Outputs {
        scaled,
        grid,
        halved,
        weighed,
    }
}

/// Polls a pending task with a no-op waker until it is ready; returns the value and the number of polls.
pub fn poll_to_end<O>(mut pending: Pending<'_, O>) -> (Result<O, Error>, usize) {
    let mut context = Context::from_waker(Waker::noop());
    let mut polls = 0;
    loop {
        polls += 1;
        match pending.as_mut().poll(&mut context) {
            Poll::Ready(value) => return (value, polls),
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// Waits for a task in the given way.
macro_rules! finish {
    ($how:expr, $task:expr) => {
        match $how {
            How::Dispatch => $task.dispatch()?,
            How::Submit => poll_to_end($task.submit()?).0?,
        }
    };
}

/// Defines a function that runs the four stages on a device of the given type, in the given way.
macro_rules! define_run {
    ($name:ident, $device_type:ty) => {
        pub fn $name(
            device: &std::sync::Arc<$device_type>,
            inputs: &Inputs,
            how: How,
        ) -> Result<Outputs, Error> {
            let a = &inputs.a[..];
            let n = a.len();
            let k = 3.0_f32;
            let b = 0.25_f32;
            let scaled = finish!(
                how,
                gpu_task!(on(std::sync::Arc::clone(device)); n, [a: &[f32], k: f32, b: f32], |i| -> f32 {
                    a[i] * k + b
                })
            );

            let bias = &inputs.bias[..];
            let grid = finish!(
                how,
                gpu_task!(on(std::sync::Arc::clone(device)); (WIDTH, HEIGHT), [bias: &[u32]], |x, y| -> u32 {
                    bias[x] + y * 100u32
                })
            );

            let mut data = inputs.data.clone();
            finish!(
                how,
                gpu_task!(on(std::sync::Arc::clone(device)); &mut data, [data: &mut [f32]], |i| {
                    data[i] = data[i] * 0.5 + 1.0;
                })
            );

            let weights = &inputs.weights[..];
            let weighed = finish!(
                how,
                gpu_task!(on(std::sync::Arc::clone(device)); n, [a: &[f32], weights: &[f32]], |i| -> f32 {
                    weights[i % 4u32] + a[i]
                })
            );
            Ok(Outputs {
                scaled,
                grid,
                halved: data,
                weighed,
            })
        }
    };
}

#[cfg(any(test, gust_dual))]
define_run!(run_cpu, CpuDevice);
define_run!(run_gpu, HeadlessDevice);

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle: DeviceHandle = default_device()?;
    let info = handle.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    let inputs = inputs();
    let want = reference(&inputs);
    for how in [How::Dispatch, How::Submit] {
        let got = run_gpu(&handle, &inputs, how)?;
        if got != want {
            return Err(format!("{how:?}: the GPU result differs from the host reference").into());
        }
        println!(
            "gpu, {how:?}: scale {} values, grid {WIDTH} x {HEIGHT}, halve {LEN} values in place, weigh {} values; equal to the host",
            got.scaled.len(),
            got.weighed.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn cpu() -> Arc<CpuDevice> {
        Arc::new(CpuDevice::new())
    }

    #[test]
    fn twin_matches_the_host_with_dispatch() {
        let inputs = inputs();
        let got = run_cpu(&cpu(), &inputs, How::Dispatch).unwrap();
        assert_eq!(got, reference(&inputs));
    }

    #[test]
    fn twin_matches_the_host_with_submit() {
        let inputs = inputs();
        let got = run_cpu(&cpu(), &inputs, How::Submit).unwrap();
        assert_eq!(got, reference(&inputs));
    }

    #[test]
    fn twin_an_empty_shape_runs_nothing() {
        let device = cpu();
        let a: &[f32] = &[];
        let n = a.len();
        let nothing = gpu_task!(on(Arc::clone(&device)); n, [a: &[f32]], |i| -> f32 { a[i] })
            .dispatch()
            .unwrap();
        assert!(nothing.is_empty());
        let mut data: Vec<f32> = Vec::new();
        gpu_task!(on(Arc::clone(&device)); &mut data, [data: &mut [f32]], |i| { data[i] = 1.0; })
            .dispatch()
            .unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn twin_a_submitted_task_is_ready_after_its_polls_and_the_target_holds_the_answer() {
        let device = cpu();
        let mut data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let pending =
            gpu_task!(on(Arc::clone(&device)); &mut data, [data: &mut [f32]], |i| { data[i] = data[i] * 0.5 + 1.0; })
                .submit()
                .unwrap();
        let (result, polls) = poll_to_end(pending);
        result.unwrap();
        assert!(polls >= 1);
        let want: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 + 1.0).collect();
        assert_eq!(data, want);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_host() {
        let handle = default_device().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        });
        let info = handle.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        let inputs = inputs();
        let want = reference(&inputs);
        for how in [How::Dispatch, How::Submit] {
            let got = run_gpu(&handle, &inputs, how).unwrap();
            assert_eq!(got, want, "{how:?}");
            println!("gpu on {}, {how:?}: equal to the host", info.name);
        }
    }
}

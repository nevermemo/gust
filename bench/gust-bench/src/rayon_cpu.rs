//! CPU comparators: rayon with an explicit thread count, and a serial loop as the sanity baseline.
//!
//! All of them work on host slices. There is no transfer, so for them "end to end" and "dispatch" are the same
//! computation; the output vector is allocated by the caller, outside the timed region.
//!
//! The histogram (W4): the rayon rows count into one shared array of `AtomicU32` (the counterpart of the GPU's atomic
//! add, contended the same way: a Zipf stream hits few counters from every thread); the serial row counts into a plain
//! `u32` array. Every call clears its bins first, so a call is one complete histogram, like a GPU end-to-end sample.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{Inputs, Workload, polynomial, saxpy, vector_add},
};

pub struct CpuLoop {
    id: &'static str,
    description: &'static str,
    pool: Option<ThreadPool>,
    workload: Workload,
    inputs: Option<Inputs>,
    out: Vec<f32>,
    atomic_bins: Vec<AtomicU32>,
    plain_bins: Vec<u32>,
}

impl CpuLoop {
    /// A plain `for` loop on the calling thread.
    pub fn serial() -> Self {
        Self::build("serial", "plain loop, one thread (sanity baseline)", None)
    }

    /// rayon in a dedicated pool of `threads` workers.
    pub fn rayon(threads: usize) -> Self {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        let (id, description) = if threads == 1 {
            ("rayon-1", "rayon par_iter in a 1-thread pool")
        } else {
            ("rayon-N", "rayon par_iter in a pool of all logical cores")
        };
        Self::build(id, description, Some(pool))
    }

    fn build(id: &'static str, description: &'static str, pool: Option<ThreadPool>) -> Self {
        Self {
            id,
            description,
            pool,
            workload: Workload::VectorAdd,
            inputs: None,
            out: Vec::new(),
            atomic_bins: Vec::new(),
            plain_bins: Vec::new(),
        }
    }

    /// One complete computation of the workload from `inputs` into `out` (resized to the output length).
    fn compute(&mut self, workload: Workload, inputs: &Inputs, out: &mut Vec<f32>) {
        out.resize(workload.out_len(inputs.n), 0.0);
        match workload.family() {
            // The workload is matched once, outside the loop, so each loop body is the bare element function.
            "W1" => self.run(&inputs.x, &inputs.y, out, vector_add::element),
            "W2" => {
                let scale = inputs.scale;
                self.run(&inputs.x, &inputs.y, out, move |x, y| {
                    saxpy::element(scale, x, y)
                });
            }
            "W3" => self.run_poly(&inputs.coeffs, &inputs.x, out),
            _ => self.histogram(&inputs.keys, inputs.bins, out),
        }
    }

    fn run(&self, x: &[f32], y: &[f32], out: &mut [f32], f: impl Fn(f32, f32) -> f32 + Sync) {
        match &self.pool {
            None => {
                for ((o, x), y) in out.iter_mut().zip(x).zip(y) {
                    *o = f(*x, *y);
                }
            }
            Some(pool) => pool.install(|| {
                out.par_iter_mut()
                    .zip(x.par_iter())
                    .zip(y.par_iter())
                    .for_each(|((o, x), y)| *o = f(*x, *y));
            }),
        }
    }

    fn run_poly(&self, coeffs: &[f32], x: &[f32], out: &mut [f32]) {
        match &self.pool {
            None => {
                for (o, x) in out.iter_mut().zip(x) {
                    *o = polynomial::element(coeffs, *x);
                }
            }
            Some(pool) => pool.install(|| {
                out.par_iter_mut()
                    .zip(x.par_iter())
                    .for_each(|(o, x)| *o = polynomial::element(coeffs, *x));
            }),
        }
    }

    fn histogram(&mut self, keys: &[u32], bins: usize, out: &mut [f32]) {
        match &self.pool {
            None => {
                self.plain_bins.clear();
                self.plain_bins.resize(bins, 0);
                for key in keys {
                    self.plain_bins[*key as usize] += 1;
                }
                for (o, count) in out.iter_mut().zip(&self.plain_bins) {
                    *o = f32::from_bits(*count);
                }
            }
            Some(pool) => {
                if self.atomic_bins.len() != bins {
                    self.atomic_bins = (0..bins).map(|_| AtomicU32::new(0)).collect();
                }
                let counters = &self.atomic_bins;
                for counter in counters {
                    counter.store(0, Relaxed);
                }
                pool.install(|| {
                    keys.par_iter().for_each(|key| {
                        counters[*key as usize].fetch_add(1, Relaxed);
                    });
                });
                for (o, counter) in out.iter_mut().zip(counters) {
                    *o = f32::from_bits(counter.load(Relaxed));
                }
            }
        }
    }
}

impl Backend for CpuLoop {
    fn id(&self) -> &'static str {
        self.id
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn kind(&self) -> Kind {
        Kind::Cpu
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        self.workload = workload;
        self.inputs = Some(inputs.clone());
        self.out = vec![workload.poison_word(); workload.out_len(inputs.n)];
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        self.out.fill(self.workload.poison_word());
        Ok(())
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        self.compute(self.workload, inputs, out);
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        let inputs = self.inputs.take().expect("prepare was called");
        let mut out = std::mem::take(&mut self.out);
        self.compute(self.workload, &inputs, &mut out);
        self.out = out;
        self.inputs = Some(inputs);
        Ok(())
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        out.clone_from(&self.out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{check, gen_inputs, reference};

    /// Every workload on every CPU backend passes its check, through `end_to_end` and through `dispatch` +
    /// `read_back`, and the poison alone fails it (the check can fail on the CPU path too).
    #[test]
    fn every_workload_passes_on_every_cpu_backend_and_the_poison_fails() {
        for w in Workload::ALL {
            let inputs = gen_inputs(w, 5000);
            let reference = reference(w, &inputs);
            for mut backend in [CpuLoop::serial(), CpuLoop::rayon(1), CpuLoop::rayon(3)] {
                backend.prepare(w, &inputs).unwrap();
                let mut out = Vec::new();
                backend.poison().unwrap();
                backend.read_back(&mut out).unwrap();
                assert!(
                    check(w, &inputs, &reference, &out).is_err(),
                    "{} poison",
                    w.id()
                );
                backend.end_to_end(&inputs, &mut out).unwrap();
                check(w, &inputs, &reference, &out)
                    .unwrap_or_else(|e| panic!("{} {}: {e}", backend.id, w.id()));
                backend.poison().unwrap();
                backend.dispatch().unwrap();
                backend.dispatch().unwrap();
                backend.read_back(&mut out).unwrap();
                check(w, &inputs, &reference, &out)
                    .unwrap_or_else(|e| panic!("{} {} dispatch: {e}", backend.id, w.id()));
            }
        }
    }
}

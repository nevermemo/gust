//! CPU comparators: rayon with an explicit thread count, and a serial loop as the sanity baseline.
//!
//! All of them work on host slices. There is no transfer, so for them "end to end" and "dispatch" are the same
//! computation; the output vector is allocated by the caller, outside the timed region.
//!
//! The histogram (W4): the rayon rows count into one shared array of `AtomicU32` (the counterpart of the GPU's atomic
//! add, contended the same way: a Zipf stream hits few counters from every thread); the serial row counts into a plain
//! `u32` array. Every call clears its bins first, so a call is one complete histogram, like a GPU end-to-end sample.
//!
//! The particle step (W5): the serial row steps each particle in turn (`particles::step_words`), rayon takes the particles
//! in parallel. The sum (W7): rayon folds a chunk per task and reduces the partial sums (`wrapping_add` for `u32`, `+` for `f32`, so
//! its order of summation is a tree whose shape depends on the splits); the serial row adds left to right. The naive
//! matrix product (W9): the serial loop is `matmul::row` for every row, rayon takes the rows in parallel.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{Inputs, Workload, matmul, particles, polynomial, saxpy, sum, vector_add},
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
            "W5" => self.particles(inputs, out),
            "W7" => out[0] = self.sum(workload, inputs),
            "W9" => self.matmul(inputs, out),
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

    /// The sum as one output word (the `u32` sum as its bit pattern).
    fn sum(&self, workload: Workload, inputs: &Inputs) -> f32 {
        if workload == Workload::SumU32 {
            let values = &inputs.values;
            let total = match &self.pool {
                None => sum::wrapping_sum(values),
                Some(pool) => pool.install(|| {
                    values
                        .par_iter()
                        .fold(|| 0u32, |sum, v| sum.wrapping_add(*v))
                        .reduce(|| 0u32, u32::wrapping_add)
                }),
            };
            return f32::from_bits(total);
        }
        let values = &inputs.x;
        match &self.pool {
            None => sum::serial_f32(values),
            Some(pool) => pool.install(|| {
                values
                    .par_iter()
                    .fold(|| 0.0f32, |sum, v| sum + *v)
                    .reduce(|| 0.0f32, |a, b| a + b)
            }),
        }
    }

    /// The particle step (W5): one particle (7 words) and its acceleration (2 words) at a time, 16 steps each.
    fn particles(&self, inputs: &Inputs, out: &mut [f32]) {
        let (x, y) = (&inputs.x, &inputs.y);
        match &self.pool {
            None => {
                for ((o, p), a) in out
                    .chunks_mut(particles::WORDS)
                    .zip(x.chunks(particles::WORDS))
                    .zip(y.chunks(particles::ACCEL_WORDS))
                {
                    particles::step_words(p, a, o);
                }
            }
            Some(pool) => pool.install(|| {
                out.par_chunks_mut(particles::WORDS)
                    .zip(x.par_chunks(particles::WORDS))
                    .zip(y.par_chunks(particles::ACCEL_WORDS))
                    .for_each(|((o, p), a)| particles::step_words(p, a, o));
            }),
        }
    }

    /// The naive product, a row of `C` at a time.
    fn matmul(&self, inputs: &Inputs, out: &mut [f32]) {
        let n = inputs.n;
        let (a, b) = (&inputs.x, &inputs.y);
        match &self.pool {
            None => {
                for (i, row) in out.chunks_mut(n).enumerate() {
                    matmul::row(n, a, b, i, row);
                }
            }
            Some(pool) => pool.install(|| {
                out.par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(i, row)| matmul::row(n, a, b, i, row));
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
            // W9's `n` is the side of the matrices: a small one keeps the cubic naive product short.
            let inputs = gen_inputs(w, crate::workloads::test_size(w));
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

//! CPU comparators: rayon with an explicit thread count, and a serial loop as the sanity baseline.
//!
//! All of them work on host slices. There is no transfer, so for them "end to end" and "dispatch" are the same
//! computation; the output vector is allocated by the caller, outside the timed region.

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{Inputs, Workload, saxpy, vector_add},
};

pub struct CpuLoop {
    id: &'static str,
    description: &'static str,
    pool: Option<ThreadPool>,
    workload: Workload,
    inputs: Option<Inputs>,
    out: Vec<f32>,
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
        }
    }

    fn compute(&self, workload: Workload, x: &[f32], y: &[f32], scale: f32, out: &mut [f32]) {
        // The workload is matched once, outside the loop, so each loop body is the bare element function.
        match workload {
            Workload::VectorAdd => self.run(x, y, out, vector_add::element),
            Workload::Saxpy => self.run(x, y, out, move |x, y| saxpy::element(scale, x, y)),
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
        self.inputs = Some(Inputs {
            x: inputs.x.clone(),
            y: inputs.y.clone(),
            scale: inputs.scale,
        });
        self.out = vec![f32::NAN; inputs.x.len()];
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        self.out.fill(f32::NAN);
        Ok(())
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        self.compute(self.workload, &inputs.x, &inputs.y, inputs.scale, out);
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        let inputs = self.inputs.as_ref().expect("prepare was called");
        let mut out = std::mem::take(&mut self.out);
        self.compute(self.workload, &inputs.x, &inputs.y, inputs.scale, &mut out);
        self.out = out;
        Ok(())
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        out.clone_from(&self.out);
        Ok(())
    }
}

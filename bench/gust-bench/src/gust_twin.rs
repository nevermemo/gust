//! The GUST CPU twin: the shadow body run once per invocation, single-threaded, in safe Rust (`gust::cpu`). It exists
//! only in a `--cfg gust_dual` build (D26). It is a semantic reference, not the production CPU target (that is M5).

use gust::{
    Independent,
    cpu::{CpuBuffer, CpuDevice},
};

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{Inputs, Workload, saxpy, vector_add},
};

struct Resident {
    workload: Workload,
    n: usize,
    x: CpuBuffer<f32>,
    y: CpuBuffer<f32>,
    scale: CpuBuffer<f32>,
    out: CpuBuffer<f32>,
}

pub struct GustTwin {
    device: CpuDevice,
    resident: Option<Resident>,
}

impl GustTwin {
    pub fn new() -> Self {
        Self {
            device: CpuDevice::new(),
            resident: None,
        }
    }

    fn resident(&self) -> &Resident {
        self.resident.as_ref().expect("prepare was called")
    }

    /// One typed `dispatch` on the twin: the call a user of the generated API makes.
    fn run(&self, r: &Resident) -> BenchResult<()> {
        let n = r.n as u32;
        match r.workload {
            Workload::VectorAdd => {
                vector_add::kernel::add::dispatch(&self.device, n, &r.x, &r.y, &r.out)?
            }
            Workload::Saxpy => saxpy::kernel::saxpy::dispatch(
                &self.device,
                n,
                Independent(&r.scale),
                &r.x,
                &r.y,
                &r.out,
            )?,
        }
        Ok(())
    }
}

impl Backend for GustTwin {
    fn id(&self) -> &'static str {
        "gust-twin"
    }

    fn description(&self) -> &'static str {
        "GUST CPU twin (--cfg gust_dual), single thread, semantic reference with per-call validation"
    }

    fn kind(&self) -> Kind {
        Kind::Cpu
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        let n = inputs.x.len();
        self.resident = Some(Resident {
            workload,
            n,
            x: self.device.buffer(&inputs.x),
            y: self.device.buffer(&inputs.y),
            scale: self.device.buffer(&[inputs.scale]),
            out: self.device.buffer(&vec![f32::NAN; n]),
        });
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        let r = self.resident();
        r.out.write(&vec![f32::NAN; r.n])?;
        Ok(())
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        r.x.write(&inputs.x)?;
        r.y.write(&inputs.y)?;
        self.run(r)?;
        *out = r.out.read();
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        self.run(self.resident())
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        *out = self.resident().out.read();
        Ok(())
    }
}

//! The GUST CPU backends: kernels run on the CPU through `gust::cpu::CpuDevice`, the kernel's own Rust shadow body once
//! per invocation, single-threaded, in ascending invocation order.
//!
//! - In a `--cfg gust_dual` GPU build it is `gust-twin` (D26): a semantic reference, with per-call validation.
//! - In the `GUST_PROFILE=cpu` build (`--no-default-features`, no GPU runtime in the binary) it is `gust-cpu`, the
//!   production CPU target (D21): the same dispatch, as the macro emits it in every build of that profile. Its parallel
//!   path is the generated map (`gust-cpu-par`, W1 and W2 only).
//!
//! The kernels are the ones the GPU backends run (`workloads::*::kernel`), so the arithmetic is the same source.

use std::cell::Cell;

use gust::{
    Extent, Independent,
    cpu::{CpuBuffer, CpuDevice},
};

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{
        Inputs, Workload, counts_as_words, hist_threads, histogram, matmul, normalize_counts,
        particles, polynomial, saxpy, signal_pipeline, vector_add,
    },
};

enum Data {
    Add {
        x: CpuBuffer<f32>,
        y: CpuBuffer<f32>,
        out: CpuBuffer<f32>,
    },
    Saxpy {
        scale: CpuBuffer<f32>,
        x: CpuBuffer<f32>,
        y: CpuBuffer<f32>,
        out: CpuBuffer<f32>,
    },
    Poly {
        coeffs: CpuBuffer<f32>,
        x: CpuBuffer<f32>,
        out: CpuBuffer<f32>,
    },
    Hist {
        keys: CpuBuffer<u32>,
        bins: CpuBuffer<u32>,
    },
    MatMul {
        dims: CpuBuffer<u32>,
        a: CpuBuffer<f32>,
        b: CpuBuffer<f32>,
        c: CpuBuffer<f32>,
    },
    /// W5: the typed particle and acceleration buffers, and the typed host copies an end-to-end sample uploads (built once
    /// in `prepare`, so no sample converts the harness's words).
    Particles {
        input: CpuBuffer<particles::kernel::Particle>,
        acceleration: CpuBuffer<particles::kernel::Pair>,
        out: CpuBuffer<particles::kernel::Particle>,
        host_input: Vec<particles::kernel::Particle>,
        host_acceleration: Vec<particles::kernel::Pair>,
    },
    /// W6: the five input streams and the six stage outputs (the last is the output), run as six dispatches in stage order.
    Pipeline {
        inputs: Vec<CpuBuffer<f32>>,
        stages: Vec<CpuBuffer<f32>>,
    },
}

struct Resident {
    workload: Workload,
    n: usize,
    data: Data,
}

pub struct GustCpu {
    device: CpuDevice,
    resident: Option<Resident>,
    /// Dispatches since the bins were last cleared (a histogram accumulates, as on the GPU).
    passes: Cell<u32>,
}

impl GustCpu {
    pub fn new() -> Self {
        Self {
            device: CpuDevice::new(),
            resident: None,
            passes: Cell::new(0),
        }
    }

    fn resident(&self) -> &Resident {
        self.resident.as_ref().expect("prepare was called")
    }

    /// One typed `dispatch` on the CPU device: the call a user of the generated API makes.
    fn run(&self, r: &Resident) -> BenchResult<()> {
        let n = r.n as u32;
        match &r.data {
            Data::Add { x, y, out } => {
                vector_add::kernel::add::dispatch(&self.device, n, x, y, out)?
            }
            Data::Saxpy { scale, x, y, out } => {
                saxpy::kernel::saxpy::dispatch(&self.device, n, Independent(scale), x, y, out)?
            }
            Data::Poly { coeffs, x, out } => {
                polynomial::kernel::horner::dispatch(&self.device, n, Independent(coeffs), x, out)?
            }
            Data::Hist { keys, bins } => histogram::kernel::histogram::dispatch(
                &self.device,
                hist_threads(r.n) as u32,
                Independent(keys),
                Independent(bins),
            )?,
            Data::MatMul { dims, a, b, c } => matmul::kernel::matmul::dispatch_extent(
                &self.device,
                Extent::new(n, n, 1),
                Independent(dims),
                Independent(a),
                Independent(b),
                Independent(c),
            )?,
            Data::Particles {
                input,
                acceleration,
                out,
                ..
            } => particles::kernel::step::dispatch(&self.device, n, input, acceleration, out)?,
            Data::Pipeline {
                inputs: i,
                stages: s,
            } => {
                use signal_pipeline::kernel;
                let d = &self.device;
                kernel::center::dispatch(d, n, &i[0], &i[1], &s[0])?;
                kernel::amplify::dispatch(d, n, &s[0], &i[2], &s[1])?;
                kernel::shift::dispatch(d, n, &s[1], &i[3], &s[2])?;
                kernel::rescale::dispatch(d, n, &s[2], &i[4], &s[3])?;
                kernel::energy::dispatch(d, n, &s[3], &s[4])?;
                kernel::combine::dispatch(d, n, &s[4], &s[3], &s[5])?;
            }
        }
        Ok(())
    }

    fn words(&self) -> Vec<f32> {
        match &self.resident().data {
            Data::Add { out, .. } | Data::Saxpy { out, .. } | Data::Poly { out, .. } => out.read(),
            Data::Hist { bins, .. } => counts_as_words(bins.read()),
            Data::MatMul { c, .. } => c.read(),
            Data::Particles { out, .. } => particles::from_particles(&out.read()),
            Data::Pipeline { stages, .. } => stages[signal_pipeline::STAGES - 1].read(),
        }
    }
}

impl Backend for GustCpu {
    fn id(&self) -> &'static str {
        if gust::profile::IS_CPU {
            "gust-cpu"
        } else {
            "gust-twin"
        }
    }

    fn description(&self) -> &'static str {
        if gust::profile::IS_CPU {
            "GUST production CPU target (GUST_PROFILE=cpu, no GPU runtime in the build): the kernel's shadow body through CpuDevice, one thread, typed dispatch with per-call validation"
        } else {
            "GUST CPU twin (--cfg gust_dual), single thread, semantic reference with per-call validation"
        }
    }

    fn kind(&self) -> Kind {
        Kind::Cpu
    }

    /// The twin and the production CPU target both run W5 (the struct kernel's shadow body). The twin runs W9 (a 2-D kernel
    /// with a twin); the production CPU target is not on W9's list, and no GUST CPU
    /// backend sums (W7 is `gust-algo`'s, which has no CPU path of its own to time here).
    fn supports(&self, workload: Workload) -> bool {
        // W6 in `StagedGraph` form needs the GPU runtime; the hand-ordered form (six dispatches, no graph) runs here.
        if workload.is_graph_form() {
            return false;
        }
        match workload.family() {
            "W7" => false,
            "W9" => !gust::profile::IS_CPU,
            _ => true,
        }
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        let d = &self.device;
        let n = inputs.n;
        let data = match workload.family() {
            "W1" | "W2" => {
                let x = d.buffer(&inputs.x);
                let y = d.buffer(&inputs.y);
                let out = d.buffer(&vec![f32::NAN; n]);
                if workload == Workload::VectorAdd {
                    Data::Add { x, y, out }
                } else {
                    Data::Saxpy {
                        scale: d.buffer(&[inputs.scale]),
                        x,
                        y,
                        out,
                    }
                }
            }
            "W3" => Data::Poly {
                coeffs: d.buffer(&inputs.coeffs),
                x: d.buffer(&inputs.x),
                out: d.buffer(&vec![f32::NAN; n]),
            },
            "W5" => {
                let host_input = particles::to_particles(&inputs.x);
                let host_acceleration = particles::to_pairs(&inputs.y);
                Data::Particles {
                    input: d.buffer(&host_input),
                    acceleration: d.buffer(&host_acceleration),
                    out: d.buffer(&particles::to_particles(&vec![
                        f32::NAN;
                        n * particles::WORDS
                    ])),
                    host_input,
                    host_acceleration,
                }
            }
            "W6" => Data::Pipeline {
                inputs: inputs
                    .pipeline
                    .iter()
                    .map(|values| d.buffer(values))
                    .collect(),
                stages: (0..signal_pipeline::STAGES)
                    .map(|_| d.buffer(&vec![f32::NAN; n]))
                    .collect(),
            },
            "W7" => return Err("the GUST CPU backends have no W7".into()),
            "W9" => Data::MatMul {
                dims: d.buffer(&[n as u32]),
                a: d.buffer(&inputs.x),
                b: d.buffer(&inputs.y),
                c: d.buffer(&vec![f32::NAN; n * n]),
            },
            _ => Data::Hist {
                keys: d.buffer(&inputs.keys),
                bins: d.buffer(&vec![0u32; inputs.bins]),
            },
        };
        self.passes.set(0);
        self.resident = Some(Resident { workload, n, data });
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        let r = self.resident();
        match &r.data {
            Data::Add { out, .. } | Data::Saxpy { out, .. } | Data::Poly { out, .. } => {
                out.write(&vec![f32::NAN; r.n])?
            }
            Data::Hist { bins, .. } => bins.write(&vec![0u32; bins.read().len()])?,
            Data::MatMul { c, .. } => c.write(&vec![f32::NAN; r.n * r.n])?,
            Data::Particles { out, .. } => out.write(&particles::to_particles(&vec![
                f32::NAN;
                r.n * particles::WORDS
            ]))?,
            Data::Pipeline { stages, .. } => {
                for stage in stages {
                    stage.write(&vec![f32::NAN; r.n])?;
                }
            }
        }
        self.passes.set(0);
        Ok(())
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        match &r.data {
            Data::Add { x, y, .. } | Data::Saxpy { x, y, .. } => {
                x.write(&inputs.x)?;
                y.write(&inputs.y)?;
            }
            Data::Poly { x, .. } => x.write(&inputs.x)?,
            Data::Hist { keys, bins } => {
                keys.write(&inputs.keys)?;
                bins.write(&vec![0u32; inputs.bins])?;
            }
            Data::MatMul { a, b, .. } => {
                a.write(&inputs.x)?;
                b.write(&inputs.y)?;
            }
            Data::Particles {
                input,
                acceleration,
                host_input,
                host_acceleration,
                ..
            } => {
                input.write(host_input)?;
                acceleration.write(host_acceleration)?;
            }
            Data::Pipeline {
                inputs: buffers, ..
            } => {
                for (buffer, values) in buffers.iter().zip(&inputs.pipeline) {
                    buffer.write(values)?;
                }
            }
        }
        self.run(r)?;
        self.passes.set(1);
        *out = self.words();
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        self.run(self.resident())?;
        self.passes.set(self.passes.get().wrapping_add(1));
        Ok(())
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        *out = self.words();
        if self.resident().workload.accumulates() {
            normalize_counts(out, self.passes.get());
        }
        Ok(())
    }
}

/// The production CPU target's parallel path: the generated map (`gust::cpu::parallel`) over host slices, W1 and W2
/// only, in the `cpu`-profile build. A map takes host slices and returns a fresh result vector, so every call includes
/// the buffer copies the map makes; `dispatch` and `end_to_end` are the same call.
#[cfg(not(feature = "gpu"))]
pub struct GustCpuPar {
    workload: Workload,
    inputs: Option<Inputs>,
    out: Vec<f32>,
}

#[cfg(not(feature = "gpu"))]
impl GustCpuPar {
    pub fn new() -> Self {
        Self {
            workload: Workload::VectorAdd,
            inputs: None,
            out: Vec::new(),
        }
    }

    fn map(workload: Workload, inputs: &Inputs) -> BenchResult<Vec<f32>> {
        use gust::map::Broadcast;

        use crate::workloads::maps;
        Ok(match workload {
            Workload::VectorAdd => maps::add_map::map(&inputs.x, &inputs.y)?,
            Workload::Saxpy => maps::saxpy_map::map(&inputs.x, &inputs.y, Broadcast(inputs.scale))?,
            other => return Err(format!("gust-cpu-par has no map form of {}", other.id()).into()),
        })
    }
}

#[cfg(not(feature = "gpu"))]
impl Backend for GustCpuPar {
    fn id(&self) -> &'static str {
        "gust-cpu-par"
    }

    fn description(&self) -> &'static str {
        "GUST production CPU target, the generated element-wise map (gust::cpu::parallel: chunked over scoped threads), host slices in, fresh result vector out; W1 and W2"
    }

    fn kind(&self) -> Kind {
        Kind::Cpu
    }

    fn supports(&self, workload: Workload) -> bool {
        matches!(workload, Workload::VectorAdd | Workload::Saxpy)
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        self.workload = workload;
        self.inputs = Some(inputs.clone());
        // `supports` keeps the run from asking for a workload without a map form; this fails loudly if it does.
        self.out = Self::map(workload, inputs)?;
        self.out.fill(f32::NAN);
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        self.out.fill(f32::NAN);
        Ok(())
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        *out = Self::map(self.workload, inputs)?;
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        let inputs = self.inputs.as_ref().expect("prepare was called");
        self.out = Self::map(self.workload, inputs)?;
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

    /// Runs every workload through `backend` (end to end, then dispatch twice and read back) and checks each against
    /// its bound or exact counts; the poison alone must fail.
    fn passes_everything(mut backend: impl Backend, workloads: &[Workload]) {
        let supported = workloads
            .iter()
            .copied()
            .filter(|w| backend.supports(*w))
            .collect::<Vec<_>>();
        for w in supported {
            // W9's `n` is the matrix side: a small one keeps the cubic naive product short.
            let inputs = gen_inputs(w, crate::workloads::test_size(w));
            let want = reference(w, &inputs);
            backend.prepare(w, &inputs).unwrap();
            let mut out = Vec::new();
            backend.poison().unwrap();
            backend.read_back(&mut out).unwrap();
            assert!(check(w, &inputs, &want, &out).is_err(), "{} poison", w.id());
            backend.end_to_end(&inputs, &mut out).unwrap();
            check(w, &inputs, &want, &out)
                .unwrap_or_else(|e| panic!("{} {}: {e}", backend.id(), w.id()));
            backend.poison().unwrap();
            backend.dispatch().unwrap();
            backend.dispatch().unwrap();
            backend.read_back(&mut out).unwrap();
            check(w, &inputs, &want, &out)
                .unwrap_or_else(|e| panic!("{} {} dispatch: {e}", backend.id(), w.id()));
        }
    }

    #[test]
    fn the_cpu_kernels_pass_every_workload_and_the_poison_fails() {
        passes_everything(GustCpu::new(), &Workload::ALL);
        assert_eq!(
            GustCpu::new().id(),
            if gust::profile::IS_CPU {
                "gust-cpu"
            } else {
                "gust-twin"
            }
        );
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn the_generated_maps_pass_w1_and_w2_and_the_other_workloads_are_left_out() {
        passes_everything(GustCpuPar::new(), &[Workload::VectorAdd, Workload::Saxpy]);
        let par = GustCpuPar::new();
        assert!(par.supports(Workload::Saxpy));
        assert!(!par.supports(Workload::Poly4) && !par.supports(Workload::Hist256Zipf));
    }
}

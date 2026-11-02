//! The run crate: a generated crate (`emit.rs`) plus the `#[cfg(test)]` module that runs every program on the CPU twin
//! and, unless told otherwise, on the GPU, and the data files that module reads.
//!
//! `emit_crate_source` is untouched (the corpus hashes cover it); the run crate's `src/lib.rs` is that text followed by
//! the test module. The test module is compiled only by `cargo test` (the twin exists only under `cfg(test)`), and it
//! reads the inputs and the interpreter's output from `data/<module>.txt` at run time, so a 4096-element program does
//! not become a 4096-element source literal.

use std::fmt::Write as _;
use std::io;
use std::path::Path;

use crate::ast::{Program, Ty};
use crate::emit::{
    BUFFER_NAMES, emit_crate_source, lock_for, manifest, module_name, path_for_toml,
};
use crate::generator::DIFFTEST_FORMAT;

const PRELUDE: &str = include_str!("run_prelude.txt");

/// One program of a run: the program, its position within its seed's batch, and what the interpreter says it computes.
#[derive(Clone, Debug)]
pub struct Planned {
    pub program: Program,
    /// Position in the seed's batch (0 to 15): `--shrink <seed>:<index>` names a program this way.
    pub index: usize,
    /// The interpreter's output buffer (bit patterns; `bool` as 0 or 1).
    pub expected: Vec<u32>,
    /// For an `f32` output, the interval of every element that a conforming implementation may hold; empty otherwise.
    pub intervals: Vec<(f64, f64)>,
}

/// The programs of `seed` with the interpreter's output for each (the batch holds only in-domain programs, so the
/// run cannot fail).
pub fn plan_seed(seed: u64) -> Result<Vec<Planned>, String> {
    let batch = crate::generator::generate_batch(seed)?;
    batch
        .programs
        .into_iter()
        .enumerate()
        .map(|(index, program)| {
            let run = crate::interp::run_full(&program).map_err(|f| {
                format!("seed {seed} program {index}: {f:?} after the batch accepted it")
            })?;
            Ok(Planned {
                program,
                index,
                expected: run.values,
                intervals: run.intervals,
            })
        })
        .collect()
}

/// The Rust element type a buffer of `ty` has in the generated test (`bool` is never a buffer type).
fn rust_ty(ty: Ty) -> &'static str {
    match ty {
        Ty::U32 | Ty::Bool => "u32",
        Ty::I32 => "i32",
        Ty::F32 => "f32",
        Ty::Vec(..) | Ty::Mat(_) => unreachable!("no vector or matrix buffer"),
    }
}

/// An expression that turns the `u32` bit pattern `word` into a value of buffer type `ty`.
fn from_word(ty: Ty, word: &str) -> String {
    match ty {
        Ty::U32 | Ty::Bool => word.to_owned(),
        Ty::I32 => format!("{word} as i32"),
        Ty::F32 => format!("f32::from_bits({word})"),
        Ty::Vec(..) | Ty::Mat(_) => unreachable!("no vector or matrix buffer"),
    }
}

/// An expression that turns the value `value` of buffer type `ty` into its `u32` bit pattern.
fn to_word(ty: Ty, value: &str) -> String {
    match ty {
        Ty::U32 | Ty::Bool => value.to_owned(),
        Ty::I32 => format!("{value} as u32"),
        Ty::F32 => format!("{value}.to_bits()"),
        Ty::Vec(..) | Ty::Mat(_) => unreachable!("no vector or matrix buffer"),
    }
}

fn zero(ty: Ty) -> &'static str {
    match ty {
        Ty::U32 | Ty::Bool => "0u32",
        Ty::I32 => "0i32",
        Ty::F32 => "0.0f32",
        Ty::Vec(..) | Ty::Mat(_) => unreachable!("no vector or matrix buffer"),
    }
}

/// The runner function of one program.
fn runner(planned: &Planned) -> String {
    let program = &planned.program;
    let module = module_name(program);
    let out_ty = program.out_ty();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "    fn run_{module}(data: &Data, gpu: Option<&HeadlessDevice>) -> Outcome {{"
    );
    for (k, input) in program.inputs.iter().enumerate() {
        let name = BUFFER_NAMES[k];
        let _ = writeln!(
            out,
            "        let {name}: Vec<{}> = data.inputs[{k}].iter().map(|&w| {}).collect();",
            rust_ty(input.ty),
            from_word(input.ty, "w")
        );
    }
    let _ = writeln!(
        out,
        "        let zero: Vec<{}> = vec![{}; data.count as usize];",
        rust_ty(out_ty),
        zero(out_ty)
    );
    // The twin.
    let _ = writeln!(
        out,
        "        let twin = catch_unwind(AssertUnwindSafe(|| -> Result<(Vec<u32>, u128), String> {{"
    );
    let _ = writeln!(out, "            let cpu = CpuDevice::new();");
    let mut twin_args = String::new();
    for name in &BUFFER_NAMES[..program.inputs.len()] {
        let _ = writeln!(out, "            let c{name} = cpu.buffer(&{name});");
        let _ = write!(twin_args, ", &c{name}");
    }
    let _ = writeln!(out, "            let cout = cpu.buffer(&zero);");
    let _ = writeln!(out, "            let start = Instant::now();");
    let _ = writeln!(
        out,
        "            super::{module}::k::dispatch(&cpu, data.count{twin_args}, &cout).map_err(|e| format!(\"twin dispatch: {{e:?}}\"))?;"
    );
    let _ = writeln!(out, "            let ns = start.elapsed().as_nanos();");
    let _ = writeln!(
        out,
        "            Ok((cout.read().iter().map(|&v| {}).collect(), ns))",
        to_word(out_ty, "v")
    );
    let _ = writeln!(
        out,
        "        }})).unwrap_or_else(|p| Err(panic_text(\"twin\", p)));"
    );
    // The GPU: a first dispatch (it compiles the kernel with slangc and the helper, then wgpu creates the shader module
    // and the pipeline, which validates the WGSL with wgpu's default capabilities) and a second one for the warm time.
    let _ = writeln!(
        out,
        "        let gpu = gpu.map(|device| catch_unwind(AssertUnwindSafe(|| -> Result<GpuRun, String> {{"
    );
    let _ = writeln!(
        out,
        "            let descriptor = &super::{module}::k::DESCRIPTOR;"
    );
    let mut bindings = String::new();
    for name in &BUFFER_NAMES[..program.inputs.len()] {
        let _ = writeln!(
            out,
            "            let g{name} = device.create_typed_buffer(\"{name}\", &{name}, GpuBufferAccess::ReadOnly);"
        );
        let _ = write!(bindings, "BufferBinding::read_only(&g{name}), ");
    }
    let _ = writeln!(
        out,
        "            let gout = device.create_typed_buffer(\"out\", &zero, GpuBufferAccess::ReadWrite);"
    );
    let _ = writeln!(
        out,
        "            let bindings = [{bindings}BufferBinding::read_write(&gout)];"
    );
    let _ = writeln!(out, "            let start = Instant::now();");
    let _ = writeln!(
        out,
        "            device.dispatch_buffers(descriptor, data.count, &bindings).map_err(|e| format!(\"dispatch: {{e:?}}\"))?;"
    );
    let _ = writeln!(
        out,
        "            let first_ns = start.elapsed().as_nanos();"
    );
    // `DIFFTEST_WGSL_OUT` names a directory that receives the WGSL of each kernel (the structural short-circuit test
    // reads it); unset, nothing is compiled twice.
    let _ = writeln!(
        out,
        "            if let Some(dir) = std::env::var_os(\"DIFFTEST_WGSL_OUT\") {{"
    );
    let _ = writeln!(
        out,
        "                let wgsl = gust::slang::compile_wgsl(descriptor).map_err(|e| format!(\"compile_wgsl: {{e:?}}\"))?;"
    );
    let _ = writeln!(
        out,
        "                std::fs::write(PathBuf::from(dir).join(\"{module}.wgsl\"), wgsl).map_err(|e| e.to_string())?;"
    );
    let _ = writeln!(out, "            }}");
    let _ = writeln!(
        out,
        "            let values = device.read_typed_buffer(&gout).map_err(|e| format!(\"readback: {{e:?}}\"))?;"
    );
    let _ = writeln!(out, "            let start = Instant::now();");
    let _ = writeln!(
        out,
        "            device.dispatch_buffers(descriptor, data.count, &bindings).map_err(|e| format!(\"second dispatch: {{e:?}}\"))?;"
    );
    let _ = writeln!(out, "            let warm_ns = start.elapsed().as_nanos();");
    let _ = writeln!(out, "            Ok(GpuRun {{");
    let _ = writeln!(
        out,
        "                bits: values.iter().map(|&v| {}).collect(),",
        to_word(out_ty, "v")
    );
    let _ = writeln!(out, "                first_ns,");
    let _ = writeln!(out, "                warm_ns,");
    let _ = writeln!(out, "            }})");
    let _ = writeln!(
        out,
        "        }})).unwrap_or_else(|p| Err(panic_text(\"gpu\", p))));"
    );
    let _ = writeln!(out, "        Outcome {{ twin, gpu }}");
    let _ = writeln!(out, "    }}");
    out
}

/// The `#[cfg(test)]` module of a run crate.
pub fn run_tests_source(plan: &[Planned]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "#[cfg(test)]");
    let _ = writeln!(out, "mod difftest_run {{");
    out.push_str(
        &PRELUDE
            .replace("\r\n", "\n")
            .replace("__FORMAT__", &DIFFTEST_FORMAT.to_string()),
    );
    for planned in plan {
        out.push('\n');
        out.push_str(&runner(planned));
    }
    let _ = writeln!(out, "\n    const CASES: &[Case] = &[");
    for planned in plan {
        let program = &planned.program;
        let module = module_name(program);
        let _ = writeln!(
            out,
            "        Case {{ name: \"{module}\", seed: {}, attempt: {}, index: {}, out_ty: \"{}\", run: run_{module} }},",
            program.seed,
            program.attempt,
            planned.index,
            program.out_ty().name()
        );
    }
    let _ = writeln!(out, "    ];");
    let _ = writeln!(out, "}}");
    out
}

/// The `src/lib.rs` of a run crate: the kernels of `emit_crate_source`, then the test module.
pub fn run_crate_source(plan: &[Planned]) -> String {
    let programs: Vec<Program> = plan.iter().map(|p| p.program.clone()).collect();
    let mut out = emit_crate_source(&programs);
    out.push('\n');
    out.push_str(&run_tests_source(plan));
    out
}

/// The content of `data/<module>.txt` for one program.
pub fn data_file(planned: &Planned) -> String {
    let program = &planned.program;
    let words = |data: &[u32]| {
        data.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut out = format!("count {}\n", program.count);
    for (k, input) in program.inputs.iter().enumerate() {
        let _ = writeln!(out, "in{k} {} {}", input.ty.name(), words(&input.data));
    }
    let _ = writeln!(out, "interp {}", words(&planned.expected));
    out
}

/// Writes `path` only if its content would change, so that an unchanged file keeps its modification time and cargo
/// does not rebuild.
pub fn write_if_changed(path: &Path, content: &str) -> io::Result<()> {
    if let Ok(existing) = std::fs::read_to_string(path)
        && existing == content
    {
        return Ok(());
    }
    std::fs::write(path, content)
}

/// Writes a run crate named `package` into `dir`: `Cargo.toml`, the template lockfile for that name, `src/lib.rs`
/// and `data/<module>.txt`. Files whose content is unchanged are left alone; data files of programs that are no longer
/// in the plan are removed.
pub fn write_run_crate(
    dir: &Path,
    package: &str,
    plan: &[Planned],
    repo_root: &Path,
) -> io::Result<()> {
    let root = path_for_toml(repo_root).map_err(io::Error::other)?;
    let lock = lock_for(package).map_err(io::Error::other)?;
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::create_dir_all(dir.join("data"))?;
    write_if_changed(&dir.join("Cargo.toml"), &manifest(&root, package))?;
    write_if_changed(&dir.join("Cargo.lock"), &lock)?;
    write_if_changed(&dir.join("src").join("lib.rs"), &run_crate_source(plan))?;
    let mut wanted = Vec::new();
    for planned in plan {
        let name = format!("{}.txt", module_name(&planned.program));
        write_if_changed(&dir.join("data").join(&name), &data_file(planned))?;
        wanted.push(name);
    }
    for entry in std::fs::read_dir(dir.join("data"))? {
        let entry = entry?;
        if !wanted.contains(&entry.file_name().to_string_lossy().into_owned()) {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::generate_batch;
    use crate::interp::run_full;

    fn plan_of(seed: u64) -> Vec<Planned> {
        generate_batch(seed)
            .unwrap()
            .programs
            .into_iter()
            .enumerate()
            .map(|(index, program)| {
                let run = run_full(&program).unwrap();
                Planned {
                    program,
                    index,
                    expected: run.values,
                    intervals: run.intervals,
                }
            })
            .collect()
    }

    #[test]
    fn the_run_crate_source_is_the_kernel_source_plus_a_test_module() {
        let plan = plan_of(1);
        let programs: Vec<Program> = plan.iter().map(|p| p.program.clone()).collect();
        let text = run_crate_source(&plan);
        assert!(text.starts_with(&emit_crate_source(&programs)));
        assert!(!text.contains('\r'));
        assert_eq!(text.matches("#[cfg(test)]\nmod difftest_run {").count(), 1);
        assert_eq!(text.matches("    fn run_s").count(), plan.len());
        assert_eq!(text.matches("        Case { name: ").count(), plan.len());
        assert!(!text.contains("__FORMAT__"));
        assert!(text.contains(&format!("\"format\", \"{DIFFTEST_FORMAT}\"")));
    }

    #[test]
    fn a_data_file_carries_the_inputs_and_the_interpreter_output() {
        let plan = plan_of(2);
        let planned = &plan[0];
        let text = data_file(planned);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], format!("count {}", planned.program.count));
        assert_eq!(lines.len(), 2 + planned.program.inputs.len());
        assert!(lines.last().unwrap().starts_with("interp "));
        let interp: Vec<u32> = lines
            .last()
            .unwrap()
            .split(' ')
            .skip(1)
            .map(|w| w.parse().unwrap())
            .collect();
        assert_eq!(interp, planned.expected);
    }

    #[test]
    fn writing_twice_keeps_unchanged_files_and_drops_stale_data() {
        let dir = std::env::temp_dir().join(format!("difftest-runsrc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let plan = plan_of(1);
        let root = Path::new("/r");
        write_run_crate(&dir, "difftest-case-t", &plan, root).unwrap();
        let lib = dir.join("src").join("lib.rs");
        let before = std::fs::metadata(&lib).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));
        write_run_crate(&dir, "difftest-case-t", &plan, root).unwrap();
        assert_eq!(std::fs::metadata(&lib).unwrap().modified().unwrap(), before);
        assert_eq!(
            std::fs::read_dir(dir.join("data")).unwrap().count(),
            plan.len()
        );
        write_run_crate(&dir, "difftest-case-t", &plan[..3], root).unwrap();
        assert_eq!(std::fs::read_dir(dir.join("data")).unwrap().count(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

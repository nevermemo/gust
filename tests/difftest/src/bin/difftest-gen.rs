//! `difftest-gen`: print or write what the generator decodes. The batch runner is `difftest-run`.
//!
//! ```text
//! difftest-gen seeds FIRST LAST        the seeds.txt text for the seeds FIRST..=LAST
//! difftest-gen show SEED [INDEX]       the Rust text of the in-domain programs of SEED (or only program INDEX)
//! difftest-gen emit SEED DIR [ROOT]    write the generated crate of SEED into DIR (ROOT: repository root, default ../..)
//! difftest-gen emit-run SEEDS DIR [ROOT]  write the run crate (kernels, test module, data) of SEEDS (e.g. 1-2) into DIR
//! difftest-gen time SEEDS              generation and interpreter time of SEEDS (milliseconds)
//! difftest-gen stats SEEDS             attempts and discards by class of SEEDS, the vector part apart
//! difftest-gen forms                   the vector and matrix operator names the generator can write, one per line
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use difftest::corpus::corpus_text;
use difftest::emit::{emit_module, package_name, write_crate};
use difftest::generator::generate_batch;
use difftest::harness::parse_seeds;
use difftest::runsrc::{plan_seed, write_run_crate};

fn number(arg: Option<&String>, name: &str) -> Result<u64, String> {
    arg.ok_or_else(|| format!("missing {name}"))?
        .parse()
        .map_err(|e| format!("{name}: {e}"))
}

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("seeds") => {
            let (first, last) = (number(args.get(1), "FIRST")?, number(args.get(2), "LAST")?);
            print!("{}", corpus_text(first, last)?);
        }
        Some("show") => {
            let batch = generate_batch(number(args.get(1), "SEED")?)?;
            let only = args
                .get(2)
                .map(|_| number(args.get(2), "INDEX"))
                .transpose()?;
            for (index, program) in batch.programs.iter().enumerate() {
                if only.is_none_or(|wanted| wanted == index as u64) {
                    println!("// program {index}\n{}", emit_module(program));
                }
            }
        }
        Some("emit") => {
            let batch = generate_batch(number(args.get(1), "SEED")?)?;
            let dir = PathBuf::from(args.get(2).ok_or("missing DIR")?);
            let root = PathBuf::from(args.get(3).map_or("../..", String::as_str));
            let root =
                std::fs::canonicalize(&root).map_err(|e| format!("{}: {e}", root.display()))?;
            let package = package_name(&format!("seed-{}", batch.seed))?;
            write_crate(&dir, &package, &batch.programs, &root).map_err(|e| e.to_string())?;
            println!(
                "wrote {} programs to {}",
                batch.programs.len(),
                dir.display()
            );
        }
        Some("emit-run") => {
            let seeds = parse_seeds(args.get(1).ok_or("missing SEEDS")?)?;
            let dir = PathBuf::from(args.get(2).ok_or("missing DIR")?);
            let root = PathBuf::from(args.get(3).map_or("../..", String::as_str));
            let root =
                std::fs::canonicalize(&root).map_err(|e| format!("{}: {e}", root.display()))?;
            let mut plan = Vec::new();
            for seed in &seeds {
                plan.extend(plan_seed(*seed)?);
            }
            let package = package_name("run-probe")?;
            write_run_crate(&dir, &package, &plan, &root).map_err(|e| e.to_string())?;
            println!("wrote {} programs to {}", plan.len(), dir.display());
        }
        Some("stats") => {
            let seeds = parse_seeds(args.get(1).ok_or("missing SEEDS")?)?;
            let mut totals: std::collections::BTreeMap<String, u64> =
                std::collections::BTreeMap::new();
            let mut add = |key: String, n: u64| *totals.entry(key).or_default() += n;
            for seed in &seeds {
                let batch = generate_batch(*seed)?;
                add("programs".into(), batch.programs.len() as u64);
                add("attempts".into(), batch.attempts);
                add("vector-attempts".into(), batch.vector_attempts);
                for (reason, n) in &batch.discarded {
                    add(format!("discard {reason}"), u64::from(*n));
                }
                for (reason, n) in &batch.vector_discarded {
                    add(format!("vector-discard {reason}"), u64::from(*n));
                }
                let aggregate = batch.programs.iter().filter(|p| p.has_aggregate()).count();
                add("programs-with-an-aggregate".into(), aggregate as u64);
                for program in &batch.programs {
                    for (name, n) in difftest::ast::operator_counts(program) {
                        if name.contains("Vec")
                            || name.contains("Mat")
                            || name == difftest::ast::ASSIGN_COMP
                        {
                            add(format!("operator {name}"), n as u64);
                        }
                    }
                }
            }
            for (key, n) in totals {
                println!("{key}: {n}");
            }
        }
        Some("forms") => {
            for form in difftest::ast::writable_aggregate_forms() {
                println!("{form}");
            }
        }
        Some("time") => {
            let seeds = parse_seeds(args.get(1).ok_or("missing SEEDS")?)?;
            let (mut generate, mut interpret, mut programs) = (0.0, 0.0, 0);
            for seed in &seeds {
                let start = std::time::Instant::now();
                let batch = generate_batch(*seed)?;
                generate += start.elapsed().as_secs_f64() * 1e3;
                for program in &batch.programs {
                    let start = std::time::Instant::now();
                    difftest::interp::run_full(program).map_err(|f| format!("{f:?}"))?;
                    interpret += start.elapsed().as_secs_f64() * 1e3;
                    programs += 1;
                }
            }
            println!(
                "seeds {} programs {programs}: generation {generate:.1} ms in all (rejected attempts included), interpreter {interpret:.1} ms in all, {:.2} ms per program",
                seeds.len(),
                interpret / f64::from(programs)
            );
        }
        _ => {
            return Err(concat!(
                "usage: difftest-gen seeds FIRST LAST | show SEED [INDEX] | emit SEED DIR [ROOT]",
                " | emit-run SEEDS DIR [ROOT] | time SEEDS | stats SEEDS | forms"
            )
            .into());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("difftest-gen: {message}");
            ExitCode::FAILURE
        }
    }
}

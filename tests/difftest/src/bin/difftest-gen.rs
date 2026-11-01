//! `difftest-gen`: print or write what the generator decodes. Not the runner (that is m5.c1b).
//!
//! ```text
//! difftest-gen seeds FIRST LAST        the seeds.txt text for the seeds FIRST..=LAST
//! difftest-gen show SEED [INDEX]       the Rust text of the in-domain programs of SEED (or only program INDEX)
//! difftest-gen emit SEED DIR [ROOT]    write the generated crate of SEED into DIR (ROOT: repository root, default ../..)
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use difftest::corpus::corpus_text;
use difftest::emit::{emit_module, package_name, write_crate};
use difftest::generator::generate_batch;

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
        _ => {
            return Err(
                "usage: difftest-gen seeds FIRST LAST | show SEED [INDEX] | emit SEED DIR [ROOT]"
                    .into(),
            );
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

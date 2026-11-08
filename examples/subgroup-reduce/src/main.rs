use std::process::ExitCode;

fn main() -> ExitCode {
    match subgroup_reduce::run() {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("subgroup-reduce: FAILED: {error}");
            ExitCode::FAILURE
        }
    }
}

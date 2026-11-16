//! Builds the rust-gpu shader crate to one multi-entry SPIR-V module and writes it, with the entry-point list, into an
//! output directory. Usage: `builder <shader-crate-dir> <out-dir>`; run on the pinned nightly with
//! `RUSTUP_TOOLCHAIN=nightly-2026-08-15` in the environment (bench/README.md has the full command). The manifest
//! (`rustgpu.manifest.json`, with the sha256) is written by the caller from this output, not here.
use spirv_builder::SpirvBuilder;
use std::{fs, path::PathBuf};

fn main() {
    let mut args = std::env::args().skip(1);
    let shader = PathBuf::from(args.next().expect("shader crate dir"));
    let out_dir = PathBuf::from(args.next().expect("out dir"));
    let result = SpirvBuilder::new(&shader, "spirv-unknown-vulkan1.1")
        .build()
        .expect("rust-gpu build");
    let module = result.module.unwrap_single().to_owned();
    let bytes = fs::read(&module).expect("read module");
    fs::create_dir_all(&out_dir).expect("out dir");
    fs::write(out_dir.join("rustgpu.spv"), &bytes).expect("write module");
    let mut entry_points = result.entry_points;
    entry_points.sort();
    println!("module bytes {}", bytes.len());
    println!("entry_points {}", entry_points.join(","));
}

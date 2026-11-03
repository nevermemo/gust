// Declares `cfg(gust_cpu_profile)`, set when this crate is built for the production CPU target (`GUST_PROFILE=cpu`).
// The map functions of `src/map_case.rs` exist only then: under any other profile the host functions that `#[gpu]`
// generates for a function (`map`, `map_gpu`, ..) name `gust-wgpu`, which this crate does not depend on.

fn main() {
    println!("cargo:rerun-if-env-changed=GUST_PROFILE");
    println!("cargo:rustc-check-cfg=cfg(gust_cpu_profile)");
    if std::env::var("GUST_PROFILE").as_deref() == Ok("cpu") {
        println!("cargo:rustc-cfg=gust_cpu_profile");
    }
}

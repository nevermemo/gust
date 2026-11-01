// The `slang-runtime` feature of `gust` under `GUST_PROFILE=native`. A strict profile builds no run-time Slang
// bridge, so `gust`'s build script refuses the feature (D44). It does so whoever turns the feature on; here the
// manifest does, as a dependency of a production crate would.

fn main() {}

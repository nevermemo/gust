//! The example `batch-basics`: see [`app`].
//!
//! A build block may also hold the library ops of `gust-algo` (`let total = gust::sum(&blocks);`,
//! `let (idx, n) = gust::compact(&flags);`): `tests/reduce.rs` shows them. Reductions in a batch are native-only: the
//! reduce, scan and compact kernels ship no `gust-artifacts`, so only `gust-wgpu`'s `slang-runtime` feature builds
//! them, and on the `web` profile (wasm32) the build block returns the "run-time compile unavailable" error.

pub mod app;

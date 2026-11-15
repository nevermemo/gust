//! The crate that owns the dependency on `gust-compute` and hands it on: a crate that depends on this one reaches the
//! facade as `inner::gust`, with no direct dependency of its own.

pub use gust;

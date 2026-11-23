//! Slang reflection for the restricted storage and constant-buffer ABIs.
//! Native compilation returns the artifact and complete layout from one linked program.
//! CLI JSON remains an explicit partial-evidence compatibility path.
//! Only global structured-buffer resources (and, under schema 2, constant buffers) are supported. Entry-point
//! parameters must be the generated `SV_DispatchThreadID` system value, not resources/uniforms.
//!
//! # Schema versions of the native reflection JSON
//!
//! The reader accepts `schemaVersion` 1, 2, 3 and 4. Schema 1 describes exactly what it always did: 32-bit scalars and
//! structs of them, in structured buffers. Schema 2 adds, beside those, the type kinds `vector`
//! (`scalarType`, `elementCount`) and `matrix` (`scalarType`, `rows`, `columns`, `matrixLayout`; its `stride` is the
//! column stride), the parameter type `constantBuffer` (`elementType`, `size`), and a mandatory top-level
//! `matrixLayout` that must equal `MATRIX_LAYOUT` (column-major). Every type carries `size`, `alignment` and
//! `stride`. A schema-1 document checked against a descriptor with a vector, a matrix or a uniform is refused with
//! the instruction to re-export it ([`Reflection::cross_check_kernel`]); nothing committed before schema 2 needs
//! re-exporting otherwise.
//!
//! Schema 3 (mx.t0b, decision D91) is schema 2's grammar (its `matrixLayout` included) plus, at the top level, a
//! `stage` (`compute` or `fragment`) and, as parameters, the type kinds `texture` (`shape`, `access`, `result`,
//! `format`) and `sampler`; a fragment document has `targets` (`index`, `type`, in `SV_Target` order) and no
//! `workgroupSize`. The helper writes it only when a 2-D texture, a sampler or a non-compute stage is present, so
//! every document of a compute kernel over buffers stays schema 1 or 2, byte for byte, and the reader reads those
//! exactly as before (a texture in a schema-1 or schema-2 document is refused with the old message). A document of
//! schema 1 or 2 describes a compute entry point: [`Reflection::stage`] is [`Stage::Compute`] for it.
//!
//! Schema 4 (m9.r2h) is schema 3's grammar plus one type kind, `array`: `elementCount` (at least 1), `elementStride`
//! (the bytes between two elements) and `elementType` (any type, an array or a struct included), beside the `size`,
//! `alignment` and `stride` every type has. The helper writes it only when an array occurs in a buffer element, so every
//! other document keeps its schema and its bytes, and the reader refuses an `array` in a document of schema 3 or below
//! with the old message. It is read into [`ReflectedTypeKind::Array`], and a host
//! [`TypeLayoutKind::Array`](crate::TypeLayoutKind::Array) is checked against it (m9.r2b): the count, the element
//! stride and the element type, recursively, beside the size, alignment and
//! stride of the array itself.
//!
//! # Where the CLI gate would sit
//!
//! The plan's section 3.3 had a second witness for the facts the helper could not see: slangc's
//! `-reflection-json`, run on the same compilation, parsed into `CliFacts` and stored in the record. X-S2 ended in
//! outcome A (mx.t0a): the helper reports the texture format, the access and a fragment's targets itself, so the
//! helper's facts are the D17 witness and that gate is **not built**. `reflect` (the CLI path) stays what it was: a
//! compute-only partial-evidence path for tests, and an optional agreement check against the helper is later work.
//! Were the helper to stop seeing a fact, the gate belongs next to `CompiledReflection::from_output`, with its
//! comparison in [`Reflection::cross_check_kernel`] beside the texture checks.
//!
//! # The `slang-runtime` feature
//!
//! The functions that launch a compiler process (`reflect`, `compile_reflected`, `compile_reflected_with_helper`,
//! `native_compiler_path`) exist only with the `slang-runtime` feature, which is off by default (D44). Parsing
//! and checking reflection JSON (`Reflection::from_json`, `cross_check_*`, `CompiledReflection::from_output`) needs
//! no feature, so [`crate::artifact::verify_artifact`] works without a compiler.

mod check;
mod compile;
mod error;
mod json;
mod model;
mod pod;
mod read;
#[cfg(test)]
mod tests;

#[cfg(feature = "slang-runtime")]
pub use compile::{
    CleanCompilation, compile_reflected, compile_reflected_clean, compile_reflected_with_helper,
    native_compiler_path, reflect,
};
pub use error::Error;
pub use model::{
    CompiledReflection, PodCrossCheck, ReflectedField, ReflectedResource, ReflectedResourceKind,
    ReflectedSampler, ReflectedTarget, ReflectedTexture, ReflectedType, ReflectedTypeKind,
    Reflection, Stage,
};
pub use pod::cross_check_pod;

//! The AST of the generated kernels: integers, booleans, `f32` (from DIFFTEST_FORMAT 2) and, for the vector programs
//! (DIFFTEST_FORMAT 6, slice m6.d), the `gust::math` vectors and matrices.
//!
//! Every scalar value is a 32-bit pattern (`u32`); `bool` is 0 or 1. Nodes that the interpreter and the emitter need to
//! type carry their operand type, so `Expr::ty` is O(1). `Program::check` verifies the invariants the generator
//! promises, and the unit tests run it over every generated program.
//!
//! The vector types never reach a buffer: inputs and the output are scalar, so the result file, the classifier and the
//! shrinker see scalars only. A vector exists as a local or an intermediate, is built from scalars (`Vec4::new`,
//! `splat`, `Mat4::from_cols`) and is read back through a component, a `dot`, or a swizzle that is read in turn. The
//! aggregate operators have their own nodes (`AggBin`, `AggNeg`, `Dot`, ...), so every pattern that matches `Bin` or
//! `Un` still sees scalars only.

use std::collections::{BTreeMap, BTreeSet};

mod check;
mod expr;
mod methods;
mod nodes;
mod program;
mod shapes;
mod shapes_more;
mod types;
mod walk;

pub use check::*;
pub use expr::*;
pub use nodes::*;
pub use program::*;
pub use shapes::*;
pub use shapes_more::*;
pub use types::*;
pub use walk::*;

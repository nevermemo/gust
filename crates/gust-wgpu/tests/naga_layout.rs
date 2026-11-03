//! The naga layout gate (roadmap MX, slice mx.n; M6 close ruling 4).
//!
//! Every WGSL module that a committed golden produces is parsed and validated by the naga that wgpu itself runs
//! (`wgpu::naga`, the lockfile's version), and the size, alignment and field offsets that naga's `proc::Layouter`
//! computes for each storage and uniform binding's element type are compared with the layout the descriptor carries
//! (`ParameterDescriptor::resource_layout`, the `GpuPod::LAYOUT` the runtime checks every buffer against). A
//! disagreement between the macro's computed offsets and the layout of the WGSL that slangc emits is a defect in one
//! of them; this gate finds it for every kernel of every golden, not only for the layouts a GPU test reads back.
//!
//! The WGSL set (the test names its size and fails on drift):
//!
//! - the 17 Slang goldens of `tests/fixtures/*.slang`, which are all of the goldens (the test
//!   `the_goldens_are_the_fixtures_directory` pins the file list against the directory, and the fixture `.rs` list
//!   with it). Each golden is split at its `// ==== kernel <name> ====` markers where it has them, and each
//!   `[shader("compute")]` entry of each section is compiled to WGSL with the `slangc` command line through
//!   `gust::slang::compile_source`: 85 entries. The descriptor of an entry comes from the fixture's Rust source,
//!   included below the way the other test files include it;
//! - every fixture kernel, 95 in all (the 85 above and the 10 that no golden holds: `atomics.slang` keeps only the
//!   first of the ten kernels of `atomics.rs`, and `numeric.slang` only the first of the two of `numeric.rs`),
//!   compiled from its descriptor's own Slang through the native reflection helper, `gust::reflect::compile_reflected`.
//!   That is the route `HeadlessDevice` takes at run time, so these are the bytes a device is handed when it compiles
//!   at run time. `export` takes `compile_reflected_clean` (the same helper on input without `#line` directives), so
//!   exported WGSL is gated only through the committed precompiled file below. The helper needs no adapter. 180 WGSL
//!   modules in all;
//! - the committed precompiled WGSL (`tests/fixtures/precompiled/52e85c5f03f72272.wgsl`), which is the only WGSL file
//!   the repository commits; its five other copies are required to be byte-identical to it.
//!
//! Validation matches `wgpu_core::Device::create_shader_module`: the WGSL front end is given the capability set, then
//! the validator runs with `ValidationFlags::all()` and the same capability set. The set here is
//! `Capabilities::empty()`, which is what wgpu derives from a device that requests no shader features. The devices of
//! `HeadlessDevice` request `TIMESTAMP_QUERY` and `MEMORY_DECORATION_COHERENT` when the adapter has them, and neither
//! maps to a naga capability the goldens use; wgpu also adds the capabilities of three downlevel flags (f16 in f32,
//! multisampled shading, cube array textures), which no compute golden touches. So the gate is at least as strict as
//! any device of the crate. It must never be `Capabilities::all()`, the default of `naga::front::wgsl::Options`: that
//! passed WGSL that wgpu refuses (docs/development/VALIDATION.md); `default_capabilities_refuse_what_all_accept`
//! pins the difference.
//!
//! The test needs `slangc` on `PATH` and the native reflection helper (`GUST_SLANG_REFLECT`, or the repository's
//! `target/slang-reflect`), both of the Slangc tier, and no adapter. It fails, never skips, without either, and the
//! message names the tool. The `slangc` version is not checked here; the CI job that runs the test installs the
//! version the repository pins.

use std::collections::{BTreeMap, BTreeSet};

use gust::{
    Access, KernelDescriptor, ParameterDescriptor, ParameterKind, ScalarKind, ShaderLayoutRules,
    TypeLayout, TypeLayoutKind, reflect,
    slang::{self, Target},
};
use wgpu::naga::{
    self, AddressSpace, ArraySize, GlobalVariable, Handle, Module, StorageAccess, Type, TypeInner,
    front::wgsl,
    proc::Layouter,
    valid::{Capabilities, ValidationFlags, Validator},
};

// The Rust sources of the goldens. Each is included into a module of its own (the test files that run them define one
// module per file); `KERNELS` is the descriptor list of the fixture's `#[gpu]` module. The two lints are the ones the
// fixtures trip on as Rust: `coretests.rs` compares against the type's minimum on purpose (its test file allows the
// same lint), and `vectors.rs` writes `w = w * s` on purpose (its golden pins that spelling beside `w *= v`).
macro_rules! fixture {
    ($name:ident, $module:ident, $path:literal) => {
        #[allow(clippy::absurd_extreme_comparisons, clippy::assign_op_pattern)]
        mod $name {
            include!($path);
            pub const KERNELS: &[gust::KernelDescriptor] = $module::MODULE_DESCRIPTOR.kernels;
        }
    };
}

// `map.rs` holds five `#[gpu]` functions (m7.a1), so it is five modules of one kernel each and no single
// `MODULE_DESCRIPTOR`: this second form names the functions and joins their descriptors in file order, which is the
// order of the sections of `map.slang`. `Vec2` is the one name the functions use from the prelude of the file.
macro_rules! fixture_functions {
    ($name:ident, $path:literal, [$($function:ident),+]) => {
        #[allow(dead_code)]
        mod $name {
            use gust::math::Vec2;
            include!($path);
            $(const _: () = assert!($function::MODULE_DESCRIPTOR.kernels.len() == 1);)+
            pub const KERNELS: &[gust::KernelDescriptor] = &[$($function::MODULE_DESCRIPTOR.kernels[0]),+];
        }
    };
}

fixture!(f_atomics, atomics, "../../../tests/fixtures/atomics.rs");
fixture!(f_compact, compact, "../../../tests/fixtures/compact.rs");
fixture!(
    f_coretests,
    coretests,
    "../../../tests/fixtures/coretests.rs"
);
fixture!(f_grid, grid, "../../../tests/fixtures/grid.rs");
fixture!(f_loops, loops, "../../../tests/fixtures/loops.rs");
fixture_functions!(
    f_map,
    "../../../tests/fixtures/map.rs",
    [saxpy, clamp_signed, even_and, mirror, add_one]
);
fixture!(f_mismatch, mismatch, "../../../tests/fixtures/mismatch.rs");
fixture!(f_numeric, numeric, "../../../tests/fixtures/numeric.rs");
fixture!(f_option, optional, "../../../tests/fixtures/option.rs");
fixture!(f_products, products, "../../../tests/fixtures/products.rs");
fixture!(f_reduce, reduce, "../../../tests/fixtures/reduce.rs");
fixture!(f_result, results, "../../../tests/fixtures/result.rs");
fixture!(f_scan, scan, "../../../tests/fixtures/scan.rs");
fixture!(
    f_semantics,
    semantics,
    "../../../tests/fixtures/semantics.rs"
);
fixture!(f_uniforms, uniforms, "../../../tests/fixtures/uniforms.rs");
fixture!(f_vectors, vectors, "../../../tests/fixtures/vectors.rs");
fixture!(
    f_workgroup,
    workgroup,
    "../../../tests/fixtures/workgroup.rs"
);

/// Every golden: its file name under `tests/fixtures`, its text, the number of compute entries it holds (pinned, so a
/// golden that gains a kernel is a deliberate edit here) and the descriptors of its fixture.
struct Golden {
    file: &'static str,
    text: &'static str,
    entries: usize,
    kernels: &'static [KernelDescriptor],
}

fn goldens() -> [Golden; 17] {
    [
        Golden {
            file: "atomics.slang",
            text: include_str!("../../../tests/fixtures/atomics.slang"),
            entries: 1,
            kernels: f_atomics::KERNELS,
        },
        Golden {
            file: "compact.slang",
            text: include_str!("../../../tests/fixtures/compact.slang"),
            entries: 5,
            kernels: f_compact::KERNELS,
        },
        Golden {
            file: "coretests.slang",
            text: include_str!("../../../tests/fixtures/coretests.slang"),
            entries: 5,
            kernels: f_coretests::KERNELS,
        },
        Golden {
            file: "grid.slang",
            text: include_str!("../../../tests/fixtures/grid.slang"),
            entries: 3,
            kernels: f_grid::KERNELS,
        },
        Golden {
            file: "loops.slang",
            text: include_str!("../../../tests/fixtures/loops.slang"),
            entries: 1,
            kernels: f_loops::KERNELS,
        },
        Golden {
            file: "map.slang",
            text: include_str!("../../../tests/fixtures/map.slang"),
            entries: 5,
            kernels: f_map::KERNELS,
        },
        Golden {
            file: "mismatch.slang",
            text: include_str!("../../../tests/fixtures/mismatch.slang"),
            entries: 15,
            kernels: f_mismatch::KERNELS,
        },
        Golden {
            file: "numeric.slang",
            text: include_str!("../../../tests/fixtures/numeric.slang"),
            entries: 1,
            kernels: f_numeric::KERNELS,
        },
        Golden {
            file: "option.slang",
            text: include_str!("../../../tests/fixtures/option.slang"),
            entries: 1,
            kernels: f_option::KERNELS,
        },
        Golden {
            file: "products.slang",
            text: include_str!("../../../tests/fixtures/products.slang"),
            entries: 1,
            kernels: f_products::KERNELS,
        },
        Golden {
            file: "reduce.slang",
            text: include_str!("../../../tests/fixtures/reduce.slang"),
            entries: 30,
            kernels: f_reduce::KERNELS,
        },
        Golden {
            file: "result.slang",
            text: include_str!("../../../tests/fixtures/result.slang"),
            entries: 1,
            kernels: f_result::KERNELS,
        },
        Golden {
            file: "scan.slang",
            text: include_str!("../../../tests/fixtures/scan.slang"),
            entries: 11,
            kernels: f_scan::KERNELS,
        },
        Golden {
            file: "semantics.slang",
            text: include_str!("../../../tests/fixtures/semantics.slang"),
            entries: 1,
            kernels: f_semantics::KERNELS,
        },
        Golden {
            file: "uniforms.slang",
            text: include_str!("../../../tests/fixtures/uniforms.slang"),
            entries: 1,
            kernels: f_uniforms::KERNELS,
        },
        Golden {
            file: "vectors.slang",
            text: include_str!("../../../tests/fixtures/vectors.slang"),
            entries: 1,
            kernels: f_vectors::KERNELS,
        },
        Golden {
            file: "workgroup.slang",
            text: include_str!("../../../tests/fixtures/workgroup.slang"),
            entries: 2,
            kernels: f_workgroup::KERNELS,
        },
    ]
}

/// Goldens whose WGSL naga refuses today, by compute entry, each with the reason. Empty: naga accepts every golden.
/// An entry here is a finding, not a skip: `allow_listed_entries_still_fail` requires each one to still be refused,
/// so the list cannot go stale.
const REFUSED_BY_NAGA: &[(&str, &str)] = &[];

/// The capability set of the gate (see the module documentation).
const DEFAULT_CAPABILITIES: Capabilities = Capabilities::empty();

/// Parses and validates `source` as wgpu's `create_shader_module` does, with `capabilities` for both steps.
fn parse_and_validate(source: &str, capabilities: Capabilities) -> Result<Module, String> {
    let mut options = wgsl::Options::new();
    options.capabilities = capabilities;
    let module = wgsl::Frontend::new_with_options(options)
        .parse(source)
        .map_err(|error| format!("parse: {}", error.emit_to_string(source)))?;
    Validator::new(ValidationFlags::all(), capabilities)
        .validate(&module)
        .map_err(|error| format!("validate: {}", error.emit_to_string(source)))?;
    Ok(module)
}

/// The compute entry points of a Slang text: the name after each `[shader("compute")]` line, from the next line that
/// starts with `void`.
fn compute_entries(slang: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut pending = false;
    for line in slang.lines().map(str::trim) {
        if line == "[shader(\"compute\")]" {
            pending = true;
        } else if pending && let Some(rest) = line.strip_prefix("void ") {
            if let Some((name, _)) = rest.split_once('(') {
                entries.push(name.trim().to_owned());
            }
            pending = false;
        }
    }
    entries
}

/// The Slang modules of a golden: one per `// ==== kernel <name> ====` section, or the whole text when it has no
/// marker (the layout of `coretests`, `mismatch` and `reduce` against the single-kernel files).
fn sections(golden: &str) -> Vec<String> {
    let text = golden.replace("\r\n", "\n");
    if !text.contains("// ==== kernel ") {
        return vec![text];
    }
    text.split("// ==== kernel ")
        .skip(1)
        .map(|section| {
            let (_, body) = section
                .split_once(" ====\n")
                .expect("a kernel marker ends with ` ====`");
            body.to_owned()
        })
        .collect()
}

/// A name in the WGSL against the descriptor's: slangc appends `_<digits>` to a name it renames.
fn same_name(wgsl: &str, descriptor: &str) -> bool {
    wgsl.strip_prefix(descriptor).is_some_and(|rest| {
        rest.is_empty()
            || rest.strip_prefix('_').is_some_and(|digits| {
                !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
            })
    })
}

fn scalar_kind(scalar: naga::Scalar) -> Option<ScalarKind> {
    match (scalar.kind, scalar.width) {
        (naga::ScalarKind::Uint, 4) => Some(ScalarKind::U32),
        (naga::ScalarKind::Sint, 4) => Some(ScalarKind::I32),
        (naga::ScalarKind::Float, 4) => Some(ScalarKind::F32),
        _ => None,
    }
}

/// What a comparison of one kernel carries: the module, its layouts, which rules apply and where findings go.
struct Comparison<'a> {
    kernel: &'a str,
    module: &'a Module,
    layouter: &'a Layouter,
    /// Uniform address space: WGSL requires a struct's alignment, and so its size, to be a multiple of 16 there
    /// (`RequiredAlignOf(S, uniform)`), which `Layouter` does not apply and the descriptor's `Std140V1` layout does.
    uniform: bool,
    problems: &'a mut Vec<String>,
}

impl Comparison<'_> {
    fn problem(&mut self, path: &str, what: &str, descriptor: impl ToString, wgsl: impl ToString) {
        self.problems.push(format!(
            "{}: {path}: {what}: descriptor {}, WGSL {}",
            self.kernel,
            descriptor.to_string(),
            wgsl.to_string()
        ));
    }

    /// Compares the descriptor's layout of a type with naga's layout of `handle`, recursively through struct fields.
    fn compare_type(&mut self, path: &str, want: &TypeLayout, handle: Handle<Type>) {
        let layout = self.layouter[handle];
        let inner = &self.module.types[handle].inner;
        // `Alignment * u32` is naga's public way to the number (`Alignment` has no getter): `* 1` is the alignment.
        let mut alignment = layout.alignment * 1;
        let mut size = layout.size;
        if self.uniform && matches!(inner, TypeInner::Struct { .. }) {
            alignment = alignment.max(16);
            size = size.next_multiple_of(alignment);
        }
        if want.size != size {
            self.problem(path, "size", want.size, size);
        }
        if want.alignment != alignment {
            self.problem(path, "alignment", want.alignment, alignment);
        }
        match (&want.kind, inner) {
            (
                TypeLayoutKind::Scalar(kind),
                TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar),
            ) => {
                if scalar_kind(*scalar) != Some(*kind) {
                    self.problem(
                        path,
                        "scalar type",
                        format!("{kind:?}"),
                        format!("{scalar:?}"),
                    );
                }
            }
            (
                TypeLayoutKind::Vector { scalar, count },
                TypeInner::Vector {
                    size,
                    scalar: wgsl_scalar,
                },
            ) => {
                if scalar_kind(*wgsl_scalar) != Some(*scalar) || *size as u32 != *count {
                    self.problem(
                        path,
                        "vector type",
                        format!("{count} x {scalar:?}"),
                        format!("{} x {wgsl_scalar:?}", *size as u32),
                    );
                }
            }
            (
                TypeLayoutKind::Matrix {
                    scalar,
                    rows,
                    cols,
                    column_stride,
                },
                TypeInner::Matrix {
                    columns,
                    rows: wgsl_rows,
                    scalar: wgsl_scalar,
                },
            ) => {
                if scalar_kind(*wgsl_scalar) != Some(*scalar)
                    || *wgsl_rows as u32 != *rows
                    || *columns as u32 != *cols
                {
                    self.problem(
                        path,
                        "matrix type",
                        format!("{rows}x{cols} {scalar:?}"),
                        format!("{}x{} {wgsl_scalar:?}", *wgsl_rows as u32, *columns as u32),
                    );
                }
                let wgsl_stride = layout.size / (*columns as u32);
                if *column_stride != wgsl_stride {
                    self.problem(path, "column stride", *column_stride, wgsl_stride);
                }
            }
            // In the uniform address space slangc spells a matrix as a struct of one array of column vectors
            // (`struct Matrix_0 { data_0 : array<vec4<f32>, 4> }`), the layout WGSL's own matrix type has there.
            (
                TypeLayoutKind::Matrix {
                    scalar,
                    rows,
                    cols,
                    column_stride,
                },
                TypeInner::Struct { members, .. },
            ) if self.uniform => {
                // A `let` chain in an `if`, not an `if let` match guard: the guard form needs a newer Rust than the
                // workspace's `rust-version`.
                let columns = match members.as_slice() {
                    [member] if member.offset == 0 => {
                        if let TypeInner::Array {
                            base,
                            size: ArraySize::Constant(length),
                            stride,
                        } = self.module.types[member.ty].inner
                            && let TypeInner::Vector {
                                size: column,
                                scalar: wgsl_scalar,
                            } = self.module.types[base].inner
                        {
                            Some((length.get(), stride, column as u32, wgsl_scalar))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                match columns {
                    Some((length, stride, column, wgsl_scalar))
                        if length == *cols
                            && stride == *column_stride
                            && column == *rows
                            && scalar_kind(wgsl_scalar) == Some(*scalar) => {}
                    _ => self.problem(
                        path,
                        "matrix as array of columns",
                        format!("{rows}x{cols} {scalar:?}, column stride {column_stride}"),
                        format!("{members:?}"),
                    ),
                }
            }
            (TypeLayoutKind::Struct(fields), TypeInner::Struct { members, .. }) => {
                if fields.len() != members.len() {
                    self.problem(path, "field count", fields.len(), members.len());
                    return;
                }
                for (field, member) in fields.iter().zip(members) {
                    let wgsl_name = member.name.as_deref().unwrap_or("<unnamed>");
                    let field_path = format!("{path}.{}", field.name);
                    if !same_name(wgsl_name, field.name) {
                        self.problem(&field_path, "field name", field.name, wgsl_name);
                    }
                    if field.offset != member.offset {
                        self.problem(&field_path, "offset", field.offset, member.offset);
                    }
                    self.compare_type(&field_path, &field.ty, member.ty);
                }
            }
            (want_kind, found) => {
                self.problem(
                    path,
                    "type kind",
                    format!("{want_kind:?}"),
                    format!("{found:?}"),
                );
            }
        }
    }

    /// One storage or uniform parameter against its WGSL global.
    fn compare_binding(&mut self, parameter: &ParameterDescriptor, var: &GlobalVariable) {
        let name = parameter.name;
        let Some(layout) = parameter.resource_layout else {
            self.problem(name, "resource layout", "none", "a binding");
            return;
        };
        match (parameter.kind, &var.space) {
            (ParameterKind::Uniform, AddressSpace::Uniform) => {
                if layout.rules != ShaderLayoutRules::Std140V1 {
                    self.problem(
                        name,
                        "layout rules",
                        format!("{:?}", layout.rules),
                        "uniform",
                    );
                }
                self.uniform = true;
                self.compare_type(name, &layout.element, var.ty);
                if layout.element_stride != layout.element.size {
                    self.problem(
                        name,
                        "element stride of a constant buffer",
                        layout.element_stride,
                        layout.element.size,
                    );
                }
            }
            (ParameterKind::Storage, AddressSpace::Storage { access }) => {
                if layout.rules == ShaderLayoutRules::Std140V1 {
                    self.problem(
                        name,
                        "layout rules",
                        format!("{:?}", layout.rules),
                        "storage",
                    );
                }
                let wanted = match parameter.access {
                    Access::ReadOnly => StorageAccess::LOAD,
                    _ => StorageAccess::LOAD | StorageAccess::STORE,
                };
                let found = *access & (StorageAccess::LOAD | StorageAccess::STORE);
                if found != wanted {
                    self.problem(name, "access", format!("{wanted:?}"), format!("{found:?}"));
                }
                self.uniform = false;
                let TypeInner::Array { base, size, stride } = self.module.types[var.ty].inner
                else {
                    let found = format!("{:?}", self.module.types[var.ty].inner);
                    self.problem(name, "storage type", "a runtime array", found);
                    return;
                };
                if size != ArraySize::Dynamic {
                    self.problem(name, "array length", "runtime-sized", format!("{size:?}"));
                }
                if layout.element_stride != stride {
                    self.problem(name, "array stride", layout.element_stride, stride);
                }
                let own_stride = self.layouter[base].to_stride();
                if own_stride != stride {
                    self.problem(
                        name,
                        "array stride against the element's own",
                        own_stride,
                        stride,
                    );
                }
                self.compare_type(&format!("{name}[]"), &layout.element, base);
            }
            (kind, space) => {
                self.problem(
                    name,
                    "address space",
                    format!("{kind:?}"),
                    format!("{space:?}"),
                );
            }
        }
    }
}

/// Every finding of one kernel's WGSL `module` against its descriptor; empty when they agree.
fn compare_kernel(kernel: &KernelDescriptor, module: &Module) -> Vec<String> {
    let mut problems = Vec::new();
    let mut layouter = Layouter::default();
    if let Err(error) = layouter.update(module.to_ctx()) {
        return vec![format!("{}: Layouter: {error}", kernel.entry_point)];
    }
    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    if names != [kernel.entry_point] {
        problems.push(format!(
            "{}: entry points: descriptor [{}], WGSL {names:?}",
            kernel.entry_point, kernel.entry_point
        ));
    }
    let mut globals: BTreeMap<(u32, u32), &GlobalVariable> = module
        .global_variables
        .iter()
        .filter_map(|(_, var)| var.binding.map(|b| ((b.group, b.binding), var)))
        .collect();
    let mut comparison = Comparison {
        kernel: kernel.entry_point,
        module,
        layouter: &layouter,
        uniform: false,
        problems: &mut problems,
    };
    for parameter in kernel.parameters {
        let (ParameterKind::Storage | ParameterKind::Uniform) = parameter.kind else {
            continue;
        };
        let Some(binding) = parameter.binding else {
            comparison.problem(parameter.name, "binding", "none", "a resource");
            continue;
        };
        match globals.remove(&(binding.group, binding.binding)) {
            Some(var) => comparison.compare_binding(parameter, var),
            None => comparison.problem(
                parameter.name,
                "binding",
                format!("group {} binding {}", binding.group, binding.binding),
                "absent",
            ),
        }
    }
    for ((group, binding), var) in globals {
        let name = var.name.as_deref().unwrap_or("<unnamed>");
        comparison.problem(
            name,
            "binding",
            "absent",
            format!("group {group} binding {binding}"),
        );
    }
    problems
}

/// The compute entries the seventeen goldens hold in all (pinned per golden in [`goldens`]).
const GOLDEN_ENTRIES: usize = 85;

/// The fixture kernels that no golden holds: the macro's golden test keeps only the first kernel of `atomics.rs`
/// (`atomics.slang` has one entry for the ten kernels of the module) and of `numeric.rs` (one of two).
const FIXTURE_KERNELS_WITHOUT_GOLDEN: usize = 10;

/// Every kernel of every fixture: the ones with a golden entry and the ones without.
const FIXTURE_KERNELS: usize = GOLDEN_ENTRIES + FIXTURE_KERNELS_WITHOUT_GOLDEN;

/// One WGSL module: where it came from (a golden file for the `slangc` command line, `helper` for the native
/// reflection helper), its entry and text.
struct Case {
    origin: &'static str,
    entry: String,
    wgsl: String,
}

/// Fails with a message naming the missing tool, before any compile: the gate never skips. `slangc` is launched by
/// name, so a launch error is a missing `PATH` entry; the helper is the file `GUST_SLANG_REFLECT` names, or else the
/// repository's `target/slang-reflect`, or else a bare name on `PATH` (which only the compile itself can tell).
fn require_tools() {
    assert!(
        std::process::Command::new("slangc")
            .arg("-version")
            .output()
            .is_ok(),
        "the naga layout gate needs `slangc` on PATH (the Slangc tier; the CI job `slang` puts the \
         pinned install's bin directory there); it fails and never skips without it"
    );
    let helper = gust::reflect::native_compiler_path();
    assert!(
        helper.is_file() || helper.components().count() == 1,
        "the naga layout gate needs the native reflection helper, `gust-slang-reflect`: GUST_SLANG_REFLECT is {:?} and \
         names no file (build it with `cargo xtask build-slang-reflect` or point the variable at a built helper); it \
         fails and never skips without it",
        std::env::var_os("GUST_SLANG_REFLECT")
    );
}

/// Where a case's WGSL comes from.
enum Source {
    /// `slangc` on the text of one golden section.
    Slangc(String),
    /// The native reflection helper on a descriptor's own Slang.
    Helper(&'static KernelDescriptor),
}

struct Job {
    origin: &'static str,
    entry: String,
    source: Source,
}

impl Job {
    fn compile(&self) -> Result<String, String> {
        match &self.source {
            Source::Slangc(section) => slang::compile_source(section, &self.entry, Target::Wgsl)
                .map_err(|error| {
                    format!(
                        "{}: slangc could not compile `{}` (is `slangc` on PATH?): {error}",
                        self.origin, self.entry
                    )
                })
                .and_then(|bytes| {
                    String::from_utf8(bytes).map_err(|error| format!("{}: {error}", self.entry))
                }),
            Source::Helper(kernel) => reflect::compile_reflected(kernel, Target::Wgsl)
                .map_err(|error| {
                    format!(
                        "{}: the native reflection helper could not compile the kernel (is GUST_SLANG_REFLECT set to \
                         a built `gust-slang-reflect`?): {error}",
                        self.entry
                    )
                })
                .and_then(|compiled| {
                    String::from_utf8(compiled.artifact)
                        .map_err(|error| format!("{}: {error}", self.entry))
                }),
        }
    }
}

/// Runs every job on up to four threads (each compile is a process of its own in a directory of its own, and the
/// other test files already compile in parallel), in the order of `jobs`; every failure is reported, not the first.
fn compile_all(jobs: Vec<Job>) -> Vec<Case> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(4));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else { break };
                    let result = job.compile();
                    done.lock().expect("no job panics").push((index, result));
                }
            });
        }
    });
    let mut done = done.into_inner().expect("no job panics");
    done.sort_by_key(|(index, _)| *index);
    let failures: Vec<&str> = done
        .iter()
        .filter_map(|(_, result)| result.as_ref().err().map(String::as_str))
        .collect();
    assert!(
        failures.is_empty(),
        "{} compile(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    jobs.into_iter()
        .zip(done)
        .map(|(job, (_, result))| Case {
            origin: match job.source {
                Source::Slangc(_) => job.origin,
                Source::Helper(_) => "helper",
            },
            entry: job.entry,
            wgsl: result.expect("failures were reported above"),
        })
        .collect()
}

/// The WGSL of every golden entry (`slangc` on the golden's own text) and of every fixture kernel through the helper
/// (`compile_reflected` on its descriptor, the route `HeadlessDevice` compiles through).
fn all_cases() -> Vec<Case> {
    require_tools();
    let mut jobs = Vec::new();
    let mut in_a_golden = BTreeSet::new();
    for golden in goldens() {
        let mut entries = 0;
        for section in sections(golden.text) {
            for entry in compute_entries(&section) {
                in_a_golden.insert(entry.clone());
                jobs.push(Job {
                    origin: golden.file,
                    entry,
                    source: Source::Slangc(section.clone()),
                });
                entries += 1;
            }
        }
        assert_eq!(
            entries, golden.entries,
            "{}: the golden holds {entries} compute entries, the gate pins {}; a changed golden changes this number \
             in the same commit",
            golden.file, golden.entries
        );
    }
    assert_eq!(
        jobs.len(),
        GOLDEN_ENTRIES,
        "the compute entries of all goldens"
    );
    let mut without_golden = 0;
    for golden in goldens() {
        for kernel in golden.kernels {
            without_golden += usize::from(!in_a_golden.contains(kernel.entry_point));
            jobs.push(Job {
                origin: "helper",
                entry: kernel.entry_point.to_owned(),
                source: Source::Helper(kernel),
            });
        }
    }
    assert_eq!(
        without_golden, FIXTURE_KERNELS_WITHOUT_GOLDEN,
        "the fixture kernels that no golden holds"
    );
    assert_eq!(
        jobs.len() - GOLDEN_ENTRIES,
        FIXTURE_KERNELS,
        "the fixture kernels compiled through the helper"
    );
    compile_all(jobs)
}

/// The descriptor of a compute entry, from all fixtures (the entry names are unique across them).
fn descriptor_of(entry: &str) -> &'static KernelDescriptor {
    let mut found = goldens()
        .into_iter()
        .flat_map(|golden| golden.kernels.iter())
        .filter(|kernel| kernel.entry_point == entry);
    let kernel = found
        .next()
        .unwrap_or_else(|| panic!("no fixture kernel has the entry point `{entry}`"));
    assert!(
        found.next().is_none(),
        "two fixture kernels share the entry `{entry}`"
    );
    kernel
}

#[test]
fn the_goldens_are_the_fixtures_directory() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let on_disk: BTreeSet<String> = std::fs::read_dir(&directory)
        .expect("tests/fixtures is readable")
        .map(|entry| {
            entry
                .expect("a directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    let pinned: BTreeSet<String> = goldens()
        .iter()
        .flat_map(|golden| [golden.file.to_owned(), golden.file.replace(".slang", ".rs")])
        .collect();
    let in_scope = |set: &BTreeSet<String>| -> BTreeSet<String> {
        set.iter()
            .filter(|name| name.ends_with(".slang") || name.ends_with(".rs"))
            .cloned()
            .collect()
    };
    assert_eq!(
        in_scope(&on_disk),
        pinned,
        "tests/fixtures/*.slang and *.rs against the goldens and fixtures this gate walks (a new fixture `.rs` is a \
         `fixture!` line and a `Golden` row; a fixture without a golden is a decision for the coordinator)"
    );
    // Each golden's entries are kernels of its fixture, in the fixture's order.
    let mut total = 0;
    for golden in goldens() {
        let entries: Vec<String> = sections(golden.text)
            .iter()
            .flat_map(|s| compute_entries(s))
            .collect();
        let kernels: Vec<&str> = golden
            .kernels
            .iter()
            .map(|kernel| kernel.entry_point)
            .collect();
        assert!(
            kernels.starts_with(&entries.iter().map(String::as_str).collect::<Vec<_>>()),
            "{}: golden entries {entries:?} against the fixture's kernels {kernels:?}",
            golden.file
        );
        total += entries.len();
    }
    assert_eq!(total, GOLDEN_ENTRIES);
}

#[test]
fn every_fixture_wgsl_validates_and_has_the_descriptor_layouts() {
    let cases = all_cases();
    let mut problems = Vec::new();
    for Case {
        origin,
        entry,
        wgsl,
    } in &cases
    {
        match parse_and_validate(wgsl, DEFAULT_CAPABILITIES) {
            Ok(module) => problems.extend(
                compare_kernel(descriptor_of(entry), &module)
                    .into_iter()
                    .map(|problem| format!("{origin}: {problem}")),
            ),
            Err(_) if REFUSED_BY_NAGA.iter().any(|(name, _)| name == entry) => {}
            Err(error) => {
                problems.push(format!("{origin}: {entry}: naga refuses the WGSL: {error}"))
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} finding(s):\n{}",
        problems.len(),
        problems.join("\n")
    );
    assert_eq!(cases.len(), GOLDEN_ENTRIES + FIXTURE_KERNELS);
}

#[test]
fn allow_listed_entries_still_fail() {
    if REFUSED_BY_NAGA.is_empty() {
        return;
    }
    let cases = all_cases();
    for (entry, reason) in REFUSED_BY_NAGA {
        let case = cases
            .iter()
            .find(|case| case.entry == *entry)
            .unwrap_or_else(|| panic!("`{entry}` is not a fixture kernel"));
        assert!(
            parse_and_validate(&case.wgsl, DEFAULT_CAPABILITIES).is_err(),
            "`{entry}` is allow-listed ({reason}) but naga accepts it now: delete the entry"
        );
    }
}

/// The kernel the committed precompiled WGSL belongs to: a copy of `vector_add::add` of `tests/precompiled.rs`, whose
/// `fixture_is_current` test ties the committed file to that kernel's source (a drift shows here as a finding too).
#[gust::gpu]
mod vector_add {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }
}

const PRECOMPILED: &[u8] = include_bytes!("fixtures/precompiled/52e85c5f03f72272.wgsl");

#[test]
fn the_committed_precompiled_wgsl_validates_and_has_the_descriptor_layouts() {
    // The same file is committed five more times (the fixtures of gust, the web example, the precompiled fixture and
    // two ui fixtures); all must stay byte-identical, so one validation covers the six.
    let copies: [(&str, &[u8]); 5] = [
        (
            "crates/gust/tests/fixtures/precompiled",
            include_bytes!("../../gust/tests/fixtures/precompiled/52e85c5f03f72272.wgsl"),
        ),
        (
            "examples/web-vector-add/gust-artifacts",
            include_bytes!("../../../examples/web-vector-add/gust-artifacts/52e85c5f03f72272.wgsl"),
        ),
        (
            "tests/precompiled/kernels/gust-artifacts",
            include_bytes!(
                "../../../tests/precompiled/kernels/gust-artifacts/52e85c5f03f72272.wgsl"
            ),
        ),
        (
            "tests/ui/profile-native-artifact/gust-artifacts",
            include_bytes!(
                "../../../tests/ui/profile-native-artifact/gust-artifacts/52e85c5f03f72272.wgsl"
            ),
        ),
        (
            "tests/ui/profile-native-no-manifest/gust-artifacts",
            include_bytes!(
                "../../../tests/ui/profile-native-no-manifest/gust-artifacts/52e85c5f03f72272.wgsl"
            ),
        ),
    ];
    for (place, bytes) in copies {
        assert_eq!(
            bytes, PRECOMPILED,
            "{place}: a committed copy of the precompiled WGSL differs"
        );
    }
    let text = std::str::from_utf8(PRECOMPILED).expect("UTF-8 WGSL");
    let module =
        parse_and_validate(text, DEFAULT_CAPABILITIES).unwrap_or_else(|error| panic!("{error}"));
    let problems = compare_kernel(&vector_add::add::DESCRIPTOR, &module);
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

// ---- Controls: the gate fails on the defects it exists for. None of these needs slangc. ----

#[gust::gpu]
mod control {
    pub struct Body {
        pub position: Vec2,
        pub velocity: Vec2,
        pub mass: f32,
        pub tag: u32,
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut bodies: RWStructuredBuffer<Body>) {
        let i = id.x;
        bodies[i].mass = 1.0;
    }
}

/// The WGSL slangc emits for `control::run`, written by hand in its spelling (names with `_0`, the storage array).
const CONTROL_WGSL: &str = "struct Body_0
{
    position_0 : vec2<f32>,
    velocity_0 : vec2<f32>,
    mass_0 : f32,
    tag_0 : u32,
};

@binding(0) @group(0) var<storage, read_write> bodies_0 : array<Body_0>;

@compute
@workgroup_size(1, 1, 1)
fn gpu_control_run(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    bodies_0[id_0.x].mass_0 = 1.0f;
    return;
}
";

fn control_findings(wgsl: &str, kernel: &KernelDescriptor) -> Vec<String> {
    let module =
        parse_and_validate(wgsl, DEFAULT_CAPABILITIES).unwrap_or_else(|error| panic!("{error}"));
    compare_kernel(kernel, &module)
}

#[test]
fn the_control_kernel_agrees() {
    let findings = control_findings(CONTROL_WGSL, &control::run::DESCRIPTOR);
    assert!(findings.is_empty(), "{}", findings.join("\n"));
}

#[test]
fn a_changed_descriptor_offset_names_the_kernel_and_the_field() {
    let mut parameters = control::run::DESCRIPTOR.parameters.to_vec();
    let bodies = parameters
        .iter_mut()
        .find(|parameter| parameter.name == "bodies")
        .expect("the control kernel has a `bodies` parameter");
    let mut layout = bodies.resource_layout.expect("a storage layout");
    let TypeLayoutKind::Struct(fields) = layout.element.kind else {
        panic!("the element is a struct");
    };
    let mut fields = fields.to_vec();
    let mass = fields
        .iter_mut()
        .find(|field| field.name == "mass")
        .expect("a `mass` field");
    assert_eq!(mass.offset, 16);
    mass.offset = 20;
    layout.element.kind = TypeLayoutKind::Struct(Box::leak(fields.into_boxed_slice()));
    bodies.resource_layout = Some(layout);
    let mutated = KernelDescriptor {
        parameters: Box::leak(parameters.into_boxed_slice()),
        ..control::run::DESCRIPTOR
    };
    let findings = control_findings(CONTROL_WGSL, &mutated);
    assert_eq!(
        findings,
        ["gpu_control_run: bodies[].mass: offset: descriptor 20, WGSL 16"]
    );
}

#[test]
fn a_swapped_wgsl_member_order_names_the_kernel_and_the_fields() {
    let swapped = CONTROL_WGSL.replace(
        "    mass_0 : f32,\n    tag_0 : u32,\n",
        "    tag_0 : u32,\n    mass_0 : f32,\n",
    );
    assert_ne!(swapped, CONTROL_WGSL);
    let findings = control_findings(&swapped, &control::run::DESCRIPTOR);
    assert_eq!(
        findings,
        [
            "gpu_control_run: bodies[].mass: field name: descriptor mass, WGSL tag_0",
            "gpu_control_run: bodies[].mass: scalar type: descriptor F32, WGSL Scalar { kind: Uint, width: 4 }",
            "gpu_control_run: bodies[].tag: field name: descriptor tag, WGSL mass_0",
            "gpu_control_run: bodies[].tag: scalar type: descriptor U32, WGSL Scalar { kind: Float, width: 4 }",
        ]
    );
}

#[test]
fn a_wgsl_field_that_moves_is_an_offset_size_and_stride_finding() {
    // A `@size(16)` on the second field moves `mass` and `tag` by 8 bytes and grows the struct from 24 to 32.
    let moved = CONTROL_WGSL.replace("    velocity_0 :", "    @size(16) velocity_0 :");
    assert_ne!(moved, CONTROL_WGSL);
    let findings = control_findings(&moved, &control::run::DESCRIPTOR);
    assert_eq!(
        findings,
        [
            "gpu_control_run: bodies: array stride: descriptor 24, WGSL 32",
            "gpu_control_run: bodies[]: size: descriptor 24, WGSL 32",
            "gpu_control_run: bodies[].mass: offset: descriptor 16, WGSL 24",
            "gpu_control_run: bodies[].tag: offset: descriptor 20, WGSL 28",
        ]
    );
}

#[test]
fn default_capabilities_refuse_what_all_accept() {
    // `f16` needs the shader-f16 capability, which no device of this crate requests. The default of the WGSL front end
    // (`Options::new`, every capability) accepts it; wgpu would not, so the gate must not use that default.
    let source = "enable f16;
@compute @workgroup_size(1)
fn main()
{
    var x : f16 = 1h;
}
";
    assert!(parse_and_validate(source, Capabilities::all()).is_ok());
    assert!(parse_and_validate(source, DEFAULT_CAPABILITIES).is_err());
    assert_eq!(DEFAULT_CAPABILITIES, Capabilities::empty());
}

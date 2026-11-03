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
//! - the 12 Slang goldens of `tests/fixtures/*.slang`, which are all of the goldens (the test
//!   `the_goldens_are_the_fixtures_directory` pins the file list against the directory). Each golden is split at its
//!   `// ==== kernel <name> ====` markers where it has them, and each `[shader("compute")]` entry of each section is
//!   compiled to WGSL with the pinned slangc through `gust::slang::compile_source` (the same arguments as the
//!   run-time compile of a kernel): 47 entries. The descriptor of an entry comes from the fixture's Rust source,
//!   included below the way the other test files include it;
//! - the 9 fixture kernels that no golden holds (`atomics.slang` keeps only the first of the ten kernels of
//!   `atomics.rs`), compiled from their descriptors' own Slang with `gust::slang::compile_wgsl`, the run-time route,
//!   so that every kernel of every golden fixture is covered: 56 WGSL modules from the fixtures;
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
//! The test needs the pinned `slangc` on `PATH` (Slangc tier) and no adapter. It fails, never skips, without slangc.

use std::collections::{BTreeMap, BTreeSet};

use gust::{
    Access, KernelDescriptor, ParameterDescriptor, ParameterKind, ScalarKind, ShaderLayoutRules,
    TypeLayout, TypeLayoutKind,
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

fixture!(f_atomics, atomics, "../../../tests/fixtures/atomics.rs");
fixture!(
    f_coretests,
    coretests,
    "../../../tests/fixtures/coretests.rs"
);
fixture!(f_loops, loops, "../../../tests/fixtures/loops.rs");
fixture!(f_mismatch, mismatch, "../../../tests/fixtures/mismatch.rs");
fixture!(f_numeric, numeric, "../../../tests/fixtures/numeric.rs");
fixture!(f_option, optional, "../../../tests/fixtures/option.rs");
fixture!(f_products, products, "../../../tests/fixtures/products.rs");
fixture!(f_reduce, reduce, "../../../tests/fixtures/reduce.rs");
fixture!(f_result, results, "../../../tests/fixtures/result.rs");
fixture!(
    f_semantics,
    semantics,
    "../../../tests/fixtures/semantics.rs"
);
fixture!(f_uniforms, uniforms, "../../../tests/fixtures/uniforms.rs");
fixture!(f_vectors, vectors, "../../../tests/fixtures/vectors.rs");

/// Every golden: its file name under `tests/fixtures`, its text, the number of compute entries it holds (pinned, so a
/// golden that gains a kernel is a deliberate edit here) and the descriptors of its fixture.
struct Golden {
    file: &'static str,
    text: &'static str,
    entries: usize,
    kernels: &'static [KernelDescriptor],
}

fn goldens() -> [Golden; 12] {
    [
        Golden {
            file: "atomics.slang",
            text: include_str!("../../../tests/fixtures/atomics.slang"),
            entries: 1,
            kernels: f_atomics::KERNELS,
        },
        Golden {
            file: "coretests.slang",
            text: include_str!("../../../tests/fixtures/coretests.slang"),
            entries: 5,
            kernels: f_coretests::KERNELS,
        },
        Golden {
            file: "loops.slang",
            text: include_str!("../../../tests/fixtures/loops.slang"),
            entries: 1,
            kernels: f_loops::KERNELS,
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
            entries: 18,
            kernels: f_reduce::KERNELS,
        },
        Golden {
            file: "result.slang",
            text: include_str!("../../../tests/fixtures/result.slang"),
            entries: 1,
            kernels: f_result::KERNELS,
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
                let columns = match members.as_slice() {
                    [member]
                        if member.offset == 0
                            && let TypeInner::Array {
                                base,
                                size: ArraySize::Constant(length),
                                stride,
                            } = self.module.types[member.ty].inner
                            && let TypeInner::Vector {
                                size: column,
                                scalar: wgsl_scalar,
                            } = self.module.types[base].inner =>
                    {
                        Some((length.get(), stride, column as u32, wgsl_scalar))
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

/// The compute entries the twelve goldens hold in all (pinned per golden in [`goldens`]).
const GOLDEN_ENTRIES: usize = 47;

/// The fixture kernels that no golden holds: the macro's golden test keeps only the first kernel of `atomics.rs`
/// (`atomics.slang` has one entry for the ten kernels of the module). The gate takes their WGSL too, from the
/// descriptor's own Slang through the run-time route, so that every kernel of every golden fixture is covered.
const FIXTURE_KERNELS_WITHOUT_GOLDEN: usize = 9;

/// One WGSL module: where it came from (a golden file, or `fixture` for a kernel without a golden), its entry and text.
struct Case {
    origin: &'static str,
    entry: String,
    wgsl: String,
}

/// The WGSL of every golden entry (the pinned slangc, `compile_source` on the golden's own text) and of every other
/// fixture kernel (`compile_wgsl` on its descriptor, which is what the run-time route compiles).
fn all_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut in_a_golden = BTreeSet::new();
    for golden in goldens() {
        let mut entries = 0;
        for section in sections(golden.text) {
            for entry in compute_entries(&section) {
                let bytes = slang::compile_source(&section, &entry, Target::Wgsl).unwrap_or_else(|error| {
                    panic!(
                        "{}: slangc could not compile `{entry}` (the pinned slangc must be on PATH): {error}",
                        golden.file
                    )
                });
                let wgsl = String::from_utf8(bytes).expect("slangc writes UTF-8 WGSL");
                in_a_golden.insert(entry.clone());
                cases.push(Case {
                    origin: golden.file,
                    entry,
                    wgsl,
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
        cases.len(),
        GOLDEN_ENTRIES,
        "the compute entries of all goldens"
    );
    for golden in goldens() {
        for kernel in golden
            .kernels
            .iter()
            .filter(|k| !in_a_golden.contains(k.entry_point))
        {
            let wgsl = slang::compile_wgsl(kernel).unwrap_or_else(|error| {
                panic!(
                    "{}: slangc could not compile the fixture kernel: {error}",
                    kernel.entry_point
                )
            });
            cases.push(Case {
                origin: "fixture",
                entry: kernel.entry_point.to_owned(),
                wgsl,
            });
        }
    }
    assert_eq!(
        cases.len() - GOLDEN_ENTRIES,
        FIXTURE_KERNELS_WITHOUT_GOLDEN,
        "the fixture kernels that no golden holds"
    );
    cases
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
        .filter(|name| name.ends_with(".slang"))
        .collect();
    let pinned: BTreeSet<String> = goldens()
        .iter()
        .map(|golden| golden.file.to_owned())
        .collect();
    assert_eq!(
        on_disk, pinned,
        "tests/fixtures/*.slang against the goldens this gate walks"
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
    assert_eq!(cases.len(), GOLDEN_ENTRIES + FIXTURE_KERNELS_WITHOUT_GOLDEN);
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

//! The portable floor (M8, D36, D84, D143): the limits of WebGPU's defaults, checked against every kernel body of a crate
//! that sets `limits = "portable"` in `[package.metadata.gust]`.
//!
//! The check counts only what the source of a kernel decides, from the same facts the descriptor is made of (the
//! `workgroup_size` option, the parameters' types, the `Workgroup` arrays), and applies the numbers of
//! `gust::limits::PORTABLE`. The macro cannot depend on `gust`, so [`FLOOR`] is a copy of that table; a unit test below
//! pins it to the list of values that `gust`'s own test and `gust-compute-wgpu`'s equality test (against
//! `wgpu::Limits::default()` and the limits of an opened device) also pin.
//!
//! The check runs BEFORE the validator ([`check_declared`], one call in `expand_in`). The validator reports the workgroup
//! memory limit as GUST-E0718, and the floor's limit for it is the same 16 KiB, so for a portable crate E0746 has to be
//! reached first: that is how E0718 is folded into E0746. A crate without the setting never reaches this code, and E0718
//! stays what it was. Reading the manifest here, instead of taking the declarations the expansion reads after the
//! validator, is what makes that order possible; a malformed table is left to the expansion's own read (GUST-E0753).
//!
//! What it does not count, because it cannot: a `Workgroup` array whose length is not an integer literal or whose element
//! type the macro cannot size contributes 0 bytes (as in E0718); a kernel attribute that does not parse is skipped here and
//! reported by the validator; and the other entry points of the macro, `#[gpu] fn` maps and `par_for!`, do not pass
//! through this check.

use std::path::Path;

use crate::diagnostics::{self, Code};
use crate::expand::kernel_attribute_facts;
use crate::manifest::{self, Limits};
use crate::profile::{GRID, PIXEL, StageProfile};
use crate::slang::{ParameterFlavor, parameter_flavor_in};

use super::barrier;
use super::classify::Shape;

/// One limit of the floor: wgpu's name, the value, and how a message names what was counted.
struct Limit {
    name: &'static str,
    max: u64,
    /// What the kernel has, as "its <what> is <value>".
    what: &'static str,
    /// The unit that follows a value in the message, `""` for a count.
    unit: &'static str,
}

const WORKGROUP_X: usize = 0;
const WORKGROUP_Y: usize = 1;
const WORKGROUP_Z: usize = 2;
const INVOCATIONS: usize = 3;
const STORAGE_BUFFERS: usize = 4;
const UNIFORM_BUFFERS: usize = 5;
const SAMPLED_TEXTURES: usize = 6;
const STORAGE_TEXTURES: usize = 7;
const SAMPLERS: usize = 8;
const WORKGROUP_MEMORY: usize = 9;

/// The floor, in the order the check reports a violation (the order of `check_kernel_limits` in `gust-compute-wgpu`,
/// with the workgroup memory last). A copy of `gust::limits::PORTABLE`.
const FLOOR: [Limit; 10] = [
    Limit {
        name: "max_compute_workgroup_size_x",
        max: 256,
        what: "workgroup size along x",
        unit: "",
    },
    Limit {
        name: "max_compute_workgroup_size_y",
        max: 256,
        what: "workgroup size along y",
        unit: "",
    },
    Limit {
        name: "max_compute_workgroup_size_z",
        max: 64,
        what: "workgroup size along z",
        unit: "",
    },
    Limit {
        name: "max_compute_invocations_per_workgroup",
        max: 256,
        what: "workgroup invocation count (x * y * z)",
        unit: "",
    },
    Limit {
        name: "max_storage_buffers_per_shader_stage",
        max: 8,
        what: "storage buffer count",
        unit: "",
    },
    Limit {
        name: "max_uniform_buffers_per_shader_stage",
        max: 12,
        what: "uniform buffer count",
        unit: "",
    },
    Limit {
        name: "max_sampled_textures_per_shader_stage",
        max: 16,
        what: "sampled texture count",
        unit: "",
    },
    Limit {
        name: "max_storage_textures_per_shader_stage",
        max: 4,
        what: "storage texture count",
        unit: "",
    },
    Limit {
        name: "max_samplers_per_shader_stage",
        max: 16,
        what: "sampler count",
        unit: "",
    },
    Limit {
        name: "max_compute_workgroup_storage_size",
        max: 16384,
        what: "workgroup memory",
        unit: " bytes",
    },
];

/// Checks the kernels of `module` against the floor when the crate whose manifest directory is `manifest_dir` declares
/// `limits = "portable"`. No directory, no `Cargo.toml`, no `limits` key and a manifest the reader refuses (GUST-E0753,
/// which the expansion reports after the validator) are all `Ok`.
pub(crate) fn check_declared(
    module: &syn::ItemMod,
    manifest_dir: Option<&Path>,
) -> syn::Result<()> {
    match manifest::read_dir(manifest_dir) {
        Ok(Some(declarations)) => check(module, declarations.limits),
        Ok(None) | Err(_) => Ok(()),
    }
}

/// Checks the kernels of `module` against the floor when `limits` is the portable one. Every `#[kernel]` function is a
/// body here, a variant's included (the variants have been lowered to kernels). The first violation, in the order of
/// [`FLOOR`], is GUST-E0746 at the kernel's signature.
pub(crate) fn check(module: &syn::ItemMod, limits: Option<Limits>) -> syn::Result<()> {
    match limits {
        Some(Limits::Portable) => {}
        None => return Ok(()),
    }
    let Some((_, items)) = &module.content else {
        return Ok(());
    };
    for item in items {
        if let syn::Item::Fn(function) = item {
            check_kernel(function)?;
        }
    }
    Ok(())
}

fn check_kernel(function: &syn::ItemFn) -> syn::Result<()> {
    // An attribute the validator will refuse is its own error; a function without `#[kernel]` is a helper.
    let Ok(Some((options, _, facts))) = kernel_attribute_facts(&GRID, function) else {
        return Ok(());
    };
    let pixel = facts
        .shape
        .as_ref()
        .is_some_and(|pin| pin.shape == Shape::Pixel);
    let stage: &dyn StageProfile = if pixel { &PIXEL } else { &GRID };
    let measured = measure(stage, function, options.workgroup_size, pixel);
    for (limit, value) in FLOOR.iter().zip(measured) {
        let Some(value) = value else { continue };
        if value > limit.max {
            return Err(diagnostics::error(
                Code::PortableFloor,
                &function.sig,
                format!(
                    "kernel `{}` is over the portable floor: its {} is {value}{unit}, over `{}` = {}{unit} (the crate sets \
                     `limits = \"portable\"` in `[package.metadata.gust]`, which keeps every kernel within WebGPU's \
                     default limits, `gust::limits::PORTABLE`); lower it, or remove the setting if the crate only \
                     targets devices with higher limits",
                    function.sig.ident,
                    limit.what,
                    limit.name,
                    limit.max,
                    unit = limit.unit,
                ),
            ));
        }
    }
    Ok(())
}

/// What the kernel has for each limit of [`FLOOR`], in its order; `None` for a limit that does not apply (a `pixel`
/// kernel has no workgroup).
fn measure(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
    workgroup: [u32; 3],
    pixel: bool,
) -> [Option<u64>; 10] {
    let mut counts = [0_u64; 10];
    for argument in &function.sig.inputs {
        let syn::FnArg::Typed(argument) = argument else {
            continue;
        };
        let slot = match parameter_flavor_in(stage, &argument.ty) {
            ParameterFlavor::StorageRead | ParameterFlavor::StorageReadWrite => STORAGE_BUFFERS,
            ParameterFlavor::Uniform => UNIFORM_BUFFERS,
            ParameterFlavor::SampledTexture => SAMPLED_TEXTURES,
            ParameterFlavor::StorageTexture | ParameterFlavor::WriteTexture => STORAGE_TEXTURES,
            ParameterFlavor::Sampler => SAMPLERS,
            _ => continue,
        };
        counts[slot] += 1;
    }
    counts[WORKGROUP_MEMORY] = barrier::workgroup_bytes(function);
    let mut measured = counts.map(Some);
    if pixel {
        measured[WORKGROUP_X] = None;
        measured[WORKGROUP_Y] = None;
        measured[WORKGROUP_Z] = None;
        measured[INVOCATIONS] = None;
    } else {
        let [x, y, z] = workgroup.map(u64::from);
        measured[WORKGROUP_X] = Some(x);
        measured[WORKGROUP_Y] = Some(y);
        measured[WORKGROUP_Z] = Some(z);
        measured[INVOCATIONS] = Some(x.saturating_mul(y).saturating_mul(z));
    }
    measured
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::diagnostics::testing::assert_code;

    /// The floor as `gust::limits::PORTABLE.entries()` is pinned to in `crates/gust/src/limits.rs`, pasted verbatim.
    const GOLDEN: [(&str, u32); 10] = [
        ("max_compute_workgroup_size_x", 256),
        ("max_compute_workgroup_size_y", 256),
        ("max_compute_workgroup_size_z", 64),
        ("max_compute_invocations_per_workgroup", 256),
        ("max_storage_buffers_per_shader_stage", 8),
        ("max_uniform_buffers_per_shader_stage", 12),
        ("max_sampled_textures_per_shader_stage", 16),
        ("max_storage_textures_per_shader_stage", 4),
        ("max_samplers_per_shader_stage", 16),
        ("max_compute_workgroup_storage_size", 16384),
    ];

    /// A module with one kernel of the given shape. `size` is the `workgroup_size` option (`None`: the default),
    /// the counts are numbers of parameters of each kind and `memory` the length of an `f32` workgroup array
    /// (0: none).
    struct Spec {
        size: Option<[u32; 3]>,
        storage: usize,
        uniform: usize,
        sampled: usize,
        storage_textures: usize,
        samplers: usize,
        memory: u32,
    }

    impl Spec {
        const fn new() -> Self {
            Self {
                size: Some([64, 1, 1]),
                storage: 1,
                uniform: 0,
                sampled: 0,
                storage_textures: 0,
                samplers: 0,
                memory: 0,
            }
        }

        fn module(&self, attribute_extra: &str) -> syn::ItemMod {
            let mut parameters = vec!["id: SV_DispatchThreadID".to_owned()];
            for index in 0..self.storage {
                parameters.push(format!("s{index}: StructuredBuffer<f32>"));
            }
            for index in 0..self.uniform {
                parameters.push(format!("u{index}: Uniform<Params>"));
            }
            for index in 0..self.sampled {
                parameters.push(format!("t{index}: Texture2D<f32>"));
            }
            for index in 0..self.storage_textures {
                parameters.push(format!("w{index}: WriteTexture2D<Rgba32Float>"));
            }
            for index in 0..self.samplers {
                parameters.push(format!("m{index}: Sampler"));
            }
            if self.memory > 0 {
                parameters.push(format!("mut scratch: Workgroup<[f32; {}]>", self.memory));
            }
            let size = self
                .size
                .map(|[x, y, z]| format!("workgroup_size({x}, {y}, {z})"));
            let options: Vec<String> = size
                .into_iter()
                .chain((!attribute_extra.is_empty()).then(|| attribute_extra.to_owned()))
                .collect();
            let attribute = if options.is_empty() {
                "#[kernel]".to_owned()
            } else {
                format!("#[kernel({})]", options.join(", "))
            };
            let source = format!(
                "mod demo {{ {attribute} fn k({}) {{ }} }}",
                parameters.join(", ")
            );
            crate::diagnostics::testing::parse_str(&source).unwrap()
        }
    }

    fn portable(module: &syn::ItemMod) -> syn::Result<()> {
        check(module, Some(Limits::Portable))
    }

    #[track_caller]
    fn refused(shape: &Spec) -> String {
        let error = portable(&shape.module("")).unwrap_err();
        assert_code(&error, Code::PortableFloor);
        error.to_string()
    }

    #[test]
    fn the_copy_of_the_floor_is_the_golden_table() {
        let copy: Vec<(&str, u64)> = FLOOR.iter().map(|limit| (limit.name, limit.max)).collect();
        let golden: Vec<(&str, u64)> = GOLDEN
            .iter()
            .map(|(name, value)| (*name, u64::from(*value)))
            .collect();
        assert_eq!(copy, golden);
    }

    /// m9.0: `workgroup_size` has no upper bound, so the invocation count of the largest dimensions (about 2^96) does not
    /// fit the `u64` of the measurement: it saturates, which is over every limit, and the first violation in the
    /// order of the floor (the x axis) is the one reported. No arithmetic here can panic.
    #[test]
    fn the_largest_dimensions_are_over_the_floor_without_an_overflow() {
        let shape = Spec {
            size: Some([u32::MAX; 3]),
            ..Spec::new()
        };
        let text = refused(&shape);
        assert!(
            text.contains(
                "its workgroup size along x is 4294967295, over `max_compute_workgroup_size_x` = 256 "
            ),
            "{text}"
        );
        let module = shape.module("");
        let syn::Item::Fn(function) = &module.content.as_ref().unwrap().1[0] else {
            panic!("the kernel is the one item");
        };
        let measured = measure(&GRID, function, [u32::MAX; 3], false);
        assert_eq!(measured[INVOCATIONS], Some(u64::MAX));
        assert_eq!(measured[WORKGROUP_Z], Some(u64::from(u32::MAX)));
    }

    #[test]
    fn a_kernel_exactly_at_every_limit_passes() {
        // One kernel cannot sit at the axis maxima and the invocation maximum together, so the axes take three.
        let at_floor = Spec {
            size: Some([256, 1, 1]),
            storage: 8,
            uniform: 12,
            sampled: 16,
            storage_textures: 4,
            samplers: 16,
            memory: 4096,
        };
        portable(&at_floor.module("")).unwrap();
        for size in [[1, 256, 1], [1, 1, 64], [16, 16, 1], [4, 8, 8]] {
            let shape = Spec {
                size: Some(size),
                ..Spec::new()
            };
            portable(&shape.module("")).unwrap();
        }
    }

    #[test]
    fn one_over_each_limit_is_e0746_with_the_limit_and_both_values() {
        let cases = [
            (
                Spec {
                    size: Some([257, 1, 1]),
                    ..Spec::new()
                },
                "its workgroup size along x is 257, over `max_compute_workgroup_size_x` = 256 ",
            ),
            (
                Spec {
                    size: Some([1, 257, 1]),
                    ..Spec::new()
                },
                "its workgroup size along y is 257, over `max_compute_workgroup_size_y` = 256 ",
            ),
            (
                Spec {
                    size: Some([1, 1, 65]),
                    ..Spec::new()
                },
                "its workgroup size along z is 65, over `max_compute_workgroup_size_z` = 64 ",
            ),
            (
                Spec {
                    size: Some([17, 16, 1]),
                    ..Spec::new()
                },
                "its workgroup invocation count (x * y * z) is 272, over `max_compute_invocations_per_workgroup` = 256 ",
            ),
            (
                Spec {
                    storage: 9,
                    ..Spec::new()
                },
                "its storage buffer count is 9, over `max_storage_buffers_per_shader_stage` = 8 ",
            ),
            (
                Spec {
                    uniform: 13,
                    ..Spec::new()
                },
                "its uniform buffer count is 13, over `max_uniform_buffers_per_shader_stage` = 12 ",
            ),
            (
                Spec {
                    sampled: 17,
                    ..Spec::new()
                },
                "its sampled texture count is 17, over `max_sampled_textures_per_shader_stage` = 16 ",
            ),
            (
                Spec {
                    storage_textures: 5,
                    ..Spec::new()
                },
                "its storage texture count is 5, over `max_storage_textures_per_shader_stage` = 4 ",
            ),
            (
                Spec {
                    samplers: 17,
                    ..Spec::new()
                },
                "its sampler count is 17, over `max_samplers_per_shader_stage` = 16 ",
            ),
            (
                Spec {
                    memory: 4097,
                    ..Spec::new()
                },
                "its workgroup memory is 16388 bytes, over `max_compute_workgroup_storage_size` = 16384 bytes ",
            ),
        ];
        for (shape, wanted) in cases {
            let message = refused(&shape);
            assert!(
                message.starts_with("[GUST-E0746] kernel `k` is over the portable floor: "),
                "{message}"
            );
            assert!(message.contains(wanted), "{wanted:?} not in {message}");
            assert!(message.contains("gust::limits::PORTABLE"), "{message}");
        }
    }

    #[test]
    fn the_whole_message_of_one_case_is_pinned() {
        let shape = Spec {
            storage: 9,
            ..Spec::new()
        };
        assert_eq!(
            refused(&shape),
            "[GUST-E0746] kernel `k` is over the portable floor: its storage buffer count is 9, over \
             `max_storage_buffers_per_shader_stage` = 8 (the crate sets `limits = \"portable\"` in \
             `[package.metadata.gust]`, which keeps every kernel within WebGPU's default limits, \
             `gust::limits::PORTABLE`); lower it, or remove the setting if the crate only targets devices with higher \
             limits"
        );
    }

    #[test]
    fn a_crate_without_the_setting_is_not_checked() {
        let shape = Spec {
            size: Some([512, 1, 1]),
            storage: 9,
            uniform: 13,
            sampled: 17,
            storage_textures: 5,
            samplers: 17,
            memory: 5000,
        };
        check(&shape.module(""), None).unwrap();
    }

    #[test]
    fn the_first_violation_in_the_order_of_the_table_is_reported() {
        let shape = Spec {
            size: Some([512, 1, 1]),
            storage: 9,
            memory: 5000,
            ..Spec::new()
        };
        assert!(refused(&shape).contains("`max_compute_workgroup_size_x`"));
        let shape = Spec {
            storage: 9,
            memory: 5000,
            ..Spec::new()
        };
        assert!(refused(&shape).contains("`max_storage_buffers_per_shader_stage`"));
    }

    #[test]
    fn a_kernel_without_a_workgroup_size_takes_the_grid_default() {
        let shape = Spec {
            size: None,
            ..Spec::new()
        };
        let module = shape.module("");
        portable(&module).unwrap();
        // The default itself, not just "within the floor": 64 x 1 x 1, 64 invocations.
        let (_, items) = module.content.as_ref().unwrap();
        let function = items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(function) => Some(function),
                _ => None,
            })
            .unwrap();
        let (options, _, _) = kernel_attribute_facts(&GRID, function).unwrap().unwrap();
        let measured = measure(&GRID, function, options.workgroup_size, false);
        assert_eq!(
            [
                measured[WORKGROUP_X],
                measured[WORKGROUP_Y],
                measured[WORKGROUP_Z],
                measured[INVOCATIONS],
            ],
            [Some(64), Some(1), Some(1), Some(64)]
        );
    }

    #[test]
    fn a_pixel_kernel_has_no_workgroup_but_keeps_the_resource_limits() {
        // `workgroup_size` is not a pixel option (the validator refuses it), so the axes are skipped here.
        let wide = Spec {
            size: Some([512, 1, 1]),
            ..Spec::new()
        };
        portable(&wide.module("shape = pixel")).unwrap();
        let many = Spec {
            size: None,
            storage: 9,
            ..Spec::new()
        };
        let message = portable(&many.module("shape = pixel"))
            .unwrap_err()
            .to_string();
        assert!(message.contains("storage buffer count is 9"), "{message}");
    }

    #[test]
    fn helpers_and_attributes_the_validator_refuses_are_skipped() {
        // A function without `#[kernel]` is a helper; its parameters are not a kernel's bindings.
        let helper: syn::ItemMod = syn::parse_quote! {
            mod demo {
                fn helper(a: StructuredBuffer<f32>, b: StructuredBuffer<f32>, c: StructuredBuffer<f32>,
                    d: StructuredBuffer<f32>, e: StructuredBuffer<f32>, f: StructuredBuffer<f32>,
                    g: StructuredBuffer<f32>, h: StructuredBuffer<f32>, i: StructuredBuffer<f32>) { }
            }
        };
        portable(&helper).unwrap();
        // A malformed attribute is the validator's error (GUST-E0708), not this check's.
        let malformed: syn::ItemMod = syn::parse_quote! {
            mod demo {
                #[kernel(workgroup_size(512, 1))]
                fn k(id: SV_DispatchThreadID) { }
            }
        };
        portable(&malformed).unwrap();
    }

    #[test]
    fn a_module_without_content_is_left_to_the_validator() {
        let module: syn::ItemMod = syn::parse_quote! { mod demo; };
        portable(&module).unwrap();
    }

    /// A scratch crate directory with a `Cargo.toml`, removed on drop.
    struct Crate(std::path::PathBuf);

    impl Crate {
        fn new(name: &str, manifest: Option<&str>) -> Self {
            let directory =
                std::env::temp_dir().join(format!("gust-m8f-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&directory);
            fs::create_dir_all(&directory).unwrap();
            if let Some(text) = manifest {
                fs::write(directory.join("Cargo.toml"), text).unwrap();
            }
            Self(directory)
        }
    }

    impl Drop for Crate {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const PORTABLE_MANIFEST: &str = "[package]\nname = \"demo\"\nversion = \"0.0.0\"\n\n[package.metadata.gust]\nlimits = \"portable\"\n";
    const PLAIN_MANIFEST: &str = "[package]\nname = \"demo\"\nversion = \"0.0.0\"\n";

    fn too_wide() -> syn::ItemMod {
        Spec {
            size: Some([512, 1, 1]),
            ..Spec::new()
        }
        .module("")
    }

    #[test]
    fn the_manifest_decides() {
        let portable_crate = Crate::new("portable", Some(PORTABLE_MANIFEST));
        let message = check_declared(&too_wide(), Some(&portable_crate.0))
            .unwrap_err()
            .to_string();
        assert!(message.starts_with("[GUST-E0746] "), "{message}");
        let plain = Crate::new("plain", Some(PLAIN_MANIFEST));
        check_declared(&too_wide(), Some(&plain.0)).unwrap();
        let empty = Crate::new("empty", None);
        check_declared(&too_wide(), Some(&empty.0)).unwrap();
        check_declared(&too_wide(), None).unwrap();
    }

    #[test]
    fn a_malformed_manifest_is_left_to_the_expansion() {
        let broken = Crate::new(
            "broken",
            Some("[package]\nname = \"demo\"\n\n[package.metadata.gust]\nlimits = \"device\"\n"),
        );
        check_declared(&too_wide(), Some(&broken.0)).unwrap();
        let error =
            crate::expand::expand_with_directories(too_wide(), Some(&broken.0), None, Some("demo"))
                .unwrap_err()
                .to_string();
        assert!(error.starts_with("[GUST-E0753] "), "{error}");
    }

    /// A barrier kernel with 16388 bytes of workgroup memory: E0718 in an ordinary crate, E0746 in a portable one.
    fn memory_kernel() -> syn::ItemMod {
        syn::parse_quote! {
            mod block {
                #[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]
                pub fn reduce(
                    id: SV_DispatchThreadID,
                    lid: LocalId,
                    gid: GroupId,
                    input: StructuredBuffer<f32>,
                    mut partial: RWStructuredBuffer<f32>,
                    mut scratch: Workgroup<[f32; 4097]>,
                ) {
                    scratch[lid.x] = 1.0f32;
                    workgroup_barrier();
                    partial[gid.x] = scratch[0u32] + input[id.x];
                }
            }
        }
    }

    #[test]
    fn workgroup_memory_is_e0746_for_a_portable_crate_and_e0718_for_the_rest() {
        let plain = Crate::new("memory-plain", Some(PLAIN_MANIFEST));
        let error = crate::expand::expand_with_directories(
            memory_kernel(),
            Some(&plain.0),
            None,
            Some("demo"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.starts_with("[GUST-E0718] "), "{error}");
        let portable_crate = Crate::new("memory-portable", Some(PORTABLE_MANIFEST));
        let error = crate::expand::expand_with_directories(
            memory_kernel(),
            Some(&portable_crate.0),
            None,
            Some("demo"),
        )
        .unwrap_err();
        assert_code(&error, Code::PortableFloor);
        let error = error.to_string();
        assert!(
            error.contains("its workgroup memory is 16388 bytes, over `max_compute_workgroup_storage_size` = 16384 bytes"),
            "{error}"
        );
    }

    /// The same kernel with a `#[variant(of = reduce, requires())]` sibling of its signature (D139).
    fn memory_kernel_with_a_variant() -> syn::ItemMod {
        syn::parse_quote! {
            mod block {
                #[kernel(workgroup_size(256, 1, 1), cpu = "unsupported", requires(subgroups))]
                pub fn reduce(
                    id: SV_DispatchThreadID,
                    lid: LocalId,
                    gid: GroupId,
                    input: StructuredBuffer<f32>,
                    mut partial: RWStructuredBuffer<f32>,
                    mut scratch: Workgroup<[f32; 4097]>,
                ) {
                    scratch[lid.x] = 1.0f32;
                    workgroup_barrier();
                    partial[gid.x] = scratch[0u32] + input[id.x];
                }

                #[variant(of = reduce, requires())]
                pub fn reduce_portable(
                    id: SV_DispatchThreadID,
                    lid: LocalId,
                    gid: GroupId,
                    input: StructuredBuffer<f32>,
                    mut partial: RWStructuredBuffer<f32>,
                    mut scratch: Workgroup<[f32; 4097]>,
                ) {
                    scratch[lid.x] = 2.0f32;
                    workgroup_barrier();
                    partial[gid.x] = scratch[0u32] + input[id.x];
                }
            }
        }
    }

    #[test]
    fn a_variant_body_is_a_kernel_body_to_the_floor() {
        // After the lowering the variant is a `#[kernel]` function like the kernel, and the check refuses each of them.
        let (lowered, _) = crate::expand::variant::lower(memory_kernel_with_a_variant()).unwrap();
        let (_, items) = lowered.content.as_ref().unwrap();
        let bodies: Vec<&syn::ItemFn> = items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Fn(function) => Some(function),
                _ => None,
            })
            .filter(|function| matches!(kernel_attribute_facts(&GRID, function), Ok(Some(_))))
            .collect();
        assert_eq!(bodies.len(), 2, "the kernel and its variant");
        for body in bodies {
            let error = check_kernel(body).unwrap_err();
            assert_code(&error, Code::PortableFloor);
            assert!(
                error
                    .to_string()
                    .contains("workgroup memory is 16388 bytes"),
                "{error}"
            );
        }
        // And through the whole expansion of a portable crate.
        let portable_crate = Crate::new("memory-variant", Some(PORTABLE_MANIFEST));
        let error = crate::expand::expand_with_directories(
            memory_kernel_with_a_variant(),
            Some(&portable_crate.0),
            None,
            Some("demo"),
        )
        .unwrap_err();
        assert_code(&error, Code::PortableFloor);
    }
}

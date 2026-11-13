//! The SPIR-V route (Cargo feature `spirv-route`; slice m9.rt2, D172, D174).
//!
//! With the feature on, `compile_kernel` hands slangc's own SPIR-V for a kernel to Vulkan through
//! `Device::create_shader_module_passthrough`, instead of the kernel's WGSL through naga. Passthrough gets none of
//! naga's work: no run-time guards (the macro emits them as Slang, see `gust-compute-macros`'s `guards`), no interface
//! check and no binding remap. This module holds the pure parts: reading the module ([`scan`]), checking it against the
//! pipeline layout ([`check_bindings`]), renumbering its bindings the way wgpu-hal numbers the layout
//! ([`densify_bindings`]), and the record of which route each kernel took ([`RouteEvent`]).
//!
//! Every case that cannot take the route keeps the WGSL route and says why in a [`RouteEvent`], which
//! `HeadlessDevice::route_events` returns: a precompiled WGSL artifact (its bytes are the contract), no run-time
//! compiler, a device without `PASSTHROUGH_SHADERS`, a SPIR-V compile error, a binding check that fails.

// Without `slang-runtime` there is no SPIR-V to read: the pure functions are then only used by the unit tests.
#![cfg_attr(not(feature = "slang-runtime"), allow(dead_code))]

use std::collections::HashMap;

/// Which route compiled a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// slangc's SPIR-V through passthrough.
    Spirv,
    /// The kernel's WGSL through naga (the default route, and the fallback).
    Wgsl,
}

/// The route one kernel took when its pipeline was built, and why it did not take the SPIR-V route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteEvent {
    /// The qualified name of the kernel.
    pub kernel: String,
    /// The route the pipeline was built from.
    pub route: Route,
    /// For [`Route::Wgsl`], the reason the SPIR-V route was not taken; `None` for [`Route::Spirv`].
    pub reason: Option<String>,
}

/// What the binding check reads from a SPIR-V module.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Scan {
    /// `(execution model, name)` of each `OpEntryPoint`.
    pub(crate) entries: Vec<(u32, String)>,
    /// `(descriptor set, binding, storage class)` of each resource variable the module declares.
    pub(crate) resources: Vec<(u32, u32, u32)>,
}

const MAGIC: u32 = 0x0723_0203;
const OP_ENTRY_POINT: u32 = 15;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const CLASS_UNIFORM_CONSTANT: u32 = 0;
const CLASS_UNIFORM: u32 = 2;
const CLASS_STORAGE_BUFFER: u32 = 12;
/// `OpEntryPoint` execution models: fragment 4, GLCompute 5.
const MODEL_FRAGMENT: u32 = 4;
const MODEL_GL_COMPUTE: u32 = 5;

/// Little-endian words of a SPIR-V byte stream; `None` if the length is not a multiple of four.
pub(crate) fn words(bytes: &[u8]) -> Option<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    Some(
        chunks
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect(),
    )
}

/// The instructions of a module as `(offset of the first word, word count)`, after the 5-word header.
fn instructions(words: &[u32]) -> Result<Vec<(usize, usize)>, String> {
    if words.len() < 5 || words[0] != MAGIC {
        return Err("not a SPIR-V module".into());
    }
    let mut at = 5;
    let mut found = Vec::new();
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return Err("truncated instruction".into());
        }
        found.push((at, count));
        at += count;
    }
    Ok(found)
}

/// Read the entry points and the resource variables of a SPIR-V module.
pub(crate) fn scan(words: &[u32]) -> Result<Scan, String> {
    let mut variables: Vec<(u32, u32)> = Vec::new(); // (id, storage class)
    let mut sets: HashMap<u32, u32> = HashMap::new();
    let mut bindings: HashMap<u32, u32> = HashMap::new();
    let mut out = Scan::default();
    for (at, count) in instructions(words)? {
        let opcode = words[at] & 0xffff;
        let operands = &words[at + 1..at + count];
        match opcode {
            OP_ENTRY_POINT if operands.len() >= 3 => {
                let mut bytes = Vec::new();
                'name: for word in &operands[2..] {
                    for byte in word.to_le_bytes() {
                        if byte == 0 {
                            break 'name;
                        }
                        bytes.push(byte);
                    }
                }
                out.entries
                    .push((operands[0], String::from_utf8_lossy(&bytes).into_owned()));
            }
            OP_VARIABLE if operands.len() >= 3 => variables.push((operands[1], operands[2])),
            OP_DECORATE if operands.len() >= 3 => match operands[1] {
                DECORATION_BINDING => {
                    bindings.insert(operands[0], operands[2]);
                }
                DECORATION_DESCRIPTOR_SET => {
                    sets.insert(operands[0], operands[2]);
                }
                _ => {}
            },
            _ => {}
        }
    }
    for (id, class) in variables {
        if matches!(
            class,
            CLASS_UNIFORM_CONSTANT | CLASS_UNIFORM | CLASS_STORAGE_BUFFER
        ) && let Some(&binding) = bindings.get(&id)
        {
            out.resources
                .push((sets.get(&id).copied().unwrap_or(0), binding, class));
        }
    }
    Ok(out)
}

/// Every binding the module declares is in the layout (same binding number, a compatible kind), and the module has the
/// entry point at the execution model of the stage. The error names the first problem.
pub(crate) fn check_bindings(
    scan: &Scan,
    layout: &[wgpu::BindGroupLayoutEntry],
    entry: &str,
    pixel: bool,
) -> Result<(), String> {
    let model = if pixel {
        MODEL_FRAGMENT
    } else {
        MODEL_GL_COMPUTE
    };
    if !scan.entries.iter().any(|(m, n)| *m == model && n == entry) {
        return Err(format!(
            "no entry point `{entry}` at execution model {model} (has {:?})",
            scan.entries
        ));
    }
    for &(set, binding, class) in &scan.resources {
        if set != 0 {
            return Err(format!("binding ({set}, {binding}) is outside set 0"));
        }
        let Some(declared) = layout.iter().find(|entry| entry.binding == binding) else {
            return Err(format!("binding {binding} is not in the layout"));
        };
        let kind_ok = match (&declared.ty, class) {
            (wgpu::BindingType::Buffer { ty, .. }, CLASS_UNIFORM) => {
                matches!(ty, wgpu::BufferBindingType::Uniform)
            }
            (wgpu::BindingType::Buffer { ty, .. }, CLASS_STORAGE_BUFFER) => {
                matches!(ty, wgpu::BufferBindingType::Storage { .. })
            }
            (
                wgpu::BindingType::Texture { .. }
                | wgpu::BindingType::StorageTexture { .. }
                | wgpu::BindingType::Sampler(_),
                CLASS_UNIFORM_CONSTANT,
            ) => true,
            _ => false,
        };
        if !kind_ok {
            return Err(format!(
                "binding {binding}: SPIR-V storage class {class} does not fit layout entry {:?}",
                declared.ty
            ));
        }
    }
    Ok(())
}

/// wgpu-hal's Vulkan backend renumbers the entries of a bind group layout densely (0, 1, 2, ... in ascending order of
/// the WebGPU binding number) and tells naga's SPIR-V writer through a binding map. Passthrough SPIR-V gets no such
/// remap, so a sparse layout leaves the module's `Binding` decorations pointing at slots that do not exist: measured on
/// the RTX 5090 as a silently lost write and on the Radeon 890M as an access violation (m9.rt1 item 3). This rewrites
/// each `OpDecorate ... Binding n` to the rank of `n` among the layout's binding numbers. An error leaves `words`
/// possibly partly rewritten: the caller drops the module.
pub(crate) fn densify_bindings(
    words: &mut [u32],
    layout: &[wgpu::BindGroupLayoutEntry],
) -> Result<(), String> {
    let mut sorted: Vec<u32> = layout.iter().map(|entry| entry.binding).collect();
    sorted.sort_unstable();
    sorted.dedup();
    for (at, count) in instructions(words)? {
        if words[at] & 0xffff == OP_DECORATE && count >= 4 && words[at + 2] == DECORATION_BINDING {
            let Some(rank) = sorted.iter().position(|&binding| binding == words[at + 3]) else {
                return Err(format!("binding {} is not in the layout", words[at + 3]));
            };
            words[at + 3] = u32::try_from(rank).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One instruction: the opcode and operand words.
    fn op(opcode: u32, operands: &[u32]) -> Vec<u32> {
        let mut words = vec![((operands.len() as u32 + 1) << 16) | opcode];
        words.extend_from_slice(operands);
        words
    }

    fn name_words(name: &str) -> Vec<u32> {
        let mut bytes = name.as_bytes().to_vec();
        bytes.push(0);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        bytes
            .chunks(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// A module with a compute entry `main` and storage buffers at the given bindings (variable ids 10, 11, ...).
    fn module(bindings: &[u32]) -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0500, 0, 100, 0];
        let mut entry = vec![MODEL_GL_COMPUTE, 1];
        entry.extend(name_words("main"));
        words.extend(op(OP_ENTRY_POINT, &entry));
        for (index, &binding) in bindings.iter().enumerate() {
            let id = 10 + index as u32;
            words.extend(op(OP_DECORATE, &[id, DECORATION_DESCRIPTOR_SET, 0]));
            words.extend(op(OP_DECORATE, &[id, DECORATION_BINDING, binding]));
        }
        for index in 0..bindings.len() {
            words.extend(op(
                OP_VARIABLE,
                &[2, 10 + index as u32, CLASS_STORAGE_BUFFER],
            ));
        }
        words
    }

    fn storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    #[test]
    fn scan_reads_the_entry_point_and_the_resources() {
        let found = scan(&module(&[3, 999])).unwrap();
        assert_eq!(found.entries, vec![(MODEL_GL_COMPUTE, "main".to_owned())]);
        assert_eq!(
            found.resources,
            vec![(0, 3, CLASS_STORAGE_BUFFER), (0, 999, CLASS_STORAGE_BUFFER)]
        );
    }

    #[test]
    fn scan_refuses_what_is_not_a_module() {
        assert!(scan(&[1, 2, 3]).is_err());
        let mut truncated = module(&[0]);
        truncated.pop();
        assert_eq!(scan(&truncated).unwrap_err(), "truncated instruction");
        let mut zero = module(&[0]);
        zero.push(0);
        assert_eq!(scan(&zero).unwrap_err(), "truncated instruction");
        assert_eq!(words(&[1, 2, 3]), None);
        assert_eq!(words(&MAGIC.to_le_bytes()), Some(vec![MAGIC]));
    }

    #[test]
    fn the_check_accepts_a_matching_layout_and_names_the_first_problem() {
        let found = scan(&module(&[3, 999])).unwrap();
        let layout = [storage(3), storage(999), storage(5)];
        assert_eq!(check_bindings(&found, &layout, "main", false), Ok(()));
        // A binding the layout lacks.
        let error = check_bindings(&found, &[storage(3)], "main", false).unwrap_err();
        assert_eq!(error, "binding 999 is not in the layout");
        // The wrong entry point name and the wrong stage.
        assert!(
            check_bindings(&found, &layout, "other", false)
                .unwrap_err()
                .starts_with("no entry point `other`")
        );
        assert!(
            check_bindings(&found, &layout, "main", true)
                .unwrap_err()
                .contains("execution model 4")
        );
        // The wrong kind: a uniform entry for a storage variable.
        let mut uniform = storage(3);
        uniform.ty = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let error = check_bindings(&found, &[uniform, storage(999)], "main", false).unwrap_err();
        assert!(
            error.starts_with("binding 3: SPIR-V storage class 12 does not fit"),
            "{error}"
        );
    }

    #[test]
    fn a_module_outside_set_zero_is_refused() {
        let mut words = module(&[1]);
        let position = words
            .iter()
            .position(|w| *w == DECORATION_DESCRIPTOR_SET)
            .unwrap();
        words[position + 1] = 1;
        let found = scan(&words).unwrap();
        let error = check_bindings(&found, &[storage(1)], "main", false).unwrap_err();
        assert_eq!(error, "binding (1, 1) is outside set 0");
    }

    #[test]
    fn densify_renumbers_by_rank_in_the_layout() {
        for (module_bindings, layout, expected) in [
            (vec![3], vec![3], vec![0]),
            (vec![999], vec![0, 999], vec![1]),
            (vec![0, 999], vec![999, 0], vec![0, 1]),
            (vec![0, 1], vec![0, 1], vec![0, 1]),
            (vec![31, 5], vec![5, 31, 100], vec![1, 0]),
        ] {
            let mut words = module(&module_bindings);
            let layout: Vec<_> = layout.into_iter().map(storage).collect();
            densify_bindings(&mut words, &layout).unwrap();
            let found = scan(&words).unwrap();
            let got: Vec<u32> = found.resources.iter().map(|r| r.1).collect();
            assert_eq!(got, expected, "{module_bindings:?}");
        }
    }

    #[test]
    fn densify_refuses_a_binding_the_layout_lacks() {
        let mut words = module(&[7]);
        let error = densify_bindings(&mut words, &[storage(0)]).unwrap_err();
        assert_eq!(error, "binding 7 is not in the layout");
    }
}

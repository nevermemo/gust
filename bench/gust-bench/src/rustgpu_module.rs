//! The committed rust-gpu module of the `rustgpu-wgpu` comparator (m10.b4), its manifest and the CPU-only checks on both,
//! and the decision of whether the backend runs. Compiled in a `rust-gpu` build (the run file records the module's
//! identity from here) and in every test build, so `cargo test -p gust-bench` checks the module against its manifest
//! without a GPU and without the feature. No CI job runs the bench's unit tests; the same check runs on every push as a
//! step of `cargo xtask lint` (`xtask/src/lint.rs`, `rust_gpu_module_problems`).

// In a test build without the `rust-gpu` feature only the tests use most of this.
#![cfg_attr(not(feature = "rust-gpu"), allow(dead_code))]

use std::{error::Error, fmt};

use serde_json::{Value, json};

/// The committed module: little-endian SPIR-V words as bytes.
pub const MODULE: &[u8] = include_bytes!("../../rust-gpu-shaders/rustgpu.spv");

/// The manifest beside the module: toolchain, crate versions, entry points and the module's sha256.
pub const MANIFEST: &str = include_str!("../../rust-gpu-shaders/rustgpu.manifest.json");

/// Every entry point of the module, the names its listing in the passthrough descriptor needs (sorted, as the manifest).
pub const ENTRY_POINTS: [&str; 5] = ["w1_f32", "w1_u32", "w2_saxpy", "w3_horner", "w4_histogram"];

/// What a run file records about the comparator (`versions.rust_gpu`): the rust-gpu version, the toolchain and target
/// that built the module, and the sha256 of the module bytes this binary embeds (computed here, not read from the
/// manifest), so a run file ties to exactly one module.
pub fn identity() -> Value {
    let manifest: Value = serde_json::from_str(MANIFEST).unwrap_or(Value::Null);
    json!({
        "rust_gpu": manifest["rust_gpu"],
        "toolchain": manifest["toolchain"],
        "target": manifest["target"],
        "module_sha256": sha256(MODULE),
    })
}

/// SHA-256 of `data`, as lowercase hex (a small FIPS 180-4 implementation: the bench has no hash dependency).
pub fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (state, value) in h.iter_mut().zip(v) {
            *state = state.wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

/// The device of the comparator lacks a capability the module needs (wgpu's passthrough shaders, or the Vulkan memory
/// model the module declares). This, and only this, leaves the backend out of a run ("not run", with the reason); any
/// other error while opening the device is a failure of the run.
#[derive(Debug)]
pub struct Refusal(pub String);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for Refusal {}

/// Vulkan 1.3 as the packed version number (variant 0, major 1, minor 3, patch 0): the version that requires
/// `vulkanMemoryModel` to be supported.
pub const VULKAN_1_3: u32 = (1 << 22) | (3 << 12);

/// The capabilities the module needs of an adapter, from what was read of it: whether it has wgpu's
/// `PASSTHROUGH_SHADERS` and its physical device's Vulkan API version (`None` when there is no Vulkan HAL adapter).
/// Each missing one is a typed `Refusal`. Pure, so the gate is tested without a GPU.
pub fn check_capabilities(
    name: &str,
    passthrough: bool,
    api_version: Option<u32>,
) -> Result<(), Refusal> {
    if !passthrough {
        return Err(Refusal(format!(
            "adapter {name} lacks wgpu's PASSTHROUGH_SHADERS feature, so the rust-gpu SPIR-V module cannot be loaded"
        )));
    }
    match api_version {
        Some(version) if version >= VULKAN_1_3 => Ok(()),
        Some(version) => Err(Refusal(format!(
            "adapter {name} reports Vulkan {}.{}, below 1.3, so it is not known to support the Vulkan memory model the rust-gpu module declares",
            version >> 22,
            (version >> 12) & 0x3ff
        ))),
        None => Err(Refusal(format!(
            "adapter {name} has no Vulkan HAL adapter, so its Vulkan memory model support cannot be read"
        ))),
    }
}

/// What the run does about the backend.
#[derive(Debug)]
pub enum Decision<T> {
    /// The device opened: the backend runs.
    Run(T),
    /// The device refused for a missing capability: the backend is left out, recorded in the run file's
    /// `config.backends_not_run` with this entry.
    NotRun(Value),
}

/// Whether the comparator runs, from the outcome of opening its device and the run's `--backends` list: a refusal
/// (`Refusal`) leaves it out and records the reason, unless `--backends` names it, which is an error; any other error
/// propagates. Pure, so its three outcomes are unit-tested without a GPU.
pub fn decide<T>(
    opened: Result<T, Box<dyn Error>>,
    wanted: Option<&[String]>,
) -> Result<Decision<T>, Box<dyn Error>> {
    match opened {
        Ok(opened) => Ok(Decision::Run(opened)),
        Err(error) => {
            let Some(refusal) = error.downcast_ref::<Refusal>() else {
                return Err(error);
            };
            let reason = refusal.0.clone();
            if wanted.is_some_and(|names| names.iter().any(|name| name == "rustgpu-wgpu")) {
                return Err(
                    format!("--backends names rustgpu-wgpu, which cannot run: {reason}").into(),
                );
            }
            Ok(Decision::NotRun(
                json!({ "backend": "rustgpu-wgpu", "reason": reason }),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sha256_helper_matches_the_standard_vectors() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// The committed module is the one its manifest describes (CPU only, in the default build): the sha256, the size, the
    /// SPIR-V header, the entry points the passthrough descriptor lists, and the toolchain and crate pins against both
    /// `rust-toolchain.toml` files and both `Cargo.toml` pins.
    #[test]
    fn the_committed_module_matches_its_manifest() {
        let manifest: Value = serde_json::from_str(MANIFEST).expect("manifest is JSON");
        assert_eq!(manifest["sha256"], sha256(MODULE), "module sha256");
        assert_eq!(manifest["bytes"], MODULE.len());
        let listed: Vec<&str> = manifest["entry_points"]
            .as_array()
            .expect("entry_points")
            .iter()
            .map(|name| name.as_str().expect("name"))
            .collect();
        assert_eq!(listed, ENTRY_POINTS);
        // A SPIR-V module: whole words, the magic number first (little-endian), the version the manifest names (1.3).
        assert_eq!(MODULE.len() % 4, 0);
        assert_eq!(MODULE[..4], 0x0723_0203u32.to_le_bytes());
        assert_eq!(MODULE[4..8], 0x0001_0300u32.to_le_bytes());
        assert_eq!(manifest["spirv_version"], "1.3");
        assert_eq!(manifest["target"], "spirv-unknown-vulkan1.1");
        assert_eq!(manifest["rust_gpu"], "0.11.0");
        // The identity a run file records is the embedded module's.
        assert_eq!(identity()["module_sha256"], manifest["sha256"]);

        let toolchain = manifest["toolchain"].as_str().expect("toolchain");
        assert_eq!(toolchain, "nightly-2026-08-15");
        for (name, file) in [
            (
                "shader",
                include_str!("../../rust-gpu-shaders/shader/rust-toolchain.toml"),
            ),
            (
                "builder",
                include_str!("../../rust-gpu-shaders/builder/rust-toolchain.toml"),
            ),
        ] {
            assert!(
                file.contains(&format!("channel = \"{toolchain}\"")),
                "{name} rust-toolchain.toml pins {toolchain}"
            );
        }
        let version = manifest["rust_gpu"].as_str().expect("rust_gpu");
        assert!(
            include_str!("../../rust-gpu-shaders/shader/Cargo.toml")
                .contains(&format!("spirv-std = \"={version}\"")),
            "the shader crate pins spirv-std ={version}"
        );
        assert!(
            include_str!("../../rust-gpu-shaders/builder/Cargo.toml")
                .contains(&format!("spirv-builder = \"={version}\"")),
            "the builder pins spirv-builder ={version}"
        );
    }

    /// The capability gate refuses 1.2, a missing HAL adapter and a missing passthrough feature, and passes 1.3 and later.
    /// Its refusal reaches `decide` as a refusal through the same `?` conversion `open_with` uses, so a later `map_err`
    /// that turned it into another error would fail here. The call from `open_with` itself needs a real adapter
    /// (`as_hal`), which a unit test cannot make; this pins the function and the conversion it relies on.
    #[test]
    fn the_capability_gate_refuses_and_its_refusal_survives_the_open_path() {
        let v = |major: u32, minor: u32| (major << 22) | (minor << 12);
        assert!(check_capabilities("a", true, Some(v(1, 3))).is_ok());
        assert!(check_capabilities("a", true, Some(v(1, 4))).is_ok());
        for refused in [
            check_capabilities("a", true, Some(v(1, 2))),
            check_capabilities("a", true, None),
            check_capabilities("a", false, Some(v(1, 3))),
        ] {
            assert!(refused.is_err());
        }
        // The shape of `open_with`: a `Refusal` returned with `?` into a boxed error.
        fn open(version: u32) -> Result<u32, Box<dyn Error>> {
            check_capabilities("a", true, Some(version))?;
            Ok(1)
        }
        assert!(matches!(
            decide(open(v(1, 3)), None).unwrap(),
            Decision::Run(1)
        ));
        match decide(open(v(1, 2)), None).unwrap() {
            Decision::NotRun(entry) => {
                assert!(entry["reason"].as_str().unwrap().contains("below 1.3"));
            }
            Decision::Run(_) => panic!("Vulkan 1.2 must be refused"),
        }
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| (*name).to_owned()).collect()
    }

    fn refused() -> Result<u32, Box<dyn Error>> {
        Err(Box::new(Refusal("no vulkanMemoryModel".to_owned())))
    }

    #[test]
    fn a_device_that_opened_runs_the_backend() {
        assert!(matches!(decide(Ok(7), None).unwrap(), Decision::Run(7)));
        let wanted = names(&["rustgpu-wgpu"]);
        assert!(matches!(
            decide(Ok(7), Some(&wanted)).unwrap(),
            Decision::Run(7)
        ));
    }

    #[test]
    fn a_refusal_leaves_the_backend_out_and_records_the_reason() {
        for wanted in [None, Some(names(&["wgsl-prebuilt", "gust-gpu"]))] {
            match decide(refused(), wanted.as_deref()).unwrap() {
                Decision::NotRun(entry) => {
                    assert_eq!(entry["backend"], "rustgpu-wgpu");
                    assert_eq!(entry["reason"], "no vulkanMemoryModel");
                }
                Decision::Run(_) => panic!("a refused backend must not run"),
            }
        }
    }

    #[test]
    fn a_refusal_of_a_backend_named_by_backends_is_an_error_and_other_errors_propagate() {
        let wanted = names(&["gust-gpu", "rustgpu-wgpu"]);
        let error = decide(refused(), Some(&wanted)).unwrap_err().to_string();
        assert!(error.contains("--backends names rustgpu-wgpu"), "{error}");
        assert!(error.contains("no vulkanMemoryModel"), "{error}");
        // Any other failure to open the device fails the run, named or not.
        for wanted in [None, Some(wanted)] {
            let other: Result<u32, Box<dyn Error>> = Err("out of device memory".into());
            let error = decide(other, wanted.as_deref()).unwrap_err().to_string();
            assert_eq!(error, "out of device memory");
        }
    }
}

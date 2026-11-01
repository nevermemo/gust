// A kernel with a committed precompiled artifact. `env.txt` builds this crate under `GUST_PROFILE=native`, a strict
// profile (D44): the macro finds `gust-artifacts/<key>.wgsl` and its `.wgsl.reflection.json` beside this crate's
// manifest, embeds them, and emits `require_artifact(true)` and `require_manifest(true)`, so the crate compiles.
// `profile-native-no-manifest` is this crate without `gust-artifacts/manifest.json`.
//
// The key is FNV-1a 64 of the entry point `gpu_vector_add_add`, a NUL byte and the clean Slang text, so it changes
// when the kernel does; a kernel edit that leaves the files behind makes this fixture fail with the missing-artifact
// error of `profile-native-no-artifact`, which is the point. The files are the ones `crates/gust/tests/fixtures/
// precompiled` holds for this same kernel: the helper's output, not hand-written (see that directory's manifest).

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

// The macro embedded the pair: proven while compiling, not at run time (`check-ui` only checks).
const _: () = {
    let artifacts = vector_add::add::DESCRIPTOR.artifacts;
    assert!(artifacts.len() == 1);
    assert!(!artifacts[0].bytes.is_empty());
    assert!(!artifacts[0].reflection.is_empty());
};

fn main() {}

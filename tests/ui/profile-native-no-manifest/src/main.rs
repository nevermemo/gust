// `profile-native-artifact` without `gust-artifacts/manifest.json`. The artifact pair is there, so the macro embeds it
// and emits `require_artifact(true)`; the manifest is not, so `require_manifest(false)` fails const evaluation under
// `GUST_PROFILE=native`: one `E0080`, at the kernel's line in this file, telling the user to run `cargo gust init`
// and `cargo gust build` (R4.3).

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

fn main() {}

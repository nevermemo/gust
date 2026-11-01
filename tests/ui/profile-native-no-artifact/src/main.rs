// A kernel without a precompiled artifact. `env.txt` builds this crate under `GUST_PROFILE=native`, a strict profile
// (D44), where `gust::profile::require_artifact(false)` fails const evaluation: one `E0080`, reported at the kernel's
// line in this file, not inside `gust`. `profile-dev-no-artifact` builds this very file under `dev` and must compile.
// No artifact can exist yet (the macro embeds none before slice m4.c), so this fixture needs no artifact file.

#[gust::gpu]
mod kernels {
    #[kernel]
    fn fill(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        out[id.x] = id.x;
    }
}

fn main() {}

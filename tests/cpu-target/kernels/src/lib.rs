// The kernel crate of the CPU-target fixture. Built under `GUST_PROFILE=cpu` the macro emits, for each kernel that has
// a typed call, `impl Run<kernel, N> for gust::cpu::CpuDevice` in every build (no `cfg(test)`, no `--cfg gust_dual`),
// so a dependent crate can dispatch on the CPU in a production binary. Built under any other profile the impl is
// absent and the dispatch is the E0277 that `app/src/bin/single_dispatch.rs` expects (e-d-expected.txt).

#[gust::gpu]
pub mod vector_add {
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

// The expression of `examples/polynomial`, so its derived bound applies: v^3 - 2 v^2 + 0.5 v + 1, written order.
#[gust::gpu]
pub mod polynomial {
    #[kernel(workgroup_size(128, 1, 1))]
    pub fn evaluate(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            let value = input[i];
            let squared = value * value;
            let cubed = squared * value;
            let curved = cubed - (2.0 * squared);
            let linear = 0.5 * value;
            out[i] = (curved + linear) + 1.0;
        }
    }
}

// Every invocation increments the same counter: the result is the number of invocations that executed the add.
#[gust::gpu]
pub mod atomic_counter {
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn increment(_id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
        atomic_add(&mut counter[0], 1u32);
    }
}

// The kernel that opted out of the CPU: it has no `Run` impl for `CpuDevice` under any profile, `cpu` included.
#[gust::gpu]
pub mod opt_out {
    #[kernel(cpu = "unsupported")]
    pub fn negate(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = -input[i];
        }
    }
}

// A vector kernel (M6): `Vec4` buffers, component-wise `+`, a scalar product, a constructor, a swizzle and `dot`. It
// runs in the production CPU build like the scalar kernels (`gust::math` is plain Rust there). The inputs of `app` and
// of the test below are small integers, so every result is exact in `f32`.
use gust::math::Vec4;

#[gust::gpu]
pub mod vector_ops {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn blend(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<Vec4>,
        b: StructuredBuffer<Vec4>,
        mut blended: RWStructuredBuffer<Vec4>,
        mut dots: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < blended.len() {
            let sum = a[i] + b[i].wzyx();
            blended[i] = sum * 2.0 + Vec4::splat(0.5);
            dots[i] = a[i].dot(b[i]);
        }
    }
}

// A texture kernel (MX, slice mx.t4): `load` and `store` on a read-write storage texture and a `store` on a write-only
// one, in the production CPU build like the buffer kernels. Each invocation reads and writes only its own texel of
// `image` and writes one texel of `mirrored` (the texel of the same row at the opposite end), so the kernel is
// race-free and every texel has one writer. The inputs of `app` are halves, so the doubling is exact in `f32`.
use gust::texture::{R32Float, R32Uint};

#[gust::gpu]
pub mod texture_ops {
    /// Doubles every texel of `image` in place and records, in `mirrored` at the mirrored column, the row-major index
    /// of the texel it came from. The dispatch is a whole number of workgroups and may be larger than the texture, so
    /// the kernel guards against the texture's own size.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn double_and_mirror(
        id: SV_DispatchThreadID,
        mut image: StorageTexture2D<R32Float>,
        mut mirrored: WriteTexture2D<R32Uint>,
    ) {
        let width = image.width();
        if id.x < width && id.y < image.height() {
            let value = image.load(id.x, id.y);
            image.store(id.x, id.y, value * 2.0f32);
            mirrored.store(width - 1u32 - id.x, id.y, id.y * width + id.x);
        }
    }
}

// A `pixel` kernel (MX, slice mx.p1a): `shape = pixel` is a plain Rust function on the CPU (the CPU is a target, D51,
// never a shape), so it builds and runs in the production CPU build like the `grid` kernels, with no fragment, no
// scissor and no draw. Every element is written once at the element's own index, as the `pixel` shape requires. The
// inputs of `app` are multiples of a quarter, so `2v + 1` is exact in `f32`.
#[gust::gpu]
pub mod pixel_shade {
    #[kernel(shape = pixel)]
    pub fn shade(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        out[i] = (input[i] * 2.0) + 1.0;
    }
}

// The map functions (M7, D60): `#[gust::gpu]` on a function. Built under `GUST_PROFILE=cpu` only, because under any
// other profile the generated host functions name `gust-wgpu`, which this crate does not depend on; `map_case::run` is
// the one entry `app` calls in every build. Under another profile it is an error, never a silent pass:
// `check-cpu-target` requires the line that only the real case prints (`CPU target: map calls match (`), exactly
// once.
#[cfg(gust_cpu_profile)]
pub mod map_case;

#[cfg(not(gust_cpu_profile))]
pub mod map_case {
    /// No map functions are built under this profile, so the case cannot run.
    pub fn run() -> Result<(), String> {
        Err("the map case needs GUST_PROFILE=cpu (cfg gust_cpu_profile); it did not run".to_owned())
    }
}

// `gust::par_for!` (M7, slice m7.p), the same way: under another profile its expansion names `gust-wgpu`. Its line
// (`CPU target: par_for calls match (`) is not paired with a list by `check-cpu-target`; `app` exits with 1 and prints
// no final line when the case fails or did not run.
#[cfg(gust_cpu_profile)]
pub mod par_for_case;

#[cfg(not(gust_cpu_profile))]
pub mod par_for_case {
    /// No `par_for!` is built under this profile, so the case cannot run.
    pub fn run() -> Result<(), String> {
        Err("the par_for case needs GUST_PROFILE=cpu (cfg gust_cpu_profile); it did not run".to_owned())
    }
}

// Capability variants (M8, slice m8.s): a portable body and a `requires(subgroups)` variant. The CPU form is the portable
// body in every profile, so the module builds and its tests run under all of them.
pub mod variant;

#[cfg(test)]
mod tests {
    use gust::{cpu::CpuDevice, math::Vec4};

    /// The map case of the CPU target passes. Only under the cpu profile: elsewhere the case does not exist.
    #[cfg(gust_cpu_profile)]
    #[test]
    fn the_map_case_matches_its_host_references() {
        super::map_case::run().unwrap();
    }

    /// The `par_for!` case of the CPU target passes. Only under the cpu profile, like the map case.
    #[cfg(gust_cpu_profile)]
    #[test]
    fn the_par_for_case_matches_its_host_loops() {
        super::par_for_case::run().unwrap();
    }

    use gust::{Extent, texture::TextureUse};

    use super::{R32Float, R32Uint, texture_ops, vector_ops};

    /// A 3 x 2 texture, worked by hand: every texel doubles, and the index of the texel at `(x, y)` lands at
    /// `(2 - x, y)` of the other texture. The dispatch is one 8 x 8 workgroup, larger than the texture.
    #[test]
    fn double_and_mirror_matches_the_hand_computed_values() {
        let device = CpuDevice::new();
        let image = device
            .texture_2d::<R32Float>(3, 2, TextureUse::STORAGE)
            .unwrap();
        image.write(&[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).unwrap();
        let mirrored = device
            .texture_2d::<R32Uint>(3, 2, TextureUse::STORAGE)
            .unwrap();
        texture_ops::double_and_mirror::dispatch_extent(
            &device,
            Extent::new(8, 8, 1),
            &image,
            &mirrored,
        )
        .unwrap();
        assert_eq!(image.read(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(mirrored.read(), [2, 1, 0, 5, 4, 3]);
    }

    /// Three elements, worked by hand: a = (1, 2, 3, 4), b = (5, 6, 7, 8) gives
    /// `(a + b.wzyx) * 2 + 0.5` = ((1 + 8) * 2 + 0.5, (2 + 7) * 2 + 0.5, (3 + 6) * 2 + 0.5, (4 + 5) * 2 + 0.5) and
    /// `dot` = 5 + 12 + 21 + 32 = 70. The second and third elements make the swizzle and the dot visible.
    #[test]
    fn blend_matches_the_hand_computed_values() {
        let device = CpuDevice::new();
        let a = device.buffer(&[
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(-1.0, 0.5, 2.0, 0.0),
        ]);
        let b = device.buffer(&[
            Vec4::new(5.0, 6.0, 7.0, 8.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(4.0, 4.0, 4.0, 4.0),
        ]);
        let blended = device.buffer(&[Vec4::splat(0.0); 3]);
        let dots = device.buffer(&[0.0f32; 3]);
        vector_ops::blend::dispatch(&device, 3, &a, &b, &blended, &dots).unwrap();
        let lanes = |v: Vec4| [v.x, v.y, v.z, v.w];
        let got: Vec<[f32; 4]> = blended.read().into_iter().map(lanes).collect();
        assert_eq!(
            got,
            [
                [18.5, 18.5, 18.5, 18.5],
                // a + b.wzyx = (0, 0, 0, 1) + (0, 0, 0, 1) = (0, 0, 0, 2); times 2, plus 0.5.
                [0.5, 0.5, 0.5, 4.5],
                // (-1, 0.5, 2, 0) + (4, 4, 4, 4) = (3, 4.5, 6, 4); times 2, plus 0.5.
                [6.5, 9.5, 12.5, 8.5],
            ]
        );
        assert_eq!(dots.read(), [70.0, 0.0, 6.0]);
    }
}

// Control case for a map over a struct (m14.mapw1): the function is written inside the `#[gpu_module]` module that
// declares the struct, and the module of its calls is `shaders::scale::map(..)`. Built under `GUST_PROFILE=cpu`
// (env.txt), where the calls run on `gust::cpu::CpuDevice` and nothing names a GPU API.

use gust::map::Broadcast;

#[gust::gpu_module]
mod shaders {
    pub struct Particle {
        pub x: f32,
        pub y: f32,
        pub id: u32,
        pub tag: u32,
    }

    #[gust::gpu_fn]
    pub fn scale(p: Particle, k: f32) -> Particle {
        let mut scaled = p;
        scaled.x = p.x * k;
        scaled.y = p.y * k;
        scaled
    }

    #[gust::gpu_fn]
    pub fn weight(p: Particle) -> f32 {
        p.x + p.y
    }
}

use shaders::Particle;

fn main() {
    let input = [Particle {
        x: 1.0,
        y: 2.0,
        id: 7,
        tag: 0,
    }];
    let scaled = shaders::scale::map(&input, Broadcast(2.0)).unwrap();
    assert_eq!(scaled[0].x, 2.0);
    assert_eq!(shaders::weight::map(&scaled).unwrap(), [6.0]);
}

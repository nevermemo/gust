// Step E-e (the negative): a binary whose graph holds the GPU runtime. Under `GUST_PROFILE=cpu` a build holds one
// target (D21), so building this is refused by `gust::profile::require_gpu_target`, which `gust-wgpu` evaluates.
// Under every other profile it builds.

fn main() {
    println!("{}", std::any::type_name::<gust_wgpu::HeadlessDevice>());
}

// Only the handle that `#[gust::gpu_module]` generates for a kernel implements `Kernel`; any other
// type passed where a kernel is expected gets the wording of `#[diagnostic::on_unimplemented]`
// on `Kernel`, checked by expected.txt.

fn descriptor_name<K: gust::Kernel>() -> &'static str {
    K::DESCRIPTOR.name
}

fn main() {
    println!("{}", descriptor_name::<u32>());
}

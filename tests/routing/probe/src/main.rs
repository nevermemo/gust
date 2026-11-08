// Prints `<tag> <kernel> <target>` for each kernel of `route-kernels`, one line each, in a fixed order. `cargo xtask
// check-routing` compares these lines with the driver's `routes.json` (the two resolutions must agree) and with its golden.

use route_kernels::{broadphase, fusion, physics};

fn main() {
    println!("physics scale {}", physics::scale::DESCRIPTOR.target.name());
    println!(
        "physics.broadphase pairs {}",
        broadphase::pairs::DESCRIPTOR.target.name()
    );
    println!("fusion blend {}", fusion::blend::DESCRIPTOR.target.name());
}

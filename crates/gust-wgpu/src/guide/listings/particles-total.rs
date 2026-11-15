/// The energy of the particles through `par_for!` on `device`, summed on the host in `f64`.
fn total_energy(ps: &[particles::Particle], device: &Dev) -> f64 {
    let y: Vec<f32> = ps.iter().map(|p| p.pos[1]).collect();
    let vy: Vec<f32> = ps.iter().map(|p| p.vel[1]).collect();
    let energy = energies(&y, &vy, device).expect("par_for runs");
    energy.iter().map(|e| f64::from(*e)).sum()
}

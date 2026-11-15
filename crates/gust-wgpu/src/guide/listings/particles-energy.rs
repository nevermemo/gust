/// The mechanical energy of every particle (unit mass, gravity 8): `par_for!` writes the loop at the call site, over
/// flat arrays that the host split out of the particles.
fn energies(y: &[f32], vy: &[f32], device: &Dev) -> Result<Vec<f32>, gust::map::Error> {
    let mut out = vec![0.0f32; y.len()];
    let n = u32::try_from(y.len()).expect("fewer than 2^32 particles");
    gust::par_for!(on(device); i in 0..n, [y: &[f32], vy: &[f32], out: &mut [f32]] {
        out[i] = 0.5 * vy[i] * vy[i] + 8.0 * y[i];
    })?;
    Ok(out)
}

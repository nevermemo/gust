/// The uniform value for `a`; the padding is zero.
fn params(a: f32) -> saxpy::Params {
    saxpy::Params {
        a,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

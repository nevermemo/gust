// The vertex stage of every `pixel` pass (slice mx.p1b, decisions D61 and D120): one triangle that covers the whole
// attachment, from the vertex index alone (no vertex buffer). The vertices are (-1,-1), (3,-1) and (-1,3), counter-clockwise.
// Hand-written and the same text as the scissor probe measured on the RTX 5090 and the Radeon 890M (0 differing words).
@vertex
fn gust_fullscreen_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
}

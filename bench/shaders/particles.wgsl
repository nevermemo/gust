// Hand-written W5 (particle step) for plain wgpu. The same struct buffer GUST's kernel takes: a 28-byte std430 element
// (a test pins size, alignment, stride and the field order against GUST's reflected layout), the same 16 explicit
// Euler steps in registers per invocation, the same operation order as `workloads/particles.rs`. `mass`, `tag` and
// `charge` are copied through. Out of place: out = step^16(input, acceleration).
struct Pair {
    x: f32,
    y: f32,
}

struct Particle {
    position: Pair,
    velocity: Pair,
    mass: f32,
    tag: u32,
    charge: i32,
}

@group(0) @binding(0) var<storage, read> input: array<Particle>;
@group(0) @binding(1) var<storage, read> acceleration: array<Pair>;
@group(0) @binding(2) var<storage, read_write> out: array<Particle>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&out) && i < arrayLength(&input) && i < arrayLength(&acceleration)) {
        let particle = input[i];
        let force = acceleration[i];
        var vx = particle.velocity.x;
        var vy = particle.velocity.y;
        var px = particle.position.x;
        var py = particle.position.y;
        for (var step = 0u; step < 16u; step = step + 1u) {
            vx = vx + force.x * 0.016;
            vy = vy + force.y * 0.016;
            px = px + vx * 0.016;
            py = py + vy * 0.016;
        }
        var result = particle;
        result.velocity.x = vx;
        result.velocity.y = vy;
        result.position.x = px;
        result.position.y = py;
        out[i] = result;
    }
}

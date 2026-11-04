// A pixel pin on a kernel with nine outputs; a pixel has at most eight (GUST-E0723, decision D62).
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, mut o0: RWStructuredBuffer<f32>, mut o1: RWStructuredBuffer<f32>, mut o2: RWStructuredBuffer<f32>, mut o3: RWStructuredBuffer<f32>, mut o4: RWStructuredBuffer<f32>, mut o5: RWStructuredBuffer<f32>, mut o6: RWStructuredBuffer<f32>, mut o7: RWStructuredBuffer<f32>, mut o8: RWStructuredBuffer<f32>) {
        o0[id.x] = 1.0f32;
        o1[id.x] = 1.0f32;
        o2[id.x] = 1.0f32;
        o3[id.x] = 1.0f32;
        o4[id.x] = 1.0f32;
        o5[id.x] = 1.0f32;
        o6[id.x] = 1.0f32;
        o7[id.x] = 1.0f32;
        o8[id.x] = 1.0f32;
    }
}

fn main() {}

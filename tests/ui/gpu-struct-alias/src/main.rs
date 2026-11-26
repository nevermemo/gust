// A field type must BE the type its name stands for: a local alias that reuses an accepted name (`UVec2` as `IVec2`,
// the same size and alignment, so no padding gives it away) is refused by the derive's per-field identity check, at the
// field. With an alias of another size (`type UVec2 = u32`) the size assertions of the `GpuPod` impl fail as well.

use gust::GpuStruct;
use gust::prelude::*;

type UVec2 = IVec2;

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row {
    a: UVec2, // error here
    b: Vec2,
}

fn main() {}

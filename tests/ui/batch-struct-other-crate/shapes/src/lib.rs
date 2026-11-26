// The struct of a batch, declared in another crate: the derive's macro and the type are both items of this crate.

use gust::prelude::*;

#[derive(Clone, Copy, gust::GpuStruct)]
#[repr(C)]
pub struct Row {
    pub pos: Vec2,
    pub life: f32,
    pub flags: u32,
}

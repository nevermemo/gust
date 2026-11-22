use gust::gpu_module;
// The GPU side is only built outside the tests: the tests run the loop on the CPU twin (see `Dev` below).
#[cfg(not(test))]
use gust_wgpu::{BufferDispatch, GpuBufferAccess, HeadlessDevice};

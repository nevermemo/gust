//! Helpers of the two host-overhead programs of this crate (`graph-overhead` and `batch_shape`): the CPU time of the
//! calling thread and a median of three runs. Nothing here asserts a time.

/// The kernel and user CPU time the calling thread has used, in nanoseconds (Windows only: `GetThreadTimes`, whose tick
/// is coarse, so a run is thousands of calls). Zero elsewhere.
#[cfg(windows)]
pub fn thread_cpu_ns() -> u64 {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn GetThreadTimes(
            thread: isize,
            created: *mut FileTime,
            exited: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    let ticks = |time: &FileTime| (u64::from(time.high) << 32 | u64::from(time.low)) * 100;
    let (mut created, mut exited, mut kernel, mut user) = (
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
    );
    // SAFETY: the pseudo handle of the calling thread and four live out-structures of the documented layout.
    let ok = unsafe {
        GetThreadTimes(
            GetCurrentThread(),
            &raw mut created,
            &raw mut exited,
            &raw mut kernel,
            &raw mut user,
        )
    };
    assert_ne!(ok, 0, "GetThreadTimes failed");
    ticks(&kernel) + ticks(&user)
}

/// The kernel and user CPU time the calling thread has used, in nanoseconds (Windows only; zero elsewhere).
#[cfg(not(windows))]
pub fn thread_cpu_ns() -> u64 {
    0
}

/// The middle of three values.
pub fn median(mut values: [f64; 3]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[1]
}

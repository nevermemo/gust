// A map argument is keyed on the function's host parameter type, so a `u32` parameter does not take `bool`s even though
// both are `u32` words in a buffer: `add_one::map(&flags)` is rustc's E0277 (`&[bool; 2]` does not implement the
// `MapArg<u32, _>` bound). The other direction is map-u32-for-bool. Built under `GUST_PROFILE=cpu` (env.txt).

#[gust::gpu]
pub fn add_one(n: u32) -> u32 {
    n + 1u32
}

fn main() {
    let flags = [true, false];
    let _ = add_one::map(&flags);
}

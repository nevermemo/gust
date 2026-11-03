// The mirror of map-bool-for-u32: `even_and` takes a `bool` flag, and the `&[2u32, 5]` given for it is rustc's E0277
// (`&[u32; 2]` does not implement the `MapArg<bool, _>` bound). Built under `GUST_PROFILE=cpu` (env.txt).

#[gust::gpu]
pub fn even_and(n: u32, flag: bool) -> bool {
    n % 2u32 == 0u32 && flag
}

fn main() {
    let n = [4u32, 6];
    let _ = even_and::map(&n, &[2u32, 5]);
}

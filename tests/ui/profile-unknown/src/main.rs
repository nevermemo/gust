// `env.txt` sets GUST_PROFILE to a value that is no profile. `gust`'s build script refuses it, naming the three
// profiles, before rustc compiles anything of this crate (expected.txt holds the exact text).

fn main() {}

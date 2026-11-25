// A free-standing enum with `#[derive(GpuEnum)]` compiles, and the word conversions are what the discriminants say
// (D157, D270). Explicit discriminants, a gap and an implicit one after an explicit one are all accepted.

use gust::GpuEnum;

#[derive(Clone, Copy, Debug, PartialEq, Eq, GpuEnum)]
#[repr(u32)]
enum Mode {
    Linear = 0,
    Curve = 1,
    Step = 5,
    After,
}

fn main() {
    assert_eq!(Mode::After.to_word(), 6);
    assert_eq!(Mode::from_word(5), Some(Mode::Step));
    assert_eq!(Mode::from_word(2), None);
    assert_eq!(Mode::try_from(1), Ok(Mode::Curve));
    assert_eq!(Mode::try_from(u32::MAX).unwrap_err().word(), u32::MAX);
    let _ = Mode::Linear;
}

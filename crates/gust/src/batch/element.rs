//! What a batch's data calls take as an element (plan m14.s2, slice s2e2c, D275).
//!
//! A device stores a buffer as plain words ([`GpuPod`]); a `#[derive(GpuEnum)]` enum is not one (D157: a bit pattern that
//! is no variant must never become a value of the enum). [`Element`] is the type of the data calls ([`Builder::input`],
//! [`FrameRun::write`], [`FrameRun::ticket`], `Blocking::read`): a [`GpuPod`] type is its own element and passes through
//! as it is, and an enum is an element whose word is a `u32`, packed on the way in and checked on the way out.
//!
//! [`Builder::input`]: super::Builder::input
//! [`FrameRun::write`]: super::FrameRun::write
//! [`FrameRun::ticket`]: super::FrameRun::ticket

use std::borrow::Cow;

use crate::{GpuEnum, GpuPod};

/// A word of a read-back that is no variant of the enum the handle holds: the lowest such index (row-major for a 2-D
/// handle) and the word. [`Error::InvalidEnum`](super::Error::InvalidEnum) carries both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidWord {
    index: usize,
    word: u32,
}

impl InvalidWord {
    /// The index of the first word that is no variant.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }

    /// That word.
    #[must_use]
    pub const fn word(self) -> u32 {
        self.word
    }
}

/// The type of a batch's data calls: what a handle holds as seen from the host.
///
/// Every [`GpuPod`] type is its own element (the words are the values, no copy is made). A `#[derive(GpuEnum)]` enum is an
/// element too, and the derive writes its impl: the device holds a `u32` per element, `write` packs the enum into words
/// (it cannot fail), and a read-back is checked word by word, so an enum value on the host is always a variant (see
/// [`Error::InvalidEnum`](super::Error::InvalidEnum)). Implement it with the derive, never by hand.
pub trait Element: Copy + 'static {
    /// What the device stores for one element.
    type Word: GpuPod;

    /// The words of `data`: borrowed when the element is its own word, packed otherwise.
    fn pack(data: &[Self]) -> Cow<'_, [Self::Word]>;

    /// The elements of a read-back.
    ///
    /// # Errors
    ///
    /// The first word that is no value of the element.
    fn unpack(words: Vec<Self::Word>) -> Result<Vec<Self>, InvalidWord>;
}

impl<T: GpuPod> Element for T {
    type Word = T;

    fn pack(data: &[Self]) -> Cow<'_, [T]> {
        Cow::Borrowed(data)
    }

    fn unpack(words: Vec<T>) -> Result<Vec<Self>, InvalidWord> {
        Ok(words)
    }
}

/// The words of `data`, for the impl of [`Element`] that `#[derive(GpuEnum)]` writes.
#[doc(hidden)]
#[must_use]
pub fn pack_enum<E: GpuEnum>(data: &[E]) -> Cow<'_, [u32]> {
    Cow::Owned(data.iter().map(|value| value.to_word()).collect())
}

/// The enums of a read-back, for the impl of [`Element`] that `#[derive(GpuEnum)]` writes.
///
/// # Errors
///
/// The lowest index whose word is no variant, with the word.
#[doc(hidden)]
pub fn unpack_enum<E: GpuEnum>(words: Vec<u32>) -> Result<Vec<E>, InvalidWord> {
    words
        .iter()
        .enumerate()
        .map(|(index, &word)| E::from_word(word).ok_or(InvalidWord { index, word }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Element, InvalidWord, unpack_enum};
    use crate::GpuEnum;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, GpuEnum)]
    #[repr(u32)]
    enum Mode {
        Off = 0,
        On = 1,
        Auto = 5,
    }

    #[test]
    fn an_enum_packs_to_its_words_and_a_pod_type_is_borrowed() {
        let words = Mode::pack(&[Mode::Auto, Mode::Off, Mode::On]);
        assert_eq!(&*words, &[5, 0, 1]);
        let floats = [1.0_f32, 2.0];
        assert!(matches!(
            <f32 as Element>::pack(&floats),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn a_read_back_of_variants_unpacks_in_order() {
        assert_eq!(
            Mode::unpack(vec![1, 5, 0, 1]),
            Ok(vec![Mode::On, Mode::Auto, Mode::Off, Mode::On])
        );
        assert_eq!(Mode::unpack(Vec::new()), Ok(Vec::new()));
        assert_eq!(<u32 as Element>::unpack(vec![7, 9]), Ok(vec![7, 9]));
    }

    #[test]
    fn the_lowest_bad_index_and_its_word_are_reported() {
        let error = Mode::unpack(vec![0, 1, 2, 9, 5, 7]).unwrap_err();
        assert_eq!((error.index(), error.word()), (2, 2));
        let error = unpack_enum::<Mode>(vec![5, 6]).unwrap_err();
        assert_eq!(error, InvalidWord { index: 1, word: 6 });
    }

    /// The device side of the host rule: an input of enums is written as words (the macro's `words()` handle is the
    /// only way to put a word that is no variant in), and a read of it checks every word.
    #[cfg(not(target_arch = "wasm32"))]
    mod on_the_twin {
        use std::future::Future;
        use std::sync::Arc;
        use std::task::{Context, Poll, Waker};

        use super::Mode;
        use crate::batch::{Batch, D1, D2, Error, Stage, build};
        use crate::cpu::CpuDevice;

        fn block_on<F: Future>(future: F) -> F::Output {
            let mut future = core::pin::pin!(future);
            let mut context = Context::from_waker(Waker::noop());
            loop {
                if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                    return value;
                }
                std::thread::yield_now();
            }
        }

        struct Held {
            line: Stage<Mode, D1>,
            grid: Stage<Mode, D2>,
        }

        fn batch() -> (Batch<CpuDevice>, Held) {
            #[allow(clippy::arc_with_non_send_sync)]
            let device = Arc::new(CpuDevice::new());
            build("elements", &device, 1, |b| {
                let line = b.input::<Mode, _>(4);
                let grid = b.input::<Mode, _>((3, 2));
                b.expose(&line);
                b.expose(&grid);
                Held { line, grid }
            })
            .unwrap()
        }

        /// One frame that writes `line` and `grid` as words and reads both back through tickets.
        fn read_back(
            batch: &mut Batch<CpuDevice>,
            held: &Held,
            line: &[u32],
            grid: &[u32],
        ) -> (Result<Vec<Mode>, Error>, Result<Vec<Mode>, Error>) {
            block_on(async {
                let mut frame = batch.begin().await.unwrap();
                frame.write(&raw(&held.line), line).unwrap();
                frame.write(&raw(&held.grid), grid).unwrap();
                let first = frame.ticket(&held.line).unwrap();
                let second = frame.ticket(&held.grid).unwrap();
                let _submitted = frame.submit().unwrap();
                (first.await, second.await)
            })
        }

        /// The handle a test writes words through: what `words()` hands a kernel is refused by `write`, so the test
        /// builds the plain `u32` handle of the same buffer, which only this module can do.
        fn raw<R>(handle: &Stage<Mode, R>) -> Stage<u32, R> {
            Stage::new(handle.batch, handle.id, handle.dims)
        }

        #[test]
        fn the_words_handle_of_a_kernel_is_no_handle_to_write() {
            let (mut batch, held) = batch();
            let refused = block_on(async {
                let mut frame = batch.begin().await.unwrap();
                frame.write(&held.line.words(), &[0, 1, 5, 0])
            });
            assert!(
                matches!(refused, Err(Error::NotInput { .. })),
                "{refused:?}"
            );
        }

        #[test]
        fn a_retyped_words_handle_is_still_no_handle_to_write() {
            let (mut batch, held) = batch();
            let refused = block_on(async {
                let mut frame = batch.begin().await.unwrap();
                frame.write(&held.line.words().retype::<Mode>(), &[Mode::Off; 4])
            });
            assert!(
                matches!(refused, Err(Error::NotInput { .. })),
                "{refused:?}"
            );
        }

        #[test]
        fn a_ticket_of_variants_resolves_to_the_enums() {
            let (mut batch, held) = batch();
            let (line, grid) = read_back(&mut batch, &held, &[0, 1, 5, 0], &[5, 5, 1, 0, 0, 1]);
            assert_eq!(line.unwrap(), [Mode::Off, Mode::On, Mode::Auto, Mode::Off]);
            assert_eq!(
                grid.unwrap(),
                [
                    Mode::Auto,
                    Mode::Auto,
                    Mode::On,
                    Mode::Off,
                    Mode::Off,
                    Mode::On
                ]
            );
        }

        #[test]
        fn a_ticket_with_a_bad_word_is_invalid_enum_with_the_lowest_index() {
            let (mut batch, held) = batch();
            let (line, grid) = read_back(&mut batch, &held, &[0, 3, 1, 9], &[5, 1, 0, 0, 1, 0]);
            match line {
                Err(Error::InvalidEnum {
                    batch,
                    stage,
                    index,
                    word,
                }) => {
                    assert_eq!((batch.as_str(), index, word), ("elements", 1, 3));
                    assert!(!stage.is_empty());
                }
                other => panic!("expected InvalidEnum, got {other:?}"),
            }
            // The other ticket of the same frame still resolves, and the batch takes the next frame.
            assert_eq!(grid.unwrap().len(), 6);
            let (line, _) = read_back(&mut batch, &held, &[1, 1, 1, 1], &[0; 6]);
            assert_eq!(line.unwrap(), [Mode::On; 4]);
        }

        #[test]
        fn a_two_dimensional_handle_reports_the_row_major_index() {
            let (mut batch, held) = batch();
            // Two bad words, 7 at row-major index 4 and 2 at index 5: the lowest is reported.
            let (_, grid) = read_back(&mut batch, &held, &[0; 4], &[0, 0, 0, 0, 7, 2]);
            assert!(matches!(
                grid,
                Err(Error::InvalidEnum {
                    index: 4,
                    word: 7,
                    ..
                })
            ));
        }

        #[test]
        fn a_blocking_read_checks_the_words_too() {
            let (mut batch, held) = batch();
            block_on(async {
                let mut frame = batch.begin().await.unwrap();
                frame.write(&raw(&held.line), &[1, 5, 4, 0]).unwrap();
                let _submitted = frame.submit().unwrap();
            });
            let after = batch.dispatch().unwrap();
            assert!(matches!(
                after.read(&held.line),
                Err(Error::InvalidEnum {
                    index: 2,
                    word: 4,
                    ..
                })
            ));
        }
    }
}

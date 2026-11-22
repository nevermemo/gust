//! Test support: each generated batch is built once per test binary and shared by every test that asks for it.
//!
//! The generator is deterministic in `(space, seed)`, and the slow tests of the trait, closure and array spaces read
//! the same seed ranges again and again. Generating a batch is the costly part and the tests only read it, so the
//! first caller generates it and every other caller waits for that result. Tests keep their own seeds, ranges and
//! assertions.

use std::sync::{Mutex, OnceLock};

use crate::generator::{Batch, Space, generate_batch, generate_batch_in};

/// `None` is the full batch of `generate_batch`; `Some(space)` is `generate_batch_in(space, ..)`.
type Key = (Option<Space>, u64);

type Slot = &'static OnceLock<Batch>;

/// The batch for `key`. The table lock is held only to find or add the slot, so batches of different keys are
/// generated in parallel by different tests while the callers of one key wait for the same generation.
fn shared(key: Key, generate: impl FnOnce() -> Result<Batch, String>) -> &'static Batch {
    static SLOTS: Mutex<Vec<(Key, Slot)>> = Mutex::new(Vec::new());
    let slot: Slot = {
        let mut slots = SLOTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match slots.iter().find(|(k, _)| *k == key) {
            Some((_, slot)) => slot,
            None => {
                let slot: Slot = Box::leak(Box::new(OnceLock::new()));
                slots.push((key, slot));
                slot
            }
        }
    };
    slot.get_or_init(|| generate().unwrap_or_else(|e| panic!("generating {key:?} failed: {e}")))
}

/// The batch of `generate_batch(seed)`.
pub(crate) fn full_batch(seed: u64) -> &'static Batch {
    shared((None, seed), || generate_batch(seed))
}

/// The batch of `generate_batch_in(space, seed)`.
pub(crate) fn space_batch(space: Space, seed: u64) -> &'static Batch {
    shared((Some(space), seed), || generate_batch_in(space, seed))
}

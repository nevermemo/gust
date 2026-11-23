//! Helpers of the unit tests of `check-precompiled`: the committed fixture, read from the repository.

use std::fs;

use super::PRECOMPILED_DIR;
use crate::workspace::workspace_root;

pub(super) fn fixture_file(relative: &str) -> String {
    let path = workspace_root()
        .unwrap()
        .join(PRECOMPILED_DIR)
        .join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{PRECOMPILED_DIR}/{relative}: {error}"))
}

use std::collections::BTreeSet;

use super::{
    UNIFORM_GOLDEN_SPIRV, UNIFORM_GOLDEN_STEMS, UNIFORM_MACRO_FILTERS, VECTOR_GOLDEN_SPIRV,
    VECTOR_GOLDEN_STEMS, VECTOR_MACRO_FILTERS, explanation,
};
use crate::{
    shader::{compute_entries, golden_file_name},
    workspace::FEATURE_AREAS,
};

/// The source of `checks.rs`, read as text: `check_feature` is a `match` over the area name that runs commands, so
/// the areas it accepts can only be listed without running it by reading its arms.
const CHECKS_SOURCE: &str = include_str!("../checks.rs");

/// The area names of the arms of `check_feature`: a line of eight spaces, a quoted name and `=>`.
fn accepted_areas() -> Vec<String> {
    let source = CHECKS_SOURCE.replace("\r\n", "\n");
    let body = source
        .split_once("pub(crate) fn check_feature(")
        .expect("check_feature is in checks.rs")
        .1;
    // The function ends at the first closing brace in column 0.
    let body = body.split_once("\n}\n").expect("check_feature ends").0;
    body.lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("        \"")?;
            let (name, after) = rest.split_once('"')?;
            let is_name = !name.is_empty()
                && name.bytes().all(|b| b.is_ascii_lowercase() || b == b'-')
                && after.trim_start().starts_with("=>");
            is_name.then(|| name.to_owned())
        })
        .collect()
}

#[test]
fn every_area_check_feature_accepts_is_listed_and_every_listed_area_is_accepted() {
    let accepted = accepted_areas();
    // The reader must find the arms, or the comparison below proves nothing.
    assert!(accepted.len() >= 15, "{accepted:?}");
    let accepted_set: BTreeSet<&str> = accepted.iter().map(String::as_str).collect();
    assert_eq!(accepted_set.len(), accepted.len(), "an arm is repeated");
    let listed: BTreeSet<&str> = FEATURE_AREAS.iter().copied().collect();
    assert_eq!(listed.len(), FEATURE_AREAS.len(), "an area is listed twice");
    let unlisted: Vec<&&str> = accepted_set.difference(&listed).collect();
    assert!(
        unlisted.is_empty(),
        "check-feature accepts areas that FEATURE_AREAS (the usage text) does not list: {unlisted:?}"
    );
    let unaccepted: Vec<&&str> = listed.difference(&accepted_set).collect();
    assert!(
        unaccepted.is_empty(),
        "FEATURE_AREAS lists areas that check-feature refuses: {unaccepted:?}"
    );
}

#[test]
fn the_areas_of_m6_and_m7_are_listed() {
    for area in ["vectors", "uniforms", "map"] {
        assert!(FEATURE_AREAS.contains(&area), "{area}");
    }
}

#[test]
fn every_listed_area_is_explained_and_named_by_the_overview() {
    let overview = explanation(None).expect("the overview exists");
    let known = overview
        .strip_prefix("known areas: ")
        .expect("the overview starts with the known areas");
    let known: BTreeSet<&str> = known.split(", ").collect();
    for area in FEATURE_AREAS {
        let text = explanation(Some(area)).unwrap_or_else(|error| panic!("{area}: {error}"));
        assert!(text.starts_with(&format!("{area}: ")), "{area}: {text}");
        assert!(
            known.contains(area),
            "explain-check's overview omits {area}"
        );
    }
}

#[test]
fn the_filters_are_distinct_lowercase_words() {
    for filters in [&VECTOR_MACRO_FILTERS[..], &UNIFORM_MACRO_FILTERS[..]] {
        let distinct: BTreeSet<&&str> = filters.iter().collect();
        assert_eq!(distinct.len(), filters.len(), "{filters:?}");
        assert!(
            filters.iter().all(|f| !f.is_empty()
                && f.bytes()
                    .all(|b| b.is_ascii_lowercase() || b == b'_' || b.is_ascii_digit())),
            "{filters:?}"
        );
    }
}

/// The SPIR-V names the goldens give, read from the goldens themselves (no slangc): they must equal the pinned names.
fn names_of(stems: &[&str]) -> Vec<String> {
    let mut names = Vec::new();
    for stem in stems {
        let text = match *stem {
            "vectors" => include_str!("../../../tests/fixtures/vectors.slang"),
            "products" => include_str!("../../../tests/fixtures/products.slang"),
            "uniforms" => include_str!("../../../tests/fixtures/uniforms.slang"),
            other => panic!("the test does not know the golden {other}"),
        };
        let entries = compute_entries(text);
        assert!(!entries.is_empty(), "{stem} has no compute entry");
        for entry in entries {
            names.push(golden_file_name(stem, &entry).unwrap());
        }
    }
    names.sort();
    names
}

#[test]
fn the_pinned_golden_spirv_names_are_the_ones_the_goldens_produce() {
    assert_eq!(names_of(&VECTOR_GOLDEN_STEMS), VECTOR_GOLDEN_SPIRV);
    assert_eq!(names_of(&UNIFORM_GOLDEN_STEMS), UNIFORM_GOLDEN_SPIRV);
    for expected in [&VECTOR_GOLDEN_SPIRV[..], &UNIFORM_GOLDEN_SPIRV[..]] {
        assert!(expected.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(expected.iter().all(|name| name.ends_with(".spv")));
    }
}

use std::collections::BTreeSet;

use super::{
    GOLDEN_SPIRV_SETS, UNIFORM_GOLDEN_SPIRV, UNIFORM_GOLDEN_STEMS, UNIFORM_MACRO_FILTERS,
    VECTOR_GOLDEN_SPIRV, VECTOR_GOLDEN_STEMS, VECTOR_MACRO_FILTERS, explanation,
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
        .filter_map(|line| match parse_arm(line) {
            Ok(name) => name,
            Err(error) => panic!("{error}"),
        })
        .collect()
}

/// The area an arm line names: `Ok(Some(name))` for `        "name" =>`, `Ok(None)` for any other line, and an error
/// for an arm that lists alternatives (`"a" | "b" =>`), which this reader would otherwise skip silently and so miss
/// an accepted area.
fn parse_arm(line: &str) -> Result<Option<String>, String> {
    let Some(rest) = line.strip_prefix("        \"") else {
        return Ok(None);
    };
    let Some((name, after)) = rest.split_once('"') else {
        return Ok(None);
    };
    let is_name = !name.is_empty() && name.bytes().all(|b| b.is_ascii_lowercase() || b == b'-');
    if !is_name {
        return Ok(None);
    }
    let after = after.trim_start();
    if after.starts_with("=>") {
        Ok(Some(name.to_owned()))
    } else if after.starts_with('|') {
        Err(format!(
            "check_feature has an arm with alternatives (`{line}`): write one arm per area or teach accepted_areas"
        ))
    } else {
        Ok(None)
    }
}

#[test]
fn an_arm_with_alternatives_is_refused_instead_of_skipped() {
    assert_eq!(
        parse_arm("        \"loops\" => run(),"),
        Ok(Some("loops".to_owned()))
    );
    assert_eq!(
        parse_arm("        \"gpu-smoke\" => {"),
        Ok(Some("gpu-smoke".to_owned()))
    );
    let error = parse_arm("        \"a\" | \"b\" => run(),").unwrap_err();
    assert!(error.contains("alternatives"), "{error}");
    // Not arms: another indentation, a string that is not a name, a call argument.
    assert_eq!(parse_arm("            \"loops\" => 1,"), Ok(None));
    assert_eq!(parse_arm("        \"Not A Name\" => 1,"), Ok(None));
    assert_eq!(parse_arm("        \"unknown feature area\","), Ok(None));
    assert_eq!(parse_arm("        \"loops\","), Ok(None));
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
    for area in ["vectors", "uniforms", "map", "algo"] {
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
    // The sets `verify` (check-artifacts, the Spirv suite, check-full) exports are these two, and no others.
    let areas: Vec<&str> = GOLDEN_SPIRV_SETS.iter().map(|(area, _, _)| *area).collect();
    assert_eq!(areas, ["vectors", "uniforms"]);
    for (_, stems, expected) in GOLDEN_SPIRV_SETS {
        assert_eq!(names_of(stems), expected);
    }
    for expected in [&VECTOR_GOLDEN_SPIRV[..], &UNIFORM_GOLDEN_SPIRV[..]] {
        assert!(expected.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(expected.iter().all(|name| name.ends_with(".spv")));
    }
}

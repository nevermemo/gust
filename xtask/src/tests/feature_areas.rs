use std::collections::BTreeSet;

use super::{
    COMPACT_GOLDEN_SPIRV, COMPACT_GOLDEN_STEMS, DISPATCH_CORE_FILTERS, DISPATCH_GPU_TESTS,
    DISPATCH_MACRO_MODULES, GOLDEN_SPIRV_SETS, GRID_GOLDEN_SPIRV, GRID_GOLDEN_STEMS,
    MAP_GOLDEN_SPIRV, MAP_GOLDEN_STEMS, PAR_FOR_GOLDEN_SPIRV, PAR_FOR_GOLDEN_STEMS,
    PIXEL_GOLDEN_SPIRV, PIXEL_GOLDEN_STEMS, REDUCE_GOLDEN_SPIRV, REDUCE_GOLDEN_STEMS,
    SCAN_GOLDEN_SPIRV, SCAN_GOLDEN_STEMS, SUBGROUP_GOLDEN_SPIRV, SUBGROUP_GOLDEN_STEMS,
    UNIFORM_GOLDEN_SPIRV, UNIFORM_GOLDEN_STEMS, UNIFORM_MACRO_FILTERS, VARIANT_GOLDEN_SPIRV,
    VARIANT_GOLDEN_STEMS, VECTOR_GOLDEN_SPIRV, VECTOR_GOLDEN_STEMS, VECTOR_MACRO_FILTERS,
    WORKGROUP_GOLDEN_SPIRV, WORKGROUP_GOLDEN_STEMS, explanation,
};
use crate::{
    shader::{golden_file_name, shader_entries},
    workspace::{FEATURE_AREAS, workspace_root},
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
    for area in ["vectors", "uniforms", "map", "dispatch", "algo"] {
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

/// The golden `tests/fixtures/<stem>.slang`, read at compile time (no slangc).
fn golden_text(stem: &str) -> &'static str {
    match stem {
        "vectors" => include_str!("../../../tests/fixtures/vectors.slang"),
        "products" => include_str!("../../../tests/fixtures/products.slang"),
        "uniforms" => include_str!("../../../tests/fixtures/uniforms.slang"),
        "map" => include_str!("../../../tests/fixtures/map.slang"),
        "grid" => include_str!("../../../tests/fixtures/grid.slang"),
        "workgroup" => include_str!("../../../tests/fixtures/workgroup.slang"),
        "par_for" => include_str!("../../../tests/fixtures/par_for.slang"),
        "reduce" => include_str!("../../../tests/fixtures/reduce.slang"),
        "scan" => include_str!("../../../tests/fixtures/scan.slang"),
        "compact" => include_str!("../../../tests/fixtures/compact.slang"),
        "textures" => include_str!("../../../tests/fixtures/textures.slang"),
        "pixel" => include_str!("../../../tests/fixtures/pixel.slang"),
        "variant" => include_str!("../../../tests/fixtures/variant.slang"),
        "subgroup" => include_str!("../../../tests/fixtures/subgroup.slang"),
        other => panic!("the test does not know the golden {other}"),
    }
}

/// The SPIR-V names the goldens give, read from the goldens themselves (no slangc): they must equal the pinned names.
fn names_of(stems: &[&str]) -> Vec<String> {
    let mut names = Vec::new();
    for stem in stems {
        let entries = shader_entries(golden_text(stem));
        assert!(
            !entries.is_empty(),
            "{stem} has no compute or fragment entry"
        );
        for (entry, _) in entries {
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
    assert_eq!(names_of(&MAP_GOLDEN_STEMS), MAP_GOLDEN_SPIRV);
    assert_eq!(names_of(&GRID_GOLDEN_STEMS), GRID_GOLDEN_SPIRV);
    assert_eq!(names_of(&WORKGROUP_GOLDEN_STEMS), WORKGROUP_GOLDEN_SPIRV);
    assert_eq!(names_of(&PAR_FOR_GOLDEN_STEMS), PAR_FOR_GOLDEN_SPIRV);
    assert_eq!(names_of(&REDUCE_GOLDEN_STEMS), REDUCE_GOLDEN_SPIRV);
    assert_eq!(names_of(&SCAN_GOLDEN_STEMS), SCAN_GOLDEN_SPIRV);
    assert_eq!(names_of(&COMPACT_GOLDEN_STEMS), COMPACT_GOLDEN_SPIRV);
    assert_eq!(names_of(&PIXEL_GOLDEN_STEMS), PIXEL_GOLDEN_SPIRV);
    assert_eq!(names_of(&VARIANT_GOLDEN_STEMS), VARIANT_GOLDEN_SPIRV);
    assert_eq!(names_of(&SUBGROUP_GOLDEN_STEMS), SUBGROUP_GOLDEN_SPIRV);
    // The sets `verify` (check-artifacts, the Spirv suite, check-full) exports are these, and no others (m7.x1 added
    // the goldens of M7 to the vectors and uniforms of M6).
    let areas: Vec<&str> = GOLDEN_SPIRV_SETS.iter().map(|(area, _, _)| *area).collect();
    assert_eq!(
        areas,
        [
            "vectors",
            "uniforms",
            "map",
            "grid",
            "workgroup",
            "par_for",
            "reduce",
            "scan",
            "compact",
            "textures",
            "pixel",
            "variant",
            "subgroup"
        ]
    );
    for (_, stems, expected) in GOLDEN_SPIRV_SETS {
        assert_eq!(names_of(stems), expected);
        assert!(expected.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(expected.iter().all(|name| name.ends_with(".spv")));
    }
}

#[test]
fn every_golden_of_m7_is_in_a_set_and_the_file_names_are_unique() {
    let stems: Vec<&str> = GOLDEN_SPIRV_SETS
        .iter()
        .flat_map(|(_, stems, _)| stems.iter().copied())
        .collect();
    for golden in [
        "map",
        "grid",
        "workgroup",
        "par_for",
        "reduce",
        "scan",
        "compact",
    ] {
        assert!(
            stems.contains(&golden),
            "{golden} is not exported by verify"
        );
    }
    let mut names: Vec<&str> = GOLDEN_SPIRV_SETS
        .iter()
        .flat_map(|(_, _, expected)| expected.iter().copied())
        .collect();
    let count = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), count, "two sets expect the same file name");
    // 2 (vectors, products) + 1 + 5 + 3 + 2 + 10 + 30 + 11 + 5 + 8 (textures) + 3 (pixel) + 2 (variant) + 12 (subgroup): one file per compute or fragment entry of the
    // goldens.
    assert_eq!(count, 94);
}

// ---- `check-feature dispatch` and the parallel-map additions to `map` (m7.x1) ----

/// The body of the arm `area` of `check_feature`, up to the next arm of the list `next`.
fn arm_body(area: &str, next: &str) -> String {
    let source = CHECKS_SOURCE.replace("\r\n", "\n");
    let body = source
        .split_once("pub(crate) fn check_feature(")
        .expect("check_feature is in checks.rs")
        .1;
    let start = body
        .split_once(&format!("\n        \"{area}\" => {{\n"))
        .unwrap_or_else(|| panic!("no arm for {area}"))
        .1;
    start
        .split_once(&format!("\n        \"{next}\" => "))
        .unwrap_or_else(|| panic!("no arm {next} after {area}"))
        .0
        .to_owned()
}

#[test]
fn the_dispatch_area_runs_every_dispatch_test_and_counts_each_call() {
    let arm = arm_body("dispatch", "workgroup");
    for constant in [
        "DISPATCH_MACRO_MODULES",
        "DISPATCH_CORE_FILTERS",
        "DISPATCH_GPU_TESTS",
    ] {
        assert!(arm.contains(constant), "the arm does not use {constant}");
    }
    // Counted, so that a call that runs no test fails; an uncounted call would pass with nothing.
    assert!(!arm.contains("run_direct"), "{arm}");
    assert_eq!(arm.matches("run_counted").count(), 3, "{arm}");
    // The lists name the tests of the brief: the `par_for!` and grid unit tests of the macros, the gust tests of
    // Extent, and the four GPU files.
    assert_eq!(DISPATCH_MACRO_MODULES, ["par_for_tests::", "grid_tests::"]);
    assert_eq!(
        DISPATCH_CORE_FILTERS,
        [
            "dispatch::",
            "cpu::tests::a_shared_buffer_holds_the_product_of_the_extent",
            "cpu::tests::invocations_cover_whole_workgroups_in_two_and_three_dimensions"
        ]
    );
    // Each filter is its own counted call (no ORed `--` list): a filter that matches nothing fails by itself.
    assert!(arm.contains("for filter in DISPATCH_CORE_FILTERS"), "{arm}");
    assert!(!arm.contains("core.extend"), "{arm}");
    assert_eq!(
        DISPATCH_GPU_TESTS,
        ["par_for", "grid", "concurrency", "async_readback"]
    );
}

#[test]
fn the_dispatch_tests_exist_where_the_area_looks_for_them() {
    let root = workspace_root().unwrap();
    let macros = std::fs::read_to_string(root.join("crates/gust-macros/src/lib.rs"))
        .unwrap()
        .replace("\r\n", "\n");
    for module in DISPATCH_MACRO_MODULES {
        let module = module.trim_end_matches("::");
        assert!(macros.contains(&format!("mod {module};")), "{module}");
        let file = module.trim_end_matches("_tests");
        assert!(
            macros.contains(&format!("#[path = \"tests/{file}.rs\"]")),
            "{module}"
        );
    }
    for test in DISPATCH_GPU_TESTS {
        let path = root.join(format!("crates/gust-wgpu/tests/{test}.rs"));
        assert!(path.is_file(), "{}", path.display());
    }
    // The gust tests of the filters: the dispatch module has tests of its own, and the two named CPU tests exist
    // under the names the filters give.
    let dispatch = std::fs::read_to_string(root.join("crates/gust/src/dispatch.rs")).unwrap();
    assert!(dispatch.contains("mod tests"));
    let cpu = std::fs::read_to_string(root.join("crates/gust/src/cpu.rs")).unwrap();
    for filter in &DISPATCH_CORE_FILTERS[1..] {
        let name = filter.strip_prefix("cpu::tests::").unwrap();
        assert!(cpu.contains(&format!("fn {name}(")), "{name}");
    }
}

#[test]
fn the_map_area_counts_the_parallel_cpu_map_tests() {
    // rustfmt breaks a call that grows past the line width over several lines, so compare without whitespace and
    // without the trailing comma it adds.
    let squash = |text: &str| {
        text.split_whitespace()
            .collect::<String>()
            .replace(",)", ")")
    };
    let arm = squash(&arm_body("map", "dispatch"));
    for call in [
        "&[\"test\", \"-p\", \"gust-compute\", \"--lib\", \"cpu::parallel\"]",
        "&[\"test\", \"-p\", \"gust-compute\", \"--test\", \"parallel\"]",
    ] {
        let line = format!("run_counted(&root, {call})?;");
        assert!(arm.contains(&squash(&line)), "the map arm lacks {line}");
    }
    let root = workspace_root().unwrap();
    assert!(root.join("crates/gust/tests/parallel.rs").is_file());
    let parallel = std::fs::read_to_string(root.join("crates/gust/src/cpu.rs")).unwrap();
    assert!(parallel.contains("pub mod parallel;"));
}

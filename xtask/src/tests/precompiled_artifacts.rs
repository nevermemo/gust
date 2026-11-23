use super::{super::test_support::fixture_file, *};

#[test]
fn the_mutation_changes_the_first_plus_only() {
    let wgsl = b"a_0[i] + b_0[i] + c";
    let mutated = mutate_wgsl(wgsl).unwrap();
    assert_eq!(mutated, b"a_0[i] - b_0[i] + c");
    assert_eq!(mutated.len(), wgsl.len());
    assert_ne!(mutated, wgsl);
    assert!(
        mutate_wgsl(b"no operator here")
            .unwrap_err()
            .contains("no `+`")
    );
}

const IDENTITY: &str = "error: dispatch failed: could not validate kernel reflection: native target, entry point, source or artifact identity mismatch: artifact hash (stored \"0123\", expected \"4567\")";

#[test]
fn the_identity_error_must_name_the_artifact_hash_alone_and_the_app_must_not_verify() {
    assert_eq!(
        identity_error(false, "precompiled vector-add\n", &format!("{IDENTITY}\n")).as_deref(),
        Ok(IDENTITY)
    );

    // The mutation did nothing: the app ran and verified.
    let verified = format!("{VERIFIED_PREFIX}1000 elements\n");
    let problems = identity_error(true, &verified, "").unwrap_err();
    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(problems[0].contains("accepted"));
    assert!(problems[1].contains("verified"));
    assert!(problems[2].contains("no identity error"));

    // Failed, but with another error.
    let other = identity_error(false, "", "error: no GPU adapter: none\n").unwrap_err();
    assert_eq!(other.len(), 1);
    assert!(other[0].contains("no GPU adapter"));

    // More than the artifact hash differs: the mutation reached something else.
    let more = IDENTITY.replace(
        "artifact hash (stored \"0123\"",
        "artifact hash (stored \"0123\", expected \"4567\"); source hash (stored \"a\"",
    );
    let problems = identity_error(false, "", &more).unwrap_err();
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("source hash ("))
    );
}

#[test]
fn the_committed_e4_expectation_is_one_e0080_at_the_kernel_line() {
    let wanted = parse_expected(&fixture_file(E4_EXPECTED)).unwrap();
    assert_eq!(wanted.code, "E0080");
    assert_eq!(wanted.file, "kernels/src/lib.rs");
    let source = fixture_file("kernels/src/lib.rs");
    let line = usize::try_from(wanted.line).unwrap();
    assert!(
        source
            .lines()
            .nth(line - 1)
            .unwrap()
            .contains("pub fn add("),
        "line {line} of kernels/src/lib.rs is not the kernel's `fn`"
    );
    assert_eq!(wanted.messages.len(), 1);
    assert!(wanted.messages[0].contains("needs a precompiled artifact for this kernel"));
}

#[test]
fn the_committed_artifacts_are_the_ones_e3_mutates_and_e4_deletes() {
    let manifest = fixture_file("kernels/gust-artifacts/manifest.json");
    // Schema 4 since D209: the kernel's split is a second entry that names the body it belongs to (`split_of`).
    assert!(manifest.contains("\"schema\": 4") && manifest.contains("\"format\": \"wgsl\""));
    let key = "52e85c5f03f72272";
    assert!(
        manifest.contains(&format!("\"split_of\": \"{key}\"")),
        "the manifest links the split to the body E3 mutates"
    );
    // Only the split's entry has a `split_of`; the body's own entry, which E3 and E4 act on, has none.
    assert_eq!(manifest.matches("\"split_of\"").count(), 1);
    let wgsl = fixture_file(&format!("kernels/gust-artifacts/{key}.wgsl"));
    assert!(
        mutate_wgsl(wgsl.as_bytes()).is_ok(),
        "E3 needs a `+` to change"
    );
    let reflection = fixture_file(&format!(
        "kernels/gust-artifacts/{key}.wgsl.reflection.json"
    ));
    assert!(reflection.contains("artifactHash"));
}

#[test]
fn the_fixture_app_has_the_observer_that_e5_needs() {
    let lib = fixture_file("app/src/lib.rs");
    assert!(lib.contains("#[cfg(gust_dual)]") && lib.contains("fn twin_smoke"));
    assert!(
        crate::checks::PRECOMPILED_TWIN_TESTS
            .iter()
            .all(|name| lib.contains(name.rsplit("::").next().unwrap()))
    );
}

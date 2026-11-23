//! Tests of the build profile: injected, never read from the process (m5.e1, R1-9).

use super::*;

#[test]
fn only_the_exact_text_cpu_is_the_cpu_profile() {
    assert_eq!(Profile::from_value(Some("cpu")), Profile::Cpu);
    assert!(Profile::from_value(Some("cpu")).is_cpu());
    // The valid targets keep their name (M8: the profile carries a target and capabilities now); they are not
    // `Cpu`.
    for (value, target) in [
        ("dev", ProfileTarget::Dev),
        ("native", ProfileTarget::Native),
        ("web", ProfileTarget::Web),
    ] {
        let profile = Profile::from_value(Some(value));
        assert_eq!(profile.target, Some(target), "{value}");
        assert_ne!(profile, Profile::Cpu, "{value}");
        assert!(!profile.is_cpu(), "{value}");
    }
    // An unset variable and every value the build script refuses is `Other`, `cpu+simd` among them.
    for other in [
        None,
        Some("CPU"),
        Some("Cpu"),
        Some(" cpu"),
        Some("cpu "),
        Some("cpu+simd"),
        Some("cpu+subgroups"),
        Some(""),
        Some("bogus"),
    ] {
        assert_eq!(Profile::from_value(other), Profile::Other, "{other:?}");
    }
}

#[test]
fn under_cpu_a_complete_pair_is_neither_embedded_nor_tracked() {
    let (scratch, _) = crate_with_pair("cpu-pair-present", true);
    let environment = environment(Some(&scratch.0), None);
    // The control: the same crate directory under every other profile embeds the pair and tracks the manifest.
    let other = expanded_for(KERNEL, &environment, Profile::Other).to_string();
    assert!(other.contains("include_bytes"), "{other}");
    assert!(other.contains("manifest.json"), "{other}");
    let cpu = expanded_for(KERNEL, &environment, Profile::Cpu).to_string();
    assert!(cpu.contains("artifacts : & []"), "{cpu}");
    for needle in [
        "include_bytes",
        "include_str",
        "manifest.json",
        "gust-artifacts",
        "require_manifest",
    ] {
        assert!(!cpu.contains(needle), "`{needle}` in {cpu}");
    }
    // The per-kernel check is still emitted; the build script's profile makes it a no-op under `cpu`.
    assert_eq!(cpu.matches("require_artifact (false)").count(), 1, "{cpu}");
}

#[test]
fn under_cpu_half_a_pair_is_no_error_because_no_artifact_is_looked_at() {
    let (key, _) = exported(KERNEL);
    let half = Scratch::new("cpu-half-pair");
    half.artifact(&format!("{key}.wgsl"), "x");
    let environment = environment(Some(&half.0), None);
    let other = expanded_for(KERNEL, &environment, Profile::Other).to_string();
    assert!(other.contains("[GUST-E0711]"), "{other}");
    let cpu = expanded_for(KERNEL, &environment, Profile::Cpu).to_string();
    assert!(!cpu.contains("compile_error"), "{cpu}");
    assert!(!cpu.contains("E0711"), "{cpu}");
}

#[test]
fn every_expansion_of_every_profile_emits_the_profile_tracking_item() {
    let item = "const _ : :: core :: option :: Option < & :: core :: primitive :: str > = :: core :: option_env ! (\"GUST_PROFILE\") ;";
    for profile in [Profile::Cpu, Profile::Other] {
        for source in [KERNEL, "mod empty {}"] {
            let text = expanded_for(source, &Environment::default(), profile).to_string();
            assert_eq!(text.matches(item).count(), 1, "{profile:?}: {text}");
        }
    }
}

/// D44's rebuild tracking has two mechanisms, and S5.4 measured (probe/s54/s54.log) that each alone suffices to
/// re-expand a kernel crate after `GUST_PROFILE` changes: the `option_env!("GUST_PROFILE")` item every expansion
/// emits (rustc tracks it), and `rerun-if-env-changed=GUST_PROFILE` in the build script of `gust` (the cascade
/// rebuilds every dependent). With both gone an expansion goes stale: a `cpu` build keeps the `dev` expansion. The
/// macro reads the variable with `std::env::var`, which rustc does not track. This guard fails when both are gone;
/// each has its own test besides.
#[test]
fn a_profile_change_is_tracked_by_at_least_one_mechanism() {
    let emitted = [Profile::Cpu, Profile::Other].iter().all(|profile| {
        expanded_for(KERNEL, &Environment::default(), *profile)
            .to_string()
            .contains("option_env ! (\"GUST_PROFILE\")")
    });
    // Only the part before the unit tests: the tests of the script name the directive too.
    let script = include_str!("../../../../gust/build.rs");
    let production = script.split("#[cfg(test)]").next().unwrap();
    assert!(
        production.len() < script.len(),
        "the script has no test module to cut off"
    );
    let rerun = production.contains("\"cargo:rerun-if-env-changed=GUST_PROFILE\"");
    assert!(
        emitted || rerun,
        "neither `option_env!(\"GUST_PROFILE\")` in the expansion nor `rerun-if-env-changed=GUST_PROFILE` in \
             crates/gust/build.rs remains: a profile change would leave stale expansions (S5.4, V3)"
    );
}

//! Names of build-time routes (D65 to D67, D137).
//!
//! A route tag names a group of kernels (`physics`, `physics.broadphase`) whose target a build chooses. The choice
//! reaches the `#[gpu]` macro through an environment variable per tag, `GUST_ROUTE__<package>__<tag>`. This module is
//! the spelling of that variable for the driver (`cargo gust`). The macro cannot call it, because `gust-compute`
//! depends on `gust-compute-macros` and not the other way round: `gust-compute-macros` keeps a token-for-token copy
//! (the same arrangement as `content_key` in `gust-macros/src/expand/artifact.rs`), and both crates assert one golden
//! table of `(package, tag, variable)` triples, so the two spellings cannot drift apart unnoticed. The table is the
//! constant `GOLDEN` of this file's tests; a copy pastes it verbatim.

/// The prefix of every route variable.
pub const VARIABLE_PREFIX: &str = "GUST_ROUTE__";

/// The variable that chooses the target of `tag` in the Cargo package `package`: `GUST_ROUTE__<package>__<tag>`,
/// with `-` in the package name spelled `_` and every `.` of the tag spelled `__`.
///
/// `package` is the Cargo package name as `cargo metadata` reports it, with its hyphens (`gust-compute`), and not the
/// library name (`gust`).
///
/// The mapping is injective on the tag half only for valid tags (see [`is_valid_tag`]): the tags `a_.b` and `a._b`
/// would both map to `a___b`, and `a__b` collides with `a.b`, which is why a valid tag has no `__` and no segment
/// that begins or ends with `_`. It is not injective across packages: Cargo allows package names with `__` or `--`,
/// names that differ only in `-` against `_`, and (on Windows, where environment names ignore case) names that differ
/// only in case. So `variable_name("a__b", "c")`, `variable_name("a--b", "c")` and `variable_name("a", "b.c")` are one
/// variable, and so are `a-b` and `a_b`. Two declared (package, tag) pairs that produce one variable name are an error
/// of the driver that sets the variables, not something this function detects. The function does not check its
/// arguments; the macro and the driver check the tag first.
///
/// ```
/// assert_eq!(
///     gust::routes::variable_name("kernel-lib", "physics.broadphase"),
///     "GUST_ROUTE__kernel_lib__physics__broadphase",
/// );
/// ```
pub fn variable_name(package: &str, tag: &str) -> String {
    let mut name = String::with_capacity(VARIABLE_PREFIX.len() + package.len() + 2 + tag.len() + 2);
    name.push_str(VARIABLE_PREFIX);
    name.extend(package.chars().map(|c| if c == '-' { '_' } else { c }));
    name.push_str("__");
    for c in tag.chars() {
        if c == '.' {
            name.push_str("__");
        } else {
            name.push(c);
        }
    }
    name
}

/// Whether `tag` is a well-formed route tag: segments of `[a-z0-9_]+` joined by single `.`, with no `__` anywhere
/// (the variable's separator), no segment that begins or ends with `_`, and no leading, trailing or doubled `.`.
pub fn is_valid_tag(tag: &str) -> bool {
    !tag.is_empty()
        && tag.split('.').all(|segment| {
            !segment.is_empty()
                && !segment.starts_with('_')
                && !segment.ends_with('_')
                && !segment.contains("__")
                && segment
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(package, tag, variable)`: the golden spelling. `gust-compute-macros` keeps a copy of `variable_name` and
    /// asserts this table verbatim (m8.r1); change both together.
    const GOLDEN: &[(&str, &str, &str)] = &[
        ("kernel-lib", "add", "GUST_ROUTE__kernel_lib__add"),
        ("kernels", "physics", "GUST_ROUTE__kernels__physics"),
        (
            "a-b-c",
            "physics.broadphase.pairs",
            "GUST_ROUTE__a_b_c__physics__broadphase__pairs",
        ),
        ("x", "t_1", "GUST_ROUTE__x__t_1"),
        (
            "gust-compute",
            "physics.broadphase",
            "GUST_ROUTE__gust_compute__physics__broadphase",
        ),
    ];

    #[test]
    fn the_variable_spells_the_package_and_the_tag() {
        for (package, tag, variable) in GOLDEN {
            assert_eq!(variable_name(package, tag), *variable, "{package} {tag}");
        }
    }

    #[test]
    fn the_first_collision_a_trailing_or_leading_underscore_is_refused() {
        // `a_.b` and `a._b` spell the same variable, so a valid tag has neither.
        assert_eq!(variable_name("c", "a_.b"), variable_name("c", "a._b"));
        assert!(!is_valid_tag("a_.b"));
        assert!(!is_valid_tag("a._b"));
        assert!(!is_valid_tag("_a"));
        assert!(!is_valid_tag("a_"));
    }

    #[test]
    fn the_second_collision_a_doubled_underscore_is_refused() {
        // `a__b` and `a.b` spell the same variable, so a valid tag has no `__`.
        assert_eq!(variable_name("c", "a__b"), variable_name("c", "a.b"));
        assert!(!is_valid_tag("a__b"));
        assert!(is_valid_tag("a.b"));
    }

    #[test]
    fn packages_that_spell_one_variable_are_a_known_collision() {
        // The package half is not injective (see `variable_name`): the driver must refuse two declared
        // (package, tag) pairs that land on one variable, because this function cannot.
        let one = variable_name("a__b", "c");
        assert_eq!(one, "GUST_ROUTE__a__b__c");
        assert_eq!(variable_name("a", "b.c"), one);
        assert_eq!(variable_name("a--b", "c"), one);
        assert_eq!(variable_name("a-b", "t"), variable_name("a_b", "t"));
    }

    #[test]
    fn valid_tags_map_to_distinct_variables() {
        let tags = [
            "a",
            "a.b",
            "a_b",
            "a.b.c",
            "a_b.c",
            "a.b_c",
            "ab",
            "a1",
            "a1.b2",
            "physics.broadphase",
            "physics_broadphase",
        ];
        for tag in tags {
            assert!(is_valid_tag(tag), "{tag}");
        }
        for (index, left) in tags.iter().enumerate() {
            for right in &tags[index + 1..] {
                assert_ne!(
                    variable_name("c", left),
                    variable_name("c", right),
                    "{left} {right}"
                );
            }
        }
    }

    #[test]
    fn a_malformed_tag_is_refused() {
        for tag in [
            "", ".", "a.", ".a", "a..b", "A", "a-b", "a b", "a.B", "a/b", "é", "a._", "_",
        ] {
            assert!(!is_valid_tag(tag), "{tag:?}");
        }
    }
}

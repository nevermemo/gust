//! Names of build-time routes (D65 to D67, D137).
//!
//! A route tag names a group of kernels (`physics`, `physics.broadphase`) whose target a build chooses. The choice
//! reaches the `#[gpu]` macro through an environment variable per tag, `GUST_ROUTE__<crate>__<tag>`. This module owns
//! the one mapping from a crate and a tag to that variable name, so that the macro and the driver (`cargo gust`)
//! cannot disagree.

/// The prefix of every route variable.
pub const VARIABLE_PREFIX: &str = "GUST_ROUTE__";

/// The variable that chooses the target of `tag` in the crate `crate_name`: `GUST_ROUTE__<crate>__<tag>`, with `-`
/// in the crate name spelled `_` and every `.` of the tag spelled `__`.
///
/// The mapping is injective only on valid tags (see [`is_valid_tag`]): the tags `a_.b` and `a._b` would both map to
/// `a___b`, and `a__b` collides with `a.b`, which is why a valid tag has no `__` and no segment that begins or ends
/// with `_`. The function does not check its arguments; the macro and the driver check the tag first.
///
/// ```
/// assert_eq!(
///     gust::routes::variable_name("kernel-lib", "physics.broadphase"),
///     "GUST_ROUTE__kernel_lib__physics__broadphase",
/// );
/// ```
pub fn variable_name(crate_name: &str, tag: &str) -> String {
    let mut name =
        String::with_capacity(VARIABLE_PREFIX.len() + crate_name.len() + 2 + tag.len() + 2);
    name.push_str(VARIABLE_PREFIX);
    name.extend(crate_name.chars().map(|c| if c == '-' { '_' } else { c }));
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

    #[test]
    fn the_variable_spells_the_crate_and_the_tag() {
        assert_eq!(
            variable_name("kernel-lib", "add"),
            "GUST_ROUTE__kernel_lib__add"
        );
        assert_eq!(
            variable_name("kernels", "physics"),
            "GUST_ROUTE__kernels__physics"
        );
        assert_eq!(
            variable_name("a-b-c", "physics.broadphase.pairs"),
            "GUST_ROUTE__a_b_c__physics__broadphase__pairs"
        );
        assert_eq!(variable_name("x", "t_1"), "GUST_ROUTE__x__t_1");
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

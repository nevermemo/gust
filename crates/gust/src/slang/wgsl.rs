//! The WGSL fix-up for subgroup operations (M8, D140; spike M8-S1).
//!
//! `slangc` 2026.20 writes WGSL that the pinned `naga` 30.0.1 does not accept as it stands when the kernel uses a
//! subgroup operation. `fixup_subgroups` repairs the two differences and nothing else. They are independent, and each
//! has its own retire condition, so each is its own function:
//!
//! 1. `remove_enable_subgroups` drops the line `enable subgroups;`. naga takes the feature from the device and
//!    refuses the directive. Retire it when the pinned naga accepts `enable subgroups;` (gfx-rs/wgpu issue 5555).
//! 2. `rewrite_lane_arguments` rewrites the lane argument `i32(<expr>)` of `subgroupShuffle` and `subgroupBroadcast`
//!    to `u32(<expr>)` (`i32(3)` to `3u`): Slang spells the lane as `i32`, naga wants `u32`. Retire it when Slang emits
//!    `u32` or naga accepts `i32` (wgpu issue 10818).
//!
//! The fix-up is a text edit. It is the identity on text that has neither pattern, so it is safe to run on every WGSL
//! artifact, and it keeps the line endings `slangc` wrote (CRLF on Windows). A WGSL lexer would be a second front end
//! to keep in step with Slang, and the function has to disappear at a pin bump; if a case ever needs a parse, the
//! decision is to stop and report it (M8 plan, spike S1's kill rule), not to grow this module.
//!
//! The bridge (`slang::compile` for the WGSL target, and `reflect::compile_reflected`) runs it on the compiler's text.
//! An export runs it on the stored artifact too, through `fixup_artifact`, which keeps the artifact hash of the
//! reflection document equal to the hash of the bytes it stores.
//!
//! This module is `#[doc(hidden)]`: it is not part of the stable API and goes away at a pin bump (D141).
//!
//! Part 1 is a fix-up for naga, not for WGSL. On wasm32 `ShaderSource::Wgsl` bypasses naga and the browser compiles
//! the text itself, so once a wgpu release maps the subgroups feature for the web, part 1 must not run on a web
//! artifact: a browser may need the `enable subgroups;` directive (M8 plan, the m8.u0 note; inferred, not run).
//! Today a web profile with `subgroups` is a typed refusal, so no web artifact reaches this module.

/// Both parts of the fix-up: part 1, then part 2. The identity on WGSL without the patterns.
pub fn fixup_subgroups(wgsl: &str) -> String {
    rewrite_lane_arguments(&remove_enable_subgroups(wgsl))
}

/// Part 1: removes every line that is exactly `enable subgroups;` (surrounding whitespace ignored), line ending
/// included. Another directive, `enable f16;` for one, stays.
pub fn remove_enable_subgroups(wgsl: &str) -> String {
    let mut out = String::with_capacity(wgsl.len());
    for line in wgsl.split_inclusive('\n') {
        if line.trim() == "enable subgroups;" {
            continue;
        }
        out.push_str(line);
    }
    out
}

/// The functions whose second argument is a lane.
const LANE_FUNCTIONS: [&str; 2] = ["subgroupShuffle", "subgroupBroadcast"];

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Part 2: the lane argument of `subgroupShuffle(value, i32(lane))` and `subgroupBroadcast(value, i32(lane))` becomes
/// `u32(lane)`, and `i32(<digits>)` becomes `<digits>u`. A call that does not have exactly two arguments, one
/// whose lane is not a single `i32(..)` call, a longer name (`subgroupShuffleXor`) and a name that is part of another
/// identifier are left as they are. Nested calls are rewritten too.
pub fn rewrite_lane_arguments(wgsl: &str) -> String {
    let bytes = wgsl.as_bytes();
    let mut out = String::with_capacity(wgsl.len());
    let mut position = 0;
    while position < wgsl.len() {
        // The next candidate at or after `position`.
        let mut next: Option<(usize, &str)> = None;
        for name in LANE_FUNCTIONS {
            if let Some(offset) = wgsl[position..].find(name) {
                let at = position + offset;
                if next.is_none_or(|(best, _)| at < best) {
                    next = Some((at, name));
                }
            }
        }
        let Some((at, name)) = next else { break };
        let after_name = at + name.len();
        let boundary_before = at == 0 || !is_identifier_byte(bytes[at - 1]);
        // Optional whitespace, then the opening parenthesis; `subgroupShuffleXor(` fails here.
        let mut open = after_name;
        while open < bytes.len() && bytes[open].is_ascii_whitespace() {
            open += 1;
        }
        if !boundary_before || open >= bytes.len() || bytes[open] != b'(' {
            out.push_str(&wgsl[position..after_name]);
            position = after_name;
            continue;
        }
        let Some(close) = matching_parenthesis(bytes, open) else {
            out.push_str(&wgsl[position..after_name]);
            position = after_name;
            continue;
        };
        let arguments = &wgsl[open + 1..close];
        let commas = top_level_commas(arguments);
        out.push_str(&wgsl[position..=open]);
        if let [comma] = commas.as_slice() {
            let (value, lane) = (&arguments[..*comma], &arguments[comma + 1..]);
            out.push_str(&rewrite_lane_arguments(value));
            out.push(',');
            out.push_str(&rewrite_lane(&rewrite_lane_arguments(lane)));
        } else {
            // Not a two-argument call: the arguments stay, and nested calls in them are still rewritten.
            out.push_str(&rewrite_lane_arguments(arguments));
        }
        out.push(')');
        position = close + 1;
    }
    out.push_str(&wgsl[position..]);
    out
}

/// `  i32(expr) ` to `  u32(expr) ` or, for a decimal literal, `  3u `; anything that is not exactly one `i32(..)` call
/// is returned as it is.
fn rewrite_lane(argument: &str) -> String {
    let trimmed = argument.trim();
    let leading = &argument[..argument.len() - argument.trim_start().len()];
    let trailing = &argument[argument.trim_end().len()..];
    let Some(rest) = trimmed.strip_prefix("i32") else {
        return argument.to_owned();
    };
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return argument.to_owned();
    }
    let open = trimmed.len() - rest.len();
    let Some(close) = matching_parenthesis(trimmed.as_bytes(), open) else {
        return argument.to_owned();
    };
    if close != trimmed.len() - 1 {
        return argument.to_owned();
    }
    let inner = &trimmed[open + 1..close];
    let literal = inner.trim();
    if !literal.is_empty() && literal.bytes().all(|byte| byte.is_ascii_digit()) {
        format!("{leading}{literal}u{trailing}")
    } else {
        format!("{leading}u32({inner}){trailing}")
    }
}

/// The index of the `)` that closes the `(` at `open`.
fn matching_parenthesis(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, &byte) in bytes.iter().enumerate().skip(open) {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// The byte offsets of the commas of `arguments` that sit at parenthesis, bracket and brace depth zero.
///
/// Angle brackets are not tracked: the comma of a template list (`array<T, N>(..)`) in an argument would count as a
/// top-level comma, and the call would then have the wrong number of arguments and be left unrewritten. That cannot
/// happen with the values a subgroup operation takes (a `u32`, `i32`, `f32` or `bool` scalar), and tracking `<` would
/// need a parse to tell it from a comparison.
fn top_level_commas(arguments: &str) -> Vec<usize> {
    let mut depth = 0_i32;
    let mut commas = Vec::new();
    for (index, byte) in arguments.bytes().enumerate() {
        match byte {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => commas.push(index),
            _ => {}
        }
    }
    commas
}

/// The hash of a stored artifact, as the reflection document spells it: sixteen hex digits of FNV-1a 64.
fn fingerprint(bytes: &[u8]) -> String {
    format!("{:016x}", crate::artifact::fnv1a64(&[bytes]))
}

/// [`fixup_subgroups`] for the WGSL artifact of an export, with its reflection document. When the fix-up changes the
/// text, the hash of the bytes changes, and `artifactHash` of the document has to name the stored bytes (that is what
/// `artifact::verify_artifact` compares at run time): the old hash is replaced by the new one in the document's text,
/// with no parse. The text of a document that does not hold the old hash exactly once is an error, not a guess. An
/// artifact that is not UTF-8, or whose text the fix-up leaves alone, is returned as it was.
///
/// # Errors
///
/// Returns the reason when the old hash does not appear exactly once in `reflection_json`.
pub fn fixup_artifact(
    artifact: Vec<u8>,
    reflection_json: String,
) -> Result<(Vec<u8>, String), String> {
    let Ok(text) = std::str::from_utf8(&artifact) else {
        return Ok((artifact, reflection_json));
    };
    let fixed = fixup_subgroups(text);
    if fixed == text {
        return Ok((artifact, reflection_json));
    }
    let (old, new) = (fingerprint(&artifact), fingerprint(fixed.as_bytes()));
    let occurrences = reflection_json.matches(old.as_str()).count();
    if occurrences != 1 {
        return Err(format!(
            "the subgroup fix-up changed the artifact, and the reflection document names its old hash {old} \
             {occurrences} times instead of once"
        ));
    }
    Ok((fixed.into_bytes(), reflection_json.replace(&old, &new)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // What slangc 2026.20 writes for a kernel that shuffles by a computed lane, a constant lane and a broadcast, from the
    // spike's output (M8-S1), shortened. slangc on Windows writes CRLF; the tests keep it where it matters.
    const SHUFFLE: &str = "enable subgroups;\r\n@binding(0) @group(0) var<storage, read_write> buf_0 : array<u32>;\r\n\r\n@compute\r\n@workgroup_size(64, 1, 1)\r\nfn main(@builtin(global_invocation_id) gid_0 : vec3<u32>, @builtin(subgroup_invocation_id) lane_0 : u32)\r\n{\r\n    var _S1 : u32 = gid_0.x;\r\n    var _S2 : u32 = subgroupShuffle(buf_0[_S1], i32((lane_0 & (u32(7)))));\r\n    var _S3 : u32 = subgroupShuffle(buf_0[_S1], i32(3));\r\n    var _S4 : u32 = subgroupBroadcast(buf_0[_S1], i32(2));\r\n    buf_0[_S1] = _S2 + _S3 + _S4;\r\n    return;\r\n}\r\n";

    fn lf(text: &str) -> String {
        text.replace("\r\n", "\n")
    }

    #[test]
    fn part_one_removes_the_enable_line_and_only_it() {
        let fixed = remove_enable_subgroups(SHUFFLE);
        assert_eq!(fixed, SHUFFLE.replacen("enable subgroups;\r\n", "", 1));
        assert!(!fixed.contains("enable"));
        assert_eq!(
            remove_enable_subgroups("enable f16;\nfn f() {}\n"),
            "enable f16;\nfn f() {}\n"
        );
        assert_eq!(
            remove_enable_subgroups("  enable subgroups;  \r\nfn f() {}"),
            "fn f() {}"
        );
        assert_eq!(remove_enable_subgroups("enable subgroups;"), "");
    }

    #[test]
    fn part_two_rewrites_a_literal_lane_to_an_unsigned_literal() {
        let fixed = fixup_subgroups(SHUFFLE);
        assert!(
            fixed.contains("subgroupShuffle(buf_0[_S1], 3u);"),
            "{fixed}"
        );
        assert!(
            fixed.contains("subgroupBroadcast(buf_0[_S1], 2u);"),
            "{fixed}"
        );
    }

    #[test]
    fn part_two_rewrites_a_computed_lane_to_a_u32_cast() {
        let fixed = fixup_subgroups(SHUFFLE);
        assert!(
            fixed.contains("subgroupShuffle(buf_0[_S1], u32((lane_0 & (u32(7)))));"),
            "{fixed}"
        );
        assert!(!fixed.contains("i32"), "{fixed}");
    }

    #[test]
    fn the_whole_golden_and_the_line_endings() {
        let expected = "@binding(0) @group(0) var<storage, read_write> buf_0 : array<u32>;\n\n@compute\n@workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) gid_0 : vec3<u32>, @builtin(subgroup_invocation_id) lane_0 : u32)\n{\n    var _S1 : u32 = gid_0.x;\n    var _S2 : u32 = subgroupShuffle(buf_0[_S1], u32((lane_0 & (u32(7)))));\n    var _S3 : u32 = subgroupShuffle(buf_0[_S1], 3u);\n    var _S4 : u32 = subgroupBroadcast(buf_0[_S1], 2u);\n    buf_0[_S1] = _S2 + _S3 + _S4;\n    return;\n}\n";
        let fixed = fixup_subgroups(SHUFFLE);
        assert_eq!(lf(&fixed), expected);
        // Only the removed line loses its ending: every other line keeps the CRLF slangc wrote.
        assert_eq!(
            fixed.matches("\r\n").count(),
            SHUFFLE.matches("\r\n").count() - 1
        );
        assert_eq!(fixed.matches('\n').count(), fixed.matches("\r\n").count());
    }

    #[test]
    fn nested_calls_commas_and_several_calls() {
        let text = "a = subgroupShuffle(subgroupShuffle(x, i32(1)), i32(f(a, b)));\nb = subgroupBroadcast(y, i32( 12 ));";
        let want = "a = subgroupShuffle(subgroupShuffle(x, 1u), u32(f(a, b)));\nb = subgroupBroadcast(y, 12u);";
        assert_eq!(rewrite_lane_arguments(text), want);
        // A comma inside the first argument is not the separator.
        assert_eq!(
            rewrite_lane_arguments("subgroupShuffle(vec2<u32>(a, b).x, i32(c))"),
            "subgroupShuffle(vec2<u32>(a, b).x, u32(c))"
        );
    }

    #[test]
    fn text_without_the_patterns_is_the_identity() {
        let plain = "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n    let a = i32(3);\n}\n";
        assert_eq!(fixup_subgroups(plain), plain);
        for same in [
            "subgroupShuffleXor(v, i32(1))",
            "subgroupShuffleUp(v, i32(1))",
            "subgroupAdd(i32(3))",
            "my_subgroupShuffle(v, i32(1))",
            "subgroupShuffle(v, 3u)",
            "subgroupShuffle(v, u32(a))",
            "subgroupShuffle(v, i32(a) + i32(b))",
            "subgroupShuffle(v)",
            "subgroupShuffle(v, i32(a), i32(b))",
            "subgroupShuffle(",
            "subgroupShuffle",
        ] {
            assert_eq!(rewrite_lane_arguments(same), same, "{same}");
        }
    }

    #[test]
    fn the_fix_up_is_idempotent() {
        let once = fixup_subgroups(SHUFFLE);
        assert_eq!(fixup_subgroups(&once), once);
    }

    #[test]
    fn whitespace_before_the_parenthesis_is_kept() {
        assert_eq!(
            rewrite_lane_arguments("subgroupShuffle (v,  i32(2) )"),
            "subgroupShuffle (v,  2u )"
        );
    }

    #[test]
    fn an_artifact_that_changes_gets_its_hash_in_the_document() {
        let artifact = SHUFFLE.as_bytes().to_vec();
        let old = fingerprint(&artifact);
        let json = format!("{{\"sourceHash\":\"0123456789abcdef\",\"artifactHash\":\"{old}\"}}");
        let (bytes, json) =
            fixup_artifact(artifact, json).expect("the document holds the hash once");
        assert_eq!(bytes, fixup_subgroups(SHUFFLE).into_bytes());
        let new = fingerprint(&bytes);
        assert_ne!(new, old);
        assert_eq!(
            json,
            format!("{{\"sourceHash\":\"0123456789abcdef\",\"artifactHash\":\"{new}\"}}")
        );
    }

    #[test]
    fn an_artifact_the_fix_up_leaves_alone_is_returned_as_it_was() {
        let plain = b"fn main() {}\n".to_vec();
        let json = String::from("{\"artifactHash\":\"whatever\"}");
        assert_eq!(
            fixup_artifact(plain.clone(), json.clone()),
            Ok((plain, json.clone()))
        );
        let not_text = vec![0xff, 0xfe, 0x00];
        assert_eq!(
            fixup_artifact(not_text.clone(), json.clone()),
            Ok((not_text, json))
        );
    }

    #[test]
    fn a_document_without_the_old_hash_is_an_error() {
        let result = fixup_artifact(SHUFFLE.as_bytes().to_vec(), "{}".to_owned());
        let message = result.expect_err("no hash to replace");
        assert!(message.contains("0 times"), "{message}");
    }
}

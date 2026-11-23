//! Tests of the recorded path of a source file.

use super::*;

#[test]
fn recorded_path_keeps_relative_paths_and_normalises_separators() {
    assert_eq!(
        recorded_path("crates/demo/src/lib.rs", UNIX_ROOT),
        "crates/demo/src/lib.rs"
    );
    assert_eq!(
        recorded_path(r"crates\demo\src\lib.rs", DRIVE_ROOT),
        "crates/demo/src/lib.rs"
    );
    assert_eq!(recorded_path("src/lib.rs", None), "src/lib.rs");
}

#[test]
fn recorded_path_makes_absolute_paths_under_the_manifest_dir_relative() {
    assert_eq!(
        recorded_path("/srv/build/gust/crates/demo/src/lib.rs", UNIX_ROOT),
        "src/lib.rs"
    );
    assert_eq!(
        recorded_path(r"C:\build_gust\crates\demo\src\kernels\a.rs", DRIVE_ROOT),
        "src/kernels/a.rs"
    );
    // Drive letter case and a verbatim prefix do not defeat the match, and a
    // trailing separator on the manifest dir is harmless.
    assert_eq!(
        recorded_path(
            r"\\?\c:\build_gust\crates\demo\src\lib.rs",
            Some(r"C:\build_gust\crates\demo\")
        ),
        "src/lib.rs"
    );
    assert_eq!(
        recorded_path(
            r"\\?\UNC\build\share\crates\demo\src\lib.rs",
            Some(r"\\build\share\crates\demo")
        ),
        "src/lib.rs"
    );
}

#[test]
fn recorded_path_keeps_only_the_file_name_of_absolute_paths_outside_the_manifest_dir() {
    for (reported, manifest_dir) in [
        ("/opt/other/crate/src/lib.rs", UNIX_ROOT),
        // A sibling that merely shares the manifest dir as a string prefix.
        ("/srv/build/gust/crates/demo-two/src/lib.rs", UNIX_ROOT),
        (r"D:\elsewhere\src\lib.rs", DRIVE_ROOT),
        (r"\\server\share\src\lib.rs", DRIVE_ROOT),
        (r"\\?\C:\Windows\src\lib.rs", DRIVE_ROOT),
        (r"\\?\UNC\server\share\src\lib.rs", DRIVE_ROOT),
        ("/opt/other/crate/src/lib.rs", None),
        (r"C:\build_gust\crates\demo\src\lib.rs", None),
        // The manifest dir itself is not a file inside it.
        ("/srv/build/gust/crates/demo", UNIX_ROOT),
    ] {
        let stored = recorded_path(reported, manifest_dir);
        let expected = reported.rsplit(['/', '\\']).next().unwrap().to_owned();
        assert_eq!(stored, expected, "{reported}");
        assert!(
            !stored.contains('/') && !stored.contains('\\'),
            "{reported}"
        );
    }
}

#[test]
fn recorded_path_collapses_separators_and_dot_segments_before_the_containment_check() {
    // Cases the manifest dir `/repo` and the reported path must agree on once
    // both are normalised: a doubled separator used to leave `/src/lib.rs`.
    for (reported, manifest_dir, expected) in [
        ("/repo//src/lib.rs", REPO, "src/lib.rs"),
        ("/repo/./src/lib.rs", REPO, "src/lib.rs"),
        ("/repo/src/../src/lib.rs", REPO, "src/lib.rs"),
        ("/repo/../repo/src/lib.rs", REPO, "src/lib.rs"),
        ("/../repo/src/lib.rs", REPO, "src/lib.rs"),
        ("/repo/src/.//lib.rs", Some("/repo//"), "src/lib.rs"),
        (r"C:\repo\src\lib.rs", Some(r"C:\repo"), "src/lib.rs"),
        (r"C:\repo\\src\.\lib.rs", Some(r"C:\repo"), "src/lib.rs"),
        (r"c:\REPO\src\lib.rs", Some(r"C:\repo"), "src/lib.rs"),
        (
            r"C:\repo\src\..\src\lib.rs",
            Some(r"C:\repo\."),
            "src/lib.rs",
        ),
        // Relative paths lose their `.` and repeated separators, and resolve `..`.
        ("./src/lib.rs", None, "src/lib.rs"),
        ("src//a/./lib.rs", None, "src/a/lib.rs"),
        ("src/a/../lib.rs", None, "src/lib.rs"),
    ] {
        assert_eq!(
            recorded_path(reported, manifest_dir),
            expected,
            "{reported}"
        );
        // The whitelist never has to replace these.
        assert!(
            is_valid_record(&contained(reported, manifest_dir)),
            "{reported}"
        );
    }
}

#[test]
fn recorded_path_never_follows_dot_dot_out_of_the_manifest_dir_or_a_relative_root() {
    for (reported, manifest_dir, expected) in [
        // Leaves the manifest dir by way of `..` into a sibling tree.
        ("/repo/../home/alice/kernel.rs", REPO, "kernel.rs"),
        ("/repo/src/../../x.rs", REPO, "x.rs"),
        ("/repo/..", REPO, ""),
        (r"C:\repo\..\other\lib.rs", Some(r"C:\repo"), "lib.rs"),
        // A relative path that leaves its root.
        ("../other/src/lib.rs", REPO, "lib.rs"),
        ("../other/src/lib.rs", None, "lib.rs"),
        ("src/../../lib.rs", None, "lib.rs"),
        (r"..\..\lib.rs", None, "lib.rs"),
        ("..", None, ""),
    ] {
        assert_eq!(
            recorded_path(reported, manifest_dir),
            expected,
            "{reported}"
        );
        let stored = recorded_path(reported, manifest_dir);
        assert!(stored.is_empty() || is_valid_record(&stored), "{reported}");
    }
}

#[test]
fn recorded_path_whitelist_replaces_a_relative_result_that_looks_absolute() {
    // `c:` as the first component of a relative path is a drive on Windows.
    assert_eq!(contained("./c:/x.rs", None), "c:/x.rs");
    assert!(!is_valid_record("c:/x.rs"));
    assert_eq!(recorded_path("./c:/x.rs", None), "x.rs");
}

#[test]
fn recorded_path_rejects_drive_like_components_even_in_the_file_name() {
    // A colon inside a component can never be stored, so when the file name
    // has one too there is nothing safe left and the origin becomes unknown.
    for (reported, manifest_dir) in [
        ("/repo/C:..", REPO),
        ("/repo/C:D:x.rs", REPO),
        ("/repo/C:x.rs", REPO),
        ("C:D:x.rs", None),
        ("a/C:x.rs", None),
        (r"\\?\C:", None),
        (r"\\?\C:lib.rs", None),
        ("C:", None),
    ] {
        assert_eq!(recorded_path(reported, manifest_dir), "", "{reported}");
    }
}

#[test]
fn recorded_path_treats_unrecognised_verbatim_forms_as_opaque_roots() {
    // Only `\\?\X:\` and `\\?\UNC\` name a drive or UNC path. GLOBALROOT,
    // Volume{GUID} and device forms used to lose the prefix and be kept as a
    // relative path, leaking `Users/someone`.
    for (reported, manifest_dir, expected) in [
        (
            r"\\?\GLOBALROOT\Device\HarddiskVolume1\Users\someone\kernel.rs",
            None,
            "kernel.rs",
        ),
        (
            r"\\?\GLOBALROOT\Device\HarddiskVolume1\Users\someone\kernel.rs",
            Some(r"\\?\GLOBALROOT\Device\HarddiskVolume1\Users\someone"),
            "kernel.rs",
        ),
        (
            r"\\?\Volume{12345678-1234-1234-1234-123456789abc}\src\lib.rs",
            Some(r"C:\build_gust\crates\demo"),
            "lib.rs",
        ),
        (r"\\?\pipe\kernel.rs", UNIX_ROOT, "kernel.rs"),
        (r"//?/GLOBALROOT/x/kernel.rs", None, "kernel.rs"),
        // The recognised forms still resolve against the manifest dir.
        (
            r"\\?\unc\build\share\crates\demo\src\lib.rs",
            Some(r"\\build\share\crates\demo"),
            "src/lib.rs",
        ),
        (
            r"\\?\C:\build_gust\crates\demo\src\lib.rs",
            DRIVE_ROOT,
            "src/lib.rs",
        ),
    ] {
        assert_eq!(
            recorded_path(reported, manifest_dir),
            expected,
            "{reported}"
        );
    }
}

#[test]
fn recorded_path_has_no_origin_for_empty_and_dot_only_inputs() {
    for reported in [
        "", ".", "..", "./", "../..", "/", r"\", "//", "./.", "src/..",
    ] {
        for manifest_dir in [None, REPO, DRIVE_ROOT] {
            assert_eq!(recorded_path(reported, manifest_dir), "", "{reported:?}");
        }
    }
}

#[test]
fn recorded_path_keeps_unicode_letters_and_rejects_everything_else() {
    for (reported, expected) in [
        ("src/ünï/kernel_é.rs", "src/ünï/kernel_é.rs"),
        ("日本語/ファイル.rs", "日本語/ファイル.rs"),
        ("src/a.b/lib-1+2.rs", "src/a.b/lib-1+2.rs"),
        // A component outside the alphabet falls back to the file name.
        ("src/my kernels/lib.rs", "lib.rs"),
        ("src/.../lib.rs", "lib.rs"),
        ("src/a\u{ff1a}b/lib.rs", "lib.rs"),
        ("src/a\u{202e}b/lib.rs", "lib.rs"),
        ("src/a\nb/lib.rs", "lib.rs"),
        ("src/a\"b/lib.rs", "lib.rs"),
        ("src/a?b/lib.rs", "lib.rs"),
        // ... and when the file name is outside it too there is no origin.
        ("src/my kernel.rs", ""),
        ("src/kernel\u{0}.rs", ""),
        ("src/k\u{ff1a}.rs", ""),
        ("src/\u{1f600}.rs", ""),
        ("/opt/other/\"quoted\".rs", ""),
    ] {
        assert_eq!(recorded_path(reported, None), expected, "{reported:?}");
    }
}

#[test]
fn recorded_path_unicode_boundary_follows_the_alphabetic_property() {
    // U+0345 COMBINING GREEK YPOGEGRAMMENI is a combining mark that is Alphabetic.
    assert!('\u{345}'.is_alphabetic());
    // U+0301 COMBINING ACUTE ACCENT (the decomposed form of an accent) is not.
    assert!(!'\u{301}'.is_alphabetic());
    for (reported, expected) in [
        // An Alphabetic combining mark survives, inside a component and on its own.
        ("src/a\u{345}b/lib.rs", "src/a\u{345}b/lib.rs"),
        ("src/\u{345}/lib.rs", "src/\u{345}/lib.rs"),
        // A decomposed accent falls back to the file name.
        ("src/e\u{301}/lib.rs", "lib.rs"),
        ("src/\u{301}/lib.rs", "lib.rs"),
        // ... and to no origin when the file name carries it.
        ("src/e\u{301}.rs", ""),
        // The precomposed letter is Alphabetic and survives.
        ("src/\u{e9}/lib.rs", "src/\u{e9}/lib.rs"),
    ] {
        assert_eq!(recorded_path(reported, None), expected, "{reported:?}");
    }
}

#[test]
fn recorded_path_keeps_a_remapped_relative_prefix_as_reported() {
    // rustc `--remap-path-prefix` can leave a relative path naming a home
    // directory; the lexical whitelist cannot tell and keeps it.
    assert_eq!(
        recorded_path("home/alice/shared/kernel.rs", Some("/repo")),
        "home/alice/shared/kernel.rs"
    );
}

#[test]
fn recorded_path_handles_genuinely_mixed_separators() {
    for (reported, manifest_dir, expected) in [
        (r"src\a/b\lib.rs", None, "src/a/b/lib.rs"),
        (r"C:/repo\src/lib.rs", Some(r"C:\repo"), "src/lib.rs"),
        (r"/repo\src\..\lib.rs", REPO, "lib.rs"),
        (r"\repo/src\lib.rs", REPO, "src/lib.rs"),
        (
            r"\\build/share\crates/demo\src/lib.rs",
            Some(r"\\build\share\crates\demo"),
            "src/lib.rs",
        ),
        (r"//?/C:\repo/src\lib.rs", Some(r"C:\repo"), "src/lib.rs"),
        (r"src\/\a//b\\lib.rs", None, "src/a/b/lib.rs"),
    ] {
        assert_eq!(
            recorded_path(reported, manifest_dir),
            expected,
            "{reported}"
        );
    }
}

#[test]
fn recorded_path_is_always_empty_or_a_safe_relative_path() {
    for reported in [
        "/a/b.rs",
        r"C:\a\b.rs",
        "C:b.rs",
        r"\\h\s\b.rs",
        r"\\?\C:\a\b.rs",
        r"\\?\UNC\h\s\b.rs",
        "relative/b.rs",
        "../a/b.rs",
        r"a\..\..\b.rs",
        "/a/../../b.rs",
        "/a//b/./../c.rs",
        r"C:\a\..\..\b.rs",
        "C:..\\b.rs",
        r"\\h\s\..\..\b.rs",
        "/repo/C:..",
        "/repo/C:D:x.rs",
        r"\\?\GLOBALROOT\Device\HarddiskVolume1\Users\someone\kernel.rs",
        r"\\?\Volume{12345678-1234-1234-1234-123456789abc}\src\lib.rs",
        r"\\.\pipe\kernel.rs",
        "",
        ".",
        "..",
        "src/ünï/kernel_é.rs",
        "src/a b/c\td.rs",
        "src/a\u{0}b.rs",
        "a/\u{ff1a}/b.rs",
    ] {
        for manifest_dir in [
            None,
            UNIX_ROOT,
            DRIVE_ROOT,
            REPO,
            Some("/a"),
            Some(r"C:\a"),
            Some(r"\\?\GLOBALROOT\Device\HarddiskVolume1"),
            Some(""),
        ] {
            let stored = recorded_path(reported, manifest_dir);
            let context = format!("{reported:?} in {manifest_dir:?} -> {stored:?}");
            assert!(!stored.starts_with('/'), "{context}");
            assert!(!stored.contains('\\'), "{context}");
            assert!(!stored.contains(':'), "{context}");
            assert!(!stored.contains('?'), "{context}");
            assert!(!stored.split('/').any(|part| part == ".."), "{context}");
            assert!(
                stored.chars().all(|c| c.is_ascii_alphanumeric()
                    || "_.+-/".contains(c)
                    || (!c.is_ascii() && c.is_alphabetic())),
                "{context}"
            );
            assert!(stored.is_empty() || is_valid_record(&stored), "{context}");
        }
    }
}

#[test]
fn an_empty_recorded_path_drops_the_whole_line_table() {
    use crate::slang::LineRecord;
    let lines = [LineRecord {
        slang_start: 3,
        slang_end: 4,
        rust_line: 12,
    }];
    let unknown = "SourceOrigin :: UNKNOWN";
    // No safe file: no origin, never `LineSpan`s with an empty file.
    for reported in [".", "..", r"\\?\C:", "/repo/C:..", "src/my kernel.rs"] {
        let output = origin_tokens(reported, REPO, &lines).to_string();
        assert!(output.ends_with(unknown), "{reported}: {output}");
        assert!(!output.contains("LineSpan"), "{reported}: {output}");
    }
    // A safe file keeps its table.
    let output = origin_tokens("/repo/src/lib.rs", REPO, &lines).to_string();
    assert!(output.contains("\"src/lib.rs\""), "{output}");
    assert!(output.contains("rust_line : 12u32"), "{output}");
    // ... and no lines is still unknown.
    let output = origin_tokens("/repo/src/lib.rs", REPO, &[]).to_string();
    assert!(output.ends_with(unknown), "{output}");
}

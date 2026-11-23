//! Unit tests of the expansion, by concern; the helpers they share live here.

use std::{fs, path::Path};

use super::*;

mod descriptor;
mod export;
mod paths;
mod profile;

/// The lines on which the identifiers called `name` of `tokens` start, depth first.
pub(super) fn ident_lines(tokens: TokenStream, name: &str, lines: &mut Vec<usize>) {
    for tree in tokens {
        match tree {
            proc_macro2::TokenTree::Ident(ident) if ident == name => {
                lines.push(ident.span().start().line);
            }
            proc_macro2::TokenTree::Group(group) => ident_lines(group.stream(), name, lines),
            _ => {}
        }
    }
}

use artifact::tests::Scratch;

pub(super) const KERNEL: &str = "mod layout {\n#[kernel]\npub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n}";

/// An environment for a crate named `demo-crate` at `manifest` and, optionally, with export mode on.
pub(super) fn environment(manifest: Option<&Path>, export: Option<&Path>) -> Environment {
    Environment {
        manifest_dir: manifest.map(Path::to_path_buf),
        export_dir: export.map(Path::to_path_buf),
        package: Some("demo-crate".to_owned()),
        reported_file: None,
    }
}

pub(super) fn expanded(source: &str, environment: &Environment) -> TokenStream {
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
    expand_in(module, environment, Profile::Other).unwrap()
}

/// Expands `source` in export mode once and returns every record it wrote as (key, clean Slang text, record text),
/// sorted by key. A grid kernel writes two: its body and its split (D209).
pub(super) fn exported_all(source: &str) -> Vec<(String, String, String)> {
    let scratch = Scratch::new("exported");
    let records = scratch.0.join("records");
    expanded(source, &environment(None, Some(&records)));
    let directory = records.join("demo-crate");
    let mut keys: Vec<String> = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter_map(|name| name.strip_suffix(".slang").map(str::to_owned))
        .collect();
    keys.sort();
    keys.into_iter()
        .map(|key| {
            let slang = fs::read_to_string(directory.join(format!("{key}.slang"))).unwrap();
            let record = fs::read_to_string(directory.join(format!("{key}.kernel.json"))).unwrap();
            (key, slang, record)
        })
        .collect()
}

/// The key and clean Slang text of the one body `source` exports (its split, if any, is not it).
pub(super) fn exported(source: &str) -> (String, String) {
    let mut bodies: Vec<_> = exported_all(source)
        .into_iter()
        .filter(|(_, _, record)| !record.contains("\"split_of\""))
        .collect();
    assert_eq!(
        bodies.len(),
        1,
        "{:?}",
        bodies.iter().map(|b| &b.0).collect::<Vec<_>>()
    );
    let (key, slang, _) = bodies.remove(0);
    (key, slang)
}

/// The key and clean Slang text of the split `source` exports, and the key of the body it names.
pub(super) fn exported_split(source: &str) -> (String, String, String) {
    let mut splits: Vec<_> = exported_all(source)
        .into_iter()
        .filter(|(_, _, record)| record.contains("\"split_of\""))
        .collect();
    assert_eq!(
        splits.len(),
        1,
        "{:?}",
        splits.iter().map(|b| &b.0).collect::<Vec<_>>()
    );
    let (key, slang, record) = splits.remove(0);
    let at = record.find("\"split_of\"").unwrap();
    let owner = record[at..]
        .split('"')
        .nth(3)
        .expect("split_of has a string value")
        .to_owned();
    (key, slang, owner)
}

/// A crate directory holding the artifact pair of `KERNEL` and, if asked, the manifest.
pub(super) fn crate_with_pair(name: &str, manifest: bool) -> (Scratch, String) {
    let (key, _) = exported(KERNEL);
    let scratch = Scratch::new(name);
    scratch.artifact(&format!("{key}.wgsl"), "wgsl bytes");
    scratch.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
    if manifest {
        scratch.artifact("manifest.json", "{}");
    }
    (scratch, key)
}

pub(super) fn environment_without_file(records: &Path) -> Environment {
    environment(None, Some(records))
}

pub(super) const UNIX_ROOT: Option<&str> = Some("/srv/build/gust/crates/demo");
pub(super) const DRIVE_ROOT: Option<&str> = Some(r"C:\build_gust\crates\demo");
pub(super) const REPO: Option<&str> = Some("/repo");

// ---- the build profile (m5.e1, R1-9): injected, never read from the process ----

/// Expands `source` for `profile` in `environment`, in process: no test here sets or removes a variable.
pub(super) fn expanded_for(
    source: &str,
    environment: &Environment,
    profile: Profile,
) -> TokenStream {
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
    expand_in(module, environment, profile).unwrap()
}

/// [`contained_record`] from the reported string.
pub(super) fn contained(reported: &str, manifest_dir: Option<&str>) -> String {
    contained_record(&NormalizedPath::new(reported), manifest_dir)
}

# Slang bundle licences (pinned 2026.21)

This file records the audits of the Slang releases that GUST's pinned-Slang manager downloads (D46, M4 slices m4.f and
m4.r). GUST does not redistribute the bundle: the manager fetches it from Slang's GitHub release into the user's cache,
keeps the archive layout, and leaves the bundle's own `LICENSE`, `LICENSES/` and `third-party-notices/` beside the
binaries. `cargo gust slang install` lists them when it finishes, and `cargo gust doctor` points to them.

The 2026.21 archives were audited on 2026-11-08 (slice m8.p), each against the 2026.20 archive of the same platform, and
every archive was checked against its published SHA-256 before it was extracted or read (the 2026.21 digests are the
table in `crates/cargo-gust/src/slang.rs`, the 2026.20 ones are in the section below). What was run: file listings and
text diffs of `LICENSE`, `LICENSES/`, `third-party-notices/`, `README.md` and `share/slang/agent-skills/` between the
two versions on each platform, a read of every changed text, and a string scan for "GNU General Public" and "GNU
Lesser" of the executables and libraries in `bin/` and `lib/` of all three 2026.21 archives (Windows: 11 `.exe` and
`.dll` files; Linux: 11; macOS: 13; no hit in any). The Linux and macOS archives were extracted on Windows, which
could not create their few symlinks (3 and 1 of them); every regular file was extracted and the listings come from
the archives themselves. Nothing from any archive was run in this audit. Rule (D46) as below.

| Archive | SHA-256 | Verdict |
| --- | --- | --- |
| `slang-2026.21-windows-x86_64.zip` | `4b6f509e73397175ed36a341beb7ad15f4fff5feaf5f6d95fecf896c0d86b8de` | accept |
| `slang-2026.21-linux-x86_64.tar.gz` | `691f9341e5bec099eec97ee30a83bcbef541cee7fe3b4de68a821d42f9506c2f` | accept |
| `slang-2026.21-macos-aarch64.tar.gz` | `c3d3c959ba3e0c8fdb6c2a5cf6a471ed097ed19c41820af9fb21625779794fb2` | accept |

## What changed from 2026.20 to 2026.21

- `LICENSE`, every text in `LICENSES/` (the same seven) and `README.md` are byte for byte the 2026.20 ones, on all three
  platforms. The third-party notice files are the same six (and `mimalloc-LICENSE` on Windows only).
- `third-party-notices/spirv-tools-LICENSE` gains a section headed `File: utils/git-sync-deps`: a BSD-3-Clause text
  (Google Inc.), 29 lines, on all three platforms; the rest of the file is unchanged. It is a permissive licence for a
  SPIRV-Tools helper script; the archives list no such script, so it is a notice for source that is not shipped (inferred
  from the listings).
- Windows `mimalloc-LICENSE` (MIT): only the copyright years change, 2018-2021 to 2018-2025. The other notices
  (`cmark-COPYING`, `glslang-LICENSE.txt`, `lz4-LICENSE`, `miniz-LICENSE`, `unordered_dense-LICENSE`) are byte for byte
  the same on every platform.
- `share/slang/agent-skills/` has the same files; its `PROVENANCE.json` differs only in `slangVersion`.
- The listings differ by two documentation examples that 2026.21 no longer ships
  (`share/doc/slang/generated/tests/design/target-pipelines/metal/nonuniform-resource-index-rejected-metal.slang` and
  `.../wgsl/non-uniform-resource-index-rejected.slang`) and by the version in library file names. Nothing was added.
- Windows `slang-compiler.dll` carries the same number of `mi_malloc` strings in both versions (14 by this count), so
  the set of bundled components did not change on that platform. The Linux and macOS compiler libraries have two
  such strings each in 2026.21; they were not counted in 2026.20.

## The named checks (2026.21)

| Check | Result | Evidence |
| --- | --- | --- |
| No GPL-2.0 lz4 code | pass | `third-party-notices/lz4-LICENSE` is unchanged (the BSD-2-Clause library licence, "Copyright (c) 2011-2020, Yann Collet"). The string scan above found no "GNU General Public" or "GNU Lesser" in any scanned binary on any platform. |
| cmark's CC-BY-SA part | pass | `cmark-COPYING` is unchanged and still names `test/spec.txt` as CC-BY-SA-4.0; no `spec.txt` and no cmark source tree is in any listing. |
| glslang's GPL-3.0 text | pass | `glslang-LICENSE.txt` is unchanged: the GPL-3.0 text "with special bison exception" beside BSD, MIT and Apache-2.0 texts, with the note that Bison was removed from the build. |
| optix-dev absent | pass (listing) | No OptiX SDK and no `optix.h` in any 2026.21 listing; the 12 listing names that contain "optix" are the same documentation examples as in 2026.20. From the listings, not from running anything. |

## Earlier pin: 2026.20 (audited 2026-11-01, slice m4.r)

The 2026.20 archives were audited on 2026-11-01 (slice m4.r, stage 1) and checked against GitHub's published SHA-256
before extraction. What was run: file listings and text diffs of all three archives against the 2026.13.1 ones, a read
of every notice text, and string scans of the Windows binaries. The Linux and macOS binaries were listed but not
string-scanned, and nothing from those two archives was run. Rule (D46): permissive licences are accepted with their
notices kept; a copyleft or non-commercial component stops the download path for that platform.

| Archive | SHA-256 | Verdict |
| --- | --- | --- |
| `slang-2026.20-windows-x86_64.zip` | `e6dd32bbddf1720ebb4e679d55e9635ce792865a5bd84f3d2cd9af28647c0174` | accept |
| `slang-2026.20-linux-x86_64.tar.gz` | `e863e568c204810d6e8bc31852f056aef6763c55ea9925db2db7affa060347c9` | accept |
| `slang-2026.20-macos-aarch64.tar.gz` | `07fa8833fdbb343c3674c84fdd885910f495f3daaaa0c732b32980151e8b58c4` | accept |

## What the 2026.20 bundle contains

- `LICENSE`: Apache-2.0 WITH LLVM-exception, unchanged from 2026.13.1 (the same text). `README.md` is CC-BY-4.0
  documentation and differs from the old one only in formatting and links.
- `LICENSES/` (new): the texts Apache-2.0, BSL-1.0, CC-BY-4.0, LLVM-exception, LicenseRef-UOI-NCSA (the LLVM release
  licence, NCSA), MIT and Unlicense.
- `third-party-notices/` (new): `cmark-COPYING`, `glslang-LICENSE.txt`, `lz4-LICENSE`, `miniz-LICENSE`,
  `spirv-tools-LICENSE` and `unordered_dense-LICENSE`, and in the Windows archive also `mimalloc-LICENSE` (MIT).
- `share/slang/agent-skills/` (new): documentation for AI agents, with its own `LICENSE` (Apache-2.0 WITH
  LLVM-exception) and a `PROVENANCE.json`. GUST does not use it.
- The two directories and the files in them are the same on all three platforms, except for `mimalloc-LICENSE`.
- No licence was removed. The notices are new texts for components that the 2026.13.1 binaries already carried (the
  earlier audit recorded their attributions inside the binaries); mimalloc is the one new component on Windows: the
  2026.13.1 binaries had no `mi_malloc` string and the 2026.20 `slang-compiler.dll` has twelve. Its licence is MIT.
- GUST uses `slangc` and the compiler library (`slang-compiler`, plus the `slang` compatibility name), and probably
  `slang-glslang` (glslang and SPIRV-Tools) on the SPIR-V path. That last one is inferred from strings, not run.
  GUST does not use `slang-llvm` (LLVM and Clang), `slang-rt`, `gfx`, the GLSL module or the `slang`, `slangd` and
  `slangi` executables. Their licences are permissive too, so nothing needs pruning for compliance.

## The named checks (2026.20)

| Check | Result | Evidence |
| --- | --- | --- |
| No GPL-2.0 lz4 code | pass | `third-party-notices/lz4-LICENSE` is the BSD-2-Clause library licence. A scan of every Windows executable and DLL for "GNU General Public" and "GNU Lesser" found no hit. |
| cmark's CC-BY-SA part | pass | `third-party-notices/cmark-COPYING` says the CommonMark specification text `test/spec.txt` is CC-BY-SA-4.0 and the `test/` software is BSD-2-Clause; neither is in the bundle (it ships no cmark source tree), only cmark's code, which is BSD-2-Clause with MIT parts. |
| glslang's GPL-3.0 text | pass | `third-party-notices/glslang-LICENSE.txt` contains the GPL-3.0 text "with special bison exception", beside BSD, MIT and Apache-2.0 texts. It says that Bison is no longer part of the build, and the exception lets Bison-generated parser code be used under any terms. So it is not copyleft in effect for the redistributed binaries. |
| optix-dev absent | pass (listing) | The 2026.20 archive listings show no OptiX SDK and no `optix.h`, as in 2026.13.1; this is from the listings, not from running anything. |

## Earlier pin: 2026.13.1 (audited 2026-11-01, slice m4.f)

| Archive | SHA-256 | Verdict |
| --- | --- | --- |
| `slang-2026.13.1-windows-x86_64.zip` | `fa1c9bcab2cdcd3626f7a1e250dd35d606c1b84745b64627f1dd63fca3746a70` | accept |
| `slang-2026.13.1-linux-x86_64.tar.gz` | `c1ed948af94c6fd2034cc0f82f6892b8b287d5362939758cb4955676f26d893e` | accept |
| `slang-2026.13.1-macos-aarch64.tar.gz` | `cf58b42ba87f66f58e0de297da57f4a5c92d00b7e7f38a708d2a4244abd8d003` | accept |

That bundle had one licence file, `LICENSE` (Apache-2.0 WITH LLVM-exception), and no third-party notices file;
third-party attributions sat inside the binaries. Its checks: no lz4 directory in any archive and no GPL or LGPL strings
in the scanned binaries; cmark's code BSD-2-Clause with MIT parts and its CC-BY-SA-4.0 specification text absent; no
OptiX SDK or `optix.h` (the compiler only names it in a message asking the user to install the SDK).

Re-run this audit whenever the pin changes. The audit notes and scripts live outside the repository.

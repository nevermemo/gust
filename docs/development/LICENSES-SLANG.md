# Slang bundle licences (pinned 2026.13.1)

This file records the audit of the Slang release that GUST's pinned-Slang manager downloads (D46, M4 slice m4.f). GUST
does not redistribute the bundle: the manager fetches it from Slang's GitHub release into the user's cache, keeps the
archive layout, and leaves the bundle's own `LICENSE` beside the binaries. `cargo gust doctor` points to it.

Audited on 2026-11-01. The three archives were checked against GitHub's published SHA-256 before extraction, and
nothing from them was run. Rule (D46): permissive licences are accepted with their notices kept; a copyleft or
non-commercial component stops the download path for that platform.

| Archive | SHA-256 | Verdict |
| --- | --- | --- |
| `slang-2026.13.1-windows-x86_64.zip` | `fa1c9bcab2cdcd3626f7a1e250dd35d606c1b84745b64627f1dd63fca3746a70` | accept |
| `slang-2026.13.1-linux-x86_64.tar.gz` | `c1ed948af94c6fd2034cc0f82f6892b8b287d5362939758cb4955676f26d893e` | accept |
| `slang-2026.13.1-macos-aarch64.tar.gz` | `cf58b42ba87f66f58e0de297da57f4a5c92d00b7e7f38a708d2a4244abd8d003` | accept |

## What the bundle contains

- One licence file, `LICENSE`: Apache-2.0 WITH LLVM-exception. `README.md` is CC-BY-4.0 documentation. There is no
  separate third-party notices file; third-party attributions sit inside the binaries.
- GUST uses `slangc` and the compiler library (`slang-compiler`, plus the `slang` compatibility name), and probably
  `slang-glslang` (glslang and SPIRV-Tools) on the SPIR-V path. That last one is inferred from strings, not run.
- GUST does not use `slang-llvm` (LLVM 21 and Clang, 107 to 152 MB), `slang-rt`, `gfx`, the GLSL module or the
  `slang`, `slangd` and `slangi` executables. Their licences are permissive too, so nothing needs pruning for
  compliance.

## The three named checks

| Check | Result | Evidence |
| --- | --- | --- |
| No GPL-2.0 lz4 code | pass | No lz4 directory exists in any archive; the binaries carry only lz4 library symbols (`lib/`, BSD-2-Clause). No GPL or LGPL strings appear in the scanned binaries. |
| cmark's CC-BY-SA part | pass | cmark's code is BSD-2-Clause, with MIT parts. Its CC-BY-SA-4.0 part is the CommonMark specification text (`test/spec.txt`), which the bundle does not contain (upstream `COPYING`, checked 2026-11-01). |
| optix-dev absent | pass | The bundle ships no OptiX SDK or `optix.h`. The compiler only names it in a message asking the user to install the SDK. |

glslang's upstream `LICENSE.txt` lists GPL-3.0 with the Bison special exception for Bison-generated parser code, beside
BSD, MIT and Apache-2.0. The exception lets the generated parser be used under any terms, and upstream notes that Bison
is no longer part of the build. So it is not copyleft in effect, and it does not change the verdict.

Re-run this audit whenever the pin changes. The audit notes and scripts live outside the repository.

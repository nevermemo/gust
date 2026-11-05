# DX19: Sidecar diagnostic maps

## Goal

Emit a deterministic `kernel.map.json` beside compiler-temporary `kernel.slang`
and use it to enrich Slang errors with a stable generated construct label.

## Contract

- The map is derived directly from emitted Slang source; it is not a shader IR.
- Each segment records inclusive Slang line range, kernel name, construct kind,
  and stable ordinal.
- Kinds: kernel, prologue, local, conditional, loop, match, assignment, return,
  and expression.
- A construct that opens a block (`if`, `for`, a lowered `match`) is one range
  that ends at its closing brace; `else` and `else if` continue the preceding
  conditional. A `match` marker also ends at the next marker at the same depth
  or the end of the enclosing block. Lowered `match` scaffolding (scrutinee
  temporary, branch test, arm bindings) belongs to the `match` range; user
  statements inside arms get their own nested segments.
- The `kernel` segment covers the marker through the entry point's closing
  brace. The generated resource-length prologue (`__gpu_len_*`,
  `__gpu_stride_*`, `GetDimensions`) is one `prologue` segment per kernel and
  does not consume construct ordinals; construct ordinals restart at 0 for each
  kernel.
- The map is a text heuristic, valid only for GUST's own emitter output (Allman
  braces, one statement per line, `//` comments only, reserved `__gpu_` and
  `__gust_` temporaries). Lines before the first `// @rust kernel:` marker
  (structs, helpers, resources) get no attribution.
- A `match` segment's ordinal is the number in its `// @gust construct: match #N`
  marker, which is the emitter's own statement index; every other construct
  uses the per-kernel map ordinal, and the diagnostic label prints the segment
  ordinal either way. The emitter's statement index is block-local: it counts
  statements within one emitted lexical block, so two matches in different
  blocks of one kernel can both be `match #0` (also in `__gust_match_N`, which
  stays unique within its block). `match` ordinals are therefore not unique per
  kernel; the segment's Slang line range disambiguates, and a diagnostic label
  such as `match #0` is read together with the reported Slang line. Numbering
  matches per kernel in the sidecar is deliberately not done.
- A marker whose number does not parse is an emitter bug: tests fail on it
  (debug assertion), and a release build falls back to the map ordinal.
- The compiler bridge writes `kernel.map.json` into its exclusively owned
  temporary directory before invoking `slangc`.
- Error mapping selects the innermost segment containing the reported Slang
  line, then falls back to the nearest preceding kernel marker.
- Schema version 2 (M1.1) adds the Rust location. The top level carries
  `"rust_file"`: the kernel's Rust source path with `/` separators, or `null` when
  no location is known (hand-written descriptors, probes, any compile outside a
  real macro expansion). The path is never absolute (the descriptor ships in public
  binaries): both paths are first normalised lexically (repeated separators
  collapsed, `.` dropped, `..` resolved). A relative path is kept as its normalised
  components; an absolute path under the expanding crate's `CARGO_MANIFEST_DIR` is
  stored relative to it (for example `src/kernels/a.rs`); any other absolute path
  (Unix root, drive, UNC, `\\?\`, including one that leaves the manifest dir through
  `..`) and any relative path that starts with `..` after normalisation is reduced
  to its file name (for example `lib.rs`). The result is a whitelist: relative, every
  component non-empty, not `.` or `..`, and made of ASCII letters and digits, `_ . + -`
  and Unicode Alphabetic characters (so no `:`, `\`, spaces or control characters). A result that
  fails is replaced by the file name; if that fails too (for example `C:..`) the
  origin is unknown, `rust_file` is `null` and the line table is dropped. Of the
  `\\?\` forms only `\\?\X:\` and `\\?\UNC\` count as drive and UNC paths; any other
  (GLOBALROOT, `Volume{GUID}`, device) is an absolute path reduced to its file
  name. The result never contains `..`. A segment
  carries `"rust_line"` (after `"ordinal"`) when the innermost line record containing the
  segment's first Slang line exists; the key is omitted otherwise, so the
  kernel and prologue segments never have it.
- The Rust lines come from the emitter's line table, carried by
  `KernelDescriptor::source` (`SourceOrigin { file, lines }`, each `LineSpan` an
  inclusive Slang range plus a 1-based Rust line, outer construct first). It is a
  line index like this map, not an IR. Records are statement level: one per
  statement in a kernel or helper body, plus one per helper and one per struct
  declaration. A multi-line Rust statement maps to the line it starts on, an
  `else if` header maps to the line of the `if` that owns the chain, and columns
  are never mapped. Prologue, resource declarations, the kernel signature and the
  `// @rust kernel:` marker have no record. The table does not change the emitted
  Slang, the exported artifacts or the pipeline cache identity.
- Both compile paths (`slang::compile*` and the reflection helper) append the same
  text after the compiler output when the reported error line is known:
  `originating Rust kernel`, `originating Rust construct` (as before; for a helper
  or struct above the first kernel marker `slang::compile*` omits both and the
  reflection path still names the kernel) and, new,
  `originating Rust source: <file>:<rust_line>` when the descriptor carries a
  table. A descriptor without a table (`SourceOrigin::UNKNOWN`) gets exactly the
  old text on both paths: on the reflection path that is the compiler output, then
  `originating Rust kernel: <module>::<kernel>` as the last line, with no construct
  and no source line.
- `#line` injection (M1.3). The text the compiler reads is a compile-time copy of
  the Slang source, never the exported text. For each Slang line the descriptor's
  table attributes to a Rust line (the innermost record, rule above) the copy has
  `#line <rust_line> "<file>"` on the line before it, repeated for every such line
  so a multi-line lowering keeps its statement's line instead of counting up; the
  first line without a record after an attributed one is preceded by
  `#line <slang_line> "kernel.slang"`, which returns the compiler to Slang
  coordinates. Lines before the first record are untouched. The file string is the
  descriptor's `rust_file` (`/` separators, whitelisted characters). With no file, no
  table, a record for Rust line 0, or a file string that is not
  `[A-Za-z0-9_.+/-]` or is named `kernel.slang`, the copy equals the source byte
  for byte. Slangc and the native reflection helper both honour the directives
  (their errors read `--> <file>:<line>:<column>`, the column being a Slang column);
  the helper's `sourceHash` is the hash of the copy it read. WGSL and SPIR-V output
  is identical with and without the directives (tested for a struct body, a helper,
  a loop, a lowered `match` and struct literals). Directives also land inside struct
  and helper bodies, since those lines are attributed to their struct or helper. The
  no-semantic-change claim covers emitter-produced source only: a hand-written
  `slang_source` paired with a table gets directives wherever the table says.
- The compile-copy eligibility guard is shared: a recorded file that injection refuses
  (see above) never classifies a location as Rust, so a file recorded as `kernel.slang`
  leaves `kernel.slang(6)` a Slang line.
- The error parser accepts a Rust location only when its path text equals the
  descriptor's `rust_file` exactly, in the `-->` and the legacy `file(line[,col])`
  shapes; any other file stays unattributed. For a Rust location the construct label
  comes from the lowest Slang start line of any record with that Rust line, through
  the same segment lookup (many Slang lines can come from one Rust line, so the
  choice is documented and deterministic), and the annotation ends with
  `(column is in the generated Slang)`. A Slang location (`kernel.slang`, an
  unattributed line) is handled as before, including the record lookup of the M1.1
  contract above.
- The sidecar gains a top-level `"line_directives": true|false` (after
  `"rust_file"`; `schema_version` stays 2, the key is additive): whether the compiled
  copy differs from the clean text. When it does, the temporary directory has
  `kernel.slang` (what the compiler read) and `kernel.export.slang` (the clean text).
- The sidecar is written for inspection and is never parsed back; the lookup
  uses the same map held in memory. "Malformed or missing map" therefore means
  a line with no containing segment, which falls back to kernel attribution.
- The sidecar is temporary diagnostics metadata, not a committed generated
  artifact or runtime ABI field.

## Checks

```sh
cargo test -p gust-compute --lib slang
cargo test -p gust-compute --test diagnostics
cargo xtask check-feature core
cargo xtask check-full
```

## Completion

- Unit tests cover map schema, line ranges, construct selection, malformed/no
  map fallback, and existing kernel attribution.
- An intentional compiler failure reports both kernel and construct label.
- Full validation and independent review pass; record tool/adapter limits in
  `STATUS.md`.

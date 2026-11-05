# Engineering agreement

GUST is the project and the Rust → Slang subsystem. Read
`.ai/AGENT_CONTEXT.json`, `docs/development/STATUS.md`,
`docs/development/NEXT_TASKS.md`, and `docs/development/HANDOFF.md` before editing.
Read the matching `docs/development/tasks/` contract before implementation;
consult `docs/development/history/` only when active state leaves an invariant
unresolved. Consult
`docs/development/REPO_STRUCTURE.md`, `docs/development/VALIDATION.md`,
`docs/development/AGENT_WORKFLOW.md`, `docs/DECISIONS.md`, and
`docs/ARCHITECTURE.md` for constraints. Design aspirations are not implemented
features or permission to perform external actions.

# AI Swarm Instructions

## Goal

Build and maintain this project using a small heterogeneous AI swarm.

The repository, tests and documentation are the shared source of truth.
Do not use long conversations as project memory.

## Routing

Use the cheapest capable worker.

- Scout:
  repository discovery, symbol search, dependency tracing, context gathering.

- Builder:
  implementation, Rust/compiler work, Slang generation, refactoring and debugging.

- Verifier:
  independent review, adversarial testing, compiler/test execution and failure analysis.

The orchestrator coordinates. It should avoid doing implementation itself.

## Workflow

For meaningful changes:

Scout → Builder → Verifier

Skip Scout when the affected files are already obvious.

Skip Builder for read-only questions.

Skip Verifier only for trivial changes.

Do not have multiple agents perform identical work unless:
- there is disagreement,
- confidence is low,
- tests fail,
- or the decision is architecturally important.

## Token discipline

Workers receive only the context required for their subtask.

Worker reports must be concise.
Do not return raw logs unless a failure requires them.
Do not paste large source files into reports.
Prefer file paths, symbols and line references.

Use repository files for durable knowledge. Human-readable operational state lives
under `docs/development/`; `.ai/` is reserved for generated or machine-readable
state such as `VALIDATION.json` and `BASELINE.sha256`.

## Truth hierarchy

Prefer, in order:

1. Tests and compiler/runtime results
2. Existing project invariants
3. Official language/tool documentation
4. Independent agent reasoning
5. Agent consensus

Never treat majority vote as proof.

## Architecture

Preserve the current working Rust AST → Slang pipeline unless explicitly asked to redesign it.

Load relevant Rust/Slang/testing skills only when needed.

## Completion

A meaningful code change is complete only when:

- Builder reports what changed.
- Relevant deterministic checks have run.
- Verifier reports PASS, or remaining uncertainty is explicitly surfaced.

Codex is used separately for important review, explanation and architectural guidance.
Do not wait for Codex during normal local swarm work.

## Invariants

- Preserve `#[gpu]` whole-module authoring, direct syn → Slang, readable artifacts,
  real GPU tests, typed buffers, pipeline caching, ordered batches, and async jobs.
- No new custom shader instruction IR or handwritten SPIR-V backend. An eventual
  execution graph is a high-level CPU/GPU scheduling model, not a replacement for Slang.
- Targets are chosen at build time (D21). A production build contains one target, a
  GPU target or the CPU target, never both, so there is no runtime CPU fallback.
  Test builds may contain both pipelines through `cfg(test)` or `--cfg gust_dual`,
  never through a Cargo feature (D26). Shadow bodies execute only under the
  semantics contract (D22, `docs/SEMANTICS.md`): since M3 they run as CPU twins in
  test and dual builds, and in no production build except the `cpu` target (D51), whose
  runtime they are. Independent host references stay
  required for semantics fixtures; a twin assertion is added to them and to the GPU
  assertion, never in place of either, and is never GPU evidence.
- Workflow commands live in `cargo xtask`; do not add shell or PowerShell scripts
  (D27). The ordered plan is `docs/ROADMAP.md`; read its "Now and next" first.
- Do not claim compiler reflection in the macro: bindings and restricted POD layouts
  are assigned by the macro. The runtime verifies them against Slang's reflection
  through the native helper before creating a pipeline (D17); that check is a gate,
  not a source of bindings.
- Portable behavior is required; native accelerations need capability gates and
  the same visible semantics. No GPU inter-workgroup spin waits.
- Prefer a diagnostic over silently translating unsupported Rust semantics.
- Do not rename crates, initialize Git, remove working examples, add dependencies,
  or expand architecture merely to align branding. Explain necessary changes. The published package names are D48's
  (`gust-compute`, `gust-compute-macros`, `gust-compute-wgpu`); the directories and the library names (`gust`,
  `gust_macros`, `gust_wgpu`) stay.
- Parallel lanes may each use one branch and one Git worktree (owner decision
  2026-10-23). Update a branch by merging `main` into it; never rebase commits that
  were already pushed.

## Approvals

The owner pre-authorizes all actions on this project, including commits, pushes to
GitHub, branches, and file deletions (2026-10-23). Do not stop to ask; report every
public action (push, PR, release, repository setting) after doing it. Machine-sharing
limits agreed with the owner still apply: shared cargo build slots, one heavy GPU job at
a time, and no use of the owner's model-service ports. Codex is approved for
cross-family reviews on this project (2026-10-23); other AI families still need an
owner decision.

## Workflow and verification

1. Check the working tree (or hashes if no Git), installed tools, and baseline tests.
2. Claim a bounded task and file scope in `docs/development/STATUS.md` before work. If another
   agent owns overlapping files, coordinate with the human; do not overwrite them.
3. Add a failing regression first when fixing a bug. Test compiler output and actual
   GPU behavior, not only substring assertions. No silent GPU/compiler test skips
   in new validation; clearly distinguish optional external validation.
Run from the workspace root:

```sh
cargo xtask check-fast
cargo xtask check-workspace
```

Use stable Rust, the pinned `slangc` on PATH (`cargo gust slang install` installs it;
`GUST_SLANG_HOME` moves that install), and a Vulkan adapter. For release evidence run
`cargo xtask check-full`: it requires SPIRV-Tools, runs every example, validates
all exported SPIR-V files, and runs the whole `check-precompiled` and `check-web`
(`--skip-web` leaves out the browser run and is recorded). Missing tools are blockers, not passes. Baseline versions
and actual results belong in `docs/development/STATUS.md`; commands alone are not evidence.
The command overwrites `.ai/VALIDATION.json` with the latest pass/failure record;
inspect its timestamp and scope before using it as evidence. That file is local and
untracked, and xtask redacts the workspace root and home directory in it; copy the
summary into `STATUS.md`. Never commit machine-specific paths: the repository is public.

## Artifact and collaboration rules

- `generated-wgpu/` contains local derived `.slang`, `.wgsl`, `.spv`, and readable
  `.rs` host examples. It is ignored and must not be committed. Regenerate with
  `cargo xtask export-artifacts`, `cargo xtask check-artifacts`, `cargo xtask
  check-full`, or the relevant example binary; never hand-edit exports. The generated
  host snippets are inspectable examples, not standalone packages.
- `tests/fixtures/semantics.slang` is a reviewed expectation, not an auto-updated
  export. A translation change requires an explained golden diff and GPU tests.
- Keep commits small if Git is available. This takeover began without Git;
  `.ai/BASELINE.sha256` records source/artifact hashes before edits, not a backup.
- Do not change account allowances, spend credits, or contact another AI family
  without authority; see Approvals. A local AI resumes manually from these files.
- Before handing off, clear ownership, record exact files/tests/known failures and
  the next command. Do not mark researched plans complete. Record reliable allowance
  readings only; the human's allowance stop policy applies to this Codex run, not
  an endless instruction for a future local agent.

# GUST handoff

## Current state

- Branch: `main`
- Active: roadmap M4 (build-time targets and distribution), plan adopted (D44 to D47); see the
  "Now and next" section of [ROADMAP.md](../ROADMAP.md) and [STATUS.md](STATUS.md).
- M0 is closed (D28), and M0.15 (the CI `unit` job is a required check on `main`) followed on
  2026-11-01. M1 (diagnostics) is closed (D33), M2 (typed host API) is closed (D37), and
  M3 (dual test builds) is closed (D43; `check-full` passed at `5971c12`).
- T13 GPU active-count proof is deferred by owner. Its work in progress is the git
  stash `WIP T13 active count deferred`, not the working tree. Do not apply,
  continue, validate, commit, or drop it without explicit direction.
- Last committed compiler feature: T12 statement Result match.

## Pickup

1. Read `.ai/AGENT_CONTEXT.json`, AGENTS.md, and (for Claude Code) CLAUDE.md.
2. Read the "Now and next" section of [ROADMAP.md](../ROADMAP.md) and the active
   section of [STATUS.md](STATUS.md).
3. Read the matching task contract or plan.
4. Run `git status --short --branch`, `git stash list`, and `cargo xtask check-fast`.

## Evidence

- Last full record: `cargo xtask check-full` passed on a clean tree at `5971c12` (2026-11-01,
  NVIDIA GeForce RTX 5090 / Vulkan: workspace tests, the release-only tests by name, every
  example test with `--include-ignored`, `check-dual` in full mode, 13 SPIR-V artifacts
  validated); details in STATUS.
- CI `unit` and `msrv` jobs green on main.
- Evidence tiers and labels: [VALIDATION.md](VALIDATION.md).

Historical handoff records are in
[history/HANDOFF-2026-09-09.md](history/HANDOFF-2026-09-09.md).

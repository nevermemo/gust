# GUST handoff

## Current state

- Branch: `main`
- Active: roadmap M7 (GPU concepts in Rust); M6 (language breadth I) is closed (D86). See the
  "Now and next" section of [ROADMAP.md](../ROADMAP.md) and [STATUS.md](STATUS.md).
- M0 is closed (D28), and M0.15 (the CI `unit` job is a required check on `main`) followed on
  2026-11-01. M1 (diagnostics) is closed (D33), M2 (typed host API) is closed (D37), and
  M3 (dual test builds) is closed (D43; `check-full` passed at `5971c12`). M4 (build-time targets and
  distribution) is closed (D73), and M5 (CPU target and evidence breadth) is closed (D76; `check-full` passed at
  `7dd9c3d`).
- T13 (GPU active count to indirect dispatch) landed in m7.t13 and m7.t14; no stash is kept for it. The latest
  slices and features are in the "Now and next" section of [ROADMAP.md](../ROADMAP.md).

## Pickup

1. Read `.ai/AGENT_CONTEXT.json`, AGENTS.md, and (for Claude Code) CLAUDE.md.
2. Read the "Now and next" section of [ROADMAP.md](../ROADMAP.md) and the active
   section of [STATUS.md](STATUS.md).
3. Read the matching task contract or plan.
4. Run `git status --short --branch`, `git stash list`, and `cargo xtask check-fast`.

## Evidence

- Last full record: `cargo xtask check-full` passed at `7dd9c3d` (2026-11-02, M5 close, NVIDIA GeForce RTX 5090 /
  Vulkan, Slang v2026.20, clean tree: workspace tests, the release-only tests
  by name, every example test with `--include-ignored`, `check-dual` in full mode, 13 SPIR-V artifacts validated,
  the whole `check-precompiled` and `check-web`, no skipped check); details in STATUS.
- CI `unit` and `msrv` jobs green on main.
- Evidence tiers and labels: [VALIDATION.md](VALIDATION.md).

Historical handoff records are in
[history/HANDOFF-2026-09-09.md](history/HANDOFF-2026-09-09.md).

# GUST handoff

## Current state

- Branch: `main`
- Active: M8 (capability variants, 0.2.0) is closed (D153); M9 (language breadth II) is next, and its plan comes
  first. The 0.2.0 publish stays the owner's act (D148, RELEASE.md). M7 (D126) and MX (D129) are closed, and M6
  (language breadth I) is closed (D86). See the "Now and next" section of [ROADMAP.md](../ROADMAP.md) and
  [STATUS.md](STATUS.md).
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

- Last full record: `cargo xtask check-full` passed at `2f96e69` (2026-11-09, 05:11 to 05:53Z, M8 close, RTX 5090 /
  Vulkan, driver 610.88, Slang v2026.21, with the `dev+subgroups` Gpu step); details in STATUS and D153.
- CI `unit` and `msrv` jobs green on main.
- Evidence tiers and labels: [VALIDATION.md](VALIDATION.md).

Historical handoff records are in
[history/HANDOFF-2026-09-09.md](history/HANDOFF-2026-09-09.md).

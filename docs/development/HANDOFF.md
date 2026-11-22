# GUST handoff

## Current state

- Branch: `main`
- Active: M11 (Realignment, the owner's joint review of the whole project, D198). M10 (publication) closed on 2026-11-17
  (D185). The release hold D189 is in force, so M10 published nothing: the 0.2.0 candidate `83124ad` stays gated and
  unpublished. D226 (2026-11-22) abandoned the 0.1 and 0.2 plans: everything is prerelease, no release line is
  maintained, and the first release waits for maturity. Api2b, api3, the publishable benchmark runs and every public
  speed claim carry to NEXT_TASKS, "M10 follow-ups". Publishing, tagging and yanking stay the owner's acts (D148, RELEASE.md). M9 (language
  breadth II) is closed (D176), M8 (capability variants) is closed (D153), M7 (D126) and MX (D129) are closed, and M6
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

- Last full record: `cargo xtask check-full` and the full `check-release` at `0b51612` (the M10 close, 2026-11-17, 11:32
  to 12:39Z, RTX 5090 / Vulkan, driver 610.88, Slang 2026.22); details in STATUS and D185. Before it: both passed on the
  0.2.0 candidate `83124ad` (2026-11-15, RTX 5090 / Vulkan) and at `ef2a73b` (2026-11-14, 16:16 to 17:40Z, clean tree,
  M9 close, driver 610.88, Slang 2026.22; D176).
- CI `unit` and `msrv` jobs green on main.
- Evidence tiers and labels: [VALIDATION.md](VALIDATION.md).

Historical handoff records are in
[history/HANDOFF-2026-09-09.md](history/HANDOFF-2026-09-09.md).

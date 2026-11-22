# CLAUDE.md

Read [AGENTS.md](AGENTS.md) first; it is the source of truth for every agent. Then read the
"Now and next" section of [docs/ROADMAP.md](docs/ROADMAP.md) and the active section of
[docs/development/STATUS.md](docs/development/STATUS.md). This file adds only Claude Code specifics.

## Roles

AGENTS.md names three worker roles. In Claude Code they map to:

| Role | Who | Rules |
| --- | --- | --- |
| Coordinator | The main session | Plans, lanes, commits, pushes, and the durable state files (roadmap, decisions, status). Avoids implementation. |
| Scout | Agent tool, `model: "sonnet"` (or `"haiku"` for narrow lookups) | Read-only discovery; returns paths, symbols and line references. |
| Builder | Agent tool, `model: "sonnet"` | One bounded slice with a written plan and an allowed-file list; runs the gates the plan names; no commits. |
| Verifier / reviewer | Agent tool, `model: "opus"` for code reviews and `model: "fable"` for the sanity check at the end of each major milestone (owner, 2026-10-23), or Codex | Read-only; never the author of the diff it reviews. Verdict plus findings ranked must-fix, should-fix, nit. |

Set `model` explicitly on every subagent. Use the user-level `fable-planner` skill only when the owner asks for it.
After each slice, check usage; usage is shared with another project on the owner's account.

## Skills

Project skills live in `.agents/skills/<name>/SKILL.md`. Claude Code does not load that folder automatically, so
read the matching `SKILL.md` before work in its area:

- `rust-gpu-ast-validation`: accepting or rejecting Rust constructs in `#[gpu_module]` modules.
- `rust-to-slang-lowering`: syn AST to Slang emission rules.
- `slang-language`: generated Slang and slangc target issues.
- `compiler-testing`: regression, golden and differential tests.
- `wgpu-runtime`: headless execution, buffers, dispatch, readback.
- `spirv-validation`: exported SPIR-V and `spirv-val`.
- `gpu-vertical-slice-verification`: end-to-end evidence claims.

## Commits

End commit messages with the attribution trailer the harness supplies. Never commit machine-specific paths; the
repository is public and `cargo xtask lint` checks for them.

## Compact instructions

A summary must keep:

- the owner's standing mandate and approvals (pointer to memory), quoted decisions not yet in `docs/DECISIONS.md`;
- branch, last pushed commit, and uncommitted work;
- in-flight work: running agents and background commands, and what to do when each finishes;
- machine-sharing state: quiet windows announced by other sessions, lanes in use;
- the current roadmap item and its next step.

Drop tool output, diffs, and tables already recorded in the roadmap or STATUS. After compacting, re-read the
roadmap's "Now and next", the STATUS active section, and `git status --short --branch` before acting.

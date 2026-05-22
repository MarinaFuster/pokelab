---
name: experiment-executor
description: Execute an ML experiment defined in execution_plan.md, using README.md for context. Use when the user asks to run, execute, or carry out an experiment plan. Always use this skill when both an execution_plan.md and README.md are present and the user wants to proceed with execution.
---

# Experiment Executor

Execute the experiment defined in `<folder>/execution_plan.md`, using `<folder>/README.md` for full context.

## Parameters
- **folder** (required): path containing both files. Ask if not provided.

## Behavior
1. Read `<folder>/README.md` for context (question, simplifications, experiment idea)
2. Read `<folder>/execution_plan.md` as the authoritative instruction set
3. Follow the Steps section sequentially
4. Track every deviation from the plan as it occurs (see Output)
5. Stop and ask the user before any dangerous operation (see below)

## Stop for human approval before
- Accessing, reading, or writing credentials or API keys (does not include loading from a `.env` file)
- Installing packages not listed in `execution_plan.md`
- Installing anything from an unverified or non-standard source
- Deleting or overwriting files not explicitly mentioned in the plan
- Making network requests to external services not mentioned in the plan
- Any operation that cannot be easily undone
- Committing to GitHub

## Output: `<folder>/execution_report.md`

Write this file when all steps are complete (or execution is halted).

```markdown
# Execution Report

## Status
[Completed / Halted at step N — <reason>]

## Artifacts Produced
- <path>: <what it contains>

## Deviations from Plan
- [Step N] <what the plan said> → <what was actually done> — <reason>
- None (if fully followed)

## Token Usage
| Component | Tokens |
|-----------|--------|
| System prompt | — |
| Tool definitions | — |
| Skills | — |
| MCP | — |
| Conversation | — |
| **Total** | **— / 300K (—% full)** |

## Errors & Warnings
- [Step N] <error that was bypassed, suppressed, or worked around without fixing the root cause> — <what was done instead>
- None (do not list bugs that were simply fixed and resolved)
```

## Rules
- Do not deviate from the execution plan without asking first
- If something in the plan is ambiguous, stop and clarify before proceeding
- Prefer failing loudly over guessing silently
- **Do not interpret, analyse, or draw conclusions from results** — record outputs and artifacts only; leave conclusions to the user
- Every deviation, even minor ones (e.g. version mismatch resolved automatically), must be logged
- **Do not complete 🔎 Final Comments** in README.md, just complete 📊 Results section.
- Do not repeat deviations, constraints, etc. everywhere. If you have filed them in execution_report.md, do not include that in the .ipynb or README.md
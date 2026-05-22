---
name: execution-plan-generator
description: Read a writeup.md and produce a detailed execution_plan.md for an ML experiment. Trigger when user asks to create an execution plan, generate a plan from a writeup, or prepare an experiment for agent execution. Always use this skill when a writeup.md or experiment description needs to be turned into a concrete, executable plan.
---

# Execution Plan Generator

Read `<folder>/writeup.md` and produce `<folder>/execution_plan.md` that an agent can execute without ambiguity.

## Parameters
- **folder** (required): path containing `writeup.md`. Ask if not provided.

## Behavior

1. Read `<folder>/writeup.md` thoroughly
2. Focus on the "Experiment Idea" section — that is what the agent will implement
3. Check system constraints before planning:
   - Run `nvidia-smi` (GPU availability/VRAM) and `df -h` (disk space) and `nproc` / `free -h` (CPU/RAM)
   - Factor these into model size, batch size, and parallelism decisions
4. Identify blockers — missing model IDs, datasets, metrics, hyperparameters, credentials
5. Resolve blockers by delegating to sub-skills (see below) or asking the user — one question at a time
6. Once all blockers are resolved, write `execution_plan.md`

## Sub-skill Delegation

When something is unspecified, delegate rather than guess:

| Need | Skill to invoke |
|------|----------------|
| Model not chosen | `select-ml-model` — pass description of what's needed |
| Dataset not chosen | `select-dataset` — pass use-case and requirements |
| Hyperparameters unspecified | `hyperparameters-specification` — pass experiment description |
| Metrics not defined | `metric-definition` — pass what needs to be measured |

## Output: `execution_plan.md`

```markdown
# Execution Plan

## Context
[One-paragraph summary of the experiment]

## System Constraints
- GPU: [model, VRAM] or CPU-only
- Disk available: [GB]
- RAM: [GB]

## Models
[HuggingFace model IDs with reason; sized to fit system constraints]

## Reproducibility
- Random seeds: [explicit values for Python, NumPy, PyTorch/TF, transformers]
- Pin library versions in requirements

## Prerequisites
[.env variables or secrets a human must set]

## Dependencies
[Key libraries and versions]

## Steps
[Ordered, specific, unambiguous steps]
```

## Rules
- No placeholders or TODOs — every section fully resolved
- Bullet points only, keep sections short
- Don't invent specifics; infer only from writeup + conversation
- Always include explicit seeds for all RNG sources
- Size models/batches to fit available VRAM/RAM
- If prerequisites exist, also create `<folder>/instructions.md` with human-readable setup steps
- Install Python packages under `pokelab/.venv`
- **PoC bias**: prefer the simplest approach that can validate the hypothesis — avoid over-engineering
- **Widely adopted first**: prefer well-known libraries and tools (e.g. `transformers`, `sklearn`, `pytorch-lightning`) over niche alternatives
- **Idempotency**: every heavy step (dataset download, feature extraction, embedding computation, model training, etc.) must check whether its output already exists and skip if so — include the exact check pattern in the step (e.g. `if not Path("data/features.npy").exists(): ...`)
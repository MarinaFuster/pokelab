---
name: select-ml-model
description: Choose appropriate HuggingFace (or other) ML models given a description of what's needed. Use when a model needs to be selected for an experiment, task, or pipeline and the user hasn't specified one. Triggers on phrases like "which model should I use", "pick a model for", or when invoked by another skill that needs model selection.
---

# Select ML Model

Given a description of what's needed, recommend one or more concrete model IDs.

## Inputs
- **description** (required): what the model needs to do, constraints (size, speed, license, modality)
- **system_constraints** (optional): available VRAM/RAM, CPU-only flag

## Process

1. Clarify if needed (one question): task type, size budget, open vs proprietary, inference speed requirements
2. Search HuggingFace or known sources for candidates
3. Filter by system constraints (VRAM/RAM fit, quantization if needed)
4. Return 1–2 ranked options

## Output

```
## Recommended Models

1. **<model-id>** — <1-line reason, parameter count, VRAM estimate>
2. **<model-id>** — <alternative with tradeoff noted>

**Recommended**: <model-id> for <primary reason>
```

## Rules
- Always include HuggingFace model ID (e.g. `mistralai/Mistral-7B-v0.1`)
- If system constraints provided, only recommend models that fit
- Prefer models with permissive licenses unless told otherwise
- **PoC bias**: prefer smaller, well-known models (e.g. `bert-base`, `distilbert`, `mistral-7b`) over large or exotic ones — good enough to validate the idea is sufficient
- **Widely adopted first**: favour models with high download counts and active community support
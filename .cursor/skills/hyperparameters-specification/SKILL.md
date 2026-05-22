---
name: hyperparameters-specification
description: Specify concrete hyperparameter values for any algorithm or model — neural nets, UMAP, clustering, dimensionality reduction, classical ML, etc. Use when hyperparameters need to be defined and the user hasn't provided them. Triggers on "what hyperparameters should I use", "set up config", or when invoked by another skill that needs parameter values.
---

# Hyperparameters Specification

Given an algorithm/model and use-case, return concrete parameter values ready to paste into code.

## Inputs
- **description** (required): algorithm name, task, dataset size/shape, any known constraints
- **system_constraints** (optional): RAM, CPU-only, time budget

## Process

1. Clarify if needed (one question): the single most impactful unknown (e.g. dataset size for UMAP, task type for sklearn estimators)
2. Start from the algorithm's documented defaults or widely cited community practice
3. Adjust only where the use-case gives a clear reason — note each adjustment and why
4. Flag parameters where the right value is genuinely unknown without more experimentation

## Output

```python
# <AlgorithmName> parameters
# Source: <library defaults / paper / community standard>
config = {
    "param_a": <value>,   # default; safe starting point
    "param_b": <value>,   # adjusted because <specific reason from use-case>
    "param_c": <value>,   # ⚠️ dataset-dependent — tune if results are poor
    "random_state": 42,
}
```

**Start here if tuning**: `<most impactful param>` — <why>

## Rules
- Always include `random_state` / `seed` where the algorithm supports it
- Defaults first — only deviate when the use-case specifically justifies it; state the reason inline
- If the right value is unknown or dataset-dependent, say so explicitly (use ⚠️) — do not invent a confident number
- Do not assume neural-net framing; parameters like `n_neighbors`, `min_dist`, `eps`, `C`, `max_iter` are all fair game
- **PoC bias**: lean toward defaults — they exist because they work broadly; exotic tuning can come later
- **Widely adopted**: cite the source (library docs, original paper, common community practice) so the user can verify
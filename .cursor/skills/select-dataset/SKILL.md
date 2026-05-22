---
name: select-dataset
description: Choose an appropriate dataset for an ML experiment or task. Use when a dataset needs to be selected and the user hasn't specified one. Triggers on "which dataset should I use", "find a dataset for", or when invoked by another skill that needs dataset selection.
---

# Select Dataset

Given a use-case description, recommend concrete datasets with access instructions.

## Inputs
- **description** (required): task, domain, data modality, size constraints
- **constraints** (optional): license, language, max size, must be downloadable without auth

## Process

1. Clarify if needed (one question): task type, size budget, license requirements
2. Identify 1–3 candidate datasets (HuggingFace Hub preferred for easy access)
3. Verify they fit constraints

## Output

```
## Recommended Datasets

1. **<dataset-id>** — <size, license, why it fits>
2. **<dataset-id>** — <alternative with tradeoff>

**Recommended**: `datasets.load_dataset("<id>")` — <primary reason>
```

## Rules
- Prefer HuggingFace Hub datasets (easy `load_dataset` access)
- Include dataset size and license
- Include the exact load call if possible
- Note any required auth (HF token, manual download)
- Flag if dataset is too large for stated disk constraints and include any special handling to download just a subset
- **PoC bias**: prefer small, well-known benchmark datasets (e.g. `imdb`, `squad`, `glue`) over large custom ones — enough to validate the approach
- **Widely adopted first**: favour datasets commonly used in papers/benchmarks for the task; familiar baselines make results easier to interpret
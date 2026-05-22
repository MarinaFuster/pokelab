---
name: metric-definition
description: Choose and define evaluation metrics for an ML experiment or task. Use when metrics need to be selected and the user hasn't specified them. Triggers on "what metrics should I use", "how should I evaluate", or when invoked by another skill that needs metric definitions.
---

# Metric Definition

Given a description of what needs to be measured, return concrete metrics with implementation references.

## Inputs
- **description** (required): task type, what success looks like, any constraints (interpretability, speed)
- **context** (optional): dataset properties (class imbalance, multi-label, etc.)

## Process

1. Clarify if needed (one question): primary goal (ranking vs threshold, offline vs online)
2. Recommend 1 primary metric + 1–2 secondary metrics
3. Note any dataset properties that affect metric choice (e.g. class imbalance → F1 over accuracy)

## Output

```
## Metrics

**Primary**: <metric> — <why it's the right main signal>
- `<library>.<function>(<args>)`

**Secondary**:
- <metric>: <what it adds> — `<library>.<function>(<args>)`
- <metric>: <what it adds> — `<library>.<function>(<args>)`

**Watch out for**: <any known pitfall for this setup>
```

## Rules
- Always give a concrete `sklearn.metrics` or equivalent call
- Justify the primary metric choice relative to the task
- Flag class imbalance, multi-label, or regression-specific concerns
- Keep secondary metrics to ≤2 — don't overwhelm.
- **PoC bias**: pick the single clearest metric that confirms or refutes the hypothesis — avoid metric overload. If one metric is enough for a PoC, return just that metric.
- **Widely adopted first**: prefer standard metrics for the task (e.g. F1 for NER, BLEU for translation, accuracy for balanced classification) so results are easy to compare
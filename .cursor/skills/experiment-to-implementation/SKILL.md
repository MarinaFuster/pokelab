# Execution Plan Generator

Read `<folder>/writeup.md` (full context: question, simplifications, thought process, experiment idea) and produce `<folder>/execution_plan.md` that an agent can execute without ambiguity.

## Parameters
- **folder** (required): path containing `writeup.md`. Ask if not provided.

## Behavior
1. Read `<folder>/writeup.md` thoroughly
2. Focus on the "Experiment Idea" section as the target to execute — that is what the agent will implement
3. Identify any open questions or missing details that would block execution
4. Iterate with the user — ask one question at a time until all blockers are resolved
5. Only then write `execution_plan.md`

## Output file: `execution_plan.md`

```markdown
# Execution Plan

## Context
[One-paragraph summary of the experiment drawn from writeup.md]

## Models
[HuggingFace model IDs to use, with brief reason for each]

## Prerequisites
[.env variables or secrets a human must set before the agent runs]

## Dependencies
[Key libraries or tools needed]

## Steps
[Ordered, specific steps an agent can follow to execute the experiment]
```

## Rules
- Every section must be fully resolved — no placeholders or TODOs
- Keep each section short — bullet points only
- Don't invent specifics; infer only what the writeup and conversation support
- If any prerequisites exist (API keys, .env variables, account setup), also create `<folder>/instructions.md` with human-readable steps to get those set up before running the agent
- Python packages should be installed under `pokelab/.venv` environment
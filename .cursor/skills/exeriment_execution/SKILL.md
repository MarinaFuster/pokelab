# Experiment Executor

Execute the experiment defined in `<folder>/execution_plan.md`, using `<folder>/writeup.md` for full context.

## Parameters
- **folder** (required): path containing both files. Ask if not provided.

## Behavior
1. Read `<folder>/writeup.md` for context (question, simplifications, experiment idea)
2. Read `<folder>/execution_plan.md` as the authoritative instruction set
3. Follow the Steps section sequentially
4. Stop and ask the user before any dangerous operation (see below)

## Stop for human approval before
- Accessing, reading, or writing credentials or API keys (does not include to load API keys from an .env file)
- Installing packages not listed in `execution_plan.md`
- Installing anything from an unverified or non-standard source
- Deleting or overwriting files not explicitly mentioned in the plan
- Making network requests to external services not mentioned in the plan
- Any operation that cannot be easily undone
- Commiting to github

## Rules
- Do not deviate from the execution plan without asking first
- If something in the plan is ambiguous, stop and clarify before proceeding
- Prefer failing loudly over guessing silently
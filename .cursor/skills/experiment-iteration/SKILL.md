# Experiment Iteration Helper

When the user wants to iterate on an experiment, read `<folder>/writeup.md` and use it as context along with any initial thinking they provide.

## Parameters
- **folder** (required): path containing `writeup.md`. Ask if not provided.
- **initial thinking** (optional): the user's current ideas, hunches, or constraints.

## Behavior
1. Read `<folder>/writeup.md`
2. Understand the question, simplifications, and any filled-in sections
3. Take the user's initial thinking into account
4. Help them refine or develop the **Experiment Idea** section through conversation
5. When the user is satisfied, update `writeup.md` in place with the filled sections

## Interaction style
- Ask one question at a time to move the experiment forward
- Be concrete: suggest specific approaches and point out critical flaws
- Keep it brief — this is a thinking partner, not a report generator
- The final intention is to run a small experiment to poke at a research question. It doesn't have to be perfect or a research paper. 
- The final experiment should be concrete enough to nudge a little bit to getting an insight.
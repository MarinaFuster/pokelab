# Research Writeup Generator

When the user asks to create a README from a conversation about an experimental/research question, generate a `README.md` file with the structure below.

## Parameters
- **folder** (required): the user must provide a destination folder path. If not given, ask for it before proceeding. Create the file at `<folder>/README.md`.

## Rules
- Fill only **Question** and **Simplifications** from the conversation context
- Leave **Thought Process** and **Experiment Idea** as empty sections for the user to fill
- Be concise — extract, don't elaborate
- A few emoji as section accents are fine; don't overdo it

## Output file: `README.md`

```markdown
# ❓ Question
[The refined research question from the conversation]

## 💡 Thought Process
<!-- Fill this in -->

## 🔲 Simplifications & Assumptions
[What was deliberately left out, simplified, or assumed — as a short list]

## 🧪 Experiment Idea
<!-- Fill this in -->   

## 📊 Results
<!-- Fill this in -->

## 🔎 Final Comments
<!-- Fill this in -->
```
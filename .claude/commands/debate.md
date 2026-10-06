---
description: Run a structured, one-round agent debate on a decision and get a recommendation.
argument-hint: <decision or question to debate>
---

Run a structured debate on: **$ARGUMENTS**

Follow these steps in order, and keep each agent's output in its required format:

1. **focus**: Confirm the decision is in scope and state the project question it serves. If it's out of scope, stop here and report.
2. **researcher** (only if the decision depends on facts we don't have): Find the most relevant evidence using generic method terms only.
3. **depth-maxer**: Propose the most rigorous in-scope approach.
4. **discourse**: Challenge that proposal (max 3 counterarguments, one round).
5. **You (main session) moderate**: Do not run further rounds. Summarize:
   - **Agreement:** where the agents align
   - **Disagreement:** the real trade-off(s), in one line each
   - **Recommendation:** one path, with the deciding reason
   - **Decision needed from user:** yes/no, and what exactly

Never read data files or `.env`, and never include data values in any step.

---
description: Run a structured two-round agent debate on a decision and log the outcome.
argument-hint: <decision or question to debate>
---

Run a structured debate on: **$ARGUMENTS**

All agents read `docs/decisions.md` first. Keep each agent's output in its required format. Never read data files or `.env`, and never include data values in any step.

## Round 1: Positions
1. **focus**: Confirm scope and state the project question this serves. If out of scope, stop and report.
2. **depth-maxer**: Propose the most rigorous in-scope approach.
3. **discourse**: Challenge the proposal (max 3 objections).
4. **researcher**: Resolve any `HANDOFF → researcher` requests from steps 1–3 (generic search terms only). Skip if there are none.

## Round 2: Rebuttal
5. Send discourse's objections (and any researcher evidence) back to **depth-maxer**: respond to each objection with concede, revise, or defend.
6. **discourse**: Final reply to depth-maxer's responses only. No new objections.
7. **focus**: Final scope check on the revised proposal.

Stop after round 2. No further rounds.

## Moderation (main session)
Summarize:
- **Agreement:** where the agents converged
- **Remaining disagreement:** the real trade-off(s), one line each
- **Recommendation:** one path, with the deciding reason
- **Decision needed from user:** yes/no, and exactly what

Then **append an entry to the top of `docs/decisions.md`** using its template, with `Decision:` marked "PENDING: awaiting user" until the user decides. Once the user decides, update the entry.

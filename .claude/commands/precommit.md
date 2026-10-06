---
description: Run quality-control on staged changes before committing.
---

Use the **quality-control** agent to review the currently staged changes (`git diff --cached`).

Report BLOCKING issues first. If there are none, say "Clear to commit" and suggest a concise commit message. Do not commit, push, or modify files yourself.

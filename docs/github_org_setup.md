# Proposal: move `predictleads-hackathon` into a shared GitHub organization

Hi Jesse, here's what I'm proposing for where our merged project lives on GitHub.

## The idea

I create a free GitHub organization with **both of us as owners**. You move `predictleads-hackathon` into it (same name), and later my `project_leads` merges into it as described in the merge plan.

**Proposed organization name: `career-compass-lab`.** It names the core idea (the retraining compass), doesn't use PredictLeads' name, and still fits if we keep building together after the externship. Happy to pick something else if you prefer.

## Why

- **It's shared:** we both have full control; neither of our personal accounts "owns" the project.
- **It shows for both of us:** the organization appears on both our profiles, our commits count on both contribution graphs, and we can both pin the repo.
- **Nothing breaks:** a GitHub transfer keeps all history, branches and issues, and the old URL redirects to the new one, so existing clones and links keep working.
- **It's free and stays private** (the free plan includes private repositories). It should stay private, since the repo describes PredictLeads data under our agreement.

## Steps

**Me:**
1. Create the organization (Free plan), with repositories private by default.
2. Invite you as an **Owner** (your username: `JGM-droid`).

**You:**
3. **Accept the invite** (GitHub email, or the banner on github.com). This has to come first: GitHub only lets you move a repo into an organization you belong to.
4. **Transfer the repo:** `predictleads-hackathon` → **Settings** → **Danger zone** → **Transfer ownership** → `career-compass-lab`. It stays private, with the same name.
5. **Optional, on your laptop:** point your clone at the new address (the redirect also works without this):
   ```
   git remote set-url origin https://github.com/career-compass-lab/predictleads-hackathon.git
   ```

**Later, both of us:** when we're ready to merge, my `project_leads` goes into this repo as planned.

## Before we do it

If TripleTen or PredictLeads set up a specific place for externship code (an organization or repo), we should use that instead. I'll check with the advisor.

Let me know if the name works for you, and I'll send the invite.

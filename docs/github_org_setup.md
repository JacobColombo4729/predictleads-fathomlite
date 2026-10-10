# Proposal: move `predictleads-hackathon` into a shared GitHub organization

Hi Jesse, here's what I'm proposing for where our merged project lives on GitHub.

## The idea

We create a free GitHub organization with **both of us as owners**, move `predictleads-hackathon` into it (same name), and later merge my `project_leads` into it as described in the merge plan.

## Why

- **It's shared:** we both have full control; neither of our personal accounts "owns" the project.
- **It shows for both of us:** the organization appears on both our profiles, our commits count on both contribution graphs, and we can both pin the repo.
- **Nothing breaks:** a GitHub transfer keeps all history, branches and issues, and the old URL redirects to the new one, so existing clones and links keep working.
- **It's free and stays private** (the free plan includes private repositories). It should stay private, since the repo describes PredictLeads data under our agreement.

## Steps (about 10 minutes)

1. **One of us creates the organization:** GitHub → avatar → **Your organizations** → **New organization** → **Free**. Name to agree, for example `predictleads-team`.
2. **Invite the other as an owner:** organization → **People** → **Invite member** → role **Owner**.
3. **You transfer the repo:** `predictleads-hackathon` → **Settings** → **Danger zone** → **Transfer ownership** → the organization. It stays private, with the same name.
4. **Optional, on each laptop:** point the local clone at the new address (the redirect also works without this):
   ```
   git remote set-url origin https://github.com/<organization>/predictleads-hackathon.git
   ```
5. **Later:** when we're ready to merge, my `project_leads` goes into this repo as planned.

## Before we do it

If TripleTen or PredictLeads set up a specific place for externship code (an organization or repo), we should use that instead. Worth checking with the advisor first.

Let me know if this works for you, and which of us should create the organization.

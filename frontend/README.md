# PredictLeads job seeker prototype

Frontend mock for the job seeker platform. All data is fictional; there is no backend, authentication, or external service.

## Run

Requires Node.js 18 or newer.

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173/. Pages are hash routes: `#/discover`, `#/saved`, `#/profile`.

Production build and preview:

```powershell
npm run build
npm run preview
```

The preview is served at http://127.0.0.1:4173/.

## Milestones

Roadmap: [design/ui-roadmap/UI_ROADMAP.md](design/ui-roadmap/UI_ROADMAP.md).

| Milestone | Status |
| --- | --- |
| 1. Navigation and Discover Jobs | Done |
| 2. My Profile | Done |
| 3. Saved Jobs with stages and notes | Done |
| 4. Job and company details | Done |
| 5. Skills & Paths | Not started; pending review |

## Structure

- `src/types.ts`: `Recommendation`, `Profile`, `TrackedJob` and related types, the contract for later backend integration.
- `src/data/`: fictional jobs, candidate profile and seed saved jobs. Replace through a data adapter when connecting real data.
- `src/state/`: `useProfile` and `useTracker` hooks with versioned, validated `localStorage` persistence.
- `src/pages/`: Discover Jobs, Saved Jobs, My Profile.
- `src/components/`: shared presentational components (`JobCard`, `JobDetails`, `FilterBar`, `SideNav`, `StageSelect`, `Toast`).
- `src/styles/tokens.css`: design tokens (color, type, spacing, radius).
- `design/ui-roadmap/`: roadmap and design-study reference images (not shipped in the app).

## Behavior notes

- Result order, fit labels and explanations are demo fixtures, not a validated matching result. Editing the profile does not change them; the app says so wherever they appear.
- Profile edits and saved-job stages and notes persist in this browser's `localStorage` (keys `pl.profile`, `pl.tracker`, schema version 1). Unreadable stored data is discarded and the user is told. The browser is not an account or a secure store; do not enter real personal information.
- First run seeds three fictional saved jobs. Clear site data to reseed.
- A saved job's stage is the user's own workflow state. It is separate from the source listing's historical status.
- Source links are placeholders on `example.com` and open with `noopener noreferrer`.

## Limitations

- Fit evidence is not recomputed from the profile.
- Unsaved profile edits are lost when navigating to another page.
- Company details come from fixtures; unknown values are shown as unknown.

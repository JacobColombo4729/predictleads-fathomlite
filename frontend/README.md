# PredictLeads job-seeker app (frontend)

React 18 + TypeScript + Vite. It talks to the demo backend (see [docs/backend-api.md](../docs/backend-api.md)) through the dev server's `/api` proxy, so the session cookie stays same-origin with the page. The browser keeps no copy of record; the server session does.

## Run

Requires Node.js 18 or newer, and the backend running on http://127.0.0.1:8000 (`..\run_demo_backend.ps1`).

```powershell
cd frontend
npm install
npm run dev        # http://127.0.0.1:5173
```

Pages are hash routes: `#/discover`, `#/saved`, `#/profile`. Other commands:

```powershell
npx tsc -b         # type-check
npm run build      # production build into dist/ (git-ignored)
npm run preview    # serve the build at http://127.0.0.1:4173 (same /api proxy)
```

`..\run_tests.ps1 -Frontend` runs the type-check and build with the Python tests. There are no automated frontend tests.

## What the app does

- **Profile:** starts blank. Fill the form, import a candidate-profile JSON, or use **Magic fill** (a text-based PDF or `.txt` resume, after a consent checkbox; needs the backend started with `-ResumeFill`). Magic fill replaces name, headline, location, experience and skills with what the resume says. Each experience entry has a **Type** dropdown (Job, Project, Education). Desired roles are never filled automatically. Nothing is saved until **Save profile**. **Start over** deletes the server session (profile, saved jobs, ratings).
- **Discover:** ranked listings for the saved profile, with search and location filters. Each job can be marked 👍 *Good match* or 👎 *Not for me*, which refreshes the list. With a blank profile it shows "Create your profile to see matches".
- **Job details:** description, source link and historical-snapshot labelling, plus a keyword-based skills check. "Not evidenced in your profile" means unknown, not missing. The grounded gap analysis is not connected here yet.
- **Saved:** saved jobs with a workflow stage and notes. Tick jobs (up to 20) and press **Send to Compass**.
- **Compass:** an analysis of the jobs you sent, built by the backend from your saved profile: an animated "analyzing" sequence, then a compass map (you at the centre, closer means a closer fit, angle groups jobs into themes, diamonds are suggestions you didn't pick), a detail panel with why a job fits, tiers you can shift with a stretch slider, what your picks ask for next to the closest line of your own profile, suggested next steps, and "hidden fit" suggestions. Needs the backend's RRF ranker. A job's stage is the user's own state and is separate from the source listing's historical status.

## Structure

- `src/App.tsx`: routing and page wiring. `src/types.ts`: shared types.
- `src/lib/api.ts`: the typed API client. `src/state/useLive.ts`: session, profile, recommendations, ratings and saved jobs.
- `src/lib/profile.ts`: limits, validation, entry types and the blank profile. `src/lib/importProfile.ts`: JSON import. `src/lib/gap.ts`: the keyword skills check.
- `src/pages/`: Discover, Saved Jobs, Compass, Profile. `src/components/`: shared components; `src/components/compass/` holds the Compass map, tier board, requirement bars and the rest. `src/state/useCompass.ts` runs the analysis; `src/lib/compass.ts` has the geometry and tier logic. `src/styles/tokens.css`: design tokens.
- `design/ui-roadmap/`: early design study and reference images (not shipped).

## Theme

Dark (near-black with a green accent) is the default; the sun/moon button at the bottom of the side nav switches to light and the choice is remembered. All colours are tokens in `src/styles/tokens.css`; the Compass chart colours (three theme hues, one sequential hue, reserved status colours) are in `src/styles/compass.css` with separate light and dark steps.

## Limitations

- Profile and saved-job data live in an anonymous server session (24-hour sliding expiry); there are no accounts.
- Unsaved profile edits are lost when navigating to another page.
- Match quality is unvalidated, and listings are historical, not verified openings.
- The Compass result lives in memory: reloading the page clears it, and you send the picks again from Saved.
- The Compass's evidence matching and thresholds are heuristics and are not validated; its design preview is in `design/compass-preview/`.
- "Skills & Paths" (milestone 5 of the design roadmap) is largely covered by the Compass; the learning-path view is a simple status list.

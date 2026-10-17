/**
 * The talking points shown in admin mode: for each machine-learning part of the app, what it does, why we chose it,
 * the data it uses, how it was checked and where it is weak. Wording follows app/docs/ml-inventory.md and the
 * analysis notebooks; change a number there and here together. Nothing here is computed: it is plain text.
 */
export interface MlNoteText {
  title: string;
  what: string;
  why: string;
  data: string;
  check: string;
  limits: string;
}

export type NoteId =
  | "ranker" | "feedback"
  | "compass" | "compass-map" | "tiers" | "requirements" | "gap" | "next-steps"
  | "signals" | "ask-market" | "company-research"
  | "magic-fill";

interface NoteEntry extends MlNoteText {
  /** Replacements for particular setups, such as the ranker the server runs or how the resume is read. */
  variants?: Record<string, Partial<MlNoteText>>;
}

export const ML_NOTES: Record<NoteId, NoteEntry> = {
  ranker: {
    title: "The job ranker",
    what: "Two retrievers run side by side: TF-IDF (keyword match, so exact terms like “Salesforce” count) and BGE embeddings (bge-base-en-v1.5, 768 numbers per listing, so “customer-facing demos” can match “sales engineering”). Their rankings are fused with reciprocal-rank fusion (depth 50, k = 60, equal weight). Nothing is trained.",
    why: "There are no match labels, so a pretrained model plus a transparent fusion rule is the honest cold-start. Keyword and meaning fail in different ways, and fusing ranks (not scores) needs no calibration or tuning.",
    data: "PredictLeads job openings (one historical snapshot). Only listings with a usable description; reposts merged when company, title and description match. Each listing is embedded once, ahead of time; your profile is embedded when you search.",
    check: "Not yet shown to beat plain keywords on this audience. On Jacob's held-out career moves the text methods tie: RRF 70.0% vs TF-IDF 69.8% for “right role in the top 3”. Call it a defensible default, not a proven winner.",
    limits: "The number on each card is an order, not a match percentage or a hiring chance. Dates are when the crawler saw a listing, not when it was posted.",
    variants: {
      combined: {
        title: "The combined ranker (text match + Jacob's models)",
        what: "Three layers. (1) The text match above finds listings that read like your resume. (2) ML_Fit, a logistic regression trained on 2,813 real career moves, says which of 8 role families people with your background actually move into; the best text matches inside those roles join the pool. (3) ML_Rank's queue orders the pool by four ratings times tuned weights: fit, chance the listing is still open (XGBoost), accessibility for career changers, and pay. Likely ghost listings and manager-level roles are hidden.",
        why: "Text match alone favors your old field: for people moving within tech it would put their current role first about 74% of the time. The trained layer learns from where switchers really went, so it does not (0.1%).",
        data: "Public resumes (Kaggle, 2,813 career changers, titles and dates only), O*NET skills tables, and the listing snapshot. The resume you enter is read on this computer.",
        check: "On 563 held-out people, the right role is in the top 3 for 81.7% [78.5, 84.7] and the top 5 for 96.3% [94.7, 97.7], against 63.2% and 87.0% for always suggesting the most common roles. Rebuilt independently on a second computer with the same numbers.",
        limits: "It shows where switchers went, not how likely a switch is to work (the resumes only show people who switched). Eight role families only; new grads are the weakest group (72.2%, not shown to beat the baseline). The listing-level ordering has been backtested on a later month, not on human relevance labels.",
      },
      ml_rank_queue: {
        title: "The ranker with ML_Rank's queue",
        what: "The text match picks the top 200 candidates, then ML_Rank's queue reorders them by fit, chance still open (XGBoost), accessibility and pay. Here “fit” is only the text-match score rescaled so the best is 100; Jacob's role-fit model is not in this version.",
        why: "It puts timing and career-change accessibility into the order, not only similarity.",
        data: "The listing snapshot plus ML_Rank's saved scores for the in-scope listings.",
        check: "The queue weights were tuned on one month and tested once on the next.",
        limits: "The fit factor is a stand-in, so the weights were tuned on a different fit than the one used here.",
      },
    },
  },

  feedback: {
    title: "👍 / 👎 feedback",
    what: "Rocchio relevance feedback (1971): a 👍 pulls your query toward that job (weight 1.0), a 👎 pushes it away (weight 0.5), on both the keyword and the embedding side. Jobs you rated are hidden from new results.",
    why: "It is classic, well understood and instant, and needs no labels or retraining.",
    data: "Only your own ratings in this session, applied to the same listing vectors.",
    check: "Usability feedback only. We do not report the team's ratings as accuracy because they are personal judgments.",
    limits: "It adapts one session; nothing is retrained and ratings end with the session. The weights are conventional defaults, not tuned. You can only rate what the ranker already showed you (selection bias).",
  },

  compass: {
    title: "The Compass",
    what: "From 2 to 20 jobs you pick, it finds what they have in common. Each listing is a BGE embedding. Picks are grouped into at most 3 themes (agglomerative clustering). Requirement sentences are merged when their cosine distance is under 0.22. Each is matched to the closest verbatim line of your own profile.",
    why: "Embeddings compare meaning, so “build dashboards” can match “reporting”. Using your own words as evidence keeps it explainable.",
    data: "The listings you picked, your saved profile (split into up to 90 lines) and the stored embeddings. Sentences are cleaned first: boilerplate and eligibility lines are dropped, lines under 5 words are skipped, at most 18 per job. Everything runs on this computer; nothing is sent to an AI service.",
    check: "Unit tests cover the logic. The thresholds were set by looking at a few real examples: they are heuristics, not validated.",
    limits: "Scores are similarities, never the chance of being hired. “Not mentioned” means your profile doesn't say, which is unknown, not a verdict.",
  },

  "compass-map": {
    title: "The map and its scope",
    what: "Each job sits at an angle set by its role and a distance set by how closely it fits your profile (similarity scaled against all listings). Up to 4 axes pair the least alike role families, taken from the provider's O*NET occupation labels grouped into 16 families. The scope is the direction with the most well-fitting jobs, found with a von Mises kernel.",
    why: "It turns a list of similarity scores into something you can read at a glance and steer.",
    data: "The same embeddings and your profile; the role families come from the provider's own occupation labels.",
    check: "Layout and scope are heuristics. They were checked visually on a few examples.",
    limits: "Distances are comparisons, not probabilities. The family grouping follows the provider's labels, which track the job title closely.",
  },

  tiers: {
    title: "Roles you didn't pick, and the stretch slider",
    what: "Suggestions are listings near your picks and profile, chosen by maximal marginal relevance (relevance minus similarity to what is already chosen, λ = 0.7) so you don't get near-copies. The stretch slider skips further down the similarity ranking. Tiers: Best fit 75 and up, Reach 55 to 74, Stretch below 55.",
    why: "Career changers need ideas that differ in title but are similar in work; plain top-N similarity returns repeats.",
    data: "The stored listing embeddings, your picks and your profile.",
    check: "Tier cut-offs and the redundancy weight were hand-set on a few examples.",
    limits: "A tier says how similar a role is to your profile, not how likely you are to be hired.",
  },

  requirements: {
    title: "What your picks ask for",
    what: "Requirement sentences from your picks are clustered by meaning. For each group we find your closest profile line by cosine similarity and map it to a coverage score: 0.50 or below counts as no evidence (0), 0.80 as strong evidence (100). 70 and up reads “supported”, 40 to 70 “partial”.",
    why: "It shows which expectations your own words already cover, with the evidence in front of you.",
    data: "The picks' descriptions and your profile, embedded with BGE. Local only.",
    check: "Thresholds are heuristics from a few real examples.",
    limits: "Years, degrees and certifications are never auto-marked “supported”. It only knows what your profile says.",
  },

  gap: {
    title: "The AI skills gap analysis (GenAI)",
    what: "One request to a hosted model (gpt-4.1-mini) over all picked jobs returns three things per job: transferable experience, a qualification assessment and a basic skills gap. The answer must follow a strict JSON schema, and every claim must cite evidence IDs from the lines we supplied. A validator rejects any citation that isn't there. With too little evidence it says so instead of guessing.",
    why: "Matching duties to experience takes reading, which a language model does well, so we let it write the explanation but never the evidence.",
    data: "Your profile's experience and skills (not your name, email, phone or links; only an allow-list of fields is passed) and the descriptions of the picked jobs. It is off unless the server enables it, needs your tick-box each time, and the result is cached in your session.",
    check: "Schema and citation validation, and offline tests with a mock provider. Not measured at scale.",
    limits: "Employer names and free text in your entries are still sent. This is not a full anonymizer. The model can still be wrong; every claim shows its evidence so you can check it.",
  },

  "next-steps": {
    title: "Suggested next steps",
    what: "Requirement groups are ordered by how often they appear in your picks and how little your profile shows them. No model is involved beyond the similarity scores above.",
    why: "It turns the gap list into an order you can act on.",
    data: "The requirement groups and coverage scores already computed.",
    check: "Rule-based ordering; tested with unit tests.",
    limits: "Frequency in your picks is not frequency in the market. It is only as good as the coverage scores.",
  },

  signals: {
    title: "Hiring signals",
    what: "For each company: new openings in the last 30 days against that company's own pace in the prior 90 days, using a z-score anomaly test with a count-based variance. The result is burst, sustained, cooling or quiet. Reposts are merged by company, title and description; bulk-import days and crawler dropouts are flagged by rules; nothing looks ahead.",
    why: "A company's own history is a fairer baseline than a global average.",
    data: "PredictLeads listing dates (when the crawler saw a listing) for AI/ML, data and technical-sales roles.",
    check: "Temporal holdout on 23,961 companies, tested on 2026-08-31. Flagged companies stayed above their own baseline next month 59% of the time vs 16% for unflagged, but kept only about 64% of their volume (sustained hirers 99%). We also trained gradient-boosting models; they did not beat “how many openings last month”, so we did not ship them.",
    limits: "Four months of data and one test month. Dates are crawler observation times, not posting dates. A burst is a timing trigger, not a volume forecast.",
  },

  "ask-market": {
    title: "Ask the market",
    what: "A hosted model only translates your question into a structured search. Our code validates it (places, technologies and companies must literally appear in the question or they are removed, and the screen says so), runs it on our listing table and computes the counts and medians. Forecasts and advice are refused by a rule, not by the model.",
    why: "The model never writes the numbers, so it can't invent them, and you can see how the question was read.",
    data: "Only the typed question goes to the model (nothing from your profile); the answer comes from our own listings. Talk records up to 30 seconds, turns it into text and does not keep it.",
    check: "Tests plus live checks on public questions. The engine's numbers are counts and medians of real listings.",
    limits: "The model can misread an unusual question (the read-back chips let you fix that). Counts come from one crawler snapshot.",
  },

  "company-research": {
    title: "Company research",
    what: "An optional step asks a hosted model with web search to summarize one company. We keep only sentences that carry a citation and show the sources.",
    why: "Our data says who is hiring, not what the company does; the web can fill that in, with sources you can open.",
    data: "Only the company's name and website are sent.",
    check: "Tried on one company; not measured at scale.",
    limits: "Web content can be wrong or out of date, and the card says so.",
  },

  "magic-fill": {
    title: "Magic fill (resume to profile)",
    what: "Reads resume text into the profile form. Every extracted item must carry a quote that actually appears in the resume; items whose quote can't be found are dropped and counted. It never guesses desired roles.",
    why: "It saves typing, and the quote check keeps it from inventing anything.",
    data: "The resume you choose. It is not stored by the app.",
    check: "Checked by hand on two real resumes, not measured at scale.",
    limits: "Ordinary one-column resumes read best; check every entry before saving.",
    variants: {
      local: {
        what: "A reader that runs on this computer, with no AI model: it splits the resume by its own headings, anchors each job, project or school on its date range, takes the skills list as written and adds the tools O*NET names. Then the same quote check as the hosted version.",
        data: "The resume you choose, read on this computer. Nothing is sent anywhere.",
        limits: "Rules, not a model: two-column or heavily designed PDFs can come out scrambled. Always review the form before saving.",
      },
      openai: {
        what: "A hosted model (gpt-4.1-mini, temperature 0) reads the resume text and returns structured entries; each must carry a verbatim quote from the resume or it is dropped.",
        data: "With your tick-box, the resume text is sent after emails, phone numbers and links are removed by pattern matching. It is not stored or logged.",
        limits: "Your name and street address are not removed by code (the prompt tells the model to ignore contact details). This is not a full anonymizer.",
      },
    },
  },
};

/** The note for a part, with any replacements for the given setup applied. */
export function noteFor(id: NoteId, variant?: string): MlNoteText {
  const { variants, ...base } = ML_NOTES[id];
  return { ...base, ...(variant ? variants?.[variant] : undefined) };
}

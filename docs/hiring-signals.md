# Hiring signals layer

`code/hiring_signals.py` turns job-listing timestamps into per-company hiring signals. It uses structured fields only (no embeddings), so it scales unchanged to the full 13.4M-listing snapshot. Read [signal-semantics.md](signal-semantics.md) first: `first_seen_at` / `last_seen_at` are crawl observation times, not posting / closing dates.

Build: `python code/hiring_signals.py build --pool <targeted_pool.jsonl> --out data/signals/hiring_signals_YYYYMMDD`
Backtest: `python code/hiring_signals.py backtest --pool <targeted_pool.jsonl> --out <backtest.json>`
Tests: `tests/test_hiring_signals.py` (24).

## What it produces (320k targeted pool → 269,891 listings after merging 50,079 reposts, 42,494 companies)

| File | Contents |
|---|---|
| `company_signals.csv` | per company: new openings in 30 days, baseline, active, gone, surge z-score, `pattern` |
| `intent_signals.csv` | companies hiring in AI/ML, data and technical-sales families (buying-intent proxy) |
| `technology_leads.csv` | technologies named in a company's new openings |
| `company_weekly.csv` | weekly new-opening counts (sparklines) |

Guards against data artifacts: reposts merged; batch-onboarding days and mass-disappearance (crawl dropout) flagged; a listing counts as gone only after a 7-day grace; at least 90 days of history before any surge claim; every signal computed with no look-ahead.

## Patterns

- **burst**: z-score ≥ 2 against the company's own prior-90-day rate, ≥ 3 new openings (771 companies)
- **sustained**: baseline ≥ 3 new openings/month and this month within 0.5–1.6× of it (920)
- **cooling**, **quiet** (692 / 40,111)

## Backtest (train cutoffs 2026-07-01/07-15/08-01, test cutoff 2026-08-31, predict next 30 days; 23,961 companies)

Honest findings:

1. **No model beats "how many did they open last month" for predicting volume.** MAE (log1p): naive 0.249, gradient boosting 0.274, change-model 0.266, linear shrinkage 0.251. Spearman ≈ 0.53–0.55 for all. We do not claim a forecast.
2. **A surge is a trigger, not a forecast.** Surging companies keep 63% of their volume the next month (mean 13.4 → 8.5; 76% still hiring), versus 106% for sustained hirers (94% still hiring). Flagged companies still beat the same-size unflagged ones on being above their own baseline (60% vs 16%), so the flag finds real campaigns; they just taper.
3. **Product framing**: show *burst* as "time-sensitive, act now", *sustained* as "reliable ongoing hiring", with the measured carry-over next to each.

Limits: 4 months of data, one test month, seasonal drift (mean next/this ratio 0.89 → 1.07 across cutoffs), crawl-observed rather than true posting dates, role family from the targeted-pool rules.

## Open questions for PredictLeads

What `last_seen_at` guarantees (see signal-semantics.md), and whether true posting dates exist.

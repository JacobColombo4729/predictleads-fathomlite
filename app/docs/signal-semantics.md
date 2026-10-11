# What `first_seen_at`, `last_seen_at` and `closed` mean

Measured on the 244,558-listing corpus (snapshot 2026-09-30) on 2026-10-10. These are observation timestamps from PredictLeads' crawl. The data guide says neither is a posting or closing date, and the checks below agree: they are good *proxies*, with known artifacts. Signals built on them should be called **observed** hiring activity.

## `first_seen_at` is roughly "went live", usually within a few days

Compared with the listing's own `posted_at` (available for 130,346 listings):

| `first_seen_at` minus `posted_at` | Share |
|---|---|
| under 1 day | 29% |
| 1 to 7 days | 46% |
| 7 to 30 days | 11% |
| 30 to 90 days | 3% |
| over 90 days | 1% |
| first seen *before* `posted_at` | 10% |

Median lag 1.4 days, 75% within 3.3 days. So it tracks when a listing appeared, not when it was created, and about one listing in ten carries a posted date later than the first sighting.

**Artifact: batch onboarding.** 8.9% of companies with 5 or more listings had 60% or more of them first seen on a single day, and single days reach about 5,000 first sightings against a median of about 1,300 (2026-05-15, 2026-04-14, 2026-04-10). That looks like a company being added to the crawl or a site being rebuilt, not a hiring surge. A momentum signal must exclude or flag these.

## `last_seen_at` is roughly "last still online", but it is not a close date

- 88% of listings with unknown status were last seen within 7 days of the snapshot (still observed).
- 63% of `closed` listings were last seen more than 60 days before the snapshot, and the rest fade out gradually.
- About 12% of unknown-status listings had already disappeared 8 or more days before the snapshot **without ever being marked closed**: the `closed` label under-reports removals.
- Only 5.0% of companies show a crawl-dropout pattern (60% or more of their vanished listings last seen on one day). For the other 95%, disappearances are staggered, which is what individual closures look like.

## Rules for building signals

1. Describe them as "observed openings" and "observed disappearances", never "posted" or "filled".
2. Active now = last seen within 7 days of the snapshot and not `closed`.
3. Exclude or flag company-days with a batch of first sightings, and companies whose disappearances cluster on one day.
4. Use windows of at least two weeks; single days are dominated by crawl noise.
5. Judge momentum against the company's own history, since 17.9% of all listings were first seen in the last 30 days.
6. Validate by backtest: compute signals at an earlier cutoff and check whether they predict later openings.

## Questions worth asking PredictLeads

- How often is each site crawled, and how is `last_seen_at` set when a crawl fails?
- Is the "deleted job opening" webhook a better closure event than `last_seen_at`?
- Are company onboarding dates available, so batch discovery can be separated from real hiring?

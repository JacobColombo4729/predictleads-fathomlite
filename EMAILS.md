# Emails

Drafts ready to send. Fill in the bracketed parts. When answers come back, record them in the [Decision log](PROJECT_PLAN.md#8-decision-log) and remove the answered questions from [`PRIORITY_QUEUE.md`](PRIORITY_QUEUE.md).

Some questions appear in both emails, such as whether real postings can appear in the final presentation. If the answers differ, follow the stricter one.

---

## 1. Project advisor

**To:** [Advisor's name and email]
**Subject:** Questions on our PredictLeads project direction and data use

Hi [Name],

I hope you're doing well. Our team has narrowed the focus of our PredictLeads project, and before we go further, I'd like to check a few things with you, especially around data use and AI tools.

**Our proposed direction.** A job listing prioritization project. A job seeker sees far more listings than they can apply to, so we want to rank job listings for a given applicant by which to apply to first. The ranking would combine time series analysis (how long listings like it stay open, how fast new ones arrive) with NLP on the listing text (employer intent, competition signals, fit with the applicant's résumé), for one occupation group in one country. We'd also like to build a web app that shows this ranking as a priority queue, with a multi-agent design using MCP.

My questions:

1. **Scope:** does job listing prioritization seem an appropriate focus for the project?
2. **Deliverables:** what's the final due date, and what format do you expect: a written report, a presentation, a demo, or all three?
3. **Web app and multi-agent requirements:** is there a rubric or a list of must-haves for the web app and the multi-agent part?
4. **AI agents and data:** our current understanding is that no confidential project data may be uploaded to outside AI tools. For the app's agents, may an agent backed by an outside model (such as Claude or GPT) process posting text? Or should it only see aggregate statistics, our own model outputs and public data, with anything that reads posting text running locally? Are locally run models fine to use on the data?
5. **Résumés:** for the app's "fit" feature, may team members use their own résumés, with names, contact details and employers removed? Are there rules on how we store them?
6. **External data:** may we combine public datasets, such as BLS job openings and wage statistics or O*NET occupation data, with the PredictLeads data?
7. **Hosting:** may the app be hosted somewhere, and if so, where is approved? Or should it stay a local demo, since the data can't leave approved locations?
8. **Presenting results:** may our final presentation show real job postings and company names, or should we use only aggregate results and synthetic examples?
9. **Data deletion:** what's the procedure for deleting the data at the end of the project, and do you need written confirmation?

Thank you for your time. Happy to set up a quick call if that's easier than email.

Best,
Jacob Colombo

---

## 2. PredictLeads

**To:** [PredictLeads contact's name and email]
**Subject:** TripleTen externship: project direction and questions about the job openings data

Hi [Name],

Thank you for sharing the companies and job openings datasets with our TripleTen team. We've loaded and explored a sample, and we have a proposed direction for the project, plus some questions about the data so we interpret it correctly.

**Our proposed direction.** A job listing prioritization project: ranking job listings for a given applicant by which to apply to first. The ranking would combine how long listings like it stay open, how fast new ones arrive on different hiring platforms and company sites, competition signals such as reposts, pay, what the listing text says, and fit with the applicant's résumé, for one occupation group in one country.

1. **Project fit.** Does that direction fit what you hoped to get from this project? Is there a use case, customer type or deliverable format you'd like us to aim for instead?

2. **Schema documentation.** Do you have documentation for the job openings and companies datasets, such as field definitions, allowed values and how each field is populated? We especially want it for the date fields (`first_seen_at`, `last_seen_at`, `posted_at`); for example, `posted_at` values sit at 22:00 or 23:00 UTC, so are they local dates in Central European time? and for `onet_data` (`code`, `family`, `occupation_name`): how are O*NET codes assigned, from the title, the description or both?

3. **Crawl timing.** How often do you crawl each company's careers page or hiring platform? Is `first_seen_at` the first crawl that found a posting, and `last_seen_at` the last crawl that still saw it? Does crawl frequency vary by company, platform or over time? We want to know how precise these dates are, in days.

4. **Daily snapshots and history.** Is each daily file a full copy of all postings, including ones that have closed, or only what changed since the previous day? Our snapshot holds listings first seen in the six months before it. Is that a fixed rolling window? Are older snapshots available if we need them to measure seasonality?

5. **Posting status.** What values can `status` take, what does each one mean, and how is it determined? In our sample it's either "closed" or empty, never "open". What's the most reliable way to tell whether a posting is still open, `status` or `last_seen_at`? And if it's `last_seen_at`, how recent should it be?

6. **Sources.** Do postings come only from company careers pages and hiring platforms, or also from job boards? Is `url` always the original source of the posting?

7. **Field origins.** Are `normalized_title`, `categories`, `tags` or `seniority` derived from O*NET, or produced by the same classifier? And what do `tags` contain: skills, keywords or something else?

8. **File order and sampling.** Is the job openings file sorted in any way, for example by company, date or crawl order? We've been working from the first 200,000 documents and want to know whether that's representative. If it isn't, is there a smaller or pre-sampled extract you'd recommend instead of streaming the full 19 GB file?

9. **Duplicates and reposts.** Does the same job ever appear as more than one posting, for example when it's reposted or listed in several locations? If so, is there a field that links the copies?

10. **Description text.** Is `description` the original posting text, or has it been cleaned, for example by stripping HTML or removing boilerplate? How is `language` detected?

11. **Final presentation.** Can we show real posting titles, company names and short description excerpts in the final presentation, or should we use synthetic examples only?

Any answers or documentation you can share would help a lot. Happy to set up a short call if that's easier.

Best,
Jacob Colombo
TripleTen externship team

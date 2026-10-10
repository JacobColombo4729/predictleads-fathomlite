# Ask the market

A question box on the **Signals** tab. You ask about who is hiring in plain words, by typing or by pressing **Talk**, and the answer comes only from our own listings and hiring signals. An optional second step researches one company on the web.

Built 2026-10-10: `backend/market.py` (the engine), `backend/market_llm.py` (the two OpenAI helpers), `/api/v1/market/*`, `frontend/src/components/signals/AskMarket.tsx`, and a per-listing table written by `code/hiring_signals.py market`.

## How a question becomes an answer

1. **Understand.** An OpenAI model (`gpt-4.1-mini`, strict JSON schema, temperature 0) turns the question into a *search*: a task (who is hiring, technologies, example openings, trend, pay, compare), filters (role family, place, title words, technologies, level, company pattern, first-time hiring, saved companies) and a window. It only fills in filters. It never writes an answer.
2. **Check.** Our code validates the search. Anything outside the allowed values is rejected. Any place, technology, company or title word the model added that the question itself does not contain is removed, and the screen says so. Forecasts and advice ("will", "next month", "should I", "best companies") are refused before the model is used.
3. **Run.** The engine runs the search on a table of 269,891 openings and on the company signals. Every figure is a count, a median or a percentile of real listings.
4. **Read back.** The screen shows "How we read your question" as chips. Removing a chip loosens that filter and re-runs the search with no AI call. A technology or company that does not exist in the data is reported with the closest real names, never replaced silently.

The preset example questions and the chips run without any AI at all, so they work with the AI switched off.

## What it can answer

Who is hiring (by role family, place, title, technology, level, hiring pattern, first-time hiring, or among your saved companies), which technologies appear in listings, example openings, new openings per week, pay stated in listings (listings that state a yearly pay only; hourly and other units are left out, not converted), and side-by-side numbers for named companies.

## What it will not do, and says so

Company size, funding, revenue, industry, headquarters, remote or on-site status, why a company is hiring, whether a company will hire you, forecasts, advice or rankings of employers, and anything outside the three role families (AI/ML, data, technical sales). The data is one crawler snapshot from spring to 2026-09-30; dates are when the crawler saw a listing, not when it was posted ([signal-semantics.md](signal-semantics.md)). Trend answers carry a caveat that coverage of the crawler can change.

## Optional: research this company

A separate button on each company, chosen by the person. It sends only the company's **name and website** to OpenAI and asks for a short synopsis from **web search** results (OpenAI's `web_search_preview` tool). Rules enforced in code:

- Only sentences a web citation covers are kept. A citation covers the passage before it (the model is told to cite every sentence, and often cites a whole paragraph at once); anything written after the last citation has no source and is dropped, and an answer with no citation at all shows nothing.
- The sources are listed (tracking tags removed), and the result is labelled "From the web through OpenAI. This is not PredictLeads data and can be wrong or out of date."
- If no sentence has a source, nothing is shown.
- Results are cached for the session.

## Talk

A **Talk** button records up to 30 seconds from the microphone (the browser asks permission each time it is first used), sends the recording to OpenAI (`gpt-4o-mini-transcribe`, set another model with `PREDICTLEADS_TRANSCRIBE_MODEL`) to be turned into text, puts the text in the box and asks it. The recording is not stored by this app. It uses the browser's microphone recorder, so it works in Firefox, Chrome and Edge on `http://127.0.0.1` or `https`.

## Switching it on, and limits

AI parts are **off by default**. Start the backend with `.\run_demo_backend.ps1 -Market` (or `--market` on Mac) and `OPENAI_API_KEY` in `.env`. Without it the box still shows the presets. Per demo session: 120 typed questions, 40 company researches, 60 voice questions (`MARKET_ASK_CAP`, `MARKET_RESEARCH_CAP`, `MARKET_VOICE_CAP` in `backend/api.py`). A typed question or a spoken one costs a fraction of a cent; a company research costs a few cents.

## Data

`python code/hiring_signals.py market --pool <targeted_pool.jsonl> --out data/signals/hiring_signals_YYYYMMDD/market_listings.csv.gz` writes one row per opening (reposts merged): company, title, role family, first and last seen, level, city, state, country, stated pay, technology tags, and whether it was first seen on a bulk-import day. `hiring_signals.py build` writes it too. The engine loads it at startup (about 3 seconds).

## What goes where

| Data sent to OpenAI | When | Not sent |
|---|---|---|
| The question text | typed or spoken, to turn it into a search | anything from your profile |
| A recording of up to 30 seconds | only when you press Talk | nothing is stored by this app |
| A company name and website | only when you press Research this company | anything else |

## Where it sits in the ML and GenAI inventory

The search understanding, the voice transcription and the web research are generative or speech AI; the engine that answers is plain data analysis (counts, medians, percentiles) with no learning in it. See [ml-inventory.md](ml-inventory.md).

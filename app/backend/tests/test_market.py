"""Ask the market: the query engine, the API and the two AI helpers. Synthetic data and a mock provider only (no network)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import api as api_module
from backend import market, market_llm
from backend.api import MARKET_EXAMPLES, Settings, create_app
from backend.tests.test_api import SNAPSHOT_ID, source_job
from backend.tests.test_search import PROFILE

CUTOFF = 272


def listing(i, company, name, family, title, city, state, tags, first, last=272, batch=False, level="mid_senior", pay=None, unit="year"):
    low, high = pay if pay else (None, None)
    return {"job_id": f"j{i}", "company_id": company, "company_name": name, "domain": f"{name.split()[0].lower()}.com", "title": title, "family": family,
            "first_day": first, "last_day": last, "closed": False, "batch_start": batch, "seniority": level, "city": city, "state": state,
            "country": "United States", "salary_low_usd": low, "salary_high_usd": high, "salary_unit": unit if pay else None, "tags": tags}


def build_engine() -> market.MarketEngine:
    rows, n = [], 0

    def add(*args, **kwargs):
        nonlocal n
        n += 1
        rows.append(listing(n, *args, **kwargs))

    for _ in range(4):  # Acme: data roles in Austin, recent, Snowflake + dbt
        add("co-a", "Acme, Inc.", "data", "Data Engineer", "Austin", "Texas", "Snowflake|DBT|SQL", 265, pay=(150000, 170000))
    add("co-a", "Acme, Inc.", "data", "Data Analyst", "Austin", "Texas", "SQL", 100, last=150)             # old, no longer observed
    for _ in range(3):  # Borealis: AI roles in Dallas, recent
        add("co-b", "Borealis Labs", "ai_ml", "AI Engineer", "Dallas", "Texas", "PyTorch|Python", 262, pay=(180000, 220000))
    add("co-b", "Borealis Labs", "ai_ml", "AI Engineer", "Dallas", "Texas", "PyTorch|Python", 120)         # earlier hiring
    for _ in range(2):  # Cobalt: data roles in New York, Snowflake only
        add("co-c", "Cobalt Data", "data", "Data Scientist", "New York", "New York", "Snowflake|Python", 268)
    for _ in range(6):  # Delta: a bulk import that must never count as new hiring
        add("co-d", "Delta Corp", "technical_sales", "Sales Engineer", "San Francisco", "California", "Salesforce", 270, batch=True)
    for _ in range(5):  # Echo: technical sales with stated pay, one hourly
        add("co-e", "Echo Systems", "technical_sales", "Solutions Engineer", "Austin", "Texas", "Salesforce|SQL", 266, pay=(120000, 140000))
    add("co-e", "Echo Systems", "technical_sales", "Solutions Engineer", "Austin", "Texas", "Salesforce", 266, pay=(40, 50), unit="hour")
    add("co-f", "Acme Robotics", "ai_ml", "ML Engineer", "Boston", "Massachusetts", "Python", 250)         # shares a first word with Acme
    listings = pd.DataFrame(rows)
    signals = pd.DataFrame([
        {"company_id": "co-a", "company_name": "Acme, Inc.", "domain": "acme.com", "pattern": "burst", "surge_z": 4.0, "new_30": 4},
        {"company_id": "co-b", "company_name": "Borealis Labs", "domain": "borealis.com", "pattern": "sustained", "surge_z": 0.5, "new_30": 3},
        {"company_id": "co-c", "company_name": "Cobalt Data", "domain": "cobalt.com", "pattern": "quiet", "surge_z": 0.1, "new_30": 2},
        {"company_id": "co-d", "company_name": "Delta Corp", "domain": "delta.com", "pattern": "quiet", "surge_z": 0.0, "new_30": 0},
        {"company_id": "co-e", "company_name": "Echo Systems", "domain": "echo.com", "pattern": "burst", "surge_z": 3.0, "new_30": 6},
        {"company_id": "co-f", "company_name": "Acme Robotics", "domain": "acmerobotics.com", "pattern": "cooling", "surge_z": -1.0, "new_30": 0},
    ])
    return market.MarketEngine(listings, signals, CUTOFF)


class QueryTests(unittest.TestCase):
    def test_limits_and_state_abbreviations_are_normalised(self) -> None:
        q = market.MarketQuery(states=["tx", "New York"], window_days=900, limit=5000)
        self.assertEqual(q.states, ["Texas", "New York"])
        self.assertEqual((q.window_days, q.limit), (90, 50))
        self.assertEqual(market.MarketQuery(window_days=1).window_days, 7)

    def test_unknown_fields_and_values_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            market.MarketQuery(task="forecast")
        with self.assertRaises(ValidationError):
            market.MarketQuery.model_validate({"task": "companies", "revenue_over": 5})
        with self.assertRaises(ValidationError):
            market.MarketQuery(patterns=["thriving"])

    def test_the_ai_schema_matches_the_query_fields(self) -> None:
        schema = market_llm.PARSE_SCHEMA
        self.assertEqual(set(schema["properties"]) - {"supported", "unsupported_reason"}, set(market.MarketQuery.model_fields))
        self.assertEqual(set(schema["required"]), set(schema["properties"]))  # strict mode needs every field required
        self.assertFalse(schema["additionalProperties"])


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = build_engine()

    def ask(self, **fields):
        return self.engine.run(market.MarketQuery(**fields))

    def names(self, result):
        return [r["name"] for r in result["rows"]]

    def test_technology_names_are_matched_case_insensitively_and_by_alias(self) -> None:
        self.assertEqual(self.engine.resolve_technologies(["dbt", "snowflake"])[0], ["DBT", "Snowflake"])
        self.assertEqual(self.engine.resolve_technologies(["SQL"])[0], ["SQL"])

    def test_unknown_technology_is_reported_not_replaced(self) -> None:
        found, missing = self.engine.resolve_technologies(["Pyth"])
        self.assertEqual(found, [])
        self.assertIn("Pyth", missing)
        self.assertIn("Python", missing["Pyth"])  # real names that contain it, offered as suggestions only
        result = self.ask(technologies=["Flurbo"])
        self.assertEqual(result["rows"], [])
        self.assertTrue(any("Flurbo" in n for n in result["notes"]))

    def test_snowflake_or_dbt_versus_both(self) -> None:
        either = self.ask(technologies=["Snowflake", "dbt"])
        self.assertEqual(set(self.names(either)), {"Acme, Inc.", "Cobalt Data"})
        both = self.ask(technologies=["Snowflake", "dbt"], technology_match="all")
        self.assertEqual(self.names(both), ["Acme, Inc."])

    def test_place_family_and_pattern_filters(self) -> None:
        self.assertEqual(self.names(self.ask(role_family=["data"], states=["TX"])), ["Acme, Inc."])
        self.assertEqual(set(self.names(self.ask(states=["Texas"], patterns=["burst"]))), {"Acme, Inc.", "Echo Systems"})
        self.assertEqual(self.names(self.ask(cities=["Dallas"])), ["Borealis Labs"])

    def test_bulk_import_openings_never_count_as_new(self) -> None:
        delta = self.ask(role_family=["technical_sales"], states=["California"])
        self.assertEqual([r for r in delta["rows"] if r["name"] == "Delta Corp" and r["new"] > 0], [])

    def test_new_and_active_counts(self) -> None:
        acme = next(r for r in self.ask(role_family=["data"], states=["Texas"])["rows"])
        self.assertEqual((acme["new"], acme["active"], acme["listings"]), (4, 4, 5))  # the old analyst is neither new nor still observed
        self.assertEqual(sum(acme["weekly"]), 5)         # the weekly history covers every opening first seen in the last 26 weeks
        self.assertEqual(sum(acme["weekly"][-4:]), 4)    # of which four arrived in the last four weeks

    def test_window_changes_what_is_new(self) -> None:
        week = self.ask(cities=["Dallas"], window_days=7)["rows"][0]
        wide = self.ask(cities=["Dallas"], window_days=30)["rows"][0]
        self.assertEqual(week["new"], 0)   # the three recent AI roles were first seen 10 days before the cutoff
        self.assertEqual(wide["new"], 3)

    def test_first_time_hiring(self) -> None:
        result = self.ask(role_family=["ai_ml"], first_time_hiring=True)
        self.assertNotIn("Borealis Labs", self.names(result))  # it hired AI roles earlier (day 120), so this is not a first
        self.assertEqual(self.names(result), ["Acme Robotics"])  # its only AI opening appeared in the last 30 days
        self.assertTrue(any("First time" in n for n in result["notes"]))
        data = self.ask(role_family=["data"], first_time_hiring=True)
        self.assertEqual(set(self.names(data)), {"Cobalt Data"})  # Acme's earliest data listing is old

    def test_compare_resolves_company_names_the_way_people_write_them(self) -> None:
        result = self.ask(task="compare", companies=["acme", "Borealis"])
        self.assertEqual(self.names(result), ["Acme, Inc.", "Borealis Labs"])
        missing = self.ask(task="compare", companies=["Borealis", "Nonesuch"])
        self.assertTrue(any("Nonesuch" in n for n in missing["notes"]))
        self.assertEqual(self.names(missing), ["Borealis Labs"])
        self.assertIn("top_technologies", result["rows"][0])

    def test_pay_uses_only_yearly_figures_and_says_what_it_left_out(self) -> None:
        result = self.ask(task="pay", role_family=["technical_sales"], cities=["Austin"])
        self.assertEqual(result["summary"]["with_yearly_pay"], 5)
        self.assertEqual(result["summary"]["median"], 130000)
        self.assertTrue(any("left out, not converted" in n for n in result["notes"]))
        few = self.ask(task="pay", cities=["Dallas"], role_family=["ai_ml"], seniority=["junior"])
        self.assertEqual(few["rows"], [])
        self.assertTrue(any("Too few" in n for n in few["notes"]))

    def test_technologies_task_lists_real_tags_and_leaves_out_the_one_asked_about(self) -> None:
        result = self.ask(task="technologies", role_family=["data"], technologies=["Snowflake"])
        tags = [r["technology"] for r in result["rows"]]
        self.assertNotIn("Snowflake", tags)
        self.assertIn("SQL", tags)

    def test_trend_has_a_value_for_every_week(self) -> None:
        result = self.ask(task="trend", states=["Texas"])
        self.assertEqual(len(result["rows"]), market.WEEKS)
        self.assertEqual(result["summary"]["last_4_weeks"], sum(r["new"] for r in result["rows"][-4:]))
        self.assertTrue(any("crawler" in n for n in result["notes"]))

    def test_listings_are_newest_first_with_pay(self) -> None:
        rows = self.ask(task="listings", role_family=["ai_ml"], limit=3)["rows"]
        self.assertEqual({r["title"] for r in rows}, {"AI Engineer"})
        self.assertTrue(all(r["pay"] for r in rows))

    def test_saved_companies_filter(self) -> None:
        mine = self.engine.run(market.MarketQuery(only_my_saved=True), ["co-b"])
        self.assertEqual(self.names(mine), ["Borealis Labs"])
        none = self.engine.run(market.MarketQuery(only_my_saved=True), [])
        self.assertTrue(any("no saved jobs" in n for n in none["notes"]))

    def test_the_interpretation_lists_what_was_applied(self) -> None:
        labels = {i["label"]: i["value"] for i in self.ask(technologies=["dbt"], states=["TX"], patterns=["burst"])["interpretation"]}
        self.assertEqual(labels["Technology"], "DBT")
        self.assertEqual(labels["Place"], "Texas")
        self.assertEqual(labels["Company pattern"], "burst")

    def test_forecasts_and_advice_are_refused_before_any_search(self) -> None:
        for q in ("Which companies will hire the most next month?", "who is likely to hire data engineers", "Should I apply to Acme?", "best companies to work for"):
            self.assertIsNotNone(market.refuse_in_advance(q), q)
        for q in ("Who is hiring data engineers in Texas?", "Which companies have been hiring since June?", "Compare Databricks and Snowflake"):
            self.assertIsNone(market.refuse_in_advance(q), q)

    def test_only_what_the_question_names_survives(self) -> None:
        fields = {"states": ["Texas"], "cities": ["Austin"], "technologies": ["Snowflake", "Kafka"], "companies": ["Databricks"], "title_words": ["engineer"], "countries": ["United States"]}
        cleaned, notes = market.ground_in_question(fields, "What do solutions engineers get paid in Austin, and who uses Snowflake?")
        self.assertEqual(cleaned["cities"], ["Austin"])
        self.assertEqual(cleaned["states"], [])               # the question never said Texas
        self.assertEqual(cleaned["technologies"], ["Snowflake"])
        self.assertEqual(cleaned["companies"], [])
        self.assertEqual(cleaned["title_words"], ["engineer"])  # "engineers" names it
        self.assertEqual(cleaned["countries"], [])
        self.assertTrue(notes and "Texas" in notes[0] and "Kafka" in notes[0])

    def test_a_state_can_be_named_by_its_abbreviation(self) -> None:
        cleaned, notes = market.ground_in_question({"states": ["Texas"]}, "data jobs in TX")
        self.assertEqual((cleaned["states"], notes), (["Texas"], []))
        cleaned, _ = market.ground_in_question({"states": ["Texas"]}, "text jobs")  # "tx" inside another word is not a mention
        self.assertEqual(cleaned["states"], [])

    def test_naming_all_three_role_families_means_no_filter(self) -> None:
        self.assertEqual(market.MarketQuery(role_family=["ai_ml", "data", "technical_sales"]).role_family, [])
        self.assertEqual(market.MarketQuery(role_family=["data"]).role_family, ["data"])

    def test_every_preset_example_validates_and_runs(self) -> None:
        for example in MARKET_EXAMPLES:
            query = market.MarketQuery.model_validate(example["query"])
            result = self.engine.run(query, ["co-a"])
            self.assertIn("interpretation", result, example["question"])


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        rows = [source_job("saved-1", "Data Engineer", "Build pipelines with Snowflake.", company_id="co-a"),
                source_job("other", "Forklift Operator", "Move pallets.", company_id="co-zzz")]
        (root / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600, secure_cookie=False,
                                 allowed_origins=("http://127.0.0.1:5173",))
        self.engine = build_engine()

    def start(self, ai=None):
        self.ai = ai
        self.client = TestClient(create_app(settings=self.settings, market_data=self.engine, market_ai=ai))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/demo-sessions")

    def good(self, **overrides):
        base = {"supported": True, "unsupported_reason": "", "task": "companies", "role_family": [], "states": [], "cities": [], "countries": [],
                "title_words": [], "technologies": [], "technology_match": "any", "seniority": [], "patterns": [], "window_days": 30,
                "first_time_hiring": False, "companies": [], "only_my_saved": False, "sort": "new", "limit": 20}
        return {**base, **overrides}

    def test_overview_reports_what_is_switched_on(self) -> None:
        self.start()
        off = self.client.get("/api/v1/market/overview").json()
        self.assertEqual((off["available"], off["ask_enabled"], off["research_enabled"]), (True, False, False))
        self.assertEqual(len(off["examples"]), len(MARKET_EXAMPLES))
        self.start(market_llm.MockMarketProvider())
        self.assertTrue(self.client.get("/api/v1/market/overview").json()["ask_enabled"])

    def test_run_needs_no_ai(self) -> None:
        self.start()
        body = self.client.post("/api/v1/market/run", json={"query": {"task": "companies", "technologies": ["dbt"]}}).json()
        self.assertEqual(body["state"], "ok")
        self.assertEqual([r["name"] for r in body["result"]["rows"]], ["Acme, Inc."])
        self.assertEqual(self.client.post("/api/v1/market/run", json={"query": {"task": "guess"}}).status_code, 422)

    def test_ask_turns_a_question_into_a_search_and_answers_from_the_data(self) -> None:
        ai = market_llm.MockMarketProvider(parse=lambda q: self.good(technologies=["Snowflake", "dbt"]))
        self.start(ai)
        body = self.client.post("/api/v1/market/ask", json={"question": "Who uses Snowflake or dbt?"}).json()
        self.assertEqual(body["state"], "ok")
        self.assertEqual(body["query"]["technologies"], ["Snowflake", "dbt"])
        self.assertEqual({r["name"] for r in body["result"]["rows"]}, {"Acme, Inc.", "Cobalt Data"})
        self.assertEqual(ai.parse_calls, 1)

    def test_filters_the_ai_added_are_dropped_and_the_person_is_told(self) -> None:
        ai = market_llm.MockMarketProvider(parse=lambda q: self.good(cities=["Austin"], states=["Texas"], role_family=["technical_sales"], task="pay"))
        self.start(ai)
        body = self.client.post("/api/v1/market/ask", json={"question": "What do solutions engineers get paid in Austin?"}).json()
        self.assertEqual(body["state"], "ok")
        self.assertEqual(body["query"]["states"], [])
        self.assertEqual(body["query"]["cities"], ["Austin"])
        self.assertTrue(any("Texas" in n and "did not name" in n for n in body["result"]["notes"]))

    def test_forecast_questions_are_refused_without_calling_the_ai(self) -> None:
        ai = market_llm.MockMarketProvider(parse=lambda q: self.good(patterns=["burst"]))
        self.start(ai)
        body = self.client.post("/api/v1/market/ask", json={"question": "Which companies will hire the most next month?"}).json()
        self.assertEqual(body["state"], "unsupported")
        self.assertIn("predict", body["reason"])
        self.assertEqual(ai.parse_calls, 0)

    def test_unsupported_questions_say_what_is_missing(self) -> None:
        ai = market_llm.MockMarketProvider(parse=lambda q: self.good(supported=False, unsupported_reason="We do not have company size."))
        self.start(ai)
        body = self.client.post("/api/v1/market/ask", json={"question": "Which small companies are hiring?"}).json()
        self.assertEqual((body["state"], body["reason"]), ("unsupported", "We do not have company size."))

    def test_an_answer_that_is_not_a_valid_search_is_not_run(self) -> None:
        ai = market_llm.MockMarketProvider(parse=lambda q: self.good(patterns=["thriving"]))
        self.start(ai)
        self.assertEqual(self.client.post("/api/v1/market/ask", json={"question": "who is thriving"}).json()["state"], "unsupported")

    def test_saved_companies_come_from_the_session(self) -> None:
        ai = market_llm.MockMarketProvider(parse=lambda q: self.good(only_my_saved=True))
        self.start(ai)
        empty = self.client.post("/api/v1/market/ask", json={"question": "my saved companies"}).json()
        self.assertTrue(any("no saved jobs" in n for n in empty["result"]["notes"]))
        self.client.post("/api/v1/saved-jobs", json={"job_id": "saved-1", "snapshot_id": SNAPSHOT_ID})
        mine = self.client.post("/api/v1/market/ask", json={"question": "my saved companies"}).json()
        self.assertEqual([r["name"] for r in mine["result"]["rows"]], ["Acme, Inc."])

    def test_ai_off_failure_and_cap(self) -> None:
        self.start()
        self.assertEqual(self.client.post("/api/v1/market/ask", json={"question": "anything"}).json()["state"], "unavailable")
        failing = market_llm.MockMarketProvider()  # parse raises MarketAIFailure
        self.start(failing)
        self.assertEqual(self.client.post("/api/v1/market/ask", json={"question": "anything"}).status_code, 502)
        working = market_llm.MockMarketProvider(parse=lambda q: self.good())
        self.start(working)
        with mock.patch.object(api_module, "MARKET_ASK_CAP", 2):
            self.assertEqual(self.client.post("/api/v1/market/ask", json={"question": "one"}).status_code, 200)
            self.assertEqual(self.client.post("/api/v1/market/ask", json={"question": "two"}).status_code, 200)
            self.assertEqual(self.client.post("/api/v1/market/ask", json={"question": "three"}).status_code, 429)

    def test_question_length_is_limited(self) -> None:
        self.start(market_llm.MockMarketProvider(parse=lambda q: self.good()))
        self.assertEqual(self.client.post("/api/v1/market/ask", json={"question": "x" * 301}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/market/ask", json={"question": "hi"}).status_code, 422)

    def test_research_returns_a_labelled_synopsis_with_sources_and_is_cached(self) -> None:
        seen = {}
        def research(name, domain):
            seen["args"] = (name, domain)
            return {"text": "Acme builds data tools.", "sources": [{"url": "https://acme.com/about", "title": "About Acme"}]}
        ai = market_llm.MockMarketProvider(research=research)
        self.start(ai)
        first = self.client.post("/api/v1/market/company-research", json={"company_id": "co-a"}).json()
        self.assertEqual(first["state"], "ok")
        self.assertEqual(first["sources"][0]["url"], "https://acme.com/about")
        self.assertIn("not PredictLeads data", first["label"])
        self.assertEqual(seen["args"], ("Acme, Inc.", "acme.com"))  # only the name and website go out
        again = self.client.post("/api/v1/market/company-research", json={"company_id": "co-a"}).json()
        self.assertEqual(again["cache"], "hit")
        self.assertEqual(ai.research_calls, 1)

    def test_research_without_sources_shows_nothing(self) -> None:
        self.start(market_llm.MockMarketProvider(research=lambda n, d: {"text": "Probably a software company.", "sources": []}))
        body = self.client.post("/api/v1/market/company-research", json={"company_id": "co-a"}).json()
        self.assertEqual(body["state"], "no_sources")
        self.assertNotIn("text", body)

    def test_research_unknown_company_disabled_and_failure(self) -> None:
        self.start(market_llm.MockMarketProvider(research=lambda n, d: {"text": "x", "sources": []}))
        self.assertEqual(self.client.post("/api/v1/market/company-research", json={"company_id": "nope"}).status_code, 404)
        self.start()
        self.assertEqual(self.client.post("/api/v1/market/company-research", json={"company_id": "co-a"}).json()["state"], "unavailable")
        self.start(market_llm.MockMarketProvider())  # research raises MarketAIFailure
        self.assertEqual(self.client.post("/api/v1/market/company-research", json={"company_id": "co-a"}).status_code, 502)

    def test_requires_a_session(self) -> None:
        self.start()
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/api/v1/market/overview").status_code, 401)

    # ---- voice ------------------------------------------------------------------------------------------------------------

    def speak(self, data=b"fake-audio", content_type="audio/webm;codecs=opus"):
        return self.client.post("/api/v1/market/transcribe", content=data, headers={"Content-Type": content_type})

    def test_a_spoken_question_becomes_text(self) -> None:
        seen = {}
        def transcribe(audio, content_type):
            seen["args"] = (audio, content_type)
            return "Which companies are hiring people who use Snowflake or dbt?"
        self.start(market_llm.MockMarketProvider(transcribe=transcribe))
        self.assertTrue(self.client.get("/api/v1/market/overview").json()["voice_enabled"])
        body = self.speak().json()
        self.assertEqual(body, {"state": "ok", "text": "Which companies are hiring people who use Snowflake or dbt?"})
        self.assertEqual(seen["args"][0], b"fake-audio")
        self.assertTrue(seen["args"][1].startswith("audio/webm"))

    def test_voice_is_unavailable_when_the_ai_is_off_and_validates_the_upload(self) -> None:
        self.start()
        self.assertFalse(self.client.get("/api/v1/market/overview").json()["voice_enabled"])
        self.assertEqual(self.speak().json()["state"], "unavailable")
        self.start(market_llm.MockMarketProvider(transcribe=lambda a, c: "hello"))
        self.assertEqual(self.speak(content_type="application/json").status_code, 415)
        self.assertEqual(self.speak(data=b"").status_code, 422)
        with mock.patch.object(api_module, "VOICE_MAX_BYTES", 10):
            self.assertEqual(self.speak(data=b"x" * 11).status_code, 413)

    def test_voice_failure_empty_transcript_and_cap(self) -> None:
        self.start(market_llm.MockMarketProvider())  # transcribe raises MarketAIFailure
        self.assertEqual(self.speak().status_code, 502)
        self.start(market_llm.MockMarketProvider(transcribe=lambda a, c: "   "))
        self.assertEqual(self.speak().json()["state"], "empty")
        self.start(market_llm.MockMarketProvider(transcribe=lambda a, c: "a question"))
        with mock.patch.object(api_module, "MARKET_VOICE_CAP", 1):
            self.assertEqual(self.speak().status_code, 200)
            self.assertEqual(self.speak().status_code, 429)

    def test_a_long_transcript_is_cut_to_the_question_limit(self) -> None:
        self.start(market_llm.MockMarketProvider(transcribe=lambda a, c: "word " * 200))
        self.assertLessEqual(len(self.speak().json()["text"]), 300)


class AiHelperTests(unittest.TestCase):
    def test_clean_synopsis_removes_citation_links_and_markdown(self) -> None:
        text = "Acme **builds** data tools ([acme.com](https://acme.com/about)). See https://acme.com for more [Acme](https://acme.com)."
        self.assertEqual(market_llm.clean_synopsis(text), "Acme builds data tools. See for more Acme.")
        self.assertEqual(market_llm.clean_synopsis("It serves banks ."), "It serves banks.")

    def test_parse_reads_the_structured_answer(self) -> None:
        provider = market_llm.OpenAIMarketProvider("sk-test")
        reply = {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({"supported": True})}]}]}
        with mock.patch.object(market_llm, "_post", return_value=reply) as post:
            self.assertEqual(provider.parse("hi"), {"supported": True})
        body = post.call_args.args[1]
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["temperature"], 0)

    def test_parse_rejects_an_unusable_answer(self) -> None:
        provider = market_llm.OpenAIMarketProvider("sk-test")
        with mock.patch.object(market_llm, "_post", return_value={"output": []}):
            with self.assertRaises(market_llm.MarketAIFailure):
                provider.parse("hi")

    def test_research_collects_sources_from_citations_and_uses_web_search(self) -> None:
        provider = market_llm.OpenAIMarketProvider("sk-test")
        text = "Acme builds tools. ([acme.com](https://acme.com/about?utm_source=openai))"
        a = text.index("([")
        reply = {"output": [{"type": "web_search_call"}, {"type": "message", "content": [{"type": "output_text", "text": text, "annotations": [
            {"type": "url_citation", "url": "https://acme.com/about?utm_source=openai", "title": "About", "start_index": a, "end_index": len(text)}]}]}]}
        with mock.patch.object(market_llm, "_post", return_value=reply) as post:
            found = provider.research("Acme", "acme.com")
        self.assertEqual(found, {"text": "Acme builds tools.", "sources": [{"url": "https://acme.com/about", "title": "About"}]})  # tracking tag removed
        tools = post.call_args.args[1]["tools"]
        self.assertEqual(tools[0]["type"], "web_search_preview")
        self.assertIn("Acme (website: acme.com)", json.dumps(post.call_args.args[1]["input"]))

    def cite(self, text, *pairs):
        notes = []
        for sub, url in pairs:
            start = text.index(sub)
            notes.append({"type": "url_citation", "url": url, "title": "t", "start_index": start, "end_index": start + len(sub)})
        return notes

    def test_a_citation_covers_the_passage_before_it_and_text_after_the_last_one_is_dropped(self) -> None:
        text = "Acme builds data tools. It serves banks and insurers. ([acme.com](https://acme.com/about)) It is probably the largest in its field."
        kept, urls = market_llm.keep_cited_sentences(text, self.cite(text, ("([acme.com](https://acme.com/about))", "https://acme.com/about")))
        self.assertEqual(kept, "Acme builds data tools. It serves banks and insurers.")
        self.assertEqual(urls, ["https://acme.com/about"])  # the closing guess has no source after it, so it is gone

    def test_each_sentence_can_carry_its_own_source(self) -> None:
        text = "Acme builds data tools. ([acme.com](https://acme.com/about)) It serves banks. ([acme.com](https://acme.com/customers))"
        kept, urls = market_llm.keep_cited_sentences(text, self.cite(
            text, ("([acme.com](https://acme.com/about))", "https://acme.com/about"), ("([acme.com](https://acme.com/customers))", "https://acme.com/customers")))
        self.assertEqual(kept, "Acme builds data tools. It serves banks.")
        self.assertEqual(urls, ["https://acme.com/about", "https://acme.com/customers"])

    def test_a_synopsis_with_no_citations_is_shown_as_nothing(self) -> None:
        kept, urls = market_llm.keep_cited_sentences("Acme is probably a software company.", [])
        self.assertEqual((kept, urls), ("", []))
        provider = market_llm.OpenAIMarketProvider("sk-test")
        reply = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Acme is probably a software company.", "annotations": []}]}]}
        with mock.patch.object(market_llm, "_post", return_value=reply):
            self.assertEqual(provider.research("Acme", None), {"text": "", "sources": []})

    def test_tracking_tags_and_fragments_are_removed_from_links(self) -> None:
        self.assertEqual(market_llm.clean_url("https://x.com/p?utm_source=openai&id=3#frag"), "https://x.com/p?id=3")

    def test_transcribe_sends_the_audio_as_a_form_upload_and_returns_the_text(self) -> None:
        provider = market_llm.OpenAIMarketProvider("sk-test")
        captured = {}

        class Reply:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps({"text": "  Who is hiring\n data engineers? "}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            captured["url"], captured["body"], captured["headers"] = request.full_url, request.data, dict(request.header_items())
            return Reply()

        with mock.patch.object(market_llm.urllib.request, "urlopen", fake_urlopen):
            text = provider.transcribe(b"RIFFfake", "audio/webm;codecs=opus")
        self.assertEqual(text, "Who is hiring data engineers?")
        self.assertTrue(captured["url"].endswith("/audio/transcriptions"))
        self.assertIn(b'name="model"', captured["body"])
        self.assertIn(b"RIFFfake", captured["body"])
        self.assertIn(b'filename="question.webm"', captured["body"])
        self.assertTrue(captured["headers"]["Content-type"].startswith("multipart/form-data; boundary="))
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")

    def test_transcribe_failures_become_a_clean_error(self) -> None:
        provider = market_llm.OpenAIMarketProvider("sk-test")
        def boom(request, timeout=0):
            raise market_llm.urllib.error.URLError("down")
        with mock.patch.object(market_llm.urllib.request, "urlopen", boom):
            with self.assertRaises(market_llm.MarketAIFailure):
                provider.transcribe(b"x", "audio/webm")

    def test_provider_needs_a_key(self) -> None:
        with self.assertRaises(market_llm.MarketAIUnavailable):
            market_llm.OpenAIMarketProvider("  ")
        with self.assertRaises(market_llm.MarketAIUnavailable):
            market_llm.OpenAIMarketProvider.from_environment({})


if __name__ == "__main__":
    unittest.main()

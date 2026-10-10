"""The hiring-signals endpoints, against a tiny hand-made signals folder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.api import Settings, create_app
from backend.signals import SignalStore
from backend.tests.test_api import source_job

COLS = ["company_code", "seen", "new_30", "new_prev_90", "active", "active_30_ago", "gone_30", "batch_recent", "history_days", "baseline",
        "net_30", "ratio", "surge_z", "enough_history", "surging", "cooling", "sustained", "pattern", "new_30_ai_ml", "active_ai_ml",
        "new_30_data", "active_data", "new_30_technical_sales", "active_technical_sales", "company_id", "company_name", "domain"]


def company(code, cid, name, pattern, new_30, ai=0, sales=0):
    return [code, 50, new_30, 10, 12, 8, 2, False, 120, 5.0, 4, 2.0, 3.5, True, pattern == "burst", pattern == "cooling", pattern == "sustained",
            pattern, ai, 3, 0, 0, sales, 1, cid, name, f"{name.lower()}.com"]


class SignalsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        sig = root / "sig"
        sig.mkdir()
        pd.DataFrame([
            company(1, "c-burst", "Alpha", "burst", 30, ai=25),
            company(2, "c-steady", "Beta", "sustained", 12, sales=9),
            company(3, "c-quiet", "Gamma", "quiet", 1),
            company(4, "c-cool", "Delta", "cooling", 0),
        ], columns=COLS).to_csv(sig / "company_signals.csv", index=False)
        pd.DataFrame({"company_code": [1, 1, 2], "week_end": [97, 104, 97], "new": [4, 9, 3], "gone": [0, 0, 0], "active": [1, 2, 3],
                      "company_id": ["c-burst", "c-burst", "c-steady"], "company_name": ["Alpha", "Alpha", "Beta"], "domain": ["a", "a", "b"]}
                     ).to_csv(sig / "company_weekly.csv", index=False)
        pd.DataFrame([{"company_code": 1, "company_id": "c-burst", "company_name": "Alpha", "domain": "a", "family": "ai_ml", "signal": "AI/ML buildout",
                       "meaning": "hiring ML roles", "new_30": 25, "baseline_monthly": 3.0, "surge_z": 9.0, "active": 3, "top_tech": "Python|SQL",
                       "evidence": json.dumps([{"job_id": "j1", "title": "ML Engineer", "first_seen": "2026-09-30"}])}]).to_csv(sig / "intent_signals.csv", index=False)
        pd.DataFrame([{"technology": "Python", "company_code": 1, "company_id": "c-burst", "company_name": "Alpha", "domain": "a", "new_30": 20, "still_observed": 3}]
                     ).to_csv(sig / "technology_leads.csv", index=False)
        (sig / "summary.json").write_text(json.dumps({"snapshot": "2026-09-30"}), encoding="utf-8")
        (sig / "backtest.json").write_text(json.dumps({"by_pattern": [{"pattern": "burst", "next_as_share_of_this_month": 0.63}]}), encoding="utf-8")
        (root / "jobs.jsonl").write_text(json.dumps(source_job("j", "Engineer", "Build things.", company_id="c1")) + "\n", encoding="utf-8")
        settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        self.client = TestClient(create_app(settings=settings, signal_store=SignalStore.load(sig)))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/demo-sessions")

    def test_overview_counts_and_carry_over(self) -> None:
        body = self.client.get("/api/v1/signals/overview").json()
        self.assertEqual(body["pattern_counts"], {"burst": 1, "sustained": 1, "cooling": 1, "quiet": 1})
        self.assertEqual(body["carry_over"][0]["next_as_share_of_this_month"], 0.63)
        self.assertEqual(body["week_ends"][0], "2026-04-08")

    def test_leaderboard_filters_and_sorts(self) -> None:
        everyone = self.client.get("/api/v1/signals/companies").json()
        self.assertEqual([c["name"] for c in everyone["companies"]], ["Alpha", "Beta", "Gamma", "Delta"])
        self.assertEqual(self.client.get("/api/v1/signals/companies?pattern=sustained").json()["total"], 1)
        sales = self.client.get("/api/v1/signals/companies?family=technical_sales").json()
        self.assertEqual([c["name"] for c in sales["companies"]], ["Beta"])
        self.assertEqual(self.client.get("/api/v1/signals/companies?q=alp").json()["total"], 1)

    def test_weekly_series_is_dense_and_ordered(self) -> None:
        alpha = self.client.get("/api/v1/signals/companies?pattern=burst").json()["companies"][0]
        self.assertEqual(alpha["weekly"], [4, 9])

    def test_company_detail_has_evidence_and_technologies(self) -> None:
        body = self.client.get("/api/v1/signals/companies/c-burst").json()
        self.assertEqual(body["evidence"][0]["title"], "ML Engineer")
        self.assertEqual(body["technologies"][0]["technology"], "Python")
        self.assertEqual(self.client.get("/api/v1/signals/companies/nope").status_code, 404)

    def test_patterns_map_leaves_out_quiet_companies(self) -> None:
        self.assertEqual(self.client.get("/api/v1/signals/patterns").json()["patterns"], {"c-burst": "burst", "c-steady": "sustained", "c-cool": "cooling"})

    def test_requires_a_session(self) -> None:
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/api/v1/signals/overview").status_code, 401)

    def test_unavailable_when_not_built(self) -> None:
        self.client.app.state.signals = None
        self.assertEqual(self.client.get("/api/v1/signals/overview").status_code, 503)


if __name__ == "__main__":
    unittest.main()

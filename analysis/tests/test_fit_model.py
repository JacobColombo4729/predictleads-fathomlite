"""Skills-based matching and the trained layer (src/fit_model.py) on made-up O*NET ratings and people. No real data."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import fit_model as fm
from src.fit_eval import ROLES

CODES = {"it support": "15-1232.00", "networks and systems": "15-1244.00", "security": "15-1212.00",
         "database": "15-1242.00", "software development": "15-1252.00", "quality assurance": "15-1253.00",
         "data science": "15-2051.00", "it project management": "15-1299.09"}
SKILLS = ["Programming", "Troubleshooting", "Negotiation", "Instructing", "Systems Analysis", "Service Orientation"]


def made_up_matrix():
    """Occupation x skill importance (1-5): tech roles lean on programming and systems; teachers on instructing; sales on
    negotiation and service."""
    r = np.random.default_rng(0)
    rows = {code: [4.5 if role in ("software development", "data science", "quality assurance") else 2.5, 4.0, 2.0, 2.0,
                   4.0 if role != "it support" else 3.0, 4.0 if role == "it support" else 2.5]
            for role, code in CODES.items()}
    rows.update({"25-2021.00": [1.5, 2.0, 2.5, 4.8, 2.0, 3.5], "25-1011.00": [2.0, 2.0, 2.5, 4.5, 2.5, 3.0],
                 "41-2031.00": [1.2, 2.0, 4.6, 2.0, 1.5, 4.5], "41-3091.00": [1.5, 2.5, 4.4, 2.5, 2.0, 4.0]})
    m = pd.DataFrame(rows, index=SKILLS).T
    return m + r.normal(0, 0.05, m.shape)


def made_up_people(n=600, seed=1):
    """People whose role depends on their background and training field, with a little noise."""
    r = np.random.default_rng(seed)
    area = r.choice(["education", "sales and retail", "unknown"], n)
    field = r.choice(["data science and analytics", "networking", "software and web development"], n)
    target = np.where(field == "data science and analytics", "data science",
                      np.where(field == "networking", "networks and systems", "software development"))
    noise = r.random(n) < 0.2
    target = np.where(noise, r.choice(ROLES, n), target)
    return pd.DataFrame({"origin": "other-field job", "from_level": None, "background_area": area, "retrain_field": field,
                         "from_bucket": None, "retrained": True, "study_undated": False,
                         "years_before": r.uniform(0, 10, n).round(1), "n_jobs_before": r.integers(0, 5, n),
                         "study_level_before": r.integers(0, 4, n), "target_bucket": target},
                        index=[f"p{i}" for i in range(n)])


class SkillsScores(unittest.TestCase):
    def setUp(self):
        self.m = made_up_matrix()
        self.roles = fm.role_profiles(self.m)

    def test_role_profiles_use_the_roles_own_occupations(self):
        self.assertEqual(list(self.roles.index), ROLES)
        self.assertAlmostEqual(self.roles.loc["it support", "Service Orientation"], self.m.loc["15-1232.00", "Service Orientation"])

    def test_backgrounds_from_area_or_current_role(self):
        people = pd.DataFrame({"from_bucket": [None, "database", None], "background_area": ["education", "tech: database", "other"]},
                              index=["a", "b", "c"])
        b = fm.background_profiles(people, self.m, self.roles)
        self.assertAlmostEqual(b.loc["a", "Instructing"], self.m.loc[["25-2021.00", "25-1011.00"], "Instructing"].mean())
        pd.testing.assert_series_equal(b.loc["b"], self.roles.loc["database"], check_names=False)
        self.assertTrue(b.loc["c"].isna().all())

    def test_scores_are_bounded_and_zero_without_a_background(self):
        people = pd.DataFrame({"from_bucket": [None, None], "background_area": ["sales and retail", "unknown"]}, index=["a", "b"])
        b = fm.background_profiles(people, self.m, self.roles)
        co, cos = fm.carry_over_scores(b, self.roles), fm.profile_cosine_scores(b, self.roles, self.m)
        self.assertEqual(co.shape, (2, len(ROLES)))
        self.assertTrue(((co >= 0) & (co <= 1)).all() and (np.abs(cos) <= 1 + 1e-9).all())
        self.assertTrue((co[1] == 0).all() and (cos[1] == 0).all())

    def test_a_tech_background_matches_its_neighbours_best(self):
        people = pd.DataFrame({"from_bucket": ["software development"], "background_area": ["tech: software development"]}, index=["a"])
        cos = fm.profile_cosine_scores(fm.background_profiles(people, self.m, self.roles), self.roles, self.m)[0]
        self.assertGreater(cos[ROLES.index("data science")], cos[ROLES.index("it support")])


class TrainedLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        people = made_up_people()
        r = np.random.default_rng(3)
        scores = {"tf-idf": r.random((len(people), len(ROLES))), "skills carry-over": r.random((len(people), len(ROLES)))}
        cls.X = fm.features(people, scores)
        cls.y = np.array([ROLES.index(t) for t in people.target_bucket])
        cls.fold = pd.Series(np.arange(len(people)) % 5, index=people.index)

    def test_features_have_no_labels_and_both_score_forms(self):
        self.assertNotIn("target_bucket", self.X)
        self.assertIn("tfidf: data science", self.X)
        self.assertIn("skills_carryover vs own mean: data science", self.X)
        self.assertFalse(self.X[fm.CATEGORICAL].isna().any().any())

    def test_compare_finds_the_signal_and_out_of_fold_covers_everyone(self):
        table, searches = fm.compare(self.X, self.y, self.fold)
        self.assertEqual(set(table.index), set(fm.candidates()))
        self.assertGreater(table["top-3 hit rate (mean)"].max(), 0.75)  # the field decides 80% of roles
        best = searches[table.index[0]].best_estimator_
        oof = fm.out_of_fold(best, self.X, self.y, self.fold)
        self.assertEqual(oof.shape, (len(self.X), len(ROLES)))
        np.testing.assert_allclose(oof.sum(axis=1), 1.0, atol=1e-6)


class WhyGrowthAndIdeas(unittest.TestCase):
    def test_groups_cover_every_kind_of_input(self):
        self.assertEqual(fm.group_of("origin_other-field job"), "prior career")
        self.assertEqual(fm.group_of("background_area_tech: database"), "prior career")
        self.assertEqual(fm.group_of("retrain_field_networking"), "training")
        self.assertEqual(fm.group_of("retrained"), "training")
        self.assertEqual(fm.group_of("missingindicator_years_before"), "prior career")
        self.assertEqual(fm.group_of("tfidf vs own mean: security"), "text similarity")
        self.assertEqual(fm.group_of("skills_profile_cosine: database"), "skills that carry over")

    def test_linear_contributions_add_up_to_the_logit_gap(self):
        people = made_up_people(300)
        X = fm.features(people, {"tf-idf": np.random.default_rng(0).random((300, len(ROLES)))})
        y = np.array([ROLES.index(t) for t in people.target_bucket])
        pipe, _ = fm.candidates()["logistic regression"]
        pipe.fit(X, y)
        contrib, names = fm.linear_contributions(pipe, X, X.iloc[:5])
        self.assertEqual(contrib.shape[:2], (5, len(ROLES)))
        logits = pipe.decision_function(X)
        model_classes = list(pipe.named_steps["model"].classes_)
        for c in model_classes:  # sum of contributions = the person's logit minus the average logit
            np.testing.assert_allclose(contrib[:, c, :].sum(axis=1), logits[:5, model_classes.index(c)] - logits[:, model_classes.index(c)].mean(), atol=1e-8)
        g, groups = fm.grouped(contrib, names)
        np.testing.assert_allclose(g.sum(axis=2), contrib.sum(axis=2), atol=1e-10)
        self.assertEqual(groups[-1], "other")

    def test_growth_adjust_is_visible_and_optional(self):
        p = np.array([[0.5, 0.45, 0.05, 0, 0, 0, 0, 0]])
        growth = {"networks and systems": 0.30, "it support": -0.05}
        np.testing.assert_allclose(fm.growth_adjust(p, growth, weight=0), p)
        adj = fm.growth_adjust(p, growth, weight=1)
        self.assertGreater(adj[0, 1], adj[0, 0])  # networks overtakes it support
        self.assertEqual(fm.top_k_changed(p, adj, k=1), 1.0)
        self.assertTrue((fm.growth_adjust(p, {"it support": -5}, 1) >= 0).all())

    def test_skill_demand_and_skills_to_build(self):
        roles = pd.Series({"a": "data science", "b": "data science", "c": "security"})
        skills = pd.DataFrame({"id": ["a", "a", "b", "c", "c"], "name": ["Python", "SQL", "Python", "Python", "Splunk"],
                               "kind": ["tool"] * 5, "section": ["required"] * 5})
        d = fm.role_skill_demand(skills, roles)
        self.assertAlmostEqual(d.set_index(["role", "name"]).loc[("data science", "SQL"), "share"], 0.5)
        b = fm.skills_to_build(d, {"education": {"Python"}})
        self.assertTrue(b.set_index(["role", "name"]).loc[("data science", "Python"), "covered"])
        self.assertFalse(b.set_index(["role", "name"]).loc[("security", "Splunk"), "covered"])

    def test_background_pay_weights_by_listings(self):
        occ = pd.DataFrame({"O*NET-SOC Code": ["25-2021.00", "25-1011.00", "41-2031.00"],
                            "Title": ["Elementary School Teachers", "Professors", "Retail Salespersons"]})
        pay = pd.DataFrame({"median hourly pay (USD)": [30.0, 50.0, 16.0], "listings": [300, 100, 50]},
                           index=["Elementary School Teachers", "Professors", "Retail Salespersons"])
        t = fm.background_pay(pay, occ, {"education": ["25-"], "sales and retail": ["41-"], "military": ["55-"]})
        self.assertAlmostEqual(t.loc["education", "typical hourly pay"], 35.0)
        self.assertTrue(np.isnan(t.loc["military", "typical hourly pay"]))


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from evaluate_requirement_rerank import calculate_metrics


class RequirementRerankMetricTests(unittest.TestCase):
    def test_method_specific_p5_survives_incomplete_shared_pool(self):
        profile_id = "synthetic_1"
        pool = [{"profile_id": profile_id, "job_id": "job_{}".format(index)} for index in range(6)]
        rankings = []
        rankings.extend({"profile_id": profile_id, "job_id": "job_{}".format(index),
                         "method": "tfidf_cosine", "rank": index + 1} for index in range(5))
        rankings.extend({"profile_id": profile_id, "job_id": "job_{}".format(index + 1),
                         "method": "embedding_section_aware_v1", "rank": index + 1} for index in range(5))
        assessments = [{"profile_id": profile_id, "job_id": "job_{}".format(index),
                        "assessment": {"requirements": [], "constraints": []}}
                       for index in range(6)]
        judgments = [{"profile_id": profile_id, "job_id": "job_{}".format(index),
                      "status": "judged" if index < 5 else "unknown",
                      "grade": 2 if index < 5 else None}
                     for index in range(6)]
        result = calculate_metrics(pool, rankings, assessments, judgments)
        row = result["per_profile"][0]["methods"]
        self.assertEqual(row["tfidf_cosine"]["p_at_5_status"], "reported")
        self.assertEqual(row["tfidf_cosine"]["plausible_or_strong_precision_at_5"], 1.0)
        self.assertEqual(row["embedding_section_aware_v1"]["p_at_5_status"], "incomplete_or_fewer_than_five")
        self.assertEqual(row["tfidf_cosine"]["ndcg_status"], "incomplete_common_pool_judgments")
        self.assertEqual(result["paired_ndcg_denominator"], 0)

    def test_independent_complete_p5_is_reported(self):
        profile_id = "synthetic_1"
        pool = [{"profile_id": profile_id, "job_id": "job_{}".format(index)} for index in range(6)]
        rankings = [{"profile_id": profile_id, "job_id": "job_{}".format(index),
                     "method": "tfidf_cosine", "rank": index + 1} for index in range(5)]
        assessments = [{"profile_id": profile_id, "job_id": "job_{}".format(index),
                        "assessment": {"requirements": [], "constraints": []}}
                       for index in range(6)]
        judgments = [{"profile_id": profile_id, "job_id": "job_{}".format(index),
                      "status": "judged" if index < 5 else "unknown",
                      "grade": 3 if index < 5 else None} for index in range(6)]
        result = calculate_metrics(pool, rankings, assessments, judgments)
        metrics = result["per_profile"][0]["methods"]["tfidf_cosine"]
        self.assertEqual(metrics["p_at_5_status"], "reported")
        self.assertEqual(metrics["plausible_or_strong_precision_at_5"], 1.0)
        self.assertEqual(metrics["strong_only_precision_at_5"], 1.0)
        self.assertEqual(metrics["ndcg_status"], "incomplete_common_pool_judgments")
        self.assertEqual(result["paired_p5_denominator"], 0)


if __name__ == "__main__":
    unittest.main()

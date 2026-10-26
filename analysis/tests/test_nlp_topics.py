"""src/nlp_topics.py: topics from TF-IDF + NMF, and clusters from embeddings. Made-up text, no real data.

The corpus below is two obvious themes written in twelve ways, so what a topic model should find is known in
advance: at k=2 the two themes must come apart, and the words describing each topic must be that theme's words.
That is the only claim about topic modelling a test can make honestly - whether ten topics over real requirement
text are *interesting* is answered in the notebook, by reading them.

The clustering tests use made-up vectors rather than a sentence model, for the same reason `test_nlp_sections.py`
uses a stub encoder: the mechanics (which lines are nearest the centre, what happens to an empty cluster) are
testable, and BGE's judgement is not.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from src import nlp_topics as nt

DASHBOARDS = [
    "Build dashboards and reports for the sales team",
    "Building reports and dashboards in Tableau",
    "You will build reporting dashboards for stakeholders",
    "Dashboard building and report writing for the business",
    "Builds dashboards, reports and visualisations",
    "Experience building reports and dashboards",
]
PIPELINES = [
    "Write SQL queries and maintain data pipelines",
    "Writing SQL and maintaining the data pipeline",
    "You will write SQL queries against the data warehouse",
    "SQL query writing and pipeline maintenance",
    "Maintains data pipelines and writes SQL",
    "Experience writing SQL and building data pipelines",
]


class Lemmas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = nt.lemmatizer()

    def test_inflections_collapse_to_one_feature(self):
        """Where the tagger reads the part of speech right, "dashboards" and "dashboard", "Built" and "build" are
        one feature each. That is the point of lemmatizing before TF-IDF."""
        self.assertIn("dashboard", nt.lemma_tokens(self.nlp("You will own several dashboards")))
        self.assertIn("build", nt.lemma_tokens(self.nlp("Built a dashboard")))
        self.assertIn("build", nt.lemma_tokens(self.nlp("You will build dashboards")))

    def test_a_bullet_opening_with_a_gerund_is_read_as_a_noun(self):
        """A known cost of tagging fragments with the small model, recorded here rather than worked around:
        "Building dashboards" is not a sentence, so "Building" is tagged NOUN and lemmatizes to "building" rather
        than "build". It changes which words describe a topic, not whether the theme is found - the Topics tests
        below separate two themes written in exactly this mixture of forms."""
        self.assertIn("building", nt.lemma_tokens(self.nlp("Building dashboards")))

    def test_stopwords_punctuation_and_numbers_are_dropped(self):
        tokens = nt.lemma_tokens(self.nlp("You will have 5+ years of experience with SQL, and more."))
        self.assertNotIn("you", tokens)
        self.assertNotIn("of", tokens)
        self.assertNotIn("5", tokens)
        self.assertNotIn(",", tokens)
        self.assertEqual(sorted(tokens), ["experience", "sql", "year"])

    def test_one_token_list_per_text_in_order(self):
        got = nt.lemmatize(["Build dashboards", "", "Write SQL"], self.nlp)
        self.assertEqual(len(got), 3)
        self.assertEqual(got[1], [])

    def test_pairs_of_lemmas_become_features_too(self):
        got = nt._features(["write", "sql", "query"], ngram_max=2)
        self.assertEqual(got, ["write", "sql", "query", "write sql", "sql query"])

    def test_one_word_has_no_pair(self):
        self.assertEqual(nt._features(["sql"]), ["sql"])


class Topics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nlp = nt.lemmatizer()
        cls.documents = nt.lemmatize(DASHBOARDS + PIPELINES, nlp)
        cls.pipeline = nt.topic_pipeline(2, min_df=1, max_df=1.0)
        cls.weights = cls.pipeline.fit_transform(cls.documents)

    def test_two_themes_come_apart(self):
        """The first six lines are about dashboards and the last six about SQL. One topic each, whichever way
        round NMF happens to number them."""
        dominant = nt.dominant_topic(self.weights)
        self.assertEqual(len(set(dominant[:6])), 1, dominant)
        self.assertEqual(len(set(dominant[6:])), 1, dominant)
        self.assertNotEqual(dominant[0], dominant[6])

    def test_a_topic_is_described_by_its_own_theme_words(self):
        words = nt.topic_words(self.pipeline, n_words=6)
        self.assertEqual(len(words), 2)
        dominant = nt.dominant_topic(self.weights)
        dashboard_topic = words.loc[dominant[0], "top words"]
        sql_topic = words.loc[dominant[6], "top words"]
        self.assertIn("dashboard", dashboard_topic)
        self.assertIn("sql", sql_topic)
        self.assertNotIn("sql", dashboard_topic)

    def test_a_document_with_no_known_words_has_no_topic(self):
        """An empty document must not be counted into topic 0 just because argmax has to return something."""
        weights = np.array([[0.4, 0.1], [0.0, 0.0]])
        self.assertEqual(list(nt.dominant_topic(weights)), [0, nt.NO_TOPIC])

    def test_prevalence_is_a_share_per_group(self):
        groups = ["dashboards"] * 6 + ["pipelines"] * 6
        table = nt.prevalence(self.weights, groups)
        self.assertEqual(sorted(table.index), ["dashboards", "pipelines"])
        for _, row in table.iterrows():
            self.assertAlmostEqual(row.sum(), 1.0, places=6)
            self.assertAlmostEqual(row.max(), 1.0, places=6)      # each group is all one topic here

    def test_prevalence_weights_a_line_by_how_often_it_was_written(self):
        """The corpus is distinct lines, so a line written forty thousand times has to count forty thousand times."""
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        table = nt.prevalence(weights, ["bucket", "bucket"], counts=[9, 1])
        self.assertAlmostEqual(table.loc["bucket", 0], 0.9, places=6)

    def test_the_curve_reports_one_row_per_k(self):
        curve = nt.fit_curve(self.documents, [1, 2], min_df=1, max_df=1.0)
        self.assertEqual(list(curve.index), [1, 2])
        self.assertLess(curve.loc[2, "reconstruction error"], curve.loc[1, "reconstruction error"])


class Clusters(unittest.TestCase):
    # Six unit vectors: three on one axis, three on another, so the two groups must cluster apart.
    VECTORS = np.array([[1.0, 0.0], [0.99, 0.14], [0.98, 0.17],
                        [0.0, 1.0], [0.14, 0.99], [0.17, 0.98]])
    TEXTS = ["dash a", "dash b", "dash c", "sql a", "sql b", "sql c"]

    def test_the_two_groups_cluster_apart(self):
        _, labels = nt.cluster(self.VECTORS, 2)
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[3:])), 1)
        self.assertNotEqual(labels[0], labels[3])

    def test_the_nearest_lines_describe_each_cluster(self):
        model, labels = nt.cluster(self.VECTORS, 2)
        near = nt.nearest_to_centre(self.VECTORS, labels, self.TEXTS, model, per_cluster=2)
        self.assertEqual(len(near), 4)                                  # two clusters, two lines each
        for cluster_id, rows in near.groupby("cluster"):
            kinds = {text.split()[0] for text in rows.text}
            self.assertEqual(len(kinds), 1, f"cluster {cluster_id} mixes {kinds}")
            self.assertEqual(rows["lines in the cluster"].tolist(), [3, 3])
            self.assertTrue(rows["distance to centre"].is_monotonic_increasing)

    def test_asking_for_more_lines_than_a_cluster_has_is_fine(self):
        model, labels = nt.cluster(self.VECTORS, 2)
        near = nt.nearest_to_centre(self.VECTORS, labels, self.TEXTS, model, per_cluster=99)
        self.assertEqual(len(near), 6)


if __name__ == "__main__":
    unittest.main()

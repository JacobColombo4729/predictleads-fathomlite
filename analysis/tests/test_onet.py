"""src/onet.py on made-up O*NET tables (real headers) and made-up requirement lines. No real data."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import onet

RATING_HEAD = ("O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value\tN\tStandard Error\tLower CI Bound\tUpper CI Bound\t"
               "Recommend Suppress\tNot Relevant\tDate\tDomain Source")


def rating(code, eid, name, scale, value, suppress="N", irrelevant="n/a"):
    return f"{code}\t{eid}\t{name}\t{scale}\t{value}\t8\t0.1\t3\t4\t{suppress}\t{irrelevant}\t08/2025\tAnalyst"


TABLES = {
    "Occupation Data": "O*NET-SOC Code\tTitle\tDescription\n15-1252.00\tSoftware Developers\tWrites \"code\" for users\n"
                       "15-1232.00\tComputer User Support Specialists\tHelps users\n43-9061.00\tOffice Clerks\tFiles",
    "Essential Skills": "\n".join([RATING_HEAD,
        rating("15-1252.00", "2.A.1.a", "Programming", "IM", 4.5), rating("15-1252.00", "2.A.1.a", "Programming", "LV", 5.0),
        rating("15-1252.00", "2.A.2.a", "Critical Thinking", "IM", 4.0),
        rating("15-1252.00", "2.A.1.b", "Active Listening", "IM", 2.0, suppress="Y"),       # suppressed: dropped
        rating("15-1252.00", "2.B.1.a", "Negotiation", "LV", 3.0, irrelevant="Y"),          # not relevant: dropped
        rating("15-1252.00", "2.A.1.c", "Writing", "XX", 3.0),                              # other scale: dropped
        rating("15-1232.00", "2.A.2.a", "Critical Thinking", "IM", 3.5),
        rating("15-1232.00", "2.B.1.b", "Service Orientation", "IM", 4.2)]),
    "Transferable Skills": "\n".join([RATING_HEAD,
        rating("15-1252.00", "2.A.2.a", "Critical Thinking", "IM", 4.0), rating("15-1252.00", "2.A.1.d", "Reading Comprehension", "IM", 3.5),
        rating("15-1232.00", "2.A.2.a", "Critical Thinking", "IM", 3.5), rating("15-1232.00", "2.A.1.d", "Reading Comprehension", "IM", 3.2),
        rating("15-1232.00", "2.B.1.b", "Service Orientation", "IM", 4.2),
        rating("43-9061.00", "2.A.2.a", "Critical Thinking", "IM", 3.1), rating("43-9061.00", "2.A.1.d", "Reading Comprehension", "IM", 2.0),
        rating("43-9061.00", "2.B.1.b", "Service Orientation", "IM", 3.9)]),
    "Software Skills": "O*NET-SOC Code\tWorkplace Example\tElement ID\tElement Name\tHot Technology\tIn Demand\n"
                       "15-1252.00\tPython\t43232707\tObject or component oriented development software\tY\tY\n"
                       "15-1252.00\tJava\t43232707\tObject or component oriented development software\tN\tY\n"
                       "15-1252.00\tJavaScript\t43232707\tWeb platform development software\tY\tY\n"
                       "15-1252.00\tC++\t43232707\tObject or component oriented development software\tN\tN\n"
                       "15-1252.00\tGo\t43232707\tObject or component oriented development software\tN\tN\n"
                       "15-1252.00\tR\t43232707\tAnalytical software\tN\tN\n"
                       "15-1252.00\tMicrosoft Excel\t43232110\tSpreadsheet software\tY\tY\n"
                       "15-1232.00\tMicrosoft Excel\t43232110\tSpreadsheet software\tY\tY\n"
                       "15-1232.00\tZendesk\t43232201\tHelp desk software\tN\tN\n"
                       "43-9061.00\tMicrosoft Excel\t43232110\tSpreadsheet software\tY\tY",
    "Job Zones": "O*NET-SOC Code\tJob Zone\tDate\tDomain Source\n15-1252.00\t4\t08/2025\tAnalyst\n15-1232.00\t3\t08/2025\tAnalyst\n43-9061.00\t2\t08/2025\tAnalyst",
    "Related Occupations": "O*NET-SOC Code\tRelated O*NET-SOC Code\tRelatedness Tier\tIndex\n15-1252.00\t15-1232.00\tPrimary-Short\t1",
    "Content Model Reference": "Element ID\tElement Name\tDescription\n2.A.1.a\tProgramming\tWriting computer programs",
}

CLEAN = {  # id -> cleaned description (lines joined by newlines)
    "a": "About the role\nYou will build services all day.\nRequirements\nExperience with Python and JS\nSQL Server is not needed\n"
         "Strong critical-thinking skills\nPreferred Qualifications\nKnowledge of Java\nMS Excel is a plus",
    "b": "Requirements\nProficiency in JavaScript required\nHands-on with Java and Python",
    "c": "Responsibilities\nWe use Python every day with the team.\nRequirements\nA friendly attitude",    # tool only outside requirements
    "d": "Must have 2+ years of C++ and Node.js; familiarity with the Go programming language",
    "e": "",
}


def write_tables(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in TABLES.items():
        (folder / f"{name}.txt").write_text(text + "\n", encoding="utf-8", newline="\n")


def matcher_for(folder):
    soft = onet.software(folder)
    return onet.Matcher(onet.vocabulary(soft, [onet.ratings("Essential Skills", folder), onet.ratings("Transferable Skills", folder)]))


class Readers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_tables(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_occupations_keep_quotes_as_text(self):
        o = onet.occupations(self.dir)
        self.assertEqual(list(o.columns), ["O*NET-SOC Code", "Title", "Description"])
        self.assertEqual(len(o), 3)
        self.assertIn('"code"', o.Description[0])

    def test_ratings_keep_importance_and_level_and_drop_suppressed_or_not_relevant(self):
        r = onet.ratings("Essential Skills", self.dir)
        self.assertEqual(set(r.scale), {"IM", "LV"})
        self.assertNotIn("Active Listening", set(r.element))   # suppressed
        self.assertNotIn("Negotiation", set(r.element))        # not relevant
        self.assertNotIn("Writing", set(r.element))            # other scale
        self.assertEqual(r.value.dtype, float)

    def test_software_flags_are_booleans(self):
        s = onet.software(self.dir)
        py = s[s.tool == "Python"].iloc[0]
        self.assertTrue(py.hot and py.in_demand)
        self.assertFalse(s[s.tool == "Java"].iloc[0].hot)

    def test_job_zones_related_and_content_model(self):
        self.assertEqual(onet.job_zones(self.dir).job_zone.tolist(), [4, 3, 2])
        self.assertEqual(onet.related(self.dir).related.tolist(), ["15-1232.00"])
        self.assertEqual(onet.content_model(self.dir).element.tolist(), ["Programming"])


class Normalizing(unittest.TestCase):
    def test_punctuation_and_special_names(self):
        self.assertEqual(onet.normalize("C++, C#, and .NET"), "cpp csharp and dotnet")
        self.assertEqual(onet.normalize("Node.js/React"), "nodejs react")
        self.assertEqual(onet.normalize("Critical-thinking"), "critical thinking")

    def test_aliases(self):
        self.assertEqual(onet.normalize("JS and MS Excel"), "javascript and microsoft excel")
        self.assertEqual(onet.normalize("TS/SCI clearance"), "ts sci clearance")  # not a TypeScript alias


class Matching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_tables(cls.dir)
        cls.m = matcher_for(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def names(self, text):
        return {n for n, _, _ in self.m.find(text)}

    def test_vocabulary_has_tools_and_skills_and_skips_ambiguous_names(self):
        v = onet.vocabulary(onet.software(self.dir), [onet.ratings("Essential Skills", self.dir)])
        self.assertEqual(set(v.kind), {"tool", "skill"})
        self.assertTrue(v[v.phrase.isin(["go", "r"])].context.all())  # kept, matched only in context
        self.assertFalse(v[v.name == "Python"].context.iloc[0])
        self.assertTrue(v[v.name == "Python"].hot.iloc[0])
        self.assertEqual(v[v.name == "Microsoft Excel"].n_occupations.iloc[0], 3)

    def test_r_and_go_only_in_context(self):
        self.assertIn("R", self.names("Python or R for analysis"))
        self.assertIn("Go", self.names("Experience with Golang microservices"))
        self.assertIn("Go", self.names("Java, Go or Rust"))
        self.assertNotIn("Go", self.names("Ready to go the extra mile"))
        self.assertNotIn("R", self.names("Work with our R&D team"))
        self.assertNotIn("R", self.names("Grade R accounting"))

    def test_whole_phrases_only(self):
        self.assertEqual(self.names("We use JavaScript daily"), {"JavaScript"})   # not Java inside JavaScript
        self.assertEqual(self.names("Java and Python"), {"Java", "Python"})
        self.assertEqual(self.names("pythonic scripting; javanese cooking"), set())

    def test_aliases_and_special_names_match(self):
        self.assertEqual(self.names("js, MS Excel and C++"), {"JavaScript", "Microsoft Excel", "C++"})

    def test_skill_elements_match_with_other_punctuation(self):
        self.assertEqual(self.names("Strong critical-thinking skills"), {"Critical Thinking"})
        self.assertEqual(self.names("programming"), {"Programming"})

    def test_ambiguous_words_do_not_match(self):
        self.assertEqual(self.names("go to market, then R&D"), set())
        self.assertEqual(self.names("the Go language"), {"Go"})  # in context: "language" next to it


class Extraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_tables(cls.dir)
        cls.m = matcher_for(cls.dir)
        cls.clean = cls.dir / "clean.parquet"
        pq.write_table(pa.table({"id": list(CLEAN), "desc_clean": list(CLEAN.values())}), cls.clean)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pairs(self, d):
        return {(n, s) for n, _, s in ((k[0], k[1], v) for k, v in d.items())}

    def test_required_vs_preferred(self):
        got = self.pairs(onet.listing_skills(CLEAN["a"], self.m))
        self.assertIn(("Python", "required"), got)
        self.assertIn(("JavaScript", "required"), got)
        self.assertIn(("Critical Thinking", "required"), got)
        self.assertIn(("Java", "preferred"), got)
        self.assertIn(("Microsoft Excel", "preferred"), got)
        self.assertNotIn(("Java", "required"), got)

    def test_required_wins_over_preferred(self):
        text = "Requirements\nPython\nPreferred\nPython and Java"
        got = self.pairs(onet.listing_skills(text, self.m))
        self.assertIn(("Python", "required"), got)
        self.assertNotIn(("Python", "preferred"), got)
        self.assertIn(("Java", "preferred"), got)

    def test_only_requirement_lines_are_read(self):
        self.assertEqual(onet.listing_skills(CLEAN["c"], self.m), {})
        self.assertEqual(onet.listing_skills("", self.m), {})

    def test_cue_line_without_a_heading_counts_as_required(self):
        got = self.pairs(onet.listing_skills(CLEAN["d"], self.m))
        self.assertEqual(got, {("C++", "required"), ("Programming", "required"), ("Go", "required")})  # "the Go programming language"; Node.js is not in the made-up vocabulary

    def test_extract_writes_rows_and_cache_check(self):
        out = self.dir / "skills.parquet"
        self.assertFalse(onet.cache_valid(out, self.clean, self.m))
        n = onet.extract(self.clean, self.m, out, batch_size=2, workers=1)
        s = pq.read_table(out).to_pandas()
        self.assertEqual(n, len(s))
        self.assertEqual(list(s.columns), ["id", "name", "kind", "section", "element_id"])
        self.assertEqual(set(s.id), {"a", "b", "d"})  # c and e have no match in requirement lines
        self.assertEqual(set(s[s.id == "b"].name), {"JavaScript", "Java", "Python"})
        self.assertFalse(list(self.dir.glob("*.part")))
        self.assertTrue(onet.cache_valid(out, self.clean, self.m))

    def test_batches_and_workers_give_the_same_rows(self):
        a, b = self.dir / "s1.parquet", self.dir / "s2.parquet"
        onet.extract(self.clean, self.m, a, batch_size=1, workers=1)
        onet.extract(self.clean, self.m, b, batch_size=5, workers=2)
        key = ["id", "name", "section"]
        pd.testing.assert_frame_equal(pq.read_table(a).to_pandas().sort_values(key).reset_index(drop=True),
                                      pq.read_table(b).to_pandas().sort_values(key).reset_index(drop=True))

    def test_cache_invalid_after_other_vocabulary_or_other_text(self):
        out = self.dir / "s3.parquet"
        onet.extract(self.clean, self.m, out, workers=1)
        other = onet.Matcher(onet.vocabulary(onet.software(self.dir).head(3), [onet.ratings("Essential Skills", self.dir)]))
        self.assertFalse(onet.cache_valid(out, self.clean, other))
        longer = self.dir / "clean2.parquet"
        pq.write_table(pa.table({"id": ["a", "b", "x"], "desc_clean": ["p", "q", "r"]}), longer)
        self.assertFalse(onet.cache_valid(out, longer, self.m))
        self.assertFalse(onet.cache_valid(self.dir / "missing.parquet", self.clean, self.m))

    def test_failed_run_leaves_no_part_file(self):
        out = self.dir / "s4.parquet"
        bad = self.dir / "bad.parquet"
        pq.write_table(pa.table({"id": ["a"], "desc_clean": [None]}), bad)   # a null description is read as no text
        onet.extract(bad, self.m, out, workers=1)
        self.assertEqual(pq.read_table(out).num_rows, 0)
        self.assertFalse(list(self.dir.glob("*.part")))


class Profiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_tables(cls.dir)
        cls.ess = onet.ratings("Essential Skills", cls.dir)
        cls.soft = onet.software(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_agreement_with_own_occupation(self):
        skills = pd.DataFrame({"id": ["a", "a", "a", "b", "z"], "name": ["Python", "Zendesk", "Critical Thinking", "Python", "Python"],
                               "kind": ["tool", "tool", "skill", "tool", "tool"]})
        codes = pd.Series({"a": "15-1252.00", "b": "15-1232", "z": None})   # b uses a 7-character code
        listed = onet.listed_by_code(codes, self.soft, self.ess)
        g = onet.agreement(skills, codes, listed)
        self.assertEqual(g.loc["a", ["n_matched", "n_listed"]].tolist(), [3, 2])  # Zendesk is a support tool
        self.assertAlmostEqual(g.loc["a", "share"], 2 / 3)
        self.assertEqual(g.loc["b", "n_listed"], 0)
        self.assertNotIn("z", g.index)

    def test_job_zone_exact_and_prefix(self):
        z = onet.job_zone_for(pd.Series(["15-1252.00", "15-12", "99-9999.00", None]), onet.job_zones(self.dir))
        self.assertEqual(z.iloc[0], 4)
        self.assertEqual(z.iloc[1], 3.5)  # median of 4 and 3
        self.assertTrue(pd.isna(z.iloc[2]) and pd.isna(z.iloc[3]))

    def test_carry_over_between_background_and_target(self):
        mat = onet.skill_matrix(onet.ratings("Transferable Skills", self.dir))
        target = onet.profile(mat, ["15-1252", "15-1232"], weights={"15-1252": 3, "15-1232": 1})
        self.assertAlmostEqual(target["Critical Thinking"], (4.0 * 3 + 3.5) / 4)
        clerk = onet.profile(mat, ["43-9061.00"])
        shared = onet.carry_over(clerk, target, threshold=3.0)
        self.assertEqual(list(shared.index), ["Service Orientation", "Critical Thinking"])  # reading comprehension: the clerk's rating is low
        self.assertAlmostEqual(target["Service Orientation"], 4.2)  # only one occupation rates it, so it isn't diluted
        self.assertEqual(onet.profile(mat, ["99-9999"]).empty, True)


if __name__ == "__main__":
    unittest.main()

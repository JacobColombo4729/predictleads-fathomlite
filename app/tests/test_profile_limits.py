"""The profile limits agree across Magic fill, the Profile form and the backend, so a filled profile always saves.

Magic fill's clean-up (code/resume_profile_extraction.py LIMITS) must never return more than the form accepts
(frontend/src/lib/profile.ts LIMITS), and the form must never accept more than the backend (backend/api.py ProfileData).
A mismatch here once made "Save profile" do nothing after Magic fill (2026-10-15).
"""
import re
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "code"))
sys.path.insert(0, str(APP))

import resume_profile_extraction as rpe  # noqa: E402
from backend import api  # noqa: E402

# Magic fill's key -> the form's key
FILL_TO_FORM = {"name": "name", "headline": "headline", "title": "title", "organization": "organization",
                "description": "experienceText", "skill": "skillName", "evidence": "evidence", "location": "location",
                "max_experience": "maxExperience", "max_skills": "maxSkills"}


def form_limits():
    text = (APP / "frontend" / "src" / "lib" / "profile.ts").read_text(encoding="utf-8")
    block = text[text.index("export const LIMITS = {"):text.index("} as const;")]
    return {k: int(v) for k, v in re.findall(r"(\w+):\s*(\d+)", block)}


def backend_limit(model, field):
    for meta in model.model_fields[field].metadata:
        if getattr(meta, "max_length", None) is not None:
            return meta.max_length
    raise AssertionError(f"{model.__name__}.{field} has no max_length")


class ProfileLimitsAgree(unittest.TestCase):
    def test_magic_fill_never_returns_more_than_the_form_accepts(self):
        form = form_limits()
        for fill_key, form_key in FILL_TO_FORM.items():
            self.assertLessEqual(rpe.LIMITS[fill_key], form[form_key], f"Magic fill's {fill_key} vs the form's {form_key}")

    def test_the_form_never_accepts_more_than_the_backend(self):
        form = form_limits()
        backend = {"title": backend_limit(api.ExperienceEntry, "title"),
                   "organization": backend_limit(api.ExperienceEntry, "organization"),
                   "experienceText": backend_limit(api.ExperienceEntry, "description"),
                   "skillName": backend_limit(api.SkillEntry, "name"),
                   "evidence": backend_limit(api.SkillEntry, "evidence"),
                   "name": backend_limit(api.ProfileData, "name"),
                   "headline": backend_limit(api.ProfileData, "headline"),
                   "location": backend_limit(api.ProfileData, "location"),
                   "constraints": backend_limit(api.ProfileData, "constraints"),
                   "maxExperience": backend_limit(api.ProfileData, "experience"),
                   "maxSkills": backend_limit(api.ProfileData, "skills")}
        for key, limit in backend.items():
            self.assertLessEqual(form[key], limit, f"the form's {key} vs the backend")


if __name__ == "__main__":
    unittest.main()

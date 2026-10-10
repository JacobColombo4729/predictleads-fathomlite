# Grounded Match Explanations

Standalone service for explaining one selected candidate/job pair. It does not retrieve or rank jobs. The service uses source-linked evidence IDs, keeps required/preferred/unclear modality separate, treats profile silence as unknown, and rejects references outside the supplied evidence catalogs.

## Contract

`code/grounded_match_explanations.py` exports:

- `explain_match(candidate, job, provider=...)`: reusable service function. Inject a provider implementing `generate(payload) -> Mapping`.
- `explain_match_with_openai(candidate, job, ...)`: convenience function that constructs the OpenAI provider from environment configuration.
- `RESPONSE_SCHEMA`: strict structured-output JSON Schema.
- `MockProvider`: offline provider for local development and tests.

Inputs are mappings based on the existing synthetic candidate fixture and prepared job fields. Candidate evidence fields include `experience`/`work_history`, `evidenced_skills`/`skills`, `education`, `training`, `certifications`, `credentials`, `projects`, `evidence`, and `qualifications`. Job evidence fields include `title`, `title_searchable`, `description_text`/`description`, and explicit requirements/responsibilities fields. Identity, URL, status, dates, pay, and company metadata are not treated as evidence for a qualification. Multiline text is split into citeable nonblank lines while preserving source order and text.

Response fields:

- `profile_id`, `job_id`, `schema_version`
- `evidence_sufficiency`: `sufficient` or `insufficient`; insufficient responses include a reason and empty assessment arrays.
- `transferable_experience`: job duty, candidate/job evidence IDs, and rationale.
- `qualifications`: stable qualification ID, source wording, `required`/`preferred`/`unclear` classification, `supported`/`explicit_gap`/`unknown` status, evidence IDs, and rationale.
- `next_steps`: practical action, qualification IDs addressed, and supporting evidence IDs. Steps may only address unknown or explicit-gap qualifications.
- `evidence_catalog`: evidence ID, source field, and exact source text for each citation.

`explicit_gap` means a directly evidenced contradiction, not missing profile information. An unknown qualification may cite related candidate evidence as context, but its status stays unknown and the rationale must say why that evidence does not establish the qualification. A transfer claim must cite both candidate and job evidence. Unknown and preferred qualifications are not eligibility or hiring conclusions.

A null, blank, missing, or non-text job description returns an `insufficient` result without calling the provider. A profile with no supported evidence does the same. Provider and response-validation errors are raised; there is no fallback to fabricated output.

## Local use

The saved fictional pair is `code/fixtures/grounded_explanation_pair.json`. Mock mode requires no network or credentials:

```python
import json
from pathlib import Path

from grounded_match_explanations import MockProvider, explain_match

fixture = json.loads(Path("code/fixtures/grounded_explanation_pair.json").read_text(encoding="utf-8"))
provider = MockProvider(response_mapping)  # supply a response conforming to RESPONSE_SCHEMA
result = explain_match(fixture["candidate"], fixture["job"], provider=provider)
```

OpenAI mode reads `OPENAI_API_KEY` from the process environment first, then the repository-root `.env` using the already-declared `python-dotenv` package. An optional `PREDICTLEADS_EXPLANATION_MODEL` overrides the default model `gpt-4.1-mini`.

```python
from grounded_match_explanations import explain_match_with_openai

result = explain_match_with_openai(candidate_mapping, selected_job_mapping)
```

The convenience function makes a paid network request; it is deliberately not used by the test suite. Do not print credentials, add them to fixtures, or commit `.env`. The source description and candidate evidence are sent to the provider when OpenAI mode is used.

## Endpoint and integration

No FastAPI, Flask, or other backend web application exists in the current repository, so this feature adds no HTTP endpoint and introduces no framework dependency. A future API layer can validate a request body containing `candidate` and `job`, call `explain_match`, and serialize its returned mapping. The selected job should be supplied by the caller; this service must not rank or choose it.

The contract is intentionally independent of retrieval internals. A future adapter can map the `CandidateProfile`/prepared job records into the accepted evidence fields. Store the returned IDs and evidence catalog together so the UI can render each assessment with its source text. Treat provider failures as unavailable explanations, not as a negative match. No result should be represented as verified current availability, employer intent, competition, eligibility, or a hiring probability.

## Verification and limitations

Run only the focused offline tests for this feature:

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -p 'test_grounded_match_explanations.py' -v
```

The tests use the saved fictional pair and `MockProvider`; they check evidence references, unknown and explicit-gap rules, required/preferred distinctions, missing descriptions, response identity, invalid references, next-step grounding, and provider failure handling. They do not make API calls. Live OpenAI schema compatibility, generation quality, and semantic correctness remain unverified until a separately approved live test is run.

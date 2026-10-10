# Claude Handoff: Qualification Protocol Experiment

**As of:** 2026-10-09  
**Current instruction:** Stop provider work here. Claude's next task is offline inspection of saved incomplete responses and semantic failures, then a proposal for a smaller response contract. No paid rerun is authorized beyond the existing $3 cumulative cap.

## Scope and Guardrails

This is an evaluator-protocol experiment on 12 purposively selected development pairs. It is not improved matching, general validation, independent human review, or evidence for a ranking change. Do not inspect held-out profiles. Do not call an API, restart an evaluator, retry a pair, reset/alter the ledger, edit the frontend, change branches, commit, or push. Preserve all existing inputs, raw Responses, original/v2 judgments, grades, metrics, and failed attempts.

The wastewater operator case is a development regression only. The original reviewer input contained the complete 4,351-character description, including driver-license and one-year wastewater-license wording; it was not truncated. See [the evidence sheet](../data/matching/weighted_rrf_20261008T233000Z/diagnostics/dev_top5_strong_ceiling_20261009/wastewater_operator_ai_evidence_sheet.md).

## Code and Tests

- Runner: [code/run_weighted_rrf_qualification_protocol.py](../code/run_weighted_rrf_qualification_protocol.py)
- Focused tests: [tests/test_weighted_rrf_qualification_protocol.py](../tests/test_weighted_rrf_qualification_protocol.py)
- Full report: [comparison_report.md](../data/matching/weighted_rrf_20261008T233000Z/diagnostics/dev_top5_strong_ceiling_20261009/blinded_grader_audit_20261009/qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/comparison_report.md)
- Per-pair comparison: [protocol_comparison.jsonl](../data/matching/weighted_rrf_20261008T233000Z/diagnostics/dev_top5_strong_ceiling_20261009/blinded_grader_audit_20261009/qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/protocol_comparison.jsonl)
- Offline v3.2 summary recalculation: [python_eligibility_recalculation_v3_2.jsonl](../data/matching/weighted_rrf_20261008T233000Z/diagnostics/dev_top5_strong_ceiling_20261009/blinded_grader_audit_20261009/qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/python_eligibility_recalculation_v3_2.jsonl)
- Recalculation provenance: [python_eligibility_recalculation_v3_2_manifest.json](../data/matching/weighted_rrf_20261008T233000Z/diagnostics/dev_top5_strong_ceiling_20261009/blinded_grader_audit_20261009/qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/python_eligibility_recalculation_v3_2_manifest.json)

Experiment root: `data/matching/weighted_rrf_20261008T233000Z/diagnostics/dev_top5_strong_ceiling_20261009/blinded_grader_audit_20261009/`.

- Failed initial v3 package: `qualification_protocol_v3_20261009/`
- Validator-only correction and final frozen prompt/schema/input: `qualification_protocol_v3_1_validator_correction_20261009/`
- GPT-4.1 run and saved outputs: `qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/`

## Reproduction Commands

The following commands are offline only:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -p 'test_weighted_rrf_qualification_protocol.py' -v
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' '.\code\run_weighted_rrf_qualification_protocol.py' compare
```

The last command recomputes derived comparison/report files from saved artifacts; it does not call a provider. The protocol's `freeze`, `run-first`, and `run-rest` commands are not part of the handoff workflow. Do not use `run-rest`; it can make paid provider requests. The first pair was already replayed offline under v3.1; there is no need to rerun it.

For offline response inspection, start with:

- Failed v3 Responses (two completed responses rejected by the initial adapter): `qualification_protocol_v3_20261009/model_comparison_gpt41_20261009/raw_responses.jsonl`
- v3.1 Responses, including incomplete/truncated attempts: `qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/raw_responses.jsonl`
- Validation outcomes: `qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/validation_attempts.jsonl`
- Pair outcomes: `qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/pair_status.jsonl`
- Unchanged blinded inputs: `qualification_protocol_v3_1_validator_correction_20261009/model_comparison_gpt41_20261009/reviewer_input_blinded.jsonl`

## Frozen Settings and Provenance

- Pinned model: `gpt-4.1-2025-04-14`.
- Prompt SHA-256: `0aa2f47c63865cf6a9a9c25b97bc3e7a8843ba3e3dfbc36d336681a7682a9557`.
- Schema SHA-256: `40a969d125fd0743773ae5f6c0d3ba4b02f59628ec461f2f7b53ac6b1649ceae`.
- v3.1 corrected only validator/orchestration behavior after v3's failed adapter. The v3.1 prompt, schema, and normalized 12-pair input are byte-identical to failed v3; their raw responses remain in v3 and were hash-checked in the v3.1 manifest.
- Request settings: temperature 0, `store=false`, maximum output 2,400 tokens per attempt, at most two attempts per request. Existing credential loading, request caching, and shared bounded retry/ledger mechanisms were used. No credentials are included in the artifacts.
- The provider schema requests independent qualification findings and no aggregate eligibility field. Each qualification includes exact job excerpt, line range, job-line evidence IDs, classification (`mandatory`, `preferred`, `unclear`), timing (`at_entry`, `explicitly_obtainable_after_joining`, `unclear`), deadline excerpt, candidate status (`supported`, `explicitly_contradicted`, `unknown`), candidate excerpt/evidence IDs, and explanation. The full description is supplied once; line-level job evidence IDs carry source locations and context.
- Frozen prompt/schema/input hashes and cost preflight are in `qualification_protocol_v3_1_validator_correction_20261009/protocol_manifest.json` and the model comparison's `preflight_manifest.json`.

## Technical Outcome

All 12 pair positions have a terminal status: 6 mechanically validated (the first is a saved v3 response replayed offline), 6 failed/unvalidated, and 0 not attempted. Five requests exhausted the output-token cap on both bounded attempts. One pair failed exact job-excerpt/location and unknown-candidate-evidence validation. These partials are preserved; do not reconstruct, repair, or silently complete them.

The first v3 request itself completed at the provider, but its local adapter rejected structured `material_gaps` and `explicit_constraint_conflicts` objects because it passed them through a legacy validator requiring string arrays. The same response also cited a disclosed unknown experience while marking that qualification unknown. v3.1 separated common grade checks from its structured claim validator and replayed the already-paid response with no new request. This is a validator correction, not a prompt retune.

## Semantic Concerns

Mechanical source identity and excerpt-location checks do not establish entailment. The validated set contains five responses with grade-blocking semantic concerns:

- Four unsupported material-gap claims, including wording that promotes an explicitly unknown candidate experience to a gap.
- Two unsupported explicit-conflict claims, including a drinking-water credential cited against a distinct wastewater or water-plant credential and/or a post-entry obligation.
- One cross-credential wastewater finding unsupported by the cited drinking-water evidence.
- Two `explicitly_contradicted` labels lacking direct negative wording in the candidate source; hotel banquet coordination was cited against minimum logistics supervision and 50+ direct reports.

Counts are concern instances, not unique cases. Per-pair claims, excerpts, IDs, and concern codes are in `protocol_comparison.jsonl` and `validated_judgments.jsonl`. Do not treat model classifications as ground truth, infer credential equivalence, or include grades from semantically flagged records in aggregate conclusions. Only one of the six validated responses has no grade-blocking concern; grade agreement on that eligible subset is 0/1, which is not a quality estimate.

## v3.2 Offline Eligibility Recalculation

The v3.2 artifact is a derived-data-only recalculation. It preserves each v3.1 summary beside the recalculated summary; provider raw responses and validated findings are unchanged.

- Entry conflict is counted only for a mandatory-at-entry finding with candidate status `explicitly_contradicted` that is not blocked by semantic flags.
- If a contradiction is flagged `contradiction_not_explicit_in_candidate_source` or `cross_credential_evidence_does_not_support_wastewater_finding`, the calculated candidate status for eligibility is effectively unknown. The original provider status remains visible in its finding.
- In the absence of an evidenced entry conflict, any unknown mandatory entry item, unresolved mandatory timing/classification, insufficient/unclear extraction coverage, or no identified mandatory entry requirement prevents an eligibility-supported conclusion.
- Entry status `supported` requires every identified mandatory entry requirement to be supported, sufficient extraction coverage, and no unresolved requirement classification or timing.
- Mandatory post-entry obligations and preferred qualifications are listed separately and do not become entry conflicts. A future deadline does not prove candidate ability to qualify.

After this recalculation, the six valid responses have four `unknown` and two `supported` entry summaries; none is reported as an entry conflict. The hotel/supply positive-coordination evidence no longer becomes a Python-derived conflict merely because the model used `explicitly_contradicted`. This is a cautious offline summary correction, not a changed provider finding or human adjudication.

## Budget and Process State

Shared append-only ledger: `data/matching/bge_transition_20261008T142818Z/openai_evaluator/budget_ledger.sqlite3`  
Authorized cumulative cap: **$3.00**; the final reserved total remains under this cap.  
Final reserved maximum: **$2.915392**  
Final usage-priced amount: **$1.567338**

The evaluator process check on 2026-10-09 found **0 active** `run_weighted_rrf_qualification_protocol`, `run_weighted_rrf_model_comparison`, or `run_weighted_rrf_grader_audit` processes. No paid rerun beyond the existing cap is authorized. Claude's next work should be offline response inspection and a proposal for a smaller response contract only.

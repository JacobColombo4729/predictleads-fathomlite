"""The combined model: ML_Fit's trained layer as the fit factor of ML_Rank's priority queue, with Jesse's TF-IDF + BGE
text match picking the candidates, over the agreed scope's listings. A third model, run as its own app by combined_demo/.

Plan and design: app/docs/combined-model-plan.md ("Modular design"). Built so far:
    corpus.py       the serving corpus from the scope, in the app's corpus format (step 1a)
    role_fit.py     a person's fit for every role, with reasons (step 3; steps 1-2 live in analysis/src)
    candidates.py   the candidate pool: text matches overall and within the top roles (step 4)
    queue.py        ML_Rank's priority queue over the pool (step 5)
    model.py        CombinedModel: steps 3-5 end to end
"""

VERSION = "combined-v2"  # v2: the roles come from the whole profile, not the trained layer alone

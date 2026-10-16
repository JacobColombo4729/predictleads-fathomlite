"""The combined model: ML_Fit's trained layer as the fit factor of ML_Rank's priority queue, with Jesse's TF-IDF + BGE
text match picking the candidates, over the agreed scope's listings. A third model, pluggable into either app.

Plan and design: app/docs/combined-model-plan.md ("Modular design"). Built so far:
    corpus.py   the serving corpus from the scope, in the app's corpus format (step 1a)
"""

VERSION = "combined-v1"

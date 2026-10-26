"""What requirement text is about: topics from TF-IDF + NMF, and clusters from sentence embeddings.

**No regular expressions** (`EDA_SDA_3_Lean`, decided 2026-10-25), and this is the one place where keeping that
promise changes the method rather than just the code. `TfidfVectorizer`'s default tokenizer *is* a regular
expression (`token_pattern`, a character pattern for "a run of two or more word characters"), so it is not used at
all here. The corpus is tokenized and lemmatized by spaCy first and TF-IDF is handed the token lists, with
`analyzer` set to a function over them.

That is not a workaround, it is better: `machine`/`machines`, `build`/`building`/`built` and `year`/`years` each
become one feature instead of several, stopwords are dropped by the model's own judgement (`token.is_stop`) rather
than by a hand-written list, and `3`, `+`, `&` and `(` never reach the vocabulary. A character pattern cannot do
any of that - which is why the original `EDA_SDA_3` lemmatized a sample with spaCy and then fed the raw strings to
TF-IDF anyway, with nothing downstream reading the lemmas.

What it costs: the tagger is reading bullet fragments, not sentences, and the small model reads a fragment that
opens with a gerund as a noun - "Building dashboards" lemmatizes to `building`, not `build`. So some inflections
survive as separate features. It changes the words a topic is described by, not which themes are found, and the
alternative (a bigger model, or stemming, which is a character rule) costs more than it is worth here.

Two unsupervised readings of the same text, because they fail differently:

  * **TF-IDF -> NMF topics.** Word-level, and readable: each topic is a list of words, so a reader can say what it
    is. A document gets a weight on every topic. Blind to meaning - two bullets saying the same thing in different
    words land in different topics.
  * **Embedding clusters (BGE + k-means).** Meaning-level, so paraphrases land together, but a cluster has no
    words of its own: it is described by the real lines closest to its centre, which is what `nearest_to_centre`
    returns for the notebook to print.

Neither is a label on a listing. They answer "what is requirement text about, and does that differ by role
bucket?", which is `EDA_SDA_3_Lean` section 6 and 7.

Teaching doc: `docs/nlp_methods.md` (part 4).
"""
from functools import partial

import numpy as np
import pandas as pd

MODEL = "en_core_web_sm"
SEED = 12345            # the project's seed (src/fit_model.py)
MIN_LEMMA_LENGTH = 2    # "ml" and "qa" are real; one-character tokens never are
NO_TOPIC = -1           # a document with no feature in the vocabulary has no strongest topic


def lemmatizer(model=MODEL):
    """The pipeline lemmatizing needs and nothing more: the tagger and lemmatizer, with the parser and NER
    disabled. Roughly three times the throughput of the full pipeline `nlp_requirements` uses."""
    import spacy

    return spacy.load(model, disable=["parser", "ner"])


def lemma_tokens(doc):
    """One document's content words, lemmatized and lowercased. Stopwords, punctuation, numbers, URLs and
    whitespace are dropped - `token.is_stop` is the language model's own list, not one written here."""
    return [token.lemma_.lower() for token in doc
            if token.is_alpha and not token.is_stop and len(token.lemma_) >= MIN_LEMMA_LENGTH]


def lemmatize(texts, nlp, batch_size=256, n_process=1):
    """Every text as a list of lemmas, in order. Done once for the whole corpus, outside the pipeline below, so
    that fitting the same corpus at several values of k does not re-run spaCy each time."""
    return [lemma_tokens(doc) for doc in nlp.pipe(list(texts), batch_size=batch_size, n_process=n_process)]


def _features(tokens, ngram_max=2):
    """The features of one already-lemmatized document: its lemmas, plus adjacent pairs of them. `ngram_range`
    cannot do this once `analyzer` is a function, and pairs are worth having - "machine learning" and "year
    experience" say more than their halves."""
    features = list(tokens)
    for n in range(2, ngram_max + 1):
        features += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return features


def vectorizer(ngram_max=2, min_df=5, max_df=0.4, max_features=20_000):
    """TF-IDF over pre-lemmatized documents. `analyzer` is the function above, so no `token_pattern` is involved
    and the input is token lists rather than strings. `max_df` drops the words every bullet uses, which in job
    descriptions is most of what makes topics unreadable."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(analyzer=partial(_features, ngram_max=ngram_max), lowercase=False,
                           min_df=min_df, max_df=max_df, max_features=max_features, sublinear_tf=True)


def topic_pipeline(k, seed=SEED, max_iter=400, **vectorizer_kwargs):
    """TF-IDF then NMF as one estimator, so the vocabulary is fitted inside the pipeline (the project's rule) and
    changing k refits both. NMF factors the documents x terms matrix into documents x topics and topics x terms,
    with everything held non-negative, which is what makes a topic readable as a list of words."""
    from sklearn.decomposition import NMF
    from sklearn.pipeline import Pipeline

    return Pipeline([("tfidf", vectorizer(**vectorizer_kwargs)),
                     ("nmf", NMF(n_components=k, init="nndsvd", random_state=seed, max_iter=max_iter))])


def topic_words(pipeline, n_words=12):
    """One row per topic: its strongest words, in order. This is the whole description of a topic, and the only
    thing that makes the method defensible - a topic nobody can name is a topic nobody should report."""
    terms = pipeline["tfidf"].get_feature_names_out()
    weights = pipeline["nmf"].components_
    rows = [{"topic": t, "top words": ", ".join(terms[i] for i in row.argsort()[::-1][:n_words])}
            for t, row in enumerate(weights)]
    return pd.DataFrame(rows).set_index("topic")


def fit_curve(documents, ks, seed=SEED, **kwargs):
    """Reconstruction error at each k, for choosing k from the data rather than from habit. The error always falls
    as k rises, so what to look for is where it stops falling steeply."""
    rows = []
    for k in ks:
        pipeline = topic_pipeline(k, seed=seed, **kwargs)
        pipeline.fit(documents)
        rows.append({"k": k, "reconstruction error": pipeline["nmf"].reconstruction_err_,
                     "terms in the vocabulary": len(pipeline["tfidf"].get_feature_names_out())})
    return pd.DataFrame(rows).set_index("k")


def dominant_topic(weights):
    """Each document's strongest topic, or `NO_TOPIC` where a document has no weight on any of them (every word it
    used was dropped from the vocabulary). Guessing topic 0 for those would quietly inflate it."""
    weights = np.asarray(weights)
    strongest = weights.argmax(axis=1)
    return np.where(weights.sum(axis=1) > 0, strongest, NO_TOPIC)


def prevalence(weights, groups, counts=None):
    """Group x topic: the share of each group's text whose strongest topic is that topic.

    `counts` weights each document by how many blocks it stands for, since the corpus is the *distinct* requirement
    lines - without it a line written once counts as much as one written forty thousand times."""
    frame = pd.DataFrame({"group": list(groups), "topic": dominant_topic(weights),
                          "weight": 1.0 if counts is None else np.asarray(counts, dtype="float64")})
    table = frame.pivot_table(index="group", columns="topic", values="weight", aggfunc="sum", fill_value=0.0)
    return table.div(table.sum(axis=1), axis=0)


def cluster(vectors, k, seed=SEED, batch_size=4_096):
    """k-means over unit-length sentence embeddings. On unit vectors the straight-line distance k-means minimises
    and cosine distance order pairs the same way, so this clusters by meaning. MiniBatch because the corpus is
    tens of thousands of lines wide and the result is the same shape."""
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=batch_size, n_init=10)
    labels = model.fit_predict(np.asarray(vectors))
    return model, labels


def nearest_to_centre(vectors, labels, texts, model, per_cluster=5):
    """The lines closest to each cluster's centre: a cluster has no words of its own, so these *are* its
    description, and the notebook prints them for a reader to name the cluster by."""
    vectors, labels, texts = np.asarray(vectors), np.asarray(labels), list(texts)
    rows = []
    for c in range(model.n_clusters):
        members = np.flatnonzero(labels == c)
        if members.size == 0:
            continue
        distance = np.linalg.norm(vectors[members] - model.cluster_centers_[c], axis=1)
        for rank, position in enumerate(distance.argsort()[:per_cluster], start=1):
            rows.append({"cluster": c, "rank": rank, "lines in the cluster": members.size,
                         "distance to centre": float(distance[position]), "text": texts[members[position]]})
    return pd.DataFrame(rows)

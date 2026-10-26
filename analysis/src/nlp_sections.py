"""What a section of a job description is about, decided from the meaning of its heading.

**No regular expressions and no keyword lists** (`EDA_SDA_3_Lean`, decided 2026-10-25). The question "is this
section the requirements, the nice-to-haves, or the company blurb?" is answered the way a reader answers it: by
what the heading *means*. Each heading is encoded as a sentence embedding (BGE) and compared with a handful of
**anchor sentences** that describe each class in plain English. The heading joins the class whose anchors it is
closest to. This is zero-shot classification by semantic similarity: no training data, no word list, and a heading
nobody anticipated ("What Success Looks Like", "Your Toolkit") still lands somewhere sensible, which is exactly what
a word list cannot do.

Two safeguards, because a similarity is not a certainty:

  * **margin** - the gap between the best class and the runner-up. A heading whose two best classes are within
    `MIN_MARGIN` is labelled `unclear` rather than guessed at, and the notebook reports how many those are.
  * **distinct headings only** - headings repeat across hundreds of thousands of listings, so the encoder runs on
    the distinct strings (tens of thousands) and the labels are joined back. That is what makes this affordable at
    all; it is also why the notebook can print every distinct heading above a frequency and let a reader check the
    labels by eye.

The same function labels an orphan block - a bullet in a listing that has no headings at all - from its own text,
since there is no heading above it to inherit from.

Teaching doc: `docs/nlp_methods.md` (part 2).
"""
import numpy as np
import pandas as pd

# Anchor sentences: plain-English descriptions of each class, not keywords. Several per class, because a class has
# several shapes ("Minimum Qualifications" and "Who You Are" are both the requirements).
ANCHORS = {
    "required": [
        "These are the qualifications and skills the applicant must have.",
        "Minimum requirements and required experience for this position.",
        "What you need to bring to this role to be considered.",
        "The education, certifications and years of experience we require.",
    ],
    "preferred": [
        "These skills are nice to have but are not required.",
        "Preferred qualifications that would give a candidate an edge.",
        "Bonus points for additional experience beyond the basics.",
        "Desirable extras we would like to see but can do without.",
    ],
    "other": [
        "What you will be doing day to day in this role.",
        "The duties and responsibilities of the position.",
        "About our company, our mission and our culture.",
        "The salary, benefits and perks we offer employees.",
        "How to apply, and our interview and hiring process.",
        "We are an equal opportunity employer and do not discriminate.",
    ],
}
CLASSES = tuple(ANCHORS)
MIN_MARGIN = 0.02  # cosine gap below which the best class is not meaningfully better than the runner-up
UNCLEAR = "unclear"


def bge_encoder(model_dir=None, batch_size=64, show_progress=False):
    """An `encode(texts) -> unit-length embeddings` function backed by the local BGE model. Imported lazily and
    called nowhere in this file, so tests can pass their own encoder and never load a model."""
    from sentence_transformers import SentenceTransformer
    from fetch.paths import PRETRAINED_DIR

    model = SentenceTransformer(str(model_dir or PRETRAINED_DIR / "bge-base-en-v1.5"), device="cpu")
    model.max_seq_length = 512

    def encode(texts):
        return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=show_progress)

    return encode


def anchor_matrix(encode, anchors=ANCHORS):
    """(class order, anchors x dims). Every anchor is kept separate rather than averaged: a class with two shapes
    would average into the middle of neither."""
    texts, owners = [], []
    for name, sentences in anchors.items():
        texts += list(sentences)
        owners += [name] * len(sentences)
    return np.asarray(owners), np.asarray(encode(texts), dtype="float32")


def scores(texts, encode, anchors=ANCHORS):
    """texts x classes cosine similarity: a text's score for a class is its closest anchor in that class."""
    owners, anchor_vecs = anchor_matrix(encode, anchors)
    vecs = np.asarray(encode(list(texts)), dtype="float32")
    sims = vecs @ anchor_vecs.T                                    # both sides unit length, so this is cosine
    return pd.DataFrame({name: sims[:, owners == name].max(axis=1) for name in anchors}, index=list(texts))


def label(texts, encode, anchors=ANCHORS, min_margin=MIN_MARGIN):
    """One row per text: its class scores, the class it joins, and the margin over the runner-up. A text whose
    margin is under `min_margin` is labelled "unclear" instead of guessed at."""
    table = scores(texts, encode, anchors)
    ordered = np.sort(table.to_numpy(), axis=1)
    table["margin"] = ordered[:, -1] - ordered[:, -2]
    table["section"] = np.where(table.margin >= min_margin, table[list(anchors)].idxmax(axis=1), UNCLEAR)
    return table.reset_index(names="text")


def propagate(block_rows, heading_section):
    """The section each block sits in: the nearest heading above it, within its own listing.

    block_rows: a DataFrame with `id`, `kind` and `text`, in reading order (EDA_SDA_3_Lean's blocks table).
    heading_section: text -> section, for heading blocks (from `label`).
    Returns a Series aligned to block_rows; blocks above the first heading come back as NA, because nothing has been
    said about them yet - the notebook labels those from their own text instead of assuming."""
    opened = block_rows.text.where(block_rows.kind.eq("heading")).map(heading_section)
    return opened.groupby(block_rows.id).ffill()

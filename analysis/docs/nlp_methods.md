# NLP methods used in EDA_SDA_3_Lean

`EDA_SDA_3_Lean` is the project's natural language processing notebook, and it contains **no regular expressions**
(decided 2026-10-25). This doc teaches each method it uses: the general method first, the way a textbook would,
then how this project uses it. One part per source file, so a reader can go from a cell in the notebook to the
method behind it without reading the code first.

The order below is the order the notebook applies them.

| Part | Method | File |
|:-:|:--|:--|
| 1 | Markup parsing and sentence segmentation | `src/nlp_text.py` |
| 2 | Sentence embeddings and zero-shot classification by anchors | `src/nlp_sections.py` |
| 3 | Dependency parsing and named entity recognition for extraction | `src/nlp_requirements.py` |
| 4 | Topic models and embedding clusters | `src/nlp_topics.py` |
| 5 | Phrase matching against a controlled vocabulary | *written with `src/nlp_skills.py`* |

## Why not regular expressions

A regular expression is a pattern over **characters**. It is the right tool when the thing you are looking for is
defined by its characters - a postcode, a date in a fixed format, a column of digits. It is the wrong tool when the
thing you are looking for is defined by its **meaning or its grammar**, because then the pattern can only
approximate the real question, and every approximation needs an exception list.

Two examples from this project, both real:

- *"Founded in 1998, Northwind has 25 years of experience serving the midwest."* A character pattern for "a number,
  then `years`, then within 50 characters `experience`" matches this sentence and reads 25 as a job requirement. The
  only fix available to a pattern is a list of sentences to ignore.
- *"Experience with Airflow"*. A character pattern for a heading - "a short line starting with a capital letter" -
  matches this bullet. If the pattern's heading words include *experience*, the bullet becomes a heading, and every
  line after it is attributed to the wrong section.

Both questions are answerable, but not at the level of characters. The first is a question about grammar: *what does
the number modify, and what are the years years of?* The second is a question about meaning: *does this line name a
section, or state a qualification?* Parts 2 and 3 are those two methods.

---

## Part 1: Markup parsing and sentence segmentation

### The method in general

**Markup is structure, and structure is information.** An HTML document is a tree. `<h3>Requirements</h3>` is a
heading element containing text; `<li>` is one item of a list; `<td>` is one cell of a row. A browser uses the tree
to decide what a reader sees on its own line: *block-level* elements (`p`, `div`, `li`, `h1`-`h6`, `tr`, `td`,
`table`, ...) each start a new line, while *inline* elements (`b`, `a`, `span`, `em`) sit inside a line and only
change how it looks.

An HTML **parser** builds that tree and lets you walk it, so you keep the answer to "what kind of thing is this
text?" A pattern-based approach deletes the tags to get at the text, and the structure goes with them.

Three things a parser gets right that tag-deletion does not:

1. **Elements that are not visible text.** The contents of `<script>` and `<style>`, and HTML comments, are in the
   document but are never shown. A parser distinguishes them by type.
2. **Spacing.** `<td>Education</td><td>Bachelor's degree</td>` must not become `EducationBachelor's degree`, and
   `know <b>Python</b>and SQL` must not become `Pythonand`. The parser knows which boundaries are real.
3. **Nesting.** A list inside a paragraph inside a table cell has to come out in reading order, once each.

**Sentence segmentation** is the next question: inside one block of text, where does one statement end and the next
begin? Splitting on full stops is wrong ("Ph.D.", "e.g.", "U.S."), so this is done by a model or by a trained rule
component. spaCy's `sentencizer` is the cheap option: it finds sentence boundaries and nothing else, with no
part-of-speech tagger, parser or entity recogniser loaded.

### How this project uses it

`src/nlp_text.py` turns one description's HTML into **blocks** - one per thing a reader sees on its own line - each
carrying the tag that produced it and a `kind` derived from that tag alone:

- `heading`: `h1`-`h6`, `th`, `dt`, `caption`, `legend`; a paragraph whose entire content is bold or italic (listings
  that style their headings instead of tagging them); and any block whose text ends in a colon.
- `item`: `li`, `dd`, `td` - a bullet or a cell, which is where requirements are usually written.
- `text`: everything else.

**`kind` never depends on the words in the block.** That is the whole point, and `tests/test_nlp_text.py` pins it:
the same sentence is a `heading` inside an `<h3>` and an `item` inside an `<li>`, whatever it says. What a heading
*means* is part 2's question.

Two details worth knowing:

- A `<br>` splits a block in two, because that is what it does on screen.
- A block over 120 words is one unbroken wall of text, so it is replaced by its sentences (the tag and kind carry
  over). A requirement stated in the middle of a paragraph is findable once the paragraph is sentences.

A description with no markup at all is treated as plain text, where the newlines **are** the layout - unlike in
HTML, where a browser collapses them.

---

## Part 2: Sentence embeddings and zero-shot classification by anchors

### The method in general

A **sentence embedding** is a vector of a few hundred numbers representing what a sentence means. The model that
produces it is trained so that sentences with similar meanings get vectors pointing in similar directions, whether
or not they share any words. Similarity is measured by the **cosine** of the angle between two vectors: 1 is the
same direction, 0 is unrelated. When the vectors are scaled to unit length, the cosine is just their dot product,
which is why embeddings are usually normalised before being compared.

This gives a way to classify text **with no training data at all**, called zero-shot classification by similarity:

1. Write a short description of each class in plain language. These are the **anchors**.
2. Embed the anchors and embed the text.
3. The text joins the class whose anchor it is closest to.

What is gained over a keyword list is generalisation. A keyword list can only recognise what is in it. An embedding
comparison places text it has never seen, because the model's knowledge of language is doing the work: `Your
Toolkit` lands near "the qualifications the applicant must have" without anyone having anticipated that heading.

Three practical points:

- **Keep several anchors per class, and do not average them.** A class with two shapes ("Minimum Qualifications"
  and "Who You Are" are both requirements) averages into a point that is near neither. Score a class by its *closest*
  anchor instead.
- **A similarity is not a certainty.** Report the **margin**: the gap between the best class and the runner-up. A
  small margin means the method cannot tell, and saying so is more useful than guessing.
- **Embed distinct strings only.** Encoding is the expensive step, so text that repeats should be encoded once.

### How this project uses it

`src/nlp_sections.py` decides what each part of a listing is about, from the meaning of its heading. The three
classes - `required`, `preferred`, `other` - have four to six anchor sentences each, written as descriptions of the
class rather than as words to look for. The encoder is the project's existing BGE model
(`pretrained/bge-base-en-v1.5`, the same one `ML_Fit` and the app use), so no new model is introduced.

- A heading whose margin is under `MIN_MARGIN` (0.02 cosine) is labelled `unclear` rather than guessed at. An
  `unclear` heading leaves the current section as it was instead of resetting it, so an ambiguous heading costs
  nothing.
- Only **distinct** headings are encoded. Headings repeat across hundreds of thousands of listings, so this is both
  what makes the step affordable and what lets a reader check the result: the notebook prints the most common
  distinct headings beside their labels, and the classifier is judged by eye there rather than trusted.
- A section reaches a bullet by **propagation**: each block takes the section of the nearest heading above it, within
  its own listing. Blocks above the first heading come back as unknown rather than being assumed to be anything.

The encoder is passed in as a function, so the tests use a stub and never load a model: they check the mechanics
(closest anchor wins, the margin guards a close call, a section carries down and does not leak between listings),
which is what a unit test can check. Whether BGE places headings *well* is not a unit-test question, and it is
answered in the notebook instead.

---

## Part 3: Dependency parsing and named entity recognition for extraction

### The method in general

**Part-of-speech tagging** labels each word with its grammatical category (noun, verb, number, adjective).
**Lemmatisation** reduces a word to its dictionary form, so *years*, *year* and *yrs* are one thing.

**Dependency parsing** is the method that matters here. It states, for every word in a sentence, which word it
depends on and how. The result is a tree whose edges are labelled relations. A few that come up constantly:

| Relation | Means | Example |
|:--|:--|:--|
| `nummod` | this number modifies that noun | **5** -> years |
| `quantmod` | this number modifies that number | **3** -> 5 (in "3-5") |
| `prep` / `pobj` | a preposition and its object | years -> **of** -> experience |
| `conj` | coordinated with ("X or Y") | degree -> **experience** |
| `amod` | adjective modifying a noun | **equivalent** -> experience |

Two operations on the tree do most of the work. A token's **subtree** is the token and everything hanging off it -
the whole phrase it heads. A token's **head** is the word it depends on, so walking up one hop widens the scope by
one phrase.

With those, questions that a character window can only approximate become statements about structure:

> *"5+ years of experience with Python"* - the number `5` is a `nummod` of `years`, and `experience` is in the
> subtree of `years` (via `of`). So: a number modifying a noun meaning *year*, where that noun's phrase mentions
> experience.

Apply the same statement to *"Founded in 1998, Northwind has 25 years of history"* and it does not match: the
subtree of `years` mentions *history*, not experience. **No exception list is needed** - the grammar rules it out.

**Named entity recognition** (NER) is a different, statistical method: a model tags spans as `DATE`, `ORG`,
`PERSON` and so on, from the words and their context rather than from a tree. It is useful here as an independent
second reading - if the parse rule finds a requirement and the entity model also tags a date span, two unrelated
methods agree.

A note on vocabulary lists. `{"year", "yr", "yrs"}` and `{"experience", "background", ...}` are lists of words, and
it is fair to ask how that differs from a keyword rule. The difference is what the list is for. These lists say
which *words express a concept*; the parse decides what those words are **doing** in the sentence. A keyword rule
has to carry both jobs in one pattern, and that is where it breaks.

### How this project uses it

`src/nlp_requirements.py` reads two things from requirement text.

**Years.** A number that modifies a noun meaning *year*, where that noun's own subtree or its head's subtree
mentions experience. Within one mention a range takes its lower end ("3-5 years" asks for 3, found through the
`quantmod` edge that hangs the 3 off the 5); across mentions the largest wins, since that is the bar an applicant
has to clear. Above `MAX_YEARS` (20) the number is treated as the company talking about itself.

**Degree.** The level comes from the degree words present, mapped to `associate` < `bachelor` < `master` < `phd`,
with the highest winning. Whether an alternative is accepted is a parse question: the equivalence word
(*equivalent*, *comparable*, *in lieu of*) must be grammatically connected to an alternative - heading its own
phrase ("in lieu of a degree"), coordinated with the degree ("diploma or equivalent"), or modifying an alternative
("or equivalent experience"). Merely sharing a sentence is not enough.

`date_entities` returns the `DATE` spans NER tags that are about years. It produces none of the numbers; the
notebook reports how often it agrees with the parse rule, as a cross-check.

**Degree named, and degree level, are two questions.** Plenty of listings ask for "a degree" and never say which
one, so `mentions_degree` is reported separately from the level. Conflating them would turn every unlevelled ask
into a missing value and undercount the requirement.

Parsing is the expensive step, and it is paid twice over if nothing is done about it: description bullets repeat
heavily across listings, and most blocks of a description cannot state either requirement at all. So two things
happen before the parser is loaded. Each **distinct** block is parsed once and the facts are joined back. And
`may_state_requirement` screens the distinct blocks with the tokenizer alone - a years requirement must contain a
number, which the tokenizer sees without any model, and a degree requirement must contain a degree word - so a
block with neither is never read. The screen is a deliberate superset ("a 401k match" and "5 direct reports" both
pass it) and it makes no decisions: everything it passes is still put to the grammar above.
`tests/test_nlp_requirements.py` pins the only property that matters, which is that nothing a rule could find is
screened out: every positive case in the file passes the screen.

### What it costs

Worth stating plainly, because it is the one case the grammar still gets wrong in the project's favour: a
requirement stated outside any requirements section is still read as a requirement. If a listing's company blurb
says *"a current AWS certification is required"*, that is picked up. The reading is arguably correct - it is a
stated requirement - but it is not what a reader would call a requirements bullet. This is one of the reasons every
measure in this notebook is a tag and an opt-in filter in the app, never a hide (decided 2026-10-13).

---

## Part 4: Topic models and embedding clusters

### The method in general

Parts 2 and 3 answered questions that were known in advance. A **topic model** answers the open one: given a pile
of documents and nothing else, what are they about? Nothing is looked for and no classes are described, so what
comes out is a description of the text rather than a judgement about it.

**TF-IDF** turns a document into a row of numbers, one per term in the vocabulary. The weight of a term rises with
how often it appears in *this* document (TF) and falls with how many documents it appears in anywhere (IDF, the
inverse document frequency). So the words every document uses weigh almost nothing and the words that single a
document out weigh a lot - which is the whole trick, because in job descriptions the common words are nearly all
boilerplate.

**NMF** (non-negative matrix factorisation) then approximates that documents x terms matrix as the product of two
smaller ones: documents x topics and topics x terms. Every number in both is held at or above zero. That
constraint is what makes the result readable: a topic is a non-negative mixture of words, so it can be printed as
a list of words and named, and a document is a non-negative mixture of topics, so it can belong partly to several.
(The comparable method without the constraint, truncated SVD / LSA, produces topics with positive *and negative*
ends, which nobody can name.) The number of topics, k, is a choice, and the honest way to make it is to look at
the reconstruction error as k rises - it always falls, so what matters is where it stops falling steeply - and
then to read the topics at that k and see whether they can be named.

**Clustering embeddings** is the same question asked of meanings instead of words. The documents are encoded with
a sentence model (part 2) and **k-means** groups the vectors: it places k centres, assigns every vector to its
nearest centre, moves each centre to the mean of its members, and repeats. On unit-length vectors, straight-line
distance and cosine distance rank pairs the same way, so this is grouping by meaning.

The two methods are worth running together because **they fail in opposite directions:**

| | TF-IDF -> NMF | Embeddings + k-means |
|:--|:--|:--|
| Works on | words | meanings |
| A topic or cluster is described by | its strongest words | the real documents nearest its centre |
| Gets wrong | the same thing said in different words lands in different topics | a cluster can be real and have no words that name it |
| Membership | partial: a weight on every topic | hard: one cluster each |

When both find the same division of a corpus, the division is in the text and not in the method. **Normalized
mutual information** puts a number on that: 0 means the two labellings are unrelated, 1 means they are the same
division under different names.

### How this project uses it

`src/nlp_topics.py` reads requirement text two ways, in `EDA_SDA_3_Lean` sections 6 and 7. The corpus is every
block section 3 placed in a `required` or `preferred` section, headings excluded.

**The tokenizer is spaCy's, not sklearn's, and that is a method decision rather than a coding one.**
`TfidfVectorizer`'s default tokenizer is a regular expression - `token_pattern`, a character pattern meaning "a run
of two or more word characters" - so it is not used at all here. The corpus is lemmatized by spaCy first and TF-IDF
is handed the token lists with `analyzer` set to a function over them. What that buys:

- `year` and `years`, `pipeline` and `pipelines`, `build` and `built` are one feature each, so a topic is described
  by concepts instead of by spellings.
- Stopwords are dropped by the language model's own judgement (`token.is_stop`), not by a list written here.
- `3`, `+`, `&` and `(` never enter the vocabulary.
- Adjacent pairs of lemmas are features too, because `machine learning` and `year experience` say more than their
  halves. (`ngram_range` stops working once `analyzer` is a function, so the pairs are built in `_features`.)

What it costs: the tagger is reading bullet fragments rather than sentences, and the small model reads a fragment
opening with a gerund as a noun - "Building dashboards" lemmatizes to `building`, not `build`. Some inflections
therefore survive as separate features. It changes which words describe a topic, not which themes are found, and
`tests/test_nlp_topics.py` records it rather than working around it.

**Sampling and weighting.** The same requirement line is written by thousands of employers, so the corpus is the
**distinct** lines. Topics are fitted on a random sample of distinct lines per role bucket (the project's seed,
12345), and prevalence is then weighted by how many blocks each line stands for. A sample of the distinct lines
represents the variety of what gets written; the weights represent the volume. Without the weights a line written
once would count as much as a line written forty thousand times.

**Judging the result.** A topic is reported only if it can be named from its words (section 6c prints them), and a
cluster only if it can be named from the real lines nearest its centre (`nearest_to_centre`, section 7c). Both of
those cells are tagged `data-rows`, so their output is cleared before committing. Section 7d reports the mutual
information between the word topics and the meaning clusters.

**Neither is a label on a listing.** These two sections answer "what is requirement text about, and does that
differ by role bucket?" Nothing downstream consumes a topic number as a feature; the features this notebook hands
on are the years, the degree and the skills.

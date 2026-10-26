"""A job description's HTML turned into the blocks a reader sees, and long blocks split into sentences.

**No regular expressions anywhere in this file**, by design (`EDA_SDA_3_Lean`, decided 2026-10-25). The HTML is
parsed by an HTML parser, not matched by character patterns, and sentences are found by spaCy's sentencizer. The
point is not purity for its own sake: a parser knows that `<li>` is a list item and `<h3>` is a heading, which is
information a pattern over characters throws away and then has to guess back from word lists. Section 3 of the
notebook needs exactly that information, because it decides what a heading *means* from its own words rather than
from a list someone wrote.

What a block is: one thing a reader sees on its own line - a heading, a paragraph, a list item, a table cell. Each
block carries the tag that produced it and a `kind` taken from that tag:

  heading   h1-h6, th, dt, caption, legend; a paragraph whose whole content is bold or italic; any block whose
            text ends in a colon ("Requirements:" written as an ordinary paragraph)
  item      li, dd, td - a bullet or a cell, which is where requirements are usually written
  text      everything else: paragraphs and loose text

`kind` says what a block *is*. What it is *about* (requirements, preferences, the company) is not decided here and
never from a word list - `src/nlp_sections.py` decides it from the heading's own sentence embedding.

Teaching doc: `docs/nlp_methods.md` (part 1).
"""
from collections import namedtuple

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, NavigableString
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)  # short plain text looks like a file name to bs4

Block = namedtuple("Block", "text tag kind")

# Elements a browser lays out on their own line. Each becomes a block; text inside an inline element (<b>, <a>,
# <span>) belongs to the block around it.
BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "caption", "dd", "div", "dl", "dt", "fieldset", "figcaption",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "legend", "li", "main", "nav",
    "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
})
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "th", "dt", "caption", "legend"})
ITEM_TAGS = frozenset({"li", "dd", "td"})
EMPHASIS_TAGS = frozenset({"b", "strong", "em", "i", "u"})
SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})  # never text a reader sees

DEFAULT_MAX_WORDS = 120  # a block longer than this is one unbroken wall of text: split it into sentences


def _is_tag(node):
    return getattr(node, "name", None) is not None


def _chunks(node):
    """The text `node` holds itself, split where a <br> breaks the line. Nested block elements are skipped - each one
    becomes a block of its own, in document order, so a list's items don't collapse into their <ul>."""
    chunks, current = [], []
    for child in node.children:
        name = getattr(child, "name", None)
        if name == "br":
            chunks.append(current)
            current = []
        elif name in SKIP_TAGS or name in BLOCK_TAGS:
            continue                                  # emitted on its own (block) or never shown (script, style)
        elif name is None:
            # Exactly a text node. bs4 gives comments, doctypes and script bodies as NavigableString *subclasses*,
            # and none of those is text a reader sees, so the type has to match exactly rather than by isinstance.
            if type(child) is NavigableString:
                current.append(str(child))
        elif child.find(BLOCK_TAGS) is not None:
            continue                                  # an inline tag wrapping blocks: those blocks are emitted below
        else:
            current.append(child.get_text(" "))       # an ordinary inline tag: <b>Python</b> belongs to this block
    chunks.append(current)
    return [" ".join(" ".join(c).split()) for c in chunks]


def _emphasis_only(node):
    """True when everything the node holds sits inside one emphasis tag: a paragraph in bold is a heading."""
    tags = [c for c in node.children if _is_tag(c) and c.name != "br"]
    loose = [str(c).strip() for c in node.children if not _is_tag(c)]
    return len(tags) == 1 and tags[0].name in EMPHASIS_TAGS and not any(loose)


def _kind(node, text, first_chunk):
    if node.name in HEADING_TAGS:
        return "heading"
    if text.endswith(":"):
        return "heading"                              # "Requirements:" as an ordinary paragraph
    if first_chunk and _emphasis_only(node):
        return "heading"
    if node.name in ITEM_TAGS:
        return "item"
    return "text"


def _walk(node, out):
    name = getattr(node, "name", None)
    if name in SKIP_TAGS:
        return
    if name in BLOCK_TAGS or name == "[document]":
        for i, text in enumerate(_chunks(node)):
            if text:
                out.append(Block(text, name if name != "[document]" else "", _kind(node, text, i == 0)))
    for child in node.children:
        if _is_tag(child):
            _walk(child, out)


def _plain_text_blocks(text):
    lines = (" ".join(line.split()) for line in text.split("\n"))
    return [Block(line, "", "heading" if line.endswith(":") else "text") for line in lines if line]


def blocks(description):
    """One description (HTML, or plain text) -> its blocks in reading order. Plain text with no tags comes back as
    one block per line, since that is what a reader sees."""
    if not isinstance(description, str) or not description.strip():
        return []
    soup = BeautifulSoup(description, "html.parser")
    if soup.find() is None:
        # No markup at all. Some listings arrive as plain text, where the newlines *are* the layout - unlike in
        # HTML, where a browser collapses them - so each line is what a reader sees on its own line.
        return _plain_text_blocks(soup.get_text())
    out = []
    _walk(soup, out)
    return out or _plain_text_blocks(soup.get_text())


def sentencizer():
    """A blank English pipeline that only finds sentence boundaries - the cheapest spaCy pipe there is. Used to split
    a block that is one wall of text; the tagger, parser and NER are not loaded, because nothing here needs them."""
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def split_long(blocks_in, nlp, max_words=DEFAULT_MAX_WORDS):
    """Blocks in, blocks out, with any block of more than `max_words` words replaced by its sentences (same tag and
    kind). A requirement stated inside a wall of text is findable once the wall is sentences."""
    long_ones = [b for b in blocks_in if len(b.text.split()) > max_words]
    if not long_ones:
        return list(blocks_in)
    by_text = {}
    for block, doc in zip(long_ones, nlp.pipe([b.text for b in long_ones])):
        by_text[block.text] = [s.text.strip() for s in doc.sents if s.text.strip()]
    out = []
    for block in blocks_in:
        for piece in by_text.get(block.text, [block.text]):
            out.append(Block(piece, block.tag, block.kind))
    return out


def as_rows(listing_id, blocks_in):
    """Blocks of one listing as (id, ord, tag, kind, text) tuples - the shape EDA_SDA_3_Lean's blocks table holds."""
    return [(listing_id, i, b.tag, b.kind, b.text) for i, b in enumerate(blocks_in)]

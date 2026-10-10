"""The two places Ask the market uses OpenAI (both off unless the backend starts with PREDICTLEADS_MARKET=openai).

1. `parse`: turns a question into a MarketQuery (a strict JSON schema). The model only fills in filters; it never writes an
   answer. Our own code runs the query on our own tables and reports what could not be matched.
2. `research`: an optional, separate step the person chooses for one company. It uses OpenAI's web search, writes a short
   synopsis only from what the search returned, and returns the sources. It is clearly labelled as web information, not
   PredictLeads data. Only the company's name and website are sent.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4.1-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"
TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
TRANSCRIBE_HINT = "A short question about hiring: companies, job titles, technologies such as Snowflake, dbt, Databricks, Python, AWS, and US states and cities."


class MarketAIUnavailable(RuntimeError):
    """No key or the feature is switched off."""


class MarketAIFailure(RuntimeError):
    """The request failed or the answer was not usable. Nothing was kept."""


_STR_LIST = {"type": "array", "items": {"type": "string"}}
PARSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported_reason": {"type": "string"},
        "task": {"type": "string", "enum": ["companies", "technologies", "listings", "trend", "pay", "compare"]},
        "role_family": {"type": "array", "items": {"type": "string", "enum": ["ai_ml", "data", "technical_sales"]}},
        "states": _STR_LIST, "cities": _STR_LIST, "countries": _STR_LIST, "title_words": _STR_LIST, "technologies": _STR_LIST,
        "technology_match": {"type": "string", "enum": ["any", "all"]},
        "seniority": {"type": "array", "items": {"type": "string", "enum": ["junior", "mid_senior", "manager", "director", "head", "vice_president", "c_level", "founder"]}},
        "patterns": {"type": "array", "items": {"type": "string", "enum": ["burst", "sustained", "cooling"]}},
        "window_days": {"type": "integer"},
        "first_time_hiring": {"type": "boolean"},
        "companies": _STR_LIST,
        "only_my_saved": {"type": "boolean"},
        "sort": {"type": "string", "enum": ["new", "active", "surge"]},
        "limit": {"type": "integer"},
    },
    "required": ["supported", "unsupported_reason", "task", "role_family", "states", "cities", "countries", "title_words", "technologies",
                 "technology_match", "seniority", "patterns", "window_days", "first_time_hiring", "companies", "only_my_saved", "sort", "limit"],
}

PARSE_PROMPT = """You turn one question about the job market into a search over a fixed table of job listings. You only fill in filters. You never answer the question and never invent facts.

The table covers about 270,000 openings seen by a crawler from spring to September 2026 in three role families only: ai_ml (AI and machine learning roles), data (analytics, data engineering, data science) and technical_sales (solutions and sales engineering). Each opening has a title, company, city, state, country (almost all United States), seniority, technology tags, and sometimes a stated pay. Dates are when the crawler first saw an opening, not when it was posted. Each company also has a pattern from its own history: burst (opened far more roles than its usual pace), sustained (steady hiring) or cooling (slowing).

Choose a task:
- companies: who is hiring (the default for "who", "which companies").
- technologies: which tools or technologies appear in listings.
- listings: example openings.
- trend: how many new openings per week over time.
- pay: what stated pay looks like.
- compare: side by side numbers for companies the person names (put the names in companies).

Rules:
- Fill in only what the person said. Do not add filters they did not ask for. Leave lists empty and use the defaults (window_days 30, technology_match any, sort new, limit 20) when they said nothing. Leave role_family empty unless they named a kind of role (AI or machine learning, data, or sales / solutions engineering); never list all three.
- Use patterns only when the person used words like these: "hot", "surging", "ramping up", "growing fast" mean ["burst"]; "steady" or "consistent" mean ["sustained"]; "slowing" or "cooling" mean ["cooling"]. Words like "most" or "top" are not a pattern; they just mean the default sort.
- If they name only a city, fill cities and leave states empty. If they name only a state, fill states. Never add a place they did not name.
- A job title they name goes in title_words as one phrase (for example "data engineer" or "solutions engineer").
- "just started hiring", "first time hiring" means first_time_hiring true.
- Use full state names (Texas, not TX). Put a technology exactly as the person wrote it in technologies; if they list several and want any of them, technology_match is any; if they say "and"/"both", use all.
- only_my_saved is true only when they refer to their own saved jobs or companies.
- "recently", "lately", "this month" mean window_days 30; "this week" 7; "last two months" 60.
- Questions about the future or predictions ("will", "next month", "likely to") and requests for advice or opinions ("should I", "best companies") are not supported.
- If the question needs something the table does not have, set supported to false and say in unsupported_reason, in one short sentence, exactly what is missing. Not available: company size, funding, revenue, industry, headquarters, remote or on-site status, why a company is hiring, whether a company will hire the person, forecasts or predictions, jobs outside the three role families, opinions or advice. If the question is not about the job market at all, supported is false.
- When supported is false, still return valid values for every field (use the defaults).
Return only the requested JSON."""

RESEARCH_PROMPT = """Write a short, factual synopsis of one company for a job seeker, using only what the web search returns.
- 3 to 5 sentences: what the company does, who it serves, and anything notable and recent that a source states.
- Put the source citation right after every sentence, not only at the end.
- Use only information that appears in the search results. Do not guess, do not fill gaps from memory, and do not speculate about hiring plans, finances or culture unless a source says so.
- Make sure the results are about the company at the website given. If you cannot find reliable information about that exact company, say so in one sentence.
- Plain text, no markdown, no links, no bullet points. The sources are shown separately."""


def _multipart(fields: Mapping[str, str], filename: str, content_type: str, data: bytes) -> tuple[bytes, str]:
    boundary = "----predictleads" + os.urandom(12).hex()
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode("utf-8"))
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def keep_cited_sentences(text: str, annotations: list[Mapping[str, Any]]) -> tuple[str, list[str]]:
    """Only the sentences that a web citation covers, and the pages those citations point to.

    A citation covers the passage before it: the sentence it follows and any sentences since the previous citation, which is how
    web-search answers are normally cited. Text after the last citation has no source and is dropped, and an answer with no citation
    at all yields nothing, so what is shown can be traced to a page.
    """
    spans = sorted(
        (int(n["start_index"]), int(n["end_index"]), str(n["url"]))
        for n in annotations
        if n.get("type") == "url_citation" and n.get("url") and isinstance(n.get("start_index"), int) and isinstance(n.get("end_index"), int)
    )
    cleaned, cites, position = "", [], 0
    for start, end, url in spans:
        if start < position:
            continue
        cleaned += text[position:start]
        cites.append((len(cleaned), url))
        position = end
    cleaned += text[position:]
    # offsets of each sentence without its leading spaces, so a citation just before a sentence is not counted as inside it
    sentences = [(m.start() + len(m.group()) - len(m.group().lstrip()), m.end(), m.group().strip()) for m in re.finditer(r"[^.!?]+[.!?]+[\"')\]]*|[^.!?]+$", cleaned) if m.group().strip()]
    cited: dict[int, str] = {}
    for index, (start, _, _) in enumerate(sentences):
        url = next((u for at, u in cites if at > start), None)  # the first citation that comes after this sentence begins
        if url is not None:
            cited[index] = url
    kept = [sentences[i][2] for i in sorted(cited)]
    urls = list(dict.fromkeys(cited[i] for i in sorted(cited)))
    return " ".join(kept), urls


def clean_url(url: str) -> str:
    """The page address without tracking tags or fragments."""
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.urlencode([(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if not k.lower().startswith("utm_")])
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def _post(api_key: str, body: Mapping[str, Any], timeout: int) -> Mapping[str, Any]:
    request = urllib.request.Request(
        RESPONSES_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise MarketAIFailure(f"OpenAI request failed with HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MarketAIFailure(f"OpenAI request failed ({type(exc).__name__})") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise MarketAIFailure("OpenAI returned invalid JSON") from None


def _message_parts(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parts: list[Mapping[str, Any]] = []
    for item in result.get("output", []) or []:
        if item.get("type") == "message":
            parts += [c for c in item.get("content", []) or [] if c.get("type") == "output_text"]
    return parts


class OpenAIMarketProvider:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, parse_timeout: int = 30, research_timeout: int = 90):
        if not api_key.strip():
            raise MarketAIUnavailable("OPENAI_API_KEY is not set")
        self._key, self.model = api_key, model
        self._parse_timeout, self._research_timeout = parse_timeout, research_timeout

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "OpenAIMarketProvider":
        values = os.environ if environ is None else environ
        key = str(values.get("OPENAI_API_KEY", "")).strip()
        if not key and environ is None:
            try:
                from dotenv import load_dotenv
            except ImportError as exc:  # pragma: no cover
                raise MarketAIUnavailable("python-dotenv is unavailable; set OPENAI_API_KEY in the environment") from exc
            load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
            key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise MarketAIUnavailable("Set OPENAI_API_KEY in the process environment or project-root .env")
        return cls(key, str(values.get("PREDICTLEADS_MARKET_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL)

    def transcribe(self, audio: bytes, content_type: str) -> str:
        """Speech to text for one short recording. The audio is sent to OpenAI for this one request and is not kept by this app."""
        extension = "mp4" if "mp4" in content_type else "ogg" if "ogg" in content_type else "wav" if "wav" in content_type else "webm"
        model = os.environ.get("PREDICTLEADS_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIBE_MODEL).strip() or DEFAULT_TRANSCRIBE_MODEL
        payload, header = _multipart({"model": model, "response_format": "json", "temperature": "0", "prompt": TRANSCRIBE_HINT},
                                     f"question.{extension}", content_type.split(";")[0] or "audio/webm", audio)
        request = urllib.request.Request(TRANSCRIBE_URL, data=payload, headers={"Authorization": f"Bearer {self._key}", "Content-Type": header}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._parse_timeout * 2) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise MarketAIFailure(f"OpenAI transcription failed with HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise MarketAIFailure(f"OpenAI transcription failed ({type(exc).__name__})") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise MarketAIFailure("OpenAI returned invalid JSON") from None
        text = result.get("text") if isinstance(result, Mapping) else None
        if not isinstance(text, str):
            raise MarketAIFailure("OpenAI did not return a transcript")
        return " ".join(text.split())

    def parse(self, question: str) -> Mapping[str, Any]:
        body = {
            "model": self.model, "temperature": 0,
            "input": [{"role": "system", "content": [{"type": "input_text", "text": PARSE_PROMPT}]},
                      {"role": "user", "content": [{"type": "input_text", "text": question}]}],
            "text": {"format": {"type": "json_schema", "name": "market_query", "strict": True, "schema": PARSE_SCHEMA}},
        }
        parts = _message_parts(_post(self._key, body, self._parse_timeout))
        try:
            parsed = json.loads(parts[0]["text"]) if parts else None
        except (json.JSONDecodeError, KeyError):
            parsed = None
        if not isinstance(parsed, Mapping):
            raise MarketAIFailure("OpenAI did not return a usable search")
        return parsed

    def research(self, company_name: str, domain: str | None) -> dict[str, Any]:
        """A short synopsis from web search results, with the pages it came from. Returns text '' when nothing reliable was found."""
        who = f"{company_name} (website: {domain})" if domain else company_name
        body = {
            "model": self.model,
            "input": [{"role": "system", "content": [{"type": "input_text", "text": RESEARCH_PROMPT}]},
                      {"role": "user", "content": [{"type": "input_text", "text": f"Company: {who}"}]}],
            "tools": [{"type": "web_search_preview", "search_context_size": "low"}],
        }
        parts = _message_parts(_post(self._key, body, self._research_timeout))
        if not parts:
            raise MarketAIFailure("OpenAI did not return a synopsis")
        titles: dict[str, str] = {}
        sentences: list[str] = []
        urls: list[str] = []
        for part in parts:
            notes = [n for n in part.get("annotations", []) or [] if isinstance(n, Mapping)]
            for note in notes:
                if note.get("type") == "url_citation" and note.get("url"):
                    titles.setdefault(clean_url(str(note["url"])), str(note.get("title") or note["url"]))
            kept, kept_urls = keep_cited_sentences(str(part.get("text", "")), notes)
            if kept:
                sentences.append(kept)
                urls += [clean_url(u) for u in kept_urls]
        sources = [{"url": u, "title": titles.get(u, u)} for u in dict.fromkeys(urls)][:6]
        return {"text": clean_synopsis(" ".join(sentences)), "sources": sources}


def clean_synopsis(text: str) -> str:
    """Drop inline citation links and markdown the model sometimes adds, leaving plain sentences."""
    text = re.sub(r"\s*\(\[[^\]]*\]\([^)]*\)\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\(https?://[^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return " ".join(text.split())


class MockMarketProvider:
    """Offline stand-in for tests: give it functions for parse and research."""

    model = "mock"

    def __init__(self, parse: Callable[[str], Mapping[str, Any]] | None = None, research: Callable[[str, str | None], dict[str, Any]] | None = None,
                 transcribe: Callable[[bytes, str], str] | None = None):
        self._parse, self._research, self._transcribe = parse, research, transcribe
        self.parse_calls = 0
        self.research_calls = 0
        self.transcribe_calls = 0

    def transcribe(self, audio: bytes, content_type: str) -> str:
        self.transcribe_calls += 1
        if self._transcribe is None:
            raise MarketAIFailure("no transcriber")
        return self._transcribe(audio, content_type)

    def parse(self, question: str) -> Mapping[str, Any]:
        self.parse_calls += 1
        if self._parse is None:
            raise MarketAIFailure("no parser")
        return self._parse(question)

    def research(self, company_name: str, domain: str | None) -> dict[str, Any]:
        self.research_calls += 1
        if self._research is None:
            raise MarketAIFailure("no research")
        return self._research(company_name, domain)

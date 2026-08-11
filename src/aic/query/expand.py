"""Turn a Vietnamese competition query into per-branch search inputs.

Two problems are solved here:

1. **Language.** Queries arrive in Vietnamese; CLIP and SigLIP text towers are
   trained overwhelmingly on English, so querying them in Vietnamese loses a lot
   of signal. Visual branches get English text; the OCR and ASR branches keep the
   Vietnamese, since the indexed text is itself Vietnamese.

2. **Facets.** A query like *"diễn giả mặc áo đỏ phát biểu tại họp báo ngoài trời"*
   mixes what is *seen* (red shirt, outdoors, trees) with what might be *written on
   screen* (the speaker's name, "HỌP BÁO") and what is *said*. Routing the whole
   sentence into every branch adds noise; splitting it first is what makes fusion
   worth doing.

The LLM path uses Gemini when ``GEMINI_API_KEY`` is set. Without a key everything
degrades to a rule-based split so the system still runs offline — reduced quality,
never a crash, because a dead API during the contest must not stop the tool.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class ExpandedQuery:
    """A query decomposed into the inputs each retrieval branch wants."""

    original: str
    #: English visual descriptions for the image-embedding branch (several phrasings).
    visual_en: list[str] = field(default_factory=list)
    #: Vietnamese terms likely to appear as on-screen text.
    ocr_terms: list[str] = field(default_factory=list)
    #: Vietnamese terms likely to be spoken.
    asr_terms: list[str] = field(default_factory=list)
    #: OpenImages-style class names for the object filter.
    objects: list[str] = field(default_factory=list)
    used_llm: bool = False

    def is_empty(self) -> bool:
        return not (self.visual_en or self.ocr_terms or self.asr_terms)


_PROMPT = """You are helping a video retrieval system for a Vietnamese TV news archive.
Given a Vietnamese search query describing a moment in a video, decompose it.

Return ONLY a JSON object with these keys:
  "visual_en": list of {n} short ENGLISH sentences describing what the frame LOOKS like.
               Vary the phrasing; these query a CLIP/SigLIP image-text model.
  "ocr_terms": list of Vietnamese words/phrases likely shown as on-screen text
               (headlines, tickers, name captions). Empty list if none apply.
  "asr_terms": list of Vietnamese keywords likely SPOKEN in the audio.
  "objects":   list of OpenImages V4 English class names visible in the frame
               (e.g. "Person", "Car", "Microphone"). Empty list if unsure.

Query: {query}"""


def _rule_based(query: str, num_expansions: int) -> ExpandedQuery:
    """Offline fallback: keep Vietnamese for text branches, reuse query for visual.

    Deliberately conservative. Without translation the visual branch is weak, which
    the UI surfaces so the operator knows to lean on the OCR/ASR results.
    """
    cleaned = " ".join(query.strip().split())
    # Quoted spans are near-verbatim on-screen text; the strongest OCR signal available.
    quoted = re.findall(r'"([^"]+)"|“([^”]+)”', cleaned)
    ocr_terms = [a or b for a, b in quoted]

    content_words = [w for w in re.split(r"[\s,.;:!?]+", cleaned) if len(w) > 3]
    return ExpandedQuery(
        original=query,
        visual_en=[cleaned][:num_expansions],
        ocr_terms=ocr_terms or content_words[:8],
        asr_terms=content_words[:8],
        objects=[],
        used_llm=False,
    )


def _parse_llm_json(text: str) -> dict:
    """Extract the JSON object from a model reply that may be fenced or prefixed."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start : end + 1])


def expand_query(
    query: str,
    num_expansions: int = 3,
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
) -> ExpandedQuery:
    """Decompose ``query``, using Gemini when available and rules otherwise."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return _rule_based(query, num_expansions)

    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model, contents=_PROMPT.format(n=num_expansions, query=query)
        )
        payload = _parse_llm_json(response.text or "")
    except Exception:  # noqa: BLE001 - any failure must fall back, not abort
        # Network failure, quota, or malformed reply: fall back rather than abort.
        # Losing decomposition quality mid-contest beats losing the tool entirely.
        return _rule_based(query, num_expansions)

    def _strings(key_name: str) -> list[str]:
        value = payload.get(key_name) or []
        if isinstance(value, str):
            value = [value]
        return [str(v).strip() for v in value if str(v).strip()]

    expanded = ExpandedQuery(
        original=query,
        visual_en=_strings("visual_en")[:num_expansions],
        ocr_terms=_strings("ocr_terms"),
        asr_terms=_strings("asr_terms"),
        objects=_strings("objects"),
        used_llm=True,
    )
    return expanded if not expanded.is_empty() else _rule_based(query, num_expansions)


def expand_trake_query(
    query: str,
    event_descriptions: list[str],
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
) -> tuple[ExpandedQuery, list[ExpandedQuery]]:
    """Expand a TRAKE query plus each event in its sequence.

    Returns the whole-sequence expansion (used to rank candidate *videos*) and one
    expansion per event (used to pick the exact frame for that event during
    alignment).
    """
    overall = expand_query(query, model=model, api_key=api_key)
    per_event = [
        expand_query(f"{query} — khoảnh khắc: {description}", num_expansions=2, model=model, api_key=api_key)
        for description in event_descriptions
    ]
    return overall, per_event

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
    #: Minimum instance count per OpenImages label (e.g. {"Person": 6} for "hơn 5
    #: người") - a frame only counts as a match for that label if the detector
    #: found at least this many instances of it.
    min_object_counts: dict[str, int] = field(default_factory=dict)
    used_llm: bool = False

    def is_empty(self) -> bool:
        return not (self.visual_en or self.ocr_terms or self.asr_terms)


#: Labels in :attr:`ExpandedQuery.min_object_counts` whose count means "at least N"
#: (more instances is a better match, never a worse one). All extracted labels
#: default to this: even a worn-item detail phrased with an exclusivity marker
#: ("chỉ có một người đeo kính" - only one wears glasses) is safer treated as a
#: floor than an exact target, for the same reason as the person count - a
#: detector is far more likely to miss/undercount an instance than invent one,
#: so an "exactly N" filter risks excluding the true frame if it over-detects by
#: even one. ObjectIndex.search_by_target_count (ranked by closeness, not just
#: floor) stays available for a future label where overcounting is the more
#: likely failure mode, but nothing currently routes to it by default. Shared
#: with aic.query.search (which picks the ranking function) and the Streamlit UI
#: (which picks the "≥" vs "≈" display prefix).
AT_LEAST_LABELS = {"person", "glasses", "hat"}

#: Vietnamese count phrasing around "người" (person/people) - the one count-bearing
#: noun common enough across queries to be worth a dedicated, LLM-independent
#: extractor. "hơn/trên/quá N" (more than N) literally means the true count is N+1
#: or more, but the search-side floor is deliberately kept at N, not N+1: a
#: detector undercounting occluded/small people is far more common than it
#: overcounting, so a true 6-person scene can easily register as 5 detected. Using
#: N+1 as the SQL ">=" floor would then wrongly exclude that frame outright, while
#: N still finds it (RRF ranks by count descending regardless, so a real 8-person
#: frame still outranks a miscounted 5-person one - the floor only decides what's
#: considered at all, not the final order).
#:
#: The plain form accepts digits and word-numbers *except* "một" ("one"): found
#: live on "Ba người đang đi bộ..." (three people walking...), where the query's
#: real subject count is later restated in a subset clause with a digit - "có 2
#: người cầm dù" (2 of them hold umbrellas) - and since re.search finds the
#: leftmost match, excluding word-numbers entirely meant the digit-only regex
#: skipped "Ba người" (the real total) and grabbed "2 người" (a subset) instead,
#: extracting a materially wrong count (2 vs the true 3). "một người" stays
#: excluded on its own: it is usually just an article ("một người đàn ông" = "a
#: man"), not a deliberate headcount, so treating it as one would manufacture a
#: min_count=1 filter (a no-op, since almost every person-containing frame
#: already has >=1) out of ordinary phrasing. "hai/ba/bốn..." are not similarly
#: ambiguous - Vietnamese does not use them as articles - so they are safe to
#: treat as real counts.
_VN_NUMBER_WORDS = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
}
_NUMBER_TOKEN = r"(\d+|" + "|".join(_VN_NUMBER_WORDS) + r")"
_PLAIN_NUMBER_TOKEN = r"(\d+|" + "|".join(w for w in _VN_NUMBER_WORDS if w != "một") + r")"

_MORE_THAN_RE = re.compile(rf"(?:hơn|trên|quá)\s*{_NUMBER_TOKEN}\s*người", re.IGNORECASE)
_PLAIN_COUNT_RE = re.compile(rf"{_PLAIN_NUMBER_TOKEN}\s*người", re.IGNORECASE)

#: Temporarily disabled at the user's request: extensive checking (per-frame joint,
#: per-video co-occurrence, broadened hat taxonomy, actual pixel-color
#: verification on detected Hat boxes) never found a real match for query 1's
#: "1 glasses + 3 red hats" detail - only coincidental ones (police uniform caps,
#: conical farmer hats). With this off, only the Person floor is extracted, so a
#: frame no longer needs a lucky Glasses/Hat detection to rank well. Flip back to
#: True to re-enable - the extraction/filtering machinery underneath is unchanged
#: and already tested independently of this toggle.
EXTRACT_ATTRIBUTE_COUNTS = False

#: A person's specific worn item IS a deliberate, discriminative headcount even at
#: N=1 ("chỉ có một người đeo kính" - only one person wears glasses) - unlike a bare
#: "N người" this always names a specific attribute, so it is never just an article.
#: Maps each OpenImages label to the Vietnamese verb phrase that names it; ``.{0,20}``
#: between the count and the verb absorbs "chỉ có", "trong nhóm", etc.
_ATTRIBUTE_PATTERNS = {
    "Glasses": re.compile(rf"{_NUMBER_TOKEN}\s*người.{{0,20}}?(?:đeo|mang)\s*kính", re.IGNORECASE),
    "Hat": re.compile(rf"{_NUMBER_TOKEN}\s*người.{{0,20}}?đội\s*(?:nón|mũ)", re.IGNORECASE),
}


def _number_token_to_int(token: str) -> int:
    return int(token) if token.isdigit() else _VN_NUMBER_WORDS.get(token.lower(), 0)


def _extract_person_count(query: str) -> dict[str, int]:
    """Instance-count constraints the object-detection branch can act on.

    Color attributes ("nón có màu đỏ" - red hat) are deliberately NOT extracted:
    the OpenImages detector this corpus was labeled with has no color output, only
    generic classes like "Hat", so a red-hat constraint can only be captured as an
    (uncolored) hat count here - the color itself has to come from the visual/CLIP
    branch's embedding of "red hat" in visual_en, not from object detection.
    """
    counts: dict[str, int] = {}

    more_than = _MORE_THAN_RE.search(query)
    if more_than:
        n = _number_token_to_int(more_than.group(1))
        if n:
            counts["Person"] = n
    else:
        plain = _PLAIN_COUNT_RE.search(query)
        if plain:
            n = _number_token_to_int(plain.group(1))
            if n:
                counts["Person"] = n

    if EXTRACT_ATTRIBUTE_COUNTS:
        for label, pattern in _ATTRIBUTE_PATTERNS.items():
            match = pattern.search(query)
            if match:
                n = _number_token_to_int(match.group(1))
                if n:
                    counts[label] = n

    return counts


_PROMPT = """You are helping a video retrieval system for a Vietnamese TV news archive.
Given a Vietnamese search query describing a moment in a video, decompose it.

Return ONLY a JSON object with these keys:
  "visual_en": list of {n} short ENGLISH sentences describing what the frame LOOKS like.
               Vary the phrasing; these query a CLIP/SigLIP image-text model.
               Keep every distinguishing visual detail from the query - counts
               ("more than 5 people"), colors ("red hat"), and accessories
               ("one person wearing glasses") - even if the sentence gets longer.
               Dropping these for brevity is the most common way this step loses
               the one detail that would have told the video apart from others.
               Do NOT add a posture, position, or pose that the query does not
               state ("xếp thành hàng" is "lined up / arranged in a row" - it
               says nothing about standing versus sitting versus kneeling, so
               translating it as "standing in a row" invents a detail that can
               wrongly exclude the real video during visual matching).
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
        min_object_counts=_extract_person_count(query),
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
        from google.genai import types

        # A network stall here (seen once as a request that neither completed nor
        # raised) otherwise hangs the whole batch forever - the except below only
        # helps once the call actually errors, so it needs a hard deadline too.
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20_000))
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
        # Extracted directly from the Vietnamese text rather than trusted to the
        # LLM: seen firsthand to drop a "hơn 5 người" (more than 5 people) count
        # from its visual_en translation while still generating plausible-looking
        # prose, so this constraint would silently vanish if left LLM-only.
        min_object_counts=_extract_person_count(query),
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

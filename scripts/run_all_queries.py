"""Batch-run every exam query against the current (partial) index.

Throwaway script for racing a download against a deadline: loads the model
once and searches every query file in one process instead of paying model-load
cost 25 times over via the CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.query.expand import ExpandedQuery, expand_query  # noqa: E402
from aic.query.search import SearchEngine  # noqa: E402
from query_translations import VISUAL_EN  # noqa: E402


def _expand(text: str, query_id: str, config) -> ExpandedQuery:
    """Real Gemini expansion when it works (needed for ``objects``); fall back
    to the hand-checked English translation, never untranslated Vietnamese."""
    expanded = expand_query(
        text, num_expansions=int(config.query.num_expansions), model=str(config.query.llm_model)
    )
    if expanded.used_llm:
        return expanded
    return ExpandedQuery(
        original=text, visual_en=VISUAL_EN[query_id], ocr_terms=expanded.ocr_terms,
        asr_terms=expanded.asr_terms, objects=expanded.objects, used_llm=False,
    )

QUESTIONS_DIR = Path("/Users/dangphuong/AI/AIO2026/HCM/aic2026-Luminary/Exam1/SOTUYEN1-bo-de-thi")
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "derived" / "query_results.json"


def main() -> None:
    config = load_config()
    engine = SearchEngine(config)

    results = {}
    for path in sorted(QUESTIONS_DIR.glob("query-p1-*.txt")):
        query_id = path.stem
        task = query_id.rsplit("-", 1)[-1]
        text = path.read_text(encoding="utf-8").strip()

        expanded = _expand(text, query_id, config)
        result = engine.search(text, top_n=20, expanded=expanded)
        hits = [
            {"video_id": h.video_id, "frame_idx": h.frame_idx, "score": h.score, "branches": h.branches}
            for h in result.hits
        ]
        results[query_id] = {"task": task, "query": text, "hits": hits}
        top = hits[0] if hits else None
        print(f"{query_id:22} [{task:5}] top: {top}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()

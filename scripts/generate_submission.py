"""Best-effort answers for all 25 exam queries, written to the submission folder.

Uses the existing search + policy + writer pipeline end to end. Honesty notes:
- KIS/TRAKE get real value from partial credit (R@20/50/100), so a diversified
  candidate list is worth submitting even under real uncertainty about rank 1.
- QA needs an exact video+frame+answer match to score anything at all. Where
  the actual displayed value was not visually confirmed, the answer text below
  is a placeholder guess, not a read result - flagged as such in FALLBACK_QA.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aic  # noqa: E402,F401
from aic.config import load_config  # noqa: E402
from aic.query.expand import ExpandedQuery, expand_query  # noqa: E402
from aic.query.search import SearchEngine  # noqa: E402
from aic.submit.policy import (  # noqa: E402
    Candidate,
    TrakeCandidate,
    build_kis_answers,
    build_trake_answers,
    build_vqa_answers,
)
from aic.submit.writer import write_submission  # noqa: E402
from aic.tasks.trake import align_events, rank_videos  # noqa: E402
from aic.data.verify import _find_video, probe_video  # noqa: E402
from aic.query.encoder import get_encoder  # noqa: E402
from query_translations import TRAKE_EVENTS, VISUAL_EN  # noqa: E402

QUESTIONS_DIR = Path("/Users/dangphuong/AI/AIO2026/HCM/Exam1/SOTUYEN1-bo-de-thi")
SUBMISSION_DIR = Path("/Users/dangphuong/AI/AIO2026/HCM/Exam1/submission")

# TRAKE stage-2 alignment (dense CPU re-decode + SigLIP re-encode per candidate) can
# take many minutes per query. Set to skip it and keep whatever CSV already exists
# for the TRAKE query - useful for quickly regenerating the other 24 KIS/QA answers.
SKIP_TRAKE = os.environ.get("SKIP_TRAKE") == "1"
ONLY_QUERY = os.environ.get("ONLY_QUERY")

# Unconfirmed placeholder guesses for QA queries where the displayed value was
# never visually located in the current partial corpus. These are NOT reads -
# they will very likely score 0 unless the guess happens to be right.
FALLBACK_QA = {
    "query-p1-3-qa": "2",
    "query-p1-9-qa": "50",
    "query-p1-15-qa": "3",
    # A real-world educated guess (2023 Bao Loc landslide on QL20), not a visual
    # read - moderate plausibility given the description, but unconfirmed.
    "query-p1-17-qa": "Bảo Lộc",
}


def _expand(text: str, query_id: str, config) -> ExpandedQuery:
    """Real Gemini expansion when it works (needed for the ``objects`` field the
    object-detection branch relies on); fall back to the hand-checked English
    translation - never the untranslated Vietnamese - if the LLM call fails.
    """
    expanded = expand_query(
        text, num_expansions=int(config.query.num_expansions), model=str(config.query.llm_model)
    )
    if expanded.used_llm:
        return expanded
    return ExpandedQuery(
        original=text, visual_en=VISUAL_EN[query_id], ocr_terms=expanded.ocr_terms,
        asr_terms=expanded.asr_terms, objects=expanded.objects, used_llm=False,
    )


def to_candidates(hits) -> list[Candidate]:
    return [
        Candidate(video_id=h.video_id, frame_idx=h.frame_idx, score=h.score)
        for h in hits
        if h.frame_idx is not None
    ]


def main() -> None:
    config = load_config()
    engine = SearchEngine(config)
    sub_cfg = config.submission

    written = []
    for path in sorted(QUESTIONS_DIR.glob("query-p1-*.txt")):
        query_id = path.stem
        task = query_id.rsplit("-", 1)[-1]
        text = path.read_text(encoding="utf-8").strip()
        out_path = SUBMISSION_DIR / f"{query_id}.csv"

        if ONLY_QUERY and query_id != ONLY_QUERY:
            continue

        if task == "trake" and SKIP_TRAKE:
            print(f"SKIP {query_id}: SKIP_TRAKE=1, leaving existing CSV in place")
            continue

        if task == "trake":
            per_event = [
                ExpandedQuery(original=d, visual_en=[d]) for d in TRAKE_EVENTS[query_id]
            ]
            n_events = len(per_event)
            # Rank a wide pool on coarse (independent-per-event) hits, then keep only
            # full-coverage videos - stage 2 needs all N events present to align.
            video_candidates = rank_videos(engine, per_event, max_videos=100, per_event_top_n=300)
            full_coverage = [vc for vc in video_candidates if len(vc.event_hits) == n_events]

            # Stage 2: dense re-decode + order-enforcing DP. Coarse hits are picked
            # independently per event, so they are almost never chronologically
            # ordered on their own - build_trake_answers drops any non-increasing
            # frame tuple, so skipping this stage yields an empty submission.
            space = config.active_space
            encoder = get_encoder(space.model_id, int(space.dim))
            trake_candidates: list[TrakeCandidate] = []
            # SigLIP2-so400m encoding measured at ~29s per 32-image chunk on CPU
            # (no faster MPS path available - see encoder.py's encode_images
            # docstring). The config defaults (window_seconds=2.0, frame_step=1)
            # give ~120 frames/event = 4 chunks/event = 12 chunks/candidate =
            # ~6min/candidate, which is why this stage kept stalling out under
            # time pressure. Halving the window and doubling the step here (an
            # override local to this one-off batch run, not the shared config)
            # drops that to ~15 frames/event = 1 chunk/event = ~90s/candidate,
            # still comfortably inside TRAKE's acceptance window.
            trake_window_seconds = 1.0
            trake_frame_step = 2
            print(f"  {query_id}: {len(full_coverage)} full-coverage candidates, aligning top 3")
            for vc in full_coverage[:3]:
                print(f"  {query_id}: aligning {vc.video_id} coarse={[h.frame_idx for h in vc.event_hits]}")
                video_path = _find_video(Path(config.paths.raw.videos), vc.video_id)
                if video_path is None:
                    continue
                try:
                    fps = probe_video(video_path).fps or 25.0
                except Exception:  # noqa: BLE001
                    fps = 25.0
                frames, scores = align_events(
                    video_path,
                    vc.event_hits,
                    per_event,
                    encoder,
                    fps=fps,
                    window_seconds=trake_window_seconds,
                    frame_step=trake_frame_step,
                )
                if not frames or any(a >= b for a, b in zip(frames, frames[1:])):
                    continue
                trake_candidates.append(
                    TrakeCandidate(
                        video_id=vc.video_id, frame_ids=frames, score=vc.score, per_event_scores=scores
                    )
                )
                print(f"  {query_id}: {vc.video_id} coarse={[h.frame_idx for h in vc.event_hits]} -> aligned={frames}")
            rows = build_trake_answers(
                trake_candidates,
                max_answers=int(sub_cfg.max_answers),
                diversify_head=int(sub_cfg.diversify_head),
                jitter=tuple(sub_cfg.trake_jitter),
            )
            write_submission(rows, out_path, "trake", expected_events=n_events)

        else:
            expanded = _expand(text, query_id, config)
            result = engine.search(text, top_n=int(sub_cfg.max_answers) * 2, expanded=expanded)
            candidates = to_candidates(result.hits)
            if not candidates:
                print(f"SKIP {query_id}: no candidates at all")
                continue

            if task == "kis":
                rows = build_kis_answers(
                    candidates,
                    max_answers=int(sub_cfg.max_answers),
                    diversify_head=int(sub_cfg.diversify_head),
                    frames_per_shot=int(sub_cfg.frames_per_shot),
                    frame_spread=int(sub_cfg.frame_spread),
                )
                write_submission(rows, out_path, "kis")
            else:  # qa
                rows = build_vqa_answers(
                    candidates,
                    fallback_answer=FALLBACK_QA.get(query_id, ""),
                    max_answers=int(sub_cfg.max_answers),
                    diversify_head=int(sub_cfg.diversify_head),
                    frames_per_shot=int(sub_cfg.frames_per_shot),
                    frame_spread=int(sub_cfg.frame_spread),
                )
                write_submission(rows, out_path, "vqa")

        written.append(out_path)
        print(f"wrote {out_path} ({task}, {len(rows)} rows)")

    print(f"\n{len(written)} files written to {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()

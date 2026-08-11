"""Q&A: locate the moment, then read the answer out of it.

Q&A scores 1.0 only when video, frame, *and* answer are all correct, so this stage
runs after retrieval has produced candidate moments and adds the third component.

Two sources feed the answer:

* the frames themselves, sent to a vision-language model, and
* the ASR transcript around that moment, which for Vietnamese news often *states*
  the answer outright (counts, names, places) far more reliably than it can be read
  off pixels.

Without an API key the module still returns candidates with an empty answer rather
than failing; the operator types the answer in the UI, which is exactly the
"manual inspection" the task description anticipates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from aic.submit.policy import Candidate

_PROMPT = """You are answering a question about a specific moment in a Vietnamese TV video.

Question: {question}

{transcript_block}
Look at the frames provided (consecutive moments from the video) and answer the
question as briefly as possible - a number, a name, a colour, or a short phrase.
Do not explain. If the answer is a number, give the digits.
Answer in Vietnamese unless the question is in English."""


@dataclass
class VQAResult:
    answer: str
    confidence: float
    source: str  # "vlm" | "manual" | "none"


def gather_context_frames(
    frame_paths: list[Path], max_frames: int = 6, repo_root: Path | None = None
) -> list[Image.Image]:
    """Load up to ``max_frames`` keyframes as context for the model."""
    root = repo_root or Path.cwd()
    images: list[Image.Image] = []
    for path in frame_paths[:max_frames]:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = root / resolved
        if resolved.exists():
            images.append(Image.open(resolved).convert("RGB"))
    return images


def transcript_around(
    text_index, video_id: str, start_frame: int, end_frame: int, fps: float = 25.0
) -> str:
    """Pull ASR text covering a frame range, to ground the model in what was said."""
    if text_index is None:
        return ""
    start_time = max(0.0, (start_frame / fps) - 5.0)
    end_time = (end_frame / fps) + 5.0
    rows = text_index.conn.execute(
        "SELECT text FROM segments WHERE kind='asr' AND video_id=? "
        "AND (start_time IS NULL OR start_time <= ?) AND (end_time IS NULL OR end_time >= ?) "
        "ORDER BY start_time LIMIT 20",
        (video_id, end_time, start_time),
    ).fetchall()
    return " ".join(row["text"] for row in rows).strip()


def answer_question(
    question: str,
    frames: list[Image.Image],
    transcript: str = "",
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
) -> VQAResult:
    """Ask a VLM the question over the given frames."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not frames:
        return VQAResult(answer="", confidence=0.0, source="none")

    transcript_block = (
        f"Transcript of the audio around this moment:\n\"{transcript}\"\n\n" if transcript else ""
    )
    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=[_PROMPT.format(question=question, transcript_block=transcript_block), *frames],
        )
        answer = (response.text or "").strip().strip('".')
    except Exception:  # noqa: BLE001 - fall back to manual entry, never abort
        return VQAResult(answer="", confidence=0.0, source="none")

    # Long replies mean the model explained instead of answering; keep the first
    # line so the submitted answer stays comparable to a short ground truth.
    answer = answer.splitlines()[0].strip() if answer else ""
    return VQAResult(answer=answer, confidence=1.0 if answer else 0.0, source="vlm")


def solve_vqa(
    engine,
    query: str,
    question: str,
    top_moments: int = 5,
    config=None,
) -> list[Candidate]:
    """Retrieve moments for ``query`` and attach an answer to each."""
    result = engine.search(query, top_n=50)
    if not result.hits:
        return []

    model = str(config.query.llm_model) if config else "gemini-2.0-flash"
    fps_guess = 25.0

    # Group hits by video so each answered moment is a distinct place in the corpus
    # rather than five near-identical frames from one shot.
    by_video: dict[str, list] = {}
    for hit in result.hits:
        by_video.setdefault(hit.video_id, []).append(hit)

    candidates: list[Candidate] = []
    for video_id, hits in list(by_video.items())[:top_moments]:
        best = hits[0]
        if best.frame_idx is None:
            continue
        images = gather_context_frames([h.path for h in hits[:6]])
        transcript = transcript_around(
            engine.text, video_id, best.frame_idx, hits[-1].frame_idx or best.frame_idx, fps_guess
        )
        answer = answer_question(question, images, transcript, model=model)
        candidates.append(
            Candidate(
                video_id=video_id,
                frame_idx=best.frame_idx,
                score=best.score,
                answer=answer.answer or None,
            )
        )
    return candidates

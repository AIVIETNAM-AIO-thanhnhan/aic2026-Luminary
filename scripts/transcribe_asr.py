"""Transcribe every locally-available raw video with faster-whisper.

Only 143 of the corpus's 873 videos still have their .mp4 on disk (the rest were
deleted after keyframe/embedding extraction to save space), so this covers that
subset only - not the full corpus. Benchmarked at ~36x realtime on CPU (int8,
"small" model), so the full ~14.7 hours of available audio takes well under an
hour, unlike OCR (~62h estimated, declined) or dense TRAKE alignment.

Resumable like scripts/embed_siglip.py: one parquet per video under
data/derived/asr/segments/, skipped on rerun if already present, so an
interruption loses at most the one video in flight.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from faster_whisper import WhisperModel

import aic  # noqa: F401
from aic.config import load_config
from aic.data.verify import probe_video

VIDEOS_DIR = Path("data_exam1/raw/videos/video")
SEGMENTS_DIR = Path("data_exam1/derived/asr/segments")
OUT_PARQUET = Path("data_exam1/derived/asr/asr.parquet")
MODEL_SIZE = "small"


def transcribe_one(model: WhisperModel, video_path: Path) -> pd.DataFrame:
    video_id = video_path.stem
    try:
        fps = probe_video(video_path).fps or 25.0
    except Exception:  # noqa: BLE001
        fps = 25.0

    segments, _info = model.transcribe(str(video_path), language="vi", vad_filter=True)
    rows = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        rows.append(
            {
                "video_id": video_id,
                "start_frame": round(seg.start * fps),
                "end_frame": round(seg.end * fps),
                "start_time": seg.start,
                "end_time": seg.end,
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    load_config()
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    pending = [v for v in videos if not (SEGMENTS_DIR / f"{v.stem}.parquet").exists()]
    print(f"{len(videos)} videos available, {len(pending)} pending transcription")

    if pending:
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        for i, video_path in enumerate(pending, start=1):
            t0 = time.time()
            df = transcribe_one(model, video_path)
            df.to_parquet(SEGMENTS_DIR / f"{video_path.stem}.parquet")
            print(f"[{i}/{len(pending)}] {video_path.stem}: {len(df)} segments in {time.time() - t0:.1f}s")

    all_dfs = [pd.read_parquet(p) for p in sorted(SEGMENTS_DIR.glob("*.parquet"))]
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    combined.to_parquet(OUT_PARQUET)
    print(f"wrote {OUT_PARQUET}: {len(combined):,} segments from {len(all_dfs)} videos")


if __name__ == "__main__":
    main()

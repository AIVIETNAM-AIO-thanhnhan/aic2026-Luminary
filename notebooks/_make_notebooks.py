"""Generate the Colab notebooks from the cell definitions below.

Notebooks are generated rather than hand-edited so their setup boilerplate stays
identical and reviewable in one place. Run ``python notebooks/_make_notebooks.py``
after changing anything here.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

REPO_URL = "https://github.com/YOUR_ORG/new_aic2026.git"  # set once, then commit

SETUP = f"""# --- Colab setup -------------------------------------------------------------
# Runtime > Change runtime type > T4 GPU before running.
!nvidia-smi -L
!git clone -q {REPO_URL} /content/aic || (cd /content/aic && git pull -q)
%cd /content/aic
!pip install -q pandas pyarrow pillow tqdm
import sys; sys.path.insert(0, "/content/aic/src")"""

DRIVE = """# --- Mount Drive -------------------------------------------------------------
# Keyframes go in, artifacts come out. Keeping both on Drive means an interrupted
# runtime resumes instead of restarting from zero.
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
DATA = Path('/content/drive/MyDrive/aic2026')
KEYFRAMES = DATA / 'raw/keyframes'
DERIVED   = DATA / 'derived'
DERIVED.mkdir(parents=True, exist_ok=True)
print('keyframes:', KEYFRAMES, KEYFRAMES.exists())"""


def notebook(cells: list[tuple[str, str]]) -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": body.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, body in cells
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_siglip.ipynb": [
        ("markdown", """# 01 · Encode keyframes with SigLIP2

Produces `derived/embeddings/siglip/*.npy` — the visual index for search.

Row *i* of the output is the frame with `gid == i` in `catalog.parquet`. Shards are
written as they finish, so a disconnected runtime resumes rather than restarting.

**Before running:** build the catalog locally (`aic build-catalog`) and upload
`catalog.parquet` to Drive, so gid ordering is identical on both machines."""),
        ("code", SETUP),
        ("code", DRIVE),
        ("code", """!pip install -q transformers torch --upgrade

import torch
from transformers import AutoModel, AutoImageProcessor

MODEL_ID = "google/siglip2-so400m-patch14-384"
DIM = 1152

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
print(model.config.name_or_path, "on", device)"""),
        ("code", """import numpy as np

def encode(images):
    \"\"\"PIL images -> L2-normalized float32 vectors.\"\"\"
    batch = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(pixel_values=batch["pixel_values"].half())
    vectors = features.float().cpu().numpy()
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)"""),
        ("code", """import pandas as pd
from tqdm.auto import tqdm
from aic.index.embed import embed_catalog

catalog = pd.read_parquet(DERIVED / 'catalog.parquet')
print(f'{len(catalog):,} frames across {catalog.video_id.nunique()} videos')

bar = tqdm(total=len(catalog))
embed_catalog(
    catalog=catalog,
    encode_fn=encode,
    output_dir=DERIVED / 'embeddings/siglip',
    dim=DIM,
    batch_size=64,
    shard_size=20_000,
    repo_root=DATA,                    # catalog paths are relative to the data root
    progress=lambda done, total: bar.update(done - bar.n),
)
bar.close()"""),
        ("markdown", """## Download

Copy `derived/embeddings/siglip/` to `data/derived/embeddings/siglip/` locally, then:

```bash
# set embedding.active: siglip in configs/default.yaml first
aic build-index
```"""),
    ],
    "02_asr.ipynb": [
        ("markdown", """# 02 · Vietnamese ASR with faster-whisper

Produces `derived/asr/asr.parquet`.

High value for this corpus: the videos are TV news, so the narration frequently
states outright what a query is describing. Segment times are converted to frame
numbers here, because the search layer joins ASR spans onto catalog frames by
`frame_idx`."""),
        ("code", SETUP),
        ("code", DRIVE),
        ("code", """!pip install -q faster-whisper

from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda", compute_type="float16")"""),
        ("code", """import subprocess, json
from pathlib import Path

VIDEOS = DATA / 'raw/videos'

def video_fps(path):
    \"\"\"Frame rate, needed to convert segment times into submittable frame numbers.\"\"\"
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate', '-of', 'json', str(path)],
        capture_output=True, text=True).stdout
    rate = json.loads(out)['streams'][0]['r_frame_rate']
    num, _, den = rate.partition('/')
    return float(num) / float(den or 1)"""),
        ("code", """import pandas as pd
from tqdm.auto import tqdm

OUT = DERIVED / 'asr'; OUT.mkdir(parents=True, exist_ok=True)
videos = sorted(VIDEOS.glob('*.mp4'))
rows = []

for video in tqdm(videos):
    shard = OUT / f'{video.stem}.parquet'
    if shard.exists():           # resume
        rows.append(pd.read_parquet(shard)); continue

    fps = video_fps(video)
    segments, _ = model.transcribe(str(video), language='vi', vad_filter=True)
    records = [{
        'video_id':   video.stem,
        'text':       s.text.strip(),
        'start_time': s.start,
        'end_time':   s.end,
        'start_frame': int(s.start * fps),
        'end_frame':   int(s.end * fps),
    } for s in segments if s.text.strip()]

    table = pd.DataFrame(records)
    table.to_parquet(shard, index=False)
    rows.append(table)

asr = pd.concat(rows, ignore_index=True)
asr.to_parquet(OUT / 'asr.parquet', index=False)
print(f'{len(asr):,} segments')
asr.head()"""),
        ("markdown", """## Download

Copy to `data/derived/asr/asr.parquet`, then `aic build-text`."""),
    ],
    "03_ocr.ipynb": [
        ("markdown", """# 03 · Vietnamese OCR on keyframes

Produces `derived/ocr/ocr.parquet`.

Likely the **highest-precision branch** for this corpus. Vietnamese news carries
headlines, tickers and name captions, and matching that text is close to an exact
match on the event — far sharper than visual similarity.

Only frames whose text is long enough to be meaningful are kept; single stray
glyphs are noise that would dilute BM25."""),
        ("code", SETUP),
        ("code", DRIVE),
        ("code", """!pip install -q paddlepaddle-gpu paddleocr

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)"""),
        ("code", """import pandas as pd, numpy as np
from PIL import Image
from tqdm.auto import tqdm

MIN_CHARS = 4          # discard stray glyphs
MIN_CONFIDENCE = 0.60

catalog = pd.read_parquet(DERIVED / 'catalog.parquet')
OUT = DERIVED / 'ocr'; OUT.mkdir(parents=True, exist_ok=True)

def read_frame(path):
    image = Image.open(DATA / path).convert('RGB')
    result = ocr.ocr(np.array(image), cls=True)
    if not result or not result[0]:
        return ''
    parts = [text for _, (text, score) in result[0]
             if score >= MIN_CONFIDENCE and len(text.strip()) >= MIN_CHARS]
    return ' '.join(parts).strip()"""),
        ("code", """rows = []
for video_id, group in tqdm(catalog.groupby('video_id')):
    shard = OUT / f'{video_id}.parquet'
    if shard.exists():
        rows.append(pd.read_parquet(shard)); continue

    records = []
    for item in group.itertuples():
        text = read_frame(item.path)
        if not text:
            continue
        records.append({
            'video_id':    video_id,
            'text':        text,
            # One keyframe's text is attributed to that exact frame; the search
            # layer widens it to the surrounding shot when joining.
            'start_frame': int(item.frame_idx),
            'end_frame':   int(item.frame_idx),
            'start_time':  float(item.pts_time) if pd.notna(item.pts_time) else None,
            'end_time':    float(item.pts_time) if pd.notna(item.pts_time) else None,
        })

    table = pd.DataFrame(records)
    table.to_parquet(shard, index=False)
    rows.append(table)

ocr_table = pd.concat(rows, ignore_index=True)
ocr_table.to_parquet(OUT / 'ocr.parquet', index=False)
print(f'{len(ocr_table):,} frames with text')
ocr_table.head()"""),
        ("markdown", """## Download

Copy to `data/derived/ocr/ocr.parquet`, then `aic build-text`."""),
    ],
    "04_transnet.ipynb": [
        ("markdown", """# 04 · Shot detection with TransNet V2

Produces extra keyframes tagged `source=transnet`.

The organizers' I-frames are a *compression* artifact, not a semantic one: short
shots can fall between them entirely. TransNet V2 finds real shot boundaries, and
taking start/middle/end of each shot raises recall on brief events.

Frame numbers come straight from the model's boundary indices, so they are already
in the units we submit — no conversion, no rounding."""),
        ("code", SETUP),
        ("code", DRIVE),
        ("code", """!pip install -q tensorflow ffmpeg-python
!git clone -q https://github.com/soCzech/TransNetV2.git /content/TransNetV2
import sys; sys.path.insert(0, '/content/TransNetV2/inference')

from transnetv2 import TransNetV2
transnet = TransNetV2(model_dir='/content/TransNetV2/inference/transnetv2-weights')"""),
        ("code", """import subprocess, numpy as np, pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

VIDEOS = DATA / 'raw/videos'
OUT_FRAMES = DATA / 'raw/keyframes_transnet'; OUT_FRAMES.mkdir(parents=True, exist_ok=True)

def shot_frames(video_path):
    \"\"\"Return (start, middle, end) frame numbers for every detected shot.\"\"\"
    _, single, _ = transnet.predict_video(str(video_path))
    scenes = transnet.predictions_to_scenes(single)
    return [(int(s), int((s + e) // 2), int(e)) for s, e in scenes]"""),
        ("code", """rows = []
for video in tqdm(sorted(VIDEOS.glob('*.mp4'))):
    target = OUT_FRAMES / video.stem
    if target.exists():
        continue
    target.mkdir(parents=True, exist_ok=True)

    for shot_index, (start, middle, end) in enumerate(shot_frames(video)):
        for label, frame_no in (('s', start), ('m', middle), ('e', end)):
            name = f'{shot_index:05d}{label}_{frame_no}.jpg'
            subprocess.run(
                ['ffmpeg', '-v', 'error', '-i', str(video),
                 '-vf', f'select=eq(n\\\\,{frame_no})', '-vsync', '0',
                 '-frames:v', '1', str(target / name)], check=False)
            rows.append({'video_id': video.stem, 'frame_idx': frame_no,
                         'shot': shot_index, 'position': label})

pd.DataFrame(rows).to_parquet(DERIVED / 'transnet_frames.parquet', index=False)
print(f'{len(rows):,} keyframes from shot boundaries')"""),
        ("markdown", """## Merge locally

Copy `keyframes_transnet/` into `data/raw/`, then rebuild the catalog including
both sources and re-run the SigLIP notebook over the enlarged catalog:

```bash
aic build-catalog --source transnet   # then merge with the btc_iframe catalog
```

Note the ordering constraint: **the catalog must be final before embeddings are
computed**, since `gid` is the embedding row index."""),
    ],
}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        path = HERE / name
        path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

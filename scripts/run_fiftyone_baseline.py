"""Adapted from AIC_2026_Baseline_v1.ipynb (the organizers' fiftyone baseline).

Original notebook assumes a single Windows lot directory (D:\\AICBaseline\\...);
this adapts it to our actual merged macOS data layout (multiple lots under one
tree, with inconsistent zip-wrapper nesting) and skips the optional object-
detection step for speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fiftyone as fo  # noqa: E402
import fiftyone.brain as fob  # noqa: E402
import numpy as np  # noqa: E402

import aic  # noqa: E402,F401
from aic.config import load_config  # noqa: E402

config = load_config()
keyframes_root = config.raw_path("keyframes")
clip_dir = config.raw_path("clip_features")

print(f"loading images from {keyframes_root} ...")
dataset = fo.Dataset.from_images_dir(str(keyframes_root), recursive=True, name="aic2026")
print(f"loaded {len(dataset)} samples")

for sample in dataset:
    path = Path(sample.filepath)
    sample["video"] = path.parent.name
    sample["frameid"] = path.stem
    sample.save()

video_keyframe_dict: dict[str, list[str]] = {}
for sample in dataset:
    video_keyframe_dict.setdefault(sample["video"], []).append(sample["frameid"])
for video in video_keyframe_dict:
    video_keyframe_dict[video] = sorted(video_keyframe_dict[video])

print("loading clip embeddings ...")
embedding_dict: dict[str, dict[str, np.ndarray]] = {}
for video, keyframes in video_keyframe_dict.items():
    npy_path = clip_dir / f"{video}.npy"
    if not npy_path.exists():
        continue
    vectors = np.load(npy_path)
    embedding_dict[video] = {kf: vectors[i] for i, kf in enumerate(keyframes) if i < len(vectors)}

clip_embeddings = []
keep_ids = []
for sample in dataset:
    per_video = embedding_dict.get(sample["video"], {})
    vector = per_video.get(sample["frameid"])
    if vector is not None:
        clip_embeddings.append(vector)
        keep_ids.append(sample.id)

print(f"{len(clip_embeddings)}/{len(dataset)} samples have a matching embedding")
view = dataset.select(keep_ids)

fob.compute_similarity(
    view,
    model="clip-vit-base32-torch",
    embeddings=np.stack(clip_embeddings).astype(np.float32),
    brain_key="img_sim",
)

session = fo.launch_app(dataset, port=5151, address="0.0.0.0", auto=False)
print("fiftyone app launched on :5151")
session.wait(-1)

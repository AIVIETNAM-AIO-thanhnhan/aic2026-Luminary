"""Encode every catalogued keyframe with SigLIP2, locally, on Apple's GPU (MPS).

The organizers' README expects this to run on Colab GPU via notebooks/01_siglip.ipynb,
but re-uploading a 100+ GB corpus over a slow connection is a non-starter. This machine
has an Apple GPU; torch's MPS backend runs the same model just as well locally.

Uses aic.index.embed.embed_catalog for resumable, shard-by-shard writing - a multi-hour
job on a laptop that might sleep or get interrupted needs to pick back up, not restart.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

import aic  # noqa: E402,F401
from aic.config import REPO_ROOT, load_config  # noqa: E402
from aic.data.catalog import load_catalog  # noqa: E402
from aic.index.embed import embed_catalog, pending_shards, plan_shards  # noqa: E402

MODEL_ID = "google/siglip2-so400m-patch14-384"
DIM = 1152
DEVICE = "mps"
BATCH_SIZE = 32
SHARD_SIZE = 5_000  # ~11 min/shard at ~7.3 img/s - keeps interruption cost low


def main() -> None:
    config = load_config()
    catalog = load_catalog(config.catalog_path)
    output_dir = REPO_ROOT / "data" / "derived" / "embeddings" / "siglip"

    shards = plan_shards(len(catalog), SHARD_SIZE, output_dir)
    todo = pending_shards(shards)
    print(f"{len(catalog)} frames, {len(shards)} shards, {len(todo)} remaining")

    print(f"loading {MODEL_ID} onto {DEVICE} ...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def encode_fn(images: list) -> np.ndarray:
        batch = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.get_image_features(**batch)
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out
        return pooled.cpu().numpy().astype(np.float32)

    start = time.time()
    seen = 0

    def progress(done: int, total: int) -> None:
        nonlocal seen
        seen = done
        elapsed = time.time() - start
        rate = seen / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else float("inf")
        print(f"  {done:,}/{total:,} ({rate:.1f} img/s, ~{remaining / 60:.0f} min left)")

    embed_catalog(
        catalog,
        encode_fn,
        output_dir,
        dim=DIM,
        batch_size=BATCH_SIZE,
        shard_size=SHARD_SIZE,
        repo_root=REPO_ROOT,
        progress=progress,
    )
    print("done")


if __name__ == "__main__":
    main()

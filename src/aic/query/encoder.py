"""Encode query text into the active image-embedding space.

Only the *text tower* runs locally. Images were encoded on Colab GPU; a query is a
handful of short strings per search, which any CPU handles in well under a second.
That asymmetry is the whole reason the hybrid split works without a local GPU.

The encoder must match the space the index was built in — a CLIP ViT-B/32 index
queried with SigLIP text vectors returns confident nonsense, so the dimension is
checked on load.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np


class EncoderUnavailableError(RuntimeError):
    """Raised when no local text tower can be loaded."""


@dataclass
class TextEncoder:
    model_id: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class TransformersTextEncoder(TextEncoder):
    """CPU text tower via transformers. Works for both CLIP and SigLIP model ids."""

    def __init__(self, model_id: str, dim: int) -> None:
        super().__init__(model_id=model_id, dim=dim)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EncoderUnavailableError(
                "Install the query encoder: pip install 'aic2026[onnx]' torch --index-url "
                "https://download.pytorch.org/whl/cpu"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).eval()
        # Threads are the only knob that matters here; leave one core for the UI.
        torch.set_num_threads(max(1, (torch.get_num_threads() or 2) - 1))

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        batch = self._tokenizer(
            texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            features = self._model.get_text_features(**batch)
            features = _project(features, self._model, "text_projection")
        vectors = features.cpu().numpy().astype(np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.model_id} produced {vectors.shape[1]}-d text vectors but the index "
                f"is {self.dim}-d. embedding.active in configs/default.yaml does not match "
                "the model that built the index."
            )
        return _unit(vectors)

    def encode_images(self, images: list) -> np.ndarray:
        """Encode PIL images with the matching image tower.

        Only used by TRAKE alignment, which scores a few hundred densely extracted
        frames per query on CPU. Toggling this model to Apple's MPS backend and
        back was tried to speed this up, but reproducibly hung inside
        MPSStream::synchronize (observed via a live `sample` stack trace: stuck
        indefinitely at the exact same native frame) - staying on CPU is slower
        but does not have that failure mode. Chunking still bounds peak memory.
        The processor is loaded lazily so text-only searches never pay for it.
        """
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        if not hasattr(self, "_processor"):
            from transformers import AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), 32):
                batch = self._processor(images=images[start : start + 32], return_tensors="pt")
                features = self._model.get_image_features(**batch)
                features = _project(features, self._model, "visual_projection")
                chunks.append(features.cpu().numpy().astype(np.float32))
        return _unit(np.concatenate(chunks, axis=0))


def _project(features, model, projection_attr: str):
    """Recover a plain embedding tensor across transformers API versions and architectures.

    Some ``transformers`` releases changed ``get_text_features`` to return the
    raw pre-projection pooled output instead of the final CLIP embedding, while
    ``get_image_features`` on the same release already returns the projected
    embedding in ``pooler_output`` - the two are not symmetric. Comparing the
    pooled width against the projection layer's expected input width (rather
    than just checking whether ``pooler_output`` exists) tells which case this
    is: only project when the width still matches the *un*-projected side.

    SigLIP has no separate projection head at all (its pooling attention head
    already outputs the joint embedding directly), so ``projection_attr`` may
    not exist on the model - that is not an error, just nothing left to do.
    """
    if not hasattr(features, "pooler_output"):
        return features
    pooled = features.pooler_output
    projection = getattr(model, projection_attr, None)
    if projection is not None and pooled.shape[-1] == projection.in_features:
        return projection(pooled)
    return pooled


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@functools.lru_cache(maxsize=2)
def get_encoder(model_id: str, dim: int) -> TextEncoder:
    """Load and cache the text tower. Cached because loading dominates query cost."""
    return TransformersTextEncoder(model_id=model_id, dim=dim)


def encode_texts(texts: list[str], model_id: str, dim: int) -> np.ndarray:
    return get_encoder(model_id, dim).encode(texts)

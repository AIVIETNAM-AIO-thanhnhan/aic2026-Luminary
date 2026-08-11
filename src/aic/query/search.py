"""Search orchestrator: expand the query, run every branch, fuse the rankings.

This is the single entry point the UI, the CLI, and the task modules all call.
Branches degrade independently — a missing text index or an unavailable encoder
disables that branch and records why, rather than failing the search. During a
timed contest a partial result now beats a complete result after a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aic.config import Config
from aic.index import text_index as text_index_mod
from aic.index import vector_index as vector_mod
from aic.query.encoder import EncoderUnavailableError, encode_texts
from aic.query.expand import ExpandedQuery, expand_query
from aic.query.fusion import BranchResult, FusedHit, reciprocal_rank_fusion


@dataclass
class SearchHit:
    """A fused result, resolved back to something submittable and viewable."""

    gid: int
    video_id: str
    frame_idx: int | None
    pts_time: float | None
    path: str
    score: float
    branches: list[str] = field(default_factory=list)
    #: OCR/ASR text that caused a text-branch match, shown in the UI as evidence.
    evidence: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    hits: list[SearchHit]
    expanded: ExpandedQuery
    #: Branch name -> why it contributed nothing, for the UI's diagnostics panel.
    disabled_branches: dict[str, str] = field(default_factory=dict)

    @property
    def active_branches(self) -> list[str]:
        return sorted({b for hit in self.hits for b in hit.branches})


class SearchEngine:
    """Holds the loaded index, catalog, and text database across queries."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._catalog: pd.DataFrame | None = None
        self._index = None
        self._text: text_index_mod.TextIndex | None = None
        self._frame_lookup: dict[str, np.ndarray] | None = None

    # -- lazy resources -------------------------------------------------------------

    @property
    def catalog(self) -> pd.DataFrame:
        if self._catalog is None:
            from aic.data.catalog import load_catalog

            self._catalog = load_catalog(self.config.catalog_path)
        return self._catalog

    @property
    def index(self):
        if self._index is None:
            self._index = vector_mod.load_index(self.config.index_path)
        return self._index

    @property
    def text(self) -> text_index_mod.TextIndex | None:
        if self._text is None:
            db_path = Path(self.config.derived_path("text_db"))
            if not db_path.exists():
                return None
            self._text = text_index_mod.TextIndex(db_path)
        return self._text

    # -- branch: visual -------------------------------------------------------------

    def _visual_branch(self, expanded: ExpandedQuery, top_k: int) -> tuple[BranchResult | None, str | None]:
        queries = expanded.visual_en or [expanded.original]
        space = self.config.active_space
        try:
            vectors = encode_texts(list(queries), space.model_id, int(space.dim))
        except EncoderUnavailableError as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the search
            return None, f"text encoder failed: {exc}"

        try:
            pooled = vector_mod.search_pooled(self.index, vectors, top_k=top_k)
        except (FileNotFoundError, ImportError) as exc:
            return None, str(exc)

        return BranchResult(
            name="visual",
            gids=[gid for gid, _ in pooled],
            scores=[score for _, score in pooled],
        ), None

    # -- branch: text (OCR / ASR) ---------------------------------------------------

    def _frames_for_video(self, video_id: str) -> pd.DataFrame:
        if self._frame_lookup is None:
            self._frame_lookup = {}
        return self.catalog[self.catalog["video_id"] == video_id]

    def _text_branch(
        self, kind: str, terms: list[str], top_k: int
    ) -> tuple[BranchResult | None, str | None, dict[int, str]]:
        """Map text-segment hits onto the catalogued frames covering that time span.

        OCR/ASR results are time spans, not frames, so each hit is attributed to the
        catalog frames whose ``frame_idx`` falls inside the span. When frame bounds
        are absent the whole video's frames are used, which is weak but still a
        useful video-level vote inside RRF.
        """
        if not terms:
            return None, None, {}
        index = self.text
        if index is None:
            return None, "no text index built (run the OCR/ASR notebooks, then `aic build-text`)", {}

        hits = index.search(terms, kind=kind, limit=top_k)
        if not hits:
            return None, None, {}

        ordered_gids: list[int] = []
        evidence: dict[int, str] = {}
        seen: set[int] = set()
        for hit in hits:
            frames = self._frames_for_video(hit.video_id)
            if frames.empty:
                continue
            if hit.start_frame is not None and hit.end_frame is not None:
                mask = frames["frame_idx"].between(hit.start_frame, hit.end_frame)
                matched = frames[mask]
                if matched.empty:
                    # Span fell between sparse keyframes: take the nearest one so the
                    # video still gets a vote at roughly the right moment.
                    target = hit.mid_frame
                    if target is not None and frames["frame_idx"].notna().any():
                        nearest = (frames["frame_idx"] - target).abs().idxmin()
                        matched = frames.loc[[nearest]]
            else:
                matched = frames

            for gid in matched["gid"].tolist():
                gid = int(gid)
                if gid not in seen:
                    seen.add(gid)
                    ordered_gids.append(gid)
                evidence.setdefault(gid, hit.text)

        return BranchResult(name=kind, gids=ordered_gids), None, evidence

    # -- public API -----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_n: int = 200,
        expanded: ExpandedQuery | None = None,
        video_filter: list[str] | None = None,
    ) -> SearchResult:
        """Run all branches for ``query`` and return the fused ranking."""
        if expanded is None:
            expanded = expand_query(
                query,
                num_expansions=int(self.config.query.num_expansions),
                model=str(self.config.query.llm_model),
            )

        per_branch = int(self.config.fusion.candidates_per_branch)
        branches: list[BranchResult] = []
        disabled: dict[str, str] = {}
        evidence: dict[int, list[str]] = {}

        visual, reason = self._visual_branch(expanded, per_branch)
        if visual:
            branches.append(visual)
        elif reason:
            disabled["visual"] = reason

        for kind, terms in (("ocr", expanded.ocr_terms), ("asr", expanded.asr_terms)):
            branch, reason, branch_evidence = self._text_branch(kind, terms, per_branch)
            if branch:
                branches.append(branch)
                for gid, text in branch_evidence.items():
                    evidence.setdefault(gid, []).append(f"[{kind}] {text}")
            elif reason:
                disabled[kind] = reason

        if not branches:
            return SearchResult(hits=[], expanded=expanded, disabled_branches=disabled)

        fused = reciprocal_rank_fusion(
            branches,
            weights=self.config.fusion.weights.as_dict(),
            k=int(self.config.fusion.rrf_k),
            top_n=None,
        )
        return SearchResult(
            hits=self._resolve(fused, evidence, top_n, video_filter),
            expanded=expanded,
            disabled_branches=disabled,
        )

    def _resolve(
        self,
        fused: list[FusedHit],
        evidence: dict[int, list[str]],
        top_n: int,
        video_filter: list[str] | None,
    ) -> list[SearchHit]:
        """Attach catalog fields to fused gids, filtering after fusion, not before."""
        if not fused:
            return []
        catalog = self.catalog.set_index("gid")
        allowed = set(video_filter) if video_filter else None

        hits: list[SearchHit] = []
        for item in fused:
            if item.gid not in catalog.index:
                continue
            row = catalog.loc[item.gid]
            if allowed is not None and row["video_id"] not in allowed:
                continue
            frame_idx = row["frame_idx"]
            hits.append(
                SearchHit(
                    gid=item.gid,
                    video_id=str(row["video_id"]),
                    frame_idx=int(frame_idx) if pd.notna(frame_idx) else None,
                    pts_time=float(row["pts_time"]) if pd.notna(row["pts_time"]) else None,
                    path=str(row["path"]),
                    score=item.score,
                    branches=item.branches,
                    evidence=evidence.get(item.gid, []),
                )
            )
            if len(hits) >= top_n:
                break
        return hits

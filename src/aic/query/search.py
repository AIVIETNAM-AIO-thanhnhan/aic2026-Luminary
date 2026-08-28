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
from aic.index import object_index as object_index_mod
from aic.index import text_index as text_index_mod
from aic.index import vector_index as vector_mod
from aic.query.encoder import EncoderUnavailableError, encode_texts
from aic.query.expand import AT_LEAST_LABELS, ExpandedQuery, expand_query
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
        self._objects: object_index_mod.ObjectIndex | None = None
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

    @property
    def objects(self) -> object_index_mod.ObjectIndex | None:
        if self._objects is None:
            db_path = Path(self.config.derived_path("objects_db"))
            if not db_path.exists():
                return None
            self._objects = object_index_mod.ObjectIndex(db_path)
        return self._objects

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
        if index.count(kind=kind) == 0:
            # The FTS5 schema exists (so `index` is not None) but is empty for this
            # kind - e.g. OCR was explicitly skipped this session, or ASR was never
            # transcribed at all. Left unchecked, this looks identical to "the terms
            # just didn't match anything" everywhere the UI reports it, silently
            # implying the branch ran when it never had any data to search.
            return None, f"{kind} index has 0 segments - no {kind.upper()} data has been indexed yet", {}

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

    # -- branch: objects -------------------------------------------------------------

    def _object_branch(
        self, labels: list[str], top_k: int
    ) -> tuple[BranchResult | None, str | None, dict[int, str]]:
        """Frames whose detections include one of the query's expected object classes.

        Detections are already per-frame, so unlike OCR/ASR this needs no
        time-span-to-frame mapping - a hit's gid is used directly.

        Each label is ranked on its own and merged by *rank*, not raw detector
        confidence: "Person" sits in ~40% of all frames while "Lion" sits in six,
        so a frame's top Lion match would never survive a shared confidence
        ranking against the flood of high-confidence Person hits. Rank-based
        merging is exactly what RRF already does for OCR vs. ASR, so it is reused
        here rather than inventing a second scale-correction scheme.
        """
        if not labels:
            return None, None, {}
        index = self.objects
        if index is None:
            return None, "no object index built (run `aic build-objects`)", {}

        per_label: list[BranchResult] = []
        evidence: dict[int, str] = {}
        for label in labels:
            if not index.is_discriminative(label):
                continue
            hits = index.search([label], limit=top_k)
            if not hits:
                continue
            per_label.append(BranchResult(name=label, gids=[h.gid for h in hits]))
            for hit in hits:
                evidence.setdefault(hit.gid, hit.label)

        if not per_label:
            return None, None, {}

        merged = reciprocal_rank_fusion(per_label, k=60, top_n=top_k)
        return BranchResult(name="object", gids=[hit.gid for hit in merged]), None, evidence

    def _object_count_branch(
        self, min_counts: dict[str, int], top_k: int
    ) -> tuple[BranchResult | None, str | None, dict[int, str]]:
        """Frames meeting a "more than N X" / "exactly N X" instance-count constraint.

        Separate from :meth:`_object_branch` because the two need opposite
        filters: a plain presence query on "Person" is useless (discriminative
        check rejects it, correctly - it is in ~40% of frames), but a *count*
        query on "Person" is exactly the case where that commonness stops
        mattering, since requiring six-plus simultaneous instances is itself
        a strong, rare signal that the discriminative check would otherwise
        block outright.
        """
        if not min_counts:
            return None, None, {}
        index = self.objects
        if index is None:
            return None, "no object index built (run `aic build-objects`)", {}

        # Deliberately NOT capped at top_k like every other branch: search_by_min_count
        # ranks by raw instance count, so a fixed small cap only ever admits the
        # highest-count frames - e.g. with candidates_per_branch=500, any frame with
        # exactly 5 people (satisfying "more than 5") was silently excluded whenever
        # 500+ frames elsewhere had 6, 10, or 20+ people, since it never ranked into
        # the cap at all and so contributed nothing to that frame's fused score. The
        # actual admission decision now happens post-fusion in
        # _filter_by_object_counts, so this just needs to cover every frame that
        # could plausibly qualify - a few thousand rows is cheap either way.
        UNCAPPED_LIMIT = 10_000

        per_label: list[BranchResult] = []
        evidence: dict[int, str] = {}
        for label, count in min_counts.items():
            is_at_least = label.strip().lower() in AT_LEAST_LABELS
            if is_at_least:
                hits = index.search_by_min_count(label, count, limit=UNCAPPED_LIMIT)
                # search_by_min_count orders by raw count descending, which is the
                # right rank for evidence/display but the wrong one to feed RRF here:
                # "more than 5 people" is satisfied just as fully by 6 as by 60, so
                # ranking by count turns this branch into a pure biggest-crowd
                # finder, burying a frame that is actually the right scene but has
                # a modest headcount under every unrelated frame with a bigger one.
                # Re-sorting by gid removes that correlation, so every frame at or
                # above the floor gets roughly the same RRF contribution and the
                # visual/OCR/ASR branches are left to do the actual discriminating.
                hits = sorted(hits, key=lambda h: h.gid)
            else:
                hits = index.search_by_target_count(label, count, limit=UNCAPPED_LIMIT)
            if not hits:
                continue
            per_label.append(BranchResult(name=label, gids=[h.gid for h in hits]))
            for hit in hits:
                evidence.setdefault(hit.gid, f"{int(hit.score)}x {hit.label}")

        if not per_label:
            return None, None, {}

        merged = reciprocal_rank_fusion(per_label, k=60, top_n=top_k)
        return BranchResult(name="object_count", gids=[hit.gid for hit in merged]), None, evidence

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

        obj_branch, obj_reason, obj_evidence = self._object_branch(expanded.objects, per_branch)
        if obj_branch:
            branches.append(obj_branch)
            for gid, label in obj_evidence.items():
                evidence.setdefault(gid, []).append(f"[object] {label}")
        elif obj_reason:
            disabled["object"] = obj_reason

        # Not fed into `branches`/RRF, unlike every branch above: an "at least N"
        # constraint is binary (satisfied or not), so *any* order this branch could
        # rank its own qualifying frames in - even the current gid-order tiebreak,
        # tried after count-magnitude order proved biased toward big crowds - is an
        # arbitrary proxy standing in for actual relevance. _filter_by_object_counts
        # below is the real enforcement (drops anything that fails the constraint
        # after fusion); this call now exists only to surface *why* a surviving
        # frame passed, as evidence text in the UI, and to report the diagnostic
        # reason when the index is missing entirely.
        _, count_reason, count_evidence = self._object_count_branch(expanded.min_object_counts, per_branch)
        for gid, label in count_evidence.items():
            evidence.setdefault(gid, []).append(f"[count] {label}")
        if count_reason:
            disabled["object_count"] = count_reason

        if not branches:
            return SearchResult(hits=[], expanded=expanded, disabled_branches=disabled)

        fused = reciprocal_rank_fusion(
            branches,
            weights=self.config.fusion.weights.as_dict(),
            k=int(self.config.fusion.rrf_k),
            top_n=None,
        )
        fused = self._filter_by_object_counts(fused, expanded.min_object_counts)
        return SearchResult(
            hits=self._resolve(fused, evidence, top_n, video_filter),
            expanded=expanded,
            disabled_branches=disabled,
        )

    def _filter_by_object_counts(
        self, fused: list[FusedHit], min_counts: dict[str, int]
    ) -> list[FusedHit]:
        """Drop fused hits that don't actually satisfy a count constraint.

        The object_count branch only contributes RRF *votes* - a frame can still
        rank highly on visual/OCR/ASR alone with zero matching detections, which
        is how a single-person interview clip ends up in results for a "more
        than 5 people" query. This is a real hard filter instead, applied after
        fusion so it sees every branch's candidates, not just object_count's own.

        Tries every constraint together first, then drops the *lowest-priority*
        one at a time until something matches - some combinations of these
        constraints have zero matches anywhere in the corpus (confirmed for
        Person+Glasses+Hat here), and returning nothing is worse than
        satisfying fewer constraints. Priority is insertion order, which is
        always Person first (the query's primary countable subject, when
        present) followed by worn-item attributes - trying same-size
        combinations in an arbitrary order (e.g. alphabetical) risks keeping
        "Glasses" and dropping "Person" entirely, which defeats the point of a
        "more than N people" filter far more than dropping a secondary detail
        does.
        """
        index = self.objects
        if not min_counts or index is None:
            return fused
        counts_by_label = {label: index.counts_by_gid(label) for label in min_counts}

        items = list(min_counts.items())
        for size in range(len(items), 0, -1):
            subset = items[:size]
            filtered = [
                hit
                for hit in fused
                if all(counts_by_label[label].get(hit.gid, 0) >= n for label, n in subset)
            ]
            if filtered:
                return filtered
        return fused

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

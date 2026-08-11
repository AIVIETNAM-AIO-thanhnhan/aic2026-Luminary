"""Score the current system against the self-labelled dev set.

This is the number to watch when changing anything in retrieval: swapping the
embedding space, retuning fusion weights, widening the frame spread. It reports
per-query and mean Final Score using the same code path as the real submission
(search -> policy -> answers), so an improvement here is an improvement in the
thing that actually gets graded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aic.config import Config, load_config
from aic.eval.devset import DevQuery, load_devset
from aic.eval.metric import (
    TOP_K_THRESHOLDS,
    KISAnswer,
    KISGroundTruth,
    TRAKEAnswer,
    TRAKEGroundTruth,
    VQAAnswer,
    VQAGroundTruth,
    final_score,
    kis_r_score,
    r_at_k,
    trake_r_score,
    vqa_r_score,
)
from aic.query.search import SearchEngine
from aic.submit.policy import (
    Candidate,
    build_kis_answers,
    build_trake_answers,
    build_vqa_answers,
)


@dataclass
class QueryReport:
    query_id: str
    task: str
    score: float
    r_at: dict[int, float]
    n_answers: int
    note: str = ""


def _candidates_from_hits(hits, limit: int = 60) -> list[Candidate]:
    return [
        Candidate(video_id=h.video_id, frame_idx=h.frame_idx, score=h.score)
        for h in hits[:limit]
        if h.frame_idx is not None
    ]


def evaluate_query(engine: SearchEngine, dev: DevQuery, config: Config) -> QueryReport:
    submission = config.submission
    note = ""

    if dev.task == "trake":
        from aic.tasks.trake import solve_trake

        try:
            candidates = solve_trake(
                engine, dev.query, dev.events, config.raw_path("videos"), config
            )
        except Exception as exc:  # noqa: BLE001 - report and score zero, keep going
            return QueryReport(dev.id, dev.task, 0.0, {k: 0.0 for k in TOP_K_THRESHOLDS}, 0, str(exc))

        rows = build_trake_answers(
            candidates,
            max_answers=int(submission.max_answers),
            diversify_head=int(submission.diversify_head),
            jitter=tuple(submission.trake_jitter),
        )
        assert isinstance(dev.truth, TRAKEGroundTruth)
        scores = [trake_r_score(TRAKEAnswer(v, f), dev.truth) for v, f in rows]

    else:
        result = engine.search(dev.query, top_n=200)
        candidates = _candidates_from_hits(result.hits)
        if result.disabled_branches:
            note = "; ".join(f"{k}: {v}" for k, v in result.disabled_branches.items())

        if dev.task == "kis":
            rows = build_kis_answers(
                candidates,
                max_answers=int(submission.max_answers),
                diversify_head=int(submission.diversify_head),
                frames_per_shot=int(submission.frames_per_shot),
                frame_spread=int(submission.frame_spread),
            )
            assert isinstance(dev.truth, KISGroundTruth)
            scores = [kis_r_score(KISAnswer(v, f), dev.truth) for v, f in rows]
        else:
            from aic.tasks.vqa import solve_vqa

            vqa_candidates = solve_vqa(engine, dev.query, dev.question or dev.query, config=config)
            rows = build_vqa_answers(
                vqa_candidates or candidates,
                max_answers=int(submission.max_answers),
                diversify_head=int(submission.diversify_head),
                frames_per_shot=int(submission.frames_per_shot),
                frame_spread=int(submission.frame_spread),
            )
            assert isinstance(dev.truth, VQAGroundTruth)
            scores = [vqa_r_score(VQAAnswer(v, f, a), dev.truth) for v, f, a in rows]

    return QueryReport(
        query_id=dev.id,
        task=dev.task,
        score=final_score(scores),
        r_at={k: r_at_k(scores, k) for k in TOP_K_THRESHOLDS},
        n_answers=len(scores),
        note=note,
    )


def run_evaluation(devset_path: Path | None = None, config: Config | None = None) -> list[QueryReport]:
    config = config or load_config()
    engine = SearchEngine(config)
    return [evaluate_query(engine, dev, config) for dev in load_devset(devset_path or Path("configs/devset.jsonl"))]


def format_report(reports: list[QueryReport]) -> str:
    if not reports:
        return "no dev queries evaluated"

    header = f"{'query':<16}{'task':<8}{'R@1':>7}{'R@5':>7}{'R@20':>7}{'R@50':>7}{'R@100':>7}{'final':>8}"
    lines = [header, "-" * len(header)]
    for report in reports:
        lines.append(
            f"{report.query_id:<16}{report.task:<8}"
            + "".join(f"{report.r_at[k]:>7.2f}" for k in TOP_K_THRESHOLDS)
            + f"{report.score:>8.3f}"
        )

    lines.append("-" * len(header))
    overall = sum(r.score for r in reports) / len(reports)
    lines.append(f"{'MEAN':<24}" + " " * 35 + f"{overall:>8.3f}")

    by_task: dict[str, list[float]] = {}
    for report in reports:
        by_task.setdefault(report.task, []).append(report.score)
    for task, scores in sorted(by_task.items()):
        lines.append(f"  {task:<10} n={len(scores):<3} mean={sum(scores) / len(scores):.3f}")

    notes = [f"  {r.query_id}: {r.note}" for r in reports if r.note]
    if notes:
        lines.append("\nDegraded branches / errors:")
        lines.extend(notes)
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    print(format_report(run_evaluation()))


if __name__ == "__main__":  # pragma: no cover
    main()

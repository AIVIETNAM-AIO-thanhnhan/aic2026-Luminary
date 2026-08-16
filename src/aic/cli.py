"""Command line interface.

The offline half of the workflow lives here — fetching data, building the catalog
and indexes, verifying frame alignment, evaluating. Interactive searching happens
in the Streamlit app, since KIS realistically needs a human scanning a keyframe grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic.config import load_config


def _cmd_fetch(args: argparse.Namespace) -> int:
    from aic.data.download import fetch, read_manifest

    config = load_config(args.config)
    entries = read_manifest(Path(args.manifest))
    destinations = {kind: config.raw_path(kind) for kind in {e.kind for e in entries}}
    for path in destinations.values():
        path.mkdir(parents=True, exist_ok=True)
    fetch(entries, destinations, extract=not args.no_extract, keep_archives=args.keep_archives)
    return 0


def _cmd_build_catalog(args: argparse.Namespace) -> int:
    from aic.config import REPO_ROOT
    from aic.data.catalog import build_catalog, catalog_summary, save_catalog

    config = load_config(args.config)
    catalog = build_catalog(
        keyframes_dir=config.raw_path("keyframes"),
        map_dir=config.raw_path("metadata"),
        source=args.source,
        repo_root=REPO_ROOT,
    )
    save_catalog(catalog, config.catalog_path)

    summary = catalog_summary(catalog)
    print(f"wrote {config.catalog_path}")
    print(f"  frames : {summary['frames']:,}")
    print(f"  videos : {summary['videos']:,}")
    print(f"  sources: {summary['sources']}")
    if summary["missing_frame_idx"]:
        print(
            f"  WARNING: {summary['missing_frame_idx']:,} frames have no frame_idx "
            f"(videos: {summary['videos_missing_map'][:5]}...). These cannot be submitted."
        )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from aic.config import REPO_ROOT
    from aic.data.catalog import load_catalog
    from aic.data.verify import verify_frame_alignment

    config = load_config(args.config)
    report = verify_frame_alignment(
        catalog=load_catalog(config.catalog_path),
        videos_dir=config.raw_path("videos"),
        sample_size=args.sample_size or int(config.verify.sample_size),
        mae_threshold=float(config.verify.mae_threshold),
        offsets_to_try=tuple(config.verify.offsets_to_try),
        repo_root=REPO_ROOT,
    )
    print(report.summary())
    if args.verbose:
        for sample in report.samples:
            print(
                f"  {sample.video_id} frame {sample.frame_idx}: "
                f"best offset {sample.best_offset} (MAE {sample.best_mae:.2f})"
            )
    return 0 if report.ok else 1


def _cmd_build_index(args: argparse.Namespace) -> int:
    from aic.data.catalog import load_catalog
    from aic.index.vector_index import build_index, load_embeddings, save_index

    config = load_config(args.config)
    catalog = load_catalog(config.catalog_path)
    space = config.active_space

    embeddings = load_embeddings(Path(space.source), expected_rows=len(catalog))
    index = build_index(embeddings, expected_dim=int(space.dim))
    save_index(index, config.index_path)
    print(f"wrote {config.index_path} ({index.ntotal:,} vectors, dim {embeddings.shape[1]})")
    return 0


def _cmd_build_text(args: argparse.Namespace) -> int:
    import pandas as pd

    from aic.index.text_index import TextIndex

    config = load_config(args.config)
    with TextIndex(config.derived_path("text_db")) as index:
        for kind, key in (("ocr", "ocr"), ("asr", "asr")):
            path = config.derived_path(key)
            if not Path(path).exists():
                print(f"skip {kind}: {path} not found (run notebooks/0{2 if kind == 'asr' else 3}_*)")
                continue
            table = pd.read_parquet(path)
            index.clear(kind)
            rows = [{"kind": kind, **row} for row in table.to_dict("records")]
            print(f"{kind}: indexed {index.add_segments(rows):,} segments")
        print(f"wrote {config.derived_path('text_db')} ({index.count():,} segments total)")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from aic.query.search import SearchEngine

    config = load_config(args.config)
    result = SearchEngine(config).search(args.query, top_n=args.top_n)

    if result.disabled_branches:
        for name, reason in result.disabled_branches.items():
            print(f"[branch disabled] {name}: {reason}", file=sys.stderr)
    if not result.hits:
        print("no results", file=sys.stderr)
        return 1

    print(f"{'rank':<6}{'video':<14}{'frame':>8}{'score':>9}  branches")
    for rank, hit in enumerate(result.hits, start=1):
        print(
            f"{rank:<6}{hit.video_id:<14}{hit.frame_idx if hit.frame_idx is not None else '-':>8}"
            f"{hit.score:>9.4f}  {','.join(hit.branches)}"
        )
    return 0


def _cmd_package(args: argparse.Namespace) -> int:
    """Zip the CSVs in a folder into the submission/ layout the graders require."""
    from aic.submit.package import SUBMISSION_DIRNAME, build_package, inspect_csv
    from aic.submit.writer import SubmissionError

    config = load_config(args.config)
    source = Path(args.source) if args.source else config.submissions_dir
    csv_files = sorted(p for p in Path(source).rglob("*.csv"))
    if not csv_files:
        print(f"error: no .csv files under {source}", file=sys.stderr)
        return 1

    from aic.submit.package import parse_query_filename

    answers: dict[str, tuple[str, list]] = {}
    problems: list[str] = []
    for csv_file in csv_files:
        try:
            query_id, task = parse_query_filename(csv_file.name)
        except SubmissionError as exc:
            problems.append(f"{csv_file.name}: {exc}")
            continue

        found = inspect_csv(csv_file, task)
        if found:
            problems.extend(f"{csv_file.name}: {p}" for p in found)
            continue

        import csv as csv_mod

        with open(csv_file, encoding="utf-8", newline="") as handle:
            cells = [row for row in csv_mod.reader(handle) if row]
        if task == "trake":
            rows = [(r[0].strip(), tuple(int(c) for c in r[1:])) for r in cells]
        elif task == "vqa":
            rows = [(r[0].strip(), int(r[1]), r[2] if len(r) > 2 else "") for r in cells]
        else:
            rows = [(r[0].strip(), int(r[1])) for r in cells]
        answers[query_id] = (task, rows)

    if problems:
        print("Refusing to package - fix these first:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    report = build_package(answers, Path(args.output), work_dir=Path(args.output).parent)
    print(report.summary())
    print(f"\nArchive layout verified: every CSV sits in {SUBMISSION_DIRNAME}/")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Validate an existing submission zip without rebuilding it."""
    from aic.submit.package import verify_package
    from aic.submit.writer import SubmissionError

    try:
        names = verify_package(Path(args.zipfile))
    except (SubmissionError, FileNotFoundError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"VALID: {Path(args.zipfile).name} contains {len(names)} CSV file(s)")
    for name in sorted(names):
        print(f"  {name}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from aic.eval.run import format_report, run_evaluation

    config = load_config(args.config)
    print(format_report(run_evaluation(Path(args.devset) if args.devset else None, config)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aic", description="AIC 2026 video retrieval toolkit")
    parser.add_argument("--config", help="path to a config YAML (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download batch data from a manifest CSV")
    fetch.add_argument("manifest", help="CSV exported from the organizers' spreadsheet")
    fetch.add_argument("--no-extract", action="store_true")
    fetch.add_argument("--keep-archives", action="store_true")
    fetch.set_defaults(func=_cmd_fetch)

    catalog = sub.add_parser("build-catalog", help="scan keyframes into catalog.parquet")
    catalog.add_argument("--source", default="btc_iframe")
    catalog.set_defaults(func=_cmd_build_catalog)

    verify = sub.add_parser("verify", help="check frame_idx against the source videos")
    verify.add_argument("--sample-size", type=int)
    verify.add_argument("--verbose", "-v", action="store_true")
    verify.set_defaults(func=_cmd_verify)

    index = sub.add_parser("build-index", help="build the FAISS index for the active space")
    index.set_defaults(func=_cmd_build_index)

    text = sub.add_parser("build-text", help="load OCR/ASR parquet into the FTS5 database")
    text.set_defaults(func=_cmd_build_text)

    search = sub.add_parser("search", help="run one query and print the ranking")
    search.add_argument("query")
    search.add_argument("--top-n", type=int, default=20)
    search.set_defaults(func=_cmd_search)

    package = sub.add_parser("package", help="zip result CSVs into the submission/ layout")
    package.add_argument("--source", help="folder of result CSVs (default: submissions/)")
    package.add_argument("-o", "--output", default="submission.zip", help="output .zip path")
    package.set_defaults(func=_cmd_package)

    check = sub.add_parser("check", help="validate an existing submission zip")
    check.add_argument("zipfile")
    check.set_defaults(func=_cmd_check)

    evaluate = sub.add_parser("eval", help="score the system on the dev set")
    evaluate.add_argument("--devset")
    evaluate.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

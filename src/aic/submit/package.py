"""Assemble query CSVs into the exact zip the organizers expect.

The guide is specific and unforgiving about the container, and two of its listed
"5 lỗi thường gặp" live here:

* the zip must contain a directory literally named ``submission/``;
* the CSVs must **not** be zipped directly at the archive root.

Getting this wrong costs one of only three attempts per query package, so
:func:`build_package` re-validates every CSV it is given, writes them into a clean
``submission/`` tree, zips that tree, and then reads the archive back to confirm
the internal paths really are ``submission/<name>.csv``.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from aic.submit.writer import SubmissionError, task_suffix, validate_rows, write_submission

SUBMISSION_DIRNAME = "submission"

#: query-1-kis.txt -> ("query-1-kis", "kis").  Suffix decides the task.
_QUERY_NAME = re.compile(r"^(?P<stem>.+)-(?P<suffix>kis|qa|trake)$", re.IGNORECASE)

#: Organizers' filename suffix -> the task name used inside this codebase.
SUFFIX_TO_TASK = {"kis": "kis", "qa": "vqa", "trake": "trake"}


@dataclass
class QuerySpec:
    """One query from a package: its id, its task, and its text."""

    query_id: str          # "query-1-kis" - also the output CSV stem
    task: str              # "kis" | "vqa" | "trake"
    text: str
    path: Path | None = None

    @property
    def csv_name(self) -> str:
        return f"{self.query_id}.csv"


def parse_query_filename(name: str) -> tuple[str, str]:
    """Split ``query-3-qa.txt`` into ``("query-3-qa", "vqa")``.

    The task is carried by the filename suffix, not the contents, so reading it
    here removes a manual step that is easy to get wrong under time pressure.
    """
    stem = Path(name).stem
    match = _QUERY_NAME.match(stem)
    if not match:
        raise SubmissionError(
            f"cannot tell the query type from {name!r}: the name must end in "
            "-kis, -qa or -trake (e.g. query-1-kis.txt)"
        )
    return stem, SUFFIX_TO_TASK[match.group("suffix").lower()]


def load_query_package(directory: Path) -> list[QuerySpec]:
    """Read every ``query-*.txt`` in a package directory, in natural order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"query package directory not found: {directory}")

    specs: list[QuerySpec] = []
    for path in sorted(directory.glob("*.txt"), key=_natural_key):
        query_id, task = parse_query_filename(path.name)
        specs.append(
            QuerySpec(
                query_id=query_id,
                task=task,
                text=path.read_text(encoding="utf-8").strip(),
                path=path,
            )
        )
    if not specs:
        raise FileNotFoundError(f"no query .txt files found in {directory}")
    return specs


def _natural_key(path: Path) -> tuple:
    """Sort query-2 before query-10 rather than lexicographically."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    )


@dataclass
class PackageReport:
    zip_path: Path
    csv_names: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Wrote {self.zip_path} with {len(self.csv_names)} CSV file(s):"]
        lines.extend(
            f"  {SUBMISSION_DIRNAME}/{name}  ({self.row_counts.get(name, 0)} rows)"
            for name in self.csv_names
        )
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        return "\n".join(lines)


def build_package(
    answers: dict[str, tuple[str, list]],
    output_zip: Path,
    work_dir: Path | None = None,
    expected_events: dict[str, int] | None = None,
) -> PackageReport:
    """Write ``submission/*.csv`` and zip it.

    ``answers`` maps query id -> ``(task, rows)``. ``expected_events`` optionally
    gives the required event count per TRAKE query id, which the guide requires to
    match exactly.
    """
    if not answers:
        raise SubmissionError("refusing to build an empty submission package")

    output_zip = Path(output_zip)
    if output_zip.suffix.lower() != ".zip":
        raise SubmissionError(f"submission archive must be a .zip, got {output_zip.name!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_zip.name):
        raise SubmissionError(
            f"zip name {output_zip.name!r} should use only letters, digits, '-', '_' "
            "(the guide recommends alphanumeric names)"
        )

    staging = Path(work_dir or output_zip.parent) / SUBMISSION_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)
    # Clear stale CSVs so a rerun cannot ship a file from a previous package.
    for stale in staging.glob("*.csv"):
        stale.unlink()

    report = PackageReport(zip_path=output_zip)
    expected_events = expected_events or {}

    for query_id, (task, rows) in answers.items():
        stem = Path(str(query_id)).stem
        if not _QUERY_NAME.match(stem):
            report.warnings.append(
                f"{stem!r} does not end in -kis/-qa/-trake; the organizers name each "
                f"CSV after its query file (expected e.g. {stem}-{task_suffix(task)})"
            )
        csv_path = staging / f"{stem}.csv"
        write_submission(
            rows, csv_path, task, expected_events=expected_events.get(query_id)
        )
        report.csv_names.append(csv_path.name)
        report.row_counts[csv_path.name] = len(rows)
        if len(rows) < 100:
            report.warnings.append(
                f"{csv_path.name}: only {len(rows)} of 100 allowed rows used - "
                "unused slots can only add score, never subtract it"
            )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(report.csv_names):
            # arcname keeps the required submission/ prefix inside the archive.
            archive.write(staging / name, arcname=f"{SUBMISSION_DIRNAME}/{name}")

    verify_package(output_zip)
    return report


def verify_package(zip_path: Path) -> list[str]:
    """Read the archive back and confirm its layout. Returns the member names.

    Written as an independent check rather than trusting the writer, so it can also
    be pointed at a zip produced by hand or by an earlier version of the tool.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"submission archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]

    if not names:
        raise SubmissionError(f"{zip_path.name} is empty")

    misplaced = [n for n in names if not n.startswith(f"{SUBMISSION_DIRNAME}/")]
    if misplaced:
        raise SubmissionError(
            f"{zip_path.name}: {misplaced[:3]} are not inside a '{SUBMISSION_DIRNAME}/' "
            "directory. The guide requires zipping the submission folder itself, not "
            "the CSV files directly."
        )

    nested = [n for n in names if n.count("/") > 1]
    if nested:
        raise SubmissionError(
            f"{zip_path.name}: {nested[:3]} sit in a sub-directory; every CSV must be "
            f"directly inside '{SUBMISSION_DIRNAME}/'"
        )

    non_csv = [n for n in names if not n.lower().endswith(".csv")]
    if non_csv:
        raise SubmissionError(
            f"{zip_path.name}: {non_csv[:3]} are not .csv files. Excel workbooks "
            "(.xlsx/.xls) are rejected by the grader."
        )
    return names


def inspect_csv(path: Path, task: str, expected_events: int | None = None) -> list[str]:
    """Re-read a written CSV and re-run validation on the parsed content.

    Catches problems that only exist once the file has been serialized — a stray
    BOM, a semicolon delimiter, a header row accidentally left in — which are
    exactly the failure modes the guide's checklist calls out.
    """
    import csv as csv_mod

    path = Path(path)
    raw = path.read_bytes()
    problems: list[str] = []

    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("file starts with a UTF-8 BOM; write plain UTF-8 instead")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [*problems, "file is not valid UTF-8"]

    rows = list(csv_mod.reader(text.splitlines()))
    if not rows:
        return [*problems, "file has no rows"]

    if any(cell.strip().lower() in {"video_id", "video", "frame_id"} for cell in rows[0]):
        problems.append(f"first line looks like a header row: {rows[0]}")

    parsed: list[tuple] = []
    for line_number, cells in enumerate(rows, start=1):
        if len(cells) < 2:
            problems.append(f"line {line_number}: fewer than 2 fields - wrong delimiter?")
            continue
        video = cells[0].strip()
        try:
            if task == "trake":
                parsed.append((video, tuple(int(c.strip()) for c in cells[1:])))
            elif task == "vqa":
                parsed.append((video, int(cells[1].strip()), cells[2] if len(cells) > 2 else ""))
            else:
                parsed.append((video, int(cells[1].strip())))
        except ValueError:
            problems.append(f"line {line_number}: non-integer frame id in {cells}")

    if parsed and not problems:
        try:
            validate_rows(parsed, task, expected_events=expected_events)
        except SubmissionError as exc:
            problems.append(str(exc))
    return problems

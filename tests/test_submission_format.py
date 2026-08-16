"""Conformance tests against "Hướng dẫn nộp bài sơ tuyển".

Each test names the rule it enforces. These are cheap to run and guard the step
where a perfect retrieval result can still score zero.
"""

from __future__ import annotations

import itertools
import zipfile

import pytest

from aic.submit.package import (
    SUBMISSION_DIRNAME,
    build_package,
    inspect_csv,
    load_query_package,
    parse_query_filename,
    verify_package,
)
from aic.submit.policy import TrakeCandidate, build_trake_answers
from aic.submit.writer import (
    ANSWER_MAX_LEN,
    SubmissionError,
    submission_path,
    validate_rows,
    write_submission,
)

# -- row formats -------------------------------------------------------------------


def test_kis_csv_matches_the_documented_example(tmp_path) -> None:
    rows = [("L00_V000", 1234), ("L00_V055", 5555), ("L01_V028", 25300)]
    path = write_submission(rows, tmp_path / "query-1-kis.csv", "kis")
    assert path.read_text(encoding="utf-8") == "L00_V000,1234\nL00_V055,5555\nL01_V028,25300\n"


def test_trake_csv_matches_the_documented_example(tmp_path) -> None:
    rows = [
        ("L10_V001", (1200, 1850, 2100, 2450)),
        ("L10_V001", (1180, 1820, 2080, 2420)),
        ("L11_V003", (5100, 5700, 6200, 6800)),
    ]
    path = write_submission(rows, tmp_path / "query-3-trake.csv", "trake")
    assert path.read_text(encoding="utf-8").splitlines()[0] == "L10_V001,1200,1850,2100,2450"


def test_qa_quotes_only_when_required(tmp_path) -> None:
    """The guide: quotes are mandatory for commas/quotes, optional otherwise."""
    rows = [
        ("L01_V028", 3450, "5"),
        ("L02_V011", 1200, "Năm người"),
        ("L03_V005", 2800, "Màu đỏ, rất đẹp"),
        ("L04_V012", 4100, 'Anh ấy nói "Tuyệt vời"'),
    ]
    lines = write_submission(rows, tmp_path / "q.csv", "vqa").read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines[0] == "L01_V028,3450,5"                       # simple: unquoted
    assert lines[1] == "L02_V011,1200,Năm người"               # spaces need no quotes
    assert lines[2] == 'L03_V005,2800,"Màu đỏ, rất đẹp"'       # comma forces quotes
    assert lines[3] == 'L04_V012,4100,"Anh ấy nói ""Tuyệt vời"""'  # quotes doubled


def test_answer_with_newline_is_quoted_and_survives_a_round_trip(tmp_path) -> None:
    path = write_submission([("L01_V028", 3450, "Dòng 1\nDòng 2")], tmp_path / "q.csv", "vqa")
    import csv

    with open(path, encoding="utf-8", newline="") as handle:
        assert next(iter(csv.reader(handle)))[2] == "Dòng 1\nDòng 2"


def test_written_files_are_utf8_without_bom_and_have_no_header(tmp_path) -> None:
    path = write_submission([("L01_V028", 3450, "Màu đỏ")], tmp_path / "q.csv", "vqa")
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8").splitlines()[0].startswith("L01_V028")


# -- validation rules ---------------------------------------------------------------


def test_video_name_must_not_carry_the_mp4_extension() -> None:
    """Listed in the guide's summary table as a wrong-vs-right example."""
    with pytest.raises(SubmissionError, match="must not include the file extension"):
        validate_rows([("L01_V028.mp4", 25300)], "kis")


def test_video_name_must_not_have_stray_whitespace() -> None:
    with pytest.raises(SubmissionError, match="surrounding whitespace"):
        validate_rows([(" L01_V028", 25300)], "kis")


def test_answer_over_100_characters_is_rejected() -> None:
    long_answer = "a" * (ANSWER_MAX_LEN + 1)
    with pytest.raises(SubmissionError, match="over the 100-character limit"):
        validate_rows([("L01_V028", 3450, long_answer)], "vqa")
    # Exactly at the limit is fine.
    validate_rows([("L01_V028", 3450, "a" * ANSWER_MAX_LEN)], "vqa")


def test_trake_frame_count_must_match_the_requested_event_count() -> None:
    rows = [("L10_V001", (1200, 1850, 2100))]
    with pytest.raises(SubmissionError, match="the query has 4 events"):
        validate_rows(rows, "trake", expected_events=4)
    validate_rows([("L10_V001", (1200, 1850, 2100, 2450))], "trake", expected_events=4)


def test_trake_frames_must_be_in_chronological_order() -> None:
    with pytest.raises(SubmissionError, match="not in chronological order"):
        validate_rows([("L10_V001", (1200, 1850, 1800, 2450))], "trake")
    with pytest.raises(SubmissionError, match="not in chronological order"):
        validate_rows([("L10_V001", (1200, 1200))], "trake")  # equal is not increasing


def test_more_than_100_rows_is_rejected() -> None:
    with pytest.raises(SubmissionError, match="exceeds the 100"):
        validate_rows([("L01_V028", i) for i in range(101)], "kis")


# -- the policy must not generate invalid rows ---------------------------------------


def test_trake_jitter_never_emits_out_of_order_frames() -> None:
    """Events 4 frames apart with ±12 jitter would otherwise cross over."""
    candidate = TrakeCandidate(video_id="L10_V001", frame_ids=(100, 104, 108, 112), score=0.9)
    rows = build_trake_answers([candidate], jitter=(0, -12, 12, -4, 4))

    for _, frames in rows:
        assert all(a < b for a, b in itertools.pairwise(frames)), frames
    validate_rows(rows, "trake", expected_events=4)  # the writer would accept all of them


def test_policy_output_passes_validation_for_every_task() -> None:
    from aic.submit.policy import Candidate, build_kis_answers, build_vqa_answers

    kis = build_kis_answers([Candidate("L01_V001", 500, 0.9), Candidate("L02_V001", 20, 0.8)])
    validate_rows(kis, "kis")

    vqa = build_vqa_answers([Candidate("L05_V005", 888, 0.9, answer="màu xanh")])
    validate_rows(vqa, "vqa")

    trake = build_trake_answers([TrakeCandidate("L10_V010", (100, 150, 200, 250), 0.9)])
    validate_rows(trake, "trake", expected_events=4)


# -- filenames ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("query-1-kis.txt", ("query-1-kis", "kis")),
        ("query-3-qa.txt", ("query-3-qa", "vqa")),
        ("query-4-trake.txt", ("query-4-trake", "trake")),
        ("query-10-KIS.TXT", ("query-10-KIS", "kis")),
    ],
)
def test_query_filename_determines_the_task(filename: str, expected: tuple[str, str]) -> None:
    assert parse_query_filename(filename) == expected


def test_unrecognisable_query_filename_is_rejected() -> None:
    with pytest.raises(SubmissionError, match="cannot tell the query type"):
        parse_query_filename("query-5-unknown.txt")


def test_submission_path_is_flat_and_named_after_the_query() -> None:
    path = submission_path("submissions", "query-1-kis", "kis")
    assert path.name == "query-1-kis.csv"
    assert path.parent.name == "submissions"          # no per-task sub-folder
    # A bare id gets the organizers' suffix appended (vqa -> "qa").
    assert submission_path("submissions", "query-3", "vqa").name == "query-3-qa.csv"


def test_load_query_package_sorts_naturally(tmp_path) -> None:
    for name in ("query-1-kis.txt", "query-2-qa.txt", "query-10-trake.txt"):
        (tmp_path / name).write_text("nội dung truy vấn", encoding="utf-8")

    specs = load_query_package(tmp_path)
    assert [s.query_id for s in specs] == ["query-1-kis", "query-2-qa", "query-10-trake"]
    assert [s.task for s in specs] == ["kis", "vqa", "trake"]
    assert specs[0].csv_name == "query-1-kis.csv"


# -- packaging ----------------------------------------------------------------------


def test_zip_contains_a_submission_folder(tmp_path) -> None:
    """The guide's #2 most common error: zipping CSVs directly."""
    answers = {
        "query-1-kis": ("kis", [("L00_V000", 1234)]),
        "query-3-qa": ("vqa", [("L01_V028", 3450, "5")]),
        "query-4-trake": ("trake", [("L10_V001", (1200, 1850, 2100, 2450))]),
    }
    report = build_package(answers, tmp_path / "team_ABC_round1.zip", work_dir=tmp_path)

    with zipfile.ZipFile(report.zip_path) as archive:
        names = archive.namelist()
    assert sorted(names) == [
        f"{SUBMISSION_DIRNAME}/query-1-kis.csv",
        f"{SUBMISSION_DIRNAME}/query-3-qa.csv",
        f"{SUBMISSION_DIRNAME}/query-4-trake.csv",
    ]


def test_packaged_csv_content_survives_the_zip(tmp_path) -> None:
    answers = {"query-1-kis": ("kis", [("L00_V000", 1234), ("L00_V055", 5555)])}
    report = build_package(answers, tmp_path / "s.zip", work_dir=tmp_path)

    with zipfile.ZipFile(report.zip_path) as archive:
        body = archive.read(f"{SUBMISSION_DIRNAME}/query-1-kis.csv").decode("utf-8")
    assert body == "L00_V000,1234\nL00_V055,5555\n"


def test_package_warns_when_answer_slots_are_unused(tmp_path) -> None:
    report = build_package(
        {"query-1-kis": ("kis", [("L00_V000", 1234)])}, tmp_path / "s.zip", work_dir=tmp_path
    )
    assert any("of 100 allowed rows" in w for w in report.warnings)


def test_package_rejects_a_non_alphanumeric_zip_name(tmp_path) -> None:
    with pytest.raises(SubmissionError, match="letters, digits"):
        build_package(
            {"query-1-kis": ("kis", [("L00_V000", 1)])},
            tmp_path / "team ABC!.zip",
            work_dir=tmp_path,
        )


def test_package_rejects_a_non_zip_extension(tmp_path) -> None:
    with pytest.raises(SubmissionError, match="must be a .zip"):
        build_package({"q-1-kis": ("kis", [("L0", 1)])}, tmp_path / "s.rar", work_dir=tmp_path)


def test_stale_csvs_are_cleared_between_builds(tmp_path) -> None:
    build_package({"query-1-kis": ("kis", [("L00_V000", 1)])}, tmp_path / "s.zip", work_dir=tmp_path)
    build_package({"query-2-kis": ("kis", [("L00_V001", 2)])}, tmp_path / "s.zip", work_dir=tmp_path)

    with zipfile.ZipFile(tmp_path / "s.zip") as archive:
        assert archive.namelist() == [f"{SUBMISSION_DIRNAME}/query-2-kis.csv"]


def test_verify_package_rejects_bare_csvs_at_the_archive_root(tmp_path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("query-1-kis.csv", "L00_V000,1234\n")
    with pytest.raises(SubmissionError, match="not inside a 'submission/'"):
        verify_package(bad)


def test_verify_package_rejects_an_excel_file(tmp_path) -> None:
    """The guide's #1 most common error."""
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr(f"{SUBMISSION_DIRNAME}/query-1-kis.xlsx", "binary junk")
    with pytest.raises(SubmissionError, match="not .csv files"):
        verify_package(bad)


def test_verify_package_rejects_nested_directories(tmp_path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr(f"{SUBMISSION_DIRNAME}/round1/query-1-kis.csv", "L00,1\n")
    with pytest.raises(SubmissionError, match="sub-directory"):
        verify_package(bad)


# -- post-write inspection -----------------------------------------------------------


def test_inspect_csv_flags_a_semicolon_delimiter(tmp_path) -> None:
    path = tmp_path / "query-1-kis.csv"
    path.write_text("L00_V000;1234\nL00_V055;5555\n", encoding="utf-8")
    assert any("wrong delimiter" in p for p in inspect_csv(path, "kis"))


def test_inspect_csv_flags_a_header_row(tmp_path) -> None:
    path = tmp_path / "query-1-kis.csv"
    path.write_text("video_id,frame_id\nL00_V000,1234\n", encoding="utf-8")
    assert any("header row" in p for p in inspect_csv(path, "kis"))


def test_inspect_csv_flags_a_bom(tmp_path) -> None:
    path = tmp_path / "query-1-kis.csv"
    path.write_bytes(b"\xef\xbb\xbfL00_V000,1234\n")
    assert any("BOM" in p for p in inspect_csv(path, "kis"))


def test_inspect_csv_accepts_a_file_we_wrote(tmp_path) -> None:
    path = write_submission([("L00_V000", 1234)], tmp_path / "query-1-kis.csv", "kis")
    assert inspect_csv(path, "kis") == []

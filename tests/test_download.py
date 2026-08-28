"""Tests for manifest parsing and the download/extract pipeline.

No real network access: ``fetch`` is exercised with ``_download`` monkeypatched
to write a canned file, and the lower-level ``_download`` is tested against a
fake ``requests.Session`` so resume/skip/sha256 behaviour is checked without a
server.
"""

from __future__ import annotations

import hashlib
import zipfile

import pytest

from aic.data.download import (
    KNOWN_KINDS,
    ManifestEntry,
    _download,
    _drive_confirm_token,
    _drive_file_id,
    _extract_zip,
    _is_complete,
    _resolve_download_url,
    _sha256,
    fetch,
    read_manifest,
)

EXAMPLE_HEADER = "kind,name,url,sha256\n"


def _write_manifest(tmp_path, body: str):
    path = tmp_path / "manifest.csv"
    path.write_text(EXAMPLE_HEADER + body, encoding="utf-8")
    return path


# -- read_manifest ---------------------------------------------------------------------


def test_read_manifest_parses_rows_and_skips_comments_and_blanks(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        "# a comment line\n"
        "\n"
        "videos,Videos_L01.zip,https://example.com/v1.zip,\n"
        "keyframes,Keyframes_L01.zip,https://example.com/k1.zip,deadbeef\n",
    )
    entries = read_manifest(path)

    assert entries == [
        ManifestEntry(kind="videos", name="Videos_L01.zip", url="https://example.com/v1.zip", sha256=None),
        ManifestEntry(
            kind="keyframes", name="Keyframes_L01.zip", url="https://example.com/k1.zip", sha256="deadbeef"
        ),
    ]


def test_read_manifest_raises_on_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "nope.csv")


def test_read_manifest_raises_on_missing_header_column(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("kind,name\nvideos,Videos_L01.zip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_manifest(path)


def test_read_manifest_raises_on_unknown_kind(tmp_path) -> None:
    path = _write_manifest(tmp_path, "audio,Foo.zip,https://example.com/f.zip,\n")
    with pytest.raises(ValueError, match="kind"):
        read_manifest(path)


def test_read_manifest_raises_on_empty_required_field(tmp_path) -> None:
    path = _write_manifest(tmp_path, "videos,,https://example.com/f.zip,\n")
    with pytest.raises(ValueError):
        read_manifest(path)


def test_known_kinds_matches_the_raw_path_categories() -> None:
    assert set(KNOWN_KINDS) == {"videos", "keyframes", "objects", "clip_features", "metadata"}


# -- Google Drive link rewriting --------------------------------------------------------


def test_resolve_download_url_leaves_plain_urls_untouched() -> None:
    url, params = _resolve_download_url("https://aic-data.ledo.io.vn/Videos_L01.zip")
    assert url == "https://aic-data.ledo.io.vn/Videos_L01.zip"
    assert params == {}


def test_resolve_download_url_rewrites_a_drive_share_link() -> None:
    url, params = _resolve_download_url("https://drive.google.com/file/d/ABC123/view?usp=sharing")
    assert url == "https://drive.google.com/uc"
    assert params == {"id": "ABC123", "export": "download"}


def test_drive_file_id_handles_the_id_query_param_form() -> None:
    assert _drive_file_id("https://drive.google.com/open?id=XYZ789") == "XYZ789"


def test_drive_file_id_returns_none_for_non_drive_urls() -> None:
    assert _drive_file_id("https://example.com/file/d/ABC123") is None


class _FakeResponse:
    def __init__(self, cookies=None, headers=None, text=""):
        self.cookies = cookies or {}
        self.headers = headers or {}
        self.text = text


def test_drive_confirm_token_reads_the_warning_cookie() -> None:
    response = _FakeResponse(cookies={"download_warning_abc": "tok123"})
    assert _drive_confirm_token(response) == "tok123"


def test_drive_confirm_token_reads_the_html_body_fallback() -> None:
    response = _FakeResponse(
        headers={"content-type": "text/html; charset=utf-8"},
        text="<html>confirm=tok456&amp;other</html>",
    )
    assert _drive_confirm_token(response) == "tok456"


def test_drive_confirm_token_is_none_when_absent() -> None:
    assert _drive_confirm_token(_FakeResponse()) is None


# -- hashing -----------------------------------------------------------------------------


def test_sha256_matches_hashlib(tmp_path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert _sha256(path) == hashlib.sha256(b"hello world").hexdigest()


def test_is_complete_true_without_sha256_if_file_exists(tmp_path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"anything")
    assert _is_complete(path, None)


def test_is_complete_false_when_file_missing(tmp_path) -> None:
    assert not _is_complete(tmp_path / "missing.bin", None)


def test_is_complete_checks_sha256_case_insensitively(tmp_path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    digest = hashlib.sha256(b"hello world").hexdigest()
    assert _is_complete(path, digest.upper())
    assert not _is_complete(path, "0" * 64)


# -- zip extraction -----------------------------------------------------------------------


def test_extract_zip_writes_files_into_the_destination(tmp_path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/data.txt", "hello")
    dest = tmp_path / "out"
    dest.mkdir()

    _extract_zip(archive, dest)
    assert (dest / "inner" / "data.txt").read_text() == "hello"


def test_extract_zip_rejects_path_traversal(tmp_path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(ValueError, match="unsafe path"):
        _extract_zip(archive, dest)
    assert not (tmp_path / "escape.txt").exists()


# -- _download (resume / skip / verify) ---------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, content: bytes, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.cookies = {}
        self.text = ""

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def get(self, url, params=None, headers=None, stream=True):
        self.requests.append({"url": url, "params": params, "headers": headers})
        return self._responses.pop(0)


def test_download_writes_the_full_response_body(tmp_path) -> None:
    session = _FakeSession([_FakeStreamResponse(b"hello world", headers={"content-length": "11"})])
    entry = ManifestEntry(kind="videos", name="v.zip", url="https://example.com/v.zip")
    dest = tmp_path / "v.zip"

    _download(session, entry, dest)

    assert dest.read_bytes() == b"hello world"
    assert not dest.with_name("v.zip.part").exists()


def test_download_skips_when_sha256_already_matches(tmp_path) -> None:
    dest = tmp_path / "v.zip"
    dest.write_bytes(b"already here")
    digest = hashlib.sha256(b"already here").hexdigest()
    entry = ManifestEntry(kind="videos", name="v.zip", url="https://example.com/v.zip", sha256=digest)

    class ExplodingSession:
        def get(self, *a, **k):
            raise AssertionError("must not hit the network when already complete")

    _download(ExplodingSession(), entry, dest)
    assert dest.read_bytes() == b"already here"


def test_download_resumes_from_an_existing_part_file(tmp_path) -> None:
    dest = tmp_path / "v.zip"
    part = tmp_path / "v.zip.part"
    part.write_bytes(b"hello ")
    session = _FakeSession([_FakeStreamResponse(b"world", status_code=206, headers={"content-length": "5"})])
    entry = ManifestEntry(kind="videos", name="v.zip", url="https://example.com/v.zip")

    _download(session, entry, dest)

    assert dest.read_bytes() == b"hello world"
    assert session.requests[0]["headers"] == {"Range": "bytes=6-"}


def test_download_raises_and_cleans_up_on_sha256_mismatch(tmp_path) -> None:
    session = _FakeSession([_FakeStreamResponse(b"wrong bytes", headers={"content-length": "11"})])
    entry = ManifestEntry(
        kind="videos", name="v.zip", url="https://example.com/v.zip", sha256="0" * 64
    )
    dest = tmp_path / "v.zip"

    with pytest.raises(ValueError, match="sha256 mismatch"):
        _download(session, entry, dest)
    assert not dest.exists()
    assert not dest.with_name("v.zip.part").exists()


# -- fetch orchestration ------------------------------------------------------------------


def test_fetch_downloads_extracts_and_removes_the_archive_by_default(tmp_path, monkeypatch) -> None:
    def fake_download(session, entry, dest):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("payload.txt", entry.name)

    monkeypatch.setattr("aic.data.download._download", fake_download)

    entries = [ManifestEntry(kind="videos", name="v.zip", url="https://example.com/v.zip")]
    dest_dir = tmp_path / "raw" / "videos"
    written = fetch(entries, {"videos": dest_dir})

    assert written == [dest_dir / "v.zip"]
    assert not (dest_dir / "v.zip").exists()  # archive removed after extraction
    assert (dest_dir / "payload.txt").read_text() == "v.zip"


def test_fetch_keeps_the_archive_when_asked(tmp_path, monkeypatch) -> None:
    def fake_download(session, entry, dest):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("payload.txt", "x")

    monkeypatch.setattr("aic.data.download._download", fake_download)

    entries = [ManifestEntry(kind="videos", name="v.zip", url="https://example.com/v.zip")]
    dest_dir = tmp_path / "raw" / "videos"
    fetch(entries, {"videos": dest_dir}, keep_archives=True)

    assert (dest_dir / "v.zip").exists()
    assert (dest_dir / "payload.txt").exists()


def test_fetch_skips_extraction_when_disabled(tmp_path, monkeypatch) -> None:
    def fake_download(session, entry, dest):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("payload.txt", "x")

    monkeypatch.setattr("aic.data.download._download", fake_download)

    entries = [ManifestEntry(kind="videos", name="v.zip", url="https://example.com/v.zip")]
    dest_dir = tmp_path / "raw" / "videos"
    fetch(entries, {"videos": dest_dir}, extract=False)

    assert (dest_dir / "v.zip").exists()
    assert not (dest_dir / "payload.txt").exists()

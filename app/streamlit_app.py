"""Interactive search UI for the preliminary round.

KIS is not a fully automatic task — the training material is explicit that the
winning systems put a human in the loop, scanning a keyframe grid and confirming
the moment. So this app optimizes for that loop: search, scan a dense grid, open
the video at the exact frame, drop answers into a cart, export.

All retrieval logic lives in ``src/aic/``; this file only draws. That separation is
what makes the planned move to FastAPI + React a rewrite of this file alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic.config import load_config
from aic.query.search import SearchEngine
from aic.submit.policy import (
    Candidate,
    TrakeCandidate,
    build_kis_answers,
    build_trake_answers,
    build_vqa_answers,
)
from aic.submit.writer import SubmissionError, submission_path, write_submission

st.set_page_config(page_title="AIC 2026 Search", layout="wide", page_icon="🎬")


@st.cache_resource
def get_engine():
    config = load_config()
    return SearchEngine(config), config


def _init_state() -> None:
    st.session_state.setdefault("cart", [])          # list of dicts
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("trake_cart", [])


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def render_sidebar(config) -> dict:
    st.sidebar.title("🎬 AIC 2026")
    task = st.sidebar.radio(
        "Dạng truy vấn",
        ["kis", "vqa", "trake"],
        format_func={"kis": "1 · Textual KIS", "vqa": "2 · Hỏi–Đáp", "trake": "3 · TRAKE"}.get,
    )
    st.sidebar.divider()
    query_id = st.sidebar.text_input("Mã truy vấn", value="query-01")
    grid_columns = st.sidebar.slider("Số cột lưới", 3, 10, 6)
    top_n = st.sidebar.slider("Số kết quả", 20, 500, 120, step=20)

    st.sidebar.divider()
    st.sidebar.caption(f"Không gian embedding: `{config.active_space.name}`")
    st.sidebar.caption(f"Index: `{config.index_path.name}`")

    return {"task": task, "query_id": query_id, "grid_columns": grid_columns, "top_n": top_n}


def render_cart(settings: dict, config) -> None:
    """The answer cart, ordered — position 1 is worth a fifth of the query's score."""
    task = settings["task"]
    cart = st.session_state.trake_cart if task == "trake" else st.session_state.cart

    st.subheader(f"🧺 Giỏ đáp án ({len(cart)})")
    if not cart:
        st.caption("Chưa chọn đáp án nào. Bấm **➕** dưới mỗi khung hình để thêm.")
        return

    st.caption(
        "Thứ tự quyết định điểm: hạng 1 chiếm 1/5 điểm truy vấn. "
        "Nên để các video **khác nhau** ở 5 vị trí đầu."
    )
    for position, item in enumerate(cart):
        columns = st.columns([0.5, 3, 1, 1])
        columns[0].markdown(f"**{position + 1}**")
        if task == "trake":
            columns[1].markdown(f"`{item['video_id']}` → {', '.join(map(str, item['frame_ids']))}")
        else:
            columns[1].markdown(f"`{item['video_id']}` @ frame **{item['frame_idx']}**")
            if task == "vqa":
                item["answer"] = columns[1].text_input(
                    "Câu trả lời", value=item.get("answer", ""), key=f"ans-{position}",
                    label_visibility="collapsed", placeholder="Nhập câu trả lời…",
                )
        if columns[2].button("↑", key=f"up-{position}", disabled=position == 0):
            cart[position - 1], cart[position] = item, cart[position - 1]
            st.rerun()
        if columns[3].button("🗑", key=f"del-{position}"):
            cart.pop(position)
            st.rerun()

    st.divider()
    if st.button("📤 Xuất file nộp", type="primary", use_container_width=True):
        _export(task, cart, settings["query_id"], config)


def _export(task: str, cart: list[dict], query_id: str, config) -> None:
    """Expand the cart into up to 100 ranked rows and write the CSV."""
    submission = config.submission
    # The submission format requires the frame count to equal the query's event
    # count exactly, so take it from the events the operator typed in.
    events_text = st.session_state.get("events", "") or ""
    event_lines = [line for line in events_text.splitlines() if line.strip()]
    expected_events = len(event_lines) if (task == "trake" and event_lines) else None

    try:
        if task == "trake":
            candidates = [
                TrakeCandidate(
                    video_id=item["video_id"],
                    frame_ids=tuple(item["frame_ids"]),
                    score=1.0 - index / 1000,  # cart order is the ranking
                    per_event_scores=tuple(item.get("per_event_scores", ())),
                )
                for index, item in enumerate(cart)
            ]
            rows = build_trake_answers(
                candidates,
                max_answers=int(submission.max_answers),
                diversify_head=int(submission.diversify_head),
                jitter=tuple(submission.trake_jitter),
            )
        else:
            candidates = [
                Candidate(
                    video_id=item["video_id"],
                    frame_idx=int(item["frame_idx"]),
                    score=1.0 - index / 1000,
                    answer=item.get("answer"),
                )
                for index, item in enumerate(cart)
            ]
            builder = build_vqa_answers if task == "vqa" else build_kis_answers
            rows = builder(
                candidates,
                max_answers=int(submission.max_answers),
                diversify_head=int(submission.diversify_head),
                frames_per_shot=int(submission.frames_per_shot),
                frame_spread=int(submission.frame_spread),
            )

        path = submission_path(config.submissions_dir, query_id, task)
        write_submission(rows, path, task, expected_events=expected_events)
    except SubmissionError as exc:
        st.error(f"Không xuất được: {exc}")
        return

    st.success(f"Đã ghi {len(rows)} đáp án → `{path}`")
    st.caption(
        "Sau khi xuất đủ các truy vấn trong gói, chạy `aic package -o team_round1.zip` "
        "để đóng gói đúng cấu trúc `submission/` mà BTC yêu cầu."
    )
    st.download_button(
        "⬇️ Tải file",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
        use_container_width=True,
    )


def render_results(settings: dict, config) -> None:
    result = st.session_state.results
    if result is None:
        return
    if not result.hits:
        st.warning("Không có kết quả. Thử diễn đạt khác hoặc kiểm tra index đã dựng chưa.")
        return

    if result.disabled_branches:
        with st.expander(f"⚠️ {len(result.disabled_branches)} nhánh tìm kiếm bị tắt", expanded=False):
            for name, reason in result.disabled_branches.items():
                st.write(f"**{name}** — {reason}")

    with st.expander("🔍 Truy vấn đã mở rộng", expanded=False):
        expanded = result.expanded
        st.write(f"**Nguồn:** {'Gemini' if expanded.used_llm else 'rule-based (không có API key)'}")
        st.write(f"**Visual (EN):** {expanded.visual_en}")
        st.write(f"**OCR:** {expanded.ocr_terms}")
        st.write(f"**ASR:** {expanded.asr_terms}")

    st.caption(f"{len(result.hits)} kết quả · nhánh đang hoạt động: {', '.join(result.active_branches)}")

    columns = st.columns(settings["grid_columns"])
    for position, hit in enumerate(result.hits):
        with columns[position % settings["grid_columns"]]:
            image_path = _resolve(hit.path)
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.caption(f"(thiếu ảnh: {hit.path})")

            st.caption(
                f"**{hit.video_id}** · f`{hit.frame_idx}`"
                + (f" · {hit.pts_time:.1f}s" if hit.pts_time is not None else "")
            )
            st.caption(f"{hit.score:.4f} · {'+'.join(hit.branches)}")
            if hit.evidence:
                st.caption(f"💬 {hit.evidence[0][:80]}")

            if st.button("➕", key=f"add-{hit.gid}", use_container_width=True):
                _add_to_cart(settings["task"], hit)
                st.rerun()

            with st.popover("▶", use_container_width=True):
                _render_player(hit, config)


def _add_to_cart(task: str, hit) -> None:
    if hit.frame_idx is None:
        st.toast("Khung hình này không có frame_idx — không nộp được.", icon="⚠️")
        return
    if task == "trake":
        st.session_state.trake_cart.append(
            {"video_id": hit.video_id, "frame_ids": [hit.frame_idx]}
        )
    else:
        st.session_state.cart.append(
            {"video_id": hit.video_id, "frame_idx": hit.frame_idx, "answer": ""}
        )


def _render_player(hit, config) -> None:
    """Play the source video seeked to this frame, to confirm before submitting."""
    from aic.data.verify import _find_video

    video_path = _find_video(config.raw_path("videos"), hit.video_id)
    if video_path is None:
        st.caption(f"Không tìm thấy video `{hit.video_id}`")
        return
    start = int(hit.pts_time) if hit.pts_time is not None else 0
    st.video(str(video_path), start_time=max(0, start - 2))
    st.caption(f"frame {hit.frame_idx} ≈ {hit.pts_time or 0:.1f}s")


def main() -> None:
    _init_state()
    try:
        engine, config = get_engine()
    except Exception as exc:  # noqa: BLE001 - show setup errors in the UI
        st.error(f"Không nạp được hệ thống: {exc}")
        st.info("Chạy `aic build-catalog` rồi `aic build-index` trước khi mở giao diện.")
        return

    settings = render_sidebar(config)
    st.title("Tìm kiếm sự kiện trong video")

    search_column, cart_column = st.columns([3, 1])

    with search_column:
        with st.form("search"):
            query = st.text_area(
                "Mô tả sự kiện (tiếng Việt)",
                height=90,
                placeholder="Ví dụ: một diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời…",
            )
            if settings["task"] == "vqa":
                st.session_state["question"] = st.text_input(
                    "Câu hỏi", placeholder="Có bao nhiêu người lên sân khấu?"
                )
            if settings["task"] == "trake":
                st.session_state["events"] = st.text_area(
                    "Các khoảnh khắc (mỗi dòng một sự kiện, đúng thứ tự)",
                    height=90,
                    placeholder="chạy đà\ngiậm nhảy\nbay qua xà\ntiếp đất",
                )
            submitted = st.form_submit_button("🔍 Tìm kiếm", type="primary", use_container_width=True)

        if submitted and query.strip():
            with st.spinner("Đang tìm…"):
                st.session_state.results = engine.search(query, top_n=settings["top_n"])

        render_results(settings, config)

    with cart_column:
        render_cart(settings, config)


if __name__ == "__main__":
    main()

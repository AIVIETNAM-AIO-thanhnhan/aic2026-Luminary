# AIC 2026 — Hệ thống truy vấn video

Hệ thống tìm kiếm sự kiện trong kho video cho **Hội thi Thử thách Trí tuệ Nhân tạo
TP.HCM 2026**, vòng sơ tuyển. Hỗ trợ cả ba dạng truy vấn: Textual KIS, Hỏi–Đáp (Q&A)
và TRAKE.

---

## Ba quyết định thiết kế quan trọng

**1. Hybrid GPU — máy local không cần card NVIDIA.**
Việc nặng (encode ảnh, ASR, OCR) chạy trên Colab/Kaggle GPU rồi xuất ra `.npy` /
`.parquet`. Máy local chỉ nạp artifact vào FAISS-CPU và chạy giao diện. Encode
*truy vấn* chỉ là vài câu ngắn nên CPU thừa sức — chính sự bất đối xứng này khiến
kiến trúc tách đôi hoạt động được.

**2. `frame_idx` được kiểm chứng bằng thực nghiệm, không phải bằng giả định.**
Số nộp lên BTC là chỉ số frame **trong video gốc**. Lệch một đơn vị là toàn bộ đáp án
bằng 0 trong khi giao diện vẫn hiển thị đúng. `aic verify` trích lại frame bằng
ffmpeg và so pixel với keyframe đã lưu — chạy **trước** khi index bất cứ thứ gì.

**3. Chính sách xếp hạng đáp án là một module riêng.**
`Final Score = mean(R@1, R@5, R@20, R@50, R@100)` có cấu trúc khai thác được: hạng 1
chiếm 1/5 điểm, còn các slot 6–100 gần như miễn phí. Xem `src/aic/submit/policy.py`.

---

## Cài đặt

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e ".[dev]"

# Bộ mã hoá truy vấn (CPU) — cần cho nhánh tìm kiếm bằng hình ảnh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[onnx]"

# Tuỳ chọn: phân rã truy vấn & trả lời Q&A bằng Gemini
pip install -e ".[llm]"
echo GEMINI_API_KEY=... > .env
```

Yêu cầu: Python 3.11+, `ffmpeg` và `ffprobe` trong PATH.

---

## Quy trình

### Bước 1 — Tải dữ liệu

Xuất các dòng link từ Google Sheets của BTC ra `configs/manifest.csv`
(`kind,name,url`), rồi:

```bash
aic fetch configs/manifest.csv
```

### Bước 2 — Dựng catalog và **kiểm chứng frame index** ⚠️

```bash
aic build-catalog
aic verify --verbose
```

`aic verify` phải in **PASS**. Nếu báo cần offset khác 0, sửa `frame_idx` theo đúng
độ lệch đó rồi chạy lại. **Không index khi bước này chưa PASS** — mọi thứ phía sau
đều dựa vào nó.

### Bước 3 — Baseline chạy ngay bằng CLIP features của BTC

```bash
aic build-index                 # embedding.active: btc_clip
aic search "một người đang mở laptop"
streamlit run app/streamlit_app.py
```

### Bước 4 — Index phong phú hơn (Colab GPU)

| Notebook | Sản phẩm | Vì sao đáng làm |
|---|---|---|
| `notebooks/01_siglip.ipynb` | `embeddings/siglip/*.npy` | mạnh hơn hẳn CLIP B/32 |
| `notebooks/02_asr.ipynb` | `asr/asr.parquet` | bản tin có lời dẫn → tín hiệu giàu |
| `notebooks/03_ocr.ipynb` | `ocr/ocr.parquet` | chữ chạy/tiêu đề → **chính xác nhất** |
| `notebooks/04_transnet.ipynb` | keyframe theo shot | I-frame bỏ sót cảnh ngắn |

Tải artifact về `data/derived/`, đổi `embedding.active: siglip` trong
`configs/default.yaml`, rồi:

```bash
aic build-index
aic build-text
```

> Ràng buộc thứ tự: **catalog phải chốt trước khi tính embedding**, vì `gid` chính là
> chỉ số dòng trong ma trận embedding.

### Bước 5 — Đo chất lượng

```bash
aic eval        # cần configs/devset.jsonl
```

BTC không cung cấp ground truth, nên phải tự gán nhãn ~30 truy vấn qua giao diện
Streamlit. Không có bước này thì mọi tinh chỉnh sau đó đều là đoán mò.

---

## Kiến trúc

```
OFFLINE (Colab GPU)                         LOCAL (CPU)
  videos ─┬─ TransNet V2 ─→ keyframes         catalog.parquet  (gid ↔ frame_idx)
          ├─ SigLIP2      ─→ embeddings.npy   FAISS IndexFlatIP
          ├─ Whisper      ─→ asr.parquet   ─→ SQLite FTS5
          └─ PaddleOCR    ─→ ocr.parquet   ─→ SQLite FTS5

truy vấn tiếng Việt
  → phân rã (Gemini): visual EN | OCR vi | ASR vi | objects
  → 4 nhánh tìm kiếm song song
  → Reciprocal Rank Fusion
  → giao diện Streamlit (người duyệt, chọn đáp án)
  → policy xếp 100 dòng → CSV
```

**Vì sao là FAISS + SQLite chứ không phải Qdrant + Elasticsearch:** cả hai đều cài
bằng `pip`, không cần Docker hay daemon. Với ~300k keyframe, FAISS flat tốn ~450 MB
RAM và truy vấn dưới 50 ms trên CPU. Bớt được hai thành phần phải giữ sống trong lúc
thi.

**Vì sao là RRF chứ không phải tổng có trọng số:** bốn nhánh cho điểm trên các thang
hoàn toàn khác nhau (cosine, BM25, nhị phân). Dò trọng số cần tập nhãn lớn mà ta
không có; RRF chỉ dùng *thứ hạng* nên không nhánh nào áp đảo chỉ vì thang điểm.

---

## Cấu trúc mã nguồn

```
src/aic/
  config.py            nạp YAML → object, mọi đường dẫn quy về gốc repo
  data/    catalog.py  ★ catalog.parquet: hợp đồng dữ liệu trung tâm
           verify.py   ★ kiểm chứng frame_idx bằng ffmpeg
           download.py tải theo manifest, có resume
  index/   embed.py    encode keyframe theo shard, chạy lại được
           vector_index.py   FAISS, giữ bất biến gid == chỉ số dòng
           text_index.py     FTS5, khớp cả có dấu lẫn không dấu
  query/   expand.py   phân rã truy vấn theo 4 mặt (có fallback offline)
           encoder.py  text/image tower chạy CPU
           fusion.py   Reciprocal Rank Fusion
           search.py   điều phối; nhánh hỏng thì tắt nhánh, không sập
  tasks/   trake.py    ★ hai giai đoạn: xếp hạng video → căn chỉnh frame dày đặc
           vqa.py      định vị khoảnh khắc → VLM đọc đáp án
  submit/  policy.py   ★ khai thác công thức Final Score
           writer.py   validate rồi mới ghi CSV
  eval/    metric.py   ★ công thức chấm điểm chính thức
           devset.py, run.py

app/streamlit_app.py   giao diện; chỉ vẽ, mọi logic nằm ở src/aic
notebooks/             driver mỏng cho Colab (sinh bởi _make_notebooks.py)
```

★ = các tệp then chốt về mặt chính xác.

---

## Kiểm thử

```bash
pytest -q                      # 55 tests
```

Đáng chú ý:

- `test_metric.py` — dùng thẳng hai ví dụ trong tài liệu BTC: TRAKE
  `L10_V010, 101, 156, 203, 251` → **0.75**, và ví dụ Final Score → **0.74**.
- `test_policy.py` — chứng minh việc rải frame cứu được đáp án lệch nhẹ, và jitter
  của TRAKE nâng 3/4 lên 4/4 ở các hạng sau.
- `test_embed_sharding.py` — bảo vệ bất biến `gid == chỉ số dòng embedding`, kể cả
  khi có ảnh hỏng giữa chừng.
- `test_end_to_end.py` — chạy trọn catalog → index → search → policy → CSV → chấm điểm
  trên dữ liệu giả.

---

## Ghi chú vận hành

- **Truy vấn tiếng Việt, mô hình tiếng Anh.** Nhánh hình ảnh nhận mô tả đã dịch sang
  tiếng Anh; nhánh OCR/ASR giữ nguyên tiếng Việt vì văn bản được index cũng là tiếng Việt.
- **Không có API key vẫn chạy được.** Phân rã truy vấn lùi về rule-based, Q&A chuyển
  sang nhập tay. Mất chất lượng, không mất công cụ.
- **Batch 2.** Các bước index đều cộng dồn theo `video_id`; chạy lại chỉ xử lý video mới.
- **Streamlit trước, FastAPI sau.** Toàn bộ logic ở `src/aic/`; chuyển sang React chỉ
  cần viết lại `app/`.

# RAG Evaluation Results

## Framework sử dụng

RAGAS 0.1.21 — LLM: OpenRouter (Task 10 `LLM_MODEL`), Embeddings: `BAAI/bge-m3` (Task 4)

> **Lưu ý về mẫu đánh giá:** Do giới hạn rate-limit 50 request/ngày của tài khoản OpenRouter free-tier
> (dùng chung cho toàn bộ generation + RAGAS judge calls), kết quả dưới đây chạy trên **3/15 câu** của
> golden dataset thay vì toàn bộ, và một vài job bị `TimeoutError`/`RateLimitError` giữa chừng (xử lý
> gracefully, không crash pipeline nhờ `raise_exceptions=False`). Vì vậy số liệu — đặc biệt
> `faithfulness = 0.000` của config `dense_only` — mang tính minh hoạ cho việc pipeline A/B evaluation
> hoạt động đúng, chưa phải con số đại diện thống kê đầy đủ. Chạy lại không giới hạn (`python -m
> group_project.evaluation.eval_pipeline`) khi có quota rộng hơn để có số liệu đầy đủ trên cả 15 câu.

---

## Overall Scores

| Metric | Config A (hybrid_rerank) | Config B (dense_only) | Δ |
|--------|-------------------------|----------------------|---|
| faithfulness | 0.625 | 0.000 | +0.625 |
| answer_relevancy | 0.729 | 0.767 | -0.038 |
| context_recall | 0.667 | 1.000 | -0.333 |
| context_precision | 0.708 | 0.806 | -0.097 |
| **Average** | **0.682** | **0.643** | **+0.039** |

---

## A/B Comparison Analysis

**Config A (hybrid_rerank):**
> Semantic search (BAAI/bge-m3) + BM25 (rank-bm25) → gộp bằng RRF (k=60) → rerank lại toàn bộ danh sách.

**Config B (dense_only):**
> Chỉ dùng semantic search (dense retrieval), không kết hợp BM25, không rerank.

**Kết luận:** `hybrid_rerank` đạt điểm trung bình cao hơn (0.682 vs 0.643). Việc kết hợp BM25 giúp bắt được các câu hỏi có từ khóa/số liệu chính xác (VD: số ngày, mức %) mà embedding đơn thuần có thể bỏ sót.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Thời giờ làm việc bình thường tối đa trong một ngày là bao n... | 0.75 | 0.84 | 0.00 | Retrieval | Context lấy về thiếu evidence đúng — retriever không tìm ra chunk chứa câu trả lời |
| 2 | Lương thử việc tối thiểu phải bằng bao nhiêu phần trăm lương... | nan | 0.65 | 1.00 | Relevance | Câu trả lời lạc đề so với câu hỏi dù context có thể đủ |
| 3 | Thời gian thử việc tối đa đối với vị trí yêu cầu trình độ đạ... | 0.50 | 0.69 | 1.00 | Relevance | Câu trả lời lạc đề so với câu hỏi dù context có thể đủ |

---

## Recommendations

### Cải tiến 1
**Action:** Áp dụng config `hybrid_rerank` làm mặc định cho production.
**Expected impact:** Cải thiện điểm trung bình ~0.039 so với config còn lại.

### Cải tiến 2
**Action:** Với các câu Retrieval-stage failure, tăng `top_k` hoặc giảm `CHUNK_SIZE` để tăng recall.
**Expected impact:** Giảm tỷ lệ context_recall thấp ở nhóm worst performers.

### Cải tiến 3
**Action:** Với các câu Generation-stage failure, siết chặt SYSTEM_PROMPT (bắt buộc trích dẫn từng câu) hoặc giảm TEMPERATURE.
**Expected impact:** Tăng faithfulness, giảm hiện tượng LLM suy diễn ngoài context.
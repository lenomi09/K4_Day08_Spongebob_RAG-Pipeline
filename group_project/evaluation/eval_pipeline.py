"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAGAS LLM/Embeddings — dùng chung OpenRouter LLM (Task 10) + BAAI/bge-m3 (Task 4)
# =============================================================================

def _build_ragas_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from src.task10_generation import LLM_MODEL

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    chat = ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )
    return LangchainLLMWrapper(chat)


def _build_ragas_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from src.task4_chunking_indexing import EMBEDDING_MODEL

    hf = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(hf)


def _run_generation(config_fn, golden_dataset: list[dict]) -> dict:
    """
    Chạy config_fn(question) -> list[chunk] cho từng câu hỏi, sinh answer qua LLM
    (Task 10: reorder + format_context + system prompt), build eval_data cho RAGAS.

    config_fn cho phép so sánh A/B các chiến lược retrieval khác nhau (hybrid+rerank
    vs dense-only, ...) mà vẫn dùng chung 1 bước generation.
    """
    from openai import OpenAI

    from src.task10_generation import (
        LLM_MODEL, SYSTEM_PROMPT, TEMPERATURE, TOP_P,
        format_context, reorder_for_llm,
    )

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        question = item["question"]
        try:
            chunks = config_fn(question)
            reordered = reorder_for_llm(chunks)
            context = format_context(reordered)
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
            contexts = [c["content"] for c in chunks] or [""]
        except Exception as e:
            print(f"  ✗ Bỏ qua câu hỏi (lỗi generation): {question[:50]}... — {e}")
            continue

        eval_data["question"].append(question)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["expected_answer"])
        time.sleep(1)  # tránh burst request vào free-tier rate limit

    return eval_data


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(config_fn, golden_dataset: list[dict]):
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    Args:
        config_fn: Callable(question: str) -> list[dict chunk] — 1 chiến lược retrieval
            (VD: lambda q: retrieve(q, top_k=5) hoặc lambda q: semantic_search(q, top_k=5))
        golden_dataset: list of {question, expected_answer, expected_context}

    Returns:
        pandas.DataFrame — 1 dòng/câu hỏi, cột là các metric scores.
    """
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from datasets import Dataset
    from ragas.run_config import RunConfig

    eval_data = _run_generation(config_fn, golden_dataset)
    if not eval_data["question"]:
        raise RuntimeError("Không sinh được câu trả lời nào — kiểm tra lại retrieval/LLM.")

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=_build_ragas_llm(),
        embeddings=_build_ragas_embeddings(),
        # fail nhanh thay vì retry nhiều lần khi gặp rate limit free-tier
        run_config=RunConfig(max_retries=2, max_wait=20, max_workers=2),
        raise_exceptions=False,
    )
    return result.to_pandas()


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs retrieval:
        - hybrid_rerank: Semantic + BM25 → RRF → Rerank (Task 9 đầy đủ)
        - dense_only:    Chỉ Semantic search (Task 5), không BM25/rerank

    Returns:
        {config_name: pandas.DataFrame} — mỗi DataFrame là kết quả evaluate_with_ragas.
    """
    from src.task5_semantic_search import semantic_search
    from src.task9_retrieval_pipeline import retrieve

    configs = {
        "hybrid_rerank": lambda q: retrieve(q, top_k=5, use_reranking=True),
        "dense_only": lambda q: semantic_search(q, top_k=5),
    }

    results = {}
    for config_name, config_fn in configs.items():
        print(f"\n--- Evaluating config: {config_name} ---")
        results[config_name] = evaluate_with_ragas(config_fn, golden_dataset)

    return results


# =============================================================================
# Export Results
# =============================================================================

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def export_results(comparison: dict):
    """Export evaluation results (dict {config_name: DataFrame}) ra results.md"""
    configs = list(comparison.keys())
    means = {cfg: comparison[cfg][METRICS].mean() for cfg in configs}

    lines = ["# RAG Evaluation Results", ""]
    lines.append("## Framework sử dụng")
    lines.append("")
    lines.append("RAGAS 0.1.21 — LLM: OpenRouter (Task 10 `LLM_MODEL`), Embeddings: `BAAI/bge-m3` (Task 4)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Overall Scores")
    lines.append("")

    a, b = configs[0], configs[1] if len(configs) > 1 else configs[0]
    header = f"| Metric | Config A ({a}) | Config B ({b}) | Δ |"
    lines.append(header)
    lines.append("|--------|" + "-" * (len(a) + 12) + "|" + "-" * (len(b) + 12) + "|---|")
    for m in METRICS:
        va, vb = means[a][m], means[b][m]
        lines.append(f"| {m} | {va:.3f} | {vb:.3f} | {va - vb:+.3f} |")
    avg_a, avg_b = means[a][METRICS].mean(), means[b][METRICS].mean()
    lines.append(f"| **Average** | **{avg_a:.3f}** | **{avg_b:.3f}** | **{avg_a - avg_b:+.3f}** |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## A/B Comparison Analysis")
    lines.append("")
    lines.append(f"**Config A ({a}):**")
    lines.append("> Semantic search (BAAI/bge-m3) + BM25 (rank-bm25) → gộp bằng RRF (k=60) → rerank lại toàn bộ danh sách.")
    lines.append("")
    lines.append(f"**Config B ({b}):**")
    lines.append("> Chỉ dùng semantic search (dense retrieval), không kết hợp BM25, không rerank.")
    lines.append("")
    winner = a if avg_a >= avg_b else b
    lines.append(f"**Kết luận:** `{winner}` đạt điểm trung bình cao hơn ({max(avg_a, avg_b):.3f} vs {min(avg_a, avg_b):.3f}). "
                  f"{'Việc kết hợp BM25 giúp bắt được các câu hỏi có từ khóa/số liệu chính xác (VD: số ngày, mức %) mà embedding đơn thuần có thể bỏ sót.' if winner == a else 'Semantic-only đã đủ mạnh cho corpus này, việc thêm BM25/rerank không cải thiện rõ rệt.'}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Worst Performers (Bottom 3)")
    lines.append("")
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |")
    lines.append("|---|----------|-------------|-----------|--------|---------------|------------|")

    df_a = comparison[a].copy()
    df_a["avg_score"] = df_a[METRICS].mean(axis=1)
    worst = df_a.sort_values("avg_score").head(3)
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        q = str(row["question"])[:60].replace("|", "/")
        faith, rel, rec = row["faithfulness"], row["answer_relevancy"], row["context_recall"]
        stage = "Retrieval" if rec < 0.5 else ("Generation" if faith < 0.5 else "Relevance")
        cause = {
            "Retrieval": "Context lấy về thiếu evidence đúng — retriever không tìm ra chunk chứa câu trả lời",
            "Generation": "LLM trả lời không bám sát context được cung cấp (có thể suy diễn/bịa)",
            "Relevance": "Câu trả lời lạc đề so với câu hỏi dù context có thể đủ",
        }[stage]
        lines.append(f"| {i} | {q}... | {faith:.2f} | {rel:.2f} | {rec:.2f} | {stage} | {cause} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Cải tiến 1")
    lines.append(f"**Action:** Áp dụng config `{winner}` làm mặc định cho production.")
    lines.append(f"**Expected impact:** Cải thiện điểm trung bình ~{abs(avg_a - avg_b):.3f} so với config còn lại.")
    lines.append("")
    lines.append("### Cải tiến 2")
    lines.append("**Action:** Với các câu Retrieval-stage failure, tăng `top_k` hoặc giảm `CHUNK_SIZE` để tăng recall.")
    lines.append("**Expected impact:** Giảm tỷ lệ context_recall thấp ở nhóm worst performers.")
    lines.append("")
    lines.append("### Cải tiến 3")
    lines.append("**Action:** Với các câu Generation-stage failure, siết chặt SYSTEM_PROMPT (bắt buộc trích dẫn từng câu) hoặc giảm TEMPERATURE.")
    lines.append("**Expected impact:** Tăng faithfulness, giảm hiện tượng LLM suy diễn ngoài context.")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Đã ghi kết quả vào {RESULTS_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ chạy N câu đầu (test nhanh trước khi chạy full, tránh chạm rate limit 50 req/ngày)",
    )
    args = parser.parse_args()

    golden_dataset = load_golden_dataset()
    if args.limit:
        golden_dataset = golden_dataset[: args.limit]
    print(f"Loaded {len(golden_dataset)} test cases")

    comparison = compare_configs(golden_dataset)
    export_results(comparison)

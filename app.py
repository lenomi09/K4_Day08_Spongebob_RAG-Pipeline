"""
RAG Chatbot — Trợ Lý Luật Lao Động
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Trợ Lý Luật Lao Động",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROLE_LABELS = {"employee": "Người lao động", "employer": "Người sử dụng LĐ", "both": "Cả hai bên"}
SOURCE_BADGE = {
    "hybrid": "🔵 Hybrid (Semantic + BM25)",
    "pageindex": "🟠 PageIndex (fallback — hybrid không đủ tin cậy)",
}

# =============================================================================
# HELPERS
# =============================================================================


def _count_indexed_docs() -> int:
    """Đếm số file .md đã chuẩn hoá — dùng để hiện trạng thái nguồn dữ liệu ở sidebar."""
    std_dir = PROJECT_ROOT / "data" / "standardized"
    if not std_dir.exists():
        return 0
    return len(list(std_dir.rglob("*.md")))


def _highlight(text: str, query: str) -> str:
    """In đậm các từ trong content trùng với từ khoá trong câu hỏi (không phân biệt hoa/thường)."""
    words = {w for w in re.findall(r"\w{3,}", query.lower())}
    if not words:
        return text

    def repl(m: re.Match) -> str:
        return f"**{m.group(0)}**" if m.group(0).lower() in words else m.group(0)

    return re.sub(r"\w{3,}", repl, text)


def render_sources(sources: list[dict], retrieval_source: str | None, query: str = ""):
    """Hiển thị khối nguồn tham khảo — dùng chung cho lịch sử chat và câu trả lời mới."""
    if not sources:
        return

    if retrieval_source in SOURCE_BADGE:
        st.caption(SOURCE_BADGE[retrieval_source])

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} đoạn)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            source_name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            role = ROLE_LABELS.get(meta.get("customer_role", "both"), "Cả hai bên")
            score = src.get("score", 0)

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**[{i}] {source_name}** `{doc_type}` · 🏷️ {role}")
            with col2:
                st.caption(f"score `{score:.4f}`")

            content = src.get("content", "")[:300]
            st.markdown(_highlight(content, query) + "...")
            st.divider()


def build_history() -> list[dict]:
    """Lịch sử hội thoại (trừ câu vừa hỏi) để Task 10 hiểu được câu hỏi nối tiếp."""
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]


# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("⚖️ Trợ Lý Luật Lao Động")
    st.caption("Hỏi đáp về thử việc, hợp đồng, tiền lương, làm thêm giờ, nghỉ phép, chấm dứt hợp đồng")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời gian thử việc tối đa là bao lâu?",
        "Lương thử việc tối thiểu bằng bao nhiêu % lương chính thức?",
        "Người lao động được nghỉ phép năm bao nhiêu ngày?",
        "Làm thêm giờ tối đa bao nhiêu giờ một tháng?",
        "Trợ cấp thôi việc được tính như thế nào?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số đoạn văn lấy về (top_k)", 3, 10, 5)
    use_reranking = st.toggle(
        "Bật RRF Rerank",
        value=True,
        help="Tắt để so sánh: hybrid thô (Semantic + BM25 merge) vs có rerank lại — "
        "dùng cho phần đánh giá A/B của bài nhóm.",
    )

    st.divider()
    if st.button("🗑️ Xoá cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    n_docs = _count_indexed_docs()
    st.caption("**Trạng thái hệ thống**")
    st.caption(f"📄 {n_docs} văn bản đã chuẩn hoá & index vào ChromaDB")
    st.caption("**Kiến trúc:** Semantic (`bge-m3`) + BM25 → RRF → Rerank → PageIndex Fallback → LLM có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("⚖️ Trợ Lý Hỏi Đáp Luật Lao Động")
st.caption("Hệ thống RAG trả lời dựa trên Bộ luật Lao động 2019 và các Nghị định hướng dẫn")

if not st.session_state.messages:
    st.info("👋 Chọn câu hỏi gợi ý ở thanh bên hoặc gõ câu hỏi của bạn để bắt đầu.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(
                msg.get("sources", []),
                msg.get("retrieval_source"),
                msg.get("query", ""),
            )

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về luật lao động...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    history=build_history(),
                    use_reranking=use_reranking,
                )
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source")

            except NotImplementedError:
                answer = "⚠️ **Pipeline chưa được implement đầy đủ.** Kiểm tra lại `src/task9_retrieval_pipeline.py` và `src/task10_generation.py`."
                sources, retrieval_source = [], None
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources, retrieval_source = [], None

            st.markdown(answer)
            render_sources(sources, retrieval_source, query)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "query": query,
        }
    )

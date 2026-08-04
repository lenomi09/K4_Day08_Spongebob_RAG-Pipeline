"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_CACHE_DIR = Path(__file__).parent.parent / "data" / "pageindex_pdfs"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

# DejaVu Sans hỗ trợ dấu tiếng Việt — font core của fpdf2 (Helvetica...) thì không.
_UNICODE_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _md_to_pdf(md_path: Path, pdf_path: Path):
    """PageIndex chỉ nhận PDF — convert .md sang PDF đơn giản bằng fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", fname=_UNICODE_FONT_PATH)
    pdf.set_font("DejaVu", size=11)

    text = md_path.read_text(encoding="utf-8")
    pdf.multi_cell(0, 6, text)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def upload_documents(wait_ready: bool = True, timeout_s: int = 300) -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex (convert sang PDF trước).

    Returns:
        dict {md_filename: doc_id} — cũng được cache vào DOC_IDS_FILE để pageindex_search()
        dùng lại mà không phải upload lại mỗi lần chạy.
    """
    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids: dict[str, str] = _load_doc_ids()  # giữ lại doc đã upload thành công ở lần chạy trước

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.name in doc_ids:
            continue  # đã upload rồi, khỏi tốn credit upload lại

        pdf_path = PDF_CACHE_DIR / f"{md_file.stem}.pdf"
        if not pdf_path.exists():
            _md_to_pdf(md_file, pdf_path)

        try:
            resp = client.submit_document(str(pdf_path))
        except Exception as e:
            # Free tier có giới hạn số document/credit — bỏ qua file lỗi, giữ nguyên
            # các doc đã upload thành công thay vì crash toàn bộ hàm.
            print(f"  ✗ Bỏ qua {md_file.name}: {e}")
            continue

        doc_id = resp["doc_id"]
        doc_ids[md_file.name] = doc_id
        DOC_IDS_FILE.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    if wait_ready:
        deadline = time.time() + timeout_s
        pending = set(doc_ids.values())
        while pending and time.time() < deadline:
            pending = {d for d in pending if not client.is_retrieval_ready(d)}
            if pending:
                print(f"  ... chờ {len(pending)} doc xử lý xong (tree + OCR)")
                time.sleep(5)
        if pending:
            print(f"  ⚠ {len(pending)} doc chưa sẵn sàng sau {timeout_s}s, có thể cần chờ thêm")

    return doc_ids


def _load_doc_ids() -> dict[str, str]:
    if not DOC_IDS_FILE.exists():
        return {}
    return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids = _load_doc_ids()
    if not doc_ids:
        raise RuntimeError("Chưa có doc nào trên PageIndex — chạy upload_documents() trước.")

    results = []
    # Mỗi query chỉ scope được 1 doc_id/lần — lặp qua từng doc đã upload rồi gộp lại.
    for filename, doc_id in doc_ids.items():
        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp["retrieval_id"]

        # Ngay sau submit_query, retrieval_id có thể chưa kịp đăng ký ở backend
        # (get_retrieval trả "Retrieval task not found") — retry vài giây trước khi poll status.
        retrieval = None
        for _ in range(10):
            try:
                retrieval = client.get_retrieval(retrieval_id)
                break
            except Exception:
                time.sleep(1)
        if retrieval is None:
            continue

        for _ in range(30):  # poll tối đa ~30s cho tới khi status == "completed"
            if retrieval.get("status") == "completed":
                break
            time.sleep(1)
            retrieval = client.get_retrieval(retrieval_id)

        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    results.append({
                        "content": item.get("relevant_content", ""),
                        # PageIndex không trả similarity score — gán điểm giảm dần theo thứ hạng
                        "score": round(1.0 - 0.01 * len(results), 4),
                        "metadata": {"section": item.get("section_title"), "source": filename},
                        "source": "pageindex",
                    })

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("mức lương tối thiểu vùng hiện nay là bao nhiêu", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")

"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Bai viet ve luat lao dong (thu viec, tang ca, nghi phep, tro cap thoi viec, sa thai)
# Luu y: thuvienphapluat.vn chan bot (crawl ve rong) -> chi dung luatvietnam.vn, da test crawl thanh cong
ARTICLE_URLS = [
    "https://luatvietnam.vn/lao-dong-tien-luong/thoi-gian-thu-viec-la-bao-lau-562-29407-article.html",
    "https://luatvietnam.vn/lao-dong-tien-luong/quy-dinh-thoi-gian-lam-them-gio-562-92730-article.html",
    "https://luatvietnam.vn/lao-dong-tien-luong/cach-tinh-ngay-phep-562-19640-article.html",
    "https://luatvietnam.vn/lao-dong-tien-luong/cach-tinh-tro-cap-thoi-viec-tu-2021-145-562-28112-article.html",
    "https://luatvietnam.vn/lao-dong-tien-luong/don-phuong-cham-dut-hop-dong-lao-dong-562-33360-article.html",
    "https://luatvietnam.vn/tin-phap-luat/hau-qua-khi-don-phuong-cham-dut-hop-dong-lao-dong-trai-phap-luat-230-19047-article.html",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter

    # PruningContentFilter loai bo menu/navbar/footer (boilerplate co mat do chu thap),
    # chi giu lai phan noi dung chinh cua bai viet — neu khong content se lan menu trang web.
    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed")
        )
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)

        markdown = result.markdown
        # fit_markdown = ban da loc boilerplate; raw_markdown = ban day du (menu, footer,...)
        content = getattr(markdown, "fit_markdown", None) or str(markdown or "")

        metadata = result.metadata or {}
        return {
            "url": url,
            "title": metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as e:
            print(f"  LOI, bo qua: {e}")
            continue

        if len(article.get("content_markdown", "").strip()) < 500:
            print(f"  CANH BAO: noi dung qua ngan ({len(article.get('content_markdown', ''))} ky tu) — co the la trang chan bot, can doi URL khac")

        # Lưu file JSON — bat buoc encoding="utf-8", Windows mac dinh cp1252 se loi voi tieng Viet
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  OK: Saved {filepath.name} ({len(article.get('content_markdown', ''))} ky tu)")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())

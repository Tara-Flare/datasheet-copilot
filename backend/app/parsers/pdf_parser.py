
from pathlib import Path

import fitz


class PDFParser:
    """PDF 文档解析器"""

    def parse(self, pdf_path: str) -> dict:
        """
        解析 PDF，返回逐页文本与文本块坐标。
        """

        path = Path(pdf_path)

        if not path.is_file():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        pages = []

        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise ValueError("暂不支持加密 PDF")

            page_count = len(doc)

            for page_index in range(page_count):
                page = doc.load_page(page_index)

                # 提取当前页面的完整文本
                text = page.get_text("text", sort=True)

                # 提取文本块及其页面坐标
                blocks = []

                for block in page.get_text("blocks", sort=True):
                    x0, y0, x1, y1, content, *_ = block

                    if not isinstance(content, str):
                        continue

                    content = content.strip()

                    if not content:
                        continue

                    blocks.append({
                        "text": content,
                        "bbox": [
                            x0, y0, x1, y1
                        ]
                    })

                pages.append({
                    "page": page_index + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "text": text,
                    "blocks": blocks
                })

        return {
            "filename": path.name,
            "page_count": page_count,
            "pages": pages,
            "status": "parsed"
        }


if __name__ == "__main__":
    parser = PDFParser()

    result = parser.parse("data/samples/test.pdf")

    print("文件名:", result["filename"])
    print("总页数:", result["page_count"])

    for page in result["pages"]:
        print(f"\n第 {page['page']} 页")
        print(page["text"][:500])

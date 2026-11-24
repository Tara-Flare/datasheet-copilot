from pathlib import Path
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from app.parsers.raw_structure import (
    build_raw_structure,
    file_sha256,
    parse_config_from_options,
    table_to_raw_structure,
)


def _runtime_device_label(pipeline_options):
    """记录实际运行设备（含强制 CPU 的情况）。"""

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"

    accelerator_options = getattr(
        pipeline_options,
        "accelerator_options",
        None,
    )

    device = getattr(
        accelerator_options,
        "device",
        None,
    )

    if device is None:
        return None

    return getattr(device, "value", None) or str(device)


def _apply_flat_glyph_restoration(
    tables,
    raw_tables,
    pdf_path,
):
    """把 PDF 字形证据应用到 Flat 表格。

    只在 Docling 文本与 PDF 文本仅相差已登记字形时
    修改单元格；原始 Docling 文本写入 docling_cells，
    供审计。没有字形证据时不改动任何表。
    """

    page_numbers = sorted({
        cell.get("page_no")
        for raw_table in raw_tables
        for cell in raw_table.get("cells", [])
        if isinstance(cell.get("page_no"), int)
    })

    if not page_numbers:
        return

    import pymupdf as fitz

    from app.parsers.structure_recovery import (
        extract_page_words,
        restore_flat_table_glyphs,
    )

    try:
        document = fitz.open(str(pdf_path))
    except Exception:  # pragma: no cover - 防御性分支
        return

    try:
        words_by_page = {}
        page_heights = {}

        for page_no in page_numbers:
            if not 1 <= page_no <= document.page_count:
                continue

            try:
                page = document[page_no - 1]
                words_by_page[page_no] = extract_page_words(page)
                page_heights[page_no] = float(page.rect.height)
            except Exception:  # pragma: no cover - 防御性分支
                continue
    finally:
        document.close()

    raw_by_id = {
        raw_table.get("table_id"): raw_table
        for raw_table in raw_tables
    }

    for table in tables:
        raw_table = raw_by_id.get(table.get("table_id"))

        if raw_table is None:
            continue

        restored, evidence = restore_flat_table_glyphs(
            table,
            raw_table,
            words_by_page=words_by_page,
            page_heights=page_heights,
        )

        if not evidence:
            continue

        table["rows"] = restored["rows"]
        table["docling_cells"] = [
            {
                "cell_id": entry["cell_id"],
                "row_index": updated["row_index"],
                "column_index": updated["column_index"],
                "docling_text": entry["docling_text"],
                "pdf_text": entry["pdf_text"],
            }
            for entry in evidence
            for updated in entry["updated_cells"]
        ]
        table["glyph_evidence"] = evidence


class TableParser:
    """Datasheet 表格结构化解析器。"""

    def __init__(self, artifacts_path=None):
        # 第一版只处理具有文本层的 PDF。
        # 扫描版 OCR 在后续模块中启用。
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
        )

        # 离线部署可指向仓库内模型目录；未设置时保持
        # Docling 原有行为（内部自行解析/下载）。
        artifacts = artifacts_path or os.getenv(
            "DOCLING_ARTIFACTS_PATH"
        )

        if artifacts:
            pipeline_options.artifacts_path = str(artifacts)

        # 保留配置，供缓存失效判定使用
        self.pipeline_options = pipeline_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def parse(
        self,
        pdf_path: str,
        include_raw_structure: bool = False,
        page_range=None,
    ) -> dict:
        """解析 PDF 表格。

        默认返回结构与旧版本完全一致。
        include_raw_structure=True 时，
        额外在结果中附加 raw_structure（原始 TableCell 结构），
        现有 rows / columns / table_id / row_count 不受影响。

        page_range 可选（如 (1, 6)），用于限定解析页范围；
        由于 table_id 按 document.tables 顺序编号，
        选择覆盖目标表格的连续前缀页不会改变其 table_id。
        """

        path = Path(pdf_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"PDF 文件不存在: {pdf_path}"
            )

        convert_kwargs = {}

        if page_range is not None:
            convert_kwargs["page_range"] = page_range

        result = self.converter.convert(
            str(path),
            **convert_kwargs,
        )

        document = result.document

        tables = []
        raw_tables = []

        for index, table in enumerate(
            document.tables,
            start=1
        ):
            # Docling TableItem 转 DataFrame
            df = table.export_to_dataframe(
                doc=document
            )

            # 获取表格来源页码
            page_numbers = sorted({
                item.page_no
                for item in table.prov
            })

            # 转换为 JSON 友好的数据
            rows = [
                [
                    None if value is None else str(value)
                    for value in row
                ]
                for row in df.itertuples(
                    index=False,
                    name=None
                )
            ]

            columns = [
                str(col) for col in df.columns
            ]

            table_result = {
                "table_id": index,
                "page_numbers": page_numbers,
                "columns": columns,
                "rows": rows,
                "row_count": len(df),
                "column_count": len(df.columns),
            }

            tables.append(table_result)

            # 在 TableItem 仍然可用时（DataFrame 扁平化不破坏
            # table.data）记录原始单元格结构；
            # 该分支不影响上面的既有字段。
            # B3.8：即使不导出 raw_structure，也保留一份临时
            # 原始结构用于 Flat 单元格式字形恢复。
            raw_tables.append(
                table_to_raw_structure(
                    table,
                    index,
                    columns=columns,
                    flattened_row_count=len(df),
                )
            )

        _apply_flat_glyph_restoration(
            tables, raw_tables, path
        )

        output = {
            "filename": path.name,
            "table_count": len(tables),
            "tables": tables,
            "status": "parsed",
        }

        if include_raw_structure:
            output["raw_structure"] = build_raw_structure(
                raw_tables,
                pdf_path=path,
                pdf_sha256=file_sha256(path),
                parse_config=parse_config_from_options(
                    self.pipeline_options
                ),
                device=_runtime_device_label(
                    self.pipeline_options
                ),
                page_range=page_range,
            )

        return output


if __name__ == "__main__":
    parser = TableParser()

    result = parser.parse(
        "data/samples/test.pdf"
    )

    print("表格数量:", result["table_count"])

    for table in result["tables"]:
        print(
            "表格 ID:",
            table["table_id"]
        )
        print(
            "所在页码:",
            table["page_numbers"]
        )
        print(
            "列名:",
            table["columns"]
        )

        for row in table["rows"][:5]:
            print(row)

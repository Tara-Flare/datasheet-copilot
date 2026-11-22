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


class TableParser:
    """Datasheet 表格结构化解析器。"""

    def __init__(self):
        # 第一版只处理具有文本层的 PDF。
        # 扫描版 OCR 在后续模块中启用。
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
        )

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
            if include_raw_structure:
                raw_tables.append(
                    table_to_raw_structure(
                        table,
                        index,
                        columns=columns,
                        flattened_row_count=len(df),
                    )
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

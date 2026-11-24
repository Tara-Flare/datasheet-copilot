"""B3.5b：离线 Docling 环境辅助（仅脚本层使用）。

生产 TableParser 默认不设置 artifacts_path，离线环境下会尝试
联网下载模型。B3.5b 的预检与基线脚本在不修改生产代码的前提下，
于进程内把 converter 指向仓库内 docling-models 目录。

该调整只影响模型文件加载位置，不改变 do_ocr / do_table_structure
配置，因此缓存 key 与生产完全一致。
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = PROJECT_ROOT / "docling-models"


def configure_environment(device="cpu"):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not os.environ.get("DOCLING_ARTIFACTS_PATH"):
        os.environ["DOCLING_ARTIFACTS_PATH"] = str(
            DEFAULT_ARTIFACTS
        )


def build_offline_table_parser(artifacts_path=None):
    """返回 converter 指向本地模型目录的 TableParser。"""

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )

    from app.parsers.table_parser import TableParser

    artifacts = Path(
        artifacts_path or DEFAULT_ARTIFACTS
    )

    parser = TableParser()
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
    )
    pipeline_options.artifacts_path = str(artifacts)

    parser.pipeline_options = pipeline_options
    parser.converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    return parser

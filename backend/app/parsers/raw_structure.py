"""原始 Docling TableCell 结构持久化（阶段 A）。

只记录 Docling 解析出的原始单元格结构，
不做任何单位继承、数值拆分或结构自动修复。

设计要点：
- 原始结构写入独立文件（带版本号），不覆盖现有表格缓存；
- 现有 rows / columns / table_id / row_index 语义保持不变；
- 通过 header_row_count + row_map 与原扁平表格建立稳定映射；
- 所有信息可追溯至原始 PDF（页码 / bbox / provenance）。
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from app.parsers.table_cache import TableParseCache


RAW_STRUCTURE_VERSION = 1
RAW_STRUCTURE_SCHEMA = "table_raw_structure/v1"

_OFFSET_SEMANTICS = {
    "row_interval": "[start, end)",
    "col_interval": "[start, end)",
    "end_offset": "exclusive",
    "row_span": "end_row_offset_idx - start_row_offset_idx",
    "col_span": "end_col_offset_idx - start_col_offset_idx",
    "verified_against": "docling-core 2.107.1 TableData.grid",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _enum_value(value):
    if value is None:
        return None
    return getattr(value, "value", str(value))


def file_sha256(path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def bbox_to_dict(bbox):
    """序列化 BoundingBox，保留坐标系。"""

    if bbox is None:
        return None

    return {
        "l": float(bbox.l),
        "t": float(bbox.t),
        "r": float(bbox.r),
        "b": float(bbox.b),
        "coord_origin": _enum_value(
            getattr(bbox, "coord_origin", None)
        ),
    }


def _charspan_to_list(charspan):
    if charspan is None:
        return None

    try:
        return [
            int(charspan[0]),
            int(charspan[1]),
        ]
    except (TypeError, ValueError, IndexError):
        return None


def provenance_to_dict(prov):
    """序列化单个 ProvenanceItem。"""

    if prov is None:
        return None

    return {
        "page_no": getattr(prov, "page_no", None),
        "bbox": bbox_to_dict(
            getattr(prov, "bbox", None)
        ),
        "charspan": _charspan_to_list(
            getattr(prov, "charspan", None)
        ),
    }


def table_provenance(table) -> list[dict]:
    """序列化表格级 provenance。"""

    provs = getattr(table, "prov", None) or []

    return [
        provenance_to_dict(prov)
        for prov in provs
        if prov is not None
    ]


def _table_page_numbers(table) -> list[int]:
    pages = []

    for prov in getattr(table, "prov", None) or []:
        page_no = getattr(prov, "page_no", None)

        if page_no is not None and page_no not in pages:
            pages.append(page_no)

    return sorted(pages)


def _table_bbox(table):
    provs = getattr(table, "prov", None) or []

    if not provs:
        return None

    return bbox_to_dict(
        getattr(provs[0], "bbox", None)
    )


def _detect_header_row_count(data):
    """用 Docling 的列头判定计算表头行数（可能失败）。"""

    try:
        grid = data.grid
        return int(
            data._count_column_header_rows(grid=grid)
        )
    except Exception:
        return None


def _ref_value(cell):
    ref = getattr(cell, "ref", None)

    if ref is None:
        return None

    return getattr(ref, "cref", None) or str(ref)


def cell_to_dict(cell, cell_id, page_no=None):
    """序列化单个原始 TableCell。"""

    start_row = int(cell.start_row_offset_idx)
    end_row = int(cell.end_row_offset_idx)
    start_col = int(cell.start_col_offset_idx)
    end_col = int(cell.end_col_offset_idx)

    is_rich = hasattr(cell, "ref")

    return {
        "cell_id": cell_id,
        "text": getattr(cell, "text", "") or "",
        "row_span": int(
            getattr(cell, "row_span", 1) or 1
        ),
        "col_span": int(
            getattr(cell, "col_span", 1) or 1
        ),
        "start_row_offset_idx": start_row,
        "end_row_offset_idx": end_row,
        "start_col_offset_idx": start_col,
        "end_col_offset_idx": end_col,
        "bbox": bbox_to_dict(
            getattr(cell, "bbox", None)
        ),
        "page_no": page_no,
        "column_header": bool(
            getattr(cell, "column_header", False)
        ),
        "row_header": bool(
            getattr(cell, "row_header", False)
        ),
        "row_section": bool(
            getattr(cell, "row_section", False)
        ),
        "is_rich": is_rich,
        "ref": _ref_value(cell),
    }


def _flattened_span(cell, header_row_count):
    """把原始网格行映射到扁平 rows 的 row_index。

    扁平 rows 由 grid[header_row_count:] 生成，
    因此 flattened_row_index = grid_row - header_row_count。
    """

    start = int(cell.start_row_offset_idx)
    end = int(cell.end_row_offset_idx)

    flat_start = start - header_row_count
    flat_end = end - header_row_count

    if flat_end <= 0:
        # 完全位于表头区域
        return None, None

    return max(flat_start, 0), flat_end


def _resolve_header_row_count(
    data,
    flattened_row_count,
    header_row_count,
):
    num_rows = int(data.num_rows)

    if header_row_count is not None:
        return int(header_row_count)

    if flattened_row_count is not None:
        # 扁平表行数 = num_rows - 表头行数，
        # 由此反推与现有 row_index 完全一致的表头偏移。
        return max(num_rows - int(flattened_row_count), 0)

    detected = _detect_header_row_count(data)

    return detected if detected is not None else 0


def table_to_raw_structure(
    table,
    table_id,
    *,
    columns=None,
    flattened_row_count=None,
    header_row_count=None,
):
    """把一个 Docling TableItem 转成原始结构快照。"""

    data = getattr(table, "data", None)

    if data is None:
        raise ValueError(
            f"table_id={table_id} 缺少 Docling TableData"
        )

    num_rows = int(data.num_rows)
    num_cols = int(data.num_cols)

    resolved_header = _resolve_header_row_count(
        data,
        flattened_row_count,
        header_row_count,
    )

    detected_header = _detect_header_row_count(data)

    page_numbers = _table_page_numbers(table)

    # 单元格自身不带页码；单页表格可安全归属，
    # 多页表格标记为不确定，避免误标。
    cell_page_no = (
        page_numbers[0]
        if len(page_numbers) == 1
        else None
    )

    cells = []

    for index, cell in enumerate(data.table_cells):
        cell_dict = cell_to_dict(
            cell,
            index,
            page_no=cell_page_no,
        )

        flat_start, flat_end = _flattened_span(
            cell,
            resolved_header,
        )

        cell_dict["flattened_row_start"] = flat_start
        cell_dict["flattened_row_end"] = flat_end

        cells.append(cell_dict)

    body_row_count = max(
        num_rows - resolved_header,
        0,
    )

    row_map = [
        {
            "row_index": row_index,
            "grid_row": row_index + resolved_header,
        }
        for row_index in range(body_row_count)
    ]

    return {
        "table_id": table_id,
        "page_numbers": page_numbers,
        "page_ambiguous": len(page_numbers) > 1,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "header_row_count": resolved_header,
        "header_row_count_detected": detected_header,
        "mapping_consistent": (
            detected_header is None
            or detected_header == resolved_header
        ),
        "row_count": body_row_count,
        "columns": list(columns) if columns else [],
        "table_bbox": _table_bbox(table),
        "provenance": table_provenance(table),
        "row_map": row_map,
        "cells": cells,
    }


def parse_config_from_options(pipeline_options) -> dict:
    """记录影响解析输出的配置。"""

    if pipeline_options is None:
        return {}

    accelerator_options = getattr(
        pipeline_options,
        "accelerator_options",
        None,
    )

    return {
        "do_ocr": getattr(
            pipeline_options, "do_ocr", None
        ),
        "do_table_structure": getattr(
            pipeline_options,
            "do_table_structure",
            None,
        ),
        "accelerator_device": _enum_value(
            getattr(
                accelerator_options,
                "device",
                None,
            )
        ),
    }


def build_raw_structure(
    raw_tables,
    *,
    pdf_path=None,
    pdf_sha256=None,
    parse_config=None,
    device=None,
    page_range=None,
):
    """构造顶层 raw_structure 文档。"""

    return {
        "schema": RAW_STRUCTURE_SCHEMA,
        "raw_structure_version": RAW_STRUCTURE_VERSION,
        "created_at": _utc_now(),
        "source": {
            "pdf_path": (
                str(pdf_path) if pdf_path else None
            ),
            "pdf_filename": (
                Path(pdf_path).name
                if pdf_path else None
            ),
            "pdf_sha256": pdf_sha256,
            "docling_version": _package_version(
                "docling"
            ),
            "docling_core_version": _package_version(
                "docling-core"
            ),
            "docling_ibm_models_version": (
                _package_version("docling-ibm-models")
            ),
            "parse_config": parse_config or {},
            "device": device,
            "page_range": (
                list(page_range)
                if page_range is not None
                else None
            ),
        },
        "offset_semantics": dict(_OFFSET_SEMANTICS),
        "table_count": len(raw_tables),
        "tables": list(raw_tables),
    }


class RawStructureStore:
    """原始结构独立文件存储。

    文件名基于主缓存 key 派生并带结构版本号，
    因此不会覆盖旧的表格缓存。
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    @classmethod
    def from_env(cls):
        cache_dir = os.getenv("TABLE_CACHE_DIR")

        if not cache_dir:
            return None

        return cls(cache_dir)

    def build_key(
        self,
        pdf_path,
        config_signature,
    ) -> str:
        base_key = TableParseCache(
            str(self.cache_dir)
        ).build_key(pdf_path, config_signature)

        stem = (
            base_key[:-5]
            if base_key.endswith(".json")
            else base_key
        )

        return (
            f"{stem}.raw_structure"
            f".v{RAW_STRUCTURE_VERSION}.json"
        )

    def load(self, key):
        path = self.cache_dir / key

        if not path.is_file():
            return None

        try:
            return json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    def save(self, key, data) -> Path:
        path = self.cache_dir / key

        try:
            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            tmp = path.with_name(
                path.name + ".tmp"
            )

            tmp.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            tmp.replace(path)

        except OSError:
            # 与现有缓存一致：失败不阻断主流程
            return path

        return path


def save_raw_structure(data, path) -> Path:
    """把原始结构写入指定独立文件。"""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)

    return path


def load_raw_structure(path) -> dict:
    """读取原始结构文件。"""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"原始结构文件不存在: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )

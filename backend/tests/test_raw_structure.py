"""原始 TableCell 结构持久化测试（阶段 A）。"""

import json
from pathlib import Path
from types import SimpleNamespace

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    TableCell,
    TableData,
)
from docling_core.types.doc.common.reference import (
    ProvenanceItem,
)

from app.parsers.raw_structure import (
    RAW_STRUCTURE_VERSION,
    RawStructureStore,
    bbox_to_dict,
    build_raw_structure,
    cell_to_dict,
    load_raw_structure,
    save_raw_structure,
    table_to_raw_structure,
)
from app.parsers.table_cache import TableParseCache


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_bbox(
    l=10.0,
    t=20.0,
    r=30.0,
    b=40.0,
    origin=CoordOrigin.TOPLEFT,
):
    return BoundingBox(
        l=l,
        t=t,
        r=r,
        b=b,
        coord_origin=origin,
    )


def make_cell(
    text,
    sr,
    er,
    sc,
    ec,
    *,
    bbox=None,
    row_span=None,
    col_span=None,
    column_header=False,
    row_header=False,
    row_section=False,
):
    return TableCell(
        text=text,
        start_row_offset_idx=sr,
        end_row_offset_idx=er,
        start_col_offset_idx=sc,
        end_col_offset_idx=ec,
        bbox=bbox,
        row_span=(
            row_span if row_span is not None else er - sr
        ),
        col_span=(
            col_span if col_span is not None else ec - sc
        ),
        column_header=column_header,
        row_header=row_header,
        row_section=row_section,
    )


def make_table(
    cells,
    num_rows,
    num_cols,
    *,
    page_no=6,
    with_prov=True,
):
    data = TableData(
        table_cells=cells,
        num_rows=num_rows,
        num_cols=num_cols,
    )

    prov = []

    if with_prov:
        prov = [
            ProvenanceItem(
                page_no=page_no,
                bbox=make_bbox(),
                charspan=(0, 0),
            )
        ]

    return SimpleNamespace(data=data, prov=prov)


def test_cell_serialization_roundtrip():
    cell = make_cell(
        "mV",
        2,
        5,
        1,
        2,
        bbox=make_bbox(),
        column_header=False,
    )

    data = cell_to_dict(cell, 7, page_no=6)

    restored = json.loads(
        json.dumps(data, ensure_ascii=False)
    )

    assert restored == data
    assert restored["cell_id"] == 7
    assert restored["text"] == "mV"
    assert restored["row_span"] == 3
    assert restored["col_span"] == 1
    assert restored["start_row_offset_idx"] == 2
    assert restored["end_row_offset_idx"] == 5
    assert restored["bbox"]["coord_origin"] == "TOPLEFT"


def test_span_offset_bbox_preserved():
    header = make_cell(
        "UNIT", 0, 1, 1, 2,
        bbox=make_bbox(t=0.0, b=10.0),
        column_header=True,
    )
    mv = make_cell(
        "mV", 1, 4, 1, 2,
        bbox=make_bbox(t=10.0, b=40.0),
        row_span=3,
        col_span=1,
    )

    table = make_table(
        [header, mv],
        num_rows=4,
        num_cols=2,
    )

    raw = table_to_raw_structure(
        table,
        9,
        columns=["PARAM", "UNIT"],
        flattened_row_count=3,
    )

    assert raw["table_id"] == 9
    assert raw["num_rows"] == 4
    assert raw["num_cols"] == 2
    assert raw["header_row_count"] == 1
    assert raw["row_count"] == 3

    mv_raw = next(
        c for c in raw["cells"]
        if c["text"] == "mV"
    )

    assert mv_raw["row_span"] == 3
    assert mv_raw["col_span"] == 1
    assert mv_raw["start_row_offset_idx"] == 1
    assert mv_raw["end_row_offset_idx"] == 4
    assert mv_raw["start_col_offset_idx"] == 1
    assert mv_raw["end_col_offset_idx"] == 2
    assert mv_raw["bbox"]["l"] == 10.0
    assert mv_raw["bbox"]["t"] == 10.0
    assert mv_raw["bbox"]["r"] == 30.0
    assert mv_raw["bbox"]["b"] == 40.0
    assert mv_raw["bbox"]["coord_origin"] == "TOPLEFT"


def test_end_offset_is_exclusive():
    mv = make_cell(
        "mV", 1, 4, 1, 2,
        row_span=3,
    )

    data = TableData(
        table_cells=[mv],
        num_rows=4,
        num_cols=2,
    )

    # grid 覆盖 [start, end) = 1,2,3，不含 4
    assert [
        data.grid[i][1].text
        for i in range(1, 4)
    ] == ["mV", "mV", "mV"]

    raw = cell_to_dict(mv, 0, page_no=6)

    assert (
        raw["row_span"]
        == raw["end_row_offset_idx"]
        - raw["start_row_offset_idx"]
    )


def test_table_id_and_row_index_mapping_unchanged():
    header = make_cell(
        "UNIT", 0, 1, 1, 2,
        column_header=True,
    )
    body = make_cell(
        "mV", 1, 2, 1, 2,
    )

    table = make_table(
        [header, body],
        num_rows=4,
        num_cols=2,
    )

    raw = table_to_raw_structure(
        table,
        9,
        columns=["PARAM", "UNIT"],
        flattened_row_count=3,
    )

    # table_id 保持 Docling enumerate 语义
    assert raw["table_id"] == 9

    # 扁平 row_index 0..2 -> 原始 grid row 1..3
    assert raw["row_map"] == [
        {"row_index": 0, "grid_row": 1},
        {"row_index": 1, "grid_row": 2},
        {"row_index": 2, "grid_row": 3},
    ]

    body_raw = next(
        c for c in raw["cells"]
        if c["text"] == "mV"
    )

    assert body_raw["flattened_row_start"] == 0
    assert body_raw["flattened_row_end"] == 1


def test_raw_structure_source_linkage():
    cell = make_cell(
        "VIN", 1, 2, 0, 1,
        bbox=make_bbox(),
    )

    table = make_table([cell], num_rows=2, num_cols=1)

    raw_table = table_to_raw_structure(
        table,
        7,
        columns=["PARAM"],
        flattened_row_count=1,
    )

    raw = build_raw_structure(
        [raw_table],
        pdf_path="/tmp/TPS7A02.pdf",
        pdf_sha256="abc123",
        parse_config={
            "do_ocr": False,
            "do_table_structure": True,
        },
    )

    assert raw["schema"] == "table_raw_structure/v1"
    assert (
        raw["raw_structure_version"]
        == RAW_STRUCTURE_VERSION
    )
    assert raw["source"]["pdf_filename"] == "TPS7A02.pdf"
    assert raw["source"]["pdf_sha256"] == "abc123"
    assert raw["source"]["parse_config"]["do_ocr"] is False

    table_out = raw["tables"][0]

    assert table_out["table_id"] == 7
    assert table_out["page_numbers"] == [6]
    assert table_out["provenance"][0]["page_no"] == 6
    assert (
        table_out["provenance"][0]["bbox"]["coord_origin"]
        == "TOPLEFT"
    )
    assert table_out["cells"][0]["page_no"] == 6
    assert table_out["cells"][0]["bbox"] is not None

    # 版本信息存在（具体值随环境变化）
    assert "docling_version" in raw["source"]
    assert "docling_core_version" in raw["source"]


def test_missing_bbox_or_provenance_is_safe():
    cell = make_cell(
        "no_bbox", 0, 1, 0, 1,
        bbox=None,
    )

    table = make_table(
        [cell],
        num_rows=1,
        num_cols=1,
        with_prov=False,
    )

    raw = table_to_raw_structure(
        table,
        1,
        columns=["X"],
        flattened_row_count=0,
    )

    assert raw["table_bbox"] is None
    assert raw["provenance"] == []
    assert raw["page_numbers"] == []
    assert raw["page_ambiguous"] is False
    assert raw["cells"][0]["bbox"] is None
    assert raw["cells"][0]["page_no"] is None

    # bbox_to_dict 也要安全处理 None
    assert bbox_to_dict(None) is None


def test_old_cache_is_readable_and_unchanged(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 old-cache-test")

    cache = TableParseCache(str(tmp_path / "cache"))
    key = cache.build_key(
        str(pdf),
        {"do_ocr": False, "do_table_structure": True},
    )

    old_result = {
        "filename": "sample.pdf",
        "table_count": 1,
        "tables": [
            {
                "table_id": 1,
                "page_numbers": [6],
                "columns": ["A"],
                "rows": [["x"]],
                "row_count": 1,
                "column_count": 1,
            }
        ],
        "status": "parsed",
    }

    cache.save(key, old_result)

    loaded = cache.load(key)

    assert loaded == old_result
    # 旧缓存没有 raw_structure，仍可正常读取
    assert "raw_structure" not in loaded

    # 新逻辑不会修改传入的旧结果对象
    before = json.loads(
        json.dumps(old_result, ensure_ascii=False)
    )
    build_raw_structure(
        [],
        pdf_path=str(pdf),
        pdf_sha256="x",
    )
    assert old_result == before


def test_real_cache_has_no_raw_structure_field():
    cache_dir = PROJECT_ROOT / "data" / "cache" / "docling"

    if not cache_dir.is_dir():
        return

    files = sorted(cache_dir.glob("*.json"))

    if not files:
        return

    data = json.loads(
        files[0].read_text(encoding="utf-8")
    )

    assert "tables" in data

    for table in data["tables"]:
        # 旧缓存字段保持原样
        assert "table_id" in table
        assert "rows" in table
        assert "columns" in table
        # 旧缓存不包含 raw_structure
        assert "cells" not in table


def test_raw_structure_store_uses_separate_versioned_key(
    tmp_path,
):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 store-test")

    config = {
        "do_ocr": False,
        "do_table_structure": True,
    }

    base_cache = TableParseCache(str(tmp_path))
    base_key = base_cache.build_key(str(pdf), config)

    store = RawStructureStore(str(tmp_path))
    raw_key = store.build_key(str(pdf), config)

    assert raw_key != base_key
    assert raw_key.endswith(
        f".raw_structure.v{RAW_STRUCTURE_VERSION}.json"
    )
    assert raw_key.startswith(base_key[:-5])

    payload = {"schema": "table_raw_structure/v1", "tables": []}

    store.save(raw_key, payload)

    assert store.load(raw_key) == payload
    # 旧缓存文件未被写入/覆盖
    assert base_cache.load(base_key) is None


def test_table_parser_default_output_shape_unchanged(
    tmp_path,
):
    """include_raw_structure=False 时输出与旧版本完全一致。"""

    import pandas as pd

    from app.parsers import table_parser as table_parser_module

    pdf = tmp_path / "dummy.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    cell = make_cell(
        "x", 1, 2, 0, 1, bbox=make_bbox()
    )
    data = TableData(
        table_cells=[cell],
        num_rows=2,
        num_cols=2,
    )

    class FakeTable:
        def __init__(self):
            self.data = data
            self.prov = [
                ProvenanceItem(
                    page_no=6,
                    bbox=make_bbox(),
                    charspan=(0, 0),
                )
            ]

        def export_to_dataframe(self, doc=None):
            return pd.DataFrame(
                [["x", "y"]],
                columns=["A", "B"],
            )

    class FakeDocument:
        tables = [FakeTable()]

    class FakeResult:
        document = FakeDocument()

    class FakeConverter:
        def convert(self, path, **kwargs):
            return FakeResult()

    parser = table_parser_module.TableParser.__new__(
        table_parser_module.TableParser
    )
    parser.pipeline_options = None
    parser.converter = FakeConverter()

    old = parser.parse(str(pdf))

    assert "raw_structure" not in old
    assert set(old.keys()) == {
        "filename",
        "table_count",
        "tables",
        "status",
    }
    assert old["tables"][0]["columns"] == ["A", "B"]
    assert old["tables"][0]["row_count"] == 1

    new = parser.parse(
        str(pdf),
        include_raw_structure=True,
    )

    assert "raw_structure" in new
    assert new["raw_structure"]["table_count"] == 1
    # 既有扁平行数据不受新字段影响
    assert new["tables"] == old["tables"]


def test_save_and_load_raw_structure_file(tmp_path):
    payload = {
        "schema": "table_raw_structure/v1",
        "tables": [{"table_id": 9, "cells": []}],
    }

    path = tmp_path / "raw.json"

    saved = save_raw_structure(payload, path)

    assert saved == path
    assert load_raw_structure(path) == payload

"""阶段 B1 结构化恢复测试。"""

import copy
import hashlib
import json
from pathlib import Path

import pymupdf as fitz
import pytest

from app.parsers.structure_recovery import (
    extract_page_geometry,
    recover_from_pdf,
    recover_structure,
    to_topleft,
)


pytestmark = pytest.mark.requires_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.raw_structure.v1.json"
)
PDF_PATH = PROJECT_ROOT / "data/samples/TPS7A02.pdf"
CACHE_DIR = PROJECT_ROOT / "data/cache/docling"

if not RAW_PATH.is_file():
    pytest.skip(
        "缺少本地 data/ fixtures；见 docs/DATA_LAYOUT.md",
        allow_module_level=True,
    )

# 在模块导入时（任何恢复调用之前）记录原始文件哈希
RAW_SHA_AT_IMPORT = hashlib.sha256(
    RAW_PATH.read_bytes()
).hexdigest()
CACHE_SHAS_AT_IMPORT = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in CACHE_DIR.glob("*.json")
}


@pytest.fixture(scope="module")
def raw_structure():
    return json.loads(RAW_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recovered(raw_structure):
    return recover_from_pdf(
        raw_structure, 9, str(PDF_PATH), 6
    )


def rows_by_id(result):
    return {
        row["logical_row_id"]: row
        for row in result["logical_rows"]
    }


def test_normal_row_unchanged(recovered):
    """1. 正常表格行保持不变。"""

    row = rows_by_id(recovered)["t9.p19"]

    assert row["parameter_symbol"] == "VDO"
    assert row["parameter_name"] == "Dropout voltage(4)"
    assert "1.0 V ≤ VOUT < 1.2 V" in row["test_conditions"]
    assert row["max"] == "790"
    assert row["unit"] == "mV"
    assert row["needs_review"] is False


def test_unit_span_recovery(recovered):
    """2. 单位跨行恢复。"""

    rows = rows_by_id(recovered)

    for row_id in ("t9.p8", "t9.p9"):
        assert rows[row_id]["unit"] == "nA"
        assert (
            "unit_span_recovery"
            in rows[row_id]["recovery_actions"]
        )

    for row_id in ("t9.p10", "t9.p11", "t9.p12"):
        assert rows[row_id]["unit"] == "%"
        assert (
            "unit_span_recovery"
            in rows[row_id]["recovery_actions"]
        )

    for row_id in ("t9.p18", "t9.p30", "t9.p31"):
        assert rows[row_id]["unit"] == "mV"
        assert (
            "unit_span_recovery"
            in rows[row_id]["recovery_actions"]
        )


def test_no_cross_parameter_unit_propagation(recovered):
    """3. 禁止跨参数区域错误传播。"""

    rows = rows_by_id(recovered)

    # nA 不跨到 IGND/IOUT
    assert rows["t9.p9"]["unit"] == "nA"
    assert rows["t9.p10"]["unit"] == "%"

    # mV 不跨到 PSRR / VN
    assert rows["t9.p32"]["unit"] == "dB"
    assert rows["t9.p33"]["unit"] == "µVRMS"

    # VDO 的 mV 不跨到相邻的 IGND(DO)
    assert rows["t9.p13"]["unit"] == "nA"


def test_multi_value_split(recovered):
    """4. 360/310 正确拆分。"""

    rows = rows_by_id(recovered)

    first = rows["t9.p30"]
    second = rows["t9.p31"]

    assert first["max"] == "360"
    assert second["max"] == "310"
    assert "2.5 V ≤ VOUT < 3.3 V" in first["test_conditions"]
    assert "3.3 V ≤ VOUT ≤ 5.0 V" in second["test_conditions"]
    assert first["unit"] == second["unit"] == "mV"

    # 保留共同的原始 Cell ID
    assert 120 in first["source_cell_ids"]
    assert 121 in first["source_cell_ids"]
    assert 120 in second["source_cell_ids"]
    assert 121 in second["source_cell_ids"]

    assert first["needs_review"] is False
    assert second["needs_review"] is False


def test_multi_value_count_mismatch_rejects_split(
    raw_structure,
):
    """5. 两值/条件数量不一致时拒绝拆分。"""

    modified = copy.deepcopy(raw_structure)
    table = next(
        item for item in modified["tables"]
        if item["table_id"] == 9
    )
    cell = next(
        item for item in table["cells"]
        if item["cell_id"] == 121
    )
    cell["text"] = "360 310 999"

    result = recover_from_pdf(
        modified, 9, str(PDF_PATH), 6
    )

    issues = [
        issue for issue in result["unresolved_issues"]
        if 121 in (issue.get("source_cell_ids") or [])
    ]

    assert issues
    assert any(
        "不一致" in issue["reason"]
        for issue in issues
    )

    rows = rows_by_id(result)

    assert rows["t9.p30"]["needs_review"] is True
    assert rows["t9.p31"]["needs_review"] is True


def test_ground_current_realignment(recovered):
    """6. Ground Current 错行纠正。"""

    rows = rows_by_id(recovered)

    # IGND: TJ=-40~+85 只有 MAX=60 nA
    assert rows["t9.p9"]["max"] == "60"
    assert rows["t9.p9"]["typ"] == ""
    assert rows["t9.p9"]["unit"] == "nA"
    assert "–40°C to +85°C" in rows["t9.p9"][
        "test_conditions"
    ]

    # IGND/IOUT: 5 µA–1 mA 的 TYP=1 %
    assert rows["t9.p10"]["typ"] == "1"
    assert rows["t9.p10"]["unit"] == "%"
    assert "5 µA ≤ IOUT < 1 mA" in rows["t9.p10"][
        "test_conditions"
    ]

    assert rows["t9.p11"]["typ"] == "0.25"
    assert rows["t9.p12"]["typ"] == "0.15"

    assert (
        "ground_current_realignment"
        in rows["t9.p9"]["recovery_actions"]
    )


def test_condition_fragment_merge(recovered):
    """7. IOUT ≥ 100 mA 条件片段恢复。"""

    row = rows_by_id(recovered)["t9.p12"]

    assert "IOUT ≥ 100 mA" in row["test_conditions"]
    assert (
        "condition_fragment_merge"
        in row["recovery_actions"]
    )


def test_source_cell_ids_traceable(recovered):
    """8. source_cell_ids 可追溯。"""

    rows = rows_by_id(recovered)

    assert 47 in rows["t9.p8"]["source_cell_ids"]
    assert 49 in rows["t9.p9"]["source_cell_ids"]
    assert 54 in rows["t9.p10"]["source_cell_ids"]
    assert 121 in rows["t9.p30"]["source_cell_ids"]

    for row_id in (
        "t9.p8", "t9.p9", "t9.p10",
        "t9.p11", "t9.p12", "t9.p30", "t9.p31",
    ):
        row = rows[row_id]

        assert row["source_cell_ids"]
        assert row["source_word_bboxes"]

        for item in row["source_word_bboxes"]:
            assert item["text"]
            assert len(item["bbox"]) == 4


def test_coordinate_conversion():
    """9. 坐标系转换正确。"""

    bottom_left = {
        "l": 10,
        "t": 90,
        "r": 50,
        "b": 20,
        "coord_origin": "BOTTOMLEFT",
    }

    assert to_topleft(bottom_left, 100) == (
        10,
        10,
        50,
        80,
    )

    top_left = {
        "l": 10,
        "t": 10,
        "r": 50,
        "b": 80,
        "coord_origin": "TOPLEFT",
    }

    assert to_topleft(top_left, 100) == (
        10,
        10,
        50,
        80,
    )


def test_ambiguous_unit_region_needs_review(
    raw_structure,
):
    """10. 无法确定时进入 needs_review。"""

    document = fitz.open(str(PDF_PATH))
    page = document[5]

    words, horizontal, vertical = extract_page_geometry(
        page
    )

    # 在 nA 单位区域内注入第二个不同单位，制造歧义
    words.append({
        "text": "mA",
        "bbox": (531.0, 250.0, 546.0, 260.0),
    })

    result = recover_structure(
        raw_structure,
        9,
        words=words,
        horizontal_segments=horizontal,
        vertical_segments=vertical,
        page_height=float(page.rect.height),
        page_no=6,
    )

    assert any(
        "多个不同单位" in issue["reason"]
        for issue in result["unresolved_issues"]
    )

    rows = rows_by_id(result)

    assert rows["t9.p8"]["needs_review"] is True
    assert rows["t9.p8"]["unit"] == ""
    assert rows["t9.p9"]["needs_review"] is True


def test_raw_structure_and_cache_unchanged(recovered):
    """11. raw_structure 与旧缓存保持不变。"""

    assert (
        hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
        == RAW_SHA_AT_IMPORT
    )

    current = {
        path.name: hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in CACHE_DIR.glob("*.json")
    }

    assert current == CACHE_SHAS_AT_IMPORT


def test_deterministic(raw_structure):
    """12. 同一输入重复运行得到相同结果。"""

    first = recover_from_pdf(
        raw_structure, 9, str(PDF_PATH), 6
    )
    second = recover_from_pdf(
        raw_structure, 9, str(PDF_PATH), 6
    )

    assert first == second

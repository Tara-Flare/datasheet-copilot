"""阶段 B3.1：logical_rows -> LLM 输入适配器测试。"""

import copy
import inspect
import json
from pathlib import Path

import pytest

from app.extraction import logical_table_adapter
from app.extraction.deepseek_extractor import (
    DeepSeekExtractor,
)
from app.extraction.logical_table_adapter import (
    LOGICAL_INPUT_COLUMNS,
    LogicalTableAdapterError,
    adapt_logical_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGICAL_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.logical_rows.v1.json"
)
RAW_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.raw_structure.v1.json"
)
V2_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.logical_rows.v2.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inputs():
    return {
        "logical_rows": load(LOGICAL_PATH),
        "raw_structure": load(RAW_PATH),
    }


@pytest.fixture(scope="module")
def adapted(inputs):
    return adapt_logical_rows(
        inputs["logical_rows"],
        raw_structure=inputs["raw_structure"],
        table_ids=[9],
    )


def records_by_id(adapted):
    table = adapted["tables"][0]
    return {
        record["logical_row_id"]: record
        for record in table["records"]
    }


def logical_by_id(inputs):
    return {
        row["logical_row_id"]: row
        for row in inputs["logical_rows"]["logical_rows"]
    }


def test_normal_logical_row_conversion(adapted):
    """1. 正常逻辑行转换。"""

    record = records_by_id(adapted)["t9.p11"]

    assert record["table_id"] == 9
    assert record["page"] == 6
    assert record["parameter_symbol"] == "IGND/IOUT"
    assert (
        record["parameter_name"]
        == "Ground current vs load current"
    )
    assert "1 mA ≤ IOUT < 100 mA" in record[
        "test_conditions"
    ]
    assert record["min"] == ""
    assert record["typ"] == "0.25"
    assert record["max"] == ""
    assert record["unit"] == "%"
    assert record["needs_review"] is False
    assert record["source_row_indices"] == [10]
    assert record["source_cell_ids"]

    # 必填列齐全
    required = {
        "logical_row_id",
        "table_id",
        "page",
        "parameter_name",
        "test_conditions",
        "min",
        "typ",
        "max",
        "unit",
        "source_row_indices",
        "source_cell_ids",
        "needs_review",
    }
    assert required <= set(LOGICAL_INPUT_COLUMNS)


def test_unit_span_recovery_input(adapted):
    """2. 单位跨行恢复后的输入。"""

    records = records_by_id(adapted)

    assert records["t9.p8"]["unit"] == "nA"
    assert records["t9.p9"]["unit"] == "nA"

    for row_id in ("t9.p10", "t9.p11", "t9.p12"):
        assert records[row_id]["unit"] == "%"

    for row_id in ("t9.p18", "t9.p24", "t9.p30"):
        assert records[row_id]["unit"] == "mV"


def test_complete_test_conditions_preserved(inputs, adapted):
    """3. 完整测试条件保留。"""

    logical = logical_by_id(inputs)["t9.p26"]
    record = records_by_id(adapted)["t9.p26"]

    assert (
        record["test_conditions"]
        == logical["test_conditions"]
    )
    assert "TJ = –40°C to +125°C" in record[
        "test_conditions"
    ]
    assert "1.0 V ≤ VOUT < 1.2 V" in record[
        "test_conditions"
    ]


def test_360_310_split_into_two_inputs(adapted):
    """4. 360/310 拆分为两条独立输入。"""

    records = records_by_id(adapted)

    first = records["t9.p30"]
    second = records["t9.p31"]

    assert first["max"] == "360"
    assert second["max"] == "310"
    assert "2.5 V ≤ VOUT < 3.3 V" in first[
        "test_conditions"
    ]
    assert "3.3 V ≤ VOUT ≤ 5.0 V" in second[
        "test_conditions"
    ]

    # 两条都来自原始 row_index 30，但逻辑行 ID 不同
    assert first["primary_row_index"] == 30
    assert second["primary_row_index"] == 30
    assert (
        first["logical_row_id"]
        != second["logical_row_id"]
    )


def test_ground_current_identity(adapted):
    """5. Ground Current 参数身份正确。"""

    records = records_by_id(adapted)

    ground = records["t9.p9"]
    load = records["t9.p10"]

    assert ground["parameter_symbol"] == "IGND"
    assert ground["parameter_name"] == "Ground current"
    assert ground["max"] == "60"
    assert ground["unit"] == "nA"

    assert load["parameter_symbol"] == "IGND/IOUT"
    assert (
        load["parameter_name"]
        == "Ground current vs load current"
    )
    assert load["typ"] == "1"
    assert load["unit"] == "%"

    assert (
        ground["logical_row_id"]
        != load["logical_row_id"]
    )


def test_needs_review_not_trusted(inputs):
    """6. needs_review 禁止自动信任。"""

    adapted = adapt_logical_rows(
        inputs["logical_rows"],
        raw_structure=inputs["raw_structure"],
        table_ids=[9],
    )

    table = adapted["tables"][0]
    trusted_ids = {
        record["logical_row_id"]
        for record in table["records"]
    }

    excluded_ids = {
        row["logical_row_id"]
        for row in table["excluded_rows"]
    }

    for row_id in ("t9.p34", "t9.p35", "t9.p36"):
        assert row_id not in trusted_ids
        assert row_id in excluded_ids

    assert any(
        "needs_review" in row["reason"]
        for row in table["excluded_rows"]
    )

    # 显式包含时仍保留 needs_review 标记
    included = adapt_logical_rows(
        inputs["logical_rows"],
        raw_structure=inputs["raw_structure"],
        table_ids=[9],
        include_needs_review=True,
    )
    included_records = records_by_id(included)

    assert included_records["t9.p34"]["needs_review"] is True


def test_row_index_mapping_unchanged(inputs, adapted):
    """7. 原始 row_index 映射不变。"""

    logical = logical_by_id(inputs)
    table = adapted["tables"][0]

    for record in table["records"]:
        source_row_indices = logical[
            record["logical_row_id"]
        ]["source_row_indices"]

        assert (
            record["source_row_indices"]
            == source_row_indices
        )
        assert record["primary_row_index"] in source_row_indices
        assert isinstance(record["primary_row_index"], int)

        # logical_row_id 不是 row_index
        assert record["logical_row_id"] != str(
            record["primary_row_index"]
        )

    records = records_by_id(adapted)

    assert records["t9.p12"]["primary_row_index"] == 12
    assert (
        records["t9.p12"]["source_row_indices"]
        == [10, 11, 12]
    )

    # row_indices 是原始行号数组，不是 logical_row_id
    assert all(
        isinstance(value, int)
        for value in table["row_indices"]
    )
    assert table["row_indices"].count(30) == 2


def test_source_cell_ids_preserved(inputs, adapted):
    """8. source_cell_ids 完整保留。"""

    logical = logical_by_id(inputs)
    table = adapted["tables"][0]

    for record in table["records"]:
        assert (
            record["source_cell_ids"]
            == logical[record["logical_row_id"]][
                "source_cell_ids"
            ]
        )

    source_map = {
        item["logical_row_id"]: item
        for item in table["source_map"]
    }

    assert source_map["t9.p30"]["source_cell_ids"] == [
        91,
        92,
        96,
        120,
        121,
    ]


def test_legacy_input_compatibility(inputs, adapted):
    """9. Legacy 输入兼容。"""

    before = copy.deepcopy(inputs["logical_rows"])

    adapt_logical_rows(
        inputs["logical_rows"],
        raw_structure=inputs["raw_structure"],
        table_ids=[9],
    )

    # 适配器不修改输入
    assert inputs["logical_rows"] == before

    # 旧 extractor 的表格载荷形状保持不变
    extractor = DeepSeekExtractor.__new__(
        DeepSeekExtractor
    )
    table = {
        "table_id": 1,
        "page_numbers": [5],
        "columns": ["A", "B"],
        "column_count": 2,
    }
    payload = extractor._build_table_payload(
        table,
        [["x", "y"]],
        [0],
    )

    assert set(payload.keys()) == {
        "table_id",
        "page_numbers",
        "columns",
        "rows",
        "row_indices",
        "row_count",
        "column_count",
    }
    assert adapted["mode"] == "logical_rows"


def test_deterministic(inputs):
    """10. 重复转换结果一致。"""

    first = adapt_logical_rows(
        inputs["logical_rows"],
        raw_structure=inputs["raw_structure"],
        table_ids=[9],
    )
    second = adapt_logical_rows(
        inputs["logical_rows"],
        raw_structure=inputs["raw_structure"],
        table_ids=[9],
    )

    assert first == second


def test_unmappable_rows_excluded(inputs):
    """11. 无法映射的行进入待审核。"""

    logical_rows = copy.deepcopy(inputs["logical_rows"])

    broken = copy.deepcopy(
        logical_rows["logical_rows"][0]
    )
    broken["logical_row_id"] = "t9.broken"
    broken["source_row_indices"] = []
    logical_rows["logical_rows"].append(broken)

    adapted = adapt_logical_rows(
        logical_rows,
        raw_structure=inputs["raw_structure"],
        table_ids=[9],
    )

    table = adapted["tables"][0]
    excluded = {
        row["logical_row_id"]: row
        for row in table["excluded_rows"]
    }

    assert "t9.broken" in excluded
    assert "source_row_indices" in excluded[
        "t9.broken"
    ]["reason"]

    assert all(
        record["logical_row_id"] != "t9.broken"
        for record in table["records"]
    )

    with pytest.raises(LogicalTableAdapterError):
        adapt_logical_rows(
            inputs["logical_rows"],
            raw_structure=inputs["raw_structure"],
            table_ids=[999],
        )


def test_no_real_llm():
    """12. 不调用真实 LLM。"""

    source = inspect.getsource(logical_table_adapter)

    for token in (
        "openai",
        "deepseek_extractor",
        "requests",
        "http",
        "chat.completions",
    ):
        assert token not in source


# ---------------------------------------------------------------
# B3.2b：UVLO v2 人工确认后的 Adapter 行为
# ---------------------------------------------------------------

def test_v2_uvlo_rows_not_split_or_merged():
    """13. v2 中三条 UVLO 记录独立且不再待审核。"""

    v2 = load(V2_PATH)
    raw = load(RAW_PATH)

    adapted = adapt_logical_rows(
        v2,
        raw_structure=raw,
        table_ids=[9],
    )

    assert adapted["row_count"] == 36
    assert adapted["excluded_count"] == 0

    table = adapted["tables"][0]
    records = {
        record["logical_row_id"]: record
        for record in table["records"]
    }

    uvlo_ids = [
        row_id
        for row_id in records
        if row_id in {"t9.p34", "t9.p35", "t9.p36"}
    ]

    assert uvlo_ids == [
        "t9.p34",
        "t9.p35",
        "t9.p36",
    ]

    p34 = records["t9.p34"]
    p35 = records["t9.p35"]
    p36 = records["t9.p36"]

    # 三条记录必须是三个独立逻辑行，不能互相合并。
    assert len({
        p34["logical_row_id"],
        p35["logical_row_id"],
        p36["logical_row_id"],
    }) == 3

    assert p34["needs_review"] is False
    assert p35["needs_review"] is False
    assert p36["needs_review"] is False

    assert p34["test_conditions"] == "VIN rising"
    assert p35["test_conditions"] == "VIN falling"
    assert p36["test_conditions"] == "VIN hysteresis"

    assert p34["primary_row_index"] == 33
    assert p35["primary_row_index"] == 34
    assert p36["primary_row_index"] == 34

    assert p34["source_row_indices"] == [33]
    assert p35["source_row_indices"] == [33, 34]
    assert p36["source_row_indices"] == [34]

    assert p34["min"] == "1.23"
    assert p34["typ"] == "1.3"
    assert p34["max"] == "1.47"

    assert p35["min"] == "1.0"
    assert p35["typ"] == "1.12"
    assert p35["max"] == "1.41"

    assert p36["min"] == ""
    assert p36["typ"] == "180"
    assert p36["max"] == ""


def test_v2_uvlo_source_cells_and_unit_inheritance():
    """14. v2 修复 UVLO 来源 cell、行号和单位继承。"""

    v2 = load(V2_PATH)
    raw = load(RAW_PATH)

    adapted = adapt_logical_rows(
        v2,
        raw_structure=raw,
        table_ids=[9],
    )
    table = adapted["tables"][0]
    records = {
        record["logical_row_id"]: record
        for record in table["records"]
    }

    p34 = records["t9.p34"]
    p35 = records["t9.p35"]
    p36 = records["t9.p36"]

    assert p34["unit"] == "V"
    assert p35["unit"] == "V"
    assert p36["unit"] == "mV"

    assert 138 in p34["source_cell_ids"]
    assert 138 in p35["source_cell_ids"]
    assert 145 in p36["source_cell_ids"]

    assert p35["unit_source"] == {
        "cell_id": 138,
        "text": "V",
        "mode": "shared_with_sibling",
        "shared_with": "t9.p34",
    }
    assert p36["unit_source"]["cell_id"] == 145

    assert p35["source_cell_fragments"] == [
        {
            "cell_id": 139,
            "text": "VIN falling",
            "role": "test_conditions",
            "fragment_of": "VIN falling VIN hysteresis",
        },
        {
            "cell_id": 141,
            "text": "1.12",
            "role": "typ",
            "fragment_of": "1.12 180",
        },
    ]
    assert p36["source_cell_fragments"] == [
        {
            "cell_id": 139,
            "text": "VIN hysteresis",
            "role": "test_conditions",
            "fragment_of": "VIN falling VIN hysteresis",
        },
        {
            "cell_id": 141,
            "text": "180",
            "role": "typ",
            "fragment_of": "1.12 180",
        },
    ]

    source_map = {
        item["logical_row_id"]: item
        for item in table["source_map"]
    }

    assert source_map["t9.p35"]["primary_row_index"] == 34
    assert source_map["t9.p36"]["primary_row_index"] == 34
    assert source_map["t9.p35"]["unit_source"][
        "shared_with"
    ] == "t9.p34"

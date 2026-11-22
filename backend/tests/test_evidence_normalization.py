
"""EvidenceValidator 原文引用归一化回归测试。

覆盖原文引用匹配的格式差异修复，
确保仅放宽 "|" 分隔符与空白差异，
其余校验保持不变。
"""

import pytest

from app.extraction.evidence import (
    normalize_text,
    verify_evidence,
)


def vin_table():
    """真实 TPS7A02.pdf 中 table_id=7 的结构。"""
    return {
        "table_id": 7,
        "page_numbers": [5],
        "columns": ["", "", "MIN", "NOM", "MAX", "UNIT"],
        "rows": [
            ["VIN", "Input voltage", "1.5", "", "6.0", "V"],
        ],
    }


def build_extraction(
    source_text,
    value=1.5,
    value_type="min",
    unit="V",
    table_id=7,
    row_index=0,
    page=5,
):
    return {
        "parameters": [
            {
                "name": "input_voltage",
                "value": value,
                "unit": unit,
                "value_type": value_type,
                "test_conditions": "",
                "evidence": {
                    "table_id": table_id,
                    "row_index": row_index,
                    "page": page,
                    "source_text": source_text,
                },
            }
        ]
    }


def status_of(extraction, tables=None):
    if tables is None:
        tables = [vin_table()]

    result = verify_evidence(extraction, tables)

    return result["parameters"][0]


def test_exact_quote_passes():
    """1. 原始数据与引用文本完全一致 → 通过。"""
    parameter = status_of(
        build_extraction("Input voltage")
    )

    assert parameter["validation_status"] == "validated"


def test_pipe_and_space_difference_passes():
    """2. 只存在 "|" 和空格差异 → 通过。"""
    parameter = status_of(
        build_extraction(
            "VIN | Input voltage | 1.5 | | 6.0 | V"
        )
    )

    assert parameter["validation_status"] == "validated"


def test_wrong_value_rejected():
    """3. 模型数值错误 → 拒绝。"""
    parameter = status_of(
        build_extraction(
            "VIN | Input voltage | 1.5 | | 6.0 | V",
            value=2.0,
        )
    )

    assert parameter["validation_status"] == "invalid"
    assert any(
        "提取数值" in error
        for error in parameter["validation_errors"]
    )


def test_min_max_column_swap_rejected():
    """4. Min 和 Max 对应错误 → 拒绝。"""
    # value_type 声明为 max，却填了 MIN 列的 1.5
    parameter = status_of(
        build_extraction(
            "VIN | Input voltage | 1.5 | | 6.0 | V",
            value=1.5,
            value_type="max",
        )
    )

    assert parameter["validation_status"] == "invalid"
    assert any(
        "提取数值" in error
        for error in parameter["validation_errors"]
    )


def test_wrong_unit_rejected():
    """5. 单位错误 → 拒绝。"""
    parameter = status_of(
        build_extraction(
            "VIN | Input voltage | 1.5 | | 6.0 | V",
            unit="mV",
        )
    )

    assert parameter["validation_status"] == "invalid"
    assert any(
        "提取单位" in error
        for error in parameter["validation_errors"]
    )


@pytest.mark.parametrize(
    "overrides, expected_message",
    [
        ({"table_id": 99}, "来源表格不存在"),
        ({"row_index": 99}, "来源行号无效"),
        ({"page": 99}, "来源页码不匹配"),
    ],
)
def test_wrong_location_rejected(overrides, expected_message):
    """6. table_id、row_index、page 错误 → 拒绝。"""
    parameter = status_of(
        build_extraction(
            "VIN | Input voltage | 1.5 | | 6.0 | V",
            **overrides,
        )
    )

    assert parameter["validation_status"] == "invalid"
    assert expected_message in parameter["validation_errors"]


def test_empty_source_text_rejected():
    """7. source_text 为空 → 拒绝。"""
    parameter = status_of(
        build_extraction("")
    )

    assert parameter["validation_status"] == "invalid"
    assert "原文引用不匹配" in parameter["validation_errors"]


def test_genuinely_different_quote_rejected():
    """8. 引用内容确实与原始行不同 → 拒绝。"""
    # 数值和单位都正确，但原文引用的是另一行
    parameter = status_of(
        build_extraction(
            "VEN | Enable voltage | 0 | | 6.0 | V"
        )
    )

    assert parameter["validation_status"] == "invalid"
    assert "原文引用不匹配" in parameter["validation_errors"]


def test_normalize_text_handles_separators_and_whitespace():
    assert (
        normalize_text("VIN | Input voltage | 1.5 | | 6.0 | V")
        == "VIN Input voltage 1.5 6.0 V"
    )

    assert (
        normalize_text("VIN\nInput\tvoltage   1.5    6.0  V")
        == "VIN Input voltage 1.5 6.0 V"
    )

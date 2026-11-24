"""阶段 B3.2b：UVLO v2 Logical Rows 与 51 条 Gold 回归测试。

只读已生成 v2 产物，不调用 DeepSeek API。
"""

import json
from pathlib import Path

import pytest

from app.validation.recovered_evidence import (
    RecoveredEvidenceValidator,
)


pytestmark = pytest.mark.requires_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_PATH = (
    PROJECT_ROOT / "data/outputs/TPS7A02.logical_rows.v1.json"
)
V2_PATH = (
    PROJECT_ROOT / "data/outputs/TPS7A02.logical_rows.v2.json"
)
RAW_PATH = (
    PROJECT_ROOT / "data/outputs/TPS7A02.raw_structure.v1.json"
)
OLD_GOLD_PATH = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.json"
)
GOLD_V2_PATH = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.v2.json"
)
MAPPING_PATH = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "uvlo_source_mapping.v2.json"
)
DIFF_PATH = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "logical_rows.v1_to_v2.diff.json"
)
COMPLETENESS_PATH = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.v2.completeness.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifacts():
    return {
        "v1": load(V1_PATH),
        "v2": load(V2_PATH),
        "raw": load(RAW_PATH),
        "old_gold": load(OLD_GOLD_PATH),
        "gold_v2": load(GOLD_V2_PATH),
        "mapping": load(MAPPING_PATH),
        "diff": load(DIFF_PATH),
        "completeness": load(COMPLETENESS_PATH),
    }


def rows_by_id(logical_rows):
    return {
        row["logical_row_id"]: row
        for row in logical_rows["logical_rows"]
    }


def uvlo_parameters(gold):
    return {
        (item["logical_row_id"], item["value_type"]): item
        for item in gold["expected_parameters"]
        if item["logical_row_id"]
        in {"t9.p34", "t9.p35", "t9.p36"}
    }


def test_old_v1_artifacts_remain_needs_review(artifacts):
    """1. 旧 v1 needs_review 状态未被覆盖。"""

    v1 = artifacts["v1"]

    assert v1["summary"] == {
        "logical_row_count": 36,
        "confirmed_count": 33,
        "needs_review_count": 3,
    }

    rows = rows_by_id(v1)

    for row_id in ("t9.p34", "t9.p35", "t9.p36"):
        assert rows[row_id]["needs_review"] is True

    assert len(v1["unresolved_issues"]) == 3
    assert (
        artifacts["old_gold"]["expected_parameter_count"]
        == 44
    )


def test_v2_logical_rows_summary(artifacts):
    """2. v2 中 UVLO 三条已 confirmed。"""

    v2 = artifacts["v2"]

    assert v2["summary"] == {
        "logical_row_count": 36,
        "confirmed_count": 36,
        "needs_review_count": 0,
    }
    assert v2["unresolved_issues"] == []

    rows = rows_by_id(v2)

    for row_id in ("t9.p34", "t9.p35", "t9.p36"):
        assert rows[row_id]["needs_review"] is False
        assert rows[row_id]["review_status"] == "confirmed"


def test_v2_gold_51_records_complete(artifacts):
    """3. Gold v2 = 44 + 7 = 51 且无重复/遗漏。"""

    gold = artifacts["gold_v2"]
    completeness = artifacts["completeness"]

    assert gold["schema"] == "ab_gold_standard/v2"
    assert gold["trusted_logical_row_count"] == 36
    assert gold["expected_parameter_count"] == 51
    assert len(gold["expected_parameters"]) == 51
    assert gold["excluded_needs_review"] == []

    assert completeness["ok"] is True
    assert completeness["logical_row_count"] == 36
    assert completeness["gold_parameter_count"] == 51
    assert completeness["uvlo_parameter_count"] == 7
    assert completeness["uvlo_counts_by_row"] == {
        "t9.p34": 3,
        "t9.p35": 3,
        "t9.p36": 1,
    }
    assert completeness["duplicate_keys"] == []
    assert completeness["missing_keys"] == []
    assert completeness["extra_keys"] == []
    assert completeness["condition_mismatches"] == []
    assert completeness["unit_mismatches"] == []
    assert completeness["value_mismatches"] == []
    assert completeness["min_typ_max_violations"] == []
    assert completeness["empty_condition_ids"] == []


def test_old_44_gold_values_unchanged(artifacts):
    """4. v2 中原始 44 条 Gold 值不变。"""

    def key(item):
        return (item["logical_row_id"], item["value_type"])

    old = {
        key(item): item
        for item in artifacts["old_gold"][
            "expected_parameters"
        ]
    }
    new = {
        key(item): item
        for item in artifacts["gold_v2"][
            "expected_parameters"
        ]
        if key(item) not in {
            ("t9.p34", "min"),
            ("t9.p34", "typ"),
            ("t9.p34", "max"),
            ("t9.p35", "min"),
            ("t9.p35", "typ"),
            ("t9.p35", "max"),
            ("t9.p36", "typ"),
        }
    }

    assert old.keys() == new.keys()

    for key_value, old_item in old.items():
        new_item = new[key_value]

        assert new_item["value"] == old_item["value"]
        assert new_item["unit"] == old_item["unit"]
        assert (
            new_item["value_type"]
            == old_item["value_type"]
        )
        assert (
            new_item["test_conditions"]
            == old_item["test_conditions"]
        )


def test_uvlo_gold_values_and_conditions(artifacts):
    """5. 七条 UVLO 规格与人工作业值一致。"""

    uvlo = uvlo_parameters(artifacts["gold_v2"])

    expected = {
        ("t9.p34", "min"): ("1.23", "V", "VIN rising"),
        ("t9.p34", "typ"): ("1.3", "V", "VIN rising"),
        ("t9.p34", "max"): ("1.47", "V", "VIN rising"),
        ("t9.p35", "min"): ("1.0", "V", "VIN falling"),
        ("t9.p35", "typ"): ("1.12", "V", "VIN falling"),
        ("t9.p35", "max"): ("1.41", "V", "VIN falling"),
        ("t9.p36", "typ"): ("180", "mV", "VIN hysteresis"),
    }

    assert set(uvlo) == set(expected)

    for key_value, (value, unit, conditions) in expected.items():
        item = uvlo[key_value]

        assert item["value"] == value
        assert item["unit"] == unit
        assert item["test_conditions"] == conditions
        assert item["needs_review"] is False
        assert item["review_status"] == "confirmed"


def test_seven_uvlo_records_pass_strict_validator(artifacts):
    """6. 七条 UVLO 规格全部通过 RecoveredEvidenceValidator。"""

    validator = RecoveredEvidenceValidator(
        artifacts["v2"],
        raw_structure=artifacts["raw"],
    )
    uvlo = uvlo_parameters(artifacts["gold_v2"])

    parameters = []

    for item in uvlo.values():
        name = (
            "uvlo_hysteresis"
            if "hysteresis" in item[
                "parameter_name"
            ].lower()
            else "uvlo_threshold"
        )
        parameters.append({
            "name": name,
            "value": float(item["value"]),
            "unit": item["unit"],
            "value_type": item["value_type"],
            "test_conditions": item[
                "test_conditions"
            ],
            "evidence": {
                "table_id": item["table_id"],
                "row_index": item[
                    "primary_row_index"
                ],
                "page": item["page"],
                "source_text": "",
                "logical_row_id": item[
                    "logical_row_id"
                ],
                "source_row_indices": item[
                    "source_row_indices"
                ],
                "source_cell_ids": item[
                    "source_cell_ids"
                ],
            },
        })

    report = validator.validate({
        "component_model": "TPS7A02",
        "manufacturer": "TI",
        "parameters": parameters,
    })

    assert len(report["parameters"]) == 7

    for result in report["parameters"]:
        assert result["recovered_status"] == "validated"
        assert result["conflicts"] == []
        assert (
            result["selected_logical_row_id"]
            == result["evidence"]["logical_row_id"]
        )


def test_p35_and_p36_are_distinct_despite_same_row(artifacts):
    """7. p35 / p36 共享原始 row 34 但仍独立匹配。"""

    uvlo = uvlo_parameters(artifacts["gold_v2"])

    p35 = uvlo[("t9.p35", "typ")]
    p36 = uvlo[("t9.p36", "typ")]

    assert p35["primary_row_index"] == 34
    assert p36["primary_row_index"] == 34
    assert p35["source_row_indices"] == [33, 34]
    assert p36["source_row_indices"] == [34]

    validator = RecoveredEvidenceValidator(
        artifacts["v2"],
        raw_structure=artifacts["raw"],
    )
    parameters = []

    for item in (p35, p36):
        parameters.append({
            "name": (
                "uvlo_threshold"
                if item["logical_row_id"] == "t9.p35"
                else "uvlo_hysteresis"
            ),
            "value": float(item["value"]),
            "unit": item["unit"],
            "value_type": item["value_type"],
            "test_conditions": item[
                "test_conditions"
            ],
            "evidence": {
                "table_id": 9,
                "row_index": 34,
                "page": 6,
                "source_text": "",
                "logical_row_id": item[
                    "logical_row_id"
                ],
                "source_row_indices": item[
                    "source_row_indices"
                ],
                "source_cell_ids": item[
                    "source_cell_ids"
                ],
            },
        })

    report = validator.validate({
        "component_model": "",
        "manufacturer": "",
        "parameters": parameters,
    })

    assert [
        item["selected_logical_row_id"]
        for item in report["parameters"]
    ] == ["t9.p35", "t9.p36"]
    assert all(
        item["recovered_status"] == "validated"
        for item in report["parameters"]
    )


def test_uvlo_source_mapping_evidence(artifacts):
    """8. 三条 UVLO 的来源 cell / 行号 / 单位继承证据完整。"""

    mapping = artifacts["mapping"]["records"]

    p34 = mapping["t9.p34"]
    p35 = mapping["t9.p35"]
    p36 = mapping["t9.p36"]

    assert p34["primary_row_index"] == 33
    assert p35["primary_row_index"] == 34
    assert p36["primary_row_index"] == 34

    assert 138 in p35["source_cell_ids"]
    assert p35["unit_source"] == {
        "cell_id": 138,
        "text": "V",
        "mode": "shared_with_sibling",
        "shared_with": "t9.p34",
    }
    assert p36["unit_source"]["cell_id"] == 145
    assert p36["unit_source"]["text"] == "mV"

    assert p35["source_cell_fragments"][0][
        "text"
    ] == "VIN falling"
    assert p36["source_cell_fragments"][0][
        "text"
    ] == "VIN hysteresis"
    assert p35["source_cell_fragments"][1][
        "text"
    ] == "1.12"
    assert p36["source_cell_fragments"][1][
        "text"
    ] == "180"

    assert "VIN falling" in p35["pdf_text_in_band"]
    assert "VIN hysteresis" in p36["pdf_text_in_band"]


def test_v1_v2_diff_summary(artifacts):
    """9. v1 -> v2 差异摘要明确。"""

    diff = artifacts["diff"]
    summary = diff["summary"]

    assert summary["v1_logical_rows"] == 36
    assert summary["v2_logical_rows"] == 36
    assert summary["v1_confirmed"] == 33
    assert summary["v2_confirmed"] == 36
    assert summary["v1_needs_review"] == 3
    assert summary["v2_needs_review"] == 0
    assert summary["v1_unresolved_issues"] == 3
    assert summary["v2_unresolved_issues"] == 0

    changed = {
        item["logical_row_id"]
        for item in diff["changed_rows"]
    }
    assert changed == {"t9.p34", "t9.p35", "t9.p36"}

    assert diff["strict_uvlo_validation"][
        "validated_count"
    ] == 7
    assert diff["strict_uvlo_validation"][
        "all_validated"
    ] is True

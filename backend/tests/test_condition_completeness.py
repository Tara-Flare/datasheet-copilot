"""阶段 B2.1.1：测试条件完整性审计测试。"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from app.validation.recovered_evidence import (
    evaluate_conditions,
    validate_from_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.extraction.json"
)
LOGICAL_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.logical_rows.v1.json"
)
BASELINE_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.final.validation.json"
)
RAW_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.raw_structure.v1.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.recovered_validation.v1.json"
)

BASELINE_SHA_AT_IMPORT = hashlib.sha256(
    BASELINE_PATH.read_bytes()
).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inputs():
    return {
        "extraction": load(EXTRACTION_PATH),
        "logical_rows": load(LOGICAL_PATH),
        "baseline": load(BASELINE_PATH),
        "raw_structure": load(RAW_PATH),
    }


@pytest.fixture(scope="module")
def report(inputs):
    return validate_from_files(
        inputs["extraction"],
        inputs["logical_rows"],
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )


def by_index(report):
    return {
        item["index"]: item
        for item in report["parameters"]
    }


def test_complete_condition_passes():
    result = evaluate_conditions(
        "TJ = -40°C to +85°C; 1.0 V ≤ VOUT < 1.2 V",
        "TJ = –40°C to +85°C | 1.0 V ≤ VOUT < 1.2 V",
    )

    assert result["status"] == "match"
    assert result["missing_conditions"] == []
    assert result["conflicting_conditions"] == []


def test_missing_temperature_not_passed():
    result = evaluate_conditions(
        "3.3 V ≤ VOUT ≤ 5.0 V",
        "TJ = –40°C to +85°C | 3.3 V ≤ VOUT ≤ 5.0 V",
    )

    assert result["status"] == "missing"
    assert result["missing_conditions"]
    assert any(
        "tj" in item for item in result["missing_conditions"]
    )


def test_different_temperature_range_rejected():
    result = evaluate_conditions(
        "TJ = -40°C to +85°C; 3.3 V ≤ VOUT ≤ 5.0 V",
        "TJ = –40°C to +125°C | 3.3 V ≤ VOUT ≤ 5.0 V",
    )

    assert result["status"] == "conflict"
    assert result["conflicting_conditions"]


def test_missing_vout_interval_rejected():
    result = evaluate_conditions(
        "TJ = -40°C to +85°C",
        "TJ = –40°C to +85°C | 3.3 V ≤ VOUT ≤ 5.0 V",
    )

    assert result["status"] == "missing"
    assert any(
        "vout" in item
        for item in result["missing_conditions"]
    )


def test_different_comparison_operator_rejected():
    result = evaluate_conditions(
        "TJ = -40°C to +85°C; 3.3 V ≤ VOUT < 5.0 V",
        "TJ = –40°C to +85°C | 3.3 V ≤ VOUT ≤ 5.0 V",
    )

    assert result["status"] == "conflict"
    assert result["conflicting_conditions"]


def test_different_unit_rejected():
    result = evaluate_conditions(
        "TJ = -40°C to +85°C; 3.3 V ≤ VOUT ≤ 5000 mV",
        "TJ = –40°C to +85°C | 3.3 V ≤ VOUT ≤ 5.0 V",
    )

    assert result["status"] == "conflict"
    assert result["conflicting_conditions"]


def test_different_order_same_semantics_passes():
    result = evaluate_conditions(
        "1.0 V ≤ VOUT < 1.2 V; TJ = -40°C to +125°C",
        "TJ = –40°C to +125°C | 1.0 V ≤ VOUT < 1.2 V",
    )

    assert result["status"] == "match"


def test_needs_review_forbidden(inputs):
    logical_rows = copy.deepcopy(inputs["logical_rows"])

    for row in logical_rows["logical_rows"]:
        if row["logical_row_id"] == "t9.p8":
            row["needs_review"] = True

    result = validate_from_files(
        inputs["extraction"],
        logical_rows,
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )

    row = by_index(result)[8]

    assert row["recovered_status"] == "pending_review"


def test_ground_current_wrong_conditions_rejected(report):
    params = by_index(report)

    # 60 nA：值/单位正确，但 IOUT 条件冲突 -> invalid
    assert params[11]["recovered_status"] == "invalid"
    assert "test_conditions" in params[11]["conflicts"]

    # 1 nA：身份/单位/条件均冲突 -> invalid
    assert params[10]["recovered_status"] == "invalid"
    assert "test_conditions" in params[10]["conflicts"]


def test_baseline_parameters_not_modified(report):
    assert (
        hashlib.sha256(
            BASELINE_PATH.read_bytes()
        ).hexdigest()
        == BASELINE_SHA_AT_IMPORT
    )

    assert report["summary"]["baseline_validated"] == 27


def test_dropout_conditions_completeness(report):
    """重点检查 270/650/490/400/310/850/700/560/450 mV。"""

    params = by_index(report)

    # 270 mV：缺温度条件 -> correctable
    row = params[23]
    assert row["value"] == 270.0
    assert row["recovered_status"] == "correctable"
    assert "test_conditions" in row["missing_fields"]
    assert row["field_checks"]["test_conditions"][
        "missing_conditions"
    ]

    # 650 / 400 / 310 mV：缺单位 + 缺温度
    for index in (19, 21, 22):
        row = params[index]
        assert row["recovered_status"] == "correctable"
        assert "unit" in row["missing_fields"]
        assert "test_conditions" in row["missing_fields"]

    # 490 mV：仅缺单位，温度条件完整
    row = params[20]
    assert row["recovered_status"] == "correctable"
    assert row["missing_fields"] == ["unit"]

    # 850 / 700 / 560 / 450 mV：条件完整 -> validated
    for index, value in (
        (25, 850.0),
        (26, 700.0),
        (27, 560.0),
        (28, 450.0),
    ):
        row = params[index]
        assert row["value"] == value
        assert row["recovered_status"] == "validated"
        assert row["missing_fields"] == []


def test_reclassification_summary(report):
    summary = report["summary"]

    assert summary["recovered_validated"] == 20
    assert summary["correctable"] == 8
    assert summary["invalid"] == 2
    assert summary["pending_review"] == 0
    assert summary["newly_validated"] == 4
    assert summary["regressed"] == 3

    assert summary["parameter_field_missing"] == {
        "test_conditions": 7,
        "unit": 4,
    }


def test_b21_comparison_if_present():
    previous_path = (
        PROJECT_ROOT
        / "data/outputs/"
        "TPS7A02.recovered_validation.B2.1.json"
    )

    if not previous_path.is_file() or not REPORT_PATH.is_file():
        return

    report = load(REPORT_PATH)
    comparison = report.get("comparison_to_previous")

    if comparison is None:
        return

    assert (
        comparison["counts"][
            "downgraded_validated_to_correctable"
        ]
        == 4
    )
    downgraded = {
        change["index"]
        for change in comparison["changes"]
        if change["change_type"]
        == "validated_to_correctable"
    }
    assert downgraded == {4, 5, 23, 24}

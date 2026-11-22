"""阶段 B3.2b：A/B 参数级评分器单元测试。"""

import json
from pathlib import Path

import pytest

from app.evaluation.ab_scoring import (
    score_extraction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.v2.eval.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gold():
    return load(GOLD_PATH)


def gold_item(gold, row_id, value_type):
    return next(
        item
        for item in gold["expected_parameters"]
        if item["logical_row_id"] == row_id
        and item["value_type"] == value_type
    )


def prediction(
    item,
    *,
    name,
    value=None,
    unit=None,
    conditions=None,
    logical_row_id=True,
):
    evidence = {
        "table_id": item["table_id"],
        "row_index": item["primary_row_index"],
        "page": item["page"],
        "source_text": "",
    }

    if logical_row_id:
        evidence["logical_row_id"] = item[
            "logical_row_id"
        ]

    return {
        "name": name,
        "value": (
            item["value"] if value is None else value
        ),
        "unit": item["unit"] if unit is None else unit,
        "value_type": item["value_type"],
        "test_conditions": (
            item["test_conditions_row_level"]
            if conditions is None
            else conditions
        ),
        "evidence": evidence,
    }


def test_uvlo_perfect_predictions(gold):
    items = [
        gold_item(gold, "t9.p34", "min"),
        gold_item(gold, "t9.p34", "typ"),
        gold_item(gold, "t9.p34", "max"),
        gold_item(gold, "t9.p35", "min"),
        gold_item(gold, "t9.p35", "typ"),
        gold_item(gold, "t9.p35", "max"),
        gold_item(gold, "t9.p36", "typ"),
    ]

    extraction = {
        "parameters": [
            prediction(
                item,
                name=(
                    "uvlo_hysteresis"
                    if item["logical_row_id"] == "t9.p36"
                    else "uvlo_threshold"
                ),
            )
            for item in items
        ]
    }

    result = score_extraction(gold, extraction)

    assert result["predicted_count"] == 7
    assert result["matched_keys"] == 7
    assert result["strict_matched"] == 7
    assert result["precision"] == 1.0
    assert result["recall"] == round(7 / 51, 4)
    assert result["field_accuracy"]["identity"] == 1.0
    assert result["field_accuracy"]["value"] == 1.0
    assert result["field_accuracy"]["unit"] == 1.0
    assert result["field_accuracy"]["test_conditions"] == 1.0
    assert result["evidence_source"]["correct"] == 7
    assert result["evidence_source"][
        "logical_row_id_presence_rate"
    ] == 1.0
    assert result["special"]["uvlo"]["strict_matched"] == 7


def test_condition_mismatch_is_not_strict(gold):
    item = gold_item(gold, "t9.p35", "typ")
    extraction = {
        "parameters": [
            prediction(
                item,
                name="uvlo_threshold",
                conditions="VIN rising",
            )
        ]
    }

    result = score_extraction(gold, extraction)

    assert result["matched_keys"] == 1
    assert result["strict_matched"] == 0
    assert result["precision"] == 0.0
    assert result["field_accuracy"]["value"] == 1.0
    assert result["field_accuracy"]["test_conditions"] == 0.0
    assert result["field_mismatches"][0]["fields"] == [
        "test_conditions"
    ]


def test_value_mismatch_is_not_strict(gold):
    item = gold_item(gold, "t9.p34", "max")
    extraction = {
        "parameters": [
            prediction(
                item,
                name="uvlo_threshold",
                value=1.5,
            )
        ]
    }

    result = score_extraction(gold, extraction)

    assert result["matched_keys"] == 1
    assert result["strict_matched"] == 0
    assert result["field_accuracy"]["value"] == 0.0
    assert "value" in result["field_mismatches"][0][
        "fields"
    ]


def test_shared_row_without_id_is_not_guessed(gold):
    p35 = gold_item(gold, "t9.p35", "typ")
    p36 = gold_item(gold, "t9.p36", "typ")

    extraction = {
        "parameters": [
            prediction(
                p35,
                name="uvlo_threshold",
                logical_row_id=False,
            ),
            prediction(
                p36,
                name="uvlo_hysteresis",
                logical_row_id=False,
            ),
        ]
    }

    result = score_extraction(gold, extraction)

    # row 34 同时属于 p35/p36；没有 logical id 或
    # recovered 选择时不得猜测。
    assert result["matched_keys"] == 0
    assert len(result["false_positives"]) == 2
    assert all(
        item["reason"] == "unmapped_source"
        for item in result["false_positives"]
    )
    assert result["special"]["uvlo"]["missing"] == 7


def test_recovered_selected_row_can_resolve_shared_row(gold):
    p35 = gold_item(gold, "t9.p35", "typ")
    p36 = gold_item(gold, "t9.p36", "typ")

    extraction = {
        "parameters": [
            prediction(
                p35,
                name="uvlo_threshold",
                logical_row_id=False,
            ),
            prediction(
                p36,
                name="uvlo_hysteresis",
                logical_row_id=False,
            ),
        ]
    }
    recovered = {
        "parameters": [
            {
                "selected_logical_row_id": "t9.p35",
                "recovered_status": "validated",
                "field_checks": {
                    "source": {"status": "match"}
                },
            },
            {
                "selected_logical_row_id": "t9.p36",
                "recovered_status": "validated",
                "field_checks": {
                    "source": {"status": "match"}
                },
            },
        ]
    }

    result = score_extraction(
        gold, extraction, recovered
    )

    assert result["matched_keys"] == 2
    assert result["strict_matched"] == 2
    assert result["evidence_source"]["correct"] == 2
    assert result["evidence_source"][
        "logical_row_id_presence_rate"
    ] == 0.0
    assert all(
        item["evidence_check"]["method"]
        == "recovered_selected_row"
        for item in result["gold_details"]
        if item["match_status"] == "matched"
    )


def test_untrusted_mapping_blocks_recovered_selected_row(gold):
    p35 = gold_item(gold, "t9.p35", "typ")
    p36 = gold_item(gold, "t9.p36", "typ")

    p35_pred = prediction(
        p35,
        name="uvlo_threshold",
        logical_row_id=False,
    )
    p36_pred = prediction(
        p36,
        name="uvlo_hysteresis",
        logical_row_id=False,
    )
    p35_pred["source_mapping_status"] = (
        "ambiguous_row_index"
    )
    p36_pred["source_mapping_status"] = (
        "ambiguous_row_index"
    )

    extraction = {
        "parameters": [p35_pred, p36_pred]
    }
    recovered = {
        "parameters": [
            {
                "selected_logical_row_id": "t9.p35",
                "recovered_status": "pending_review",
                "field_checks": {
                    "source": {"status": "match"}
                },
            },
            {
                "selected_logical_row_id": "t9.p36",
                "recovered_status": "pending_review",
                "field_checks": {
                    "source": {"status": "match"}
                },
            },
        ]
    }

    result = score_extraction(
        gold, extraction, recovered
    )

    assert result["matched_keys"] == 0
    assert len(result["false_positives"]) == 2


def test_duplicate_prediction_is_false_positive(gold):
    item = gold_item(gold, "t9.p34", "typ")
    single = prediction(
        item, name="uvlo_threshold"
    )

    extraction = {
        "parameters": [
            single,
            dict(single),
        ]
    }

    result = score_extraction(gold, extraction)

    assert result["matched_keys"] == 1
    assert result["strict_matched"] == 1
    assert result["predicted_count"] == 2
    assert result["precision"] == 0.5
    assert result["duplicate_count"] == 1

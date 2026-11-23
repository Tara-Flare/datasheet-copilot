"""B3.4d：生产 RecoveredEvidenceValidator 身份归一化测试。

覆盖正例、反例和 t9.p33 的 validated / invalid / pending_review
状态边界。不调用 API。
"""

import copy

import pytest

from app.validation.recovered_evidence import (
    FAMILY_NORMALIZATION_VERSION,
    RecoveredEvidenceValidator,
    _identity_match,
    logical_family,
    original_family,
)


def row(
    symbol,
    name,
    *,
    row_id="t9.test",
    value_type="typ",
    value="1.0",
    unit="V",
    conditions="TJ = 25°C",
):
    return {
        "logical_row_id": row_id,
        "table_id": 9,
        "page": 6,
        "parameter_symbol": symbol,
        "parameter_name": name,
        "min": value if value_type == "min" else "",
        "typ": value if value_type == "typ" else "",
        "max": value if value_type == "max" else "",
        "unit": unit,
        "test_conditions": conditions,
        "source_row_indices": [32],
        "source_cell_ids": [127, 128, 129, 130, 131],
        "source_word_bboxes": [
            {"text": symbol or name, "bbox": [0, 0, 1, 1]}
        ],
        "needs_review": False,
    }


def param(
    name,
    *,
    value=1.0,
    unit="V",
    value_type="typ",
    conditions="TJ = 25°C",
    logical_row_id="t9.test",
    table_id=9,
    row_index=32,
    source_row_indices=None,
):
    evidence = {
        "table_id": table_id,
        "row_index": row_index,
        "page": 6,
        "source_text": "",
        "logical_row_id": logical_row_id,
        "source_row_indices": (
            [32]
            if source_row_indices is None
            else source_row_indices
        ),
        "source_cell_ids": [127, 128, 129, 130, 131],
    }
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "value_type": value_type,
        "test_conditions": conditions,
        "evidence": evidence,
    }


@pytest.mark.parametrize(
    "symbol,name,expected",
    [
        ("VN", "Output voltage noise", "output_noise"),
        ("V_N", "Output voltage noise", "output_noise"),
        ("", "Output voltage noise", "output_noise"),
        ("", "Output voltage accuracy", "accuracy"),
        ("", "Nominal accuracy", "accuracy"),
        ("", "Accuracy over temperature", "accuracy"),
        ("VOUT", "Output voltage", "output_voltage"),
        ("VIN", "Input voltage", "input_voltage"),
        ("", "Input voltage noise", "input_noise"),
        ("", "Line regulation", "line_regulation"),
        ("", "Load regulation(2)", "load_regulation"),
        (
            "IGND/IOUT",
            "Ground current vs load current",
            "ground_current_load",
        ),
        ("IGND", "Ground current in dropout", "ground_current"),
        ("", "Shutdown current", "shutdown_current"),
        ("ICL", "Output current limit", "current_limit"),
        (
            "ISC",
            "Short-circuit current limit",
            "short_circuit_current_limit",
        ),
        ("VUVLO", "UVLO threshold", "uvlo_threshold"),
        (
            "VUVLO(HYST)",
            "UVLO hysteresis",
            "uvlo_hysteresis",
        ),
        ("PSRR", "Power-supply rejection ratio", "psrr"),
        ("VDO", "Dropout voltage", "dropout"),
    ],
)
def test_logical_family_normalization(symbol, name, expected):
    assert logical_family({
        "parameter_symbol": symbol,
        "parameter_name": name,
    }) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("output_voltage_noise", "output_noise"),
        ("V_N", "output_noise"),
        ("VN", "output_noise"),
        ("output_noise", "output_noise"),
        ("input_voltage_noise", "input_noise"),
        ("output_voltage_accuracy", "accuracy"),
        ("output_voltage", "output_voltage"),
        ("line_regulation", "line_regulation"),
        ("load_regulation", "load_regulation"),
        (
            "ground_current_vs_load_current",
            "ground_current_load",
        ),
        ("ground_current_in_dropout", "ground_current"),
        ("shutdown_current", "shutdown_current"),
        ("output_current_limit", "current_limit"),
        (
            "short_circuit_current_limit",
            "short_circuit_current_limit",
        ),
        ("uvlo_threshold", "uvlo_threshold"),
        ("uvlo_hysteresis", "uvlo_hysteresis"),
    ],
)
def test_original_family_normalization(name, expected):
    assert original_family(name) == expected


def test_output_noise_identity_matches():
    logical = row("VN", "Output voltage noise")
    predicted = param("output_voltage_noise")

    result = _identity_match(predicted, logical)

    assert result["status"] == "match"
    assert result["original_family"] == "output_noise"
    assert result["logical_family"] == "output_noise"


@pytest.mark.parametrize(
    "prediction_name,logical_row",
    [
        ("output_voltage", row("VN", "Output voltage noise")),
        ("output_voltage_noise", row("VOUT", "Output voltage")),
        ("uvlo_hysteresis", row("VUVLO", "UVLO threshold")),
        ("uvlo_threshold", row("VUVLO(HYST)", "UVLO hysteresis")),
        (
            "output_current_limit",
            row("ISC", "Short-circuit current limit"),
        ),
        (
            "short_circuit_current_limit",
            row("ICL", "Output current limit"),
        ),
        (
            "ground_current",
            row("IGND/IOUT", "Ground current vs load current"),
        ),
        (
            "ground_current_vs_load_current",
            row("IGND", "Ground current"),
        ),
        ("psrr", row("VN", "Output voltage noise")),
    ],
)
def test_identity_negative_cases_remain_conflicts(
    prediction_name,
    logical_row,
):
    result = _identity_match(
        param(prediction_name),
        logical_row,
    )

    assert result["status"] == "conflict"


def p33_row():
    return row(
        "VN",
        "Output voltage noise",
        row_id="t9.p33",
        value="130",
        unit="µVRMS",
        conditions=(
            "BW = 10 Hz to 100 kHz, "
            "VOUT = 0.8 V, IOUT = 30 mA"
        ),
    )


def p33_param():
    return param(
        "output_voltage_noise",
        value=130.0,
        unit="µVRMS",
        conditions=(
            "BW = 10 Hz to 100 kHz, "
            "VOUT = 0.8 V, IOUT = 30 mA"
        ),
        logical_row_id="t9.p33",
    )


def p33_raw_structure():
    return {
        "tables": [{
            "table_id": 9,
            "cells": [
                {"cell_id": cell_id}
                for cell_id in (127, 128, 129, 130, 131)
            ],
        }]
    }


def validate_one(
    logical_row,
    prediction,
    raw_structure=None,
):
    validator = RecoveredEvidenceValidator(
        {"logical_rows": [logical_row]},
        raw_structure=(
            p33_raw_structure()
            if raw_structure is None
            else raw_structure
        ),
    )

    return validator.validate({
        "parameters": [prediction],
    })["parameters"][0]


def test_p33_is_validated_after_fix():
    result = validate_one(p33_row(), p33_param())

    assert result["recovered_status"] == "validated"
    assert result["conflicts"] == []
    assert result["field_checks"]["identity"]["status"] == "match"
    assert result["field_checks"]["value"]["status"] == "match"
    assert result["field_checks"]["unit"]["status"] == "match"
    assert result["field_checks"]["value_type"]["status"] == "match"
    assert result["field_checks"]["test_conditions"][
        "status"
    ] == "match"
    assert result["field_checks"]["source"]["status"] == "match"


@pytest.mark.parametrize(
    "mutation,expected_status",
    [
        ("wrong_name", "invalid"),
        ("wrong_value", "invalid"),
        ("wrong_unit", "invalid"),
        ("wrong_conditions", "invalid"),
        ("wrong_logical_id", "pending_review"),
        ("wrong_source_rows", "pending_review"),
        ("needs_review", "pending_review"),
    ],
)
def test_p33_invalid_or_pending_is_not_relaxed(
    mutation,
    expected_status,
):
    logical_row = p33_row()
    prediction = p33_param()

    if mutation == "wrong_name":
        prediction["name"] = "output_voltage"
    elif mutation == "wrong_value":
        prediction["value"] = 131.0
    elif mutation == "wrong_unit":
        prediction["unit"] = "mV"
    elif mutation == "wrong_conditions":
        prediction["test_conditions"] = (
            "BW = 1 Hz, VOUT = 0.8 V, IOUT = 30 mA"
        )
    elif mutation == "wrong_logical_id":
        prediction["evidence"]["logical_row_id"] = "t9.p999"
    elif mutation == "wrong_source_rows":
        prediction["evidence"]["source_row_indices"] = [99]
    elif mutation == "needs_review":
        logical_row["needs_review"] = True

    result = validate_one(logical_row, prediction)

    assert result["recovered_status"] == expected_status


def test_ambiguous_mapping_stays_pending_review():
    logical_row = p33_row()
    prediction = p33_param()
    prediction["source_mapping_status"] = (
        "ambiguous_row_index"
    )

    result = validate_one(logical_row, prediction)

    assert result["recovered_status"] == "pending_review"


def test_not_covered_table_stays_not_covered():
    logical_row = p33_row()
    prediction = p33_param()
    prediction["evidence"]["table_id"] = 999

    result = validate_one(logical_row, prediction)

    assert result["recovered_status"] == "not_covered"


def test_validator_normalization_version_is_marked():
    assert FAMILY_NORMALIZATION_VERSION == (
        "b3.4d-validator-v1"
    )

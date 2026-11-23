"""B3.4c：评估层参数族归一化与优先级测试。

只测试 ab_scoring.py 的评估语义，不修改生产 Validator。
"""

import pytest

from app.evaluation.ab_scoring import (
    FAMILY_NORMALIZATION_VERSION,
    SCORER_VERSION,
    gold_family,
    prediction_family,
)


@pytest.mark.parametrize(
    "symbol,name,expected",
    [
        ("VN", "Output voltage noise", "output_noise"),
        ("V_N", "Output voltage noise", "output_noise"),
        ("", "Output voltage noise", "output_noise"),
        ("", "output_noise", "output_noise"),
        ("", "noise voltage", "output_noise"),
        ("", "input voltage noise", "input_noise"),
        ("", "Output voltage accuracy", "accuracy"),
        ("", "Nominal accuracy", "accuracy"),
        ("", "Accuracy over temperature", "accuracy"),
        ("VOUT", "Output voltage", "output_voltage"),
        ("VIN", "Input voltage", "input_voltage"),
        ("", "Enable threshold", "enable_threshold"),
        ("PSRR", "Power-supply rejection ratio", "psrr"),
        ("ICL", "Output current limit", "current_limit"),
        (
            "ISC",
            "Short-circuit current limit",
            "short_circuit_current_limit",
        ),
        (
            "IGND/IOUT",
            "Ground current vs load current",
            "ground_current_load",
        ),
        (
            "IGND",
            "Ground current in dropout",
            "ground_current",
        ),
        ("VUVLO", "UVLO threshold", "uvlo_threshold"),
        (
            "VUVLO(HYST)",
            "UVLO hysteresis",
            "uvlo_hysteresis",
        ),
        ("VDO", "Dropout voltage", "dropout"),
    ],
)
def test_gold_family_priority(symbol, name, expected):
    assert gold_family({
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
        ("noise voltage", "output_noise"),
        ("input_voltage_noise", "input_noise"),
        ("output_voltage_accuracy", "accuracy"),
        ("output_voltage", "output_voltage"),
        ("input_voltage", "input_voltage"),
        ("enable_threshold", "enable_threshold"),
        ("psrr", "psrr"),
        ("dropout_voltage", "dropout"),
        ("output_current_limit", "current_limit"),
        (
            "short_circuit_current_limit",
            "short_circuit_current_limit",
        ),
        ("ground_current", "ground_current"),
        (
            "ground_current_vs_load_current",
            "ground_current_load",
        ),
        ("uvlo_threshold", "uvlo_threshold"),
        ("uvlo_hysteresis", "uvlo_hysteresis"),
    ],
)
def test_prediction_family_semantics(name, expected):
    assert prediction_family(name) == expected


def test_output_noise_aliases_are_equivalent():
    assert (
        gold_family({
            "parameter_symbol": "V_N",
            "parameter_name": "Output voltage noise",
        })
        == prediction_family("output_voltage_noise")
        == "output_noise"
    )


def test_noise_does_not_absorb_output_voltage():
    assert gold_family({
        "parameter_symbol": "VOUT",
        "parameter_name": "Output voltage",
    }) == "output_voltage"
    assert gold_family({
        "parameter_symbol": "VOUT",
        "parameter_name": "Output voltage noise",
    }) == "output_noise"


def test_accuracy_does_not_absorb_output_voltage():
    assert gold_family({
        "parameter_symbol": "VOUT",
        "parameter_name": "Output voltage accuracy",
    }) == "accuracy"
    assert prediction_family(
        "output_voltage_accuracy"
    ) == "accuracy"
    assert prediction_family(
        "output_voltage"
    ) == "output_voltage"


def test_input_noise_is_not_output_noise():
    assert gold_family({
        "parameter_symbol": "VIN",
        "parameter_name": "Input voltage noise",
    }) == "input_noise"
    assert prediction_family(
        "input_voltage_noise"
    ) == "input_noise"


def test_non_confusable_categories_keep_their_family():
    assert gold_family({
        "parameter_symbol": "EN",
        "parameter_name": "Enable threshold",
    }) == "enable_threshold"
    assert prediction_family(
        "enable_threshold"
    ) == "enable_threshold"

    assert gold_family({
        "parameter_symbol": "PSRR",
        "parameter_name": "Power-supply rejection ratio",
    }) == "psrr"
    assert prediction_family("psrr") == "psrr"

    assert gold_family({
        "parameter_symbol": "ICL",
        "parameter_name": "Output current limit",
    }) == "current_limit"
    assert prediction_family(
        "output_current_limit"
    ) == "current_limit"

    assert gold_family({
        "parameter_symbol": "ISC",
        "parameter_name": "Short-circuit current limit",
    }) == "short_circuit_current_limit"
    assert prediction_family(
        "short_circuit_current_limit"
    ) == "short_circuit_current_limit"


def test_scorer_version_is_marked():
    assert SCORER_VERSION == "b3.4c-v1"
    assert FAMILY_NORMALIZATION_VERSION == (
        "b3.4c-family-v1"
    )

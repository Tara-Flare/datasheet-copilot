
from app.extraction.evidence import verify_evidence
from app.validation.validator import ParameterValidator


TABLES = [
    {
        "table_id": 1,
        "page_numbers": [5],
        "columns": [
            "Parameter",
            "Conditions",
            "Min",
            "Typ",
            "Max",
            "Unit"
        ],
        "rows": [
            [
                "Dropout Voltage",
                "IOUT=500mA",
                "",
                "200",
                "300",
                "mV"
            ]
        ]
    }
]


def build_extraction(value, value_type):
    return {
        "parameters": [
            {
                "name": "dropout_voltage",
                "value": value,
                "unit": "mV",
                "value_type": value_type,
                "test_conditions": "IOUT=500mA",
                "evidence": {
                    "table_id": 1,
                    "row_index": 0,
                    "page": 5,
                    "source_text": "Dropout Voltage"
                }
            }
        ]
    }


def test_correct_typical_value():
    extraction = build_extraction(
        200, "typ"
    )

    result = verify_evidence(
        extraction,
        TABLES
    )

    assert (
        result["parameters"][0]["validation_status"]
        == "validated"
    )


def test_wrong_typical_value():
    extraction = build_extraction(
        300, "typ"
    )

    result = verify_evidence(
        extraction,
        TABLES
    )

    assert (
        result["parameters"][0]["validation_status"]
        == "invalid"
    )


def test_correct_maximum_value():
    extraction = build_extraction(
        300, "max"
    )

    result = verify_evidence(
        extraction,
        TABLES
    )

    assert (
        result["parameters"][0]["validation_status"]
        == "validated"
    )


def test_invalid_table_id():
    extraction = build_extraction(
        200, "typ"
    )

    extraction["parameters"][0]["evidence"][
        "table_id"
    ] = 99

    result = verify_evidence(
        extraction,
        TABLES
    )

    assert (
        result["parameters"][0]["validation_status"]
        == "invalid"
    )


def test_validator_preserves_evidence_error():
    extraction = build_extraction(
        300, "typ"
    )

    evidence_result = verify_evidence(
        extraction,
        TABLES
    )

    validator = ParameterValidator()

    result = validator.validate(
        evidence_result
    )

    parameter = result["parameters"][0]

    assert parameter["validation_status"] == "invalid"

    assert parameter["validation_errors"]

    assert (
        result["validation_summary"]["invalid"]
        == 1
    )

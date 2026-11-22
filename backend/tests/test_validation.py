
from app.validation.units import (
    same_physical_value
)

from app.validation.rules import (
    check_min_typ_max
)

from app.validation.validator import (
    ParameterValidator
)


def test_unit_conversion():
    assert same_physical_value(
        200, "mV",
        0.2, "V"
    )


def test_min_typ_max_valid():
    parameters = [
        {
            "name": "dropout_voltage",
            "value": 200,
            "unit": "mV",
            "value_type": "typ",
            "test_conditions": "IOUT=500mA"
        },
        {
            "name": "dropout_voltage",
            "value": 300,
            "unit": "mV",
            "value_type": "max",
            "test_conditions": "IOUT=500mA"
        }
    ]

    errors = check_min_typ_max(parameters)

    assert errors == {}


def test_min_typ_max_invalid():
    parameters = [
        {
            "name": "dropout_voltage",
            "value": 350,
            "unit": "mV",
            "value_type": "typ",
            "test_conditions": "IOUT=500mA"
        },
        {
            "name": "dropout_voltage",
            "value": 300,
            "unit": "mV",
            "value_type": "max",
            "test_conditions": "IOUT=500mA"
        }
    ]

    errors = check_min_typ_max(parameters)

    assert 0 in errors
    assert 1 in errors


def test_validator():
    extraction = {
        "parameters": [
            {
                "name": "dropout_voltage",
                "value": 200,
                "unit": "mV",
                "value_type": "typ",
                "test_conditions": "IOUT=500mA",
                "validation_status": "validated"
            }
        ]
    }

    validator = ParameterValidator()

    result = validator.validate(extraction)

    assert (
        result["parameters"][0]["validation_status"]
        == "validated"
    )

    assert (
        result["validation_summary"]["validated"]
        == 1
    )

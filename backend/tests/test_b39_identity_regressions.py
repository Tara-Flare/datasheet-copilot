"""B3.9 在线验证发现的身份回归测试。

覆盖：
- Thermal Shutdown Hysteresis 不得被折叠成 UVLO hysteresis；
- PGFB Hysteresis 保留独立身份；
- 真正的 UVLO hysteresis 仍正常匹配；
- EN Pin Threshold Voltage 与模型输出 en_threshold_voltage 同义匹配；
- output_current_limit 对 internal/programmable 的同名歧义仍被拒绝。

不调用 LLM，不读取冻结 Gold。
"""

from app.validation.recovered_evidence import (
    _identity_match,
    logical_family,
    original_family,
)


def _row(symbol, name=""):
    return {
        "parameter_symbol": symbol,
        "parameter_name": name,
    }


def test_thermal_shutdown_hysteresis_keeps_its_identity():
    row = _row("Thermal Shutdown Hysteresis")

    assert logical_family(row) == "thermal_shutdown_hysteresis"
    assert (
        _identity_match(
            {"name": "thermal_shutdown_hysteresis"}, row
        )["status"]
        == "match"
    )


def test_pgfb_hysteresis_keeps_its_identity():
    row = _row("PGFB Hysteresis")

    assert logical_family(row) == "pgfb_hysteresis"
    assert (
        _identity_match({"name": "pgfb_hysteresis"}, row)[
            "status"
        ]
        == "match"
    )


def test_uvlo_hysteresis_still_matches():
    row = _row("VUVLO(HYST)", "UVLO hysteresis")

    assert logical_family(row) == "uvlo_hysteresis"
    assert (
        _identity_match({"name": "uvlo_hysteresis"}, row)[
            "status"
        ]
        == "match"
    )


def test_en_threshold_synonym_matches():
    row = _row("EN Pin Threshold Voltage")

    assert logical_family(row) == "en_pin_threshold_voltage"
    assert (
        _identity_match({"name": "en_threshold_voltage"}, row)[
            "status"
        ]
        == "match"
    )
    assert (
        _identity_match(
            {"name": "enable_threshold_voltage"}, row
        )["status"]
        == "match"
    )


def test_en_pin_current_synonym_matches():
    row = _row("EN Pin Current")

    assert logical_family(row) == "en_pin_current"
    assert (
        _identity_match({"name": "en_pin_current"}, row)[
            "status"
        ]
        == "match"
    )
    assert (
        _identity_match({"name": "enable_pin_current"}, row)[
            "status"
        ]
        == "match"
    )


def test_current_limit_ambiguity_is_still_rejected():
    row = _row("Internal Current Limit (Note 12)")

    assert original_family("output_current_limit") == "current_limit"
    result = _identity_match(
        {"name": "output_current_limit"}, row
    )

    assert result["status"] == "conflict"
    assert result["ambiguous"] is True

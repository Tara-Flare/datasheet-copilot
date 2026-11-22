
from dataclasses import dataclass


@dataclass
class NormalizedValue:
    value: float
    unit: str
    dimension: str


UNIT_MAP = {
    # 电压
    "V": ("voltage", 1.0, "V"),
    "mV": ("voltage", 0.001, "V"),
    "uV": ("voltage", 0.000001, "V"),

    # 电流
    "A": ("current", 1.0, "A"),
    "mA": ("current", 0.001, "A"),
    "uA": ("current", 0.000001, "A"),
    "nA": ("current", 0.000000001, "A"),

    # 温度
    "°C": ("temperature", 1.0, "°C"),

    # 频率
    "Hz": ("frequency", 1.0, "Hz"),
    "kHz": ("frequency", 1000.0, "Hz"),
    "MHz": ("frequency", 1000000.0, "Hz"),

    # 无量纲/特殊单位
    "%": ("percentage", 1.0, "%"),
    "dB": ("decibel", 1.0, "dB"),
}


def normalize_unit(unit: str) -> str:
    """
    统一单位字符表示。
    """

    unit = unit.strip()

    unit = unit.replace("μ", "u")
    unit = unit.replace("µ", "u")

    return unit


def convert_to_base(
    value: float,
    unit: str
) -> NormalizedValue:
    """
    将参数转换为基础单位。
    """

    unit = normalize_unit(unit)

    if unit not in UNIT_MAP:
        raise ValueError(
            f"不支持的单位: {unit}"
        )

    dimension, factor, base_unit = UNIT_MAP[unit]

    return NormalizedValue(
        value=value * factor,
        unit=base_unit,
        dimension=dimension
    )


def same_physical_value(
    value1: float,
    unit1: str,
    value2: float,
    unit2: str,
    tolerance: float = 1e-9
) -> bool:
    """
    判断两个带单位的值是否等价。
    """

    a = convert_to_base(value1, unit1)
    b = convert_to_base(value2, unit2)

    if a.dimension != b.dimension:
        return False

    return abs(a.value - b.value) <= (
        tolerance * max(
            abs(a.value),
            abs(b.value),
            1e-12
        )
    )


if __name__ == "__main__":
    print(convert_to_base(200, "mV"))
    print(convert_to_base(0.2, "V"))

    print(
        same_physical_value(
            200, "mV",
            0.2, "V"
        )
    )

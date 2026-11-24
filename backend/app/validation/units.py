
import re
from dataclasses import dataclass


@dataclass
class NormalizedValue:
    value: float
    unit: str
    dimension: str


# B3.8：补齐四份 PDF 中真实出现、但旧白名单未覆盖的复合单位。
# 复合单位各自使用独立 dimension，避免把 % 与 %/V、%/A 混为一谈，
# 也避免 min/typ/max 跨维度比较。
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

    # 时间
    "s": ("time", 1.0, "s"),
    "ms": ("time", 0.001, "s"),
    "us": ("time", 0.000001, "s"),

    # 电阻
    "ohm": ("resistance", 1.0, "ohm"),
    "kohm": ("resistance", 1000.0, "ohm"),
    "Mohm": ("resistance", 1000000.0, "ohm"),

    # 无量纲/特殊单位
    "%": ("percentage", 1.0, "%"),
    "dB": ("decibel", 1.0, "dB"),

    # 线性调整率 / 负载调整率（百分比每伏/安）
    "%/V": ("percentage_per_voltage", 1.0, "%/V"),
    "%/A": ("percentage_per_current", 1.0, "%/A"),
    "%/W": ("percentage_per_power", 1.0, "%/W"),

    # RMS 噪声电压
    "Vrms": ("rms_voltage", 1.0, "Vrms"),
    "VRMS": ("rms_voltage", 1.0, "Vrms"),
    "mVrms": ("rms_voltage", 0.001, "Vrms"),
    "uVrms": ("rms_voltage", 0.000001, "Vrms"),
    "uVRMS": ("rms_voltage", 0.000001, "Vrms"),

    # 温度系数 / 热阻 / 电流-电阻乘积
    "ppm/°C": ("temperature_coefficient", 1.0, "ppm/°C"),
    "°C/W": ("thermal_resistance", 1.0, "°C/W"),
    "mA*kohm": (
        "current_resistance_product", 1.0, "mA*kohm"
    ),
}


def normalize_unit(unit: str) -> str:
    """
    统一单位字符表示。

    只做字符级等价替换（微符号、欧姆符号、乘号、空白），
    不根据数值大小推断或改写单位。
    """

    text = str(unit or "").strip()

    text = text.replace("μ", "u")
    text = text.replace("µ", "u")
    text = text.replace("Ω", "ohm")
    text = text.replace("Ω", "ohm")
    text = text.replace("·", "*")
    text = text.replace("⋅", "*")

    return re.sub(r"\s+", "", text)


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

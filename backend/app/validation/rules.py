
from collections import defaultdict
from math import isfinite

from app.validation.units import (
    convert_to_base
)


def check_numeric_value(param: dict) -> list[str]:
    """检查数值和单位是否合法。"""

    errors = []

    value = param.get("value")
    unit = param.get("unit", "")

    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        return ["参数值不是有效数字"]

    if not isfinite(value):
        errors.append("参数值不是有限数字")

    try:
        convert_to_base(value, unit)
    except ValueError as exc:
        errors.append(str(exc))

    return errors


def check_min_typ_max(
    parameters: list[dict]
) -> dict[int, list[str]]:
    """
    检查同一参数、同一测试条件下
    Min <= Typ <= Max。

    返回：
    {参数索引: [错误信息]}
    """

    groups = defaultdict(list)

    for index, param in enumerate(parameters):
        key = (
            param["name"].strip().lower(),
            param.get(
                "test_conditions", ""
            ).strip().lower()
        )

        groups[key].append((index, param))

    errors = defaultdict(list)

    for group in groups.values():
        values = {}

        for index, param in group:
            value_type = param["value_type"]

            if value_type not in ("min", "typ", "max"):
                continue

            try:
                normalized = convert_to_base(
                    param["value"],
                    param["unit"]
                )
            except (ValueError, TypeError):
                continue

            values.setdefault(value_type, []).append(
                (index, normalized)
            )

        # 只有维度一致才可以比较
        for left_type, right_type in [
            ("min", "typ"),
            ("typ", "max"),
            ("min", "max"),
        ]:
            for left_idx, left in values.get(
                left_type, []
            ):
                for right_idx, right in values.get(
                    right_type, []
                ):
                    if left.dimension != right.dimension:
                        continue

                    if left.value > right.value:
                        message = (
                            f"{left_type} 大于 "
                            f"{right_type}"
                        )

                        errors[left_idx].append(message)
                        errors[right_idx].append(message)

    return dict(errors)

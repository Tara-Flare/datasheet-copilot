
import re
from collections import defaultdict
from math import isfinite

from app.validation.units import (
    convert_to_base
)


def _variant_key(param: dict):
    """区分同一参数/条件名下的不同表格变体。

    Datasheet 常把同一参数在多个输出型号下重复列出，
    LLM 可能给出相同的 test_conditions 文本。仅按
    (name, test_conditions) 分组会把不同变体的
    typ 与 max 交叉比较，产生假冲突。
    """

    evidence = param.get("evidence") or {}
    source_text = str(
        evidence.get("source_text") or ""
    )
    # 只保留字母/数字，避免同一行的 "|"、标点差异
    # 把 min/typ/max 拆成不同变体而漏掉真实冲突。
    source_text = re.sub(
        r"[^a-z0-9]+", " ", source_text.lower()
    )
    source_text = re.sub(
        r"\s+", " ", source_text
    ).strip()

    return (
        evidence.get("table_id"),
        evidence.get("row_index"),
        source_text,
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
        # 同一参数名/条件下再按来源变体细分，避免跨型号
        # 交叉比较。
        variants = defaultdict(list)

        for index, param in group:
            variants[_variant_key(param)].append(
                (index, param)
            )

        for variant in variants.values():
            values = {}

            for index, param in variant:
                value_type = param["value_type"]

                if value_type not in (
                    "min", "typ", "max",
                ):
                    continue

                try:
                    normalized = convert_to_base(
                        param["value"],
                        param["unit"]
                    )
                except (ValueError, TypeError):
                    continue

                values.setdefault(
                    value_type, []
                ).append((index, normalized))

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

                            errors[left_idx].append(
                                message
                            )
                            errors[right_idx].append(
                                message
                            )

    return dict(errors)

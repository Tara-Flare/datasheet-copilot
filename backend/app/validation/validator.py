
from copy import deepcopy

from app.validation.rules import (
    check_numeric_value,
    check_min_typ_max,
)


class ParameterValidator:

    def validate(self, extraction: dict) -> dict:
        result = deepcopy(extraction)

        parameters = result.get("parameters", [])

        relation_errors = check_min_typ_max(
            parameters
        )

        summary = {
            "total": len(parameters),
            "validated": 0,
            "invalid": 0,
            "pending_review": 0,
        }

        for index, param in enumerate(parameters):

            # 保留前面来源校验产生的错误
            evidence_errors = list(
                param.get("validation_errors", [])
            )

            # 结构性/数值性错误是硬错误；
            # 证据不足（如多值歧义）只是 pending_review，
            # 不能被压缩成 invalid。
            hard_errors = list(
                check_numeric_value(param)
            )
            hard_errors.extend(
                relation_errors.get(index, [])
            )

            # 错误信息去重
            errors = list(dict.fromkeys(
                evidence_errors + hard_errors
            ))

            previous_status = param.get(
                "validation_status",
                "pending_review"
            )

            if previous_status == "invalid":
                status = "invalid"

            elif hard_errors:
                status = "invalid"

            elif previous_status == "pending_review":
                status = "pending_review"

            elif previous_status == "validated":
                status = "validated"

            else:
                status = "pending_review"

            param["validation_status"] = status
            param["validation_errors"] = errors

            summary[status] += 1

        result["validation_summary"] = summary

        return result

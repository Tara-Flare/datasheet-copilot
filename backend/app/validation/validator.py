
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
            errors = list(
                param.get("validation_errors", [])
            )

            errors.extend(
                check_numeric_value(param)
            )

            errors.extend(
                relation_errors.get(index, [])
            )

            # 错误信息去重
            errors = list(dict.fromkeys(errors))

            previous_status = param.get(
                "validation_status",
                "pending_review"
            )

            if errors or previous_status == "invalid":
                status = "invalid"

            elif previous_status == "validated":
                status = "validated"

            else:
                status = "pending_review"

            param["validation_status"] = status
            param["validation_errors"] = errors

            summary[status] += 1

        result["validation_summary"] = summary

        return result

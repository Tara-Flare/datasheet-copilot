
from app.agents.state import DatasheetState


def route_after_validation(
    state: DatasheetState
) -> str:
    """根据校验状态选择下一节点。"""

    if state.get("status") == "validated":
        return "finish"

    result = state.get(
        "validation_result", {}
    )

    parameters = result.get(
        "parameters", []
    )

    invalid_parameters = [
        p for p in parameters
        if p.get("validation_status") == "invalid"
    ]

    pending_parameters = [
        p for p in parameters
        if p.get("validation_status") == "pending_review"
    ]

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # 证据不确定的字段需要人工复核
    if pending_parameters:
        return "review"

    # 没有明确错误，但未完成验证
    if not invalid_parameters:
        return "review"

    # 达到最大重试次数
    if retry_count >= max_retries:
        return "review"

    return "retry"



def route_after_review(state) -> str:
    """人工复核之后的路由。"""

    if state.get("status") == "reviewed":
        return "validate"

    return "end"

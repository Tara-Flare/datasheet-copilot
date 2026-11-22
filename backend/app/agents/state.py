
from typing import TypedDict, Literal, Any


class DatasheetState(TypedDict, total=False):
    """Datasheet Agent 共享状态。"""

    document_path: str

    document_info: dict[str, Any]
    tables: list[dict[str, Any]]

    extraction: dict[str, Any]
    validation_result: dict[str, Any]

    # B3.2a：Logical 输入装配结果。
    # 不覆盖 state["tables"]（原始 Flat 表格）。
    extraction_input_tables: list[dict[str, Any]]
    extraction_input_mode: str
    extraction_input_summary: dict[str, Any]

    retry_count: int
    max_retries: int

    evidence_validation_mode: str

    enable_human_review: bool

    status: Literal[
        "pending",
        "parsed",
        "extracted",
        "validated",
        "retrying",
        "pending_review",
        "completed",
        "failed",
        "reviewed",
        "rejected",
    ]

    errors: list[str]

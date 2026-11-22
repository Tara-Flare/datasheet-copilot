
from typing import Literal

from pydantic import BaseModel, Field


class ParameterEvidence(BaseModel):
    """参数来源信息。"""

    table_id: int = Field(
        description="来源表格 ID"
    )

    row_index: int = Field(
        description="来源行号，从 0 开始"
    )

    page: int = Field(
        description="来源 PDF 页码，从 1 开始"
    )

    source_text: str = Field(
        description="来源表格行中的原文摘录"
    )

    # B3.2a 向后兼容的可选字段。
    # Flat 输入模式下不会输出这些字段。
    logical_row_id: str | None = Field(
        default=None,
        description=(
            "Logical 输入模式下的逻辑行标识；"
            "不是 row_index"
        ),
    )

    source_row_indices: list[int] | None = Field(
        default=None,
        description=(
            "该逻辑行引用的全部原始表格行号"
        ),
    )

    source_cell_ids: list[int] | None = Field(
        default=None,
        description=(
            "该逻辑行引用的原始表格单元格 ID"
        ),
    )


class ElectricalParameter(BaseModel):
    """单个电气参数。"""

    name: str = Field(
        description="标准化参数名称"
    )

    value: float = Field(
        description="参数数值，不包含单位"
    )

    unit: str = Field(
        description="原始参数单位"
    )

    value_type: Literal[
        "min",
        "typ",
        "max",
        "nominal"
    ]

    test_conditions: str = Field(
        description="测试条件，未注明则为空字符串"
    )

    evidence: ParameterEvidence


class ExtractionResult(BaseModel):
    """LLM 结构化提取结果。"""

    component_model: str = Field(
        description="芯片型号，未知则为空字符串"
    )

    manufacturer: str = Field(
        description="制造商，未知则为空字符串"
    )

    parameters: list[ElectricalParameter]

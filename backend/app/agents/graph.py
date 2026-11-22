
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.state import DatasheetState
from app.agents.nodes import create_nodes


from app.agents.routers import (
    route_after_validation,
    route_after_review,
)


def build_graph(
    pdf_parser,
    table_parser,
    extractor,
    checkpointer=None,
    evidence_validation_mode=None,
    recovered_evidence=None,
    extraction_input_mode=None,
    logical_extraction=None,
):
    """构建 Datasheet Agent 工作流。

    只增加输入模式配置，不改变任何 LangGraph 路由拓扑。
    """

    nodes = create_nodes(
        pdf_parser,
        table_parser,
        extractor,
        evidence_validation_mode=evidence_validation_mode,
        recovered_evidence=recovered_evidence,
        extraction_input_mode=extraction_input_mode,
        logical_extraction=logical_extraction,
    )

    builder = StateGraph(DatasheetState)

    # 注册节点
    for name, node in nodes.items():
        builder.add_node(name, node)

    # 主流程
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "extract")
    builder.add_edge("extract", "validate")

    # 重试后重新执行完整参数校验
    builder.add_edge("retry", "validate")

    # 条件路由
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "finish": "finish",
            "retry": "retry",
            "review": "review",
        }
    )

    builder.add_edge("finish", END)
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {
            "validate": "validate",
            "end": END,
        }
    )

    # 默认使用内存 Checkpointer
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )

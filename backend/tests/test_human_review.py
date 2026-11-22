
from langgraph.types import Command

from app.agents.graph import build_graph

from test_agent_graph import (
    MockPDFParser,
    MockTableParser,
    MockExtractor,
)


def create_review_graph():
    """创建一个会提取错误参数的测试 Graph。"""

    return build_graph(
        MockPDFParser(),
        MockTableParser(),
        MockExtractor(value=300),
    )


def initial_state():
    return {
        "document_path": "test.pdf",
        "status": "pending",
        "retry_count": 0,
        "max_retries": 0,
        "enable_human_review": True,
        "errors": [],
    }


def test_human_review_correct():
    graph = create_review_graph()

    config = {
        "configurable": {
            "thread_id": "human-review-correct"
        }
    }

    # 第一次执行，等待人工复核
    result = graph.invoke(
        initial_state(),
        config=config
    )

    # 应触发中断
    assert "__interrupt__" in result

    # 查看当前暂停位置
    snapshot = graph.get_state(config)

    assert "review" in snapshot.next

    # 模拟人工将 Typ=300 修改为 Typ=200
    resumed = graph.invoke(
        Command(
            resume={
                "action": "correct",
                "corrections": [
                    {
                        "index": 0,
                        "value": 200
                    }
                ]
            }
        ),
        config=config
    )

    assert resumed["status"] == "completed"

    assert (
        resumed["validation_result"]
        ["validation_summary"]["validated"]
        == 1
    )


def test_human_review_reject():
    graph = create_review_graph()

    config = {
        "configurable": {
            "thread_id": "human-review-reject"
        }
    }

    result = graph.invoke(
        initial_state(),
        config=config
    )

    assert "__interrupt__" in result

    resumed = graph.invoke(
        Command(
            resume={
                "action": "reject"
            }
        ),
        config=config
    )

    assert resumed["status"] == "rejected"

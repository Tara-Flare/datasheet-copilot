
from app.agents.graph import build_graph
from test_agent_graph import (
    MockPDFParser,
    MockTableParser,
    MockExtractor,
)
from test_deepseek_extractor import (
    make_extractor,
    make_response,
    make_extraction,
    make_parameter,
)


class RetryExtractor(MockExtractor):
    """模拟第一次错误、重试后正确。"""

    def __init__(self):
        super().__init__(value=300)
        self.retry_calls = 0

    def retry_parameters(
        self,
        failed_parameters,
        tables
    ):
        self.retry_calls += 1

        corrected = self.extract(tables)

        corrected["parameters"][0]["value"] = 200

        return corrected


class AlwaysWrongExtractor(MockExtractor):
    """模拟模型一直无法纠正。"""

    def __init__(self):
        super().__init__(value=300)
        self.retry_calls = 0

    def retry_parameters(
        self,
        failed_parameters,
        tables
    ):
        self.retry_calls += 1

        return self.extract(tables)


def run_graph(extractor, thread_id):
    graph = build_graph(
        MockPDFParser(),
        MockTableParser(),
        extractor
    )

    result = graph.invoke(
        {
            "document_path": "test.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 2,
            "errors": [],
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return graph, result


def test_retry_success():
    extractor = RetryExtractor()

    graph, result = run_graph(
        extractor,
        "test-retry-success"
    )

    assert result["status"] == "completed"
    assert result["retry_count"] == 1
    assert extractor.retry_calls == 1


def test_retry_exhausted():
    extractor = AlwaysWrongExtractor()

    graph, result = run_graph(
        extractor,
        "test-retry-exhausted"
    )

    assert result["status"] == "pending_review"
    assert result["retry_count"] == 2
    assert extractor.retry_calls == 2



def test_checkpoint_state():
    extractor = RetryExtractor()

    graph = build_graph(
        MockPDFParser(),
        MockTableParser(),
        extractor
    )

    config = {
        "configurable": {
            "thread_id": "test-checkpoint"
        }
    }

    graph.invoke(
        {
            "document_path": "test.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 2,
            "errors": [],
        },
        config=config
    )

    snapshot = graph.get_state(config)

    assert snapshot.values["status"] == "completed"
    assert snapshot.values["retry_count"] == 1

    assert snapshot.values["validation_result"][
        "validation_summary"
    ]["validated"] == 1


def test_retry_does_not_lose_parameters_on_failure():
    """真实 DeepSeekExtractor + Mock 客户端：
    纠错始终失败时，原参数不被破坏，重试次数受 max_retries 限制。
    """

    # extract 返回错误值 300（校验失败），
    # 后续两次 retry 均返回空（无法纠正）。
    responses = [
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        value=300.0,
                        table_id=1,
                        row_index=0,
                        page=5,
                    )
                ]
            )
        ),
        make_response(make_extraction(parameters=[])),
        make_response(make_extraction(parameters=[])),
    ]

    extractor = make_extractor(responses)

    graph = build_graph(
        MockPDFParser(),
        MockTableParser(),
        extractor,
    )

    result = graph.invoke(
        {
            "document_path": "test.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 2,
            "errors": [],
        },
        config={
            "configurable": {
                "thread_id": "test-retry-no-loss"
            }
        },
    )

    assert result["status"] == "pending_review"
    assert result["retry_count"] == 2

    params = result["validation_result"]["parameters"]
    assert len(params) == 1
    assert params[0]["value"] == 300.0
    assert params[0]["validation_status"] == "invalid"


def test_retry_corrects_and_finishes():
    """真实 DeepSeekExtractor + Mock 客户端：
    纠错成功后可重新校验并完成。
    """

    responses = [
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        value=300.0,
                        table_id=1,
                        row_index=0,
                        page=5,
                    )
                ]
            )
        ),
        # retry 纠正为正确值 200
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        value=200.0,
                        table_id=1,
                        row_index=0,
                        page=5,
                    )
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)

    graph = build_graph(
        MockPDFParser(),
        MockTableParser(),
        extractor,
    )

    result = graph.invoke(
        {
            "document_path": "test.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 2,
            "errors": [],
        },
        config={
            "configurable": {
                "thread_id": "test-retry-correct"
            }
        },
    )

    assert result["status"] == "completed"
    assert result["retry_count"] == 1

    params = result["validation_result"]["parameters"]
    assert params[0]["value"] == 200.0
    assert params[0]["validation_status"] == "validated"

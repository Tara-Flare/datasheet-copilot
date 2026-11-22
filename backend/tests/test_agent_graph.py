
from app.agents.graph import build_graph


MOCK_TABLE = {
    "table_id": 1,
    "page_numbers": [5],
    "columns": [
        "Parameter",
        "Conditions",
        "Min",
        "Typ",
        "Max",
        "Unit",
    ],
    "rows": [
        [
            "Dropout Voltage",
            "IOUT=500mA",
            "",
            "200",
            "300",
            "mV",
        ]
    ],
}


class MockPDFParser:

    def parse(self, path):
        return {
            "filename": "test.pdf",
            "page_count": 10,
        }


class MockTableParser:

    def parse(self, path):
        return {
            "table_count": 1,
            "tables": [MOCK_TABLE],
        }


class MockExtractor:

    def __init__(self, value):
        self.value = value

    def extract(self, tables):
        return {
            "component_model": "TEST-LDO",
            "manufacturer": "TEST",
            "parameters": [
                {
                    "name": "dropout_voltage",
                    "value": self.value,
                    "unit": "mV",
                    "value_type": "typ",
                    "test_conditions": "IOUT=500mA",
                    "evidence": {
                        "table_id": 1,
                        "row_index": 0,
                        "page": 5,
                        "source_text": "Dropout Voltage",
                    }
                }
            ],
        }


def test_graph_success():
    graph = build_graph(
        MockPDFParser(),
        MockTableParser(),
        MockExtractor(value=200),
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
                "thread_id": "test-graph-success"
            }
        }
    )

    assert result["status"] == "completed"

    assert (
        result["validation_result"]
        ["validation_summary"]["validated"]
        == 1
    )


def test_graph_review():
    graph = build_graph(
        MockPDFParser(),
        MockTableParser(),
        MockExtractor(value=300),
    )

    result = graph.invoke(
        {
            "document_path": "test.pdf",
            "status": "pending",
            "retry_count": 0,

            # 禁止自动重试
            "max_retries": 0,

            "errors": [],
        },
        config={
            "configurable": {
                "thread_id": "test-graph-review"
            }
        }
    )

    assert result["status"] == "pending_review"

    assert (
        result["validation_result"]
        ["validation_summary"]["invalid"]
        == 1
    )

    # 确认没有执行 Retry Node
    assert result["retry_count"] == 0
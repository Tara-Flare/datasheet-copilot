
import json
from types import SimpleNamespace

import pytest

from app.extraction.deepseek_extractor import (
    DeepSeekExtractor,
    OutputTruncationError,
)
from app.extraction.evidence import verify_evidence


def make_response(
    content,
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=5,
):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=message,
    )
    return SimpleNamespace(
        usage=usage,
        choices=[choice],
    )


class FakeCompletions:
    """记录每次 create 调用参数并依次返回响应。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        item = self._responses.pop(0)

        if callable(item):
            item = item(kwargs)

        if isinstance(item, Exception):
            raise item

        return item


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, responses):
        completions = FakeCompletions(responses)
        self.chat = FakeChat(completions)
        self.completions = completions


def make_extractor(responses):
    """绕过真实 API 初始化，注入 Mock 客户端。"""

    extractor = DeepSeekExtractor.__new__(
        DeepSeekExtractor
    )
    extractor.model = "test-model"
    extractor.client = FakeClient(responses)
    extractor.prompt_tokens = 0
    extractor.completion_tokens = 0
    extractor.max_tokens = 6000
    extractor.rows_per_chunk = 10
    extractor.max_split_depth = 6
    return extractor


def make_table(table_id, page=1):
    return {
        "table_id": table_id,
        "page_numbers": [page],
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


def make_parameter(
    table_id=1,
    row_index=0,
    page=1,
    name="dropout_voltage",
    value=200.0,
    unit="mV",
    value_type="typ",
    test_conditions="IOUT=500mA",
    source_text="Dropout Voltage",
):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "value_type": value_type,
        "test_conditions": test_conditions,
        "evidence": {
            "table_id": table_id,
            "row_index": row_index,
            "page": page,
            "source_text": source_text,
        },
    }


def make_extraction(
    component_model="",
    manufacturer="",
    parameters=None,
):
    return json.dumps({
        "component_model": component_model,
        "manufacturer": manufacturer,
        "parameters": parameters or [],
    })


def make_failed_param(
    name="dropout_voltage",
    value=300.0,
    table_id=1,
    row_index=0,
    page=5,
):
    return {
        "name": name,
        "value": value,
        "unit": "mV",
        "value_type": "typ",
        "test_conditions": "IOUT=500mA",
        "validation_errors": [
            "提取数值与来源数值不一致"
        ],
        "evidence": {
            "table_id": table_id,
            "row_index": row_index,
            "page": page,
            "source_text": "Dropout Voltage",
        },
    }


def make_simple_table(table_id, row_count, page=1):
    """构造只有 Parameter/Typ 两列的简单表格。"""

    return {
        "table_id": table_id,
        "page_numbers": [page],
        "columns": ["Parameter", "Typ"],
        "rows": [
            ["r%s" % i]
            for i in range(row_count)
        ],
    }


def call_payload(call):
    return json.loads(
        call["messages"][1]["content"]
    )["tables"][0]


# ---------------------------------------------------------------
# 1. 单表正常提取
# ---------------------------------------------------------------

def test_extract_single_table():
    response = make_response(
        make_extraction(
            component_model="TPS7A02",
            manufacturer="TI",
            parameters=[make_parameter()],
        )
    )

    extractor = make_extractor([response])

    result = extractor.extract([make_table(1)])

    assert result["component_model"] == "TPS7A02"
    assert result["manufacturer"] == "TI"
    assert len(result["parameters"]) == 1
    assert (
        result["parameters"][0]["evidence"]["table_id"]
        == 1
    )

    assert len(extractor.client.completions.calls) == 1


# ---------------------------------------------------------------
# 2. 多表分批提取
# ---------------------------------------------------------------

def test_extract_multiple_tables_batched():
    responses = [
        make_response(
            make_extraction(
                parameters=[make_parameter(table_id=1)]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=2,
                        name="quiescent_current",
                        value=35.0,
                    )
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)

    result = extractor.extract([
        make_table(1),
        make_table(2),
    ])

    assert len(result["parameters"]) == 2

    calls = extractor.client.completions.calls
    assert len(calls) == 2

    for call in calls:
        payload = json.loads(
            call["messages"][1]["content"]
        )
        assert len(payload["tables"]) == 1

        assert call["max_tokens"] == 6000
        assert call["response_format"] == {
            "type": "json_object"
        }


# ---------------------------------------------------------------
# 3. 多批参数合并
# ---------------------------------------------------------------

def test_extract_merges_batches():
    responses = [
        make_response(
            make_extraction(
                component_model="TPS7A02",
                manufacturer="TI",
                parameters=[make_parameter(table_id=1)],
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=2,
                        name="psrr",
                        value=70.0,
                    )
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=3,
                        name="output_current",
                        value=500.0,
                    )
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)

    result = extractor.extract([
        make_table(1),
        make_table(2),
        make_table(3),
    ])

    assert result["component_model"] == "TPS7A02"
    assert result["manufacturer"] == "TI"

    assert [p["name"] for p in result["parameters"]] == [
        "dropout_voltage",
        "psrr",
        "output_current",
    ]


# ---------------------------------------------------------------
# 4. 空表格输入
# ---------------------------------------------------------------

def test_extract_empty_tables():
    extractor = make_extractor([])

    result = extractor.extract([])

    assert result == {
        "component_model": "",
        "manufacturer": "",
        "parameters": [],
    }

    assert len(extractor.client.completions.calls) == 0


# ---------------------------------------------------------------
# 5. 大表按行拆分
# ---------------------------------------------------------------

def test_big_table_split_into_chunks():
    table = make_simple_table(5, row_count=7, page=2)

    responses = [
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(table_id=5, row_index=0)
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(table_id=5, row_index=3)
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(table_id=5, row_index=6)
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)
    extractor.rows_per_chunk = 3

    result = extractor.extract([table])

    assert len(extractor.client.completions.calls) == 3
    assert len(result["parameters"]) == 3


# ---------------------------------------------------------------
# 6. 原始行号映射正确
# ---------------------------------------------------------------

def test_row_index_mapping_preserved():
    table = make_simple_table(9, row_count=5, page=4)

    responses = [
        make_response(make_extraction(parameters=[])),
        make_response(make_extraction(parameters=[])),
        make_response(make_extraction(parameters=[])),
    ]

    extractor = make_extractor(responses)
    extractor.rows_per_chunk = 2

    extractor.extract([table])

    row_indices_per_call = [
        call_payload(call)["row_indices"]
        for call in extractor.client.completions.calls
    ]

    assert row_indices_per_call == [
        [0, 1],
        [2, 3],
        [4],
    ]


# ---------------------------------------------------------------
# 7. 多个行块结果合并
# ---------------------------------------------------------------

def test_multiple_row_chunks_merged():
    table = make_simple_table(3, row_count=6, page=1)

    responses = [
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=3,
                        row_index=0,
                        name="dropout_voltage",
                    )
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=3,
                        row_index=3,
                        name="quiescent_current",
                        value=35.0,
                        unit="uA",
                        source_text="Quiescent Current",
                    )
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)
    extractor.rows_per_chunk = 3

    result = extractor.extract([table])

    assert [p["name"] for p in result["parameters"]] == [
        "dropout_voltage",
        "quiescent_current",
    ]

    assert result["parameters"][0]["evidence"]["row_index"] == 0
    assert result["parameters"][1]["evidence"]["row_index"] == 3


# ---------------------------------------------------------------
# 8. 单个行块截断后自动二分
# ---------------------------------------------------------------

def test_chunk_truncation_binary_split():
    table = make_simple_table(1, row_count=4, page=1)

    responses = [
        make_response("", finish_reason="length"),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(table_id=1, row_index=0)
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(table_id=1, row_index=2)
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)

    result = extractor.extract([table])

    calls = extractor.client.completions.calls
    assert len(calls) == 3

    # 首次整块 4 行被截断，随后二分为两个 2 行子块
    assert len(call_payload(calls[0])["rows"]) == 4
    assert len(call_payload(calls[1])["rows"]) == 2
    assert len(call_payload(calls[2])["rows"]) == 2

    assert len(result["parameters"]) == 2


# ---------------------------------------------------------------
# 9. 二分后的两个子块全部成功并合并
# ---------------------------------------------------------------

def test_binary_split_halves_merged():
    table = make_simple_table(2, row_count=4, page=3)

    responses = [
        make_response("", finish_reason="length"),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=2,
                        row_index=0,
                        name="dropout_voltage",
                    )
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=2,
                        row_index=2,
                        name="quiescent_current",
                        value=35.0,
                    )
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)

    result = extractor.extract([table])

    assert [p["name"] for p in result["parameters"]] == [
        "dropout_voltage",
        "quiescent_current",
    ]

    assert result["parameters"][0]["evidence"]["row_index"] == 0
    assert result["parameters"][1]["evidence"]["row_index"] == 2


# ---------------------------------------------------------------
# 10. 超过最大拆分深度时正确报错
# ---------------------------------------------------------------

def test_max_split_depth_exceeded():
    table = make_simple_table(1, row_count=4, page=1)

    extractor = make_extractor([
        make_response("", finish_reason="length")
    ])
    extractor.max_split_depth = 0

    with pytest.raises(OutputTruncationError) as exc_info:
        extractor._extract_chunk(
            table,
            table["rows"],
            [0, 1, 2, 3],
            depth=0,
        )

    assert "最大拆分深度" in str(exc_info.value)


# ---------------------------------------------------------------
# 11. 单行仍然截断时正确报错
# ---------------------------------------------------------------

def test_single_row_truncation():
    table = make_simple_table(1, row_count=1, page=1)

    extractor = make_extractor([
        make_response("", finish_reason="length")
    ])

    with pytest.raises(OutputTruncationError) as exc_info:
        extractor._extract_chunk(
            table,
            table["rows"],
            [0],
            depth=0,
        )

    assert "单行" in str(exc_info.value)


def test_extract_single_row_truncation_raises():
    table = make_simple_table(1, row_count=1, page=1)

    extractor = make_extractor([
        make_response("", finish_reason="length")
    ])

    with pytest.raises(ValueError) as exc_info:
        extractor.extract([table])

    message = str(exc_info.value)

    assert "1 个行块" in message
    assert "OutputTruncationError" in message


# ---------------------------------------------------------------
# 12. 单批失败（非截断）时的异常处理
# ---------------------------------------------------------------

def test_extract_single_batch_failure_raises():
    responses = [
        make_response(
            make_extraction(
                parameters=[make_parameter(table_id=1)]
            )
        ),
        ValueError("模拟批次失败"),
        make_response(
            make_extraction(
                parameters=[make_parameter(table_id=3)]
            )
        ),
    ]

    extractor = make_extractor(responses)

    with pytest.raises(ValueError) as exc_info:
        extractor.extract([
            make_table(1),
            make_table(2),
            make_table(3),
        ])

    message = str(exc_info.value)

    assert "1 个行块" in message
    assert "table_id" in message
    assert "ValueError" in message

    assert len(extractor.client.completions.calls) == 3


# ---------------------------------------------------------------
# 13. Token 使用量累计
# ---------------------------------------------------------------

def test_token_accumulation():
    responses = [
        make_response(
            make_extraction(
                parameters=[make_parameter(table_id=1)]
            ),
            prompt_tokens=100,
            completion_tokens=50,
        ),
        make_response(
            make_extraction(
                parameters=[make_parameter(table_id=2)]
            ),
            prompt_tokens=200,
            completion_tokens=80,
        ),
    ]

    extractor = make_extractor(responses)

    extractor.extract([
        make_table(1),
        make_table(2),
    ])

    usage = extractor.get_usage()

    assert usage["prompt_tokens"] == 300
    assert usage["completion_tokens"] == 130
    assert usage["total_tokens"] == 430


# ---------------------------------------------------------------
# 14. 分批后的 Evidence 元数据保持不变
# ---------------------------------------------------------------

def test_evidence_preserved_across_batches():
    p1 = make_parameter(
        table_id=7,
        row_index=1,
        page=3,
        name="dropout_voltage",
    )
    p2 = make_parameter(
        table_id=8,
        row_index=2,
        page=6,
        name="quiescent_current",
        value=35.0,
        unit="uA",
        source_text="Quiescent Current",
    )

    responses = [
        make_response(
            make_extraction(parameters=[p1])
        ),
        make_response(
            make_extraction(parameters=[p2])
        ),
    ]

    extractor = make_extractor(responses)

    result = extractor.extract([
        make_simple_table(7, row_count=3, page=3),
        make_simple_table(8, row_count=3, page=6),
    ])

    assert (
        result["parameters"][0]["evidence"]
        == p1["evidence"]
    )
    assert (
        result["parameters"][1]["evidence"]
        == p2["evidence"]
    )


# ---------------------------------------------------------------
# 15. Evidence 来源校验仍然有效
# ---------------------------------------------------------------

def test_evidence_validation_still_works():
    table = {
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
            ],
            [
                "Quiescent Current",
                "IOUT=0mA",
                "",
                "35",
                "",
                "uA",
            ],
        ],
    }

    p1 = make_parameter(
        table_id=1,
        row_index=0,
        page=5,
        name="dropout_voltage",
    )
    p2 = make_parameter(
        table_id=1,
        row_index=1,
        page=5,
        name="quiescent_current",
        value=35.0,
        unit="uA",
        source_text="Quiescent Current",
    )

    responses = [
        make_response(make_extraction(parameters=[p1])),
        make_response(make_extraction(parameters=[p2])),
    ]

    extractor = make_extractor(responses)
    extractor.rows_per_chunk = 1

    merged = extractor.extract([table])

    verified = verify_evidence(merged, [table])

    assert all(
        p["validation_status"] == "validated"
        for p in verified["parameters"]
    )


# ---------------------------------------------------------------
# 16. 截断异常由 _request_json 直接抛出
# ---------------------------------------------------------------

def test_request_json_truncation():
    extractor = make_extractor([
        make_response("", finish_reason="length")
    ])

    with pytest.raises(ValueError) as exc_info:
        extractor._request_json(
            payload={"tables": [make_table(1)]},
            instruction="测试指令",
        )

    assert "截断" in str(exc_info.value)


# ---------------------------------------------------------------
# 17. max_tokens 环境变量配置
# ---------------------------------------------------------------

def test_max_tokens_default(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    extractor = DeepSeekExtractor()

    assert extractor.max_tokens == 6000


def test_max_tokens_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_TOKENS", "7000")

    extractor = DeepSeekExtractor()

    assert extractor.max_tokens == 7000


# ===============================================================
# 参数纠错（retry_parameters）测试
# ===============================================================

def test_retry_single_param_corrected():
    table = make_table(1, page=5)

    failed = [make_failed_param(value=300.0)]

    response = make_response(
        make_extraction(
            parameters=[make_parameter(value=200.0)]
        )
    )

    extractor = make_extractor([response])

    result = extractor.retry_parameters(
        failed,
        [table],
    )

    assert len(result["parameters"]) == 1

    param = result["parameters"][0]

    assert param["value"] == 200.0
    assert param["evidence"]["table_id"] == 1
    assert param["evidence"]["row_index"] == 0
    assert param["evidence"]["page"] == 5


def test_retry_multiple_params_batched():
    table = make_simple_table(1, row_count=6, page=2)

    failed = [
        make_failed_param(
            name="p%s" % i,
            table_id=1,
            row_index=i,
            page=2,
        )
        for i in range(6)
    ]

    responses = [
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        name="p0",
                        table_id=1,
                        row_index=0,
                        page=2,
                    )
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        name="p3",
                        table_id=1,
                        row_index=3,
                        page=2,
                    )
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)
    extractor.rows_per_chunk = 3

    result = extractor.retry_parameters(
        failed,
        [table],
    )

    assert len(extractor.client.completions.calls) == 2
    assert len(result["parameters"]) == 2


def test_retry_groups_by_table_id():
    table1 = make_table(1, page=5)
    table2 = make_table(2, page=7)

    failed = [
        make_failed_param(
            name="dropout_voltage",
            table_id=1,
            row_index=0,
            page=5,
        ),
        make_failed_param(
            name="dropout_voltage",
            table_id=2,
            row_index=0,
            page=7,
        ),
    ]

    responses = [
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=1,
                        row_index=0,
                        page=5,
                    )
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        table_id=2,
                        row_index=0,
                        page=7,
                    )
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)
    extractor.rows_per_chunk = 1

    result = extractor.retry_parameters(
        failed,
        [table1, table2],
    )

    calls = extractor.client.completions.calls
    assert len(calls) == 2
    assert len(result["parameters"]) == 2

    first = json.loads(
        calls[0]["messages"][1]["content"]
    )["failed_parameters"][0]

    second = json.loads(
        calls[1]["messages"][1]["content"]
    )["failed_parameters"][0]

    assert first["table_id"] == 1
    assert first["source_row"] == table1["rows"][0]

    assert second["table_id"] == 2
    assert second["source_row"] == table2["rows"][0]


def test_retry_preserves_original_evidence():
    table = make_table(1, page=5)

    failed = [make_failed_param(value=300.0)]

    wrong_param = make_parameter(value=200.0)
    wrong_param["evidence"] = {
        "table_id": 999,
        "row_index": 7,
        "page": 9,
        "source_text": "Dropout Voltage",
    }

    response = make_response(
        make_extraction(parameters=[wrong_param])
    )

    extractor = make_extractor([response])

    result = extractor.retry_parameters(
        failed,
        [table],
    )

    evidence = result["parameters"][0]["evidence"]

    assert evidence["table_id"] == 1
    assert evidence["row_index"] == 0
    assert evidence["page"] == 5


def test_retry_truncation_binary_split():
    table = make_simple_table(1, row_count=4, page=1)

    failed = [
        make_failed_param(
            name="p%s" % i,
            table_id=1,
            row_index=i,
            page=1,
        )
        for i in range(4)
    ]

    responses = [
        make_response("", finish_reason="length"),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        name="p0",
                        table_id=1,
                        row_index=0,
                        page=1,
                    )
                ]
            )
        ),
        make_response(
            make_extraction(
                parameters=[
                    make_parameter(
                        name="p2",
                        table_id=1,
                        row_index=2,
                        page=1,
                    )
                ]
            )
        ),
    ]

    extractor = make_extractor(responses)

    result = extractor.retry_parameters(
        failed,
        [table],
    )

    assert len(extractor.client.completions.calls) == 3
    assert len(result["parameters"]) == 2


def test_retry_invalid_table_id():
    table = make_table(1)

    failed = [
        make_failed_param(
            table_id=999,
            row_index=0,
            page=5,
        )
    ]

    extractor = make_extractor([])

    result = extractor.retry_parameters(
        failed,
        [table],
    )

    assert result["parameters"] == []
    assert len(extractor.client.completions.calls) == 0


def test_retry_invalid_row_index():
    table = make_table(1)

    failed = [
        make_failed_param(
            table_id=1,
            row_index=5,
            page=5,
        )
    ]

    extractor = make_extractor([])

    result = extractor.retry_parameters(
        failed,
        [table],
    )

    assert result["parameters"] == []
    assert len(extractor.client.completions.calls) == 0


def test_retry_corrected_passes_evidence_validation():
    table = make_table(1, page=5)

    failed = [make_failed_param(value=300.0)]

    response = make_response(
        make_extraction(
            parameters=[make_parameter(value=200.0)]
        )
    )

    extractor = make_extractor([response])

    corrected = extractor.retry_parameters(
        failed,
        [table],
    )

    extraction = {
        "component_model": "",
        "manufacturer": "",
        "parameters": corrected["parameters"],
    }

    verified = verify_evidence(extraction, [table])

    assert (
        verified["parameters"][0]["validation_status"]
        == "validated"
    )

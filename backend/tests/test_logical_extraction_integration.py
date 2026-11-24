"""阶段 B3.2a：LogicalTableAdapter 与 DeepSeekExtractor 安全集成。

全部测试使用 Mock LLM，不访问真实 DeepSeek API，
不写入 data/outputs、PDF 或缓存。
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents.graph import build_graph
from app.extraction.deepseek_extractor import (
    DeepSeekExtractor,
    OutputTruncationError,
)
from app.extraction.evidence import verify_evidence
from app.extraction.input_mode import (
    ExtractionInputModeError,
    build_extraction_input,
    resolve_extraction_input_mode,
)
from app.extraction.logical_table_adapter import (
    adapt_logical_rows,
)
from app.extraction.logical_source import (
    canonicalize_parameters,
)
from app.validation.recovered_evidence import (
    RecoveredEvidenceValidator,
)
from app.validation.recovered_integration import (
    build_recovered_validation_result,
)
from app.validation.validator import ParameterValidator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGICAL_PATH = (
    PROJECT_ROOT / "data/outputs/TPS7A02.logical_rows.v1.json"
)
RAW_PATH = (
    PROJECT_ROOT / "data/outputs/TPS7A02.raw_structure.v1.json"
)
CACHE_DIR = PROJECT_ROOT / "data/cache/docling"

pytestmark = pytest.mark.requires_data


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached_table(table_id):
    for path in CACHE_DIR.glob("*.json"):
        data = load(path)

        for table in data.get("tables", []):
            if table.get("table_id") == table_id:
                return table

    raise KeyError(f"缓存中无 table_id={table_id}")


@pytest.fixture(scope="module")
def artifacts():
    return {
        "logical_rows": load(LOGICAL_PATH),
        "raw_structure": load(RAW_PATH),
        "table9": load_cached_table(9),
        "table7": load_cached_table(7),
    }


@pytest.fixture(scope="module")
def logical_input(artifacts):
    return build_extraction_input(
        [
            artifacts["table9"],
            artifacts["table7"],
        ],
        mode="logical",
        logical_rows=artifacts["logical_rows"],
        raw_structure=artifacts["raw_structure"],
    )


def logical_records(logical_input):
    table = next(
        table for table in logical_input["tables"]
        if table.get("table_id") == 9
    )
    return {
        record["logical_row_id"]: record
        for record in table["records"]
    }


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


def make_extractor(responses, *, rows_per_chunk=10):
    extractor = DeepSeekExtractor.__new__(
        DeepSeekExtractor
    )
    extractor.model = "mock-model"
    extractor.client = FakeClient(responses)
    extractor.prompt_tokens = 0
    extractor.completion_tokens = 0
    extractor.max_tokens = 6000
    extractor.rows_per_chunk = rows_per_chunk
    extractor.max_split_depth = 6
    extractor.input_mode = "flat"
    extractor.logical_rows = None
    extractor.raw_structure = None
    extractor.logical_table_ids = None
    extractor.include_needs_review = False
    return extractor


def make_param(
    name,
    value,
    unit,
    value_type,
    test_conditions,
    *,
    table_id=9,
    row_index=0,
    page=6,
    logical_row_id=None,
    source_text="",
    source_row_indices=None,
):
    evidence = {
        "table_id": table_id,
        "row_index": row_index,
        "page": page,
        "source_text": source_text,
    }

    if logical_row_id is not None:
        evidence["logical_row_id"] = logical_row_id

    if source_row_indices is not None:
        evidence["source_row_indices"] = (
            source_row_indices
        )

    return {
        "name": name,
        "value": value,
        "unit": unit,
        "value_type": value_type,
        "test_conditions": test_conditions,
        "evidence": evidence,
    }


def make_extraction(parameters, model="TPS7A02"):
    return json.dumps({
        "component_model": model,
        "manufacturer": "TI",
        "parameters": parameters,
    })


def payload_of_call(call):
    return json.loads(
        call["messages"][1]["content"]
    )["tables"][0]


def recovered_config(artifacts):
    return {
        "logical_rows": artifacts["logical_rows"],
        "raw_structure": artifacts["raw_structure"],
        "pdf_sha256": artifacts["raw_structure"]["source"][
            "pdf_sha256"
        ],
        "on_error": "pending_review",
    }


def logical_table(artifacts, *, include_needs_review=False):
    adapted = adapt_logical_rows(
        artifacts["logical_rows"],
        raw_structure=artifacts["raw_structure"],
        table_ids=[9],
        include_needs_review=include_needs_review,
    )
    return adapted["tables"][0]


def raw_report(artifacts, extraction):
    """直接运行 RecoveredEvidenceValidator，查看字段级证据。"""

    return RecoveredEvidenceValidator(
        artifacts["logical_rows"],
        raw_structure=artifacts["raw_structure"],
    ).validate(extraction)


# ---------------------------------------------------------------
# 1. 模式选择
# ---------------------------------------------------------------

def test_input_mode_resolution(monkeypatch):
    monkeypatch.delenv(
        "EXTRACTION_INPUT_MODE", raising=False
    )
    assert resolve_extraction_input_mode() == "flat"
    assert resolve_extraction_input_mode("flat") == "flat"
    assert (
        resolve_extraction_input_mode("logical")
        == "logical"
    )

    monkeypatch.setenv(
        "EXTRACTION_INPUT_MODE", "logical"
    )
    assert resolve_extraction_input_mode() == "logical"

    with pytest.raises(ExtractionInputModeError):
        resolve_extraction_input_mode("auto")


def test_extractor_reads_input_mode_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv(
        "EXTRACTION_INPUT_MODE", "logical"
    )

    extractor = DeepSeekExtractor()

    assert extractor.input_mode == "logical"


# ---------------------------------------------------------------
# 2. Logical / Flat 混合装配
# ---------------------------------------------------------------

def test_logical_input_keeps_unrecovered_table_flat(
    artifacts, logical_input
):
    assert logical_input["effective_mode"] == "logical"
    assert logical_input["logical_table_ids"] == [9]
    assert logical_input["flat_table_ids"] == [7]

    table9 = logical_input["tables"][0]
    table7 = logical_input["tables"][1]

    assert table9["input_mode"] == "logical"
    assert table9["row_count"] == 33
    assert table9["row_indices"].count(30) == 2
    assert all(
        row["logical_row_id"] not in {
            "t9.p34", "t9.p35", "t9.p36"
        }
        for row in table9["records"]
    )

    # 未恢复表格继续使用原始 Flat 对象内容。
    assert table7["table_id"] == 7
    assert "input_mode" not in table7
    assert table7 == artifacts["table7"]


def test_no_recovered_table_falls_back_flat(
    artifacts, monkeypatch
):
    monkeypatch.delenv(
        "EXTRACTION_INPUT_MODE", raising=False
    )
    info = build_extraction_input(
        [artifacts["table7"]],
        mode="logical",
        logical_rows=artifacts["logical_rows"],
        raw_structure=artifacts["raw_structure"],
    )

    assert info["effective_mode"] == "flat_fallback"
    assert info["tables"] == [artifacts["table7"]]


# ---------------------------------------------------------------
# 3. Flat 兼容
# ---------------------------------------------------------------

def test_flat_mode_payload_and_evidence_unchanged():
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
            ]
        ],
    }

    parameter = make_param(
        "dropout_voltage",
        200.0,
        "mV",
        "typ",
        "IOUT=500mA",
        table_id=1,
        row_index=0,
        page=5,
        source_text="Dropout Voltage",
    )

    extractor = make_extractor([
        make_response(
            make_extraction([parameter])
        )
    ])

    result = extractor.extract([table])
    call_payload = payload_of_call(
        extractor.client.completions.calls[0]
    )

    assert set(call_payload) == {
        "table_id",
        "page_numbers",
        "columns",
        "rows",
        "row_indices",
        "row_count",
        "column_count",
    }

    evidence = result["parameters"][0]["evidence"]

    assert evidence == {
        "table_id": 1,
        "row_index": 0,
        "page": 5,
        "source_text": "Dropout Voltage",
    }
    assert "logical_row_id" not in evidence
    assert "source_row_indices" not in evidence
    assert "source_mapping_status" not in (
        result["parameters"][0]
    )


# ---------------------------------------------------------------
# 4. 360 / 310 独立来源映射
# ---------------------------------------------------------------

def test_360_310_independent_mapping_and_validation(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    first = records["t9.p30"]
    second = records["t9.p31"]

    # source_text 仍由程序按 canonical record 重写；
    # 来源行号合法，来源字段检查应通过。
    p30 = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "max",
        first["test_conditions"],
        row_index=30,
        logical_row_id="t9.p30",
        source_text="LLM invented source",
        source_row_indices=[18, 25, 30],
    )
    p31 = make_param(
        "dropout_voltage",
        310.0,
        "mV",
        "max",
        second["test_conditions"],
        row_index=30,
        logical_row_id="t9.p31",
        source_text="LLM invented source",
        source_row_indices=[18, 25, 30],
    )

    extractor = make_extractor(
        [make_response(make_extraction([p30, p31]))],
        rows_per_chunk=33,
    )

    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )

    result = extractor.extract([table9])
    call_payload = payload_of_call(
        extractor.client.completions.calls[0]
    )

    assert call_payload["input_mode"] == "logical"
    assert call_payload["logical_row_ids"].count(
        "t9.p30"
    ) == 1
    assert call_payload["row_indices"].count(30) == 2

    parameters = result["parameters"]

    assert len(parameters) == 2

    first_param, second_param = parameters

    assert (
        first_param["evidence"]["logical_row_id"]
        == "t9.p30"
    )
    assert (
        second_param["evidence"]["logical_row_id"]
        == "t9.p31"
    )
    assert first_param["evidence"]["row_index"] == 30
    assert second_param["evidence"]["row_index"] == 30
    assert (
        first_param["evidence"]["source_row_indices"]
        == [18, 25, 30]
    )
    assert (
        second_param["evidence"]["source_row_indices"]
        == [18, 25, 30]
    )
    assert "360" in first_param["evidence"]["source_text"]
    assert "310" in second_param["evidence"]["source_text"]
    assert (
        first_param["source_mapping_status"] == "mapped"
    )

    report = raw_report(artifacts, result)

    validated = {
        item["selected_logical_row_id"]: item
        for item in report["parameters"]
    }

    assert validated["t9.p30"]["recovered_status"] == "validated"
    assert validated["t9.p31"]["recovered_status"] == "validated"
    assert report["summary"]["recovered_validated"] == 2


# ---------------------------------------------------------------
# 5. 850 mV 完整条件 / 270 mV 缺少温度
# ---------------------------------------------------------------

def test_850_complete_and_270_missing_temperature(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    p850 = make_param(
        "dropout_voltage",
        850.0,
        "mV",
        "max",
        records["t9.p26"]["test_conditions"],
        row_index=26,
        logical_row_id="t9.p26",
    )
    p270 = make_param(
        "dropout_voltage",
        270.0,
        "mV",
        "max",
        # 故意缺少 TJ = -40..+85°C
        "3.3 V ≤ VOUT ≤ 5.0 V",
        row_index=24,
        logical_row_id="t9.p24",
    )

    extractor = make_extractor(
        [make_response(make_extraction([p850, p270]))],
        rows_per_chunk=33,
    )

    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )
    result = extractor.extract([table9])

    report = raw_report(artifacts, result)

    by_row = {
        item["selected_logical_row_id"]: item
        for item in report["parameters"]
    }

    assert by_row["t9.p26"]["recovered_status"] == "validated"
    assert (
        by_row["t9.p24"]["recovered_status"]
        == "correctable"
    )
    assert "test_conditions" in by_row["t9.p24"][
        "missing_fields"
    ]


# ---------------------------------------------------------------
# 6. Ground Current 归属
# ---------------------------------------------------------------

def test_ground_current_1_percent_and_60na_attribution(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    p1_percent = make_param(
        "quiescent_current",
        1.0,
        "%",
        "typ",
        records["t9.p10"]["test_conditions"],
        row_index=9,
        logical_row_id="t9.p10",
    )
    p60_na = make_param(
        "quiescent_current",
        60.0,
        "nA",
        "max",
        records["t9.p9"]["test_conditions"],
        row_index=9,
        logical_row_id="t9.p9",
    )

    extractor = make_extractor(
        [make_response(make_extraction([
            p1_percent, p60_na
        ]))],
        rows_per_chunk=33,
    )

    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )
    result = extractor.extract([table9])

    report = raw_report(artifacts, result)
    params = report["parameters"]

    first, second = params

    # 1% 必须归属 t9.p10，不能归属 60 nA 的 t9.p9。
    assert (
        first["selected_logical_row_id"] == "t9.p10"
    )
    assert first["field_checks"]["value"]["status"] == "match"
    assert first["value_type"] == "typ"
    assert first["unit"] == "%"
    assert first["selected_logical_row_id"] != "t9.p9"

    # 60 nA 必须归属 t9.p9，完整条件可严格通过。
    assert (
        second["selected_logical_row_id"] == "t9.p9"
    )
    assert second["recovered_status"] == "validated"
    assert second["field_checks"]["value"]["status"] == "match"


# ---------------------------------------------------------------
# 7. needs_review 不自动通过
# ---------------------------------------------------------------

def test_needs_review_row_never_auto_validates(
    artifacts,
):
    table9 = logical_table(
        artifacts, include_needs_review=True
    )
    record = next(
        row for row in table9["records"]
        if row["logical_row_id"] == "t9.p34"
    )

    assert record["needs_review"] is True

    param = make_param(
        "input_voltage",
        float(record["typ"]),
        record["unit"],
        "typ",
        "VIN rising",
        row_index=record["primary_row_index"],
        logical_row_id="t9.p34",
    )

    extractor = make_extractor(
        [make_response(make_extraction([param]))],
        rows_per_chunk=36,
    )

    result = extractor.extract([table9])

    report = raw_report(artifacts, result)
    item = report["parameters"][0]

    assert item["selected_logical_row_id"] == "t9.p34"
    assert item["recovered_status"] == "pending_review"
    assert (
        item["field_checks"]["needs_review"][
            "logical_row_needs_review"
        ]
        is True
    )


def test_llm_invented_logical_row_id_is_not_trusted(
    artifacts, logical_input
):
    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )

    param = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "max",
        "TJ = -40°C to +125°C; 2.5 V ≤ VOUT < 3.3 V",
        row_index=30,
        logical_row_id="t9.p999",
    )

    extractor = make_extractor(
        [make_response(make_extraction([param]))],
        rows_per_chunk=33,
    )
    result = extractor.extract([table9])

    mapped = result["parameters"][0]

    assert (
        mapped["source_mapping_status"]
        == "unknown_logical_row_id"
    )
    assert mapped["source_mapping"]["trusted"] is False

    report = raw_report(artifacts, result)
    assert (
        report["parameters"][0]["recovered_status"]
        == "pending_review"
    )


def test_ambiguous_row_index_without_id_is_not_trusted(
    artifacts, logical_input
):
    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )

    # t9.p30 / t9.p31 都引用 row 30；没有 ID 不能猜测。
    param = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "max",
        "TJ = -40°C to +125°C; 2.5 V ≤ VOUT < 3.3 V",
        row_index=30,
    )

    extractor = make_extractor(
        [make_response(make_extraction([param]))],
        rows_per_chunk=33,
    )
    result = extractor.extract([table9])

    mapped = result["parameters"][0]

    assert (
        mapped["source_mapping_status"]
        == "ambiguous_row_index"
    )
    assert mapped["source_mapping"]["trusted"] is False

    report = raw_report(artifacts, result)
    assert (
        report["parameters"][0]["recovered_status"]
        == "pending_review"
    )


# ---------------------------------------------------------------
# 8. 未恢复表格正常处理
# ---------------------------------------------------------------

def test_unrecovered_table_keeps_flat_validation(
    artifacts, logical_input
):
    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )
    table7 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 7
    )

    records = logical_records(logical_input)
    p30 = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "max",
        records["t9.p30"]["test_conditions"],
        row_index=30,
        logical_row_id="t9.p30",
    )

    flat_param = make_param(
        "input_voltage",
        1.5,
        "V",
        "min",
        "",
        table_id=7,
        row_index=0,
        page=5,
        source_text=(
            "VIN Input voltage 1.5 6.0 V"
        ),
    )

    extractor = make_extractor(
        [
            make_response(make_extraction([p30])),
            make_response(make_extraction([flat_param])),
        ],
        rows_per_chunk=33,
    )

    result = extractor.extract([table9, table7])

    assert "source_mapping_status" not in (
        result["parameters"][1]
    )

    # 校验需要同时包含 table 7 的 Flat 数据，
    # 按两张表构造 legacy 基线。
    legacy = ParameterValidator().validate(
        verify_evidence(result, [table9, table7])
    )
    report = build_recovered_validation_result(
        result,
        legacy,
        recovered_config(artifacts),
    )

    flat = report["parameters"][1]

    assert flat["validation_source"] == "legacy"
    assert flat["detailed_recovery_status"] == "not_covered"
    assert flat["validation_status"] == "validated"
    assert report["validation_summary"]["legacy_validated"] == 1


# ---------------------------------------------------------------
# 9. 分块 / 截断 / Token 统计保留
# ---------------------------------------------------------------

def test_logical_chunking_and_truncation_preserved(
    artifacts, logical_input
):
    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )

    extractor = make_extractor(
        [
            make_response("", finish_reason="length"),
            make_response(make_extraction([
                make_param(
                    "dropout_voltage",
                    360.0,
                    "mV",
                    "max",
                    "TJ = -40°C to +125°C; "
                    "2.5 V ≤ VOUT < 3.3 V",
                    row_index=30,
                    logical_row_id="t9.p30",
                )
            ])),
            make_response(make_extraction([])),
        ],
        rows_per_chunk=2,
    )

    result = extractor._extract_chunk(
        table9,
        table9["rows"][:2],
        table9["row_indices"][:2],
        record_positions=[0, 1],
    )

    calls = extractor.client.completions.calls

    assert len(calls) == 3
    assert len(payload_of_call(calls[0])["rows"]) == 2
    assert len(payload_of_call(calls[1])["rows"]) == 1
    assert len(payload_of_call(calls[2])["rows"]) == 1

    # 分块后的 row_indices 仍来自原始行号，而不是 0/1 局部编号。
    assert payload_of_call(calls[1])["row_indices"] == [
        table9["row_indices"][0]
    ]

    assert result["parameters"][0]["evidence"][
        "logical_row_id"
    ] == "t9.p30"

    usage = extractor.get_usage()
    assert usage["prompt_tokens"] == 30
    assert usage["completion_tokens"] == 15
    assert usage["total_tokens"] == 45


# ---------------------------------------------------------------
# 10. Retry 仍严格校验
# ---------------------------------------------------------------

def test_retry_recanonicalizes_and_revalidates(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )

    failed = make_param(
        "dropout_voltage",
        275.0,
        "mV",
        "max",
        records["t9.p24"]["test_conditions"],
        row_index=24,
        logical_row_id="t9.p24",
    )

    corrected = make_param(
        "dropout_voltage",
        270.0,
        "mV",
        "max",
        records["t9.p24"]["test_conditions"],
        row_index=24,
        # 故意给错误 logical id；retry 后必须重新 canonicalize。
        logical_row_id="t9.p999",
    )

    extractor = make_extractor(
        [make_response(make_extraction([corrected]))]
    )

    retry_result = extractor.retry_parameters(
        [failed], [table9]
    )

    item = json.loads(
        extractor.client.completions.calls[0][
            "messages"
        ][1]["content"]
    )["failed_parameters"][0]

    assert item["logical_row_id"] == "t9.p24"
    assert item["source_row_indices"] == [18, 21, 24, 25]

    parameter = retry_result["parameters"][0]

    assert (
        parameter["evidence"]["logical_row_id"]
        == "t9.p24"
    )
    assert (
        parameter["evidence"]["source_row_indices"]
        == [18, 21, 24, 25]
    )
    assert parameter["value"] == 270.0

    report = raw_report(artifacts, retry_result)
    assert (
        report["parameters"][0]["recovered_status"]
        == "validated"
    )


def test_retry_wrong_value_still_fails_strict_validation(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    table9 = next(
        table for table in logical_input["tables"]
        if table["table_id"] == 9
    )

    failed = make_param(
        "dropout_voltage",
        275.0,
        "mV",
        "max",
        records["t9.p24"]["test_conditions"],
        row_index=24,
        logical_row_id="t9.p24",
    )

    corrected = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "max",
        records["t9.p24"]["test_conditions"],
        row_index=24,
        logical_row_id="t9.p24",
    )

    extractor = make_extractor(
        [make_response(make_extraction([corrected]))]
    )

    retry_result = extractor.retry_parameters(
        [failed], [table9]
    )

    report = raw_report(artifacts, retry_result)
    merged = report["parameters"][0]

    assert merged["recovered_status"] == "invalid"
    assert "value" in merged["conflicts"]


# ---------------------------------------------------------------
# 11. Mock LangGraph 端到端
# ---------------------------------------------------------------

class MockPDFParser:
    def parse(self, path):
        return {
            "filename": Path(path).name,
            "page_count": 37,
        }


class MockTableParser:
    def __init__(self, tables):
        self.tables = tables

    def parse(self, path):
        return {
            "table_count": len(self.tables),
            "tables": [
                copy.deepcopy(table)
                for table in self.tables
            ],
        }


def run_graph(
    artifacts,
    extractor,
    thread_id,
    *,
    max_retries=1,
):
    graph = build_graph(
        MockPDFParser(),
        MockTableParser([artifacts["table9"]]),
        extractor,
        evidence_validation_mode="recovered",
        recovered_evidence=recovered_config(artifacts),
        extraction_input_mode="logical",
        logical_extraction={
            "logical_rows": artifacts["logical_rows"],
            "raw_structure": artifacts["raw_structure"],
        },
    )

    config = {
        "configurable": {"thread_id": thread_id}
    }

    result = graph.invoke(
        {
            "document_path": "TPS7A02.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": max_retries,
            "errors": [],
        },
        config=config,
    )

    return graph, config, result


def test_graph_logical_360_310_completed(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    response = make_response(make_extraction([
        make_param(
            "dropout_voltage",
            360.0,
            "mV",
            "max",
            records["t9.p30"]["test_conditions"],
            row_index=30,
            logical_row_id="t9.p30",
        ),
        make_param(
            "dropout_voltage",
            310.0,
            "mV",
            "max",
            records["t9.p31"]["test_conditions"],
            row_index=30,
            logical_row_id="t9.p31",
        ),
    ]))

    extractor = make_extractor(
        [response], rows_per_chunk=33
    )

    graph, config, result = run_graph(
        artifacts, extractor, "b32a-360-310"
    )

    assert result["status"] == "completed"
    assert result["evidence_validation_mode"] == "recovered"
    assert result["extraction_input_mode"] == "logical"
    assert result["extraction_input_summary"][
        "logical_table_ids"
    ] == [9]
    assert result["extraction_input_summary"][
        "flat_table_ids"
    ] == []

    params = result["validation_result"]["parameters"]

    assert len(params) == 2
    assert {
        param["recovered_logical_row_id"]
        for param in params
    } == {"t9.p30", "t9.p31"}
    assert all(
        param["validation_status"] == "validated"
        for param in params
    )

    snapshot = graph.get_state(config)

    assert (
        snapshot.values["extraction_input_mode"]
        == "logical"
    )


def test_graph_retry_revalidates_strictly(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    initial = make_response(make_extraction([
        make_param(
            "dropout_voltage",
            275.0,
            "mV",
            "max",
            records["t9.p24"]["test_conditions"],
            row_index=24,
            logical_row_id="t9.p24",
        )
    ]))

    # Retry 返回了错误数值；即使来源 ID 正确，
    # 也必须重新执行严格校验，不能自动通过。
    corrected = make_response(make_extraction([
        make_param(
            "dropout_voltage",
            360.0,
            "mV",
            "max",
            records["t9.p24"]["test_conditions"],
            row_index=24,
            logical_row_id="t9.p24",
        )
    ]))

    extractor = make_extractor(
        [initial, corrected],
        rows_per_chunk=33,
    )

    graph, config, result = run_graph(
        artifacts,
        extractor,
        "b32a-retry-strict",
        max_retries=1,
    )

    assert result["retry_count"] == 1
    assert result["status"] == "pending_review"

    parameter = result["validation_result"]["parameters"][0]

    assert parameter["validation_source"] == "recovered"
    assert parameter["validation_status"] == "invalid"
    assert (
        parameter["detailed_recovery_status"] == "invalid"
    )
    assert "value" in parameter["conflicting_fields"]
    assert (
        parameter["recovered_logical_row_id"] == "t9.p24"
    )
    assert parameter["value"] == 360.0

    # Retry 的纠错输入与输出都保留 canonical logical 来源。
    retry_payload = json.loads(
        extractor.client.completions.calls[1][
            "messages"
        ][1]["content"]
    )
    retry_item = retry_payload["failed_parameters"][0]

    assert retry_item["logical_row_id"] == "t9.p24"
    assert retry_item["source_row_indices"] == [
        18, 21, 24, 25
    ]

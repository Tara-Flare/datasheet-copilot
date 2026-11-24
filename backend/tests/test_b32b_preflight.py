"""阶段 B3.2b：Logical/Recovered 安全预检测试。

只使用 Mock LLM，不访问真实 DeepSeek API。
"""

import copy
import json
from pathlib import Path

import pytest
from langgraph.types import Command

from app.agents.graph import build_graph
from app.extraction.artifact_store import (
    build_artifact,
    validate_extraction,
)
from app.extraction.input_mode import (
    build_extraction_input,
)
from app.extraction.logical_table_adapter import (
    LogicalTableAdapterError,
)
from app.validation.recovered_evidence import (
    RecoveredEvidenceValidator,
)
from app.validation.recovered_integration import (
    build_recovered_validation_result,
)
from app.validation.validator import ParameterValidator
from app.extraction.evidence import verify_evidence

from test_logical_extraction_integration import (
    MockPDFParser,
    MockTableParser,
    load_cached_table,
    logical_records,
    make_extraction,
    make_extractor,
    make_param,
    make_response,
    raw_report,
    recovered_config,
)


pytestmark = pytest.mark.requires_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGICAL_PATH = (
    PROJECT_ROOT / "data/outputs/TPS7A02.logical_rows.v1.json"
)
RAW_PATH = (
    PROJECT_ROOT / "data/outputs/TPS7A02.raw_structure.v1.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifacts():
    return {
        "logical_rows": load(LOGICAL_PATH),
        "raw_structure": load(RAW_PATH),
        "table9": load_cached_table(9),
    }


@pytest.fixture(scope="module")
def logical_input(artifacts):
    return build_extraction_input(
        [artifacts["table9"]],
        mode="logical",
        logical_rows=artifacts["logical_rows"],
        raw_structure=artifacts["raw_structure"],
        table_ids=[9],
    )


def table9_of(logical_input):
    return logical_input["tables"][0]


def extract_with_response(
    logical_input,
    parameters,
    *,
    rows_per_chunk=33,
):
    extractor = make_extractor(
        [make_response(make_extraction(parameters))],
        rows_per_chunk=rows_per_chunk,
    )
    result = extractor.extract([table9_of(logical_input)])
    return extractor, result


# ---------------------------------------------------------------
# 1. 错误来源字段不得被 canonical 重写洗白
# ---------------------------------------------------------------

def test_wrong_source_fields_force_pending(
    artifacts, logical_input
):
    records = logical_records(logical_input)
    conditions = records["t9.p30"]["test_conditions"]

    base = dict(
        name="dropout_voltage",
        value=360.0,
        unit="mV",
        value_type="max",
        test_conditions=conditions,
        logical_row_id="t9.p30",
    )

    wrong_table = make_param(
        **base,
        table_id=7,
        row_index=30,
        page=6,
    )
    wrong_row = make_param(
        **base,
        table_id=9,
        row_index=31,
        page=6,
    )
    wrong_page = make_param(
        **base,
        table_id=9,
        row_index=30,
        page=5,
    )
    wrong_indices = make_param(
        **base,
        table_id=9,
        row_index=30,
        page=6,
        source_row_indices=[1],
    )
    wrong_cells = make_param(
        **base,
        table_id=9,
        row_index=30,
        page=6,
    )
    wrong_cells["evidence"]["source_cell_ids"] = [1]

    _, extraction = extract_with_response(
        logical_input,
        [
            wrong_table,
            wrong_row,
            wrong_page,
            wrong_indices,
            wrong_cells,
        ],
    )

    for param in extraction["parameters"]:
        assert (
            param["source_mapping_status"]
            == "mapped_source_conflict"
        )
        assert param["source_mapping"]["trusted"] is False
        assert param["evidence"]["table_id"] == 9
        assert param["evidence"]["row_index"] == 30
        assert param["evidence"]["page"] == 6
        assert param["evidence"]["logical_row_id"] == "t9.p30"

    report = raw_report(artifacts, extraction)

    for item in report["parameters"]:
        assert item["recovered_status"] == "pending_review"
        assert (
            "来源映射不可信"
            in item["reason"]
        )


def test_content_conflicts_are_independently_validated(
    artifacts, logical_input
):
    records = logical_records(logical_input)
    p30_conditions = records["t9.p30"][
        "test_conditions"
    ]
    p31_conditions = records["t9.p31"][
        "test_conditions"
    ]

    wrong_value = make_param(
        "dropout_voltage",
        361.0,
        "mV",
        "max",
        p30_conditions,
        row_index=30,
        logical_row_id="t9.p30",
    )
    wrong_unit = make_param(
        "dropout_voltage",
        360.0,
        "V",
        "max",
        p30_conditions,
        row_index=30,
        logical_row_id="t9.p30",
    )
    wrong_value_type = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "typ",
        p30_conditions,
        row_index=30,
        logical_row_id="t9.p30",
    )
    wrong_conditions = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "max",
        p31_conditions,
        row_index=30,
        logical_row_id="t9.p30",
    )

    _, extraction = extract_with_response(
        logical_input,
        [
            wrong_value,
            wrong_unit,
            wrong_value_type,
            wrong_conditions,
        ],
    )

    report = raw_report(artifacts, extraction)
    params = report["parameters"]

    assert params[0]["recovered_status"] == "invalid"
    assert "value" in params[0]["conflicts"]

    assert params[1]["recovered_status"] == "invalid"
    assert "unit" in params[1]["conflicts"]

    assert params[2]["recovered_status"] == "invalid"
    assert "value_type" in params[2]["conflicts"]

    assert params[3]["recovered_status"] == "invalid"
    assert "test_conditions" in params[3]["conflicts"]


def test_unique_row_index_mapping_but_ambiguous_pending(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    unique = make_param(
        "dropout_voltage",
        850.0,
        "mV",
        "max",
        records["t9.p26"]["test_conditions"],
        row_index=26,
        logical_row_id=None,
    )
    ambiguous = make_param(
        "quiescent_current",
        60.0,
        "nA",
        "max",
        records["t9.p9"]["test_conditions"],
        row_index=9,
        logical_row_id=None,
    )

    _, extraction = extract_with_response(
        logical_input, [unique, ambiguous]
    )

    first, second = extraction["parameters"]

    assert first["source_mapping_status"] == "mapped"
    assert first["evidence"]["logical_row_id"] == "t9.p26"

    assert (
        second["source_mapping_status"]
        == "ambiguous_row_index"
    )
    assert second["source_mapping"]["trusted"] is False

    report = raw_report(artifacts, extraction)

    assert (
        report["parameters"][0]["recovered_status"]
        == "validated"
    )
    assert (
        report["parameters"][1]["recovered_status"]
        == "pending_review"
    )


# ---------------------------------------------------------------
# 2. 模式组合
# ---------------------------------------------------------------

def test_logical_legacy_guard_blocks_false_pass(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    # t9.p15 的 Flat 行完整，legacy 只查值/单位/原文，
    # 不查 test_conditions，因此空条件会被错误放行。
    parameter = make_param(
        "output_current_limit",
        750.0,
        "mA",
        "max",
        "",
        row_index=15,
        logical_row_id="t9.p15",
    )

    # 先证明单纯 legacy 会错误放行。
    _, direct_extraction = extract_with_response(
        logical_input, [parameter]
    )
    legacy = ParameterValidator().validate(
        verify_evidence(
            direct_extraction, [artifacts["table9"]]
        )
    )
    assert (
        legacy["parameters"][0]["validation_status"]
        == "validated"
    )

    # 在正式 Graph 中，logical + legacy 必须被安全阻断。
    extractor = make_extractor(
        [make_response(make_extraction([parameter]))],
        rows_per_chunk=33,
    )
    graph = build_graph(
        MockPDFParser(),
        MockTableParser([artifacts["table9"]]),
        extractor,
        evidence_validation_mode="legacy",
        extraction_input_mode="logical",
        logical_extraction={
            "logical_rows": artifacts["logical_rows"],
            "raw_structure": artifacts["raw_structure"],
            "table_ids": [9],
        },
    )

    result = graph.invoke(
        {
            "document_path": "TPS7A02.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 0,
            "enable_human_review": False,
            "errors": [],
        },
        config={
            "configurable": {
                "thread_id": "b32b-legacy-guard"
            }
        },
    )

    assert result["status"] == "pending_review"

    merged = result["validation_result"]["parameters"][0]

    assert merged["validation_status"] == "pending_review"
    assert merged["validation_source"] == "legacy_blocked"
    assert merged["logical_validation_required"] is True
    assert any(
        "recovered 校验" in error
        for error in merged["validation_errors"]
    )

    summary = result["validation_result"][
        "validation_summary"
    ]
    assert summary["validated"] == 0
    assert summary["logical_legacy_blocked"] == 1


def test_logical_recovered_remains_strict(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    parameter = make_param(
        "output_current_limit",
        750.0,
        "mA",
        "max",
        "",
        row_index=15,
        logical_row_id="t9.p15",
    )

    _, extraction = extract_with_response(
        logical_input, [parameter]
    )

    # legacy 会 validated，但 recovered 必须看到条件缺失。
    legacy = ParameterValidator().validate(
        verify_evidence(
            extraction, [artifacts["table9"]]
        )
    )
    assert (
        legacy["parameters"][0]["validation_status"]
        == "validated"
    )

    recovered = build_recovered_validation_result(
        extraction,
        legacy,
        recovered_config(artifacts),
    )
    merged = recovered["parameters"][0]

    assert merged["validation_source"] == "recovered"
    assert merged["validation_status"] == "pending_review"
    assert (
        merged["detailed_recovery_status"]
        == "correctable"
    )
    assert "test_conditions" in merged["missing_fields"]


# ---------------------------------------------------------------
# 3. 回退策略
# ---------------------------------------------------------------

def test_malformed_logical_source_does_not_fallback_flat(
    artifacts, logical_input
):
    broken = copy.deepcopy(
        artifacts["logical_rows"]
    )
    broken["schema"] = "unsupported/logical"

    with pytest.raises(LogicalTableAdapterError):
        build_extraction_input(
            [artifacts["table9"]],
            mode="logical",
            logical_rows=broken,
            raw_structure=artifacts["raw_structure"],
            table_ids=[9],
        )


def test_recovered_source_mismatch_not_silent_fallback(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    parameter = make_param(
        "dropout_voltage",
        850.0,
        "mV",
        "max",
        records["t9.p26"]["test_conditions"],
        row_index=26,
        logical_row_id="t9.p26",
    )

    extractor = make_extractor(
        [make_response(make_extraction([parameter]))],
        rows_per_chunk=33,
    )

    bad_config = recovered_config(artifacts)
    bad_config["pdf_sha256"] = "deadbeef"

    graph = build_graph(
        MockPDFParser(),
        MockTableParser([artifacts["table9"]]),
        extractor,
        evidence_validation_mode="recovered",
        recovered_evidence=bad_config,
        extraction_input_mode="logical",
        logical_extraction={
            "logical_rows": artifacts["logical_rows"],
            "raw_structure": artifacts["raw_structure"],
            "table_ids": [9],
        },
    )

    result = graph.invoke(
        {
            "document_path": "TPS7A02.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 0,
            "errors": [],
        },
        config={
            "configurable": {
                "thread_id": "b32b-sha-mismatch"
            }
        },
    )

    merged = result["validation_result"]["parameters"][0]

    assert result["status"] == "pending_review"
    assert merged["validation_status"] == "pending_review"
    assert (
        merged["detailed_recovery_status"] == "unavailable"
    )
    assert result["validation_result"][
        "validation_summary"
    ]["recovered_evidence_error"]


# ---------------------------------------------------------------
# 4. 结果持久化
# ---------------------------------------------------------------

def test_artifact_preserves_logical_metadata(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    parameter = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "max",
        records["t9.p30"]["test_conditions"],
        row_index=30,
        logical_row_id="t9.p30",
    )

    _, extraction = extract_with_response(
        logical_input, [parameter]
    )

    artifact = build_artifact(
        extraction,
        final_extraction=extraction,
    )

    for source_name in (
        "extraction",
        "final_extraction",
    ):
        param = artifact[source_name]["parameters"][0]

        assert (
            param["source_mapping_status"] == "mapped"
        )
        assert "source_mapping" in param
        assert (
            param["evidence"]["logical_row_id"]
            == "t9.p30"
        )
        assert (
            param["evidence"]["source_row_indices"]
            == [18, 25, 30]
        )
        assert param["evidence"]["source_cell_ids"]

    validate_extraction(artifact["extraction"])
    validate_extraction(artifact["final_extraction"])


def test_retry_preserves_logical_metadata(
    artifacts, logical_input
):
    records = logical_records(logical_input)

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
        logical_row_id="t9.p24",
    )

    extractor = make_extractor([
        make_response(make_extraction([corrected]))
    ])

    result = extractor.retry_parameters(
        [failed], [table9_of(logical_input)]
    )
    param = result["parameters"][0]

    assert param["source_mapping_status"] == "mapped"
    assert param["source_mapping"]["trusted"] is True
    assert param["evidence"]["logical_row_id"] == "t9.p24"
    assert param["evidence"]["source_row_indices"] == [
        18, 21, 24, 25
    ]
    assert param["evidence"]["source_cell_ids"]


def test_review_preserves_logical_metadata(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    parameter = make_param(
        "dropout_voltage",
        900.0,
        "mV",
        "max",
        records["t9.p26"]["test_conditions"],
        row_index=26,
        logical_row_id="t9.p26",
    )

    extractor = make_extractor(
        [make_response(make_extraction([parameter]))],
        rows_per_chunk=33,
    )

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
            "table_ids": [9],
        },
    )

    config = {
        "configurable": {
            "thread_id": "b32b-review-meta"
        }
    }

    first = graph.invoke(
        {
            "document_path": "TPS7A02.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 0,
            "enable_human_review": True,
            "errors": [],
        },
        config=config,
    )

    assert "__interrupt__" in first

    resumed = graph.invoke(
        Command(
            resume={
                "action": "correct",
                "corrections": [
                    {"index": 0, "value": 850.0}
                ],
            }
        ),
        config=config,
    )

    assert resumed["status"] == "completed"

    param = resumed["extraction"]["parameters"][0]

    assert param["source_mapping_status"] == "mapped"
    assert "source_mapping" in param
    assert param["evidence"]["logical_row_id"] == "t9.p26"
    assert param["evidence"]["source_row_indices"] == [
        18, 25, 26
    ]
    assert param["evidence"]["source_cell_ids"]

    # Review 重建参数时，错误/冲突元数据也不能丢失。
    assert "validation_errors" in param
    assert "conflicting_fields" in param
    assert "detailed_recovery_status" in param

    validated = resumed["validation_result"][
        "parameters"
    ][0]
    assert validated["validation_status"] == "validated"
    assert (
        validated["recovered_logical_row_id"] == "t9.p26"
    )


def test_raw_llm_output_is_recorded(
    artifacts, logical_input
):
    records = logical_records(logical_input)

    parameter = make_param(
        "dropout_voltage",
        360.0,
        "mV",
        "max",
        records["t9.p30"]["test_conditions"],
        row_index=30,
        logical_row_id="t9.p30",
    )

    extractor, _ = extract_with_response(
        logical_input, [parameter]
    )

    assert len(extractor.raw_responses) == 1

    raw = extractor.raw_responses[0]

    assert raw["finish_reason"] == "stop"
    assert raw["validated"] is True
    assert raw["prompt_tokens"] == 10
    assert raw["completion_tokens"] == 5
    assert json.loads(raw["content"])["parameters"][0][
        "evidence"
    ]["logical_row_id"] == "t9.p30"

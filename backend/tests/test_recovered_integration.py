"""阶段 B2.2：RecoveredEvidenceValidator 与 Agent 的安全集成测试。"""

import copy
import json
from pathlib import Path

import pytest

from app.agents.graph import build_graph
from app.extraction.evidence import verify_evidence
from app.validation.recovered_integration import (
    RecoveredEvidenceError,
    build_recovered_validation_result,
    check_recovered_source_consistency,
    resolve_evidence_validation_mode,
)
from app.validation.validator import ParameterValidator

from test_agent_graph import (
    MockPDFParser as LegacyMockPDFParser,
    MockTableParser as LegacyMockTableParser,
    MockExtractor as LegacyMockExtractor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_PATH = (
    PROJECT_ROOT / "data/outputs/TPS7A02.extraction.json"
)
LOGICAL_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.logical_rows.v1.json"
)
RAW_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.raw_structure.v1.json"
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
        "extraction": load(EXTRACTION_PATH),
        "table9": load_cached_table(9),
        "table7": load_cached_table(7),
    }


def make_param(**overrides):
    param = {
        "name": "dropout_voltage",
        "value": 270.0,
        "unit": "mV",
        "value_type": "max",
        "test_conditions": "3.3 V ≤ VOUT ≤ 5.0 V",
        "evidence": {
            "table_id": 9,
            "row_index": 24,
            "page": 6,
            "source_text": "",
        },
    }
    param.update(overrides)
    return param


def make_extraction(parameters):
    return {
        "component_model": "",
        "manufacturer": "",
        "parameters": parameters,
    }


def source_text_from_row(table, row_index):
    row = table["rows"][row_index]
    return " ".join(
        str(cell) for cell in row
        if cell is not None
    )


def legacy_validate(extraction, table):
    return ParameterValidator().validate(
        verify_evidence(extraction, [table])
    )


def recovered_config(artifacts, **overrides):
    config = {
        "logical_rows": artifacts["logical_rows"],
        "raw_structure": artifacts["raw_structure"],
        "pdf_sha256": artifacts["raw_structure"]["source"][
            "pdf_sha256"
        ],
        "on_error": "pending_review",
    }
    config.update(overrides)
    return config


def validate_one(artifacts, param, table=None):
    extraction = make_extraction([param])
    table = table or artifacts["table9"]
    legacy = legacy_validate(extraction, table)

    return build_recovered_validation_result(
        extraction,
        legacy,
        recovered_config(artifacts),
    )


def test_legacy_default_mode(monkeypatch):
    """1. legacy 默认行为不变。"""

    monkeypatch.delenv(
        "EVIDENCE_VALIDATION_MODE", raising=False
    )

    assert resolve_evidence_validation_mode() == "legacy"

    graph = build_graph(
        LegacyMockPDFParser(),
        LegacyMockTableParser(),
        LegacyMockExtractor(value=200),
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
                "thread_id": "b22-legacy-default"
            }
        },
    )

    assert result["status"] == "completed"
    assert result["evidence_validation_mode"] == "legacy"
    assert (
        result["validation_result"][
            "validation_summary"
        ]["validated"]
        == 1
    )

    # legacy 模式不添加 recovered 字段
    parameter = result["validation_result"]["parameters"][0]
    assert "validation_source" not in parameter
    assert "detailed_recovery_status" not in parameter


def test_recovered_mode_selection(monkeypatch):
    """2. recovered 模式正确选择（仅显式启用）。"""

    assert (
        resolve_evidence_validation_mode("recovered")
        == "recovered"
    )
    assert (
        resolve_evidence_validation_mode("legacy")
        == "legacy"
    )

    with pytest.raises(ValueError):
        resolve_evidence_validation_mode("auto")

    monkeypatch.setenv(
        "EVIDENCE_VALIDATION_MODE", "recovered"
    )
    assert resolve_evidence_validation_mode() == "recovered"


def test_270mv_missing_temperature_correctable(artifacts):
    """3. 270 mV 缺温度 → correctable。"""

    param = make_param(
        value=270.0,
        unit="mV",
        test_conditions="3.3 V ≤ VOUT ≤ 5.0 V",
        evidence={
            "table_id": 9,
            "row_index": 24,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 24
            ),
        },
    )

    result = validate_one(artifacts, param)
    merged = result["parameters"][0]

    assert merged["validation_status"] == "pending_review"
    assert merged["validation_source"] == "recovered"
    assert (
        merged["detailed_recovery_status"] == "correctable"
    )
    assert "test_conditions" in merged["missing_fields"]
    assert merged["suggested_correction"] is not None
    assert any(
        "tj" in item
        for item in merged["missing_conditions"]
    )

    assert result["validation_summary"]["correctable"] == 1
    assert (
        result["validation_summary"]["recovered_validated"]
        == 0
    )


def test_850mv_complete_conditions_validated(artifacts):
    """4. 850 mV 完整条件 → validated。"""

    param = make_param(
        value=850.0,
        unit="mV",
        test_conditions=(
            "TJ = -40°C to +125°C; 1.0 V ≤ VOUT < 1.2 V"
        ),
        evidence={
            "table_id": 9,
            "row_index": 26,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 26
            ),
        },
    )

    result = validate_one(artifacts, param)
    merged = result["parameters"][0]

    assert merged["validation_status"] == "validated"
    assert merged["validation_source"] == "recovered"
    assert (
        merged["detailed_recovery_status"] == "validated"
    )
    assert (
        result["validation_summary"]["recovered_validated"]
        == 1
    )


def test_ground_current_1na_invalid(artifacts):
    """5. Ground Current 1 nA → invalid。"""

    param = make_param(
        name="quiescent_current",
        value=1.0,
        unit="nA",
        value_type="typ",
        test_conditions=(
            "5 µA ≤ IOUT < 1 mA, TJ = -40°C to +85°C"
        ),
        evidence={
            "table_id": 9,
            "row_index": 9,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 9
            ),
        },
    )

    result = validate_one(artifacts, param)
    merged = result["parameters"][0]

    assert merged["validation_status"] == "invalid"
    assert merged["validation_source"] == "recovered"
    assert "identity" in merged["conflicting_fields"]
    assert "unit" in merged["conflicting_fields"]
    assert merged["suggested_correction"] is not None


def test_ground_current_60na_condition_conflict(artifacts):
    """6. Ground Current 60 nA 条件冲突 → invalid。"""

    param = make_param(
        name="quiescent_current",
        value=60.0,
        unit="nA",
        value_type="max",
        test_conditions=(
            "5 µA ≤ IOUT < 1 mA, TJ = -40°C to +85°C"
        ),
        evidence={
            "table_id": 9,
            "row_index": 9,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 9
            ),
        },
    )

    result = validate_one(artifacts, param)
    merged = result["parameters"][0]

    assert merged["validation_status"] == "invalid"
    assert "test_conditions" in merged["conflicting_fields"]
    assert merged["conflicting_conditions"]


def test_vuvlo_needs_review_not_passed(artifacts):
    """7. VUVLO needs_review 不自动通过。"""

    vuvlo = next(
        row for row in artifacts["logical_rows"][
            "logical_rows"
        ]
        if row["logical_row_id"] == "t9.p34"
    )

    param = make_param(
        name="input_voltage",
        value=1.3,
        unit="V",
        value_type="typ",
        test_conditions="VIN rising",
        evidence={
            "table_id": 9,
            "row_index": vuvlo["source_row_indices"][0],
            "page": 6,
            "source_text": "VIN rising",
        },
    )

    result = validate_one(artifacts, param)
    merged = result["parameters"][0]

    assert merged["validation_status"] == "pending_review"
    assert (
        merged["detailed_recovery_status"] == "pending_review"
    )


def test_uncovered_table_marked_legacy(artifacts):
    """8. 未覆盖表格明确标记 legacy。"""

    param = {
        "name": "input_voltage",
        "value": 1.5,
        "unit": "V",
        "value_type": "min",
        "test_conditions": "VIN input voltage",
        "evidence": {
            "table_id": 7,
            "row_index": 0,
            "page": 5,
            "source_text": source_text_from_row(
                artifacts["table7"], 0
            ),
        },
    }

    result = validate_one(
        artifacts, param, table=artifacts["table7"]
    )
    merged = result["parameters"][0]

    assert merged["validation_source"] == "legacy"
    assert (
        merged["detailed_recovery_status"] == "not_covered"
    )
    assert merged["validation_status"] == "validated"
    assert result["validation_summary"]["legacy_validated"] == 1
    assert (
        result["validation_summary"]["recovered_validated"]
        == 0
    )


def test_covered_conflict_no_legacy_fallback(artifacts):
    """9. 已覆盖但冲突不得 legacy 回退。"""

    param = make_param(
        value=1100.0,
        unit="mV",
        test_conditions="0.8 V ≤ VOUT < 1.0 V",
        evidence={
            "table_id": 9,
            "row_index": 25,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 25
            ),
        },
    )

    extraction = make_extraction([param])
    legacy = legacy_validate(
        extraction, artifacts["table9"]
    )

    # legacy 认为该参数通过
    assert (
        legacy["parameters"][0]["validation_status"]
        == "validated"
    )

    result = build_recovered_validation_result(
        extraction,
        legacy,
        recovered_config(artifacts),
    )
    merged = result["parameters"][0]

    # recovered 因缺少温度条件不得回退 legacy validated
    assert merged["validation_source"] == "recovered"
    assert merged["validation_status"] == "pending_review"
    assert (
        merged["detailed_recovery_status"] == "correctable"
    )
    assert result["validation_summary"]["legacy_validated"] == 0
    assert (
        result["validation_summary"]["recovered_validated"]
        == 0
    )


def test_recovered_source_mismatch_not_used(artifacts):
    """10. 恢复文件与 PDF 不匹配不得使用。"""

    param = make_param(
        evidence={
            "table_id": 9,
            "row_index": 24,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 24
            ),
        }
    )
    extraction = make_extraction([param])
    legacy = legacy_validate(
        extraction, artifacts["table9"]
    )

    result = build_recovered_validation_result(
        extraction,
        legacy,
        recovered_config(
            artifacts, pdf_sha256="deadbeef"
        ),
    )
    merged = result["parameters"][0]

    assert check_recovered_source_consistency(
        recovered_config(artifacts)
    )["ok"] is True

    assert merged["validation_status"] == "pending_review"
    assert (
        merged["detailed_recovery_status"] == "unavailable"
    )
    assert result["validation_summary"][
        "recovered_evidence_error"
    ]
    assert result["validation_summary"]["validated"] == 0

    consistency = check_recovered_source_consistency(
        recovered_config(
            artifacts, pdf_sha256="deadbeef"
        )
    )
    assert consistency["ok"] is False
    assert any(
        "SHA256" in error
        for error in consistency["errors"]
    )

    with pytest.raises(RecoveredEvidenceError):
        build_recovered_validation_result(
            extraction,
            legacy,
            recovered_config(
                artifacts,
                pdf_sha256="deadbeef",
                on_error="fail",
            ),
        )


class MockPDFParser:
    def __init__(self, page_count=37):
        self.page_count = page_count

    def parse(self, path):
        return {
            "filename": Path(path).name,
            "page_count": self.page_count,
        }


class MockTableParser:
    def __init__(self, table):
        self.table = table

    def parse(self, path):
        return {
            "table_count": 1,
            "tables": [copy.deepcopy(self.table)],
        }


class RetryExtractor:
    def __init__(self, initial, corrected):
        self.initial = initial
        self.corrected = corrected
        self.retry_calls = 0

    def extract(self, tables):
        return copy.deepcopy(self.initial)

    def retry_parameters(self, failed_parameters, tables):
        self.retry_calls += 1
        return copy.deepcopy(self.corrected)


def run_recovered_graph(artifacts, extractor, thread_id):
    config = {
        "configurable": {"thread_id": thread_id}
    }

    graph = build_graph(
        MockPDFParser(),
        MockTableParser(artifacts["table9"]),
        extractor,
        evidence_validation_mode="recovered",
        recovered_evidence=recovered_config(artifacts),
    )

    result = graph.invoke(
        {
            "document_path": "TPS7A02.pdf",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 1,
            "errors": [],
        },
        config=config,
    )

    return graph, config, result


def test_retry_reruns_strict_recovered_validation(artifacts):
    """11. Retry 后仍执行严格校验。"""

    complete_conditions = (
        "TJ = -40°C to +85°C; 3.3 V ≤ VOUT ≤ 5.0 V"
    )

    initial = make_extraction([make_param(
        value=275.0,
        unit="mV",
        test_conditions=complete_conditions,
        evidence={
            "table_id": 9,
            "row_index": 24,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 24
            ),
        },
    )])
    corrected = make_extraction([make_param(
        value=270.0,
        unit="mV",
        test_conditions=complete_conditions,
        evidence={
            "table_id": 9,
            "row_index": 24,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 24
            ),
        },
    )])

    extractor = RetryExtractor(initial, corrected)

    graph, config, result = run_recovered_graph(
        artifacts, extractor, "b22-retry"
    )

    assert extractor.retry_calls == 1
    assert result["retry_count"] == 1
    assert result["status"] == "completed"
    assert result["evidence_validation_mode"] == "recovered"

    parameter = result["validation_result"]["parameters"][0]

    assert parameter["validation_status"] == "validated"
    assert parameter["validation_source"] == "recovered"
    assert (
        parameter["detailed_recovery_status"] == "validated"
    )


def test_checkpoint_and_state_compatible(artifacts):
    """12. Checkpoint 与 State 兼容。"""

    complete_conditions = (
        "TJ = -40°C to +125°C; 1.0 V ≤ VOUT < 1.2 V"
    )

    extraction = make_extraction([make_param(
        value=850.0,
        unit="mV",
        test_conditions=complete_conditions,
        evidence={
            "table_id": 9,
            "row_index": 26,
            "page": 6,
            "source_text": source_text_from_row(
                artifacts["table9"], 26
            ),
        },
    )])

    extractor = RetryExtractor(extraction, extraction)

    graph, config, result = run_recovered_graph(
        artifacts, extractor, "b22-checkpoint"
    )

    snapshot = graph.get_state(config)

    assert (
        snapshot.values["evidence_validation_mode"]
        == "recovered"
    )
    assert (
        snapshot.values["validation_result"][
            "validation_summary"
        ]["recovered_validated"]
        == 1
    )

    parameter = snapshot.values["validation_result"][
        "parameters"
    ][0]
    assert parameter["validation_source"] == "recovered"
    assert parameter["validation_status"] == "validated"

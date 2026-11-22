"""阶段 B2.1：离线证据匹配测试。"""

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from app.validation import recovered_evidence
from app.validation.recovered_evidence import (
    RecoveredEvidenceValidator,
    validate_from_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.extraction.json"
)
LOGICAL_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.logical_rows.v1.json"
)
BASELINE_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.final.validation.json"
)
RAW_PATH = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.raw_structure.v1.json"
)

EXTRACTION_SHA_AT_IMPORT = hashlib.sha256(
    EXTRACTION_PATH.read_bytes()
).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inputs():
    return {
        "extraction": load(EXTRACTION_PATH),
        "logical_rows": load(LOGICAL_PATH),
        "baseline": load(BASELINE_PATH),
        "raw_structure": load(RAW_PATH),
    }


@pytest.fixture(scope="module")
def report(inputs):
    return validate_from_files(
        inputs["extraction"],
        inputs["logical_rows"],
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )


def by_index(report):
    return {
        item["index"]: item
        for item in report["parameters"]
    }


def test_correct_param_unique_match(report):
    """1. 正确参数唯一匹配。"""

    params = by_index(report)

    row = params[8]
    assert row["name"] == "quiescent_current"
    assert row["recovered_status"] == "validated"
    assert row["selected_logical_row_id"] == "t9.p8"
    assert row["mapping"]["method"] == "unique"
    assert row["conflicts"] == []

    row = params[25]
    assert row["value"] == 850.0
    assert row["recovered_status"] == "validated"
    assert row["selected_logical_row_id"] == "t9.p26"

    # 1100 mV 缺温度条件，按条件完整性规则应为 correctable
    row = params[24]
    assert row["value"] == 1100.0
    assert row["recovered_status"] == "correctable"
    assert row["selected_logical_row_id"] == "t9.p25"
    assert "test_conditions" in row["missing_fields"]


def test_same_value_different_params(report):
    """2. 不同参数具有相同数值，靠条件区分。"""

    params = by_index(report)

    # 240 mA 同时出现在 t9.p15 / t9.p16
    first = params[13]
    second = params[16]

    assert first["value"] == second["value"] == 240.0
    assert first["selected_logical_row_id"] == "t9.p15"
    assert second["selected_logical_row_id"] == "t9.p16"

    assert first["recovered_status"] == "validated"
    assert second["recovered_status"] == "validated"


def test_value_correct_unit_wrong(inputs):
    """3. 数值正确但单位错误。"""

    extraction = copy.deepcopy(inputs["extraction"])
    extraction["final_extraction"]["parameters"][8]["unit"] = "µA"

    result = validate_from_files(
        extraction,
        inputs["logical_rows"],
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )

    row = by_index(result)[8]

    assert row["recovered_status"] == "invalid"
    assert "unit" in row["conflicts"]
    assert row["field_checks"]["value"]["status"] == "match"


def test_value_correct_conditions_wrong(report):
    """4. 数值正确但测试条件冲突（明确冲突 → invalid）。"""

    row = by_index(report)[11]

    assert row["value"] == 60.0
    assert row["unit"] == "nA"
    assert row["recovered_status"] == "invalid"
    assert row["recovered_status"] != "validated"
    assert "test_conditions" in row["conflicts"]

    conditions = row["field_checks"]["test_conditions"]
    assert conditions["status"] == "conflict"
    assert conditions["conflicting_conditions"]
    assert row["correction_suggestions"]


def test_one_row_split_into_two_logical_rows(report):
    """5. 一条原始行拆成两条逻辑行。"""

    params = by_index(report)

    # 原始 row_index=9 同时映射到 t9.p9 与 t9.p10
    first = params[10]   # 1 nA -> 实际属于 1 %
    second = params[11]  # 60 nA -> 实际属于 Ground current

    assert first["evidence"]["row_index"] == 9
    assert second["evidence"]["row_index"] == 9

    assert first["selected_logical_row_id"] == "t9.p10"
    assert second["selected_logical_row_id"] == "t9.p9"
    assert (
        first["selected_logical_row_id"]
        != second["selected_logical_row_id"]
    )


def test_missing_unit_candidate_completion(report):
    """6. 缺失单位候选补全。"""

    row = by_index(report)[19]

    assert row["value"] == 650.0
    assert row["unit"] == ""
    assert row["recovered_status"] == "correctable"
    assert row["field_checks"]["unit"]["status"] == "missing"

    suggestions = row["correction_suggestions"]
    assert suggestions
    assert suggestions[0]["unit"] == "mV"
    assert suggestions[0]["source_cell_ids"]


def test_needs_review_forbidden(inputs):
    """7. needs_review 禁止自动通过。"""

    logical_rows = copy.deepcopy(inputs["logical_rows"])

    for row in logical_rows["logical_rows"]:
        if row["logical_row_id"] == "t9.p8":
            row["needs_review"] = True

    result = validate_from_files(
        inputs["extraction"],
        logical_rows,
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )

    row = by_index(result)[8]

    assert row["recovered_status"] == "pending_review"
    assert row["field_checks"]["needs_review"][
        "logical_row_needs_review"
    ] is True


def test_ambiguous_mapping_pending_review(inputs):
    """8. logical_row 来源映射不唯一。"""

    logical_rows = copy.deepcopy(inputs["logical_rows"])

    source = next(
        row for row in logical_rows["logical_rows"]
        if row["logical_row_id"] == "t9.p8"
    )
    duplicate = copy.deepcopy(source)
    duplicate["logical_row_id"] = "t9.p8_dup"

    logical_rows["logical_rows"].append(duplicate)

    result = validate_from_files(
        inputs["extraction"],
        logical_rows,
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )

    row = by_index(result)[8]

    assert row["mapping"]["method"] == "ambiguous"
    assert row["recovered_status"] == "pending_review"


def test_original_extraction_not_modified(inputs, report):
    """9. 原始提取结果不被修改。"""

    extraction = inputs["extraction"]

    # 运行时未被就地修改：再次校验应得到相同结果
    again = validate_from_files(
        extraction,
        inputs["logical_rows"],
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )

    assert again == report

    # 文件未被修改
    assert (
        hashlib.sha256(
            EXTRACTION_PATH.read_bytes()
        ).hexdigest()
        == EXTRACTION_SHA_AT_IMPORT
    )


def test_regression_against_old_validator(inputs, report):
    """10. 对旧 Validator 的结果回归比较。"""

    baseline = inputs["baseline"]["summary"]

    assert (
        report["summary"]["baseline_validated"]
        == baseline["validated"]
    )
    assert report["summary"]["regressed"] == 3
    assert report["summary"]["newly_validated"] == 4

    # 条件完整性收紧后，4/5/24 因缺少温度条件由 validated 降为 correctable
    expected_downgrade = {4, 5, 24}

    for item in report["parameters"]:
        if (
            item["baseline_status"] == "validated"
            and item["recovered_status"] != "not_covered"
        ):
            if item["index"] in expected_downgrade:
                assert item["recovered_status"] == "correctable"
            else:
                assert item["recovered_status"] == "validated"


def test_deterministic(inputs):
    """11. 同一输入重复运行结果一致。"""

    first = validate_from_files(
        inputs["extraction"],
        inputs["logical_rows"],
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )
    second = validate_from_files(
        inputs["extraction"],
        inputs["logical_rows"],
        baseline=inputs["baseline"],
        raw_structure=inputs["raw_structure"],
    )

    assert first == second


def test_fully_offline_module():
    """12. 全部操作离线执行。"""

    source = inspect.getsource(recovered_evidence)

    forbidden = (
        "openai",
        "deepseek_extractor",
        "requests",
        "pymupdf",
        "fitz",
        "langgraph",
    )

    for token in forbidden:
        assert token not in source

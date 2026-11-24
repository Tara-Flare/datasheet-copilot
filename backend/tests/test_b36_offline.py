"""B3.6 离线回归：结构修复、Prompt 开关、Validator 回归。

只读 B3.5b Gold / B3.5c artifact / B3.6 产物，
不调用任何 LLM。
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from app.extraction import prompts  # noqa: E402
from app.extraction.deepseek_extractor import (  # noqa: E402
    DeepSeekExtractor,
    EXTRACT_INSTRUCTION,
)
from app.extraction.logical_table_adapter import (  # noqa: E402
    adapt_logical_rows,
)
from app.parsers.structure_recovery import (  # noqa: E402
    recover_from_pdf,
)
from app.validation.rules import (  # noqa: E402
    check_min_typ_max,
)
from app.validation.validator import (  # noqa: E402
    ParameterValidator,
)


EXPERIMENT = (
    PROJECT_ROOT / "data/outputs/experiments/B3.6"
)
STRUCTURE = EXPERIMENT / "structure/structure_compare.json"
VALIDATOR = EXPERIMENT / "validator/regression.json"
GOLD = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5b/gold/gold.frozen.json"
)
GOLD_RECEIPT = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5b/gold"
    / "gold.frozen.receipt.json"
)
RAW_DIR = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5d/structure/raw"
)
PDF_DIR = PROJECT_ROOT / "data/samples/b35b"

CASES = {
    "adi_lt3045": (4, 4),
    "diodes_ap2112": (4, 4),
    "onsemi_ncp170": (4, 3),
    "renesas_isl80510": (7, 4),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recovered():
    out = {}

    for sample_id, (table_id, page) in CASES.items():
        raw = load(
            RAW_DIR
            / f"{sample_id}.raw_structure.v1.json"
        )
        out[sample_id] = recover_from_pdf(
            raw,
            table_id,
            str(PDF_DIR / f"{sample_id}.pdf"),
            page,
        )

    return out


@pytest.fixture(scope="module")
def structure():
    return load(STRUCTURE)


@pytest.fixture(scope="module")
def validator_regression():
    return load(VALIDATOR)


def rows_by_id(recovered_sample):
    return {
        row["logical_row_id"]: row
        for row in recovered_sample["logical_rows"]
    }


@pytest.mark.requires_data
def test_gold_is_unchanged():
    receipt = load(GOLD_RECEIPT)
    actual = hashlib.sha256(
        GOLD.read_bytes()
    ).hexdigest()

    assert actual == receipt["gold_sha256"]
    assert receipt["record_count"] == 76


@pytest.mark.requires_data
def test_recovered_and_adapter_coverage(structure):
    totals = structure["totals"]

    assert totals["gold_records"] == 76
    assert totals["recovered_rows_complete"] == 76
    assert totals["recovered_coverage"] == 1.0
    assert totals["adapter_identity_matched"] == 76
    assert totals["adapter_value_present"] == 76
    assert totals["adapter_unit_present"] == 76
    assert totals["adapter_row_index_correct"] == 76
    assert totals["adapter_complete"] == 70
    assert totals["adapter_coverage"] == 0.9211


@pytest.mark.requires_data
def test_adi_table4_values_fully_recovered(recovered):
    sample = recovered["adi_lt3045"]
    rows = rows_by_id(sample)

    assert rows["t4.p2.s0"]["typ"] == "0.3"
    assert rows["t4.p2.s1"]["max"] == "10"
    assert rows["t4.p2.s2"]["max"] == "20"
    assert rows["t4.p3.s0"]["min"] == "570"
    assert rows["t4.p3.s0"]["typ"] == "710"
    assert rows["t4.p3.s0"]["max"] == "850"
    assert rows["t4.p3.s2"]["min"] == "230"
    assert rows["t4.p3.s2"]["max"] == "430"
    assert rows["t4.p4.s0"]["typ"] == "150"
    assert rows["t4.p4.s0"]["unit"] == "mA·kΩ"
    assert rows["t4.p4.s0"]["parameter_symbol"] == (
        "Programming Scale Factor"
    )
    assert rows["t4.p4.s1"]["min"] == "450"
    assert rows["t4.p4.s2"]["max"] == "110"

    # 只允许 Start-Up Time 的 3 条进入人工复核。
    review = {
        row["logical_row_id"]
        for row in sample["logical_rows"]
        if row["needs_review"]
    }
    assert review == {
        "t4.p14.s0", "t4.p14.s1", "t4.p14.s2",
    }

    assert len(
        sample["unresolved_issues"]
    ) == 1
    issue = sample["unresolved_issues"][0]
    assert issue["value_line_count"] == 3
    assert issue["condition_line_count"] == 4


@pytest.mark.requires_data
def test_adi_no_fabricated_values(recovered):
    """恢复出的每个值必须能在 raw Docling 结构中找到。"""

    raw = load(
        RAW_DIR
        / "adi_lt3045.raw_structure.v1.json"
    )
    raw_text = " ".join(
        cell.get("text") or ""
        for table in raw["tables"]
        if table["table_id"] == 4
        for cell in table["cells"]
    )
    raw_text = (
        raw_text.replace("−", "-")
        .replace("–", "-")
        .replace("±", "")
    )

    for row in recovered["adi_lt3045"]["logical_rows"]:
        for role in ("min", "typ", "max"):
            value = row[role]

            if not value:
                continue

            tokens = re.findall(
                r"[+-]?\d+(?:\.\d+)?", value
            )

            for token in tokens:
                normalized = token.lstrip("+")
                assert normalized in raw_text, (
                    row["logical_row_id"],
                    role,
                    value,
                )


@pytest.mark.parametrize(
    "sample_id",
    [
        "adi_lt3045",
        "diodes_ap2112",
        "onsemi_ncp170",
        "renesas_isl80510",
    ],
)

@pytest.mark.requires_data
def test_recovered_numeric_values_exist_in_raw_cells(
    recovered, sample_id
):
    """所有 min/typ/max 数值都必须来自原始 cell 文本。"""

    table_id = CASES[sample_id][0]
    raw = load(
        RAW_DIR
        / f"{sample_id}.raw_structure.v1.json"
    )
    raw_text = " ".join(
        cell.get("text") or ""
        for table in raw["tables"]
        if table["table_id"] == table_id
        for cell in table["cells"]
    )
    raw_text = (
        raw_text.replace("−", "-")
        .replace("–", "-")
        .replace("±", "")
        .replace("+", "")
    )

    for row in recovered[sample_id]["logical_rows"]:
        for role in ("min", "typ", "max"):
            value = row[role]

            if not value:
                continue

            for token in re.findall(
                r"[+-]?\d+(?:\.\d+)?", value
            ):
                assert token.lstrip("+") in raw_text, (
                    sample_id,
                    row["logical_row_id"],
                    role,
                    value,
                )


@pytest.mark.requires_data
def test_ncp170_units_use_pdf_glyph_evidence(recovered):
    rows = rows_by_id(recovered["onsemi_ncp170"])

    assert rows["t4.p9"]["unit"] == "µA"
    assert rows["t4.p9"]["docling_unit_text"] == "mA"
    assert rows["t4.p10"]["unit"] == "µA"
    assert rows["t4.p15"]["unit"] == "µVrms"
    assert rows["t4.p15"]["docling_unit_text"] == "mVrms"
    assert "COUT = 1 µF" in rows["t4.p15"]["test_conditions"]

    evidence = rows["t4.p9"]["glyph_evidence"]
    assert evidence
    assert evidence[0]["char"] == "µ"
    left, top, right, bottom = evidence[0]["bbox"]
    assert 529 <= left <= 532
    assert 534 <= right <= 536
    assert bottom > top

    # 数值本身不能被单位恢复改动。
    assert rows["t4.p9"]["typ"] == "0.5"
    assert rows["t4.p9"]["max"] == "0.9"


@pytest.mark.requires_data
def test_ncp170_minus_and_full_coverage(recovered):
    sample = recovered["onsemi_ncp170"]
    rows = rows_by_id(sample)

    assert rows["t4.p5"]["min"] == "-20"
    assert rows["t4.p5"]["typ"] == "1"
    assert rows["t4.p5"]["max"] == "20"
    assert rows["t4.p5"]["unit"] == "mV"

    assert not any(
        row["needs_review"]
        for row in sample["logical_rows"]
    )
    assert sample["unresolved_issues"] == []


@pytest.mark.requires_data
def test_ap2112_cross_line_conditions_merged(recovered):
    rows = rows_by_id(recovered["diodes_ap2112"])

    assert rows["t4.p3"]["test_conditions"] == (
        "VIN = 2.5V, 1mA ≤ IOUT ≤ 600mA"
    )
    assert rows["t4.p9"]["test_conditions"] == (
        "VIN = 2.5V, VEN in OFF mode"
    )
    assert rows["t4.p14"]["test_conditions"] == (
        "No Load, 10Hz ≤ f ≤ 100kHz"
    )


@pytest.mark.requires_data
def test_renesas_footnote_min_max_columns_recovered(recovered):
    rows = rows_by_id(recovered["renesas_isl80510"])

    assert rows["t7.p5"]["min"] == "-0.9"
    assert rows["t7.p5"]["max"] == "0.9"
    assert rows["t7.p6"]["min"] == "-0.9"
    assert rows["t7.p6"]["max"] == "0.9"

    assert rows["t7.p7"]["typ"] == "2.2"
    assert rows["t7.p7"]["max"] == "4.6"
    assert rows["t7.p8"]["max"] == "5.7"

    assert rows["t7.p11"]["min"] == "1.35"
    assert rows["t7.p11"]["typ"] == "1.75"
    assert rows["t7.p11"]["max"] == "2.15"


@pytest.mark.requires_data
def test_adapter_preserves_logical_source_fields(recovered):
    raw = load(
        RAW_DIR
        / "adi_lt3045.raw_structure.v1.json"
    )
    adapted = adapt_logical_rows(
        recovered["adi_lt3045"],
        raw_structure=raw,
        table_ids=[4],
    )
    table = adapted["tables"][0]

    assert table["input_mode"] == "logical"
    assert table["row_count"] == len(table["records"])

    for record in table["records"]:
        assert record["logical_row_id"]
        assert record["logical_row_id"].startswith("t4.p")
        assert isinstance(
            record["primary_row_index"], int
        )
        assert record["primary_row_index"] in (
            record["source_row_indices"]
        )
        assert record["source_row_indices"]


def test_flat_and_logical_prompts_are_frozen_by_default():
    assert (
        prompts.LOGICAL_INPUT_INSTRUCTION
        == prompts.LOGICAL_INPUT_INSTRUCTION_BASE
        + prompts.LOGICAL_FAMILY_COVERAGE
    )


def test_family_coverage_prompt_lists_all_missing_families():
    families = [
        "line_regulation",
        "load_regulation",
        "shutdown_current",
        "short_circuit_current_limit",
        "output_voltage_noise",
    ]

    for family in families:
        assert family in prompts.LOGICAL_FAMILY_COVERAGE
        assert family in prompts.FLAT_FAMILY_COVERAGE

    # Prompt 只能写规则，不能写入具体型号数值。
    forbidden = [
        "1.23", "1.47", "180 mV", "240 mA",
        "450 mA", "750 mA", "65 mA", "130 µVRMS",
        "0.05", "225",
    ]

    for value in forbidden:
        assert value not in prompts.LOGICAL_FAMILY_COVERAGE
        assert value not in prompts.FLAT_FAMILY_COVERAGE


def _capture_instruction(table, family_coverage):
    extractor = DeepSeekExtractor.__new__(
        DeepSeekExtractor
    )
    extractor.rows_per_chunk = 10
    extractor.max_split_depth = 6
    extractor.family_coverage = family_coverage
    captured = {}

    def fake_request_json(payload, instruction):
        captured["instruction"] = instruction
        return {
            "component_model": "",
            "manufacturer": "",
            "parameters": [],
        }

    extractor._request_json = fake_request_json
    extractor._extract_chunk(
        table,
        table["rows"],
        table["row_indices"],
    )

    return captured["instruction"]


def test_prompt_family_coverage_toggle_isolated():
    flat = {
        "table_id": 4,
        "page_numbers": [4],
        "columns": ["PARAMETER", "MIN", "TYP", "MAX"],
        "rows": [["Line Regulation", "-1", "0.2", "1"]],
        "row_indices": [0],
    }
    logical = dict(flat)
    logical["input_mode"] = "logical"
    logical["logical_row_ids"] = ["t4.p1"]

    flat_off = _capture_instruction(flat, False)
    flat_on = _capture_instruction(flat, True)
    logical_off = _capture_instruction(logical, False)
    logical_on = _capture_instruction(logical, True)

    assert flat_off == EXTRACT_INSTRUCTION
    assert flat_on == (
        EXTRACT_INSTRUCTION + prompts.FLAT_FAMILY_COVERAGE
    )
    assert logical_off == (
        EXTRACT_INSTRUCTION
        + prompts.LOGICAL_INPUT_INSTRUCTION_BASE
    )
    assert logical_on == (
        EXTRACT_INSTRUCTION
        + prompts.LOGICAL_INPUT_INSTRUCTION
    )


@pytest.mark.requires_data
def test_validator_regression_removes_false_rejects(
    validator_regression,
):
    assert validator_regression["record_count"] == 13
    assert validator_regression["before_status_counts"] == {
        "invalid": 13,
    }
    assert validator_regression["after_status_counts"] == {
        "validated": 11,
        "pending_review": 2,
    }

    for record in validator_regression["records"]:
        assert record["after"][
            "final_validation_status"
        ] != "invalid", record["gold_id"]

        if record["root_cause"] == "multi_value_cell":
            assert record["after"][
                "final_validation_status"
            ] == "pending_review"


def test_real_typ_max_conflict_still_invalid():
    params = [
        {
            "name": "dropout_voltage",
            "value": 350,
            "unit": "mV",
            "value_type": "typ",
            "test_conditions": "IOUT = 500mA",
            "evidence": {
                "table_id": 4,
                "row_index": 4,
                "source_text": "VDROP 350 300",
            },
        },
        {
            "name": "dropout_voltage",
            "value": 300,
            "unit": "mV",
            "value_type": "max",
            "test_conditions": "IOUT = 500mA",
            "evidence": {
                "table_id": 4,
                "row_index": 4,
                "source_text": "VDROP 350 300",
            },
        },
    ]

    errors = check_min_typ_max(params)
    assert errors.get(0) == ["typ 大于 max"]
    assert errors.get(1) == ["typ 大于 max"]

    validated = ParameterValidator().validate(
        {"parameters": [dict(p) for p in params]}
    )["parameters"]

    assert all(
        item["validation_status"] == "invalid"
        for item in validated
    )


def test_same_row_punctuation_difference_still_conflicts():
    base = {
        "name": "dropout_voltage",
        "unit": "mV",
        "test_conditions": "IOUT = 500mA",
        "evidence": {
            "table_id": 4,
            "row_index": 4,
        },
    }
    params = [
        {
            **base,
            "value": 350,
            "value_type": "typ",
            "evidence": {
                **base["evidence"],
                "source_text": "VDROP | 350 | 300",
            },
        },
        {
            **base,
            "value": 300,
            "value_type": "max",
            "evidence": {
                **base["evidence"],
                "source_text": "VDROP  350  300",
            },
        },
    ]

    errors = check_min_typ_max(params)
    assert errors.get(0) == ["typ 大于 max"]
    assert errors.get(1) == ["typ 大于 max"]


def test_different_variants_do_not_conflict():
    params = [
        {
            "name": "dropout_voltage",
            "value": 1000,
            "unit": "mV",
            "value_type": "typ",
            "test_conditions": "IOUT = 10mA",
            "evidence": {
                "table_id": 4,
                "row_index": 4,
                "source_text": "VDROP 1000 1300",
            },
        },
        {
            "name": "dropout_voltage",
            "value": 700,
            "unit": "mV",
            "value_type": "max",
            "test_conditions": "IOUT = 10mA",
            "evidence": {
                "table_id": 4,
                "row_index": 4,
                "source_text": "VDROP 500 700",
            },
        },
    ]

    assert check_min_typ_max(params) == {}


def test_evidence_gap_stays_pending_review():
    validation = ParameterValidator().validate(
        {
            "parameters": [
                {
                    "name": "psrr",
                    "value": 57.0,
                    "unit": "dB",
                    "value_type": "typ",
                    "test_conditions": "f = 1kHz",
                    "validation_status": "pending_review",
                    "validation_errors": [
                        "来源单元格不是可直接比较的单一数字"
                    ],
                    "evidence": {
                        "table_id": 4,
                        "row_index": 13,
                        "page": 3,
                    },
                }
            ]
        }
    )
    parameter = validation["parameters"][0]

    assert parameter["validation_status"] == "pending_review"
    assert parameter["validation_errors"] == [
        "来源单元格不是可直接比较的单一数字"
    ]

"""B3.5d 离线回归测试：只读实验产物，不调用 LLM。

覆盖：
- 50 条 FN 归因与 B3.5c 指标一致；
- ADI table 4 非 Selector/非截断证据；
- 13 条 Validator 误拒的 B3.5d 快照保留，B3.6 生产函数已修复；
- Flat vs Recovered 结构指标不被当作真实 F1；
- 成本与调用数可复算。
"""

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from app.extraction.evidence import (  # noqa: E402
    find_column,
    verify_single_parameter,
)
from app.validation.rules import check_min_typ_max  # noqa: E402
from app.validation.validator import ParameterValidator  # noqa: E402


EXPERIMENT = (
    PROJECT_ROOT / "data/outputs/experiments/B3.5d"
)
TRACE = EXPERIMENT / "trace/fn_attribution.json"
ADI = EXPERIMENT / "trace/adi_table4.json"
STRUCTURE = EXPERIMENT / "structure/structure_compare.json"
VALIDATOR = EXPERIMENT / "validator/review.json"
COST = EXPERIMENT / "cost/cost_breakdown.json"
B35C_METRICS = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5c/results/metrics.json"
)
RUN_ROOT = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5c/runs/flat_baseline_r1"
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def trace():
    return load(TRACE)


@pytest.fixture(scope="module")
def adi():
    return load(ADI)


@pytest.fixture(scope="module")
def structure():
    return load(STRUCTURE)


@pytest.fixture(scope="module")
def validator_review():
    return load(VALIDATOR)


@pytest.fixture(scope="module")
def cost():
    return load(COST)


@pytest.mark.requires_data
def test_artifacts_exist():
    for path in (
        TRACE,
        ADI,
        STRUCTURE,
        VALIDATOR,
        COST,
        EXPERIMENT / "REPORT.md",
        EXPERIMENT / "reports/adi_table4.md",
        EXPERIMENT / "reports/flat_vs_recovered.md",
        EXPERIMENT / "reports/fix_priority.md",
        EXPERIMENT / "reports/validator_review.md",
    ):
        assert path.is_file(), path


@pytest.mark.requires_data
def test_trace_accounts_for_all_50_fn(trace):
    assert trace["fn_total"] == 50
    assert len(trace["records"]) == 50
    assert trace["fn_total"] == sum(
        trace["stage_counts"].values()
    )
    assert trace["fn_total"] == sum(
        trace["stage_group_counts"].values()
    )
    assert set(trace["stage_counts"]) <= {
        "structure_recovery_gap",
        "table_parser_unit_error",
        "table_parser_row_continuation_error",
        "llm_omitted",
        "llm_wrong_value",
        "llm_wrong_unit",
        "llm_wrong_value_type",
        "retry_or_merge_lost",
        "validator_dropped",
        "scoring_mismatch",
        "unknown",
    }


@pytest.mark.requires_data
def test_trace_per_sample_fn_matches_b35c_metrics(trace):
    metrics = load(B35C_METRICS)
    expected = {
        sample["sample_id"]: sample["fn"]
        for sample in metrics["samples"]
    }
    actual = {}

    for record in trace["records"]:
        actual[record["sample_id"]] = (
            actual.get(record["sample_id"], 0) + 1
        )

    assert actual == expected
    assert expected == {
        "adi_lt3045": 18,
        "diodes_ap2112": 11,
        "onsemi_ncp170": 11,
        "renesas_isl80510": 10,
    }


@pytest.mark.requires_data
def test_trace_has_no_validator_or_scoring_fn(trace):
    assert (
        trace["stage_group_counts"].get(
            "validator_retry", 0
        )
        == 0
    )
    assert (
        trace["stage_group_counts"].get("scoring", 0) == 0
    )
    assert not any(
        record["primary_stage"]
        in {
            "validator_dropped",
            "retry_or_merge_lost",
            "scoring_mismatch",
        }
        for record in trace["records"]
    )


@pytest.mark.requires_data
def test_all_fn_have_gold_and_evidence_chain(trace):
    for record in trace["records"]:
        assert record["gold_id"]
        assert record["table_selected"] is True
        assert record["flat_input"]["table_present"] is True
        assert "raw_llm" in record
        assert "structure_recovery" in record
        assert "final_predictions" in record


@pytest.mark.requires_data
def test_adi_table4_selected_and_not_truncated(adi):
    assert 4 in adi["selected_table_ids"]
    calls = adi["table4_calls"]

    assert calls["calls"] == 2
    assert calls["finish_reasons"] == {"stop": 2}
    assert calls["parameter_counts"] == [0, 0]
    assert calls["content_lengths"] == [56, 56]
    assert adi["table4_final_parameters"] == 0
    assert adi["final_parameters_by_table"] == {
        "3": 24,
        "11": 33,
    }


@pytest.mark.requires_data
def test_adi_table4_multi_value_and_adapter_gap(adi):
    assert adi["gold_record_count"] == 18
    assert adi["gold_records_in_multi_value_cell"] == 17
    assert adi["flat_multi_value_row_count"] == 6
    assert adi["logical_rows_without_values"] == 12
    assert adi["adapter_excluded_rows"] == 12
    assert adi["adapter_excluded_reasons"] == {
        "缺少 min/typ/max 数值": 12
    }
    assert adi["structure_recovered_complete"] == 0
    assert adi["adapter_complete"] == 0


@pytest.mark.requires_data
def test_adi_table4_rows_contain_merged_values(adi):
    rows = {
        row_index: row
        for row_index, row in enumerate(
            adi["gold_table"]["rows"]
        )
    }

    assert rows[2][3] == "570 230"
    assert rows[2][4] == "710 700 330"
    assert rows[2][5] == "850 430"
    assert rows[3][6] == "mA · kΩ mA mA"


@pytest.mark.requires_data
def test_validator_review_reproduces_all_13(validator_review):
    assert validator_review["total"] == 13
    assert validator_review["all_reproduced"] is True
    assert all(
        record["content_matches_gold"]
        for record in validator_review["records"]
    )
    assert all(
        record["reproduced"]["final_validation_status"]
        == "invalid"
        for record in validator_review["records"]
    )
    assert validator_review["root_cause_counts"] == {
        "cross_variant_grouping": 7,
        "ambiguous_column_header": 3,
        "multi_value_cell": 2,
        "merged_unit_cell": 1,
    }


@pytest.mark.requires_data
def test_validator_b36_fixes_b35d_misrejects(
    validator_review,
):
    """B3.5d 快照保留旧结果；B3.6 生产函数已修复误拒。"""

    expected_status = {
        "merged_unit_cell": "validated",
        "ambiguous_column_header": "validated",
        "multi_value_cell": "pending_review",
        "cross_variant_grouping": "validated",
    }
    cache = {}

    for record in validator_review["records"]:
        sample_id = record["sample_id"]

        if sample_id not in cache:
            selected = load(
                RUN_ROOT / sample_id / "selected_tables.json"
            )
            validation = load(
                RUN_ROOT / sample_id / "validation.json"
            )
            cache[sample_id] = {
                "tables": {
                    table["table_id"]: table
                    for table in selected["tables"]
                },
                "params": validation["parameters"],
            }

        state = cache[sample_id]
        param = state["params"][
            record["prediction_index"]
        ]
        status, issues = verify_single_parameter(
            param, state["tables"]
        )
        root_cause = record["root_cause"]

        assert status == expected_status[root_cause], (
            record["gold_id"],
            status,
            issues,
        )

        if root_cause in (
            "merged_unit_cell",
            "ambiguous_column_header",
        ):
            assert issues == [], record["gold_id"]


def test_cross_variant_grouping_no_longer_conflicts():
    params = [
        {
            "name": "dropout_voltage",
            "value": 1000.0,
            "unit": "mV",
            "value_type": "typ",
            "test_conditions": "IOUT = 10mA",
            "evidence": {"table_id": 4, "row_index": 4},
        },
        {
            "name": "dropout_voltage",
            "value": 700.0,
            "unit": "mV",
            "value_type": "max",
            "test_conditions": "IOUT = 10mA",
            "evidence": {"table_id": 5, "row_index": 4},
        },
    ]

    errors = check_min_typ_max(params)

    # B3.6：不同 table_id/source_text 视为不同变体，不交叉比较。
    assert errors == {}


def test_footnote_header_is_recognized():
    # B3.6：MIN (Note 9) 去除脚注后可匹配 min。
    assert (
        find_column(
            ["MIN (Note 9)", "TYP", "MAX (Note 9)", "UNIT"],
            {"min", "minimum"},
        )
        == 0
    )
    assert (
        find_column(
            ["MIN (Note 9)", "TYP", "MAX (Note 9)", "UNIT"],
            {"max", "maximum"},
        )
        == 2
    )
    assert find_column(["MIN", "MAX"], {"min"}) == 0


def test_pending_review_is_not_promoted_to_invalid():
    # B3.6：证据不足的 pending_review 不得被压缩成 invalid。
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


@pytest.mark.requires_data
def test_ncp170_micro_amp_lost_in_flat_input(trace):
    records = {
        record["gold_id"]: record
        for record in trace["records"]
    }
    record = records["onsemi_ncp170:t4:r8:typ"]

    assert record["gold_unit"] == "ua"
    assert record["flat_input"]["unit_cell"] == "mA"
    assert record["primary_stage"] == "table_parser_unit_error"
    assert record["raw_llm"]["same_identity_indices"]

    raw_units = {
        item["unit"]
        for item in record["raw_llm"]["params"]
    }
    assert raw_units == {"mA"}


@pytest.mark.requires_data
def test_structure_compare_is_not_claimed_as_f1(structure):
    totals = structure["totals"]

    assert totals["gold_records"] == 76
    assert totals["recovered_rows_complete"] == 40
    assert totals["adapter_complete"] == 33
    assert totals["recovered_coverage"] == 0.5263
    assert totals["adapter_coverage"] == 0.4342
    assert "不是真实提取 F1" in structure["note"]
    assert "未进入" in structure["note"]

    by_id = {
        sample["sample_id"]: sample
        for sample in structure["samples"]
    }
    assert by_id["adi_lt3045"]["recovered_coverage"] == 0.0
    assert by_id["diodes_ap2112"]["recovered_coverage"] == 1.0
    assert by_id["onsemi_ncp170"]["recovered_coverage"] == 0.5789
    assert (
        by_id["renesas_isl80510"]["recovered_coverage"]
        == 0.4444
    )


@pytest.mark.requires_data
def test_cost_totals_are_consistent(cost):
    totals = cost["totals"]

    assert totals["calls"] == 68
    assert totals["prompt_tokens"] == 94028
    assert totals["completion_tokens"] == 215522
    assert totals["reasoning_tokens"] == 189609
    assert (
        totals["prompt_tokens"]
        + totals["completion_tokens"]
        == totals["total_tokens"]
    )
    assert sum(
        sample["total"]["calls"]
        for sample in cost["samples"]
    ) == totals["calls"]
    assert totals["non_gold_table_call_ratio"] == 0.75
    assert totals["retry_call_ratio"] == 0.4706
    assert totals["length_call_ratio"] == 0.2206
    assert totals["length_calls"]["calls"] == 15
    assert (
        totals["length_calls"]["completion_tokens"]
        <= totals["completion_tokens"]
    )

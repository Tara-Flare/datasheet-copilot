"""B3.4b：Logical Prompt 完整性优化的回归测试。

只检查 Prompt 内容、大小和注入边界，不调用真实 API。
"""

import hashlib

from app.extraction import prompts
from app.extraction.deepseek_extractor import (
    EXTRACT_INSTRUCTION,
    DeepSeekExtractor,
)


BASELINE_SYSTEM_PROMPT_SHA256 = (
    "e79bff0b8112438fbd684ed746ee27dbee26b70539064f4d2e657f31d0dfa908"
)
BASELINE_EXTRACT_INSTRUCTION_SHA256 = (
    "d05776f90044f0e5e34013af9705bccf5dcd15b9fc0079afe5511bd24abe58c9"
)
BASELINE_LOGICAL_RETRY_SHA256 = (
    "112ac739733bdc21cf113ec37752c48762007fb04f866a789cd6e115ae41f8b0"
)
B34A_LOGICAL_INPUT_SHA256 = (
    "0be6d99c97c1d4132b9fa8e9f16bd5b40e6f786992173739b6ed34daacfb439c"
)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_flat_and_retry_prompts_unchanged():
    assert (
        sha256_text(prompts.SYSTEM_PROMPT)
        == BASELINE_SYSTEM_PROMPT_SHA256
    )
    assert (
        sha256_text(EXTRACT_INSTRUCTION)
        == BASELINE_EXTRACT_INSTRUCTION_SHA256
    )
    assert (
        sha256_text(prompts.LOGICAL_RETRY_INSTRUCTION)
        == BASELINE_LOGICAL_RETRY_SHA256
    )


def test_logical_prompt_changed_for_b34b():
    assert prompts.LOGICAL_INPUT_INSTRUCTION_VERSION == "b3.4b-v1"
    assert (
        sha256_text(prompts.LOGICAL_INPUT_INSTRUCTION)
        != B34A_LOGICAL_INPUT_SHA256
    )


def test_logical_prompt_contains_completeness_rules():
    text = prompts.LOGICAL_INPUT_INSTRUCTION

    required = [
        "逐条检查每个 logical_row_id",
        "完成全部逻辑行",
        "min、typ、max",
        "必须输出一条记录",
        "不得合并",
        "不得把相邻行数值拼接到当前行",
    ]

    for phrase in required:
        assert phrase in text


def test_logical_prompt_covers_required_parameter_families():
    text = prompts.LOGICAL_INPUT_INSTRUCTION

    required = [
        "output_voltage_accuracy",
        "line_regulation",
        "load_regulation",
        "ground_current",
        "ground_current_vs_load_current",
        "Ground Current in Dropout",
        "shutdown_current",
        "output_current_limit",
        "short_circuit_current_limit",
        "output_voltage_noise",
        "dropout_voltage",
        "psrr",
        "uvlo_threshold",
        "uvlo_hysteresis",
        "VIN rising",
        "VIN falling",
        "VIN hysteresis",
    ]

    for phrase in required:
        assert phrase in text


def test_logical_prompt_requires_clean_conditions_and_source():
    text = prompts.LOGICAL_INPUT_INSTRUCTION

    required = [
        "test_conditions 直接复制",
        "不得省略明确的 TJ/温度条件",
        "不得写入参数名称",
        "logical_row_id",
        "source_row_indices",
    ]

    for phrase in required:
        assert phrase in text


def test_logical_prompt_is_general_ldo_not_tps7a02_specific():
    text = prompts.LOGICAL_INPUT_INSTRUCTION

    forbidden = [
        "TPS7A02",
        "t9.",
        "1.23",
        "1.47",
        "180 mV",
        "240 mA",
        "450 mA",
        "750 mA",
        "65 mA",
        "130 µVRMS",
    ]

    for value in forbidden:
        assert value not in text


def test_logical_prompt_size_budget():
    # B3.4a 为 674 字符；B3.4b 允许增加完整性规则，
    # 但必须控制 Prompt Token 开销。
    assert len(prompts.LOGICAL_INPUT_INSTRUCTION) <= 2600


def _capture_chunk_instruction(table):
    extractor = DeepSeekExtractor.__new__(
        DeepSeekExtractor
    )
    extractor.rows_per_chunk = 10
    extractor.max_split_depth = 6
    captured = {}

    def fake_request_json(payload, instruction):
        captured["payload"] = payload
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
        table.get("row_indices")
        or list(range(len(table["rows"]))),
    )

    return captured


def test_logical_instruction_only_injected_for_logical_tables():
    flat_table = {
        "table_id": 1,
        "page_numbers": [1],
        "columns": ["Parameter", "Min"],
        "rows": [["VOUT", "1.0"]],
        "row_indices": [0],
    }
    logical_table = {
        "table_id": 1,
        "page_numbers": [1],
        "columns": ["Parameter", "Min"],
        "rows": [["VOUT", "1.0"]],
        "row_indices": [0],
        "input_mode": "logical",
        "records": [{
            "logical_row_id": "t1.p1",
            "source_row_indices": [0],
            "source_cell_ids": [1, 2],
        }],
    }

    flat = _capture_chunk_instruction(flat_table)
    logical = _capture_chunk_instruction(logical_table)

    assert flat["instruction"] == EXTRACT_INSTRUCTION
    assert (
        prompts.LOGICAL_INPUT_INSTRUCTION
        not in flat["instruction"]
    )
    assert logical["instruction"] == (
        EXTRACT_INSTRUCTION
        + prompts.LOGICAL_INPUT_INSTRUCTION
    )
    assert (
        logical["payload"]["tables"][0]["input_mode"]
        == "logical"
    )
    assert logical["payload"]["tables"][0][
        "logical_row_ids"
    ] == ["t1.p1"]

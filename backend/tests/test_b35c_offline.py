"""B3.5c 离线回归测试：不访问 DeepSeek，不修改冻结 Gold。

覆盖：
- 冻结 Gold / 回执门禁只读且可通过；
- 评分器归一化、匹配与 out_of_scope 分类；
- 使用缓存表格 + 假 LLM 客户端跑通完整 Flat graph、
  artifact 保存与评分链路。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import b35c_common  # noqa: E402
import b35c_score_flat  # noqa: E402


@pytest.mark.requires_data
def test_gold_gate_passes_and_is_readonly():
    before = {
        name: b35c_common.sha256_file(path)
        for name, path in (
            ("gold", b35c_common.GOLD),
            ("receipt", b35c_common.GOLD_RECEIPT),
            ("manifest", b35c_common.MANIFEST),
        )
    }

    gate = b35c_common.validate_gold_gate()

    assert gate["gold"]["status"] == "frozen"
    assert gate["gold"]["record_count"] == 76
    assert len(gate["samples"]) == 4
    assert all(
        sample["record_count"] > 0 for sample in gate["samples"]
    )
    assert gate["sync_receipt"]["record_count"] == 88

    after = {
        name: b35c_common.sha256_file(path)
        for name, path in (
            ("gold", b35c_common.GOLD),
            ("receipt", b35c_common.GOLD_RECEIPT),
            ("manifest", b35c_common.MANIFEST),
        )
    }
    assert before == after


def test_identity_and_unit_normalization():
    assert (
        b35c_score_flat.normalize_identity("Ground Current")
        == b35c_score_flat.normalize_identity("ground_current")
    )
    assert (
        b35c_score_flat.normalize_identity("Dropout Voltage")
        == "dropout"
    )
    assert (
        b35c_score_flat.normalize_identity("VDO")
        == "dropout"
    )
    assert (
        b35c_score_flat.normalize_unit("µA")
        == b35c_score_flat.normalize_unit("uA")
    )
    assert (
        b35c_score_flat.normalize_unit("mA·kΩ")
        == b35c_score_flat.normalize_unit("mA*kohm")
    )


def test_value_and_condition_matching():
    assert b35c_score_flat.value_match(
        {"value": "1.0"}, {"value": "1"}
    )
    assert not b35c_score_flat.value_match(
        {"value": "1.1"}, {"value": "1"}
    )

    gold_tokens = b35c_score_flat.condition_tokens(
        "VIN = 2V; TA = 25°C"
    )
    pred_tokens = b35c_score_flat.condition_tokens(
        "TA = 25°C; VIN = 2V; note extra"
    )
    assert gold_tokens.issubset(pred_tokens)

    ranged = b35c_score_flat.condition_tokens(
        "2.5V <= VIN <= 6V"
    )
    assert any("vin:" in token for token in ranged)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_score_sample_tp_fp_fn_and_out_of_scope(tmp_path):
    sample = {
        "sample_id": "demo",
        "vendor": "DemoVendor",
        "model": "DEMO1",
        "document_version": "Rev A",
        "gold_scope_table_ids": [4],
    }
    gold_sample = {
        "records": [
            {
                "gold_id": "demo:t4:r0:typ",
                "source": {
                    "table_id": 4,
                    "row_index": 0,
                    "page": 1,
                    "row_text": "Ground Current | 1 | µA",
                },
                "parameter_identity": "ground_current",
                "value_type": "typ",
                "value": "1",
                "unit": "µA",
                "test_conditions": "VIN = 5V",
                "condition_source": "row",
                "review": {"status": "confirmed"},
            }
        ]
    }
    run_dir = tmp_path / "demo"
    _write_json(
        run_dir / "artifact.json",
        {
            "extraction": {"parameters": []},
            "final_extraction": None,
        },
    )
    _write_json(
        run_dir / "validation.json",
        {
            "parameters": [
                {
                    "name": "Ground Current",
                    "value": 1,
                    "unit": "µA",
                    "value_type": "typ",
                    "test_conditions": "",
                    "evidence": {
                        "table_id": 9,
                        "row_index": 0,
                        "page": 2,
                        "source_text": "",
                    },
                    "validation_status": "validated",
                },
                {
                    "name": "Ground Current",
                    "value": 1,
                    "unit": "µA",
                    "value_type": "typ",
                    "test_conditions": "VIN = 5V",
                    "evidence": {
                        "table_id": 4,
                        "row_index": 0,
                        "page": 1,
                        "source_text": "Ground Current | 1 | µA",
                    },
                    "validation_status": "validated",
                },
                {
                    "name": "Ground Current",
                    "value": 2,
                    "unit": "µA",
                    "value_type": "max",
                    "test_conditions": "",
                    "evidence": {
                        "table_id": 4,
                        "row_index": 0,
                        "page": 1,
                        "source_text": "",
                    },
                    "validation_status": "invalid",
                },
                {
                    "name": "Dropout",
                    "value": 1,
                    "unit": "mV",
                    "value_type": "typ",
                    "test_conditions": "",
                    "evidence": {
                        "table_id": 4,
                        "row_index": 5,
                        "page": 1,
                        "source_text": "",
                    },
                    "validation_status": "validated",
                },
            ],
            "validation_summary": {"total": 4},
        },
    )

    result = b35c_score_flat.score_sample(
        sample, gold_sample, run_dir
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["predictions_in_scope"] == 2
    assert result["out_of_scope_table_predictions"] == 1
    assert result["out_of_scope_parameter_predictions"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["conditions_complete_rate"] == 1.0
    assert (
        len(
            [
                error
                for error in result["errors"]
                if error["kind"] == "FP"
            ]
        )
        == result["fp"]
    )


class _FakeCompletions:
    def __init__(self, responder):
        self._responder = responder

    def create(self, *args, **kwargs):
        return self._responder(kwargs)


class _FakeClient:
    def __init__(self, responder):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(responder)
        )


def _fake_response(parameters):
    content = json.dumps(
        {
            "component_model": "",
            "manufacturer": "",
            "parameters": parameters,
        },
        ensure_ascii=False,
    )

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 10
        total_tokens = 20
        completion_tokens_details = SimpleNamespace(
            reasoning_tokens=0
        )

        def model_dump(self):
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    usage = _Usage()
    return SimpleNamespace(
        id="fake",
        model="deepseek-flash",
        created=0,
        system_fingerprint="fake",
        usage=usage,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
    )


@pytest.mark.requires_data
def test_run_one_and_score_with_fake_llm(tmp_path, monkeypatch):
    import b35c_run_flat_baseline as runner
    from app.extraction.deepseek_extractor import DeepSeekExtractor

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")

    gate = b35c_common.validate_gold_gate()
    manifest_by_id = {
        sample["sample_id"]: sample
        for sample in gate["manifest"]["samples"]
    }
    gold_by_id = {
        sample["sample_id"]: sample
        for sample in gate["gold"]["samples"]
    }
    sample = manifest_by_id["renesas_isl80510"]
    gold_sample = gold_by_id["renesas_isl80510"]
    scope_table = sample["gold_scope_table_ids"][0]
    gold_records = [
        record
        for record in gold_sample["records"]
        if record["source"]["table_id"] == scope_table
    ][:2]

    def responder(kwargs):
        payload = json.loads(
            kwargs["messages"][-1]["content"]
        )

        if "tables" not in payload:
            return _fake_response([])

        table_id = payload["tables"][0]["table_id"]

        if table_id != scope_table:
            return _fake_response([])

        parameters = []

        for record in gold_records:
            parameters.append(
                {
                    "name": record["parameter_identity"],
                    "value": float(record["value"]),
                    "unit": record["unit"],
                    "value_type": record["value_type"],
                    "test_conditions": record[
                        "test_conditions"
                    ],
                    "evidence": {
                        "table_id": record["source"][
                            "table_id"
                        ],
                        "row_index": record["source"][
                            "row_index"
                        ],
                        "page": record["source"]["page"],
                        "source_text": record["source"][
                            "row_text"
                        ],
                    },
                }
            )

        return _fake_response(parameters)

    real_extractor = DeepSeekExtractor()
    real_extractor.rows_per_chunk = 10000
    real_extractor.client = _FakeClient(responder)

    monkeypatch.setattr(
        runner,
        "DeepSeekExtractor",
        lambda: real_extractor,
    )

    class _CachedOnlyParser:
        def __init__(self):
            self.pipeline_options = SimpleNamespace(
                do_ocr=False, do_table_structure=True
            )

        def parse(self, pdf_path):
            raise AssertionError(
                "缓存在测试中应命中，不应真正解析 PDF"
            )

    monkeypatch.setattr(
        runner,
        "build_offline_table_parser",
        lambda: _CachedOnlyParser(),
    )

    from app.parsers.table_cache import TableParseCache

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    error = runner.run_one(
        sample,
        gold_sample,
        run_dir / "renesas_isl80510",
        TableParseCache(str(b35c_common.DEFAULT_CACHE)),
        max_retries=1,
    )

    assert error is None
    assert (run_dir / "renesas_isl80510" / "artifact.json").is_file()

    result = b35c_score_flat.score_sample(
        sample,
        gold_sample,
        run_dir / "renesas_isl80510",
    )

    assert result["gold_records"] == len(gold_sample["records"])
    assert result["tp"] >= 1
    assert result["predictions_in_scope"] >= result["tp"]

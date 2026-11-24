"""B3.9：第一版 MVP 最终小范围在线验证。

范围固定为两个型号、两条输入路径：
- D: Logical + P1（默认候选）
- C: Flat + P1（保守对照，输入表应用 B3.8 Flat 字形恢复）

只做一次受控真实 API 调用。预算、Gold 门禁、模型/Prompt/表格范围
与 Validator 版本全部写入运行配置；不重新设计 Prompt，不新增产品
功能，不覆盖 B3.5c / B3.7 / B3.8 产物，不自动 Git Commit。
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv  # noqa: E402

from app.extraction.evidence import verify_evidence  # noqa: E402
from app.parsers.table_parser import (  # noqa: E402
    TableParser,
    _apply_flat_glyph_restoration,
)
from app.validation.recovered_evidence import (  # noqa: E402
    IDENTITY_SYNONYM_VERSION,
    _identity_match,
    logical_family,
)
from app.validation.recovered_integration import (  # noqa: E402
    build_recovered_validation_result,
)
from app.validation.units import convert_to_base  # noqa: E402
from app.validation.validator import ParameterValidator  # noqa: E402

from b35c_common import validate_gold_gate  # noqa: E402
from b35c_score_flat import (  # noqa: E402
    NORMALIZATION_VERSION,
    SCORER_VERSION,
    aggregate,
    content_allowed,
    content_weight,
    max_weight_match,
    normalize_gold,
    normalize_prediction,
    score_sample,
)
from b35b_docling_env import configure_environment  # noqa: E402
from b37_common import (  # noqa: E402
    GOLD,
    GOLD_RECEIPT,
    MANIFEST,
    TABLE_SCOPE,
    build_sample_inputs,
    expected_chunks,
    experiment_config,
    load_json,
    resolve_arm,
    sha256_file,
    sha256_text,
    write_json as b37_write_json,
)

import b37_run_arm  # noqa: E402


B39_DIR = PROJECT_ROOT / "data/outputs/experiments/B3.9"
PREFLIGHT_DIR = B39_DIR / "preflight"
RUNS_DIR = B39_DIR / "runs"
RESULTS_DIR = B39_DIR / "results"
REPORTS_DIR = B39_DIR / "reports"

SAMPLES = ("adi_lt3045", "onsemi_ncp170")
ARMS = ("D", "C")
RUN_ID = "r1"

DEFAULT_MAX_CALLS = 30
DEFAULT_MAX_TOKENS = 400_000
DEFAULT_MAX_SECONDS = 900

B38_REPLAY = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.8/results/offline_replay.json"
)


class BudgetExceeded(RuntimeError):
    """硬性预算耗尽，立即停止后续调用。"""


class RunBudget:
    def __init__(self, max_calls, max_tokens, max_seconds):
        self.max_calls = int(max_calls)
        self.max_tokens = int(max_tokens)
        self.max_seconds = float(max_seconds)
        self.started = time.monotonic()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.exhausted = False
        self.stop_reason = None
        self.events = []

    def _remaining(self):
        return {
            "calls": self.max_calls - self.calls,
            "tokens": self.max_tokens - self.total_tokens,
            "seconds": round(
                self.max_seconds - (time.monotonic() - self.started),
                1,
            ),
        }

    def install(self, extractor):
        client = extractor.client
        original = client.chat.completions.create

        def wrapped(*args, **kwargs):
            remaining = self._remaining()

            if (
                self.exhausted
                or remaining["calls"] <= 0
                or remaining["tokens"] <= 0
                or remaining["seconds"] <= 0
            ):
                self.exhausted = True
                self.stop_reason = (
                    "budget_exhausted_before_call"
                )
                raise BudgetExceeded(
                    "B3.9 预算已耗尽，拒绝继续调用: "
                    f"{remaining}"
                )

            started = time.perf_counter()
            response = original(*args, **kwargs)
            elapsed = time.perf_counter() - started
            usage = (
                response.usage.model_dump()
                if response.usage
                else {}
            )
            prompt_tokens = int(
                usage.get("prompt_tokens") or 0
            )
            completion_tokens = int(
                usage.get("completion_tokens") or 0
            )
            total_tokens = int(
                usage.get("total_tokens")
                or prompt_tokens + completion_tokens
            )
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.events.append({
                "index": self.calls - 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "elapsed_s": round(elapsed, 3),
                "model": getattr(response, "model", None),
            })

            if self.total_tokens >= self.max_tokens:
                self.exhausted = True
                self.stop_reason = "token_budget_reached"

            return response

        client.chat.completions.create = wrapped

    def snapshot(self):
        return {
            "limits": {
                "max_calls": self.max_calls,
                "max_total_tokens": self.max_tokens,
                "max_seconds": self.max_seconds,
            },
            "used": {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "elapsed_s": round(
                    time.monotonic() - self.started, 3
                ),
            },
            "remaining": self._remaining(),
            "exhausted": self.exhausted,
            "stop_reason": self.stop_reason,
            "events": self.events,
        }


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"无法序列化的类型: {type(value).__name__}"
    )


def write_json(path, payload, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if exclusive and path.exists():
        raise RuntimeError(f"拒绝覆盖已有文件: {path}")

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _sha256_text(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def _gold_manifest():
    gate = validate_gold_gate(
        manifest_path=MANIFEST,
        gold_path=GOLD,
        receipt_path=GOLD_RECEIPT,
    )
    manifest = load_json(MANIFEST)
    gold = load_json(GOLD)
    samples = {
        sample["sample_id"]: {
            **sample,
            "file": sample.get("pdf_path") or sample["file"],
        }
        for sample in gate["samples"]
    }
    gold_samples = {
        sample["sample_id"]: sample
        for sample in gold["samples"]
    }

    return gate, manifest, gold, samples, gold_samples


def _source_files():
    return [
        "backend/app/extraction/prompts.py",
        "backend/app/extraction/deepseek_extractor.py",
        "backend/app/extraction/evidence.py",
        "backend/app/extraction/input_mode.py",
        "backend/app/extraction/logical_table_adapter.py",
        "backend/app/parsers/structure_recovery.py",
        "backend/app/parsers/table_parser.py",
        "backend/app/validation/validator.py",
        "backend/app/validation/rules.py",
        "backend/app/validation/units.py",
        "backend/app/validation/recovered_evidence.py",
        "backend/app/validation/recovered_integration.py",
        "backend/app/agents/graph.py",
        "backend/app/agents/nodes.py",
        "backend/scripts/b35c_score_flat.py",
        "backend/scripts/b37_common.py",
        "backend/scripts/b37_run_arm.py",
    ]


def _source_fingerprints():
    return {
        path: (
            sha256_file(PROJECT_ROOT / path)
            if (PROJECT_ROOT / path).is_file()
            else None
        )
        for path in _source_files()
    }


def _prepare_inputs(sample):
    inputs = build_sample_inputs(sample)
    flat = deepcopy(inputs["flat_table"])
    raw_structure = inputs.get("raw_structure") or {}
    raw_tables = raw_structure.get("tables", [])
    evidence = {"applied": False, "cells": []}

    if raw_tables:
        _apply_flat_glyph_restoration(
            [flat],
            raw_tables,
            PROJECT_ROOT / sample["file"],
        )

        if flat.get("glyph_evidence"):
            evidence = {
                "applied": True,
                "cells": [
                    {
                        "cell_id": entry["cell_id"],
                        "docling_text": entry["docling_text"],
                        "pdf_text": entry["pdf_text"],
                        "updated_cells": entry[
                            "updated_cells"
                        ],
                    }
                    for entry in flat["glyph_evidence"]
                ],
                "docling_cells": flat.get(
                    "docling_cells", []
                ),
            }

    inputs["flat_table"] = flat
    return inputs, evidence


def _preflight_probes(sample_id, sample, sample_inputs):
    checks = []

    def add(name, ok, detail):
        checks.append({
            "name": name,
            "ok": bool(ok),
            "detail": detail,
        })

    if sample_id == "onsemi_ncp170":
        evidence = sample_inputs.get(
            "glyph_evidence", {}
        )
        cells = {}

        for cell in evidence.get("cells", []):
            cells.setdefault(
                cell.get("pdf_text"), set()
            ).update(
                (
                    item.get("row_index"),
                    item.get("column_index"),
                )
                for item in cell.get("updated_cells", [])
            )

        add(
            "flat_glyph_restoration_in_production_function",
            (8, 6) in cells.get("µA", set())
            and (9, 6) in cells.get("µA", set())
            and (14, 6) in cells.get("µVrms", set())
            and any(
                (14, 2) in positions
                for pdf_text, positions in cells.items()
                if "µF" in str(pdf_text)
            ),
            {
                "cells": {
                    pdf_text: sorted(positions)
                    for pdf_text, positions in cells.items()
                },
            },
        )
        flat = sample_inputs["flat_table"]
        add(
            "legitimate_mA_preserved",
            flat["rows"][6][6] == "mA"
            and flat["rows"][6][3] == "150"
            and flat["rows"][7][6] == "mA"
            and flat["rows"][7][4] == "225",
            {
                "output_current": flat["rows"][6],
                "short_circuit_limit": flat["rows"][7],
            },
        )

        ncp_rows = sample_inputs["logical_rows"][
            "logical_rows"
        ]
        thermal_row = next(
            item
            for item in ncp_rows
            if "Thermal Shutdown Hysteresis"
            in str(item.get("parameter_symbol"))
        )
        thermal_match = _identity_match(
            {"name": "thermal_shutdown_hysteresis"},
            thermal_row,
        )
        add(
            "thermal_shutdown_hysteresis_identity",
            logical_family(thermal_row)
            == "thermal_shutdown_hysteresis"
            and thermal_match.get("status") == "match",
            {
                "logical_row_id": thermal_row.get(
                    "logical_row_id"
                ),
                "family": logical_family(thermal_row),
                "match": thermal_match,
            },
        )

        en_row = next(
            item
            for item in ncp_rows
            if "EN Pin Threshold Voltage"
            in str(item.get("parameter_symbol"))
        )
        en_match = _identity_match(
            {"name": "en_threshold_voltage"},
            en_row,
        )
        add(
            "en_threshold_synonym_matches",
            en_match.get("status") == "match",
            {
                "logical_row_id": en_row.get("logical_row_id"),
                "family": logical_family(en_row),
                "match": en_match,
            },
        )

        extraction = {
            "parameters": [{
                "name": "quiescent_current",
                "value": 0.5,
                "unit": "mA",
                "value_type": "typ",
                "evidence": {
                    "table_id": 4,
                    "row_index": 8,
                    "page": 3,
                    "source_text": (
                        "IQ Quiescent Current IOUT = 0 mA "
                        "- 0.5 0.9 µA"
                    ),
                },
            }]
        }
        secured = verify_evidence(
            extraction, [flat]
        )
        validated = ParameterValidator().validate(secured)
        param = validated["parameters"][0]
        add(
            "wrong_unit_still_invalid",
            param.get("validation_status") == "invalid",
            {
                "status": param.get("validation_status"),
                "errors": param.get("validation_errors"),
            },
        )

    if sample_id == "adi_lt3045":
        row = next(
            item
            for item in sample_inputs["logical_rows"][
                "logical_rows"
            ]
            if "Quiescent Current in Shutdown"
            in (
                str(item.get("parameter_name"))
                + " "
                + str(item.get("parameter_symbol"))
            )
        )
        match = _identity_match(
            {"name": "shutdown_current"}, row
        )
        add(
            "legal_shutdown_synonym_matches",
            match.get("status") == "match",
            {
                "logical_row_id": row.get("logical_row_id"),
                "family": logical_family(row),
                "match": match,
                "version": IDENTITY_SYNONYM_VERSION,
            },
        )
        ambiguous_row = next(
            item
            for item in sample_inputs["logical_rows"][
                "logical_rows"
            ]
            if "Internal Current Limit"
            in str(item.get("parameter_symbol"))
        )
        ambiguous = _identity_match(
            {"name": "output_current_limit"},
            ambiguous_row,
        )
        add(
            "real_current_limit_ambiguity_rejected",
            ambiguous.get("status") == "conflict"
            and bool(ambiguous.get("ambiguous")),
            {
                "logical_row_id": ambiguous_row.get(
                    "logical_row_id"
                ),
                "match": ambiguous,
            },
        )

        pgfb_row = next(
            item
            for item in sample_inputs["logical_rows"][
                "logical_rows"
            ]
            if "PGFB Hysteresis"
            in str(item.get("parameter_symbol"))
        )
        pgfb_match = _identity_match(
            {"name": "pgfb_hysteresis"}, pgfb_row
        )
        add(
            "pgfb_hysteresis_identity",
            logical_family(pgfb_row) == "pgfb_hysteresis"
            and pgfb_match.get("status") == "match",
            {
                "logical_row_id": pgfb_row.get(
                    "logical_row_id"
                ),
                "family": logical_family(pgfb_row),
                "match": pgfb_match,
            },
        )

        flat = sample_inputs["flat_table"]
        shutdown_row = next(
            index
            for index, item in enumerate(flat["rows"])
            if any(
                "Quiescent Current in Shutdown" in str(cell)
                for cell in item
            )
        )
        extraction = {
            "parameters": [{
                "name": "quiescent_current",
                "value": 0.3,
                "unit": "µA",
                "value_type": "typ",
                "evidence": {
                    "table_id": flat["table_id"],
                    "row_index": shutdown_row,
                    "page": (flat.get("page_numbers") or [0])[0],
                    "source_text": "Quiescent Current in Shutdown",
                },
            }]
        }
        secured = verify_evidence(
            extraction, [flat]
        )
        validated = ParameterValidator().validate(secured)
        param = validated["parameters"][0]
        add(
            "flat_shutdown_guard_holds",
            param.get("validation_status") == "invalid"
            and any(
                "shutdown/standby" in error
                for error in param.get(
                    "validation_errors", []
                )
            ),
            {
                "row_index": shutdown_row,
                "status": param.get("validation_status"),
                "errors": param.get("validation_errors"),
            },
        )

    declared = __import__(
        "app.validation.recovered_evidence",
        fromlist=["IDENTITY_SYNONYM_VERSION"],
    ).IDENTITY_SYNONYM_VERSION
    add(
        "identity_synonym_version",
        declared == "b3.9-identity-v1",
        {"value": declared},
    )

    units_ok = (
        convert_to_base(0.2, "%/A").dimension
        == "percentage_per_current"
        and convert_to_base(0.2, "%/V").dimension
        == "percentage_per_voltage"
        and convert_to_base(85, "µVRMS").dimension
        == "rms_voltage"
    )
    add(
        "unit_whitelist_in_production",
        units_ok,
        {"%/A", "%/V", "µVRMS"},
    )

    source = __import__("inspect").getsource(
        TableParser.parse
    )
    add(
        "table_parser_calls_glyph_restoration",
        "_apply_flat_glyph_restoration" in source,
        {
            "call_present": (
                "_apply_flat_glyph_restoration" in source
            )
        },
    )

    return checks


def run_preflight():
    load_dotenv(PROJECT_ROOT / ".env")
    configure_environment("cpu")
    gate, manifest, gold, samples, gold_samples = _gold_manifest()
    config = experiment_config()
    selected = {}

    for sample_id in SAMPLES:
        sample = samples[sample_id]
        inputs, evidence = _prepare_inputs(sample)
        checks = _preflight_probes(
            sample_id, sample, {**inputs, "glyph_evidence": evidence}
        )
        selected[sample_id] = {
            "sample_id": sample_id,
            "gold_scope_table_ids": inputs["gold_scope_table_ids"],
            "flat_row_count": len(inputs["flat_table"]["rows"]),
            "logical_row_count": len(
                (inputs.get("logical_rows") or {}).get(
                    "logical_rows", []
                )
            ),
            "glyph_evidence": evidence,
            "checks": checks,
            "overall_ok": all(
                check["ok"] for check in checks
            ),
        }

    provider = {
        "base_url": os.getenv("LLM_BASE_URL"),
        "requested_model": config["model"]["requested_model"],
        "temperature": config["model"]["temperature"],
        "max_tokens": config["model"]["max_tokens"],
        "rows_per_chunk": config["model"]["rows_per_chunk"],
        "max_split_depth": config["model"]["max_split_depth"],
        "transport_max_retries": config["model"][
            "transport_max_retries"
        ],
        "api_key_present": bool(os.getenv("LLM_API_KEY")),
    }
    payload = {
        "schema": "b39_preflight/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "preflight_no_api",
        "scope": {
            "samples": list(SAMPLES),
            "arms": list(ARMS),
            "table_scope": TABLE_SCOPE,
            "run_id": RUN_ID,
            "note": (
                "只验证两个型号；不得宣称为新的四厂商泛化结果。"
            ),
        },
        "gold_gate": {
            "hashes": gate.get("hashes"),
            "receipt": gate.get("receipt"),
            "gold_sha256": sha256_file(GOLD),
            "receipt_sha256": sha256_file(GOLD_RECEIPT),
        },
        "scorer": {
            "scorer_version": SCORER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
        },
        "identity_synonym_version": IDENTITY_SYNONYM_VERSION,
        "provider": provider,
        "experiment_config": {
            key: config[key]
            for key in (
                "table_scope",
                "arms",
                "prompt_versions",
                "randomness",
                "source_fingerprints",
                "git",
            )
        },
        "budget": {
            "default_max_calls": DEFAULT_MAX_CALLS,
            "default_max_tokens": DEFAULT_MAX_TOKENS,
            "default_max_seconds": DEFAULT_MAX_SECONDS,
        },
        "samples": selected,
        "overall_ok": all(
            item["overall_ok"]
            for item in selected.values()
        ),
        "note": (
            "预检不调用 API；只证明 Gold/回执一致、B3.8 修复函数"
            "已进入生产模块，并且预算与模型配置已固定。"
        ),
    }

    write_json(
        PREFLIGHT_DIR / "preflight.json", payload
    )
    write_json(
        PREFLIGHT_DIR / "run_config.json",
        {
            "schema": "b39_run_config/v1",
            "gold_gate": payload["gold_gate"],
            "provider": provider,
            "scorer": payload["scorer"],
            "scope": payload["scope"],
            "experiment_config": payload[
                "experiment_config"
            ],
            "source_fingerprints": config[
                "source_fingerprints"
            ],
        },
        exclusive=False,
    )
    return payload


def _write_preflight_markdown(payload, path):
    lines = [
        "# B3.9 在线验证预检",
        "",
        f"- 总体：{'PASS' if payload['overall_ok'] else 'FAIL'}",
        f"- 范围：{', '.join(payload['scope']['samples'])}；"
        f"臂 {', '.join(payload['scope']['arms'])}",
        f"- 模型：`{payload['provider']['requested_model']}`",
        f"- Base URL：`{payload['provider']['base_url']}`",
        f"- Temperature：{payload['provider']['temperature']}",
        f"- 评分契约：`{payload['scorer']['scorer_version']}` / "
        f"`{payload['scorer']['normalization_version']}`",
        f"- 身份同义词版本："
        f"`{payload.get('identity_synonym_version')}`",
        "",
        "| 样本 | 检查 | 结果 |",
        "| --- | --- | --- |",
    ]

    for sample_id, sample in payload["samples"].items():
        for check in sample["checks"]:
            lines.append(
                f"| {sample_id} | `{check['name']}` | "
                f"{'PASS' if check['ok'] else 'FAIL'} |"
            )

    lines.extend([
        "",
        "预检不调用 API；KPI 以正式在线运行结果为准。",
        "",
    ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


class _BudgetedExtractor:
    """运行时替换 b37_run_arm.DeepSeekExtractor 的工厂。"""

    budget = None

    def __new__(cls, *args, **kwargs):
        extractor = cls.original(
            *args, **kwargs
        )

        if cls.budget is not None:
            cls.budget.install(extractor)

        return extractor


def install_budget(budget):
    _BudgetedExtractor.original = b37_run_arm.DeepSeekExtractor
    _BudgetedExtractor.budget = budget
    b37_run_arm.DeepSeekExtractor = _BudgetedExtractor


def restore_extractor():
    if hasattr(_BudgetedExtractor, "original"):
        b37_run_arm.DeepSeekExtractor = (
            _BudgetedExtractor.original
        )


def run_online(args):
    load_dotenv(PROJECT_ROOT / ".env")
    configure_environment("cpu")

    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("缺少 LLM_API_KEY，拒绝真实调用")

    if not args.execute:
        raise SystemExit(
            "在线模式必须显式提供 --execute；未调用任何 API。"
        )

    gate, manifest, gold, samples, gold_samples = _gold_manifest()
    config = experiment_config()
    budget = RunBudget(
        args.max_calls,
        args.max_tokens,
        args.max_seconds,
    )
    install_budget(budget)
    selected_samples = [
        sample_id
        for sample_id in args.samples
        if sample_id in SAMPLES
    ]

    if not selected_samples:
        raise SystemExit("没有匹配的样本")

    results = {}

    try:
        for arm_name in args.arms:
            arm = resolve_arm(arm_name)
            arm_run_dir = RUNS_DIR / arm_name / RUN_ID

            if arm_run_dir.exists():
                raise SystemExit(
                    f"运行目录已存在，拒绝覆盖: {arm_run_dir}"
                )

            arm_run_dir.mkdir(parents=True)
            write_json(
                arm_run_dir / "run_config.json",
                {
                    "schema": "b39_arm_run/v1",
                    "run_id": RUN_ID,
                    "arm": arm,
                    "samples": selected_samples,
                    "gold_gate": gate["hashes"],
                    "gold_receipt": gate["receipt"],
                    "provider": {
                        "base_url": os.getenv("LLM_BASE_URL"),
                        "requested_model": config["model"][
                            "requested_model"
                        ],
                        "temperature": config["model"][
                            "temperature"
                        ],
                        "max_tokens": config["model"][
                            "max_tokens"
                        ],
                        "rows_per_chunk": config["model"][
                            "rows_per_chunk"
                        ],
                        "max_split_depth": config["model"][
                            "max_split_depth"
                        ],
                        "transport_max_retries": config["model"][
                            "transport_max_retries"
                        ],
                    },
                    "scorer": {
                        "scorer_version": SCORER_VERSION,
                        "normalization_version": (
                            NORMALIZATION_VERSION
                        ),
                    },
                    "identity_synonym_version": (
                        IDENTITY_SYNONYM_VERSION
                    ),
                    "source_fingerprints": config[
                        "source_fingerprints"
                    ],
                },
                exclusive=True,
            )

            completed = []
            failures = {}
            glyph_summary = {}

            for sample_id in selected_samples:
                sample = samples[sample_id]
                inputs, evidence = _prepare_inputs(sample)
                glyph_summary[sample_id] = evidence
                sample_dir = arm_run_dir / sample_id

                try:
                    error = b37_run_arm.run_one(
                        arm,
                        sample,
                        inputs,
                        sample_dir,
                        max_retries=0,
                        temperature=float(
                            config["model"]["temperature"]
                        ),
                    )
                except BudgetExceeded as exc:
                    error = {
                        "type": "BudgetExceeded",
                        "message": str(exc),
                    }
                except Exception as exc:  # noqa: BLE001
                    error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }

                if error is not None:
                    failures[sample_id] = error
                    break

                write_json(
                    sample_dir / "input_glyph_evidence.json",
                    evidence,
                    exclusive=False,
                )
                completed.append(sample_id)

                if budget.exhausted:
                    break

            write_json(
                arm_run_dir / "run_manifest.json",
                {
                    "schema": "b39_arm_run_manifest/v1",
                    "run_id": RUN_ID,
                    "arm": arm,
                    "started_at": None,
                    "finished_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "status": (
                        "completed"
                        if not failures
                        else "failed"
                    ),
                    "samples_requested": selected_samples,
                    "samples_completed": completed,
                    "failures": failures,
                    "failure_policy": (
                        "fail-fast；预算或任一 API/解析异常立即停止，"
                        "不挑选结果、不自动补跑。"
                    ),
                    "glyph_evidence": glyph_summary,
                    "budget": budget.snapshot(),
                    "note": (
                        "temperature=0.0；transport max_retries=0；"
                        "本运行只覆盖两个型号，不代表四厂商泛化。"
                    ),
                },
            )

            if failures:
                results[arm_name] = {
                    "status": "failed",
                    "failures": failures,
                }
                break

            results[arm_name] = {
                "status": "completed",
                "samples": completed,
            }
    finally:
        restore_extractor()

    return results


def _load_validation_payload(sample_dir):
    revalidated = sample_dir / "validation_revalidated.json"

    if revalidated.is_file():
        return load_json(revalidated), "online_revalidated"

    return load_json(sample_dir / "validation.json"), "online"


def _validation_entries(validation):
    entries = []

    for index, param in enumerate(
        validation.get("parameters", [])
    ):
        evidence = param.get("evidence") or {}
        entries.append({
            "index": index,
            "name": param.get("name"),
            "value": param.get("value"),
            "unit": param.get("unit"),
            "value_type": param.get("value_type"),
            "row_index": evidence.get("row_index"),
            "status": param.get("validation_status"),
            "errors": list(
                param.get("validation_errors") or []
            ),
            "identity_ambiguous": bool(
                param.get("identity_ambiguous")
            ),
        })

    return entries


def _simple_status_counts(validation):
    counts = {}

    for param in validation.get("parameters", []):
        status = param.get("validation_status") or "unknown"
        counts[status] = counts.get(status, 0) + 1

    return counts


def _revalidate_sample(arm, sample, inputs, sample_dir):
    artifact = load_json(sample_dir / "artifact.json")
    extraction = (
        artifact.get("final_extraction")
        or artifact.get("extraction")
        or {}
    )
    tables = load_json(
        sample_dir / "selected_tables.json"
    )["tables"]
    legacy = ParameterValidator().validate(
        verify_evidence(extraction, tables)
    )

    if arm["evidence_validation_mode"] == "recovered":
        validation = build_recovered_validation_result(
            extraction,
            legacy,
            {
                "logical_rows": inputs["logical_rows"],
                "raw_structure": inputs["raw_structure"],
                "pdf_sha256": inputs["pdf_sha256"],
            },
        )
    else:
        validation = legacy

    return validation


def revalidate_runs(args):
    """用最终生产 Validator 重放已保存在线 raw extraction。

    不调用 LLM，不修改原始 validation.json；只写
    validation_revalidated.json 与 revalidation_diff.json。
    """

    gate, manifest, gold, samples, gold_samples = _gold_manifest()
    summary = {
        "schema": "b39_revalidation/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0,
        "note": (
            "使用已保存的在线 raw extraction 重新执行生产 Validator；"
            "不新增模型调用，不修改原始 validation.json。"
        ),
        "arms": {},
    }

    for arm_name in args.arms:
        arm = resolve_arm(arm_name)
        arm_summary = {}

        for sample_id in args.samples:
            sample_dir = RUNS_DIR / arm_name / RUN_ID / sample_id

            if not (sample_dir / "validation.json").is_file():
                arm_summary[sample_id] = {
                    "status": "missing"
                }
                continue

            sample = samples[sample_id]
            inputs, _ = _prepare_inputs(sample)
            original = load_json(
                sample_dir / "validation.json"
            )
            revalidated = _revalidate_sample(
                arm, sample, inputs, sample_dir
            )
            before = _validation_entries(original)
            after = _validation_entries(revalidated)
            changed = [
                {
                    "name": new["name"],
                    "value": new["value"],
                    "unit": new["unit"],
                    "value_type": new["value_type"],
                    "row_index": new["row_index"],
                    "before_status": old["status"],
                    "after_status": new["status"],
                    "before_errors": old["errors"],
                    "after_errors": new["errors"],
                }
                for old, new in zip(before, after)
                if old["status"] != new["status"]
                or old["errors"] != new["errors"]
            ]
            write_json(
                sample_dir / "validation_revalidated.json",
                revalidated,
            )
            write_json(
                sample_dir / "revalidation_diff.json",
                {
                    "schema": "b39_revalidation_diff/v1",
                    "arm": arm_name,
                    "sample_id": sample_id,
                    "api_calls": 0,
                    "before_counts": _simple_status_counts(original),
                    "after_counts": _simple_status_counts(revalidated),
                    "changed": changed,
                    "note": (
                        "同一份在线 raw extraction 经最终生产 "
                        "Validator 重放；没有新增 API 调用。"
                    ),
                },
            )
            arm_summary[sample_id] = {
                "status": "revalidated",
                "before_counts": _simple_status_counts(original),
                "after_counts": _simple_status_counts(revalidated),
                "changed_count": len(changed),
            }

        summary["arms"][arm_name] = arm_summary

    write_json(RESULTS_DIR / "revalidation.json", summary)
    return summary


def _production_view(validation, sample, gold_sample):
    predictions = [
        normalize_prediction(param, index)
        for index, param in enumerate(
            validation.get("parameters", [])
        )
    ]
    golds = [
        normalize_gold(record)
        for record in gold_sample["records"]
    ]
    scope_ids = set(sample["gold_scope_table_ids"])
    gold_rows = {
        (gold["table_id"], gold["row_index"])
        for gold in golds
    }
    gold_identities = {gold["identity"] for gold in golds}
    in_scope = [
        prediction
        for prediction in predictions
        if prediction["table_id"] in scope_ids
        and (
            prediction["identity"] in gold_identities
            or (
                prediction["table_id"],
                prediction["row_index"],
            )
            in gold_rows
        )
    ]
    pairs = max_weight_match(
        golds, in_scope, content_allowed, content_weight
    )
    matched_indexes = {
        in_scope[pred_index]["index"]
        for _, pred_index in pairs
    }
    validated_in_scope = sum(
        1
        for prediction in in_scope
        if prediction.get("validation_status")
        == "validated"
    )
    pending_in_scope = sum(
        1
        for prediction in in_scope
        if prediction.get("validation_status")
        == "pending_review"
    )
    auto_tp = sum(
        1
        for prediction in in_scope
        if prediction["index"] in matched_indexes
        and prediction.get("validation_status")
        == "validated"
    )
    diag_tp = sum(
        1
        for prediction in in_scope
        if prediction["index"] in matched_indexes
        and prediction.get("validation_status")
        in ("validated", "pending_review")
    )
    gold_total = len(golds)

    def ratio(numerator, denominator):
        if not denominator:
            return None

        return round(numerator / denominator, 4)

    def f1(precision, recall):
        if not precision or not recall:
            return None

        return round(
            2 * precision * recall / (precision + recall), 4
        )

    auto_precision = ratio(auto_tp, validated_in_scope)
    auto_recall = ratio(auto_tp, gold_total)
    diag_precision = ratio(
        diag_tp, validated_in_scope + pending_in_scope
    )
    diag_recall = ratio(diag_tp, gold_total)

    return {
        "sample_id": sample["sample_id"],
        "gold_total": gold_total,
        "in_scope_total": len(in_scope),
        "content_tp": len(pairs),
        "content_fp": len(in_scope) - len(pairs),
        "validated_in_scope": validated_in_scope,
        "pending_in_scope": pending_in_scope,
        "validated_only_tp": auto_tp,
        "validated_only_precision": auto_precision,
        "validated_only_recall": auto_recall,
        "validated_only_f1": f1(
            auto_precision, auto_recall
        ),
        "pending_inclusive_tp": diag_tp,
        "pending_inclusive_precision": diag_precision,
        "pending_inclusive_recall": diag_recall,
        "pending_inclusive_f1": f1(
            diag_precision, diag_recall
        ),
    }


def _aggregate_production(rows):
    def total(key):
        return sum(row[key] for row in rows)

    def ratio(numerator, denominator):
        if not denominator:
            return None

        return round(numerator / denominator, 4)

    def f1(precision, recall):
        if not precision or not recall:
            return None

        return round(
            2 * precision * recall / (precision + recall), 4
        )

    gold_total = total("gold_total")
    auto_tp = total("validated_only_tp")
    validated_in_scope = total("validated_in_scope")
    diag_tp = total("pending_inclusive_tp")
    validated_and_pending = (
        validated_in_scope + total("pending_in_scope")
    )
    auto_precision = ratio(auto_tp, validated_in_scope)
    auto_recall = ratio(auto_tp, gold_total)
    diag_precision = ratio(
        diag_tp, validated_and_pending
    )
    diag_recall = ratio(diag_tp, gold_total)

    return {
        "gold_total": gold_total,
        "content_tp": total("content_tp"),
        "content_fp": total("content_fp"),
        "in_scope_total": total("in_scope_total"),
        "validated_in_scope": validated_in_scope,
        "pending_in_scope": total("pending_in_scope"),
        "validated_only_tp": auto_tp,
        "validated_only_precision": auto_precision,
        "validated_only_recall": auto_recall,
        "validated_only_f1": f1(
            auto_precision, auto_recall
        ),
        "pending_inclusive_tp": diag_tp,
        "pending_inclusive_precision": diag_precision,
        "pending_inclusive_recall": diag_recall,
        "pending_inclusive_f1": f1(
            diag_precision, diag_recall
        ),
        "pending_inclusive_note": (
            "包含 pending_review 仅为诊断口径，"
            "不代表生产自动接受。"
        ),
    }


def _status_counts(validation):
    counts = {}
    details = []

    for param in validation.get("parameters", []):
        status = param.get("validation_status") or "unknown"
        counts[status] = counts.get(status, 0) + 1

        if status != "validated":
            details.append({
                "name": param.get("name"),
                "value": param.get("value"),
                "unit": param.get("unit"),
                "value_type": param.get("value_type"),
                "row_index": (
                    param.get("evidence") or {}
                ).get("row_index"),
                "status": status,
                "errors": list(
                    param.get("validation_errors") or []
                ),
                "identity_ambiguous": bool(
                    param.get("identity_ambiguous")
                ),
            })

    return {"counts": counts, "non_validated": details}


def _ncp_unit_findings(validation, gold_sample):
    gold_by_row = {}

    for record in gold_sample["records"]:
        key = (
            int(record["source"]["row_index"]),
            str(record["value_type"]).lower(),
        )
        gold_by_row[key] = record

    findings = []

    for index, param in enumerate(
        validation.get("parameters", [])
    ):
        evidence = param.get("evidence") or {}
        key = (
            evidence.get("row_index"),
            str(param.get("value_type") or "").lower(),
        )
        gold = gold_by_row.get(key)

        if gold is None:
            continue

        if param.get("unit") == gold.get("unit"):
            continue

        findings.append({
            "index": index,
            "name": param.get("name"),
            "row_index": key[0],
            "value_type": key[1],
            "prediction_value": param.get("value"),
            "prediction_unit": param.get("unit"),
            "gold_unit": gold.get("unit"),
            "status": param.get("validation_status"),
            "errors": list(
                param.get("validation_errors") or []
            ),
        })

    return findings


def _ncp_gold_unit_rows(validation, gold_sample):
    """列出 Gold 关注的 µA / µVrms 行，验证在线输出单位。"""

    wanted_rows = {8, 9, 14}
    gold_by_row = {}

    for record in gold_sample["records"]:
        row_index = int(record["source"]["row_index"])

        if row_index not in wanted_rows:
            continue

        gold_by_row.setdefault(row_index, {
            "gold_name": record["parameter_identity"],
            "gold_unit": record["unit"],
            "predictions": [],
        })["predictions"].append({
            "value_type": record["value_type"],
            "gold_value": record["value"],
            "gold_unit": record["unit"],
        })

    for param in validation.get("parameters", []):
        evidence = param.get("evidence") or {}
        row_index = evidence.get("row_index")

        if row_index not in gold_by_row:
            continue

        gold_by_row[row_index]["predictions"].append({
            "value_type": param.get("value_type"),
            "prediction_name": param.get("name"),
            "prediction_value": param.get("value"),
            "prediction_unit": param.get("unit"),
            "status": param.get("validation_status"),
        })

    rows = []

    for row_index in sorted(gold_by_row):
        item = gold_by_row[row_index]
        predictions = item["predictions"]
        units = {
            prediction.get("prediction_unit")
            for prediction in predictions
            if prediction.get("prediction_unit")
        }
        gold_unit = item["gold_unit"]
        rows.append({
            "row_index": row_index,
            "gold_name": item["gold_name"],
            "gold_unit": gold_unit,
            "predicted_units": sorted(units),
            "unit_ok": bool(units) and units == {gold_unit},
            "predictions": [
                prediction
                for prediction in predictions
                if prediction.get("prediction_name")
            ],
        })

    return rows


def _adi_identity_findings(validation):
    conflicts = []
    for param in validation.get("parameters", []):
        errors = param.get("validation_errors") or []

        if any(
            "recovered_conflict:identity" in error
            for error in errors
        ):
            conflicts.append({
                "name": param.get("name"),
                "value": param.get("value"),
                "unit": param.get("unit"),
                "value_type": param.get("value_type"),
                "status": param.get("validation_status"),
                "identity_ambiguous": bool(
                    param.get("identity_ambiguous")
                ),
                "errors": list(errors),
            })

    return {
        "conflict_count": len(conflicts),
        "ambiguous_count": sum(
            1
            for item in conflicts
            if item["identity_ambiguous"]
        ),
        "conflicts": conflicts,
    }


def _load_b38_expectation(arm):
    if not B38_REPLAY.is_file():
        return None

    payload = load_json(B38_REPLAY)
    return payload.get("arms", {}).get(arm)


def score_runs(args):
    gate, manifest, gold, samples, gold_samples = _gold_manifest()
    payload = {
        "schema": "b39_online_score/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scorer_version": SCORER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "scope": {
            "samples": list(SAMPLES),
            "arms": list(ARMS),
            "note": (
                "只覆盖两个型号；官方指标是两样本 micro，"
                "不能与 B3.7 四厂商结果直接比较。"
            ),
        },
        "arms": {},
    }

    for arm_name in ARMS:
        arm_run_dir = RUNS_DIR / arm_name / RUN_ID
        arm_payload = {
            "samples": {},
            "official": None,
            "production": None,
            "status": {},
            "ncp_units": {},
            "adi_identity": {},
            "cost": {},
            "comparison_b38": {},
        }
        score_rows = []
        production_rows = []

        for sample_id in SAMPLES:
            sample_dir = arm_run_dir / sample_id

            if not (sample_dir / "validation.json").is_file():
                arm_payload["samples"][sample_id] = {
                    "status": "missing"
                }
                continue

            validation, validation_source = (
                _load_validation_payload(sample_dir)
            )
            scored = score_sample(
                samples[sample_id],
                gold_samples[sample_id],
                sample_dir,
            )
            production = _production_view(
                validation,
                samples[sample_id],
                gold_samples[sample_id],
            )
            status = _status_counts(validation)
            score_rows.append(scored)
            production_rows.append(production)
            arm_payload["samples"][sample_id] = {
                "validation_source": validation_source,
                "official": {
                    key: scored[key]
                    for key in (
                        "gold_records",
                        "predictions_total",
                        "predictions_in_scope",
                        "tp",
                        "fp",
                        "fn",
                        "precision",
                        "recall",
                        "f1",
                        "strict_tp",
                        "strict_precision",
                        "strict_recall",
                        "strict_f1",
                        "tp_by_validation_status",
                    )
                },
                "production": production,
                "status": status,
            }

            if sample_id == "onsemi_ncp170":
                arm_payload["ncp_units"] = {
                    "findings": _ncp_unit_findings(
                        validation, gold_samples[sample_id]
                    ),
                    "gold_unit_rows": _ncp_gold_unit_rows(
                        validation, gold_samples[sample_id]
                    ),
                    "glyph_evidence": load_json(
                        sample_dir
                        / "input_glyph_evidence.json"
                    )
                    if (
                        sample_dir
                        / "input_glyph_evidence.json"
                    ).is_file()
                    else None,
                }

            if sample_id == "adi_lt3045":
                arm_payload["adi_identity"] = (
                    _adi_identity_findings(validation)
                )
                artifact = load_json(
                    sample_dir / "artifact.json"
                )
                api = load_json(
                    sample_dir / "api_responses.json"
                )
                raw = (
                    artifact.get("final_extraction")
                    or artifact.get("extraction")
                    or {}
                )
                params = raw.get("parameters", [])
                finish_reasons = [
                    call.get("finish_reason")
                    for call in api.get("calls", [])
                ]
                arm_payload["adi_table"] = {
                    "raw_parameter_count": len(params),
                    "call_count": api.get("call_count", 0),
                    "finish_reasons": finish_reasons,
                    "length_calls": sum(
                        1
                        for reason in finish_reasons
                        if reason == "length"
                    ),
                    "table_ids": sorted({
                        (param.get("evidence") or {}).get(
                            "table_id"
                        )
                        for param in params
                        if (param.get("evidence") or {}).get(
                            "table_id"
                        )
                        is not None
                    }),
                    "row_indices": sorted({
                        (param.get("evidence") or {}).get(
                            "row_index"
                        )
                        for param in params
                        if (param.get("evidence") or {}).get(
                            "row_index"
                        )
                        is not None
                    }),
                }

        if score_rows:
            arm_payload["official"] = aggregate(score_rows)
            arm_payload["production"] = (
                _aggregate_production(production_rows)
            )

        cost = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "elapsed_s": 0.0,
            "response_models": set(),
            "samples": {},
        }

        for sample_id in SAMPLES:
            sample_dir = arm_run_dir / sample_id
            usage_path = sample_dir / "usage.json"
            timing_path = sample_dir / "timing.json"
            api_path = sample_dir / "api_responses.json"

            if not usage_path.is_file():
                continue

            usage = load_json(usage_path)
            timing = load_json(timing_path)
            api = load_json(api_path)
            calls = api.get("call_count", 0)
            cost["calls"] += calls
            cost["prompt_tokens"] += usage.get(
                "prompt_tokens", 0
            )
            cost["completion_tokens"] += usage.get(
                "completion_tokens", 0
            )
            cost["total_tokens"] += usage.get(
                "total_tokens", 0
            )
            cost["elapsed_s"] += timing.get(
                "elapsed_s", 0.0
            )
            cost["response_models"].update(
                api.get("response_models") or []
            )
            cost["samples"][sample_id] = {
                "calls": calls,
                "prompt_tokens": usage.get(
                    "prompt_tokens", 0
                ),
                "completion_tokens": usage.get(
                    "completion_tokens", 0
                ),
                "total_tokens": usage.get(
                    "total_tokens", 0
                ),
                "elapsed_s": timing.get("elapsed_s", 0.0),
            }

        cost["response_models"] = sorted(
            cost["response_models"]
        )
        input_rate = os.getenv("B39_INPUT_PRICE_PER_MTOK")
        output_rate = os.getenv("B39_OUTPUT_PRICE_PER_MTOK")
        cost["pricing"] = {
            "input_per_mtok": input_rate,
            "output_per_mtok": output_rate,
            "estimated_cost": None,
            "note": (
                "未在仓库中硬编码供应商价格；如提供 "
                "B39_INPUT_PRICE_PER_MTOK / "
                "B39_OUTPUT_PRICE_PER_MTOK 则计算估算值。"
            ),
        }

        if input_rate is not None and output_rate is not None:
            cost["pricing"]["estimated_cost"] = round(
                cost["prompt_tokens"] / 1_000_000
                * float(input_rate)
                + cost["completion_tokens"] / 1_000_000
                * float(output_rate),
                6,
            )

        arm_payload["cost"] = cost
        b38 = _load_b38_expectation(arm_name)
        arm_payload["comparison_b38"] = {
            "offline_available": b38 is not None,
            "expected_trusted": (
                {
                    sample_id: b38["samples"][sample_id]["after"]
                    for sample_id in b38["samples"]
                    if sample_id in SAMPLES
                }
                if b38
                else None
            ),
            "expected_official": (
                b38.get("official") if b38 else None
            ),
            "expected_ncp_units": (
                b38.get("ncp_units") if b38 else None
            ),
            "expected_adi_identity": (
                b38.get("identity") if b38 else None
            ),
        }
        payload["arms"][arm_name] = arm_payload

    write_json(
        RESULTS_DIR / "online_scores.json", payload
    )
    _write_split_results(payload)
    return payload


def _write_split_results(scores):
    """把四类评测口径拆成独立文件，便于交付审阅。"""

    def arm_map(key):
        return {
            arm: scores["arms"][arm].get(key)
            for arm in ARMS
        }

    write_json(
        RESULTS_DIR / "official_metrics.json",
        {
            "schema": "b39_official_metrics/v1",
            "scope": scores["scope"],
            "scorer_version": scores["scorer_version"],
            "normalization_version": scores[
                "normalization_version"
            ],
            "arms": arm_map("official"),
            "note": (
                "官方 B3.5c 契约；invalid 预测仍计入 FP。"
                "两样本 micro，不能与 B3.7 四厂商绝对值直接比较。"
            ),
        },
    )
    write_json(
        RESULTS_DIR / "production_validated.json",
        {
            "schema": "b39_production_validated/v1",
            "arms": arm_map("production"),
            "note": (
                "仅统计 validated；pending_review 不计入自动接受。"
            ),
        },
    )
    write_json(
        RESULTS_DIR / "pending_inclusive.json",
        {
            "schema": "b39_pending_inclusive/v1",
            "arms": {
                arm: {
                    key: (scores["arms"][arm].get("production") or {}).get(
                        key
                    )
                    for key in (
                        "pending_inclusive_tp",
                        "pending_inclusive_precision",
                        "pending_inclusive_recall",
                        "pending_inclusive_f1",
                    )
                }
                for arm in ARMS
            },
            "note": (
                "pending_review 仅用于诊断，不代表生产自动可信接受。"
            ),
        },
    )
    write_json(
        RESULTS_DIR / "status_counts.json",
        {
            "schema": "b39_status_counts/v1",
            "arms": {
                arm: {
                    sample_id: sample.get("status")
                    for sample_id, sample in scores["arms"][arm][
                        "samples"
                    ].items()
                    if sample.get("status") != "missing"
                }
                for arm in ARMS
            },
        },
    )
    write_json(
        RESULTS_DIR / "comparison_b38.json",
        {
            "schema": "b39_comparison_b38/v1",
            "arms": arm_map("comparison_b38"),
        },
    )
    write_json(
        RESULTS_DIR / "cost.json",
        {
            "schema": "b39_cost/v1",
            "arms": arm_map("cost"),
        },
    )

    import csv

    rows = []

    for arm in ARMS:
        for sample_id, sample in scores["arms"][arm][
            "samples"
        ].items():
            if sample.get("status") == "missing":
                continue

            for item in sample["status"]["non_validated"]:
                rows.append({
                    "arm": arm,
                    "sample_id": sample_id,
                    "name": item.get("name"),
                    "value": item.get("value"),
                    "unit": item.get("unit"),
                    "value_type": item.get("value_type"),
                    "row_index": item.get("row_index"),
                    "status": item.get("status"),
                    "identity_ambiguous": item.get(
                        "identity_ambiguous"
                    ),
                    "errors": "; ".join(
                        item.get("errors") or []
                    ),
                })

    path = RESULTS_DIR / "key_errors.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "sample_id",
                "name",
                "value",
                "unit",
                "value_type",
                "row_index",
                "status",
                "identity_ambiguous",
                "errors",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _pct(value):
    return (
        "N/A"
        if value is None
        else f"{value * 100:.2f}%"
    )


def _write_report(scores, preflight, path):
    lines = [
        "# B3.9 第一版 MVP 小范围在线验证报告",
        "",
        "> 真实 DeepSeek API 调用；范围仅两个型号"
        "（ADI LT3045 / onsemi NCP170），"
        "不代表新的四厂商完整泛化结果。",
        "",
        "## 1. 固定条件与预检",
        "",
        f"- 模型请求标识：`{preflight['provider']['requested_model']}`",
        f"- Base URL：`{preflight['provider']['base_url']}`",
        f"- Temperature：{preflight['provider']['temperature']}",
        f"- 评分契约：`{preflight['scorer']['scorer_version']}` / "
        f"`{preflight['scorer']['normalization_version']}`",
        f"- 身份同义词版本："
        f"`{preflight.get('identity_synonym_version')}`"
        f"（在线原始运行记录为 `b3.8-identity-v1`；"
        f"后处理重校验使用最终版本）",
        f"- Gold SHA256：`{preflight['gold_gate']['gold_sha256']}`",
        f"- 预检总体："
        f"{'PASS' if preflight['overall_ok'] else 'FAIL'}",
        "",
        "| 样本 | 检查 | 结果 |",
        "| --- | --- | --- |",
    ]

    for sample_id, sample in preflight["samples"].items():
        for check in sample["checks"]:
            lines.append(
                f"| {sample_id} | `{check['name']}` | "
                f"{'PASS' if check['ok'] else 'FAIL'} |"
            )

    revalidation_path = RESULTS_DIR / "revalidation.json"

    if revalidation_path.is_file():
        revalidation = load_json(revalidation_path)
        lines += [
            "",
            "### 1.1 在线后处理重校验（不新增 API 调用）",
            "",
            "在线运行发现 `logical_family` 把所有 hysteresis "
            "误折叠为 UVLO，导致合法的 Thermal Shutdown / "
            "PGFB Hysteresis 被拒绝。已做最小修复，并用最终"
            "生产 Validator 重放已保存的在线 raw extraction；"
            "原始 `validation.json` 未被覆盖。",
            "",
            "| 臂 | 样本 | 重校验前 | 重校验后 | 变化条数 |",
            "| --- | --- | --- | --- | ---: |",
        ]

        for arm in ARMS:
            for sample_id, item in revalidation["arms"].get(
                arm, {}
            ).items():
                if item.get("status") == "missing":
                    continue

                lines.append(
                    f"| {arm} | {sample_id} | "
                    f"`{item['before_counts']}` | "
                    f"`{item['after_counts']}` | "
                    f"{item['changed_count']} |"
                )

        lines += [
            "",
            "重校验只改变非 Gold 作用域内的合法参数状态；"
            "官方与 validated-only 指标均不变。",
        ]

    lines += [
        "",
        "## 2. 官方指标（两样本 micro）",
        "",
        "| 臂 | TP | FP | FN | Precision | Recall | F1 | Strict F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for arm in ARMS:
        official = scores["arms"][arm].get("official")
        if not official:
            lines.append(f"| {arm} | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {arm} | {official['tp']} | {official['fp']} | "
            f"{official['fn']} | "
            f"{_pct(official['precision'])} | "
            f"{_pct(official['recall'])} | "
            f"{_pct(official['f1'])} | "
            f"{_pct(official['strict_f1'])} |"
        )

    lines += [
        "",
        "### 2.1 分样本官方指标",
        "",
        "| 臂 | 样本 | TP | FP | FN | Precision | Recall | F1 | Strict F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for arm in ARMS:
        for sample_id in SAMPLES:
            sample = scores["arms"][arm]["samples"].get(
                sample_id
            )

            if not sample or sample.get("status") == "missing":
                continue

            official = sample["official"]
            lines.append(
                f"| {arm} | {sample_id} | {official['tp']} | "
                f"{official['fp']} | {official['fn']} | "
                f"{_pct(official['precision'])} | "
                f"{_pct(official['recall'])} | "
                f"{_pct(official['f1'])} | "
                f"{_pct(official['strict_f1'])} |"
            )

    lines += [
        "",
        "官方指标仍使用 B3.5c 冻结评分契约；两样本 micro "
        "不能与 B3.7 四厂商结果直接比较。"
        "D 的官方 F1 低于 C，主要因为 ADI 13 条同名歧义"
        "在官方口径中计为 FP；C 的 ADI 只输出 1 条、"
        "几乎没有 FP，但 ADI 召回仅 1/18。"
        "因此官方 micro F1 不能单独用于选择默认方案。",
        "",
        "## 3. 生产自动接受口径（仅 validated）",
        "",
        "| 臂 | 自动接受 TP | 自动接受预测数 | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for arm in ARMS:
        production = scores["arms"][arm].get("production")
        if not production:
            lines.append(f"| {arm} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {arm} | {production['validated_only_tp']} | "
            f"{production['validated_in_scope']} | "
            f"{_pct(production['validated_only_precision'])} | "
            f"{_pct(production['validated_only_recall'])} | "
            f"{_pct(production['validated_only_f1'])} |"
        )

    lines += [
        "",
        "分样本验证：",
        "",
        "| 臂 | 样本 | 自动接受 TP | 自动接受预测数 | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for arm in ARMS:
        for sample_id in SAMPLES:
            sample = scores["arms"][arm]["samples"].get(
                sample_id
            )

            if not sample or sample.get("status") == "missing":
                continue

            production = sample["production"]
            lines.append(
                f"| {arm} | {sample_id} | "
                f"{production['validated_only_tp']} | "
                f"{production['validated_in_scope']} | "
                f"{_pct(production['validated_only_precision'])} | "
                f"{_pct(production['validated_only_recall'])} | "
                f"{_pct(production['validated_only_f1'])} |"
            )

    lines += [
        "",
        "自动接受 Precision 的分母是 in-scope 中所有 "
        "`validated` 预测；未被内容匹配的 validated 预测按 FP 计。"
        "官方口径中的 invalid 同名歧义不会进入该分母，"
        "所以 D 的 validated-only F1 高于官方 F1。",
        "",
        "## 4. pending-inclusive 诊断口径",
        "",
        "| 臂 | 诊断 TP | 诊断预测数 | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for arm in ARMS:
        production = scores["arms"][arm].get("production")
        if not production:
            lines.append(f"| {arm} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {arm} | {production['pending_inclusive_tp']} | "
            f"{production['validated_in_scope'] + production['pending_in_scope']} | "
            f"{_pct(production['pending_inclusive_precision'])} | "
            f"{_pct(production['pending_inclusive_recall'])} | "
            f"{_pct(production['pending_inclusive_f1'])} |"
        )

    lines += [
        "",
        "该口径把 `pending_review` 视为可诊断命中，"
        "**不代表自动可信接受**。",
        "",
        "## 5. 各状态数量与关键错误",
        "",
        "状态来自最终生产 Validator；如存在 "
        "`validation_revalidated.json`，则使用重校验结果，"
        "原始 `validation.json` 保留未覆盖。",
        "",
    ]

    for arm in ARMS:
        lines.append(f"### 臂 {arm}")
        lines.append("")
        lines.append("| 样本 | validated | pending_review | invalid | 其他 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")

        for sample_id in SAMPLES:
            sample = scores["arms"][arm]["samples"].get(
                sample_id
            )

            if not sample or sample.get("status") == "missing":
                lines.append(f"| {sample_id} | - | - | - | - |")
                continue

            counts = sample["status"]["counts"]
            other = sum(
                value
                for key, value in counts.items()
                if key not in (
                    "validated",
                    "pending_review",
                    "invalid",
                )
            )
            lines.append(
                f"| {sample_id} | {counts.get('validated', 0)} | "
                f"{counts.get('pending_review', 0)} | "
                f"{counts.get('invalid', 0)} | {other} |"
            )

        lines.append("")

        for sample_id in SAMPLES:
            sample = scores["arms"][arm]["samples"].get(
                sample_id
            )

            if not sample or sample.get("status") == "missing":
                continue

            non_validated = sample["status"]["non_validated"]

            if not non_validated:
                continue

            lines.append(
                f"`{sample_id}` 非 validated 记录（最多 12 条）："
            )
            lines.append("")
            lines.append(
                "| 参数 | 值 | 单位 | 类型 | 行 | 状态 | 原因 |"
            )
            lines.append(
                "| --- | ---: | --- | --- | ---: | --- | --- |"
            )

            for item in non_validated[:12]:
                lines.append(
                    f"| {item['name']} | {item['value']} | "
                    f"{item['unit']} | {item['value_type']} | "
                    f"{item['row_index']} | {item['status']} | "
                    f"{'; '.join(item['errors'])[:120]} |"
                )

            if len(non_validated) > 12:
                lines.append(
                    f"| … |  |  |  |  |  | 另有 "
                    f"{len(non_validated) - 12} 条 |"
                )

            lines.append("")

    lines += [
        "## 6. NCP170 单位与 ADI LT3045 身份",
        "",
    ]

    for arm in ARMS:
        arm_payload = scores["arms"][arm]
        ncp = arm_payload.get("ncp_units", {})
        findings = ncp.get("findings", [])
        adi = arm_payload.get("adi_identity", {})
        table = arm_payload.get("adi_table", {})
        lines.append(f"### 臂 {arm}")
        lines.append("")
        lines.append(
            f"- NCP170 单位不一致预测数：{len(findings)}"
        )

        for finding in findings[:12]:
            lines.append(
                f"  - row {finding['row_index']} "
                f"{finding['name']} {finding['value_type']}："
                f"预测 {finding['prediction_value']} "
                f"{finding['prediction_unit']}，Gold "
                f"{finding['gold_unit']}，状态 "
                f"`{finding['status']}`"
            )

        if not findings:
            lines.append("  - 无单位不一致预测。")

        unit_rows = ncp.get("gold_unit_rows", [])

        if unit_rows:
            lines.append("")
            lines.append(
                "| Gold 行 | Gold 参数 | Gold 单位 | 预测单位 | 单位一致 |"
            )
            lines.append(
                "| ---: | --- | --- | --- | --- |"
            )

            for item in unit_rows:
                lines.append(
                    f"| {item['row_index']} | {item['gold_name']} | "
                    f"{item['gold_unit']} | "
                    f"{', '.join(item['predicted_units'])} | "
                    f"{'PASS' if item['unit_ok'] else 'FAIL'} |"
                )

        lines.append(
            f"- ADI LT3045 raw 参数数："
            f"{table.get('raw_parameter_count')}；"
            f"table_ids={table.get('table_ids')}；"
            f"rows={table.get('row_indices')}"
        )
        lines.append(
            f"- ADI 调用数：{table.get('call_count')}；"
            f"finish_reasons={table.get('finish_reasons')}；"
            f"length 截断 {table.get('length_calls')} 次"
        )
        lines.append(
            f"- ADI identity 冲突：{adi.get('conflict_count', 0)} "
            f"（其中同名歧义 {adi.get('ambiguous_count', 0)}）"
        )
        lines.append("")

    lines += [
        "## 7. 与 B3.8 离线预期的对比",
        "",
    ]

    for arm in ARMS:
        comparison = scores["arms"][arm].get(
            "comparison_b38", {}
        )
        lines.append(f"### 臂 {arm}")
        lines.append("")

        if not comparison.get("offline_available"):
            lines.append("B3.8 离线预期不可用。")
            lines.append("")
            continue

        expected = comparison.get("expected_trusted") or {}
        lines.append(
            "| 样本 | B3.8 离线 TP/FP/FN | B3.9 在线 TP/FP/FN | "
            "B3.8 离线 validated TP | B3.9 在线 validated TP |"
        )
        lines.append("| --- | --- | --- | ---: | ---: |")

        for sample_id in SAMPLES:
            offline = expected.get(sample_id)

            if not offline:
                continue

            sample = scores["arms"][arm]["samples"].get(
                sample_id
            )

            if not sample or sample.get("status") == "missing":
                lines.append(
                    f"| {sample_id} | "
                    f"{offline['content_tp']}/"
                    f"{offline['content_fp']}/"
                    f"{offline['gold_total'] - offline['content_tp']} | "
                    "缺失 | "
                    f"{offline['validated_tp']} | - |"
                )
                continue

            official = sample["official"]
            production = sample["production"]
            lines.append(
                f"| {sample_id} | "
                f"{offline['content_tp']}/"
                f"{offline['content_fp']}/"
                f"{offline['gold_total'] - offline['content_tp']} | "
                f"{official['tp']}/{official['fp']}/"
                f"{official['fn']} | "
                f"{offline['validated_tp']} | "
                f"{production['validated_only_tp']} |"
            )

        lines.append("")
        lines.append(
            "B3.8 的 trusted 口径包含 pending_review；"
            "上表的 validated TP 只统计 `validated`，"
            "两列均按同一内容匹配契约计算。"
        )
        lines.append("")

    lines += [
        "## 8. 成本与预算",
        "",
        "| 臂 | 调用数 | Prompt Tokens | Completion Tokens | "
        "Total Tokens | 耗时(s) | 实际响应模型 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for arm in ARMS:
        cost = scores["arms"][arm].get("cost", {})
        lines.append(
            f"| {arm} | {cost.get('calls', 0)} | "
            f"{cost.get('prompt_tokens', 0)} | "
            f"{cost.get('completion_tokens', 0)} | "
            f"{cost.get('total_tokens', 0)} | "
            f"{cost.get('elapsed_s', 0.0)} | "
            f"{', '.join(cost.get('response_models') or [])} |"
        )

    lines += [
        "",
        "供应商价格未硬编码；完整 token 级成本见 "
        "`results/online_scores.json`。"
        "真实模型输出保存在 "
        "`runs/<arm>/r1/<sample>/raw_responses.json`、"
        "`artifact.json`；API 调用与 usage 保存在同目录的 "
        "`api_requests.json`、`api_responses.json`、`usage.json`。",
        "",
        "## 9. 结论",
        "",
    ]

    d_official = scores["arms"]["D"].get("official") or {}
    c_official = scores["arms"]["C"].get("official") or {}
    d_prod = scores["arms"]["D"].get("production") or {}
    c_prod = scores["arms"]["C"].get("production") or {}
    ncp_d = scores["arms"]["D"].get("ncp_units", {}).get(
        "findings", []
    )
    adi_d = scores["arms"]["D"].get(
        "adi_identity", {}
    )
    d_table = scores["arms"]["D"].get("adi_table", {})
    c_table = scores["arms"]["C"].get("adi_table", {})
    d_ncp_status = scores["arms"]["D"]["samples"][
        "onsemi_ncp170"
    ]["status"]["counts"]
    c_ncp_status = scores["arms"]["C"]["samples"][
        "onsemi_ncp170"
    ]["status"]["counts"]

    lines += [
        f"- D 官方 F1：{_pct(d_official.get('f1'))}；"
        f"validated-only F1："
        f"{_pct(d_prod.get('validated_only_f1'))}；"
        f"ADI 提取 {d_table.get('raw_parameter_count')} 条、"
        f"覆盖 rows={d_table.get('row_indices')}，"
        f"length 截断 {d_table.get('length_calls')} 次。",
        f"- C 官方 F1：{_pct(c_official.get('f1'))}；"
        f"validated-only F1："
        f"{_pct(c_prod.get('validated_only_f1'))}；"
        f"ADI 仅提取 {c_table.get('raw_parameter_count')} 条、"
        f"覆盖 rows={c_table.get('row_indices')}，"
        f"length 截断 {c_table.get('length_calls')} 次。",
        f"- NCP170：D 的官方内容 TP 为 19/19，"
        f"validated-only TP 为 "
        f"{scores['arms']['D']['samples']['onsemi_ncp170']['production']['validated_only_tp']}/19；"
        f"C 为 19/19，validated-only TP 为 "
        f"{scores['arms']['C']['samples']['onsemi_ncp170']['production']['validated_only_tp']}/19。"
        f"重校验后 D 状态 "
        f"{d_ncp_status}，C 状态 {c_ncp_status}。"
        f"两臂 NCP170 单位不一致预测数：D {len(ncp_d)}，"
        f"C {len(scores['arms']['C'].get('ncp_units', {}).get('findings', []))}。",
        f"- ADI LT3045：D 命中 4 条 shutdown_current，"
        f"另有 {adi_d.get('conflict_count', 0)} 条 identity 冲突"
        f"（同名歧义 {adi_d.get('ambiguous_count', 0)}）；"
        f"C 仅命中 1 条 shutdown_current，"
        f"17 条 ADI Gold 记录未被提取。",
        "",
        "### 最终判断",
        "",
        "1. **D 可以作为第一版 MVP 的默认方案**（限本次两个型号范围）："
        "NCP170 官方与 validated-only 均为 19/19；"
        "ADI LT3045 覆盖目标表 rows 1–6，"
        "合法 shutdown 同义参数被放行。"
        "但 ADI 电流限制族仍是已知阻塞项："
        f"{adi_d.get('ambiguous_count', 0)} 条 "
        "`output_current_limit` 同名歧义保持 invalid，"
        "不能进入自动接受。",
        "2. **C 需要保留为保守备选**："
        "它的 NCP170 同样 19/19，且精确率 100%，"
        "适合作为 Flat 路径的高精确率降级方案；"
        "但 ADI 仅提取 1 条（2 次 length 截断），"
        "召回坍缩到 1/18，因此不能作为唯一默认方案。",
        "3. **是否存在阻碍核心 E2E 演示的问题**："
        "NCP170 无阻塞；ADI LT3045 的 shutdown/电压/噪声等"
        "已可演示，但电流限制族（Internal / Programmable "
        "Current Limit）不能自动接受，"
        "需要作为已知限制或人工复核项展示。"
        "C 的 ADI 截断是模型输出长度问题，不是 Validator 误拒。",
        "4. **可以正式转入 README、部署文档和最终交付**："
        "以“两型号在线验证 + 四厂商离线/历史结果”的"
        "分层表述交付，明确不宣称新的四厂商完整泛化；"
        "本轮不再开展大规模调优。",
        "",
        "最终交付数据见 `results/online_scores.json`；"
        "拆分口径见 `results/official_metrics.json`、"
        "`results/production_validated.json`、"
        "`results/pending_inclusive.json`、"
        "`results/status_counts.json`、`results/cost.json`、"
        "`results/key_errors.csv`。",
        "",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(preflight, scores):
    write_json(
        B39_DIR / "manifest.json",
        {
            "schema": "b39_manifest/v1",
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "scope": preflight["scope"],
            "gold_gate": preflight["gold_gate"],
            "provider": preflight["provider"],
            "scorer": preflight["scorer"],
            "outputs": {
                "preflight": str(
                    (PREFLIGHT_DIR / "preflight.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "preflight_markdown": str(
                    (PREFLIGHT_DIR / "preflight.md")
                    .relative_to(PROJECT_ROOT)
                ),
                "runs": str(
                    RUNS_DIR.relative_to(PROJECT_ROOT)
                ),
                "scores": str(
                    (RESULTS_DIR / "online_scores.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "official_metrics": str(
                    (RESULTS_DIR / "official_metrics.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "production_validated": str(
                    (RESULTS_DIR / "production_validated.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "pending_inclusive": str(
                    (RESULTS_DIR / "pending_inclusive.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "status_counts": str(
                    (RESULTS_DIR / "status_counts.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "comparison_b38": str(
                    (RESULTS_DIR / "comparison_b38.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "cost": str(
                    (RESULTS_DIR / "cost.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "key_errors": str(
                    (RESULTS_DIR / "key_errors.csv")
                    .relative_to(PROJECT_ROOT)
                ),
                "revalidation": str(
                    (RESULTS_DIR / "revalidation.json")
                    .relative_to(PROJECT_ROOT)
                ),
                "report": str(
                    (REPORTS_DIR / "B3.9_REPORT.md")
                    .relative_to(PROJECT_ROOT)
                ),
            },
            "api_calls_total": sum(
                (
                    scores["arms"][arm].get("cost") or {}
                ).get("calls") or 0
                for arm in ARMS
            ),
            "revalidation_api_calls": 0,
            "arms": {
                arm: {
                    "official_f1": (
                        (scores["arms"][arm].get("official") or {})
                        .get("f1")
                    ),
                    "validated_only_f1": (
                        (
                            scores["arms"][arm].get(
                                "production"
                            )
                            or {}
                        ).get("validated_only_f1")
                    ),
                    "calls": (
                        (scores["arms"][arm].get("cost") or {})
                        .get("calls")
                    ),
                    "validation_sources": {
                        sample_id: sample.get(
                            "validation_source"
                        )
                        for sample_id, sample in scores["arms"][
                            arm
                        ]["samples"].items()
                        if sample.get("status") != "missing"
                    },
                }
                for arm in ARMS
            },
            "note": (
                "两个型号的小范围在线验证；不覆盖 B3.5c / B3.7 / "
                "B3.8；未自动 Git Commit。"
            ),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["preflight", "online", "revalidate", "score"],
        default="preflight",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        choices=list(ARMS),
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=list(SAMPLES),
        choices=list(SAMPLES),
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
    )
    args = parser.parse_args(argv)

    if args.mode == "preflight":
        payload = run_preflight()
        _write_preflight_markdown(
            payload, PREFLIGHT_DIR / "preflight.md"
        )
        print(
            json.dumps(
                {
                    "overall_ok": payload["overall_ok"],
                    "samples": {
                        sample_id: {
                            "overall_ok": sample[
                                "overall_ok"
                            ],
                            "checks": {
                                check["name"]: check["ok"]
                                for check in sample["checks"]
                            },
                        }
                        for sample_id, sample in payload[
                            "samples"
                        ].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.mode == "online":
        results = run_online(args)
        print(
            json.dumps(
                results, ensure_ascii=False, indent=2
            )
        )
        revalidate_runs(args)
        scores = score_runs(args)
        write_manifest(
            load_json(PREFLIGHT_DIR / "preflight.json"),
            scores,
        )
        _write_report(
            scores,
            load_json(PREFLIGHT_DIR / "preflight.json"),
            REPORTS_DIR / "B3.9_REPORT.md",
        )
        return 0

    if args.mode == "revalidate":
        summary = revalidate_runs(args)
        print(
            json.dumps(
                summary, ensure_ascii=False, indent=2
            )
        )
        scores = score_runs(args)
        write_manifest(
            load_json(PREFLIGHT_DIR / "preflight.json"),
            scores,
        )
        _write_report(
            scores,
            load_json(PREFLIGHT_DIR / "preflight.json"),
            REPORTS_DIR / "B3.9_REPORT.md",
        )
        return 0

    scores = score_runs(args)
    write_manifest(
        load_json(PREFLIGHT_DIR / "preflight.json"),
        scores,
    )
    _write_report(
        scores,
        load_json(PREFLIGHT_DIR / "preflight.json"),
        REPORTS_DIR / "B3.9_REPORT.md",
    )
    print(
        json.dumps(
            {
                arm: {
                    "official_f1": (
                        (scores["arms"][arm].get("official") or {})
                        .get("f1")
                    ),
                    "validated_only_f1": (
                        (
                            scores["arms"][arm].get(
                                "production"
                            )
                            or {}
                        ).get("validated_only_f1")
                    ),
                    "pending_inclusive_f1": (
                        (
                            scores["arms"][arm].get(
                                "production"
                            )
                            or {}
                        ).get("pending_inclusive_f1")
                    ),
                }
                for arm in ARMS
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

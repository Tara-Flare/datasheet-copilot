"""B3.7：四臂受控 A/B 运行器（真实 API）。

臂：
- A: Flat + P0
- B: Logical + P0
- C: Flat + P1
- D: Logical + P1

只替换输入结构与 Prompt 族覆盖开关。模型、temperature、表格范围、
分块、重试与评分契约保持一致。预检与正式运行都写入独立目录，
拒绝覆盖历史产物。
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.agents.graph import build_graph  # noqa: E402
from app.extraction.artifact_store import (  # noqa: E402
    RecordingExtractor,
    artifact_from_state,
    file_sha256,
    save_artifact,
)
from app.extraction.deepseek_extractor import DeepSeekExtractor  # noqa: E402
from app.extraction.evidence import verify_evidence  # noqa: E402
from app.parsers.pdf_parser import PDFParser  # noqa: E402
from app.validation.recovered_integration import (  # noqa: E402
    build_recovered_validation_result,
)
from app.validation.validator import ParameterValidator  # noqa: E402

from b35b_docling_env import configure_environment  # noqa: E402
from b35c_common import validate_gold_gate  # noqa: E402
from b37_common import (  # noqa: E402
    B37_DIR,
    B37Error,
    FORMAL_MAX_RETRIES,
    GOLD,
    GOLD_RECEIPT,
    MANIFEST,
    PREFLIGHT_DIR,
    PREFLIGHT_MAX_RETRIES,
    RUNS_DIR,
    TABLE_SCOPE,
    build_sample_inputs,
    expected_chunks,
    experiment_config,
    load_json,
    prompt_flags,
    resolve_arm,
    sha256_text,
    write_json,
)


EXTRACTED_EVIDENCE_FIELDS = (
    "table_id",
    "row_index",
    "page",
    "source_text",
)

UNTRUSTED_SOURCE_STATUSES = {
    "mapped_source_conflict",
    "unknown_logical_row_id",
    "ambiguous_row_index",
    "missing_source",
}


class RunError(RuntimeError):
    """单样本运行前置条件失败。"""


class StaticTableParser:
    """把已冻结的表格直接交给 LangGraph parse 节点。"""

    def __init__(self, tables):
        self.tables = deepcopy(tables)

    def parse(self, path):  # noqa: ARG002
        return {
            "table_count": len(self.tables),
            "tables": deepcopy(self.tables),
        }


def configure_logging(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    return logging.getLogger("b37")


def _table_ids_from_payload(payload_text):
    try:
        payload = json.loads(payload_text)
    except (TypeError, ValueError):
        return []

    tables = payload.get("tables")

    if not isinstance(tables, list):
        return []

    return [
        table.get("table_id")
        for table in tables
        if isinstance(table, dict)
    ]


def _table_modes_from_payload(payload_text):
    try:
        payload = json.loads(payload_text)
    except (TypeError, ValueError):
        return []

    tables = payload.get("tables")

    if not isinstance(tables, list):
        return []

    return [
        table.get("input_mode", "flat")
        for table in tables
        if isinstance(table, dict)
    ]


def install_api_recorder(extractor, *, temperature):
    """包装 client.create，记录请求配置与响应元数据。

    只做实验观测，不改变返回对象。temperature 由实验层显式注入，
    并禁止 OpenAI SDK 的隐藏传输层重试，保证 calls 可审计。
    """

    extractor.client.max_retries = 0
    requests = []
    responses = []
    original = extractor.client.chat.completions.create

    def wrapped(*args, **kwargs):
        request_kwargs = dict(kwargs)
        injected_temperature = "temperature" not in request_kwargs

        if injected_temperature:
            request_kwargs["temperature"] = temperature

        messages = request_kwargs.get("messages") or []
        system_prompt = ""
        user_payload = ""

        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""

            if role == "system":
                system_prompt = str(content)
            elif role == "user":
                user_payload = str(content)

        flags = prompt_flags(system_prompt)
        request_record = {
            "index": len(requests),
            "requested_model": request_kwargs.get(
                "model", extractor.model
            ),
            "temperature": request_kwargs.get("temperature"),
            "temperature_injected": injected_temperature,
            "max_tokens": request_kwargs.get("max_tokens"),
            "response_format": request_kwargs.get("response_format"),
            "system_prompt": system_prompt,
            "system_prompt_sha256": sha256_text(system_prompt),
            "system_prompt_chars": len(system_prompt),
            "prompt_flags": flags,
            "user_payload_sha256": sha256_text(user_payload),
            "user_payload_chars": len(user_payload),
            "table_ids": _table_ids_from_payload(user_payload),
            "table_input_modes": _table_modes_from_payload(user_payload),
        }
        request_record["request_sha256"] = sha256_text(
            json.dumps(
                {
                    "model": request_record["requested_model"],
                    "temperature": request_record["temperature"],
                    "max_tokens": request_record["max_tokens"],
                    "messages": messages,
                    "response_format": request_record[
                        "response_format"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        requests.append(request_record)

        started = time.perf_counter()

        try:
            response = original(*args, **request_kwargs)
        except Exception as exc:  # noqa: BLE001
            responses.append({
                "index": len(responses),
                "request_index": request_record["index"],
                "requested_model": request_record["requested_model"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": round(
                    time.perf_counter() - started, 3
                ),
            })
            raise

        usage = (
            response.usage.model_dump()
            if response.usage
            else None
        )
        finish_reason = (
            response.choices[0].finish_reason
            if response.choices
            else None
        )
        content = (
            response.choices[0].message.content
            if response.choices
            else None
        ) or ""
        reasoning_tokens = None

        if usage:
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens")

        response_record = {
            "index": len(responses),
            "request_index": request_record["index"],
            "requested_model": request_record["requested_model"],
            "response_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "system_fingerprint": getattr(
                response, "system_fingerprint", None
            ),
            "finish_reason": finish_reason,
            "content_chars": len(content),
            "content_sha256": sha256_text(content),
            "content_preview": content[:200],
            "reasoning_tokens": reasoning_tokens,
            "usage": usage,
            "elapsed_s": round(
                time.perf_counter() - started, 3
            ),
        }
        responses.append(response_record)
        return response

    extractor.client.chat.completions.create = wrapped
    return requests, responses


def _json_or_none(path):
    if not Path(path).is_file():
        return None

    return load_json(path)


def _summarize_retry(artifact):
    initial = artifact.get("extraction", {}).get("parameters", [])
    final = (
        artifact.get("final_extraction")
        or artifact.get("extraction")
        or {}
    ).get("parameters", [])
    retry_results = artifact.get("retry_results", [])

    return {
        "initial_parameter_count": len(initial),
        "final_parameter_count": len(final),
        "retry_result_batches": len(retry_results),
        "retry_candidate_count": sum(
            len(result.get("parameters", []))
            for result in retry_results
        ),
    }


def _save_tables(tables, output_dir):
    payload = {
        "table_count": len(tables),
        "tables": tables,
        "table_summaries": [
            {
                "table_id": table.get("table_id"),
                "page_numbers": table.get("page_numbers", []),
                "columns": table.get("columns", []),
                "row_count": len(table.get("rows", [])),
                "column_count": table.get(
                    "column_count",
                    len(table.get("columns", [])),
                ),
                "input_mode": table.get("input_mode", "flat"),
            }
            for table in tables
        ],
    }
    write_json(output_dir / "selected_tables.json", payload)


def _save_input_tables(tables, output_dir):
    payload = {
        "table_count": len(tables),
        "tables": tables,
        "table_summaries": [
            {
                "table_id": table.get("table_id"),
                "page_numbers": table.get("page_numbers", []),
                "columns": table.get("columns", []),
                "row_count": len(table.get("rows", [])),
                "column_count": table.get(
                    "column_count",
                    len(table.get("columns", [])),
                ),
                "input_mode": table.get("input_mode", "flat"),
                "excluded_rows": len(
                    table.get("excluded_rows", [])
                ),
            }
            for table in tables
        ],
    }
    write_json(output_dir / "extraction_input_tables.json", payload)


def _first_pass_validation(artifact, tables, arm, sample_inputs):
    raw = artifact.get("extraction") or {}
    legacy = ParameterValidator().validate(
        verify_evidence(raw, tables)
    )

    if arm["input_mode"] != "logical":
        return {
            "source": "recomputed_legacy_first_pass",
            "note": (
                "对 recorder.extract_result 重新执行 "
                "verify_evidence + ParameterValidator；不调用 LLM。"
            ),
            "validation_mode": "legacy",
            "validation": legacy,
        }

    config = {
        "logical_rows": sample_inputs["logical_rows"],
        "raw_structure": sample_inputs["raw_structure"],
        "pdf_sha256": sample_inputs["pdf_sha256"],
    }
    recovered = build_recovered_validation_result(
        raw, legacy, config
    )

    return {
        "source": "recomputed_recovered_first_pass",
        "note": (
            "对 recorder.extract_result 重新执行 verify_evidence + "
            "legacy ParameterValidator，再执行生产 "
            "build_recovered_validation_result；不调用 LLM。"
        ),
        "validation_mode": "recovered",
        "validation": recovered,
    }


def run_one(
    arm,
    sample,
    sample_inputs,
    output_dir,
    *,
    max_retries,
    temperature,
):
    output_dir = Path(output_dir)

    if output_dir.exists():
        raise RunError(f"样本目录已存在: {output_dir}")

    pdf_path = PROJECT_ROOT / sample["file"]
    pdf_sha = file_sha256(pdf_path)

    if pdf_sha != sample["pdf_sha256"]:
        raise RunError(
            f"{sample['sample_id']}: PDF SHA256 与 Gold 不一致"
        )

    output_dir.mkdir(parents=True)

    tables = [deepcopy(sample_inputs["flat_table"])]
    table_parser = StaticTableParser(tables)
    pdf_parser = PDFParser()
    extractor = DeepSeekExtractor(
        input_mode=arm["input_mode"],
        family_coverage=arm["family_coverage"],
    )
    request_records, response_records = install_api_recorder(
        extractor, temperature=temperature
    )
    recording_extractor = RecordingExtractor(extractor)

    logical_extraction = None
    recovered_evidence = None

    if arm["input_mode"] == "logical":
        logical_extraction = {
            "logical_rows": sample_inputs["logical_rows"],
            "raw_structure": sample_inputs["raw_structure"],
            "table_ids": sample_inputs["gold_scope_table_ids"],
            "include_needs_review": False,
        }
        recovered_evidence = {
            "logical_rows": sample_inputs["logical_rows"],
            "raw_structure": sample_inputs["raw_structure"],
            "pdf_sha256": sample_inputs["pdf_sha256"],
        }

    graph = build_graph(
        pdf_parser=pdf_parser,
        table_parser=table_parser,
        extractor=recording_extractor,
        evidence_validation_mode=arm[
            "evidence_validation_mode"
        ],
        recovered_evidence=recovered_evidence,
        extraction_input_mode=arm["input_mode"],
        logical_extraction=logical_extraction,
    )

    task_id = str(uuid4())
    initial_state = {
        "document_path": str(pdf_path),
        "status": "pending",
        "retry_count": 0,
        "max_retries": max_retries,
        "enable_human_review": False,
        "errors": [],
    }

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    error = None
    result = None

    try:
        result = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": task_id}},
        )
    except Exception as exc:  # noqa: BLE001
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "task_id": task_id,
        }

    elapsed_s = round(time.perf_counter() - started, 3)
    response_models = sorted({
        item.get("response_model")
        for item in response_records
        if item.get("response_model")
    })
    fingerprints = sorted({
        item.get("system_fingerprint")
        for item in response_records
        if item.get("system_fingerprint")
    })
    usage = extractor.get_usage()

    write_json(
        output_dir / "timing.json",
        {
            "task_id": task_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed_s,
        },
    )
    write_json(output_dir / "usage.json", usage)
    write_json(
        output_dir / "api_requests.json",
        {
            "configured_model": extractor.model,
            "base_url": os.getenv("LLM_BASE_URL"),
            "temperature": temperature,
            "max_tokens": extractor.max_tokens,
            "transport_max_retries": 0,
            "call_count": len(request_records),
            "requests": request_records,
        },
    )
    write_json(
        output_dir / "api_responses.json",
        {
            "configured_model": extractor.model,
            "base_url": os.getenv("LLM_BASE_URL"),
            "max_tokens": extractor.max_tokens,
            "rows_per_chunk": extractor.rows_per_chunk,
            "max_split_depth": extractor.max_split_depth,
            "temperature": temperature,
            "call_count": len(response_records),
            "response_models": response_models,
            "system_fingerprints": fingerprints,
            "calls": response_records,
        },
    )
    write_json(
        output_dir / "raw_responses.json",
        {
            "model": extractor.model,
            "max_tokens": extractor.max_tokens,
            "rows_per_chunk": extractor.rows_per_chunk,
            "max_split_depth": extractor.max_split_depth,
            "temperature": temperature,
            "raw_responses": extractor.raw_responses,
        },
    )
    write_json(
        output_dir / "input_source.json",
        {
            "sample_id": sample["sample_id"],
            "table_scope": TABLE_SCOPE,
            "gold_scope_table_ids": sample_inputs[
                "gold_scope_table_ids"
            ],
            "input_sources": sample_inputs["input_sources"],
            "flat_table_id": sample_inputs["flat_table"].get(
                "table_id"
            ),
            "flat_row_count": len(
                sample_inputs["flat_table"].get("rows", [])
            ),
            "logical_row_count": len(
                (sample_inputs.get("logical_rows") or {}).get(
                    "logical_rows", []
                )
            ),
        },
    )
    write_json(
        output_dir / "arm.json",
        {
            "arm": arm["arm"],
            "label": arm["label"],
            "input_mode": arm["input_mode"],
            "prompt_family": arm["prompt_family"],
            "family_coverage": arm["family_coverage"],
            "evidence_validation_mode": arm[
                "evidence_validation_mode"
            ],
            "max_retries": max_retries,
            "temperature": temperature,
            "table_scope": TABLE_SCOPE,
        },
    )

    if error is not None:
        write_json(output_dir / "errors.json", error)
        write_json(
            output_dir / "state_summary.json",
            {
                "status": "failed",
                "error": error,
                "extraction_input_mode": None,
                "extraction_input_summary": None,
                "validation_summary": None,
            },
        )
        return error

    write_json(output_dir / "errors.json", {"error": None})
    result_tables = result.get("tables", [])
    _save_tables(result_tables, output_dir)
    _save_input_tables(
        result.get("extraction_input_tables") or [],
        output_dir,
    )
    write_json(
        output_dir / "selection.json",
        {
            "table_scope": TABLE_SCOPE,
            "gold_scope_table_ids": sample_inputs[
                "gold_scope_table_ids"
            ],
            "actual_parsed_table_ids": [
                table.get("table_id") for table in result_tables
            ],
            "extraction_input_summary": result.get(
                "extraction_input_summary"
            ),
        },
    )

    artifact = artifact_from_state(
        result,
        recording_extractor,
        pdf_path=pdf_path,
        pdf_sha256=pdf_sha,
        table_cache_dir=None,
        table_cache_key=None,
        model=extractor.model,
    )
    save_artifact(artifact, output_dir / "artifact.json")
    write_json(
        output_dir / "retry_summary.json",
        _summarize_retry(artifact),
    )

    first_pass = _first_pass_validation(
        artifact, result_tables, arm, sample_inputs
    )
    write_json(
        output_dir / "first_pass_validation.json",
        first_pass,
    )

    validation = result.get("validation_result", {})
    write_json(output_dir / "validation.json", validation)
    write_json(
        output_dir / "state_summary.json",
        {
            "status": result.get("status"),
            "retry_count": result.get("retry_count"),
            "evidence_validation_mode": result.get(
                "evidence_validation_mode"
            ),
            "extraction_input_mode": result.get(
                "extraction_input_mode"
            ),
            "extraction_input_summary": result.get(
                "extraction_input_summary"
            ),
            "document_info": result.get("document_info"),
            "selected_table_ids": [
                table.get("table_id")
                for table in result_tables
            ],
            "validation_summary": validation.get(
                "validation_summary"
            ),
            "validation_errors": result.get("errors", []),
        },
    )

    return None


def _check(ok, detail, *, skipped=False):
    return {
        "ok": bool(ok),
        "skipped": bool(skipped),
        "detail": detail,
    }


def evaluate_preflight(run_dir, sample, arm, sample_inputs):
    """逐项验证 B3.7 预检的 5 个条件。"""

    api_requests = load_json(run_dir / "api_requests.json")
    api_responses = load_json(run_dir / "api_responses.json")
    artifact = _json_or_none(run_dir / "artifact.json")
    validation = _json_or_none(run_dir / "validation.json")
    state = _json_or_none(run_dir / "state_summary.json")
    input_tables = _json_or_none(
        run_dir / "extraction_input_tables.json"
    )

    requests = api_requests.get("requests", [])
    calls = api_responses.get("calls", [])
    expected = expected_chunks(
        sample_inputs["flat_table"],
        logical_rows=(
            sample_inputs["logical_rows"]
            if arm["input_mode"] == "logical"
            else None
        ),
        table_ids=sample_inputs["gold_scope_table_ids"],
    )
    expected_logical = arm["input_mode"] == "logical"
    table_id = sample_inputs["gold_scope_table_ids"][0]

    logical_requests = [
        request
        for request in requests
        if table_id in (request.get("table_ids") or [])
        and "logical" in (request.get("table_input_modes") or [])
    ]
    logical_in_model = _check(
        (
            bool(logical_requests)
            and state.get("extraction_input_mode") == "logical"
            and table_id
            in (
                (state.get("extraction_input_summary") or {}).get(
                    "logical_table_ids"
                )
                or []
            )
        )
        if expected_logical
        else (
            state.get("extraction_input_mode") == "flat"
            and not any(
                "logical" in (
                    request.get("table_input_modes") or []
                )
                for request in requests
            )
        ),
        {
            "expected_input_mode": arm["input_mode"],
            "effective_input_mode": state.get(
                "extraction_input_mode"
            ),
            "logical_table_ids": (
                (state.get("extraction_input_summary") or {}).get(
                    "logical_table_ids"
                )
            ),
            "request_table_ids": [
                request.get("table_ids") for request in requests
            ],
            "request_table_modes": [
                request.get("table_input_modes")
                for request in requests
            ],
        },
    )

    prompt_expected = bool(arm["family_coverage"])
    prompt_flags_ok = all(
        bool(request.get("prompt_flags", {}).get(
            "family_coverage"
        )) == prompt_expected
        for request in requests
    )
    expanded_prompt = _check(
        bool(requests) and prompt_flags_ok,
        {
            "prompt_family": arm["prompt_family"],
            "expected_family_coverage": prompt_expected,
            "request_family_coverage": [
                request.get("prompt_flags", {}).get(
                    "family_coverage"
                )
                for request in requests
            ],
            "system_prompt_chars": [
                request.get("system_prompt_chars")
                for request in requests
            ],
        },
    )

    raw_params = (
        (artifact or {}).get("extraction") or {}
    ).get("parameters", [])
    final_params = (
        (artifact or {}).get("final_extraction")
        or (artifact or {}).get("extraction")
        or {}
    ).get("parameters", [])
    validator_params = (validation or {}).get("parameters", [])
    artifact_table_ids = {
        (param.get("evidence") or {}).get("table_id")
        for param in raw_params
    }
    adi_ok = True
    adi_skipped = True
    adi_detail = {"skipped": True, "reason": "非 ADI 样本"}

    if sample["sample_id"] == "adi_lt3045":
        adi_ok = table_id in artifact_table_ids and bool(raw_params)
        adi_skipped = False
        adi_detail = {
            "skipped": False,
            "gold_scope_table_id": table_id,
            "artifact_table_ids": sorted(
                value for value in artifact_table_ids
                if value is not None
            ),
            "raw_parameter_count": len(raw_params),
        }

    adi_table4 = _check(adi_ok, adi_detail, skipped=adi_skipped)

    evidence_errors = []
    for param in validator_params:
        evidence = param.get("evidence") or {}

        for field in EXTRACTED_EVIDENCE_FIELDS:
            if field not in evidence:
                evidence_errors.append(
                    f"missing_evidence_field:{field}"
                )

        if param.get("source_mapping_status") in UNTRUSTED_SOURCE_STATUSES:
            evidence_errors.append(
                "untrusted_source:"
                + str(param.get("source_mapping_status"))
            )

    parse_evidence = _check(
        artifact is not None
        and validation is not None
        and len(validator_params) == len(final_params)
        and not evidence_errors,
        {
            "artifact_present": artifact is not None,
            "validation_present": validation is not None,
            "raw_parameter_count": len(raw_params),
            "final_parameter_count": len(final_params),
            "validator_parameter_count": len(validator_params),
            "evidence_errors": evidence_errors[:20],
            "validation_status_counts": (
                (validation or {}).get(
                    "validation_summary", {}
                )
            ),
        },
    )

    finish_reasons = [
        call.get("finish_reason") for call in calls
    ]
    completion_tokens = [
        ((call.get("usage") or {}).get("completion_tokens") or 0)
        for call in calls
    ]
    request_hashes = [
        request.get("request_sha256") for request in requests
    ]
    call_count_ok = (
        api_requests.get("call_count") == len(requests)
        == api_responses.get("call_count") == len(calls)
    )
    no_truncation = all(
        reason != "length" for reason in finish_reasons
    )
    no_hidden_retry = (
        state.get("retry_count") in (0, None)
        and all(call.get("error") is None for call in calls)
    )
    no_duplicate_requests = (
        len(request_hashes) == len(set(request_hashes))
    )
    tokens_ok = (
        bool(calls)
        and all(
            tokens <= api_responses.get("max_tokens", 6000)
            for tokens in completion_tokens
        )
        and sum(completion_tokens) > 0
    )
    calls_ok = (
        call_count_ok
        and len(calls) == expected
        and no_truncation
        and no_hidden_retry
        and no_duplicate_requests
        and tokens_ok
    )
    calls_check = _check(
        calls_ok,
        {
            "expected_chunks": expected,
            "request_count": len(requests),
            "response_count": len(calls),
            "call_count_fields": {
                "requests": api_requests.get("call_count"),
                "responses": api_responses.get("call_count"),
            },
            "finish_reasons": finish_reasons,
            "completion_tokens": completion_tokens,
            "max_tokens": api_responses.get("max_tokens"),
            "retry_count": state.get("retry_count"),
            "duplicate_request_hashes": (
                len(request_hashes) - len(set(request_hashes))
            ),
            "all_calls_have_no_error": all(
                call.get("error") is None for call in calls
            ),
        },
    )

    checks = {
        "logical_enters_model": logical_in_model,
        "expanded_prompt_enters_request": expanded_prompt,
        "adi_table4_extractable": adi_table4,
        "output_parse_and_evidence": parse_evidence,
        "no_abnormal_tokens_or_retry": calls_check,
    }
    overall = all(item["ok"] for item in checks.values())

    return {
        "schema": "b37_preflight_checks/v1",
        "sample_id": sample["sample_id"],
        "arm": arm["arm"],
        "arm_label": arm["label"],
        "overall_ok": overall,
        "checks": checks,
        "input_tables_summary": (
            (input_tables or {}).get("table_summaries")
        ),
        "note": (
            "预检只验证链路、Prompt 是否进入请求、输出是否可解析/"
            "可校验以及调用是否可审计；预检指标不得当作最终 F1。"
        ),
    }


def _write_preflight_markdown(payload, path):
    lines = [
        f"# B3.7 预检：{payload['arm_label']} / "
        f"{payload['sample_id']}",
        "",
        f"- 总体：{'PASS' if payload['overall_ok'] else 'FAIL'}",
        "",
        "| 检查 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]

    for name, check in payload["checks"].items():
        lines.append(
            f"| `{name}` | {'PASS' if check['ok'] else 'FAIL'} | "
            f"{json.dumps(check['detail'], ensure_ascii=False)} |"
        )

    lines.extend(["", payload["note"], ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _collect_run_totals(run_dir, completed):
    total_calls = 0
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    total_elapsed = 0.0
    response_models = set()
    fingerprints = set()
    preflight = {}

    for sample_id in completed:
        sample_dir = Path(run_dir) / sample_id
        api_path = sample_dir / "api_responses.json"
        usage_path = sample_dir / "usage.json"
        timing_path = sample_dir / "timing.json"

        if api_path.is_file():
            api = load_json(api_path)
            total_calls += api.get("call_count", 0)
            response_models.update(api.get("response_models") or [])
            fingerprints.update(
                api.get("system_fingerprints") or []
            )

        if usage_path.is_file():
            usage = load_json(usage_path)

            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

        if timing_path.is_file():
            total_elapsed += load_json(timing_path).get(
                "elapsed_s", 0.0
            )

        check_path = sample_dir / "preflight_checks.json"

        if check_path.is_file():
            preflight[sample_id] = load_json(check_path)

    return {
        "total_api_calls": total_calls,
        "total_usage": total_usage,
        "total_elapsed_s": round(total_elapsed, 3),
        "response_models": sorted(response_models),
        "system_fingerprints": sorted(fingerprints),
        "preflight": preflight,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument(
        "--mode", choices=["preflight", "formal"], default="formal"
    )
    parser.add_argument("--sample", action="append", default=None)
    parser.add_argument("--run-id", default="r1")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="默认预检 0、正式 1；显式覆盖会写入运行配置。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="必须显式提供；否则只做门禁检查，不调用 API。",
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    configure_environment("cpu")

    arm = resolve_arm(args.arm)
    max_retries = (
        args.max_retries
        if args.max_retries is not None
        else (
            PREFLIGHT_MAX_RETRIES
            if args.mode == "preflight"
            else FORMAL_MAX_RETRIES
        )
    )

    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("缺少 LLM_API_KEY，拒绝真实调用")

    gate = validate_gold_gate(
        manifest_path=MANIFEST,
        gold_path=GOLD,
        receipt_path=GOLD_RECEIPT,
    )
    manifest_samples = {
        sample["sample_id"]: {
            **sample,
            "file": sample["pdf_path"],
        }
        for sample in gate["samples"]
    }
    requested = set(args.sample or [])

    if requested - set(manifest_samples):
        raise SystemExit(f"未知样本: {sorted(requested - set(manifest_samples))}")

    if args.mode == "formal" and not requested:
        selected_ids = list(manifest_samples)
    else:
        selected_ids = [
            sample_id
            for sample_id in manifest_samples
            if not requested or sample_id in requested
        ]

    if args.mode == "formal" and set(selected_ids) != set(manifest_samples):
        raise SystemExit(
            "正式运行必须覆盖全部四份 PDF；预检请使用 --mode preflight"
        )

    if not selected_ids:
        raise SystemExit("没有匹配的样本")

    run_root = PREFLIGHT_DIR if args.mode == "preflight" else RUNS_DIR
    run_dir = run_root / arm["arm"] / args.run_id

    if run_dir.exists():
        raise SystemExit(f"运行目录已存在: {run_dir}")

    prepared_inputs = {}

    for sample_id in selected_ids:
        prepared_inputs[sample_id] = build_sample_inputs(
            manifest_samples[sample_id]
        )

    if not args.execute:
        print("门禁通过：Gold 已冻结，输入范围可构造。")
        print("模式:", args.mode)
        print("臂:", arm)
        print("样本:", selected_ids)
        print(
            "输入行数:",
            {
                sample_id: {
                    "flat_rows": len(
                        item["flat_table"].get("rows", [])
                    ),
                    "logical_rows": len(
                        (item.get("logical_rows") or {}).get(
                            "logical_rows", []
                        )
                    ),
                }
                for sample_id, item in prepared_inputs.items()
            },
        )
        print("将写入:", run_dir)
        print("未提供 --execute，未调用任何 API。")
        return 0

    run_dir.mkdir(parents=True)
    logger = configure_logging(run_dir / "run.log")

    config = experiment_config()
    config.update({
        "run_id": args.run_id,
        "mode": args.mode,
        "arm": arm,
        "max_retries": max_retries,
        "samples": selected_ids,
        "gold_gate": gate["hashes"],
        "gold_receipt": gate["receipt"],
    })
    write_json(run_dir / "run_config.json", config, exclusive=True)

    logger.info(
        "B3.7 开始: mode=%s arm=%s samples=%s model=%s temp=%s",
        args.mode,
        arm["arm"],
        selected_ids,
        config["model"]["requested_model"],
        config["model"]["temperature"],
    )

    failures = {}
    completed = []
    started_at = datetime.now(timezone.utc).isoformat()
    temperature = float(config["model"]["temperature"])

    try:
        for sample_id in selected_ids:
            sample = manifest_samples[sample_id]
            sample_inputs = build_sample_inputs(sample)
            logger.info("[%s] %s 开始", arm["arm"], sample_id)

            try:
                error = run_one(
                    arm,
                    sample,
                    sample_inputs,
                    run_dir / sample_id,
                    max_retries=max_retries,
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }

            if error is not None:
                failures[sample_id] = error
                logger.error(
                    "[%s] %s 失败: %s: %s",
                    arm["arm"],
                    sample_id,
                    error["type"],
                    error["message"],
                )
                break

            if args.mode == "preflight":
                checks = evaluate_preflight(
                    run_dir / sample_id,
                    sample,
                    arm,
                    sample_inputs,
                )
                write_json(
                    run_dir / sample_id / "preflight_checks.json",
                    checks,
                )
                _write_preflight_markdown(
                    checks,
                    run_dir / sample_id / "preflight_checks.md",
                )

                if not checks["overall_ok"]:
                    failures[sample_id] = {
                        "type": "PreflightCheckError",
                        "message": "预检检查未全部通过",
                        "checks": {
                            name: check["ok"]
                            for name, check in checks["checks"].items()
                        },
                    }
                    logger.error(
                        "[%s] %s 预检未通过: %s",
                        arm["arm"],
                        sample_id,
                        failures[sample_id]["checks"],
                    )
                    break

            completed.append(sample_id)
            logger.info("[%s] %s 完成", arm["arm"], sample_id)
    finally:
        totals = _collect_run_totals(run_dir, completed)
        status = "completed" if not failures else "failed"
        preflight_overall = (
            all(
                payload["overall_ok"]
                for payload in totals["preflight"].values()
            )
            if totals["preflight"]
            else None
        )
        write_json(
            run_dir / "run_manifest.json",
            {
                "schema": "b37_arm_run/v1",
                "run_id": args.run_id,
                "mode": args.mode,
                "arm": arm,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "samples_requested": selected_ids,
                "samples_completed": completed,
                "failures": failures,
                "failure_policy": (
                    "fail-fast：任一 API/解析/预检异常立即停止后续样本，"
                    "不挑选结果，不自动补跑。"
                ),
                "table_scope": TABLE_SCOPE,
                "max_retries": max_retries,
                "config": config,
                "gold_gate": gate["hashes"],
                "gold_receipt": gate["receipt"],
                "total_api_calls": totals["total_api_calls"],
                "total_usage": totals["total_usage"],
                "total_elapsed_s": totals["total_elapsed_s"],
                "response_models": totals["response_models"],
                "system_fingerprints": totals["system_fingerprints"],
                "preflight": totals["preflight"],
                "preflight_overall_ok": preflight_overall,
                "note": (
                    "温度由运行器统一注入 0.0；transport max_retries=0；"
                    "预检结果不计入最终 F1。"
                ),
            },
        )

    if failures:
        logger.error("运行失败，未完成全部样本")
        return 1

    logger.info(
        "B3.7 完成: arm=%s samples=%s calls=%s",
        arm["arm"],
        completed,
        totals["total_api_calls"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

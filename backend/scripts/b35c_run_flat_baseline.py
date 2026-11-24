"""B3.5c 跨型号 Flat Baseline 运行器。

严格约束：
- 只使用生产默认配置：flat 输入 + legacy 校验、production Prompt、
  TableSelector、StructureRecovery、Validator；
- 不使用 LogicalTableAdapter，不做评测期调优；
- 每份 PDF 只运行一次；失败明确记录并停止后续样本，不挑选结果；
- 保存每次 API 响应的模型元数据、原始输出、结构化结果、校验状态与日志；
- 冻结 Gold 未通过回执校验时拒绝运行。
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
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
from app.extraction.deepseek_extractor import (  # noqa: E402
    DeepSeekExtractor,
)
from app.parsers.pdf_parser import PDFParser  # noqa: E402
from app.parsers.table_cache import TableParseCache  # noqa: E402
from app.parsers.table_selector import (  # noqa: E402
    SelectedTableParser,
    TableSelector,
)

from b35b_docling_env import (  # noqa: E402
    build_offline_table_parser,
    configure_environment,
)
from b35c_common import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_OUTPUT,
    GOLD,
    GOLD_RECEIPT,
    MANIFEST,
    PRODUCTION_CONFIG,
    GateError,
    load_json,
    runtime_config,
    sha256_file,
    validate_gold_gate,
    write_json,
)


class RunError(RuntimeError):
    """运行前置条件或单样本执行失败。"""


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

    return logging.getLogger("b35c")


def install_api_recorder(extractor):
    """包装 client.create，只记录响应元数据，不改变返回对象。"""

    records = []
    original = extractor.client.chat.completions.create

    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        response = original(*args, **kwargs)
        usage = (
            response.usage.model_dump()
            if response.usage
            else None
        )
        records.append(
            {
                "index": len(records),
                "requested_model": kwargs.get(
                    "model", extractor.model
                ),
                "response_model": getattr(
                    response, "model", None
                ),
                "response_id": getattr(response, "id", None),
                "created": getattr(response, "created", None),
                "system_fingerprint": getattr(
                    response, "system_fingerprint", None
                ),
                "finish_reason": (
                    response.choices[0].finish_reason
                    if response.choices
                    else None
                ),
                "usage": usage,
                "elapsed_s": round(
                    time.perf_counter() - started, 3
                ),
            }
        )
        return response

    extractor.client.chat.completions.create = wrapped
    return records


def _semantic_projection(param):
    evidence = param.get("evidence") or {}
    return {
        "name": param.get("name"),
        "value": str(param.get("value")),
        "unit": param.get("unit"),
        "value_type": param.get("value_type"),
        "test_conditions": param.get("test_conditions"),
        "table_id": evidence.get("table_id"),
        "row_index": evidence.get("row_index"),
        "page": evidence.get("page"),
    }


def summarize_retry(artifact):
    initial = artifact.get("extraction", {}).get(
        "parameters", []
    )
    final = (
        artifact.get("final_extraction")
        or artifact.get("extraction")
        or {}
    ).get("parameters", [])
    retry_results = artifact.get("retry_results", [])
    candidates = [
        param
        for result in retry_results
        for param in result.get("parameters", [])
    ]

    initial_projection = [
        _semantic_projection(param) for param in initial
    ]
    final_projection = [
        _semantic_projection(param) for param in final
    ]
    changed = []

    for index in range(max(len(initial), len(final))):
        before = (
            initial_projection[index]
            if index < len(initial_projection)
            else None
        )
        after = (
            final_projection[index]
            if index < len(final_projection)
            else None
        )

        if before != after:
            changed.append(
                {
                    "index": index,
                    "before": before,
                    "after": after,
                }
            )

    return {
        "initial_parameter_count": len(initial),
        "final_parameter_count": len(final),
        "retry_result_batches": len(retry_results),
        "retry_candidate_count": len(candidates),
        "changed_parameter_count": len(changed),
        "changed_parameters": changed,
    }


def save_tables(result, output_dir):
    tables = result.get("tables", [])

    payload = {
        "table_count": len(tables),
        "tables": tables,
        "table_summaries": [
            {
                "table_id": table.get("table_id"),
                "page_numbers": table.get("page_numbers", []),
                "columns": table.get("columns", []),
                "row_count": len(table.get("rows", [])),
                "column_count": table.get("column_count"),
            }
            for table in tables
        ],
    }
    write_json(output_dir / "selected_tables.json", payload)


def run_one(
    sample,
    gold_sample,
    output_dir,
    cache,
    *,
    max_retries=1,
):
    pdf_path = PROJECT_ROOT / sample["file"]
    pdf_sha = file_sha256(pdf_path)

    if pdf_sha != sample["pdf_sha256"]:
        raise RunError(
            f"{sample['sample_id']}: PDF SHA256 与 manifest 不一致"
        )

    if pdf_sha != gold_sample["pdf_sha256"]:
        raise RunError(
            f"{sample['sample_id']}: PDF SHA256 与 Gold 不一致"
        )

    if output_dir.exists():
        raise RunError(f"样本目录已存在: {output_dir}")

    output_dir.mkdir(parents=True)

    pdf_parser = PDFParser()
    raw_table_parser = build_offline_table_parser()
    selector = TableSelector(
        max_tables=PRODUCTION_CONFIG["table_selector"]["max_tables"],
        max_chars=PRODUCTION_CONFIG["table_selector"]["max_chars"],
    )
    table_parser = SelectedTableParser(
        raw_table_parser, selector, cache=cache
    )
    extractor = DeepSeekExtractor()
    api_records = install_api_recorder(extractor)
    recording_extractor = RecordingExtractor(extractor)

    graph = build_graph(
        pdf_parser=pdf_parser,
        table_parser=table_parser,
        extractor=recording_extractor,
        evidence_validation_mode="legacy",
        recovered_evidence=None,
        extraction_input_mode="flat",
        logical_extraction=None,
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

    elapsed = round(time.perf_counter() - started, 3)

    write_json(
        output_dir / "timing.json",
        {
            "task_id": task_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed,
        },
    )
    write_json(output_dir / "usage.json", extractor.get_usage())
    write_json(
        output_dir / "api_responses.json",
        {
            "configured_model": extractor.model,
            "base_url": os.getenv(
                "LLM_BASE_URL", "https://api.deepseek.com"
            ),
            "max_tokens": extractor.max_tokens,
            "rows_per_chunk": extractor.rows_per_chunk,
            "max_split_depth": extractor.max_split_depth,
            "call_count": len(api_records),
            "calls": api_records,
        },
    )
    write_json(
        output_dir / "raw_responses.json",
        {
            "model": extractor.model,
            "max_tokens": extractor.max_tokens,
            "rows_per_chunk": extractor.rows_per_chunk,
            "max_split_depth": extractor.max_split_depth,
            "raw_responses": extractor.raw_responses,
        },
    )
    write_json(
        output_dir / "selection.json",
        {
            "manifest_selected_table_ids": sample[
                "selector_selected_table_ids"
            ],
            "ec_table_ids": sample["ec_table_ids"],
            "gold_scope_table_ids": sample[
                "gold_scope_table_ids"
            ],
            "actual_selected_table_ids": (
                [
                    table.get("table_id")
                    for table in result.get("tables", [])
                ]
                if result
                else []
            ),
        },
    )

    if error is not None:
        write_json(output_dir / "errors.json", error)
        return error

    write_json(output_dir / "errors.json", {"error": None})
    save_tables(result, output_dir)

    artifact = artifact_from_state(
        result,
        recording_extractor,
        pdf_path=pdf_path,
        pdf_sha256=pdf_sha,
        table_cache_dir=(
            Path(cache.cache_dir).resolve()
            if cache is not None
            else None
        ),
        table_cache_key=(
            cache.build_key(
                pdf_path,
                table_parser._config_signature(),
            )
            if cache is not None
            else None
        ),
        model=extractor.model,
    )
    save_artifact(artifact, output_dir / "artifact.json")
    write_json(
        output_dir / "retry_summary.json",
        summarize_retry(artifact),
    )

    # 首轮校验状态：用生产 Validator 对原始提取结果做确定性重算，
    # 不调用 LLM，仅用于区分 validated_first_pass 与 retry 纠正。
    from app.extraction.evidence import verify_evidence
    from app.validation.validator import ParameterValidator

    first_pass = ParameterValidator().validate(
        verify_evidence(
            artifact.get("extraction") or {},
            result.get("tables", []),
        )
    )
    write_json(
        output_dir / "first_pass_validation.json",
        {
            "source": "recomputed_from_raw_artifact",
            "note": (
                "对 recorder.extract_result 重新执行 verify_evidence + "
                "ParameterValidator，等价于首轮 validate 节点；"
                "不调用 LLM。"
            ),
            "validation": first_pass,
        },
    )

    validation = result.get("validation_result", {})
    write_json(
        output_dir / "validation.json",
        validation,
    )
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
                for table in result.get("tables", [])
            ],
            "validation_summary": validation.get(
                "validation_summary"
            ),
            "validation_errors": result.get("errors", []),
        },
    )

    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--receipt", default=str(GOLD_RECEIPT))
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT / "runs")
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--sample", action="append", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="必须显式提供；否则只做门禁检查，不调用 API",
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    configure_environment("cpu")

    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("缺少 LLM_API_KEY，拒绝真实调用")

    if os.getenv("EXTRACTION_INPUT_MODE", "flat") != "flat":
        raise SystemExit("B3.5c 要求 EXTRACTION_INPUT_MODE=flat")

    if os.getenv("EVIDENCE_VALIDATION_MODE", "legacy") != "legacy":
        raise SystemExit(
            "B3.5c 要求 EVIDENCE_VALIDATION_MODE=legacy"
        )

    gate = validate_gold_gate(
        manifest_path=args.manifest,
        gold_path=args.gold,
        receipt_path=args.receipt,
    )
    manifest = gate["manifest"]
    gold_by_id = {
        sample["sample_id"]: sample
        for sample in gate["gold"]["samples"]
    }
    requested = set(args.sample or [])

    samples = [
        sample
        for sample in manifest["samples"]
        if not requested or sample["sample_id"] in requested
    ]

    if not samples:
        raise SystemExit("没有匹配的样本")

    missing = [
        sample["sample_id"]
        for sample in samples
        if sample["sample_id"] not in gold_by_id
    ]

    if missing:
        raise SystemExit(f"Gold 缺少样本: {missing}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = Path(args.output_root) / run_id

    if run_dir.exists():
        raise SystemExit(f"运行目录已存在: {run_dir}")

    if not args.execute:
        print("门禁通过：Gold 已冻结，环境为 flat + legacy。")
        print("样本:", [sample["sample_id"] for sample in samples])
        print("将写入:", run_dir)
        print("未提供 --execute，未调用任何 API。")
        return 0

    run_dir.mkdir(parents=True)
    logger = configure_logging(run_dir / "run.log")
    cache = TableParseCache(args.cache_dir)
    config = runtime_config(
        {
            "run_id": run_id,
            "samples": [sample["sample_id"] for sample in samples],
            "gold_gate": gate["hashes"],
            "gold_receipt": gate["receipt"],
            "cache_dir": str(Path(args.cache_dir).resolve()),
        }
    )
    write_json(run_dir / "run_config.json", config)

    logger.info(
        "B3.5c Flat Baseline 开始: run_id=%s samples=%s",
        run_id,
        [sample["sample_id"] for sample in samples],
    )

    failures = {}
    completed = []
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        for sample in samples:
            sample_id = sample["sample_id"]
            logger.info("[baseline] %s", sample_id)

            try:
                error = run_one(
                    sample,
                    gold_by_id[sample_id],
                    run_dir / sample_id,
                    cache,
                    max_retries=PRODUCTION_CONFIG["max_retries"],
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
                    "[baseline] %s 失败: %s: %s",
                    sample_id,
                    error["type"],
                    error["message"],
                )
                break

            completed.append(sample_id)
            logger.info("[baseline] %s 完成", sample_id)
    finally:
        total_calls = 0
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for sample_id in completed:
            sample_dir = run_dir / sample_id
            api_path = sample_dir / "api_responses.json"
            usage_path = sample_dir / "usage.json"

            if api_path.is_file():
                total_calls += load_json(api_path).get(
                    "call_count", 0
                )

            if usage_path.is_file():
                usage = load_json(usage_path)

                for key in total_usage:
                    total_usage[key] += usage.get(key, 0)

        status = "completed" if not failures else "failed"
        write_json(
            run_dir / "run_manifest.json",
            {
                "schema": "b35c_baseline_run/v1",
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "samples_requested": [
                    sample["sample_id"] for sample in samples
                ],
                "samples_completed": completed,
                "failures": failures,
                "failure_policy": (
                    "fail-fast：任一 API/流程异常立即停止后续样本，"
                    "不自动重试或挑选结果"
                ),
                "config": config,
                "gold_gate": gate["hashes"],
                "gold_receipt": gate["receipt"],
                "total_api_calls": total_calls,
                "total_usage": total_usage,
            },
        )

    if failures:
        logger.error("运行失败，未完成全部样本")
        return 1

    logger.info(
        "B3.5c Flat Baseline 完成: %s", completed
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

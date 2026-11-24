"""B3.5c 预检：冻结 Gold、样本 scope、缓存、Selector 与 API 配置。

只读校验，不调用提取流程；`--check-api` 会发起 1 次最小健康检查调用。
所有输出写入 B3.5c 新目录，拒绝覆盖既有产物。
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.parsers.table_cache import TableParseCache  # noqa: E402
from app.parsers.table_selector import TableSelector  # noqa: E402

from b35b_docling_env import build_offline_table_parser  # noqa: E402
from b35c_common import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_OUTPUT,
    GateError,
    MANIFEST,
    GOLD,
    GOLD_RECEIPT,
    PRODUCTION_CONFIG,
    load_json,
    runtime_config,
    sha256_file,
    validate_gold_gate,
    write_json,
)


CACHE_SIGNATURE = {"do_ocr": False, "do_table_structure": True}


def read_cached_tables(pdf_path, cache, allow_parse=False):
    key = cache.build_key(str(pdf_path), CACHE_SIGNATURE)
    cached = cache.load(key)

    if cached is not None:
        return cached, True

    if not allow_parse:
        raise GateError(
            f"Docling 缓存未命中且未允许解析: {pdf_path.name}"
        )

    parser = build_offline_table_parser()
    parsed = parser.parse(str(pdf_path))
    cache.save(key, parsed)
    return parsed, False


def api_health_check():
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv(
        "LLM_BASE_URL", "https://api.deepseek.com"
    )
    model = os.getenv("LLM_MODEL", "deepseek-flash")

    if not api_key:
        return {
            "ok": False,
            "error": "LLM_API_KEY 未设置",
            "base_url": base_url,
            "requested_model": model,
        }

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
        max_retries=0,
    )
    started = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        '只返回 JSON：{"ok": true}'
                    ),
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
        )
        usage = (
            response.usage.model_dump()
            if response.usage
            else None
        )

        return {
            "ok": True,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "base_url": base_url,
            "requested_model": model,
            "response_model": response.model,
            "response_id": response.id,
            "created": response.created,
            "system_fingerprint": getattr(
                response, "system_fingerprint", None
            ),
            "finish_reason": response.choices[0].finish_reason,
            "content": response.choices[0].message.content,
            "usage": usage,
            "call_count": 1,
            "retries": 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "base_url": base_url,
            "requested_model": model,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "call_count": 1,
            "retries": 0,
        }


def build_preflight(
    *,
    manifest_path=MANIFEST,
    gold_path=GOLD,
    receipt_path=GOLD_RECEIPT,
    cache_dir=DEFAULT_CACHE,
    check_api=False,
    allow_parse=False,
):
    gate = validate_gold_gate(
        manifest_path=manifest_path,
        gold_path=gold_path,
        receipt_path=receipt_path,
    )
    cache = TableParseCache(str(cache_dir))
    selector = TableSelector(
        max_tables=PRODUCTION_CONFIG["table_selector"]["max_tables"],
        max_chars=PRODUCTION_CONFIG["table_selector"]["max_chars"],
    )
    samples = []
    total_estimated_calls = 0

    for sample in gate["samples"]:
        pdf_path = PROJECT_ROOT / sample["pdf_path"]
        parsed, cache_hit = read_cached_tables(
            pdf_path, cache, allow_parse=allow_parse
        )
        tables = parsed.get("tables", [])
        selected = selector.select(tables)
        selected_ids = [
            int(table["table_id"]) for table in selected
        ]
        expected_ids = [
            int(value)
            for value in sample["selector_selected_table_ids"]
        ]

        if sorted(selected_ids) != sorted(expected_ids):
            raise GateError(
                f"{sample['sample_id']}: 当前 Selector 选中 "
                f"{selected_ids} 与 manifest 记录 {expected_ids} 不一致"
            )

        scope_ids = set(sample["gold_scope_table_ids"])
        covered = scope_ids.issubset(set(selected_ids))

        if not covered:
            raise GateError(
                f"{sample['sample_id']}: Selector 未覆盖 Gold scope 表"
            )

        table_by_id = {
            int(table["table_id"]): table for table in tables
        }
        estimated_calls = 0
        selected_rows = {}

        for table_id in selected_ids:
            table = table_by_id[table_id]
            rows = table.get("rows") or []
            calls = max(
                1,
                math.ceil(
                    len(rows)
                    / PRODUCTION_CONFIG["rows_per_chunk"]
                ),
            )
            estimated_calls += calls
            selected_rows[table_id] = len(rows)

        total_estimated_calls += estimated_calls
        samples.append(
            {
                **sample,
                "cache_hit": cache_hit,
                "table_count": len(tables),
                "selected_table_ids": selected_ids,
                "gold_scope_covered": covered,
                "selected_rows": selected_rows,
                "estimated_initial_llm_calls": estimated_calls,
            }
        )

    api_health = api_health_check() if check_api else None

    if check_api and not api_health.get("ok"):
        raise GateError(
            "API 健康检查失败: "
            f"{api_health.get('error_type')}: "
            f"{api_health.get('error')}"
        )

    report = {
        "schema": "b35c_preflight/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "stage": "B3.5c",
        "gold_gate": {
            "hashes": gate["hashes"],
            "gold_receipt": gate["receipt"],
            "sync_receipt": gate["sync_receipt"],
            "record_count": gate["gold"]["record_count"],
            "sample_count": gate["gold"]["sample_count"],
            "only_confirmed": all(
                record["review"]["status"] == "confirmed"
                for sample in gate["gold"]["samples"]
                for record in sample["records"]
            ),
        },
        "production_config": dict(PRODUCTION_CONFIG),
        "runtime_config": runtime_config(),
        "samples": samples,
        "estimated_initial_llm_calls": total_estimated_calls,
        "estimated_total_llm_calls_note": (
            "不含生产 retry、截断二分与坏输出重试；"
            "实际调用次数以运行日志与 usage 为准。"
        ),
        "api_health": api_health,
    }

    return report


def write_markdown(report, path):
    lines = [
        "# B3.5c 预检：跨型号 Flat Baseline",
        "",
        f"- 生成时间（UTC）：{report['generated_at']}",
        f"- 状态：**{report['status']}**",
        f"- 冻结 Gold：{report['gold_gate']['record_count']} 条 / "
        f"{report['gold_gate']['sample_count']} 型号，仅 confirmed："
        f"{report['gold_gate']['only_confirmed']}",
        f"- Gold SHA256：`{report['gold_gate']['hashes']['gold']}`",
        f"- 冻结回执 SHA256："
        f"`{report['gold_gate']['hashes']['gold_receipt']}`",
        f"- 预计初始 LLM 调用次数："
        f"**{report['estimated_initial_llm_calls']}**"
        "（不含 retry / 截断拆分）",
        "",
        "## 生产配置",
        "",
        "```json",
        json.dumps(report["production_config"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 样本与选表",
        "",
        "| sample | vendor | Gold 条数 | Gold scope | Selector 选中 | "
        "缓存命中 | 预计调用 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for sample in report["samples"]:
        lines.append(
            "| {sample_id} | {vendor} | {record_count} | {scope} | "
            "{selected} | {cache} | {calls} |".format(
                sample_id=sample["sample_id"],
                vendor=sample["vendor"],
                record_count=sample["record_count"],
                scope=sample["gold_scope_table_ids"],
                selected=sample["selected_table_ids"],
                cache=sample["cache_hit"],
                calls=sample["estimated_initial_llm_calls"],
            )
        )

    lines.extend(
        [
            "",
            "## API 健康检查",
            "",
        ]
    )

    health = report.get("api_health")

    if health is None:
        lines.append("- 未启用 `--check-api`，本轮预检未调用 API。")
    elif health.get("ok"):
        lines.extend(
            [
                "- 状态：**OK**",
                f"- 请求模型：`{health['requested_model']}`",
                f"- 响应模型：`{health['response_model']}`",
                f"- response_id：`{health['response_id']}`",
                f"- system_fingerprint："
                f"`{health['system_fingerprint']}`",
                f"- 用量：`{health['usage']}`",
                f"- 调用次数：{health['call_count']}，重试："
                f"{health['retries']}",
            ]
        )
    else:
        lines.append(
            f"- 状态：**失败** — {health.get('error_type')}: "
            f"{health.get('error')}"
        )

    lines.extend(
        [
            "",
            "## 评分范围",
            "",
            "- 仅使用 76 条 confirmed 记录作为正向 Gold；"
            "12 条 rejected 不进入标准答案。",
            "- 每份 PDF 只取 manifest 的 `gold_scope_table_ids`；"
            "其他表格的预测标记为 `out_of_scope_table`，不计入 P/R。",
            "- 同表内、Gold 身份集合之外的预测标记为 "
            "`out_of_scope_parameter`，不计入 false positive。",
            "- 原始 Flat Baseline 不使用 LogicalTableAdapter，"
            "因此该错误类别预期为 0/N/A。",
        ]
    )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--receipt", default=str(GOLD_RECEIPT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT / "preflight"),
    )
    parser.add_argument(
        "--check-api",
        action="store_true",
        help="发起 1 次最小 API 健康检查（不提取参数）",
    )
    parser.add_argument(
        "--allow-parse",
        action="store_true",
        help="缓存未命中时允许运行 Docling 解析（不调用 LLM）",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    json_path = output_dir / "preflight.json"
    md_path = output_dir / "preflight.md"
    config_path = output_dir.parent / "config/run_config.json"
    health_path = output_dir / "api_health.json"

    existing = [
        path
        for path in (json_path, md_path, config_path)
        if path.exists()
    ]

    if existing and not args.force:
        raise SystemExit(
            "预检产物已存在，拒绝覆盖: "
            + ", ".join(str(path) for path in existing)
        )

    report = build_preflight(
        manifest_path=args.manifest,
        gold_path=args.gold,
        receipt_path=args.receipt,
        cache_dir=args.cache_dir,
        check_api=args.check_api,
        allow_parse=args.allow_parse,
    )

    write_json(json_path, report)
    write_markdown(report, md_path)
    write_json(
        config_path,
        {
            "schema": "b35c_run_config/v1",
            "generated_at": report["generated_at"],
            "production_config": report["production_config"],
            "runtime_config": report["runtime_config"],
            "gold_gate": report["gold_gate"],
            "preflight": str(json_path),
            "estimated_initial_llm_calls": report[
                "estimated_initial_llm_calls"
            ],
        },
    )

    if report.get("api_health") is not None:
        write_json(health_path, report["api_health"])

    print("预检通过:", json_path)
    print("报告:", md_path)
    print("配置:", config_path)

    if report.get("api_health") is not None:
        print("API 健康检查:", health_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

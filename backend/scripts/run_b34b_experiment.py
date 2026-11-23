"""阶段 B3.4b：Logical Prompt 提取完整性优化实验。

控制变量：
- 只修改 backend/app/extraction/prompts.py 中的
  LOGICAL_INPUT_INSTRUCTION；
- Flat Prompt、Schema、EvidenceValidator、
  RecoveredEvidenceValidator、来源映射和 Retry 不变；
- rows_per_chunk 保持 10，max_tokens 保持 6000，
  max_split_depth 保持 6；
- 只运行 Logical 单臂，不重跑 Flat 基线；
- 不启用 LangGraph Retry，保留截断自动二分；
- 使用同一份 Logical Rows v2 和 51 条 Gold v2；
- 默认只做安全预检，必须显式 --execute 才调用真实 API；
- 使用全新输出目录，拒绝覆盖任何已有实验产物。
"""

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_ROOT = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from run_ab_experiment import (
    ExperimentError,
    find_flat_table,
    load_json,
    run_arm,
    sha256_file,
    write_json_exclusive,
    write_text_exclusive,
)
from run_b34a_experiment import (
    build_record_rows,
    chunk_summary,
    detail_index,
    detail_status,
    failed_fields,
    metric_block,
    metric_delta,
    replay_chunk_tree,
)
from app.extraction import prompts
from app.extraction.deepseek_extractor import (
    EXTRACT_INSTRUCTION,
)
from app.extraction.input_mode import (
    build_extraction_input,
)


DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b"
    / "20261122T143428Z"
    / "ab_result.json"
)
DEFAULT_B34A_PREFLIGHT = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.4a"
    / "preflight"
    / "20261122T152016Z"
    / "preflight.json"
)
DEFAULT_LOGICAL_ROWS = (
    PROJECT_ROOT / "data/outputs/TPS7A02.logical_rows.v2.json"
)
DEFAULT_RAW_STRUCTURE = (
    PROJECT_ROOT / "data/outputs/TPS7A02.raw_structure.v1.json"
)
DEFAULT_GOLD = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.v2.eval.json"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data/cache/docling"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data/outputs/experiments/B3.4b"
)
DEFAULT_TABLE_ID = 9
EXPECTED_BRANCH = "experiment/b3.4b-prompt"
BASE_COMMIT = "21ac3f26acf893bc30a3808415352c663f1f00d3"
ROWS_PER_CHUNK = 10
MAX_SPLIT_DEPTH = 6
MAX_TOKENS = 6000
CANDIDATE_PROMPT_VERSION = "b3.4b-v1"
PROMPT_SIZE_BUDGET = 2600

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

ALLOWED_TRACKED_DIFF = {
    "backend/app/extraction/prompts.py",
    "backend/tests/test_b34b_prompt.py",
    "backend/scripts/run_b34b_experiment.py",
}

GROUPS = {
    "accuracy": {f"t9.p{i}" for i in range(1, 5)},
    "line_regulation": {"t9.p5"},
    "load_regulation": {"t9.p6", "t9.p7"},
    "ground_current": {f"t9.p{i}" for i in range(8, 14)},
    "shutdown_current": {"t9.p14"},
    "current_limit": {"t9.p15", "t9.p16", "t9.p17"},
    "vdo": {f"t9.p{i}" for i in range(18, 32)},
    "psrr": {"t9.p32"},
    "output_noise": {"t9.p33"},
    "uvlo": {"t9.p34", "t9.p35", "t9.p36"},
}


def utc_now():
    return datetime.now(timezone.utc)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ExperimentError(
            f"git {' '.join(args)} 失败"
        ) from exc


def read_prompt_from_commit(commit, name):
    source = git_output(
        "show", f"{commit}:backend/app/extraction/prompts.py"
    )
    tree = ast.parse(source)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        targets = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]

        if name in targets:
            return ast.literal_eval(node.value)

    raise ExperimentError(
        f"基线提交中找不到 Prompt: {name}"
    )


def load_baseline(path):
    baseline = load_json(path)

    if baseline.get("schema") != "ab_experiment_result/v1":
        raise ExperimentError(
            "基线文件 schema 不是 ab_experiment_result/v1"
        )

    arms = {
        arm.get("arm"): arm
        for arm in baseline.get("arms", [])
    }
    logical = arms.get("logical")

    if logical is None:
        raise ExperimentError("基线缺少 logical 单臂")

    config = logical.get("extractor_config") or {}

    if config.get("rows_per_chunk") != ROWS_PER_CHUNK:
        raise ExperimentError(
            "基线 Logical rows_per_chunk 不是 10"
        )

    return baseline, logical, config


def check_code_scope():
    branch = git_output(
        "rev-parse", "--abbrev-ref", "HEAD"
    )
    head = git_output("rev-parse", "HEAD")
    status = git_output(
        "status", "--porcelain=v1"
    )
    changed = git_output(
        "diff",
        "--name-only",
        BASE_COMMIT,
        "--",
        "backend/app",
        "backend/scripts",
        "backend/tests",
        "pyproject.toml",
    )
    changed_files = [
        line.strip()
        for line in changed.splitlines()
        if line.strip()
    ]
    unexpected = sorted(
        set(changed_files) - ALLOWED_TRACKED_DIFF
    )

    if branch != EXPECTED_BRANCH:
        raise ExperimentError(
            "当前分支不是 "
            f"{EXPECTED_BRANCH}: {branch}"
        )

    if unexpected:
        raise ExperimentError(
            "出现 Prompt 之外的 tracked 代码修改: "
            f"{unexpected}"
        )

    git_output(
        "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"
    )

    return {
        "branch": branch,
        "head": head,
        "base_commit": BASE_COMMIT,
        "status_porcelain": status.splitlines(),
        "tracked_changed_files": changed_files,
        "unexpected_tracked_changes": unexpected,
    }


def check_prompts(base_logical_prompt):
    current_logical = prompts.LOGICAL_INPUT_INSTRUCTION
    candidate = {
        "version": getattr(
            prompts,
            "LOGICAL_INPUT_INSTRUCTION_VERSION",
            None,
        ),
        "sha256": sha256_text(current_logical),
        "chars": len(current_logical),
    }
    base_logical = {
        "sha256": sha256_text(base_logical_prompt),
        "chars": len(base_logical_prompt),
    }
    checks = {
        "Flat SYSTEM_PROMPT 未修改": (
            sha256_text(prompts.SYSTEM_PROMPT)
            == BASELINE_SYSTEM_PROMPT_SHA256
        ),
        "Flat EXTRACT_INSTRUCTION 未修改": (
            sha256_text(EXTRACT_INSTRUCTION)
            == BASELINE_EXTRACT_INSTRUCTION_SHA256
        ),
        "Logical Retry Prompt 未修改": (
            sha256_text(prompts.LOGICAL_RETRY_INSTRUCTION)
            == BASELINE_LOGICAL_RETRY_SHA256
        ),
        "Logical 输入 Prompt 已修改": (
            candidate["sha256"]
            != B34A_LOGICAL_INPUT_SHA256
            and candidate["sha256"]
            != base_logical["sha256"]
        ),
        "Logical Prompt 版本正确": (
            candidate["version"]
            == CANDIDATE_PROMPT_VERSION
        ),
        "Logical Prompt 大小在预算内": (
            candidate["chars"] <= PROMPT_SIZE_BUDGET
        ),
    }

    for name, passed in checks.items():
        if not passed:
            raise ExperimentError(
                f"Prompt 预检失败: {name}"
            )

    return {
        "checks": checks,
        "baseline_logical": base_logical,
        "candidate_logical": candidate,
        "char_delta": (
            candidate["chars"] - base_logical["chars"]
        ),
        "flat_system_prompt_sha256": (
            sha256_text(prompts.SYSTEM_PROMPT)
        ),
        "flat_extract_instruction_sha256": (
            sha256_text(EXTRACT_INSTRUCTION)
        ),
        "logical_retry_sha256": (
            sha256_text(
                prompts.LOGICAL_RETRY_INSTRUCTION
            )
        ),
    }


def check_other_code_unchanged(b34a_preflight_path):
    preflight = load_json(b34a_preflight_path)
    rows = []

    for item in preflight.get("code_provenance", []):
        relative = item["path"]

        if relative.endswith("prompts.py"):
            continue

        current = sha256_file(
            PROJECT_ROOT / relative
        )
        rows.append({
            "path": relative,
            "expected_sha256": item["sha256"],
            "current_sha256": current,
            "unchanged": current == item["sha256"],
        })

    if not rows:
        raise ExperimentError(
            "B3.4a 预检中没有可用的代码哈希"
        )

    if not all(row["unchanged"] for row in rows):
        changed = [
            row["path"]
            for row in rows
            if not row["unchanged"]
        ]
        raise ExperimentError(
            "Prompt 之外存在代码变化: "
            f"{changed}"
        )

    return rows


def build_prepared(args):
    code_scope = check_code_scope()
    baseline_path = Path(args.baseline)
    baseline, baseline_arm, baseline_config = (
        load_baseline(baseline_path)
    )
    base_logical_prompt = read_prompt_from_commit(
        BASE_COMMIT,
        "LOGICAL_INPUT_INSTRUCTION",
    )
    prompt_state = check_prompts(
        base_logical_prompt
    )
    other_code = check_other_code_unchanged(
        args.b34a_preflight
    )

    logical_rows = load_json(args.logical_rows)
    raw_structure = load_json(args.raw_structure)
    gold = load_json(args.gold)

    logical_rows_sha = sha256_file(args.logical_rows)
    raw_structure_sha = sha256_file(args.raw_structure)
    gold_sha = sha256_file(args.gold)
    baseline_gold = baseline["plan"]["gold"]
    baseline_source = baseline["gold_standard"]["source"]
    baseline_pdf = baseline["plan"]["target"].get(
        "pdf_sha256"
    )

    checks = {
        "基线文件存在且 schema 正确": True,
        "基线 Logical rows_per_chunk=10": True,
        "Logical Rows v2 与基线相同": (
            logical_rows_sha
            == baseline_source.get(
                "logical_rows_v2_sha256"
            )
        ),
        "Gold v2 eval 与基线相同": (
            gold_sha == baseline_gold.get("sha256")
        ),
        "PDF 与基线相同": (
            (raw_structure.get("source") or {}).get(
                "pdf_sha256"
            )
            == baseline_pdf
        ),
        "Gold 为 51 条": (
            gold.get("expected_parameter_count") == 51
            and len(gold.get("expected_parameters", []))
            == 51
        ),
        "Gold 含表级默认条件": bool(
            gold.get("table_default_conditions")
        ),
        "Gold 含脚注 (1)～(4)": all(
            str(number) in gold.get("footnotes", {})
            for number in (1, 2, 3, 4)
        ),
    }
    checks.update(prompt_state["checks"])

    for name, passed in checks.items():
        if not passed:
            raise ExperimentError(
                f"预检失败: {name}"
            )

    flat_table, table_source = find_flat_table(
        args.table_cache_dir,
        args.table_id,
    )

    if len(flat_table.get("rows", [])) != baseline[
        "plan"
    ]["target"].get("flat_row_count"):
        raise ExperimentError(
            "Flat 行数与基线不一致"
        )

    logical_prepared = build_extraction_input(
        [flat_table],
        mode="logical",
        logical_rows=logical_rows,
        raw_structure=raw_structure,
        table_ids=[args.table_id],
    )

    if logical_prepared["effective_mode"] != "logical":
        raise ExperimentError(
            "Logical 输入未覆盖目标表格"
        )

    logical_table = logical_prepared["tables"][0]

    if len(logical_table.get("records", [])) != baseline[
        "plan"
    ]["target"].get("logical_trusted_row_count"):
        raise ExperimentError(
            "Logical 可信行数与基线不一致"
        )

    if logical_table.get("excluded_rows"):
        raise ExperimentError(
            "Logical 输入存在 needs_review 排除行"
        )

    candidate_model = os.getenv(
        "LLM_MODEL", "deepseek-flash"
    )
    planned_config = {
        "model": candidate_model,
        "input_mode": "logical",
        "max_tokens": baseline_config.get("max_tokens"),
        "rows_per_chunk": ROWS_PER_CHUNK,
        "max_split_depth": baseline_config.get(
            "max_split_depth"
        ),
        "temperature": baseline_config.get(
            "temperature", "provider_default"
        ),
    }
    config_checks = {
        "模型与基线一致": (
            planned_config["model"]
            == baseline_config.get("model")
        ),
        "max_tokens 与基线一致": (
            planned_config["max_tokens"]
            == baseline_config.get("max_tokens")
            == MAX_TOKENS
        ),
        "rows_per_chunk=10": (
            planned_config["rows_per_chunk"]
            == ROWS_PER_CHUNK
        ),
        "max_split_depth 与基线一致": (
            planned_config["max_split_depth"]
            == baseline_config.get(
                "max_split_depth"
            )
            == MAX_SPLIT_DEPTH
        ),
        "temperature 与基线一致": (
            planned_config["temperature"]
            == baseline_config.get(
                "temperature",
                "provider_default",
            )
        ),
    }
    checks.update(config_checks)

    for name, passed in config_checks.items():
        if not passed:
            raise ExperimentError(
                f"配置预检失败: {name}"
            )

    refs = [
        record["logical_row_id"]
        for record in logical_table["records"]
    ]
    baseline_replay = replay_chunk_tree(
        "baseline-logical",
        baseline_arm.get("raw_llm_responses", []),
        refs,
        ROWS_PER_CHUNK,
        baseline_config.get("max_split_depth") or 6,
    )
    baseline_chunks = chunk_summary(
        baseline_replay,
        ROWS_PER_CHUNK,
        baseline_config.get("max_split_depth") or 6,
    )

    return {
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "generated_at": baseline["generated_at"],
            "extractor_config": baseline_config,
        },
        "inputs": {
            "logical_rows_path": str(args.logical_rows),
            "logical_rows_sha256": logical_rows_sha,
            "gold_path": str(args.gold),
            "gold_sha256": gold_sha,
            "raw_structure_path": str(args.raw_structure),
            "raw_structure_sha256": raw_structure_sha,
            "pdf_sha256": (
                raw_structure.get("source") or {}
            ).get("pdf_sha256"),
            "flat_table_source": str(table_source),
            "flat_row_count": len(
                flat_table.get("rows", [])
            ),
            "logical_trusted_row_count": len(
                logical_table.get("records", [])
            ),
            "logical_row_ids": refs,
        },
        "prompt": prompt_state,
        "other_code": other_code,
        "git": code_scope,
        "planned_config": planned_config,
        "baseline_config": baseline_config,
        "checks": checks,
        "baseline_replay": baseline_replay,
        "baseline_chunks": baseline_chunks,
        "gold": gold,
        "logical_rows": logical_rows,
        "raw_structure": raw_structure,
        "flat_table": flat_table,
        "logical_table": logical_table,
        "baseline_arm": baseline_arm,
    }


def plan_payload(prepared, execute):
    return {
        "schema": "b34b_plan/v1",
        "stage": "B3.4b",
        "created_at": utc_now().isoformat(),
        "api_call": bool(execute),
        "git": prepared["git"],
        "baseline": prepared["baseline"],
        "candidate": {
            "arm": "logical",
            "extractor_config": prepared[
                "planned_config"
            ],
        },
        "prompt": prepared["prompt"],
        "other_code": prepared["other_code"],
        "inputs": {
            key: value
            for key, value in prepared["inputs"].items()
            if key != "logical_row_ids"
        },
        "control_variable": {
            "changed": "LOGICAL_INPUT_INSTRUCTION",
            "flat_prompt_unchanged": True,
            "retry_prompt_unchanged": True,
            "schema_unchanged": True,
            "validators_unchanged": True,
            "source_mapping_unchanged": True,
            "rows_per_chunk": ROWS_PER_CHUNK,
            "max_tokens": MAX_TOKENS,
            "max_split_depth": MAX_SPLIT_DEPTH,
        },
        "controls": {
            "only_logical_arm": True,
            "flat_arm_not_rerun": True,
            "retry_enabled": False,
            "truncation_bisection_kept": True,
            "same_gold": True,
            "same_logical_rows": True,
            "overwrite_existing_outputs": False,
        },
        "checks": prepared["checks"],
    }


def preflight_payload(prepared, execute):
    return {
        "schema": "b34b_preflight/v1",
        "stage": "B3.4b",
        "created_at": utc_now().isoformat(),
        "execute": bool(execute),
        "passed": all(prepared["checks"].values()),
        "checks": prepared["checks"],
        "git": prepared["git"],
        "prompt": prepared["prompt"],
        "baseline": {
            **prepared["baseline"],
            "rows_per_chunk": ROWS_PER_CHUNK,
            "chunk_summary": prepared[
                "baseline_chunks"
            ],
        },
        "candidate": {
            "rows_per_chunk": ROWS_PER_CHUNK,
            "config": prepared["planned_config"],
        },
        "inputs": {
            key: value
            for key, value in prepared["inputs"].items()
            if key != "logical_row_ids"
        },
        "other_code": prepared["other_code"],
    }


def group_metrics(
    gold_details,
    group_ids,
):
    selected = [
        item for item in gold_details
        if item["logical_row_id"] in group_ids
    ]
    total = len(selected)
    matched = sum(
        1 for item in selected
        if item["match_status"] == "matched"
    )
    strict = sum(
        1 for item in selected
        if item["strict_match"]
    )
    missing = sum(
        1 for item in selected
        if item["match_status"] == "missing"
    )

    def ratio(numerator):
        return (
            round(numerator / total, 4)
            if total else None
        )

    return {
        "gold_records": total,
        "matched_keys": matched,
        "strict_matched": strict,
        "missing": missing,
        "key_recall": ratio(matched),
        "strict_recall": ratio(strict),
    }


def build_group_comparison(
    baseline_arm,
    candidate_arm,
):
    baseline_details = baseline_arm["evaluation"][
        "gold_details"
    ]
    candidate_details = candidate_arm["evaluation"][
        "gold_details"
    ]
    groups = {}

    for name, ids in GROUPS.items():
        groups[name] = {
            "baseline": group_metrics(
                baseline_details, ids
            ),
            "candidate": group_metrics(
                candidate_details, ids
            ),
        }

    return groups


def record_changes(
    all_rows,
    candidate_arm,
):
    recovered = [
        row for row in all_rows
        if row["recovered_strict"]
    ]
    regressed = [
        row for row in all_rows
        if row["regressed_strict"]
    ]
    status_changes = [
        row for row in all_rows
        if row["baseline_status"]
        != row["candidate_status"]
    ]
    candidate_field_errors = []

    for detail in candidate_arm["evaluation"][
        "field_mismatches"
    ]:
        candidate_field_errors.append({
            "logical_row_id": detail["logical_row_id"],
            "value_type": detail["value_type"],
            "fields": detail["fields"],
        })

    return {
        "recovered_strict": recovered,
        "regressed_strict": regressed,
        "status_changes": status_changes,
        "candidate_field_errors": (
            candidate_field_errors
        ),
        "candidate_missing": candidate_arm[
            "evaluation"
        ]["missing"],
    }


def _status_cell(status):
    return {
        "strict": "STRICT",
        "key_only": "KEY_ONLY",
        "missing": "MISSING",
    }.get(status, status)


def build_markdown(
    prepared,
    baseline_arm,
    candidate_arm,
    chunk_analysis,
    groups,
    changes,
    all_rows,
):
    baseline_block = metric_block(baseline_arm)
    candidate_block = metric_block(candidate_arm)
    delta = metric_delta(
        baseline_block, candidate_block
    )
    baseline_chunks = prepared["baseline_chunks"]
    prompt = prepared["prompt"]
    baseline_special = baseline_arm["evaluation"][
        "special"
    ]
    candidate_special = candidate_arm["evaluation"][
        "special"
    ]

    lines = [
        "# B3.4b Logical Prompt 完整性实验报告",
        "",
        f"- 阶段：B3.4b",
        f"- 生成时间：{utc_now().isoformat()}",
        f"- Git 分支：`{prepared['git']['branch']}`",
        f"- Git HEAD：`{prepared['git']['head']}`",
        f"- 基线：`{prepared['baseline']['path']}`",
        f"- Gold：51 条 Gold v2，"
        f"sha256=`{prepared['inputs']['gold_sha256']}`",
        "",
        "> 本阶段只修改 Logical 输入 Prompt。",
        "> Flat Prompt、Schema、EvidenceValidator、"
        "RecoveredEvidenceValidator、来源映射和 Retry 未修改。",
        "> rows_per_chunk 保持 10，未重跑 Flat 基线。",
        "> 未进入 B4。",
        "",
        "## 一、安全预检",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]

    for name, value in prepared["checks"].items():
        lines.append(
            f"| {name} | {'PASS' if value else 'FAIL'} |"
        )

    lines += [
        "",
        f"- 当前分支：`{prepared['git']['branch']}`",
        f"- 当前 HEAD：`{prepared['git']['head']}`",
        f"- 基线提交：`{BASE_COMMIT}`",
        "",
        "### Prompt 哈希",
        "",
        "| Prompt | 基线 SHA256 | 候选 SHA256 | 字符数 |",
        "|---|---|---|---:|",
        f"| SYSTEM_PROMPT（Flat） | "
        f"`{BASELINE_SYSTEM_PROMPT_SHA256[:16]}…` | "
        f"`{prompt['flat_system_prompt_sha256'][:16]}…` | "
        f"— |",
        f"| EXTRACT_INSTRUCTION（Flat） | "
        f"`{BASELINE_EXTRACT_INSTRUCTION_SHA256[:16]}…` | "
        f"`{prompt['flat_extract_instruction_sha256'][:16]}…` | "
        f"— |",
        f"| LOGICAL_RETRY_INSTRUCTION | "
        f"`{BASELINE_LOGICAL_RETRY_SHA256[:16]}…` | "
        f"`{prompt['logical_retry_sha256'][:16]}…` | "
        f"— |",
        f"| LOGICAL_INPUT_INSTRUCTION | "
        f"`{prompt['baseline_logical']['sha256'][:16]}…` | "
        f"`{prompt['candidate_logical']['sha256'][:16]}…` | "
        f"{prompt['baseline_logical']['chars']} → "
        f"{prompt['candidate_logical']['chars']} |",
        "",
        f"Logical Prompt 字符增量："
        f"**{prompt['char_delta']:+d}**；"
        f"版本：`{prompt['candidate_logical']['version']}`。",
        "",
        "## 二、实验配置",
        "",
        "| 配置 | 基线 rpc=10 | 候选 rpc=10 |",
        "|---|---:|---:|",
        f"| model | {baseline_arm['extractor_config'].get('model')} | "
        f"{candidate_arm['extractor_config'].get('model')} |",
        f"| input_mode | {baseline_arm['extractor_config'].get('input_mode')} | "
        f"{candidate_arm['extractor_config'].get('input_mode')} |",
        f"| max_tokens | {baseline_arm['extractor_config'].get('max_tokens')} | "
        f"{candidate_arm['extractor_config'].get('max_tokens')} |",
        f"| rows_per_chunk | {baseline_arm['extractor_config'].get('rows_per_chunk')} | "
        f"{candidate_arm['extractor_config'].get('rows_per_chunk')} |",
        f"| max_split_depth | {baseline_arm['extractor_config'].get('max_split_depth')} | "
        f"{candidate_arm['extractor_config'].get('max_split_depth')} |",
        f"| temperature | {baseline_arm['extractor_config'].get('temperature')} | "
        f"{candidate_arm['extractor_config'].get('temperature')} |",
        "",
        "## 三、核心指标",
        "",
        "| 指标 | 基线 | 候选 | Δ |",
        "|---|---:|---:|---:|",
    ]

    for key in (
        "strict_matched",
        "precision",
        "recall",
        "f1",
        "matched_keys",
        "key_precision",
        "key_recall",
        "key_f1",
        "missing_count",
        "false_positive_count",
        "duplicate_count",
        "predicted_count",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "request_count",
        "elapsed_seconds",
    ):
        lines.append(
            f"| {key} | {baseline_block.get(key)} | "
            f"{candidate_block.get(key)} | "
            f"{delta.get(key)} |"
        )

    lines += [
        "",
        "## 四、分块、截断与空响应",
        "",
        "| 指标 | 基线 | 候选 |",
        "|---|---:|---:|",
        f"| API 请求数 | {baseline_chunks['api_request_count']} | "
        f"{chunk_analysis['api_request_count']} |",
        f"| 截断响应数 | {baseline_chunks['truncated_request_count']} | "
        f"{chunk_analysis['truncated_request_count']} |",
        f"| 空叶子响应数 | {baseline_chunks['empty_leaf_response_count']} | "
        f"{chunk_analysis['empty_leaf_response_count']} |",
        f"| 解析失败响应数 | {baseline_chunks['parse_failure_request_count']} | "
        f"{chunk_analysis['parse_failure_request_count']} |",
        "",
        "### 候选 Chunk 明细",
        "",
        "| request | parent | rows | leaf | 状态 | 参数数 | 空响应 |",
        "|---:|---:|---|---|---|---:|---|",
    ]

    for chunk in chunk_analysis["chunks"]:
        lines.append(
            f"| {chunk['request_index']} | "
            f"{chunk['parent_request_index']} | "
            f"{chunk['row_start']}–{chunk['row_end']} | "
            f"{chunk['leaf']} | "
            f"{chunk['error'] or 'stop'} | "
            f"{chunk['param_count']} | "
            f"{chunk['empty_response']} |"
        )

    lines += [
        "",
        "## 五、参数族覆盖",
        "",
        "| 参数族 | Gold | 基线严格 | 候选严格 | 基线 Key | 候选 Key |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, group in groups.items():
        base = group["baseline"]
        cand = group["candidate"]
        lines.append(
            f"| {name} | {base['gold_records']} | "
            f"{base['strict_matched']} | "
            f"{cand['strict_matched']} | "
            f"{base['matched_keys']} | "
            f"{cand['matched_keys']} |"
        )

    lines += [
        "",
        f"- UVLO 严格覆盖："
        f"{baseline_special['uvlo']['strict_matched']}/"
        f"{baseline_special['uvlo']['gold_records']} → "
        f"{candidate_special['uvlo']['strict_matched']}/"
        f"{candidate_special['uvlo']['gold_records']}",
        f"- VDO 严格覆盖："
        f"{baseline_special['vdo']['strict_matched']}/"
        f"{baseline_special['vdo']['gold_records']} → "
        f"{candidate_special['vdo']['strict_matched']}/"
        f"{candidate_special['vdo']['gold_records']}",
        "",
        "## 六、逐条变化",
        "",
        f"- 恢复为严格匹配："
        f"{len(changes['recovered_strict'])} 条",
        f"- 严格匹配回退："
        f"{len(changes['regressed_strict'])} 条",
        f"- 状态变化："
        f"{len(changes['status_changes'])} 条",
        f"- 候选键匹配但字段错误："
        f"{len(changes['candidate_field_errors'])} 条",
        "",
    ]

    if changes["status_changes"]:
        lines += [
            "| ID | 类型 | 基线 | 候选 | 候选失败字段 |",
            "|---|---|---|---|---|",
        ]

        for row in changes["status_changes"]:
            lines.append(
                f"| {row['logical_row_id']} | "
                f"{row['value_type']} | "
                f"{_status_cell(row['baseline_status'])} | "
                f"{_status_cell(row['candidate_status'])} | "
                f"{', '.join(row['candidate_failed_fields']) or '—'} |"
            )

        lines.append("")

    if changes["candidate_field_errors"]:
        lines += [
            "候选键匹配但字段错误：",
            "",
            "| ID | 类型 | 失败字段 |",
            "|---|---|---|",
        ]

        for item in changes["candidate_field_errors"]:
            lines.append(
                f"| {item['logical_row_id']} | "
                f"{item['value_type']} | "
                f"{', '.join(item['fields'])} |"
            )

        lines.append("")

    lines += [
        "## 七、候选遗漏明细",
        "",
    ]

    if not changes["candidate_missing"]:
        lines.append("- 无缺失。")
    else:
        lines += [
            "| ID | 类型 | 参数名 |",
            "|---|---|---|",
        ]

        for item in changes["candidate_missing"]:
            lines.append(
                f"| {item['logical_row_id']} | "
                f"{item['value_type']} | "
                f"{item.get('parameter_name')} |"
            )

    lines += [
        "",
        "## 八、结论与限制",
        "",
        "- 本阶段只验证 Logical Prompt 完整性优化。",
        "- 单次真实运行，不构成统计显著性结论。",
        "- temperature 使用 provider_default，"
        "未显式固定随机种子。",
        "- 未启用 LangGraph Retry。",
        "- 只运行 Logical 单臂；Flat 基线未重跑。",
        "- 是否进入下一阶段需由用户根据本报告决定。",
        "- 未进入 B4。",
        "",
    ]

    return "\n".join(lines)


def write_tracking_csv(path, rows):
    fields = [
        "logical_row_id",
        "value_type",
        "parameter_name",
        "baseline_status",
        "baseline_strict",
        "baseline_failed_fields",
        "candidate_status",
        "candidate_strict",
        "candidate_failed_fields",
        "recovered_strict",
        "regressed_strict",
    ]

    with Path(path).open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

        for row in rows:
            output = {
                key: row.get(key)
                for key in fields
            }
            output["baseline_failed_fields"] = "|".join(
                row.get("baseline_failed_fields") or []
            )
            output["candidate_failed_fields"] = "|".join(
                row.get("candidate_failed_fields") or []
            )
            writer.writerow(output)


def write_chunk_csv(path, chunks):
    fields = [
        "node_id",
        "request_index",
        "parent_request_index",
        "depth",
        "row_start",
        "row_end",
        "leaf",
        "truncated",
        "error",
        "parse_success",
        "empty_response",
        "param_count",
        "ancestor_truncated",
    ]

    with Path(path).open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

        for chunk in chunks:
            writer.writerow({
                key: chunk.get(key)
                for key in fields
            })


def run_candidate(prepared):
    os.environ["LLM_ROWS_PER_CHUNK"] = str(
        ROWS_PER_CHUNK
    )
    os.environ["LLM_MAX_TOKENS"] = str(
        MAX_TOKENS
    )
    os.environ["LLM_MAX_SPLIT_DEPTH"] = str(
        MAX_SPLIT_DEPTH
    )

    return run_arm(
        "logical",
        [copy.deepcopy(prepared["logical_table"])],
        prepared["flat_table"],
        prepared["logical_rows"],
        prepared["raw_structure"],
        prepared["inputs"]["pdf_sha256"],
        "logical",
        prepared["gold"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "B3.4b：Logical Prompt 完整性优化"
            "（默认只预检）"
        )
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
    )
    parser.add_argument(
        "--b34a-preflight",
        default=str(DEFAULT_B34A_PREFLIGHT),
    )
    parser.add_argument(
        "--logical-rows",
        default=str(DEFAULT_LOGICAL_ROWS),
    )
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD),
    )
    parser.add_argument(
        "--raw-structure",
        default=str(DEFAULT_RAW_STRUCTURE),
    )
    parser.add_argument(
        "--table-cache-dir",
        default=str(DEFAULT_CACHE_DIR),
    )
    parser.add_argument(
        "--table-id",
        type=int,
        default=DEFAULT_TABLE_ID,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="显式允许调用真实 DeepSeek API",
    )
    args = parser.parse_args(argv)

    prepared = build_prepared(args)

    if not args.execute:
        stamp = utc_now().strftime(
            "%Y%m%dT%H%M%SZ"
        )
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else DEFAULT_OUTPUT_ROOT
            / "preflight"
            / stamp
        )

        if output_dir.exists():
            raise ExperimentError(
                f"拒绝覆盖已有目录: {output_dir}"
            )

        output_dir.mkdir(
            parents=True, exist_ok=False
        )
        preflight_path = write_json_exclusive(
            output_dir / "preflight.json",
            preflight_payload(prepared, False),
        )
        plan_path = write_json_exclusive(
            output_dir / "plan.json",
            plan_payload(prepared, False),
        )
        gold_path = write_json_exclusive(
            output_dir / "gold_standard.used.json",
            prepared["gold"],
        )

        print("=" * 64)
        print("B3.4b 安全预检通过（未调用 API）")
        print("=" * 64)
        print("输出目录:", output_dir)
        print("预检:", preflight_path)
        print("计划:", plan_path)
        print("Gold:", gold_path)
        print(
            "分支:", prepared["git"]["branch"]
        )
        print(
            "HEAD:", prepared["git"]["head"]
        )
        print(
            "Logical Prompt:",
            prepared["prompt"]["baseline_logical"]["chars"],
            "→",
            prepared["prompt"]["candidate_logical"]["chars"],
            "字符，版本",
            prepared["prompt"]["candidate_logical"]["version"],
        )

        for name, passed in prepared["checks"].items():
            print(
                ("PASS" if passed else "FAIL"),
                name,
            )

        return 0

    if not os.getenv("LLM_API_KEY"):
        raise ExperimentError(
            "真实实验缺少 LLM_API_KEY"
        )

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / stamp
    )

    if output_dir.exists():
        raise ExperimentError(
            f"拒绝覆盖已有目录: {output_dir}"
        )

    output_dir.mkdir(
        parents=True, exist_ok=False
    )
    plan_path = write_json_exclusive(
        output_dir / "plan.json",
        plan_payload(prepared, True),
    )
    gold_path = write_json_exclusive(
        output_dir / "gold_standard.used.json",
        prepared["gold"],
    )
    preflight_path = write_json_exclusive(
        output_dir / "preflight.json",
        preflight_payload(prepared, True),
    )
    prompt_diff = git_output(
        "diff",
        BASE_COMMIT,
        "--",
        "backend/app/extraction/prompts.py",
    )
    prompt_diff_path = write_text_exclusive(
        output_dir / "prompt.diff",
        prompt_diff + "\n",
    )

    baseline_sha_before = prepared["baseline"][
        "sha256"
    ]

    try:
        candidate_arm = run_candidate(prepared)
    except Exception as exc:
        write_json_exclusive(
            output_dir / "failure.json",
            {
                "schema": "b34b_failure/v1",
                "stage": "B3.4b",
                "created_at": utc_now().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "baseline_sha256": baseline_sha_before,
            },
        )
        raise

    baseline_sha_after = sha256_file(
        prepared["baseline"]["path"]
    )

    if baseline_sha_after != baseline_sha_before:
        raise ExperimentError(
            "基线实验文件在运行期间发生变化"
        )

    baseline_arm = prepared["baseline_arm"]
    refs = prepared["inputs"]["logical_row_ids"]
    candidate_replay = replay_chunk_tree(
        "candidate-logical",
        candidate_arm.get("raw_llm_responses", []),
        refs,
        ROWS_PER_CHUNK,
        MAX_SPLIT_DEPTH,
    )
    chunk_analysis = chunk_summary(
        candidate_replay,
        ROWS_PER_CHUNK,
        MAX_SPLIT_DEPTH,
    )
    baseline_details = detail_index(baseline_arm)
    candidate_details = detail_index(candidate_arm)
    gold_records = prepared["gold"][
        "expected_parameters"
    ]
    all_rows = build_record_rows(
        gold_records,
        baseline_details,
        candidate_details,
    )
    groups = build_group_comparison(
        baseline_arm,
        candidate_arm,
    )
    changes = record_changes(
        all_rows,
        candidate_arm,
    )
    checks = dict(prepared["checks"])
    checks.update({
        "基线文件运行前后未变化": True,
        "候选实际 rows_per_chunk=10": (
            candidate_arm["extractor_config"].get(
                "rows_per_chunk"
            )
            == ROWS_PER_CHUNK
        ),
        "候选实际 max_tokens=6000": (
            candidate_arm["extractor_config"].get(
                "max_tokens"
            )
            == MAX_TOKENS
        ),
        "候选实际 max_split_depth=6": (
            candidate_arm["extractor_config"].get(
                "max_split_depth"
            )
            == MAX_SPLIT_DEPTH
        ),
        "候选实际模型与基线一致": (
            candidate_arm["extractor_config"].get(
                "model"
            )
            == prepared["baseline_config"].get(
                "model"
            )
        ),
        "未调用 LangGraph Retry": True,
    })

    baseline_block = metric_block(baseline_arm)
    candidate_block = metric_block(candidate_arm)
    comparison = {
        "baseline_logical": baseline_block,
        "candidate_logical": candidate_block,
        "delta_candidate_minus_baseline": metric_delta(
            baseline_block, candidate_block
        ),
        "special": {
            "baseline": baseline_arm["evaluation"][
                "special"
            ],
            "candidate": candidate_arm["evaluation"][
                "special"
            ],
        },
        "groups": groups,
    }

    result = {
        "schema": "b34b_result/v1",
        "stage": "B3.4b",
        "generated_at": utc_now().isoformat(),
        "api_call": True,
        "plan": plan_payload(prepared, True),
        "gold_standard": prepared["gold"],
        "baseline": {
            **prepared["baseline"],
            "chunk_summary": prepared[
                "baseline_chunks"
            ],
        },
        "prompt": prepared["prompt"],
        "arms": [candidate_arm],
        "candidate_chunk_analysis": chunk_analysis,
        "comparison": comparison,
        "record_changes": changes,
        "record_comparison": all_rows,
        "checks": checks,
        "limitations": [
            "单次真实运行，不构成统计显著性结论。",
            "temperature 使用 provider_default，"
            "未显式固定随机种子。",
            "未启用 LangGraph Retry。",
            "只运行 Logical 单臂；Flat 基线未重跑。",
        ],
    }

    report = build_markdown(
        prepared,
        baseline_arm,
        candidate_arm,
        chunk_analysis,
        groups,
        changes,
        all_rows,
    )
    result["report_markdown"] = report

    result_path = write_json_exclusive(
        output_dir / "b34b_result.json",
        result,
    )
    report_path = write_text_exclusive(
        output_dir / "b34b_report.md",
        report,
    )
    tracking_path = output_dir / "b34b_tracking.csv"
    write_tracking_csv(tracking_path, all_rows)
    chunk_path = output_dir / "b34b_chunks.csv"
    write_chunk_csv(
        chunk_path,
        chunk_analysis["chunks"],
    )

    print("=" * 64)
    print("B3.4b 真实实验完成")
    print("=" * 64)
    print("输出目录:", output_dir)
    print("结果:", result_path)
    print("报告:", report_path)
    print("追踪 CSV:", tracking_path)
    print("Chunk CSV:", chunk_path)
    print("Prompt Diff:", prompt_diff_path)
    print(
        "baseline:",
        {
            "precision": baseline_block["precision"],
            "recall": baseline_block["recall"],
            "f1": baseline_block["f1"],
            "strict": baseline_block["strict_matched"],
            "key": baseline_block["matched_keys"],
            "tokens": baseline_block["total_tokens"],
            "requests": baseline_block["request_count"],
            "elapsed": baseline_block["elapsed_seconds"],
        },
    )
    print(
        "candidate:",
        {
            "precision": candidate_block["precision"],
            "recall": candidate_block["recall"],
            "f1": candidate_block["f1"],
            "strict": candidate_block["strict_matched"],
            "key": candidate_block["matched_keys"],
            "tokens": candidate_block["total_tokens"],
            "requests": candidate_block["request_count"],
            "elapsed": candidate_block["elapsed_seconds"],
        },
    )
    print(
        "UVLO:",
        comparison["special"]["candidate"]["uvlo"][
            "strict_matched"
        ],
        "/",
        comparison["special"]["candidate"]["uvlo"][
            "gold_records"
        ],
    )
    print(
        "VDO:",
        comparison["special"]["candidate"]["vdo"][
            "strict_matched"
        ],
        "/",
        comparison["special"]["candidate"]["vdo"][
            "gold_records"
        ],
    )
    print(
        "chunk events: truncated="
        f"{chunk_analysis['truncated_request_count']}, "
        "empty_leaf="
        f"{chunk_analysis['empty_leaf_response_count']}, "
        "requests="
        f"{chunk_analysis['api_request_count']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

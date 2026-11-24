"""B3.7：受控 2x2 A/B 公共配置与输入装配。

设计约束：
- 四个臂只改变两个因子：输入结构（Flat / Logical）与 Prompt 族覆盖
  （P0 / P1）；
- 模型、温度、表格范围、分块、重试、Validator 模式选择逻辑一致；
- Gold 只读，不修改 B3.5b / B3.5c / B3.6 产物；
- 正式主实验只使用每份 PDF 的 Gold scope 表，保证四臂表格范围一致，
  同时把 API 预算控制在可复核范围内。
"""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS = BACKEND_ROOT / "scripts"
B37_DIR = PROJECT_ROOT / "data/outputs/experiments/B3.7"
PREFLIGHT_DIR = B37_DIR / "preflight"
RUNS_DIR = B37_DIR / "runs"
RESULTS_DIR = B37_DIR / "results"
REPORTS_DIR = B37_DIR / "reports"

B35B_DIR = PROJECT_ROOT / "data/outputs/experiments/B3.5b"
GOLD = B35B_DIR / "gold/gold.frozen.json"
MANIFEST = B35B_DIR / "manifest.json"
GOLD_RECEIPT = B35B_DIR / "gold/gold.frozen.receipt.json"

B35C_RUN = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5c/runs/flat_baseline_r1"
)
B35C_METRICS = (
    PROJECT_ROOT / "data/outputs/experiments/B3.5c/results/metrics.json"
)
B36_STRUCTURE = (
    PROJECT_ROOT / "data/outputs/experiments/B3.6/structure"
)
B35D_RAW = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5d/structure/raw"
)
B35D_FN = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5d/trace/fn_attribution.json"
)

PDF_DIR = PROJECT_ROOT / "data/samples/b35b"

# B3.7 只跑 Gold scope 表。该范围与 four-arm 完全一致；
# 非 Gold 表预测在 B3.5c 评分契约中本来就不计入 P/R/F1。
TABLE_SCOPE = "gold_scope_only"

# 当前环境模型。运行时以 .env 为准，并在 manifest 中记录实际响应模型。
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 6000
DEFAULT_ROWS_PER_CHUNK = 10
DEFAULT_MAX_SPLIT_DEPTH = 6
DEFAULT_TRANSPORT_RETRIES = 0
FORMAL_MAX_RETRIES = 1
PREFLIGHT_MAX_RETRIES = 0

ARMS = {
    "A": {
        "label": "Flat + P0",
        "input_mode": "flat",
        "prompt_family": "P0",
        "family_coverage": False,
        "evidence_validation_mode": "legacy",
    },
    "B": {
        "label": "Logical + P0",
        "input_mode": "logical",
        "prompt_family": "P0",
        "family_coverage": False,
        "evidence_validation_mode": "recovered",
    },
    "C": {
        "label": "Flat + P1",
        "input_mode": "flat",
        "prompt_family": "P1",
        "family_coverage": True,
        "evidence_validation_mode": "legacy",
    },
    "D": {
        "label": "Logical + P1",
        "input_mode": "logical",
        "prompt_family": "P1",
        "family_coverage": True,
        "evidence_validation_mode": "recovered",
    },
}

PROMPT_MARKERS = {
    "family_coverage": (
        "低频参数族的强制覆盖与自检",
        "低频参数族补充要求",
    ),
    "logical_input": "本次输入中的部分或全部表格来自 Logical 恢复结果",
    "extract_instruction": "请从候选表格中提取 LDO 电气参数",
}


class B37Error(RuntimeError):
    """B3.7 输入或运行前置条件不满足。"""


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if exclusive and path.exists():
        raise B37Error(f"拒绝覆盖已有文件: {path}")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def resolve_arm(name):
    arm = ARMS.get(str(name).upper())

    if arm is None:
        raise B37Error(f"未知实验臂: {name}")

    return {
        "arm": str(name).upper(),
        **deepcopy(arm),
    }


def load_selected_tables(sample_id):
    path = B35C_RUN / sample_id / "selected_tables.json"

    if not path.is_file():
        raise B37Error(f"缺少 B3.5c selected_tables: {path}")

    return load_json(path), path


def load_logical_rows(sample_id):
    path = B36_STRUCTURE / f"{sample_id}.logical_rows.v1.json"

    if not path.is_file():
        raise B37Error(f"缺少 B3.6 logical_rows: {path}")

    return load_json(path), path


def load_raw_structure(sample_id):
    path = B35D_RAW / f"{sample_id}.raw_structure.v1.json"

    if not path.is_file():
        raise B37Error(f"缺少 B3.5d raw_structure: {path}")

    return load_json(path), path


def gold_scope_table(sample, selected_tables):
    scope_ids = [int(value) for value in sample["gold_scope_table_ids"]]

    if len(scope_ids) != 1:
        raise B37Error(
            f"{sample['sample_id']}: B3.7 只支持单一 Gold scope 表，"
            f"实际为 {scope_ids}"
        )

    table_id = scope_ids[0]
    matches = [
        table
        for table in selected_tables.get("tables", [])
        if int(table.get("table_id")) == table_id
    ]

    if len(matches) != 1:
        raise B37Error(
            f"{sample['sample_id']}: B3.5c selected_tables 中 "
            f"table_id={table_id} 命中 {len(matches)} 次"
        )

    return table_id, deepcopy(matches[0])


def validate_logical_table(sample_id, logical_rows, table_id):
    if int(logical_rows.get("table_id", -1)) != table_id:
        raise B37Error(
            f"{sample_id}: B3.6 logical_rows table_id="
            f"{logical_rows.get('table_id')} 与 Gold scope "
            f"{table_id} 不一致"
        )

    if logical_rows.get("schema") != "table_logical_rows/v1":
        raise B37Error(
            f"{sample_id}: 不支持的 logical_rows schema: "
            f"{logical_rows.get('schema')!r}"
        )

    if logical_rows.get("logical_rows_version") != 1:
        raise B37Error(
            f"{sample_id}: 不支持的 logical_rows_version: "
            f"{logical_rows.get('logical_rows_version')!r}"
        )

    if not logical_rows.get("logical_rows"):
        raise B37Error(f"{sample_id}: logical_rows 为空")


def build_sample_inputs(sample):
    sample_id = sample["sample_id"]
    selected, selected_path = load_selected_tables(sample_id)
    table_id, flat_table = gold_scope_table(sample, selected)

    record = {
        "sample_id": sample_id,
        "vendor": sample["vendor"],
        "model": sample["model"],
        "pdf_path": sample["file"],
        "pdf_sha256": sample["pdf_sha256"],
        "gold_scope_table_ids": [table_id],
        "table_scope": TABLE_SCOPE,
        "flat_table": flat_table,
        "input_sources": {
            "b35c_selected_tables": {
                "path": str(selected_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(selected_path),
            },
        },
        "logical_rows": None,
        "raw_structure": None,
    }

    if sample_id in {"adi_lt3045", "diodes_ap2112", "onsemi_ncp170",
                     "renesas_isl80510"}:
        logical_rows, logical_path = load_logical_rows(sample_id)
        raw_structure, raw_path = load_raw_structure(sample_id)
        validate_logical_table(sample_id, logical_rows, table_id)
        record["logical_rows"] = logical_rows
        record["raw_structure"] = raw_structure
        record["input_sources"]["b36_logical_rows"] = {
            "path": str(logical_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(logical_path),
        }
        record["input_sources"]["b35d_raw_structure"] = {
            "path": str(raw_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(raw_path),
        }

    return record


def current_model_config():
    return {
        "requested_model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
        "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": int(
            os.getenv("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
        ),
        "rows_per_chunk": int(
            os.getenv("LLM_ROWS_PER_CHUNK", str(DEFAULT_ROWS_PER_CHUNK))
        ),
        "max_split_depth": int(
            os.getenv("LLM_MAX_SPLIT_DEPTH", str(DEFAULT_MAX_SPLIT_DEPTH))
        ),
        "transport_max_retries": DEFAULT_TRANSPORT_RETRIES,
        "temperature_source": (
            "b37_runner_injects_temperature_0.0"
        ),
        "seed": None,
    }


def source_fingerprints():
    files = [
        "backend/app/extraction/prompts.py",
        "backend/app/extraction/deepseek_extractor.py",
        "backend/app/extraction/input_mode.py",
        "backend/app/extraction/logical_table_adapter.py",
        "backend/app/extraction/logical_source.py",
        "backend/app/extraction/evidence.py",
        "backend/app/parsers/structure_recovery.py",
        "backend/app/parsers/table_selector.py",
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

    return {
        path: (
            sha256_file(PROJECT_ROOT / path)
            if (PROJECT_ROOT / path).is_file()
            else None
        )
        for path in files
    }


def git_state():
    import subprocess

    def run(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip()
        except OSError:
            return None

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def experiment_config():
    model = current_model_config()

    return {
        "schema": "b37_experiment_config/v1",
        "table_scope": TABLE_SCOPE,
        "arms": deepcopy(ARMS),
        "model": model,
        "formal_max_retries": FORMAL_MAX_RETRIES,
        "preflight_max_retries": PREFLIGHT_MAX_RETRIES,
        "score_contract": {
            "scorer": "b35c-flat-v1",
            "normalization": "b35c-norm-v1",
            "gold": str(GOLD.relative_to(PROJECT_ROOT)),
            "gold_sha256": sha256_file(GOLD),
        },
        "input_sources": {
            "flat": "B3.5c selected_tables.json (gold scope table only)",
            "logical": "B3.6 logical_rows.v1.json + B3.5d raw_structure",
        },
        "prompt_versions": {
            "P0": "PROMPT_FAMILY_COVERAGE=0 (B3.5c original)",
            "P1": "PROMPT_FAMILY_COVERAGE=1 (B3.6 family coverage)",
        },
        "randomness": {
            "temperature": model["temperature"],
            "seed": None,
            "note": (
                "temperature 由 B3.7 运行器在每次请求中显式注入 0.0；"
                "provider 仍可能返回不同的 response id/created；"
                "正式主实验每臂只运行一次，不构成统计显著结论。"
            ),
        },
        "git": git_state(),
        "source_fingerprints": source_fingerprints(),
    }


def expected_chunks(table, *, logical_rows=None, table_ids=None):
    """粗略估计正常（无截断）分块数，仅用于预检诊断。"""

    from app.extraction.logical_table_adapter import adapt_logical_rows

    if logical_rows is not None:
        adapted = adapt_logical_rows(
            logical_rows,
            raw_structure=None,
            table_ids=table_ids,
        )
        rows = adapted["tables"][0].get("rows", [])
    else:
        rows = table.get("rows", [])

    return (len(rows) + DEFAULT_ROWS_PER_CHUNK - 1) // DEFAULT_ROWS_PER_CHUNK


def prompt_flags(system_prompt):
    text = str(system_prompt or "")

    return {
        "extract_instruction": (
            PROMPT_MARKERS["extract_instruction"] in text
        ),
        "logical_input": (
            PROMPT_MARKERS["logical_input"] in text
        ),
        "family_coverage": any(
            marker in text
            for marker in PROMPT_MARKERS["family_coverage"]
        ),
    }

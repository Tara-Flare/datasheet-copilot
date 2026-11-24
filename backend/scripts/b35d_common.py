"""B3.5d 离线分析公共工具（只读，不调用 LLM）。

本模块只消费 B3.5c 运行产物、B3.5b 冻结 Gold 和 B3.5d 离线结构恢复结果。
不修改生产代码、Gold 或 B3.5c 结果。
"""

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5c/runs/flat_baseline_r1"
)
GOLD_PATH = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5b/gold/gold.frozen.json"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5b/manifest.json"
)
B35C_METRICS = (
    PROJECT_ROOT / "data/outputs/experiments/B3.5c/results/metrics.json"
)
B35C_ERRORS = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5c/results/metrics.errors.json"
)
EXPERIMENT_DIR = (
    PROJECT_ROOT / "data/outputs/experiments/B3.5d"
)
RAW_DIR = EXPERIMENT_DIR / "structure/raw"
LOGICAL_DIR = EXPERIMENT_DIR / "structure/logical"
TRACE_DIR = EXPERIMENT_DIR / "trace"
REPORT_DIR = EXPERIMENT_DIR / "reports"
VALIDATOR_DIR = EXPERIMENT_DIR / "validator"


IDENTITY_KEYWORDS = {
    "shutdown_current": ["shutdown", "standby"],
    "ground_current": ["quiescent", "ground pin", "ground current"],
    "ground_current_load": ["ground current vs load"],
    "accuracy": ["accuracy", "output voltage"],
    "line_regulation": ["line regulation"],
    "load_regulation": ["load regulation"],
    "dropout": ["dropout"],
    "current_limit": ["current limit"],
    "internal_current_limit": ["internal current limit"],
    "programmable_current_limit": ["programmable current limit"],
    "current_limit_programming_scale_factor": [
        "scale factor",
        "programming scale",
    ],
    "short_circuit_current": ["short circuit", "short-circuit", "short current"],
    "noise": ["noise"],
    "psrr": ["ripple rejection", "rejection ratio", "psrr"],
    "input_voltage": ["input voltage"],
    "output_voltage": ["output voltage"],
    "output_current": ["output current", "maximum output current"],
    "operating_temperature": ["operating temperature"],
    "uvlo_threshold": ["uvlo", "trip point"],
    "uvlo_hysteresis": ["uvlo", "hysteresis"],
}

# SYSTEM_PROMPT 中“重点提取”的参数；用于区分提示词覆盖与纯执行遗漏。
FLAT_PROMPT_FOCUS = {
    "input_voltage",
    "output_voltage",
    "output_current",
    "dropout",
    "ground_current",
    "psrr",
    "accuracy",
    "operating_temperature",
}

STOPWORDS = {
    "the",
    "of",
    "in",
    "with",
    "and",
    "for",
    "to",
    "a",
    "an",
    "note",
    "notes",
    "min",
    "typ",
    "max",
    "typical",
    "minimum",
    "maximum",
}

NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
UNIT_RE = re.compile(
    r"(?:µA|μA|mA|nA|A|µV|μV|mV|kV|V|µVRMS|μVRMS|mVRMS|"
    r"VRMS|nV|dB|%|%/A|%/V|ppm/°C|mA·kΩ|kΩ|Ω|°C|nA/V|µV/V|"
    r"nV/√Hz|nARMS|µs|ms|s|W|%/W|Vrms)"
)


def load_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def normalize_number(value):
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def number_tokens(text) -> list[str]:
    return NUMBER_RE.findall(str(text or ""))


def unit_tokens(text) -> list[str]:
    return UNIT_RE.findall(str(text or ""))


def normalize_unit_token(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("Ω", "ohm").replace("ω", "ohm")
    text = text.replace("·", "*").replace("⋅", "*")
    text = text.replace(" ", "")
    return text


def normalize_unit_text(value: str) -> str:
    tokens = unit_tokens(value)

    if tokens:
        return " ".join(
            normalize_unit_token(token) for token in tokens
        )

    return normalize_unit_token(value)


def unit_contains(field_value, gold_unit) -> bool:
    target = normalize_unit_token(gold_unit)
    text = normalize_unit_token(field_value)

    if not target:
        return False

    if not text:
        return False

    if target in text:
        return True

    # 复合单位 mA·kΩ：Docling 常写成 "mA · kΩ"。
    compact_target = target.replace("*", "")
    compact_text = text.replace("*", "")

    return compact_target in compact_text


def value_contains(field_value, gold_value) -> bool:
    target = normalize_number(gold_value)

    if target is None:
        return False

    for token in number_tokens(field_value):
        parsed = normalize_number(token)

        if parsed is not None and parsed == target:
            return True

    return False


def value_is_single(field_value) -> bool:
    return len(number_tokens(field_value)) == 1


def token_set(value) -> set[str]:
    text = str(value or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)

    return {
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    }


def symbol_similarity(left, right) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)

    if not left_tokens or not right_tokens:
        return 0.0

    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return 1.0

    overlap = len(left_tokens & right_tokens)

    return overlap / len(left_tokens | right_tokens)


def identity_matches_text(identity: str, text: str) -> bool:
    lowered = str(text or "").lower()

    for keyword in IDENTITY_KEYWORDS.get(identity, []):
        if keyword in lowered:
            return True

    return False


def recovered_row_matches_identity(gold, row) -> tuple[bool, float]:
    left = gold.get("parameter_symbol") or ""
    right = " ".join(
        str(row.get(key) or "")
        for key in ("parameter_symbol", "parameter_name")
    )
    similarity = symbol_similarity(left, right)

    if similarity >= 0.6:
        return True, similarity

    if identity_matches_text(gold["identity"], right):
        return True, similarity

    return False, similarity


def parse_log_events(sample_id: str) -> list[dict]:
    """按请求顺序还原 run.log 中的每个 API 调用上下文。"""

    log_path = (RUN_ROOT / "run.log").resolve()

    if not log_path.is_file():
        raise FileNotFoundError(f"缺少运行日志: {log_path}")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    events = []
    current_sample = None
    context = None

    for line in lines:
        match = re.search(r"\[baseline\] (\w+)$", line.strip())

        if match:
            current_sample = match.group(1)
            context = None
            continue

        match = re.search(
            r"提取行块: table_id=(\d+), 深度=(\d+), "
            r"行数=(\d+), 输入字符数=(\d+), "
            r"起始原始行号=(\d+)",
            line,
        )

        if match:
            context = {
                "kind": "extract",
                "table_id": int(match.group(1)),
                "depth": int(match.group(2)),
                "rows": int(match.group(3)),
                "chars": int(match.group(4)),
                "start": int(match.group(5)),
            }
            continue

        match = re.search(
            r"纠错批次: 参数数量=(\d+), 深度=(\d+), "
            r"table_ids=\[([^\]]*)\]",
            line,
        )

        if match:
            context = {
                "kind": "retry",
                "params": int(match.group(1)),
                "depth": int(match.group(2)),
                "table_ids": [
                    int(value.strip())
                    for value in match.group(3).split(",")
                    if value.strip()
                ],
            }
            continue

        match = re.search(
            r"DeepSeek 请求完成: finish_reason=(\w+), "
            r"prompt_tokens=(\d+), completion_tokens=(\d+), "
            r"reasoning_tokens=(\d+)",
            line,
        )

        if (
            match
            and current_sample == sample_id
            and context is not None
        ):
            event = dict(context)
            event.update(
                {
                    "finish": match.group(1),
                    "prompt_tokens": int(match.group(2)),
                    "completion_tokens": int(match.group(3)),
                    "reasoning_tokens": int(match.group(4)),
                }
            )
            events.append(event)
            context = None

    return events


def load_calls(sample_id: str) -> tuple[list[dict], list[dict]]:
    sample_dir = RUN_ROOT / sample_id
    raw = load_json(sample_dir / "raw_responses.json")[
        "raw_responses"
    ]
    api = load_json(sample_dir / "api_responses.json")["calls"]

    return raw, api


def attach_events(sample_id: str) -> list[dict]:
    events = parse_log_events(sample_id)
    raw, api = load_calls(sample_id)

    if not (len(events) == len(raw) == len(api)):
        raise ValueError(
            f"{sample_id}: 事件/响应/调用数量不一致 "
            f"{len(events)}/{len(raw)}/{len(api)}"
        )

    attached = []

    for index, (event, response, call) in enumerate(
        zip(events, raw, api)
    ):
        if event["finish"] != response["finish_reason"]:
            raise ValueError(
                f"{sample_id} 调用 {index}: finish_reason 不一致"
            )

        item = dict(event)
        item["index"] = index
        item["response_id"] = call.get("response_id")
        item["response_model"] = call.get("response_model")
        item["system_fingerprint"] = call.get("system_fingerprint")
        item["content"] = response.get("content") or ""
        item["content_length"] = len(item["content"])
        item["error"] = response.get("error")
        item["attempt"] = response.get("attempt")

        try:
            parsed = json.loads(item["content"])
            item["json_valid"] = True
            item["parameters"] = parsed.get("parameters", [])
        except (json.JSONDecodeError, TypeError):
            item["json_valid"] = False
            item["parameters"] = []

        attached.append(item)

    return attached


def table_rows(sample_id: str, table_id: int) -> dict:
    selected = load_json(
        RUN_ROOT / sample_id / "selected_tables.json"
    )

    for table in selected["tables"]:
        if table["table_id"] == table_id:
            return table

    return {}


def load_gold() -> dict:
    return load_json(GOLD_PATH)


def load_manifest() -> dict:
    return load_json(MANIFEST_PATH)


def load_logical(sample_id: str, table_id: int):
    names = {
        ("adi_lt3045", 4): "adi_lt3045.table4",
        ("diodes_ap2112", 4): "diodes_ap2112.table4",
        ("onsemi_ncp170", 4): "onsemi_ncp170.table4",
        ("renesas_isl80510", 7): "renesas_isl80510.table7",
    }
    name = names.get((sample_id, table_id))

    if not name:
        return None

    path = LOGICAL_DIR / f"{name}.logical_rows.v1.json"

    if not path.is_file():
        return None

    return load_json(path)


def load_raw_structure(sample_id: str):
    path = RAW_DIR / f"{sample_id}.raw_structure.v1.json"

    if not path.is_file():
        return None

    return load_json(path)


def row_cells_for_gold(sample_id: str, table_id: int, row_index: int):
    table = table_rows(sample_id, table_id)

    if not table:
        return None

    rows = table.get("rows", [])

    if not isinstance(row_index, int) or not (
        0 <= row_index < len(rows)
    ):
        return None

    return rows[row_index]


def column_index(table: dict, aliases: set[str]):
    matches = []

    for index, column in enumerate(table.get("columns", [])):
        text = re.sub(r"\([^)]*\)", " ", str(column).lower())
        normalized = re.sub(r"[^a-z]", "", text)

        if normalized in aliases:
            matches.append(index)

    return matches[0] if len(matches) == 1 else None


def value_column_index(table: dict, value_type: str):
    if value_type == "min":
        return column_index(table, {"min", "minimum"})
    if value_type == "typ":
        return column_index(table, {"typ", "typical"})
    if value_type == "max":
        return column_index(table, {"max", "maximum"})
    return None


def unit_column_index(table: dict):
    return column_index(table, {"unit", "units"})

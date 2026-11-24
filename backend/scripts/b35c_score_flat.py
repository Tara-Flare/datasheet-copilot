"""B3.5c 跨型号 Flat Baseline 评分与错误分析（只读，不调用 LLM）。

匹配定义（版本 b35c-flat-v1）：
- 只评估 Gold scope table 内、且身份属于该样本 Gold 身份集合的预测；
- out_of_scope_table：来自非 Gold scope 表的预测，不计入 P/R；
- out_of_scope_parameter：Gold 表内、来源行不属于 Gold 且身份不在 Gold
  身份集合中的预测，不计入 false positive；
- TP：一对一匹配，参数身份 + value_type + 数值 + 单位全部一致；
- FP：in-scope 预测中未成为 TP 的记录；
- FN：Gold 中未被 TP 匹配的记录；
- 测试条件、来源行与严格匹配单独统计，不混入主 P/R/F1。

评分归一化代码在本文件内冻结，带版本号，避免与生产语义耦合。
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from scipy.optimize import linear_sum_assignment

import numpy as np


SCORER_VERSION = "b35c-flat-v1"
NORMALIZATION_VERSION = "b35c-norm-v1"

IDENTITY_ALIASES = {
    "accuracy": "accuracy",
    "nominalaccuracy": "accuracy",
    "accuracyovertemperature": "accuracy",
    "outputvoltageaccuracy": "accuracy",
    "outputvoltage": "accuracy",
    "outputvoltageaccuracyovertemperature": "accuracy",
    "quiescentcurrent": "ground_current",
    "quiescentcurrentinshutdown": "shutdown_current",
    "groundpincurrent": "ground_current",
    "groundpincurrentinshutdown": "shutdown_current",
    "iq": "ground_current",
    "groundcurrent": "ground_current",
    "groundcurrentindropout": "ground_current",
    "groundcurrentvsloadcurrent": "ground_current_load",
    "dropoutvoltage": "dropout",
    "vdo": "dropout",
    "outputcurrentlimit": "current_limit",
    "currentlimit": "current_limit",
    "internalcurrentlimit": "internal_current_limit",
    "programmablecurrentlimit": "programmable_current_limit",
    "currentlimitprogrammingscalefactor": (
        "current_limit_programming_scale_factor"
    ),
    "outputcurrentlimitprogrammingscalefactor": (
        "current_limit_programming_scale_factor"
    ),
    "icl": "current_limit",
    "shortcircuitcurrentlimit": "short_circuit_current",
    "outputshortcircuitcurrent": "short_circuit_current",
    "shortcircuitcurrent": "short_circuit_current",
    "shortcurrentlimit": "short_circuit_current",
    "isc": "short_circuit_current",
    "shutdowncurrent": "shutdown_current",
    "standbycurrent": "shutdown_current",
    "istd": "shutdown_current",
    "lineregulation": "line_regulation",
    "linereg": "line_regulation",
    "loadregulation": "load_regulation",
    "loadreg": "load_regulation",
    "powersupplyrejectionratio": "psrr",
    "inputsupplyripplerejection": "psrr",
    "ripplerejection": "psrr",
    "psrr": "psrr",
    "outputvoltagenoise": "noise",
    "outputnoisevoltage": "noise",
    "rmsoutputnoise": "noise",
    "outputnoise": "noise",
    "noise": "noise",
    "uvlothreshold": "uvlo_threshold",
    "uvlohysteresis": "uvlo_hysteresis",
    "uvlo": "uvlo",
    "inputvoltage": "input_voltage",
    "outputcurrent": "output_current",
    "operatingtemperature": "operating_temperature",
}

NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
VAL_TYPE = {"min", "typ", "max", "nominal"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def normalize_identity(value):
    text = str(value or "").strip().lower()

    while "(" in text and ")" in text:
        start = text.find("(")
        end = text.find(")", start)

        if end == -1:
            break

        text = text[:start] + text[end + 1:]

    key = re.sub(r"[^a-z0-9]", "", text)

    return IDENTITY_ALIASES.get(key, key)


def normalize_unit(value):
    text = str(value or "").strip().lower()
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("Ω", "ohm").replace("ω", "ohm")
    text = text.replace("·", "*").replace("⋅", "*")
    text = text.replace(" ", "")

    return text


def normalize_conditions(value):
    text = str(value or "").lower()
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("Ω", "ohm").replace("ω", "ohm")
    text = text.replace("–", "-").replace("−", "-")
    text = text.replace("≤", "<=").replace("≥", ">=")
    text = text.replace("×", "x")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([<>=])\s*", r"\1", text)
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s*;\s*", ";", text)

    return text.strip(" ;,|")


def normalize_number(value):
    text = str(value).strip().replace("+", "")

    try:
        number = Decimal(text)

        if number == number.to_integral_value():
            return str(number.quantize(Decimal("1")))

        return format(number.normalize(), "f")
    except InvalidOperation:
        return text


def parse_decimal(value):
    text = str(value or "").strip()
    text = text.replace("–", "-").replace("−", "-")
    text = text.replace(",", "")

    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def value_match(prediction, gold):
    left = parse_decimal(prediction["value"])
    right = parse_decimal(gold["value"])

    return left is not None and right is not None and left == right


def key_of(record):
    return (
        to_int(record.get("table_id")),
        to_int(record.get("row_index")),
        str(record.get("value_type") or "").strip().lower(),
    )


def source_of(record):
    return (
        to_int(record.get("table_id")),
        to_int(record.get("row_index")),
        to_int(record.get("page")),
    )


CONDITION_UNITS = r"[a-z%°/]+"


def _token_from_parts(var, a, u1, b=None, u2=None, op=None):
    var = var.strip(" _")
    unit1 = normalize_unit(u1 or "")
    left = normalize_number(a)

    if b is not None:
        unit2 = normalize_unit(u2 or u1 or "")
        return f"{var}:{left}{unit1}..{normalize_number(b)}{unit2}"

    if op in {"=", ""}:
        return f"{var}:{left}{unit1}"

    return f"{var}:{op}{left}{unit1}"


def condition_tokens(value):
    """把测试条件拆成可比较的约束集合。

    支持：
    - `VIN = 2.5V`、`TA = +25°C`、`f = 100Hz`；
    - `2.5V <= VIN <= 6V`、`-40°C <= TJ <= 85°C`；
    - `VIN = 2.2V to 6V`、`ILOAD = 0A to 1A`；
    - 其余非数值文字子句（No Load / ENABLE Pin = 0V 等）。
    """

    text = normalize_conditions(value)

    if not text:
        return frozenset()

    tokens = set()
    remaining = text

    chained = re.compile(
        rf"(?P<a>[+-]?\d+(?:\.\d+)?)\s*(?P<u1>{CONDITION_UNITS})?"
        rf"\s*(?P<op1><=|<|>=|>)\s*"
        rf"(?P<var>[a-z][a-z0-9_/]*)\s*"
        rf"(?P<op2><=|<|>=|>)\s*"
        rf"(?P<b>[+-]?\d+(?:\.\d+)?)\s*(?P<u2>{CONDITION_UNITS})?"
    )

    for match in list(chained.finditer(remaining)):
        tokens.add(
            _token_from_parts(
                match.group("var"),
                match.group("a"),
                match.group("u1"),
                match.group("b"),
                match.group("u2"),
            )
        )
        remaining = remaining.replace(match.group(0), " ; ")

    ranged = re.compile(
        rf"(?P<var>[a-z][a-z0-9_/]*)\s*(?:=|<=|>=|<|>)\s*"
        rf"(?P<a>[+-]?\d+(?:\.\d+)?)\s*(?P<u1>{CONDITION_UNITS})?"
        rf"\s*(?:to|through|\.\.\.|~|-)\s*"
        rf"(?P<b>[+-]?\d+(?:\.\d+)?)\s*(?P<u2>{CONDITION_UNITS})?"
    )

    for match in list(ranged.finditer(remaining)):
        tokens.add(
            _token_from_parts(
                match.group("var"),
                match.group("a"),
                match.group("u1"),
                match.group("b"),
                match.group("u2"),
            )
        )
        remaining = remaining.replace(match.group(0), " ; ")

    simple = re.compile(
        rf"(?P<var>[a-z][a-z0-9_/]*)\s*(?P<op>=|<=|>=|<|>)\s*"
        rf"(?P<val>[+-]?\d+(?:\.\d+)?)\s*(?P<u>{CONDITION_UNITS})?"
    )

    for match in list(simple.finditer(remaining)):
        tokens.add(
            _token_from_parts(
                match.group("var"),
                match.group("val"),
                match.group("u"),
                op=match.group("op"),
            )
        )
        remaining = remaining.replace(match.group(0), " ; ")

    for part in re.split(r"[;,]", remaining):
        part = part.strip(" .:-")

        if not part or len(part) < 2:
            continue

        if part in {
            "unless otherwise noted",
            "unless otherwise specified",
            "note",
        }:
            continue

        tokens.add("text:" + re.sub(r"\s+", " ", part))

    return frozenset(tokens)


def conditions_complete(prediction, gold):
    gold_tokens = condition_tokens(gold.get("test_conditions"))

    if not gold_tokens:
        return True

    return gold_tokens.issubset(
        condition_tokens(prediction.get("test_conditions"))
    )


def conditions_exact(prediction, gold):
    return normalize_conditions(
        prediction.get("test_conditions")
    ) == normalize_conditions(gold.get("test_conditions"))


def normalize_prediction(param, index):
    evidence = param.get("evidence") or {}

    return {
        "index": index,
        "table_id": to_int(evidence.get("table_id")),
        "row_index": to_int(evidence.get("row_index")),
        "page": to_int(evidence.get("page")),
        "value_type": str(
            param.get("value_type") or ""
        ).strip().lower(),
        "name": param.get("name"),
        "identity": normalize_identity(param.get("name")),
        "value": param.get("value"),
        "unit": normalize_unit(param.get("unit")),
        "raw_unit": param.get("unit"),
        "test_conditions": param.get("test_conditions", ""),
        "validation_status": param.get("validation_status"),
        "validation_errors": list(
            param.get("validation_errors") or []
        ),
        "evidence": evidence,
    }


def normalize_gold(record):
    return {
        "gold_id": record["gold_id"],
        "table_id": to_int(record["source"]["table_id"]),
        "row_index": to_int(record["source"]["row_index"]),
        "page": to_int(record["source"].get("page")),
        "value_type": str(record["value_type"]).lower(),
        "identity": normalize_identity(
            record["parameter_identity"]
        ),
        "value": record["value"],
        "unit": normalize_unit(record["unit"]),
        "raw_unit": record["unit"],
        "test_conditions": record.get("test_conditions", ""),
        "condition_source": record.get("condition_source", "row"),
        "row_text": record["source"].get("row_text", ""),
        "parameter_symbol": record.get("parameter_symbol"),
        "review": record.get("review", {}),
    }


def ratio(numerator, denominator):
    if denominator in (0, None):
        return None

    return round(numerator / denominator, 4)


def f1(precision, recall):
    if precision is None or recall is None:
        return None

    if precision + recall == 0:
        return 0.0

    return round(
        2 * precision * recall / (precision + recall), 4
    )


def max_weight_match(left, right, allowed, weight):
    """最大权重一对一匹配；返回 [(left_index, right_index)]。"""

    if not left or not right:
        return []

    weights = np.zeros((len(left), len(right)))

    for i, left_item in enumerate(left):
        for j, right_item in enumerate(right):
            if allowed(left_item, right_item):
                weights[i, j] = 1_000_000 + weight(
                    left_item, right_item
                )

    size = max(weights.shape)
    matrix = np.zeros((size, size))
    matrix[: weights.shape[0], : weights.shape[1]] = weights
    rows, cols = linear_sum_assignment(-matrix)

    return [
        (int(row), int(col))
        for row, col in zip(rows, cols)
        if row < weights.shape[0]
        and col < weights.shape[1]
        and weights[row, col] > 0
    ]


def content_allowed(gold, prediction):
    return (
        gold["identity"] == prediction["identity"]
        and gold["value_type"] == prediction["value_type"]
        and value_match(prediction, gold)
        and gold["unit"] == prediction["unit"]
    )


def identity_allowed(gold, prediction):
    return (
        gold["identity"] == prediction["identity"]
        and gold["value_type"] == prediction["value_type"]
    )


def content_weight(gold, prediction):
    return (
        (1 if conditions_complete(prediction, gold) else 0) * 100
        + (1 if source_of(prediction) == source_of(gold) else 0) * 10
        + (1 if prediction["page"] == gold["page"] else 0)
    )


def identity_weight(gold, prediction):
    return (
        (1 if content_allowed(gold, prediction) else 0) * 1000
        + (1 if conditions_complete(prediction, gold) else 0) * 100
        + (1 if source_of(prediction) == source_of(gold) else 0) * 10
        + (1 if prediction["page"] == gold["page"] else 0)
    )


def classify_missing(gold, candidates, predictions, gold_scope_ids):
    if gold["table_id"] not in gold_scope_ids:
        return "table_selector_miss"

    same_identity_type = [
        prediction
        for prediction in candidates
        if prediction["identity"] == gold["identity"]
        and prediction["value_type"] == gold["value_type"]
    ]

    if not same_identity_type:
        same_identity = [
            prediction
            for prediction in candidates
            if prediction["identity"] == gold["identity"]
        ]

        if same_identity:
            return "llm_wrong_value_type"

        return "llm_omitted"

    if any(
        value_match(prediction, gold)
        and prediction["unit"] != gold["unit"]
        for prediction in same_identity_type
    ):
        return "llm_wrong_unit"

    if any(
        value_match(prediction, gold)
        for prediction in same_identity_type
    ):
        if any(
            prediction["identity"] == gold["identity"]
            and value_match(prediction, gold)
            and prediction["value_type"] != gold["value_type"]
            for prediction in predictions
            if prediction["table_id"] == gold["table_id"]
        ):
            return "llm_wrong_value_type"

        return "llm_wrong_unit"

    if any(
        source_of(prediction) == source_of(gold)
        for prediction in same_identity_type
    ):
        if any(
            source_of(prediction) == source_of(gold)
            and prediction["identity"] != gold["identity"]
            and value_match(prediction, gold)
            for prediction in candidates
        ):
            return "llm_wrong_identity"

        return "llm_wrong_value"

    return "llm_wrong_value"


def structure_suspected(gold):
    row_text = gold.get("row_text", "")

    if not row_text:
        return False

    unit_hits = re.findall(
        r"(?:µA|μA|mA|µV|mV|V|Ω|kΩ|dB|%)", row_text
    )
    number_hits = NUMBER_RE.findall(row_text)

    return len(unit_hits) >= 3 or len(number_hits) >= 6


def classify_extra(prediction, matched_content, predictions, golds):
    key = (
        prediction["identity"],
        prediction["value_type"],
        parse_decimal(prediction["value"]),
        prediction["unit"],
    )
    matched_keys = {
        (
            gold["identity"],
            gold["value_type"],
            parse_decimal(gold["value"]),
            gold["unit"],
        )
        for gold in matched_content
    }

    if key in matched_keys:
        return "duplicate_prediction"

    same_source = [
        gold
        for gold in golds
        if source_of(gold) == source_of(prediction)
    ]

    if same_source:
        if all(
            gold["identity"] != prediction["identity"]
            for gold in same_source
        ):
            return "llm_wrong_identity"

        same_identity = [
            gold
            for gold in same_source
            if gold["identity"] == prediction["identity"]
        ]

        if any(
            value_match(prediction, gold)
            and gold["unit"] != prediction["unit"]
            for gold in same_identity
        ):
            return "llm_wrong_unit"

        if any(
            gold["value_type"] != prediction["value_type"]
            for gold in same_identity
        ):
            return "llm_wrong_value_type"

        return "llm_wrong_value"

    same_identity = [
        gold
        for gold in golds
        if gold["identity"] == prediction["identity"]
    ]

    if same_identity:
        if any(
            value_match(prediction, gold)
            and gold["unit"] != prediction["unit"]
            for gold in same_identity
        ):
            return "llm_wrong_unit"

        if any(
            value_match(prediction, gold)
            for gold in same_identity
        ):
            return "llm_wrong_value_type"

        return "extra_or_duplicate_prediction"

    return "gold_coverage_ambiguity"


def error_row(
    *,
    sample,
    kind,
    category,
    gold=None,
    prediction=None,
    note,
    closest=None,
):
    gold = gold or {}
    prediction = prediction or {}

    return {
        "sample_id": sample["sample_id"],
        "vendor": sample["vendor"],
        "kind": kind,
        "category": category,
        "gold_id": gold.get("gold_id"),
        "prediction_index": prediction.get("index"),
        "page": (
            gold.get("page")
            if gold
            else prediction.get("page")
        ),
        "table_id": (
            gold.get("table_id")
            if gold
            else prediction.get("table_id")
        ),
        "row_index": (
            gold.get("row_index")
            if gold
            else prediction.get("row_index")
        ),
        "gold_identity": gold.get("identity"),
        "prediction_identity": prediction.get("identity"),
        "gold_value_type": gold.get("value_type"),
        "prediction_value_type": prediction.get("value_type"),
        "gold_value": gold.get("value"),
        "prediction_value": prediction.get("value"),
        "gold_unit": gold.get("raw_unit"),
        "prediction_unit": prediction.get("raw_unit"),
        "gold_conditions": gold.get("test_conditions", ""),
        "prediction_conditions": prediction.get(
            "test_conditions", ""
        ),
        "source_row_text": gold.get("row_text", ""),
        "prediction_source_text": (
            prediction.get("evidence", {}).get(
                "source_text", ""
            )
        ),
        "validation_status": prediction.get(
            "validation_status"
        ),
        "validation_errors": " | ".join(
            prediction.get("validation_errors") or []
        ),
        "closest_gold_id": closest.get("gold_id")
        if closest
        else None,
        "structure_recovery_suspected": structure_suspected(gold)
        if gold
        else False,
        "note": note,
    }


def _field_key(param):
    evidence = param.get("evidence") or {}

    return (
        param.get("name"),
        param.get("value_type"),
        param.get("test_conditions"),
        evidence.get("table_id"),
        evidence.get("row_index"),
    )


def load_first_pass_params(run_dir, artifact):
    """返回首轮校验参数与来源标记。

    优先读取运行器保存的 first_pass_validation.json；
    旧运行目录缺失时，用生产 verify_evidence + ParameterValidator
    对 artifact 中的原始提取结果确定性重算（不调用 LLM）。
    """

    saved = run_dir / "first_pass_validation.json"

    if saved.is_file():
        validation = load_json(saved).get("validation", {})
        return validation.get("parameters", []), "saved"

    tables_path = run_dir / "selected_tables.json"
    raw = artifact.get("extraction") or {}

    if not tables_path.is_file() or not raw.get("parameters"):
        return [], "unavailable"

    import sys

    backend_root = Path(__file__).resolve().parents[1]

    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.extraction.evidence import verify_evidence
    from app.validation.validator import ParameterValidator

    tables = load_json(tables_path).get("tables", [])
    validation = ParameterValidator().validate(
        verify_evidence(raw, tables)
    )

    return validation.get("parameters", []), "reconstructed"


def status_distribution(validation, first_pass_params, source):
    parameters = validation.get("parameters", [])
    statuses = Counter(
        param.get("validation_status") or "unknown"
        for param in parameters
    )
    first_pass_status = defaultdict(list)

    for param in first_pass_params:
        first_pass_status[_field_key(param)].append(
            param.get("validation_status")
        )

    correctable = 0
    validated_first_pass = 0
    unmatched = 0

    for param in parameters:
        if param.get("validation_status") != "validated":
            continue

        history = first_pass_status.get(_field_key(param))

        if not history:
            validated_first_pass += 1
            unmatched += 1
        elif "invalid" in history:
            correctable += 1
        else:
            validated_first_pass += 1

    return {
        "total": len(parameters),
        "validated_total": statuses.get("validated", 0),
        "validated_first_pass": validated_first_pass,
        "correctable_retry_corrected": correctable,
        "invalid": statuses.get("invalid", 0),
        "pending_review": statuses.get("pending_review", 0),
        "unknown": statuses.get("unknown", 0),
        "raw_status_counts": dict(statuses),
        "first_pass_source": source,
        "first_pass_unmatched": unmatched,
        "first_pass_invalid_count": sum(
            1
            for item in first_pass_params
            if item.get("validation_status") == "invalid"
        ),
        "first_pass_validated_count": sum(
            1
            for item in first_pass_params
            if item.get("validation_status") == "validated"
        ),
    }


def score_sample(sample, gold_sample, run_dir):
    artifact_path = run_dir / "artifact.json"
    validation_path = run_dir / "validation.json"

    if not artifact_path.is_file():
        raise SystemExit(f"缺少 artifact.json: {artifact_path}")

    artifact = load_json(artifact_path)
    validation = (
        load_json(validation_path)
        if validation_path.is_file()
        else {}
    )
    retry_path = run_dir / "retry_summary.json"
    retry_summary = (
        load_json(retry_path) if retry_path.is_file() else {}
    )
    first_pass_params, first_pass_source = load_first_pass_params(
        run_dir, artifact
    )

    source_params = validation.get("parameters")

    if not source_params:
        source_params = (
            artifact.get("final_extraction")
            or artifact.get("extraction")
            or {}
        ).get("parameters", [])

    predictions = [
        normalize_prediction(param, index)
        for index, param in enumerate(source_params)
    ]
    golds = [
        normalize_gold(record)
        for record in gold_sample["records"]
    ]

    scope_ids = set(sample["gold_scope_table_ids"])
    gold_rows = {
        (gold["table_id"], gold["row_index"]) for gold in golds
    }
    gold_identities = {gold["identity"] for gold in golds}
    out_of_scope_table = []
    out_of_scope_parameter = []
    in_scope = []

    for prediction in predictions:
        if prediction["table_id"] not in scope_ids:
            out_of_scope_table.append(prediction)
        elif (
            prediction["identity"] not in gold_identities
            and (
                prediction["table_id"],
                prediction["row_index"],
            )
            not in gold_rows
        ):
            out_of_scope_parameter.append(prediction)
        else:
            in_scope.append(prediction)

    content_pairs = max_weight_match(
        golds,
        in_scope,
        content_allowed,
        content_weight,
    )
    identity_pairs = max_weight_match(
        golds,
        in_scope,
        identity_allowed,
        identity_weight,
    )

    matched_gold = {i for i, _ in content_pairs}
    matched_pred = {j for _, j in content_pairs}
    matched_identity_gold = {i for i, _ in identity_pairs}
    matched_identity_pred = {j for _, j in identity_pairs}
    tp = len(content_pairs)
    fp = len(in_scope) - len(matched_pred)
    fn = len(golds) - len(matched_gold)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    score_f1 = f1(precision, recall)
    identity_pairs_count = len(identity_pairs)
    identity_precision = ratio(
        identity_pairs_count, len(in_scope)
    )
    identity_recall = ratio(identity_pairs_count, len(golds))
    identity_f1 = f1(identity_precision, identity_recall)
    identity_jaccard = ratio(
        identity_pairs_count,
        len(golds) + len(in_scope) - identity_pairs_count,
    )
    content_conditions = sum(
        1
        for gold_index, pred_index in content_pairs
        if conditions_complete(
            in_scope[pred_index], golds[gold_index]
        )
    )
    content_conditions_exact = sum(
        1
        for gold_index, pred_index in content_pairs
        if conditions_exact(
            in_scope[pred_index], golds[gold_index]
        )
    )
    content_source = sum(
        1
        for gold_index, pred_index in content_pairs
        if source_of(in_scope[pred_index])
        == source_of(golds[gold_index])
    )
    strict_tp = sum(
        1
        for gold_index, pred_index in content_pairs
        if conditions_complete(
            in_scope[pred_index], golds[gold_index]
        )
        and source_of(in_scope[pred_index])
        == source_of(golds[gold_index])
    )
    strict_precision = ratio(strict_tp, len(in_scope))
    strict_recall = ratio(strict_tp, len(golds))
    strict_f1 = f1(strict_precision, strict_recall)

    errors = []
    matched_content_golds = [
        golds[gold_index] for gold_index, _ in content_pairs
    ]

    for gold_index, gold in enumerate(golds):
        if gold_index in matched_gold:
            prediction = in_scope[
                dict(content_pairs)[gold_index]
            ]

            if not conditions_complete(prediction, gold):
                errors.append(
                    error_row(
                        sample=sample,
                        kind="condition",
                        category="test_conditions_incomplete",
                        gold=gold,
                        prediction=prediction,
                        note=(
                            "数值/单位/身份匹配，但测试条件不完整。"
                        ),
                    )
                )

            if source_of(prediction) != source_of(gold):
                errors.append(
                    error_row(
                        sample=sample,
                        kind="source",
                        category=(
                            "structure_recovery_row_or_unit_error"
                            if structure_suspected(gold)
                            else "source_localization_mismatch"
                        ),
                        gold=gold,
                        prediction=prediction,
                        note=(
                            "内容匹配但 table_id/row_index/page "
                            "与 Gold 来源不一致。"
                        ),
                    )
                )

            if prediction["validation_status"] == "invalid":
                errors.append(
                    error_row(
                        sample=sample,
                        kind="validator",
                        category="validator_false_reject",
                        gold=gold,
                        prediction=prediction,
                        note=(
                            "内容与 Gold 一致，但 Validator "
                            "标记为 invalid。"
                        ),
                    )
                )
            continue

        category = classify_missing(
            gold,
            [
                prediction
                for prediction in in_scope
                if prediction["table_id"] == gold["table_id"]
            ],
            predictions,
            scope_ids,
        )

        if structure_suspected(gold) and category.startswith("llm_"):
            category = "structure_recovery_row_or_unit_error"

        errors.append(
            error_row(
                sample=sample,
                kind="FN",
                category=category,
                gold=gold,
                note="Gold 记录未被任何预测以身份+数值+单位命中。",
            )
        )

    for position, prediction in enumerate(in_scope):
        if position in matched_pred:
            continue

        category = classify_extra(
            prediction,
            matched_content_golds,
            in_scope,
            golds,
        )
        closest = None

        if golds:
            closest = min(
                golds,
                key=lambda gold: (
                    0
                    if gold["identity"] == prediction["identity"]
                    else 1,
                    0
                    if gold["value_type"]
                    == prediction["value_type"]
                    else 1,
                    abs(
                        (gold["row_index"] or 0)
                        - (prediction["row_index"] or 0)
                    ),
                ),
            )

        errors.append(
            error_row(
                sample=sample,
                kind="FP",
                category=category,
                prediction=prediction,
                closest=closest,
                note="in-scope 预测未成为 TP。",
            )
        )

    error_categories = Counter(
        error["category"] for error in errors
    )
    out_of_scope_table_by_table = Counter(
        prediction["table_id"]
        for prediction in out_of_scope_table
    )
    in_scope_by_table = Counter(
        prediction["table_id"] for prediction in in_scope
    )
    tp_status = Counter(
        in_scope[pred_index].get("validation_status")
        for _, pred_index in content_pairs
    )
    status = status_distribution(
        validation, first_pass_params, first_pass_source
    )
    status["in_scope_total"] = len(in_scope)
    status["out_of_scope_table"] = len(out_of_scope_table)
    status["out_of_scope_parameter"] = len(
        out_of_scope_parameter
    )

    return {
        "sample_id": sample["sample_id"],
        "vendor": sample["vendor"],
        "model": sample["model"],
        "document_version": sample["document_version"],
        "gold_records": len(golds),
        "gold_scope_table_ids": sorted(scope_ids),
        "predictions_total": len(predictions),
        "predictions_in_scope": len(in_scope),
        "out_of_scope_table_predictions": len(
            out_of_scope_table
        ),
        "out_of_scope_parameter_predictions": len(
            out_of_scope_parameter
        ),
        "out_of_scope_table_by_table": dict(
            out_of_scope_table_by_table
        ),
        "in_scope_by_table": dict(in_scope_by_table),
        "tp_by_validation_status": dict(tp_status),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": score_f1,
        "identity_pairs": identity_pairs_count,
        "identity_precision": identity_precision,
        "identity_recall": identity_recall,
        "identity_f1": identity_f1,
        "identity_accuracy": identity_jaccard,
        "value_unit_accuracy": ratio(
            tp, identity_pairs_count
        ),
        "conditions_complete_count": content_conditions,
        "conditions_complete_rate": ratio(
            content_conditions, tp
        ),
        "conditions_exact_count": content_conditions_exact,
        "conditions_exact_rate": ratio(
            content_conditions_exact, tp
        ),
        "source_localized_count": content_source,
        "source_localization_rate": ratio(
            content_source, tp
        ),
        "strict_tp": strict_tp,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
        "validation_status_distribution": status,
        "error_category_counts": dict(error_categories),
        "errors": errors,
        "limitations": [
            "参数身份/数值/单位匹配使用脚本内冻结的归一化规则。",
            "测试条件按 Gold 约束子集匹配；"
            "额外条件不判为错误。",
            "Flat Baseline 不使用 LogicalTableAdapter，"
            "该错误类别预期为 0。",
        ],
    }


def aggregate(samples):
    totals = {
        key: sum(sample[key] for sample in samples)
        for key in (
            "gold_records",
            "predictions_total",
            "predictions_in_scope",
            "out_of_scope_table_predictions",
            "out_of_scope_parameter_predictions",
            "tp",
            "fp",
            "fn",
            "identity_pairs",
            "conditions_complete_count",
            "conditions_exact_count",
            "source_localized_count",
            "strict_tp",
        )
    }
    totals["precision"] = ratio(
        totals["tp"], totals["tp"] + totals["fp"]
    )
    totals["recall"] = ratio(
        totals["tp"], totals["tp"] + totals["fn"]
    )
    totals["f1"] = f1(
        totals["precision"], totals["recall"]
    )
    totals["identity_precision"] = ratio(
        totals["identity_pairs"], totals["predictions_in_scope"]
    )
    totals["identity_recall"] = ratio(
        totals["identity_pairs"], totals["gold_records"]
    )
    totals["identity_f1"] = f1(
        totals["identity_precision"], totals["identity_recall"]
    )
    totals["identity_accuracy"] = ratio(
        totals["identity_pairs"],
        totals["gold_records"]
        + totals["predictions_in_scope"]
        - totals["identity_pairs"],
    )
    totals["value_unit_accuracy"] = ratio(
        totals["tp"], totals["identity_pairs"]
    )
    totals["conditions_complete_rate"] = ratio(
        totals["conditions_complete_count"], totals["tp"]
    )
    totals["conditions_exact_rate"] = ratio(
        totals["conditions_exact_count"], totals["tp"]
    )
    totals["source_localization_rate"] = ratio(
        totals["source_localized_count"], totals["tp"]
    )
    totals["strict_precision"] = ratio(
        totals["strict_tp"], totals["predictions_in_scope"]
    )
    totals["strict_recall"] = ratio(
        totals["strict_tp"], totals["gold_records"]
    )
    totals["strict_f1"] = f1(
        totals["strict_precision"], totals["strict_recall"]
    )
    macro_keys = ("precision", "recall", "f1")
    totals["macro"] = {
        key: (
            round(
                sum(
                    sample[key]
                    for sample in samples
                    if sample[key] is not None
                )
                / len(
                    [
                        sample
                        for sample in samples
                        if sample[key] is not None
                    ]
                ),
                4,
            )
            if any(
                sample[key] is not None for sample in samples
            )
            else None
        )
        for key in macro_keys
    }

    statuses = Counter()

    for sample in samples:
        distribution = sample["validation_status_distribution"]
        statuses["validated_first_pass"] += distribution[
            "validated_first_pass"
        ]
        statuses["correctable_retry_corrected"] += distribution[
            "correctable_retry_corrected"
        ]
        statuses["invalid"] += distribution["invalid"]
        statuses["pending_review"] += distribution[
            "pending_review"
        ]
        statuses["unknown"] += distribution["unknown"]
        statuses["total"] += distribution["total"]

    totals["validation_status_distribution"] = dict(statuses)
    totals["error_category_counts"] = dict(
        Counter(
            error["category"]
            for sample in samples
            for error in sample["errors"]
        )
    )

    return totals


def write_details_csv(samples, path):
    fieldnames = [
        "sample_id",
        "kind",
        "category",
        "gold_id",
        "prediction_index",
        "page",
        "table_id",
        "row_index",
        "gold_identity",
        "prediction_identity",
        "gold_value_type",
        "prediction_value_type",
        "gold_value",
        "prediction_value",
        "gold_unit",
        "prediction_unit",
        "gold_conditions",
        "prediction_conditions",
        "source_row_text",
        "prediction_source_text",
        "validation_status",
        "validation_errors",
        "closest_gold_id",
        "structure_recovery_suspected",
        "note",
    ]

    with Path(path).open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()

        for sample in samples:
            for error in sample["errors"]:
                writer.writerow(error)


def write_report(metrics, samples, path):
    micro = metrics["micro"]
    first_pass_sources = sorted(
        {
            sample["validation_status_distribution"].get(
                "first_pass_source"
            )
            or "unknown"
            for sample in samples
        }
    )
    first_pass_source = ", ".join(first_pass_sources)

    def pct(value):
        return "N/A" if value is None else f"{value * 100:.2f}%"

    lines = [
        "# B3.5c 跨型号 Flat Baseline 评测报告",
        "",
        f"- 生成时间（UTC）：{metrics['generated_at']}",
        f"- Scorer：`{metrics['scorer_version']}` / "
        f"`{metrics['normalization_version']}`",
        f"- 冻结 Gold：{metrics['gold_path']}",
        f"- 运行目录：{metrics['run_root']}",
        "- 只使用 76 条 confirmed；12 条 rejected 未进入标准答案。",
        "",
        "## 1. 全局指标",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| TP / FP / FN | {micro['tp']} / {micro['fp']} / "
        f"{micro['fn']} |",
        f"| Precision | {pct(micro['precision'])} |",
        f"| Recall | {pct(micro['recall'])} |",
        f"| F1 | {pct(micro['f1'])} |",
        f"| 参数身份 Jaccard | "
        f"{pct(micro['identity_accuracy'])} |",
        f"| 身份 Precision / Recall | "
        f"{pct(micro['identity_precision'])} / "
        f"{pct(micro['identity_recall'])} |",
        f"| 数值与单位正确率 | "
        f"{pct(micro['value_unit_accuracy'])} |",
        f"| 测试条件完整匹配率 | "
        f"{pct(micro['conditions_complete_rate'])} |",
        f"| 测试条件完全一致率 | "
        f"{pct(micro['conditions_exact_rate'])} |",
        f"| 来源定位正确率 | "
        f"{pct(micro['source_localization_rate'])} |",
        f"| 严格 P / R / F1 | {pct(micro['strict_precision'])} / "
        f"{pct(micro['strict_recall'])} / "
        f"{pct(micro['strict_f1'])} |",
        f"| out_of_scope 预测 | "
        f"table={micro['out_of_scope_table_predictions']}, "
        f"parameter={micro['out_of_scope_parameter_predictions']} |",
        "",
        "## 2. 分厂商指标",
        "",
        "| 厂商 | Gold | TP | FP | FN | P | R | F1 | 身份 Jaccard | "
        "数值+单位 | 条件完整 | 严格 F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- |",
    ]

    for sample in samples:
        lines.append(
            "| {vendor} | {gold} | {tp} | {fp} | {fn} | {p} | {r} | "
            "{f1} | {identity} | {value_unit} | {conditions} | "
            "{strict} |".format(
                vendor=sample["vendor"],
                gold=sample["gold_records"],
                tp=sample["tp"],
                fp=sample["fp"],
                fn=sample["fn"],
                p=pct(sample["precision"]),
                r=pct(sample["recall"]),
                f1=pct(sample["f1"]),
                identity=pct(sample["identity_accuracy"]),
                value_unit=pct(sample["value_unit_accuracy"]),
                conditions=pct(
                    sample["conditions_complete_rate"]
                ),
                strict=pct(sample["strict_f1"]),
            )
        )

    lines.extend(
        [
            "",
            "## 2.1 表级预测分布",
            "",
            "| 厂商 | in-scope 表 | out-of-scope 表预测 | TP（validated / invalid / 其他） |",
            "| --- | --- | --- | --- |",
        ]
    )

    for sample in samples:
        tp_status = sample.get("tp_by_validation_status", {})
        tp_desc = " / ".join(
            f"{key or 'unknown'}={value}"
            for key, value in sorted(tp_status.items())
        )
        lines.append(
            "| {vendor} | {in_scope} | {out_scope} | {tp} |".format(
                vendor=sample["vendor"],
                in_scope=sample.get("in_scope_by_table", {}),
                out_scope=sample.get(
                    "out_of_scope_table_by_table", {}
                ),
                tp=tp_desc or "0",
            )
        )

    lines.extend(
        [
            "",
            "## 3. 校验状态分布",
            "",
            "| 范围 | validated（首轮） | correctable（retry 纠正） | "
            "invalid | pending_review |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for sample in samples:
        distribution = sample["validation_status_distribution"]
        lines.append(
            "| {vendor} | {validated} | {correctable} | {invalid} | "
            "{pending} |".format(
                vendor=sample["vendor"],
                validated=distribution["validated_first_pass"],
                correctable=distribution[
                    "correctable_retry_corrected"
                ],
                invalid=distribution["invalid"],
                pending=distribution["pending_review"],
            )
        )

    global_status = micro["validation_status_distribution"]
    lines.append(
        "| **全局** | {validated} | {correctable} | {invalid} | "
        "{pending} |".format(
            validated=global_status.get(
                "validated_first_pass", 0
            ),
            correctable=global_status.get(
                "correctable_retry_corrected", 0
            ),
            invalid=global_status.get("invalid", 0),
            pending=global_status.get("pending_review", 0),
        )
    )
    lines.extend(
        [
            "",
            "> 生产 Flat Baseline 使用 legacy 校验，原生状态只有 "
            "validated / invalid / pending_review。"
            "`correctable` 是评测层对“首轮 invalid、经生产 retry "
            "纠正后 validated”的标注，不是 legacy Validator 原生状态。",
            f"> 首轮状态来源：`{first_pass_source}`"
            "（saved = 运行器保存；reconstructed = 用生产 Validator "
            "对原始提取结果离线确定性重算，不调用 LLM）。",
            "",
            "## 4. 错误分类",
            "",
            "| 类别 | 数量 |",
            "| --- | --- |",
        ]
    )

    for category, count in sorted(
        micro["error_category_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"| `{category}` | {count} |")

    lines.extend(
        [
            "",
            "说明：",
            "",
            "- `logical_table_adapter_error` 为 0：Flat Baseline "
            "不启用 LogicalTableAdapter。",
            "- `table_selector_miss` 由预检保证为 0；"
            "非 Gold scope 表的预测单独计为 out_of_scope_table。",
            "- `structure_recovery_row_or_unit_error` 表示 Gold 来源行"
            "存在合并单元格/多值/重复单位迹象，且该条未严格命中。",
            "- `out_of_scope_parameter` 不计入 FP，单独统计。",
            "",
            "## 5. 分厂商主要错误",
            "",
        ]
    )

    for sample in samples:
        lines.append(f"### {sample['vendor']}")
        lines.append("")
        categories = sample["error_category_counts"]

        if not categories:
            lines.append("- 无错误记录。")
            lines.append("")
            continue

        for category, count in sorted(
            categories.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"- `{category}`: {count}")
        lines.append("")

        scope_ids = sample.get("gold_scope_table_ids") or []
        in_scope = sample.get("in_scope_by_table") or {}
        empty_scope = [
            table_id
            for table_id in scope_ids
            if in_scope.get(table_id, 0) == 0
        ]

        if empty_scope:
            out_scope = sample.get("out_of_scope_table_by_table")
            lines.append(
                "- 诊断：Gold scope 表 "
                f"{empty_scope} 没有任何 in-scope 预测；"
                f"本样本 out-of-scope 表预测为 {out_scope}。"
                "说明模型从其他表产出了参数，但没有从 Gold "
                "主表产出可匹配记录（常见于合并多值单元格）。"
            )
            lines.append("")

        failures = [
            error
            for error in sample["errors"]
            if error["kind"] in {"FN", "FP", "validator"}
        ][:8]

        if failures:
            lines.append(
                "| kind | category | page | Gold | Prediction | 说明 |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- |")

            for error in failures:
                gold_desc = (
                    f"{error['gold_identity']} "
                    f"{error['gold_value']}{error['gold_unit'] or ''} "
                    f"({error['gold_value_type']})"
                )
                pred_desc = (
                    f"{error['prediction_identity']} "
                    f"{error['prediction_value']}"
                    f"{error['prediction_unit'] or ''} "
                    f"({error['prediction_value_type']})"
                )
                lines.append(
                    f"| {error['kind']} | `{error['category']}` | "
                    f"p{error['page']} | {gold_desc} | {pred_desc} | "
                    f"{error['note']} |"
                )
            lines.append("")

    lines.extend(
        [
            "## 6. 评分定义",
            "",
            "- **TP**：Gold 与预测一对一匹配，"
            "`identity + value_type + 数值 + 单位` 全部一致。",
            "- **FP**：Gold scope 表内、属于 Gold 身份集合、"
            "但未成为 TP 的预测。",
            "- **FN**：Gold 中未被 TP 匹配的记录。",
            "- **out_of_scope_table**：来自非 Gold scope 表的预测，"
            "不计入 P/R。",
            "- **out_of_scope_parameter**：Gold 表内、来源行不属于 "
            "Gold 且身份不在 Gold 身份集合中的预测，不计入 FP。",
            "- **参数身份 Jaccard**：身份+value_type 一对一匹配数 / "
            "（Gold 条数 + in-scope 预测数 − 匹配数）；"
            "另见 `identity_precision` / `identity_recall`。",
            "- **数值与单位正确率**：内容 TP / 身份匹配数。",
            "- **测试条件完整匹配率**：内容 TP 中满足 Gold 全部条件"
            "约束的比例。",
            "- **严格 P/R/F1**：内容 TP 且测试条件完整、来源行/page "
            "一致。",
            "",
            "## 7. 限制",
            "",
            "- 每个厂商只有 1 次运行，不构成统计显著结论。",
            "- Gold 是人工确认的子集，不是 PDF 全部有效参数的穷举；"
            "同表内未知身份预测已单独标记为 out_of_scope。",
            "- Flat Baseline 不启用 StructureRecovery/LogicalTableAdapter "
            "的重建输入；相关错误只能观测，不能归因于这些模块的增强效果。",
        ]
    )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    gold = load_json(args.gold)
    manifest = load_json(args.manifest)

    if gold.get("status") != "frozen":
        raise SystemExit("Gold 未冻结，禁止评分")

    manifest_by_id = {
        sample["sample_id"]: sample
        for sample in manifest.get("samples", [])
    }
    run_root = Path(args.run_root)
    run_manifest_path = run_root / "run_manifest.json"

    if run_manifest_path.is_file():
        run_manifest = load_json(run_manifest_path)

        if run_manifest.get("status") != "completed":
            raise SystemExit(
                "运行未完成，禁止评分: "
                f"{run_manifest.get('status')}"
            )

    samples = []

    for gold_sample in gold.get("samples", []):
        sample_id = gold_sample["sample_id"]
        sample = manifest_by_id[sample_id]
        run_dir = run_root / sample_id

        if not run_dir.is_dir():
            raise SystemExit(f"缺少运行目录: {run_dir}")

        samples.append(
            score_sample(sample, gold_sample, run_dir)
        )

    micro = aggregate(samples)
    metrics = {
        "schema": "b35c_baseline_metrics/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scorer_version": SCORER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "gold_path": args.gold,
        "manifest_path": args.manifest,
        "run_root": str(run_root),
        "micro": micro,
        "samples": [
            {
                key: value
                for key, value in sample.items()
                if key != "errors"
            }
            for sample in samples
        ],
        "limitations": [
            "单次跨型号运行不构成统计显著结论。",
            "参数身份正确率取身份+value_type 的 Jaccard 匹配。",
            "测试条件按 Gold 约束子集匹配；额外条件不计错。",
        ],
    }

    output_path = Path(args.output)
    write_json(output_path, metrics)
    write_details_csv(
        samples,
        output_path.with_name(
            output_path.stem + ".records.csv"
        ),
    )
    write_json(
        output_path.with_name(
            output_path.stem + ".errors.json"
        ),
        [
            error
            for sample in samples
            for error in sample["errors"]
        ],
    )
    write_report(
        metrics,
        samples,
        output_path.with_name(output_path.stem + ".report.md"),
    )

    print("指标:", output_path)
    print(
        "TP/FP/FN:",
        micro["tp"],
        micro["fp"],
        micro["fn"],
    )
    print(
        "P/R/F1:",
        micro["precision"],
        micro["recall"],
        micro["f1"],
    )
    print(
        "identity/value_unit/conditions:",
        micro["identity_accuracy"],
        micro["value_unit_accuracy"],
        micro["conditions_complete_rate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

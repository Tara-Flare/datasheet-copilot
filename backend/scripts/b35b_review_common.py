"""B3.5b：XLSX 人工核验与 CSV 同步的共享逻辑。

设计约束：
- XLSX 是唯一人工编辑入口；CSV 只由同步脚本从 XLSX 派生；
- 同步绝不自动把 review_status 改成 confirmed；
- 所有验证失败时不得写入任何 CSV；
- 写入采用同目录临时文件 + os.replace 原子替换。
"""

import csv
import hashlib
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


SCHEMA_VERSION = "b35b_review_sync/v1"

# 与 b35b_build_gold_draft.py 输出的 CSV 列完全一致。
COLUMNS = [
    "gold_id",
    "sample_id",
    "vendor",
    "model",
    "table_id",
    "page",
    "row_index",
    "parameter_symbol_raw",
    "parameter_name_raw",
    "family_guess",
    "value_type",
    "value_raw",
    "unit_raw",
    "test_conditions_raw",
    "source_row_text",
    "layout_source",
    "layout_confidence",
    "layout_warnings",
    "needs_layout_review",
    "layout_confirmed",
    "needs_split",
    "needs_unit",
    "needs_conditions",
    "review_status",
    "reviewer",
    "review_notes",
    "gold_identity",
    "gold_symbol",
    "gold_value",
    "gold_unit",
    "gold_test_conditions",
    "condition_source",
    "split_resolved",
]

BOOLEAN_COLUMNS = {
    "needs_layout_review",
    "layout_confirmed",
    "needs_split",
    "needs_unit",
    "needs_conditions",
    "split_resolved",
}

# 人工可编辑字段 = COLUMNS；XLSX 不额外增加隐藏列，避免两份状态。
EDITABLE_COLUMNS = list(COLUMNS)

SHEET_BY_SAMPLE = {
    "adi_lt3045": "ADI LT3045",
    "diodes_ap2112": "Diodes AP2112",
    "onsemi_ncp170": "onsemi NCP170",
    "renesas_isl80510": "Renesas ISL80510",
}

VALID_REVIEW_STATUS = {"unreviewed", "confirmed", "rejected"}
VALID_VALUE_TYPES = {"min", "typ", "max"}
VALID_CONDITION_SOURCES = {
    "row",
    "table_default",
    "footnote",
    "none",
}
PLACEHOLDERS = {
    "",
    "-",
    "–",
    "—",
    "n/a",
    "na",
    "none",
    "null",
    "tbd",
}
AUTO_REVIEWERS = {
    "auto",
    "automatic",
    "llm",
    "model",
    "script",
    "system",
    "n/a",
}


class SyncError(ValueError):
    """同步验证失败，未写入任何文件。"""


class SyncReceiptError(SyncError):
    """同步回执缺失或与当前文件不一致。"""


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def truthy(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def to_text(value):
    """把 Excel 单元格值稳定转换为字符串。"""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, int):
        return str(value)

    return str(value)


def clean(value):
    return to_text(value).strip()


def normalize_status(value):
    return clean(value).lower()


def parse_decimal(value):
    text = clean(value)

    if not text:
        return None

    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None

    if not parsed.is_finite():
        return None

    return parsed


def is_placeholder(value):
    return clean(value).lower() in PLACEHOLDERS


def require_int(value, field, errors):
    text = clean(value)

    if not text:
        errors.append(f"{field} 为空")
        return None

    try:
        return int(text)
    except (TypeError, ValueError):
        errors.append(f"{field} 不是整数: {text!r}")
        return None


def expected_sheet_names(manifest):
    names = []

    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]

        if sample_id not in SHEET_BY_SAMPLE:
            raise SyncError(
                f"manifest 中存在未定义工作表名的样本: {sample_id}"
            )

        names.append(SHEET_BY_SAMPLE[sample_id])

    return names


def load_drafts(draft_dir, manifest):
    """读取 Gold 草稿，构造允许的来源行集合。"""

    drafts = {}
    allowed = {}

    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        path = (
            Path(draft_dir)
            / f"gold_draft.{sample_id}.json"
        )

        if not path.is_file():
            raise SyncError(f"缺少 Gold 草稿: {path}")

        draft = load_json(path)
        drafts[sample_id] = draft
        sources = {}

        for record in draft.get("records", []):
            key = (record["table_id"], record["row_index"])
            entry = sources.setdefault(
                key,
                {
                    "page": record.get("page"),
                    "source_row_text": record.get(
                        "source_row_text", ""
                    ),
                    "in_draft": True,
                    "records": [],
                },
            )
            entry["records"].append(
                {
                    "gold_id": record.get("gold_id"),
                    "value_type": str(
                        record.get("value_type", "")
                    ).lower(),
                }
            )

        for item in draft.get("excluded_rows", []):
            key = (item["table_id"], item["row_index"])
            sources.setdefault(
                key,
                {
                    "page": None,
                    "source_row_text": item.get(
                        "source_row_text", ""
                    ),
                    "in_draft": False,
                    "records": [],
                },
            )

        allowed[sample_id] = sources

    return drafts, allowed


def read_workbook(path):
    """读取 XLSX，返回 {sheet_name: (headers, rows)}。"""

    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover
        raise SyncError(
            "缺少 openpyxl，请在 .venv 中安装 openpyxl>=3.1"
        ) from exc

    workbook = load_workbook(
        filename=path, data_only=True, read_only=False
    )
    sheets = {}

    for worksheet in workbook.worksheets:
        values = list(worksheet.iter_rows(values_only=True))

        if not values:
            sheets[worksheet.title] = ([], [])
            continue

        headers = [clean(value) for value in values[0]]

        while headers and not headers[-1]:
            headers.pop()

        rows = []

        for raw in values[1:]:
            if all(clean(value) == "" for value in raw):
                continue

            rows.append(
                {
                    headers[index]: raw[index]
                    if index < len(raw)
                    else None
                    for index in range(len(headers))
                }
            )

        sheets[worksheet.title] = (headers, rows)

    return sheets


def validate_sheet_headers(sample_id, headers, errors):
    if len(headers) != len(set(headers)):
        errors[f"{sample_id}:header"] = [
            "表头存在重复列"
        ]
        return

    missing = [
        column for column in COLUMNS if column not in headers
    ]
    unknown = [
        column for column in headers if column not in COLUMNS
    ]

    if missing:
        errors[f"{sample_id}:header"] = [
            f"缺少列: {missing}"
        ]

    if unknown:
        errors.setdefault(f"{sample_id}:header", []).append(
            f"存在未定义列: {unknown}"
        )


def validate_rows(
    sample,
    rows,
    allowed_sources,
    *,
    allow_incomplete=False,
):
    """校验单个工作表；返回 (normalized_rows, errors)。"""

    sample_id = sample["sample_id"]
    errors = {}
    normalized = []
    parsed_rows = []
    seen_gold_ids = set()
    duplicate_keys = {}

    for row_number, raw in enumerate(rows, start=2):
        row = {
            column: clean(raw.get(column, ""))
            for column in COLUMNS
        }
        row["_row_number"] = row_number
        row_errors = []
        gold_id = row["gold_id"]

        if not gold_id:
            row_errors.append("gold_id 为空")
        elif gold_id in seen_gold_ids:
            row_errors.append(f"gold_id 重复: {gold_id}")
        else:
            seen_gold_ids.add(gold_id)

        if row["sample_id"] != sample_id:
            row_errors.append(
                f"sample_id 应为 {sample_id}，实际 "
                f"{row['sample_id']!r}"
            )

        table_id = require_int(
            row["table_id"], "table_id", row_errors
        )
        row_index = require_int(
            row["row_index"], "row_index", row_errors
        )
        page = require_int(row["page"], "page", row_errors)

        source = (
            allowed_sources.get((table_id, row_index))
            if table_id is not None
            and row_index is not None
            else None
        )

        if source is None:
            row_errors.append(
                "来源 (table_id, row_index) 不在 Docling "
                "Gold 草稿中"
            )
        elif (
            source.get("page") is not None
            and page is not None
            and page != source["page"]
        ):
            row_errors.append(
                f"page 与草稿不一致: {page} != "
                f"{source['page']}"
            )

        status = normalize_status(row["review_status"])
        row["review_status"] = status

        if status not in VALID_REVIEW_STATUS:
            row_errors.append(
                f"review_status 非法: {row['review_status']!r}"
            )

        reviewer = row["reviewer"].strip()
        notes = row["review_notes"].strip()

        if status == "confirmed":
            if not reviewer:
                row_errors.append("confirmed 但 reviewer 为空")
            elif reviewer.lower() in AUTO_REVIEWERS:
                row_errors.append(
                    f"reviewer 不能是自动值: {reviewer!r}"
                )

            if not row["gold_identity"].strip():
                row_errors.append("gold_identity 为空")

            if is_placeholder(row["gold_unit"]):
                row_errors.append("gold_unit 为空或占位符")

            if parse_decimal(row["gold_value"]) is None:
                row_errors.append(
                    "gold_value 不是单一数字"
                )

            if not truthy(row["split_resolved"]):
                row_errors.append(
                    "split_resolved 未确认（多值未拆分）"
                )

            if not truthy(row["layout_confirmed"]):
                row_errors.append(
                    "layout_confirmed 未确认（列布局未核对）"
                )

            condition_source = row["condition_source"].strip()

            if condition_source not in VALID_CONDITION_SOURCES:
                row_errors.append(
                    "condition_source 非法: "
                    f"{condition_source!r}"
                )

            if condition_source == "none" and not notes:
                row_errors.append(
                    "condition_source=none 时必须填写 "
                    "review_notes"
                )

            if row["value_type"].lower() not in VALID_VALUE_TYPES:
                row_errors.append(
                    "value_type 必须是 min/typ/max"
                )
        elif status == "rejected":
            if not reviewer:
                row_errors.append("rejected 但 reviewer 为空")
            if not notes:
                row_errors.append(
                    "rejected 必须填写 review_notes 说明原因"
                )
        else:
            if not allow_incomplete:
                row_errors.append(
                    "存在 unreviewed 记录；完成核验后重试，"
                    "或显式使用 --allow-incomplete"
                )

        # 布尔字段统一写回 True/False，避免 CSV 出现 1/0/空。
        for column in BOOLEAN_COLUMNS:
            row[column] = (
                "True" if truthy(row[column]) else "False"
            )

        if row["value_type"]:
            row["value_type"] = row["value_type"].lower()

        if row["condition_source"]:
            row["condition_source"] = row[
                "condition_source"
            ].lower()

        duplicate_key = (
            table_id,
            row_index,
            row["value_type"],
            row["gold_identity"].strip().lower(),
            row["gold_symbol"].strip().lower(),
            row["gold_test_conditions"].strip().lower(),
        )

        if all(
            value not in (None, "")
            for value in duplicate_key
        ):
            duplicate_keys.setdefault(
                duplicate_key, []
            ).append(gold_id or f"row {row_number}")

        parsed_rows.append(row)

        if row_errors:
            errors[gold_id or f"row {row_number}"] = row_errors
        else:
            normalized.append(row)

    for key, gold_ids in duplicate_keys.items():
        if len(gold_ids) > 1:
            errors[f"duplicate:{key}"] = [
                "同一来源行 + value_type + 参数身份 + 条件重复: "
                + ", ".join(gold_ids)
            ]

    return normalized, errors, parsed_rows


def validate_workbook(
    workbook_sheets,
    manifest,
    allowed_sources,
    *,
    allow_incomplete=False,
):
    """校验整个工作簿，返回 (rows_by_sample, errors)。"""

    expected_sheets = expected_sheet_names(manifest)
    errors = {}
    rows_by_sample = {}

    actual = list(workbook_sheets)

    missing = [
        name for name in expected_sheets if name not in actual
    ]
    extra = [
        name for name in actual if name not in expected_sheets
    ]

    if missing:
        errors["workbook"] = [f"缺少工作表: {missing}"]
    if extra:
        errors.setdefault("workbook", []).append(
            f"存在多余工作表: {extra}"
        )

    all_gold_ids = {}

    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        sheet_name = SHEET_BY_SAMPLE[sample_id]
        headers, rows = workbook_sheets.get(
            sheet_name, ([], [])
        )
        validate_sheet_headers(sample_id, headers, errors)

        normalized, sample_errors, parsed_rows = validate_rows(
            sample,
            rows,
            allowed_sources.get(sample_id, {}),
            allow_incomplete=allow_incomplete,
        )
        rows_by_sample[sample_id] = normalized

        # 完整性：每个（来源行, value_type）的行数不得少于草稿；
        # 允许改 gold_id，也允许把多值行拆成更多行，但禁止静默删除。
        draft_buckets = {}
        row_buckets = {}

        for key, source in allowed_sources.get(
            sample_id, {}
        ).items():
            if not source.get("in_draft"):
                continue

            for record in source.get("records", []):
                bucket = (key, record.get("value_type", ""))
                draft_buckets[bucket] = (
                    draft_buckets.get(bucket, 0) + 1
                )

        for row in parsed_rows:
            try:
                source_key = (
                    int(row["table_id"]),
                    int(row["row_index"]),
                )
            except (TypeError, ValueError):
                continue

            bucket = (
                source_key,
                str(row.get("value_type", "")).lower(),
            )
            row_buckets[bucket] = row_buckets.get(bucket, 0) + 1

        for bucket, expected_count in draft_buckets.items():
            actual_count = row_buckets.get(bucket, 0)

            if actual_count < expected_count:
                (table_id, row_index), value_type = bucket
                sample_errors.setdefault(
                    f"missing:t{table_id}:r{row_index}:{value_type}",
                    [],
                ).append(
                    "草稿记录缺失：该来源行/数值类型的记录数 "
                    f"{actual_count} < 草稿 {expected_count}"
                    "（删除记录应改为 rejected + 说明；"
                    "多值拆分请新增行）"
                )

        for gold_id in (
            row["gold_id"] for row in normalized
        ):
            if gold_id in all_gold_ids:
                sample_errors.setdefault(
                    f"gold_id:{gold_id}", []
                ).append(
                    "gold_id 在整个工作簿中重复: "
                    f"{all_gold_ids[gold_id]} / {sheet_name}"
                )
            else:
                all_gold_ids[gold_id] = sheet_name

        if sample_errors:
            errors[sample_id] = sample_errors

    if errors:
        raise SyncError(
            json.dumps(errors, ensure_ascii=False, indent=2)
        )

    return rows_by_sample


def write_csv_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    try:
        with tmp.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=COLUMNS,
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        column: row.get(column, "")
                        for column in COLUMNS
                    }
                )

        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def count_status(rows):
    counts = {"unreviewed": 0, "confirmed": 0, "rejected": 0}

    for row in rows:
        status = normalize_status(row.get("review_status"))
        counts[status] = counts.get(status, 0) + 1

    return counts


def verify_sync_receipt(
    review_dir, manifest_path, receipt_path=None
):
    """校验同步回执、XLSX 与 CSV 三者一致。"""

    review_dir = Path(review_dir)
    receipt_path = Path(
        receipt_path
        or (review_dir / "gold_sync_receipt.json")
    )

    if not receipt_path.is_file():
        raise SyncReceiptError(
            f"缺少同步回执: {receipt_path}；"
            "请先运行 b35b_sync_review_xlsx.py --apply"
        )

    receipt = load_json(receipt_path)

    if receipt.get("schema") != SCHEMA_VERSION:
        raise SyncReceiptError("同步回执 schema 不匹配")

    xlsx = Path(receipt.get("xlsx_path", ""))

    if not xlsx.is_file():
        raise SyncReceiptError(
            f"回执中的 XLSX 不存在: {xlsx}"
        )

    if sha256_file(xlsx) != receipt.get("xlsx_sha256"):
        raise SyncReceiptError(
            "XLSX 自上次同步后已被修改；请重新同步"
        )

    manifest_path = Path(manifest_path)

    if (
        sha256_file(manifest_path)
        != receipt.get("manifest_sha256")
    ):
        raise SyncReceiptError(
            "manifest 自上次同步后已被修改；请重新同步"
        )

    manifest = load_json(manifest_path)

    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        entry = receipt.get("samples", {}).get(sample_id)

        if entry is None:
            raise SyncReceiptError(
                f"回执缺少样本: {sample_id}"
            )

        csv_path = review_dir / entry["csv"]

        if not csv_path.is_file():
            raise SyncReceiptError(
                f"CSV 不存在: {csv_path}"
            )

        if sha256_file(csv_path) != entry["csv_sha256"]:
            raise SyncReceiptError(
                f"CSV 自上次同步后已被修改: {csv_path}"
            )

        with csv_path.open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))

        if len(rows) != entry["row_count"]:
            raise SyncReceiptError(
                f"CSV 行数与回执不一致: {csv_path}"
            )

    return receipt

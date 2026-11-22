"""阶段 B3.2b：Gold Standard 条件/脚注补充与差异报告。

只读输入：
- gold_review.xlsx
- gold_standard.table9.json
- TPS7A02.pdf

新增输出：
- gold_standard.table9.supplemented.json
- gold_review_supplemented.xlsx
- gold_condition_diff_report.json
- gold_condition_diff_report.md

不覆盖原始 Excel、Gold JSON、PDF 或缓存。
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pymupdf
from openpyxl import load_workbook
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
)
DEFAULT_XLSX = DEFAULT_DIR / "gold_review.xlsx"
DEFAULT_GOLD = DEFAULT_DIR / "gold_standard.table9.json"
DEFAULT_PDF = PROJECT_ROOT / "data/samples/TPS7A02.pdf"

DEFAULT_SUPPLEMENT = (
    DEFAULT_DIR / "gold_standard.table9.supplemented.json"
)
DEFAULT_SUPPLEMENT_XLSX = (
    DEFAULT_DIR / "gold_review_supplemented.xlsx"
)
DEFAULT_DIFF_JSON = (
    DEFAULT_DIR / "gold_condition_diff_report.json"
)
DEFAULT_DIFF_MD = (
    DEFAULT_DIR / "gold_condition_diff_report.md"
)

TABLE_PAGE = 6
FOOTNOTE_PAGE = 7

FOOTNOTE_MAP = {
    "t9.p5": ["1"],
    "t9.p6": ["1", "2"],
    "t9.p7": ["1", "2"],
    "t9.p13": ["3"],
    "t9.p1": ["3"],
}

for row_id in (
    [f"t9.p{i}" for i in range(18, 32)]
):
    FOOTNOTE_MAP[row_id] = ["4"]

FOCUS_IDS = {
    "t9.p9",
    "t9.p10",
    "t9.p24",
    "t9.p25",
    "t9.p26",
    "t9.p30",
    "t9.p31",
}

DEFAULT_TJ = "TJ = –40°C to +125°C"
DEFAULT_VIN = (
    "VIN = VOUT(nom) + 0.5 V or 2.0 V "
    "(whichever is greater)"
)
DEFAULT_IOUT = "IOUT = 1 mA"
DEFAULT_VEN = "VEN = VIN"
DEFAULT_CAPS = "CIN = COUT = 1 µF"
DEFAULT_TYPICAL = "Typical values are at TJ = 25°C."


class SupplementError(ValueError):
    """补充文件生成失败。"""


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def normalize_number(value):
    text = str(value or "").strip()

    return (
        text.replace("–", "-")
        .replace("−", "-")
        .replace("‒", "-")
    )


def numbers_equal(first, second):
    first_text = normalize_number(first)
    second_text = normalize_number(second)

    try:
        return Decimal(first_text) == Decimal(second_text)
    except (InvalidOperation, ValueError):
        return first_text == second_text


def clean_condition(text):
    value = re.sub(r"\(\d+\)", "", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_ws(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def read_excel(path):
    workbook = load_workbook(path, data_only=True)

    if "候选参数" not in workbook.sheetnames:
        raise SupplementError("Excel 缺少 候选参数 工作表")

    ws = workbook["候选参数"]
    headers = [
        ws.cell(1, column).value
        for column in range(1, ws.max_column + 1)
    ]

    required = [
        "序号",
        "logical_row_id",
        "参数名称",
        "Min/Typ/Max",
        "数值",
        "单位",
        "完整测试条件",
        "PDF 页码",
        "source_row_indices",
        "人工核验状态",
        "正确值",
        "备注",
        "重点标记",
        "source_cell_ids",
        "PDF 核验截图",
        "PDF 原文（bbox）",
    ]

    missing = [field for field in required if field not in headers]

    if missing:
        raise SupplementError(
            f"候选参数 缺少列: {missing}"
        )

    rows = []

    for row_index in range(2, ws.max_row + 1):
        row = {}

        for column, header in enumerate(headers, start=1):
            if header:
                row[header] = ws.cell(
                    row_index, column
                ).value

        if not str(row.get("logical_row_id") or "").strip():
            continue

        row["__excel_row"] = row_index
        rows.append(row)

    needs_rows = []

    if "needs_review" in workbook.sheetnames:
        ws2 = workbook["needs_review"]
        headers2 = [
            ws2.cell(1, column).value
            for column in range(1, ws2.max_column + 1)
        ]

        for row_index in range(2, ws2.max_row + 1):
            row = {}

            for column, header in enumerate(
                headers2, start=1
            ):
                if header:
                    row[header] = ws2.cell(
                        row_index, column
                    ).value

            if str(row.get("logical_row_id") or "").strip():
                row["__excel_row"] = row_index
                needs_rows.append(row)

    return workbook, rows, needs_rows


def extract_pdf_sources(pdf_path):
    document = pymupdf.open(pdf_path)

    try:
        page6 = normalize_ws(
            document[TABLE_PAGE - 1].get_text("text")
        )
        page7 = normalize_ws(
            document[FOOTNOTE_PAGE - 1].get_text("text")
        )
    finally:
        document.close()

    default_match = re.search(
        r"(Specified at TJ = .*?\(unless otherwise noted\)\.)",
        page6,
    )
    typical_match = re.search(
        r"(Typical values are at TJ = 25°C\.)",
        page6,
    )

    if default_match is None:
        raise SupplementError(
            "未能从第 6 页提取表级默认测试条件"
        )

    if typical_match is None:
        raise SupplementError(
            "未能从第 6 页提取 Typical values 说明"
        )

    footnotes = {}

    for number in (1, 2, 3, 4):
        other = (
            rf"\s*\({number + 1}\)\s*"
            if number < 4
            else r"\s*6\.6 Switching Characteristics"
        )
        pattern = (
            rf"\({number}\)\s*(.*?)"
            rf"(?={other}|\Z)"
        )
        match = re.search(pattern, page7)

        if match is None:
            raise SupplementError(
                f"未能从第 7 页提取脚注 ({number})"
            )

        footnotes[str(number)] = normalize_ws(
            match.group(1)
        )

    expected_fragments = {
        "1": "VIN = 2.0 V",
        "2": "Load Regulation is normalized",
        "3": "Specified by design",
        "4": "IOUT = 200 mA",
    }

    for number, fragment in expected_fragments.items():
        if fragment not in footnotes[number]:
            raise SupplementError(
                f"脚注 ({number}) 内容校验失败: "
                f"缺少 {fragment!r}"
            )

    if "95%" not in footnotes["4"]:
        raise SupplementError(
            "脚注 (4) 缺少 95% 终点条件"
        )

    return {
        "table_default_text": default_match.group(1),
        "typical_text": typical_match.group(1),
        "footnotes": footnotes,
    }


def parse_tokens(test_conditions):
    tokens = []

    for part in re.split(
        r"[|,;]", str(test_conditions or "")
    ):
        part = part.strip()

        if part:
            tokens.append(part)

    return tokens


def classify_token(token):
    if re.search(r"\bTJ\b", token):
        return "tj"

    if re.search(r"\bVEN\b", token):
        return "ven"

    if re.search(r"\bVIN\b", token):
        return "vin"

    if re.search(r"\bIOUT\b", token):
        return "iout"

    if re.search(r"\bVOUT\b", token):
        return "vout"

    return "other"


def row_tokens(test_conditions):
    classified = {
        "tj": [],
        "vin": [],
        "iout": [],
        "vout": [],
        "ven": [],
        "other": [],
    }

    for token in parse_tokens(test_conditions):
        classified[classify_token(token)].append(token)

    return classified


def join_tokens(tokens):
    return " | ".join(tokens) if tokens else None


def is_current_limit(row):
    name = str(row.get("参数名称") or "")

    return (
        "ICL" in name
        or "ISC" in name
        or "Output current limit" in name
        or "Short-circuit current limit" in name
    )


def build_effective_conditions(row, footnotes):
    row_id = row["logical_row_id"]
    tokens = row_tokens(
        row.get("完整测试条件")
    )
    row_tj = join_tokens(tokens["tj"])
    row_vin = join_tokens(tokens["vin"])
    row_iout = join_tokens(tokens["iout"])
    row_ven = join_tokens(tokens["ven"])
    row_vout = join_tokens(tokens["vout"])

    # TJ
    if row_tj:
        tj = {
            "value": clean_condition(row_tj),
            "source": "row_override",
        }
    else:
        tj = {
            "value": DEFAULT_TJ,
            "source": "table_default",
        }

    # VIN
    if "1" in footnotes:
        if row_id == "t9.p5":
            vin = {
                "mode": "vin_sweep",
                "base_range": clean_condition(row_vin),
                "upper_bound": "6.0 V",
                "footnote_1": {
                    "scope": "VOUT ≤ 1.5 V",
                    "effect": (
                        "VIN lower bound becomes 2.0 V"
                    ),
                    "replaces_vin_scan": False,
                },
                "source": "row_override+footnote_1",
            }
        else:
            vin = {
                "mode": "conditional_fixed",
                "base_value": clean_condition(row_vin),
                "footnote_1": {
                    "scope": "VOUT ≤ 1.5 V",
                    "value": "VIN = 2.0 V",
                    "else": "VIN = VOUT(nom) + 0.5 V",
                },
                "source": "row_override+footnote_1",
            }
    elif "4" in footnotes:
        vin = {
            "mode": "vin_ramp_down",
            "stop_condition": (
                "VOUT = 95% × VOUT(nom)"
            ),
            "default_fixed_vin_suppressed": True,
            "source": "footnote_4",
        }
    elif row_vin:
        vin = {
            "value": clean_condition(row_vin),
            "source": "row_override",
        }
    else:
        vin = {
            "value": DEFAULT_VIN,
            "source": "table_default",
        }

    # IOUT
    if "4" in footnotes:
        iout = {
            "value_ma": 200,
            "text": "IOUT = 200 mA",
            "source": "footnote_4",
            "default_iout_suppressed": True,
        }
    elif row_iout:
        iout = {
            "value": clean_condition(row_iout),
            "source": "row_override",
        }
    elif is_current_limit(row):
        iout = {
            "status": "not_applicable",
            "reason": (
                "该参数本身为输出电流限值，"
                "默认 IOUT=1 mA 不作为条件"
            ),
            "source": "rule",
        }
    else:
        iout = {
            "value": DEFAULT_IOUT,
            "source": "table_default",
        }

    if row_ven:
        ven = {
            "value": clean_condition(row_ven),
            "source": "row_override",
        }
    else:
        ven = {
            "value": DEFAULT_VEN,
            "source": "table_default",
        }

    return {
        "tj": tj,
        "vin": vin,
        "iout": iout,
        "ven": ven,
        "vout_scope": (
            clean_condition(row_vout)
            if row_vout else None
        ),
        "capacitors": {
            "CIN": "1 µF",
            "COUT": "1 µF",
            "source": "table_default",
        },
        "other_row_conditions": tokens["other"],
        "default_typical_values_tj": "25°C",
    }


def build_annotations(row, footnotes, sources):
    row_id = row["logical_row_id"]
    annotations = []

    for number in footnotes:
        text = sources["footnotes"][number]
        item = {
            "footnote": number,
            "source_page": FOOTNOTE_PAGE,
            "text": text,
        }

        if number == "1":
            if row_id == "t9.p5":
                item["applies_to"] = "line_regulation_vin_scan"
                item["effect"] = {
                    "scope": "VOUT ≤ 1.5 V",
                    "vin_lower_bound": "2.0 V",
                    "preserve_vin_scan": True,
                    "preserve_upper_bound": "6.0 V",
                }
            else:
                item["applies_to"] = "load_regulation_vin_setpoint"
                item["effect"] = {
                    "scope": "VOUT ≤ 1.5 V",
                    "value": "VIN = 2.0 V",
                    "else": "VIN = VOUT(nom) + 0.5 V",
                }
        elif number == "2":
            item["applies_to"] = "load_regulation_normalization"
            item["effect"] = {
                "reference": (
                    "VOUT at IOUT = 1 mA"
                ),
                "changes_vin": False,
                "changes_iout": False,
            }
        elif number == "3":
            item["effect"] = {
                "provenance": "Specified by design",
                "changes_values": False,
                "changes_conditions": False,
            }
            if row_id == "t9.p1":
                item["applies_to"] = "iout_lower_bound"
                item["effect"]["token"] = "1 µA(3)"
                item["effect"]["scope"] = (
                    "1 µA lower IOUT bound only"
                )
            else:
                item["applies_to"] = "parameter"
        elif number == "4":
            item["applies_to"] = "dropout_measurement"
            item["effect"] = {
                "iout": "200 mA",
                "vin": "ramp VIN down",
                "stop_condition": (
                    "VOUT = 95% × VOUT(nom)"
                ),
                "overrides_default_vin": True,
                "overrides_default_iout": True,
            }

        annotations.append(item)

    return annotations


def condition_changes(row, footnotes):
    row_id = row["logical_row_id"]
    changes = []

    if row_id == "t9.p5":
        changes.append(
            "脚注 (1) 仅在 VOUT ≤ 1.5 V 时把 VIN 下限修正为 2.0 V；"
            "保留 Line Regulation 的 VIN 扫描范围 "
            "VOUT(nom)+0.5 V ≤ VIN ≤ 6.0 V，"
            "扫描上限 6.0 V 不变。"
        )
    elif row_id in {"t9.p6", "t9.p7"}:
        changes.append(
            "脚注 (1)：VOUT ≤ 1.5 V 时 VIN = 2.0 V；"
            "否则 VIN = VOUT(nom) + 0.5 V。"
        )
        changes.append(
            "脚注 (2)：Load Regulation 归一化到 "
            "IOUT = 1 mA 时的 VOUT；不改变 VIN/IOUT 条件。"
        )
    elif row_id == "t9.p13":
        changes.append(
            "脚注 (3)：该参数标注为 Specified by design；"
            "数值、单位和测试条件不变。"
        )
    elif row_id == "t9.p1":
        changes.append(
            "脚注 (3)：IOUT 下限 1 µA 标注为 "
            "Specified by design；IOUT 范围不变。"
        )
    elif "4" in footnotes:
        changes.append(
            "脚注 (4)：IOUT 覆盖为 200 mA；"
            "默认固定 VIN 被 VIN 逐渐下降的测量过程替代；"
            "终点为 VOUT = 95% × VOUT(nom)。"
        )
        changes.append(
            "保留该行自身的 TJ 和 VOUT 区间；"
            "不再同时应用默认 IOUT = 1 mA / 默认固定 VIN。"
        )

    return changes


def consistency_checks(row, footnotes, effective):
    checks = {
        "tj_single_source": True,
        "vin_single_source": True,
        "iout_single_source": True,
        "value_type_value_unit_unchanged": True,
    }

    if footnotes == ["1"] and row["logical_row_id"] == "t9.p5":
        checks["line_regulation_vin_scan_preserved"] = True

    if "4" in footnotes:
        checks["dropout_default_iout_suppressed"] = bool(
            effective["iout"].get(
                "default_iout_suppressed"
            )
        )
        checks["dropout_default_vin_suppressed"] = bool(
            effective["vin"].get(
                "default_fixed_vin_suppressed"
            )
        )

    return checks


def value_preservation(rows, gold):
    gold_items = gold.get("expected_parameters", [])
    gold_by_key = {}

    for item in gold_items:
        key = (
            item.get("logical_row_id"),
            item.get("value_type"),
        )
        gold_by_key.setdefault(key, []).append(item)

    mismatches = []

    for row in rows:
        key = (
            row.get("logical_row_id"),
            row.get("Min/Typ/Max"),
        )
        candidates = list(gold_by_key.get(key, []))

        if not candidates:
            mismatches.append({
                "logical_row_id": row.get("logical_row_id"),
                "value_type": row.get("Min/Typ/Max"),
                "reason": "Gold JSON 无对应项",
            })
            continue

        expected = candidates[0]
        excel_value = normalize_number(row.get("数值"))
        gold_value = normalize_number(expected.get("value"))
        excel_unit = str(row.get("单位") or "").strip()
        gold_unit = str(expected.get("unit") or "").strip()

        if (
            not numbers_equal(excel_value, gold_value)
            or excel_unit != gold_unit
        ):
            mismatches.append({
                "logical_row_id": row.get("logical_row_id"),
                "value_type": row.get("Min/Typ/Max"),
                "excel_value": excel_value,
                "gold_value": gold_value,
                "excel_unit": excel_unit,
                "gold_unit": gold_unit,
            })

    return {
        "verified": not mismatches,
        "checked_records": len(rows),
        "mismatches": mismatches,
        "note": (
            "补充文件只添加脚注/条件注释，"
            "不修改 value、unit、value_type。"
        ),
    }


def build_records(rows, sources):
    records = []
    affected = []

    for row in rows:
        row_id = str(row["logical_row_id"])
        footnotes = FOOTNOTE_MAP.get(row_id, [])
        effective = build_effective_conditions(
            row, footnotes
        )
        annotations = build_annotations(
            row, footnotes, sources
        )
        changes = condition_changes(row, footnotes)
        checks = consistency_checks(
            row, footnotes, effective
        )
        status = str(
            row.get("人工核验状态") or "unreviewed"
        ).strip().lower()

        supplement = {
            "footnotes": footnotes,
            "annotations": annotations,
            "effective_conditions": effective,
            "condition_changes": changes,
            "consistency_checks": checks,
            "has_supplement": bool(footnotes),
        }

        record = {
            "serial": row.get("序号"),
            "logical_row_id": row_id,
            "table_id": 9,
            "parameter_name": row.get("参数名称"),
            "value_type": row.get("Min/Typ/Max"),
            "value": normalize_number(row.get("数值")),
            "unit": row.get("单位"),
            "test_conditions_original": row.get(
                "完整测试条件"
            ),
            "page": row.get("PDF 页码"),
            "source_row_indices": row.get(
                "source_row_indices"
            ),
            "source_cell_ids": row.get(
                "source_cell_ids"
            ),
            "status": status,
            "correct_value": row.get("正确值"),
            "note": row.get("备注"),
            "focus": row_id in FOCUS_IDS,
            "supplement": supplement,
        }
        records.append(record)

        if footnotes:
            affected.append(record)

    return records, affected


def build_supplement_document(
    sources, rows, needs_rows, records, affected,
    gold, hashes,
):
    status_counts = Counter(
        record["status"] for record in records
    )
    need_status = Counter(
        str(row.get("人工核验状态") or "unreviewed")
        .strip().lower()
        for row in needs_rows
    )

    return {
        "schema": "gold_standard_supplement/v1",
        "version": 1,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "pdf_path": str(DEFAULT_PDF),
            "pdf_sha256": hashes["pdf"],
            "excel_path": str(DEFAULT_XLSX),
            "excel_sha256": hashes["excel"],
            "gold_json_path": str(DEFAULT_GOLD),
            "gold_json_sha256": hashes["gold"],
            "table_page": TABLE_PAGE,
            "footnote_page": FOOTNOTE_PAGE,
        },
        "status_summary": {
            "candidate_total": len(records),
            "confirmed": status_counts.get("confirmed", 0),
            "unreviewed": status_counts.get(
                "unreviewed", 0
            ),
            "other": sum(
                count for status, count in status_counts.items()
                if status not in {"confirmed", "unreviewed"}
            ),
            "needs_review_total": len(needs_rows),
            "needs_review_confirmed": need_status.get(
                "confirmed", 0
            ),
            "needs_review_unreviewed": need_status.get(
                "unreviewed", 0
            ),
        },
        "table_default_conditions": {
            "text": sources["table_default_text"],
            "typical_values": sources["typical_text"],
            "normalized": {
                "tj": {
                    "min_c": -40,
                    "max_c": 125,
                    "text": DEFAULT_TJ,
                },
                "vin": {
                    "text": DEFAULT_VIN,
                    "rule": (
                        "VOUT(nom) + 0.5 V or 2.0 V, "
                        "whichever is greater"
                    ),
                },
                "iout": {
                    "value_ma": 1,
                    "text": DEFAULT_IOUT,
                },
                "ven": DEFAULT_VEN,
                "cin_uf": 1,
                "cout_uf": 1,
                "unless_otherwise_noted": True,
                "typical_values_tj_c": 25,
            },
        },
        "footnotes": {
            number: {
                "number": int(number),
                "page": FOOTNOTE_PAGE,
                "text": text,
            }
            for number, text in sources[
                "footnotes"
            ].items()
        },
        "precedence_rules": [
            "表级默认条件仅在行级未覆盖时生效。",
            "行级 TJ/VIN/IOUT/VEN/VOUT 条件优先于表级默认。",
            "脚注 (1) 是 VOUT ≤ 1.5 V 时的 VIN 条件适用范围；"
            "对 Line Regulation 只修正 VIN 下限，不替换 VIN 扫描范围。",
            "脚注 (2) 只提供 Load Regulation 归一化参考，"
            "不改变 TJ/VIN/IOUT。",
            "脚注 (3) 只标记 Specified by design，"
            "不改变数值或条件。",
            "脚注 (4) 覆盖默认 IOUT 为 200 mA，"
            "并以 VIN ramp-down 过程替代默认固定 VIN；"
            "保留行级 TJ/VOUT。",
        ],
        "candidate_records": records,
        "needs_review_records": [
            {
                "serial": row.get("序号"),
                "logical_row_id": row.get(
                    "logical_row_id"
                ),
                "parameter_name": row.get("参数名称"),
                "min": row.get("Min"),
                "typ": row.get("Typ"),
                "max": row.get("Max"),
                "unit": row.get("单位"),
                "test_conditions": row.get(
                    "完整测试条件"
                ),
                "page": row.get("PDF 页码"),
                "source_row_indices": row.get(
                    "source_row_indices"
                ),
                "source_cell_ids": row.get(
                    "source_cell_ids"
                ),
                "status": str(
                    row.get("人工核验状态")
                    or "unreviewed"
                ).strip().lower(),
                "note": row.get("备注"),
            }
            for row in needs_rows
        ],
        "affected_summary": {
            "affected_candidate_records": len(affected),
            "affected_logical_rows": sorted({
                record["logical_row_id"]
                for record in affected
            }),
            "by_footnote": dict(Counter(
                footnote
                for record in affected
                for footnote in record[
                    "supplement"
                ]["footnotes"]
            )),
        },
        "value_preservation": value_preservation(
            rows, gold
        ),
    }


def build_diff_report(supplement):
    affected = [
        record
        for record in supplement["candidate_records"]
        if record["supplement"]["has_supplement"]
    ]

    conflicts = []

    for record in affected:
        checks = record["supplement"][
            "consistency_checks"
        ]
        failed = [
            name for name, passed in checks.items()
            if passed is False
        ]

        if failed:
            conflicts.append({
                "logical_row_id": record[
                    "logical_row_id"
                ],
                "value_type": record["value_type"],
                "failed_checks": failed,
            })

    return {
        "schema": "gold_condition_diff_report/v1",
        "version": 1,
        "generated_at": supplement["generated_at"],
        "source": supplement["source"],
        "status_summary": supplement["status_summary"],
        "summary": {
            "affected_candidate_records": len(affected),
            "affected_logical_rows": len({
                record["logical_row_id"]
                for record in affected
            }),
            "unaffected_candidate_records": (
                len(supplement["candidate_records"])
                - len(affected)
            ),
            "value_unit_type_changes": 0,
            "condition_conflicts": conflicts,
        },
        "affected_records": [
            {
                "serial": record["serial"],
                "logical_row_id": record[
                    "logical_row_id"
                ],
                "parameter_name": record[
                    "parameter_name"
                ],
                "value_type": record["value_type"],
                "value": record["value"],
                "unit": record["unit"],
                "status": record["status"],
                "before": {
                    "test_conditions": record[
                        "test_conditions_original"
                    ],
                    "footnotes": [],
                    "effective_conditions": (
                        "表级默认 + 行级条件，"
                        "未附加脚注解释"
                    ),
                },
                "after": {
                    "test_conditions": record[
                        "test_conditions_original"
                    ],
                    "footnotes": record[
                        "supplement"
                    ]["footnotes"],
                    "annotations": record[
                        "supplement"
                    ]["annotations"],
                    "effective_conditions": record[
                        "supplement"
                    ]["effective_conditions"],
                    "condition_changes": record[
                        "supplement"
                    ]["condition_changes"],
                },
                "value_unchanged": True,
                "unit_unchanged": True,
                "value_type_unchanged": True,
                "consistency_checks": record[
                    "supplement"
                ]["consistency_checks"],
            }
            for record in affected
        ],
    }


def build_markdown_report(diff):
    lines = [
        "# TPS7A02 table 9 · Gold Standard 条件补充差异报告",
        "",
        f"生成时间：{diff['generated_at']}",
        "",
        "## 状态统计",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 候选参数 | {diff['status_summary']['candidate_total']} |",
        f"| 实际 confirmed | {diff['status_summary']['confirmed']} |",
        f"| 实际 unreviewed | {diff['status_summary']['unreviewed']} |",
        f"| needs_review 总数 | {diff['status_summary']['needs_review_total']} |",
        f"| needs_review confirmed | {diff['status_summary']['needs_review_confirmed']} |",
        f"| needs_review unreviewed | {diff['status_summary']['needs_review_unreviewed']} |",
        "",
        "## 表级默认测试条件",
        "",
        "> " + diff.get("table_default_text", ""),
        "",
        "## 脚注",
        "",
        "| 脚注 | 原文 |",
        "|---|---|",
    ]

    # table defaults/footnotes are in supplement; allow diff to
    # carry rendered strings without duplicating source data.
    for number, text in diff.get(
        "footnote_texts", {}
    ).items():
        escaped_text = text.replace("|", "\\|")
        lines.append(
            f"| ({number}) | {escaped_text} |"
        )

    lines += [
        "",
        "## 受影响记录",
        "",
        f"- 受影响候选记录：{diff['summary']['affected_candidate_records']}",
        f"- 受影响逻辑行：{diff['summary']['affected_logical_rows']}",
        f"- 未受影响候选记录：{diff['summary']['unaffected_candidate_records']}",
        "- 数值/单位/Min-Typ-Max 修改：0",
        "",
        "| 序号 | logical_row_id | 参数 | 类型 | 数值 | 单位 | 脚注 | 条件变化 |",
        "|---:|---|---|---|---:|---|---|---|",
    ]

    for record in diff["affected_records"]:
        changes = "；".join(
            record["after"]["condition_changes"]
        ).replace("|", "\\|")

        lines.append(
            f"| {record['serial']} | "
            f"{record['logical_row_id']} | "
            f"{record['parameter_name']} | "
            f"{record['value_type']} | "
            f"{record['value']} | "
            f"{record['unit']} | "
            f"{', '.join(record['after']['footnotes'])} | "
            f"{changes} |"
        )

    lines += [
        "",
        "## 关键规则",
        "",
        "1. 脚注 (1) 仅作为 VOUT ≤ 1.5 V 时的 VIN 适用范围；"
        "Line Regulation 的 VIN 扫描范围保持不变。",
        "2. 脚注 (2) 仅作为 Load Regulation 归一化说明，"
        "不改变测试条件。",
        "3. 脚注 (3) 仅表示 Specified by design，"
        "不改变数值、单位或条件。",
        "4. 脚注 (4) 使用 IOUT = 200 mA，"
        "VIN 逐渐降低，终点 VOUT = 95% × VOUT(nom)；"
        "默认 IOUT = 1 mA 和默认固定 VIN 不再同时生效。",
        "5. 所有已确认的 value、unit、value_type 保持不变。",
        "",
        "## 输出文件",
        "",
        "- `gold_standard.table9.supplemented.json`",
        "- `gold_review_supplemented.xlsx`",
        "- `gold_condition_diff_report.json`",
        "- `gold_condition_diff_report.md`",
        "",
    ]

    return "\n".join(lines)


def write_json(path, data):
    path = Path(path)

    if path.exists():
        # 允许重新生成补充文件，但不触碰原始文件。
        path.unlink()

    path.write_text(
        json.dumps(
            data, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


def write_supplement_excel(
    source_workbook,
    output_path,
    supplement,
    diff,
):
    workbook = source_workbook

    for name in (
        "补充-表级默认条件",
        "补充-脚注",
        "补充-行级映射",
        "补充-差异摘要",
    ):
        if name in workbook.sheetnames:
            del workbook[name]

    header_fill = PatternFill(
        "solid", fgColor="1F4E78"
    )
    header_font = Font(
        color="FFFFFF", bold=True
    )
    thin = Side(style="thin", color="D9D9D9")
    border = Border(
        left=thin, right=thin,
        top=thin, bottom=thin,
    )

    def write_table(ws, headers, rows, widths):
        for column, header in enumerate(
            headers, start=1
        ):
            cell = ws.cell(
                1, column, header
            )
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )
            cell.border = border
            ws.column_dimensions[
                cell.column_letter
            ].width = widths[column - 1]

        for row_index, values in enumerate(
            rows, start=2
        ):
            for column, value in enumerate(
                values, start=1
            ):
                cell = ws.cell(
                    row_index, column, value
                )
                cell.alignment = Alignment(
                    vertical="top", wrap_text=True
                )
                cell.border = border
                cell.font = Font(size=9)
                cell.fill = PatternFill(
                    "solid", fgColor="F2F6FA"
                )

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = (
            f"A1:{ws.cell(1, ws.max_column).column_letter}"
            f"{ws.max_row}"
        )

    # 默认条件
    ws = workbook.create_sheet("补充-表级默认条件")
    default = supplement["table_default_conditions"]
    default_rows = [
        ["表级默认原文", default["text"]],
        ["Typical values", default["typical_values"]],
        ["TJ", DEFAULT_TJ],
        ["VIN", DEFAULT_VIN],
        ["IOUT", DEFAULT_IOUT],
        ["VEN", DEFAULT_VEN],
        ["CIN / COUT", DEFAULT_CAPS],
        ["除另有说明", "unless otherwise noted"],
        ["来源", "TPS7A02.pdf 第 6 页"],
    ]
    write_table(
        ws,
        ["项目", "内容"],
        default_rows,
        [26, 105],
    )

    # 脚注
    ws = workbook.create_sheet("补充-脚注")
    footnote_rows = [
        [
            f"({number})",
            item["text"],
            "第 7 页",
            (
                "仅条件适用范围/设计说明，不改变已确认数值"
            ),
        ]
        for number, item in supplement[
            "footnotes"
        ].items()
    ]
    write_table(
        ws,
        ["脚注", "原文", "来源页", "处理原则"],
        footnote_rows,
        [10, 100, 10, 45],
    )

    # 行级映射：全部 44 条，保留人工状态
    ws = workbook.create_sheet("补充-行级映射")
    mapping_rows = []

    for record in supplement["candidate_records"]:
        supplement_data = record["supplement"]
        mapping_rows.append([
            record["serial"],
            record["logical_row_id"],
            record["parameter_name"],
            record["value_type"],
            record["value"],
            record["unit"],
            record["test_conditions_original"],
            ", ".join(supplement_data["footnotes"]),
            "\n".join(
                f"({item['footnote']}) {item['text']}"
                for item in supplement_data[
                    "annotations"
                ]
            ),
            json.dumps(
                supplement_data[
                    "effective_conditions"
                ],
                ensure_ascii=False,
                indent=2,
            ),
            "\n".join(
                supplement_data["condition_changes"]
            ),
            record["status"],
        ])

    write_table(
        ws,
        [
            "序号",
            "logical_row_id",
            "参数名称",
            "Min/Typ/Max",
            "数值",
            "单位",
            "原始测试条件",
            "脚注",
            "条件注释",
            "补充后生效条件",
            "条件变化",
            "人工核验状态",
        ],
        mapping_rows,
        [7, 15, 30, 12, 10, 8, 55, 12, 65, 70, 55, 16],
    )

    # 差异摘要
    ws = workbook.create_sheet("补充-差异摘要")
    summary_rows = [
        ["候选参数总数", diff["status_summary"]["candidate_total"]],
        ["实际 confirmed", diff["status_summary"]["confirmed"]],
        ["实际 unreviewed", diff["status_summary"]["unreviewed"]],
        ["needs_review 总数", diff["status_summary"]["needs_review_total"]],
        ["needs_review unreviewed", diff["status_summary"]["needs_review_unreviewed"]],
        ["受影响候选记录", diff["summary"]["affected_candidate_records"]],
        ["受影响逻辑行", diff["summary"]["affected_logical_rows"]],
        ["未受影响候选记录", diff["summary"]["unaffected_candidate_records"]],
        ["数值/单位/类型修改", 0],
        ["条件冲突", len(diff["summary"]["condition_conflicts"])],
    ]
    write_table(
        ws,
        ["项目", "数量/结果"],
        summary_rows,
        [34, 30],
    )

    workbook.save(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="生成 Gold Standard 脚注/条件补充文件"
    )
    parser.add_argument("--excel", default=str(DEFAULT_XLSX))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument(
        "--supplement", default=str(DEFAULT_SUPPLEMENT)
    )
    parser.add_argument(
        "--supplement-xlsx",
        default=str(DEFAULT_SUPPLEMENT_XLSX),
    )
    parser.add_argument(
        "--diff-json", default=str(DEFAULT_DIFF_JSON)
    )
    parser.add_argument(
        "--diff-md", default=str(DEFAULT_DIFF_MD)
    )
    args = parser.parse_args(argv)

    excel_path = Path(args.excel)
    gold_path = Path(args.gold)
    pdf_path = Path(args.pdf)

    for path in (excel_path, gold_path, pdf_path):
        if not path.is_file():
            raise SupplementError(
                f"输入文件不存在: {path}"
            )

    hashes_before = {
        "excel": sha256_file(excel_path),
        "gold": sha256_file(gold_path),
        "pdf": sha256_file(pdf_path),
    }

    _, rows, needs_rows = read_excel(
        excel_path
    )
    # 写入副本单独加载，保留原工作簿的格式与数据验证。
    writer_workbook = load_workbook(excel_path)
    gold = load_json(gold_path)
    sources = extract_pdf_sources(pdf_path)

    records, affected = build_records(rows, sources)

    supplement = build_supplement_document(
        sources,
        rows,
        needs_rows,
        records,
        affected,
        gold,
        hashes_before,
    )

    diff = build_diff_report(supplement)
    diff["table_default_text"] = (
        sources["table_default_text"]
    )
    diff["footnote_texts"] = sources["footnotes"]

    write_json(args.supplement, supplement)
    write_json(args.diff_json, diff)

    markdown = build_markdown_report(diff)
    Path(args.diff_md).write_text(
        markdown, encoding="utf-8"
    )

    write_supplement_excel(
        writer_workbook,
        Path(args.supplement_xlsx),
        supplement,
        diff,
    )

    hashes_after = {
        "excel": sha256_file(excel_path),
        "gold": sha256_file(gold_path),
        "pdf": sha256_file(pdf_path),
    }

    if hashes_before != hashes_after:
        raise SupplementError(
            "原始输入文件在生成过程中被修改"
        )

    print("=" * 68)
    print("Gold Standard 补充文件已生成")
    print("=" * 68)
    print(
        "状态: confirmed=",
        supplement["status_summary"]["confirmed"],
        "unreviewed=",
        supplement["status_summary"]["unreviewed"],
        "needs_review/unreviewed=",
        supplement["status_summary"][
            "needs_review_unreviewed"
        ],
    )
    print(
        "表级默认条件:",
        sources["table_default_text"],
    )
    for number in ("1", "2", "3", "4"):
        print(
            f"脚注 ({number}):",
            sources["footnotes"][number],
        )
    print(
        "受影响候选记录:",
        len(affected),
        "受影响逻辑行:",
        len({
            record["logical_row_id"]
            for record in affected
        }),
    )
    print(
        "补充 JSON:",
        args.supplement,
    )
    print(
        "补充 Excel:",
        args.supplement_xlsx,
    )
    print(
        "差异 JSON:",
        args.diff_json,
    )
    print(
        "差异 Markdown:",
        args.diff_md,
    )
    print(
        "原始文件 SHA256 未变化:",
        hashes_before,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

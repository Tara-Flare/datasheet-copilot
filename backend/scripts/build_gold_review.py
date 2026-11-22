"""阶段 B3.2b：生成 Gold Standard 人工核验包。

只读取：
- gold_standard.table9.json
- TPS7A02.logical_rows.v1.json
- TPS7A02.raw_structure.v1.json
- TPS7A02.pdf

生成：
- gold_review.xlsx
- gold_review.html
- pdf_regions/*.png

不修改任何原始文件。
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pymupdf
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
)
DEFAULT_GOLD = DEFAULT_DIR / "gold_standard.table9.json"
DEFAULT_LOGICAL = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.logical_rows.v1.json"
)
DEFAULT_RAW = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.raw_structure.v1.json"
)
DEFAULT_PDF = (
    PROJECT_ROOT / "data/samples/TPS7A02.pdf"
)

FOCUS_IDS = {
    "t9.p9",
    "t9.p10",
    "t9.p24",
    "t9.p25",
    "t9.p26",
    "t9.p30",
    "t9.p31",
}

STATUS_OPTIONS = [
    "unreviewed",
    "confirmed",
    "corrected",
    "rejected",
    "needs_review",
]


class ReviewBuildError(ValueError):
    """核验包生成失败。"""


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise ReviewBuildError(f"文件不存在: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def index_cells(raw_structure, table_id):
    table = next(
        (
            item
            for item in raw_structure.get("tables", [])
            if item.get("table_id") == table_id
        ),
        None,
    )

    if table is None:
        raise ReviewBuildError(
            f"raw_structure 缺少 table_id={table_id}"
        )

    return table, {
        cell.get("cell_id"): cell
        for cell in table.get("cells", [])
    }


def index_logical_rows(logical_rows, table_id):
    rows = {}

    groups = logical_rows.get("tables")

    if not isinstance(groups, list) or not groups:
        groups = [logical_rows]

    for group in groups:
        if group.get("table_id") != table_id:
            continue

        for row in group.get("logical_rows", []):
            row_id = row.get("logical_row_id")

            if row_id:
                rows[row_id] = row

    if not rows:
        raise ReviewBuildError(
            f"logical_rows 缺少 table_id={table_id}"
        )

    return rows


def normalized_number(value):
    text = str(value or "").strip()
    return (
        text.replace("–", "-")
        .replace("−", "-")
        .replace("‒", "-")
    )


def cell_texts_for(record, cells):
    result = []

    for cell_id in record.get("source_cell_ids", []):
        cell = cells.get(cell_id)

        if cell is None:
            result.append({
                "cell_id": cell_id,
                "text": "",
                "missing": True,
            })
            continue

        result.append({
            "cell_id": cell_id,
            "text": cell.get("text", ""),
        })

    return result


def extract_pdf_text(page, rect):
    words = page.get_text("words", clip=rect)

    words.sort(
        key=lambda item: (
            round(item[1], 1),
            round(item[0], 1),
        )
    )

    return " ".join(item[4] for item in words)


def row_band(row):
    band = row.get("physical_row_band") or {}
    top = band.get("top")
    bottom = band.get("bottom")

    if top is None or bottom is None:
        return None

    return float(top), float(bottom)


def crop_rect_for(table, row, page_rect):
    band = row_band(row)

    if band is None:
        raise ReviewBuildError(
            "逻辑行缺少 physical_row_band: "
            f"{row.get('logical_row_id')}"
        )

    top, bottom = band
    table_bbox = table.get("table_bbox") or {}

    left = float(table_bbox.get("l", 40.0))
    right = float(table_bbox.get("r", 570.0))

    padding_x = 6.0
    padding_y = 9.0

    rect = pymupdf.Rect(
        max(page_rect.x0, left - padding_x),
        max(page_rect.y0, top - padding_y),
        min(page_rect.x1, right + padding_x),
        min(page_rect.y1, bottom + padding_y),
    )

    return rect, pymupdf.Rect(
        left,
        top,
        right,
        bottom,
    )


def render_region(
    pdf_path,
    page_number,
    crop_rect,
    highlight_rect,
    output_path,
):
    document = pymupdf.open(pdf_path)

    try:
        page = document[page_number - 1]

        # 仅内存绘制，不写回 PDF。
        page.draw_rect(
            highlight_rect,
            color=(0.85, 0.15, 0.10),
            width=1.1,
            overlay=True,
        )

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(3.2, 3.2),
            clip=crop_rect,
            alpha=False,
        )

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        pixmap.save(str(output_path))

    finally:
        document.close()


def render_full_page(
    pdf_path,
    page_number,
    output_path,
):
    document = pymupdf.open(pdf_path)

    try:
        page = document[page_number - 1]

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5),
            alpha=False,
        )

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        pixmap.save(str(output_path))

    finally:
        document.close()


def make_display_name(record):
    symbol = str(
        record.get("parameter_symbol") or ""
    ).strip()
    name = str(
        record.get("parameter_name") or ""
    ).strip()

    if symbol and name:
        return f"{symbol} / {name}"

    return name or symbol or "(unnamed)"


def build_entries(
    gold,
    logical_rows,
    raw_structure,
    pdf_path,
    output_dir,
):
    table_id = gold.get("table_id", 9)
    table, cells = index_cells(
        raw_structure, table_id
    )
    logical_by_id = index_logical_rows(
        logical_rows, table_id
    )

    document = pymupdf.open(pdf_path)
    pages = {}

    try:
        for row_id, row in logical_by_id.items():
            page_number = int(row.get("page") or 6)
            page = document[page_number - 1]
            pages[page_number] = page

            crop_rect, highlight = crop_rect_for(
                table, row, page.rect
            )
            image_name = f"{row_id}.png"
            image_path = (
                output_dir / "pdf_regions" / image_name
            )

            render_region(
                pdf_path,
                page_number,
                crop_rect,
                highlight,
                image_path,
            )

            page_image_name = f"page{page_number}.png"
            page_image_path = (
                output_dir
                / "pdf_regions"
                / page_image_name
            )

            if not page_image_path.exists():
                render_full_page(
                    pdf_path,
                    page_number,
                    page_image_path,
                )

            row["__review_image"] = (
                f"pdf_regions/{image_name}"
            )
            row["__review_page_image"] = (
                f"pdf_regions/{page_image_name}"
            )
            row["__review_pdf_text"] = extract_pdf_text(
                page, crop_rect
            )
            row["__review_cells"] = cell_texts_for(
                row, cells
            )
    finally:
        document.close()

    entries = []

    for serial, record in enumerate(
        gold.get("expected_parameters", []),
        start=1,
    ):
        row = logical_by_id.get(
            record.get("logical_row_id"), {}
        )
        value = normalized_number(record.get("value"))

        entries.append({
            "kind": "candidate",
            "serial": serial,
            "logical_row_id": record.get(
                "logical_row_id"
            ),
            "display_name": make_display_name(record),
            "value_type": record.get("value_type"),
            "value": value,
            "raw_value": record.get("value"),
            "unit": record.get("unit"),
            "test_conditions": record.get(
                "test_conditions"
            ),
            "page": record.get("page"),
            "source_row_indices": list(
                record.get("source_row_indices") or []
            ),
            "source_cell_ids": list(
                record.get("source_cell_ids") or []
            ),
            "status": "unreviewed",
            "correct_value": "",
            "note": "",
            "focus": record.get("logical_row_id")
            in FOCUS_IDS,
            "needs_review": bool(
                record.get("needs_review")
            ),
            "image": row.get("__review_image", ""),
            "page_image": row.get(
                "__review_page_image", ""
            ),
            "pdf_text": row.get(
                "__review_pdf_text", ""
            ),
            "cell_texts": row.get(
                "__review_cells", []
            ),
        })

    needs_entries = []

    for serial, item in enumerate(
        gold.get("excluded_needs_review", []),
        start=1,
    ):
        row = logical_by_id.get(
            item.get("logical_row_id"), {}
        )
        record = {
            "parameter_symbol": row.get(
                "parameter_symbol"
            ),
            "parameter_name": row.get(
                "parameter_name"
            ),
        }

        needs_entries.append({
            "kind": "needs_review",
            "serial": serial,
            "logical_row_id": item.get(
                "logical_row_id"
            ),
            "display_name": make_display_name(record),
            "min": normalized_number(row.get("min")),
            "typ": normalized_number(row.get("typ")),
            "max": normalized_number(row.get("max")),
            "unit": row.get("unit"),
            "test_conditions": row.get(
                "test_conditions"
            ),
            "page": row.get("page"),
            "source_row_indices": list(
                item.get("source_row_indices") or []
            ),
            "source_cell_ids": list(
                item.get("source_cell_ids") or []
            ),
            "status": "unreviewed",
            "correct_value": "",
            "note": "B3.1 needs_review=true，禁止自动确认",
            "focus": item.get("logical_row_id")
            in FOCUS_IDS,
            "needs_review": True,
            "reason": item.get("reason"),
            "image": row.get("__review_image", ""),
            "page_image": row.get(
                "__review_page_image", ""
            ),
            "pdf_text": row.get(
                "__review_pdf_text", ""
            ),
            "cell_texts": row.get(
                "__review_cells", []
            ),
        })

    return entries, needs_entries


def style_header(ws, columns):
    fill = PatternFill(
        "solid", fgColor="1F4E78"
    )
    font = Font(
        color="FFFFFF", bold=True, size=10
    )
    alignment = Alignment(
        horizontal="center",
        vertical="center",
        wrap_text=True,
    )
    thin = Side(style="thin", color="BFBFBF")
    border = Border(
        left=thin, right=thin,
        top=thin, bottom=thin,
    )

    for index, (name, width) in enumerate(
        columns, start=1
    ):
        cell = ws.cell(row=1, column=index)

        if cell.value is None:
            cell.value = name

        cell.fill = fill
        cell.font = font
        cell.alignment = alignment
        cell.border = border
        ws.column_dimensions[
            get_column_letter(index)
        ].width = width

    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"


def add_status_validation(ws, column, last_row):
    validation = DataValidation(
        type="list",
        formula1='"'
        + ",".join(STATUS_OPTIONS)
        + '"',
        allow_blank=False,
        showDropDown=False,
    )
    validation.error = "请选择规定状态"
    validation.errorTitle = "状态不合法"
    validation.prompt = (
        "unreviewed / confirmed / corrected / "
        "rejected / needs_review"
    )
    validation.promptTitle = "人工核验状态"
    ws.add_data_validation(validation)
    validation.add(
        f"{column}2:{column}{last_row}"
    )

    fills = {
        "confirmed": "C6EFCE",
        "corrected": "FFEB9C",
        "rejected": "FFC7CE",
        "needs_review": "FCE4D6",
        "unreviewed": "EDEDED",
    }

    for status, color in fills.items():
        ws.conditional_formatting.add(
            f"{column}2:{column}{last_row}",
            CellIsRule(
                operator="equal",
                formula=[f'"{status}"'],
                fill=PatternFill(
                    "solid", fgColor=color
                ),
            ),
        )


def build_excel(
    entries,
    needs_entries,
    output_path,
    gold,
    pdf_path,
    pdf_sha256,
):
    workbook = Workbook()
    ws = workbook.active
    ws.title = "候选参数"

    columns = [
        ("序号", 7),
        ("logical_row_id", 15),
        ("参数名称", 31),
        ("Min/Typ/Max", 13),
        ("数值", 11),
        ("单位", 8),
        ("完整测试条件", 55),
        ("PDF 页码", 9),
        ("source_row_indices", 19),
        ("人工核验状态", 16),
        ("正确值", 14),
        ("备注", 30),
        ("重点标记", 10),
        ("source_cell_ids", 24),
        ("PDF 核验截图", 25),
        ("PDF 原文（bbox）", 60),
    ]

    for index, (name, _) in enumerate(
        columns, start=1
    ):
        ws.cell(row=1, column=index, value=name)

    style_header(ws, columns)

    thin = Side(style="thin", color="D9D9D9")
    border = Border(
        left=thin, right=thin,
        top=thin, bottom=thin,
    )
    wrap = Alignment(
        vertical="top", wrap_text=True
    )
    center = Alignment(
        vertical="top",
        horizontal="center",
        wrap_text=True,
    )
    focus_fill = PatternFill(
        "solid", fgColor="FFF2CC"
    )
    hyperlink_font = Font(
        color="0563C1", underline="single"
    )

    for row_index, entry in enumerate(
        entries, start=2
    ):
        values = [
            entry["serial"],
            entry["logical_row_id"],
            entry["display_name"],
            entry["value_type"],
            entry["value"],
            entry["unit"],
            entry["test_conditions"],
            entry["page"],
            ", ".join(
                str(value)
                for value in entry[
                    "source_row_indices"
                ]
            ),
            entry["status"],
            entry["correct_value"],
            entry["note"],
            "★ 重点" if entry["focus"] else "",
            ", ".join(
                str(value)
                for value in entry[
                    "source_cell_ids"
                ]
            ),
            entry["image"],
            entry["pdf_text"],
        ]

        for column, value in enumerate(
            values, start=1
        ):
            cell = ws.cell(
                row=row_index,
                column=column,
                value=value,
            )
            cell.border = border
            cell.alignment = (
                center
                if column in {1, 4, 6, 8, 13}
                else wrap
            )
            cell.font = Font(size=9)

            if entry["focus"]:
                cell.fill = focus_fill

        image_cell = ws.cell(
            row=row_index, column=15
        )
        image_cell.hyperlink = entry["image"]
        image_cell.font = hyperlink_font

        ws.row_dimensions[row_index].height = 46

    last_row = len(entries) + 1
    ws.auto_filter.ref = (
        f"A1:P{last_row}"
    )
    add_status_validation(ws, "J", last_row)

    # needs_review 单独工作表：绝不预填 confirmed。
    ws2 = workbook.create_sheet("needs_review")
    needs_columns = [
        ("序号", 7),
        ("logical_row_id", 15),
        ("参数名称", 31),
        ("Min", 9),
        ("Typ", 9),
        ("Max", 9),
        ("单位", 8),
        ("完整测试条件", 55),
        ("PDF 页码", 9),
        ("source_row_indices", 19),
        ("人工核验状态", 16),
        ("正确值", 14),
        ("备注", 38),
        ("重点标记", 10),
        ("source_cell_ids", 24),
        ("PDF 核验截图", 25),
        ("PDF 原文（bbox）", 60),
    ]

    for index, (name, _) in enumerate(
        needs_columns, start=1
    ):
        ws2.cell(row=1, column=index, value=name)

    style_header(ws2, needs_columns)

    for row_index, entry in enumerate(
        needs_entries, start=2
    ):
        values = [
            entry["serial"],
            entry["logical_row_id"],
            entry["display_name"],
            entry["min"],
            entry["typ"],
            entry["max"],
            entry["unit"],
            entry["test_conditions"],
            entry["page"],
            ", ".join(
                str(value)
                for value in entry[
                    "source_row_indices"
                ]
            ),
            entry["status"],
            entry["correct_value"],
            entry["note"],
            "★ 重点" if entry["focus"] else "",
            ", ".join(
                str(value)
                for value in entry[
                    "source_cell_ids"
                ]
            ),
            entry["image"],
            entry["pdf_text"],
        ]

        for column, value in enumerate(
            values, start=1
        ):
            cell = ws2.cell(
                row=row_index,
                column=column,
                value=value,
            )
            cell.border = border
            cell.alignment = (
                center
                if column in {1, 4, 5, 6, 7, 9, 14}
                else wrap
            )
            cell.font = Font(size=9)
            cell.fill = PatternFill(
                "solid", fgColor="FCE4D6"
            )

        image_cell = ws2.cell(
            row=row_index, column=16
        )
        image_cell.hyperlink = entry["image"]
        image_cell.font = hyperlink_font
        ws2.row_dimensions[row_index].height = 46

    last_row2 = len(needs_entries) + 1
    ws2.auto_filter.ref = (
        f"A1:Q{last_row2}"
    )
    add_status_validation(ws2, "K", last_row2)

    # 说明页
    ws3 = workbook.create_sheet("核验说明")
    ws3.sheet_view.showGridLines = False
    ws3.column_dimensions["A"].width = 24
    ws3.column_dimensions["B"].width = 110

    notes = [
        ("文件用途", "Gold Standard table_id=9 人工核验包"),
        ("生成时间", datetime.now(timezone.utc).isoformat()),
        ("PDF", str(pdf_path)),
        ("PDF SHA256", pdf_sha256),
        ("Gold JSON", "gold_standard.table9.json（只读）"),
        ("候选参数数", str(len(entries))),
        ("needs_review 数", str(len(needs_entries))),
        (
            "默认状态",
            "所有行均为 unreviewed；没有任何自动 confirmed。",
        ),
        (
            "重点核验行",
            ", ".join(sorted(FOCUS_IDS)),
        ),
        (
            "状态含义",
            "unreviewed=未核验；confirmed=人工确认正确；"
            "corrected=人工纠错；rejected=应删除；"
            "needs_review=需进一步讨论。",
        ),
        (
            "正确值",
            "人工发现错误时填写正确数值；可同时修改单位/条件到备注。",
        ),
        (
            "PDF 核验依据",
            "右侧 PDF 核验截图列为对应逻辑行区域；"
            "PDF 原文（bbox）列为 PyMuPDF 从该区域提取的原文。",
        ),
        (
            "重要约束",
            "StructureRecovery 的 confirmed 状态不等于人工确认；"
            "本文件没有读取或继承任何自动 confirmed 状态。",
        ),
        (
            "needs_review",
            "t9.p34 / t9.p35 / t9.p36 位于独立工作表，"
            "不得自动标记为 confirmed。",
        ),
        (
            "HTML 页面",
            "同目录 gold_review.html 提供左侧候选、右侧 PDF 截图核验视图。",
        ),
    ]

    ws3["A1"] = "Gold Standard 人工核验说明"
    ws3["A1"].font = Font(
        bold=True, size=14, color="1F4E78"
    )
    ws3.merge_cells("A1:B1")

    for row, (key, value) in enumerate(
        notes, start=3
    ):
        key_cell = ws3.cell(
            row=row, column=1, value=key
        )
        value_cell = ws3.cell(
            row=row, column=2, value=value
        )
        key_cell.font = Font(bold=True)
        key_cell.alignment = Alignment(
            vertical="top", wrap_text=True
        )
        value_cell.alignment = Alignment(
            vertical="top", wrap_text=True
        )
        ws3.row_dimensions[row].height = 34

    workbook.save(output_path)


def build_html(entries, needs_entries, output_path):
    payload = {
        "candidates": entries,
        "needs_review": needs_entries,
        "focus_ids": sorted(FOCUS_IDS),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
    data_json = json.dumps(
        payload, ensure_ascii=False
    ).replace("</", "<\\/")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>TPS7A02 table 9 Gold Standard 人工核验</title>
<style>
:root {{
  --ink:#1f2937; --muted:#6b7280; --line:#d7dde5;
  --bg:#f6f8fb; --card:#ffffff; --blue:#1f4e78;
  --focus:#fff2cc; --review:#fce4d6;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family: -apple-system, "Segoe UI", "Noto Sans CJK SC",
    "Microsoft YaHei", sans-serif;
}}
header {{
  position:sticky; top:0; z-index:5; background:#fff;
  border-bottom:1px solid var(--line); padding:14px 22px;
}}
header h1 {{ margin:0 0 4px; font-size:20px; color:var(--blue); }}
header p {{ margin:0; color:var(--muted); font-size:12px; }}
.controls {{
  margin-top:10px; display:flex; gap:14px; flex-wrap:wrap;
  align-items:center; font-size:13px;
}}
.controls input[type=search] {{
  width:300px; padding:7px 10px; border:1px solid var(--line);
  border-radius:6px;
}}
.controls label {{ cursor:pointer; user-select:none; }}
.layout {{
  display:grid; grid-template-columns: minmax(420px, 1fr) minmax(520px, 1.1fr);
  gap:18px; padding:18px; align-items:start;
}}
.left {{ min-width:0; }}
.section-title {{
  font-size:14px; font-weight:700; color:var(--blue);
  margin:10px 4px 8px;
}}
.cards {{ display:flex; flex-direction:column; gap:8px; }}
.card {{
  background:var(--card); border:1px solid var(--line);
  border-left:4px solid transparent; border-radius:8px;
  padding:10px 12px; cursor:pointer; transition:.12s;
}}
.card:hover {{ border-color:#9db4cc; transform:translateX(1px); }}
.card.active {{ border-left-color:var(--blue); box-shadow:0 2px 10px rgba(31,78,120,.16); }}
.card.focus {{ background:var(--focus); }}
.card.needs {{ background:var(--review); }}
.card-head {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.serial {{ font-size:11px; color:var(--muted); }}
.id {{ font-weight:700; color:var(--blue); }}
.name {{ font-weight:600; }}
.badge {{
  display:inline-block; padding:2px 6px; border-radius:10px;
  font-size:11px; background:#e8eef5; color:#33506b;
}}
.badge.focus {{ background:#e8a400; color:#fff; }}
.badge.review {{ background:#c85b32; color:#fff; }}
.meta {{ margin-top:5px; color:var(--muted); font-size:12px; }}
.conditions {{ margin-top:5px; font-size:12px; line-height:1.45; }}
.right {{
  position:sticky; top:112px; background:var(--card);
  border:1px solid var(--line); border-radius:10px;
  overflow:hidden; box-shadow:0 3px 16px rgba(15,35,55,.06);
}}
.right-head {{ padding:14px 16px; border-bottom:1px solid var(--line); }}
.right-head h2 {{ margin:0; font-size:17px; color:var(--blue); }}
.right-body {{ padding:14px 16px; }}
.kv {{ display:grid; grid-template-columns:140px 1fr; gap:6px 10px; font-size:13px; }}
.kv dt {{ color:var(--muted); }}
.kv dd {{ margin:0; word-break:break-word; }}
.evidence {{ margin-top:12px; }}
.evidence img {{
  width:100%; max-height:260px; object-fit:contain;
  border:1px solid var(--line); border-radius:6px; background:#fff;
}}
.evidence .caption {{ margin-top:4px; font-size:11px; color:var(--muted); }}
pre {{
  white-space:pre-wrap; word-break:break-word; background:#f7f9fc;
  border:1px solid var(--line); border-radius:6px; padding:8px;
  font-size:12px; line-height:1.45; max-height:150px; overflow:auto;
}}
.cellchips {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }}
.chip {{ background:#eef3f8; border:1px solid #d8e2ec; border-radius:5px; padding:3px 6px; font-size:11px; }}
.empty {{ color:var(--muted); padding:30px; text-align:center; }}
@media (max-width:1050px) {{
  .layout {{ grid-template-columns:1fr; }}
  .right {{ position:static; }}
}}
</style>
</head>
<body>
<header>
  <h1>TPS7A02 · table_id=9 · Gold Standard 人工核验</h1>
  <p>所有候选默认 <b>unreviewed</b>；StructureRecovery confirmed 不等于人工确认。核验状态请在 Excel 中填写。</p>
  <div class="controls">
    <input id="search" type="search" placeholder="搜索 logical_row_id / 参数名 / 条件">
    <label><input id="onlyFocus" type="checkbox"> 只看重点行</label>
    <span id="count"></span>
  </div>
</header>
<div class="layout">
  <div class="left">
    <div class="section-title">候选参数</div>
    <div id="candidateCards" class="cards"></div>
    <div class="section-title">needs_review（独立记录，禁止自动确认）</div>
    <div id="reviewCards" class="cards"></div>
  </div>
  <div class="right">
    <div class="right-head">
      <h2 id="detailTitle">请选择左侧候选</h2>
      <div class="meta" id="detailSub"></div>
    </div>
    <div class="right-body" id="detailBody">
      <div class="empty">点击左侧任一条目查看 PDF 核验依据。</div>
    </div>
  </div>
</div>
<script id="gold-data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('gold-data').textContent);
const state = {{ selected: null, onlyFocus: false, query: '' }};

function esc(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }}[ch]));
}}
function rowIndices(value) {{
  return Array.isArray(value) ? value.join(', ') : esc(value);
}}
function cardHtml(item) {{
  const badgeFocus = item.focus ? '<span class="badge focus">★ 重点</span>' : '';
  const badgeReview = item.kind === 'needs_review' ? '<span class="badge review">needs_review</span>' : '';
  const vt = item.kind === 'candidate' ? esc(item.value_type) : 'min/typ/max';
  const val = item.kind === 'candidate'
    ? `${{esc(item.value)}} ${{esc(item.unit)}}`
    : `${{esc(item.min)}} / ${{esc(item.typ)}} / ${{esc(item.max)}} ${{esc(item.unit)}}`;
  return `<div class="card ${{item.focus ? 'focus' : ''}} ${{item.kind === 'needs_review' ? 'needs' : ''}}"
      data-id="${{esc(item.logical_row_id)}}" data-kind="${{esc(item.kind)}}">
    <div class="card-head">
      <span class="serial">#${{item.serial}}</span>
      <span class="id">${{esc(item.logical_row_id)}}</span>
      <span class="name">${{esc(item.display_name)}}</span>
      ${{badgeFocus}}${{badgeReview}}
    </div>
    <div class="meta">${{vt}} · ${{val}} · page ${{esc(item.page)}} · rows ${{rowIndices(item.source_row_indices)}}</div>
    <div class="conditions">${{esc(item.test_conditions)}}</div>
  </div>`;
}}
function candidates() {{
  return DATA.candidates;
}}
function reviewItems() {{
  return DATA.needs_review;
}}
function visible(item) {{
  if (state.onlyFocus && !item.focus) return false;
  if (!state.query) return true;
  const hay = [
    item.logical_row_id, item.display_name, item.test_conditions,
    item.value, item.unit, item.min, item.typ, item.max
  ].join(' ').toLowerCase();
  return hay.includes(state.query);
}}
function render() {{
  const cs = candidates().filter(visible);
  const rs = reviewItems().filter(visible);
  document.getElementById('candidateCards').innerHTML = cs.map(cardHtml).join('') || '<div class="empty">无匹配候选</div>';
  document.getElementById('reviewCards').innerHTML = rs.map(cardHtml).join('') || '<div class="empty">无匹配 needs_review 记录</div>';
  document.getElementById('count').textContent = `候选 ${{cs.length}} / ${{DATA.candidates.length}} · needs_review ${{rs.length}} / ${{DATA.needs_review.length}}`;
  document.querySelectorAll('.card').forEach(card => {{
    card.addEventListener('click', () => {{
      const kind = card.dataset.kind;
      const id = card.dataset.id;
      const list = kind === 'candidate' ? candidates() : reviewItems();
      const item = list.find(x => x.logical_row_id === id && x.kind === kind);
      if (item) select(item, card);
    }});
  }});
}}
function detailRows(item) {{
  if (item.kind === 'candidate') {{
    return `
      <dt>状态</dt><dd>unreviewed（请在 Excel 填写）</dd>
      <dt>Min/Typ/Max</dt><dd>${{esc(item.value_type)}}</dd>
      <dt>数值</dt><dd>${{esc(item.value)}} ${{esc(item.unit)}}</dd>`;
  }}
  return `
    <dt>状态</dt><dd>unreviewed（needs_review，禁止自动确认）</dd>
    <dt>Min / Typ / Max</dt><dd>${{esc(item.min)}} / ${{esc(item.typ)}} / ${{esc(item.max)}} ${{esc(item.unit)}}</dd>`;
}}
function cellChips(item) {{
  const cells = item.cell_texts || [];
  if (!cells.length) return '';
  return `<div class="cellchips">${{cells.map(c => `<span class="chip">cell ${{esc(c.cell_id)}}: ${{esc(c.text)}}</span>`).join('')}}</div>`;
}}
function select(item, card) {{
  document.querySelectorAll('.card.active').forEach(el => el.classList.remove('active'));
  if (card) card.classList.add('active');
  state.selected = item;
  document.getElementById('detailTitle').textContent = `${{item.logical_row_id}} · ${{item.display_name}}`;
  document.getElementById('detailSub').textContent = item.kind === 'needs_review' ? 'needs_review 独立记录' : '候选参数';
  document.getElementById('detailBody').innerHTML = `
    <dl class="kv">
      <dt>logical_row_id</dt><dd>${{esc(item.logical_row_id)}}</dd>
      ${{detailRows(item)}}
      <dt>完整测试条件</dt><dd>${{esc(item.test_conditions)}}</dd>
      <dt>PDF 页码</dt><dd>${{esc(item.page)}}</dd>
      <dt>source_row_indices</dt><dd>${{rowIndices(item.source_row_indices)}}</dd>
      <dt>source_cell_ids</dt><dd>${{rowIndices(item.source_cell_ids)}}</dd>
    </dl>
    <div class="evidence">
      <div><b>PDF 区域截图（对应逻辑行）</b></div>
      <a href="${{esc(item.page_image)}}" target="_blank" rel="noopener">
        <img src="${{esc(item.image)}}" alt="${{esc(item.logical_row_id)}} PDF evidence">
      </a>
      <div class="caption">点击截图打开完整页。截图仅由 PyMuPDF 渲染，未修改 PDF。</div>
    </div>
    <div class="evidence">
      <div><b>PDF 原文（bbox 内文字）</b></div>
      <pre>${{esc(item.pdf_text)}}</pre>
    </div>
    <div class="evidence">
      <div><b>来源单元格文本</b></div>
      ${{cellChips(item)}}
    </div>`;
}}
document.getElementById('search').addEventListener('input', e => {{
  state.query = e.target.value.trim().toLowerCase();
  render();
}});
document.getElementById('onlyFocus').addEventListener('change', e => {{
  state.onlyFocus = e.target.checked;
  render();
}});
render();
</script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="生成 Gold Standard Excel/HTML 人工核验包"
    )
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--logical", default=str(DEFAULT_LOGICAL))
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    args = parser.parse_args(argv)

    gold_path = Path(args.gold)
    logical_path = Path(args.logical)
    raw_path = Path(args.raw)
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir)

    gold = load_json(gold_path)
    logical_rows = load_json(logical_path)
    raw_structure = load_json(raw_path)

    if not pdf_path.is_file():
        raise ReviewBuildError(
            f"PDF 不存在: {pdf_path}"
        )

    pdf_sha256 = sha256_file(pdf_path)
    expected_sha = (
        gold.get("source") or {}
    ).get("pdf_sha256")

    if expected_sha and expected_sha != pdf_sha256:
        raise ReviewBuildError(
            "PDF SHA256 与 Gold Standard 不一致: "
            f"expected={expected_sha}, actual={pdf_sha256}"
        )

    entries, needs_entries = build_entries(
        gold,
        logical_rows,
        raw_structure,
        pdf_path,
        output_dir,
    )

    excel_path = output_dir / "gold_review.xlsx"
    html_path = output_dir / "gold_review.html"

    build_excel(
        entries,
        needs_entries,
        excel_path,
        gold,
        pdf_path,
        pdf_sha256,
    )
    build_html(entries, needs_entries, html_path)

    print("=" * 64)
    print("Gold Standard 人工核验包已生成")
    print("=" * 64)
    print("PDF:", pdf_path)
    print("PDF SHA256:", pdf_sha256)
    print("候选参数行:", len(entries))
    print("needs_review 行:", len(needs_entries))
    print("重点行:", ", ".join(sorted(FOCUS_IDS)))
    print("Excel:", excel_path)
    print("HTML:", html_path)
    print(
        "PDF 截图目录:",
        output_dir / "pdf_regions",
    )
    print(
        "默认状态: unreviewed（未自动 confirmed）"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

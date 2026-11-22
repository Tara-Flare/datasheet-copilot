"""阶段 B3.2b：按参数名称和测试条件生成 Gold Standard 分组核验页。

只读取现有 gold_review.xlsx 和既有 PDF 截图，不修改：
- gold_review.xlsx
- Gold JSON
- PDF / 缓存 / Logical Rows
"""

import argparse
import html
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XLSX = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight/gold_review.xlsx"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight/gold_review_grouped.html"
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


class GroupBuildError(ValueError):
    """分组核验页生成失败。"""


def cell_text(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_status(value):
    status = cell_text(value).lower()

    return status or "unreviewed"


def read_sheet(workbook, name, required):
    if name not in workbook.sheetnames:
        raise GroupBuildError(
            f"Excel 缺少工作表: {name}"
        )

    ws = workbook[name]
    headers = [
        ws.cell(1, column).value
        for column in range(
            1, ws.max_column + 1
        )
    ]

    missing = [
        field for field in required
        if field not in headers
    ]

    if missing:
        raise GroupBuildError(
            f"{name} 缺少列: {missing}"
        )

    rows = []

    for row_index in range(2, ws.max_row + 1):
        row = {}

        for column, header in enumerate(
            headers, start=1
        ):
            if not header:
                continue

            row[header] = ws.cell(
                row_index, column
            ).value

        if not cell_text(row.get("logical_row_id")):
            continue

        row["__row"] = row_index
        rows.append(row)

    return headers, rows


def group_unreviewed(rows):
    groups = OrderedDict()

    for row in rows:
        if normalize_status(
            row.get("人工核验状态")
        ) != "unreviewed":
            continue

        name = cell_text(row.get("参数名称"))
        conditions = cell_text(
            row.get("完整测试条件")
        )
        key = (name, conditions)

        groups.setdefault(key, []).append(row)

    return groups


def split_values(value):
    return [
        item.strip()
        for item in cell_text(value).split(",")
        if item.strip()
    ]


def unique_evidence(rows):
    evidence = OrderedDict()

    for row in rows:
        row_id = cell_text(row.get("logical_row_id"))

        if row_id in evidence:
            continue

        evidence[row_id] = {
            "logical_row_id": row_id,
            "image": cell_text(
                row.get("PDF 核验截图")
            ),
            "pdf_text": cell_text(
                row.get("PDF 原文（bbox）")
            ),
            "page": row.get("PDF 页码"),
            "source_rows": split_values(
                row.get("source_row_indices")
            ),
            "source_cells": split_values(
                row.get("source_cell_ids")
            ),
        }

    return list(evidence.values())


def is_focus(row):
    return cell_text(
        row.get("逻辑行 ID")
    ) in FOCUS_IDS or cell_text(
        row.get("logical_row_id")
    ) in FOCUS_IDS


def render_parameter_table(rows):
    body = []

    for row in rows:
        focus = is_focus(row)
        body.append(f"""
        <tr class="{'focus-row' if focus else ''}">
          <td class="center">{html.escape(cell_text(row.get('序号')))}</td>
          <td class="mono">{html.escape(cell_text(row.get('logical_row_id')))}</td>
          <td class="center">{html.escape(cell_text(row.get('Min/Typ/Max')))}</td>
          <td class="center strong">{html.escape(cell_text(row.get('数值')))}</td>
          <td class="center">{html.escape(cell_text(row.get('单位')))}</td>
          <td class="center">{html.escape(cell_text(row.get('PDF 页码')))}</td>
          <td class="mono">{html.escape(cell_text(row.get('source_row_indices')))}</td>
          <td class="center status">{html.escape(normalize_status(row.get('人工核验状态')))}</td>
          <td>{html.escape(cell_text(row.get('正确值')))}</td>
          <td>{html.escape(cell_text(row.get('备注')))}</td>
        </tr>""")

    return "".join(body)


def render_evidence(rows):
    cards = []

    for item in unique_evidence(rows):
        image = html.escape(item["image"])
        row_id = html.escape(item["logical_row_id"])
        pdf_text = html.escape(item["pdf_text"])

        cards.append(f"""
        <figure class="evidence-card">
          <figcaption>
            <b>{row_id}</b>
            <span>PDF page {html.escape(cell_text(item['page']))}</span>
            <span>rows {html.escape(', '.join(item['source_rows']))}</span>
          </figcaption>
          <a href="{image}" target="_blank" rel="noopener">
            <img loading="lazy" src="{image}" alt="{row_id} PDF evidence">
          </a>
          <pre>{pdf_text}</pre>
        </figure>""")

    return "".join(cards)


def render_group(index, name, conditions, rows):
    focus = any(is_focus(row) for row in rows)
    focus_badge = (
        '<span class="badge focus">含重点行</span>'
        if focus else ""
    )
    anchor = f"group-{index}"
    search = f"{name} {conditions} " + " ".join(
        cell_text(row.get("logical_row_id"))
        for row in rows
    )

    return f"""
    <section class="group" id="{anchor}" data-search="{html.escape(search.lower())}">
      <div class="group-head">
        <div>
          <div class="group-kicker">GROUP {index:02d}</div>
          <h2>{html.escape(name)}</h2>
          <div class="condition">{html.escape(conditions)}</div>
        </div>
        <div class="group-meta">
          <span class="count">{len(rows)} 条参数</span>
          {focus_badge}
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>序号</th>
              <th>logical_row_id</th>
              <th>Min/Typ/Max</th>
              <th>数值</th>
              <th>单位</th>
              <th>页码</th>
              <th>source_row_indices</th>
              <th>状态</th>
              <th>正确值</th>
              <th>备注</th>
            </tr>
          </thead>
          <tbody>{render_parameter_table(rows)}</tbody>
        </table>
      </div>
      <div class="evidence-title">PDF 对应区域</div>
      <div class="evidence-grid">{render_evidence(rows)}</div>
    </section>"""


def render_needs(needs_rows):
    if not needs_rows:
        return '<div class="empty">没有 needs_review 记录。</div>'

    body = []

    for row in needs_rows:
        body.append(f"""
        <tr>
          <td class="center">{html.escape(cell_text(row.get('序号')))}</td>
          <td class="mono">{html.escape(cell_text(row.get('logical_row_id')))}</td>
          <td>{html.escape(cell_text(row.get('参数名称')))}</td>
          <td class="center">{html.escape(cell_text(row.get('Min')))}</td>
          <td class="center">{html.escape(cell_text(row.get('Typ')))}</td>
          <td class="center">{html.escape(cell_text(row.get('Max')))}</td>
          <td class="center">{html.escape(cell_text(row.get('单位')))}</td>
          <td>{html.escape(cell_text(row.get('完整测试条件')))}</td>
          <td>{html.escape(cell_text(row.get('备注')))}</td>
          <td class="center status">{html.escape(normalize_status(row.get('人工核验状态')))}</td>
        </tr>""")

    evidence_rows = []
    for row in needs_rows:
        evidence_rows.append(row)

    return f"""
    <section class="group needs-section" id="needs-review">
      <div class="group-head">
        <div>
          <div class="group-kicker">SEPARATE / DO NOT AUTO-CONFIRM</div>
          <h2>needs_review 独立记录</h2>
          <div class="condition">t9.p34 / t9.p35 / t9.p36；不得自动标记为 confirmed。</div>
        </div>
        <div class="group-meta">
          <span class="count">{len(needs_rows)} 条记录</span>
          <span class="badge review">needs_review</span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>序号</th>
              <th>logical_row_id</th>
              <th>参数名称</th>
              <th>Min</th>
              <th>Typ</th>
              <th>Max</th>
              <th>单位</th>
              <th>完整测试条件</th>
              <th>备注</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>
      <div class="evidence-title">PDF 对应区域</div>
      <div class="evidence-grid">{render_evidence(evidence_rows)}</div>
    </section>"""


def build_html(
    headers,
    candidate_rows,
    needs_rows,
    output_path,
):
    statuses = [
        normalize_status(row.get("人工核验状态"))
        for row in candidate_rows
    ]
    confirmed = statuses.count("confirmed")
    unreviewed = statuses.count("unreviewed")
    other = [
        (row, normalize_status(row.get("人工核验状态")))
        for row in candidate_rows
        if normalize_status(row.get("人工核验状态"))
        not in {"confirmed", "unreviewed"}
    ]

    groups = group_unreviewed(candidate_rows)
    group_items = list(groups.items())

    nav = []

    for index, ((name, conditions), rows) in enumerate(
        group_items, start=1
    ):
        nav.append(f"""
        <a class="nav-item" href="#group-{index}">
          <span>{index:02d}</span>
          <b>{html.escape(name)}</b>
          <em>{len(rows)}</em>
        </a>""")

    group_html = "".join(
        render_group(index, name, conditions, rows)
        for index, ((name, conditions), rows)
        in enumerate(group_items, start=1)
    )

    confirmed_rows = [
        row for row in candidate_rows
        if normalize_status(row.get("人工核验状态"))
        == "confirmed"
    ]
    confirmed_list = "".join(
        f"<li><span class='mono'>{html.escape(cell_text(row.get('logical_row_id')))}</span> "
        f"{html.escape(cell_text(row.get('参数名称')))} "
        f"({html.escape(cell_text(row.get('Min/Typ/Max')))}, "
        f"{html.escape(cell_text(row.get('数值')))} "
        f"{html.escape(cell_text(row.get('单位')))})</li>"
        for row in confirmed_rows
    )

    other_html = ""

    if other:
        other_html = (
            "<p class='warn'>存在其他状态 "
            f"{len(other)} 条，未进入 unreviewed 分组。</p>"
        )

    generated = datetime.now(timezone.utc).isoformat()
    excel_name = html.escape(
        output_path.parent.joinpath(
            "gold_review.xlsx"
        ).name
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>TPS7A02 table 9 · Gold Standard 分组核验</title>
<style>
:root {{
  --ink:#1f2937; --muted:#6b7280; --line:#d8dfe8;
  --bg:#f4f7fb; --card:#fff; --blue:#1f4e78;
  --focus:#fff2cc; --review:#fce4d6;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif;
}}
header {{
  background:#fff; border-bottom:1px solid var(--line);
  padding:20px 24px 16px;
}}
header h1 {{ margin:0 0 5px; color:var(--blue); font-size:23px; }}
header p {{ margin:0; color:var(--muted); font-size:13px; }}
.stats {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }}
.stat {{
  background:var(--bg); border:1px solid var(--line); border-radius:8px;
  padding:8px 12px; min-width:112px;
}}
.stat b {{ display:block; font-size:21px; color:var(--blue); }}
.stat span {{ color:var(--muted); font-size:12px; }}
.layout {{ display:grid; grid-template-columns:240px minmax(0,1fr); gap:18px; padding:18px; }}
nav {{ position:sticky; top:16px; align-self:start; max-height:calc(100vh - 32px); overflow:auto; }}
.nav-title {{ font-size:12px; color:var(--muted); text-transform:uppercase; margin:0 6px 8px; }}
.nav-item {{
  display:grid; grid-template-columns:26px 1fr 28px; gap:6px; align-items:center;
  text-decoration:none; color:var(--ink); background:#fff; border:1px solid var(--line);
  border-radius:7px; padding:7px 8px; margin-bottom:6px; font-size:12px;
}}
.nav-item:hover {{ border-color:#8aa8c4; }}
.nav-item span {{ color:var(--muted); }}
.nav-item b {{ font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.nav-item em {{ color:var(--muted); font-style:normal; text-align:right; }}
main {{ min-width:0; }}
.toolbar {{ display:flex; gap:12px; align-items:center; margin-bottom:12px; }}
.toolbar input {{
  width:min(520px,100%); padding:9px 11px; border:1px solid var(--line);
  border-radius:7px; background:#fff;
}}
.group {{
  background:var(--card); border:1px solid var(--line); border-radius:11px;
  margin-bottom:18px; overflow:hidden; box-shadow:0 2px 12px rgba(20,40,70,.04);
}}
.group-head {{
  display:flex; justify-content:space-between; gap:15px; align-items:flex-start;
  padding:15px 17px; border-bottom:1px solid var(--line); background:#fbfcfe;
}}
.group-kicker {{ font-size:10px; letter-spacing:.08em; color:var(--muted); }}
.group h2 {{ margin:3px 0 5px; color:var(--blue); font-size:17px; }}
.condition {{ color:#374151; font-size:12px; line-height:1.5; white-space:pre-wrap; }}
.group-meta {{ display:flex; gap:7px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }}
.count {{ font-size:12px; color:var(--muted); white-space:nowrap; }}
.badge {{ padding:3px 7px; border-radius:20px; font-size:11px; }}
.badge.focus {{ background:#e8a400; color:#fff; }}
.badge.review {{ background:#c85b32; color:#fff; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ border-bottom:1px solid #e8edf3; padding:7px 8px; text-align:left; vertical-align:top; }}
th {{ background:#eef3f8; color:#334e68; font-weight:600; white-space:nowrap; }}
td.center, th.center {{ text-align:center; }}
td.mono, .mono {{ font-family:"SFMono-Regular",Consolas,monospace; font-size:11px; }}
td.strong {{ font-weight:700; color:#0f3557; }}
td.status {{ background:#fafbfc; }}
tr.focus-row td {{ background:var(--focus); }}
.evidence-title {{ padding:11px 17px 0; color:var(--muted); font-size:12px; font-weight:600; }}
.evidence-grid {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
  gap:12px; padding:10px 17px 17px;
}}
.evidence-card {{ margin:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; }}
.evidence-card figcaption {{
  display:flex; gap:8px; flex-wrap:wrap; padding:7px 9px; font-size:11px;
  color:var(--muted); background:#f8fafc; border-bottom:1px solid var(--line);
}}
.evidence-card figcaption b {{ color:var(--blue); }}
.evidence-card img {{ display:block; width:100%; max-height:280px; object-fit:contain; background:#fff; }}
pre {{
  margin:0; padding:8px 9px; max-height:135px; overflow:auto; white-space:pre-wrap;
  word-break:break-word; background:#fbfcfe; color:#374151; font-size:11px; line-height:1.45;
}}
.needs-section {{ border-color:#e4ad94; background:#fffaf7; }}
.needs-section .group-head {{ background:#fdf1eb; }}
.confirmed-box {{
  background:#fff; border:1px solid var(--line); border-radius:11px; padding:14px 17px;
  margin-bottom:18px;
}}
.confirmed-box h3 {{ margin:0 0 7px; color:var(--blue); font-size:15px; }}
.confirmed-box ul {{ margin:0; padding-left:20px; font-size:12px; line-height:1.7; }}
.warn {{ color:#9a3412; font-size:12px; }}
.empty {{ padding:22px; color:var(--muted); }}
footer {{ padding:0 18px 25px; color:var(--muted); font-size:11px; }}
@media (max-width:900px) {{
  .layout {{ grid-template-columns:1fr; }}
  nav {{ position:static; max-height:none; }}
  .nav-item {{ display:inline-grid; width:calc(50% - 4px); vertical-align:top; }}
}}
</style>
</head>
<body>
<header>
  <h1>TPS7A02 · table_id=9 · Gold Standard 分组核验</h1>
  <p>数据源：{excel_name}。已填写的人工核验状态被原样读取；本页面为只读快照。</p>
  <div class="stats">
    <div class="stat"><b>{len(candidate_rows)}</b><span>候选参数总数</span></div>
    <div class="stat"><b>{confirmed}</b><span>实际 confirmed</span></div>
    <div class="stat"><b>{unreviewed}</b><span>实际 unreviewed</span></div>
    <div class="stat"><b>{len(group_items)}</b><span>未核验分组</span></div>
    <div class="stat"><b>{len(needs_rows)}</b><span>needs_review 记录</span></div>
  </div>
</header>
<div class="layout">
  <nav>
    <div class="nav-title">未核验分组</div>
    {''.join(nav)}
  </nav>
  <main>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索参数名称 / 条件 / logical_row_id">
    </div>
    {other_html}
    <div class="confirmed-box">
      <h3>已 confirmed（不进入未核验分组）</h3>
      <ul>{confirmed_list or '<li>无</li>'}</ul>
    </div>
    {group_html}
    {render_needs(needs_rows)}
  </main>
</div>
<footer>生成时间：{html.escape(generated)}。截图由 PyMuPDF 从原始 PDF 渲染；未修改 PDF、Excel 或 Gold JSON。</footer>
<script>
const search = document.getElementById('search');
search.addEventListener('input', () => {{
  const q = search.value.trim().toLowerCase();
  document.querySelectorAll('section.group:not(.needs-section)').forEach(section => {{
    const hit = !q || (section.dataset.search || '').includes(q);
    section.style.display = hit ? '' : 'none';
  }});
  document.querySelectorAll('.nav-item').forEach(link => {{
    const target = document.querySelector(link.getAttribute('href'));
    link.style.display = !q || !target || target.style.display !== 'none' ? '' : 'none';
  }});
}});
</script>
</body>
</html>
"""

    output_path.write_text(
        html_text, encoding="utf-8"
    )
    return {
        "confirmed": confirmed,
        "unreviewed": unreviewed,
        "other": len(other),
        "groups": len(group_items),
        "group_items": group_items,
        "needs_review": len(needs_rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="生成 Gold Standard 分组核验 HTML"
    )
    parser.add_argument(
        "--xlsx", default=str(DEFAULT_XLSX)
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT)
    )
    args = parser.parse_args(argv)

    xlsx_path = Path(args.xlsx)
    output_path = Path(args.output)

    if not xlsx_path.is_file():
        raise GroupBuildError(
            f"Excel 不存在: {xlsx_path}"
        )

    workbook = load_workbook(
        xlsx_path, data_only=True
    )

    candidate_headers, candidate_rows = read_sheet(
        workbook,
        "候选参数",
        required=[
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
            "PDF 核验截图",
            "PDF 原文（bbox）",
            "source_cell_ids",
        ],
    )
    needs_headers, needs_rows = read_sheet(
        workbook,
        "needs_review",
        required=[
            "序号",
            "logical_row_id",
            "参数名称",
            "Min",
            "Typ",
            "Max",
            "单位",
            "完整测试条件",
            "PDF 页码",
            "source_row_indices",
            "人工核验状态",
            "备注",
            "PDF 核验截图",
            "PDF 原文（bbox）",
        ],
    )

    # 检查截图存在，缺失时明确报错，不静默生成坏页面。
    missing_images = []

    for row in candidate_rows + needs_rows:
        image = cell_text(row.get("PDF 核验截图"))

        if image and not (
            output_path.parent / image
        ).is_file():
            missing_images.append(image)

    if missing_images:
        raise GroupBuildError(
            "PDF 截图缺失: "
            + ", ".join(sorted(set(missing_images)))
        )

    stats = build_html(
        candidate_headers,
        candidate_rows,
        needs_rows,
        output_path,
    )

    print("=" * 64)
    print("Gold Standard 分组核验页已生成")
    print("=" * 64)
    print("Excel:", xlsx_path)
    print("HTML:", output_path)
    print(
        "候选参数:",
        len(candidate_rows),
        "confirmed:",
        stats["confirmed"],
        "unreviewed:",
        stats["unreviewed"],
        "其他:",
        stats["other"],
    )
    print(
        "未核验分组:",
        stats["groups"],
        "覆盖参数:",
        sum(
            len(rows)
            for _, rows in stats["group_items"]
        ),
    )
    print(
        "needs_review:",
        stats["needs_review"],
    )

    for index, ((name, conditions), rows) in enumerate(
        stats["group_items"], start=1
    ):
        print(
            f"  {index:02d}. {name} | "
            f"{len(rows)} 条 | "
            f"{', '.join(cell_text(row.get('logical_row_id')) for row in rows)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

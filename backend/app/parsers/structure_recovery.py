"""阶段 B1：基于 PDF 几何证据的表格结构恢复。

将 Docling 原始 TableCell 与 PDF 页面几何证据（文字坐标 + 表格线段）
确定性地恢复为可追溯的 Logical Rows。

设计原则：
- 只用确定性算法，不调用 LLM；
- 物理行带由「覆盖数值列的水平线」推导，而不是把任意水平线当作全表行边界；
- 单位传播以 UNIT 列内部无分隔线为证据，并受该列分隔线约束，不跨单位区域；
- 每个逻辑行保留 source_cell_ids / source_word_bboxes / physical_row_band；
- 证据不足时设置 needs_review=true，不自动猜测。
"""

import json
import re
from pathlib import Path

import pymupdf as fitz


SCHEMA = "table_logical_rows/v1"
LOGICAL_ROWS_VERSION = 1

VALUE_HEADER_ALIASES = {
    "min": {"min", "minimum"},
    "typ": {"typ", "typical"},
    "max": {"max", "maximum"},
}

UNIT_HEADERS = {"unit", "units"}
CONDITION_HINT = "condition"
PARAMETER_HINT = "parameter"

# 常见单位 token（用于单位区域歧义检测）
UNIT_TOKEN_RE = re.compile(
    r"^(?:[nupµmkMG]?[AVWΩ]|%|dB|°C|µVRMS|nA|µA|mA|mV|kV|kΩ|MΩ|Hz|kHz)$"
)

NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _cluster(values, tol=1.5):
    """对一维坐标去重聚类。"""

    values = sorted(float(v) for v in values)
    out = []

    for value in values:
        if not out or abs(value - out[-1]) > tol:
            out.append(value)
        else:
            out[-1] = (out[-1] + value) / 2.0

    return out


def _bbox_center(bbox):
    return (
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0,
    )


def to_topleft(bbox, page_height):
    """Docling bbox -> TOPLEFT 页面坐标。"""

    if bbox is None:
        return None

    if bbox.get("coord_origin") == "BOTTOMLEFT":
        return (
            float(bbox["l"]),
            page_height - float(bbox["t"]),
            float(bbox["r"]),
            page_height - float(bbox["b"]),
        )

    return (
        float(bbox["l"]),
        float(bbox["t"]),
        float(bbox["r"]),
        float(bbox["b"]),
    )


def _normalize_header(value):
    return re.sub(r"[^a-z]", "", str(value).lower())


def column_roles(columns):
    """根据表头名推断每一列的角色。"""

    roles = {}
    parameter_seen = 0
    condition_seen = 0

    for index, name in enumerate(columns):
        normalized = _normalize_header(name)

        if normalized in VALUE_HEADER_ALIASES["min"]:
            roles[index] = "min"
        elif normalized in VALUE_HEADER_ALIASES["typ"]:
            roles[index] = "typ"
        elif normalized in VALUE_HEADER_ALIASES["max"]:
            roles[index] = "max"
        elif normalized in UNIT_HEADERS:
            roles[index] = "unit"
        elif CONDITION_HINT in normalized:
            condition_seen += 1
            roles[index] = (
                "condition1"
                if condition_seen == 1
                else "condition2"
            )
        elif PARAMETER_HINT in normalized:
            parameter_seen += 1
            roles[index] = (
                "parameter_symbol"
                if parameter_seen == 1
                else "parameter_name"
            )
        else:
            roles[index] = f"column{index}"

    return roles


def _covers(segment, column, fraction=0.5):
    """水平线段是否覆盖某列足够宽度。"""

    _y, x0, x1 = segment
    c0, c1 = column

    if c1 <= c0:
        return False

    overlap = max(0.0, min(x1, c1) - max(x0, c0))

    return overlap >= (c1 - c0) * fraction


def _column_boundaries(column, horizontal_segments):
    ys = [
        segment[0]
        for segment in horizontal_segments
        if _covers(segment, column)
    ]

    return _cluster(ys, tol=1.5)


def _find_region(boundaries, y, tol=0.6):
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start - tol <= y <= end + tol:
            return (start, end)

    return None


def _words_in_region(words, column, region):
    c0, c1 = column
    y0, y1 = region

    selected = []

    for word in words:
        cx, cy = _bbox_center(word["bbox"])

        if c0 - 1.0 <= cx <= c1 + 1.0 and y0 - 0.6 <= cy <= y1 + 0.6:
            selected.append(word)

    return selected


def line_aware_text(words):
    """按文本行（y 聚类）恢复阅读顺序。"""

    if not words:
        return ""

    ordered = sorted(
        words,
        key=lambda w: (
            _bbox_center(w["bbox"])[1],
            w["bbox"][0],
        ),
    )

    lines = []

    for word in ordered:
        cy = _bbox_center(word["bbox"])[1]

        if not lines or abs(cy - lines[-1]["cy"]) > 3.0:
            lines.append({"cy": cy, "words": [word]})
        else:
            lines[-1]["words"].append(word)
            lines[-1]["cy"] = sum(
                _bbox_center(w["bbox"])[1]
                for w in lines[-1]["words"]
            ) / len(lines[-1]["words"])

    texts = []

    for line in lines:
        line["words"].sort(key=lambda w: w["bbox"][0])
        texts.append(
            " ".join(w["text"] for w in line["words"])
        )

    return " ".join(texts)


def _text_line_count(words):
    if not words:
        return 0

    ys = sorted(_bbox_center(w["bbox"])[1] for w in words)
    lines = 1

    for previous, current in zip(ys[:-1], ys[1:]):
        if current - previous > 3.0:
            lines += 1

    return lines


def _join_conditions(first, second):
    """连接两个测试条件列，处理跨列溢出的短片段。"""

    first = (first or "").strip()
    second = (second or "").strip()

    if not first:
        return second

    if not second:
        return first

    # 条件文本从第一列溢出到第二列时（如 "mA" 或 "= 30 mA"），
    # 用空格连接，避免出现 " | mA"。
    if len(second) <= 5 or second[0] in "=<>≤≥":
        return f"{first} {second}"

    return f"{first} | {second}"


def _detect_columns(
    raw_table,
    vertical_segments,
    table_bbox,
):
    # 先在表体范围内过滤，再做 x 聚类；
    # 否则页面其它竖线（页框、logo 等）会污染列边界。
    segments = [
        segment for segment in vertical_segments
        if table_bbox[0] - 2 <= segment[0] <= table_bbox[2] + 2
        and segment[1] >= table_bbox[1] - 2
        and segment[2] <= table_bbox[3] + 2
        and segment[2] - segment[1] >= 5.0
    ]

    xs = _cluster(
        [segment[0] for segment in segments],
        tol=2.0,
    )

    xs = [
        x for x in xs
        if table_bbox[0] - 2 <= x <= table_bbox[2] + 2
    ]

    if len(xs) - 1 == len(raw_table.get("columns", [])):
        return [
            (xs[i], xs[i + 1])
            for i in range(len(xs) - 1)
        ]

    # 回退：按列数均分表格宽度
    columns = raw_table.get("columns", [])
    count = max(len(columns), 1)
    width = (table_bbox[2] - table_bbox[0]) / count

    return [
        (
            table_bbox[0] + i * width,
            table_bbox[0] + (i + 1) * width,
        )
        for i in range(count)
    ]


def _match_word_to_cell(word, cells):
    """把一个 PDF word 匹配到 Docling 原始 cell。"""

    cx, cy = _bbox_center(word["bbox"])
    best = None
    best_area = None

    for cell in cells:
        bbox = cell.get("_bbox_tl")

        if bbox is None:
            continue

        if (
            bbox[0] - 1 <= cx <= bbox[2] + 1
            and bbox[1] - 1 <= cy <= bbox[3] + 1
        ):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if best is None or area < best_area:
                best = cell
                best_area = area

    return best


def _nearest_cell_for_role(word, cells, roles, role, max_dy=6.0):
    """数值列的兜底匹配：找同一列最近的 cell。

    Docling 的 cell bbox 是文字 bbox，个别情况下与
    PDF word 的 y 中心有 2-3pt 偏差，这里按列内最近距离兜底。
    只用于 MIN/TYP/MAX，避免影响 UNIT 的合并判断。
    """

    cx, cy = _bbox_center(word["bbox"])
    best = None
    best_dy = None

    for cell in cells:
        if roles.get(cell.get("start_col_offset_idx")) != role:
            continue

        bbox = cell.get("_bbox_tl")

        if bbox is None:
            continue

        if not (bbox[0] - 1 <= cx <= bbox[2] + 1):
            continue

        if bbox[1] <= cy <= bbox[3]:
            dy = 0.0
        else:
            dy = min(abs(cy - bbox[1]), abs(cy - bbox[3]))

        if dy <= max_dy and (best is None or dy < best_dy):
            best = cell
            best_dy = dy

    return best


def _word_payload(words):
    return [
        {
            "text": word["text"],
            "bbox": [
                round(word["bbox"][0], 2),
                round(word["bbox"][1], 2),
                round(word["bbox"][2], 2),
                round(word["bbox"][3], 2),
            ],
        }
        for word in words
    ]


def _dedupe_words(words):
    seen = set()
    result = []

    for word in words:
        key = (
            word["text"],
            round(word["bbox"][0], 2),
            round(word["bbox"][1], 2),
            round(word["bbox"][2], 2),
            round(word["bbox"][3], 2),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(word)

    return result


def recover_structure(
    raw_structure,
    table_id,
    *,
    words,
    horizontal_segments,
    vertical_segments,
    page_height,
    page_no=None,
):
    """把一个 table 的原始 cell + PDF 几何恢复为逻辑行。

    参数：
    - raw_structure: 阶段 A 的 raw_structure 文档（或含 tables 的 dict）
    - table_id: 目标表 ID
    - words: [{"text", "bbox": (x0,y0,x1,y1)}]（TOPLEFT）
    - horizontal_segments: [(y, x0, x1)]（TOPLEFT）
    - vertical_segments: [(x, y0, y1)]（TOPLEFT）
    - page_height: 页面高度（用于 bbox 坐标转换）
    """

    table = None

    for candidate in raw_structure.get("tables", []):
        if candidate.get("table_id") == table_id:
            table = candidate
            break

    if table is None:
        raise KeyError(
            f"raw_structure 中不存在 table_id={table_id}"
        )

    table_bbox = to_topleft(
        table.get("table_bbox"),
        page_height,
    )

    if table_bbox is None:
        # 用 cell bbox 兜底
        bboxes = [
            to_topleft(c.get("bbox"), page_height)
            for c in table.get("cells", [])
        ]
        bboxes = [b for b in bboxes if b is not None]

        if not bboxes:
            raise ValueError(
                f"table_id={table_id} 缺少可用的 bbox"
            )

        table_bbox = (
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        )

    columns = table.get("columns", [])
    roles = column_roles(columns)

    column_boxes = _detect_columns(
        table,
        vertical_segments,
        table_bbox,
    )

    # 过滤到表体范围
    words = [
        word for word in words
        if table_bbox[0] - 2 <= _bbox_center(word["bbox"])[0] <= table_bbox[2] + 2
        and table_bbox[1] - 2 <= _bbox_center(word["bbox"])[1] <= table_bbox[3] + 2
    ]

    horizontal_segments = [
        seg for seg in horizontal_segments
        if seg[1] >= table_bbox[0] - 2
        and seg[2] <= table_bbox[2] + 2
        and table_bbox[1] - 2 <= seg[0] <= table_bbox[3] + 2
    ]

    vertical_segments = [
        seg for seg in vertical_segments
        if table_bbox[0] - 2 <= seg[0] <= table_bbox[2] + 2
        and seg[1] < table_bbox[3]
        and seg[2] > table_bbox[1]
    ]

    if len(column_boxes) != len(columns):
        # 列数不一致时仍按检测结果继续，但记录问题
        pass

    col_boundaries = [
        _column_boundaries(column, horizontal_segments)
        for column in column_boxes
    ]

    # 用边界最多的列（数值列）作为物理行参考
    value_indices = [
        index
        for index, role in roles.items()
        if role in ("min", "typ", "max")
        and index < len(col_boundaries)
    ]

    reference_indices = value_indices or list(
        range(len(column_boxes))
    )

    counts = [len(col_boundaries[i]) for i in reference_indices]
    max_count = max(counts) if counts else 0

    reference_indices = [
        i for i in reference_indices
        if len(col_boundaries[i]) >= max_count - 1
    ]

    reference_ys = _cluster(
        [
            y
            for index in reference_indices
            for y in col_boundaries[index]
        ],
        tol=1.5,
    )

    physical_bands = []

    for index, (top, bottom) in enumerate(
        zip(reference_ys[:-1], reference_ys[1:])
    ):
        physical_bands.append({
            "band_id": index,
            "top": round(top, 2),
            "bottom": round(bottom, 2),
        })

    # 预处理 cell bbox（TOPLEFT）
    cells = table.get("cells", [])
    header_row_count = int(
        table.get("header_row_count", 0)
    )

    for cell in cells:
        cell["_bbox_tl"] = to_topleft(
            cell.get("bbox"), page_height
        )

    # 每一列的区域文本
    column_regions = {}

    for index, column in enumerate(column_boxes):
        boundaries = col_boundaries[index]
        regions = []

        for start, end in zip(
            boundaries[:-1], boundaries[1:]
        ):
            region_words = _words_in_region(
                words, column, (start, end)
            )
            regions.append({
                "top": round(start, 2),
                "bottom": round(end, 2),
                "text": line_aware_text(region_words),
                "words": region_words,
            })

        column_regions[index] = regions

    def region_for(index, band):
        center = (
            band["top"] + band["bottom"]
        ) / 2.0

        for region in column_regions.get(index, []):
            if (
                region["top"] - 0.6
                <= center
                <= region["bottom"] + 0.6
            ):
                return region

        return None

    def role_region(role, band):
        for index, value in roles.items():
            if value == role:
                return region_for(index, band)

        return None

    # 单位区域歧义检测
    unresolved_issues = []
    unit_regions = {}
    ambiguous_unit_regions = set()

    for index, role in roles.items():
        if role != "unit":
            continue

        for region in column_regions.get(index, []):
            unit_words = [
                word for word in region["words"]
                if word["text"].strip()
            ]
            unit_tokens = {
                word["text"].strip()
                for word in unit_words
                if UNIT_TOKEN_RE.match(
                    word["text"].strip()
                )
            }

            if len(unit_tokens) > 1:
                ambiguous_unit_regions.add(
                    (region["top"], region["bottom"])
                )
                unresolved_issues.append({
                    "table_id": table_id,
                    "physical_row_band": {
                        "top": region["top"],
                        "bottom": region["bottom"],
                    },
                    "reason": "单位区域内存在多个不同单位，无法唯一传播",
                    "units": sorted(unit_tokens),
                })

            unit_regions[(
                region["top"],
                region["bottom"],
            )] = region

    # 多值拆分一致性校验
    # 只对数值列/条件列的多行 cell 生效
    multivalue_cells = {}

    for cell in cells:
        bbox = cell.get("_bbox_tl")

        if bbox is None:
            continue

        role = roles.get(
            cell.get("start_col_offset_idx")
        )

        if role not in (
            "min", "typ", "max",
            "condition1", "condition2",
        ):
            continue

        spans = [
            band for band in physical_bands
            if min(bbox[3], band["bottom"])
            - max(bbox[1], band["top"]) > 2.0
        ]

        if len(spans) <= 1:
            continue

        text = cell.get("text") or ""

        if role in ("min", "typ", "max"):
            tokens = NUMBER_RE.findall(text)
            expected = len(tokens)
        else:
            cell_words = [
                word for word in words
                if bbox[0] - 1 <= _bbox_center(word["bbox"])[0] <= bbox[2] + 1
                and bbox[1] <= _bbox_center(word["bbox"])[1] <= bbox[3]
            ]
            expected = _text_line_count(cell_words)

        multivalue_cells[cell["cell_id"]] = {
            "band_count": len(spans),
            "expected_count": expected,
            "bands": spans,
            "text": text,
            "role": role,
        }

        if expected > 1 and expected != len(spans):
            unresolved_issues.append({
                "table_id": table_id,
                "physical_row_band": {
                    "top": spans[0]["top"],
                    "bottom": spans[-1]["bottom"],
                },
                "reason": (
                    "多值单元格的数值/条件数量与物理行数不一致，"
                    "拒绝自动拆分"
                ),
                "source_cell_ids": [cell["cell_id"]],
                "expected_count": expected,
                "band_count": len(spans),
                "text": text,
            })

    logical_rows = []
    previous_parameter_name = None

    def is_header(band):
        texts = []

        for index in range(len(column_boxes)):
            region = region_for(index, band)

            if region:
                texts.append(region["text"])

        joined = " ".join(texts).upper()

        return (
            "PARAMETER" in joined
            or "TEST CONDITIONS" in joined
        )

    def band_has_content(band):
        for role in (
            "parameter_name", "condition1",
            "condition2", "min", "typ", "max",
            "unit",
        ):
            region = role_region(role, band)

            if region and region["text"].strip():
                return True

        return False

    for band in physical_bands:
        if is_header(band) or not band_has_content(band):
            continue

        role_texts = {}
        role_words = {}
        role_regions = {}

        for role in (
            "parameter_symbol", "parameter_name",
            "condition1", "condition2",
            "min", "typ", "max", "unit",
        ):
            region = role_region(role, band)

            if region is None:
                role_texts[role] = ""
                role_words[role] = []
                role_regions[role] = None
                continue

            role_texts[role] = region["text"].strip()
            role_words[role] = region["words"]
            role_regions[role] = region

        used_words = []

        for role in role_words:
            used_words.extend(role_words[role])

        used_words = _dedupe_words(used_words)

        source_cells = []

        for role, role_word_list in role_words.items():
            allow_nearest = role in ("min", "typ", "max")

            for word in role_word_list:
                cell = _match_word_to_cell(word, cells)

                if cell is None and allow_nearest:
                    cell = _nearest_cell_for_role(
                        word, cells, roles, role
                    )

                if cell is not None:
                    source_cells.append(cell)

        source_cell_ids = sorted({
            cell["cell_id"] for cell in source_cells
        })

        source_row_indices = sorted({
            int(cell["start_row_offset_idx"])
            - header_row_count
            for cell in source_cells
        })

        actions = ["physical_row_banding"]
        needs_review = False

        # 单位跨行传播
        unit_region = role_regions.get("unit")

        if (
            unit_region is not None
            and (
                unit_region["top"],
                unit_region["bottom"],
            ) in ambiguous_unit_regions
        ):
            # 证据不足：不自动填充单位
            role_texts["unit"] = ""
            role_words["unit"] = []
            needs_review = True

        if unit_region is not None:
            unit_band_count = sum(
                1
                for candidate in physical_bands
                if candidate["top"] >= unit_region["top"] - 0.6
                and candidate["bottom"] <= unit_region["bottom"] + 0.6
            )

            if unit_band_count > 1:
                actions.append("unit_span_recovery")

        # 多值拆分（仅数值列）
        for cell in source_cells:
            info = multivalue_cells.get(cell["cell_id"])

            if (
                info
                and info["role"] in ("min", "typ", "max")
                and info["expected_count"] > 1
            ):
                actions.append("multi_value_split")

                if info["expected_count"] != info["band_count"]:
                    needs_review = True

        # 条件片段合并（例如 "IOUT ≥ 100" + "mA"）
        for role in ("condition1", "condition2"):
            region = role_regions.get(role)

            if region is None:
                continue

            region_cell_texts = set()

            for word in region["words"]:
                cell = _match_word_to_cell(word, cells)

                if cell is not None:
                    region_cell_texts.add(
                        (cell.get("text") or "").strip()
                    )

            if len(region_cell_texts) > 1 and any(
                len(text) <= 4
                and UNIT_TOKEN_RE.match(text)
                for text in region_cell_texts
            ):
                actions.append(
                    "condition_fragment_merge"
                )

        # Ground Current 行归属恢复
        if role_texts.get("parameter_symbol") in (
            "IGND", "IGND/IOUT",
        ):
            actions.append("ground_current_realignment")

        # 无法匹配到 source cell 的关键值
        for role in ("min", "typ", "max", "unit"):
            if not role_texts.get(role):
                continue

            allow_nearest = role in ("min", "typ", "max")
            matched = False

            for word in role_words.get(role, []):
                cell = _match_word_to_cell(word, cells)

                if cell is None and allow_nearest:
                    cell = _nearest_cell_for_role(
                        word, cells, roles, role
                    )

                if cell is not None:
                    matched = True
                    break

            if not matched:
                needs_review = True
                unresolved_issues.append({
                    "table_id": table_id,
                    "physical_row_band": {
                        "top": band["top"],
                        "bottom": band["bottom"],
                    },
                    "reason": f"{role} 文本无法匹配到原始 cell",
                    "text": role_texts.get(role),
                })

        # 参数名切换 + 单一数值来自跨多物理行 cell：
        # 典型如 VUVLO(HYST) 的 "180" 杂散值，名称与数值错位。
        nonempty_values = [
            role for role in ("min", "typ", "max")
            if role_texts.get(role)
        ]

        if (
            previous_parameter_name is not None
            and role_texts.get("parameter_name")
            and role_texts.get("parameter_name")
            != previous_parameter_name
            and len(nonempty_values) == 1
        ):
            role = nonempty_values[0]

            for cell in source_cells:
                if roles.get(
                    cell.get("start_col_offset_idx")
                ) != role:
                    continue

                cbbox = cell.get("_bbox_tl")

                if cbbox is None:
                    continue

                band_count = sum(
                    1
                    for candidate in physical_bands
                    if min(cbbox[3], candidate["bottom"])
                    - max(cbbox[1], candidate["top"]) > 2.0
                )

                if (
                    band_count > 1
                    and len(NUMBER_RE.findall(
                        cell.get("text") or ""
                    )) > 1
                ):
                    needs_review = True
                    unresolved_issues.append({
                        "table_id": table_id,
                        "physical_row_band": {
                            "top": band["top"],
                            "bottom": band["bottom"],
                        },
                        "reason": (
                            "参数名切换处的单一数值来自跨多物理行"
                            " cell，疑似名称/数值错位，拒绝自动确认"
                        ),
                        "source_cell_ids": [
                            cell["cell_id"]
                        ],
                        "band_count": band_count,
                        "text": role_texts.get(role),
                    })
                    break

        previous_parameter_name = (
            role_texts.get("parameter_name")
            or previous_parameter_name
        )

        actions = list(dict.fromkeys(actions))

        if needs_review:
            confidence = 0.4
        elif any(
            action != "physical_row_banding"
            for action in actions
        ):
            confidence = 0.9
        else:
            confidence = 1.0

        test_conditions = _join_conditions(
            role_texts.get("condition1", ""),
            role_texts.get("condition2", ""),
        )

        logical_rows.append({
            "logical_row_id": (
                f"t{table_id}.p{band['band_id']}"
            ),
            "table_id": table_id,
            "page": page_no,
            "parameter_symbol": role_texts.get(
                "parameter_symbol", ""
            ),
            "parameter_name": role_texts.get(
                "parameter_name", ""
            ),
            "test_conditions": test_conditions,
            "conditions": {
                "condition1": role_texts.get(
                    "condition1", ""
                ),
                "condition2": role_texts.get(
                    "condition2", ""
                ),
            },
            "min": role_texts.get("min", ""),
            "typ": role_texts.get("typ", ""),
            "max": role_texts.get("max", ""),
            "unit": role_texts.get("unit", ""),
            "source_cell_ids": source_cell_ids,
            "source_row_indices": source_row_indices,
            "source_word_bboxes": _word_payload(
                used_words
            ),
            "physical_row_band": {
                "top": band["top"],
                "bottom": band["bottom"],
            },
            "recovery_actions": actions,
            "confidence": confidence,
            "needs_review": needs_review,
        })

    # 清理临时字段
    for cell in cells:
        cell.pop("_bbox_tl", None)

    action_counts = {}

    for row in logical_rows:
        for action in row["recovery_actions"]:
            action_counts[action] = (
                action_counts.get(action, 0) + 1
            )

    return {
        "schema": SCHEMA,
        "logical_rows_version": LOGICAL_ROWS_VERSION,
        "table_id": table_id,
        "page": page_no,
        "coordinate_handling": {
            "page_height": page_height,
            "table_bbox_origin": (
                table.get("table_bbox") or {}
            ).get("coord_origin"),
            "cell_bbox_origin": "TOPLEFT",
            "table_bbox_topleft": [
                round(value, 2) for value in table_bbox
            ],
        },
        "physical_row_bands": physical_bands,
        "logical_rows": logical_rows,
        "recovery_actions": action_counts,
        "unresolved_issues": unresolved_issues,
        "summary": {
            "logical_row_count": len(logical_rows),
            "confirmed_count": sum(
                1
                for row in logical_rows
                if not row["needs_review"]
            ),
            "needs_review_count": sum(
                1
                for row in logical_rows
                if row["needs_review"]
            ),
        },
    }


def extract_page_geometry(page, table_bbox_tl=None):
    """从 PyMuPDF 页面提取 words 与表格线段。"""

    words = [
        {
            "text": word[4],
            "bbox": (
                float(word[0]),
                float(word[1]),
                float(word[2]),
                float(word[3]),
            ),
        }
        for word in page.get_text("words")
    ]

    horizontal = []
    vertical = []

    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]

                if abs(p1.y - p2.y) < 0.6:
                    x0, x1 = sorted([p1.x, p2.x])
                    horizontal.append(
                        (float(p1.y), float(x0), float(x1))
                    )

                if abs(p1.x - p2.x) < 0.6:
                    y0, y1 = sorted([p1.y, p2.y])
                    vertical.append(
                        (float(p1.x), float(y0), float(y1))
                    )

            elif item[0] == "re":
                rect = item[1]

                for y in (rect.y0, rect.y1):
                    horizontal.append(
                        (float(y), float(rect.x0), float(rect.x1))
                    )

                for x in (rect.x0, rect.x1):
                    vertical.append(
                        (float(x), float(rect.y0), float(rect.y1))
                    )

    return words, horizontal, vertical


def recover_from_pdf(
    raw_structure,
    table_id,
    pdf_path,
    page_no,
):
    """从 PDF 页面几何恢复指定 table。"""

    document = fitz.open(pdf_path)
    page = document[page_no - 1]

    words, horizontal, vertical = extract_page_geometry(
        page
    )

    return recover_structure(
        raw_structure,
        table_id,
        words=words,
        horizontal_segments=horizontal,
        vertical_segments=vertical,
        page_height=float(page.rect.height),
        page_no=page_no,
    )


def save_logical_rows(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)

    return path

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
from copy import deepcopy
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

# PDF 文本层常把负数写成 U+2212（−）或 U+2013（–）。
# 仅对数值字段做字形归一，不改变数值本身。
NUMERIC_DASH_TRANSLATION = str.maketrans({
    "\u2212": "-",
    "\u2013": "-",
})


def normalize_numeric_glyphs(value):
    """把数值字段中的 Unicode 负号归一为 ASCII '-'。

    只是排版字形归一，不新增或猜测任何数值。
    """

    return str(value or "").translate(
        NUMERIC_DASH_TRANSLATION
    )


# 子行条件列中的“参数标签：条件”形式
CONDITION_LABEL_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9 ()\-/]{2,60}):\s*(.*)$"
)


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
    """标准化表头；先去掉脚注括号。

    例如 "MIN (Note 9)" -> "min"，否则无法识别
    MIN/MAX 列，导致这些列的数值整体丢失。
    """

    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)

    return re.sub(r"[^a-z]", "", text)


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


UNIT_CONNECTORS = {"·", "•", "*", "/", "×", "⋅"}


def _group_words_by_line(words, tol=3.0):
    """按 y 中心把 PDF words 聚成文本行。"""

    ordered = sorted(
        words,
        key=lambda word: (
            _bbox_center(word["bbox"])[1],
            word["bbox"][0],
        ),
    )
    groups = []

    for word in ordered:
        center = _bbox_center(word["bbox"])[1]

        if not groups or center - groups[-1]["last"] > tol:
            groups.append({
                "last": center,
                "centers": [center],
                "words": [word],
            })
        else:
            groups[-1]["words"].append(word)
            groups[-1]["centers"].append(center)
            groups[-1]["last"] = center

    return [
        {
            "center": sum(group["centers"]) / len(group["centers"]),
            "words": group["words"],
        }
        for group in groups
    ]


def _unit_expression_in_line(words):
    """从一个文本行内提取单位表达。

    返回：
    - 字符串：唯一单位（允许 "mA·kΩ" 这类复合单位）；
    - None：同一行存在多个互不相连的单位，无法判定。
    """

    ordered = sorted(words, key=lambda word: word["bbox"][0])
    tokens = []

    for word in ordered:
        text = word["text"].strip()

        if not text:
            continue

        if UNIT_TOKEN_RE.match(text) or text in UNIT_CONNECTORS:
            tokens.append(text)

    units = []
    connectors = []

    for token in tokens:
        if token in UNIT_CONNECTORS:
            connectors.append(token)
        elif token not in units:
            units.append(token)

    if not units:
        return ""

    if len(units) == 1:
        return units[0]

    if connectors:
        parts = []

        for token in tokens:
            if token in units:
                if parts:
                    parts.append("·")
                parts.append(token)

        return "".join(parts)

    return None


def _unit_line_analysis(words):
    """按文本行分析单位列。"""

    expressions = []
    ambiguous = False

    for group in _group_words_by_line(words):
        expression = _unit_expression_in_line(
            group["words"]
        )
        expressions.append(expression)

        if expression is None:
            ambiguous = True

    return {
        "expressions": expressions,
        "ambiguous": ambiguous,
        "distinct": {
            expression
            for expression in expressions
            if expression
        },
    }


def _refine_physical_bands(
    base_bands,
    raw_table,
    cells,
    column_boxes,
    roles,
    words,
):
    """把「一个 Docling 行内含多行数值」的 band 拆成子行。

    ADI LT3045 table 4 的 Docling cell 会把多个 PDF 文本行
    合并成一个 cell（如 MAX="1 10 20"）。这里只依据数值列
    的实际文本行拆分，不读取/外推任何 PDF 中不存在的数值。
    """

    condition_roles = {"condition1", "condition2"}
    refined = []
    issues = []

    for band in base_bands:
        top = band["top"]
        bottom = band["bottom"]

        def in_band(word, start=top, end=bottom):
            center = _bbox_center(word["bbox"])[1]
            return start - 0.6 <= center <= end + 0.6

        # 只有「单个 Docling 行内的 MIN/TYP/MAX cell」
        # 才作为行内拆分证据；跨物理行的合并 cell 由
        # 原有 physical band 机制处理。
        #
        # 行数使用“该列在本 band 内的 PDF 文本行”，
        # 而不是 cell bbox 内的文本行：Docling 的 bbox
        # 有时只覆盖部分子行（如 ADI typ="150 500 100"），
        # 用列范围才能拿到完整三行。
        candidates = []
        unsafe_reason = None

        for cell in cells:
            role = roles.get(
                cell.get("start_col_offset_idx")
            )

            if role not in ("min", "typ", "max"):
                continue

            start_row = cell.get("start_row_offset_idx")
            end_row = cell.get("end_row_offset_idx")

            if not isinstance(start_row, int) or not isinstance(
                end_row, int
            ):
                continue

            if end_row - start_row != 1:
                continue

            bbox = cell.get("_bbox_tl")

            if bbox is None:
                continue

            if (
                min(bbox[3], bottom)
                - max(bbox[1], top)
                <= 2.0
            ):
                continue

            text = cell.get("text") or ""
            expected = len(NUMBER_RE.findall(text))

            if expected <= 1:
                continue

            column_index = cell.get(
                "start_col_offset_idx"
            )

            if (
                not isinstance(column_index, int)
                or column_index >= len(column_boxes)
            ):
                continue

            c0, c1 = column_boxes[column_index]
            column_words = [
                word for word in words
                if c0 - 1.0
                <= _bbox_center(word["bbox"])[0]
                <= c1 + 1.0
                and in_band(word)
            ]
            line_groups = _group_words_by_line(
                column_words
            )
            line_counts = [
                sum(
                    len(NUMBER_RE.findall(word["text"]))
                    for word in group["words"]
                )
                for group in line_groups
            ]

            # 同一文本行同一列出现多个数值时，无法靠
            # 文本行拆分，明确记录，不做猜测。
            if any(count > 1 for count in line_counts):
                unsafe_reason = (
                    "同一文本行内存在多个数值，"
                    "拒绝按文本行自动拆分"
                )
                break

            # 行数与 cell 数值数量不一致：可能是跨 base
            # band 的合并 cell，交由既有机制处理，不在此
            # 拆分，也不误报为结构错误。
            if len(line_groups) != expected:
                continue

            centers = [
                group["center"]
                for group in line_groups
                if top - 0.6
                <= group["center"]
                <= bottom + 0.6
            ]

            if len(centers) <= 1:
                continue

            candidates.append({
                "cell_id": cell["cell_id"],
                "expected": expected,
                "centers": centers,
            })

        if unsafe_reason is not None:
            issues.append({
                "table_id": raw_table.get("table_id"),
                "physical_row_band": {
                    "top": top,
                    "bottom": bottom,
                },
                "reason": unsafe_reason,
            })
            refined.append({
                "band_id": len(refined),
                "top": top,
                "bottom": bottom,
                "_parent_band_id": band["band_id"],
                "_parent_top": top,
                "_parent_bottom": bottom,
                "_sub_index": 0,
                "_split": False,
                "_sub_issue": True,
            })
            continue

        centers = _cluster(
            [
                center
                for candidate in candidates
                for center in candidate["centers"]
            ],
            tol=3.0,
        )

        if len(centers) <= 1:
            refined.append({
                "band_id": len(refined),
                "top": top,
                "bottom": bottom,
                "_parent_band_id": band["band_id"],
                "_parent_top": top,
                "_parent_bottom": bottom,
                "_sub_index": 0,
                "_split": False,
                "_sub_issue": False,
            })
            continue

        groups = [
            {"center": center, "words": []}
            for center in centers
        ]

        # 条件列是否也按同样的文本行对齐。
        condition_words = []

        for index, role in roles.items():
            if role not in condition_roles:
                continue

            if index >= len(column_boxes):
                continue

            c0, c1 = column_boxes[index]

            for word in words:
                cx = _bbox_center(word["bbox"])[0]

                if (
                    c0 - 1.0 <= cx <= c1 + 1.0
                    and in_band(word)
                ):
                    condition_words.append(word)

        condition_groups = _group_words_by_line(
            condition_words
        )
        centers = [group["center"] for group in groups]
        first_center = centers[0] if centers else None
        shared_condition_words = []
        local_condition_groups = []

        # 第一条数值行以上的条件行通常是整个父行共享的
        # 表头条件（如 PSRR 的 "f = 1 kHz, VIN = ..."），
        # 需要传播到每个子行，而不是只留在第一个子行。
        for group in condition_groups:
            if (
                first_center is not None
                and group["center"] < first_center - 3.0
            ):
                shared_condition_words.extend(
                    group["words"]
                )
            else:
                local_condition_groups.append(group)

        aligned = (
            len(local_condition_groups) <= 1
            or len(local_condition_groups) == len(groups)
        )

        if not aligned:
            issues.append({
                "table_id": raw_table.get("table_id"),
                "physical_row_band": {
                    "top": top,
                    "bottom": bottom,
                },
                "reason": (
                    "多值数值行数与条件行数不一致，"
                    "拆分结果需要人工复核"
                ),
                "value_line_count": len(groups),
                "condition_line_count": len(
                    local_condition_groups
                ),
            })

        boundaries = [top]

        for left, right in zip(centers, centers[1:]):
            boundaries.append((left + right) / 2.0)

        boundaries.append(bottom)

        for sub_index, (start, end) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            refined.append({
                "band_id": len(refined),
                "top": round(start, 2),
                "bottom": round(end, 2),
                "_parent_band_id": band["band_id"],
                "_parent_top": top,
                "_parent_bottom": bottom,
                "_sub_index": sub_index,
                "_split": True,
                "_split_count": len(groups),
                "_shared_condition_words": list(
                    shared_condition_words
                ),
                "_sub_issue": not aligned,
            })

    return refined, issues


_CONDITION_OPERATOR_END_RE = re.compile(
    r"(?:[<>=≤≥]|\b(?:to|through|\.\.\.|~))\s*$",
    re.IGNORECASE,
)
_CONDITION_VAR_START_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9/]*)"
)
_CONDITION_VAR_END_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9/]*)\s*$"
)
_CONDITION_WORD_END_RE = re.compile(r"([A-Za-z0-9]+)\s*$")


def _join_conditions(first, second):
    """连接同一条件单元在跨行/跨列处被切断的片段。

    只做“去掉重复片段”的拼接，不新增条件：
    - "1mA ≤ IOUT" + "IOUT ≤ 600mA" -> "1mA ≤ IOUT ≤ 600mA"
    - "VEN in OFF" + "OFF mode"    -> "VEN in OFF mode"
    - "10Hz ≤ f ≤" + "100kHz"      -> "10Hz ≤ f ≤ 100kHz"
    不满足这些形状时仍以 " | " 保留为两个独立条件。
    """

    first = (first or "").strip()
    second = (second or "").strip()

    if not first:
        return second

    if not second:
        return first

    # 上一片以比较符/范围连接词结尾，下一片是连续文本。
    if _CONDITION_OPERATOR_END_RE.search(first):
        return f"{first} {second}"

    # 两片共享同一个变量名（如 IOUT），去掉第一片末尾的重复变量。
    first_var = _CONDITION_VAR_END_RE.search(first)
    second_var = _CONDITION_VAR_START_RE.match(second)

    if (
        first_var
        and second_var
        and first_var.group(1).lower()
        == second_var.group(1).lower()
    ):
        prefix = first[:first_var.start(1)].rstrip()
        return f"{prefix} {second}".strip()

    # 两片共享同一个词（如 "OFF"），去掉重复词。
    first_word = _CONDITION_WORD_END_RE.search(first)
    second_word = _CONDITION_VAR_START_RE.match(second)

    if (
        first_word
        and second_word
        and first_word.group(1).lower()
        == second_word.group(1).lower()
    ):
        rest = second[second_word.end():].lstrip()
        return f"{first} {rest}".strip()

    # 条件文本从第一列溢出到第二列时（如 "mA" 或 "= 30 mA"），
    # 用空格连接，避免出现 " | mA"。
    if len(second) <= 5 or second[0] in "=<>≤≥":
        return f"{first} {second}"

    return f"{first} | {second}"


def _cell_x_extent(cell):
    """返回 cell 的 x 范围（左右坐标不受坐标原点影响）。"""

    bbox = cell.get("bbox")

    if not bbox:
        return None

    try:
        left = float(bbox["l"])
        right = float(bbox["r"])
    except (KeyError, TypeError, ValueError):
        return None

    return (min(left, right), max(left, right))


def _infer_column_boxes(
    raw_table,
    vertical_xs,
    table_bbox,
):
    """用 Docling cell 的 x 范围推断列边界。

    ADI LT3045 等表格的 MIN/TYP/MAX 之间没有竖线，
    仅靠页面竖线无法得到完整列数；此时用单列 cell 的
    文字 x 范围补出缺失边界，并优先采用落在两个相邻列
    文字范围之间的真实竖线。
    """

    columns = raw_table.get("columns", [])
    count = len(columns)

    if count <= 0:
        return []

    extents = {}

    for cell in raw_table.get("cells", []):
        start = cell.get("start_col_offset_idx")
        end = cell.get("end_col_offset_idx")

        if not isinstance(start, int) or not isinstance(end, int):
            continue

        if end - start != 1:
            continue

        extent = _cell_x_extent(cell)

        if extent is None:
            continue

        left = max(extent[0], table_bbox[0])
        right = min(extent[1], table_bbox[2])

        if right <= left:
            continue

        extents.setdefault(start, {"left": [], "right": []})
        extents[start]["left"].append(left)
        extents[start]["right"].append(right)

    def median(values):
        ordered = sorted(values)

        if not ordered:
            return None

        middle = len(ordered) // 2

        if len(ordered) % 2:
            return ordered[middle]

        return (ordered[middle - 1] + ordered[middle]) / 2.0

    column_extents = {
        index: (
            median(values["left"]),
            median(values["right"]),
        )
        for index, values in extents.items()
    }

    boundaries = [table_bbox[0]]
    table_width = table_bbox[2] - table_bbox[0]
    proportional_used = False

    for index in range(count - 1):
        left = column_extents.get(index)
        right = column_extents.get(index + 1)

        if left is not None and right is not None and left[1] <= right[0]:
            low, high = left[1], right[0]
            midpoint = (low + high) / 2.0
            candidates = sorted(
                vertical_xs,
                key=lambda value: abs(value - midpoint),
            )
            chosen = None

            for candidate in candidates:
                if low - 2.0 <= candidate <= high + 2.0:
                    chosen = candidate
                    break

            boundaries.append(
                chosen if chosen is not None else midpoint
            )
        else:
            proportional_used = True
            boundaries.append(
                table_bbox[0] + (index + 1) * table_width / count
            )

    boundaries.append(table_bbox[2])

    ordered = [boundaries[0]]

    for value in boundaries[1:]:
        if value <= ordered[-1] + 0.5:
            return []
        ordered.append(value)

    boxes = [
        (ordered[i], ordered[i + 1])
        for i in range(len(ordered) - 1)
    ]

    if len(boxes) != count:
        return []

    if proportional_used and not column_extents:
        return []

    return boxes


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

    count = len(raw_table.get("columns", []))

    if len(xs) - 1 == count:
        return [
            (xs[i], xs[i + 1])
            for i in range(len(xs) - 1)
        ]

    # 竖线不足（例如数值列靠空白分隔）时，用 cell 文字范围
    # 补出缺失边界。
    inferred = _infer_column_boxes(
        raw_table, xs, table_bbox
    )

    if inferred:
        return inferred

    # 最后回退：按列数均分表格宽度
    count = max(count, 1)
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
    用于 MIN/TYP/MAX 以及 UNIT；UNIT 只在同一列内兜底，
    不会跨列传播。
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

    base_bands = []

    for index, (top, bottom) in enumerate(
        zip(reference_ys[:-1], reference_ys[1:])
    ):
        base_bands.append({
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

    # 「一个 Docling 行内含多行数值」时拆成子行；
    # 子行只由 PDF 文本行决定，不新增数值。
    physical_bands, band_split_issues = (
        _refine_physical_bands(
            base_bands,
            table,
            cells,
            column_boxes,
            roles,
            words,
        )
    )
    unresolved_issues = list(band_split_issues)

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

    # 会被按文本行拆分的 base band：其单位由子行
    # 单独判定，不再对整段做歧义报警。
    split_parent_ranges = {
        (band["_parent_top"], band["_parent_bottom"])
        for band in physical_bands
        if band.get("_split")
    }

    # 单位区域歧义检测（按文本行判断，支持复合单位）
    unit_regions = {}
    legacy_ambiguous_unit_regions = set()

    for index, role in roles.items():
        if role != "unit":
            continue

        for region in column_regions.get(index, []):
            analysis = _unit_line_analysis(region["words"])
            distinct = analysis["distinct"]

            if (
                analysis["ambiguous"] or len(distinct) > 1
            ) and (
                (region["top"], region["bottom"])
                not in split_parent_ranges
            ):
                legacy_ambiguous_unit_regions.add(
                    (region["top"], region["bottom"])
                )
                unresolved_issues.append({
                    "table_id": table_id,
                    "physical_row_band": {
                        "top": region["top"],
                        "bottom": region["bottom"],
                    },
                    "reason": (
                        "单位区域内存在多个不同单位，"
                        "无法唯一传播"
                    ),
                    "units": sorted(
                        value for value in distinct if value
                    ),
                })

            unit_regions[(
                region["top"],
                region["bottom"],
            )] = region

    # 多值拆分一致性校验：
    # - 跨多个 base band 的 cell：数值数量需等于覆盖的
    #   base band 数（原有机制）；
    # - 单个 base band 内的多行 cell：数值数量需等于该列
    #   的 PDF 实际文本行数（新增机制）。
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

        text = cell.get("text") or ""
        overlapping = [
            band for band in base_bands
            if min(bbox[3], band["bottom"])
            - max(bbox[1], band["top"]) > 2.0
        ]
        same_line_multi = False

        if len(overlapping) > 1:
            observed_lines = len(overlapping)
        elif len(overlapping) == 1:
            band = overlapping[0]
            column_index = cell.get(
                "start_col_offset_idx"
            )

            if (
                not isinstance(column_index, int)
                or column_index >= len(column_boxes)
            ):
                continue

            c0, c1 = column_boxes[column_index]
            column_words = [
                word for word in words
                if c0 - 1.0
                <= _bbox_center(word["bbox"])[0]
                <= c1 + 1.0
                and band["top"] - 0.6
                <= _bbox_center(word["bbox"])[1]
                <= band["bottom"] + 0.6
            ]
            line_groups = _group_words_by_line(
                column_words
            )
            observed_lines = len(line_groups)

            if role in ("min", "typ", "max"):
                same_line_multi = any(
                    sum(
                        len(NUMBER_RE.findall(word["text"]))
                        for word in group["words"]
                    ) > 1
                    for group in line_groups
                )
        else:
            continue

        if role in ("min", "typ", "max"):
            expected = len(NUMBER_RE.findall(text))
        else:
            expected = observed_lines

        if expected <= 1:
            continue

        splittable = (
            expected == observed_lines
            and not same_line_multi
        )

        multivalue_cells[cell["cell_id"]] = {
            "expected_count": expected,
            "observed_lines": observed_lines,
            "splittable": splittable,
            "text": text,
            "role": role,
        }

        if not splittable:
            unresolved_issues.append({
                "table_id": table_id,
                "physical_row_band": {
                    "top": bbox[1],
                    "bottom": bbox[3],
                },
                "reason": (
                    "多值单元格的数值数量与文本行数不一致，"
                    "拒绝自动拆分"
                ),
                "source_cell_ids": [cell["cell_id"]],
                "expected_count": expected,
                "observed_lines": observed_lines,
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
        split = bool(band.get("_split"))
        unit_inherited = False

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

            role_regions[role] = region
            words_for_role = region["words"]

            if split and role in (
                "min", "typ", "max",
                "condition1", "condition2", "unit",
            ):
                words_for_role = [
                    word for word in region["words"]
                    if band["top"] - 0.6
                    <= _bbox_center(word["bbox"])[1]
                    <= band["bottom"] + 0.6
                ]

                if role in (
                    "condition1", "condition2",
                ):
                    shared_words = (
                        band.get("_shared_condition_words") or []
                    )

                    if shared_words:
                        local_ids = {
                            id(word)
                            for word in words_for_role
                        }
                        words_for_role = [
                            word for word in shared_words
                            if id(word) not in local_ids
                        ] + words_for_role

                if not words_for_role:
                    if role in ("condition1", "condition2"):
                        parent_lines = _group_words_by_line(
                            region["words"]
                        )

                        if len(parent_lines) <= 1:
                            words_for_role = region["words"]
                    elif role == "unit":
                        analysis = _unit_line_analysis(
                            region["words"]
                        )
                        expressions = {
                            value
                            for value in analysis["expressions"]
                            if value
                        }

                        if len(expressions) == 1:
                            words_for_role = region["words"]
                            unit_inherited = True

            role_words[role] = words_for_role

            if role == "unit" and split:
                analysis = _unit_line_analysis(
                    words_for_role
                )
                expressions = [
                    value
                    for value in analysis["expressions"]
                    if value
                ]

                if (
                    analysis["ambiguous"]
                    or len(set(expressions)) > 1
                ):
                    role_texts[role] = ""
                elif len(expressions) == 1:
                    role_texts[role] = expressions[0]
                else:
                    role_texts[role] = line_aware_text(
                        words_for_role
                    ).strip()
            else:
                role_texts[role] = line_aware_text(
                    words_for_role
                ).strip()

        if split and (
            _unit_line_analysis(
                role_words.get("unit", [])
            )["ambiguous"]
        ):
            role_texts["unit"] = ""
            role_words["unit"] = []

        # 数值字段的 Unicode 负号归一为 ASCII '-'，
        # 以便下游解析；不做任何数值猜测。
        for role in ("min", "typ", "max"):
            if role_texts.get(role):
                role_texts[role] = normalize_numeric_glyphs(
                    role_texts[role]
                )

        # 子行条件列可能承载独立的参数标签，例如 ADI
        # “Programming Scale Factor: 2V < VIN < 20V”。
        # 此时把标签提升为 parameter_symbol，原参数名保留
        # 在 parameter_name，避免把两个参数合并成一行。
        if split:
            for role in ("condition1", "condition2"):
                text = role_texts.get(role, "")
                match = CONDITION_LABEL_RE.match(text)

                if not match:
                    continue

                label = match.group(1).strip()

                if not label or re.search(r"\d", label):
                    continue

                role_texts[role] = match.group(2).strip()
                role_texts["parameter_name"] = (
                    role_texts.get("parameter_name")
                    or role_texts.get("parameter_symbol", "")
                )
                role_texts["parameter_symbol"] = label
                break

        used_words = []

        for role in role_words:
            used_words.extend(role_words[role])

        used_words = _dedupe_words(used_words)

        source_cells = []

        for role, role_word_list in role_words.items():
            allow_nearest = role in (
                "min", "typ", "max", "unit",
            )

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

        # 保留原始 Docling 单位文本，便于审计单位纠正
        # （如 Docling 把 µA 读成 mA）。
        docling_unit_text = ""

        for cell in source_cells:
            if roles.get(
                cell.get("start_col_offset_idx")
            ) == "unit":
                docling_unit_text = (
                    cell.get("text") or ""
                ).strip()
                break

        glyph_evidence = []

        for role_name in (
            "unit", "condition1", "condition2",
            "min", "typ", "max",
        ):
            for word in role_words.get(role_name, []):
                if not word.get("glyph_restored"):
                    continue

                glyph_evidence.append({
                    "role": role_name,
                    "char": word["glyph_restored"],
                    "bbox": word["glyph_bbox"],
                    "word": word["text"],
                })

        actions = ["physical_row_banding"]
        needs_review = False

        if band.get("_sub_issue"):
            needs_review = True

        # 单位跨行传播
        unit_region = role_regions.get("unit")

        if (
            not split
            and unit_region is not None
            and (
                unit_region["top"],
                unit_region["bottom"],
            ) in legacy_ambiguous_unit_regions
        ):
            # 证据不足：不自动填充单位
            role_texts["unit"] = ""
            role_words["unit"] = []
            needs_review = True

        if unit_region is not None and not split:
            unit_band_count = sum(
                1
                for candidate in base_bands
                if candidate["top"] >= unit_region["top"] - 0.6
                and candidate["bottom"] <= unit_region["bottom"] + 0.6
            )

            if unit_band_count > 1:
                actions.append("unit_span_recovery")

        if split and unit_inherited:
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

                if not info["splittable"]:
                    needs_review = True

        if split:
            actions.append("intra_row_line_split")

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

            allow_nearest = role in (
                "min", "typ", "max", "unit",
            )
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

        parent_band_id = band.get(
            "_parent_band_id", band["band_id"]
        )

        if split:
            logical_row_id = (
                f"t{table_id}.p{parent_band_id}"
                f".s{band['_sub_index']}"
            )
        else:
            logical_row_id = (
                f"t{table_id}.p{parent_band_id}"
            )

        logical_row = {
            "logical_row_id": logical_row_id,
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
            "docling_unit_text": docling_unit_text,
            "glyph_evidence": glyph_evidence,
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
        }

        if split:
            logical_row["parent_physical_row_band"] = {
                "top": band["_parent_top"],
                "bottom": band["_parent_bottom"],
            }
            logical_row["sub_row_index"] = band[
                "_sub_index"
            ]
            logical_row["sub_row_count"] = band[
                "_split_count"
            ]

        logical_rows.append(logical_row)

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
        "physical_row_bands": [
            {
                "band_id": band["band_id"],
                "top": band["top"],
                "bottom": band["bottom"],
                "parent_band_id": band.get(
                    "_parent_band_id"
                ),
                "sub_row_index": (
                    band.get("_sub_index")
                    if band.get("_split")
                    else None
                ),
            }
            for band in physical_bands
        ],
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


# 字体私有编码中的字形映射。
# NCP170 的 µ 在 PDF 文本层是控制字符 \x02，
# PyMuPDF 的 words 模式会丢掉该字形。
# InterleafSymbols 的 \x02 是微点标记，不是 µ，禁止映射。
GLYPH_REPLACEMENTS = {
    "IntDutch801G-Roman": {"\x02": "µ"},
}


def _restore_encoded_glyphs(page, words):
    """把字体私有编码的字形拼回右侧相邻 word。

    只按字体白名单处理；不新增、不修改任何数值，
    只恢复 PDF 中真实存在但文本层丢失的字形。
    """

    try:
        raw = page.get_text("rawdict")
    except Exception:  # pragma: no cover - 防御性分支
        return words

    glyphs = []

    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                mapping = GLYPH_REPLACEMENTS.get(
                    span.get("font") or ""
                )

                if not mapping:
                    continue

                for char in span.get("chars", []):
                    replacement = mapping.get(char.get("c"))

                    if replacement is None:
                        continue

                    bbox = char.get("bbox")

                    if not bbox or len(bbox) != 4:
                        continue

                    glyphs.append((
                        replacement,
                        tuple(float(v) for v in bbox),
                    ))

    for replacement, bbox in glyphs:
        gx0, gy0, gx1, gy1 = bbox
        center_y = (gy0 + gy1) / 2.0
        best_index = None
        best_dx = None

        for index, word in enumerate(words):
            wx0, wy0, wx1, wy1 = word["bbox"]
            word_center_y = (wy0 + wy1) / 2.0

            if abs(word_center_y - center_y) > 3.0:
                continue

            dx = wx0 - gx1

            if dx < -2.5 or dx > 3.0:
                continue

            if best_index is None or abs(dx) < abs(best_dx):
                best_index = index
                best_dx = dx

        if best_index is None:
            continue

        word = words[best_index]

        if word["text"].startswith(replacement):
            continue

        words[best_index] = {
            **word,
            "text": replacement + word["text"],
            "glyph_restored": replacement,
            "glyph_bbox": list(bbox),
        }

    return words


def extract_page_words(page):
    """提取页面 words，并恢复字体私有编码的字形。"""

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

    return _restore_encoded_glyphs(page, words)


def extract_page_geometry(page, table_bbox_tl=None):
    """从 PyMuPDF 页面提取 words 与表格线段。"""

    words = extract_page_words(page)

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


_RESTORABLE_GLYPHS = {
    replacement
    for mapping in GLYPH_REPLACEMENTS.values()
    for replacement in mapping.values()
}


def _glyph_only_change(pdf_text, docling_text):
    """判断 PDF 文本与 Docling 文本是否只差恢复的字形。

    NCP170 的 Docling 文本把 µA 读成 mA（字符替换），
    而不是控制字符残留；因此允许同一位置上的
    字形差异，但要求长度与 token 数一致，避免
    把结构差异误当成字形恢复。
    """

    pdf = str(pdf_text or "").strip()
    docling = str(docling_text or "").strip()

    if not pdf or not docling or pdf == docling:
        return False

    pdf_tokens = pdf.split()
    docling_tokens = docling.split()

    if len(pdf_tokens) != len(docling_tokens):
        return False

    changed = False

    for pdf_token, docling_token in zip(
        pdf_tokens, docling_tokens
    ):
        if pdf_token == docling_token:
            continue

        if len(pdf_token) != len(docling_token):
            return False

        differences = [
            (pdf_char, docling_char)
            for pdf_char, docling_char in zip(
                pdf_token, docling_token
            )
            if pdf_char != docling_char
        ]

        if not differences:
            continue

        if any(
            pdf_char not in _RESTORABLE_GLYPHS
            for pdf_char, _ in differences
        ):
            return False

        changed = True

    return changed


def restore_flat_table_glyphs(
    table,
    raw_table,
    *,
    words_by_page,
    page_heights,
):
    """把 PDF 字形证据应用到 Flat 表格单元格。

    只修改 Docling 文本与 PDF 文本仅相差已登记字形的
    单元格；原始 Docling 文本保留在 docling_cells 中。
    返回 (restored_table, evidence)，不修改输入对象。
    """

    restored = deepcopy(table)
    rows = restored.get("rows") or []
    header_row_count = int(
        raw_table.get("header_row_count") or 0
    )
    evidence = []

    for cell in raw_table.get("cells", []):
        page_no = cell.get("page_no")
        page_words = words_by_page.get(page_no)

        if page_words is None:
            continue

        bbox = to_topleft(
            cell.get("bbox"),
            page_heights.get(page_no),
        )

        if bbox is None:
            continue

        cell_words = []

        for word in page_words:
            cx, cy = _bbox_center(word["bbox"])

            if (
                bbox[0] - 1.0 <= cx <= bbox[2] + 1.0
                and bbox[1] - 1.0 <= cy <= bbox[3] + 1.0
            ):
                cell_words.append(word)

        if not any(
            word.get("glyph_restored")
            for word in cell_words
        ):
            continue

        pdf_text = line_aware_text(cell_words).strip()
        docling_text = str(cell.get("text") or "").strip()

        if not _glyph_only_change(pdf_text, docling_text):
            continue

        start_row = int(
            cell.get("start_row_offset_idx", 0)
        ) - header_row_count
        end_row = int(
            cell.get("end_row_offset_idx", 0)
        ) - header_row_count
        start_col = int(cell.get("start_col_offset_idx", 0))
        end_col = int(cell.get("end_col_offset_idx", 0))
        updated = []

        for row_index in range(
            max(start_row, 0), min(end_row, len(rows))
        ):
            row = rows[row_index]

            for column_index in range(
                start_col, min(end_col, len(row))
            ):
                current = row[column_index]

                if current is None:
                    continue

                if str(current).strip() != docling_text:
                    continue

                row[column_index] = pdf_text
                updated.append({
                    "row_index": row_index,
                    "column_index": column_index,
                })

        if not updated:
            continue

        evidence.append({
            "cell_id": cell.get("cell_id"),
            "page": page_no,
            "docling_text": docling_text,
            "pdf_text": pdf_text,
            "bbox": [round(value, 2) for value in bbox],
            "updated_cells": updated,
            "glyph_evidence": [
                {
                    "char": word["glyph_restored"],
                    "word": word["text"],
                    "bbox": list(word["glyph_bbox"]),
                }
                for word in cell_words
                if word.get("glyph_restored")
            ],
        })

    return restored, evidence


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

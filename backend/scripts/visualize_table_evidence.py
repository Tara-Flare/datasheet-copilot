"""阶段 B0：PDF 视觉证据对齐（只读）。

- 用 PyMuPDF 渲染第 6 页；
- 统一 BOTTOMLEFT / TOPLEFT 坐标后绘制 table 与 cell；
- 输出整页渲染、cell 标注图、三类问题局部放大图；
- 对比 CPU 局部解析 raw_structure 与旧完整缓存。

不做任何单位继承、行拆分或自动恢复。
"""

import argparse
import json
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PDF = "data/samples/TPS7A02.pdf"
DEFAULT_RAW = (
    "data/outputs/TPS7A02.raw_structure.v1.json"
)
DEFAULT_CACHE_DIR = "data/cache/docling"
DEFAULT_OUT = "data/outputs/diagnostics"

FONT_PATH = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
)
FONT_BOLD = (
    "/usr/share/fonts/truetype/dejavu/"
    "DejaVuSans-Bold.ttf"
)

COLOR_TABLE = (210, 0, 0)
COLOR_DEFAULT = (150, 150, 150)
COLOR_A = (0, 150, 0)
COLOR_B = (0, 80, 220)
COLOR_C = (200, 0, 180)


def to_topleft(bbox, page_height):
    """把 Docling bbox 统一为 TOPLEFT 页面坐标。"""

    if bbox is None:
        return None

    if bbox.get("coord_origin") == "BOTTOMLEFT":
        return (
            bbox["l"],
            page_height - bbox["t"],
            bbox["r"],
            page_height - bbox["b"],
        )

    return (
        bbox["l"],
        bbox["t"],
        bbox["r"],
        bbox["b"],
    )


def load_raw(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def find_raw_table(raw, table_id):
    for table in raw["tables"]:
        if table["table_id"] == table_id:
            return table

    raise KeyError(f"raw_structure 中无 table_id={table_id}")


def find_cached_table(cache_dir, table_id):
    cache_dir = Path(cache_dir)

    if not cache_dir.is_dir():
        return None, None

    for path in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue

        for table in data.get("tables", []):
            if table.get("table_id") == table_id:
                return path, table

    return None, None


def materialize_grid(table):
    """按 Docling grid 语义（span 复制）还原扁平行。"""

    num_rows = table["num_rows"]
    num_cols = table["num_cols"]

    grid = [
        ["" for _ in range(num_cols)]
        for _ in range(num_rows)
    ]

    for cell in table["cells"]:
        text = cell.get("text") or ""

        for row in range(
            cell["start_row_offset_idx"],
            cell["end_row_offset_idx"],
        ):
            for col in range(
                cell["start_col_offset_idx"],
                cell["end_col_offset_idx"],
            ):
                if (
                    row < num_rows
                    and col < num_cols
                ):
                    grid[row][col] = text

    header = table["header_row_count"]

    return grid[header:]


def normalize_row(row):
    return [
        "" if value is None else str(value)
        for value in row
    ]


def consistency_check(raw_table, cache_path, cache_table):
    new_rows = materialize_grid(raw_table)

    result = {
        "raw_structure_table_id": raw_table["table_id"],
        "old_cache_path": (
            str(cache_path) if cache_path else None
        ),
        "old_cache_found": cache_table is not None,
        "columns_equal": None,
        "row_count_equal": None,
        "rows_equal": None,
        "row_differences": [],
        "notes": [],
    }

    if cache_table is None:
        result["notes"].append(
            "未找到旧缓存中的 table_id，无法逐行比较"
        )
        return result

    old_rows = [
        normalize_row(row)
        for row in cache_table.get("rows", [])
    ]

    new_norm = [
        normalize_row(row) for row in new_rows
    ]

    result["columns_equal"] = (
        list(raw_table["columns"])
        == list(cache_table.get("columns", []))
    )
    result["row_count_equal"] = (
        len(new_norm) == len(old_rows)
    )
    result["rows_equal"] = new_norm == old_rows

    for index in range(
        max(len(new_norm), len(old_rows))
    ):
        new_row = (
            new_norm[index]
            if index < len(new_norm)
            else None
        )
        old_row = (
            old_rows[index]
            if index < len(old_rows)
            else None
        )

        if new_row != old_row:
            result["row_differences"].append({
                "row_index": index,
                "new_row": new_row,
                "old_row": old_row,
            })

    result["notes"].append(
        "旧缓存为扁平 rows，不含原始 cell/span/bbox，"
        "因此只能比较扁平视图，无法比较 cell 级结构"
    )

    return result


def row_y_map(table, page_height):
    """仅用单行 cell 推断每个 grid row 的 y 范围。"""

    rows = {}

    for cell in table["cells"]:
        if (
            cell["end_row_offset_idx"]
            - cell["start_row_offset_idx"]
            != 1
        ):
            continue

        bb = to_topleft(
            cell["bbox"], page_height
        )

        if bb is None:
            continue

        grid_row = cell["start_row_offset_idx"]
        rows.setdefault(
            grid_row,
            [bb[1], bb[3]],
        )
        rows[grid_row][0] = min(rows[grid_row][0], bb[1])
        rows[grid_row][1] = max(rows[grid_row][1], bb[3])

    return rows


def intersects(cell, grid_rows):
    start = cell["start_row_offset_idx"]
    end = cell["end_row_offset_idx"]

    return any(start <= r < end for r in grid_rows)


def cell_color(cell, focus_a, focus_b, focus_c):
    if intersects(cell, focus_c):
        return COLOR_C
    if intersects(cell, focus_a):
        return COLOR_A
    if intersects(cell, focus_b):
        return COLOR_B
    return COLOR_DEFAULT


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default(size=size)


def render(page, clip_tl, dpi):
    rect = fitz.Rect(*clip_tl)
    pix = page.get_pixmap(clip=rect, dpi=dpi)

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )

    return img, clip_tl, dpi / 72.0


def to_px(bb, clip, scale):
    return (
        (bb[0] - clip[0]) * scale,
        (bb[1] - clip[1]) * scale,
        (bb[2] - clip[0]) * scale,
        (bb[3] - clip[1]) * scale,
    )


def draw_annotations(
    img,
    clip,
    scale,
    table,
    page_height,
    focus_a,
    focus_b,
    focus_c,
    label_size,
    show_all_ids=True,
    draw_row_labels=False,
):
    draw = ImageDraw.Draw(img)
    font = load_font(FONT_PATH, label_size)
    font_bold = load_font(
        FONT_BOLD, label_size
    )

    table_tl = to_topleft(
        table["table_bbox"], page_height
    )

    if table_tl is not None:
        draw.rectangle(
            to_px(table_tl, clip, scale),
            outline=COLOR_TABLE,
            width=max(3, int(scale * 1.2)),
        )

    for cell in table["cells"]:
        bb = to_topleft(
            cell["bbox"], page_height
        )

        if bb is None:
            continue

        if (
            bb[2] < clip[0]
            or bb[0] > clip[2]
            or bb[3] < clip[1]
            or bb[1] > clip[3]
        ):
            continue

        color = cell_color(
            cell, focus_a, focus_b, focus_c
        )
        width = (
            max(3, int(scale * 1.0))
            if color != COLOR_DEFAULT
            else max(1, int(scale * 0.4))
        )

        p = to_px(bb, clip, scale)
        draw.rectangle(p, outline=color, width=width)

        is_focus = color != COLOR_DEFAULT

        if not (show_all_ids or is_focus):
            continue

        text = f"c{cell['cell_id']}"
        fnt = font_bold if is_focus else font

        tx = p[0] + 2
        ty = p[1] + 1

        try:
            tb = draw.textbbox(
                (tx, ty), text, font=fnt
            )
            draw.rectangle(
                (
                    tb[0] - 1,
                    tb[1] - 1,
                    tb[2] + 1,
                    tb[3] + 1,
                ),
                fill=(255, 255, 210),
            )
        except Exception:
            pass

        draw.text(
            (tx, ty),
            text,
            fill=color,
            font=fnt,
        )

    if draw_row_labels:
        ymap = row_y_map(table, page_height)
        header = table["header_row_count"]
        font_row = load_font(
            FONT_BOLD, max(12, int(label_size * 0.9))
        )

        for grid_row, (top, bottom) in ymap.items():
            if not (clip[1] <= top <= clip[3]):
                continue

            row_index = grid_row - header
            text = f"row{row_index}"

            x = (
                (table_tl[0] - 30 - clip[0])
                * scale
            )
            y = (
                ((top + bottom) / 2 - clip[1])
                * scale
                - label_size * 0.6
            )

            draw.text(
                (x, y),
                text,
                fill=(0, 0, 0),
                font=font_row,
            )


def focus_clip(table, grid_rows, page_width, page_height):
    table_tl = to_topleft(
        table["table_bbox"], page_height
    )

    ymap = row_y_map(table, page_height)

    ys = [
        ymap[r]
        for r in grid_rows
        if r in ymap
    ]

    # 用覆盖 focus 行、且 bbox 更小的 cell 补充 y 范围
    for cell in table["cells"]:
        if not intersects(cell, grid_rows):
            continue

        bb = to_topleft(cell["bbox"], page_height)

        if bb is None:
            continue

        ys.append((bb[1], bb[3]))

    if not ys:
        return (0, 0, page_width, page_height)

    top = min(y[0] for y in ys) - 4
    bottom = max(y[1] for y in ys) + 4

    left = table_tl[0] - 34
    right = table_tl[2] + 4

    return (
        max(0, left),
        max(0, top),
        min(page_width, right),
        min(page_height, bottom),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="阶段 B0 视觉证据对齐（只读）"
    )
    parser.add_argument(
        "--pdf", default=DEFAULT_PDF
    )
    parser.add_argument(
        "--raw-structure", default=DEFAULT_RAW
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUT
    )
    parser.add_argument(
        "--table-id", type=int, default=9
    )
    parser.add_argument(
        "--page-no", type=int, default=6
    )

    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.raw_structure)
    table = find_raw_table(raw, args.table_id)

    doc = fitz.open(args.pdf)
    page = doc[args.page_no - 1]
    page_height = page.rect.height
    page_width = page.rect.width

    header = table["header_row_count"]

    def rows_to_grid(row_indices):
        return [
            index + header
            for index in row_indices
        ]

    focus_a = set(rows_to_grid(range(8, 13)))
    focus_b = set(rows_to_grid(range(18, 31)))
    focus_c = set(rows_to_grid(range(29, 33)))

    # 1) 整页原始渲染
    img, clip, scale = render(
        page,
        (0, 0, page_width, page_height),
        300,
    )
    img.save(out_dir / "page6_render.png")

    # 2) 整页标注
    img, clip, scale = render(
        page,
        (0, 0, page_width, page_height),
        300,
    )
    draw_annotations(
        img,
        clip,
        scale,
        table,
        page_height,
        focus_a,
        focus_b,
        focus_c,
        label_size=17,
        show_all_ids=True,
    )
    img.save(out_dir / "page6_cells_annotated.png")

    # 3) 三个局部放大
    crops = {
        "focus_A_ground_current.png": (
            focus_a,
            "A ground current",
        ),
        "focus_B_dropout_unit.png": (
            focus_b,
            "B dropout unit",
        ),
        "focus_C_multi_value.png": (
            focus_c,
            "C multi value",
        ),
    }

    for filename, (rows, _label) in crops.items():
        cclip = focus_clip(
            table,
            rows,
            page_width,
            page_height,
        )
        img, clip, scale = render(
            page, cclip, 600
        )
        draw_annotations(
            img,
            clip,
            scale,
            table,
            page_height,
            focus_a,
            focus_b,
            focus_c,
            label_size=26,
            show_all_ids=True,
            draw_row_labels=True,
        )
        img.save(out_dir / filename)

    # 4) 一致性检查
    cache_path, cache_table = find_cached_table(
        args.cache_dir, args.table_id
    )
    consistency = consistency_check(
        table, cache_path, cache_table
    )

    (out_dir / "consistency_check.json").write_text(
        json.dumps(
            consistency,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("输出目录:", out_dir)
    print(
        "整页渲染:",
        out_dir / "page6_render.png",
    )
    print(
        "cell 标注:",
        out_dir / "page6_cells_annotated.png",
    )
    for filename in crops:
        print("局部放大:", out_dir / filename)

    print()
    print("=== 一致性检查 ===")
    print(
        "旧缓存:",
        consistency["old_cache_path"],
    )
    print(
        "columns_equal:",
        consistency["columns_equal"],
    )
    print(
        "row_count_equal:",
        consistency["row_count_equal"],
    )
    print(
        "rows_equal:",
        consistency["rows_equal"],
    )
    print(
        "row_differences:",
        len(consistency["row_differences"]),
    )

    for diff in consistency["row_differences"][:10]:
        print(
            "  row",
            diff["row_index"],
            "new=",
            diff["new_row"],
        )
        print(
            "           old=",
            diff["old_row"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

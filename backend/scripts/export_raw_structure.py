"""导出 Docling 原始 TableCell 结构（阶段 A，只读+独立产物）。

用法示例：

    PYTHONPATH=backend .venv/bin/python \
        backend/scripts/export_raw_structure.py \
        data/samples/TPS7A02.pdf \
        --output data/outputs/TPS7A02.raw_structure.v1.json \
        --device cpu \
        --table-id 9 --rows 8-12,18-30

说明：
- 默认强制 CPU 解析，绕开当前环境的 CUDA/cuDNN 加载失败；
- 默认复用本地 docling-models（DOCLING_ARTIFACTS_PATH），
  不会自动下载新模型；
- 只写入独立的 raw_structure 文件，不覆盖旧表格缓存；
- 不实现任何单位继承或自动行拆分。
"""

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = PROJECT_ROOT / "docling-models"
DEFAULT_OUTPUT_DIR = Path("data/outputs")


def configure_environment(device: str) -> None:
    """在导入 docling/torch 之前固定运行环境。"""

    if device == "cpu":
        # 让 torch 看不到 GPU，从而回退到 CPU，
        # 规避 CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED。
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not os.environ.get("DOCLING_ARTIFACTS_PATH"):
        if DEFAULT_ARTIFACTS.is_dir():
            os.environ["DOCLING_ARTIFACTS_PATH"] = str(
                DEFAULT_ARTIFACTS
            )


def parse_ranges(spec: str) -> list[int]:
    """解析 '8-12,18-30' 形式的行号列表。"""

    rows = []

    for part in spec.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start, end = part.split("-", 1)
            rows.extend(
                range(int(start), int(end) + 1)
            )
        else:
            rows.append(int(part))

    return sorted(set(rows))


def parse_page_range(spec):
    """解析 '1-6' 或 '6' 形式的页范围。"""

    if not spec:
        return None

    if "-" in spec:
        start, end = spec.split("-", 1)
        return (int(start), int(end))

    page = int(spec)
    return (page, page)


def find_table(raw_structure, table_id):
    for table in raw_structure.get("tables", []):
        if table.get("table_id") == table_id:
            return table

    return None


def cells_for_row(table, row_index):
    """返回覆盖某个扁平 row_index 的原始 cell。"""

    header = int(
        table.get("header_row_count", 0)
    )
    grid_row = row_index + header

    result = []

    for cell in table.get("cells", []):
        start = cell["start_row_offset_idx"]
        end = cell["end_row_offset_idx"]

        if start <= grid_row < end:
            result.append(cell)

    return grid_row, result


def format_bbox(bbox):
    if not bbox:
        return "None"

    return (
        f"l={bbox['l']:.1f} t={bbox['t']:.1f} "
        f"r={bbox['r']:.1f} b={bbox['b']:.1f} "
        f"({bbox.get('coord_origin')})"
    )


def print_table_focus(
    raw_structure,
    table_id,
    row_indices,
):
    table = find_table(raw_structure, table_id)

    if table is None:
        print(
            f"[focus] 未找到 table_id={table_id}"
        )
        return

    print()
    print("=" * 70)
    print(
        f"table_id={table_id} 原始 cell 聚焦"
    )
    print("=" * 70)
    print(
        "page_numbers:",
        table["page_numbers"],
        " num_rows:",
        table["num_rows"],
        " num_cols:",
        table["num_cols"],
        " header_row_count:",
        table["header_row_count"],
        " row_count:",
        table["row_count"],
    )
    print("columns:", table["columns"])
    print(
        "mapping_consistent:",
        table["mapping_consistent"],
        " header_row_count_detected:",
        table["header_row_count_detected"],
    )

    for row_index in row_indices:
        grid_row, cells = cells_for_row(
            table,
            row_index,
        )

        print()
        print(
            f"--- row_index={row_index} "
            f"(grid_row={grid_row}) ---"
        )

        for cell in sorted(
            cells,
            key=lambda c: c[
                "start_col_offset_idx"
            ],
        ):
            print(
                "  cell_id={cid} r{rs}-{re} c{cs}-{ce} "
                "span=({rspan},{cspan}) bbox=({bbox}) "
                "ch={ch} rh={rh} rs={rsec} text={text!r}".format(
                    cid=cell["cell_id"],
                    rs=cell["start_row_offset_idx"],
                    re=cell["end_row_offset_idx"],
                    cs=cell["start_col_offset_idx"],
                    ce=cell["end_col_offset_idx"],
                    rspan=cell["row_span"],
                    cspan=cell["col_span"],
                    bbox=format_bbox(
                        cell["bbox"]
                    ),
                    ch=cell["column_header"],
                    rh=cell["row_header"],
                    rsec=cell["row_section"],
                    text=(cell["text"] or "")[:80],
                )
            )


def print_span_summary(raw_structure, table_id):
    table = find_table(raw_structure, table_id)

    if table is None:
        return

    print()
    print("=" * 70)
    print(
        f"table_id={table_id} 跨行/跨列 cell 汇总"
    )
    print("=" * 70)

    spanning = [
        cell
        for cell in table.get("cells", [])
        if cell["row_span"] > 1
        or cell["col_span"] > 1
    ]

    if not spanning:
        print("  (无 span>1 的 cell)")
        return

    for cell in sorted(
        spanning,
        key=lambda c: (
            c["start_row_offset_idx"],
            c["start_col_offset_idx"],
        ),
    ):
        print(
            "  cell_id={cid} r{rs}-{re} c{cs}-{ce} "
            "span=({rspan},{cspan}) text={text!r}".format(
                cid=cell["cell_id"],
                rs=cell["start_row_offset_idx"],
                re=cell["end_row_offset_idx"],
                cs=cell["start_col_offset_idx"],
                ce=cell["end_col_offset_idx"],
                rspan=cell["row_span"],
                cspan=cell["col_span"],
                text=(cell["text"] or "")[:60],
            )
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "导出 Docling 原始 TableCell 结构"
            "（阶段 A，不做自动修复）"
        )
    )

    parser.add_argument(
        "pdf",
        help="Datasheet PDF 路径",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "raw_structure 输出路径"
            "（默认 data/outputs/<stem>.raw_structure.v1.json）"
        ),
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "同时按主缓存 key 写入独立 raw_structure 文件"
            "（与旧缓存分离，不覆盖）"
        ),
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="解析设备，默认 cpu（规避当前 CUDA 问题）",
    )

    parser.add_argument(
        "--page-range",
        default=None,
        help=(
            "限定解析页范围，如 1-6；"
            "默认解析全部页。选择覆盖目标表格的连续前缀页"
            "不会改变其 table_id。"
        ),
    )

    parser.add_argument(
        "--table-id",
        type=int,
        default=9,
        help="聚焦输出的 table_id，默认 9",
    )

    parser.add_argument(
        "--rows",
        default="8-12,18-30",
        help="聚焦输出的扁平 row_index 列表，默认 8-12,18-30",
    )

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    configure_environment(args.device)

    # 必须在设置环境变量之后再导入 docling
    from app.parsers.table_parser import TableParser
    from app.parsers.raw_structure import (
        RAW_STRUCTURE_VERSION,
        RawStructureStore,
        save_raw_structure,
    )

    pdf_path = Path(args.pdf).resolve()

    if not pdf_path.is_file():
        print(
            f"错误: PDF 不存在: {pdf_path}",
            file=sys.stderr,
        )
        return 1

    output = args.output

    if output is None:
        output = (
            DEFAULT_OUTPUT_DIR
            / f"{pdf_path.stem}.raw_structure"
            f".v{RAW_STRUCTURE_VERSION}.json"
        )

    print("PDF:", pdf_path)
    print("device:", args.device)
    print(
        "DOCLING_ARTIFACTS_PATH:",
        os.environ.get("DOCLING_ARTIFACTS_PATH"),
    )

    parser = TableParser()

    print("开始解析（首次可能较慢）...")

    result = parser.parse(
        str(pdf_path),
        include_raw_structure=True,
        page_range=parse_page_range(args.page_range),
    )

    raw_structure = result.get("raw_structure")

    if raw_structure is None:
        print(
            "错误: 未生成 raw_structure",
            file=sys.stderr,
        )
        return 1

    saved = save_raw_structure(
        raw_structure,
        output,
    )

    print("原始结构已保存:", saved)

    if args.cache_dir:
        config_signature = {
            "do_ocr": getattr(
                parser.pipeline_options,
                "do_ocr",
                None,
            ),
            "do_table_structure": getattr(
                parser.pipeline_options,
                "do_table_structure",
                None,
            ),
        }

        store = RawStructureStore(args.cache_dir)
        key = store.build_key(
            str(pdf_path),
            config_signature,
        )
        store_path = store.save(key, raw_structure)
        print("缓存旁独立文件:", store_path)

    print_table_focus(
        raw_structure,
        args.table_id,
        parse_ranges(args.rows),
    )

    print_span_summary(
        raw_structure,
        args.table_id,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

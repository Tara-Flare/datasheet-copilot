"""阶段 B3.1：生成逻辑表格 LLM 输入（只读，不调用 LLM）。"""

import argparse
import json
from pathlib import Path

from app.extraction.logical_table_adapter import (
    adapt_logical_rows,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="把 logical_rows 转为 LLM 输入"
    )
    parser.add_argument(
        "--logical-rows",
        default="data/outputs/TPS7A02.logical_rows.v1.json",
    )
    parser.add_argument(
        "--raw-structure",
        default="data/outputs/TPS7A02.raw_structure.v1.json",
        help=(
            "用于按 cell 几何选择更精确的 primary_row_index"
        ),
    )
    parser.add_argument(
        "--table-id",
        type=int,
        action="append",
        default=None,
        help=(
            "只包含指定 table_id；可重复。"
            "建议显式指定通用规格表（当前为 9）。"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/outputs/"
            "TPS7A02.logical_table_input.v1.json"
        ),
    )

    args = parser.parse_args(argv)

    logical_rows = json.loads(
        Path(args.logical_rows).read_text(
            encoding="utf-8"
        )
    )
    raw_structure = None

    if Path(args.raw_structure).is_file():
        raw_structure = json.loads(
            Path(args.raw_structure).read_text(
                encoding="utf-8"
            )
        )

    table_ids = args.table_id

    if table_ids is None:
        table_ids = [9]

    adapted = adapt_logical_rows(
        logical_rows,
        raw_structure=raw_structure,
        table_ids=table_ids,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(
        json.dumps(adapted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(output)

    print("逻辑表格输入已保存:", output)
    print("table_count:", adapted["table_count"])
    print("row_count:", adapted["row_count"])
    print("excluded_count:", adapted["excluded_count"])

    for table in adapted["tables"]:
        print(
            "table_id:",
            table["table_id"],
            "page:",
            table["page"],
            "rows:",
            table["row_count"],
            "excluded:",
            len(table["excluded_rows"]),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

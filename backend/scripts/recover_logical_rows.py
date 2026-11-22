"""生成 table_id=9 的逻辑行恢复结果（阶段 B1，只读）。

用法：
    PYTHONPATH=backend .venv/bin/python \
        backend/scripts/recover_logical_rows.py \
        --raw data/outputs/TPS7A02.raw_structure.v1.json \
        --pdf data/samples/TPS7A02.pdf \
        --table-id 9 --page-no 6 \
        --output data/outputs/TPS7A02.logical_rows.v1.json
"""

import argparse
import json
from pathlib import Path

from app.parsers.structure_recovery import (
    recover_from_pdf,
    save_logical_rows,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="基于 PDF 几何证据恢复逻辑行"
    )
    parser.add_argument(
        "--raw",
        default="data/outputs/TPS7A02.raw_structure.v1.json",
    )
    parser.add_argument(
        "--pdf", default="data/samples/TPS7A02.pdf"
    )
    parser.add_argument(
        "--table-id", type=int, default=9
    )
    parser.add_argument(
        "--page-no", type=int, default=6
    )
    parser.add_argument(
        "--output",
        default="data/outputs/TPS7A02.logical_rows.v1.json",
    )

    args = parser.parse_args(argv)

    raw = json.loads(
        Path(args.raw).read_text(encoding="utf-8")
    )

    result = recover_from_pdf(
        raw,
        args.table_id,
        args.pdf,
        args.page_no,
    )

    saved = save_logical_rows(result, args.output)

    print("逻辑行结果已保存:", saved)
    print("逻辑行数量:", result["summary"]["logical_row_count"])
    print("confirmed:", result["summary"]["confirmed_count"])
    print("needs_review:", result["summary"]["needs_review_count"])
    print("recovery_actions:", result["recovery_actions"])
    print("unresolved_issues:", len(result["unresolved_issues"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

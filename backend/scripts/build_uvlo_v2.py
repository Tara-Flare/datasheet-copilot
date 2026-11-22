"""阶段 B3.2b：UVLO 三条人工 confirmed 结果回填。

只读：
- TPS7A02.logical_rows.v1.json
- TPS7A02.raw_structure.v1.json
- TPS7A02.pdf
- gold_standard.table9.json

新增：
- TPS7A02.logical_rows.v2.json
- uvlo_human_review.v1.json
- gold_standard.table9.v2.json
- uvlo_source_mapping.v2.json / .md
- logical_rows.v1_to_v2.diff.json / .md
- gold_standard.table9.v2.completeness.json / .md
"""

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.validation.recovered_evidence import (
    RecoveredEvidenceValidator,
)


DEFAULT_DIR = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
)
DEFAULT_V1 = (
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
DEFAULT_OLD_GOLD = (
    DEFAULT_DIR / "gold_standard.table9.json"
)
DEFAULT_V2 = (
    PROJECT_ROOT
    / "data/outputs/TPS7A02.logical_rows.v2.json"
)
DEFAULT_REVIEW = (
    DEFAULT_DIR / "uvlo_human_review.v1.json"
)
DEFAULT_GOLD_V2 = (
    DEFAULT_DIR / "gold_standard.table9.v2.json"
)
DEFAULT_MAPPING_JSON = (
    DEFAULT_DIR / "uvlo_source_mapping.v2.json"
)
DEFAULT_MAPPING_MD = (
    DEFAULT_DIR / "uvlo_source_mapping.v2.md"
)
DEFAULT_DIFF_JSON = (
    DEFAULT_DIR / "logical_rows.v1_to_v2.diff.json"
)
DEFAULT_DIFF_MD = (
    DEFAULT_DIR / "logical_rows.v1_to_v2.diff.md"
)
DEFAULT_COMPLETENESS_JSON = (
    DEFAULT_DIR / "gold_standard.table9.v2.completeness.json"
)
DEFAULT_COMPLETENESS_MD = (
    DEFAULT_DIR / "gold_standard.table9.v2.completeness.md"
)

TABLE_ID = 9
PAGE = 6

UVLO_CONFIRMATIONS = {
    "t9.p34": {
        "parameter_symbol": "VUVLO",
        "parameter_name": "UVLO threshold",
        "test_conditions": "VIN rising",
        "min": "1.23",
        "typ": "1.3",
        "max": "1.47",
        "unit": "V",
        "source_cell_ids": [132, 133, 134, 135, 136, 137, 138],
        "source_row_indices": [33],
        "primary_row_index": 33,
        "unit_source_cell_id": 138,
        "unit_source_mode": "direct",
        "fragments": [],
    },
    "t9.p35": {
        "parameter_symbol": "VUVLO",
        "parameter_name": "UVLO threshold",
        "test_conditions": "VIN falling",
        "min": "1.0",
        "typ": "1.12",
        "max": "1.41",
        "unit": "V",
        "source_cell_ids": [
            132, 133, 138, 139, 140, 141, 142
        ],
        "source_row_indices": [33, 34],
        "primary_row_index": 34,
        "unit_source_cell_id": 138,
        "unit_source_mode": "shared_with_sibling",
        "fragments": [
            {
                "cell_id": 139,
                "text": "VIN falling",
                "role": "test_conditions",
                "fragment_of": "VIN falling VIN hysteresis",
            },
            {
                "cell_id": 141,
                "text": "1.12",
                "role": "typ",
                "fragment_of": "1.12 180",
            },
        ],
    },
    "t9.p36": {
        "parameter_symbol": "VUVLO(HYST)",
        "parameter_name": "UVLO hysteresis",
        "test_conditions": "VIN hysteresis",
        "min": "",
        "typ": "180",
        "max": "",
        "unit": "mV",
        "source_cell_ids": [139, 141, 143, 144, 145],
        "source_row_indices": [34],
        "primary_row_index": 34,
        "unit_source_cell_id": 145,
        "unit_source_mode": "direct",
        "fragments": [
            {
                "cell_id": 139,
                "text": "VIN hysteresis",
                "role": "test_conditions",
                "fragment_of": "VIN falling VIN hysteresis",
            },
            {
                "cell_id": 141,
                "text": "180",
                "role": "typ",
                "fragment_of": "1.12 180",
            },
        ],
    },
}

EXPECTED_WORDS = {
    "t9.p34": ["VIN", "rising", "1.23", "1.3", "1.47", "V"],
    "t9.p35": ["VIN", "falling", "1.0", "1.12", "1.41", "V"],
    "t9.p36": ["VIN", "hysteresis", "180", "mV"],
}


class UvloBuildError(ValueError):
    """UVLO 修复或 Gold 构建失败。"""


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


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


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
        raise UvloBuildError(
            f"raw_structure 缺少 table_id={table_id}"
        )

    return table, {
        cell.get("cell_id"): cell
        for cell in table.get("cells", [])
    }


def index_logical(logical_rows):
    return {
        row["logical_row_id"]: row
        for row in logical_rows.get("logical_rows", [])
    }


def word_in_band(bbox, band, tolerance=8.0):
    top = float(bbox[1])
    bottom = float(bbox[3])
    center = (top + bottom) / 2.0

    return (
        band["top"] - tolerance
        <= center
        <= band["bottom"] + tolerance
    )


def verify_uvlo_geometry(
    old_rows,
    table,
    cells,
    pdf_path,
):
    """验证 PDF 几何证据是否足以支持三条 UVLO 修复。"""

    document = pymupdf.open(pdf_path)
    evidence = {}

    try:
        page = document[PAGE - 1]

        for row_id, confirmed in UVLO_CONFIRMATIONS.items():
            old = old_rows[row_id]
            band = old["physical_row_band"]

            missing_cells = [
                cell_id
                for cell_id in confirmed["source_cell_ids"]
                if cell_id not in cells
            ]

            if missing_cells:
                raise UvloBuildError(
                    f"{row_id} 缺少 source cells: {missing_cells}"
                )

            for fragment in confirmed["fragments"]:
                cell = cells[fragment["cell_id"]]
                text = str(cell.get("text") or "")

                if fragment["text"] not in text:
                    raise UvloBuildError(
                        f"{row_id} 片段不匹配: "
                        f"cell={fragment['cell_id']} "
                        f"expected={fragment['text']!r} "
                        f"actual={text!r}"
                    )

            words = old.get("source_word_bboxes") or []
            word_map = {}

            for item in words:
                word_map.setdefault(
                    item["text"], []
                ).append(item["bbox"])

            for expected in EXPECTED_WORDS[row_id]:
                bboxes = word_map.get(expected, [])

                if not bboxes:
                    raise UvloBuildError(
                        f"{row_id} 缺少 PDF word: {expected!r}"
                    )

                if not any(
                    word_in_band(bbox, band)
                    for bbox in bboxes
                ):
                    raise UvloBuildError(
                        f"{row_id} word 不在物理行带内: "
                        f"{expected!r}, band={band}"
                    )

            unit_cell = cells[
                confirmed["unit_source_cell_id"]
            ]

            if str(unit_cell.get("text") or "").strip() != (
                confirmed["unit"]
            ):
                raise UvloBuildError(
                    f"{row_id} 单位来源不匹配: "
                    f"{unit_cell.get('text')!r} != "
                    f"{confirmed['unit']!r}"
                )

            pdf_words = page.get_text(
                "words",
                clip=pymupdf.Rect(
                    table["table_bbox"]["l"],
                    band["top"] - 2,
                    table["table_bbox"]["r"],
                    band["bottom"] + 2,
                ),
            )
            pdf_words.sort(
                key=lambda item: (
                    round(item[1], 1),
                    round(item[0], 1),
                )
            )

            evidence[row_id] = {
                "logical_row_id": row_id,
                "page": PAGE,
                "physical_row_band": band,
                "source_row_indices": (
                    confirmed["source_row_indices"]
                ),
                "primary_row_index": (
                    confirmed["primary_row_index"]
                ),
                "source_cell_ids": (
                    confirmed["source_cell_ids"]
                ),
                "source_cells": [
                    {
                        "cell_id": cell_id,
                        "text": cells[cell_id].get("text"),
                        "bbox": cells[cell_id].get("bbox"),
                        "start_row_offset_idx": (
                            cells[cell_id].get(
                                "start_row_offset_idx"
                            )
                        ),
                        "end_row_offset_idx": (
                            cells[cell_id].get(
                                "end_row_offset_idx"
                            )
                        ),
                        "start_col_offset_idx": (
                            cells[cell_id].get(
                                "start_col_offset_idx"
                            )
                        ),
                        "end_col_offset_idx": (
                            cells[cell_id].get(
                                "end_col_offset_idx"
                            )
                        ),
                    }
                    for cell_id in confirmed[
                        "source_cell_ids"
                    ]
                ],
                "source_cell_fragments": (
                    confirmed["fragments"]
                ),
                "unit_source": {
                    "cell_id": confirmed[
                        "unit_source_cell_id"
                    ],
                    "text": confirmed["unit"],
                    "mode": confirmed[
                        "unit_source_mode"
                    ],
                    "shared_with": (
                        "t9.p34"
                        if row_id == "t9.p35"
                        else None
                    ),
                },
                "source_word_bboxes": words,
                "pdf_text_in_band": " ".join(
                    item[4] for item in pdf_words
                ),
                "confirmed_values": {
                    "min": confirmed["min"],
                    "typ": confirmed["typ"],
                    "max": confirmed["max"],
                    "unit": confirmed["unit"],
                    "test_conditions": confirmed[
                        "test_conditions"
                    ],
                },
            }
    finally:
        document.close()

    return evidence


def apply_uvlo_repair(v1, evidence):
    result = copy.deepcopy(v1)
    rows = []

    for old in result["logical_rows"]:
        row_id = old["logical_row_id"]

        if row_id not in UVLO_CONFIRMATIONS:
            rows.append(old)
            continue

        confirmed = UVLO_CONFIRMATIONS[row_id]
        row = copy.deepcopy(old)

        row.update({
            "parameter_symbol": confirmed[
                "parameter_symbol"
            ],
            "parameter_name": confirmed[
                "parameter_name"
            ],
            "test_conditions": confirmed[
                "test_conditions"
            ],
            "conditions": {
                "condition1": confirmed[
                    "test_conditions"
                ],
                "condition2": "",
            },
            "min": confirmed["min"],
            "typ": confirmed["typ"],
            "max": confirmed["max"],
            "unit": confirmed["unit"],
            "source_cell_ids": list(
                confirmed["source_cell_ids"]
            ),
            "source_row_indices": list(
                confirmed["source_row_indices"]
            ),
            "primary_row_index": confirmed[
                "primary_row_index"
            ],
            "needs_review": False,
            "review_status": "confirmed",
            "confidence": 0.95,
            "source_cell_fragments": copy.deepcopy(
                confirmed["fragments"]
            ),
            "unit_source": {
                "cell_id": confirmed[
                    "unit_source_cell_id"
                ],
                "text": confirmed["unit"],
                "mode": confirmed[
                    "unit_source_mode"
                ],
                "shared_with": (
                    "t9.p34"
                    if row_id == "t9.p35"
                    else None
                ),
            },
            "human_review": {
                "status": "confirmed",
                "reviewer": "human",
                "source": (
                    "TPS7A02.pdf page 6 "
                    "Electrical Characteristics"
                ),
                "confirmed_values": {
                    "min": confirmed["min"],
                    "typ": confirmed["typ"],
                    "max": confirmed["max"],
                    "unit": confirmed["unit"],
                    "test_conditions": confirmed[
                        "test_conditions"
                    ],
                },
            },
        })

        actions = list(row.get("recovery_actions") or [])

        for action in (
            "uvlo_geometry_repair",
            "human_confirmed_pdf_page6",
        ):
            if action not in actions:
                actions.append(action)

        row["recovery_actions"] = actions
        rows.append(row)

    result["logical_rows"] = rows
    result["logical_rows_revision"] = (
        "v2-uvlo-confirmed"
    )
    result["supersedes"] = (
        "TPS7A02.logical_rows.v1.json"
    )
    result["review"] = {
        "status": "confirmed",
        "reviewer": "human",
        "source_pdf": str(DEFAULT_PDF),
        "page": PAGE,
        "confirmed_logical_row_ids": sorted(
            UVLO_CONFIRMATIONS
        ),
        "historical_v1_needs_review_preserved_in_v1": True,
    }
    result["summary"] = {
        "logical_row_count": len(rows),
        "confirmed_count": sum(
            1 for row in rows
            if not row.get("needs_review")
        ),
        "needs_review_count": sum(
            1 for row in rows
            if row.get("needs_review")
        ),
    }
    result["unresolved_issues"] = []
    result["recovery_actions"] = dict(
        result.get("recovery_actions") or {}
    )
    result["recovery_actions"][
        "uvlo_geometry_repair"
    ] = 3
    result["recovery_actions"][
        "human_confirmed_pdf_page6"
    ] = 3

    return result


def build_review_document(evidence):
    return {
        "schema": "uvlo_human_review/v1",
        "version": 1,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "pdf": str(DEFAULT_PDF),
            "page": PAGE,
            "table_id": TABLE_ID,
            "reviewer": "human",
        },
        "status": "confirmed",
        "records": [
            {
                "logical_row_id": row_id,
                "parameter_symbol": UVLO_CONFIRMATIONS[
                    row_id
                ]["parameter_symbol"],
                "parameter_name": UVLO_CONFIRMATIONS[
                    row_id
                ]["parameter_name"],
                "test_conditions": UVLO_CONFIRMATIONS[
                    row_id
                ]["test_conditions"],
                "min": UVLO_CONFIRMATIONS[row_id]["min"],
                "typ": UVLO_CONFIRMATIONS[row_id]["typ"],
                "max": UVLO_CONFIRMATIONS[row_id]["max"],
                "unit": UVLO_CONFIRMATIONS[row_id]["unit"],
                "source_row_indices": evidence[
                    row_id
                ]["source_row_indices"],
                "primary_row_index": evidence[
                    row_id
                ]["primary_row_index"],
                "source_cell_ids": evidence[
                    row_id
                ]["source_cell_ids"],
                "source_cell_fragments": evidence[
                    row_id
                ]["source_cell_fragments"],
                "unit_source": evidence[row_id][
                    "unit_source"
                ],
                "review_status": "confirmed",
            }
            for row_id in sorted(UVLO_CONFIRMATIONS)
        ],
        "notes": [
            "旧 v1 中的 needs_review=true 仍然保留在 v1。",
            "本文件只记录人工核验结果，不修改旧产物。",
        ],
    }


def build_gold_v2(
    old_gold,
    v2,
    raw_structure,
    evidence,
    review_doc,
    hashes,
):
    params = copy.deepcopy(
        old_gold["expected_parameters"]
    )

    for record in params:
        record.setdefault(
            "review_status", "confirmed"
        )

    rows = index_logical(v2)

    for row_id in (
        "t9.p34", "t9.p35", "t9.p36"
    ):
        row = rows[row_id]

        for value_type in ("min", "typ", "max"):
            value = str(
                row.get(value_type) or ""
            ).strip()

            if not value:
                continue

            params.append({
                "logical_row_id": row_id,
                "table_id": TABLE_ID,
                "page": PAGE,
                "parameter_symbol": row.get(
                    "parameter_symbol"
                ),
                "parameter_name": row.get(
                    "parameter_name"
                ),
                "value": value,
                "value_type": value_type,
                "unit": row.get("unit"),
                "test_conditions": row.get(
                    "test_conditions"
                ),
                "source_row_indices": list(
                    row.get("source_row_indices") or []
                ),
                "source_cell_ids": list(
                    row.get("source_cell_ids") or []
                ),
                "primary_row_index": row.get(
                    "primary_row_index"
                ),
                "source_cell_fragments": copy.deepcopy(
                    row.get("source_cell_fragments") or []
                ),
                "unit_source": copy.deepcopy(
                    row.get("unit_source")
                ),
                "needs_review": False,
                "review_status": "confirmed",
                "human_review": row.get(
                    "human_review"
                ),
            })

    return {
        "schema": "ab_gold_standard/v2",
        "version": 2,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "table_id": TABLE_ID,
        "source": {
            "pdf_path": str(DEFAULT_PDF),
            "pdf_sha256": hashes["pdf"],
            "raw_structure_path": str(DEFAULT_RAW),
            "raw_structure_sha256": hashes["raw"],
            "logical_rows_v1_path": str(DEFAULT_V1),
            "logical_rows_v1_sha256": hashes["v1"],
            "logical_rows_v2_path": str(DEFAULT_V2),
            "logical_rows_v2_sha256": hashes["v2"],
            "old_gold_path": str(DEFAULT_OLD_GOLD),
            "old_gold_sha256": hashes["old_gold"],
            "uvlo_human_review_path": str(DEFAULT_REVIEW),
            "uvlo_human_review_sha256": hashes[
                "review"
            ],
        },
        "logical_rows_revision": "v2-uvlo-confirmed",
        "trusted_logical_row_count": len(v2["logical_rows"]),
        "expected_parameter_count": len(params),
        "status_summary": {
            "logical_rows": len(v2["logical_rows"]),
            "confirmed_logical_rows": v2["summary"][
                "confirmed_count"
            ],
            "needs_review_logical_rows": v2["summary"][
                "needs_review_count"
            ],
            "gold_parameter_records": len(params),
            "confirmed_gold_records": sum(
                1 for item in params
                if item.get("review_status")
                == "confirmed"
            ),
        },
        "expected_parameters": params,
        "excluded_needs_review": [],
        "resolved_needs_review": [
            {
                "logical_row_id": row_id,
                "old_status": "needs_review",
                "new_status": "confirmed",
                "human_review": review_doc[
                    "records"
                ][index],
            }
            for index, row_id in enumerate(
                sorted(UVLO_CONFIRMATIONS)
            )
        ],
        "notes": [
            "v2 保留原始 44 条 Gold 记录，并新增 7 条 UVLO 规格。",
            "原始 gold_standard.table9.json 未被修改。",
            "所有 UVLO 值来自人工核验 PDF 第 6 页。",
        ],
    }


def check_gold_completeness(gold, v2):
    rows = index_logical(v2)
    entries = gold["expected_parameters"]
    keys = [
        (item["logical_row_id"], item["value_type"])
        for item in entries
    ]
    duplicate_keys = sorted({
        key for key in keys if keys.count(key) > 1
    })

    expected_keys = set()

    for row_id, row in rows.items():
        for value_type in ("min", "typ", "max"):
            if str(row.get(value_type) or "").strip():
                expected_keys.add((row_id, value_type))

    actual_keys = set(keys)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)

    condition_mismatches = []
    unit_mismatches = []
    value_mismatches = []
    order_violations = []

    for item in entries:
        row = rows[item["logical_row_id"]]
        value_type = item["value_type"]

        if item["test_conditions"] != row.get(
            "test_conditions"
        ):
            condition_mismatches.append({
                "logical_row_id": item["logical_row_id"],
                "value_type": value_type,
                "gold": item["test_conditions"],
                "logical": row.get(
                    "test_conditions"
                ),
            })

        if item["unit"] != row.get("unit"):
            unit_mismatches.append({
                "logical_row_id": item["logical_row_id"],
                "value_type": value_type,
                "gold": item["unit"],
                "logical": row.get("unit"),
            })

        if str(item["value"]).strip() != str(
            row.get(value_type)
        ).strip():
            value_mismatches.append({
                "logical_row_id": item["logical_row_id"],
                "value_type": value_type,
                "gold": item["value"],
                "logical": row.get(value_type),
            })

    for row_id, row in rows.items():
        values = {}

        for value_type in ("min", "typ", "max"):
            raw = str(row.get(value_type) or "").strip()

            if not raw:
                continue

            try:
                values[value_type] = Decimal(
                    raw.replace("–", "-")
                )
            except (InvalidOperation, ValueError):
                pass

        if "min" in values and "typ" in values:
            if values["min"] > values["typ"]:
                order_violations.append({
                    "logical_row_id": row_id,
                    "reason": "min > typ",
                })

        if "typ" in values and "max" in values:
            if values["typ"] > values["max"]:
                order_violations.append({
                    "logical_row_id": row_id,
                    "reason": "typ > max",
                })

        if "min" in values and "max" in values:
            if values["min"] > values["max"]:
                order_violations.append({
                    "logical_row_id": row_id,
                    "reason": "min > max",
                })

    counts_by_row = Counter(
        item["logical_row_id"] for item in entries
    )
    empty_condition_ids = sorted({
        item["logical_row_id"]
        for item in entries
        if not str(
            item.get("test_conditions") or ""
        ).strip()
    })

    ok = not any((
        duplicate_keys,
        missing,
        extra,
        condition_mismatches,
        unit_mismatches,
        value_mismatches,
        order_violations,
        empty_condition_ids,
    ))

    return {
        "ok": ok,
        "logical_row_count": len(rows),
        "gold_parameter_count": len(entries),
        "uvlo_parameter_count": sum(
            count for row_id, count in counts_by_row.items()
            if row_id in UVLO_CONFIRMATIONS
        ),
        "uvlo_counts_by_row": {
            row_id: counts_by_row.get(row_id, 0)
            for row_id in sorted(UVLO_CONFIRMATIONS)
        },
        "duplicate_keys": duplicate_keys,
        "missing_keys": missing,
        "extra_keys": extra,
        "condition_mismatches": condition_mismatches,
        "unit_mismatches": unit_mismatches,
        "value_mismatches": value_mismatches,
        "min_typ_max_violations": order_violations,
        "empty_condition_ids": empty_condition_ids,
        "records_by_logical_row": dict(
            sorted(counts_by_row.items())
        ),
    }


def validate_seven_uvlo_records(
    v2,
    raw_structure,
    gold,
):
    uvlo_params = [
        item for item in gold["expected_parameters"]
        if item["logical_row_id"] in UVLO_CONFIRMATIONS
    ]

    parameters = []

    for item in uvlo_params:
        name = (
            "uvlo_hysteresis"
            if "hysteresis" in item[
                "parameter_name"
            ].lower()
            else "uvlo_threshold"
        )
        parameters.append({
            "name": name,
            "value": float(
                str(item["value"]).replace(
                    "–", "-"
                )
            ),
            "unit": item["unit"],
            "value_type": item["value_type"],
            "test_conditions": item[
                "test_conditions"
            ],
            "evidence": {
                "table_id": item["table_id"],
                "row_index": item[
                    "primary_row_index"
                ],
                "page": item["page"],
                "source_text": (
                    f"{item['parameter_name']} "
                    f"{item['test_conditions']}"
                ),
                "logical_row_id": item[
                    "logical_row_id"
                ],
                "source_row_indices": item[
                    "source_row_indices"
                ],
                "source_cell_ids": item[
                    "source_cell_ids"
                ],
            },
        })

    extraction = {
        "component_model": "TPS7A02",
        "manufacturer": "TI",
        "parameters": parameters,
    }
    validator = RecoveredEvidenceValidator(
        v2,
        raw_structure=raw_structure,
    )
    report = validator.validate(extraction)

    results = []

    for item, result in zip(
        uvlo_params, report["parameters"]
    ):
        results.append({
            "logical_row_id": item["logical_row_id"],
            "value_type": item["value_type"],
            "value": item["value"],
            "unit": item["unit"],
            "selected_logical_row_id": result[
                "selected_logical_row_id"
            ],
            "recovered_status": result[
                "recovered_status"
            ],
            "reason": result.get("reason"),
            "conflicts": result.get("conflicts", []),
        })

    return {
        "record_count": len(results),
        "validated_count": sum(
            1 for item in results
            if item["recovered_status"]
            == "validated"
        ),
        "all_validated": all(
            item["recovered_status"] == "validated"
            for item in results
        ),
        "records": results,
    }


def build_diff(
    v1,
    v2,
    evidence,
    mapping,
    completeness,
    strict,
):
    old_rows = index_logical(v1)
    new_rows = index_logical(v2)
    changed = []

    fields = [
        "min",
        "typ",
        "max",
        "unit",
        "test_conditions",
        "source_cell_ids",
        "source_row_indices",
        "primary_row_index",
        "needs_review",
        "review_status",
        "source_cell_fragments",
        "unit_source",
        "confidence",
    ]

    for row_id in sorted(
        set(old_rows) | set(new_rows)
    ):
        old = old_rows.get(row_id)
        new = new_rows.get(row_id)

        if old is None or new is None:
            changed.append({
                "logical_row_id": row_id,
                "change": "added" if old is None else "removed",
            })
            continue

        field_changes = {}

        for field in fields:
            if old.get(field) != new.get(field):
                field_changes[field] = {
                    "before": old.get(field),
                    "after": new.get(field),
                }

        if field_changes:
            changed.append({
                "logical_row_id": row_id,
                "parameter_name": new.get(
                    "parameter_name"
                ),
                "changed_fields": field_changes,
            })

    return {
        "schema": "logical_rows_diff/v1",
        "table_id": TABLE_ID,
        "summary": {
            "v1_logical_rows": len(v1["logical_rows"]),
            "v2_logical_rows": len(v2["logical_rows"]),
            "v1_confirmed": v1["summary"][
                "confirmed_count"
            ],
            "v2_confirmed": v2["summary"][
                "confirmed_count"
            ],
            "v1_needs_review": v1["summary"][
                "needs_review_count"
            ],
            "v2_needs_review": v2["summary"][
                "needs_review_count"
            ],
            "v1_unresolved_issues": len(
                v1.get("unresolved_issues", [])
            ),
            "v2_unresolved_issues": len(
                v2.get("unresolved_issues", [])
            ),
            "changed_logical_rows": [
                item["logical_row_id"]
                for item in changed
            ],
        },
        "changed_rows": changed,
        "uvlo_source_evidence": evidence,
        "uvlo_source_mapping": mapping,
        "gold_completeness": completeness,
        "strict_uvlo_validation": strict,
    }


def build_mapping_markdown(evidence):
    lines = [
        "# TPS7A02 UVLO 三条来源映射证据（v2）",
        "",
        "来源：TPS7A02.pdf 第 6 页 Electrical Characteristics。",
        "",
    ]

    for row_id in sorted(evidence):
        item = evidence[row_id]
        lines += [
            f"## {row_id}",
            "",
            f"- 参数：{UVLO_CONFIRMATIONS[row_id]['parameter_name']}",
            f"- 条件：{UVLO_CONFIRMATIONS[row_id]['test_conditions']}",
            f"- Min/Typ/Max：{UVLO_CONFIRMATIONS[row_id]['min'] or '—'} / "
            f"{UVLO_CONFIRMATIONS[row_id]['typ'] or '—'} / "
            f"{UVLO_CONFIRMATIONS[row_id]['max'] or '—'} "
            f"{UVLO_CONFIRMATIONS[row_id]['unit']}",
            f"- primary_row_index：{item['primary_row_index']}",
            f"- source_row_indices：{item['source_row_indices']}",
            f"- source_cell_ids：{item['source_cell_ids']}",
            f"- 单位来源：cell {item['unit_source']['cell_id']} "
            f"`{item['unit_source']['text']}` "
            f"({item['unit_source']['mode']})",
            f"- PDF 区域文字：`{item['pdf_text_in_band']}`",
            "",
            "| cell_id | text | row span | col span | bbox(top-left) |",
            "|---:|---|---|---|---|",
        ]

        for cell in item["source_cells"]:
            bbox = cell.get("bbox") or {}
            text = str(cell["text"]).replace(
                "|", "\\|"
            )
            lines.append(
                f"| {cell['cell_id']} | {text} | "
                f"{cell.get('start_row_offset_idx')}–{cell.get('end_row_offset_idx')} | "
                f"{cell.get('start_col_offset_idx')}–{cell.get('end_col_offset_idx')} | "
                f"({bbox.get('l')}, {bbox.get('t')})–"
                f"({bbox.get('r')}, {bbox.get('b')}) |"
            )

        if item["source_cell_fragments"]:
            lines += [
                "",
                "片段来源：",
            ]
            for fragment in item["source_cell_fragments"]:
                lines.append(
                    f"- cell {fragment['cell_id']} "
                    f"`{fragment['fragment_of']}` → "
                    f"`{fragment['text']}` "
                    f"({fragment['role']})"
                )

        lines.append("")

    return "\n".join(lines)


def build_diff_markdown(diff):
    summary = diff["summary"]
    lines = [
        "# Logical Rows v1 → v2 差异",
        "",
        "| 项目 | v1 | v2 |",
        "|---|---:|---:|",
        f"| 逻辑行 | {summary['v1_logical_rows']} | {summary['v2_logical_rows']} |",
        f"| confirmed | {summary['v1_confirmed']} | {summary['v2_confirmed']} |",
        f"| needs_review | {summary['v1_needs_review']} | {summary['v2_needs_review']} |",
        f"| unresolved_issues | {summary['v1_unresolved_issues']} | {summary['v2_unresolved_issues']} |",
        "",
        "## 变更行",
        "",
        "| logical_row_id | 变更字段 |",
        "|---|---|",
    ]

    for item in diff["changed_rows"]:
        if "changed_fields" not in item:
            continue

        fields = ", ".join(
            sorted(item["changed_fields"])
        )
        lines.append(
            f"| {item['logical_row_id']} | {fields} |"
        )

    lines += [
        "",
        "## UVLO 严格校验",
        "",
        f"- 待校验记录：{diff['strict_uvlo_validation']['record_count']}",
        f"- validated：{diff['strict_uvlo_validation']['validated_count']}",
        f"- 全部通过：{diff['strict_uvlo_validation']['all_validated']}",
        "",
        "## Gold 完整性",
        "",
        f"- 逻辑行：{diff['gold_completeness']['logical_row_count']}",
        f"- Gold 规格：{diff['gold_completeness']['gold_parameter_count']}",
        f"- UVLO 规格：{diff['gold_completeness']['uvlo_parameter_count']}",
        f"- 完整：{diff['gold_completeness']['ok']}",
        "",
    ]

    return "\n".join(lines)


def build_completeness_markdown(completeness):
    lines = [
        "# Gold Standard v2 完整性检查",
        "",
        f"- 结果：{'PASS' if completeness['ok'] else 'FAIL'}",
        f"- 逻辑行：{completeness['logical_row_count']}",
        f"- Gold 规格记录：{completeness['gold_parameter_count']}",
        f"- UVLO 规格记录：{completeness['uvlo_parameter_count']}",
        f"- UVLO 分布：{completeness['uvlo_counts_by_row']}",
        f"- 重复键：{completeness['duplicate_keys']}",
        f"- 遗漏键：{completeness['missing_keys']}",
        f"- 多余键：{completeness['extra_keys']}",
        f"- 条件不一致：{len(completeness['condition_mismatches'])}",
        f"- 单位不一致：{len(completeness['unit_mismatches'])}",
        f"- 数值不一致：{len(completeness['value_mismatches'])}",
        f"- Min/Typ/Max 顺序冲突：{completeness['min_typ_max_violations']}",
        f"- 空测试条件：{completeness['empty_condition_ids']}",
        "",
    ]

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="生成 UVLO v2 Logical Rows 与 51 条 Gold"
    )
    parser.add_argument("--v1", default=str(DEFAULT_V1))
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument(
        "--old-gold", default=str(DEFAULT_OLD_GOLD)
    )
    parser.add_argument(
        "--logical-v2", default=str(DEFAULT_V2)
    )
    parser.add_argument(
        "--review", default=str(DEFAULT_REVIEW)
    )
    parser.add_argument(
        "--gold-v2", default=str(DEFAULT_GOLD_V2)
    )
    parser.add_argument(
        "--mapping-json",
        default=str(DEFAULT_MAPPING_JSON),
    )
    parser.add_argument(
        "--mapping-md",
        default=str(DEFAULT_MAPPING_MD),
    )
    parser.add_argument(
        "--diff-json", default=str(DEFAULT_DIFF_JSON)
    )
    parser.add_argument(
        "--diff-md", default=str(DEFAULT_DIFF_MD)
    )
    parser.add_argument(
        "--completeness-json",
        default=str(DEFAULT_COMPLETENESS_JSON),
    )
    parser.add_argument(
        "--completeness-md",
        default=str(DEFAULT_COMPLETENESS_MD),
    )
    args = parser.parse_args(argv)

    paths = {
        "v1": Path(args.v1),
        "raw": Path(args.raw),
        "pdf": Path(args.pdf),
        "old_gold": Path(args.old_gold),
    }

    for name, path in paths.items():
        if not path.is_file():
            raise UvloBuildError(
                f"{name} 文件不存在: {path}"
            )

    hashes_before = {
        name: sha256_file(path)
        for name, path in paths.items()
    }

    v1 = load_json(args.v1)
    raw = load_json(args.raw)
    old_gold = load_json(args.old_gold)

    pdf_sha = sha256_file(args.pdf)
    expected_pdf_sha = (
        raw.get("source") or {}
    ).get("pdf_sha256")

    if expected_pdf_sha and expected_pdf_sha != pdf_sha:
        raise UvloBuildError(
            "PDF SHA256 与 raw_structure 不一致"
        )

    table, cells = index_cells(raw, TABLE_ID)
    old_rows = index_logical(v1)

    evidence = verify_uvlo_geometry(
        old_rows,
        table,
        cells,
        args.pdf,
    )
    review_doc = build_review_document(evidence)
    write_json(args.review, review_doc)
    review_sha = sha256_file(args.review)

    v2 = apply_uvlo_repair(v1, evidence)
    write_json(args.logical_v2, v2)
    v2_sha = sha256_file(args.logical_v2)

    hashes = {
        **hashes_before,
        "review": review_sha,
        "v2": v2_sha,
    }

    gold_v2 = build_gold_v2(
        old_gold,
        v2,
        raw,
        evidence,
        review_doc,
        hashes,
    )
    completeness = check_gold_completeness(
        gold_v2, v2
    )
    strict = validate_seven_uvlo_records(
        v2, raw, gold_v2
    )

    if not completeness["ok"]:
        raise UvloBuildError(
            "Gold v2 完整性检查失败: "
            f"{completeness}"
        )

    if not strict["all_validated"]:
        raise UvloBuildError(
            "UVLO 严格校验未全部通过: "
            f"{strict}"
        )

    write_json(args.gold_v2, gold_v2)

    mapping = {
        "schema": "uvlo_source_mapping/v2",
        "version": 2,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "pdf_path": str(args.pdf),
        "pdf_sha256": pdf_sha,
        "page": PAGE,
        "records": evidence,
        "strict_validation": strict,
    }
    write_json(args.mapping_json, mapping)
    Path(args.mapping_md).write_text(
        build_mapping_markdown(evidence),
        encoding="utf-8",
    )

    diff = build_diff(
        v1,
        v2,
        evidence,
        mapping,
        completeness,
        strict,
    )
    write_json(args.diff_json, diff)
    Path(args.diff_md).write_text(
        build_diff_markdown(diff),
        encoding="utf-8",
    )

    write_json(args.completeness_json, completeness)
    Path(args.completeness_md).write_text(
        build_completeness_markdown(completeness),
        encoding="utf-8",
    )

    hashes_after = {
        name: sha256_file(path)
        for name, path in paths.items()
    }

    if hashes_before != hashes_after:
        raise UvloBuildError(
            "原始输入在生成过程中被修改"
        )

    print("=" * 70)
    print("UVLO v2 Logical Rows / Gold Standard 已生成")
    print("=" * 70)
    print(
        "v1:",
        v1["summary"],
    )
    print(
        "v2:",
        v2["summary"],
    )
    print(
        "Gold v2 规格:",
        gold_v2["expected_parameter_count"],
        "UVLO:",
        completeness["uvlo_parameter_count"],
    )
    print(
        "完整性检查:",
        completeness["ok"],
    )
    print(
        "UVLO 严格校验:",
        strict["validated_count"],
        "/",
        strict["record_count"],
        "all_validated=",
        strict["all_validated"],
    )
    print("Logical v2:", args.logical_v2)
    print("Gold v2:", args.gold_v2)
    print("Human review:", args.review)
    print("Source mapping:", args.mapping_json)
    print("Diff:", args.diff_json)
    print("Completeness:", args.completeness_json)
    print("原始文件 SHA256 未变化:", hashes_before)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

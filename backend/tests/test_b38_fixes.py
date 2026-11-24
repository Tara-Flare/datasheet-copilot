"""B3.8 定向修复的最小复现测试。

覆盖：
- 单位白名单中的 %/A、%/V、uVRMS；
- Flat 合并单位单元格与原文引用的证据对齐；
- shutdown/standby 身份限定不一致；
- Validator 身份同义词与同名参数歧义；
- Flat 表格 PDF 字形恢复。

不调用 LLM。
"""

import copy
import json
from pathlib import Path

import pymupdf as fitz
import pytest

from app.extraction.evidence import (
    canonical_unit_cell,
    verify_single_parameter,
)
from app.parsers.structure_recovery import (
    extract_page_words,
    restore_flat_table_glyphs,
)
from app.validation.recovered_evidence import (
    _identity_match,
    logical_family,
    original_family,
)
from app.validation.units import (
    convert_to_base,
    same_physical_value,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
B35C_RUN = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5c/runs/flat_baseline_r1"
)
RAW_DIR = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5d/structure/raw"
)
PDF_DIR = PROJECT_ROOT / "data/samples/b35b"


def test_percentage_per_unit_units_are_supported():
    assert convert_to_base(0.2, "%/A").dimension == (
        "percentage_per_current"
    )
    assert convert_to_base(0.2, "%/V").dimension == (
        "percentage_per_voltage"
    )
    assert convert_to_base(0.2, "%/W").dimension == (
        "percentage_per_power"
    )
    assert convert_to_base(1.0, "%").dimension == "percentage"


def test_rms_voltage_units_are_equivalent():
    assert same_physical_value(85, "µVrms", 85, "uVRMS")
    assert same_physical_value(85, "mVrms", 0.085, "Vrms")
    assert not same_physical_value(85, "mVrms", 85, "µVrms")


def test_canonical_unit_cell_collapses_repeated_units():
    assert canonical_unit_cell("µA µA µA") == "uA"
    assert canonical_unit_cell("mA mA") == "mA"
    assert canonical_unit_cell("mA µA") == "mA uA"


def test_flat_merged_unit_and_quote_are_accepted():
    columns = [
        "Parameter",
        "Test Conditions",
        "Min",
        "Typ",
        "Max",
        "Unit",
    ]
    rows = [[
        "Quiescent Current",
        "VIN = 6V",
        "",
        "0.3",
        "1",
        "µA µA µA",
    ]]
    param = {
        "name": "quiescent_current",
        "value": 0.3,
        "unit": "µA",
        "value_type": "typ",
        "evidence": {
            "table_id": 4,
            "row_index": 0,
            "page": 4,
            "source_text": (
                "Quiescent Current VIN = 6V 0.3 1 "
                "µA µA µA"
            ),
        },
    }

    status, issues = verify_single_parameter(
        param,
        {4: {
            "table_id": 4,
            "page_numbers": [4],
            "columns": columns,
            "rows": rows,
        }},
    )

    assert status == "validated"
    assert issues == []


def test_quiescent_current_from_shutdown_row_is_rejected():
    columns = [
        "Symbol",
        "Parameter",
        "Test Conditions",
        "Min",
        "Typ",
        "Max",
        "Unit",
    ]
    rows = [[
        "IQ",
        "Quiescent Current in Shutdown",
        "VIN = 6V",
        "",
        "0.3",
        "1",
        "µA",
    ]]
    param = {
        "name": "quiescent_current",
        "value": 0.3,
        "unit": "µA",
        "value_type": "typ",
        "evidence": {
            "table_id": 4,
            "row_index": 0,
            "page": 4,
            "source_text": (
                "Quiescent Current in Shutdown VIN = 6V "
                "0.3 1 µA"
            ),
        },
    }

    status, issues = verify_single_parameter(
        param,
        {4: {
            "table_id": 4,
            "page_numbers": [4],
            "columns": columns,
            "rows": rows,
        }},
    )

    assert status == "invalid"
    assert any("shutdown/standby" in issue for issue in issues)


def test_shutdown_current_from_shutdown_row_is_accepted():
    columns = [
        "Symbol",
        "Parameter",
        "Test Conditions",
        "Min",
        "Typ",
        "Max",
        "Unit",
    ]
    rows = [[
        "IQ",
        "Quiescent Current in Shutdown",
        "VIN = 6V",
        "",
        "0.3",
        "1",
        "µA",
    ]]
    param = {
        "name": "shutdown_current",
        "value": 0.3,
        "unit": "µA",
        "value_type": "typ",
        "evidence": {
            "table_id": 4,
            "row_index": 0,
            "page": 4,
            "source_text": (
                "Quiescent Current in Shutdown VIN = 6V "
                "0.3 1 µA"
            ),
        },
    }

    status, issues = verify_single_parameter(
        param,
        {4: {
            "table_id": 4,
            "page_numbers": [4],
            "columns": columns,
            "rows": rows,
        }},
    )

    assert status == "validated"
    assert issues == []


@pytest.mark.parametrize(
    "prediction,row,expected",
    [
        (
            "shutdown_current",
            {
                "parameter_symbol": "IQ",
                "parameter_name": "Quiescent Current in Shutdown",
            },
            "shutdown_current",
        ),
        (
            "ground_current",
            {
                "parameter_symbol": "IQ",
                "parameter_name": "Quiescent Current",
            },
            "ground_current",
        ),
        (
            "short_circuit_current_limit",
            {
                "parameter_symbol": "ISC",
                "parameter_name": "Short Current Limit",
            },
            "short_circuit_current_limit",
        ),
        (
            "output_voltage_noise",
            {
                "parameter_symbol": "VN",
                "parameter_name": "RMS Output Noise",
            },
            "output_noise",
        ),
    ],
)
def test_identity_synonyms_match(
    prediction, row, expected
):
    assert original_family(prediction) == expected
    assert logical_family(row) == expected
    assert _identity_match(
        {"name": prediction}, row
    )["status"] == "match"


def test_scale_factor_row_prefers_symbol_over_container_name():
    row = {
        "parameter_symbol": "Programming Scale Factor",
        "parameter_name": "Programmable Current Limit",
    }

    assert logical_family(row) == (
        "current_limit_programming_scale_factor"
    )
    assert original_family(
        "programming_scale_factor"
    ) == "current_limit_programming_scale_factor"


@pytest.mark.parametrize(
    "prediction,row",
    [
        (
            "output_current_limit",
            {
                "parameter_symbol": "Internal Current Limit (Note 12)",
                "parameter_name": "",
            },
        ),
        (
            "output_current_limit",
            {
                "parameter_symbol": "Programmable Current Limit",
                "parameter_name": "",
            },
        ),
    ],
)
def test_generic_current_limit_stays_ambiguous(prediction, row):
    result = _identity_match({"name": prediction}, row)

    assert result["status"] == "conflict"
    assert result["ambiguous"] is True
    assert result["ambiguous_candidates"] == [
        "internal_current_limit",
        "programmable_current_limit",
    ]


def _raw_table():
    return {
        "table_id": 4,
        "header_row_count": 1,
        "cells": [{
            "cell_id": 9,
            "text": "mA",
            "page_no": 1,
            "start_row_offset_idx": 1,
            "end_row_offset_idx": 2,
            "start_col_offset_idx": 1,
            "end_col_offset_idx": 2,
            "bbox": {
                "l": 10.0,
                "t": 20.0,
                "r": 20.0,
                "b": 28.0,
                "coord_origin": "TOPLEFT",
            },
        }],
    }


def _words(*, glyph=True):
    word = {
        "text": "µA" if glyph else "mA",
        "bbox": (10.0, 20.0, 20.0, 28.0),
    }

    if glyph:
        word["glyph_restored"] = "µ"
        word["glyph_bbox"] = [10.0, 20.0, 11.0, 28.0]

    return [word]


def test_flat_glyph_restoration_changes_only_glyph_cells():
    table = {
        "table_id": 4,
        "columns": ["Symbol", "Unit"],
        "rows": [["IQ", "mA"]],
    }

    restored, evidence = restore_flat_table_glyphs(
        table,
        _raw_table(),
        words_by_page={1: _words()},
        page_heights={1: 100.0},
    )

    assert restored["rows"] == [["IQ", "µA"]]
    assert table["rows"] == [["IQ", "mA"]]
    assert evidence[0]["docling_text"] == "mA"
    assert evidence[0]["pdf_text"] == "µA"


def test_flat_glyph_restoration_requires_glyph_evidence():
    table = {
        "table_id": 4,
        "columns": ["Symbol", "Unit"],
        "rows": [["IQ", "mA"]],
    }

    restored, evidence = restore_flat_table_glyphs(
        table,
        _raw_table(),
        words_by_page={1: _words(glyph=False)},
        page_heights={1: 100.0},
    )

    assert restored["rows"] == [["IQ", "mA"]]
    assert evidence == []


@pytest.mark.requires_data
def test_ncp170_flat_units_use_pdf_glyph_evidence():
    table_path = (
        B35C_RUN / "onsemi_ncp170/selected_tables.json"
    )
    raw_path = RAW_DIR / "onsemi_ncp170.raw_structure.v1.json"
    pdf_path = PDF_DIR / "onsemi_ncp170.pdf"

    selected = json.loads(table_path.read_text(encoding="utf-8"))
    raw_structure = json.loads(
        raw_path.read_text(encoding="utf-8")
    )
    flat_table = next(
        table
        for table in selected["tables"]
        if table["table_id"] == 4
    )
    raw_table = next(
        table
        for table in raw_structure["tables"]
        if table["table_id"] == 4
    )
    document = fitz.open(pdf_path)
    page = document[2]
    words_by_page = {3: extract_page_words(page)}
    page_heights = {3: float(page.rect.height)}
    document.close()

    restored, evidence = restore_flat_table_glyphs(
        copy.deepcopy(flat_table),
        raw_table,
        words_by_page=words_by_page,
        page_heights=page_heights,
    )

    assert restored["rows"][8][6] == "µA"
    assert restored["rows"][9][6] == "µA"
    assert restored["rows"][14][6] == "µVrms"
    assert "1 µF" in restored["rows"][14][2]
    # 防回退：Output Current / Short Circuit Current Limit 行的 mA
    # 在 PDF 中本来就是 mA，不能被字形恢复误改成 µA。
    assert restored["rows"][6][6] == "mA"
    assert restored["rows"][6][3] == "150"
    assert restored["rows"][7][6] == "mA"
    assert restored["rows"][7][4] == "225"
    assert any(
        entry["docling_text"] == "mA"
        and entry["pdf_text"] == "µA"
        for entry in evidence
    )
    assert flat_table["rows"][8][6] == "mA"

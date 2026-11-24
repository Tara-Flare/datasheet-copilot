import hashlib
from pathlib import Path

import pytest

from app import pipeline


SCHEMA = "table_raw_structure/v1"


def _fake_pdf(path):
    return {
        "filename": Path(path).name,
        "page_count": 2,
        "pages": [{"page": 1, "text": "", "blocks": []}],
        "status": "parsed",
    }


def _table(table_id, page=1, row_count=1):
    return {
        "table_id": table_id,
        "page_numbers": [page],
        "columns": ["Parameter", "Min", "Typ", "Max", "Unit"],
        "rows": [["Vout", "1", "2", "3", "V"]] * row_count,
        "row_count": row_count,
        "column_count": 5,
    }


def _raw_structure(pdf_path):
    sha = hashlib.sha256(
        Path(pdf_path).read_bytes()
    ).hexdigest()

    return {
        "schema": SCHEMA,
        "raw_structure_version": 1,
        "source": {"pdf_sha256": sha},
        "table_count": 2,
        "tables": [
            {"table_id": 4, "cells": []},
            {"table_id": 5, "cells": []},
        ],
    }


class FakePDFParser:
    def parse(self, pdf_path):
        return _fake_pdf(pdf_path)


class FakeTableParser:
    def __init__(self, tables):
        self.tables = tables

    def parse(self, pdf_path, include_raw_structure=False):
        payload = {
            "filename": Path(pdf_path).name,
            "table_count": len(self.tables),
            "tables": self.tables,
            "status": "parsed",
        }
        if include_raw_structure:
            payload["raw_structure"] = _raw_structure(pdf_path)
        return payload


class FakeSelector:
    def __init__(self, table_ids=None):
        self.table_ids = table_ids

    def select(self, tables):
        if self.table_ids is None:
            return list(tables)
        return [t for t in tables if t["table_id"] in self.table_ids]


def _recovered(table_id, page=1):
    return {
        "schema": "table_logical_rows/v1",
        "logical_rows_version": 1,
        "table_id": table_id,
        "page": page,
        "logical_rows": [
            {
                "logical_row_id": f"t{table_id}-r0",
                "table_id": table_id,
                "page": page,
                "parameter_name": "Vout",
                "test_conditions": "",
                "min": "1",
                "typ": "2",
                "max": "3",
                "unit": "V",
                "source_row_indices": [0],
                "source_cell_ids": [1],
                "needs_review": False,
            }
        ],
    }


@pytest.fixture()
def pdf_file(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4\n% test\n")
    return path


def test_profile_default_is_d(monkeypatch):
    monkeypatch.delenv("DATASHEET_RUN_PROFILE", raising=False)
    profile = pipeline.resolve_run_profile()

    assert profile.key == "d"
    assert profile.input_mode == "logical"
    assert profile.evidence_validation_mode == "recovered"
    assert profile.family_coverage is True


def test_profile_env_c(monkeypatch):
    monkeypatch.setenv("DATASHEET_RUN_PROFILE", "C")
    profile = pipeline.resolve_run_profile()

    assert profile.key == "c"
    assert profile.input_mode == "flat"
    assert profile.evidence_validation_mode == "legacy"


def test_profile_rejects_unknown():
    with pytest.raises(pipeline.PipelineError):
        pipeline.resolve_run_profile("x")


def test_flat_profile_skips_structure_recovery(pdf_file):
    runtime = pipeline.build_runtime_inputs(
        pdf_file,
        profile="c",
        pdf_parser=FakePDFParser(),
        table_parser=FakeTableParser([_table(4), _table(5)]),
        selector=FakeSelector(),
    )

    assert runtime["logical_rows"] is None
    assert runtime["recovered_evidence"] is None
    assert runtime["diagnostics"]["structure_recovery"]["enabled"] is False
    assert runtime["table_result"]["table_count"] == 2


def test_logical_profile_runs_recovery(
    pdf_file, monkeypatch
):
    calls = []

    def fake_recover(raw_structure, table_id, pdf_path, page_no):
        calls.append(table_id)
        return _recovered(table_id, page_no)

    monkeypatch.setattr(
        pipeline, "recover_from_pdf", fake_recover
    )

    runtime = pipeline.build_runtime_inputs(
        pdf_file,
        profile="d",
        pdf_parser=FakePDFParser(),
        table_parser=FakeTableParser([_table(4), _table(5)]),
        selector=FakeSelector(),
    )

    assert calls == [4, 5]
    assert runtime["logical_extraction"]["table_ids"] == [4, 5]
    assert runtime["logical_rows"]["schema"] == (
        "table_logical_rows/v1"
    )
    assert len(runtime["logical_rows"]["logical_rows"]) == 2
    assert runtime["recovered_evidence"]["pdf_sha256"] == (
        _raw_structure(pdf_file)["source"]["pdf_sha256"]
    )
    assert runtime["diagnostics"]["structure_recovery"][
        "recovered"
    ] == 2


def test_explicit_table_ids_filter_selection(pdf_file):
    runtime = pipeline.build_runtime_inputs(
        pdf_file,
        profile="c",
        pdf_parser=FakePDFParser(),
        table_parser=FakeTableParser([_table(4), _table(5)]),
        selector=FakeSelector(),
        table_ids=[5],
    )

    assert [
        table["table_id"]
        for table in runtime["table_result"]["tables"]
    ] == [5]


def test_explicit_logical_recovery_failure_rejects(
    pdf_file, monkeypatch
):
    def fake_recover(raw_structure, table_id, pdf_path, page_no):
        raise RuntimeError("no geometry")

    monkeypatch.setattr(
        pipeline, "recover_from_pdf", fake_recover
    )

    with pytest.raises(pipeline.PipelineError):
        pipeline.build_runtime_inputs(
            pdf_file,
            profile="d",
            pdf_parser=FakePDFParser(),
            table_parser=FakeTableParser([_table(4)]),
            selector=FakeSelector(),
            table_ids=[4],
        )


def test_run_pipeline_wires_recovered_mode(
    pdf_file, monkeypatch
):
    captured = {}

    class FakeGraph:
        def invoke(self, state, config):
            return {
                "status": "completed",
                "extraction_input_mode": "logical",
                "extraction_input_summary": {},
                "evidence_validation_mode": "recovered",
                "validation_result": {
                    "parameters": [
                        {
                            "name": "Vout",
                            "value": 2,
                            "unit": "V",
                            "value_type": "typ",
                            "validation_status": "validated",
                        }
                    ],
                    "validation_summary": {},
                },
            }

    def fake_build_graph(**kwargs):
        captured.update(kwargs)
        return FakeGraph()

    def fake_recover(raw_structure, table_id, pdf_path, page_no):
        return _recovered(table_id, page_no)

    monkeypatch.setattr(
        pipeline, "recover_from_pdf", fake_recover
    )
    monkeypatch.setattr(
        pipeline, "build_graph", fake_build_graph
    )

    run = pipeline.run_pipeline(
        pdf_file,
        profile="d",
        extractor=object(),
        pdf_parser=FakePDFParser(),
        table_parser=FakeTableParser([_table(4)]),
        selector=FakeSelector(),
    )

    assert captured["evidence_validation_mode"] == "recovered"
    assert captured["extraction_input_mode"] == "logical"
    assert captured["recovered_evidence"] is not None
    assert captured["logical_extraction"] is not None
    assert run["summary"]["validated"] == 1
    assert run["summary"]["auto_accepted"] == 1

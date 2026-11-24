import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_e2e  # noqa: E402


class FakeExtractor:
    model = "fake-model"

    def get_usage(self):
        return {
            "model": self.model,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }


@pytest.fixture()
def pdf_file(tmp_path):
    path = tmp_path / "demo.pdf"
    path.write_bytes(b"%PDF-1.4\n% demo\n")
    return path


def test_run_e2e_saves_artifact_and_pipeline_sidecar(
    tmp_path, pdf_file, monkeypatch, capsys
):
    def fake_run_pipeline(
        pdf_path,
        *,
        profile=None,
        extractor=None,
        table_ids=None,
        logical_rows_override=None,
        raw_structure_override=None,
    ):
        state = {
            "status": "completed",
            "extraction": {
                "component_model": "",
                "manufacturer": "",
                "parameters": [],
            },
            "validation_result": {
                "parameters": [],
                "validation_summary": {},
            },
            "tables": [],
            "extraction_input_mode": "logical",
            "extraction_input_summary": {},
            "evidence_validation_mode": "recovered",
            "retry_count": 0,
        }

        return {
            "state": state,
            "runtime": {
                "all_tables": [],
                "table_result": {"tables": []},
                "logical_rows": None,
            },
            "diagnostics": {
                "profile": profile.to_dict(),
                "structure_recovery": {
                    "enabled": True,
                    "attempted": 1,
                    "recovered": 1,
                    "failed": [],
                },
                "effective_input_mode": "logical",
                "validation_mode": "recovered",
            },
            "validation": state["validation_result"],
            "summary": {
                "total": 0,
                "validated": 0,
                "pending_review": 0,
                "invalid": 0,
                "auto_accepted_parameters": [],
                "pending_parameters": [],
                "invalid_parameters": [],
            },
            "profile": profile,
            "extractor": extractor,
        }

    monkeypatch.setattr(
        run_e2e, "run_pipeline", fake_run_pipeline
    )
    monkeypatch.setattr(
        run_e2e,
        "DeepSeekExtractor",
        lambda **kwargs: FakeExtractor(),
    )

    output_json = tmp_path / "demo.extraction.json"

    result = run_e2e.run_e2e(
        str(pdf_file),
        output_json=str(output_json),
        profile="d",
        table_ids=[4],
    )

    assert Path(result["artifact_path"]).is_file()
    assert Path(result["sidecar_path"]).is_file()

    artifact = json.loads(output_json.read_text())
    assert artifact["extraction"]["parameters"] == []

    sidecar = json.loads(
        Path(result["sidecar_path"]).read_text()
    )
    assert sidecar["profile"]["key"] == "d"
    assert sidecar["diagnostics"]["effective_input_mode"] == (
        "logical"
    )

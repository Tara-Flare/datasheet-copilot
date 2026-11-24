from fastapi.testclient import TestClient

import app.main as main


def test_extract_endpoint_uses_production_pipeline(
    monkeypatch,
):
    captured = {}

    def fake_run_pipeline(
        pdf_path,
        *,
        profile=None,
        table_ids=None,
    ):
        captured.update({
            "pdf_path": pdf_path,
            "profile": profile,
            "table_ids": table_ids,
        })

        return {
            "pdf_document": {
                "page_count": 1,
                "pages": [{"page": 1, "text": "demo"}],
            },
            "runtime": {"all_tables": []},
            "validation": {
                "parameters": [],
                "validation_summary": {},
            },
            "summary": {
                "auto_accepted_parameters": [],
                "pending_parameters": [],
                "invalid_parameters": [],
            },
            "diagnostics": {
                "profile": {"key": "c"},
                "effective_input_mode": "flat",
                "validation_mode": "legacy",
            },
        }

    monkeypatch.setattr(
        main, "run_pipeline", fake_run_pipeline
    )

    client = TestClient(main.app)
    response = client.post(
        "/api/v1/documents/parse"
        "?extract=true&profile=c&table_ids=4",
        files={
            "file": (
                "demo.pdf",
                b"%PDF-1.4\n% demo\n",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert captured["profile"] == "c"
    assert captured["table_ids"] == [4]
    assert payload["status"] == "extracted"
    assert payload["pipeline"]["profile"]["key"] == "c"
    assert payload["auto_accept"] == []
    assert payload["pending_review"] == []
    assert payload["invalid"] == []


def test_extract_endpoint_rejects_non_pdf():
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/documents/parse?extract=true",
        files={"file": ("demo.txt", b"not a pdf")},
    )

    assert response.status_code == 400

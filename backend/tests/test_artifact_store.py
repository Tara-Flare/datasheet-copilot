"""提取结果 JSON 保存/读取与字段完整性测试。"""

import json

import pytest

from app.extraction.artifact_store import (
    ArtifactError,
    RecordingExtractor,
    artifact_from_state,
    build_artifact,
    load_artifact,
    save_artifact,
)


def sample_extraction():
    return {
        "component_model": "TPS7A02",
        "manufacturer": "TI",
        "parameters": [
            {
                "name": "input_voltage",
                "value": 1.5,
                "unit": "V",
                "value_type": "min",
                "test_conditions": "",
                "evidence": {
                    "table_id": 7,
                    "row_index": 0,
                    "page": 5,
                    "source_text": (
                        "VIN | Input voltage | 1.5 | | 6.0 | V"
                    ),
                },
            }
        ],
    }


def test_save_and_load_roundtrip(tmp_path):
    artifact = build_artifact(
        sample_extraction(),
        pdf_path="data/samples/TPS7A02.pdf",
        pdf_sha256="abc123",
        table_cache_dir="data/cache/docling",
        table_cache_key="cache-key.json",
        selected_table_ids=[7],
        model="deepseek-flash",
    )

    path = tmp_path / "extraction.json"
    save_artifact(artifact, path)

    loaded = load_artifact(path)

    assert loaded["extraction"] == artifact["extraction"]
    assert (
        loaded["source"]["pdf_filename"]
        == "TPS7A02.pdf"
    )
    assert (
        loaded["source"]["table_cache_key"]
        == "cache-key.json"
    )
    assert loaded["source"]["selected_table_ids"] == [7]
    assert loaded["source"]["model"] == "deepseek-flash"


def test_parameter_and_evidence_fields_preserved(tmp_path):
    artifact = build_artifact(sample_extraction())

    path = tmp_path / "extraction.json"
    save_artifact(artifact, path)

    loaded = load_artifact(path)
    param = loaded["extraction"]["parameters"][0]
    evidence = param["evidence"]

    assert param["name"] == "input_voltage"
    assert param["value"] == 1.5
    assert param["unit"] == "V"
    assert param["value_type"] == "min"
    assert param["test_conditions"] == ""

    assert evidence["table_id"] == 7
    assert evidence["row_index"] == 0
    assert evidence["page"] == 5
    assert evidence["source_text"].startswith("VIN")


def test_saved_json_contains_required_fields(tmp_path):
    artifact = build_artifact(sample_extraction())

    path = tmp_path / "extraction.json"
    save_artifact(artifact, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    param = data["extraction"]["parameters"][0]

    for field in (
        "name",
        "value",
        "unit",
        "value_type",
        "test_conditions",
        "evidence",
    ):
        assert field in param

    for field in (
        "table_id",
        "row_index",
        "page",
        "source_text",
    ):
        assert field in param["evidence"]


def test_missing_parameter_field_raises():
    extraction = sample_extraction()
    del extraction["parameters"][0]["test_conditions"]

    with pytest.raises(ArtifactError) as exc_info:
        build_artifact(extraction)

    assert "test_conditions" in str(exc_info.value)


def test_missing_evidence_field_raises():
    extraction = sample_extraction()
    del extraction["parameters"][0]["evidence"]["source_text"]

    with pytest.raises(ArtifactError) as exc_info:
        build_artifact(extraction)

    assert "source_text" in str(exc_info.value)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ArtifactError) as exc_info:
        load_artifact(tmp_path / "missing.json")

    assert "不存在" in str(exc_info.value)


def test_load_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ArtifactError) as exc_info:
        load_artifact(path)

    assert "JSON" in str(exc_info.value)


def test_load_unsupported_schema_raises(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({
            "schema_version": 99,
            "extraction": sample_extraction(),
        }),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError) as exc_info:
        load_artifact(path)

    assert "schema_version" in str(exc_info.value)


class FakeExtractor:
    def __init__(self, extract_result, retry_results):
        self._extract_result = extract_result
        self._retry_results = list(retry_results)

    def extract(self, tables):
        return self._extract_result

    def retry_parameters(self, failed_parameters, tables):
        return self._retry_results.pop(0)

    def get_usage(self):
        return {"model": "fake"}


def test_recording_extractor_records_outputs():
    initial = sample_extraction()
    retried = sample_extraction()

    fake = FakeExtractor(initial, [retried])
    recorder = RecordingExtractor(fake)

    recorder.extract([{"table_id": 7}])
    recorder.retry_parameters([], [{"table_id": 7}])

    assert recorder.extract_result == initial
    assert recorder.retry_results == [retried]
    assert recorder.get_usage() == {"model": "fake"}

    # 记录的是副本，外部修改原对象不应影响已保存结果
    initial["parameters"][0]["value"] = 999

    assert (
        recorder.extract_result["parameters"][0]["value"]
        == 1.5
    )


def test_artifact_from_state():
    extraction = sample_extraction()

    fake = FakeExtractor(extraction, [])
    recorder = RecordingExtractor(fake)
    recorder.extract([{"table_id": 7}])

    state = {
        "extraction": extraction,
        "tables": [{"table_id": 7}],
    }

    artifact = artifact_from_state(
        state,
        recorder,
        pdf_path="data/samples/TPS7A02.pdf",
        pdf_sha256="abc123",
        table_cache_dir="data/cache/docling",
        table_cache_key="cache-key.json",
        model="fake",
    )

    assert artifact["extraction"] == extraction
    assert artifact["final_extraction"] == extraction
    assert (
        artifact["source"]["selected_table_ids"]
        == [7]
    )


def test_artifact_from_state_without_extraction_raises():
    fake = FakeExtractor(sample_extraction(), [])
    recorder = RecordingExtractor(fake)

    with pytest.raises(ArtifactError) as exc_info:
        artifact_from_state(
            {"tables": []},
            recorder,
            pdf_path="x.pdf",
        )

    assert "提取结果" in str(exc_info.value)

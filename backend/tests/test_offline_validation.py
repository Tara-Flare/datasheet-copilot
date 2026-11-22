"""离线重校验测试：不调用 DeepSeek，不重新解析 PDF。"""

import json

import pytest

from app.extraction.artifact_store import (
    build_artifact,
    load_artifact,
    save_artifact,
)

from app.validation.offline import (
    OfflineValidationError,
    build_validation_report,
    compare_reports,
    ensure_tables_available,
    load_cached_tables,
    load_validation_report,
    revalidate,
    save_validation_report,
    select_extraction,
)


def vin_table():
    return {
        "table_id": 7,
        "page_numbers": [5],
        "columns": ["", "", "MIN", "NOM", "MAX", "UNIT"],
        "rows": [
            ["VIN", "Input voltage", "1.5", "", "6.0", "V"],
        ],
    }


def extraction_with(
    source_text="VIN | Input voltage | 1.5 | | 6.0 | V",
    value=1.5,
    value_type="min",
    unit="V",
    table_id=7,
    row_index=0,
    page=5,
):
    return {
        "component_model": "TPS7A02",
        "manufacturer": "TI",
        "parameters": [
            {
                "name": "input_voltage",
                "value": value,
                "unit": unit,
                "value_type": value_type,
                "test_conditions": "",
                "evidence": {
                    "table_id": table_id,
                    "row_index": row_index,
                    "page": page,
                    "source_text": source_text,
                },
            }
        ],
    }


def test_offline_revalidation_pipe_source_text_passes():
    result = revalidate(
        extraction_with(),
        [vin_table()],
    )

    param = result["parameters"][0]

    assert param["validation_status"] == "validated"
    assert param["validation_errors"] == []


def test_offline_revalidation_wrong_value_invalid():
    result = revalidate(
        extraction_with(value=2.0),
        [vin_table()],
    )

    param = result["parameters"][0]

    assert param["validation_status"] == "invalid"
    assert any(
        "提取数值" in error
        for error in param["validation_errors"]
    )


def test_offline_revalidation_wrong_unit_invalid():
    result = revalidate(
        extraction_with(unit="mV"),
        [vin_table()],
    )

    param = result["parameters"][0]

    assert param["validation_status"] == "invalid"
    assert any(
        "提取单位" in error
        for error in param["validation_errors"]
    )


def test_revalidation_overwrites_previous_validation_fields():
    extraction = extraction_with()
    extraction["parameters"][0]["validation_status"] = "invalid"
    extraction["parameters"][0]["validation_errors"] = ["旧错误"]

    result = revalidate(extraction, [vin_table()])

    assert (
        result["parameters"][0]["validation_status"]
        == "validated"
    )
    assert result["parameters"][0]["validation_errors"] == []


def test_report_summary_and_statistics():
    report = build_validation_report(
        extraction_with(value=2.0),
        [vin_table()],
        validator_version="test-v1",
    )

    assert report["validator_version"] == "test-v1"
    assert report["summary"]["total"] == 1
    assert report["summary"]["invalid"] == 1
    assert (
        report["error_statistics"]["total_errors"]
        >= 1
    )
    assert (
        report["error_statistics"][
            "parameters_with_errors"
        ]
        == 1
    )
    assert (
        report["error_statistics"]["by_category"]["数值"]
        == 1
    )


def test_load_cached_tables_missing_file_raises(tmp_path):
    with pytest.raises(OfflineValidationError) as exc_info:
        load_cached_tables(tmp_path / "missing.json")

    assert "不存在" in str(exc_info.value)


def test_load_cached_tables_incomplete_raises(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"filename": "x.pdf"}),
        encoding="utf-8",
    )

    with pytest.raises(OfflineValidationError) as exc_info:
        load_cached_tables(path)

    assert "tables" in str(exc_info.value)


def test_load_cached_tables_invalid_table_raises(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"tables": [{"table_id": 7}]}),
        encoding="utf-8",
    )

    with pytest.raises(OfflineValidationError) as exc_info:
        load_cached_tables(path)

    assert "rows" in str(exc_info.value)


def test_missing_referenced_table_raises():
    cache_table = dict(vin_table())
    cache_table["table_id"] = 1

    with pytest.raises(OfflineValidationError) as exc_info:
        ensure_tables_available(
            [cache_table],
            extraction_with(),
        )

    message = str(exc_info.value)

    assert "table_id" in message
    assert "7" in message


def test_select_extraction_missing_source_raises():
    artifact = {
        "extraction": extraction_with(),
        "final_extraction": None,
    }

    with pytest.raises(OfflineValidationError):
        select_extraction(artifact, "final_extraction")


def test_select_extraction_invalid_source_raises():
    artifact = {"extraction": extraction_with()}

    with pytest.raises(OfflineValidationError):
        select_extraction(artifact, "unknown")


def test_report_save_and_load_roundtrip(tmp_path):
    report = build_validation_report(
        extraction_with(),
        [vin_table()],
    )

    path = tmp_path / "report.json"
    save_validation_report(report, path)

    loaded = load_validation_report(path)

    assert loaded["summary"] == report["summary"]
    assert (
        loaded["parameters"][0]["validation_status"]
        == "validated"
    )


def test_load_validation_report_missing_file_raises(
    tmp_path,
):
    with pytest.raises(OfflineValidationError) as exc_info:
        load_validation_report(tmp_path / "missing.json")

    assert "不存在" in str(exc_info.value)


def test_compare_reports_detects_improvement_and_regression():
    old = {
        "validator_version": "old",
        "validator_fingerprint": "aaa",
        "summary": {
            "total": 2,
            "validated": 1,
            "invalid": 1,
            "pending_review": 0,
        },
        "parameters": [
            {
                "name": "a",
                "value_type": "min",
                "test_conditions": "",
                "validation_status": "invalid",
                "validation_errors": [
                    "原文引用不匹配"
                ],
            },
            {
                "name": "b",
                "value_type": "min",
                "test_conditions": "",
                "validation_status": "validated",
                "validation_errors": [],
            },
        ],
    }

    new = {
        "validator_version": "new",
        "validator_fingerprint": "bbb",
        "summary": {
            "total": 2,
            "validated": 1,
            "invalid": 1,
            "pending_review": 0,
        },
        "parameters": [
            {
                "name": "a",
                "value_type": "min",
                "test_conditions": "",
                "validation_status": "validated",
                "validation_errors": [],
            },
            {
                "name": "b",
                "value_type": "min",
                "test_conditions": "",
                "validation_status": "invalid",
                "validation_errors": [
                    "提取数值 2.0 与来源数值 1.5 不一致"
                ],
            },
        ],
    }

    diff = compare_reports(old, new)

    assert diff["improved"] == 1
    assert diff["regressed"] == 1
    assert diff["changed_count"] == 2
    assert (
        diff["old_validator_fingerprint"] == "aaa"
    )
    assert (
        diff["new_validator_fingerprint"] == "bbb"
    )


def test_end_to_end_offline_from_files(tmp_path):
    """完整离线流程：artifact 文件 + 缓存文件 → 校验报告。"""

    artifact = build_artifact(
        extraction_with(),
        pdf_path="data/samples/TPS7A02.pdf",
        table_cache_dir=str(tmp_path),
        table_cache_key="cache.json",
    )

    extraction_path = tmp_path / "extraction.json"
    save_artifact(artifact, extraction_path)

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps({
            "filename": "TPS7A02.pdf",
            "table_count": 1,
            "tables": [vin_table()],
        }),
        encoding="utf-8",
    )

    loaded = load_artifact(extraction_path)
    tables = load_cached_tables(cache_path)
    extraction = select_extraction(
        loaded,
        "extraction",
    )

    ensure_tables_available(tables, extraction)

    report = build_validation_report(
        extraction,
        tables,
        artifact_path=extraction_path,
        extraction_metadata=loaded.get("source"),
    )

    assert report["summary"]["validated"] == 1
    assert report["summary"]["invalid"] == 0
    assert (
        report["extraction_metadata"]["table_cache_key"]
        == "cache.json"
    )


def test_offline_flow_does_not_require_cache_metadata(
    tmp_path,
):
    """显式提供 --table-cache 时，不依赖 artifact 记录的缓存位置。"""

    artifact = build_artifact(extraction_with())

    extraction_path = tmp_path / "extraction.json"
    save_artifact(artifact, extraction_path)

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps({"tables": [vin_table()]}),
        encoding="utf-8",
    )

    loaded = load_artifact(extraction_path)

    assert loaded["source"]["table_cache_key"] is None

    tables = load_cached_tables(cache_path)
    extraction = select_extraction(loaded)

    ensure_tables_available(tables, extraction)

    report = build_validation_report(extraction, tables)

    assert report["summary"]["validated"] == 1

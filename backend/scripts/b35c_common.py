"""B3.5c 跨型号 Flat Baseline 公共工具。

只读校验 B3.5b 冻结 Gold 与同步回执；不修改 Gold、manifest、PDF。
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
B35B_DIR = PROJECT_ROOT / "data/outputs/experiments/B3.5b"
MANIFEST = B35B_DIR / "manifest.json"
GOLD = B35B_DIR / "gold/gold.frozen.json"
GOLD_RECEIPT = B35B_DIR / "gold/gold.frozen.receipt.json"
SYNC_RECEIPT = B35B_DIR / "gold/gold_sync_receipt.json"
FORMAL_XLSX = B35B_DIR / "gold/gold_review.B3.5b.xlsx"
DEFAULT_CACHE = PROJECT_ROOT / "data/cache/b35b-docling"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/outputs/experiments/B3.5c"

EXPECTED_SAMPLES = {
    "adi_lt3045": {"rows": 25, "confirmed": 18, "rejected": 7},
    "diodes_ap2112": {"rows": 23, "confirmed": 21, "rejected": 2},
    "onsemi_ncp170": {"rows": 22, "confirmed": 19, "rejected": 3},
    "renesas_isl80510": {"rows": 18, "confirmed": 18, "rejected": 0},
}

EXPECTED_TOTAL = {
    "rows": 88,
    "confirmed": 76,
    "rejected": 12,
    "unreviewed": 0,
}

PRODUCTION_CONFIG = {
    "extraction_input_mode": "flat",
    "evidence_validation_mode": "legacy",
    "max_retries": 1,
    "rows_per_chunk": 10,
    "max_tokens": 6000,
    "max_split_depth": 6,
    "table_selector": {"max_tables": 4, "max_chars": 40000},
}


class GateError(RuntimeError):
    """冻结 Gold 或运行前置条件不满足。"""


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if exclusive and path.exists():
        raise GateError(f"拒绝覆盖已有文件: {path}")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)

    return path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_state():
    def run(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip()
        except OSError:
            return None

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def source_fingerprints():
    files = [
        "backend/app/extraction/prompts.py",
        "backend/app/extraction/deepseek_extractor.py",
        "backend/app/extraction/input_mode.py",
        "backend/app/extraction/logical_table_adapter.py",
        "backend/app/parsers/table_selector.py",
        "backend/app/parsers/structure_recovery.py",
        "backend/app/parsers/table_parser.py",
        "backend/app/validation/validator.py",
        "backend/app/validation/rules.py",
        "backend/app/validation/units.py",
        "backend/app/extraction/evidence.py",
        "backend/app/agents/graph.py",
        "backend/app/agents/nodes.py",
    ]

    return {
        path: sha256_file(PROJECT_ROOT / path)
        for path in files
        if (PROJECT_ROOT / path).is_file()
    }


def validate_gold_gate(
    *,
    manifest_path=MANIFEST,
    gold_path=GOLD,
    receipt_path=GOLD_RECEIPT,
    sync_receipt_path=SYNC_RECEIPT,
):
    """校验冻结 Gold、冻结回执、同步回执与 PDF/CSV 来源。

    任何不一致都抛 GateError；不自动修复。
    """

    manifest_path = Path(manifest_path)
    gold_path = Path(gold_path)
    receipt_path = Path(receipt_path)

    for path, label in (
        (manifest_path, "manifest"),
        (gold_path, "frozen Gold"),
        (receipt_path, "frozen receipt"),
    ):
        if not path.is_file():
            raise GateError(f"{label} 不存在: {path}")

    manifest = load_json(manifest_path)
    gold = load_json(gold_path)
    receipt = load_json(receipt_path)

    if gold.get("status") != "frozen":
        raise GateError("gold.frozen.json status != frozen")

    if gold.get("sample_count") != 4:
        raise GateError("Gold sample_count != 4")

    if gold.get("record_count") != EXPECTED_TOTAL["confirmed"]:
        raise GateError(
            "Gold record_count != 76: "
            f"{gold.get('record_count')}"
        )

    if receipt.get("status") != "frozen":
        raise GateError("冻结回执 status != frozen")

    if receipt.get("gold_sha256") != sha256_file(gold_path):
        raise GateError("冻结回执 gold_sha256 与 Gold 不一致")

    if receipt.get("manifest_sha256") != sha256_file(manifest_path):
        raise GateError("冻结回执 manifest_sha256 与 manifest 不一致")

    if receipt.get("record_count") != EXPECTED_TOTAL["confirmed"]:
        raise GateError("冻结回执 record_count != 76")

    if receipt.get("errors"):
        raise GateError(f"冻结回执包含错误: {receipt['errors']}")

    manifest_by_id = {
        sample["sample_id"]: sample
        for sample in manifest.get("samples", [])
    }
    gold_by_id = {
        sample["sample_id"]: sample
        for sample in gold.get("samples", [])
    }

    if set(manifest_by_id) != set(EXPECTED_SAMPLES):
        raise GateError("manifest 样本集合与预期不一致")

    if set(gold_by_id) != set(EXPECTED_SAMPLES):
        raise GateError("Gold 样本集合与预期不一致")

    gold_ids = []
    summary = []

    for sample_id, expected in EXPECTED_SAMPLES.items():
        sample = manifest_by_id[sample_id]
        gold_sample = gold_by_id[sample_id]
        pdf_path = PROJECT_ROOT / sample["file"]

        if not pdf_path.is_file():
            raise GateError(f"PDF 不存在: {pdf_path}")

        if sha256_file(pdf_path) != sample["pdf_sha256"]:
            raise GateError(f"{sample_id}: PDF SHA256 与 manifest 不一致")

        if sha256_file(pdf_path) != gold_sample["pdf_sha256"]:
            raise GateError(f"{sample_id}: PDF SHA256 与 Gold 不一致")

        if gold_sample["record_count"] != expected["confirmed"]:
            raise GateError(
                f"{sample_id}: Gold confirmed 数不等于 "
                f"{expected['confirmed']}"
            )

        csv_path = B35B_DIR / "gold" / gold_sample["review_csv"]

        if not csv_path.is_file():
            raise GateError(f"{sample_id}: 审核 CSV 不存在: {csv_path}")

        if sha256_file(csv_path) != gold_sample["review_csv_sha256"]:
            raise GateError(f"{sample_id}: 审核 CSV SHA256 与 Gold 不一致")

        scope_ids = set(sample.get("gold_scope_table_ids") or [])

        if not scope_ids:
            raise GateError(f"{sample_id}: manifest 缺少 gold_scope_table_ids")

        for record in gold_sample["records"]:
            gold_ids.append(record["gold_id"])

            if record["gold_id"].split(":")[0] != sample_id:
                raise GateError(
                    f"{sample_id}: gold_id 前缀与样本不一致: "
                    f"{record['gold_id']}"
                )

            if record["source"]["table_id"] not in scope_ids:
                raise GateError(
                    f"{sample_id}: 记录超出 Gold scope 表: "
                    f"{record['gold_id']}"
                )

            if not record.get("value"):
                raise GateError(
                    f"{sample_id}: Gold 记录缺少数值: "
                    f"{record['gold_id']}"
                )

            if record.get("value_type") not in {"min", "typ", "max"}:
                raise GateError(
                    f"{sample_id}: Gold value_type 非法: "
                    f"{record['gold_id']}"
                )

        if len(gold_sample["records"]) != expected["confirmed"]:
            raise GateError(
                f"{sample_id}: Gold 记录数与 record_count 不一致"
            )

        summary.append(
            {
                "sample_id": sample_id,
                "vendor": sample["vendor"],
                "model": sample["model"],
                "document_version": sample["document_version"],
                "pdf_path": sample["file"],
                "pdf_sha256": sample["pdf_sha256"],
                "gold_scope_table_ids": sorted(scope_ids),
                "selector_selected_table_ids": list(
                    sample.get("selector_selected_table_ids") or []
                ),
                "record_count": len(gold_sample["records"]),
                "review_csv": gold_sample["review_csv"],
                "review_csv_sha256": gold_sample["review_csv_sha256"],
            }
        )

    if len(gold_ids) != len(set(gold_ids)):
        raise GateError("Gold gold_id 不是全库唯一")

    if len(gold_ids) != EXPECTED_TOTAL["confirmed"]:
        raise GateError(
            f"Gold 记录总数 != 76: {len(gold_ids)}"
        )

    sync_status = None

    if sync_receipt_path and Path(sync_receipt_path).is_file():
        import sys

        if str(BACKEND_ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

        from b35b_review_common import verify_sync_receipt

        verified = verify_sync_receipt(
            str(B35B_DIR / "gold"), str(manifest_path)
        )

        if verified.get("record_count") != 88:
            raise GateError("同步回执 record_count != 88")

        if verified.get("allow_incomplete"):
            raise GateError("同步回执 allow_incomplete=true")

        sync_status = {
            "ok": True,
            "record_count": verified["record_count"],
            "allow_incomplete": verified["allow_incomplete"],
            "xlsx_path": verified.get("xlsx_path"),
            "xlsx_sha256": verified.get("xlsx_sha256"),
        }

    return {
        "manifest": manifest,
        "gold": gold,
        "receipt": receipt,
        "samples": summary,
        "sync_receipt": sync_status,
        "hashes": {
            "manifest": sha256_file(manifest_path),
            "gold": sha256_file(gold_path),
            "gold_receipt": sha256_file(receipt_path),
            "formal_xlsx": (
                sha256_file(FORMAL_XLSX)
                if FORMAL_XLSX.is_file()
                else None
            ),
        },
    }


def runtime_config(extra=None):
    payload = {
        "schema": "b35c_runtime_config/v1",
        "generated_at": utc_now(),
        "production_config": dict(PRODUCTION_CONFIG),
        "env": {
            "LLM_MODEL": os.getenv("LLM_MODEL", "deepseek-flash"),
            "LLM_BASE_URL": os.getenv(
                "LLM_BASE_URL", "https://api.deepseek.com"
            ),
            "LLM_MAX_TOKENS": os.getenv("LLM_MAX_TOKENS", "6000"),
            "LLM_ROWS_PER_CHUNK": os.getenv(
                "LLM_ROWS_PER_CHUNK", "10"
            ),
            "LLM_MAX_SPLIT_DEPTH": os.getenv(
                "LLM_MAX_SPLIT_DEPTH", "6"
            ),
            "EXTRACTION_INPUT_MODE": os.getenv(
                "EXTRACTION_INPUT_MODE", "flat"
            ),
            "EVIDENCE_VALIDATION_MODE": os.getenv(
                "EVIDENCE_VALIDATION_MODE", "legacy"
            ),
        },
        "source_fingerprints": source_fingerprints(),
        "git": git_state(),
    }

    if extra:
        payload.update(extra)

    return payload

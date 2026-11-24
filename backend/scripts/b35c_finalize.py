"""B3.5c 汇总：把预检、运行清单与指标合并为 summary.json。

只读既有产物，不调用 LLM，不修改 Gold 或运行目录。
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data/outputs/experiments/B3.5c"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def build(experiment_dir, run_id):
    experiment_dir = Path(experiment_dir)
    run_dir = experiment_dir / "runs" / run_id
    preflight = load_json(experiment_dir / "preflight/preflight.json")
    metrics = load_json(experiment_dir / "results/metrics.json")
    run_manifest = load_json(run_dir / "run_manifest.json")

    samples = []

    for sample in metrics["samples"]:
        sample_id = sample["sample_id"]
        sample_dir = run_dir / sample_id
        api = load_json(sample_dir / "api_responses.json")
        usage = load_json(sample_dir / "usage.json")
        timing = load_json(sample_dir / "timing.json")
        artifact = load_json(sample_dir / "artifact.json")

        finishes = {}

        for call in api.get("calls", []):
            key = call.get("finish_reason")
            finishes[key] = finishes.get(key, 0) + 1

        raw_by_table = {}

        for param in artifact["extraction"].get(
            "parameters", []
        ):
            table_id = (param.get("evidence") or {}).get(
                "table_id"
            )
            raw_by_table[table_id] = (
                raw_by_table.get(table_id, 0) + 1
            )

        samples.append(
            {
                "sample_id": sample_id,
                "vendor": sample["vendor"],
                "model": sample["model"],
                "document_version": sample["document_version"],
                "gold_scope_table_ids": sample.get(
                    "gold_scope_table_ids"
                ),
                "api_calls": api.get("call_count"),
                "elapsed_s": timing.get("elapsed_s"),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get(
                        "completion_tokens"
                    ),
                },
                "finish_reasons": finishes,
                "response_models": sorted(
                    {
                        call.get("response_model")
                        for call in api.get("calls", [])
                    }
                ),
                "system_fingerprints": sorted(
                    {
                        call.get("system_fingerprint")
                        for call in api.get("calls", [])
                    }
                ),
                "raw_parameters_by_table": raw_by_table,
                "metrics": {
                    key: sample[key]
                    for key in (
                        "gold_records",
                        "predictions_total",
                        "predictions_in_scope",
                        "out_of_scope_table_predictions",
                        "out_of_scope_parameter_predictions",
                        "tp",
                        "fp",
                        "fn",
                        "precision",
                        "recall",
                        "f1",
                        "identity_accuracy",
                        "value_unit_accuracy",
                        "conditions_complete_rate",
                        "conditions_exact_rate",
                        "source_localization_rate",
                        "strict_f1",
                        "error_category_counts",
                        "validation_status_distribution",
                    )
                },
            }
        )

    return {
        "schema": "b35c_summary/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "B3.5c",
        "run_id": run_id,
        "status": run_manifest.get("status"),
        "gold_gate": preflight.get("gold_gate"),
        "production_config": preflight.get("production_config"),
        "preflight_estimated_initial_calls": preflight.get(
            "estimated_initial_llm_calls"
        ),
        "run": {
            "started_at": run_manifest.get("started_at"),
            "finished_at": run_manifest.get("finished_at"),
            "total_api_calls": run_manifest.get(
                "total_api_calls"
            ),
            "total_usage": run_manifest.get("total_usage"),
            "failures": run_manifest.get("failures"),
        },
        "metrics_micro": metrics["micro"],
        "samples": samples,
        "limitations": metrics.get("limitations", []),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir", default=str(DEFAULT_DIR)
    )
    parser.add_argument("--run-id", default="flat_baseline_r1")
    parser.add_argument(
        "--output", default=None
    )
    args = parser.parse_args(argv)

    summary = build(args.experiment_dir, args.run_id)
    output = (
        Path(args.output)
        if args.output
        else Path(args.experiment_dir) / "summary.json"
    )
    write_json(output, summary)

    micro = summary["metrics_micro"]
    print("summary:", output)
    print(
        "TP/FP/FN:",
        micro["tp"],
        micro["fp"],
        micro["fn"],
        "P/R/F1:",
        micro["precision"],
        micro["recall"],
        micro["f1"],
    )
    print(
        "calls:",
        summary["run"]["total_api_calls"],
        "usage:",
        summary["run"]["total_usage"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

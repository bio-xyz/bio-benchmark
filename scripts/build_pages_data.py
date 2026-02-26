#!/usr/bin/env python3
"""Build GitHub Pages data assets for the benchmark results site."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOCS_RESULTS_DIR = DOCS_DATA_DIR / "results"
DOCS_ASSETS_DIR = DOCS_DIR / "assets"

TARGET_CSVS = [
    "phylobiobixbench-verified-50-20260224-190546.csv",
    "phylobiobixbench-verified-50-20260224-224841.csv",
    "phylobiobixbench-verified-50-20260225-085809.csv",
]
SOURCE_MARKDOWN = "results/saved_output.md"
PERFORMANCE_PNG = "performance_analysis.png"


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * q
    low = int(idx)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = idx - low
    return float(sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction)


def parse_file_metadata(filename: str) -> tuple[str, str]:
    match = re.search(r"-(\d{8})-(\d{6})\.csv$", filename)
    if not match:
        raise ValueError(f"Unexpected CSV filename format: {filename}")
    yyyymmdd = match.group(1)
    file_code = match.group(2)
    date = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return date, file_code


def metric_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    metric_columns = {
        "direct": "direct_correct",
        "mcq_with_refusal": "mcq_with_refusal_correct",
        "mcq_without_refusal": "mcq_without_refusal_correct",
    }
    metric_labels = {
        "direct": "Direct",
        "mcq_with_refusal": "MCQ with refusal",
        "mcq_without_refusal": "MCQ without refusal",
    }
    total = len(rows)
    out: dict[str, dict[str, Any]] = {}
    for key, column in metric_columns.items():
        correct = sum(1 for row in rows if to_bool(row[column]))
        pct = round((correct / total) * 100, 1) if total else 0.0
        out[key] = {
            "label": metric_labels[key],
            "correct": correct,
            "total": total,
            "pct": pct,
        }
    return out


def load_rows(csv_paths: list[Path]) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    all_rows: list[dict[str, str]] = []
    by_file_rows: dict[str, list[dict[str, str]]] = {}
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        by_file_rows[csv_path.name] = rows
        all_rows.extend(rows)
    return all_rows, by_file_rows


def build_payload(all_rows: list[dict[str, str]], by_file_rows: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    source_meta: list[dict[str, Any]] = []
    for filename in TARGET_CSVS:
        date, file_code = parse_file_metadata(filename)
        rows = by_file_rows[filename]
        repeat_values = sorted({int(row["repeat_index"]) for row in rows})
        source_meta.append(
            {
                "filename": filename,
                "file_code": file_code,
                "date": date,
                "rows": len(rows),
                "repeats": len(repeat_values),
            }
        )

    overall = metric_summary(all_rows)

    by_file = []
    for filename in TARGET_CSVS:
        rows = by_file_rows[filename]
        _, file_code = parse_file_metadata(filename)
        latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms")]
        by_file.append(
            {
                "filename": filename,
                "file_code": file_code,
                "rows": len(rows),
                "repeats": len({row["repeat_index"] for row in rows}),
                "metrics": metric_summary(rows),
                "latency": {
                    "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                    "p95_ms": round(percentile(latencies, 0.95), 1) if latencies else 0.0,
                },
            }
        )

    repeat_map: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        file_code = row["run_id"].split("-")[-2]
        repeat_map[(file_code, row["repeat_index"])].append(row)

    by_repeat = []
    for (file_code, repeat_index), rows in repeat_map.items():
        by_repeat.append(
            {
                "file_code": file_code,
                "repeat_index": int(repeat_index),
                "repeat_label": f"r{int(repeat_index)}",
                "run_id": rows[0]["run_id"],
                "rows": len(rows),
                "metrics": metric_summary(rows),
            }
        )
    by_repeat.sort(
        key=lambda item: (
            -item["metrics"]["mcq_without_refusal"]["pct"],
            -item["metrics"]["direct"]["pct"],
            item["run_id"],
        )
    )
    for rank, row in enumerate(by_repeat, start=1):
        row["rank_mcq_without_refusal"] = rank

    question_acc: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        question_id = row["question_id"]
        if question_id not in question_acc:
            question_acc[question_id] = {
                "question_id": question_id,
                "task_group": row["task_group"],
                "n": 0,
                "direct": 0,
                "mcq_with_refusal": 0,
                "mcq_without_refusal": 0,
            }
        rec = question_acc[question_id]
        rec["n"] += 1
        rec["direct"] += int(to_bool(row["direct_correct"]))
        rec["mcq_with_refusal"] += int(to_bool(row["mcq_with_refusal_correct"]))
        rec["mcq_without_refusal"] += int(to_bool(row["mcq_without_refusal_correct"]))

    question_scores = []
    for rec in question_acc.values():
        n = rec["n"]
        direct_pct = round((rec["direct"] / n) * 100, 1)
        mcq_with_refusal_pct = round((rec["mcq_with_refusal"] / n) * 100, 1)
        mcq_without_refusal_pct = round((rec["mcq_without_refusal"] / n) * 100, 1)
        question_scores.append(
            {
                "question_id": rec["question_id"],
                "task_group": rec["task_group"],
                "n": n,
                "direct_pct": direct_pct,
                "mcq_with_refusal_pct": mcq_with_refusal_pct,
                "mcq_without_refusal_pct": mcq_without_refusal_pct,
                "refusal_gap_pp": round(mcq_without_refusal_pct - mcq_with_refusal_pct, 1),
                "mcq_without_refusal_failures": int(n - rec["mcq_without_refusal"]),
            }
        )
    question_scores.sort(key=lambda item: (item["mcq_without_refusal_pct"], item["question_id"]))

    task_group_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "direct": 0,
            "mcq_with_refusal": 0,
            "mcq_without_refusal": 0,
            "questions": set(),
        }
    )
    for row in all_rows:
        task_group = row["task_group"]
        task_group_acc[task_group]["n"] += 1
        task_group_acc[task_group]["direct"] += int(to_bool(row["direct_correct"]))
        task_group_acc[task_group]["mcq_with_refusal"] += int(
            to_bool(row["mcq_with_refusal_correct"])
        )
        task_group_acc[task_group]["mcq_without_refusal"] += int(
            to_bool(row["mcq_without_refusal_correct"])
        )
        task_group_acc[task_group]["questions"].add(row["question_id"])

    task_group_scores = []
    for task_group, rec in task_group_acc.items():
        n = rec["n"]
        task_group_scores.append(
            {
                "task_group": task_group,
                "questions": len(rec["questions"]),
                "n": n,
                "direct_pct": round(rec["direct"] / n * 100, 1),
                "mcq_with_refusal_pct": round(rec["mcq_with_refusal"] / n * 100, 1),
                "mcq_without_refusal_pct": round(rec["mcq_without_refusal"] / n * 100, 1),
            }
        )
    task_group_scores.sort(key=lambda item: (item["mcq_without_refusal_pct"], item["task_group"]))

    rows_direct_wrong = [row for row in all_rows if not to_bool(row["direct_correct"])]
    rows_direct_right = [row for row in all_rows if to_bool(row["direct_correct"])]
    rescued = sum(1 for row in rows_direct_wrong if to_bool(row["mcq_without_refusal_correct"]))
    lost = sum(
        1 for row in rows_direct_right if not to_bool(row["mcq_without_refusal_correct"])
    )

    question_mcq_counts = defaultdict(int)
    question_total_counts = defaultdict(int)
    for row in all_rows:
        question_id = row["question_id"]
        question_total_counts[question_id] += 1
        question_mcq_counts[question_id] += int(to_bool(row["mcq_without_refusal_correct"]))
    always_correct = [
        question
        for question, count in question_mcq_counts.items()
        if count == question_total_counts[question]
    ]
    always_wrong = [
        question for question, count in question_mcq_counts.items() if count == 0
    ]
    mixed = [
        question
        for question, count in question_mcq_counts.items()
        if 0 < count < question_total_counts[question]
    ]

    hardest_questions = question_scores[:12]
    refusal_gap_questions = sorted(
        [question for question in question_scores if question["refusal_gap_pp"] > 0],
        key=lambda question: (-question["refusal_gap_pp"], question["question_id"]),
    )[:12]

    latencies = [float(row["latency_ms"]) for row in all_rows if row.get("latency_ms")]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "PhyloBioBixBench-Verified-50 Results",
        "totals": {
            "rows": len(all_rows),
            "repeats": sum(meta["repeats"] for meta in source_meta),
            "questions": len(question_scores),
            "task_groups": len(task_group_scores),
        },
        "source_files": source_meta,
        "overall": overall,
        "by_file": by_file,
        "by_repeat": by_repeat,
        "question_scores": question_scores,
        "task_group_scores": task_group_scores,
        "hardest_questions": hardest_questions,
        "refusal_gap_questions": refusal_gap_questions,
        "rescue": {
            "direct_wrong": len(rows_direct_wrong),
            "direct_right": len(rows_direct_right),
            "rescued": rescued,
            "rescued_pct": round((rescued / len(rows_direct_wrong)) * 100, 1)
            if rows_direct_wrong
            else 0.0,
            "lost": lost,
            "lost_pct": round((lost / len(rows_direct_right)) * 100, 1)
            if rows_direct_right
            else 0.0,
        },
        "consistency": {
            "always_correct": len(always_correct),
            "always_wrong": len(always_wrong),
            "mixed": len(mixed),
            "always_correct_pct": round(len(always_correct) / len(question_scores) * 100, 1),
            "always_wrong_pct": round(len(always_wrong) / len(question_scores) * 100, 1),
            "mixed_pct": round(len(mixed) / len(question_scores) * 100, 1),
            "mixed_questions": sorted(mixed),
        },
        "latency": {
            "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p95_ms": round(percentile(latencies, 0.95), 1) if latencies else 0.0,
        },
        "assets": {
            "performance_plot": "assets/performance_analysis.png",
            "csv_dir": "data/results",
        },
        "notes": {
            "scoped_to_source_files": True,
            "source_markdown": SOURCE_MARKDOWN,
        },
    }


def copy_inputs(csv_paths: list[Path]) -> None:
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        shutil.copy2(csv_path, DOCS_RESULTS_DIR / csv_path.name)

    source_png = RESULTS_DIR / PERFORMANCE_PNG
    if not source_png.exists():
        raise FileNotFoundError(f"Missing image: {source_png}")
    shutil.copy2(source_png, DOCS_ASSETS_DIR / PERFORMANCE_PNG)


def main() -> None:
    csv_paths = [RESULTS_DIR / filename for filename in TARGET_CSVS]
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV: {path}")

    copy_inputs(csv_paths)
    all_rows, by_file_rows = load_rows(csv_paths)
    payload = build_payload(all_rows, by_file_rows)

    output_path = DOCS_DATA_DIR / "benchmark_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overall = payload["overall"]
    print(f"Wrote {output_path}")
    print(
        "Overall accuracy:",
        {
            "direct": overall["direct"]["pct"],
            "mcq_with_refusal": overall["mcq_with_refusal"]["pct"],
            "mcq_without_refusal": overall["mcq_without_refusal"]["pct"],
        },
    )


if __name__ == "__main__":
    main()

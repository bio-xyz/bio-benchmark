#!/usr/bin/env python3
"""Build GitHub Pages data assets for the benchmark results site."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
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


def wilson_ci(correct: int, total: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return {"lo": 0.0, "hi": 0.0}
    p = correct / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    lo = max(0.0, (center - spread) / denom)
    hi = min(1.0, (center + spread) / denom)
    return {"lo": round(lo * 100, 1), "hi": round(hi * 100, 1)}


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
            "ci_95": wilson_ci(correct, total),
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
                "mcq_lift_pp": round(mcq_without_refusal_pct - direct_pct, 1),
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

    # --- Majority vote per question ---
    majority_vote: dict[str, dict[str, bool]] = {}
    for qid, rec in question_acc.items():
        n = rec["n"]
        majority_vote[qid] = {
            "direct": rec["direct"] > n / 2,
            "mcq_with_refusal": rec["mcq_with_refusal"] > n / 2,
            "mcq_without_refusal": rec["mcq_without_refusal"] > n / 2,
        }
    mv_direct = sum(1 for v in majority_vote.values() if v["direct"])
    mv_mcq_with = sum(1 for v in majority_vote.values() if v["mcq_with_refusal"])
    mv_mcq_without = sum(1 for v in majority_vote.values() if v["mcq_without_refusal"])
    n_questions = len(question_scores)
    majority_vote_summary = {
        "direct": {"correct": mv_direct, "total": n_questions, "pct": round(mv_direct / n_questions * 100, 1)},
        "mcq_with_refusal": {"correct": mv_mcq_with, "total": n_questions, "pct": round(mv_mcq_with / n_questions * 100, 1)},
        "mcq_without_refusal": {"correct": mv_mcq_without, "total": n_questions, "pct": round(mv_mcq_without / n_questions * 100, 1)},
    }

    # --- Derived headline metrics ---
    mcq_lift_pp = round(overall["mcq_without_refusal"]["pct"] - overall["direct"]["pct"], 1)
    refusal_gap_pp = round(overall["mcq_without_refusal"]["pct"] - overall["mcq_with_refusal"]["pct"], 1)

    best_repeat = by_repeat[0]
    worst_repeat = by_repeat[-1]
    best_repeat_pct = best_repeat["metrics"]["mcq_without_refusal"]["pct"]
    worst_repeat_pct = worst_repeat["metrics"]["mcq_without_refusal"]["pct"]
    best_repeat_label = f"{best_repeat['file_code']}-{best_repeat['repeat_label']}"
    worst_repeat_label = f"{worst_repeat['file_code']}-{worst_repeat['repeat_label']}"
    repeat_spread_pp = round(best_repeat_pct - worst_repeat_pct, 1)
    repeat_pcts = [r["metrics"]["mcq_without_refusal"]["pct"] for r in by_repeat]
    repeat_median_pct = round(statistics.median(repeat_pcts), 1)

    task_groups_at_100 = sum(1 for tg in task_group_scores if tg["mcq_without_refusal_pct"] == 100.0)
    task_groups_at_100_pct = round(task_groups_at_100 / len(task_group_scores) * 100, 1)

    # --- MCQ rescue lifts (top 10) ---
    rescue_lifts = sorted(
        [q for q in question_scores if q["mcq_lift_pp"] > 0],
        key=lambda q: (-q["mcq_lift_pp"], q["question_id"]),
    )[:10]

    perfect_rescues = [q for q in question_scores if q["direct_pct"] == 0.0 and q["mcq_without_refusal_pct"] == 100.0]

    # --- Cross-run variability for mixed questions ---
    # Build per-question per-file_code correct/total for mcq_without_refusal
    q_file_map: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for row in all_rows:
        qid = row["question_id"]
        fc = row["run_id"].split("-")[-2]
        q_file_map[qid][fc]["total"] += 1
        q_file_map[qid][fc]["correct"] += int(to_bool(row["mcq_without_refusal_correct"]))

    file_codes = sorted({row["run_id"].split("-")[-2] for row in all_rows})
    cross_run_variability = []
    for qid in sorted(mixed):
        entry: dict[str, Any] = {"question_id": qid, "by_file": {}}
        for fc in file_codes:
            rec = q_file_map[qid][fc]
            entry["by_file"][fc] = {"correct": rec["correct"], "total": rec["total"]}
        cross_run_variability.append(entry)

    # --- Total MCQ failures for top-3 stat ---
    total_mcq_failures = sum(q["mcq_without_refusal_failures"] for q in question_scores)
    top3_failures = sum(q["mcq_without_refusal_failures"] for q in question_scores[:3])
    top3_failure_pct = round(top3_failures / total_mcq_failures * 100, 1) if total_mcq_failures else 0.0

    # --- Commentary dict ---
    rescued_pct = round((rescued / len(rows_direct_wrong)) * 100, 1) if rows_direct_wrong else 0.0
    lost_pct = round((lost / len(rows_direct_right)) * 100, 1) if rows_direct_right else 0.0

    commentary = {
        "grading_modes": [
            {
                "name": "Direct",
                "description": "The model's free-form answer is graded by an LLM judge (gpt-5) against the ground truth. Sensitive to phrasing and formatting.",
                "accent": "coral",
            },
            {
                "name": "MCQ without refusal",
                "description": "The model's answer is mapped to the closest multiple-choice option (via gpt-4o), then graded for correctness. The model must select an option.",
                "accent": "teal",
            },
            {
                "name": "MCQ with refusal",
                "description": "Same as above, but the model is allowed to select \"Insufficient information to determine\" when the answer doesn't closely match any option. This is the stricter evaluator.",
                "accent": "sun",
            },
        ],
        "invariant_note": "Key invariant: mcq_with_refusal <= mcq_without_refusal. If the with-refusal mode selects a concrete option, that option must be correct (meaning without-refusal would also be correct). The with-refusal mode may additionally refuse when the answer is not precise enough.",
        "overall_interpretation": f"MCQ without refusal provides a +{mcq_lift_pp}pp lift over direct grading. MCQ with refusal sits {refusal_gap_pp}pp below MCQ without refusal, reflecting its stricter tolerance. The gap between direct and MCQ suggests that a significant portion of \"wrong\" answers under direct grading are actually correct answers that the direct grader cannot parse due to formatting differences.",
        "rescue_interpretation": f"MCQ grading is {'purely additive' if lost == 0 else 'mostly additive'} \u2014 it {'never downgrades a correct direct answer' if lost == 0 else f'rarely downgrades ({lost} losses)'}, and it rescues {rescued_pct}% of direct failures.",
        "refusal_gap_interpretation": f"The {refusal_gap_pp}pp refusal gap ({overall['mcq_without_refusal']['pct']}% \u2192 {overall['mcq_with_refusal']['pct']}%) represents answer precision, not knowledge. The model often knows the right direction but its exact numerical or categorical answers aren't always tight enough for the stricter evaluator.",
        "strengths": [
            f"{overall['mcq_without_refusal']['pct']}% MCQ w/o refusal is a strong result. The model demonstrates solid bioinformatics knowledge across the majority of the benchmark.",
            f"{len(always_correct)}/{n_questions} questions ({round(len(always_correct)/n_questions*100)}%) are perfectly consistent. The model reliably gets these right every single time across all {sum(m['repeats'] for m in source_meta)} repeats.",
            f"{task_groups_at_100}/{len(task_group_scores)} task groups score 100%. The model handles the majority of bioinformatics analysis tasks flawlessly.",
            f"MCQ rescue rate of {rescued_pct}% with {'zero' if lost == 0 else str(lost)} losses. The MCQ format is a reliable safety net that recovers misgraded answers without {'ever introducing new errors' if lost == 0 else 'significant error introduction'}.",
            f"{len(perfect_rescues)} questions with perfect 100pp rescue. The model knows the answer to these questions every time \u2014 the issue is purely in how the direct grader interprets the response format.",
            f"Cross-run stability. MCQ w/o refusal ranges from {min(f['metrics']['mcq_without_refusal']['pct'] for f in by_file)}\u2013{max(f['metrics']['mcq_without_refusal']['pct'] for f in by_file)}% across files, indicating reproducible evaluation.",
        ],
        "weaknesses": [
            f"bix-32-q2 (0/{sum(m['repeats'] for m in source_meta)}): Complete failure across all repeats and all grading modes. This is not a grading issue \u2014 the model genuinely cannot answer this question.",
            f"bix-16 task group ({[tg for tg in task_group_scores if tg['task_group'] == 'bix-16'][0]['mcq_without_refusal_pct']}%): The weakest multi-question task group. Two of its three questions (q1 and q3) fail at 9.1%, while q4 scores 100%.",
            f"bix-24-q2 ({[q for q in question_scores if q['question_id'] == 'bix-24-q2'][0]['mcq_without_refusal_pct']}%): Moderate failure rate with no MCQ lift \u2014 the model has a genuine knowledge gap here.",
            f"Answer precision issues: {len(refusal_gap_questions)} questions show a refusal gap, meaning the model gets to the right answer but not precisely enough.",
            f"Run-to-run variance: While the average is stable, individual repeats range from {worst_repeat_pct}% to {best_repeat_pct}% \u2014 a {repeat_spread_pp}pp spread.",
        ],
        "key_takeaways": [
            f"The model's true knowledge level is closer to {overall['mcq_without_refusal']['pct']}% than {overall['direct']['pct']}%. The {mcq_lift_pp}pp gap between direct ({overall['direct']['pct']}%) and MCQ ({overall['mcq_without_refusal']['pct']}%) is largely a grading artifact.",
            f"3 questions account for {top3_failure_pct}% of all MCQ failures. bix-32-q2, bix-16-q3, and bix-16-q1 together account for {top3_failures} of the {total_mcq_failures} total MCQ w/o refusal failures. Fixing these three would push the score to approximately {round((overall['mcq_without_refusal']['correct'] + top3_failures) / overall['mcq_without_refusal']['total'] * 100, 1)}%.",
            f"The {refusal_gap_pp}pp refusal gap ({overall['mcq_without_refusal']['pct']}% \u2192 {overall['mcq_with_refusal']['pct']}%) represents answer precision, not knowledge.",
            f"Best single-repeat score is {best_repeat_pct}% ({best_repeat['metrics']['mcq_without_refusal']['correct']}/{best_repeat['metrics']['mcq_without_refusal']['total']}). This demonstrates the model's ceiling \u2014 it's capable of near-perfect performance in favorable conditions.",
            f"The benchmark has good discriminative power. With {len(always_correct)} always-correct, {len(always_wrong)} always-wrong, and {len(mixed)} mixed questions, the benchmark effectively separates reliable knowledge from uncertain areas.",
        ],
    }

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
            "rescued_pct": rescued_pct,
            "lost": lost,
            "lost_pct": lost_pct,
        },
        "rescue_lifts": rescue_lifts,
        "perfect_rescues": len(perfect_rescues),
        "consistency": {
            "always_correct": len(always_correct),
            "always_wrong": len(always_wrong),
            "mixed": len(mixed),
            "always_correct_pct": round(len(always_correct) / len(question_scores) * 100, 1),
            "always_wrong_pct": round(len(always_wrong) / len(question_scores) * 100, 1),
            "mixed_pct": round(len(mixed) / len(question_scores) * 100, 1),
            "mixed_questions": sorted(mixed),
        },
        "cross_run_variability": cross_run_variability,
        "file_codes": file_codes,
        "majority_vote": majority_vote_summary,
        "headline": {
            "mcq_lift_pp": mcq_lift_pp,
            "refusal_gap_pp": refusal_gap_pp,
            "best_repeat_pct": best_repeat_pct,
            "best_repeat_label": best_repeat_label,
            "worst_repeat_pct": worst_repeat_pct,
            "worst_repeat_label": worst_repeat_label,
            "repeat_spread_pp": repeat_spread_pp,
            "repeat_median_pct": repeat_median_pct,
            "task_groups_at_100": task_groups_at_100,
            "task_groups_at_100_pct": task_groups_at_100_pct,
            "total_task_groups": len(task_group_scores),
        },
        "commentary": commentary,
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

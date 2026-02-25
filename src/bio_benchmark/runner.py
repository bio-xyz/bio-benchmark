from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .agent import AgentTaskResult, run_data_analysis_question
from .config import BenchmarkConfig
from .judge import BatchThreeModeEvaluation, GradeResult, evaluate_batch_three_modes
from .strategies import get_strategy
from .strategies.base import BenchmarkRow, PreparedTaskBatch
from .template import render_template_string


@dataclass
class RunSummary:
    total: int
    passed: int
    failed: int
    direct_correct: int
    direct_incorrect: int
    mcq_with_refusal_correct: int
    mcq_with_refusal_incorrect: int
    mcq_without_refusal_correct: int
    mcq_without_refusal_incorrect: int
    request_errors: int
    judge_errors: int
    output_csv: str


def _retry_delay(previous_delay: float, multiplier: float, max_delay: float) -> float:
    return min(previous_delay * multiplier, max_delay)


def _default_grade(question_id: str, reason: str = "Missing grade") -> GradeResult:
    return GradeResult(question_id=question_id, correct=False, reasoning=reason)


def _run_task_with_retry(
    config: BenchmarkConfig,
    task_group: str,
    prompt_text: str,
    local_data_dirs: list[str],
    event_logger: Callable[[str], None] | None = None,
) -> tuple[AgentTaskResult | None, str | None, int]:
    delay = max(0.0, config.retry.initial_delay_seconds)
    last_error: str | None = None

    for attempt in range(1, config.retry.max_attempts + 1):
        try:
            result = run_data_analysis_question(
                config.agent,
                prompt_text,
                local_data_dirs,
                task_label=task_group,
                event_logger=event_logger,
            )
            return result, None, attempt
        except Exception as exc:
            last_error = str(exc)
            if attempt >= config.retry.max_attempts:
                break
            if delay > 0:
                time.sleep(delay)
            delay = _retry_delay(
                delay,
                config.retry.backoff_multiplier,
                config.retry.max_delay_seconds,
            )

    return None, last_error, config.retry.max_attempts


def _build_error_row(
    *,
    timestamp_utc: str,
    config: BenchmarkConfig,
    repeat_index: int,
    row: BenchmarkRow,
    attempts: int,
    request_error: str,
) -> dict[str, object]:
    return {
        "timestamp_utc": timestamp_utc,
        "benchmark": config.benchmark,
        "run_id": config.run_id,
        "repeat_index": repeat_index,
        "test": config.test,
        "question_id": row.question_id,
        "task_group": row.task_group,
        "status": "request_error",
        "success": False,
        "attempts": attempts,
        "uploaded_files": 0,
        "poll_count": 0,
        "latency_ms": 0,
        "error": request_error,
        "question": row.question_text,
        "ground_truth": row.ground_truth,
        "direct_answer": "",
        "answer": "",
        "direct_correct": False,
        "direct_reasoning": "",
        "mcq_with_refusal_correct": False,
        "mcq_with_refusal_reasoning": "",
        "mcq_with_refusal_mapped_answer": "",
        "mcq_with_refusal_options": "[]",
        "mcq_without_refusal_correct": False,
        "mcq_without_refusal_reasoning": "",
        "mcq_without_refusal_mapped_answer": "",
        "mcq_without_refusal_options": "[]",
        "grader_model": config.judge.grader_model,
        "option_chooser_model": config.judge.option_chooser_model,
        "judge_error": "",
        "api_task_id": "",
        "data_folder": row.data_folder,
        "local_data_dir": row.local_data_dir,
    }


def _build_success_row(
    *,
    timestamp_utc: str,
    config: BenchmarkConfig,
    repeat_index: int,
    row: BenchmarkRow,
    task_result: AgentTaskResult,
    attempts: int,
    evaluation: BatchThreeModeEvaluation,
    direct_grade: GradeResult,
    mcq_with_refusal_grade: GradeResult,
    mcq_without_refusal_grade: GradeResult,
) -> dict[str, object]:
    return {
        "timestamp_utc": timestamp_utc,
        "benchmark": config.benchmark,
        "run_id": config.run_id,
        "repeat_index": repeat_index,
        "test": config.test,
        "question_id": row.question_id,
        "task_group": row.task_group,
        "status": task_result.status,
        "success": task_result.success,
        "attempts": attempts,
        "uploaded_files": task_result.uploaded_files,
        "poll_count": task_result.poll_count,
        "latency_ms": task_result.latency_ms,
        "error": "",
        "question": row.question_text,
        "ground_truth": row.ground_truth,
        "direct_answer": task_result.direct_answer,
        "answer": task_result.answer,
        "direct_correct": direct_grade.correct,
        "direct_reasoning": direct_grade.reasoning,
        "mcq_with_refusal_correct": mcq_with_refusal_grade.correct,
        "mcq_with_refusal_reasoning": mcq_with_refusal_grade.reasoning,
        "mcq_with_refusal_mapped_answer": evaluation.mapped_with_refusal,
        "mcq_with_refusal_options": json.dumps(
            evaluation.options_with_refusal_by_question_id.get(row.question_id, []),
            ensure_ascii=False,
        ),
        "mcq_without_refusal_correct": mcq_without_refusal_grade.correct,
        "mcq_without_refusal_reasoning": mcq_without_refusal_grade.reasoning,
        "mcq_without_refusal_mapped_answer": evaluation.mapped_without_refusal,
        "mcq_without_refusal_options": json.dumps(
            evaluation.options_without_refusal_by_question_id.get(row.question_id, []),
            ensure_ascii=False,
        ),
        "grader_model": evaluation.grader_model,
        "option_chooser_model": evaluation.option_chooser_model,
        "judge_error": evaluation.judge_error,
        "api_task_id": task_result.task_id,
        "data_folder": row.data_folder,
        "local_data_dir": row.local_data_dir,
    }


def _short_error(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.*\S)?\s*$")


def _parse_numbered_answers(text: str) -> dict[int, str]:
    output: dict[int, str] = {}
    for line in text.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if not match:
            continue
        idx = int(match.group(1))
        output[idx] = (match.group(2) or "").strip()
    return output


def _normalize_cell_text(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _bool_cell(value: bool) -> str:
    return "[green]yes[/green]" if value else "[red]no[/red]"


def _ratio_pct(correct: int, incorrect: int) -> str:
    total = correct + incorrect
    if total == 0:
        return "0/0 (0.0%)"
    return f"{correct}/{total} ({(correct / total) * 100:.1f}%)"


def _repeat_progress_description(
    *,
    repeat_index: int,
    direct_correct: int,
    direct_incorrect: int,
    mcq_refusal_correct: int,
    mcq_refusal_incorrect: int,
    mcq_no_refusal_correct: int,
    mcq_no_refusal_incorrect: int,
) -> str:
    return (
        f"repeat {repeat_index} task groups "
        f"| direct {_ratio_pct(direct_correct, direct_incorrect)} "
        f"| mcq_refusal {_ratio_pct(mcq_refusal_correct, mcq_refusal_incorrect)} "
        f"| mcq_no_refusal {_ratio_pct(mcq_no_refusal_correct, mcq_no_refusal_incorrect)}"
    )


def run_benchmark(
    config: BenchmarkConfig,
    limit: int | None = None,
    console: Console | None = None,
) -> RunSummary:
    console = console or Console()

    strategy = get_strategy(config.strategy)
    console.rule(f"Benchmark Run: {config.benchmark}")
    console.log(
        f"strategy={config.strategy} test={config.test} concurrency={config.concurrency} repeats={config.repeats}"
    )
    rows = strategy.load_rows(config, limit=limit, event_logger=console.log)
    console.log(f"loaded_rows={len(rows)}")

    output_path = Path(
        render_template_string(config.output.csv_path, {"run_id": config.run_id})
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_utc",
        "benchmark",
        "run_id",
        "repeat_index",
        "test",
        "question_id",
        "task_group",
        "status",
        "success",
        "attempts",
        "uploaded_files",
        "poll_count",
        "latency_ms",
        "error",
        "question",
        "ground_truth",
        "direct_answer",
        "answer",
        "direct_correct",
        "direct_reasoning",
        "mcq_with_refusal_correct",
        "mcq_with_refusal_reasoning",
        "mcq_with_refusal_mapped_answer",
        "mcq_with_refusal_options",
        "mcq_without_refusal_correct",
        "mcq_without_refusal_reasoning",
        "mcq_without_refusal_mapped_answer",
        "mcq_without_refusal_options",
        "grader_model",
        "option_chooser_model",
        "judge_error",
        "api_task_id",
        "data_folder",
        "local_data_dir",
    ]

    total = 0
    direct_correct = 0
    direct_incorrect = 0
    mcq_with_refusal_correct = 0
    mcq_with_refusal_incorrect = 0
    mcq_without_refusal_correct = 0
    mcq_without_refusal_incorrect = 0
    request_errors = 0
    judge_errors = 0

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for repeat_index in range(1, config.repeats + 1):
            repeat_run_id = (
                config.run_id
                if config.repeats == 1
                else f"{config.run_id}-r{repeat_index}"
            )
            repeat_direct_correct = 0
            repeat_direct_incorrect = 0
            repeat_mcq_refusal_correct = 0
            repeat_mcq_refusal_incorrect = 0
            repeat_mcq_no_refusal_correct = 0
            repeat_mcq_no_refusal_incorrect = 0
            repeat_config = replace(config, run_id=repeat_run_id)
            batches = strategy.prepare_batches(repeat_config, rows)
            console.rule(f"Repeat {repeat_index}/{config.repeats}")
            console.log(f"task_groups={len(batches)} run_id={repeat_run_id}")
            if not batches:
                continue

            max_workers = min(max(1, repeat_config.concurrency), max(1, len(batches)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        _run_task_with_retry,
                        repeat_config,
                        batch.task_group,
                        batch.prompt_text,
                        batch.local_data_dirs,
                        console.log,
                    ): batch
                    for batch in batches
                }
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,
                ) as progress:
                    repeat_task = progress.add_task(
                        _repeat_progress_description(
                            repeat_index=repeat_index,
                            direct_correct=repeat_direct_correct,
                            direct_incorrect=repeat_direct_incorrect,
                            mcq_refusal_correct=repeat_mcq_refusal_correct,
                            mcq_refusal_incorrect=repeat_mcq_refusal_incorrect,
                            mcq_no_refusal_correct=repeat_mcq_no_refusal_correct,
                            mcq_no_refusal_incorrect=repeat_mcq_no_refusal_incorrect,
                        ),
                        total=len(batches),
                    )
                    for future in as_completed(future_map):
                        batch: PreparedTaskBatch = future_map[future]
                        timestamp_utc = datetime.now(UTC).isoformat()
                        total += len(batch.rows)

                        try:
                            task_result, request_error, attempts = future.result()
                        except Exception as exc:
                            task_result, request_error, attempts = (
                                None,
                                f"Worker failed: {exc}",
                                repeat_config.retry.max_attempts,
                            )

                        if task_result is None:
                            request_errors += len(batch.rows)
                            direct_incorrect += len(batch.rows)
                            mcq_with_refusal_incorrect += len(batch.rows)
                            mcq_without_refusal_incorrect += len(batch.rows)
                            repeat_direct_incorrect += len(batch.rows)
                            repeat_mcq_refusal_incorrect += len(batch.rows)
                            repeat_mcq_no_refusal_incorrect += len(batch.rows)
                            for row in batch.rows:
                                writer.writerow(
                                    _build_error_row(
                                        timestamp_utc=timestamp_utc,
                                        config=repeat_config,
                                        repeat_index=repeat_index,
                                        row=row,
                                        attempts=attempts,
                                        request_error=request_error
                                        or "unknown request error",
                                    )
                                )
                            console.log(
                                f"[red]task={batch.task_group} request_error[/red] "
                                f"attempts={attempts} "
                                f"questions={len(batch.rows)} "
                                f"error={_short_error(request_error or 'unknown')}"
                            )
                            progress.update(
                                repeat_task,
                                description=_repeat_progress_description(
                                    repeat_index=repeat_index,
                                    direct_correct=repeat_direct_correct,
                                    direct_incorrect=repeat_direct_incorrect,
                                    mcq_refusal_correct=repeat_mcq_refusal_correct,
                                    mcq_refusal_incorrect=repeat_mcq_refusal_incorrect,
                                    mcq_no_refusal_correct=repeat_mcq_no_refusal_correct,
                                    mcq_no_refusal_incorrect=repeat_mcq_no_refusal_incorrect,
                                ),
                            )
                            progress.advance(repeat_task, 1)
                            continue

                        candidate_answer = task_result.direct_answer or task_result.answer
                        evaluation = evaluate_batch_three_modes(
                            config=repeat_config.judge,
                            questions=[
                                {
                                    "question_id": row.question_id,
                                    "question": row.question_text,
                                    "ground_truth": row.ground_truth,
                                    "distractors": row.distractors,
                                }
                                for row in batch.rows
                            ],
                            agent_response=candidate_answer,
                        )
                        if evaluation.judge_error:
                            judge_errors += len(batch.rows)
                            console.log(
                                f"[yellow]task={batch.task_group} judge_error[/yellow] "
                                f"{_short_error(evaluation.judge_error)}"
                            )

                        batch_direct_correct = 0
                        batch_mcq_refusal_correct = 0
                        batch_mcq_no_refusal_correct = 0
                        direct_response_text = (
                            task_result.direct_answer.strip()
                            if task_result.direct_answer.strip()
                            else task_result.answer.strip()
                        )
                        direct_items = _parse_numbered_answers(direct_response_text)
                        mapped_with_refusal_items = _parse_numbered_answers(
                            evaluation.mapped_with_refusal
                        )
                        mapped_without_refusal_items = _parse_numbered_answers(
                            evaluation.mapped_without_refusal
                        )
                        if (
                            not direct_items
                            and len(batch.rows) == 1
                            and direct_response_text
                        ):
                            direct_items[1] = direct_response_text
                        if (
                            not mapped_with_refusal_items
                            and len(batch.rows) == 1
                            and evaluation.mapped_with_refusal.strip()
                        ):
                            mapped_with_refusal_items[1] = (
                                evaluation.mapped_with_refusal.strip()
                            )
                        if (
                            not mapped_without_refusal_items
                            and len(batch.rows) == 1
                            and evaluation.mapped_without_refusal.strip()
                        ):
                            mapped_without_refusal_items[1] = (
                                evaluation.mapped_without_refusal.strip()
                            )

                        for row in batch.rows:
                            direct_grade = evaluation.direct_by_question_id.get(
                                row.question_id, _default_grade(row.question_id)
                            )
                            mcq_with_refusal_grade = (
                                evaluation.mcq_with_refusal_by_question_id.get(
                                    row.question_id,
                                    _default_grade(row.question_id),
                                )
                            )
                            mcq_without_refusal_grade = (
                                evaluation.mcq_without_refusal_by_question_id.get(
                                    row.question_id,
                                    _default_grade(row.question_id),
                                )
                            )

                            if direct_grade.correct:
                                direct_correct += 1
                                batch_direct_correct += 1
                                repeat_direct_correct += 1
                            else:
                                direct_incorrect += 1
                                repeat_direct_incorrect += 1

                            if mcq_with_refusal_grade.correct:
                                mcq_with_refusal_correct += 1
                                batch_mcq_refusal_correct += 1
                                repeat_mcq_refusal_correct += 1
                            else:
                                mcq_with_refusal_incorrect += 1
                                repeat_mcq_refusal_incorrect += 1

                            if mcq_without_refusal_grade.correct:
                                mcq_without_refusal_correct += 1
                                batch_mcq_no_refusal_correct += 1
                                repeat_mcq_no_refusal_correct += 1
                            else:
                                mcq_without_refusal_incorrect += 1
                                repeat_mcq_no_refusal_incorrect += 1

                            writer.writerow(
                                _build_success_row(
                                    timestamp_utc=timestamp_utc,
                                    config=repeat_config,
                                    repeat_index=repeat_index,
                                    row=row,
                                    task_result=task_result,
                                    attempts=attempts,
                                    evaluation=evaluation,
                                    direct_grade=direct_grade,
                                    mcq_with_refusal_grade=mcq_with_refusal_grade,
                                    mcq_without_refusal_grade=mcq_without_refusal_grade,
                                )
                            )

                        if direct_response_text:
                            console.print(
                                Panel(
                                    _short_error(direct_response_text, limit=2500),
                                    title=f"task={batch.task_group} directAnswer",
                                    border_style="blue",
                                    expand=False,
                                )
                            )
                        else:
                            console.log(
                                f"[yellow]task={batch.task_group} directAnswer empty[/yellow]"
                            )

                        details = Table(
                            title=f"task={batch.task_group} per-question results",
                            show_lines=True,
                        )
                        details.add_column("#", justify="right")
                        details.add_column("question_id")
                        details.add_column("ground_truth")
                        details.add_column("direct_answer")
                        details.add_column("direct_ok", justify="center")
                        details.add_column("mapped_refusal")
                        details.add_column("mcq_refusal_ok", justify="center")
                        details.add_column("mapped_no_refusal")
                        details.add_column("mcq_no_refusal_ok", justify="center")

                        for idx, row in enumerate(batch.rows, start=1):
                            direct_grade = evaluation.direct_by_question_id.get(
                                row.question_id, _default_grade(row.question_id)
                            )
                            mcq_with_refusal_grade = (
                                evaluation.mcq_with_refusal_by_question_id.get(
                                    row.question_id,
                                    _default_grade(row.question_id),
                                )
                            )
                            mcq_without_refusal_grade = (
                                evaluation.mcq_without_refusal_by_question_id.get(
                                    row.question_id,
                                    _default_grade(row.question_id),
                                )
                            )
                            details.add_row(
                                str(idx),
                                row.question_id,
                                _normalize_cell_text(row.ground_truth),
                                _normalize_cell_text(direct_items.get(idx, "")),
                                _bool_cell(direct_grade.correct),
                                _normalize_cell_text(
                                    mapped_with_refusal_items.get(idx, "")
                                ),
                                _bool_cell(mcq_with_refusal_grade.correct),
                                _normalize_cell_text(
                                    mapped_without_refusal_items.get(idx, "")
                                ),
                                _bool_cell(mcq_without_refusal_grade.correct),
                            )
                        console.print(details)

                        console.log(
                            f"[green]task={batch.task_group} completed[/green] "
                            f"attempts={attempts} uploaded={task_result.uploaded_files} "
                            f"polls={task_result.poll_count} "
                            f"direct={batch_direct_correct}/{len(batch.rows)} "
                            f"mcq_refusal={batch_mcq_refusal_correct}/{len(batch.rows)} "
                            f"mcq_no_refusal={batch_mcq_no_refusal_correct}/{len(batch.rows)}"
                        )
                        progress.update(
                            repeat_task,
                            description=_repeat_progress_description(
                                repeat_index=repeat_index,
                                direct_correct=repeat_direct_correct,
                                direct_incorrect=repeat_direct_incorrect,
                                mcq_refusal_correct=repeat_mcq_refusal_correct,
                                mcq_refusal_incorrect=repeat_mcq_refusal_incorrect,
                                mcq_no_refusal_correct=repeat_mcq_no_refusal_correct,
                                mcq_no_refusal_incorrect=repeat_mcq_no_refusal_incorrect,
                            ),
                        )
                        progress.advance(repeat_task, 1)

    return RunSummary(
        total=total,
        passed=direct_correct,
        failed=direct_incorrect,
        direct_correct=direct_correct,
        direct_incorrect=direct_incorrect,
        mcq_with_refusal_correct=mcq_with_refusal_correct,
        mcq_with_refusal_incorrect=mcq_with_refusal_incorrect,
        mcq_without_refusal_correct=mcq_without_refusal_correct,
        mcq_without_refusal_incorrect=mcq_without_refusal_incorrect,
        request_errors=request_errors,
        judge_errors=judge_errors,
        output_csv=str(output_path),
    )

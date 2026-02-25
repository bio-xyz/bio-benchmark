from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

from ..config import BenchmarkConfig
from ..template import render_template_string
from .base import BenchmarkRow, BenchmarkStrategy, PreparedTaskBatch
from .bixbench_dataset import load_bixbench_rows


def _get_str(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _group_by_task(rows: list[BenchmarkRow]) -> OrderedDict[str, list[BenchmarkRow]]:
    grouped: OrderedDict[str, list[BenchmarkRow]] = OrderedDict()
    for index, row in enumerate(rows, start=1):
        task_group = row.task_group or f"group-{index}"
        if task_group not in grouped:
            grouped[task_group] = []
        grouped[task_group].append(row)
    return grouped


def _collect_local_data_dirs(rows: list[BenchmarkRow]) -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.local_data_dir:
            continue
        if row.local_data_dir in seen:
            continue
        seen.add(row.local_data_dir)
        dirs.append(row.local_data_dir)
    return dirs


def _list_available_datasets(local_data_dirs: list[str]) -> str:
    names: list[str] = []
    seen: set[str] = set()
    for local_data_dir in local_data_dirs:
        root = Path(local_data_dir)
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            label = entry.name + ("/" if entry.is_dir() else "")
            if label in seen:
                continue
            seen.add(label)
            names.append(label)
    return ", ".join(names)


def _find_hypothesis(rows: list[BenchmarkRow]) -> str:
    for row in rows:
        if row.hypothesis:
            return row.hypothesis
    return ""


def _build_questions_block(rows: list[BenchmarkRow]) -> str:
    return "\n".join(f"{i}. {row.question_text}" for i, row in enumerate(rows, start=1))


class BixBenchStrategy(BenchmarkStrategy):
    name = "bixbench"

    def load_rows(
        self,
        config: BenchmarkConfig,
        *,
        limit: int | None = None,
        event_logger: Callable[[str], None] | None = None,
    ) -> list[BenchmarkRow]:
        raw_rows = load_bixbench_rows(
            config.dataset,
            cli_limit=limit,
            event_logger=event_logger,
        )
        output: list[BenchmarkRow] = []
        for index, row in enumerate(raw_rows, start=1):
            question_id = _get_str(
                row.get(config.dataset.question_id_field),
                fallback=f"q-{index}",
            )
            task_group = _get_str(
                row.get(config.dataset.task_id_field),
                fallback=f"group-{index}",
            )
            output.append(
                BenchmarkRow(
                    raw=row,
                    question_id=question_id,
                    task_group=task_group,
                    question_text=_get_str(row.get(config.dataset.question_field), ""),
                    ground_truth=_get_str(
                        row.get(config.dataset.ground_truth_field), ""
                    ),
                    distractors=row.get("distractors"),
                    data_folder=_get_str(row.get(config.dataset.data_folder_field), ""),
                    local_data_dir=_get_str(row.get("_local_data_dir"), ""),
                    hypothesis=_get_str(row.get("hypothesis"), ""),
                )
            )
        return output

    def prepare_batches(
        self,
        config: BenchmarkConfig,
        rows: list[BenchmarkRow],
    ) -> list[PreparedTaskBatch]:
        batches: list[PreparedTaskBatch] = []
        grouped = _group_by_task(rows)
        for task_group, task_rows in grouped.items():
            local_data_dirs = _collect_local_data_dirs(task_rows)
            prompt_text = render_template_string(
                config.prompt.template,
                {
                    **task_rows[0].raw,
                    "task_id": task_group,
                    "question_text": task_rows[0].question_text,
                    "questions_block": _build_questions_block(task_rows),
                    "question_count": len(task_rows),
                    "research_hypothesis": _find_hypothesis(task_rows),
                    "available_datasets": _list_available_datasets(local_data_dirs),
                    "benchmark": config.benchmark,
                    "run_id": config.run_id,
                },
            )
            batches.append(
                PreparedTaskBatch(
                    task_group=task_group,
                    rows=task_rows,
                    prompt_text=prompt_text,
                    local_data_dirs=local_data_dirs,
                )
            )
        return batches

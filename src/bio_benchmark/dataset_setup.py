from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

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

from .agent import warm_file_cache
from .config import BenchmarkConfig
from .strategies.base import PreparedTaskBatch


@dataclass
class DatasetSetupSummary:
    bundles: int
    cache_hits: int
    cache_misses: int
    misses_with_uploads: int
    misses_without_uploads: int
    setup_failures: int
    files_discovered: int
    files_uploaded: int


def _normalize_dirs(local_data_dirs: list[str]) -> tuple[str, ...]:
    unique: set[str] = set()
    output: list[str] = []
    for local_data_dir in local_data_dirs:
        if not local_data_dir:
            continue
        resolved = str(Path(local_data_dir).resolve())
        if resolved in unique:
            continue
        unique.add(resolved)
        output.append(resolved)
    output.sort()
    return tuple(output)


def _collect_setup_targets(
    batches: list[PreparedTaskBatch],
) -> list[tuple[str, list[str]]]:
    seen: set[tuple[str, ...]] = set()
    output: list[tuple[str, list[str]]] = []
    for batch in batches:
        normalized = _normalize_dirs(batch.local_data_dirs)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append((batch.task_group, list(normalized)))
    return output


def run_dataset_setup(
    config: BenchmarkConfig,
    batches: list[PreparedTaskBatch],
    *,
    console: Console | None = None,
) -> DatasetSetupSummary:
    console = console or Console()
    targets = _collect_setup_targets(batches)
    if not targets:
        console.log("[yellow]dataset setup skipped[/yellow] no local data bundles found")
        return DatasetSetupSummary(
            bundles=0,
            cache_hits=0,
            cache_misses=0,
            misses_with_uploads=0,
            misses_without_uploads=0,
            setup_failures=0,
            files_discovered=0,
            files_uploaded=0,
        )

    cache_hits = 0
    cache_misses = 0
    misses_with_uploads = 0
    misses_without_uploads = 0
    setup_failures = 0
    files_discovered = 0
    files_uploaded = 0
    max_workers = min(max(1, config.concurrency), len(targets))

    console.rule("Dataset Setup")
    console.log(f"bundles={len(targets)} concurrency={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                warm_file_cache,
                config.agent,
                local_data_dirs,
                task_label=f"setup:{task_group}",
                event_logger=console.log,
            ): task_group
            for task_group, local_data_dirs in targets
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
            setup_task = progress.add_task("priming dataset bundles", total=len(targets))
            for future in as_completed(future_map):
                task_group = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    setup_failures += 1
                    console.log(
                        f"[red]dataset setup failed[/red] task={task_group} error={exc}"
                    )
                    progress.advance(setup_task, 1)
                    continue

                files_discovered += result.discovered_files
                files_uploaded += result.uploaded_files
                if result.cache_hit:
                    cache_hits += 1
                    cache_state = "hit"
                else:
                    cache_misses += 1
                    if result.uploaded_files > 0:
                        misses_with_uploads += 1
                    else:
                        misses_without_uploads += 1
                    cache_state = "miss"
                console.log(
                    f"dataset setup task={task_group} cache={cache_state} "
                    f"files={result.discovered_files} uploaded={result.uploaded_files}"
                )
                progress.advance(setup_task, 1)

    summary = DatasetSetupSummary(
        bundles=len(targets),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        misses_with_uploads=misses_with_uploads,
        misses_without_uploads=misses_without_uploads,
        setup_failures=setup_failures,
        files_discovered=files_discovered,
        files_uploaded=files_uploaded,
    )
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Bundles: {summary.bundles}",
                    f"Cache hits: {summary.cache_hits}",
                    f"Cache misses: {summary.cache_misses}",
                    f"Misses with uploads: {summary.misses_with_uploads}",
                    f"Misses without uploads: {summary.misses_without_uploads}",
                    f"Setup failures: {summary.setup_failures}",
                    f"Files discovered: {summary.files_discovered}",
                    f"Files uploaded: {summary.files_uploaded}",
                ]
            ),
            title="Dataset Setup Summary",
        )
    )
    return summary

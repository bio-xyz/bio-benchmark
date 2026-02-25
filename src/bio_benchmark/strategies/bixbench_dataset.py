from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ..config import DatasetConfig


def _emit(
    event_logger: Callable[[str], None] | None,
    message: str,
) -> None:
    if event_logger:
        event_logger(message)


def _extract_and_process_capsule(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(str(zip_path), str(extract_dir))

    data_dir = next(
        (path for path in extract_dir.rglob("*") if path.is_dir() and "Data" in path.name),
        None,
    )
    if data_dir is not None:
        for item in data_dir.iterdir():
            shutil.move(str(item), str(extract_dir / item.name))
        shutil.rmtree(data_dir, ignore_errors=True)

    notebook_dir = next(
        (
            path
            for path in extract_dir.rglob("*")
            if path.is_dir() and "Notebook" in path.name
        ),
        None,
    )
    if notebook_dir is not None:
        shutil.rmtree(notebook_dir, ignore_errors=True)

    for ipynb_file in extract_dir.glob("*.ipynb"):
        ipynb_file.unlink(missing_ok=True)


def _ensure_capsule_dir(
    *,
    repo_id: str,
    zip_filename: str,
    cache_dir: Path,
    local_data_dir: Path,
    hf_token: str | None,
    event_logger: Callable[[str], None] | None,
) -> Path:
    extract_dir = local_data_dir / zip_filename.replace(".zip", "")
    if extract_dir.exists() and any(extract_dir.iterdir()):
        _emit(event_logger, f"[cyan]capsule cache hit[/cyan] {zip_filename}")
        return extract_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_data_dir.mkdir(parents=True, exist_ok=True)

    _emit(event_logger, f"[cyan]downloading capsule[/cyan] {zip_filename}")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_filename,
        local_dir=str(cache_dir),
        repo_type="dataset",
        token=hf_token,
    )
    _emit(event_logger, f"[cyan]extracting capsule[/cyan] {zip_filename}")
    _extract_and_process_capsule(Path(zip_path), extract_dir)
    _emit(event_logger, f"[green]capsule ready[/green] {extract_dir}")
    return extract_dir


def load_bixbench_rows(
    config: DatasetConfig,
    cli_limit: int | None = None,
    event_logger: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Load BixBench rows and attach local extracted capsule directory per row."""
    hf_token = os.getenv(config.hf_token_env) if config.hf_token_env else None
    _emit(
        event_logger,
        f"loading dataset repo={config.repo_id} split={config.split}",
    )
    load_kwargs: dict[str, Any] = {
        "path": config.repo_id,
        "split": config.split,
    }
    if config.data_files:
        load_kwargs["data_files"] = config.data_files
    if hf_token:
        load_kwargs["token"] = hf_token
    dataset = load_dataset(**load_kwargs)
    _emit(event_logger, f"dataset loaded rows={len(dataset)}")

    limit = cli_limit if cli_limit is not None else config.limit
    if limit is not None:
        limit = max(0, min(limit, len(dataset)))
        dataset = dataset.select(range(limit))
        _emit(event_logger, f"applying limit rows={len(dataset)}")

    rows = [dict(row) for row in dataset]
    data_folder_field = config.data_folder_field
    zip_filenames = {
        str(row[data_folder_field])
        for row in rows
        if row.get(data_folder_field) and str(row.get(data_folder_field)).endswith(".zip")
    }
    _emit(event_logger, f"capsules to prepare={len(zip_filenames)}")

    cache_dir = Path(config.cache_dir)
    local_data_dir = Path(config.local_data_dir)

    extracted_dirs: dict[str, Path] = {}
    for zip_filename in sorted(zip_filenames):
        extracted_dirs[zip_filename] = _ensure_capsule_dir(
            repo_id=config.repo_id,
            zip_filename=zip_filename,
            cache_dir=cache_dir,
            local_data_dir=local_data_dir,
            hf_token=hf_token,
            event_logger=event_logger,
        )

    for row in rows:
        zip_filename = row.get(data_folder_field)
        if isinstance(zip_filename, str) and zip_filename in extracted_dirs:
            row["_local_data_dir"] = str(extracted_dirs[zip_filename])
        else:
            row["_local_data_dir"] = ""

    _emit(event_logger, f"dataset preparation complete rows={len(rows)}")
    return rows

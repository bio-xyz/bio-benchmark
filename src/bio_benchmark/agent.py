from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

import requests

from .config import AgentConfig


@dataclass
class AgentTaskResult:
    task_id: str
    status: str
    success: bool
    direct_answer: str
    answer: str
    uploaded_files: int
    poll_count: int
    latency_ms: int
    raw_response: dict[str, Any]


CACHE_PATH = Path("datasets/file_ids_cache.json")
CACHE_LOCK = threading.Lock()


def _emit(
    event_logger: Callable[[str], None] | None,
    task_label: str | None,
    message: str,
) -> None:
    if not event_logger:
        return
    if task_label:
        event_logger(f"task={task_label} {message}")
        return
    event_logger(message)


def _resolve_api_key(config: AgentConfig) -> str:
    if config.api_key:
        return config.api_key
    if config.api_key_env:
        value = os.getenv(config.api_key_env)
        if value:
            return value
    raise ValueError(
        "Missing API key. Set agent.api_key or agent.api_key_env in benchmark config."
    )


def _build_auth_headers(config: AgentConfig) -> dict[str, str]:
    return {"Authorization": f"Bearer {_resolve_api_key(config)}"}


def _build_url(endpoint: str, path: str) -> str:
    return urljoin(endpoint.rstrip("/") + "/", path.lstrip("/"))


def _auth_fingerprint(config: AgentConfig) -> str:
    token = _resolve_api_key(config)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _file_details_path(config: AgentConfig) -> str:
    upload_path = config.upload_path.rstrip("/")
    if upload_path.endswith("/upload"):
        base = upload_path[: -len("/upload")]
        return f"{base}/{{file_id}}"
    return "/files/{file_id}"


def _collect_upload_files(local_data_dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    cache_resolved = str(CACHE_PATH.resolve())
    for local_data_dir in local_data_dirs:
        if not local_data_dir:
            continue
        root = Path(local_data_dir)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key == cache_resolved:
                continue
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def _cache_payload() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "entries": {}}
    if not isinstance(data, dict):
        return {"version": 1, "entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        data["entries"] = {}
    return data


def _write_cache(payload: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _fingerprint_files(files_to_upload: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for file_path in files_to_upload:
        stat = file_path.stat()
        records.append(
            {
                "path": str(file_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return digest, records


def _cache_key(config: AgentConfig, files_to_upload: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    fingerprint, records = _fingerprint_files(files_to_upload)
    digest = hashlib.sha256(
        f"{config.endpoint}|{_auth_fingerprint(config)}|{fingerprint}".encode("utf-8")
    ).hexdigest()
    return digest, records


def _get_cached_file_ids(cache_key: str) -> list[str] | None:
    with CACHE_LOCK:
        payload = _cache_payload()
        entry = payload.get("entries", {}).get(cache_key)
        if not isinstance(entry, dict):
            return None
        file_ids = entry.get("file_ids")
        if not isinstance(file_ids, list):
            return None
        return [str(item) for item in file_ids if str(item).strip()]


def _set_cached_file_ids(
    *,
    cache_key: str,
    config: AgentConfig,
    file_ids: list[str],
    file_records: list[dict[str, Any]],
) -> None:
    with CACHE_LOCK:
        payload = _cache_payload()
        entries = payload.setdefault("entries", {})
        entries[cache_key] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "endpoint": config.endpoint,
            "test": config.test,
            "file_ids": file_ids,
            "files": file_records,
        }
        _write_cache(payload)


def _cached_file_ids_are_valid(
    session: requests.Session,
    config: AgentConfig,
    headers: dict[str, str],
    cached_ids: list[str],
) -> bool:
    if not cached_ids:
        return False
    details_template = _file_details_path(config)
    for file_id in cached_ids:
        details_url = _build_url(
            config.endpoint,
            details_template.format(file_id=file_id),
        )
        response = session.get(
            details_url,
            headers=headers,
            timeout=config.timeout_seconds,
        )
        if response.status_code != 200:
            return False
    return True


def _upload_files(
    session: requests.Session,
    config: AgentConfig,
    headers: dict[str, str],
    files_to_upload: list[Path],
    *,
    event_logger: Callable[[str], None] | None = None,
    task_label: str | None = None,
) -> list[str]:
    upload_url = _build_url(config.endpoint, config.upload_path)
    file_ids: list[str] = []
    total = len(files_to_upload)
    for index, file_path in enumerate(files_to_upload, start=1):
        content_type, _ = mimetypes.guess_type(file_path.name)
        mime = content_type or "application/octet-stream"
        size_bytes = file_path.stat().st_size
        _emit(
            event_logger,
            task_label,
            f"[cyan]uploading file {index}/{total}[/cyan] name={file_path.name} size={size_bytes}B",
        )
        with file_path.open("rb") as handle:
            response = session.post(
                upload_url,
                headers=headers,
                files={"file": (file_path.name, handle, mime)},
                timeout=config.timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
        file_id = payload.get("fileId") or payload.get("file_id")
        if not file_id:
            raise ValueError(f"Upload response missing fileId for {file_path.name}")
        file_ids.append(str(file_id))
        _emit(
            event_logger,
            task_label,
            f"[green]uploaded file[/green] name={file_path.name} file_id={file_id}",
        )
    return file_ids


def _start_task(
    session: requests.Session,
    config: AgentConfig,
    headers: dict[str, str],
    prompt_text: str,
    file_ids: list[str],
    *,
    event_logger: Callable[[str], None] | None = None,
    task_label: str | None = None,
) -> str:
    run_url = _build_url(config.endpoint, config.run_path)
    form_data: list[tuple[str, str]] = [("taskDescription", prompt_text)]
    form_data.extend(("fileIds", file_id) for file_id in file_ids)
    _emit(
        event_logger,
        task_label,
        f"[cyan]starting task[/cyan] files={len(file_ids)}",
    )

    response = session.post(
        run_url,
        headers=headers,
        data=form_data,
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    task_id = payload.get("id") or payload.get("taskId") or payload.get("task_id")
    if not task_id:
        raise ValueError("Task start response missing id/taskId")
    _emit(event_logger, task_label, f"[green]task started[/green] api_task_id={task_id}")
    return str(task_id)


def _extract_answer(payload: dict[str, Any]) -> tuple[str, str]:
    direct_answer = payload.get("directAnswer") or payload.get("direct_answer") or ""
    answer = payload.get("answer") or payload.get("result") or ""
    return str(direct_answer), str(answer)


def _poll_task(
    session: requests.Session,
    config: AgentConfig,
    headers: dict[str, str],
    task_id: str,
    *,
    event_logger: Callable[[str], None] | None = None,
    task_label: str | None = None,
) -> tuple[dict[str, Any], int]:
    poll_url = _build_url(
        config.endpoint,
        config.task_path_template.format(task_id=task_id),
    )
    start = time.monotonic()
    poll_count = 0

    while True:
        poll_count += 1
        response = session.get(
            poll_url,
            headers=headers,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "")).strip().lower()
        success = payload.get("success")

        if poll_count == 1 or poll_count % 10 == 0:
            elapsed = int(time.monotonic() - start)
            _emit(
                event_logger,
                task_label,
                f"api_task_id={task_id} poll={poll_count} status={status or '?'} success={success} elapsed={elapsed}s",
            )

        if status == "completed" and success is True:
            _emit(
                event_logger,
                task_label,
                f"[green]task completed[/green] api_task_id={task_id} polls={poll_count}",
            )
            return payload, poll_count
        if status in {"failed", "error"} or success is False:
            _emit(
                event_logger,
                task_label,
                f"[red]task failed[/red] api_task_id={task_id} status={payload.get('status')} success={success}",
            )
            raise RuntimeError(
                f"Task {task_id} failed: status={payload.get('status')} success={success} "
                f"answer={payload.get('answer')}"
            )

        if (time.monotonic() - start) > config.poll_timeout_seconds:
            _emit(
                event_logger,
                task_label,
                f"[red]task timeout[/red] api_task_id={task_id} timeout={config.poll_timeout_seconds}s",
            )
            raise TimeoutError(
                f"Task {task_id} timed out after {config.poll_timeout_seconds}s"
            )
        time.sleep(config.poll_interval_seconds)


def run_data_analysis_question(
    config: AgentConfig,
    prompt_text: str,
    local_data_dirs: list[str] | None,
    *,
    task_label: str | None = None,
    event_logger: Callable[[str], None] | None = None,
) -> AgentTaskResult:
    if config.test != "bio-data-analysis":
        raise ValueError(
            f"Unsupported test '{config.test}'. Only 'bio-data-analysis' is supported."
        )

    start_time = time.perf_counter()
    headers = _build_auth_headers(config)
    files_to_upload = _collect_upload_files(local_data_dirs or [])
    _emit(
        event_logger,
        task_label,
        f"files discovered={len(files_to_upload)} local_dirs={len(local_data_dirs or [])}",
    )
    cache_key, file_records = _cache_key(config, files_to_upload)

    with requests.Session() as session:
        cached_ids = _get_cached_file_ids(cache_key)
        if (
            cached_ids
            and len(cached_ids) == len(files_to_upload)
            and _cached_file_ids_are_valid(session, config, headers, cached_ids)
        ):
            file_ids = cached_ids
            uploaded_files = 0
            _emit(
                event_logger,
                task_label,
                f"[cyan]file cache hit[/cyan] reusing={len(file_ids)}",
            )
        else:
            if cached_ids:
                _emit(
                    event_logger,
                    task_label,
                    "[yellow]file cache invalid[/yellow] uploading fresh files",
                )
            else:
                _emit(event_logger, task_label, "file cache miss uploading files")
            file_ids = _upload_files(
                session,
                config,
                headers,
                files_to_upload,
                event_logger=event_logger,
                task_label=task_label,
            )
            uploaded_files = len(file_ids)
            _set_cached_file_ids(
                cache_key=cache_key,
                config=config,
                file_ids=file_ids,
                file_records=file_records,
            )
            _emit(
                event_logger,
                task_label,
                f"[green]file cache updated[/green] entries={len(file_ids)}",
            )
        task_id = _start_task(
            session,
            config,
            headers,
            prompt_text,
            file_ids,
            event_logger=event_logger,
            task_label=task_label,
        )
        payload, poll_count = _poll_task(
            session,
            config,
            headers,
            task_id,
            event_logger=event_logger,
            task_label=task_label,
        )

    direct_answer, answer = _extract_answer(payload)
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return AgentTaskResult(
        task_id=task_id,
        status=str(payload.get("status", "")),
        success=bool(payload.get("success", False)),
        direct_answer=direct_answer,
        answer=answer,
        uploaded_files=uploaded_files,
        poll_count=poll_count,
        latency_ms=latency_ms,
        raw_response=payload,
    )

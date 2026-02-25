from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

BENCHMARK_PRESETS: dict[str, dict[str, str]] = {
    "phylobio/BixBench-Verified-50": {
        "strategy": "bixbench",
        "repo_id": "phylobio/BixBench-Verified-50",
    },
}


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 20.0


@dataclass
class DatasetConfig:
    repo_id: str
    split: str = "train"
    data_files: str | None = "*.jsonl"
    cache_dir: str = "datasets"
    local_data_dir: str = "datasets/capsules"
    limit: int | None = None
    question_field: str = "question"
    ground_truth_field: str = "ideal"
    question_id_field: str = "question_id"
    task_id_field: str = "short_id"
    data_folder_field: str = "data_folder"
    hf_token_env: str | None = None


@dataclass
class PromptConfig:
    template: str = """Task ID: {{task_id}}
Research Hypothesis: {{research_hypothesis}}

Available Datasets: {{available_datasets}}

Questions to Answer:
{{questions_block}}

IMPORTANT: Please analyze the provided datasets and provide ONLY the direct answer.
Output Format Requirements:
- Answer with number (e.g., "1. [answer]")
- Provide concise, direct answer
- If cannot be answered from the data, state "Cannot be determined from the data"
"""


@dataclass
class AgentConfig:
    test: str
    endpoint: str
    api_key_env: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 120.0
    upload_path: str = "/files/upload"
    run_path: str = "/agents/analysis/run"
    task_path_template: str = "/agents/analysis/tasks/{task_id}"
    poll_interval_seconds: float = 10.0
    poll_timeout_seconds: int = 10800


@dataclass
class JudgeConfig:
    enabled: bool = True
    grader_model: str = "gpt-5"
    option_chooser_model: str = "gpt-5-mini"
    include_mcq_modes: bool = True
    refusal_option_text: str = "Insufficient information to answer the question"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    timeout_seconds: float = 60.0


@dataclass
class OutputConfig:
    csv_path: str = "results/{{run_id}}.csv"


@dataclass
class BenchmarkConfig:
    strategy: str
    test: str
    benchmark: str
    run_id: str
    concurrency: int
    repeats: int
    dataset_setup: bool
    dataset: DatasetConfig
    prompt: PromptConfig
    agent: AgentConfig
    retry: RetryPolicy
    judge: JudgeConfig
    output: OutputConfig


def _require_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid required string field: {key}")
    return value


def _as_dict(raw: Any, key: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected '{key}' to be a mapping")
    return raw


def _as_positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _default_run_id(benchmark: str) -> str:
    safe_name = benchmark.strip().lower().replace(" ", "-")
    safe_name = "".join(ch for ch in safe_name if ch.isalnum() or ch in {"-", "_"})
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{safe_name}-{timestamp}"


def _get_preset(benchmark: str) -> dict[str, str]:
    key = benchmark.strip()
    if key in BENCHMARK_PRESETS:
        return BENCHMARK_PRESETS[key]
    for preset_key, preset in BENCHMARK_PRESETS.items():
        if preset_key.lower() == key.lower():
            return preset
    return {}


def load_config(path: str | Path) -> BenchmarkConfig:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Root YAML config must be a mapping")

    benchmark = _require_str(raw, "benchmark")
    preset = _get_preset(benchmark)

    strategy = str(raw.get("strategy") or preset.get("strategy") or "bixbench")
    test = str(raw.get("test", "bio-data-analysis"))
    run_id = str(raw.get("run_id") or _default_run_id(benchmark))
    concurrency = _as_positive_int(raw.get("concurrency"), default=1)
    repeats = _as_positive_int(raw.get("repeats"), default=1)
    dataset_setup = _as_bool(raw.get("dataset_setup"), default=True)

    dataset_raw = _as_dict(raw.get("dataset"), "dataset")
    hf_token_env = dataset_raw.get("hf_token_env") or raw.get("hf_token_env")
    repo_id = dataset_raw.get("repo_id") or preset.get("repo_id")
    if not repo_id and "/" in benchmark:
        repo_id = benchmark
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError(
            "Missing dataset.repo_id and no preset available for this benchmark."
        )
    dataset = DatasetConfig(
        repo_id=repo_id,
        split=str(dataset_raw.get("split", "train")),
        data_files=dataset_raw.get("data_files", "*.jsonl"),
        cache_dir=str(dataset_raw.get("cache_dir", "datasets")),
        local_data_dir=str(dataset_raw.get("local_data_dir", "datasets/capsules")),
        limit=dataset_raw.get("limit"),
        question_field=str(dataset_raw.get("question_field", "question")),
        ground_truth_field=str(dataset_raw.get("ground_truth_field", "ideal")),
        question_id_field=str(dataset_raw.get("question_id_field", "question_id")),
        task_id_field=str(dataset_raw.get("task_id_field", "short_id")),
        data_folder_field=str(dataset_raw.get("data_folder_field", "data_folder")),
        hf_token_env=str(hf_token_env) if hf_token_env else None,
    )

    prompt_raw = _as_dict(raw.get("prompt"), "prompt")
    prompt = PromptConfig(
        template=str(prompt_raw.get("template", PromptConfig().template))
    )

    agent_raw = _as_dict(raw.get("agent"), "agent")
    endpoint = agent_raw.get("endpoint") or raw.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("Missing required endpoint. Set top-level `endpoint`.")

    api_key_env = agent_raw.get("api_key_env") or raw.get(
        "api_key_env", "BIOS_API_KEY"
    )
    api_key = agent_raw.get("api_key") or raw.get("api_key")
    if not api_key and not api_key_env:
        raise ValueError("Set api_key or api_key_env for agent authentication.")

    agent = AgentConfig(
        test=str(agent_raw.get("test") or test),
        endpoint=endpoint,
        api_key_env=api_key_env,
        api_key=api_key,
        timeout_seconds=float(agent_raw.get("timeout_seconds", 120)),
        upload_path=str(agent_raw.get("upload_path", "/files/upload")),
        run_path=str(agent_raw.get("run_path", "/agents/analysis/run")),
        task_path_template=str(
            agent_raw.get("task_path_template", "/agents/analysis/tasks/{task_id}")
        ),
        poll_interval_seconds=float(agent_raw.get("poll_interval_seconds", 10)),
        poll_timeout_seconds=int(agent_raw.get("poll_timeout_seconds", 10800)),
    )

    retry_raw = _as_dict(raw.get("retry"), "retry")
    retry = RetryPolicy(
        max_attempts=int(retry_raw.get("max_attempts", 3)),
        initial_delay_seconds=float(retry_raw.get("initial_delay_seconds", 2)),
        backoff_multiplier=float(retry_raw.get("backoff_multiplier", 2)),
        max_delay_seconds=float(retry_raw.get("max_delay_seconds", 20)),
    )

    judge_raw = _as_dict(raw.get("judge"), "judge")
    judge = JudgeConfig(
        enabled=bool(judge_raw.get("enabled", True)),
        grader_model=str(judge_raw.get("grader_model", "gpt-5")),
        option_chooser_model=str(judge_raw.get("option_chooser_model", "gpt-5-mini")),
        include_mcq_modes=bool(judge_raw.get("include_mcq_modes", True)),
        refusal_option_text=str(
            judge_raw.get(
                "refusal_option_text",
                "Insufficient information to answer the question",
            )
        ),
        api_key_env=str(judge_raw.get("api_key_env", "OPENAI_API_KEY")),
        api_key=judge_raw.get("api_key"),
        timeout_seconds=float(judge_raw.get("timeout_seconds", 60)),
    )

    output_raw = _as_dict(raw.get("output"), "output")
    output = OutputConfig(
        csv_path=str(output_raw.get("csv_path", "results/{{run_id}}.csv"))
    )

    return BenchmarkConfig(
        strategy=strategy,
        test=test,
        benchmark=benchmark,
        run_id=run_id,
        concurrency=concurrency,
        repeats=repeats,
        dataset_setup=dataset_setup,
        dataset=dataset,
        prompt=prompt,
        agent=agent,
        retry=retry,
        judge=judge,
        output=output,
    )

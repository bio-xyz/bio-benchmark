from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from ..config import BenchmarkConfig


@dataclass
class BenchmarkRow:
    raw: dict[str, Any]
    question_id: str
    task_group: str
    question_text: str
    ground_truth: str
    distractors: Any
    data_folder: str
    local_data_dir: str
    hypothesis: str


@dataclass
class PreparedTaskBatch:
    task_group: str
    rows: list[BenchmarkRow]
    prompt_text: str
    local_data_dirs: list[str]


class BenchmarkStrategy(ABC):
    name: str

    @abstractmethod
    def load_rows(
        self,
        config: BenchmarkConfig,
        *,
        limit: int | None = None,
        event_logger: Callable[[str], None] | None = None,
    ) -> list[BenchmarkRow]:
        """Load and normalize benchmark rows."""

    @abstractmethod
    def prepare_batches(
        self,
        config: BenchmarkConfig,
        rows: list[BenchmarkRow],
    ) -> list[PreparedTaskBatch]:
        """Group rows and generate task prompts for execution."""

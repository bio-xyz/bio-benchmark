from __future__ import annotations

from .base import BenchmarkStrategy
from .bixbench import BixBenchStrategy

_STRATEGIES: dict[str, BenchmarkStrategy] = {
    "bixbench": BixBenchStrategy(),
}


def get_strategy(name: str) -> BenchmarkStrategy:
    key = name.strip().lower()
    strategy = _STRATEGIES.get(key)
    if strategy is None:
        supported = ", ".join(sorted(_STRATEGIES.keys()))
        raise ValueError(f"Unknown benchmark strategy '{name}'. Supported: {supported}")
    return strategy


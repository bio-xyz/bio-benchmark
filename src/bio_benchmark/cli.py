from __future__ import annotations

import argparse

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_config
from .runner import run_benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bio-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark from YAML config")
    run_parser.add_argument(
        "--config",
        default="benchmark.yml",
        help="Path to benchmark YAML config (default: benchmark.yml)",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of dataset examples to run",
    )
    return parser


def _accuracy(correct: int, incorrect: int) -> str:
    total = correct + incorrect
    if total == 0:
        return "0.0%"
    return f"{(correct / total) * 100:.1f}%"


def main() -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()
    console = Console()

    if args.command == "run":
        config = load_config(args.config)
        summary = run_benchmark(config, limit=args.limit, console=console)

        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Strategy: {config.strategy}",
                        f"Benchmark: {config.benchmark}",
                        f"Run ID: {config.run_id}",
                        f"Concurrency: {config.concurrency}",
                        f"Repeats: {config.repeats}",
                        f"Target test: {config.test}",
                    ]
                ),
                title="Run Configuration",
            )
        )

        table = Table(title="Correctness Report")
        table.add_column("Mode")
        table.add_column("Correct", justify="right")
        table.add_column("Incorrect", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_row(
            "Direct Open",
            str(summary.direct_correct),
            str(summary.direct_incorrect),
            _accuracy(summary.direct_correct, summary.direct_incorrect),
        )
        table.add_row(
            "MCQ + Refusal",
            str(summary.mcq_with_refusal_correct),
            str(summary.mcq_with_refusal_incorrect),
            _accuracy(
                summary.mcq_with_refusal_correct,
                summary.mcq_with_refusal_incorrect,
            ),
        )
        table.add_row(
            "MCQ No Refusal",
            str(summary.mcq_without_refusal_correct),
            str(summary.mcq_without_refusal_incorrect),
            _accuracy(
                summary.mcq_without_refusal_correct,
                summary.mcq_without_refusal_incorrect,
            ),
        )
        console.print(table)

        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Total questions: {summary.total}",
                        f"Request errors: {summary.request_errors}",
                        f"Judge errors: {summary.judge_errors}",
                        f"CSV: {summary.output_csv}",
                    ]
                ),
                title="Run Summary",
            )
        )


if __name__ == "__main__":
    main()


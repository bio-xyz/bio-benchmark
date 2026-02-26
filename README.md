# bio-benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![BixBench-Verified-50](https://img.shields.io/badge/benchmark-BixBench--Verified--50-green.svg)](https://huggingface.co/datasets/phylobio/BixBench-Verified-50)
[![BioAgents](https://img.shields.io/badge/backend-BioAgents-purple.svg)](https://github.com/bio-xyz/BioAgents)
[![Get API Key](https://img.shields.io/badge/API%20Key-chat.bio.xyz-teal.svg)](https://chat.bio.xyz/chat?settings=account&section=api-keys)

BixBench evaluation harness for [BioAgents](https://github.com/bio-xyz/BioAgents) and its closed-source literature and data-analysis agents.

### Overall highlights (from dashboard)

| Mode                    | Accuracy | 95% Wilson CI | Correct / Total |
| ----------------------- | -------: | ------------: | --------------: |
| Direct                  |    71.3% | 67.4% - 74.9% |       392 / 550 |
| MCQ with refusal        |    85.1% | 81.9% - 87.8% |       468 / 550 |
| **MCQ without refusal** |    90.0% | 87.2% - 92.2% |       495 / 550 |

| Headline                                                  |               Value |
| --------------------------------------------------------- | ------------------: |
| MCQ lift (Direct -> **MCQ without refusal**)              |             +18.7pp |
| Refusal gap (**MCQ without refusal** -> MCQ with refusal) |              -4.9pp |
| Best repeat (**MCQ without refusal**)                     | 96.0% (`085809-r3`) |
| Task groups at 100%                                       |     22 / 32 (68.8%) |

### Showcase: Benchmark Dashboard

Interactive page from `docs/`:

- Live (GitHub Pages): [bio-xyz.github.io/bio-benchmark](https://bio-xyz.github.io/bio-benchmark/)
- Source: [`docs/index.html`](docs/index.html)

[![Benchmark dashboard preview](docs/assets/performance_analysis.png)](https://bio-xyz.github.io/bio-benchmark/)

Local preview:

```bash
python3 -m http.server 8080 --directory docs
# open http://localhost:8080
```

### Minimal config

Only these fields are required:

```yml
benchmark: phylobio/BixBench-Verified-50
endpoint: http://localhost:8000
api_key_env: BIOS_API_KEY
hf_token_env: HF_TOKEN
concurrency: 2
repeats: 1
dataset_setup: true
```

Configured env vars are validated at startup and the run exits immediately if any are missing
(for example `BIOS_API_KEY`, `HF_TOKEN`, and `OPENAI_API_KEY` when judge is enabled).

### What this does (for `benchmark: phylobio/BixBench-Verified-50`)

- Uses BixBench strategy (load + prepare + grouped prompt generation).
- Uses the full Hugging Face dataset name from `benchmark`.
- Groups by `short_id` (`bix-1`, `bix-2`, ...).
- Runs one API call per group.
- Runs up to `concurrency` task groups in parallel.
- Repeats full benchmark `repeats` times.
- Caches uploaded file IDs per capsule fingerprint in `datasets/file_ids_cache.json` and reuses them in future runs.
- Dataset setup runs before benchmark tasks by default, priming file cache for each unique task bundle.
- You can disable setup with `dataset_setup: false` or `--no-dataset-setup`.
- Grades with copied BixBench judge/option-selector logic.
- Shows interactive progress bars and per-task logs in shell.
- Prints final correctness report for all 3 modes: direct open, MCQ with refusal, MCQ without refusal.
- Writes CSV to `results/<run_id>.csv`.

### Run

```bash
cd /Users/mihailo/bio/bio-benchmark
uv sync
cp .env.example .env
# fill values in .env
# change config in benchmark.yml if needed
uv run bio-benchmark run --config benchmark.yml
# optional: disable pre-run setup
uv run bio-benchmark run --config benchmark.yml --no-dataset-setup
```

### Optional advanced overrides

You can still provide optional `dataset`, `prompt`, `agent`, `judge`, `retry`, and `output` sections to override defaults.
For the filtered dataset, use `benchmark: bixbench-verified-50` or override `dataset.repo_id`.

### Benchmark Results GitHub Pages

This repo includes a static site in `docs/` that showcases benchmark results for exactly these 3 CSV runs listed in `results/saved_output.md`:

- `phylobiobixbench-verified-50-20260224-190546.csv`
- `phylobiobixbench-verified-50-20260224-224841.csv`
- `phylobiobixbench-verified-50-20260225-085809.csv`

To rebuild page data and synced assets:

```bash
python3 scripts/build_pages_data.py
```

This script updates:

- `docs/data/benchmark_summary.json`
- `docs/data/results/*.csv`
- `docs/assets/performance_analysis.png`

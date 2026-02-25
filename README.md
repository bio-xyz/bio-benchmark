# bio-benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![BixBench-Verified-50](https://img.shields.io/badge/benchmark-BixBench--Verified--50-green.svg)](https://huggingface.co/datasets/phylobio/BixBench-Verified-50)
[![BioAgents](https://img.shields.io/badge/backend-BioAgents-purple.svg)](https://github.com/bio-xyz/BioAgents)
[![Get API Key](https://img.shields.io/badge/API%20Key-chat.bio.xyz-teal.svg)](https://chat.bio.xyz/)

BixBench evaluation harness for [BioAgents](https://github.com/bio-xyz/BioAgents) and its closed-source literature and data-analysis agents.

### Minimal config

Only these fields are required:

```yml
benchmark: phylobio/BixBench-Verified-50
endpoint: http://localhost:8000
api_key_env: BIOAGENT_PUBLIC_API_KEY
hf_token_env: HF_TOKEN
concurrency: 2
repeats: 1
dataset_setup: true
```

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

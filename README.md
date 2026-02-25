## bio-benchmark

Minimal benchmark runner.

### Minimal config

Only these fields are required:

```yml
benchmark: phylobio/BixBench-Verified-50
endpoint: http://localhost:8000
api_key_env: BIOAGENT_PUBLIC_API_KEY
hf_token_env: HF_TOKEN
concurrency: 2
repeats: 1
```

### What this does (for `benchmark: phylobio/BixBench-Verified-50`)

- Uses BixBench strategy (load + prepare + grouped prompt generation).
- Uses the full Hugging Face dataset name from `benchmark`.
- Groups by `short_id` (`bix-1`, `bix-2`, ...).
- Runs one API call per group.
- Runs up to `concurrency` task groups in parallel.
- Repeats full benchmark `repeats` times.
- Caches uploaded file IDs per capsule fingerprint in `datasets/file_ids_cache.json` and reuses them in future runs.
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
uv run bio-benchmark run --config benchmark.yml
```

### Optional advanced overrides

You can still provide optional `dataset`, `prompt`, `agent`, `judge`, `retry`, and `output` sections to override defaults.
For the filtered dataset, use `benchmark: bixbench-verified-50` or override `dataset.repo_id`.

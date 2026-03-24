After new shots, gotta rerun:

1. uv run python scripts/bootstrap_specs.py — regenerate spec drafts from the new images
2. Review/curate the specs in data/golden/specs/ — this is the most important step
3. uv run python scripts/prepare_golden.py — upload + cache depth maps
4. uv run python scripts/run_calibration.py — re-freeze thresholds
5. uv run python scripts/record_replay.py — re-record demo data

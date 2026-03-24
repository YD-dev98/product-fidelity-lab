"""Record complete runs as replay data for offline demo mode.

Runs a self-evaluation (golden vs golden) and saves the full run
as replay data in data/replay/ for the demo mode.

Usage: uv run python scripts/record_replay.py [shot_id ...]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from product_fidelity_lab.config import get_settings

os.environ.setdefault("FAL_KEY", get_settings().fal_key)

import numpy as np

from product_fidelity_lab.evaluation.calibration import load_thresholds
from product_fidelity_lab.evaluation.engine import run_evaluation
from product_fidelity_lab.evaluation.image_fetch import preload_local
from product_fidelity_lab.evaluation.spec_loader import load_all_specs, load_spec
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.models.run import RunType
from product_fidelity_lab.storage.run_store import RunStore


async def record_evaluation(
    shot_id: str,
    settings: object,
    fal_client: FalClient,
    store: RunStore,
    replay_dir: Path,
    grade_thresholds: dict[str, float],
    thresholds_path: Path,
    label: str = "replay_evaluation",
) -> dict | None:
    """Run evaluation on a golden image and save as replay data."""
    spec_path = settings.data_dir / "golden" / "specs" / f"{shot_id}.json"  # type: ignore[union-attr]
    spec = load_spec(spec_path)
    golden_image_path = settings.data_dir / "golden" / spec.image_path  # type: ignore[union-attr]

    if not golden_image_path.exists() or spec.image_url is None:
        print(f"  SKIP {shot_id}: missing image or URL")
        return None

    # Load depth
    depth_path = settings.data_dir / "cache" / "golden_depth" / f"{shot_id}.npy"  # type: ignore[union-attr]
    if not depth_path.exists():
        print(f"  SKIP {shot_id}: no depth map")
        return None

    golden_depth = np.load(str(depth_path))
    preload_local(spec.image_url, golden_image_path)

    run = await store.create_run(
        RunType.EVALUATION,
        config={"image_url": spec.image_url, "spec_id": shot_id, "label": label},
    )

    start = time.monotonic()
    try:
        report = await run_evaluation(
            spec.image_url,
            spec,
            golden_depth,
            fal_client=fal_client,
            gemini_api_key=settings.gemini_api_key,  # type: ignore[union-attr]
            gemini_model=settings.gemini_model,  # type: ignore[union-attr]
            run_store=store,
            run_id=run.id,
            grade_thresholds=grade_thresholds,
            thresholds_source=str(thresholds_path),
        )
    except Exception as e:
        print(f"  ERROR on {shot_id}: {e}")
        return None

    duration_ms = int((time.monotonic() - start) * 1000)
    r = report.model_dump()

    # Build replay run data
    replay_run = {
        "id": run.id,
        "type": "evaluation",
        "status": "complete",
        "created_at": run.created_at.isoformat(),
        "config": {
            "image_url": spec.image_url,
            "spec_id": shot_id,
            "label": label,
        },
        "result": r,
        "score": r["final"]["overall"],
        "grade": r["final"]["grade"],
        "passed": r["final"]["passed"],
        "duration_ms": duration_ms,
        "layer_states": {"afv": "complete", "depth": "complete", "brand": "complete"},
        "spec": spec.model_dump(),
    }

    # Save to replay directory
    run_replay_dir = replay_dir / run.id
    run_replay_dir.mkdir(parents=True, exist_ok=True)
    (run_replay_dir / "run.json").write_text(json.dumps(replay_run, indent=2))

    if r["final"]["outcome"] == "incomplete":
        print(f"  Recorded {shot_id}: INCOMPLETE")
    else:
        print(f"  Recorded {shot_id}: {r['final']['grade']} ({r['final']['overall']:.2f})")
    return replay_run


async def main() -> None:
    settings = get_settings()
    replay_dir = settings.data_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
    if not thresholds_path.exists():
        print("ERROR: calibration thresholds not found. Run run_calibration.py first.")
        return
    grade_thresholds = load_thresholds(thresholds_path)

    # Determine which shots to record
    if len(sys.argv) > 1:
        shot_ids = sys.argv[1:]
    else:
        # Default: pick 2-3 diverse training specs
        specs = load_all_specs(settings.data_dir / "golden" / "specs")
        training = [s for s in specs if s.category == "training"]
        # Pick first, middle, and last for variety
        picks = []
        if len(training) >= 1:
            picks.append(training[0])
        if len(training) >= 3:
            picks.append(training[len(training) // 2])
        if len(training) >= 2:
            picks.append(training[-1])
        shot_ids = [s.shot_id for s in picks]

    print(f"Recording replay data for {len(shot_ids)} shots")
    print(f"Replay directory: {replay_dir}")
    print()

    fal_client = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )
    store = RunStore(db_path=settings.db_path, runs_dir=settings.data_dir / "runs")
    await store.initialize()

    recorded = 0
    for shot_id in shot_ids:
        print(f"Recording: {shot_id}")
        result = await record_evaluation(
            shot_id, settings, fal_client, store, replay_dir, grade_thresholds, thresholds_path,
        )
        if result:
            recorded += 1

    print()
    print(f"Recorded {recorded}/{len(shot_ids)} replay runs")
    print(f"Replay data at: {replay_dir}")

    # List replay contents
    runs = list(replay_dir.iterdir())
    if runs:
        print(f"\nReplay runs ({len(runs)}):")
        for run_dir in sorted(runs):
            run_file = run_dir / "run.json"
            if run_file.exists():
                data = json.loads(run_file.read_text())
                score = data.get("score")
                score_display = f"{score:.2f}" if isinstance(score, int | float) else "n/a"
                print(
                    f"  {data['id'][:12]}... — {data['config']['spec_id']} "
                    f"— {data.get('grade', '?') or 'n/a'} ({score_display})"
                )


if __name__ == "__main__":
    asyncio.run(main())

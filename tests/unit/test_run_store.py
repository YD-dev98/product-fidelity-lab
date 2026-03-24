from __future__ import annotations

from pathlib import Path

import pytest

from product_fidelity_lab.models.run import LayerState, RunStatus, RunType
from product_fidelity_lab.storage.run_store import RunStore


@pytest.fixture
async def store(tmp_path: Path) -> RunStore:
    s = RunStore(
        db_path=tmp_path / "test.db",
        runs_dir=tmp_path / "runs",
    )
    await s.initialize()
    return s


class TestRunStore:
    @pytest.mark.asyncio
    async def test_create_and_get(self, store: RunStore) -> None:
        run = await store.create_run(RunType.EVALUATION, config={"spec_id": "hero"})
        assert run.type == RunType.EVALUATION
        assert run.status == RunStatus.PENDING
        assert run.config == {"spec_id": "hero"}

        fetched = await store.get_run(run.id)
        assert fetched is not None
        assert fetched.id == run.id
        assert fetched.type == RunType.EVALUATION

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: RunStore) -> None:
        assert await store.get_run("doesnotexist") is None

    @pytest.mark.asyncio
    async def test_list_runs(self, store: RunStore) -> None:
        await store.create_run(RunType.EVALUATION)
        await store.create_run(RunType.GENERATION)
        await store.create_run(RunType.EVALUATION)

        all_runs = await store.list_runs()
        assert len(all_runs) == 3

        eval_runs = await store.list_runs(type_filter=RunType.EVALUATION)
        assert len(eval_runs) == 2

    @pytest.mark.asyncio
    async def test_update_status(self, store: RunStore) -> None:
        run = await store.create_run(RunType.EVALUATION)
        layers = {"afv": LayerState.RUNNING, "depth": LayerState.PENDING}
        await store.update_status(run.id, RunStatus.RUNNING, layer_states=layers)

        fetched = await store.get_run(run.id)
        assert fetched is not None
        assert fetched.status == RunStatus.RUNNING
        assert fetched.layer_states["afv"] == LayerState.RUNNING

    @pytest.mark.asyncio
    async def test_update_result(self, store: RunStore) -> None:
        run = await store.create_run(RunType.EVALUATION)
        await store.update_result(
            run.id,
            result={"afv_score": 0.9},
            score=0.85,
            grade="A",
            passed=True,
            duration_ms=1234,
            cost=0.06,
        )

        fetched = await store.get_run(run.id)
        assert fetched is not None
        assert fetched.score == 0.85
        assert fetched.grade == "A"
        assert fetched.passed is True
        assert fetched.duration_ms == 1234
        assert fetched.result == {"afv_score": 0.9}

    @pytest.mark.asyncio
    async def test_artifacts(self, store: RunStore, tmp_path: Path) -> None:
        run = await store.create_run(RunType.EVALUATION)

        # Create a test artifact file
        artifact_path = store.artifact_dir(run.id) / "generated.jpg"
        artifact_path.write_bytes(b"fake image")

        artifact = await store.add_artifact(
            run.id, "generated_image", str(artifact_path), metadata={"width": 1024}
        )
        assert artifact.artifact_type == "generated_image"

        artifacts = await store.get_artifacts(run.id)
        assert len(artifacts) == 1
        assert artifacts[0].metadata == {"width": 1024}

    @pytest.mark.asyncio
    async def test_recover_interrupted(self, store: RunStore) -> None:
        run1 = await store.create_run(RunType.EVALUATION)
        run2 = await store.create_run(RunType.EVALUATION)

        await store.update_status(
            run1.id,
            RunStatus.RUNNING,
            layer_states={"afv": LayerState.RUNNING, "depth": LayerState.PENDING},
        )
        await store.update_status(run2.id, RunStatus.COMPLETE)

        count = await store.recover_interrupted()
        assert count == 1

        fetched = await store.get_run(run1.id)
        assert fetched is not None
        assert fetched.status == RunStatus.INTERRUPTED
        assert fetched.layer_states["afv"] == LayerState.INTERRUPTED
        assert fetched.layer_states["depth"] == LayerState.INTERRUPTED

        # Completed run should be untouched
        fetched2 = await store.get_run(run2.id)
        assert fetched2 is not None
        assert fetched2.status == RunStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_delete_run(self, store: RunStore) -> None:
        run = await store.create_run(RunType.EVALUATION)
        artifact_dir = store.artifact_dir(run.id)
        (artifact_dir / "test.txt").write_text("data")

        await store.delete_run(run.id)
        assert await store.get_run(run.id) is None
        assert not artifact_dir.exists()

    @pytest.mark.asyncio
    async def test_list_with_limit(self, store: RunStore) -> None:
        for _ in range(5):
            await store.create_run(RunType.EVALUATION)

        runs = await store.list_runs(limit=3)
        assert len(runs) == 3

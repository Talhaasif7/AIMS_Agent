"""Integration & statistical correctness test suite for the IRT Fitting Engine (Phase 2).

Charter references:
  - §4.2: 1PL/2PL estimation, convergence checking, sample guardrails, caching.
  - §4.1: Parameter artifact files.
  - §3.2: Async job lifecycle.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone

import pytest

from aims.db.models import (
    AIModel,
    Benchmark,
    CallerContext,
    ConvergenceStatus,
    Item,
    ModelFamily,
    Response,
)
from aims.db.sql_repositories import (
    SqlAIModelRepository,
    SqlBenchmarkRepository,
    SqlFitRepository,
    SqlItemRepository,
    SqlResponseRepository,
)
from aims.irt.artifacts import load_parameter_artifact, save_parameter_artifact
from aims.irt.estimator import estimate_irt_parameters
from aims.irt.jobs import JobManager
from aims.irt.models import FitRequest, IRTItemParameters, IRTPersonParameters, JobState


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def temp_artifact_dir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestArtifactManager:
    def test_save_and_load_parameter_artifact(self, temp_artifact_dir):
        fit_id = uuid.uuid4()
        items = [
            IRTItemParameters(
                item_id=uuid.uuid4(),
                item_key="q1",
                difficulty=0.5,
                discrimination=1.2,
            )
        ]
        persons = [
            IRTPersonParameters(
                ai_model_id=uuid.uuid4(),
                ability=1.1,
            )
        ]

        uri = save_parameter_artifact(fit_id, items, persons, base_dir=temp_artifact_dir)
        assert os.path.exists(uri)

        loaded_items, loaded_persons = load_parameter_artifact(uri, base_dir=temp_artifact_dir)
        assert len(loaded_items) == 1
        assert loaded_items[0].item_key == "q1"
        assert loaded_items[0].difficulty == 0.5
        assert loaded_items[0].discrimination == 1.2
        assert len(loaded_persons) == 1
        assert loaded_persons[0].ability == 1.1


class TestIRTEstimatorCorrectness:
    def test_guardrails_reject_underpowered_data(self):
        bench_id = uuid.uuid4()
        items = [Item(id=uuid.uuid4(), benchmark_id=bench_id, item_key="q1", created_at=_now())]
        responses = [Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=uuid.uuid4(), item_id=items[0].id, score=1, created_at=_now())]

        with pytest.raises(ValueError, match="requires at least 3 items"):
            estimate_irt_parameters(ModelFamily.RASCH_1PL, items, responses)

    def test_1pl_estimation_and_fit_stats(self):
        bench_id = uuid.uuid4()
        items = [Item(id=uuid.uuid4(), benchmark_id=bench_id, item_key=f"q{i}", created_at=_now()) for i in range(5)]
        models = [uuid.uuid4() for _ in range(10)]

        # Responses: models 0..4 get items 0..2 right, models 5..9 get items 2..4 right
        responses = []
        for i, m in enumerate(models):
            for j, item in enumerate(items):
                score = 1 if (i < 5 and j <= 2) or (i >= 5 and j >= 2) else 0
                responses.append(Response(
                    id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item.id, score=score, created_at=_now()
                ))

        item_params, person_params, conv_status, fit_stats = estimate_irt_parameters(
            ModelFamily.RASCH_1PL, items, responses, max_iter=100
        )

        assert len(item_params) == 5
        assert len(person_params) == 10
        assert conv_status == ConvergenceStatus.CONVERGED
        assert "aic" in fit_stats
        assert "bic" in fit_stats
        assert "minus_2ll" in fit_stats
        assert fit_stats["num_parameters"] == 15.0  # 5 items + 10 persons

    def test_non_convergence_returns_not_converged_status(self):
        """Charter §4.2: 'never return success on a run that hit max-iterations without convergence'"""
        bench_id = uuid.uuid4()
        items = [Item(id=uuid.uuid4(), benchmark_id=bench_id, item_key=f"q{i}", created_at=_now()) for i in range(5)]
        models = [uuid.uuid4() for _ in range(5)]

        responses = [
            Response(
                id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item.id,
                score=1 if (i + j) % 2 == 0 else 0, created_at=_now()
            )
            for i, m in enumerate(models) for j, item in enumerate(items)
        ]

        # Force max_iter = 1 to trigger max_iter cutoff
        item_params, person_params, conv_status, fit_stats = estimate_irt_parameters(
            ModelFamily.RASCH_1PL, items, responses, max_iter=1, tolerance=1e-10
        )

        assert conv_status == ConvergenceStatus.NOT_CONVERGED
        assert fit_stats["num_iterations"] == 1.0


class TestJobManagerIntegration:
    async def test_fit_job_execution_and_caching(self, session, temp_artifact_dir, caller_a):
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)
        fit_repo = SqlFitRepository(session)

        # Create benchmark, items, models, responses
        bench = await bench_repo.create(
            Benchmark(id=uuid.uuid4(), name="FitBench", owner_id=caller_a.user_id, created_at=_now()),
            caller_a,
        )
        items = await item_repo.create_bulk(
            [Item(id=uuid.uuid4(), benchmark_id=bench.id, item_key=f"q{i}", created_at=_now()) for i in range(5)],
            caller_a,
        )
        models = [
            await model_repo.create(AIModel(id=uuid.uuid4(), name=f"M{i}", owner_id=caller_a.user_id, created_at=_now()), caller_a)
            for i in range(5)
        ]

        responses = []
        for i, m in enumerate(models):
            for j, item in enumerate(items):
                responses.append(Response(
                    id=uuid.uuid4(), benchmark_id=bench.id, ai_model_id=m.id, item_id=item.id, score=1 if (i + j) % 2 == 0 else 0, created_at=_now()
                ))
        await resp_repo.create_bulk(responses, caller_a)

        manager = JobManager()
        req = FitRequest(benchmark_id=bench.id, model_family=ModelFamily.RASCH_1PL, random_seed=42)

        # 1. First submission (Cache miss)
        handle = await manager.submit_fit_job(
            req, caller_a, bench_repo, item_repo, resp_repo, fit_repo, base_artifact_dir=temp_artifact_dir
        )
        assert handle.status in (JobState.QUEUED, JobState.RUNNING, JobState.COMPLETED)

        # Wait for background task to complete
        for _ in range(50):
            job = manager.get_job(handle.job_id)
            if job and job.status == JobState.COMPLETED:
                break
            await asyncio.sleep(0.05)

        job = manager.get_job(handle.job_id)
        assert job is not None
        assert job.status == JobState.COMPLETED
        assert job.fit_result is not None
        assert job.fit_result.is_cached is False
        assert os.path.exists(job.fit_result.parameter_artifact_uri)

        # 2. Second submission (Cache hit §4.2)
        handle_cached = await manager.submit_fit_job(
            req, caller_a, bench_repo, item_repo, resp_repo, fit_repo, base_artifact_dir=temp_artifact_dir
        )
        assert handle_cached.status == JobState.COMPLETED
        assert handle_cached.fit_result is not None
        assert handle_cached.fit_result.is_cached is True
        assert handle_cached.fit_result.fit_id == job.fit_result.fit_id

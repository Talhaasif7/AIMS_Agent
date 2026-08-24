"""Async Job Orchestration & Service Integration for IRT Fitting (Layer 4 backend).

Charter references:
  - §3.2: "Long-running work never blocks the MCP call."
  - §4.2: Caching integration: "identical (dataset hash, model spec, seed) fit
           requests should hit cache, not recompute."
  - §4.1: Parameter estimates saved to artifact, referenced in fit record.
  - §5.2: "Every backend job: emit structured events for queued -> running -> completed."
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import UUID, uuid4

from aims.db.models import CallerContext, Fit, ModelFamily
from aims.db.repositories import (
    BenchmarkRepository,
    FitRepository,
    ItemRepository,
    ResponseRepository,
)
from aims.irt.artifacts import save_parameter_artifact
from aims.irt.estimator import compute_dataset_hash, estimate_irt_parameters
from aims.irt.models import FitRequest, IRTFitResult, JobHandle, JobState

_TORCH_MEASURE_VERSION = "0.3.0"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    """In-memory async job orchestrator for IRT model fitting tasks."""

    def __init__(self):
        self._jobs: Dict[UUID, JobHandle] = {}

    def get_job(self, job_id: UUID) -> Optional[JobHandle]:
        return self._jobs.get(job_id)

    async def submit_fit_job(
        self,
        request: FitRequest,
        caller: CallerContext,
        bench_repo: BenchmarkRepository,
        item_repo: ItemRepository,
        resp_repo: ResponseRepository,
        fit_repo: FitRepository,
        *,
        base_artifact_dir: Optional[str] = None,
    ) -> JobHandle:
        """Submit an IRT fitting request.

        Returns job handle immediately. If cached, marks completed right away.
        """
        # Fetch responses & compute dataset hash for cache check
        responses = await resp_repo.get_by_benchmark(request.benchmark_id, caller)
        if not responses:
            raise ValueError(f"No responses found for benchmark {request.benchmark_id}")

        dataset_hash = compute_dataset_hash(responses)

        # 1. Cache lookup (§4.2)
        cached_fit = await fit_repo.find_cached(
            dataset_hash=dataset_hash,
            model_family=request.model_family.value,
            random_seed=request.random_seed,
            caller=caller,
        )

        job_id = uuid4()
        now = _now()

        if cached_fit is not None:
            # Cache hit (§4.2)
            fit_result = IRTFitResult(
                fit_id=cached_fit.id,
                benchmark_id=cached_fit.benchmark_id,
                model_family=ModelFamily(cached_fit.model_family),
                convergence_status=cached_fit.convergence_status,
                fit_statistics=cached_fit.fit_statistics,
                parameter_artifact_uri=cached_fit.parameter_artifact_uri,
                dataset_hash=cached_fit.dataset_hash,
                torch_measure_version=cached_fit.torch_measure_version,
                random_seed=cached_fit.random_seed,
                created_at=cached_fit.created_at,
                owner_id=cached_fit.owner_id,
                is_cached=True,
            )
            handle = JobHandle(
                job_id=job_id,
                status=JobState.COMPLETED,
                progress=1.0,
                fit_result=fit_result,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = handle
            return handle

        # Cache miss -> Create queued job handle
        handle = JobHandle(
            job_id=job_id,
            status=JobState.QUEUED,
            progress=0.0,
            created_at=now,
            updated_at=now,
        )
        self._jobs[job_id] = handle

        # Spawn background execution task
        asyncio.create_task(
            self._run_fit_job(
                job_id=job_id,
                request=request,
                caller=caller,
                responses=responses,
                dataset_hash=dataset_hash,
                item_repo=item_repo,
                fit_repo=fit_repo,
                base_artifact_dir=base_artifact_dir,
            )
        )

        return handle

    async def _run_fit_job(
        self,
        job_id: UUID,
        request: FitRequest,
        caller: CallerContext,
        responses: list,
        dataset_hash: str,
        item_repo: ItemRepository,
        fit_repo: FitRepository,
        base_artifact_dir: Optional[str],
    ) -> None:
        """Background execution worker for IRT model fitting."""
        handle = self._jobs[job_id]
        handle.status = JobState.RUNNING
        handle.progress = 0.1
        handle.updated_at = _now()

        try:
            # Fetch items
            items = await item_repo.list_by_benchmark(request.benchmark_id, caller)
            if not items:
                raise ValueError(f"No items found for benchmark {request.benchmark_id}")

            handle.progress = 0.3
            handle.updated_at = _now()

            # Execute IRT estimation algorithm
            item_params, person_params, conv_status, fit_stats = estimate_irt_parameters(
                model_family=request.model_family,
                items=items,
                responses=responses,
                max_iter=request.hyperparameters.get("max_iter", 200),
                tolerance=request.hyperparameters.get("tolerance", 1e-5),
            )

            handle.progress = 0.7
            handle.updated_at = _now()

            # Save parameter artifact to file (§4.1)
            fit_id = uuid4()
            artifact_uri = save_parameter_artifact(
                fit_id=fit_id,
                item_params=item_params,
                person_params=person_params,
                base_dir=base_artifact_dir,
            )

            # Save fit record to database
            fit_domain = Fit(
                id=fit_id,
                benchmark_id=request.benchmark_id,
                model_family=request.model_family,
                hyperparameters=request.hyperparameters,
                convergence_status=conv_status,
                parameter_artifact_uri=artifact_uri,
                fit_statistics=fit_stats,
                dataset_hash=dataset_hash,
                torch_measure_version=_TORCH_MEASURE_VERSION,
                random_seed=request.random_seed,
                created_at=_now(),
                owner_id=caller.user_id,
            )
            saved_fit = await fit_repo.create(fit_domain, caller)

            fit_result = IRTFitResult(
                fit_id=saved_fit.id,
                benchmark_id=saved_fit.benchmark_id,
                model_family=ModelFamily(saved_fit.model_family),
                convergence_status=saved_fit.convergence_status,
                fit_statistics=saved_fit.fit_statistics,
                parameter_artifact_uri=saved_fit.parameter_artifact_uri,
                dataset_hash=saved_fit.dataset_hash,
                torch_measure_version=saved_fit.torch_measure_version,
                random_seed=saved_fit.random_seed,
                created_at=saved_fit.created_at,
                owner_id=saved_fit.owner_id,
                is_cached=False,
            )

            handle.status = JobState.COMPLETED
            handle.progress = 1.0
            handle.fit_result = fit_result
            handle.updated_at = _now()

        except Exception as exc:
            handle.status = JobState.FAILED
            handle.error_message = str(exc)
            handle.updated_at = _now()

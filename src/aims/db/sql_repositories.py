"""SQLAlchemy implementations of the repository interfaces.

Security notes:
  - All queries use SQLAlchemy ORM (parameterized by default). No raw SQL.
  - Access control is enforced at the query level: private data is filtered
    out in the WHERE clause, not post-retrieval.
  - MUST NOT expose SQL errors to callers. Repositories raise domain exceptions.

Charter §4.1: "Layer 4 should never touch SQL directly."
This module is the ONLY place SQL is constructed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aims.db.models import (
    AIModel,
    Benchmark,
    CallerContext,
    Fit,
    Item,
    Report,
    Response,
)
from aims.db.tables import (
    AIModelRow,
    BenchmarkRow,
    FitRow,
    ItemRow,
    ReportRow,
    ResponseRow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _access_filter_benchmark(caller: CallerContext):
    """Row-level access control: caller sees own + public benchmarks.

    Charter §4.1: "model developer's own private benchmarks vs.
    shared/public benchmark corpora — scope every query by caller identity."
    """
    return or_(
        BenchmarkRow.owner_id == str(caller.user_id),
        BenchmarkRow.is_public == True,  # noqa: E712 — SQLAlchemy requires ==
    )


def _owner_filter(column, caller: CallerContext):
    """Generic owner filter for non-benchmark tables."""
    return column == str(caller.user_id)


def _benchmark_to_domain(row: BenchmarkRow) -> Benchmark:
    return Benchmark(
        id=UUID(row.id),
        name=row.name,
        description=row.description,
        owner_id=UUID(row.owner_id),
        is_public=row.is_public,
        item_response_type=row.item_response_type,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _item_to_domain(row: ItemRow) -> Item:
    return Item(
        id=UUID(row.id),
        benchmark_id=UUID(row.benchmark_id),
        item_key=row.item_key,
        content_hash=row.content_hash,
        max_score=row.max_score,
        created_at=row.created_at,
    )


def _ai_model_to_domain(row: AIModelRow) -> AIModel:
    return AIModel(
        id=UUID(row.id),
        name=row.name,
        description=row.description,
        owner_id=UUID(row.owner_id),
        created_at=row.created_at,
    )


def _response_to_domain(row: ResponseRow) -> Response:
    return Response(
        id=UUID(row.id),
        benchmark_id=UUID(row.benchmark_id),
        ai_model_id=UUID(row.ai_model_id),
        item_id=UUID(row.item_id),
        score=row.score,
        response_version=row.response_version,
        is_current=row.is_current,
        created_at=row.created_at,
    )


def _fit_to_domain(row: FitRow) -> Fit:
    return Fit(
        id=UUID(row.id),
        benchmark_id=UUID(row.benchmark_id),
        model_family=row.model_family,
        hyperparameters=json.loads(row.hyperparameters) if isinstance(row.hyperparameters, str) else row.hyperparameters,
        convergence_status=row.convergence_status,
        parameter_artifact_uri=row.parameter_artifact_uri,
        fit_statistics=json.loads(row.fit_statistics) if isinstance(row.fit_statistics, str) else row.fit_statistics,
        dataset_hash=row.dataset_hash,
        torch_measure_version=row.torch_measure_version,
        random_seed=row.random_seed,
        created_at=row.created_at,
        owner_id=UUID(row.owner_id),
    )


def _report_to_domain(row: ReportRow) -> Report:
    return Report(
        id=UUID(row.id),
        fit_id=UUID(row.fit_id),
        format=row.format,
        artifact_uri=row.artifact_uri,
        dataset_hash=row.dataset_hash,
        torch_measure_version=row.torch_measure_version,
        created_at=row.created_at,
        owner_id=UUID(row.owner_id),
    )


# ---------------------------------------------------------------------------
# Concrete Implementations
# ---------------------------------------------------------------------------

class SqlBenchmarkRepository:
    """SQLAlchemy implementation of BenchmarkRepository."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, benchmark: Benchmark, caller: CallerContext) -> Benchmark:
        row = BenchmarkRow(
            id=str(benchmark.id),
            name=benchmark.name,
            description=benchmark.description,
            owner_id=str(caller.user_id),
            is_public=benchmark.is_public,
            item_response_type=benchmark.item_response_type,
            created_at=benchmark.created_at,
        )
        self._session.add(row)
        await self._session.flush()
        return _benchmark_to_domain(row)

    async def get_by_id(self, benchmark_id: UUID, caller: CallerContext) -> Benchmark | None:
        stmt = (
            select(BenchmarkRow)
            .where(
                BenchmarkRow.id == str(benchmark_id),
                _access_filter_benchmark(caller),
            )
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _benchmark_to_domain(row) if row else None

    async def list_accessible(self, caller: CallerContext) -> Sequence[Benchmark]:
        stmt = (
            select(BenchmarkRow)
            .where(_access_filter_benchmark(caller))
            .order_by(BenchmarkRow.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return [_benchmark_to_domain(row) for row in result.scalars().all()]

    async def delete(self, benchmark_id: UUID, caller: CallerContext) -> bool:
        # Only the owner can delete — not just anyone with read access
        stmt = select(BenchmarkRow).where(
            BenchmarkRow.id == str(benchmark_id),
            BenchmarkRow.owner_id == str(caller.user_id),
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return False
        await self._session.delete(row)
        await self._session.flush()
        return True


class SqlItemRepository:
    """SQLAlchemy implementation of ItemRepository."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_bulk(self, items: Sequence[Item], caller: CallerContext) -> Sequence[Item]:
        rows = []
        for item in items:
            row = ItemRow(
                id=str(item.id),
                benchmark_id=str(item.benchmark_id),
                item_key=item.item_key,
                content_hash=item.content_hash,
                max_score=item.max_score,
                created_at=item.created_at,
            )
            rows.append(row)
        self._session.add_all(rows)
        await self._session.flush()
        return [_item_to_domain(r) for r in rows]

    async def list_by_benchmark(
        self, benchmark_id: UUID, caller: CallerContext,
    ) -> Sequence[Item]:
        # Verify caller has access to the benchmark first
        bench_stmt = select(BenchmarkRow.id).where(
            BenchmarkRow.id == str(benchmark_id),
            _access_filter_benchmark(caller),
        )
        bench_result = await self._session.execute(bench_stmt)
        if bench_result.scalar_one_or_none() is None:
            return []

        stmt = (
            select(ItemRow)
            .where(ItemRow.benchmark_id == str(benchmark_id))
            .order_by(ItemRow.item_key)
        )
        result = await self._session.execute(stmt)
        return [_item_to_domain(row) for row in result.scalars().all()]


class SqlAIModelRepository:
    """SQLAlchemy implementation of AIModelRepository."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, ai_model: AIModel, caller: CallerContext) -> AIModel:
        row = AIModelRow(
            id=str(ai_model.id),
            name=ai_model.name,
            description=ai_model.description,
            owner_id=str(caller.user_id),
            created_at=ai_model.created_at,
        )
        self._session.add(row)
        await self._session.flush()
        return _ai_model_to_domain(row)

    async def get_by_id(self, model_id: UUID, caller: CallerContext) -> AIModel | None:
        stmt = select(AIModelRow).where(
            AIModelRow.id == str(model_id),
            _owner_filter(AIModelRow.owner_id, caller),
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _ai_model_to_domain(row) if row else None

    async def list_accessible(self, caller: CallerContext) -> Sequence[AIModel]:
        stmt = (
            select(AIModelRow)
            .where(_owner_filter(AIModelRow.owner_id, caller))
            .order_by(AIModelRow.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return [_ai_model_to_domain(row) for row in result.scalars().all()]


class SqlResponseRepository:
    """SQLAlchemy implementation of ResponseRepository.

    The response matrix is sparse: absent rows = unobserved pairs.
    This implementation never inserts sentinel values for missing data.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def _verify_benchmark_access(
        self, benchmark_id: UUID, caller: CallerContext,
    ) -> bool:
        stmt = select(BenchmarkRow.id).where(
            BenchmarkRow.id == str(benchmark_id),
            _access_filter_benchmark(caller),
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def create_bulk(
        self, responses: Sequence[Response], caller: CallerContext,
    ) -> int:
        rows = [
            ResponseRow(
                id=str(r.id),
                benchmark_id=str(r.benchmark_id),
                ai_model_id=str(r.ai_model_id),
                item_id=str(r.item_id),
                score=r.score,
                response_version=r.response_version,
                is_current=r.is_current,
                created_at=r.created_at,
            )
            for r in responses
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return len(rows)

    async def get_by_benchmark(
        self,
        benchmark_id: UUID,
        caller: CallerContext,
        *,
        current_only: bool = True,
    ) -> Sequence[Response]:
        if not await self._verify_benchmark_access(benchmark_id, caller):
            return []

        conditions = [ResponseRow.benchmark_id == str(benchmark_id)]
        if current_only:
            conditions.append(ResponseRow.is_current == True)  # noqa: E712

        stmt = select(ResponseRow).where(and_(*conditions))
        result = await self._session.execute(stmt)
        return [_response_to_domain(row) for row in result.scalars().all()]

    async def get_by_model_and_benchmark(
        self,
        ai_model_id: UUID,
        benchmark_id: UUID,
        caller: CallerContext,
        *,
        current_only: bool = True,
    ) -> Sequence[Response]:
        if not await self._verify_benchmark_access(benchmark_id, caller):
            return []

        conditions = [
            ResponseRow.ai_model_id == str(ai_model_id),
            ResponseRow.benchmark_id == str(benchmark_id),
        ]
        if current_only:
            conditions.append(ResponseRow.is_current == True)  # noqa: E712

        stmt = select(ResponseRow).where(and_(*conditions))
        result = await self._session.execute(stmt)
        return [_response_to_domain(row) for row in result.scalars().all()]

    async def get_at_version(
        self,
        benchmark_id: UUID,
        version: int,
        caller: CallerContext,
    ) -> Sequence[Response]:
        """Retrieve responses at a specific historical version.

        Charter §5.5: A fit stored against version N can be reproduced
        by re-querying version N responses.
        """
        if not await self._verify_benchmark_access(benchmark_id, caller):
            return []

        stmt = select(ResponseRow).where(
            ResponseRow.benchmark_id == str(benchmark_id),
            ResponseRow.response_version == version,
        )
        result = await self._session.execute(stmt)
        return [_response_to_domain(row) for row in result.scalars().all()]

    async def supersede(
        self,
        benchmark_id: UUID,
        new_responses: Sequence[Response],
        caller: CallerContext,
    ) -> int:
        """Append new version, mark old as non-current.

        Charter §4.1: "old responses must not be overwritten; append + supersede."
        Atomic savepoint guarantees rollback if insertion fails.
        """
        if not await self._verify_benchmark_access(benchmark_id, caller):
            return 0

        async with self._session.begin_nested():
            # Mark existing current responses as non-current
            stmt = (
                update(ResponseRow)
                .where(
                    ResponseRow.benchmark_id == str(benchmark_id),
                    ResponseRow.is_current == True,  # noqa: E712
                )
                .values(is_current=False)
            )
            await self._session.execute(stmt)

            # Insert new responses as current
            return await self.create_bulk(new_responses, caller)


class SqlFitRepository:
    """SQLAlchemy implementation of FitRepository."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, fit: Fit, caller: CallerContext) -> Fit:
        row = FitRow(
            id=str(fit.id),
            benchmark_id=str(fit.benchmark_id),
            model_family=fit.model_family,
            hyperparameters=json.dumps(fit.hyperparameters),
            convergence_status=fit.convergence_status,
            parameter_artifact_uri=fit.parameter_artifact_uri,
            fit_statistics=json.dumps(fit.fit_statistics),
            dataset_hash=fit.dataset_hash,
            torch_measure_version=fit.torch_measure_version,
            random_seed=fit.random_seed,
            created_at=fit.created_at,
            owner_id=str(caller.user_id),
        )
        self._session.add(row)
        await self._session.flush()
        return _fit_to_domain(row)

    async def get_by_id(self, fit_id: UUID, caller: CallerContext) -> Fit | None:
        stmt = select(FitRow).where(
            FitRow.id == str(fit_id),
            _owner_filter(FitRow.owner_id, caller),
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _fit_to_domain(row) if row else None

    async def list_by_benchmark(
        self, benchmark_id: UUID, caller: CallerContext,
    ) -> Sequence[Fit]:
        stmt = (
            select(FitRow)
            .where(
                FitRow.benchmark_id == str(benchmark_id),
                _owner_filter(FitRow.owner_id, caller),
            )
            .order_by(FitRow.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return [_fit_to_domain(row) for row in result.scalars().all()]

    async def find_cached(
        self,
        dataset_hash: str,
        model_family: str,
        random_seed: int,
        caller: CallerContext,
    ) -> Fit | None:
        """Cache lookup by (dataset_hash, model_family, seed).

        Charter §4.2: "identical (dataset hash, model spec, seed) fit requests
        should hit cache, not recompute."
        """
        stmt = select(FitRow).where(
            FitRow.dataset_hash == dataset_hash,
            FitRow.model_family == model_family,
            FitRow.random_seed == random_seed,
            _owner_filter(FitRow.owner_id, caller),
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _fit_to_domain(row) if row else None


class SqlReportRepository:
    """SQLAlchemy implementation of ReportRepository."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, report: Report, caller: CallerContext) -> Report:
        row = ReportRow(
            id=str(report.id),
            fit_id=str(report.fit_id),
            format=report.format,
            artifact_uri=report.artifact_uri,
            dataset_hash=report.dataset_hash,
            torch_measure_version=report.torch_measure_version,
            created_at=report.created_at,
            owner_id=str(caller.user_id),
        )
        self._session.add(row)
        await self._session.flush()
        return _report_to_domain(row)

    async def list_by_fit(
        self, fit_id: UUID, caller: CallerContext,
    ) -> Sequence[Report]:
        stmt = (
            select(ReportRow)
            .where(
                ReportRow.fit_id == str(fit_id),
                _owner_filter(ReportRow.owner_id, caller),
            )
            .order_by(ReportRow.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return [_report_to_domain(row) for row in result.scalars().all()]

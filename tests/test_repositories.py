"""Tests for AIMS Layer 5 repositories.

Covers:
  - Sub-Task 0.2: Schema correctness (models create/query without error)
  - Sub-Task 0.4: Repository CRUD operations
  - Sub-Task 0.5: Historical response versioning (supersede + version queries)
  - Sub-Task 0.6: Access control isolation (user A cannot see user B's private data)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from aims.db.models import (
    AIModel,
    Benchmark,
    CallerContext,
    ConvergenceStatus,
    Fit,
    Item,
    ItemResponseType,
    ModelFamily,
    Report,
    Response,
)
from aims.db.sql_repositories import (
    SqlAIModelRepository,
    SqlBenchmarkRepository,
    SqlFitRepository,
    SqlItemRepository,
    SqlReportRepository,
    SqlResponseRepository,
)


# Fixtures: engine, session, caller_a, caller_b are in conftest.py



def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_benchmark(
    owner_id: uuid.UUID,
    *,
    is_public: bool = False,
    name: str = "Test Benchmark",
) -> Benchmark:
    return Benchmark(
        id=uuid.uuid4(),
        name=name,
        owner_id=owner_id,
        is_public=is_public,
        item_response_type=ItemResponseType.BINARY,
        created_at=_now(),
    )


def _make_items(benchmark_id: uuid.UUID, count: int = 5) -> list[Item]:
    return [
        Item(
            id=uuid.uuid4(),
            benchmark_id=benchmark_id,
            item_key=f"item_{i}",
            max_score=1,
            created_at=_now(),
        )
        for i in range(count)
    ]


def _make_ai_model(owner_id: uuid.UUID, name: str = "GPT-Test") -> AIModel:
    return AIModel(
        id=uuid.uuid4(),
        name=name,
        owner_id=owner_id,
        created_at=_now(),
    )


# ---------------------------------------------------------------------------
# Sub-Task 0.2 + 0.4: Benchmark CRUD
# ---------------------------------------------------------------------------

class TestBenchmarkRepository:
    async def test_create_and_get(self, session: AsyncSession, caller_a: CallerContext):
        repo = SqlBenchmarkRepository(session)
        bench = _make_benchmark(caller_a.user_id)
        created = await repo.create(bench, caller_a)

        assert created.name == bench.name
        assert created.owner_id == caller_a.user_id

        fetched = await repo.get_by_id(created.id, caller_a)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_list_accessible_returns_own_and_public(
        self, session: AsyncSession, caller_a: CallerContext, caller_b: CallerContext,
    ):
        repo = SqlBenchmarkRepository(session)
        # A creates private + public
        await repo.create(_make_benchmark(caller_a.user_id, name="A-private"), caller_a)
        await repo.create(
            _make_benchmark(caller_a.user_id, name="A-public", is_public=True), caller_a,
        )
        # B creates private
        await repo.create(_make_benchmark(caller_b.user_id, name="B-private"), caller_b)

        # A sees own (2) + no public from B
        a_list = await repo.list_accessible(caller_a)
        a_names = {b.name for b in a_list}
        assert "A-private" in a_names
        assert "A-public" in a_names
        assert "B-private" not in a_names

        # B sees own (1) + A's public (1)
        b_list = await repo.list_accessible(caller_b)
        b_names = {b.name for b in b_list}
        assert "B-private" in b_names
        assert "A-public" in b_names
        assert "A-private" not in b_names

    async def test_delete_only_by_owner(
        self, session: AsyncSession, caller_a: CallerContext, caller_b: CallerContext,
    ):
        repo = SqlBenchmarkRepository(session)
        bench = await repo.create(
            _make_benchmark(caller_a.user_id, is_public=True), caller_a,
        )

        # B cannot delete A's benchmark even though it's public
        deleted = await repo.delete(bench.id, caller_b)
        assert deleted is False

        # A can delete their own
        deleted = await repo.delete(bench.id, caller_a)
        assert deleted is True

        # Verify it's gone
        assert await repo.get_by_id(bench.id, caller_a) is None


# ---------------------------------------------------------------------------
# Sub-Task 0.4: Item Repository
# ---------------------------------------------------------------------------

class TestItemRepository:
    async def test_bulk_create_and_list(
        self, session: AsyncSession, caller_a: CallerContext,
    ):
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)

        bench = await bench_repo.create(_make_benchmark(caller_a.user_id), caller_a)
        items = _make_items(bench.id, count=10)
        created = await item_repo.create_bulk(items, caller_a)
        assert len(created) == 10

        listed = await item_repo.list_by_benchmark(bench.id, caller_a)
        assert len(listed) == 10
        # Verify ordering by item_key
        keys = [i.item_key for i in listed]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Sub-Task 0.6: Access control isolation for items
# ---------------------------------------------------------------------------

class TestItemAccessControl:
    async def test_cannot_list_items_of_private_benchmark(
        self, session: AsyncSession, caller_a: CallerContext, caller_b: CallerContext,
    ):
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)

        bench = await bench_repo.create(
            _make_benchmark(caller_a.user_id, is_public=False), caller_a,
        )
        await item_repo.create_bulk(_make_items(bench.id, count=3), caller_a)

        # B cannot see items of A's private benchmark
        items = await item_repo.list_by_benchmark(bench.id, caller_b)
        assert len(items) == 0


# ---------------------------------------------------------------------------
# Sub-Task 0.4 + 0.5: Response repository + historical versioning
# ---------------------------------------------------------------------------

class TestResponseRepository:
    async def test_bulk_create_and_query(
        self, session: AsyncSession, caller_a: CallerContext,
    ):
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        bench = await bench_repo.create(_make_benchmark(caller_a.user_id), caller_a)
        items = await item_repo.create_bulk(_make_items(bench.id, count=5), caller_a)
        ai_model = await model_repo.create(_make_ai_model(caller_a.user_id), caller_a)

        responses = [
            Response(
                id=uuid.uuid4(),
                benchmark_id=bench.id,
                ai_model_id=ai_model.id,
                item_id=item.id,
                score=1 if i % 2 == 0 else 0,
                response_version=1,
                is_current=True,
                created_at=_now(),
            )
            for i, item in enumerate(items)
        ]

        count = await resp_repo.create_bulk(responses, caller_a)
        assert count == 5

        # Query by benchmark
        by_bench = await resp_repo.get_by_benchmark(bench.id, caller_a)
        assert len(by_bench) == 5

        # Query by model + benchmark
        by_model = await resp_repo.get_by_model_and_benchmark(
            ai_model.id, bench.id, caller_a,
        )
        assert len(by_model) == 5

    async def test_supersede_marks_old_non_current(
        self, session: AsyncSession, caller_a: CallerContext,
    ):
        """Charter §4.1: 'old responses must not be overwritten; append + supersede.'"""
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        bench = await bench_repo.create(_make_benchmark(caller_a.user_id), caller_a)
        items = await item_repo.create_bulk(_make_items(bench.id, count=3), caller_a)
        ai_model = await model_repo.create(_make_ai_model(caller_a.user_id), caller_a)

        # Version 1
        v1_responses = [
            Response(
                id=uuid.uuid4(),
                benchmark_id=bench.id,
                ai_model_id=ai_model.id,
                item_id=item.id,
                score=0,
                response_version=1,
                is_current=True,
                created_at=_now(),
            )
            for item in items
        ]
        await resp_repo.create_bulk(v1_responses, caller_a)

        # Version 2 via supersede (re-scoring — all correct now)
        v2_responses = [
            Response(
                id=uuid.uuid4(),
                benchmark_id=bench.id,
                ai_model_id=ai_model.id,
                item_id=item.id,
                score=1,
                response_version=2,
                is_current=True,
                created_at=_now(),
            )
            for item in items
        ]
        count = await resp_repo.supersede(bench.id, v2_responses, caller_a)
        assert count == 3

        # Current responses should be version 2
        current = await resp_repo.get_by_benchmark(bench.id, caller_a, current_only=True)
        assert len(current) == 3
        assert all(r.response_version == 2 for r in current)
        assert all(r.score == 1 for r in current)

        # Version 1 still accessible for reproducibility
        v1_historical = await resp_repo.get_at_version(bench.id, 1, caller_a)
        assert len(v1_historical) == 3
        assert all(r.score == 0 for r in v1_historical)

        # All responses (current_only=False) includes both versions
        all_responses = await resp_repo.get_by_benchmark(
            bench.id, caller_a, current_only=False,
        )
        assert len(all_responses) == 6

    async def test_supersede_rollback_on_failure(
        self, session: AsyncSession, caller_a: CallerContext,
    ):
        """Verify atomic transaction rollback if bulk insert fails during supersede."""
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        bench = await bench_repo.create(_make_benchmark(caller_a.user_id), caller_a)
        items = await item_repo.create_bulk(_make_items(bench.id, count=2), caller_a)
        ai_model = await model_repo.create(_make_ai_model(caller_a.user_id), caller_a)

        # Version 1
        v1_responses = [
            Response(
                id=uuid.uuid4(), benchmark_id=bench.id, ai_model_id=ai_model.id,
                item_id=item.id, score=1, response_version=1, is_current=True, created_at=_now(),
            )
            for item in items
        ]
        await resp_repo.create_bulk(v1_responses, caller_a)

        # Duplicate version 1 response (triggers UniqueConstraint uq_response_model_item_version)
        invalid_v2 = [
            Response(
                id=uuid.uuid4(), benchmark_id=bench.id, ai_model_id=ai_model.id,
                item_id=items[0].id, score=1, response_version=1, is_current=True, created_at=_now(),
            )
        ]

        # Expect IntegrityError / Exception
        with pytest.raises(Exception):
            await resp_repo.supersede(bench.id, invalid_v2, caller_a)

        # Verify old responses remain current because savepoint rolled back!
        current = await resp_repo.get_by_benchmark(bench.id, caller_a, current_only=True)
        assert len(current) == 2
        assert all(r.response_version == 1 for r in current)


# ---------------------------------------------------------------------------
# Sub-Task 0.6: Response access control isolation
# ---------------------------------------------------------------------------

class TestResponseAccessControl:
    async def test_cannot_query_responses_of_private_benchmark(
        self, session: AsyncSession, caller_a: CallerContext, caller_b: CallerContext,
    ):
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        bench = await bench_repo.create(
            _make_benchmark(caller_a.user_id, is_public=False), caller_a,
        )
        items = await item_repo.create_bulk(_make_items(bench.id, count=2), caller_a)
        ai_model = await model_repo.create(_make_ai_model(caller_a.user_id), caller_a)

        responses = [
            Response(
                id=uuid.uuid4(),
                benchmark_id=bench.id,
                ai_model_id=ai_model.id,
                item_id=item.id,
                score=1,
                response_version=1,
                is_current=True,
                created_at=_now(),
            )
            for item in items
        ]
        await resp_repo.create_bulk(responses, caller_a)

        # B cannot see A's private benchmark responses
        by_bench = await resp_repo.get_by_benchmark(bench.id, caller_b)
        assert len(by_bench) == 0

        by_version = await resp_repo.get_at_version(bench.id, 1, caller_b)
        assert len(by_version) == 0


# ---------------------------------------------------------------------------
# Sub-Task 0.4: Fit repository + cache lookup
# ---------------------------------------------------------------------------

class TestFitRepository:
    async def test_create_and_cache_lookup(
        self, session: AsyncSession, caller_a: CallerContext,
    ):
        bench_repo = SqlBenchmarkRepository(session)
        fit_repo = SqlFitRepository(session)

        bench = await bench_repo.create(_make_benchmark(caller_a.user_id), caller_a)

        fit = Fit(
            id=uuid.uuid4(),
            benchmark_id=bench.id,
            model_family=ModelFamily.TWO_PL,
            hyperparameters={"max_iter": 1000},
            convergence_status=ConvergenceStatus.CONVERGED,
            parameter_artifact_uri="s3://fits/test-fit-params.json",
            fit_statistics={"aic": 1234.5, "bic": 1240.1},
            dataset_hash="abc123def456",
            torch_measure_version="0.3.0",
            random_seed=42,
            created_at=_now(),
            owner_id=caller_a.user_id,
        )

        created = await fit_repo.create(fit, caller_a)
        assert created.convergence_status == ConvergenceStatus.CONVERGED

        # Cache lookup: same (hash, family, seed) should find it
        cached = await fit_repo.find_cached(
            dataset_hash="abc123def456",
            model_family=ModelFamily.TWO_PL,
            random_seed=42,
            caller=caller_a,
        )
        assert cached is not None
        assert cached.id == created.id

        # Different seed → cache miss
        miss = await fit_repo.find_cached(
            dataset_hash="abc123def456",
            model_family=ModelFamily.TWO_PL,
            random_seed=99,
            caller=caller_a,
        )
        assert miss is None

    async def test_fit_access_isolation(
        self, session: AsyncSession, caller_a: CallerContext, caller_b: CallerContext,
    ):
        bench_repo = SqlBenchmarkRepository(session)
        fit_repo = SqlFitRepository(session)

        bench = await bench_repo.create(_make_benchmark(caller_a.user_id), caller_a)

        fit = Fit(
            id=uuid.uuid4(),
            benchmark_id=bench.id,
            model_family=ModelFamily.RASCH_1PL,
            convergence_status=ConvergenceStatus.CONVERGED,
            parameter_artifact_uri="s3://fits/params.json",
            dataset_hash="hash_a",
            torch_measure_version="0.3.0",
            random_seed=42,
            created_at=_now(),
            owner_id=caller_a.user_id,
        )
        created = await fit_repo.create(fit, caller_a)

        # B cannot see A's fit
        assert await fit_repo.get_by_id(created.id, caller_b) is None

        # B cannot find A's fit via cache lookup
        cached = await fit_repo.find_cached("hash_a", ModelFamily.RASCH_1PL, 42, caller_b)
        assert cached is None


# ---------------------------------------------------------------------------
# Sub-Task 0.4: Report repository
# ---------------------------------------------------------------------------

class TestReportRepository:
    async def test_create_and_list_by_fit(
        self, session: AsyncSession, caller_a: CallerContext,
    ):
        bench_repo = SqlBenchmarkRepository(session)
        fit_repo = SqlFitRepository(session)
        report_repo = SqlReportRepository(session)

        bench = await bench_repo.create(_make_benchmark(caller_a.user_id), caller_a)
        fit = await fit_repo.create(
            Fit(
                id=uuid.uuid4(),
                benchmark_id=bench.id,
                model_family=ModelFamily.TWO_PL,
                convergence_status=ConvergenceStatus.CONVERGED,
                parameter_artifact_uri="s3://fits/params.json",
                dataset_hash="hash_report",
                torch_measure_version="0.3.0",
                random_seed=42,
                created_at=_now(),
                owner_id=caller_a.user_id,
            ),
            caller_a,
        )

        report = await report_repo.create(
            Report(
                id=uuid.uuid4(),
                fit_id=fit.id,
                format="html",
                artifact_uri="s3://reports/report.html",
                dataset_hash="hash_report",
                torch_measure_version="0.3.0",
                created_at=_now(),
                owner_id=caller_a.user_id,
            ),
            caller_a,
        )
        assert report.format == "html"

        reports = await report_repo.list_by_fit(fit.id, caller_a)
        assert len(reports) == 1
        assert reports[0].id == report.id

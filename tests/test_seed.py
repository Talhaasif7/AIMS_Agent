"""Integration test for seed data generation (Sub-Task 0.7).

Verifies that the full Layer 5 stack works end-to-end:
  - Tables create without error
  - Seed data inserts correctly
  - All repository queries work on seeded data
  - Sparse response matrix preserves missingness
"""

from __future__ import annotations

import uuid

import pytest

from aims.db.models import CallerContext
from aims.db.seed import seed_database
from aims.db.sql_repositories import (
    SqlAIModelRepository,
    SqlBenchmarkRepository,
    SqlItemRepository,
    SqlResponseRepository,
)


# Fixtures: engine, session are in conftest.py


class TestSeedDatabase:
    async def test_seed_creates_expected_counts(self, session):
        caller = CallerContext(user_id=uuid.uuid4())
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        stats = await seed_database(
            bench_repo, item_repo, model_repo, resp_repo, caller,
            num_benchmarks=3,
            items_per_benchmark=20,
            num_ai_models=5,
            sparsity=0.5,
        )

        assert stats["benchmarks"] == 3
        assert stats["items"] == 60  # 3 * 20
        assert stats["ai_models"] == 5
        # With 50% sparsity, ~50% of 5*20=100 pairs per benchmark ≈ 50 per bench
        # Total ≈ 150 ± variance. Just check it's > 0 and < max.
        assert stats["responses"] > 0
        assert stats["responses"] < 3 * 5 * 20  # Can't exceed total possible pairs

    async def test_seeded_data_queryable(self, session):
        caller = CallerContext(user_id=uuid.uuid4())
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        await seed_database(
            bench_repo, item_repo, model_repo, resp_repo, caller,
            num_benchmarks=2,
            items_per_benchmark=10,
            num_ai_models=3,
            sparsity=0.4,
        )

        # Benchmarks queryable
        benchmarks = await bench_repo.list_accessible(caller)
        assert len(benchmarks) == 2

        # Items queryable per benchmark
        for bench in benchmarks:
            items = await item_repo.list_by_benchmark(bench.id, caller)
            assert len(items) == 10

        # Responses queryable (sparse — may be < max)
        for bench in benchmarks:
            responses = await resp_repo.get_by_benchmark(bench.id, caller)
            assert len(responses) > 0  # At least some observed
            max_possible = 3 * 10
            assert len(responses) <= max_possible

    async def test_response_sparsity_preserves_missingness(self, session):
        """Verify absent rows = unobserved, not zero-filled."""
        caller = CallerContext(user_id=uuid.uuid4())
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        await seed_database(
            bench_repo, item_repo, model_repo, resp_repo, caller,
            num_benchmarks=1,
            items_per_benchmark=50,
            num_ai_models=10,
            sparsity=0.2,  # Only 20% observed
        )

        benchmarks = await bench_repo.list_accessible(caller)
        responses = await resp_repo.get_by_benchmark(benchmarks[0].id, caller)

        max_possible = 10 * 50  # 500
        # With 20% sparsity, expect ~100 responses ± variance
        # Key assertion: significantly less than max (proving sparsity works)
        assert len(responses) < max_possible * 0.5

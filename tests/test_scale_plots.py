"""Integration, Diagnostic Plots, and Scale Performance Test Suite (Phase 5).

Charter references:
  - §4.2: ICC, TIF, Wright Map plot specs.
  - §4.1 & §6.3: Scale testing & sparse response matrix query performance validation.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import pytest

from aims.db.models import CallerContext
from aims.db.seed import seed_database
from aims.db.sql_repositories import (
    SqlAIModelRepository,
    SqlBenchmarkRepository,
    SqlItemRepository,
    SqlResponseRepository,
)
from aims.irt.models import IRTItemParameters, IRTPersonParameters
from aims.plots.generator import (
    generate_diagnostic_plots,
    generate_icc_curves,
    generate_tif_curve,
    generate_wright_map,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestDiagnosticPlots:
    def test_icc_curves_valid_probabilities(self):
        item = IRTItemParameters(
            item_id=uuid.uuid4(),
            item_key="q1",
            difficulty=0.0,
            discrimination=1.5,
        )
        curves = generate_icc_curves([item], num_points=11)
        assert len(curves) == 1
        assert len(curves[0].curve) == 11
        # Monotonically increasing probability for 2PL with positive discrimination
        probs = [pt.probability for pt in curves[0].curve]
        assert probs == sorted(probs)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_tif_curve_peak_information(self):
        bench_id = uuid.uuid4()
        items = [
            IRTItemParameters(
                item_id=uuid.uuid4(),
                item_key=f"q_{i}",
                difficulty=b,
                discrimination=1.5,
            )
            for i, b in enumerate([-1.0, 0.0, 1.0])
        ]
        tif = generate_tif_curve(bench_id, items, num_points=21)
        assert tif.total_items == 3
        assert tif.peak_information > 0.0
        assert len(tif.curve) == 21
        assert all(pt.sem > 0.0 for pt in tif.curve)

    def test_wright_map_alignment(self):
        items = [
            IRTItemParameters(item_id=uuid.uuid4(), item_key="easy", difficulty=-1.5),
            IRTItemParameters(item_id=uuid.uuid4(), item_key="hard", difficulty=1.5),
        ]
        persons = [
            IRTPersonParameters(ai_model_id=uuid.uuid4(), ability=-0.5),
            IRTPersonParameters(ai_model_id=uuid.uuid4(), ability=1.0),
        ]

        wm = generate_wright_map(items, persons)
        assert wm.mean_item_difficulty == 0.0
        assert wm.mean_person_ability == 0.25
        assert len(wm.entries) == 4

    def test_plot_collection_assembly(self):
        bench_id = uuid.uuid4()
        items = [IRTItemParameters(item_id=uuid.uuid4(), item_key="q1", difficulty=0.0)]
        persons = [IRTPersonParameters(ai_model_id=uuid.uuid4(), ability=0.0)]

        collection = generate_diagnostic_plots(bench_id, items, persons)
        assert collection.benchmark_id == bench_id
        assert len(collection.icc_curves) == 1
        assert collection.tif_curve.total_items == 1
        assert len(collection.wright_map.entries) == 2


class TestScaleAndSparseQueryPerformance:
    async def test_sparse_response_query_performance_at_scale(self, session):
        """Charter §4.1 & §6.3: Sparse query performance validation (<1.0s budget)."""
        caller = CallerContext(user_id=uuid.uuid4())
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        # Seed 1,000 items x 20 models with 30% sparsity = ~6,000 response rows
        stats = await seed_database(
            bench_repo,
            item_repo,
            model_repo,
            resp_repo,
            caller,
            num_benchmarks=1,
            items_per_benchmark=500,
            num_ai_models=20,
            sparsity=0.5,
        )
        assert stats["responses"] > 1000

        benchmarks = await bench_repo.list_accessible(caller)
        bench_id = benchmarks[0].id

        # Measure query latency for get_by_benchmark
        start_t = time.perf_counter()
        responses = await resp_repo.get_by_benchmark(bench_id, caller)
        elapsed_t = time.perf_counter() - start_t

        assert len(responses) == stats["responses"]
        # Fast sparse retrieval threshold (<0.5 seconds)
        assert elapsed_t < 0.500

"""Statistical correctness test suite for Layer 4 analysis services.

Charter §5.4: "Statistical correctness tests: validate output against known
closed-form results (simulated data with known generating parameters)."
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

import pytest

from aims.analysis.items import ItemFlagPolicy, compute_item_analysis
from aims.analysis.models import ItemFlagReason
from aims.analysis.reliability import compute_reliability_analysis
from aims.db.models import Item, Response


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_item(benchmark_id: uuid.UUID, item_key: str) -> Item:
    return Item(
        id=uuid.uuid4(),
        benchmark_id=benchmark_id,
        item_key=item_key,
        max_score=1,
        created_at=_now(),
    )


def _make_responses(
    benchmark_id: uuid.UUID,
    item_id: uuid.UUID,
    model_scores: list[tuple[uuid.UUID, int]],
) -> list[Response]:
    return [
        Response(
            id=uuid.uuid4(),
            benchmark_id=benchmark_id,
            ai_model_id=m_id,
            item_id=item_id,
            score=score,
            response_version=1,
            is_current=True,
            created_at=_now(),
        )
        for m_id, score in model_scores
    ]


class TestItemAnalysisCorrectness:
    def test_difficulty_calculation_exact(self):
        bench_id = uuid.uuid4()
        item = _make_item(bench_id, "item_1")

        # 8 out of 10 models get it right → difficulty = 0.8
        models = [uuid.uuid4() for _ in range(10)]
        scores = [(m, 1 if i < 8 else 0) for i, m in enumerate(models)]
        responses = _make_responses(bench_id, item.id, scores)

        summary = compute_item_analysis(bench_id, [item], responses)
        assert len(summary.items) == 1
        assert summary.items[0].difficulty == 0.8

    def test_zero_variance_flag(self):
        bench_id = uuid.uuid4()
        item = _make_item(bench_id, "item_easy")

        # All models get it right → zero variance
        models = [uuid.uuid4() for _ in range(10)]
        scores = [(m, 1) for m in models]
        responses = _make_responses(bench_id, item.id, scores)

        summary = compute_item_analysis(bench_id, [item], responses)
        assert ItemFlagReason.ZERO_VARIANCE in summary.items[0].flags

    def test_negative_discrimination_and_key_error_flag(self):
        bench_id = uuid.uuid4()
        item_good = _make_item(bench_id, "item_good")
        item_bad = _make_item(bench_id, "item_inverted")
        item_a1 = _make_item(bench_id, "item_a1")
        item_a2 = _make_item(bench_id, "item_a2")

        items = [item_good, item_bad, item_a1, item_a2]
        models = [uuid.uuid4() for _ in range(20)]
        responses = []

        # Models 0..9: Low ability (good=0, a1=0, a2=0, inverted=1) → T - X_good = 1
        # Models 10..19: High ability (good=1, a1=1, a2=1, inverted=0) → T - X_good = 2
        for i, m in enumerate(models):
            if i < 10:  # Low ability
                responses.append(Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item_good.id, score=0, created_at=_now()))
                responses.append(Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item_a1.id, score=0, created_at=_now()))
                responses.append(Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item_a2.id, score=0, created_at=_now()))
                responses.append(Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item_bad.id, score=1, created_at=_now()))
            else:  # High ability
                responses.append(Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item_good.id, score=1, created_at=_now()))
                responses.append(Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item_a1.id, score=1, created_at=_now()))
                responses.append(Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item_a2.id, score=1, created_at=_now()))
                responses.append(Response(id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m, item_id=item_bad.id, score=0, created_at=_now()))

        summary = compute_item_analysis(bench_id, items, responses)
        stats_map = {st.item_key: st for st in summary.items}

        assert stats_map["item_good"].discrimination > 0.5
        assert ItemFlagReason.NEGATIVE_DISCRIMINATION in stats_map["item_inverted"].flags

class TestReliabilityAnalysisCorrectness:
    def test_perfect_reliability_alpha_near_one(self):
        bench_id = uuid.uuid4()
        items = [_make_item(bench_id, f"item_{i}") for i in range(5)]
        models = [uuid.uuid4() for _ in range(30)]

        responses = []
        # Perfectly consistent models: half get all 5 right, half get all 5 wrong
        for i, m in enumerate(models):
            score = 1 if i < 15 else 0
            for item in items:
                responses.append(Response(
                    id=uuid.uuid4(),
                    benchmark_id=bench_id,
                    ai_model_id=m,
                    item_id=item.id,
                    score=score,
                    created_at=_now(),
                ))

        result = compute_reliability_analysis(bench_id, items, responses)
        assert result.cronbach_alpha >= 0.95
        assert result.marginal_reliability >= 0.70
        assert len(result.sem_curve) == 41

    def test_random_responses_low_reliability(self):
        bench_id = uuid.uuid4()
        items = [_make_item(bench_id, f"item_{i}") for i in range(5)]
        models = [uuid.uuid4() for _ in range(50)]

        random.seed(42)
        responses = []
        for m in models:
            for item in items:
                responses.append(Response(
                    id=uuid.uuid4(),
                    benchmark_id=bench_id,
                    ai_model_id=m,
                    item_id=item.id,
                    score=random.choice([0, 1]),
                    created_at=_now(),
                ))

        result = compute_reliability_analysis(bench_id, items, responses)
        # Random noise → Cronbach alpha near 0
        assert result.cronbach_alpha < 0.30
        assert any("below standard reliability threshold" in w for w in result.warnings)

    def test_sem_curve_has_valid_theta_range(self):
        bench_id = uuid.uuid4()
        items = [_make_item(bench_id, f"item_{i}") for i in range(3)]
        models = [uuid.uuid4() for _ in range(10)]

        responses = [
            Response(
                id=uuid.uuid4(), benchmark_id=bench_id, ai_model_id=m,
                item_id=item.id, score=1 if i % 2 == 0 else 0, created_at=_now(),
            )
            for m in models for i, item in enumerate(items)
        ]

        result = compute_reliability_analysis(bench_id, items, responses, num_grid_points=11)
        assert len(result.sem_curve) == 11
        assert result.sem_curve[0].theta == -4.0
        assert result.sem_curve[-1].theta == 4.0
        assert all(p.sem > 0 for p in result.sem_curve)
        assert all(p.information >= 0 for p in result.sem_curve)

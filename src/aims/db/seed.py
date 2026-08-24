"""Synthetic data generator for AIMS Layer 5 testing (Sub-Task 0.7).

Generates realistic benchmark data at configurable scale:
  - Benchmarks with varying item counts
  - Items with binary and polytomous responses
  - AI models with sparse response patterns
  - Historical response versions

Charter §7 Phase 0 exit criteria:
  "Seeded with a real (or realistic synthetic) benchmark."
  "Can query items/responses for at least one benchmark at representative scale."
"""

from __future__ import annotations

import hashlib
import random
import uuid
from datetime import datetime, timezone
from typing import Sequence

from aims.db.models import (
    AIModel,
    Benchmark,
    CallerContext,
    Item,
    ItemResponseType,
    Response,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def generate_benchmarks(
    owner_id: uuid.UUID,
    count: int = 5,
    *,
    public_ratio: float = 0.4,
) -> list[Benchmark]:
    """Generate synthetic benchmarks with realistic variety."""
    benchmarks = []
    for i in range(count):
        benchmarks.append(Benchmark(
            id=uuid.uuid4(),
            name=f"Benchmark-{i:04d}",
            description=f"Synthetic benchmark #{i} for testing",
            owner_id=owner_id,
            is_public=random.random() < public_ratio,
            item_response_type=ItemResponseType.BINARY,
            created_at=_now(),
        ))
    return benchmarks


def generate_items(
    benchmark_id: uuid.UUID,
    count: int = 100,
) -> list[Item]:
    """Generate synthetic items for a benchmark."""
    return [
        Item(
            id=uuid.uuid4(),
            benchmark_id=benchmark_id,
            item_key=f"item_{i:06d}",
            content_hash=hashlib.sha256(f"item_{i}_{benchmark_id}".encode()).hexdigest()[:16],
            max_score=1,
            created_at=_now(),
        )
        for i in range(count)
    ]


def generate_ai_models(
    owner_id: uuid.UUID,
    count: int = 10,
) -> list[AIModel]:
    """Generate synthetic AI models (the systems being evaluated)."""
    model_names = [
        "GPT-4o", "Claude-3.5", "Gemini-Pro", "Llama-3-70B",
        "Mistral-Large", "Command-R+", "Qwen-72B", "DeepSeek-V2",
        "Phi-3-Medium", "Yi-34B",
    ]
    return [
        AIModel(
            id=uuid.uuid4(),
            name=model_names[i % len(model_names)] + f"-v{i // len(model_names)}",
            owner_id=owner_id,
            created_at=_now(),
        )
        for i in range(count)
    ]


def generate_sparse_responses(
    benchmark_id: uuid.UUID,
    ai_models: Sequence[AIModel],
    items: Sequence[Item],
    *,
    sparsity: float = 0.3,
    version: int = 1,
) -> list[Response]:
    """Generate sparse response matrix.

    Charter §4.1: "most model×item pairs are unobserved."
    Sparsity controls what fraction of pairs are OBSERVED (not missing).
    """
    responses = []
    for model in ai_models:
        for item in items:
            # Skip based on sparsity — absent = unobserved
            if random.random() > sparsity:
                continue
            responses.append(Response(
                id=uuid.uuid4(),
                benchmark_id=benchmark_id,
                ai_model_id=model.id,
                item_id=item.id,
                score=random.randint(0, item.max_score),
                response_version=version,
                is_current=True,
                created_at=_now(),
            ))
    return responses


async def seed_database(
    bench_repo,
    item_repo,
    model_repo,
    resp_repo,
    caller: CallerContext,
    *,
    num_benchmarks: int = 5,
    items_per_benchmark: int = 100,
    num_ai_models: int = 10,
    sparsity: float = 0.3,
) -> dict:
    """Seed the database with synthetic data and return summary stats."""
    total_items = 0
    total_responses = 0

    benchmarks = generate_benchmarks(caller.user_id, num_benchmarks)
    ai_models = generate_ai_models(caller.user_id, num_ai_models)

    # Create AI models
    for model in ai_models:
        await model_repo.create(model, caller)

    for bench in benchmarks:
        # Create benchmark
        created_bench = await bench_repo.create(bench, caller)

        # Create items
        items = generate_items(created_bench.id, items_per_benchmark)
        await item_repo.create_bulk(items, caller)
        total_items += len(items)

        # Create sparse responses
        responses = generate_sparse_responses(
            created_bench.id, ai_models, items, sparsity=sparsity,
        )
        if responses:
            await resp_repo.create_bulk(responses, caller)
            total_responses += len(responses)

    return {
        "benchmarks": num_benchmarks,
        "items": total_items,
        "ai_models": num_ai_models,
        "responses": total_responses,
        "sparsity": sparsity,
    }

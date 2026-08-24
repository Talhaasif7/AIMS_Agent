"""Repository interfaces (protocols) for AIMS Layer 5.

These define the contract between Layer 4 (backend) and Layer 5 (data).
Layer 4 imports ONLY these protocols — never SQLAlchemy or SQL.

Charter §4.1: "Write data access layer (repository pattern) that Layer 4 calls —
Layer 4 should never touch SQL directly, so this is swappable later."

Every method that returns data MUST accept a CallerContext parameter.
There is no bypass path for access control.
"""

from __future__ import annotations

from typing import Protocol, Sequence
from uuid import UUID

from aims.db.models import (
    AIModel,
    Benchmark,
    CallerContext,
    Fit,
    Item,
    Report,
    Response,
)


class BenchmarkRepository(Protocol):
    """CRUD + list-by-owner for benchmarks."""

    async def create(self, benchmark: Benchmark, caller: CallerContext) -> Benchmark: ...

    async def get_by_id(self, benchmark_id: UUID, caller: CallerContext) -> Benchmark | None: ...

    async def list_accessible(self, caller: CallerContext) -> Sequence[Benchmark]: ...

    async def delete(self, benchmark_id: UUID, caller: CallerContext) -> bool: ...


class ItemRepository(Protocol):
    """CRUD + list-by-benchmark for items."""

    async def create_bulk(
        self, items: Sequence[Item], caller: CallerContext,
    ) -> Sequence[Item]: ...

    async def list_by_benchmark(
        self, benchmark_id: UUID, caller: CallerContext,
    ) -> Sequence[Item]: ...


class AIModelRepository(Protocol):
    """CRUD for AI models (the systems being evaluated)."""

    async def create(self, ai_model: AIModel, caller: CallerContext) -> AIModel: ...

    async def get_by_id(self, model_id: UUID, caller: CallerContext) -> AIModel | None: ...

    async def list_accessible(self, caller: CallerContext) -> Sequence[AIModel]: ...


class ResponseRepository(Protocol):
    """Sparse response matrix access.

    Charter §4.1: "Response matrix must support sparse access."
    Absent rows = unobserved. Never return defaults for missing pairs.
    """

    async def create_bulk(
        self, responses: Sequence[Response], caller: CallerContext,
    ) -> int:
        """Insert responses in bulk. Returns count of rows inserted."""
        ...

    async def get_by_benchmark(
        self,
        benchmark_id: UUID,
        caller: CallerContext,
        *,
        current_only: bool = True,
    ) -> Sequence[Response]:
        """All responses for a benchmark (primary IRT access pattern)."""
        ...

    async def get_by_model_and_benchmark(
        self,
        ai_model_id: UUID,
        benchmark_id: UUID,
        caller: CallerContext,
        *,
        current_only: bool = True,
    ) -> Sequence[Response]: ...

    async def get_at_version(
        self,
        benchmark_id: UUID,
        version: int,
        caller: CallerContext,
    ) -> Sequence[Response]:
        """Historical version access for reproducibility (§5.5)."""
        ...

    async def supersede(
        self,
        benchmark_id: UUID,
        new_responses: Sequence[Response],
        caller: CallerContext,
    ) -> int:
        """Append new version, mark old as non-current (§4.1 versioning)."""
        ...


class FitRepository(Protocol):
    """CRUD + cache lookup for stored IRT fits."""

    async def create(self, fit: Fit, caller: CallerContext) -> Fit: ...

    async def get_by_id(self, fit_id: UUID, caller: CallerContext) -> Fit | None: ...

    async def list_by_benchmark(
        self, benchmark_id: UUID, caller: CallerContext,
    ) -> Sequence[Fit]: ...

    async def find_cached(
        self,
        dataset_hash: str,
        model_family: str,
        random_seed: int,
        caller: CallerContext,
    ) -> Fit | None:
        """Cache lookup: same (dataset, model, seed) → hit (§4.2)."""
        ...


class ReportRepository(Protocol):
    """CRUD for generated reports."""

    async def create(self, report: Report, caller: CallerContext) -> Report: ...

    async def list_by_fit(
        self, fit_id: UUID, caller: CallerContext,
    ) -> Sequence[Report]: ...

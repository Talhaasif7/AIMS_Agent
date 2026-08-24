"""SQLAlchemy ORM table definitions for AIMS Layer 5 (Measurement DB).

Security notes (per secure coding guidelines):
  - All queries MUST use parameterized statements via SQLAlchemy ORM/Core.
  - MUST NOT use string concatenation for SQL (§ SQL Injection prevention).
  - Row-level access control enforced at repository layer, not here.

Charter references:
  - §4.1: Full schema spec (benchmarks, items, models, responses, fits, reports).
  - §5.5: Non-nullable reproducibility columns enforced at DDL level.
  - §4.1: "fits.parameter_estimates as artifact reference, not inline blob."
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from aims.db.models import ConvergenceStatus, ItemResponseType, ModelFamily


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class BenchmarkRow(Base):
    __tablename__ = "benchmarks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    item_response_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ItemResponseType.BINARY.value,
    )
    created_at: Mapped[str] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[str | None] = mapped_column(DateTime, nullable=True, onupdate=func.now())

    # Relationships
    items = relationship("ItemRow", back_populates="benchmark", cascade="all, delete-orphan")
    responses = relationship("ResponseRow", back_populates="benchmark", cascade="all, delete-orphan")
    fits = relationship("FitRow", back_populates="benchmark", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

class ItemRow(Base):
    __tablename__ = "items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    benchmark_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("benchmarks.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    item_key: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    max_score: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[str] = mapped_column(DateTime, nullable=False, server_default=func.now())

    # Unique item_key within a benchmark
    __table_args__ = (
        UniqueConstraint("benchmark_id", "item_key", name="uq_benchmark_item_key"),
    )

    benchmark = relationship("BenchmarkRow", back_populates="items")
    responses = relationship("ResponseRow", back_populates="item", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# AI Models (the systems being evaluated — NOT IRT models)
# ---------------------------------------------------------------------------

class AIModelRow(Base):
    __tablename__ = "ai_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    created_at: Mapped[str] = mapped_column(DateTime, nullable=False, server_default=func.now())

    responses = relationship("ResponseRow", back_populates="ai_model", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Responses (sparse model × item matrix)
#
# Charter §4.1: "preserve missingness, not impute it"
# Absent rows = unobserved. Never store a row with a sentinel value for
# "not tested." The IRT math depends on this.
# ---------------------------------------------------------------------------

class ResponseRow(Base):
    __tablename__ = "responses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    benchmark_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("benchmarks.id", ondelete="CASCADE"), nullable=False,
    )
    ai_model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_models.id", ondelete="CASCADE"), nullable=False,
    )
    item_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("items.id", ondelete="CASCADE"), nullable=False,
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- Historical versioning (Charter §4.1) ---
    # "old responses must not be overwritten; append + supersede"
    response_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[str] = mapped_column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        # Fast lookups by benchmark (primary access pattern for IRT fitting)
        Index("ix_responses_benchmark", "benchmark_id"),
        # Fast lookups by model × benchmark
        Index("ix_responses_model_benchmark", "ai_model_id", "benchmark_id"),
        # Uniqueness: one response per (model, item, version)
        UniqueConstraint(
            "ai_model_id", "item_id", "response_version",
            name="uq_response_model_item_version",
        ),
        # Filter current responses efficiently
        Index("ix_responses_current", "benchmark_id", "is_current"),
    )

    benchmark = relationship("BenchmarkRow", back_populates="responses")
    ai_model = relationship("AIModelRow", back_populates="responses")
    item = relationship("ItemRow", back_populates="responses")


# ---------------------------------------------------------------------------
# Fits (stored IRT model runs)
#
# Charter §4.1 + §5.5: All reproducibility columns are NON-NULLABLE.
# "Enforce this at the data-model level (non-nullable columns),
#  not as a convention people forget."
# ---------------------------------------------------------------------------

class FitRow(Base):
    __tablename__ = "fits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    benchmark_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("benchmarks.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    model_family: Mapped[str] = mapped_column(String(10), nullable=False)
    hyperparameters: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    convergence_status: Mapped[str] = mapped_column(String(20), nullable=False)

    # Artifact reference, NOT inline blob (Charter §4.1)
    parameter_artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    fit_statistics: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    # --- Reproducibility metadata (ALL non-nullable per §5.5) ---
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    torch_measure_version: Mapped[str] = mapped_column(String(32), nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime, nullable=False, server_default=func.now())

    owner_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    __table_args__ = (
        # Cache lookup: same (dataset, model, seed) → cache hit (Charter §4.2)
        Index("ix_fits_cache_key", "dataset_hash", "model_family", "random_seed"),
    )

    benchmark = relationship("BenchmarkRow", back_populates="fits")
    reports = relationship("ReportRow", back_populates="fit", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class ReportRow(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    fit_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("fits.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)

    # Reproducibility (non-nullable)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    torch_measure_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime, nullable=False, server_default=func.now())

    owner_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    fit = relationship("FitRow", back_populates="reports")

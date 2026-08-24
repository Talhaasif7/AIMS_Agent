"""Domain models (Pydantic) for the AIMS Measurement Agent.

These are the typed data contracts that Layer 4 (backend) works with.
Layer 4 never sees SQLAlchemy — only these models.

Charter references:
  - §3.4: "Statistical outputs are typed and versioned, not prose."
  - §5.5: "Every stored fit and every generated report carries:
           torch_measure version, model spec, dataset hash, seed, timestamp."
  - §4.1: "models (the AI systems being evaluated, not IRT models)"
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelFamily(str, enum.Enum):
    """IRT model families supported by torch_measure.

    Charter §4.2: "Support model families: 1PL (Rasch), 2PL, 3PL, GPCM, GRM."
    """
    RASCH_1PL = "1PL"
    TWO_PL = "2PL"
    THREE_PL = "3PL"
    GPCM = "GPCM"
    GRM = "GRM"


class ConvergenceStatus(str, enum.Enum):
    """Convergence outcome for an IRT fit.

    Charter §4.2: "never return 'success' on a run that hit max-iterations
    without convergence; return a distinct status."
    """
    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    FAILED = "failed"


class ItemResponseType(str, enum.Enum):
    """Whether items are binary (correct/incorrect) or polytomous (graded)."""
    BINARY = "binary"
    POLYTOMOUS = "polytomous"


# ---------------------------------------------------------------------------
# Caller context (access control foundation)
# ---------------------------------------------------------------------------

class CallerContext(BaseModel):
    """Represents an authenticated caller for access-control scoping.

    Charter §4.1: "scope every query by caller identity from day one."
    Phase 0 builds the data structure; Phase 3 populates it from OAuth.
    """
    user_id: UUID
    org_id: Optional[UUID] = None


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class Benchmark(BaseModel):
    """A benchmark (test/evaluation instrument) containing items."""
    id: UUID
    name: str
    description: Optional[str] = None
    owner_id: UUID
    is_public: bool = False
    item_response_type: ItemResponseType = ItemResponseType.BINARY
    created_at: datetime
    updated_at: Optional[datetime] = None


class Item(BaseModel):
    """A single item (question/task) within a benchmark."""
    id: UUID
    benchmark_id: UUID
    item_key: str = Field(description="Unique key within the benchmark (e.g., 'q1', 'task_42')")
    content_hash: Optional[str] = Field(
        default=None,
        description="Hash of item content for change detection",
    )
    max_score: int = Field(
        default=1,
        ge=0,
        description="Maximum score (1 for binary, >1 for polytomous)",
    )
    created_at: datetime


class AIModel(BaseModel):
    """An AI system being evaluated (NOT an IRT model).

    Named 'AIModel' to avoid collision with IRT 'model family'.
    Charter §4.1: "models (the AI systems being evaluated, not IRT models)."
    """
    id: UUID
    name: str
    description: Optional[str] = None
    owner_id: UUID
    created_at: datetime


class Response(BaseModel):
    """A single model × item response (sparse — absent means unobserved).

    Charter §4.1: "Response matrix must support sparse access
    (most model×item pairs are unobserved)."
    """
    id: UUID
    benchmark_id: UUID
    ai_model_id: UUID
    item_id: UUID
    score: int = Field(ge=0, description="0/1 for binary, 0..max_score for polytomous")
    response_version: int = Field(default=1, ge=1)
    is_current: bool = True
    created_at: datetime


class Fit(BaseModel):
    """A stored IRT model fit with full reproducibility metadata.

    Charter §4.1: "Add a fits table that stores: model family, hyperparameters,
    fit timestamp, convergence status, item/person parameter estimates
    (as artifact reference, not inline blob), fit statistics, library version, seed."

    Charter §5.5: Non-nullable reproducibility fields enforced here.
    """
    id: UUID
    benchmark_id: UUID
    model_family: ModelFamily
    hyperparameters: dict = Field(default_factory=dict)
    convergence_status: ConvergenceStatus
    parameter_artifact_uri: str = Field(
        description="URI to stored parameter estimates (not inline blob per §4.1)",
    )
    fit_statistics: dict = Field(
        default_factory=dict,
        description="AIC, BIC, -2LL, item/person fit stats",
    )
    # --- Reproducibility metadata (all non-nullable per §5.5) ---
    dataset_hash: str
    torch_measure_version: str
    random_seed: int
    created_at: datetime

    owner_id: UUID


class Report(BaseModel):
    """A generated measurement report with provenance.

    Charter §4.2: "Every report must embed the reproducibility metadata."
    """
    id: UUID
    fit_id: UUID
    format: str = Field(description="pdf, html, or json")
    artifact_uri: str
    # --- Reproducibility (inherited from fit + report-specific) ---
    dataset_hash: str
    torch_measure_version: str
    created_at: datetime
    owner_id: UUID

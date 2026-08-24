"""Pydantic data models for the IRT Fitting Engine (Layer 4 backend logic).

Charter references:
  - §3.2: "Long-running work never blocks the MCP call. Every compute-heavy tool
           must support async job semantics (submit -> job_id -> poll/webhook)."
  - §3.4: "Statistical outputs are typed and versioned, not prose."
  - §4.1: "fits table: model family, hyperparameters, fit timestamp,
           convergence status, parameter estimates (as artifact reference),
           fit statistics, library version, seed."
  - §5.5: Non-nullable reproducibility metadata fields.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from aims.db.models import ConvergenceStatus, ModelFamily


class JobState(str, enum.Enum):
    """Lifecycle states for async fitting jobs.

    Charter §3.2 & §4.3: Async job pattern at protocol level.
    """
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FitRequest(BaseModel):
    """Input payload to request an IRT model fit."""
    benchmark_id: UUID
    model_family: ModelFamily = Field(
        default=ModelFamily.TWO_PL,
        description="IRT model family: 1PL (Rasch) or 2PL for Phase 2",
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional hyperparameters: max_iter, tolerance, etc.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for deterministic estimation (§5.5)",
    )


class IRTItemParameters(BaseModel):
    """Estimated parameters for a single benchmark item."""
    item_id: UUID
    item_key: str
    difficulty: float = Field(description="Item difficulty parameter b_j")
    discrimination: float = Field(default=1.0, description="Item discrimination parameter a_j")
    guessing: float = Field(default=0.0, description="Item guessing parameter c_j (0 for 1PL/2PL)")
    difficulty_se: Optional[float] = None
    discrimination_se: Optional[float] = None


class IRTPersonParameters(BaseModel):
    """Estimated ability parameter for an evaluated AI system."""
    ai_model_id: UUID
    ability: float = Field(description="Latent ability parameter theta_i")
    ability_se: Optional[float] = None


class IRTFitResult(BaseModel):
    """Result of a completed IRT model fit."""
    fit_id: UUID
    benchmark_id: UUID
    model_family: ModelFamily
    convergence_status: ConvergenceStatus
    fit_statistics: Dict[str, float] = Field(
        description="AIC, BIC, -2LL, num_parameters, max_gradient_norm",
    )
    parameter_artifact_uri: str = Field(
        description="URI/path to detailed parameter estimates file (§4.1)",
    )

    # --- Reproducibility metadata (Non-nullable per §5.5) ---
    dataset_hash: str
    torch_measure_version: str
    random_seed: int
    created_at: datetime

    owner_id: UUID
    is_cached: bool = Field(default=False, description="True if returned from cache")


class JobHandle(BaseModel):
    """Handle returned for an async job execution.

    Charter §4.3: "return a job handle immediately with a check_status-style follow-up"
    """
    job_id: UUID
    status: JobState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    fit_result: Optional[IRTFitResult] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

"""Internal representation models for Reproducible Measurement Reports.

Charter references:
  - §4.5: "Report templates (PDF/HTML/JSON) — build from a single internal
           representation so all three formats stay in sync; never hand-maintain
           three separate report builders."
  - §3.5 & §5.5: Every report must embed dataset_hash, model_spec, torch_measure_version,
           seed, and timestamp.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID

from pydantic import BaseModel, Field


class ReportData(BaseModel):
    """Single internal representation for all report formats (§4.5)."""

    report_id: UUID
    fit_id: UUID
    benchmark_id: UUID
    benchmark_name: str
    owner_id: UUID
    format: str = Field(description="html, json, or pdf")

    # --- Fit & Reliability Summaries ---
    model_family: str
    convergence_status: str
    fit_statistics: Dict[str, float]
    num_items: int
    num_models_evaluated: int
    mean_difficulty: float
    mean_discrimination: float
    cronbach_alpha: float
    marginal_reliability: float
    overall_sem: float

    # --- Recommendations & Warnings ---
    flagged_item_count: int
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    # --- Reproducibility Metadata (§5.5) ---
    dataset_hash: str
    torch_measure_version: str
    random_seed: int
    created_at: datetime

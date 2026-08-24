"""Pydantic data models for Layer 4 statistical analysis outputs.

Charter references:
  - §3.4: "Statistical outputs are typed and versioned, not prose."
  - §4.2: analyze_items & reliability_analysis backend logic.
  - §4.2: "Standard error of measurement (SE) at relevant ability levels —
           report it as a function, not a scalar."
  - §5.3: Structured summary + typed key numbers inline.
"""

from __future__ import annotations

import enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ItemFlagReason(str, enum.Enum):
    """Reasons an item might be flagged as problematic during item analysis.

    Charter §4.2: "Flag problematic items (misfitting, low discrimination,
    near-zero variance, potential key errors) with a defined threshold policy."
    """
    ZERO_VARIANCE = "zero_variance"
    LOW_DISCRIMINATION = "low_discrimination"
    NEGATIVE_DISCRIMINATION = "negative_discrimination"
    MISFITTING_INSET = "misfitting_infit"
    MISFITTING_OUTFIT = "misfitting_outfit"
    POTENTIAL_KEY_ERROR = "potential_key_error"


class ItemStats(BaseModel):
    """Statistical properties of a single benchmark item."""
    item_id: UUID
    item_key: str
    difficulty: float = Field(
        description="Proportion correct (binary) or normalized mean score (polytomous) in [0, 1]",
    )
    discrimination: float = Field(
        description="Point-biserial correlation between item score and total test score",
    )
    infit_mnsq: Optional[float] = Field(
        default=None,
        description="Infit mean-square statistic (inlier-sensitive fit)",
    )
    outfit_mnsq: Optional[float] = Field(
        default=None,
        description="Outfit mean-square statistic (outlier-sensitive fit)",
    )
    flags: List[ItemFlagReason] = Field(
        default_factory=list,
        description="List of quality flags triggered for this item",
    )


class ItemAnalysisSummary(BaseModel):
    """Aggregate summary of item analysis across a benchmark."""
    benchmark_id: UUID
    num_items: int
    num_models_evaluated: int
    mean_difficulty: float
    mean_discrimination: float
    flagged_item_count: int
    items: List[ItemStats]
    warnings: List[str] = Field(default_factory=list)


class SEMPoint(BaseModel):
    """Standard Error of Measurement at a specific ability theta level.

    Charter §4.2: "report it as a function, not a scalar."
    """
    theta: float = Field(description="Latent ability level (typically -4.0 to +4.0)")
    sem: float = Field(description="Standard error of measurement at this theta level")
    information: float = Field(description="Test information value at this theta level")


class ReliabilityAnalysisResult(BaseModel):
    """Typed result of a reliability analysis run.

    Charter §4.2: Cronbach's alpha + IRT marginal reliability + SEM curve.
    """
    benchmark_id: UUID
    num_items: int
    num_models_evaluated: int
    cronbach_alpha: float = Field(
        description="Classical Test Theory internal consistency reliability",
    )
    marginal_reliability: float = Field(
        description="IRT-native marginal reliability across ability distribution",
    )
    overall_sem: float = Field(
        description="Overall standard error of measurement",
    )
    sem_curve: List[SEMPoint] = Field(
        default_factory=list,
        description="SEM and Test Information evaluated across theta grid [-4.0, 4.0]",
    )
    warnings: List[str] = Field(default_factory=list)

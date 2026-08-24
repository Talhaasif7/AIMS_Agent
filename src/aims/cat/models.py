"""Pydantic data models for Computerized Adaptive Testing (CAT) Engine.

Charter references:
  - §4.2: "CAT simulation engine: item selection strategy (max information, Fisher info),
           stopping rule (SE threshold or fixed length), exposure control."
  - §3.4: "Statistical outputs are typed and versioned, not prose."
"""

from __future__ import annotations

import enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ItemSelectionStrategy(str, enum.Enum):
    """Item selection strategies for CAT adaptive testing."""
    MAX_INFORMATION = "max_information"
    FISHER_INFORMATION = "fisher_information"
    RANDOM_BASELINE = "random_baseline"


class StoppingCriterion(str, enum.Enum):
    """Stopping rules for CAT adaptive testing."""
    SE_THRESHOLD = "se_threshold"
    MAX_ITEMS = "max_items"
    EITHER = "either"


class CATStep(BaseModel):
    """A single step in a CAT adaptive testing sequence."""
    step_number: int
    item_id: UUID
    item_key: str
    response_score: int
    ability_estimate: float = Field(description="Updated theta estimate after administering this item")
    sem: float = Field(description="Standard Error of Measurement after this step")
    information: float = Field(description="Item information provided by the selected item at theta")


class CATSimulationResult(BaseModel):
    """Result of a CAT adaptive testing simulation run."""
    benchmark_id: UUID
    fit_id: Optional[UUID] = None
    true_theta: float = Field(description="Simulated true latent ability theta* of the test-taker")
    final_ability_estimate: float = Field(description="Final estimated ability theta hat")
    final_sem: float = Field(description="Final Standard Error of Measurement")
    num_items_administered: int
    total_bank_items: int
    selection_strategy: ItemSelectionStrategy
    stopped_reason: str
    trajectory: List[CATStep] = Field(default_factory=list)

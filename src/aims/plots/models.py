"""Pydantic data models for Diagnostic Plots Generator.

Charter references:
  - §4.2: "Diagnostics and plots: Item Characteristic Curves (ICC), Test Information
           Functions (TIF), Wright maps, residual distributions."
  - §5.3: "Return data/spec for charts (or ASCII summary inline) + reference
           to generated plot files; don't dump raw image bytes directly into tool calls."
"""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ICCPoint(BaseModel):
    """A point on an Item Characteristic Curve."""
    theta: float
    probability: float


class ICCData(BaseModel):
    """Item Characteristic Curve data for a single item."""
    item_id: UUID
    item_key: str
    difficulty: float
    discrimination: float
    curve: List[ICCPoint]


class TIFPoint(BaseModel):
    """A point on a Test Information Function curve."""
    theta: float
    information: float
    sem: float


class TIFData(BaseModel):
    """Test Information Function data for a benchmark."""
    benchmark_id: UUID
    total_items: int
    peak_theta: float
    peak_information: float
    curve: List[TIFPoint]


class WrightMapEntry(BaseModel):
    """An entry on a Wright Map (comparing item difficulties to person abilities)."""
    id: UUID
    label: str
    value: float
    entity_type: str = Field(description="item or model")


class WrightMapData(BaseModel):
    """Wright Map data comparing item difficulty and model ability distributions."""
    min_value: float
    max_value: float
    mean_item_difficulty: float
    mean_person_ability: float
    entries: List[WrightMapEntry]


class PlotCollection(BaseModel):
    """Complete collection of diagnostic plot specs for a fitted IRT model."""
    benchmark_id: UUID
    fit_id: Optional[UUID] = None
    icc_curves: List[ICCData]
    tif_curve: TIFData
    wright_map: WrightMapData

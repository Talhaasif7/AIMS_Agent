"""Diagnostic Plots Generator for AIMS Measurement Agent (Layer 4).

Generates structured plot specs for:
  - Item Characteristic Curves (ICC)
  - Test Information Function (TIF)
  - Wright Maps (comparing item difficulty vs person ability distributions)

Charter §4.2: "Diagnostics and plots: Item Characteristic Curves (ICC),
Test Information Functions (TIF), Wright maps, residual distributions."
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence
from uuid import UUID

from aims.irt.models import IRTItemParameters, IRTPersonParameters
from aims.plots.models import (
    ICCData,
    ICCPoint,
    PlotCollection,
    TIFData,
    TIFPoint,
    WrightMapData,
    WrightMapEntry,
)


def generate_icc_curves(
    item_params: Sequence[IRTItemParameters],
    *,
    num_points: int = 41,
    theta_min: float = -4.0,
    theta_max: float = 4.0,
) -> List[ICCData]:
    """Generate Item Characteristic Curve (ICC) data points for all items."""
    step = (theta_max - theta_min) / (num_points - 1)
    theta_grid = [theta_min + i * step for i in range(num_points)]

    icc_list: List[ICCData] = []
    for item in item_params:
        curve_points: List[ICCPoint] = []
        for theta in theta_grid:
            z = item.discrimination * (theta - item.difficulty)
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
            p_final = item.guessing + (1.0 - item.guessing) * p
            curve_points.append(ICCPoint(
                theta=round(theta, 2),
                probability=round(p_final, 4),
            ))

        icc_list.append(ICCData(
            item_id=item.item_id,
            item_key=item.item_key,
            difficulty=item.difficulty,
            discrimination=item.discrimination,
            curve=curve_points,
        ))

    return icc_list


def generate_tif_curve(
    benchmark_id: UUID,
    item_params: Sequence[IRTItemParameters],
    *,
    num_points: int = 41,
    theta_min: float = -4.0,
    theta_max: float = 4.0,
) -> TIFData:
    """Generate Test Information Function (TIF) curve data for a benchmark."""
    step = (theta_max - theta_min) / (num_points - 1)
    theta_grid = [theta_min + i * step for i in range(num_points)]

    tif_points: List[TIFPoint] = []
    max_info = 0.0
    peak_theta = 0.0

    for theta in theta_grid:
        total_info = 0.0
        for item in item_params:
            z = item.discrimination * (theta - item.difficulty)
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
            info_j = (item.discrimination ** 2) * p * (1.0 - p)
            total_info += info_j

        sem = 1.0 / math.sqrt(total_info) if total_info > 1e-6 else 9.99

        if total_info > max_info:
            max_info = total_info
            peak_theta = theta

        tif_points.append(TIFPoint(
            theta=round(theta, 2),
            information=round(total_info, 4),
            sem=round(sem, 4),
        ))

    return TIFData(
        benchmark_id=benchmark_id,
        total_items=len(item_params),
        peak_theta=round(peak_theta, 2),
        peak_information=round(max_info, 4),
        curve=tif_points,
    )


def generate_wright_map(
    item_params: Sequence[IRTItemParameters],
    person_params: Sequence[IRTPersonParameters],
) -> WrightMapData:
    """Generate Wright Map data aligning item difficulties and person abilities."""
    entries: List[WrightMapEntry] = []

    diffs = [it.difficulty for it in item_params]
    abilities = [pers.ability for pers in person_params]

    for it in item_params:
        entries.append(WrightMapEntry(
            id=it.item_id,
            label=it.item_key,
            value=it.difficulty,
            entity_type="item",
        ))

    for pers in person_params:
        entries.append(WrightMapEntry(
            id=pers.ai_model_id,
            label=str(pers.ai_model_id)[:8],
            value=pers.ability,
            entity_type="model",
        ))

    all_vals = diffs + abilities
    min_v = min(all_vals) if all_vals else -3.0
    max_v = max(all_vals) if all_vals else 3.0
    mean_d = sum(diffs) / len(diffs) if diffs else 0.0
    mean_a = sum(abilities) / len(abilities) if abilities else 0.0

    return WrightMapData(
        min_value=round(min_v, 3),
        max_value=round(max_v, 3),
        mean_item_difficulty=round(mean_d, 3),
        mean_person_ability=round(mean_a, 3),
        entries=entries,
    )


def generate_diagnostic_plots(
    benchmark_id: UUID,
    item_params: Sequence[IRTItemParameters],
    person_params: Sequence[IRTPersonParameters],
    fit_id: Optional[UUID] = None,
) -> PlotCollection:
    """Generate complete collection of diagnostic plot specs."""
    return PlotCollection(
        benchmark_id=benchmark_id,
        fit_id=fit_id,
        icc_curves=generate_icc_curves(item_params),
        tif_curve=generate_tif_curve(benchmark_id, item_params),
        wright_map=generate_wright_map(item_params, person_params),
    )

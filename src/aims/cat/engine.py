"""CAT Simulation Engine (Layer 4 backend logic).

Charter §4.2:
  - "CAT simulation engine: item selection strategy (max information, Fisher info),
     stopping rule (SE threshold or fixed length), exposure control."
  - "Must run as a simulation against the stored item bank + fitted model —
     not require the caller to supply the entire item bank inline."
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple
from uuid import UUID

from aims.cat.models import (
    CATSimulationResult,
    CATStep,
    ItemSelectionStrategy,
    StoppingCriterion,
)
from aims.irt.models import IRTItemParameters


def _normal_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _item_info_2pl(a_j: float, b_j: float, theta: float) -> float:
    z = a_j * (theta - b_j)
    p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
    return (a_j ** 2) * p * (1.0 - p)


def _eap_update(
    administered_items: Sequence[Tuple[float, float, int]],  # (a_j, b_j, score)
    grid_points: Sequence[float],
) -> Tuple[float, float]:
    """Expected A Posteriori (EAP) ability theta estimate and SEM."""
    if not administered_items:
        return 0.0, 1.0  # Prior N(0, 1)

    step = grid_points[1] - grid_points[0]
    weights = []
    num_sum = 0.0
    den_sum = 0.0

    for theta in grid_points:
        log_lik = 0.0
        for a_j, b_j, y in administered_items:
            z = a_j * (theta - b_j)
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
            p_clamped = max(1e-12, min(1.0 - 1e-12, p))
            log_lik += y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped)

        # Prior density N(0, 1)
        w = math.exp(max(-100.0, log_lik)) * _normal_pdf(theta) * step
        weights.append(w)
        num_sum += theta * w
        den_sum += w

    if den_sum < 1e-12:
        return 0.0, 1.0

    hat_theta = num_sum / den_sum

    # Variance / SEM calculation
    var_sum = sum(((theta - hat_theta) ** 2) * w for theta, w in zip(grid_points, weights))
    sem = math.sqrt(var_sum / den_sum)

    return round(hat_theta, 4), round(sem, 4)


def run_cat_simulation(
    benchmark_id: UUID,
    item_bank: Sequence[IRTItemParameters],
    true_theta: float,
    *,
    strategy: ItemSelectionStrategy = ItemSelectionStrategy.MAX_INFORMATION,
    target_se: float = 0.35,
    max_items: int = 15,
    exposure_control_k: int = 3,
    fit_id: Optional[UUID] = None,
    random_seed: Optional[int] = None,
) -> CATSimulationResult:
    """Run an adaptive testing simulation against a fitted item bank with exposure control."""
    if random_seed is not None:
        random.seed(random_seed)

    num_bank_items = len(item_bank)
    if num_bank_items == 0:
        return CATSimulationResult(
            benchmark_id=benchmark_id,
            fit_id=fit_id,
            true_theta=true_theta,
            final_ability_estimate=0.0,
            final_sem=1.0,
            num_items_administered=0,
            total_bank_items=0,
            selection_strategy=strategy,
            stopped_reason="Empty item bank",
            trajectory=[],
        )

    # Grid for EAP estimation
    grid_points = [-4.0 + i * 0.2 for i in range(41)]

    unadministered = list(item_bank)
    administered_tuples: List[Tuple[float, float, int]] = []
    trajectory: List[CATStep] = []

    hat_theta = 0.0
    sem = 1.0
    stopped_reason = f"Max items reached ({max_items})"

    for step_num in range(1, max_items + 1):
        if not unadministered:
            stopped_reason = "Item bank exhausted"
            break

        # Select next item based on strategy & exposure control
        if strategy in (ItemSelectionStrategy.MAX_INFORMATION, ItemSelectionStrategy.FISHER_INFORMATION):
            # Sort unadministered candidates by information at current theta_hat
            sorted_candidates = sorted(
                unadministered,
                key=lambda it: _item_info_2pl(it.discrimination, it.difficulty, hat_theta),
                reverse=True,
            )
            # Exposure control: select randomly among top-k candidates (Charter §4.2)
            k_pool = min(max(1, exposure_control_k), len(sorted_candidates))
            best_item = random.choice(sorted_candidates[:k_pool])
        else:
            best_item = random.choice(unadministered)

        unadministered.remove(best_item)

        # Simulate response from true_theta
        z_true = best_item.discrimination * (true_theta - best_item.difficulty)
        p_true = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z_true))))
        score = 1 if random.random() < p_true else 0

        # Update ability & SEM via EAP
        administered_tuples.append((best_item.discrimination, best_item.difficulty, score))
        hat_theta, sem = _eap_update(administered_tuples, grid_points)

        info_at_theta = _item_info_2pl(best_item.discrimination, best_item.difficulty, hat_theta)

        step = CATStep(
            step_number=step_num,
            item_id=best_item.item_id,
            item_key=best_item.item_key,
            response_score=score,
            ability_estimate=hat_theta,
            sem=sem,
            information=round(info_at_theta, 4),
        )
        trajectory.append(step)

        # Check stopping rule: SE threshold
        if sem <= target_se:
            stopped_reason = f"Target SE reached ({sem:.3f} <= {target_se:.3f})"
            break

    return CATSimulationResult(
        benchmark_id=benchmark_id,
        fit_id=fit_id,
        true_theta=true_theta,
        final_ability_estimate=hat_theta,
        final_sem=sem,
        num_items_administered=len(trajectory),
        total_bank_items=num_bank_items,
        selection_strategy=strategy,
        stopped_reason=stopped_reason,
        trajectory=trajectory,
    )

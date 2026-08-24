"""Item analysis service (Layer 4 backend logic).

Calculates item difficulty, point-biserial discrimination, and infit/outfit mean-square fit statistics.
Flags problematic items according to configurable policy thresholds.

Charter references:
  - §4.2: "Item-level stats: difficulty, discrimination, infit/outfit."
  - §4.2: "Flag problematic items with a defined threshold policy —
           document the thresholds, make them configurable."
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from aims.analysis.models import ItemAnalysisSummary, ItemFlagReason, ItemStats
from aims.db.models import Item, Response


class ItemFlagPolicy:
    """Configurable threshold policy for flagging problematic items.

    Charter §4.2: "document the thresholds, make them configurable,
    don't hardcode magic numbers with no justification."
    """

    def __init__(
        self,
        min_discrimination: float = 0.15,
        max_outfit_mnsq: float = 1.5,
        min_outfit_mnsq: float = 0.5,
        max_infit_mnsq: float = 1.5,
        min_infit_mnsq: float = 0.5,
        key_error_discrim_threshold: float = -0.1,
    ):
        self.min_discrimination = min_discrimination
        self.max_outfit_mnsq = max_outfit_mnsq
        self.min_outfit_mnsq = min_outfit_mnsq
        self.max_infit_mnsq = max_infit_mnsq
        self.min_infit_mnsq = min_infit_mnsq
        self.key_error_discrim_threshold = key_error_discrim_threshold


DEFAULT_FLAG_POLICY = ItemFlagPolicy()


def compute_item_analysis(
    benchmark_id: UUID,
    items: Sequence[Item],
    responses: Sequence[Response],
    policy: ItemFlagPolicy = DEFAULT_FLAG_POLICY,
) -> ItemAnalysisSummary:
    """Compute classical item analysis and fit statistics for a benchmark.

    Handles sparse response matrix (missing model-item pairs omitted).
    """
    if not items or not responses:
        return ItemAnalysisSummary(
            benchmark_id=benchmark_id,
            num_items=len(items),
            num_models_evaluated=0,
            mean_difficulty=0.0,
            mean_discrimination=0.0,
            flagged_item_count=0,
            items=[],
            warnings=["Insufficient data for item analysis (no items or responses)."],
        )

    item_map = {item.id: item for item in items}
    model_ids = {r.ai_model_id for r in responses}
    num_models = len(model_ids)

    # Organize responses: model_id -> item_id -> score
    model_item_scores: Dict[UUID, Dict[UUID, int]] = {}
    for r in responses:
        if r.item_id in item_map:
            model_item_scores.setdefault(r.ai_model_id, {})[r.item_id] = r.score

    # Compute total test score per model (for item-total correlation)
    model_totals: Dict[UUID, float] = {
        m_id: sum(scores.values()) for m_id, scores in model_item_scores.items()
    }

    # Compute item-level statistics
    item_stats_list: List[ItemStats] = []
    total_diff = 0.0
    total_disc = 0.0
    flagged_count = 0
    warnings: List[str] = []

    if num_models < 10:
        warnings.append(
            f"Sample size is low (N={num_models} models). Item discrimination and fit statistics may be unstable."
        )

    for item in items:
        # Gather scores for this item across models
        item_scores: List[Tuple[float, float]] = []  # (item_score, total_minus_item)
        for m_id, scores in model_item_scores.items():
            if item.id in scores:
                x_ij = float(scores[item.id])
                t_i_minus_j = model_totals[m_id] - x_ij
                item_scores.append((x_ij, t_i_minus_j))

        n_j = len(item_scores)
        if n_j == 0:
            stats = ItemStats(
                item_id=item.id,
                item_key=item.item_key,
                difficulty=0.0,
                discrimination=0.0,
                flags=[ItemFlagReason.ZERO_VARIANCE],
            )
            item_stats_list.append(stats)
            flagged_count += 1
            continue

        # 1. Difficulty (proportion correct / normalized score)
        raw_scores = [x for x, _ in item_scores]
        mean_score = sum(raw_scores) / n_j
        difficulty = mean_score / item.max_score if item.max_score > 0 else mean_score

        # 2. Variance & Discrimination (Point-Biserial Correlation)
        var_x = sum((x - mean_score) ** 2 for x in raw_scores) / n_j if n_j > 1 else 0.0

        flags: List[ItemFlagReason] = []
        if var_x < 1e-8:
            discrimination = 0.0
            flags.append(ItemFlagReason.ZERO_VARIANCE)
        else:
            std_x = math.sqrt(var_x)
            other_scores = [t for _, t in item_scores]
            mean_other = sum(other_scores) / n_j
            var_other = sum((t - mean_other) ** 2 for t in other_scores) / n_j if n_j > 1 else 0.0

            if var_other < 1e-8:
                discrimination = 0.0
            else:
                std_other = math.sqrt(var_other)
                cov = sum((raw_scores[k] - mean_score) * (other_scores[k] - mean_other) for k in range(n_j)) / n_j
                discrimination = cov / (std_x * std_other)
                # Clamp to [-1, 1] for floating point rounding
                discrimination = max(-1.0, min(1.0, discrimination))

            # Discrimination flags
            if discrimination < 0.0:
                flags.append(ItemFlagReason.NEGATIVE_DISCRIMINATION)
                if discrimination < policy.key_error_discrim_threshold and difficulty < 0.3:
                    flags.append(ItemFlagReason.POTENTIAL_KEY_ERROR)
            elif discrimination < policy.min_discrimination:
                flags.append(ItemFlagReason.LOW_DISCRIMINATION)

        # 3. Infit and Outfit MNSQ statistics
        # Expected score e_j and variance w_j under Rasch / CTT model
        e_j = mean_score
        w_j = e_j * (1.0 - (e_j / item.max_score if item.max_score > 0 else e_j))

        infit_mnsq: Optional[float] = None
        outfit_mnsq: Optional[float] = None

        if w_j > 1e-6 and n_j > 2:
            std_err = math.sqrt(w_j)
            squared_residuals = [(x - e_j) ** 2 for x in raw_scores]

            # Outfit MNSQ: average squared standardized residual
            outfit_mnsq = (sum(sq / w_j for sq in squared_residuals) / n_j)
            outfit_mnsq = round(outfit_mnsq, 3)

            # Infit MNSQ: weighted mean-square
            infit_mnsq = sum(squared_residuals) / (n_j * w_j)
            infit_mnsq = round(infit_mnsq, 3)

            if outfit_mnsq > policy.max_outfit_mnsq or outfit_mnsq < policy.min_outfit_mnsq:
                flags.append(ItemFlagReason.MISFITTING_OUTFIT)
            if infit_mnsq > policy.max_infit_mnsq or infit_mnsq < policy.min_infit_mnsq:
                flags.append(ItemFlagReason.MISFITTING_INSET)

        stats = ItemStats(
            item_id=item.id,
            item_key=item.item_key,
            difficulty=round(difficulty, 4),
            discrimination=round(discrimination, 4),
            infit_mnsq=infit_mnsq,
            outfit_mnsq=outfit_mnsq,
            flags=flags,
        )
        item_stats_list.append(stats)

        total_diff += difficulty
        total_disc += discrimination
        if flags:
            flagged_count += 1

    num_k = len(items)
    mean_diff = total_diff / num_k if num_k > 0 else 0.0
    mean_disc = total_disc / num_k if num_k > 0 else 0.0

    return ItemAnalysisSummary(
        benchmark_id=benchmark_id,
        num_items=num_k,
        num_models_evaluated=num_models,
        mean_difficulty=round(mean_diff, 4),
        mean_discrimination=round(mean_disc, 4),
        flagged_item_count=flagged_count,
        items=item_stats_list,
        warnings=warnings,
    )

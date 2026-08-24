"""Reliability analysis service (Layer 4 backend logic).

Computes Classical Test Theory Cronbach's Alpha, IRT-native marginal reliability,
and the Standard Error of Measurement (SEM) curve across ability theta levels.

Charter references:
  - §4.2: "Cronbach's alpha plus at least one IRT-based reliability measure."
  - §4.2: "Standard error of measurement (SE) at relevant ability levels —
           report it as a function, not a scalar."
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple
from uuid import UUID

from aims.analysis.items import compute_item_analysis
from aims.analysis.models import ReliabilityAnalysisResult, SEMPoint
from aims.db.models import Item, Response


def _normal_pdf(x: float) -> float:
    """Standard normal probability density function phi(x)."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def compute_reliability_analysis(
    benchmark_id: UUID,
    items: Sequence[Item],
    responses: Sequence[Response],
    *,
    num_grid_points: int = 41,
) -> ReliabilityAnalysisResult:
    """Compute classical & IRT reliability metrics for a benchmark dataset."""
    num_k = len(items)

    if num_k == 0 or not responses:
        return ReliabilityAnalysisResult(
            benchmark_id=benchmark_id,
            num_items=num_k,
            num_models_evaluated=0,
            cronbach_alpha=0.0,
            marginal_reliability=0.0,
            overall_sem=0.0,
            sem_curve=[],
            warnings=["Insufficient data for reliability analysis (no items or responses)."],
        )

    # First run item analysis to get difficulties (b_j) and discriminations (a_j)
    item_analysis = compute_item_analysis(benchmark_id, items, responses)
    model_ids = {r.ai_model_id for r in responses}
    num_models = len(model_ids)

    warnings: List[str] = list(item_analysis.warnings)

    # -----------------------------------------------------------------------
    # 1. Cronbach's Alpha calculation
    # -----------------------------------------------------------------------
    # Build model x item score matrix
    model_scores: Dict[UUID, Dict[UUID, float]] = {}
    for r in responses:
        model_scores.setdefault(r.ai_model_id, {})[r.item_id] = float(r.score)

    # Item variances sum
    item_variances: List[float] = []
    for item in items:
        scores_i = [
            scores[item.id] for scores in model_scores.values() if item.id in scores
        ]
        n_i = len(scores_i)
        if n_i > 1:
            m_i = sum(scores_i) / n_i
            var_i = sum((x - m_i) ** 2 for x in scores_i) / n_i
            item_variances.append(var_i)
        else:
            item_variances.append(0.0)

    sum_item_variances = sum(item_variances)

    # Total test score variance
    total_scores = [sum(scores.values()) for scores in model_scores.values()]
    n_tot = len(total_scores)
    if n_tot > 1:
        mean_tot = sum(total_scores) / n_tot
        var_total = sum((t - mean_tot) ** 2 for t in total_scores) / n_tot
    else:
        var_total = 0.0

    if num_k > 1 and var_total > 1e-8:
        cronbach_alpha = (num_k / (num_k - 1.0)) * (1.0 - (sum_item_variances / var_total))
        cronbach_alpha = max(0.0, min(1.0, cronbach_alpha))
    else:
        cronbach_alpha = 0.0

    overall_sem = math.sqrt(var_total * (1.0 - cronbach_alpha)) if var_total > 0 else 0.0

    # -----------------------------------------------------------------------
    # 2. IRT SEM Curve & Marginal Reliability
    # -----------------------------------------------------------------------
    # Theta grid from -4.0 to +4.0
    grid_min, grid_max = -4.0, 4.0
    step = (grid_max - grid_min) / (num_grid_points - 1)
    theta_grid = [grid_min + i * step for i in range(num_grid_points)]

    # Item parameters for information function
    item_params: List[Tuple[float, float]] = []  # (a_j, b_j)
    for stat in item_analysis.items:
        # Convert difficulty (p-value) to logit difficulty b_j = -ln(p / (1-p))
        p = max(0.01, min(0.99, stat.difficulty))
        b_j = -math.log(p / (1.0 - p))
        # Discrimination a_j from point-biserial (constrained >= 0.2 for stability)
        a_j = max(0.2, min(3.0, stat.discrimination * 2.0 if stat.discrimination > 0 else 0.5))
        item_params.append((a_j, b_j))

    sem_curve: List[SEMPoint] = []
    weighted_info_sum = 0.0
    total_weight = 0.0

    for theta in theta_grid:
        info_theta = 0.0
        for a_j, b_j in item_params:
            # 2PL probability P_j(theta) = 1 / (1 + exp(-a_j * (theta - b_j)))
            z = a_j * (theta - b_j)
            p_j = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
            info_j = (a_j ** 2) * p_j * (1.0 - p_j)
            info_theta += info_j

        sem_theta = 1.0 / math.sqrt(info_theta) if info_theta > 1e-6 else 9.99

        sem_curve.append(SEMPoint(
            theta=round(theta, 2),
            sem=round(sem_theta, 4),
            information=round(info_theta, 4),
        ))

        # Weight by standard normal density N(0, 1) for marginal integration
        w = _normal_pdf(theta) * step
        weighted_info_sum += info_theta * w
        total_weight += w

    # Marginal reliability: avg_info / (1 + avg_info)
    avg_info = weighted_info_sum / total_weight if total_weight > 0 else 0.0
    marginal_reliability = avg_info / (1.0 + avg_info) if avg_info > 0 else 0.0
    marginal_reliability = max(0.0, min(1.0, marginal_reliability))

    if cronbach_alpha < 0.70:
        warnings.append(
            f"Cronbach's alpha ({cronbach_alpha:.2f}) is below standard reliability threshold (0.70). "
            "Consider removing misfitting or low-discrimination items."
        )

    return ReliabilityAnalysisResult(
        benchmark_id=benchmark_id,
        num_items=num_k,
        num_models_evaluated=num_models,
        cronbach_alpha=round(cronbach_alpha, 4),
        marginal_reliability=round(marginal_reliability, 4),
        overall_sem=round(overall_sem, 4),
        sem_curve=sem_curve,
        warnings=warnings,
    )

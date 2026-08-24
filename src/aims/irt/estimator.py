"""IRT Estimation Engine (Layer 4 backend logic).

Implements 1PL (Rasch) and 2PL IRT model parameter estimation with MML / Joint ML,
strict convergence checking, sample size guardrails, and fit statistics calculation.

Charter references:
  - §4.2: "Support model families: 1PL (Rasch), 2PL, 3PL, GPCM, GRM."
  - §4.2: "Convergence checking: define and enforce concrete convergence criteria
           (log-likelihood delta, gradient norm, max iterations) — never return
           'success' on a run that hit max-iterations without convergence."
  - §4.2: "Minimum sample size / item count guardrails per model family."
  - §5.5: Dataset hashing & reproducibility metadata.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Sequence, Tuple
from uuid import UUID

from aims.db.models import ConvergenceStatus, Item, ModelFamily, Response
from aims.irt.models import IRTItemParameters, IRTPersonParameters


def compute_dataset_hash(responses: Sequence[Response]) -> str:
    """Generate deterministic SHA-256 hash of dataset responses (§5.5)."""
    sorted_tuples = sorted(
        (str(r.ai_model_id), str(r.item_id), r.score, r.response_version)
        for r in responses
    )
    raw = repr(sorted_tuples).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def check_sample_size_guardrails(
    model_family: ModelFamily,
    num_items: int,
    num_models: int,
) -> None:
    """Enforce minimum sample size and item count guardrails per model family.

    Charter §4.2: "reject or warn, don't silently fit garbage."
    """
    if model_family == ModelFamily.RASCH_1PL:
        if num_items < 3 or num_models < 3:
            raise ValueError(
                f"1PL/Rasch model requires at least 3 items and 3 models (got {num_items} items, {num_models} models)."
            )
    elif model_family == ModelFamily.TWO_PL:
        if num_items < 5 or num_models < 5:
            raise ValueError(
                f"2PL model requires at least 5 items and 5 models (got {num_items} items, {num_models} models)."
            )
    elif model_family in (ModelFamily.THREE_PL, ModelFamily.GPCM, ModelFamily.GRM):
        if num_items < 10 or num_models < 20:
            raise ValueError(
                f"{model_family.value} model requires at least 10 items and 20 models."
            )


def estimate_irt_parameters(
    model_family: ModelFamily,
    items: Sequence[Item],
    responses: Sequence[Response],
    *,
    max_iter: int = 200,
    tolerance: float = 1e-5,
) -> Tuple[List[IRTItemParameters], List[IRTPersonParameters], ConvergenceStatus, Dict[str, float]]:
    """Estimate IRT model parameters (1PL / 2PL).

    Returns: (item_parameters, person_parameters, convergence_status, fit_statistics)
    """
    item_map = {item.id: item for item in items}
    model_ids = sorted(list({r.ai_model_id for r in responses}))
    item_ids = sorted(list(item_map.keys()))

    num_models = len(model_ids)
    num_items = len(item_ids)

    # Check sample size guardrails
    check_sample_size_guardrails(model_family, num_items, num_models)

    # Build response matrix: (i, j) -> score (0 or 1)
    model_idx_map = {m_id: i for i, m_id in enumerate(model_ids)}
    item_idx_map = {it_id: j for j, it_id in enumerate(item_ids)}

    observed_responses: List[Tuple[int, int, float]] = []
    for r in responses:
        if r.item_id in item_idx_map and r.ai_model_id in model_idx_map:
            i = model_idx_map[r.ai_model_id]
            j = item_idx_map[r.item_id]
            # Clamp binary score
            score = 1.0 if r.score > 0 else 0.0
            observed_responses.append((i, j, score))

    num_observed = len(observed_responses)

    # Initial parameter estimates
    # b_j initial: -logit(p_j)
    b = [0.0] * num_items
    for j in range(num_items):
        item_scores = [y for _, j_idx, y in observed_responses if j_idx == j]
        if item_scores:
            p_j = max(0.05, min(0.95, sum(item_scores) / len(item_scores)))
            b[j] = -math.log(p_j / (1.0 - p_j))

    # theta_i initial: logit(p_i)
    theta = [0.0] * num_models
    for i in range(num_models):
        user_scores = [y for i_idx, _, y in observed_responses if i_idx == i]
        if user_scores:
            p_i = max(0.05, min(0.95, sum(user_scores) / len(user_scores)))
            theta[i] = math.log(p_i / (1.0 - p_i))

    # a_j initial (1.0 for 1PL, 1.0 for 2PL)
    a = [1.0] * num_items

    is_2pl = (model_family == ModelFamily.TWO_PL)
    prev_ll = -float("inf")
    converged = False
    final_iter = 0

    # Coordinate ascent / Joint ML iteration
    for iteration in range(1, max_iter + 1):
        final_iter = iteration

        # 1. Update theta_i (person abilities) with N(0, 1) prior penalty
        for i in range(num_models):
            obs_i = [(j_idx, y) for i_idx, j_idx, y in observed_responses if i_idx == i]
            if not obs_i:
                continue
            grad = -0.1 * theta[i]  # Prior N(0, 10)
            hess = 0.1
            for j_idx, y in obs_i:
                z = a[j_idx] * (theta[i] - b[j_idx])
                p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
                grad += a[j_idx] * (y - p)
                hess += (a[j_idx] ** 2) * p * (1.0 - p)

            if hess > 1e-6:
                theta[i] += grad / hess
                theta[i] = max(-4.0, min(4.0, theta[i]))

        # 2. Update b_j (item difficulties) and a_j (discriminations if 2PL)
        for j in range(num_items):
            obs_j = [(i_idx, y) for i_idx, j_idx, y in observed_responses if j_idx == j]
            if not obs_j:
                continue

            # Update b_j
            grad_b = -0.1 * b[j]
            hess_b = 0.1
            for i_idx, y in obs_j:
                z = a[j] * (theta[i_idx] - b[j])
                p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
                grad_b += -a[j] * (y - p)
                hess_b += (a[j] ** 2) * p * (1.0 - p)

            if hess_b > 1e-6:
                b[j] += grad_b / hess_b
                b[j] = max(-4.0, min(4.0, b[j]))

            # Update a_j (if 2PL)
            if is_2pl:
                grad_a = -0.1 * (a[j] - 1.0)
                hess_a = 0.1
                for i_idx, y in obs_j:
                    z = a[j] * (theta[i_idx] - b[j])
                    p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
                    diff_t_b = theta[i_idx] - b[j]
                    grad_a += diff_t_b * (y - p)
                    hess_a += (diff_t_b ** 2) * p * (1.0 - p)

                if hess_a > 1e-6:
                    a[j] += grad_a / hess_a
                    a[j] = max(0.1, min(4.0, a[j]))

        # 3. Compute Log-Likelihood
        curr_ll = 0.0
        for i_idx, j_idx, y in observed_responses:
            z = a[j_idx] * (theta[i_idx] - b[j_idx])
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
            p_clamped = max(1e-12, min(1.0 - 1e-12, p))
            curr_ll += y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped)

        # Check convergence
        delta_ll = abs(curr_ll - prev_ll)
        if iteration > 1 and delta_ll < tolerance:
            converged = True
            break
        prev_ll = curr_ll

    # Charter §4.2: "never return 'success' on a run that hit max-iterations without convergence"
    convergence_status = (
        ConvergenceStatus.CONVERGED if converged else ConvergenceStatus.NOT_CONVERGED
    )

    # Compute fit statistics: AIC, BIC, -2LL
    minus_2ll = -2.0 * prev_ll
    num_params = (num_items + num_models) if not is_2pl else (2 * num_items + num_models)
    aic = 2.0 * num_params + minus_2ll
    bic = num_params * math.log(max(1, num_observed)) + minus_2ll

    fit_statistics = {
        "log_likelihood": round(prev_ll, 4),
        "minus_2ll": round(minus_2ll, 4),
        "aic": round(aic, 4),
        "bic": round(bic, 4),
        "num_parameters": float(num_params),
        "num_iterations": float(final_iter),
    }

    # Calculate asymptotic standard errors from Fisher Information matrix
    item_params: List[IRTItemParameters] = []
    for j, it_id in enumerate(item_ids):
        item_obj = item_map[it_id]
        obs_j = [(i_idx, y) for i_idx, j_idx, y in observed_responses if j_idx == j]

        info_b = 0.1  # Prior contribution
        info_a = 0.1
        for i_idx, _ in obs_j:
            z = a[j] * (theta[i_idx] - b[j])
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
            w = p * (1.0 - p)
            info_b += (a[j] ** 2) * w
            info_a += ((theta[i_idx] - b[j]) ** 2) * w

        se_b = round(1.0 / math.sqrt(info_b), 4)
        se_a = round(1.0 / math.sqrt(info_a), 4) if is_2pl else None

        item_params.append(IRTItemParameters(
            item_id=it_id,
            item_key=item_obj.item_key,
            difficulty=round(b[j], 4),
            discrimination=round(a[j], 4) if is_2pl else 1.0,
            guessing=0.0,
            difficulty_se=se_b,
            discrimination_se=se_a,
        ))

    person_params: List[IRTPersonParameters] = []
    for i, m_id in enumerate(model_ids):
        obs_i = [(j_idx, y) for i_idx, j_idx, y in observed_responses if i_idx == i]
        info_theta = 0.1  # Prior contribution
        for j_idx, _ in obs_i:
            z = a[j_idx] * (theta[i] - b[j_idx])
            p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
            info_theta += (a[j_idx] ** 2) * p * (1.0 - p)

        se_theta = round(1.0 / math.sqrt(info_theta), 4)

        person_params.append(IRTPersonParameters(
            ai_model_id=m_id,
            ability=round(theta[i], 4),
            ability_se=se_theta,
        ))

    return item_params, person_params, convergence_status, fit_statistics

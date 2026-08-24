"""Tool execution service layer for AIMS MCP Server (Layer 3).

Exposes the 5 measurement tools:
  - aims_analyze_items
  - aims_reliability_analysis
  - aims_fit_irt_model
  - aims_check_job_status
  - aims_adaptive_testing (placeholder for Phase 4)

Charter references:
  - §3.1: Tool design contract ("when NOT to use" guidance, domain-namespaced names).
  - §3.2: Async job pattern for compute-heavy operations.
  - §3.7: "The agent is a translator, not an oracle. It explains what numbers mean;
           it does not make deployment decisions on user's behalf."
  - §3.8: Fixed error taxonomy mapping, no raw Python tracebacks.
  - §5.3: Structured summary inline, large artifacts referenced.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from aims.analysis.items import compute_item_analysis
from aims.analysis.reliability import compute_reliability_analysis
from aims.db.models import CallerContext, ModelFamily
from aims.db.repositories import (
    BenchmarkRepository,
    FitRepository,
    ItemRepository,
    ResponseRepository,
)
from aims.irt.jobs import JobManager
from aims.irt.models import FitRequest
from aims.server.errors import ErrorCode, format_tool_error, logger


class AIMSMCPToolService:
    """Service layer that executes MCP tool requests against Layer 4 & Layer 5."""

    def __init__(
        self,
        bench_repo: BenchmarkRepository,
        item_repo: ItemRepository,
        resp_repo: ResponseRepository,
        fit_repo: FitRepository,
        job_manager: Optional[JobManager] = None,
    ):
        self._bench_repo = bench_repo
        self._item_repo = item_repo
        self._resp_repo = resp_repo
        self._fit_repo = fit_repo
        self._job_manager = job_manager or JobManager()

    async def analyze_items(
        self,
        benchmark_id: UUID,
        caller: CallerContext,
    ) -> Dict[str, Any]:
        """Tool: aims_analyze_items

        Guidance: Use this to perform classical item-level quality analysis
        (difficulty, point-biserial discrimination, infit/outfit statistics,
        and flagging misfitting/problematic items). DO NOT use this to compute
        test-level reliability or fit latent-variable IRT models.
        """
        try:
            bench = await self._bench_repo.get_by_id(benchmark_id, caller)
            if not bench:
                return format_tool_error(
                    ErrorCode.UNKNOWN_BENCHMARK_ID,
                    f"Benchmark {benchmark_id} not found or access denied.",
                )

            items = await self._item_repo.list_by_benchmark(benchmark_id, caller)
            responses = await self._resp_repo.get_by_benchmark(benchmark_id, caller)

            if not items or not responses:
                return format_tool_error(
                    ErrorCode.INSUFFICIENT_DATA,
                    f"Benchmark {benchmark_id} has insufficient items ({len(items)}) or responses ({len(responses)}).",
                )

            summary = compute_item_analysis(benchmark_id, items, responses)
            return summary.model_dump(mode="json")

        except Exception as exc:
            logger.exception("Error in aims_analyze_items")
            return format_tool_error(ErrorCode.INTERNAL_ERROR, str(exc))

    async def reliability_analysis(
        self,
        benchmark_id: UUID,
        caller: CallerContext,
    ) -> Dict[str, Any]:
        """Tool: aims_reliability_analysis

        Guidance: Use this to evaluate test reliability metrics (Cronbach's alpha,
        IRT-native marginal reliability, and Standard Error of Measurement curves across
        ability levels). DO NOT use this to analyze individual item statistics or fit IRT models.
        """
        try:
            bench = await self._bench_repo.get_by_id(benchmark_id, caller)
            if not bench:
                return format_tool_error(
                    ErrorCode.UNKNOWN_BENCHMARK_ID,
                    f"Benchmark {benchmark_id} not found or access denied.",
                )

            items = await self._item_repo.list_by_benchmark(benchmark_id, caller)
            responses = await self._resp_repo.get_by_benchmark(benchmark_id, caller)

            if not items or not responses:
                return format_tool_error(
                    ErrorCode.INSUFFICIENT_DATA,
                    f"Benchmark {benchmark_id} has insufficient items ({len(items)}) or responses ({len(responses)}).",
                )

            result = compute_reliability_analysis(benchmark_id, items, responses)
            return result.model_dump(mode="json")

        except Exception as exc:
            logger.exception("Error in aims_reliability_analysis")
            return format_tool_error(ErrorCode.INTERNAL_ERROR, str(exc))

    async def fit_irt_model(
        self,
        benchmark_id: UUID,
        model_family_str: str,
        caller: CallerContext,
        hyperparameters: Dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Tool: aims_fit_irt_model

        Guidance: Use this to fit latent-trait IRT models (1PL/Rasch, 2PL) to estimate
        item difficulty, discrimination, and model ability. Returns a job handle immediately.
        DO NOT use this for quick item analysis or reliability checks.
        """
        try:
            # Validate model family enum
            try:
                model_family = ModelFamily(model_family_str)
            except ValueError:
                return format_tool_error(
                    ErrorCode.INVALID_MODEL_FAMILY_FOR_ITEM_TYPE,
                    f"Invalid model family '{model_family_str}'. Supported families: '1PL', '2PL'.",
                )

            bench = await self._bench_repo.get_by_id(benchmark_id, caller)
            if not bench:
                return format_tool_error(
                    ErrorCode.UNKNOWN_BENCHMARK_ID,
                    f"Benchmark {benchmark_id} not found or access denied.",
                )

            req = FitRequest(
                benchmark_id=benchmark_id,
                model_family=model_family,
                hyperparameters=hyperparameters or {},
                random_seed=random_seed,
            )

            job_handle = await self._job_manager.submit_fit_job(
                req,
                caller,
                self._bench_repo,
                self._item_repo,
                self._resp_repo,
                self._fit_repo,
            )
            return job_handle.model_dump(mode="json")

        except Exception as exc:
            logger.exception("Error in aims_fit_irt_model")
            return format_tool_error(ErrorCode.JOB_FAILED, str(exc))

    async def check_job_status(
        self,
        job_id: UUID,
    ) -> Dict[str, Any]:
        """Tool: aims_check_job_status

        Guidance: Check status of an asynchronous job handle returned by aims_fit_irt_model.
        """
        job = self._job_manager.get_job(job_id)
        if not job:
            return format_tool_error(
                ErrorCode.JOB_FAILED,
                f"Job ID {job_id} not found.",
            )
        return job.model_dump(mode="json")

    async def adaptive_testing(
        self,
        benchmark_id: UUID,
        caller: CallerContext,
        true_theta: float = 0.0,
        strategy_str: str = "max_information",
        target_se: float = 0.35,
        max_items: int = 15,
    ) -> Dict[str, Any]:
        """Tool: aims_adaptive_testing

        Guidance: Use this to run Computerized Adaptive Testing (CAT) simulations
        against stored item banks and fitted IRT models. DO NOT use this for static item analysis.
        """
        try:
            from aims.cat.engine import run_cat_simulation
            from aims.cat.models import ItemSelectionStrategy
            from aims.irt.artifacts import load_parameter_artifact

            bench = await self._bench_repo.get_by_id(benchmark_id, caller)
            if not bench:
                return format_tool_error(
                    ErrorCode.UNKNOWN_BENCHMARK_ID,
                    f"Benchmark {benchmark_id} not found or access denied.",
                )

            fits = await self._fit_repo.list_by_benchmark(benchmark_id, caller)
            if not fits:
                return format_tool_error(
                    ErrorCode.INSUFFICIENT_DATA,
                    f"No fitted IRT model found for benchmark {benchmark_id}. Run aims_fit_irt_model first.",
                )

            latest_fit = fits[0]
            item_params, _ = load_parameter_artifact(latest_fit.parameter_artifact_uri)

            try:
                strategy = ItemSelectionStrategy(strategy_str)
            except ValueError:
                strategy = ItemSelectionStrategy.MAX_INFORMATION

            cat_result = run_cat_simulation(
                benchmark_id=benchmark_id,
                item_bank=item_params,
                true_theta=true_theta,
                strategy=strategy,
                target_se=target_se,
                max_items=max_items,
                fit_id=latest_fit.id,
            )
            return cat_result.model_dump(mode="json")

        except Exception as exc:
            logger.exception("Error in aims_adaptive_testing")
            return format_tool_error(ErrorCode.INTERNAL_ERROR, str(exc))

    async def generate_report(
        self,
        benchmark_id: UUID,
        caller: CallerContext,
        format: str = "html",
    ) -> Dict[str, Any]:
        """Tool: aims_generate_report

        Guidance: Use this to assemble a reproducible measurement report (HTML, JSON, PDF)
        for a benchmark. DO NOT use this to compute raw statistics or fit models.
        """
        try:
            from uuid import uuid4
            from aims.db.models import Report
            from aims.db.tables import ReportRow
            from aims.reports.models import ReportData
            from aims.reports.renderer import render_report_artifact

            bench = await self._bench_repo.get_by_id(benchmark_id, caller)
            if not bench:
                return format_tool_error(
                    ErrorCode.UNKNOWN_BENCHMARK_ID,
                    f"Benchmark {benchmark_id} not found or access denied.",
                )

            fits = await self._fit_repo.list_by_benchmark(benchmark_id, caller)
            if not fits:
                return format_tool_error(
                    ErrorCode.INSUFFICIENT_DATA,
                    f"No fitted IRT model found for benchmark {benchmark_id}. Run aims_fit_irt_model first.",
                )

            latest_fit = fits[0]
            items = await self._item_repo.list_by_benchmark(benchmark_id, caller)
            responses = await self._resp_repo.get_by_benchmark(benchmark_id, caller)

            rel_analysis = compute_reliability_analysis(benchmark_id, items, responses)
            item_analysis = compute_item_analysis(benchmark_id, items, responses)

            report_id = uuid4()
            report_data = ReportData(
                report_id=report_id,
                fit_id=latest_fit.id,
                benchmark_id=benchmark_id,
                benchmark_name=bench.name,
                owner_id=caller.user_id,
                format=format.lower(),
                model_family=latest_fit.model_family,
                convergence_status=latest_fit.convergence_status,
                fit_statistics=latest_fit.fit_statistics,
                num_items=len(items),
                num_models_evaluated=len({r.ai_model_id for r in responses}),
                mean_difficulty=item_analysis.mean_difficulty,
                mean_discrimination=item_analysis.mean_discrimination,
                cronbach_alpha=rel_analysis.cronbach_alpha,
                marginal_reliability=rel_analysis.marginal_reliability,
                overall_sem=rel_analysis.overall_sem,
                flagged_item_count=item_analysis.flagged_item_count,
                warnings=rel_analysis.warnings,
                recommendations=[
                    f"Benchmark contains {item_analysis.flagged_item_count} flagged items needing review."
                    if item_analysis.flagged_item_count > 0 else "Benchmark items exhibit satisfactory psychometric properties."
                ],
                dataset_hash=latest_fit.dataset_hash,
                torch_measure_version=latest_fit.torch_measure_version,
                random_seed=latest_fit.random_seed,
                created_at=latest_fit.created_at,
            )

            artifact_uri = render_report_artifact(report_data)

            # Persist report domain record via repository
            report_domain = Report(
                id=report_id,
                fit_id=latest_fit.id,
                format=format.lower(),
                artifact_uri=artifact_uri,
                dataset_hash=latest_fit.dataset_hash,
                torch_measure_version=latest_fit.torch_measure_version,
                created_at=latest_fit.created_at,
                owner_id=caller.user_id,
            )
            from aims.db.sql_repositories import SqlReportRepository
            report_repo = SqlReportRepository(self._fit_repo._session)
            await report_repo.create(report_domain, caller)

            return {
                "report_id": str(report_id),
                "benchmark_id": str(benchmark_id),
                "fit_id": str(latest_fit.id),
                "format": format.lower(),
                "artifact_uri": artifact_uri,
                "reproducibility_metadata": {
                    "dataset_hash": latest_fit.dataset_hash,
                    "torch_measure_version": latest_fit.torch_measure_version,
                    "random_seed": latest_fit.random_seed,
                    "created_at": str(latest_fit.created_at),
                },
            }

        except Exception as exc:
            logger.exception("Error in aims_generate_report")
            return format_tool_error(ErrorCode.INTERNAL_ERROR, str(exc))

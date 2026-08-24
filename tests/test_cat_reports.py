"""Integration and correctness test suite for CAT Engine and Reproducible Reports (Phase 4).

Charter references:
  - §4.2: CAT simulation engine (max info, SE stopping rules) & reproducible reports.
  - §4.5: Single internal ReportData model rendering HTML/JSON.
  - §5.5: Reproducibility metadata embedding.
  - §4.4: Full 5-tool sequence chaining integration.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone

import pytest

from aims.cat.engine import run_cat_simulation
from aims.cat.models import ItemSelectionStrategy
from aims.db.models import AIModel, Benchmark, CallerContext, Item, Response
from aims.db.sql_repositories import (
    SqlAIModelRepository,
    SqlBenchmarkRepository,
    SqlFitRepository,
    SqlItemRepository,
    SqlResponseRepository,
)
from aims.irt.models import IRTItemParameters
from aims.reports.models import ReportData
from aims.reports.renderer import render_report_artifact
from aims.server.app import create_mcp_server_manifest
from aims.server.service import AIMSMCPToolService


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def temp_report_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class TestCATEngine:
    def test_cat_simulation_max_information_converges(self):
        bench_id = uuid.uuid4()
        # Item bank with varying difficulties
        item_bank = [
            IRTItemParameters(
                item_id=uuid.uuid4(),
                item_key=f"q_{i}",
                difficulty=b,
                discrimination=1.5,
            )
            for i, b in enumerate([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        ]

        true_theta = 0.5
        result = run_cat_simulation(
            benchmark_id=bench_id,
            item_bank=item_bank,
            true_theta=true_theta,
            strategy=ItemSelectionStrategy.MAX_INFORMATION,
            target_se=0.35,
            max_items=5,
            random_seed=42,
        )

        assert result.num_items_administered > 0
        assert result.num_items_administered <= 5
        # Ability estimate theta_hat should be reasonably close to true_theta=0.5
        assert abs(result.final_ability_estimate - true_theta) < 1.0
        assert len(result.trajectory) == result.num_items_administered

    def test_cat_exposure_control_varies_selected_items(self):
        bench_id = uuid.uuid4()
        item_bank = [
            IRTItemParameters(item_id=uuid.uuid4(), item_key=f"q_{i}", difficulty=0.0, discrimination=1.5)
            for i in range(10)
        ]

        res1 = run_cat_simulation(bench_id, item_bank, 0.0, exposure_control_k=3, random_seed=1)
        res2 = run_cat_simulation(bench_id, item_bank, 0.0, exposure_control_k=3, random_seed=99)

        # Different random seeds with top-k exposure control produce different item selection trajectories
        keys1 = [step.item_key for step in res1.trajectory]
        keys2 = [step.item_key for step in res2.trajectory]
        assert keys1 != keys2


class TestReportRenderer:
    def test_render_html_and_json_reports_embeds_reproducibility(self, temp_report_dir):
        report_data = ReportData(
            report_id=uuid.uuid4(),
            fit_id=uuid.uuid4(),
            benchmark_id=uuid.uuid4(),
            benchmark_name="EvalBenchmark",
            owner_id=uuid.uuid4(),
            format="html",
            model_family="2PL",
            convergence_status="converged",
            fit_statistics={"aic": 100.0, "bic": 110.0},
            num_items=10,
            num_models_evaluated=20,
            mean_difficulty=0.1,
            mean_discrimination=1.2,
            cronbach_alpha=0.85,
            marginal_reliability=0.82,
            overall_sem=0.25,
            flagged_item_count=0,
            warnings=[],
            recommendations=["All items operating within target specifications."],
            dataset_hash="dataset_hash_123456789",
            torch_measure_version="0.3.0",
            random_seed=42,
            created_at=_now(),
        )

        # Render HTML
        html_uri = render_report_artifact(report_data, base_dir=temp_report_dir)
        assert os.path.exists(html_uri)
        with open(html_uri, "r", encoding="utf-8") as f:
            html_content = f.read()

        assert "EvalBenchmark" in html_content
        assert "dataset_hash_123456789" in html_content
        assert "0.3.0" in html_content

        # Render JSON
        report_data.format = "json"
        json_uri = render_report_artifact(report_data, base_dir=temp_report_dir)
        assert os.path.exists(json_uri)
        with open(json_uri, "r", encoding="utf-8") as f:
            json_content = f.read()

        assert "dataset_hash_123456789" in json_content


class TestPhase4MCPTools:
    async def test_full_5_tool_chaining_sequence(self, session, caller_a):
        """Charter §4.4 & §4.3: Test end-to-end chaining of all tools:
        analyze_items -> reliability_analysis -> fit_irt_model -> adaptive_testing -> generate_report
        """
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)
        fit_repo = SqlFitRepository(session)

        service = AIMSMCPToolService(
            bench_repo=bench_repo,
            item_repo=item_repo,
            resp_repo=resp_repo,
            fit_repo=fit_repo,
        )

        bench = await bench_repo.create(
            Benchmark(id=uuid.uuid4(), name="FullSequenceBench", owner_id=caller_a.user_id, created_at=_now()),
            caller_a,
        )
        items = await item_repo.create_bulk(
            [Item(id=uuid.uuid4(), benchmark_id=bench.id, item_key=f"q{i}", created_at=_now()) for i in range(5)],
            caller_a,
        )
        models = [
            await model_repo.create(AIModel(id=uuid.uuid4(), name=f"M{i}", owner_id=caller_a.user_id, created_at=_now()), caller_a)
            for i in range(5)
        ]

        responses = [
            Response(
                id=uuid.uuid4(), benchmark_id=bench.id, ai_model_id=m.id, item_id=item.id,
                score=1 if (i + j) % 2 == 0 else 0, created_at=_now()
            )
            for i, m in enumerate(models) for j, item in enumerate(items)
        ]
        await resp_repo.create_bulk(responses, caller_a)

        # 1. aims_analyze_items
        analysis = await service.analyze_items(bench.id, caller_a)
        assert "items" in analysis

        # 2. aims_reliability_analysis
        reliability = await service.reliability_analysis(bench.id, caller_a)
        assert "cronbach_alpha" in reliability

        # 3. aims_fit_irt_model
        fit_handle = await service.fit_irt_model(bench.id, "1PL", caller_a)
        job_id = uuid.UUID(fit_handle["job_id"])

        for _ in range(50):
            status = await service.check_job_status(job_id)
            if status.get("status") == "completed":
                break
            await asyncio.sleep(0.05)

        # 4. aims_adaptive_testing
        cat_res = await service.adaptive_testing(bench.id, caller_a, true_theta=0.0)
        assert "final_ability_estimate" in cat_res
        assert len(cat_res["trajectory"]) > 0

        # 5. aims_generate_report
        report_res = await service.generate_report(bench.id, caller_a, format="html")
        assert "report_id" in report_res
        assert "artifact_uri" in report_res
        assert os.path.exists(report_res["artifact_uri"])

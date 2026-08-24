"""Contract and tool-chaining test suite for AIMS MCP Server (Phase 3).

Charter references:
  - §4.3: Tool contracts, namespacing, error taxonomy.
  - §4.4: Server instructions, host integration, tool chaining (analyze_items -> fit_irt_model -> status).
  - §3.8: Structured error responses.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from aims.db.models import AIModel, Benchmark, CallerContext, Item, Response
from aims.db.sql_repositories import (
    SqlAIModelRepository,
    SqlBenchmarkRepository,
    SqlFitRepository,
    SqlItemRepository,
    SqlResponseRepository,
)
from aims.server.app import SERVER_INSTRUCTIONS, create_mcp_server_manifest
from aims.server.errors import ErrorCode
from aims.server.service import AIMSMCPToolService


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestMCPServerManifest:
    def test_manifest_has_instructions_and_tools(self):
        manifest = create_mcp_server_manifest()
        assert manifest["name"] == "aims-measurement-server"
        assert SERVER_INSTRUCTIONS in manifest["instructions"]
        assert len(manifest["tools"]) == 6

        names = [t["name"] for t in manifest["tools"]]
        assert "aims_analyze_items" in names
        assert "aims_reliability_analysis" in names
        assert "aims_fit_irt_model" in names
        assert "aims_check_job_status" in names
        assert "aims_adaptive_testing" in names
        assert "aims_generate_report" in names

        # Verify "when NOT to use" guidance in descriptions (§4.3)
        for tool in manifest["tools"]:
            if tool["name"] != "aims_check_job_status":
                assert "DO NOT use" in tool["description"]


class TestMCPToolServiceContracts:
    @pytest.fixture
    async def service(self, session):
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)
        fit_repo = SqlFitRepository(session)

        return AIMSMCPToolService(
            bench_repo=bench_repo,
            item_repo=item_repo,
            resp_repo=resp_repo,
            fit_repo=fit_repo,
        )

    async def test_unknown_benchmark_returns_structured_error(self, service, caller_a):
        fake_id = uuid.uuid4()
        res = await service.analyze_items(fake_id, caller_a)
        assert res.get("is_error") is True
        assert res.get("error_code") == ErrorCode.UNKNOWN_BENCHMARK_ID.value

    async def test_invalid_model_family_returns_structured_error(self, service, session, caller_a):
        bench_repo = SqlBenchmarkRepository(session)
        bench = await bench_repo.create(
            Benchmark(id=uuid.uuid4(), name="B1", owner_id=caller_a.user_id, created_at=_now()),
            caller_a,
        )

        res = await service.fit_irt_model(bench.id, "INVALID_FAMILY", caller_a)
        assert res.get("is_error") is True
        assert res.get("error_code") == ErrorCode.INVALID_MODEL_FAMILY_FOR_ITEM_TYPE.value

    async def test_analyze_and_reliability_tools_work(self, service, session, caller_a):
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        bench = await bench_repo.create(
            Benchmark(id=uuid.uuid4(), name="B_MCP", owner_id=caller_a.user_id, created_at=_now()),
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

        responses = []
        for i, m in enumerate(models):
            for j, item in enumerate(items):
                responses.append(Response(
                    id=uuid.uuid4(), benchmark_id=bench.id, ai_model_id=m.id, item_id=item.id, score=1 if (i + j) % 2 == 0 else 0, created_at=_now()
                ))
        await resp_repo.create_bulk(responses, caller_a)

        # 1. aims_analyze_items
        analysis_res = await service.analyze_items(bench.id, caller_a)
        assert "items" in analysis_res
        assert len(analysis_res["items"]) == 5
        assert "mean_difficulty" in analysis_res

        # 2. aims_reliability_analysis
        rel_res = await service.reliability_analysis(bench.id, caller_a)
        assert "cronbach_alpha" in rel_res
        assert "marginal_reliability" in rel_res
        assert "sem_curve" in rel_res
        assert len(rel_res["sem_curve"]) == 41

    async def test_tool_chaining_workflow(self, service, session, caller_a):
        """Charter §4.4: Confirm behavior when host chains tools in sequence:
        analyze_items -> fit_irt_model -> check_job_status
        """
        bench_repo = SqlBenchmarkRepository(session)
        item_repo = SqlItemRepository(session)
        model_repo = SqlAIModelRepository(session)
        resp_repo = SqlResponseRepository(session)

        bench = await bench_repo.create(
            Benchmark(id=uuid.uuid4(), name="B_Chain", owner_id=caller_a.user_id, created_at=_now()),
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

        # Step 1: analyze_items
        analysis = await service.analyze_items(bench.id, caller_a)
        assert analysis.get("is_error") is not True

        # Step 2: fit_irt_model
        fit_handle = await service.fit_irt_model(bench.id, "1PL", caller_a)
        assert fit_handle.get("is_error") is not True
        job_id = uuid.UUID(fit_handle["job_id"])

        # Wait for background task to complete
        for _ in range(50):
            status = await service.check_job_status(job_id)
            if status.get("status") == "completed":
                break
            await asyncio.sleep(0.05)

        # Step 3: check_job_status
        final_status = await service.check_job_status(job_id)
        assert final_status["status"] == "completed"
        assert final_status["fit_result"] is not None
        assert final_status["fit_result"]["model_family"] == "1PL"

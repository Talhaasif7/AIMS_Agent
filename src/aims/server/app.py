"""AIMS Measurement App — MCP Server Application (Layer 3).

Charter references:
  - §4.3: Remote HTTP & stdio transports, namespaced tool definitions.
  - §4.4: Server-level instructions prompt establishing expectations, bounds,
           and recommended workflow order.
  - §3.7: "The agent is a translator, not an oracle... it does not make deployment decisions."
"""

from __future__ import annotations

import json
from uuid import UUID

from aims.db.models import CallerContext
from aims.server.errors import logger
from aims.server.service import AIMSMCPToolService

SERVER_INSTRUCTIONS = """You are connected to the AIMS Measurement Server powered by torch_measure.
This server provides psychometric and predictive measurement tools for evaluating AI benchmarks and systems.

RECOMMENDED WORKFLOW ORDER (§4.4):
1. `aims_analyze_items`: Inspect classical item difficulty, discrimination, and flag misfitting/problematic items.
2. `aims_reliability_analysis`: Check internal consistency (Cronbach's alpha) and Standard Error of Measurement (SEM) curves across ability levels.
3. `aims_fit_irt_model`: Fit latent-variable IRT models (1PL Rasch, 2PL) to estimate item parameters and model ability.
4. `aims_check_job_status`: Poll long-running async fit jobs returned by `aims_fit_irt_model`.

IMPORTANT BOUNDS (§3.7):
- Translate statistical numbers into actionable insights and recommendations.
- Always communicate uncertainty alongside parameter estimates.
- DO NOT make final deployment or safety sign-off decisions on behalf of the user; provide calibrated analysis to support human decision-making.
"""


def create_mcp_server_manifest() -> dict:
    """Return the MCP server manifest with tools and system instructions (§4.4)."""
    return {
        "name": "aims-measurement-server",
        "version": "0.1.0",
        "instructions": SERVER_INSTRUCTIONS,
        "tools": [
            {
                "name": "aims_analyze_items",
                "description": (
                    "Use this to perform classical item-level quality analysis "
                    "(difficulty, point-biserial discrimination, infit/outfit statistics, "
                    "and flagging misfitting/problematic items). DO NOT use this to compute "
                    "test-level reliability or fit latent-variable IRT models."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "benchmark_id": {"type": "string", "description": "UUID of the benchmark to analyze"},
                    },
                    "required": ["benchmark_id"],
                },
            },
            {
                "name": "aims_reliability_analysis",
                "description": (
                    "Use this to evaluate test reliability metrics (Cronbach's alpha, "
                    "IRT-native marginal reliability, and Standard Error of Measurement curves across "
                    "ability levels). DO NOT use this to analyze individual item statistics or fit IRT models."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "benchmark_id": {"type": "string", "description": "UUID of the benchmark to analyze"},
                    },
                    "required": ["benchmark_id"],
                },
            },
            {
                "name": "aims_fit_irt_model",
                "description": (
                    "Use this to fit latent-trait IRT models (1PL/Rasch, 2PL) to estimate "
                    "item difficulty, discrimination, and model ability. Returns a job handle immediately. "
                    "DO NOT use this for quick item analysis or reliability checks."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "benchmark_id": {"type": "string", "description": "UUID of the benchmark to fit"},
                        "model_family": {
                            "type": "string",
                            "enum": ["1PL", "2PL"],
                            "description": "IRT model family to fit",
                        },
                        "random_seed": {"type": "integer", "default": 42},
                    },
                    "required": ["benchmark_id", "model_family"],
                },
            },
            {
                "name": "aims_check_job_status",
                "description": "Check status of an asynchronous job handle returned by aims_fit_irt_model.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "UUID of the job handle"},
                    },
                    "required": ["job_id"],
                },
            },
            {
                "name": "aims_adaptive_testing",
                "description": (
                    "Use this to run Computerized Adaptive Testing (CAT) simulations "
                    "against stored item banks and fitted IRT models. DO NOT use this for static item analysis."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "benchmark_id": {"type": "string", "description": "UUID of the benchmark"},
                        "true_theta": {"type": "number", "default": 0.0, "description": "Simulated true ability theta*"},
                        "strategy": {"type": "string", "enum": ["max_information", "random_baseline"], "default": "max_information"},
                        "target_se": {"type": "number", "default": 0.35},
                        "max_items": {"type": "integer", "default": 15},
                    },
                    "required": ["benchmark_id"],
                },
            },
            {
                "name": "aims_generate_report",
                "description": (
                    "Use this to assemble a reproducible measurement report (HTML, JSON, PDF) "
                    "for a benchmark. DO NOT use this to compute raw statistics or fit models."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "benchmark_id": {"type": "string", "description": "UUID of the benchmark"},
                        "format": {"type": "string", "enum": ["html", "json", "pdf"], "default": "html"},
                    },
                    "required": ["benchmark_id"],
                },
            },
        ],
    }

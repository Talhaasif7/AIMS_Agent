"""Report Rendering Pipeline for AIMS Measurement Agent (Layer 6).

Renders HTML, JSON, and text/PDF formats from the single internal ReportData
representation (§4.5) with embedded reproducibility provenance (§5.5).

Security notes (per secure coding guidelines):
  - Jinja2 autoescaping enabled to prevent HTML injection.
  - Path boundary verification enforced before writing files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID

from jinja2 import Environment

from aims.reports.models import ReportData

DEFAULT_REPORT_DIR = os.path.abspath("artifacts/reports")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AIMS Measurement Report - {{ data.benchmark_name }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; color: #333; background: #f8f9fa; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #1a252f; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .meta-box { background: #eef2f7; padding: 15px; border-radius: 6px; margin-bottom: 20px; font-size: 0.9em; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f1f5f9; padding: 15px; border-radius: 6px; text-align: center; }
        .stat-val { font-size: 1.6em; font-weight: bold; color: #2c3e50; }
        .stat-lbl { font-size: 0.85em; color: #7f8c8d; }
        .warning { background: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; margin: 5px 0; }
        .recommendation { background: #d4edda; color: #155724; padding: 10px; border-radius: 4px; margin: 5px 0; }
        .footer { margin-top: 40px; font-size: 0.8em; color: #95a5a6; border-top: 1px solid #ddd; padding-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AIMS Measurement Report</h1>
        <div class="meta-box">
            <strong>Benchmark:</strong> {{ data.benchmark_name }} (ID: {{ data.benchmark_id }})<br>
            <strong>Fit ID:</strong> {{ data.fit_id }} | <strong>Model Family:</strong> {{ data.model_family }}<br>
            <strong>Convergence Status:</strong> {{ data.convergence_status }}
        </div>

        <h2>Measurement Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">{{ data.num_items }}</div>
                <div class="stat-lbl">Items Evaluated</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{{ data.num_models_evaluated }}</div>
                <div class="stat-lbl">AI Models Evaluated</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{{ "%.3f"|format(data.mean_difficulty) }}</div>
                <div class="stat-lbl">Mean Item Difficulty</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{{ "%.3f"|format(data.cronbach_alpha) }}</div>
                <div class="stat-lbl">Cronbach's Alpha</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{{ "%.3f"|format(data.marginal_reliability) }}</div>
                <div class="stat-lbl">Marginal Reliability</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{{ "%.3f"|format(data.overall_sem) }}</div>
                <div class="stat-lbl">Overall SEM</div>
            </div>
        </div>

        {% if data.warnings %}
        <h2>Warnings & Flags</h2>
        {% for w in data.warnings %}
        <div class="warning">⚠️ {{ w }}</div>
        {% endfor %}
        {% endif %}

        {% if data.recommendations %}
        <h2>Actionable Insights & Recommendations</h2>
        {% for r in data.recommendations %}
        <div class="recommendation">💡 {{ r }}</div>
        {% endfor %}
        {% endif %}

        <div class="footer">
            <strong>Reproducibility Provenance (§5.5):</strong><br>
            Dataset Hash: <code>{{ data.dataset_hash }}</code> |
            torch_measure: <code>{{ data.torch_measure_version }}</code> |
            Seed: <code>{{ data.random_seed }}</code> |
            Timestamp: {{ data.created_at }}
        </div>
    </div>
</body>
</html>
"""


def _get_sandbox_dir(base_dir: str | None = None) -> Path:
    target = Path(base_dir or DEFAULT_REPORT_DIR).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def render_report_artifact(
    report_data: ReportData,
    base_dir: str | None = None,
) -> str:
    """Render a report artifact (HTML, JSON, or text/PDF) from ReportData.

    Returns the written file URI.
    """
    sandbox = _get_sandbox_dir(base_dir)
    fmt = report_data.format.lower()
    file_name = f"report_{report_data.report_id.hex}.{fmt}"
    file_path = (sandbox / file_name).resolve()

    # Security check: path traversal prevention
    if not str(file_path).startswith(str(sandbox)):
        raise ValueError(f"Path traversal detected: {file_path}")

    if fmt == "json":
        content = report_data.model_dump_json(indent=2)
    elif fmt in ("html", "pdf"):
        env = Environment(autoescape=True)
        template = env.from_string(HTML_TEMPLATE)
        content = template.render(data=report_data)
    else:
        # Fallback to plain text representation
        content = f"AIMS Measurement Report ({report_data.benchmark_name})\n" + repr(report_data.model_dump())

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(file_path)

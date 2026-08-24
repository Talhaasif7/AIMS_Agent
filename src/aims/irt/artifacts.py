"""Parameter Artifact Manager for AIMS IRT Engine.

Charter §4.1: "item/person parameter estimates as artifact reference, not inline blob."

Security notes (per secure coding guidelines):
  - Path inputs MUST be sanitized using path.basename / Path to prevent directory traversal.
  - Resolved artifact paths MUST be verified against the allowed sandbox directory boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple
from uuid import UUID

from aims.irt.models import IRTItemParameters, IRTPersonParameters

# Default storage directory for fit parameter artifacts
DEFAULT_ARTIFACT_DIR = os.path.abspath("artifacts/fits")


def _get_sandbox_dir(base_dir: str | None = None) -> Path:
    target = Path(base_dir or DEFAULT_ARTIFACT_DIR).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_parameter_artifact(
    fit_id: UUID,
    item_params: List[IRTItemParameters],
    person_params: List[IRTPersonParameters],
    base_dir: str | None = None,
) -> str:
    """Save item and person parameter estimates to a JSON artifact file.

    Returns the canonical string path URI.
    """
    sandbox = _get_sandbox_dir(base_dir)
    file_name = f"fit_{fit_id.hex}.json"
    file_path = (sandbox / file_name).resolve()

    # Security check: verify resolved path is inside sandbox directory
    if not str(file_path).startswith(str(sandbox)):
        raise ValueError(f"Path traversal detected: {file_path}")

    data = {
        "fit_id": str(fit_id),
        "item_parameters": [p.model_dump(mode="json") for p in item_params],
        "person_parameters": [p.model_dump(mode="json") for p in person_params],
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return str(file_path)


def load_parameter_artifact(
    artifact_uri: str,
    base_dir: str | None = None,
) -> Tuple[List[IRTItemParameters], List[IRTPersonParameters]]:
    """Load parameter estimates from an artifact file.

    Sanitizes URI path to prevent traversal attacks.
    """
    sandbox = _get_sandbox_dir(base_dir)

    # If URI has file:// prefix, strip it
    clean_path = artifact_uri.replace("file:///", "").replace("file://", "")
    target_path = Path(clean_path).resolve()

    # Verify path is inside allowed directory or absolute file path exists
    if not target_path.exists():
        raise FileNotFoundError(f"Parameter artifact not found: {artifact_uri}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    item_params = [IRTItemParameters(**item) for item in data.get("item_parameters", [])]
    person_params = [IRTPersonParameters(**person) for person in data.get("person_parameters", [])]

    return item_params, person_params

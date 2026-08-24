"""Domain error taxonomy and MCP error content formatting (Layer 3).

Charter references:
  - §3.8: "Human-readable failure, not stack traces. Every tool has a defined
           taxonomy of failure modes that map to clear MCP tool-result error content."
  - §4.3: Domain error taxonomy:
           - NOT_CONVERGED
           - INSUFFICIENT_DATA
           - UNKNOWN_BENCHMARK_ID
           - INVALID_MODEL_FAMILY_FOR_ITEM_TYPE
           - JOB_FAILED
"""

from __future__ import annotations

import enum
import logging
import sys
from typing import Any, Dict


# Logging discipline (§4.3): stdio transport MUST NEVER write to stdout.
# Log to stderr only to avoid corrupting JSON-RPC streams.
logger = logging.getLogger("aims.mcp_server")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ErrorCode(str, enum.Enum):
    """Domain failure taxonomy (§4.3)."""
    UNKNOWN_BENCHMARK_ID = "UNKNOWN_BENCHMARK_ID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_CONVERGED = "NOT_CONVERGED"
    INVALID_MODEL_FAMILY_FOR_ITEM_TYPE = "INVALID_MODEL_FAMILY_FOR_ITEM_TYPE"
    JOB_FAILED = "JOB_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AIMSDomainError(Exception):
    """Base domain exception for AIMS tool failures."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_tool_error_dict(self) -> Dict[str, Any]:
        """Format as human-readable structured MCP tool-result error content (§3.8)."""
        return {
            "is_error": True,
            "error_code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


def format_tool_error(
    code: ErrorCode,
    message: str,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Helper to format structured error dict without raising an exception."""
    return {
        "is_error": True,
        "error_code": code.value,
        "message": message,
        "details": details or {},
    }

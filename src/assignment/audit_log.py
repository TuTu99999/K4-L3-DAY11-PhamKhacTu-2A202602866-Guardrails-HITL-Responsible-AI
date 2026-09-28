"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or user_id

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and its start time until the matching output arrives."""
        rid = request_id or f"req-{uuid.uuid4().hex[:12]}"
        self._open[self._key(user_id, request_id)] = {
            "request_id": rid,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_perf": time.perf_counter(),
        }
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish an interaction and append a forensics-friendly log row."""
        pending = self._open.pop(self._key(user_id, request_id), None)
        now = time.perf_counter()
        row = {
            "request_id": (
                pending["request_id"] if pending else request_id or f"req-{uuid.uuid4().hex[:12]}"
            ),
            "user_id": user_id,
            "input": pending["input"] if pending else None,
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "started_at": pending["started_at"] if pending else None,
            "completed_at": utc_now_iso(),
            "latency_ms": round((now - pending["started_perf"]) * 1000, 3)
            if pending
            else None,
        }
        self.logs.append(row)
        return row

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

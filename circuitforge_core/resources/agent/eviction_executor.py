from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

_DEFAULT_GRACE_S = 5.0


@dataclass(frozen=True)
class EvictionResult:
    success: bool
    method: str   # "sigterm", "sigkill", "already_gone", "not_found", "error"
    message: str


class EvictionExecutor:
    def __init__(self, grace_period_s: float = _DEFAULT_GRACE_S) -> None:
        self._default_grace = grace_period_s

    def evict_pid(
        self,
        pid: int,
        grace_period_s: float | None = None,
    ) -> EvictionResult:
        grace = grace_period_s if grace_period_s is not None else self._default_grace

        if not psutil.pid_exists(pid):
            return EvictionResult(
                success=False, method="not_found",
                message=f"PID {pid} not found"
            )

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return EvictionResult(
                success=True, method="already_gone",
                message=f"PID {pid} vanished before SIGTERM"
            )
        except PermissionError as exc:
            return EvictionResult(
                success=False, method="error",
                message=f"Permission denied terminating PID {pid}: {exc}"
            )

        # Wait for grace period
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                logger.info("PID %d exited cleanly after SIGTERM", pid)
                return EvictionResult(
                    success=True, method="sigterm",
                    message=f"PID {pid} exited after SIGTERM"
                )
            time.sleep(0.05)

        # Escalate to SIGKILL
        if psutil.pid_exists(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                logger.warning("PID %d required SIGKILL", pid)
                return EvictionResult(
                    success=True, method="sigkill",
                    message=f"PID {pid} killed with SIGKILL"
                )
            except ProcessLookupError:
                pass

        return EvictionResult(
            success=True, method="sigkill",
            message=f"PID {pid} is gone"
        )

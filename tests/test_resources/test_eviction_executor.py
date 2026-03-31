import signal
from unittest.mock import patch, call
import pytest
from circuitforge_core.resources.agent.eviction_executor import EvictionExecutor, EvictionResult


def test_evict_by_pid_sends_sigterm_then_sigkill():
    executor = EvictionExecutor(grace_period_s=0.01)
    # pid_exists always True → grace period expires → SIGKILL fires
    with patch("os.kill") as mock_kill, \
         patch("circuitforge_core.resources.agent.eviction_executor.psutil") as mock_psutil:
        mock_psutil.pid_exists.return_value = True
        result = executor.evict_pid(pid=1234, grace_period_s=0.01)

    assert result.success is True
    calls = mock_kill.call_args_list
    assert call(1234, signal.SIGTERM) in calls
    assert call(1234, signal.SIGKILL) in calls


def test_evict_pid_succeeds_on_sigterm_alone():
    executor = EvictionExecutor(grace_period_s=0.1)
    with patch("os.kill"), \
         patch("circuitforge_core.resources.agent.eviction_executor.psutil") as mock_psutil:
        mock_psutil.pid_exists.side_effect = [True, False]  # gone after SIGTERM
        result = executor.evict_pid(pid=5678, grace_period_s=0.01)
    assert result.success is True
    assert result.method == "sigterm"


def test_evict_pid_not_found_returns_failure():
    executor = EvictionExecutor()
    with patch("circuitforge_core.resources.agent.eviction_executor.psutil") as mock_psutil:
        mock_psutil.pid_exists.return_value = False
        result = executor.evict_pid(pid=9999)
    assert result.success is False
    assert "not found" in result.message.lower()


def test_eviction_result_is_immutable():
    result = EvictionResult(success=True, method="sigterm", message="ok")
    with pytest.raises((AttributeError, TypeError)):
        result.success = False  # type: ignore

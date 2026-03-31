from unittest.mock import patch
from circuitforge_core.resources.agent.gpu_monitor import GpuMonitor


SAMPLE_NVIDIA_SMI_OUTPUT = (
    "0, Quadro RTX 4000, 8192, 6843, 1349\n"
    "1, Quadro RTX 4000, 8192, 721, 7471\n"
)


def test_parse_returns_list_of_gpu_info():
    monitor = GpuMonitor()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = SAMPLE_NVIDIA_SMI_OUTPUT
        gpus = monitor.poll()
    assert len(gpus) == 2
    assert gpus[0].gpu_id == 0
    assert gpus[0].name == "Quadro RTX 4000"
    assert gpus[0].vram_total_mb == 8192
    assert gpus[0].vram_used_mb == 6843
    assert gpus[0].vram_free_mb == 1349


def test_parse_second_gpu():
    monitor = GpuMonitor()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = SAMPLE_NVIDIA_SMI_OUTPUT
        gpus = monitor.poll()
    assert gpus[1].gpu_id == 1
    assert gpus[1].vram_used_mb == 721
    assert gpus[1].vram_free_mb == 7471


def test_poll_returns_empty_list_when_nvidia_smi_unavailable():
    monitor = GpuMonitor()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        gpus = monitor.poll()
    assert gpus == []


def test_poll_returns_empty_list_on_nonzero_exit():
    monitor = GpuMonitor()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        gpus = monitor.poll()
    assert gpus == []


def test_poll_skips_malformed_lines():
    monitor = GpuMonitor()
    malformed = "0, RTX 4000, 8192, not_a_number, 1024\n1, RTX 4000, 8192, 512, 7680\n"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = malformed
        gpus = monitor.poll()
    assert len(gpus) == 1
    assert gpus[0].gpu_id == 1

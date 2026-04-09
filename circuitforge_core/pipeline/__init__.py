# circuitforge_core/pipeline — FPGA→ASIC crystallization engine
#
# Public API: call pipeline.run() from product code instead of llm.router directly.
# The module transparently checks for crystallized workflows first, falls back
# to LLM when none match, and records each run for future crystallization.
from __future__ import annotations

from typing import Any, Callable

from .crystallizer import CrystallizerConfig, crystallize, evaluate_new_run, should_crystallize
from .executor import ExecutionResult, Executor, StepResult
from .models import CrystallizedWorkflow, PipelineRun, Step, hash_input
from .multimodal import MultimodalConfig, MultimodalPipeline, PageResult
from .recorder import Recorder
from .registry import Registry
from .staging import StagingDB

__all__ = [
    # models
    "PipelineRun",
    "CrystallizedWorkflow",
    "Step",
    "hash_input",
    # recorder
    "Recorder",
    # crystallizer
    "CrystallizerConfig",
    "crystallize",
    "evaluate_new_run",
    "should_crystallize",
    # registry
    "Registry",
    # executor
    "Executor",
    "ExecutionResult",
    "StepResult",
    # multimodal
    "MultimodalPipeline",
    "MultimodalConfig",
    "PageResult",
    # legacy stub
    "StagingDB",
]

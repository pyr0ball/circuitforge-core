"""Tests for pipeline models and hash_input utility."""
import pytest
from circuitforge_core.pipeline.models import (
    CrystallizedWorkflow,
    PipelineRun,
    Step,
    hash_input,
)


class TestHashInput:
    def test_stable_across_calls(self):
        feat = {"agency": "FTB", "menu_depth": 2}
        assert hash_input(feat) == hash_input(feat)

    def test_key_order_irrelevant(self):
        a = hash_input({"b": 2, "a": 1})
        b = hash_input({"a": 1, "b": 2})
        assert a == b

    def test_different_values_differ(self):
        assert hash_input({"a": 1}) != hash_input({"a": 2})

    def test_returns_hex_string(self):
        h = hash_input({"x": "y"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex


class TestStep:
    def test_roundtrip(self):
        s = Step(action="dtmf", params={"digits": "1"}, description="Press 1")
        assert Step.from_dict(s.to_dict()) == s

    def test_description_optional(self):
        s = Step.from_dict({"action": "dtmf", "params": {}})
        assert s.description == ""


class TestPipelineRun:
    def _run(self, **kwargs) -> PipelineRun:
        defaults = dict(
            run_id="r1",
            product="osprey",
            task_type="ivr_navigate",
            input_hash="abc123",
            steps=[Step("dtmf", {"digits": "1"})],
            approved=True,
            review_duration_ms=8000,
            output_modified=False,
        )
        defaults.update(kwargs)
        return PipelineRun(**defaults)

    def test_roundtrip(self):
        run = self._run()
        assert PipelineRun.from_dict(run.to_dict()).run_id == "r1"

    def test_output_modified_false_default(self):
        d = self._run().to_dict()
        d.pop("output_modified", None)
        run = PipelineRun.from_dict(d)
        assert run.output_modified is False

    def test_timestamp_auto_set(self):
        run = self._run()
        assert run.timestamp  # non-empty


class TestCrystallizedWorkflow:
    def _wf(self) -> CrystallizedWorkflow:
        return CrystallizedWorkflow(
            workflow_id="osprey:ivr_navigate:abc123abc123",
            product="osprey",
            task_type="ivr_navigate",
            input_hash="abc123",
            steps=[Step("dtmf", {"digits": "1"})],
            crystallized_at="2026-04-08T00:00:00+00:00",
            run_ids=["r1", "r2", "r3"],
            approval_count=3,
            avg_review_duration_ms=9000,
            all_output_unmodified=True,
        )

    def test_roundtrip(self):
        wf = self._wf()
        restored = CrystallizedWorkflow.from_dict(wf.to_dict())
        assert restored.workflow_id == wf.workflow_id
        assert restored.avg_review_duration_ms == 9000
        assert restored.all_output_unmodified is True

    def test_active_default_true(self):
        d = self._wf().to_dict()
        d.pop("active", None)
        wf = CrystallizedWorkflow.from_dict(d)
        assert wf.active is True

    def test_version_default_one(self):
        d = self._wf().to_dict()
        d.pop("version", None)
        wf = CrystallizedWorkflow.from_dict(d)
        assert wf.version == 1

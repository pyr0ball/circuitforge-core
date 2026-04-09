"""Tests for pipeline.Recorder."""
import pytest
from circuitforge_core.pipeline.models import PipelineRun, Step
from circuitforge_core.pipeline.recorder import Recorder


def _run(run_id="r1", approved=True, input_hash="abc", review_ms=8000,
         modified=False, ts="2026-04-08T01:00:00+00:00") -> PipelineRun:
    return PipelineRun(
        run_id=run_id,
        product="osprey",
        task_type="ivr_navigate",
        input_hash=input_hash,
        steps=[Step("dtmf", {"digits": "1"})],
        approved=approved,
        review_duration_ms=review_ms,
        output_modified=modified,
        timestamp=ts,
    )


class TestRecorder:
    def test_record_creates_file(self, tmp_path):
        rec = Recorder(root=tmp_path)
        path = rec.record(_run())
        assert path.exists()

    def test_load_runs_empty_when_no_directory(self, tmp_path):
        rec = Recorder(root=tmp_path)
        assert rec.load_runs("osprey", "ivr_navigate") == []

    def test_load_runs_returns_recorded(self, tmp_path):
        rec = Recorder(root=tmp_path)
        rec.record(_run("r1"))
        rec.record(_run("r2"))
        runs = rec.load_runs("osprey", "ivr_navigate")
        assert len(runs) == 2

    def test_load_runs_newest_first(self, tmp_path):
        rec = Recorder(root=tmp_path)
        rec.record(_run("r_old", ts="2026-01-01T00:00:00+00:00"))
        rec.record(_run("r_new", ts="2026-04-08T00:00:00+00:00"))
        runs = rec.load_runs("osprey", "ivr_navigate")
        assert runs[0].run_id == "r_new"

    def test_load_approved_filters(self, tmp_path):
        rec = Recorder(root=tmp_path)
        rec.record(_run("r1", approved=True))
        rec.record(_run("r2", approved=False))
        approved = rec.load_approved("osprey", "ivr_navigate", "abc")
        assert all(r.approved for r in approved)
        assert len(approved) == 1

    def test_load_approved_filters_by_hash(self, tmp_path):
        rec = Recorder(root=tmp_path)
        rec.record(_run("r1", input_hash="hash_a"))
        rec.record(_run("r2", input_hash="hash_b"))
        result = rec.load_approved("osprey", "ivr_navigate", "hash_a")
        assert len(result) == 1
        assert result[0].run_id == "r1"

    def test_record_is_append_only(self, tmp_path):
        rec = Recorder(root=tmp_path)
        for i in range(5):
            rec.record(_run(f"r{i}"))
        assert len(rec.load_runs("osprey", "ivr_navigate")) == 5

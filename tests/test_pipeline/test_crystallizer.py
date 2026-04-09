"""Tests for pipeline.crystallizer — the core promotion logic."""
import warnings
import pytest
from circuitforge_core.pipeline.crystallizer import (
    CrystallizerConfig,
    crystallize,
    evaluate_new_run,
    should_crystallize,
)
from circuitforge_core.pipeline.models import PipelineRun, Step
from circuitforge_core.pipeline.recorder import Recorder


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _run(run_id, approved=True, review_ms=8000, modified=False,
         steps=None, input_hash="fixedhash",
         ts="2026-04-08T00:00:00+00:00") -> PipelineRun:
    return PipelineRun(
        run_id=run_id,
        product="osprey",
        task_type="ivr_navigate",
        input_hash=input_hash,
        steps=steps or [Step("dtmf", {"digits": "1"})],
        approved=approved,
        review_duration_ms=review_ms,
        output_modified=modified,
        timestamp=ts,
    )


_CFG = CrystallizerConfig(threshold=3, min_review_ms=5_000)


# ── should_crystallize ────────────────────────────────────────────────────────

class TestShouldCrystallize:
    def test_returns_false_below_threshold(self):
        runs = [_run(f"r{i}") for i in range(2)]
        assert should_crystallize(runs, _CFG) is False

    def test_returns_true_at_threshold(self):
        runs = [_run(f"r{i}") for i in range(3)]
        assert should_crystallize(runs, _CFG) is True

    def test_returns_true_above_threshold(self):
        runs = [_run(f"r{i}") for i in range(10)]
        assert should_crystallize(runs, _CFG) is True

    def test_unapproved_runs_not_counted(self):
        approved = [_run(f"r{i}") for i in range(2)]
        unapproved = [_run(f"u{i}", approved=False) for i in range(10)]
        assert should_crystallize(approved + unapproved, _CFG) is False

    def test_threshold_one(self):
        cfg = CrystallizerConfig(threshold=1)
        assert should_crystallize([_run("r1")], cfg) is True


# ── crystallize ───────────────────────────────────────────────────────────────

class TestCrystallize:
    def _approved_runs(self, n=3, review_ms=8000):
        return [_run(f"r{i}", review_ms=review_ms) for i in range(n)]

    def test_produces_workflow(self):
        wf = crystallize(self._approved_runs(), _CFG)
        assert wf.product == "osprey"
        assert wf.task_type == "ivr_navigate"
        assert wf.approval_count == 3

    def test_workflow_id_format(self):
        wf = crystallize(self._approved_runs(), _CFG)
        assert wf.workflow_id.startswith("osprey:ivr_navigate:")

    def test_avg_review_duration_computed(self):
        runs = [_run("r0", review_ms=6000), _run("r1", review_ms=10000),
                _run("r2", review_ms=8000)]
        wf = crystallize(runs, _CFG)
        assert wf.avg_review_duration_ms == 8000

    def test_all_output_unmodified_true(self):
        runs = self._approved_runs()
        wf = crystallize(runs, _CFG)
        assert wf.all_output_unmodified is True

    def test_all_output_unmodified_false_when_any_modified(self):
        runs = [_run("r0"), _run("r1"), _run("r2", modified=True)]
        wf = crystallize(runs, _CFG)
        assert wf.all_output_unmodified is False

    def test_raises_below_threshold(self):
        with pytest.raises(ValueError, match="Need 3"):
            crystallize([_run("r0"), _run("r1")], _CFG)

    def test_raises_on_mixed_products(self):
        r1 = _run("r1")
        r2 = PipelineRun(
            run_id="r2", product="falcon", task_type="ivr_navigate",
            input_hash="fixedhash", steps=r1.steps, approved=True,
            review_duration_ms=8000, output_modified=False,
        )
        with pytest.raises(ValueError, match="product"):
            crystallize([r1, r2, r1], _CFG)

    def test_raises_on_mixed_hashes(self):
        runs = [_run("r0", input_hash="hash_a"),
                _run("r1", input_hash="hash_b"),
                _run("r2", input_hash="hash_a")]
        with pytest.raises(ValueError, match="input_hash"):
            crystallize(runs, _CFG)

    def test_rubber_stamp_warning(self):
        runs = [_run(f"r{i}", review_ms=100) for i in range(3)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            crystallize(runs, _CFG)
        assert any("rubber-stamp" in str(w.message) for w in caught)

    def test_no_warning_when_min_review_ms_zero(self):
        cfg = CrystallizerConfig(threshold=3, min_review_ms=0)
        runs = [_run(f"r{i}", review_ms=1) for i in range(3)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            crystallize(runs, cfg)
        assert not any("rubber-stamp" in str(w.message) for w in caught)

    def test_version_increments(self):
        wf = crystallize(self._approved_runs(), _CFG, existing_version=2)
        assert wf.version == 3

    def test_strategy_most_recent_uses_latest(self):
        steps_old = [Step("dtmf", {"digits": "9"})]
        steps_new = [Step("dtmf", {"digits": "1"})]
        runs = [
            _run("r0", steps=steps_old, ts="2026-01-01T00:00:00+00:00"),
            _run("r1", steps=steps_old, ts="2026-01-02T00:00:00+00:00"),
            _run("r2", steps=steps_new, ts="2026-04-08T00:00:00+00:00"),
        ]
        cfg = CrystallizerConfig(threshold=3, strategy="most_recent")
        wf = crystallize(runs, cfg)
        assert wf.steps[0].params["digits"] == "1"

    def test_strategy_majority_picks_common_action(self):
        steps_a = [Step("dtmf", {"digits": "1"})]
        steps_b = [Step("press_key", {"key": "2"})]
        runs = [
            _run("r0", steps=steps_a),
            _run("r1", steps=steps_a),
            _run("r2", steps=steps_b),
        ]
        cfg = CrystallizerConfig(threshold=3, strategy="majority")
        wf = crystallize(runs, cfg)
        assert wf.steps[0].action == "dtmf"

    def test_strategy_majority_falls_back_on_length_mismatch(self):
        runs = [
            _run("r0", steps=[Step("dtmf", {"digits": "1"})]),
            _run("r1", steps=[Step("dtmf", {"digits": "1"}),
                               Step("dtmf", {"digits": "2"})]),
            _run("r2", steps=[Step("dtmf", {"digits": "1"})],
                 ts="2026-04-08T00:00:00+00:00"),
        ]
        cfg = CrystallizerConfig(threshold=3, strategy="majority")
        # Should not raise — falls back to most_recent
        wf = crystallize(runs, cfg)
        assert wf.steps is not None


# ── evaluate_new_run ──────────────────────────────────────────────────────────

class TestEvaluateNewRun:
    def test_returns_none_before_threshold(self, tmp_path):
        rec = Recorder(root=tmp_path)
        cfg = CrystallizerConfig(threshold=3, min_review_ms=0)
        result = evaluate_new_run(_run("r1"), rec, cfg)
        assert result is None

    def test_returns_workflow_at_threshold(self, tmp_path):
        rec = Recorder(root=tmp_path)
        cfg = CrystallizerConfig(threshold=3, min_review_ms=0)
        for i in range(2):
            evaluate_new_run(_run(f"r{i}"), rec, cfg)
        wf = evaluate_new_run(_run("r2"), rec, cfg)
        assert wf is not None
        assert wf.approval_count == 3

    def test_unapproved_run_does_not_trigger(self, tmp_path):
        rec = Recorder(root=tmp_path)
        cfg = CrystallizerConfig(threshold=1, min_review_ms=0)
        result = evaluate_new_run(_run("r1", approved=False), rec, cfg)
        assert result is None

    def test_run_is_recorded_even_if_not_approved(self, tmp_path):
        rec = Recorder(root=tmp_path)
        cfg = CrystallizerConfig(threshold=3, min_review_ms=0)
        evaluate_new_run(_run("r1", approved=False), rec, cfg)
        assert len(rec.load_runs("osprey", "ivr_navigate")) == 1

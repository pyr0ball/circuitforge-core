"""Tests for pipeline.Registry — workflow lookup."""
import pytest
from circuitforge_core.pipeline.models import CrystallizedWorkflow, Step
from circuitforge_core.pipeline.registry import Registry


def _wf(input_hash="abc", active=True, wf_id=None) -> CrystallizedWorkflow:
    wid = wf_id or f"osprey:ivr_navigate:{input_hash[:12]}"
    return CrystallizedWorkflow(
        workflow_id=wid,
        product="osprey",
        task_type="ivr_navigate",
        input_hash=input_hash,
        steps=[Step("dtmf", {"digits": "1"})],
        crystallized_at="2026-04-08T00:00:00+00:00",
        run_ids=["r1", "r2", "r3"],
        approval_count=3,
        avg_review_duration_ms=9000,
        all_output_unmodified=True,
        active=active,
    )


class TestRegistry:
    def test_register_creates_file(self, tmp_path):
        reg = Registry(root=tmp_path)
        path = reg.register(_wf())
        assert path.exists()

    def test_load_all_empty_when_no_directory(self, tmp_path):
        reg = Registry(root=tmp_path)
        assert reg.load_all("osprey", "ivr_navigate") == []

    def test_load_all_returns_registered(self, tmp_path):
        reg = Registry(root=tmp_path)
        reg.register(_wf("hash_a", wf_id="osprey:ivr_navigate:hash_a"))
        reg.register(_wf("hash_b", wf_id="osprey:ivr_navigate:hash_b"))
        assert len(reg.load_all("osprey", "ivr_navigate")) == 2

    def test_match_exact_hit(self, tmp_path):
        reg = Registry(root=tmp_path)
        reg.register(_wf("abc123"))
        wf = reg.match("osprey", "ivr_navigate", "abc123")
        assert wf is not None
        assert wf.input_hash == "abc123"

    def test_match_returns_none_on_miss(self, tmp_path):
        reg = Registry(root=tmp_path)
        reg.register(_wf("abc123"))
        assert reg.match("osprey", "ivr_navigate", "different") is None

    def test_match_ignores_inactive(self, tmp_path):
        reg = Registry(root=tmp_path)
        reg.register(_wf("abc123", active=False))
        assert reg.match("osprey", "ivr_navigate", "abc123") is None

    def test_deactivate_sets_active_false(self, tmp_path):
        reg = Registry(root=tmp_path)
        wf = _wf("abc123")
        reg.register(wf)
        reg.deactivate(wf.workflow_id, "osprey", "ivr_navigate")
        assert reg.match("osprey", "ivr_navigate", "abc123") is None

    def test_deactivate_returns_false_when_not_found(self, tmp_path):
        reg = Registry(root=tmp_path)
        assert reg.deactivate("nonexistent", "osprey", "ivr_navigate") is False

    def test_find_falls_through_to_fuzzy(self, tmp_path):
        reg = Registry(root=tmp_path,
                       similarity_fn=lambda a, b: 1.0 if a == b else 0.5,
                       fuzzy_threshold=0.4)
        reg.register(_wf("hash_stored"))
        # No exact match for "hash_query" but similarity returns 0.5 >= 0.4
        wf = reg.find("osprey", "ivr_navigate", "hash_query")
        assert wf is not None

    def test_fuzzy_match_raises_without_fn(self, tmp_path):
        reg = Registry(root=tmp_path)
        with pytest.raises(RuntimeError, match="similarity_fn"):
            reg.fuzzy_match("osprey", "ivr_navigate", "any")

    def test_fuzzy_match_below_threshold_returns_none(self, tmp_path):
        reg = Registry(root=tmp_path,
                       similarity_fn=lambda a, b: 0.1,
                       fuzzy_threshold=0.8)
        reg.register(_wf("hash_stored"))
        assert reg.fuzzy_match("osprey", "ivr_navigate", "hash_query") is None

    def test_find_exact_takes_priority(self, tmp_path):
        reg = Registry(root=tmp_path,
                       similarity_fn=lambda a, b: 0.9,
                       fuzzy_threshold=0.8)
        reg.register(_wf("exact_hash"))
        wf = reg.find("osprey", "ivr_navigate", "exact_hash")
        # Should be the exact-match workflow
        assert wf.input_hash == "exact_hash"

    def test_workflow_id_colon_safe_in_filename(self, tmp_path):
        """Colons in workflow_id must not break file creation on any OS."""
        reg = Registry(root=tmp_path)
        wf = _wf("abc", wf_id="osprey:ivr_navigate:abc123abc123")
        path = reg.register(wf)
        assert path.exists()
        assert ":" not in path.name

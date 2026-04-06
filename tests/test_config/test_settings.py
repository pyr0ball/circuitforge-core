import os
import pytest
from circuitforge_core.config import require_env, load_env


def test_require_env_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "hello")
    assert require_env("TEST_KEY") == "hello"


def test_require_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="TEST_KEY"):
        require_env("TEST_KEY")


def test_load_env_sets_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nBAZ=qux\n")
    monkeypatch.delenv("FOO", raising=False)
    load_env(env_file)
    assert os.environ.get("FOO") == "bar"
    assert os.environ.get("BAZ") == "qux"


def test_load_env_skips_missing_file(tmp_path):
    load_env(tmp_path / "nonexistent.env")  # must not raise

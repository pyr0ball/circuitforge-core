import pytest
from circuitforge_core.wizard import BaseWizard
from circuitforge_core.pipeline import StagingDB


def test_wizard_raises_not_implemented():
    wizard = BaseWizard()
    with pytest.raises(NotImplementedError):
        wizard.run()


def test_pipeline_raises_not_implemented():
    staging = StagingDB()
    with pytest.raises(NotImplementedError):
        staging.enqueue("job", {})

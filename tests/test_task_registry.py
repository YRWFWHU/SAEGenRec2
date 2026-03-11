"""Unit tests for SFT task registry."""

import pytest


def test_four_builtin_tasks_registered():
    """Test that all 4 builtin tasks are registered."""
    from SAEGenRec.datasets.task_registry import list_sft_tasks

    tasks = list_sft_tasks()
    assert "sid_seq" in tasks
    assert "item_feat" in tasks
    assert "fusion" in tasks
    assert "sid_to_title" in tasks


def test_custom_task_registration_via_decorator():
    """Test that custom tasks can be registered via decorator."""
    from SAEGenRec.datasets.task_registry import _SFT_TASK_REGISTRY, register_sft_task, SFTTask

    @register_sft_task("test_custom_task_xyz")
    class CustomTask(SFTTask):
        name = "test_custom_task_xyz"
        default_template = "templates/test.txt"
        required_inputs = ["csv"]
        required_placeholders = ["history"]

        def build_examples(self, csv_data, index_json, item_json, template, **kwargs):
            return []

    assert "test_custom_task_xyz" in _SFT_TASK_REGISTRY
    # Cleanup
    del _SFT_TASK_REGISTRY["test_custom_task_xyz"]


def test_required_placeholders_validation():
    """Test that missing required placeholders raise an error."""
    from SAEGenRec.datasets.template_utils import validate_placeholders

    template = "The user history: {history}. Response:"
    # Should pass with matching placeholder
    validate_placeholders(template, ["history"])

    # Should fail with missing placeholder
    with pytest.raises(ValueError, match="missing required placeholders"):
        validate_placeholders(template, ["history", "target"])


def test_unknown_task_raises_error():
    """Test that unknown task name raises ValueError."""
    from SAEGenRec.datasets.task_registry import get_sft_task

    with pytest.raises(ValueError, match="Unknown SFT task"):
        get_sft_task("nonexistent_task_xyz")

"""Unit tests for SID method registry."""

import pytest

# Trigger @register_sid_method decorators by importing method modules
import SAEGenRec.sid_builder.gated_sae  # noqa: F401
import SAEGenRec.sid_builder.rqkmeans  # noqa: F401
import SAEGenRec.sid_builder.rqvae  # noqa: F401


def test_all_three_methods_registered():
    """Test that rqvae, rqkmeans, gated_sae are all registered."""
    from SAEGenRec.sid_builder.registry import list_sid_methods

    methods = list_sid_methods()
    assert "rqvae" in methods
    assert "rqkmeans" in methods
    assert "gated_sae" in methods


def test_registry_lookup_by_name():
    """Test that methods can be looked up by name."""
    from SAEGenRec.sid_builder.registry import get_sid_method

    method = get_sid_method("rqvae")
    assert method is not None


def test_unknown_method_raises_error():
    """Test that unknown method names raise ValueError."""
    from SAEGenRec.sid_builder.registry import get_sid_method

    with pytest.raises(ValueError, match="Unknown SID method"):
        get_sid_method("nonexistent_method")


def test_sid_method_base_class_interface():
    """Test SIDMethod abstract base class has train/generate interface."""
    from SAEGenRec.sid_builder.base import SIDMethod

    assert hasattr(SIDMethod, "train")
    assert hasattr(SIDMethod, "generate")
    assert hasattr(SIDMethod, "name")
    assert hasattr(SIDMethod, "default_k")
    assert hasattr(SIDMethod, "token_format")


def test_token_format_auto_resolution_rqvae():
    """Test that rqvae uses positional prefix in auto mode."""
    from SAEGenRec.sid_builder.registry import get_sid_method

    method = get_sid_method("rqvae")
    assert method.token_format == "auto"


def test_token_format_auto_resolution_gated_sae():
    """Test that gated_sae uses 'f' prefix in auto mode."""
    from SAEGenRec.sid_builder.registry import get_sid_method

    method = get_sid_method("gated_sae")
    assert method.token_format == "auto"


def test_custom_token_format_override():
    """Test that token_format can be overridden."""
    from SAEGenRec.sid_builder.registry import get_sid_method

    method = get_sid_method("rqvae")
    # token_format is an instance attribute that can be overridden
    method_instance = method.__class__()
    method_instance.token_format = "v"
    assert method_instance.token_format == "v"

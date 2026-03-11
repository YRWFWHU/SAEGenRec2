"""Unit tests for reward registry."""

import pytest


def test_five_builtin_rewards_registered():
    """Test that all 5 builtin rewards are registered."""
    from SAEGenRec.training.rewards import list_rewards

    rewards = list_rewards()
    for name in ["rule", "prefix", "ranking", "semantic", "sasrec"]:
        assert name in rewards, f"'{name}' not in registry"


def test_custom_reward_registration():
    """Test that custom rewards can be registered via @register_reward."""
    from SAEGenRec.training.rewards import _REWARD_REGISTRY, register_reward

    @register_reward("test_custom_reward_xyz")
    def my_reward(predictions, target, **kwargs):
        return [1.0] * len(predictions)

    assert "test_custom_reward_xyz" in _REWARD_REGISTRY
    # Cleanup
    del _REWARD_REGISTRY["test_custom_reward_xyz"]


def test_get_reward_fn_rule():
    """Test that get_reward_fn returns correct function for 'rule'."""
    from SAEGenRec.training.rewards import get_reward_fn, rule_reward

    fn = get_reward_fn("rule")
    assert callable(fn)
    result = fn(["[a_1][b_2][c_3]", "[a_4][b_5][c_6]"], "[a_1][b_2][c_3]")
    assert result == [1.0, 0.0]


def test_unknown_reward_raises_error():
    """Test that unknown reward name raises ValueError with available list."""
    from SAEGenRec.training.rewards import get_reward_fn

    with pytest.raises(ValueError, match="Unknown reward"):
        get_reward_fn("nonexistent_reward_xyz")


def test_combined_reward_weighted_aggregation():
    """Test CombinedReward weighted aggregation."""
    from SAEGenRec.training.rewards import CombinedReward

    # rule gives [1, 0], prefix gives partial matches
    combined = CombinedReward(names=["rule", "rule"], weights=[0.6, 0.4])
    result = combined(["[a_1][b_2][c_3]", "[a_4][b_5][c_6]"], "[a_1][b_2][c_3]")
    assert len(result) == 2
    assert abs(result[0] - 1.0) < 1e-6
    assert abs(result[1] - 0.0) < 1e-6


def test_parse_reward_type_single():
    """Test parse_reward_type with single reward name."""
    from SAEGenRec.training.rewards import parse_reward_type

    fn = parse_reward_type("rule", "")
    assert callable(fn)
    result = fn(["abc"], "abc")
    assert result == [1.0]


def test_parse_reward_type_combined():
    """Test parse_reward_type with '+' syntax creates CombinedReward."""
    from SAEGenRec.training.rewards import CombinedReward, parse_reward_type

    fn = parse_reward_type("rule+prefix", "0.7,0.3")
    assert isinstance(fn, CombinedReward)


def test_exception_handling_returns_zero():
    """Test that get_reward_fn wraps exceptions to return 0.0."""
    from SAEGenRec.training.rewards import _REWARD_REGISTRY, register_reward

    @register_reward("test_raising_reward")
    def bad_reward(predictions, target, **kwargs):
        raise RuntimeError("simulated error")

    from SAEGenRec.training.rewards import get_reward_fn

    fn = get_reward_fn("test_raising_reward")
    result = fn(["abc", "def"], "abc")
    assert result == [0.0, 0.0]
    del _REWARD_REGISTRY["test_raising_reward"]

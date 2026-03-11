"""Unit tests for TrainingEvaluator callback."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


def test_callback_creation():
    """Test TrainingEvaluator can be created with parameters."""
    from SAEGenRec.evaluation.training_evaluator import TrainingEvaluator

    evaluator = TrainingEvaluator(
        eval_rec_steps=100,
        eval_rec_beams=4,
        eval_rec_samples=10,
        info_file="dummy.txt",
        test_csv="dummy.csv",
        k_values=[1, 5, 10],
    )
    assert evaluator.eval_rec_steps == 100
    assert evaluator.eval_rec_beams == 4
    assert evaluator.eval_rec_samples == 10
    assert evaluator.k_values == [1, 5, 10]


def test_disabled_by_default():
    """Test that TrainingEvaluator is effectively disabled when eval_rec_steps <= 0."""
    from SAEGenRec.evaluation.training_evaluator import TrainingEvaluator

    evaluator = TrainingEvaluator(
        eval_rec_steps=0,
        info_file="dummy.txt",
        test_csv="dummy.csv",
    )
    # on_step_end should return without calling evaluate when steps=0
    mock_args = MagicMock()
    mock_args.max_steps = 100
    mock_state = MagicMock()
    mock_state.global_step = 10
    mock_control = MagicMock()
    mock_trainer = MagicMock()

    with patch.object(evaluator, "_run_eval") as mock_run:
        evaluator.on_step_end(mock_args, mock_state, mock_control, model=MagicMock())
        mock_run.assert_not_called()


def test_step_interval_calculation():
    """Test that eval triggers at correct step intervals."""
    from SAEGenRec.evaluation.training_evaluator import TrainingEvaluator

    evaluator = TrainingEvaluator(
        eval_rec_steps=50,
        info_file="dummy.txt",
        test_csv="dummy.csv",
    )

    mock_args = MagicMock()
    mock_args.max_steps = 1000
    mock_control = MagicMock()

    triggered_steps = []

    def fake_run_eval(model, tokenizer):
        triggered_steps.append(True)

    evaluator._run_eval = fake_run_eval

    # Step 50 should trigger
    mock_state = MagicMock()
    mock_state.global_step = 50
    evaluator.on_step_end(mock_args, mock_state, mock_control, model=MagicMock())
    assert len(triggered_steps) == 1

    # Step 75 should NOT trigger
    mock_state.global_step = 75
    evaluator.on_step_end(mock_args, mock_state, mock_control, model=MagicMock())
    assert len(triggered_steps) == 1

    # Step 100 should trigger
    mock_state.global_step = 100
    evaluator.on_step_end(mock_args, mock_state, mock_control, model=MagicMock())
    assert len(triggered_steps) == 2


def test_metrics_logging_format():
    """Test that metrics are logged in expected format."""
    from SAEGenRec.evaluation.training_evaluator import TrainingEvaluator

    evaluator = TrainingEvaluator(
        eval_rec_steps=10,
        info_file="dummy.txt",
        test_csv="dummy.csv",
        k_values=[1, 5, 10],
    )

    mock_metrics = {"HR@1": 0.05, "HR@5": 0.20, "HR@10": 0.35, "NDCG@5": 0.12, "NDCG@10": 0.18}

    log_messages = []
    with patch("SAEGenRec.evaluation.training_evaluator.logger") as mock_logger:
        mock_logger.info.side_effect = lambda msg: log_messages.append(msg)
        evaluator._log_metrics(step=10, total_steps=100, metrics=mock_metrics)

    # At least one log message should contain step info and metric values
    assert any("10" in m and "100" in m for m in log_messages)
    assert any("HR@1" in m for m in log_messages)

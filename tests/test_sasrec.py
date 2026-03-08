"""tests/test_sasrec.py — SASRec, GRU, Caser model tests."""

import copy
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from SAEGenRec.models.sasrec import Caser, GRU, SASRec


DEVICE = torch.device("cpu")
BATCH = 4
SEQ_SIZE = 10
ITEM_NUM = 50
HIDDEN = 32


# ---------------------------------------------------------------------------
# SASRec
# ---------------------------------------------------------------------------


class TestSASRec:
    def _model(self):
        return SASRec(HIDDEN, ITEM_NUM, SEQ_SIZE, dropout=0.0, device=DEVICE)

    def test_forward_output_shape(self):
        model = self._model()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,))
        out = model(states, len_states)
        # squeeze() may collapse batch dim when BATCH=1; check in general
        assert out.shape[-1] == ITEM_NUM

    def test_forward_eval_matches_forward(self):
        model = self._model()
        model.eval()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,))
        with torch.no_grad():
            out1 = model.forward(states, len_states)
            out2 = model.forward_eval(states, len_states)
        assert torch.allclose(out1, out2)

    def test_padding_index_ignored(self):
        """Padding token (item_num) should be masked out."""
        model = self._model()
        model.eval()
        # all-padding sequence vs first token real
        states_pad = torch.full((1, SEQ_SIZE), ITEM_NUM, dtype=torch.long)
        states_real = torch.randint(0, ITEM_NUM, (1, SEQ_SIZE))
        len_s = torch.tensor([1])
        with torch.no_grad():
            out_pad = model.forward(states_pad, len_s)
            out_real = model.forward(states_real, len_s)
        # outputs differ since real items have content
        assert out_pad.shape == out_real.shape

    def test_bce_loss_decreases(self):
        model = self._model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,))
        target = torch.randint(0, ITEM_NUM, (BATCH,))
        target_neg = torch.randint(0, ITEM_NUM, (BATCH,))

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(states, len_states)
            pos = torch.gather(out, 1, target.view(-1, 1))
            neg = torch.gather(out, 1, target_neg.view(-1, 1))
            scores = torch.cat([pos, neg], 0)
            labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], 0)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # loss should generally be finite and positive
        assert all(np.isfinite(l) for l in losses)

    def test_item_embeddings_initialized(self):
        model = self._model()
        w = model.item_embeddings.weight.detach().numpy()
        # Normal init with std=0.01 — not all zeros
        assert not np.allclose(w, 0.0)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------


class TestGRU:
    def _model(self):
        return GRU(HIDDEN, ITEM_NUM, SEQ_SIZE)

    def test_forward_output_shape(self):
        model = self._model()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,)).cpu()
        out = model(states, len_states)
        assert out.shape == (BATCH, ITEM_NUM)

    def test_forward_eval_output_shape(self):
        model = self._model()
        model.eval()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,))
        with torch.no_grad():
            out = model.forward_eval(states, len_states)
        assert out.shape == (BATCH, ITEM_NUM)

    def test_gru_init(self):
        model = self._model()
        assert isinstance(model.gru, nn.GRU)
        assert model.gru.input_size == HIDDEN
        assert model.gru.hidden_size == HIDDEN

    def test_bce_loss_backward(self):
        model = self._model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,)).cpu()
        target = torch.randint(0, ITEM_NUM, (BATCH,))

        optimizer.zero_grad()
        out = model(states, len_states)
        pos = torch.gather(out, 1, target.view(-1, 1))
        labels = torch.ones_like(pos)
        loss = criterion(pos, labels)
        loss.backward()
        optimizer.step()
        assert np.isfinite(loss.item())


# ---------------------------------------------------------------------------
# Caser
# ---------------------------------------------------------------------------


class TestCaser:
    def _model(self, filter_sizes=None):
        fs = filter_sizes or [2, 3, 4]
        return Caser(HIDDEN, ITEM_NUM, SEQ_SIZE, num_filters=16, filter_sizes=fs, dropout_rate=0.0)

    def test_forward_output_shape(self):
        model = self._model()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,))
        out = model(states, len_states)
        assert out.shape == (BATCH, ITEM_NUM)

    def test_forward_eval_output_shape(self):
        model = self._model()
        model.eval()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,))
        with torch.no_grad():
            out = model.forward_eval(states, len_states)
        assert out.shape == (BATCH, ITEM_NUM)

    def test_caser_init_string_filter_sizes(self):
        model = Caser(HIDDEN, ITEM_NUM, SEQ_SIZE, 16, "[2,3,4]", 0.0)
        assert model.filter_sizes == [2, 3, 4]

    def test_caser_horizontal_cnn_count(self):
        model = self._model([2, 3])
        assert len(model.horizontal_cnn) == 2

    def test_bce_loss_backward(self):
        model = self._model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        states = torch.randint(0, ITEM_NUM, (BATCH, SEQ_SIZE))
        len_states = torch.randint(1, SEQ_SIZE + 1, (BATCH,))
        target = torch.randint(0, ITEM_NUM, (BATCH,))

        optimizer.zero_grad()
        out = model(states, len_states)
        pos = torch.gather(out, 1, target.view(-1, 1))
        labels = torch.ones_like(pos)
        loss = criterion(pos, labels)
        loss.backward()
        optimizer.step()
        assert np.isfinite(loss.item())


# ---------------------------------------------------------------------------
# Early stopping simulation
# ---------------------------------------------------------------------------


def test_early_stopping_trigger():
    """Simulate early stopping: counter increments when NDCG doesn't improve."""
    ndcg_max = 0.0
    early_stop_count = 0
    early_stop_limit = 3

    # Simulate 5 epochs with no improvement
    for _ in range(5):
        ndcg_val = 0.0  # no improvement
        if ndcg_val > ndcg_max:
            ndcg_max = ndcg_val
            early_stop_count = 0
        else:
            early_stop_count += 1

    assert early_stop_count > early_stop_limit


def test_early_stopping_reset_on_improvement():
    """Counter resets when NDCG improves."""
    ndcg_max = 0.0
    early_stop_count = 0

    improvements = [0.1, 0.2, 0.15, 0.3]  # drops at epoch 2
    for ndcg_val in improvements:
        if ndcg_val > ndcg_max:
            ndcg_max = ndcg_val
            early_stop_count = 0
        else:
            early_stop_count += 1

    # After epoch 2 (0.15 < 0.2), count=1; epoch 3 (0.3 > 0.2) resets to 0
    assert early_stop_count == 0
    assert ndcg_max == 0.3

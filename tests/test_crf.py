"""Unit tests for the linear-chain CRF layer (src.models.CRF)."""

import itertools

import pytest
import torch

from src.models import CRF


def _brute_force_log_partition(crf: CRF, emissions: torch.Tensor) -> torch.Tensor:
    """Exact log-partition for one unmasked sequence, by enumerating every path."""
    assert emissions.shape[0] == 1
    seq_len, num_tags = emissions.shape[1], emissions.shape[2]

    def path_score(seq):
        s = crf.start_transitions[seq[0]] + emissions[0, 0, seq[0]]
        for i in range(1, seq_len):
            s = s + crf.transitions[seq[i - 1], seq[i]] + emissions[0, i, seq[i]]
        return s + crf.end_transitions[seq[-1]]

    scores = torch.stack([path_score(p) for p in itertools.product(range(num_tags), repeat=seq_len)])
    return torch.logsumexp(scores, dim=0)


class TestInit:
    def test_shapes(self):
        crf = CRF(num_tags=5)
        assert crf.num_tags == 5
        assert crf.transitions.shape == (5, 5)
        assert crf.start_transitions.shape == (5,)
        assert crf.end_transitions.shape == (5,)

    def test_params_are_learnable(self):
        crf = CRF(num_tags=4)
        names = {n for n, _ in crf.named_parameters()}
        assert names == {"transitions", "start_transitions", "end_transitions"}
        assert all(p.requires_grad for p in crf.parameters())


class TestForward:
    @pytest.mark.parametrize("batch,seq_len,num_tags", [(2, 10, 5), (4, 7, 9), (1, 12, 3), (3, 5, 3)])
    def test_returns_finite_scalar(self, batch, seq_len, num_tags):
        crf = CRF(num_tags=num_tags)
        emissions = torch.randn(batch, seq_len, num_tags)
        tags = torch.randint(0, num_tags, (batch, seq_len))
        mask = torch.ones(batch, seq_len, dtype=torch.long)

        loss = crf(emissions, tags, mask)

        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0  # NLL of a valid path is non-negative

    def test_gradients_flow(self):
        crf = CRF(num_tags=5)
        emissions = torch.randn(3, 8, 5, requires_grad=True)
        tags = torch.randint(0, 5, (3, 8))
        mask = torch.ones(3, 8, dtype=torch.long)

        crf(emissions, tags, mask).backward()

        assert emissions.grad is not None
        assert crf.transitions.grad is not None
        assert torch.isfinite(emissions.grad).all()

    def test_matches_brute_force_nll(self):
        """The forward algorithm must agree with exhaustive path enumeration."""
        torch.manual_seed(1)
        crf = CRF(num_tags=3)
        seq_len = 4
        emissions = torch.randn(1, seq_len, 3)
        mask = torch.ones(1, seq_len, dtype=torch.long)
        gold = torch.tensor([[0, 1, 2, 0]])

        log_z = _brute_force_log_partition(crf, emissions)
        gold_score = (
            crf.start_transitions[gold[0, 0]]
            + emissions[0, 0, gold[0, 0]]
            + sum(crf.transitions[gold[0, i - 1], gold[0, i]] + emissions[0, i, gold[0, i]]
                  for i in range(1, seq_len))
            + crf.end_transitions[gold[0, -1]]
        )
        expected_nll = -(gold_score - log_z)

        got = crf(emissions, gold, mask)
        assert torch.allclose(got, expected_nll, atol=1e-4)

    def test_masked_positions_are_ignored(self):
        """Padding tokens must not change the loss regardless of their tags."""
        torch.manual_seed(2)
        crf = CRF(num_tags=4)
        emissions = torch.randn(2, 6, 4)
        tags = torch.randint(0, 4, (2, 6))
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])

        base = crf(emissions, tags, mask)
        tampered = tags.clone()
        tampered[:, 4:] = (tampered[:, 4:] + 1) % 4
        assert torch.allclose(base, crf(emissions, tampered, mask), atol=1e-5)


class TestDecode:
    @pytest.mark.parametrize("batch,seq_len,num_tags", [(2, 10, 5), (3, 6, 9), (1, 8, 3)])
    def test_shape_and_range(self, batch, seq_len, num_tags):
        crf = CRF(num_tags=num_tags)
        emissions = torch.randn(batch, seq_len, num_tags)
        mask = torch.ones(batch, seq_len, dtype=torch.long)

        decoded = crf.decode(emissions, mask)

        assert decoded.shape == (batch, seq_len)
        assert decoded.min() >= 0 and decoded.max() < num_tags

    def test_decode_prefers_dominant_emissions(self):
        """With peaked emissions and near-zero transitions, decode = argmax."""
        crf = CRF(num_tags=4)
        with torch.no_grad():
            crf.transitions.zero_()
            crf.start_transitions.zero_()
            crf.end_transitions.zero_()
        emissions = torch.full((1, 5, 4), -10.0)
        target = [2, 0, 3, 1, 2]
        for i, t in enumerate(target):
            emissions[0, i, t] = 10.0

        decoded = crf.decode(emissions, torch.ones(1, 5, dtype=torch.long))
        assert decoded[0].tolist() == target

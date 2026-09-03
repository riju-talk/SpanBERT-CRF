"""Tests for the SpanBERT-CRF model heads.

The BIO span helpers run without any pretrained weights. The full-model tests
are marked ``slow`` and skipped when the SpanBERT encoder is unavailable.
"""

import pytest
import torch

from src.models import (
    QA_TAG_B,
    QA_TAG_I,
    QA_TAG_O,
    SpanBERTForNER,
    SpanBERTForQA,
    _bio_emissions_to_span_logits,
    _spans_to_bio,
)


class TestSpansToBIO:
    def test_answerable_span_is_b_then_i(self):
        bio = _spans_to_bio(torch.tensor([3]), torch.tensor([5]), seq_len=8)
        assert bio[0].tolist() == [
            QA_TAG_O, QA_TAG_O, QA_TAG_O, QA_TAG_B, QA_TAG_I, QA_TAG_I, QA_TAG_O, QA_TAG_O
        ]

    def test_single_token_answer(self):
        bio = _spans_to_bio(torch.tensor([2]), torch.tensor([2]), seq_len=5)
        assert bio[0].tolist() == [QA_TAG_O, QA_TAG_O, QA_TAG_B, QA_TAG_O, QA_TAG_O]

    def test_unanswerable_is_all_O(self):
        bio = _spans_to_bio(torch.tensor([0]), torch.tensor([0]), seq_len=6)
        assert bio[0].sum().item() == 0

    def test_batch_mixed(self):
        starts = torch.tensor([1, 0, 4])
        ends = torch.tensor([2, 0, 4])
        bio = _spans_to_bio(starts, ends, seq_len=6)
        assert bio.shape == (3, 6)
        assert bio[1].sum().item() == 0            # unanswerable row
        assert bio[0].tolist() == [QA_TAG_O, QA_TAG_B, QA_TAG_I, QA_TAG_O, QA_TAG_O, QA_TAG_O]
        assert bio[2, 4].item() == QA_TAG_B


class TestBIOEmissionsToSpanLogits:
    def test_start_and_end_logits_track_the_right_tags(self):
        emissions = torch.zeros(1, 4, 3)
        emissions[0, 1, QA_TAG_B] = 5.0   # strong span start at token 1
        emissions[0, 2, QA_TAG_I] = 5.0   # strong span interior at token 2
        start_logits, end_logits = _bio_emissions_to_span_logits(emissions)

        assert start_logits.shape == end_logits.shape == (1, 4)
        # start is driven by B-ANS only -> token 1
        assert start_logits.argmax(dim=-1).item() == 1
        # end is driven by max(B-ANS, I-ANS) -> tokens 1 and 2 both light up,
        # and both beat the non-span tokens 0 and 3
        assert end_logits[0, 1] > 0 and end_logits[0, 2] > 0
        assert end_logits[0, 2] > end_logits[0, 3]
        assert end_logits[0, 2] > end_logits[0, 0]


@pytest.mark.slow
class TestSpanBERTForQA:
    def test_crf_is_default_and_present(self, require_encoder, spanbert_name):
        model = SpanBERTForQA(model_name=spanbert_name)
        assert model.use_crf is True
        assert hasattr(model, "crf")
        assert hasattr(model, "qa_classifier")

    def test_forward_train_and_eval(self, require_encoder, spanbert_name):
        model = SpanBERTForQA(model_name=spanbert_name)
        b, l = 2, 24
        ids = torch.randint(0, 1000, (b, l))
        mask = torch.ones(b, l, dtype=torch.long)

        model.train()
        loss, start_logits, end_logits = model(
            input_ids=ids, attention_mask=mask,
            start_positions=torch.tensor([4, 0]), end_positions=torch.tensor([6, 0]),
        )
        assert loss.dim() == 0 and torch.isfinite(loss)
        assert start_logits.shape == (b, l) and end_logits.shape == (b, l)
        loss.backward()

        model.eval()
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
            tags = model.decode_spans(ids, mask)
        assert len(out) == 2
        assert tags.shape == (b, l)

    def test_non_crf_ablation_path(self, require_encoder, spanbert_name):
        model = SpanBERTForQA(use_crf=False, model_name=spanbert_name)
        assert not hasattr(model, "crf")
        out = model(input_ids=torch.randint(0, 1000, (1, 16)),
                    attention_mask=torch.ones(1, 16, dtype=torch.long))
        assert out[0].shape == (1, 16)


@pytest.mark.slow
class TestSpanBERTForNER:
    def test_crf_forward_handles_ignored_labels(self, require_encoder, spanbert_name):
        model = SpanBERTForNER(num_ner_tags=9, use_crf=True, model_name=spanbert_name)
        b, l = 2, 20
        ids = torch.randint(0, 1000, (b, l))
        mask = torch.ones(b, l, dtype=torch.long)
        labels = torch.randint(0, 9, (b, l))
        labels[:, 0] = -100   # CLS
        labels[:, -1] = -100  # SEP

        model.train()
        loss = model(input_ids=ids, attention_mask=mask, labels=labels)[0]
        assert torch.isfinite(loss)
        loss.backward()

        model.eval()
        with torch.no_grad():
            decoded = model(input_ids=ids, attention_mask=mask)[-1]
        assert decoded.shape == (b, l)

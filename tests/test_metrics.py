"""Unit tests for src.metrics (QA and NER scoring)."""

import pytest
import torch

from src.metrics import (
    Evaluator,
    compute_f1,
    compute_ner_metrics,
    compute_qa_metrics,
    compute_span_overlap_metrics,
    normalize_text,
)


class TestNormalization:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_text("The Quick, Brown FOX!") == "the quick brown fox"

    def test_collapses_whitespace(self):
        assert normalize_text("  a   b\tc\n") == "a b c"


class TestComputeF1:
    def test_identical_strings(self):
        assert compute_f1("paris france", "paris france") == 1.0

    def test_disjoint_strings(self):
        assert compute_f1("paris", "london") == 0.0

    def test_partial_overlap(self):
        f1 = compute_f1("the quick brown fox", "quick brown fox jumps")
        assert 0.0 < f1 < 1.0

    def test_case_insensitive(self):
        assert compute_f1("Paris", "paris") == 1.0


class TestQAMetrics:
    def test_half_exact_match(self):
        preds = [
            {"id": "1", "prediction_text": "Paris"},
            {"id": "2", "prediction_text": "London"},
        ]
        gold = [
            {"id": "1", "answers": ["Paris"]},
            {"id": "2", "answers": ["Berlin"]},
        ]
        m = compute_qa_metrics(preds, gold)
        assert m["exact_match"] == 0.5
        assert 0.0 <= m["f1"] <= 1.0

    def test_perfect_match_is_case_insensitive(self):
        preds = [{"id": "1", "prediction_text": "The Quick Brown Fox"}]
        gold = [{"id": "1", "answers": ["the quick brown fox"]}]
        m = compute_qa_metrics(preds, gold)
        assert m["exact_match"] == 1.0
        assert m["f1"] == 1.0

    def test_takes_best_over_multiple_references(self):
        preds = [{"id": "1", "prediction_text": "NYC"}]
        gold = [{"id": "1", "answers": ["New York", "NYC", "New York City"]}]
        m = compute_qa_metrics(preds, gold)
        assert m["exact_match"] == 1.0

    def test_reports_expected_keys(self):
        preds = [{"id": "1", "prediction_text": "a"}]
        gold = [{"id": "1", "answers": ["a"]}]
        assert set(compute_qa_metrics(preds, gold)) == {"exact_match", "f1", "bleu", "bertscore_f1"}


class TestNERMetrics:
    ID2LABEL = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG"}

    def test_perfect_prediction(self):
        preds = [[0, 1, 2, 0, 3, 4, 0]]
        gold = [[0, 1, 2, 0, 3, 4, 0]]
        m = compute_ner_metrics(preds, gold, self.ID2LABEL)
        assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_missed_entity_lowers_recall(self):
        gold = [[1, 2, 0, 3, 4]]     # PER + ORG
        preds = [[1, 2, 0, 0, 0]]    # ORG missed
        m = compute_ner_metrics(preds, gold, self.ID2LABEL)
        assert m["recall"] == 0.5
        assert m["precision"] == 1.0

    def test_spurious_entity_lowers_precision(self):
        gold = [[1, 2, 0, 0, 0]]
        preds = [[1, 2, 0, 3, 4]]
        m = compute_ner_metrics(preds, gold, self.ID2LABEL)
        assert m["precision"] == 0.5
        assert m["recall"] == 1.0

    def test_wrong_type_is_both_fp_and_fn(self):
        gold = [[1, 2]]   # PER
        preds = [[3, 4]]  # ORG
        m = compute_ner_metrics(preds, gold, self.ID2LABEL)
        assert m["f1"] == 0.0


class TestSpanOverlap:
    def test_exact_and_partial(self):
        pred_s = torch.tensor([1, 5, 3])
        pred_e = torch.tensor([2, 9, 3])
        true_s = torch.tensor([1, 5, 4])
        true_e = torch.tensor([2, 8, 4])
        m = compute_span_overlap_metrics(pred_s, pred_e, true_s, true_e)
        assert m["exact_span_match"] == pytest.approx(1 / 3)
        assert m["partial_span_match"] == pytest.approx(2 / 3)


class TestEvaluator:
    def test_dispatches_on_task_type(self):
        qa = Evaluator("qa")
        out = qa.evaluate(
            [{"id": "1", "prediction_text": "a"}],
            [{"id": "1", "answers": ["a"]}],
        )
        assert out["exact_match"] == 1.0

    def test_rejects_unknown_task(self):
        with pytest.raises(ValueError):
            Evaluator("summarization").evaluate([], [])

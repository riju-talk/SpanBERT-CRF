"""Tests for src.data — dataset tensorisation and label handling.

These use the real SpanBERT tokenizer (cached); the ``tokenizer`` fixture skips
the module when it cannot be loaded. No model weights are needed.
"""

import torch

from src.data import CONLL_LABEL_MAP, NERDataset, QADataset, create_dataloaders


class TestQADataset:
    def _example(self):
        return {
            "question": "What is the capital of France?",
            "context": "Paris is the capital of France and a major city.",
            "answer_start": 0,
            "answer_text": "Paris",
        }

    def test_item_shapes_and_keys(self, tokenizer):
        ds = QADataset([self._example()], tokenizer, max_length=64)
        item = ds[0]
        assert {"input_ids", "attention_mask", "start_positions", "end_positions"} <= set(item)
        assert item["input_ids"].shape == (64,)
        assert item["start_positions"].dtype == torch.long
        assert 0 <= item["start_positions"].item() <= item["end_positions"].item()

    def test_unanswerable_maps_to_cls(self, tokenizer):
        ex = self._example()
        ex["answer_text"] = ""
        ex["answer_start"] = 0
        ds = QADataset([ex], tokenizer, max_length=64)
        item = ds[0]
        assert item["start_positions"].item() == 0
        assert item["end_positions"].item() == 0

    def test_len(self, tokenizer):
        ds = QADataset([self._example()] * 3, tokenizer, max_length=32)
        assert len(ds) == 3


class TestNERDataset:
    EX = {"tokens": ["EU", "rejects", "German", "call"], "ner_tags": ["B-ORG", "O", "B-MISC", "O"]}

    def test_item_shapes_and_special_token_labels(self, tokenizer):
        ds = NERDataset([self.EX], tokenizer, CONLL_LABEL_MAP, max_length=16)
        item = ds[0]
        assert item["input_ids"].shape == (16,)
        assert item["labels"].shape == (16,)
        assert item["labels"][0].item() == -100          # CLS
        assert item["labels"][5:].eq(-100).all()          # SEP + padding
        assert item["labels"][1].item() == CONLL_LABEL_MAP["B-ORG"]

    def test_attention_mask_matches_real_tokens(self, tokenizer):
        ds = NERDataset([self.EX], tokenizer, CONLL_LABEL_MAP, max_length=16)
        item = ds[0]
        assert item["attention_mask"].sum().item() == len(self.EX["tokens"]) + 2


class TestCreateDataloaders:
    def test_qa_loader_batches(self, tokenizer):
        data = [{
            "question": "q?", "context": "the answer is here", "answer_start": 4, "answer_text": "answer",
        } for _ in range(6)]
        train, val = create_dataloaders(data, data, tokenizer, batch_size=3,
                                        max_length=32, dataset_type="qa")
        batch = next(iter(train))
        assert batch["input_ids"].shape == (3, 32)
        assert len(val) == 2

    def test_ner_loader_accepts_label_map_tuple(self, tokenizer):
        raw = [TestNERDataset.EX for _ in range(4)]
        train, _ = create_dataloaders((raw, CONLL_LABEL_MAP), (raw, CONLL_LABEL_MAP),
                                      tokenizer, batch_size=2, max_length=16, dataset_type="ner")
        batch = next(iter(train))
        assert batch["labels"].shape == (2, 16)


class TestLabelMap:
    def test_bio_scheme_is_complete(self):
        assert CONLL_LABEL_MAP["O"] == 0
        for ent in ("PER", "ORG", "LOC", "MISC"):
            assert f"B-{ent}" in CONLL_LABEL_MAP and f"I-{ent}" in CONLL_LABEL_MAP
        assert sorted(CONLL_LABEL_MAP.values()) == list(range(9))

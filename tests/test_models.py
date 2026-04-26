"""Unit tests for SpanBERT-CRF models and utilities."""

import pytest
import torch
from src.models import CRF, SpanBERTForQA, SpanBERTForNER
from src.metrics import compute_qa_metrics, compute_ner_metrics


class TestCRF:
    """Tests for CRF layer."""
    
    def test_crf_initialization(self):
        """Test CRF layer initialization."""
        crf = CRF(num_tags=5)
        assert crf.num_tags == 5
        assert crf.transitions.shape == (5, 5)
        assert crf.start_transitions.shape == (5,)
        assert crf.end_transitions.shape == (5,)
    
    def test_crf_forward(self):
        """Test CRF forward pass."""
        batch_size, seq_len, num_tags = 2, 10, 5
        crf = CRF(num_tags=num_tags)
        
        emissions = torch.randn(batch_size, seq_len, num_tags)
        tags = torch.randint(0, num_tags, (batch_size, seq_len))
        
        loss = crf(emissions, tags)
        assert loss.dim() == 0
        assert loss > 0
    
    def test_crf_decode(self):
        """Test CRF Viterbi decoding."""
        batch_size, seq_len, num_tags = 2, 10, 5
        crf = CRF(num_tags=num_tags)
        
        emissions = torch.randn(batch_size, seq_len, num_tags)
        decoded = crf.decode(emissions)
        
        assert decoded.shape == (batch_size, seq_len)
        assert all(0 <= tag < num_tags for tag in decoded.flatten())


class TestSpanBERTForQA:
    """Tests for QA model."""
    
    def test_qa_model_init(self):
        """Test QA model initialization."""
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained('spanbert-base-cased', num_labels=2)
        model = SpanBERTForQA(config, use_crf=False)
        
        assert hasattr(model, 'spanbert')
        assert hasattr(model, 'qa_outputs')
    
    def test_qa_model_forward(self):
        """Test QA model forward pass."""
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained('spanbert-base-cased', num_labels=2)
        model = SpanBERTForQA(config, use_crf=False)
        model.eval()
        
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        token_type_ids = torch.zeros(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        start_logits, end_logits = outputs
        assert start_logits.shape == (batch_size, seq_len)
        assert end_logits.shape == (batch_size, seq_len)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_qa_exact_match(self):
        """Test exact match calculation."""
        predictions = [
            {'id': '1', 'prediction_text': 'Paris'},
            {'id': '2', 'prediction_text': 'London'}
        ]
        ground_truth = [
            {'id': '1', 'answers': ['Paris']},
            {'id': '2', 'answers': ['Berlin']}
        ]
        
        metrics = compute_qa_metrics(predictions, ground_truth)
        
        assert metrics['exact_match'] == 0.5
        assert 0 < metrics['f1'] < 1
    
    def test_qa_perfect_match(self):
        """Test perfect exact match and F1."""
        predictions = [
            {'id': '1', 'prediction_text': 'The quick brown fox'}
        ]
        ground_truth = [
            {'id': '1', 'answers': ['the quick brown fox']}
        ]
        
        metrics = compute_qa_metrics(predictions, ground_truth)
        
        assert metrics['exact_match'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_ner_metrics(self):
        """Test NER metric calculation."""
        id2label = {
            0: 'O', 1: 'B-PER', 2: 'I-PER',
            3: 'B-ORG', 4: 'I-ORG'
        }
        
        predictions = [[0, 1, 2, 0, 3, 4, 0]]
        ground_truth = [[0, 1, 2, 0, 3, 4, 0]]
        
        metrics = compute_ner_metrics(predictions, ground_truth, id2label)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

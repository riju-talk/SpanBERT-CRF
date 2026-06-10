"""
SpanBERT-CRF: Unified NLP Architecture for QA and NER
"""

__version__ = "1.0.0"
__author__ = "Riju"

from src.models import CRF, SpanBERTForQA, SpanBERTForNER, apply_lora_to_model
from src.data import QADataset, NERDataset, load_squad_data, load_conll_ner_data
from src.metrics import compute_qa_metrics, compute_ner_metrics, Evaluator
from src.train import Trainer
from src.inference import QAInference, NERInference, load_inference_model

__all__ = [
    'CRF',
    'SpanBERTForQA',
    'SpanBERTForNER',
    'apply_lora_to_model',
    'QADataset',
    'NERDataset',
    'load_squad_data',
    'load_conll_ner_data',
    'compute_qa_metrics',
    'compute_ner_metrics',
    'Evaluator',
    'Trainer',
    'QAInference',
    'NERInference',
    'load_inference_model'
]

"""
Evaluation metrics for QA and NER tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


def compute_qa_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """
    Compute Exact Match (EM) and F1 scores for Question Answering.
    
    Args:
        predictions: List of dicts with 'id', 'prediction_text'
        ground_truth: List of dicts with 'id', 'answers' (list of acceptable answers)
    
    Returns:
        Dictionary with EM and F1 scores
    """
    
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def compute_f1(pred: str, truth: str) -> float:
        """Compute F1 score between prediction and ground truth."""
        pred_tokens = normalize_text(pred).split()
        truth_tokens = normalize_text(truth).split()
        
        if not pred_tokens and not truth_tokens:
            return 1.0 if pred == truth else 0.0
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    exact_matches = []
    f1_scores = []
    
    for pred, truth in zip(predictions, ground_truth):
        pred_text = pred['prediction_text']
        true_answers = truth['answers']
        
        # Check for exact match with any acceptable answer
        em = max(1.0 if normalize_text(pred_text) == normalize_text(ans) else 0.0 
                for ans in true_answers)
        exact_matches.append(em)
        
        # Compute F1 with best matching answer
        f1 = max(compute_f1(pred_text, ans) for ans in true_answers)
        f1_scores.append(f1)
    
    return {
        'exact_match': np.mean(exact_matches),
        'f1': np.mean(f1_scores)
    }


def compute_ner_metrics(predictions: List[List[int]], ground_truth: List[List[int]], 
                        id2label: Dict[int, str]) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for Named Entity Recognition.
    
    Args:
        predictions: List of predicted tag sequences
        ground_truth: List of gold tag sequences
        id2label: Mapping from tag IDs to tag names
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    
    def extract_entities(tags: List[int], id2label: Dict[int, str]) -> set:
        """Extract entity spans from BIO-tagged sequence."""
        entities = set()
        current_entity = None
        start_idx = None
        
        for i, tag_id in enumerate(tags):
            if tag_id == -100:  # Ignore padding
                continue
            
            tag = id2label[tag_id]
            
            if tag.startswith('B-'):
                if current_entity is not None:
                    entities.add((current_entity, start_idx, i - 1))
                current_entity = tag[2:]
                start_idx = i
            elif tag.startswith('I-'):
                entity_type = tag[2:]
                if current_entity != entity_type:
                    if current_entity is not None:
                        entities.add((current_entity, start_idx, i - 1))
                    current_entity = None
                    start_idx = None
            else:  # O tag
                if current_entity is not None:
                    entities.add((current_entity, start_idx, i - 1))
                current_entity = None
                start_idx = None
        
        # Handle entity at end of sequence
        if current_entity is not None:
            entities.add((current_entity, start_idx, len(tags) - 1))
        
        return entities
    
    all_pred_entities = []
    all_true_entities = []
    
    for pred_tags, true_tags in zip(predictions, ground_truth):
        pred_entities = extract_entities(pred_tags, id2label)
        true_entities = extract_entities(true_tags, id2label)
        
        all_pred_entities.append(pred_entities)
        all_true_entities.append(true_entities)
    
    # Compute overall metrics
    total_tp = sum(len(p & t) for p, t in zip(all_pred_entities, all_true_entities))
    total_fp = sum(len(p - t) for p, t in zip(all_pred_entities, all_true_entities))
    total_fn = sum(len(t - p) for p, t in zip(all_pred_entities, all_true_entities))
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_span_overlap_metrics(pred_starts: torch.Tensor, pred_ends: torch.Tensor,
                                  true_starts: torch.Tensor, true_ends: torch.Tensor) -> Dict[str, float]:
    """
    Compute span overlap metrics for QA without text normalization.
    
    Args:
        pred_starts: Predicted start positions
        pred_ends: Predicted end positions
        true_starts: Gold start positions
        true_ends: Gold end positions
    
    Returns:
        Dictionary with exact match and partial match rates
    """
    exact_matches = (pred_starts == true_starts) & (pred_ends == true_ends)
    
    # Partial match: either start or end matches
    partial_matches = (pred_starts == true_starts) | (pred_ends == true_ends)
    
    return {
        'exact_span_match': exact_matches.float().mean().item(),
        'partial_span_match': partial_matches.float().mean().item()
    }


class Evaluator:
    """Unified evaluator for multiple tasks."""
    
    def __init__(self, task_type: str = 'qa', id2label: Dict[int, str] = None):
        self.task_type = task_type
        self.id2label = id2label
    
    def evaluate(self, predictions, ground_truth) -> Dict[str, float]:
        """Evaluate predictions against ground truth."""
        if self.task_type == 'qa':
            return compute_qa_metrics(predictions, ground_truth)
        elif self.task_type == 'ner':
            return compute_ner_metrics(predictions, ground_truth, self.id2label)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

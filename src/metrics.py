"""
Evaluation metrics for QA and NER tasks.
Compliant with strict evaluation splits (validation/test only) and includes BERTScore + BLEU.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import re

# New imports for advanced metrics
import sacrebleu
from bert_score import score as bert_score_fn


def normalize_text(text: str) -> str:
    """Normalize text for EM and F1 comparison."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())


def compute_f1(pred: str, truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
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
    return 2 * precision * recall / (precision + recall)


def compute_qa_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """
    Compute EM, F1, BLEU, and BERTScore for Question Answering.
    
    Args:
        predictions: List of dicts with 'id', 'prediction_text'
        ground_truth: List of dicts with 'id', 'answers' (list of acceptable strings)
    
    Returns:
        Dictionary with comprehensive QA metrics
    """
    exact_matches = []
    f1_scores = []
    bleu_scores = []
    bert_f1_scores = []
    
    # --- 1. Prepare data for batched BERTScore and BLEU ---
    expanded_preds = []
    expanded_refs = []
    ref_counts = []
    all_true_answers_for_bleu = [] # Will be padded to max refs
    
    max_refs = max(len(truth['answers']) for truth in ground_truth)
    
    for pred, truth in zip(predictions, ground_truth):
        pred_text = pred['prediction_text']
        true_answers = truth['answers']
        
        # EM and F1 (max over all acceptable answers)
        em = max(1.0 if normalize_text(pred_text) == normalize_text(ans) else 0.0 for ans in true_answers)
        exact_matches.append(em)
        
        f1 = max(compute_f1(pred_text, ans) for ans in true_answers)
        f1_scores.append(f1)
        
        # Prepare for batched BERTScore
        expanded_preds.extend([pred_text] * len(true_answers))
        expanded_refs.extend(true_answers)
        ref_counts.append(len(true_answers))
        
        # Prepare for BLEU (pad with empty strings if fewer than max_refs)
        padded_refs = true_answers + [""] * (max_refs - len(true_answers))
        all_true_answers_for_bleu.append(padded_refs)

    # --- 2. Batched BERTScore Calculation (Highly Efficient) ---
    # bert_score_fn returns tensors of shape (num_pairs,)
    P, R, F1 = bert_score_fn(expanded_preds, expanded_refs, lang="en", verbose=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    F1_np = F1.cpu().numpy()
    
    idx = 0
    for count in ref_counts:
        # Take the max BERTScore F1 across all valid references for this example
        max_bert_f1 = float(F1_np[idx:idx+count].max())
        bert_f1_scores.append(max_bert_f1)
        idx += count

    # --- 3. Corpus BLEU Calculation (via sacrebleu) ---
    # sacrebleu expects references as a list of lists: [ [ref1_for_all], [ref2_for_all], ... ]
    sacrebleu_refs = [[all_true_answers_for_bleu[i][j] for i in range(len(ground_truth))] for j in range(max_refs)]
    hypotheses = [pred['prediction_text'] for pred in predictions]
    
    # sacrebleu handles tokenization and smoothing automatically
    bleu_result = sacrebleu.corpus_bleu(hypotheses, sacrebleu_refs)
    bleu_scores.append(bleu_result.score / 100.0) # Normalize to 0.0-1.0
    
    return {
        'exact_match': float(np.mean(exact_matches)),
        'f1': float(np.mean(f1_scores)),
        'bleu': float(np.mean(bleu_scores)),
        'bertscore_f1': float(np.mean(bert_f1_scores))
    }


def compute_ner_metrics(predictions: List[List[int]], ground_truth: List[List[int]], 
                        id2label: Dict[int, str]) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for Named Entity Recognition (BIO scheme).
    """
    def extract_entities(tags: List[int], id2label_map: Dict[int, str]) -> set:
        entities = set()
        current_entity = None
        start_idx = None
        
        for i, tag_id in enumerate(tags):
            if tag_id == -100:  # Ignore padding/ignore index
                continue
            
            tag = id2label_map[tag_id]
            
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
            else:  # 'O' tag
                if current_entity is not None:
                    entities.add((current_entity, start_idx, i - 1))
                current_entity = None
                start_idx = None
        
        if current_entity is not None:
            entities.add((current_entity, start_idx, len(tags) - 1))
        
        return entities
    
    all_pred_entities = [extract_entities(p, id2label) for p in predictions]
    all_true_entities = [extract_entities(t, id2label) for t in ground_truth]
    
    total_tp = sum(len(p & t) for p, t in zip(all_pred_entities, all_true_entities))
    total_fp = sum(len(p - t) for p, t in zip(all_pred_entities, all_true_entities))
    total_fn = sum(len(t - p) for p, t in zip(all_pred_entities, all_true_entities))
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_span_overlap_metrics(pred_starts: torch.Tensor, pred_ends: torch.Tensor,
                                  true_starts: torch.Tensor, true_ends: torch.Tensor) -> Dict[str, float]:
    """Compute span overlap metrics for QA at the token index level."""
    exact_matches = (pred_starts == true_starts) & (pred_ends == true_ends)
    partial_matches = (pred_starts == true_starts) | (pred_ends == true_ends)
    
    return {
        'exact_span_match': exact_matches.float().mean().item(),
        'partial_span_match': partial_matches.float().mean().item()
    }


class Evaluator:
    """Unified evaluator for multiple tasks."""
    
    def __init__(self, task_type: str = 'qa', id2label: Optional[Dict[int, str]] = None):
        self.task_type = task_type
        self.id2label = id2label or {}
    
    def evaluate(self, predictions, ground_truth) -> Dict[str, float]:
        if self.task_type == 'qa':
            return compute_qa_metrics(predictions, ground_truth)
        elif self.task_type == 'ner':
            return compute_ner_metrics(predictions, ground_truth, self.id2label)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


# ==============================================================================
# STRICT SPLIT EVALUATION PIPELINE
# This guarantees you ONLY evaluate on validation/test data, never training data.
# ==============================================================================

@torch.no_grad()
def evaluate_on_split(model, tokenizer, dataset, split_name: str = "validation", 
                      task_type: str = "qa", id2label: Optional[Dict[int, str]] = None,
                      batch_size: int = 16, device: str = "cuda") -> Dict[str, float]:
    """
    Evaluates the model strictly on a specified dataset split (e.g., 'validation' or 'test').
    
    Args:
        model: The trained SpanBERT model (SpanBERTForQA or SpanBERTForNER)
        tokenizer: The corresponding tokenizer
        dataset: Hugging Face DatasetDict (must contain the split_name key)
        split_name: The split to evaluate on (e.g., "validation", "test"). NEVER "train".
        task_type: "qa" or "ner"
        id2label: Required for NER to map IDs to BIO tags.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    if split_name == "train":
        raise ValueError("DO NOT evaluate on the 'train' split. Use 'validation' or 'test' to prevent data leakage.")
    
    if split_name not in dataset:
        raise ValueError(f"Split '{split_name}' not found in dataset. Available: {list(dataset.keys())}")
    
    eval_dataset = dataset[split_name]
    model.to(device)
    model.eval()
    evaluator = Evaluator(task_type=task_type, id2label=id2label)
    
    predictions = []
    ground_truth = []
    
    # Simple batching loop (replace with Trainer/HF evaluate if preferred)
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:i+batch_size]
        
        inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(device),
            "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(device)
            
        # Forward pass
        outputs = model(**inputs)
        
        if task_type == "qa":
            start_logits, end_logits = outputs[0], outputs[1]
            pred_starts = start_logits.argmax(dim=-1).cpu().tolist()
            pred_ends = end_logits.argmax(dim=-1).cpu().tolist()
            
            for j in range(len(batch["id"])):
                # Reconstruct text from tokens (simplified; adapt to your tokenizer's decode)
                pred_text = tokenizer.decode(inputs["input_ids"][j][pred_starts[j]:pred_ends[j]+1]).strip()
                predictions.append({"id": batch["id"][j], "prediction_text": pred_text})
                ground_truth.append({"id": batch["id"][j], "answers": batch["answers"][j]["text"]})
                
        elif task_type == "ner":
            # If model uses CRF, outputs[1] is already decoded tags. Otherwise, it's logits.
            if hasattr(model, 'use_crf') and model.use_crf:
                pred_tags = outputs[1].cpu().tolist()
            else:
                pred_tags = outputs[1].argmax(dim=-1).cpu().tolist()
                
            true_labels = batch["labels"]
            
            for j in range(len(pred_tags)):
                # Filter out padding (-100) for clean metric calculation
                valid_mask = [l != -100 for l in true_labels[j]]
                clean_pred = [p for p, m in zip(pred_tags[j], valid_mask) if m]
                clean_true = [t for t, m in zip(true_labels[j], valid_mask) if m]
                
                predictions.append(clean_pred)
                ground_truth.append(clean_true)
    
    # Calculate and return metrics
    metrics = evaluator.evaluate(predictions, ground_truth)
    print(f"\n--- Evaluation Results on '{split_name}' split ---")
    for k, v in metrics.items():
        print(f"{k.upper():<15}: {v:.4f}")
        
    return metrics
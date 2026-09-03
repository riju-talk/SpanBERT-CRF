"""
Evaluation metrics for QA and NER tasks.
Compliant with strict evaluation splits (validation/test only).
Includes Exact Match, F1, BLEU, BERTScore, and Span Overlap metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import re
import warnings

# Import sacrebleu for robust, standardized BLEU calculation
import sacrebleu

# Try to import bert_score with error handling
try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    warnings.warn("bert_score not available. Install with: pip install bert-score")


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


def compute_qa_metrics(predictions: List[Dict], ground_truth: List[Dict], 
                       debug: bool = False) -> Dict[str, float]:
    """
    Compute EM, F1, BLEU, and BERTScore for Question Answering.
    """
    exact_matches = []
    f1_scores = []
    bleu_scores = []
    bert_f1_scores = []
    
    # --- 1. Prepare data for BLEU and BERTScore ---
    hypotheses = []
    all_references = []
    
    for idx, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        pred_text = pred['prediction_text']
        true_answers = truth['answers']
        
        # EM and F1 (max over all acceptable answers)
        em = max(1.0 if normalize_text(pred_text) == normalize_text(ans) else 0.0 for ans in true_answers)
        exact_matches.append(em)
        
        f1 = max(compute_f1(pred_text, ans) for ans in true_answers)
        f1_scores.append(f1)
        
        hypotheses.append(pred_text)
        all_references.append(true_answers)
        
        if debug and idx < 3:
            print(f"\n[Example {idx}]")
            print(f"  Prediction: '{pred_text}'")
            print(f"  Gold: {true_answers}")
            print(f"  EM: {em}, F1: {f1:.4f}")

    # --- 2. Corpus BLEU Calculation ---
    try:
        max_refs = max(len(refs) for refs in all_references)
        padded_references = [refs + [""] * (max_refs - len(refs)) for refs in all_references]
        sacrebleu_refs = [[padded_references[i][j] for i in range(len(padded_references))] for j in range(max_refs)]
        
        bleu_result = sacrebleu.corpus_bleu(hypotheses, sacrebleu_refs)
        bleu_score = bleu_result.score / 100.0
    except Exception as e:
        warnings.warn(f"BLEU calculation failed: {e}")
        bleu_score = 0.0
    
    bleu_scores = [bleu_score]  # Corpus-level BLEU
    
    # --- 3. BERTScore Calculation (with robust error handling) ---
    if BERT_SCORE_AVAILABLE:
        try:
            # Flatten for BERTScore
            expanded_preds = []
            expanded_refs = []
            ref_counts = []
            
            for pred, truth in zip(predictions, ground_truth):
                pred_text = pred['prediction_text']
                true_answers = truth['answers']
                expanded_preds.extend([pred_text] * len(true_answers))
                expanded_refs.extend(true_answers)
                ref_counts.append(len(true_answers))
            
            # Use a simpler model to avoid compatibility issues
            P, R, F1 = bert_score_fn(
                expanded_preds, 
                expanded_refs, 
                lang="en", 
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                model_type="roberta-base"  # Use base model for compatibility
            )
            F1_np = F1.cpu().numpy()
            
            # Aggregate per example
            idx = 0
            for count in ref_counts:
                max_bert_f1 = float(F1_np[idx:idx+count].max())
                bert_f1_scores.append(max_bert_f1)
                idx += count
                
        except Exception as e:
            warnings.warn(f"BERTScore failed: {type(e).__name__}: {str(e)[:100]}")
            bert_f1_scores = [0.0] * len(predictions)
    else:
        bert_f1_scores = [0.0] * len(predictions)
    
    return {
        'exact_match': float(np.mean(exact_matches)),
        'f1': float(np.mean(f1_scores)),
        'bleu': float(np.mean(bleu_scores)),
        'bertscore_f1': float(np.mean(bert_f1_scores)) if bert_f1_scores else 0.0
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
            if tag_id == -100:
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
            else:
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
    
    def evaluate(self, predictions, ground_truth, debug: bool = False) -> Dict[str, float]:
        if self.task_type == 'qa':
            return compute_qa_metrics(predictions, ground_truth, debug=debug)
        elif self.task_type == 'ner':
            return compute_ner_metrics(predictions, ground_truth, self.id2label)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


@torch.no_grad()
def evaluate_on_split(model, tokenizer, dataset, split_name: str = "validation", 
                      task_type: str = "qa", id2label: Optional[Dict[int, str]] = None,
                      batch_size: int = 16, device: str = "cuda", 
                      debug: bool = False) -> Dict[str, float]:
    """
    Evaluates the model strictly on a specified dataset split.
    Set debug=True to see first 3 examples.
    """
    if split_name == "train":
        raise ValueError("DO NOT evaluate on 'train' split!")
    
    if split_name not in dataset:
        raise ValueError(f"Split '{split_name}' not found. Available: {list(dataset.keys())}")
    
    eval_dataset = dataset[split_name]
    model.to(device)
    model.eval()
    evaluator = Evaluator(task_type=task_type, id2label=id2label)
    
    predictions = []
    ground_truth = []
    
    print(f"Processing {len(eval_dataset)} examples...")
    
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:i+batch_size]
        
        inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(device),
            "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(device)
            
        outputs = model(**inputs)
        
        if task_type == "qa":
            if len(outputs) == 3:
                _, start_logits, end_logits = outputs
            else:
                start_logits, end_logits = outputs[0], outputs[1]
                
            pred_starts = start_logits.argmax(dim=-1).cpu().tolist()
            pred_ends = end_logits.argmax(dim=-1).cpu().tolist()
            
            for j in range(len(batch["id"])):
                start_idx = min(pred_starts[j], pred_ends[j])
                end_idx = max(pred_starts[j], pred_ends[j])
                
                pred_text = tokenizer.decode(
                    inputs["input_ids"][j][start_idx:end_idx+1], 
                    skip_special_tokens=True
                ).strip()
                
                predictions.append({"id": str(batch["id"][j]), "prediction_text": pred_text})
                
                # Handle different answer formats
                raw_answers = batch.get("answers", None)
                if isinstance(raw_answers, dict):
                    # SQuAD format: {"text": [...], "answer_start": [...]}
                    answers = raw_answers.get("text", [""])
                    if isinstance(answers, list):
                        answers = answers[j] if j < len(answers) else [""]
                elif isinstance(raw_answers, list):
                    answers = raw_answers[j] if j < len(raw_answers) else [""]
                else:
                    answers = [batch.get("answer_text", "")]
                
                # Ensure answers is a list
                if isinstance(answers, str):
                    answers = [answers]
                if not answers:
                    answers = [""]
                    
                ground_truth.append({"id": str(batch["id"][j]), "answers": answers})
                
        elif task_type == "ner":
            if hasattr(model, 'use_crf') and model.use_crf:
                pred_tags = outputs[1].cpu().tolist()
            else:
                pred_tags = outputs[1].argmax(dim=-1).cpu().tolist()
                
            true_labels = batch["labels"]
            
            for j in range(len(pred_tags)):
                valid_mask = [l != -100 for l in true_labels[j]]
                clean_pred = [p for p, m in zip(pred_tags[j], valid_mask) if m]
                clean_true = [t for t, m in zip(true_labels[j], valid_mask) if m]
                
                predictions.append(clean_pred)
                ground_truth.append(clean_true)
    
    print(f"Evaluated {len(predictions)} examples")
    
    # Calculate and return metrics
    metrics = evaluator.evaluate(predictions, ground_truth, debug=debug)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results on '{split_name}' split ({len(predictions)} examples)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"{k.upper():<20}: {v:.4f}")
    print(f"{'='*60}\n")
        
    return metrics


@torch.no_grad()
def debug_evaluate_qa(model, tokenizer, qa_val_loader, device, max_batches: int = 1):
    """
    DEBUG FUNCTION: Inspect first few examples to find metric calculation bugs.
    
    Use this to debug why you're getting perfect EM/F1 but 0 BLEU and 50% span match.
    """
    print("="*70)
    print("DEBUG MODE: Inspecting predictions vs ground truth")
    print("="*70)
    
    qa_preds = []
    qa_golds = []
    pred_starts_list = []
    pred_ends_list = []
    true_starts_list = []
    true_ends_list = []
    
    model.eval()
    
    for batch_idx, batch in enumerate(qa_val_loader):
        if batch_idx >= max_batches:
            break
        
        # Move tensors to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Print batch structure on first batch
        if batch_idx == 0:
            print(f"\n📊 Batch Structure:")
            print(f"  Keys: {list(batch.keys())}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            if "answers" in batch:
                print(f"  Answers type: {type(batch['answers'])}")
                if isinstance(batch['answers'], dict):
                    print(f"  Answers keys: {list(batch['answers'].keys())}")
                    if "text" in batch['answers']:
                        print(f"  Sample answer text: {batch['answers']['text'][:2]}")
                elif isinstance(batch['answers'], list):
                    print(f"  Sample answer: {batch['answers'][0] if len(batch['answers']) > 0 else 'EMPTY'}")
        
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        
        if len(outputs) == 3:
            _, start_logits, end_logits = outputs
        else:
            start_logits, end_logits = outputs[0], outputs[1]
        
        s_idx = start_logits.argmax(dim=-1)
        e_idx = end_logits.argmax(dim=-1)
        
        # Process each example in batch
        for i in range(min(5, batch["input_ids"].size(0))):  # First 5 examples
            start_pos, end_pos = s_idx[i].item(), e_idx[i].item()
            if end_pos < start_pos:
                end_pos = start_pos
            
            # Decode prediction
            pred_text = tokenizer.decode(
                batch["input_ids"][i][start_pos:end_pos+1],
                skip_special_tokens=True
            ).strip()
            
            # Extract ground truth - handle multiple formats
            raw_ans = batch.get("answers", None)
            if isinstance(raw_ans, dict):
                # SQuAD format
                ans_list = raw_ans.get("text", [])
                if isinstance(ans_list, list) and len(ans_list) > i:
                    answers = ans_list[i]
                    if not isinstance(answers, list):
                        answers = [answers]
                else:
                    answers = [""]
            elif isinstance(raw_ans, list):
                answers = raw_ans[i] if i < len(raw_ans) else [""]
                if isinstance(answers, str):
                    answers = [answers]
            else:
                # Try answer_text
                ans_field = batch.get("answer_text", "")
                if isinstance(ans_field, list) and len(ans_field) > i:
                    answers = [ans_field[i]] if isinstance(ans_field[i], str) else ans_field[i]
                else:
                    answers = [""]
            
            if not answers or answers == [""]:
                answers = ["NO_ANSWER"]
            
            # Calculate metrics manually
            em = max(1.0 if normalize_text(pred_text) == normalize_text(ans) else 0.0 for ans in answers)
            f1 = max(compute_f1(pred_text, ans) for ans in answers)
            
            # Print detailed info
            print(f"\n[Example {batch_idx * batch['input_ids'].size(0) + i}]")
            print(f"  Pred span: [{start_pos}:{end_pos}]")
            print(f"  Pred text: '{pred_text}'")
            print(f"  Gold answers: {answers}")
            if "start_positions" in batch:
                print(f"  Gold span: [{batch['start_positions'][i].item()}:{batch['end_positions'][i].item()}]")
            print(f"  → EM: {em}, F1: {f1:.4f}")
            
            # Check for issues
            if em == 1.0 and pred_text.strip() == "":
                print("  ⚠️  WARNING: Empty prediction got EM=1.0!")
            if answers == ["NO_ANSWER"]:
                print("  ⚠️  WARNING: Could not extract ground truth!")
            
            qa_preds.append({"id": str(len(qa_preds)), "prediction_text": pred_text})
            qa_golds.append({"id": str(len(qa_golds)), "answers": answers})
            pred_starts_list.append(s_idx[i].cpu())
            pred_ends_list.append(e_idx[i].cpu())
            if "start_positions" in batch:
                true_starts_list.append(batch["start_positions"][i].cpu())
                true_ends_list.append(batch["end_positions"][i].cpu())
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    # Check for empty answers
    empty_answers = sum(1 for g in qa_golds if g['answers'] == [""] or g['answers'] == ["NO_ANSWER"])
    print(f"Examples with missing answers: {empty_answers}/{len(qa_golds)}")
    
    # Check span match if we have true positions
    if len(true_starts_list) > 0:
        pred_starts_t = torch.stack(pred_starts_list)
        pred_ends_t = torch.stack(pred_ends_list)
        true_starts_t = torch.stack(true_starts_list)
        true_ends_t = torch.stack(true_ends_list)
        
        exact_span = (pred_starts_t == true_starts_t) & (pred_ends_t == true_ends_t)
        print(f"Exact span match: {exact_span.float().mean().item():.4f}")
        
        # Check if EM matches span match
        em_scores = [1.0 if normalize_text(qa_preds[i]['prediction_text']) == normalize_text(qa_golds[i]['answers'][0]) else 0.0 
                     for i in range(len(qa_preds))]
        print(f"Text EM: {np.mean(em_scores):.4f}")
        
        if exact_span.float().mean().item() < 0.5 and np.mean(em_scores) > 0.9:
            print("\n🚨 CRITICAL ISSUE DETECTED:")
            print("   Perfect EM but poor span match suggests:")
            print("   1. Ground truth extraction is wrong (comparing to empty/wrong answers)")
            print("   2. Or your DataLoader has misaligned data")
    
    print("="*70)
    
    return qa_preds, qa_golds
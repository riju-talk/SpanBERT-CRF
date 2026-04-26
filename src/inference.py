"""
Inference utilities for QA and NER tasks.
"""

import torch
from transformers import SpanBertTokenizer
from typing import List, Dict, Tuple, Optional


class QAInference:
    """Question Answering inference with SpanBERT-CRF."""
    
    def __init__(self, model, tokenizer: SpanBertTokenizer, device: str = 'cuda',
                 max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.to(device)
        self.model.eval()
    
    def predict(self, context: str, question: str) -> Dict[str, any]:
        """
        Predict answer for a given question and context.
        
        Args:
            context: The context passage
            question: The question to answer
        
        Returns:
            Dictionary with predicted answer, confidence, and span positions
        """
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            start_logits = outputs[0]
            end_logits = outputs[1] if len(outputs) > 1 else outputs[0]
        
        # Find best span
        start_scores = torch.softmax(start_logits, dim=-1)
        end_scores = torch.softmax(end_logits, dim=-1)
        
        # Get top-k spans
        batch_size = input_ids.size(0)
        best_answer = None
        best_score = 0
        best_start = 0
        best_end = 0
        
        for i in range(batch_size):
            for start_idx in range(len(start_scores[i])):
                for end_idx in range(start_idx, min(start_idx + 50, len(end_scores[i]))):
                    score = start_scores[i][start_idx] * end_scores[i][end_idx]
                    if score > best_score:
                        best_score = score
                        best_start = start_idx
                        best_end = end_idx
        
        # Decode answer
        answer_tokens = input_ids[0][best_start:best_end+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Convert token positions to character positions
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        char_start = encoding.token_to_chars(best_start)[0] if encoding.token_to_chars(best_start) else None
        char_end = encoding.token_to_chars(best_end)[1] if encoding.token_to_chars(best_end) else None
        
        return {
            'answer': answer,
            'confidence': float(best_score),
            'start_token': int(best_start),
            'end_token': int(best_end),
            'start_char': char_start,
            'end_char': char_end
        }
    
    def predict_batch(self, contexts: List[str], questions: List[str]) -> List[Dict]:
        """Predict answers for multiple question-context pairs."""
        results = []
        for context, question in zip(contexts, questions):
            result = self.predict(context, question)
            results.append(result)
        return results


class NERInference:
    """Named Entity Recognition inference with SpanBERT-CRF."""
    
    def __init__(self, model, tokenizer: SpanBertTokenizer, 
                 id2label: Dict[int, str], device: str = 'cuda',
                 max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = device
        self.max_length = max_length
        self.model.to(device)
        self.model.eval()
    
    def predict(self, text: str) -> List[Dict[str, any]]:
        """
        Predict named entities in text.
        
        Args:
            text: Input text
        
        Returns:
            List of entities with type, text, and positions
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Truncate if needed
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length-2]
        
        # Add special tokens
        input_ids = [self.tokenizer.cls_token_id] + \
                    self.tokenizer.convert_tokens_to_ids(tokens) + \
                    [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        # Pad
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if isinstance(outputs, tuple):
                predictions = outputs[-1]  # Decoded tags from CRF
            else:
                predictions = torch.argmax(outputs, dim=-1)
        
        # Extract entities
        entities = []
        current_entity = None
        start_idx = None
        
        for i, pred_id in enumerate(predictions[0]):
            if i == 0 or i == len(predictions[0]) - 1:  # Skip CLS and SEP
                continue
            
            tag = self.id2label[pred_id.item()]
            
            if tag.startswith('B-'):
                if current_entity is not None:
                    entities.append(current_entity)
                entity_type = tag[2:]
                current_entity = {
                    'type': entity_type,
                    'tokens': [tokens[i-1]],
                    'start_token': i-1,
                    'end_token': i-1
                }
                start_idx = i-1
            elif tag.startswith('I-') and current_entity is not None:
                entity_type = tag[2:]
                if entity_type == current_entity['type']:
                    current_entity['tokens'].append(tokens[i-1])
                    current_entity['end_token'] = i-1
                else:
                    entities.append(current_entity)
                    current_entity = None
            else:  # O tag
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity is not None:
            entities.append(current_entity)
        
        # Convert to text
        for entity in entities:
            entity['text'] = self.tokenizer.convert_tokens_to_string(entity['tokens'])
            del entity['tokens']
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Predict entities for multiple texts."""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results


def load_inference_model(model_path: str, task_type: str = 'qa', 
                         use_crf: bool = False, device: str = 'cuda'):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to saved model checkpoint
        task_type: 'qa' or 'ner'
        use_crf: Whether model uses CRF layer
        device: Device to run inference on
    
    Returns:
        Inference object (QAInference or NERInference)
    """
    from src.models import SpanBERTForQA, SpanBERTForNER
    
    tokenizer = SpanBertTokenizer.from_pretrained('spanbert-base-cased')
    
    if task_type == 'qa':
        model = SpanBERTForQA.from_pretrained('spanbert-base-cased', use_crf=use_crf)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return QAInference(model, tokenizer, device)
    
    elif task_type == 'ner':
        id2label = {
            0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
        }
        model = SpanBERTForNER.from_pretrained('spanbert-base-cased', 
                                               num_ner_tags=9, use_crf=use_crf)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return NERInference(model, tokenizer, id2label, device)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")

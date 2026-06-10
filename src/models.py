"""
SpanBERT-CRF: Unified NLP Architecture for Question Answering and Named Entity Recognition
===========================================================================================
Minimal, robust implementation using AutoModel and a bug-free CRF layer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, Tuple


class CRF(nn.Module):
    """Minimal, robust Conditional Random Field layer for sequence tagging."""
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition matrices
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        # Initialize transitions to avoid extreme initial values
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood for CRF.
        Args:
            emissions: (batch_size, seq_len, num_tags)
            tags: (batch_size, seq_len)
            mask: (batch_size, seq_len)
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)  # (seq_len, batch_size, num_tags)
            tags = tags.transpose(0, 1)            # (seq_len, batch_size)
            mask = mask.transpose(0, 1)            # (seq_len, batch_size)

        seq_len, batch_size, num_tags = emissions.shape
        mask_float = mask.float()

        # --- 1. Numerator: Score of the gold (true) path ---
        emit_score = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # (seq_len, batch_size)
        trans_score = self.transitions[tags[:-1], tags[1:]]             # (seq_len-1, batch_size)
        
        emit_score = emit_score * mask_float
        trans_score = trans_score * mask_float[1:]
        
        start_score = self.start_transitions[tags[0]] * mask_float[0]
        
        # End score: find the last valid tag for each sequence in the batch
        last_tag_indices = mask.sum(dim=0).long() - 1  # (batch_size,)
        end_score = torch.zeros(batch_size, device=emissions.device)
        valid = last_tag_indices >= 0
        if valid.any():
            batch_indices = torch.arange(batch_size, device=emissions.device)[valid]
            valid_last_tags = tags[last_tag_indices[valid], batch_indices]
            end_score[valid] = self.end_transitions[valid_last_tags]

        numerator = start_score + emit_score.sum(dim=0) + trans_score.sum(dim=0) + end_score

        # --- 2. Denominator: Log partition function (Forward Algorithm) ---
        alpha = self.start_transitions + emissions[0]  # (batch_size, num_tags)
        
        for i in range(1, seq_len):
            score = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B, T, T)
            next_alpha = torch.logsumexp(score, dim=1) + emissions[i]   # (B, T)
            
            # Elegant masking: if step `i` is padded, keep the previous `alpha`.
            # This naturally ensures `end_transitions` are applied at the last valid step.
            alpha = torch.where(mask[i].unsqueeze(1).bool(), next_alpha, alpha)
        
        denominator = torch.logsumexp(alpha + self.end_transitions, dim=1)

        return -(numerator - denominator).mean()
    
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi decoding to find the best tag sequence."""
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_len, batch_size, num_tags = emissions.shape
        
        # Initialize
        score = self.start_transitions + emissions[0]
        history = []
        
        # Viterbi forward pass
        for i in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_prev_score, best_prev_tag = next_score.max(dim=1)
            score = best_prev_score + emissions[i]
            
            # Masking: keep previous score/tag if current step is padded
            score = torch.where(mask[i].unsqueeze(1).bool(), score, score) 
            history.append(best_prev_tag)
        
        # Add end transitions and find best final tag
        score = score + self.end_transitions
        best_last_tags = score.argmax(dim=1)  # (batch_size,)
        
        # Backtrack
        best_tags_list = []
        for idx in range(batch_size):
            best_tags = [best_last_tags[idx].item()]
            for hist in reversed(history):
                best_tags.append(hist[idx][best_tags[-1]].item())
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return torch.tensor(best_tags_list, device=emissions.device)


def apply_lora_to_model(model, use_lora: bool = False, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1):
    """Apply LoRA adapters to the SpanBERT encoder."""
    if not use_lora:
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value", "output.dense"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # Apply LoRA specifically to the base spanbert model
    model.spanbert = get_peft_model(model.spanbert, lora_config)
    model.spanbert.print_trainable_parameters()
    return model


class SpanBERTForQA(nn.Module):
    """SpanBERT model for Question Answering with robust dual-CRF enhancement."""
    
    def __init__(self, use_crf: bool = False, model_name: str = "SpanBERT/spanbert-base-cased"):
        super().__init__()
        self.use_crf = use_crf
        
        # Load base model exactly as requested
        self.spanbert = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        config = self.spanbert.config
        
        # QA heads for start and end positions (2 classes: 0=not position, 1=is position)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Dual CRF layers for start and end (treated as independent sequence labeling tasks)
        if use_crf:
            self.crf_start = CRF(num_tags=2, batch_first=True)
            self.crf_end = CRF(num_tags=2, batch_first=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        
        outputs = self.spanbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 2)
        
        # Split into start and end logits
        start_logits = logits[..., 0].contiguous()
        end_logits = logits[..., 1].contiguous()
        
        loss = None
        if start_positions is not None and end_positions is not None:
            seq_len = input_ids.shape[1]
            
            if self.use_crf:
                # Format emissions: (batch_size, seq_len, 2) -> [prob of 0, prob of 1]
                emissions_start = torch.stack([torch.zeros_like(start_logits), start_logits], dim=-1)
                emissions_end = torch.stack([torch.zeros_like(end_logits), end_logits], dim=-1)
                
                # Create tag sequences: (batch_size, seq_len) all zeros, then set 1 at target position
                batch_size = start_positions.shape[0]
                tags_start = torch.zeros(batch_size, seq_len, dtype=torch.long, device=start_positions.device)
                tags_end = torch.zeros(batch_size, seq_len, dtype=torch.long, device=end_positions.device)
                
                # Clamp positions to valid range to prevent out-of-bounds errors
                start_positions = start_positions.clamp(0, seq_len - 1)
                end_positions = end_positions.clamp(0, seq_len - 1)
                
                tags_start.scatter_(1, start_positions.unsqueeze(1), 1)
                tags_end.scatter_(1, end_positions.unsqueeze(1), 1)
                
                loss_start = self.crf_start(emissions_start, tags_start, attention_mask)
                loss_end = self.crf_end(emissions_end, tags_end, attention_mask)
                loss = (loss_start + loss_end) / 2.0
            else:
                # Standard cross-entropy loss fallback
                ignored_index = seq_len
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2.0
        
        return (loss, start_logits, end_logits) if loss is not None else (start_logits, end_logits)


class SpanBERTForNER(nn.Module):
    """SpanBERT model for Named Entity Recognition with CRF."""
    
    def __init__(self, num_ner_tags: int, use_crf: bool = True, model_name: str = "SpanBERT/spanbert-base-cased"):
        super().__init__()
        self.num_labels = num_ner_tags
        self.use_crf = use_crf
        
        # Load base model exactly as requested
        self.spanbert = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        config = self.spanbert.config
        
        self.classifier = nn.Linear(config.hidden_size, num_ner_tags)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if use_crf:
            self.crf = CRF(num_tags=num_ner_tags, batch_first=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        
        outputs = self.spanbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            if self.use_crf:
                loss = self.crf(logits, labels, attention_mask)
            else:
                loss_fct = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
        
        if self.use_crf and labels is None:
            tags = self.crf.decode(logits, attention_mask)
            return (loss, tags) if loss is not None else (tags,)
        
        return (loss, logits) if loss is not None else (logits,)
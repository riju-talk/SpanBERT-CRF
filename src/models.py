"""
SpanBERT-CRF: Unified NLP Architecture for Question Answering and Named Entity Recognition
===========================================================================================

This module provides the core model architecture combining SpanBERT with Conditional Random Fields (CRF)
for improved span-based predictions in QA and NER tasks.
"""

import torch
import torch.nn as nn
from transformers import SpanBertModel, SpanBertPreTrainedModel
from typing import Optional, Tuple


class CRF(nn.Module):
    """Conditional Random Field layer for span boundary prediction."""
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition matrix: transitions[i][j] = score from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def _compute_emission_score(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """Compute emission scores for given tags."""
        return emissions.gather(2, tags.unsqueeze(2)).squeeze(2)
    
    def _compute_transition_score(self, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute transition scores."""
        batch_size, seq_len = tags.shape
        
        # Add start and end tags
        start_score = self.start_transitions[tags[:, 0]]
        
        # Transition scores between tags
        trans_score = self.transitions[tags[:, :-1], tags[:, 1:]]
        trans_score = trans_score * mask[:, :-1].unsqueeze(-1)
        
        # End transitions
        last_tag_indices = mask.long().sum(dim=1) - 1
        end_score = self.end_transitions[tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)]
        
        return start_score + trans_score.sum(dim=1) + end_score
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute negative log-likelihood for CRF.
        
        Args:
            emissions: Tensor of shape (batch_size, seq_len, num_tags)
            tags: Tensor of shape (batch_size, seq_len)
            mask: Binary tensor of shape (batch_size, seq_len)
        
        Returns:
            Negative log-likelihood loss
        """
        if mask is None:
            mask = torch.ones_like(tags).float()
        
        # Numerator: score of the gold sequence
        numerator = self._compute_emission_score(emissions, tags) + \
                    self._compute_transition_score(tags, mask)
        
        # Denominator: sum of scores of all possible sequences (using forward algorithm)
        denominator = self._compute_log_partition_function(emissions, mask)
        
        return -(numerator - denominator).mean()
    
    def _compute_log_partition_function(self, emissions: torch.Tensor, 
                                        mask: torch.Tensor) -> torch.Tensor:
        """Compute log partition function using forward algorithm."""
        batch_size, seq_len, _ = emissions.shape
        
        # Initialize alpha with start transitions + first emissions
        alpha = emissions[:, 0, :] + self.start_transitions
        
        # Forward pass
        for i in range(1, seq_len):
            emit_scores = emissions[:, i, :].unsqueeze(2)
            trans_scores = self.transitions.unsqueeze(0)
            alpha = torch.logsumexp(alpha.unsqueeze(2) + trans_scores, dim=1) + emit_scores.squeeze(2)
            alpha = alpha * mask[:, i].unsqueeze(1) + alpha * (1 - mask[:, i].unsqueeze(1))
        
        # Add end transitions
        alpha = alpha + self.end_transitions
        
        return torch.logsumexp(alpha, dim=1)
    
    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Viterbi decoding to find the best tag sequence."""
        if mask is None:
            mask = torch.ones(emissions.shape[:2]).to(emissions.device)
        
        batch_size, seq_len, _ = emissions.shape
        
        # Initialize
        score = emissions[:, 0, :] + self.start_transitions
        history = []
        
        # Viterbi forward pass
        for i in range(1, seq_len):
            emit_scores = emissions[:, i, :].unsqueeze(1)
            trans_scores = self.transitions.unsqueeze(0)
            next_score = score.unsqueeze(2) + trans_scores
            
            # Find best previous tag
            best_prev_score, best_prev_tag = next_score.max(dim=1)
            score = best_prev_score + emit_scores.squeeze(1)
            history.append(best_prev_tag)
        
        # Add end transitions
        score = score + self.end_transitions
        
        # Backtrack
        best_tags_list = []
        for idx in range(batch_size):
            best_last_tag = score[idx].argmax()
            best_tags = [best_last_tag.item()]
            
            for hist in reversed(history):
                best_last_tag = hist[idx][best_last_tag]
                best_tags.append(best_last_tag.item())
            
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return torch.tensor(best_tags_list, device=emissions.device)


class SpanBERTForQA(SpanBertPreTrainedModel):
    """SpanBERT model for Question Answering with optional CRF enhancement."""
    
    def __init__(self, config, use_crf: bool = False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_crf = use_crf
        
        self.spanbert = SpanBertModel(config)
        
        # QA heads for start and end positions
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # CRF layer (optional)
        if use_crf:
            self.crf = CRF(num_tags=2, batch_first=True)  # 2 tags: start, end
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for QA task.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (segment IDs)
            start_positions: Gold start positions (for training)
            end_positions: Gold end positions (for training)
        
        Returns:
            Loss (if training), start_logits, end_logits
        """
        outputs = self.spanbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # Get start and end logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        loss = None
        if start_positions is not None and end_positions is not None:
            if self.use_crf:
                # Use CRF for joint start-end prediction
                emissions = torch.stack([start_logits, end_logits], dim=2)
                
                # Create tag sequences
                batch_size, seq_len = start_positions.shape
                tags = torch.zeros((batch_size, seq_len, 2), 
                                   dtype=torch.long, device=start_positions.device)
                
                for i in range(batch_size):
                    for j in range(seq_len):
                        if j == start_positions[i]:
                            tags[i, j, 0] = 1
                        if j == end_positions[i]:
                            tags[i, j, 1] = 1
                
                loss = self.crf(emissions, tags, attention_mask)
            else:
                # Standard cross-entropy loss
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
        
        return (loss, start_logits, end_logits) if loss is not None else (start_logits, end_logits)


class SpanBERTForNER(SpanBertPreTrainedModel):
    """SpanBERT model for Named Entity Recognition with CRF."""
    
    def __init__(self, config, num_ner_tags: int, use_crf: bool = True):
        super().__init__(config)
        self.num_labels = num_ner_tags
        self.use_crf = use_crf
        
        self.spanbert = SpanBertModel(config)
        
        # NER classification head
        self.classifier = nn.Linear(config.hidden_size, num_ner_tags)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # CRF layer
        if use_crf:
            self.crf = CRF(num_tags=num_ner_tags, batch_first=True)
        
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for NER task.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Gold NER tags (for training)
        
        Returns:
            Loss (if training), logits or decoded tags
        """
        outputs = self.spanbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
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
            # Decode using Viterbi
            tags = self.crf.decode(logits, attention_mask)
            return (loss, tags) if loss is not None else (tags,)
        
        return (loss, logits) if loss is not None else (logits,)

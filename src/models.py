"""
SpanBERT-CRF: Unified NLP Architecture for QA and NER
=====================================================
A CRF layer sits on top of SpanBERT for **both** tasks:

* NER  — CRF over the BIO entity tag set (``O``, ``B-*``, ``I-*``).
* QA   — the answer span is cast as a 3-tag BIO problem over the sequence
         (``O``, ``B-ANS``, ``I-ANS``); the CRF models the transitions
         between them and Viterbi decoding yields a contiguous span.

``SpanBERTForQA`` keeps a plain start/end head available (``use_crf=False``)
for ablations, but the default and intended configuration is ``use_crf=True``.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, Tuple

# BIO tag ids used by the QA-as-sequence-labelling formulation.
QA_TAG_O, QA_TAG_B, QA_TAG_I = 0, 1, 2
QA_NUM_TAGS = 3


class CRF(nn.Module):
    """Batch-first linear-chain Conditional Random Field for sequence tagging.

    Used by both task heads: NER over the entity tag set, QA over a 3-tag BIO
    span scheme. ``forward`` returns the mean negative log-likelihood;
    ``decode`` runs Viterbi and returns the best tag path.
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood."""
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_len, batch_size, num_tags = emissions.shape
        mask_float = mask.float()

        # --- Numerator: gold path score ---
        emit_score = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)
        emit_score = emit_score * mask_float
        
        # Transitions between consecutive tags
        trans_score = torch.zeros(seq_len - 1, batch_size, device=emissions.device)
        valid_trans = mask[1:].bool()
        if valid_trans.any():
            trans_score[valid_trans] = self.transitions[
                tags[:-1][valid_trans], tags[1:][valid_trans]
            ]
        
        start_score = self.start_transitions[tags[0]] * mask_float[0]
        
        # End transitions at last valid position
        seq_lengths = mask.sum(dim=0).long() - 1
        end_score = torch.zeros(batch_size, device=emissions.device)
        valid_end = seq_lengths >= 0
        if valid_end.any():
            last_tags = tags[seq_lengths[valid_end], valid_end.nonzero(as_tuple=True)[0]]
            end_score[valid_end] = self.end_transitions[last_tags]

        numerator = start_score + emit_score.sum(dim=0) + trans_score.sum(dim=0) + end_score

        # --- Denominator: log partition function (forward algorithm) ---
        alpha = self.start_transitions + emissions[0]  # (B, T)

        for i in range(1, seq_len):
            # alpha_prev (B, T, 1) + transitions (1, T, T) + emission_i (B, 1, T)
            inner = (alpha.unsqueeze(2)
                     + self.transitions.unsqueeze(0)
                     + emissions[i].unsqueeze(1))       # (B, T, T)
            new_alpha = torch.logsumexp(inner, dim=1)   # (B, T)
            mask_i = mask[i].unsqueeze(1).bool()        # (B, 1)
            alpha = torch.where(mask_i, new_alpha, alpha)

        denominator = torch.logsumexp(alpha + self.end_transitions, dim=1)  # (B,)
        return -(numerator - denominator).mean()
    
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi decoding."""
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_len, batch_size, num_tags = emissions.shape
        
        # Initialize
        score = self.start_transitions + emissions[0]  # (B, T)
        history = []
        
        # Forward
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # (B, T, 1)
            broadcast_emission = emissions[i].unsqueeze(1)  # (B, 1, T)
            next_score = broadcast_score + self.transitions + broadcast_emission  # (B, T, T)
            next_score, indices = next_score.max(dim=1)  # (B, T)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)
        
        # End transitions
        score = score + self.end_transitions
        best_last_tags = score.argmax(dim=1)  # (B,)
        
        # Backtrack
        best_tags = [best_last_tags]
        for hist in reversed(history):
            best_tags.append(hist.gather(1, best_tags[-1].unsqueeze(1)).squeeze(1))
        best_tags.reverse()
        
        return torch.stack(best_tags, dim=1)  # (B, seq_len)


def apply_lora_to_model(model, use_lora: bool = False, r: int = 8, 
                        lora_alpha: int = 32, lora_dropout: float = 0.1):
    """Apply LoRA to SpanBERT encoder."""
    if not use_lora:
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=lora_dropout,
        bias="none",
        # Keep every task head (and the CRF) fully trainable.
        modules_to_save=["qa_outputs", "qa_classifier", "classifier", "crf"],
    )
    
    model.spanbert = get_peft_model(model.spanbert, lora_config)
    model.spanbert.print_trainable_parameters()
    return model


def _spans_to_bio(start_positions: torch.Tensor, end_positions: torch.Tensor,
                  seq_len: int) -> torch.Tensor:
    """Turn (start, end) token indices into a ``(B, L)`` BIO tag tensor.

    ``start == 0`` marks an unanswerable question (the pointer sits on ``[CLS]``)
    and yields an all-``O`` row, which is exactly what SpanBERT should learn to
    predict for SQuAD v2.0 negatives.
    """
    device = start_positions.device
    batch_size = start_positions.shape[0]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, L)

    starts = start_positions.clamp(0, seq_len - 1).unsqueeze(1)     # (B, 1)
    ends = end_positions.clamp(0, seq_len - 1).unsqueeze(1)         # (B, 1)
    answerable = (start_positions > 0).unsqueeze(1)                 # (B, 1)

    bio = torch.full((batch_size, seq_len), QA_TAG_O, dtype=torch.long, device=device)
    inside = (positions >= starts) & (positions <= ends) & answerable
    bio = torch.where(inside, torch.full_like(bio, QA_TAG_I), bio)
    at_start = (positions == starts) & answerable
    bio = torch.where(at_start, torch.full_like(bio, QA_TAG_B), bio)
    return bio


def _bio_emissions_to_span_logits(emissions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project ``(B, L, 3)`` BIO emissions onto start/end logits.

    This keeps the CRF-QA model duck-type compatible with the plain start/end
    consumers in ``metrics.py`` and ``inference.py``: a token is a likely span
    *start* when ``B-ANS`` beats ``O``, and a likely span *end* when either
    ``B-ANS`` or ``I-ANS`` beats ``O``.
    """
    o = emissions[..., QA_TAG_O]
    start_logits = emissions[..., QA_TAG_B] - o
    end_logits = torch.maximum(emissions[..., QA_TAG_B], emissions[..., QA_TAG_I]) - o
    return start_logits.contiguous(), end_logits.contiguous()


class SpanBERTForQA(nn.Module):
    """SpanBERT for extractive Question Answering.

    ``use_crf=True`` (default): the answer span is decoded from a CRF over a
    3-tag BIO scheme (``O`` / ``B-ANS`` / ``I-ANS``). ``use_crf=False`` falls
    back to an independent start/end pointer head.
    """

    def __init__(self, use_crf: bool = True,
                 model_name: str = "SpanBERT/spanbert-base-cased"):
        super().__init__()
        self.use_crf = use_crf

        self.spanbert = AutoModel.from_pretrained(model_name).float()
        config = self.spanbert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Plain start/end pointer head (used when use_crf=False).
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.qa_outputs.weight, std=0.02)
        nn.init.zeros_(self.qa_outputs.bias)

        # BIO head + CRF (used when use_crf=True).
        if use_crf:
            self.qa_classifier = nn.Linear(config.hidden_size, QA_NUM_TAGS)
            nn.init.normal_(self.qa_classifier.weight, std=0.02)
            nn.init.zeros_(self.qa_classifier.bias)
            self.crf = CRF(num_tags=QA_NUM_TAGS, batch_first=True)

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
        seq_len = input_ids.shape[1]
        has_targets = start_positions is not None and end_positions is not None

        if not self.use_crf:
            logits = self.qa_outputs(sequence_output)  # (B, L, 2)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            loss = None
            if has_targets:
                ignored_index = seq_len
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                loss = (loss_fct(start_logits, start_positions)
                        + loss_fct(end_logits, end_positions)) / 2.0
            return (loss, start_logits, end_logits) if loss is not None else (start_logits, end_logits)

        # --- CRF path ---
        emissions = self.qa_classifier(sequence_output)  # (B, L, 3)
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        start_logits, end_logits = _bio_emissions_to_span_logits(emissions)

        loss = None
        if start_positions is not None and end_positions is not None:
            bio_tags = _spans_to_bio(start_positions, end_positions, seq_len)
            loss = self.crf(emissions, bio_tags, attention_mask)

        return (loss, start_logits, end_logits) if loss is not None else (start_logits, end_logits)

    @torch.no_grad()
    def decode_spans(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Viterbi-decode BIO tags and return ``(B, L)`` tag ids (CRF mode only)."""
        if not self.use_crf:
            raise RuntimeError("decode_spans requires use_crf=True")
        outputs = self.spanbert(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        emissions = self.qa_classifier(outputs.last_hidden_state)
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        return self.crf.decode(emissions, attention_mask)


class SpanBERTForNER(nn.Module):
    """SpanBERT for NER with optional CRF."""
    
    def __init__(self, num_ner_tags: int, use_crf: bool = True,
                 model_name: str = "SpanBERT/spanbert-base-cased"):
        super().__init__()
        self.num_labels = num_ner_tags
        self.use_crf = use_crf

        self.spanbert = AutoModel.from_pretrained(model_name).float()
        config = self.spanbert.config
        
        self.classifier = nn.Linear(config.hidden_size, num_ner_tags)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
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

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        loss = None
        if labels is not None:
            if self.use_crf:
                # The CRF needs a valid tag id at every position it scores.
                # Ignored positions (-100 for CLS/SEP/pad) are mapped to "O"
                # and excluded via the mask instead.
                crf_labels = labels.clone()
                ignore = crf_labels == -100
                crf_labels[ignore] = 0
                crf_mask = attention_mask.clone()
                crf_mask[ignore] = 0
                crf_mask[:, 0] = 1  # CRF requires the first timestep to be active
                loss = self.crf(logits, crf_labels, crf_mask)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                active = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active]
                active_labels = labels.view(-1)[active]
                loss = loss_fct(active_logits, active_labels)
        
        if self.use_crf and labels is None:
            tags = self.crf.decode(logits, attention_mask)
            return (loss, tags) if loss is not None else (tags,)
        
        return (loss, logits) if loss is not None else (logits,)
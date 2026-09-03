"""
Data loading and preprocessing utilities for QA and NER tasks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import numpy as np


class QADataset(Dataset):
    """Dataset for Question Answering (SQuAD format)."""
    
    def __init__(self, data, tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Tokenize question and context
        encoding = self.tokenizer(
            example['question'],
            example['context'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get answer character positions
        answer_start = example.get('answer_start', 0)
        answer_end = answer_start + len(example.get('answer_text', ''))
        
        # Convert character positions to token positions
        sequence_ids = encoding.sequence_ids()
        start_position = None
        end_position = None
        
        for i, seq_id in enumerate(sequence_ids):
            if seq_id == 1:  # Context tokens
                token_start = encoding.token_to_chars(i)[0] if encoding.token_to_chars(i) else None
                token_end = encoding.token_to_chars(i)[1] if encoding.token_to_chars(i) else None
                
                if token_start is not None and token_end is not None:
                    if token_start <= answer_start < token_end:
                        start_position = i
                    if token_start < answer_end <= token_end:
                        end_position = i
        
        # Handle unanswerable questions
        if start_position is None or end_position is None:
            start_position = 0
            end_position = 0
        
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['start_positions'] = torch.tensor(start_position, dtype=torch.long)
        item['end_positions'] = torch.tensor(end_position, dtype=torch.long)
        
        return item


class NERDataset(Dataset):
    """Dataset for Named Entity Recognition."""
    
    def __init__(self, data, tokenizer: AutoTokenizer, 
                 label_map: Dict[str, int], max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        self.id2label = {v: k for k, v in label_map.items()}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        tokens = example['tokens']
        ner_tags = [self.label_map[tag] for tag in example['ner_tags']]
        
        # Truncate if needed
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length-2]
            ner_tags = ner_tags[:self.max_length-2]
        
        # Add special tokens
        input_ids = [self.tokenizer.cls_token_id] + \
                    self.tokenizer.convert_tokens_to_ids(tokens) + \
                    [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        # Adjust labels for special tokens (-100 for CLS and SEP)
        labels = [-100] + ner_tags + [-100]
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def load_squad_data(split: str = 'train', max_samples: Optional[int] = None):
    """Load SQuAD v2.0 dataset."""
    dataset = load_dataset('rajpurkar/squad_v2', split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    processed_data = []
    for example in dataset:
        answers = example['answers']
        if answers['text']:
            processed_data.append({
                'question': example['question'],
                'context': example['context'],
                'answer_start': answers['answer_start'][0],
                'answer_text': answers['text'][0]
            })
        else:
            processed_data.append({
                'question': example['question'],
                'context': example['context'],
                'answer_start': 0,
                'answer_text': ''
            })
    
    return processed_data


CONLL_LABEL_MAP = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6,
    'B-MISC': 7, 'I-MISC': 8,
}


def load_conll_ner_data(split: str = 'train', max_samples: Optional[int] = None):
    """Load the CoNLL-2003 NER dataset.

    Primary source is the Hugging Face Hub (the auto-generated parquet export of
    ``eriktks/conll2003``, which needs no dataset script). If the Hub is
    unreachable it falls back to parsing raw CoNLL-format text files.

    Returns ``(examples, label_map)`` where each example is
    ``{"tokens": [...], "ner_tags": ["O", "B-PER", ...]}``.
    """
    label_map = dict(CONLL_LABEL_MAP)
    hf_split = {'train': 'train', 'validation': 'validation', 'test': 'test'}[split]

    try:
        dataset = load_dataset(
            'eriktks/conll2003', split=hf_split, revision='refs/convert/parquet'
        )
        tag_names = dataset.features['ner_tags'].feature.names
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        processed_data = [
            {"tokens": list(ex['tokens']),
             "ner_tags": [tag_names[t] for t in ex['ner_tags']]}
            for ex in dataset
        ]
        return processed_data, label_map
    except Exception as exc:  # offline / Hub issues -> raw text fallback
        print(f"HF load failed ({type(exc).__name__}); falling back to raw text files...")

    import os
    import tempfile
    import urllib.request

    file_split = {'train': 'train', 'validation': 'valid', 'test': 'test'}[split]
    urls = [
        f"https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.{file_split}",
        f"https://data.deepai.org/conll2003/{file_split}.txt",
    ]

    cache_dir = os.path.join(tempfile.gettempdir(), "spanbert_crf_conll2003")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{file_split}.txt")

    if not os.path.exists(cache_path):
        last_err = None
        for url in urls:
            try:
                print(f"Downloading CoNLL-2003 {split} split from {url} ...")
                urllib.request.urlretrieve(url, cache_path)
                break
            except Exception as e:  # try the next mirror
                last_err = e
        else:
            raise RuntimeError(
                "Could not obtain CoNLL-2003 from the Hub or any text mirror"
            ) from last_err

    processed_data = []
    with open(cache_path, "r", encoding="utf-8") as f:
        tokens, ner_tags = [], []
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("-DOCSTART-"):
                if tokens:
                    processed_data.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens, ner_tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 4:
                tokens.append(parts[0])
                ner_tags.append(parts[-1])

    if tokens:
        processed_data.append({"tokens": tokens, "ner_tags": ner_tags})

    if max_samples:
        processed_data = processed_data[:max_samples]

    return processed_data, label_map


def create_dataloaders(train_data, val_data, tokenizer, batch_size: int = 16, 
                       max_length: int = 512, dataset_type: str = 'qa'):
    """Create train and validation dataloaders."""
    
    if dataset_type == 'qa':
        train_dataset = QADataset(train_data, tokenizer, max_length)
        val_dataset = QADataset(val_data, tokenizer, max_length)
    else:
        label_map = train_data[1] if isinstance(train_data, tuple) else None
        train_dataset = NERDataset(train_data[0] if isinstance(train_data, tuple) else train_data, 
                                   tokenizer, label_map, max_length)
        val_dataset = NERDataset(val_data[0] if isinstance(val_data, tuple) else val_data,
                                 tokenizer, label_map, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

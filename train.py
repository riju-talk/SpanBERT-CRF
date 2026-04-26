"""
Main training script for SpanBERT-CRF models.
Supports both QA and NER tasks with optional CRF enhancement.
"""

import argparse
import torch
from transformers import SpanBertTokenizer, AutoConfig
from src.models import SpanBERTForQA, SpanBERTForNER
from src.data import load_squad_data, load_conll_ner_data, create_dataloaders
from src.train import Trainer


def train_qa(args):
    """Train QA model."""
    print("Loading SQuAD dataset...")
    train_data = load_squad_data('train', max_samples=args.max_samples)
    val_data = load_squad_data('validation', max_samples=args.max_samples // 5)
    
    tokenizer = SpanBertTokenizer.from_pretrained('spanbert-base-cased')
    
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_type='qa'
    )
    
    print("Initializing model...")
    config = AutoConfig.from_pretrained('spanbert-base-cased', num_labels=2)
    model = SpanBERTForQA.from_pretrained('spanbert-base-cased', config=config, use_crf=args.use_crf)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_wandb=args.use_wandb
    )
    
    print("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(f"{args.output_dir}/spanbert_qa_{'crf' if args.use_crf else 'base'}.pt")
    
    print("Training complete!")
    return history


def train_ner(args):
    """Train NER model."""
    print("Loading CoNLL-2003 dataset...")
    train_data = load_conll_ner_data('train', max_samples=args.max_samples)
    val_data = load_conll_ner_data('validation', max_samples=args.max_samples // 5)
    
    tokenizer = SpanBertTokenizer.from_pretrained('spanbert-base-cased')
    _, label_map = train_data
    
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_type='ner'
    )
    
    print("Initializing model...")
    config = AutoConfig.from_pretrained('spanbert-base-cased')
    model = SpanBERTForNER.from_pretrained(
        'spanbert-base-cased',
        config=config,
        num_ner_tags=len(label_map),
        use_crf=args.use_crf
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_wandb=args.use_wandb
    )
    
    print("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(f"{args.output_dir}/spanbert_ner_{'crf' if args.use_crf else 'base'}.pt")
    
    print("Training complete!")
    return history


def main():
    parser = argparse.ArgumentParser(description='Train SpanBERT-CRF models')
    
    # Task selection
    parser.add_argument('--task', type=str, default='qa', choices=['qa', 'ner'],
                       help='Task to train (qa or ner)')
    
    # Model options
    parser.add_argument('--use_crf', action='store_true',
                       help='Use CRF layer on top of SpanBERT')
    
    # Data options
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for debugging)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Training options
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Device and output
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save models')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Training configuration:")
    print(f"  Task: {args.task}")
    print(f"  Use CRF: {args.use_crf}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print("-" * 50)
    
    if args.task == 'qa':
        train_qa(args)
    elif args.task == 'ner':
        train_ner(args)


if __name__ == '__main__':
    main()

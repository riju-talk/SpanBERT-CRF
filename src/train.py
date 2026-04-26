"""
Training utilities for SpanBERT-CRF models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import wandb


class Trainer:
    """Trainer for QA and NER models with support for CRF."""
    
    def __init__(self, model, tokenizer, device: str = 'cuda', 
                 learning_rate: float = 2e-5, num_epochs: int = 3,
                 batch_size: int = 16, gradient_accumulation_steps: int = 1,
                 use_wandb: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project="spanbert-crf", config={
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size
            })
        
        self.optimizer = None
        self.scheduler = None
    
    def train(self, train_loader, val_loader=None) -> Dict[str, list]:
        """Train the model."""
        
        # Set up optimizer and scheduler
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
        )
        
        self.model.to(self.device)
        
        train_losses = []
        val_metrics = []
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                metrics = self._validate(val_loader)
                val_metrics.append(metrics)
                print(f"Validation Metrics: {metrics}")
                
                if self.use_wandb:
                    wandb.log({"val_" + k: v for k, v in metrics.items()})
            
            if self.use_wandb:
                wandb.log({"train_loss": train_loss, "epoch": epoch + 1})
        
        if self.use_wandb:
            wandb.finish()
        
        return {"train_losses": train_losses, "val_metrics": val_metrics}
    
    def _train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids'),
                start_positions=batch.get('start_positions'),
                end_positions=batch.get('end_positions'),
                labels=batch.get('labels')
            )
            
            loss = outputs[0]
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids'),
                    start_positions=batch.get('start_positions'),
                    end_positions=batch.get('end_positions'),
                    labels=batch.get('labels')
                )
                
                loss = outputs[0]
                total_loss += loss.item()
        
        return {"val_loss": total_loss / len(val_loader)}
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {path}")

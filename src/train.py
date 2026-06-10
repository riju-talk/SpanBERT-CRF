"""
Full training pipeline for SpanBERT-CRF models with LoRA support.
Tasks: QA (SQuAD v2.0) and NER (CoNLL-2003).
"""

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from typing import Dict, Optional
import argparse
import os

from src.models import SpanBERTForQA, SpanBERTForNER, apply_lora_to_model
from src.data import load_squad_data, load_conll_ner_data, create_dataloaders


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trainer for QA and NER models with CRF and LoRA support."""

    def __init__(self, model, tokenizer, device: str = "cuda",
                 learning_rate: float = 2e-5, num_epochs: int = 3,
                 batch_size: int = 16, gradient_accumulation_steps: int = 1,
                 use_wandb: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_wandb = use_wandb

        if use_wandb:
            import wandb
            wandb.init(project="spanbert-crf", config={
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
            })

        self.optimizer = None
        self.scheduler = None

    def train(self, train_loader, val_loader=None) -> Dict[str, list]:
        """Run the full training loop."""

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        self.model.to(self.device)

        train_losses = []
        val_metrics = []

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            train_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(train_loss)

            if val_loader is not None:
                metrics = self._validate(val_loader)
                val_metrics.append(metrics)
                print(f"Validation Metrics: {metrics}")

                if self.use_wandb:
                    import wandb
                    wandb.log({"val_" + k: v for k, v in metrics.items()})

            if self.use_wandb:
                import wandb
                wandb.log({"train_loss": train_loss, "epoch": epoch + 1})

        if self.use_wandb:
            import wandb
            wandb.finish()

        return {"train_losses": train_losses, "val_metrics": val_metrics}

    def _train_epoch(self, train_loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                start_positions=batch.get("start_positions"),
                end_positions=batch.get("end_positions"),
                labels=batch.get("labels"),
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
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    start_positions=batch.get("start_positions"),
                    end_positions=batch.get("end_positions"),
                    labels=batch.get("labels"),
                )

                loss = outputs[0]
                total_loss += loss.item()

        return {"val_loss": total_loss / len(val_loader)}

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Model loaded from {path}")


# ---------------------------------------------------------------------------
# HuggingFace Hub upload
# ---------------------------------------------------------------------------

def upload_to_huggingface(model, tokenizer, repo_id, task, hf_token=None):
    """Upload the trained model and tokenizer to the HuggingFace Hub.

    How to use:
        1. Install huggingface_hub:  pip install huggingface_hub
        2. Log in:                    huggingface-cli login
           (or set the HF_TOKEN environment variable)
        3. Call this function after training completes.

    Note: SpanBERTForQA/SpanBERTForNER are custom nn.Module classes (not
    PreTrainedModels). We save the state_dict + metadata + tokenizer, so the
    model can be restored via ``SpanBERTForQA(...).load_state_dict(...)``.

    Args:
        model: Trained SpanBERTForQA or SpanBERTForNER.
        tokenizer: Corresponding tokenizer.
        repo_id: HF repo name, e.g. "Phantomcloak19/SpanBERT-CRF".
        task: "qa" or "ner" — used in the subfolder name.
        hf_token: Optional token. Falls back to env var or cached login.
    """
    from huggingface_hub import HfApi, login
    import os
    import json

    if hf_token:
        login(token=hf_token)

    # Merge LoRA adapters into base weights before saving
    if hasattr(model, "spanbert") and hasattr(model.spanbert, "merge_and_unload"):
        print("Merging LoRA adapters into base model...")
        model.spanbert = model.spanbert.merge_and_unload()

    save_dir = f"./hf_upload_{task}"
    os.makedirs(save_dir, exist_ok=True)

    # Save full state dict (includes spanbert + task head weights)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    # Save metadata so loader knows how to reconstruct
    meta = {
        "task": task,
        "use_crf": getattr(model, "use_crf", False),
        "num_labels": getattr(model, "num_labels", None),
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(meta, f)

    # Tokenizer
    tokenizer.save_pretrained(save_dir)

    # Upload
    api = HfApi()
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {task} model — SpanBERT-CRF",
    )
    print(f"Model uploaded to https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------


def train_qa(args):
    """Train QA model on SQuAD v2.0."""
    print("Loading SQuAD v2.0 dataset...")
    train_data = load_squad_data("train", max_samples=args.max_train_samples)
    val_data = load_squad_data("validation", max_samples=args.max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_type="qa",
    )

    print("Initializing model...")
    model = SpanBERTForQA(use_crf=args.use_crf, model_name="SpanBERT/spanbert-base-cased")

    if args.use_lora:
        model = apply_lora_to_model(model, use_lora=True, r=args.lora_r, lora_alpha=args.lora_alpha)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_wandb=args.use_wandb,
    )

    print("Starting training...")
    history = trainer.train(train_loader, val_loader)

    save_path = f"{args.output_dir}/spanbert_qa_{'crf_' if args.use_crf else ''}{'lora_' if args.use_lora else ''}base.pt"
    trainer.save_model(save_path)

    if args.upload:
        upload_to_huggingface(
            model=model,
            tokenizer=tokenizer,
            repo_id=args.hf_repo_id,
            task="qa",
        )

    print("QA training complete!")
    return history


def train_ner(args):
    """Train NER model on CoNLL-2003."""
    print("Loading CoNLL-2003 dataset...")
    train_data = load_conll_ner_data("train", max_samples=args.max_train_samples)
    val_data = load_conll_ner_data("validation", max_samples=args.max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    _, label_map = train_data

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_type="ner",
    )

    print("Initializing model...")
    model = SpanBERTForNER(num_ner_tags=len(label_map), use_crf=args.use_crf, model_name="SpanBERT/spanbert-base-cased")

    if args.use_lora:
        model = apply_lora_to_model(model, use_lora=True, r=args.lora_r, lora_alpha=args.lora_alpha)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_wandb=args.use_wandb,
    )

    print("Starting training...")
    history = trainer.train(train_loader, val_loader)

    save_path = f"{args.output_dir}/spanbert_ner_{'crf_' if args.use_crf else ''}{'lora_' if args.use_lora else ''}base.pt"
    trainer.save_model(save_path)

    if args.upload:
        upload_to_huggingface(
            model=model,
            tokenizer=tokenizer,
            repo_id=args.hf_repo_id,
            task="ner",
        )

    print("NER training complete!")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train SpanBERT-CRF models with optional LoRA")

    # Task
    parser.add_argument("--task", type=str, default="qa", choices=["qa", "ner"],
                        help="Task to train: qa or ner")

    # Model
    parser.add_argument("--use_crf", action="store_true",
                        help="Enable CRF layer on top of SpanBERT")
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA fine-tuning via PEFT")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling")

    # Data
    parser.add_argument("--max_train_samples", type=int, default=15000,
                        help="Max training samples (default: 15000)")
    parser.add_argument("--max_eval_samples", type=int, default=5000,
                        help="Max evaluation samples (default: 5000)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length")

    # Training
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")

    # Device / output
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model checkpoints")

    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")

    # HuggingFace Hub upload
    parser.add_argument("--upload", action="store_true",
                        help="Upload model to HuggingFace Hub after training")
    parser.add_argument("--hf_repo_id", type=str, default=None,
                        help="HF repo ID (default: Phantomcloak19/SpanBERT-CRF for QA, Phantomcloak19/SpanBERT-CRF-NER for NER)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Default HF repo per task
    if args.hf_repo_id is None:
        args.hf_repo_id = "Phantomcloak19/SpanBERT-CRF" if args.task == "qa" else "Phantomcloak19/SpanBERT-CRF-NER"

    print("Training configuration:")
    print(f"  Task:              {args.task}")
    print(f"  Use CRF:           {args.use_crf}")
    print(f"  Use LoRA:          {args.use_lora}")
    if args.use_lora:
        print(f"  LoRA r:            {args.lora_r}")
        print(f"  LoRA alpha:        {args.lora_alpha}")
    print(f"  Train samples:     {args.max_train_samples}")
    print(f"  Eval samples:      {args.max_eval_samples}")
    print(f"  Upload to HF:      {args.upload}")
    if args.upload:
        print(f"  HF repo:           {args.hf_repo_id}")
    print(f"  Device:            {args.device}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Learning rate:     {args.learning_rate}")
    print(f"  Epochs:            {args.num_epochs}")
    print(f"  Max length:        {args.max_length}")
    print("-" * 50)

    if args.task == "qa":
        train_qa(args)
    else:
        train_ner(args)


if __name__ == "__main__":
    main()

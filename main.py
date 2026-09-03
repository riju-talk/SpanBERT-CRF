"""
main.py — Unified training orchestrator for SpanBERT-CRF.

This is the single entry point that runs the **entire** training pipeline:

    1. Question Answering       — SpanBERT-CRF on SQuAD v2.0
    2. Named Entity Recognition — SpanBERT-CRF on CoNLL-2003

The CRF head is on for both stages by default (``--no-crf`` disables it, for
ablations only).

Each stage reuses the building blocks in ``src/`` (models, data, trainer,
metrics) without modifying them. Stages are independent: you can run one,
the other, or both, and every stage writes its own checkpoint under
``--output-dir``.

Examples
--------
Run the full pipeline (QA then NER) with sensible defaults::

    python main.py

Only NER, with the CRF head and LoRA adapters, on a small subset::

    python main.py --tasks ner --use-crf --use-lora \
        --max-train-samples 2000 --max-eval-samples 500 --num-epochs 3

Full pipeline and push both checkpoints to the Hugging Face Hub::

    python main.py --tasks qa ner --upload

Quick smoke test of the wiring without touching a GPU::

    python main.py --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from src.data import create_dataloaders, load_conll_ner_data, load_squad_data
from src.models import SpanBERTForNER, SpanBERTForQA, apply_lora_to_model
from src.train import Trainer, upload_to_huggingface
from transformers import AutoTokenizer

MODEL_NAME = "SpanBERT/spanbert-base-cased"
DEFAULT_HF_REPOS = {
    "qa": "Phantomcloak19/SpanBERT-CRF",
    "ner": "Phantomcloak19/SpanBERT-CRF-NER",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PipelineConfig:
    """Everything the orchestrator needs, resolved from the CLI."""

    tasks: list[str]

    # model — the CRF head is part of the architecture for both tasks
    use_crf: bool = True
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    model_name: str = MODEL_NAME

    # data
    max_train_samples: int | None = 15_000
    max_eval_samples: int | None = 5_000
    max_length: int = 384

    # optimisation
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    gradient_accumulation: int = 1

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "models"

    # logging / distribution
    use_wandb: bool = False
    upload: bool = False
    hf_repo_qa: str = DEFAULT_HF_REPOS["qa"]
    hf_repo_ner: str = DEFAULT_HF_REPOS["ner"]

    dry_run: bool = False

    def checkpoint_path(self, task: str) -> str:
        tags = "".join((
            "crf_" if self.use_crf else "",
            "lora_" if self.use_lora else "",
        ))
        return os.path.join(self.output_dir, f"spanbert_{task}_{tags}base.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}", flush=True)


def _as_train_args(cfg: PipelineConfig, task: str) -> argparse.Namespace:
    """Adapt :class:`PipelineConfig` to the namespace ``src.train`` expects."""
    return argparse.Namespace(
        task=task,
        use_crf=cfg.use_crf,
        use_lora=cfg.use_lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_epochs=cfg.num_epochs,
        gradient_accumulation=cfg.gradient_accumulation,
        device=cfg.device,
        output_dir=cfg.output_dir,
        use_wandb=cfg.use_wandb,
        upload=False,  # uploads are handled centrally by this orchestrator
        hf_repo_id=cfg.hf_repo_qa if task == "qa" else cfg.hf_repo_ner,
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_qa_stage(cfg: PipelineConfig) -> dict[str, Any]:
    crf_note = "-CRF" if cfg.use_crf else ""
    _banner(f"STAGE: Question Answering  -  SpanBERT{crf_note}  -  SQuAD v2.0")

    print("Loading SQuAD v2.0 ...", flush=True)
    train_data = load_squad_data("train", max_samples=cfg.max_train_samples)
    val_data = load_squad_data("validation", max_samples=cfg.max_eval_samples)
    print(f"  train={len(train_data):,}  val={len(val_data):,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, tokenizer,
        batch_size=cfg.batch_size, max_length=cfg.max_length, dataset_type="qa",
    )

    model = SpanBERTForQA(use_crf=cfg.use_crf, model_name=cfg.model_name)
    if cfg.use_lora:
        model = apply_lora_to_model(model, use_lora=True, r=cfg.lora_r, lora_alpha=cfg.lora_alpha)

    trainer = Trainer(
        model=model, tokenizer=tokenizer, device=cfg.device,
        learning_rate=cfg.learning_rate, num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.gradient_accumulation,
        use_wandb=cfg.use_wandb,
    )

    history = trainer.train(train_loader, val_loader)

    ckpt = cfg.checkpoint_path("qa")
    trainer.save_model(ckpt)

    if cfg.upload:
        upload_to_huggingface(model=model, tokenizer=tokenizer, repo_id=cfg.hf_repo_qa, task="qa")

    return {"checkpoint": ckpt, "history": history}


def run_ner_stage(cfg: PipelineConfig) -> dict[str, Any]:
    crf_note = "-CRF" if cfg.use_crf else ""
    _banner(f"STAGE: Named Entity Recognition  -  SpanBERT{crf_note}  -  CoNLL-2003")

    print("Loading CoNLL-2003 ...", flush=True)
    train_data = load_conll_ner_data("train", max_samples=cfg.max_train_samples)
    val_data = load_conll_ner_data("validation", max_samples=cfg.max_eval_samples)
    _, label_map = train_data
    print(f"  train={len(train_data[0]):,}  val={len(val_data[0]):,}  labels={len(label_map)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, tokenizer,
        batch_size=cfg.batch_size, max_length=min(cfg.max_length, 256), dataset_type="ner",
    )

    model = SpanBERTForNER(num_ner_tags=len(label_map), use_crf=cfg.use_crf, model_name=cfg.model_name)
    if cfg.use_lora:
        model = apply_lora_to_model(model, use_lora=True, r=cfg.lora_r, lora_alpha=cfg.lora_alpha)

    trainer = Trainer(
        model=model, tokenizer=tokenizer, device=cfg.device,
        learning_rate=cfg.learning_rate, num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.gradient_accumulation,
        use_wandb=cfg.use_wandb,
    )

    history = trainer.train(train_loader, val_loader)

    ckpt = cfg.checkpoint_path("ner")
    trainer.save_model(ckpt)

    if cfg.upload:
        upload_to_huggingface(model=model, tokenizer=tokenizer, repo_id=cfg.hf_repo_ner, task="ner")

    return {"checkpoint": ckpt, "history": history, "label_map": label_map}


STAGES = {"qa": run_qa_stage, "ner": run_ner_stage}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Unified training orchestrator for SpanBERT-CRF (QA + NER).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--tasks", nargs="+", choices=["qa", "ner"], default=["qa", "ner"],
                   help="Which stage(s) to run, in order.")

    g = p.add_argument_group("model")
    g.add_argument("--no-crf", dest="use_crf", action="store_false",
                   help="Ablation: drop the CRF head (on by default for both tasks).")
    g.add_argument("--use-lora", action="store_true", help="Fine-tune with LoRA adapters (PEFT).")
    g.add_argument("--lora-r", type=int, default=8)
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--model-name", default=MODEL_NAME)

    g = p.add_argument_group("data")
    g.add_argument("--max-train-samples", type=int, default=15_000,
                   help="Cap on training examples per stage (use -1 for the full split).")
    g.add_argument("--max-eval-samples", type=int, default=5_000,
                   help="Cap on validation examples per stage (use -1 for the full split).")
    g.add_argument("--max-length", type=int, default=384)

    g = p.add_argument_group("optimisation")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--learning-rate", type=float, default=2e-5)
    g.add_argument("--num-epochs", type=int, default=3)
    g.add_argument("--gradient-accumulation", type=int, default=1)

    g = p.add_argument_group("runtime")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output-dir", default="models")
    g.add_argument("--dry-run", action="store_true",
                   help="Resolve config and print the plan without training.")

    g = p.add_argument_group("logging / distribution")
    g.add_argument("--use-wandb", action="store_true")
    g.add_argument("--upload", action="store_true", help="Push each checkpoint to the HF Hub.")
    g.add_argument("--hf-repo-qa", default=DEFAULT_HF_REPOS["qa"])
    g.add_argument("--hf-repo-ner", default=DEFAULT_HF_REPOS["ner"])

    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    def cap(value: int) -> int | None:
        return None if value is not None and value < 0 else value

    return PipelineConfig(
        tasks=list(dict.fromkeys(args.tasks)),  # dedupe, keep order
        use_crf=args.use_crf,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        model_name=args.model_name,
        max_train_samples=cap(args.max_train_samples),
        max_eval_samples=cap(args.max_eval_samples),
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation=args.gradient_accumulation,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        upload=args.upload,
        hf_repo_qa=args.hf_repo_qa,
        hf_repo_ner=args.hf_repo_ner,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    _banner("SpanBERT-CRF - training pipeline")
    print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    # `_as_train_args` is exercised so a signature drift in src.train fails fast.
    for task in cfg.tasks:
        _as_train_args(cfg, task)

    if cfg.dry_run:
        print("\n[dry-run] configuration is valid; exiting before training.")
        return 0

    started = time.time()
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": dataclasses.asdict(cfg),
        "stages": {},
    }

    for task in cfg.tasks:
        stage_started = time.time()
        result = STAGES[task](cfg)
        history = result.get("history", {})
        report["stages"][task] = {
            "checkpoint": result["checkpoint"],
            "minutes": round((time.time() - stage_started) / 60, 2),
            "train_losses": history.get("train_losses"),
            "val_metrics": history.get("val_metrics"),
        }

    report["total_minutes"] = round((time.time() - started) / 60, 2)

    report_path = os.path.join(cfg.output_dir, "pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    _banner("PIPELINE COMPLETE")
    for task, info in report["stages"].items():
        last_val = (info["val_metrics"] or [{}])[-1]
        print(f"  {task.upper():4s}  {info['minutes']:>6.2f} min  ->  {info['checkpoint']}")
        if last_val:
            print(f"        final val: {last_val}")
    print(f"\n  report: {report_path}")
    print(f"  total:  {report['total_minutes']:.2f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

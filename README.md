<div align="center">

# SpanBERT‑CRF

**A linear‑chain CRF on top of SpanBERT for span‑structured NLP — Question Answering *and* Named Entity Recognition.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.x-FFD21E)](https://huggingface.co/docs/transformers)
[![uv](https://img.shields.io/badge/packaged%20with-uv-DE5FE9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](tests/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

[![Open QA notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riju-talk/SpanBERT-CRF/blob/main/spanbert-crf.ipynb)
[![Model on HF Hub](https://img.shields.io/badge/🤗%20Hub-Phantomcloak19%2FSpanBERT--CRF-blue)](https://huggingface.co/Phantomcloak19/SpanBERT-CRF)

</div>

---

## Table of Contents

- [Why SpanBERT‑CRF](#why-spanbertcrf)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Inference](#inference)
- [Evaluation & metrics](#evaluation--metrics)
- [Testing](#testing)
- [Configuration reference](#configuration-reference)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)

---

## Why SpanBERT‑CRF

[SpanBERT](https://arxiv.org/abs/1907.10529) pre‑trains by masking and predicting
contiguous **spans** rather than individual tokens, which makes it a strong
encoder for anything span‑shaped. This repo adds a **linear‑chain CRF** decoding
layer so that predictions are *globally consistent sequences* instead of
independent per‑token choices:

- **NER** — the CRF learns that `I-PER` cannot follow `B-LOC`, that entities
  start with `B-`, etc., and Viterbi decoding returns the single best valid tag
  path.
- **QA** — the answer span is recast as a 3‑tag BIO problem
  (`O` / `B-ANS` / `I-ANS`) over the sequence. The CRF enforces one contiguous
  run, and an all‑`O` decode is a first‑class prediction for **unanswerable**
  SQuAD v2.0 questions.

One encoder, one CRF abstraction, two tasks.

---

## How it works

```
                       ┌───────────────────────────┐
  question + context   │                           │   BIO emissions   ┌─────┐   Viterbi   answer span
  (QA)  ───────────────▶      SpanBERT encoder      ├───────────────────▶ CRF ├────────────▶  or  "unanswerable"
  raw tokens (NER)  ───▶   (SpanBERT/spanbert-      │   (B, L, num_tags)└─────┘             / entity list
                       │        base-cased)         │
                       └───────────────────────────┘
```

| | Question Answering | Named Entity Recognition |
|---|---|---|
| Dataset | SQuAD v2.0 (`rajpurkar/squad_v2`) | CoNLL‑2003 (`eriktks/conll2003`, parquet export) |
| CRF tag set | `O`, `B-ANS`, `I-ANS` | `O`, `B-/I-` × {PER, ORG, LOC, MISC} |
| Loss | CRF negative log‑likelihood over BIO span | CRF negative log‑likelihood over entity tags |
| Decode | Viterbi → first `B-ANS …` run → detokenised text | Viterbi → BIO → entity offsets |
| Metrics | Exact Match, token‑F1 (+ BLEU, BERTScore) | entity‑level precision / recall / F1 |

The CRF (`src/models.py::CRF`) is a from‑scratch batch‑first implementation:
forward‑algorithm partition function, gold‑path scoring with masking, and
Viterbi decoding with back‑pointers. Its NLL is unit‑tested against brute‑force
path enumeration (`tests/test_crf.py`).

Optional **LoRA** adapters (via `peft`) keep the encoder frozen and train only
low‑rank deltas plus the task head and CRF.

---

## Project layout

```text
SpanBERT-CRF/
├── main.py                 # ⟵ unified orchestrator: runs the whole QA + NER pipeline
├── src/
│   ├── models.py           # CRF, SpanBERTForQA (CRF/BIO), SpanBERTForNER, LoRA helper
│   ├── data.py             # SQuAD v2 + CoNLL-2003 loaders, QADataset / NERDataset
│   ├── train.py            # Trainer + per-task CLI (python -m src.train ...)
│   ├── metrics.py          # EM / F1 / BLEU / BERTScore (QA), P/R/F1 (NER), Evaluator
│   └── inference.py        # QAInference / NERInference pipelines + loader
├── tests/
│   ├── conftest.py         # fixtures, `slow` marker, encoder-availability skip
│   ├── test_crf.py         # CRF vs brute-force, masking, Viterbi
│   ├── test_models.py      # BIO span helpers + (slow) full-model forward/backward
│   ├── test_metrics.py     # QA & NER scoring
│   ├── test_data.py        # tensorisation, label alignment, dataloaders
│   └── test_pipeline.py    # main.py config resolution & dry-run
├── spanbert-crf.ipynb      # standalone Colab notebook (QA + NER, no src/ dependency)
├── models/                 # checkpoints (git-ignored except tracked baselines)
├── hf_upload_qa/            # exported QA artifact staged for the HF Hub
├── pyproject.toml          # deps + tooling (managed with uv)
└── uv.lock
```

---

## Inference

```python
from src.inference import load_inference_model

# Question Answering
qa = load_inference_model("models/spanbert_qa_crf_base.pt", task_type="qa", use_crf=True)
print(qa.predict(
    context="SpanBERT was introduced by Joshi et al. in 2020.",
    question="Who introduced SpanBERT?",
))
# -> {'answer': 'Joshi et al.', 'confidence': ..., 'start_token': ..., ...}

# Named Entity Recognition
ner = load_inference_model("models/spanbert_ner_crf_base.pt", task_type="ner", use_crf=True)
print(ner.predict("Apple was founded by Steve Jobs in Cupertino."))
# -> [{'type': 'ORG', 'text': 'Apple'}, {'type': 'PER', 'text': 'Steve Jobs'}, {'type': 'LOC', 'text': 'Cupertino'}]
```

The custom `SpanBERTForQA` / `SpanBERTForNER` modules are also serialised to the
Hugging Face Hub as `state_dict` + `config.json` + tokenizer
(`src/train.py::upload_to_huggingface`).

---

## Evaluation & metrics

`src/metrics.py` provides:

| Task | Metrics |
|------|---------|
| QA | Exact Match, token‑level F1, corpus BLEU (`sacrebleu`), BERTScore F1 (optional), span‑overlap |
| NER | entity‑level precision / recall / F1 (strict BIO span match) |

`evaluate_on_split(...)` refuses to score the `train` split by design.

---

## Testing

```bash
pytest -m "not slow"     # fast: CRF, metrics, data, pipeline config  (no weights, seconds)
pytest -m slow           # full-model forward/backward (needs the SpanBERT encoder)
pytest                    # everything
pytest --cov=src --cov-report=term-missing
```

Slow tests skip themselves automatically when `SpanBERT/spanbert-base-cased`
cannot be loaded (offline CI with a cold cache).

---

## Configuration reference

`main.py` flags (see `python main.py --help` for the full list):

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks` | `qa ner` | Stages to run, in order |
| `--no-crf` | *(CRF on)* | Ablation: drop the CRF head |
| `--use-lora` / `--lora-r` / `--lora-alpha` | off / 8 / 32 | LoRA fine‑tuning |
| `--max-train-samples` / `--max-eval-samples` | 15000 / 5000 | Per‑stage caps (`-1` = full split) |
| `--max-length` | 384 | Max sequence length (NER is capped at 256) |
| `--batch-size` / `--learning-rate` / `--num-epochs` | 16 / 2e‑5 / 3 | Optimisation |
| `--gradient-accumulation` | 1 | Micro‑batching |
| `--device` | auto | `cuda` / `cpu` |
| `--seed` | 42 | Reproducibility |
| `--output-dir` | `models` | Checkpoints + `pipeline_report.json` |
| `--use-wandb` | off | Weights & Biases logging |
| `--upload` / `--hf-repo-qa` / `--hf-repo-ner` | off | Push checkpoints to the HF Hub |

---

## Roadmap

- [x] From‑scratch linear‑chain CRF (forward algorithm + Viterbi), brute‑force verified
- [x] SpanBERT‑CRF for **NER** (CoNLL‑2003)
- [x] SpanBERT‑CRF for **QA** via BIO span tagging (SQuAD v2.0, unanswerable‑aware)
- [x] Unified `main.py` orchestrator + per‑task `src.train` CLI
- [x] LoRA / PEFT fine‑tuning
- [x] Standalone Colab notebook
- [x] Test suite (fast unit + slow integration)
- [ ] FastAPI serving layer
- [ ] Hyper‑parameter search (Optuna)
- [ ] Multi‑GPU / mixed precision
- [ ] ONNX export & quantisation
- [ ] More datasets (OntoNotes, Natural Questions)

---

## Acknowledgments

- **SpanBERT** — [Joshi et al., 2020](https://arxiv.org/abs/1907.10529)
- **CRF for sequence labelling** — [Lafferty et al., 2001](https://repository.upenn.edu/cis_papers/159/); implementation inspired by [`pytorch-crf`](https://github.com/kmkurn/pytorch-crf)
- **Datasets** — [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/), [CoNLL‑2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- **🤗 Transformers, Datasets, PEFT**

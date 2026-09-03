"""Shared pytest fixtures and configuration for the SpanBERT-CRF test suite.

Tests are split into two groups:

* **fast** (default) — pure-tensor unit tests for the CRF, the metrics and the
  data/label plumbing. No network, no pretrained weights, runs in seconds.
* **slow** (``-m slow``) — end-to-end tests that instantiate the real
  ``SpanBERT/spanbert-base-cased`` encoder. These are skipped automatically when
  the weights cannot be loaded (e.g. offline CI with a cold cache).

Run everything::      pytest
Fast only::           pytest -m "not slow"
Slow only::           pytest -m slow
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make the repo root importable so ``import main`` works from the test suite.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_NAME = "SpanBERT/spanbert-base-cased"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: needs the pretrained SpanBERT encoder")


@pytest.fixture(scope="session")
def spanbert_name() -> str:
    return MODEL_NAME


@pytest.fixture(scope="session")
def tokenizer():
    """Real SpanBERT tokenizer, or skip the test if it cannot be fetched."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"SpanBERT tokenizer unavailable: {exc}")


@pytest.fixture(scope="session")
def _encoder_available() -> bool:
    from transformers import AutoModel

    try:
        AutoModel.from_pretrained(MODEL_NAME)
        return True
    except Exception:  # pragma: no cover - depends on environment
        return False


@pytest.fixture
def require_encoder(_encoder_available: bool) -> None:
    if not _encoder_available:
        pytest.skip("SpanBERT encoder weights unavailable")


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)

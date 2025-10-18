"""Centralised default path definitions for runtime artifacts."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SUMMARIES_DIR = DATA_DIR / "summaries"
MODELS_DIR = PROJECT_ROOT / "models"
TOPIC_MODEL_PATH = DATA_DIR / "topic_model.joblib"
CORPUS_PATH = DATA_DIR / "corpus.parquet"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CACHE_DIR",
    "SUMMARIES_DIR",
    "MODELS_DIR",
    "TOPIC_MODEL_PATH",
    "CORPUS_PATH",
]

"""Fine-tune a Seq2Seq summarisation model on meeting transcripts."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Set, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:  # Optional dependency; gracefully handle absence.
    from dotenv import set_key
except ImportError:  # pragma: no cover - optional
    set_key = None


LOGGER = logging.getLogger("meeting_summariser_train")


def _summary_files(base_dir: Path) -> Iterable[Path]:
    seen: Set[Path] = set()
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.stem.lower() != "summary":
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path


def _find_transcript(summary_file: Path) -> Path | None:
    for candidate in summary_file.parent.iterdir():
        if not candidate.is_file():
            continue
        if candidate.stem.lower() == "transcript":
            return candidate
    return None


def _stringify_summary(raw_summary) -> str:
    """Normalise nested summary payloads into a single string."""

    def _collect_from_sequence(values) -> list[str]:
        collected: list[str] = []
        for item in values:
            text = _stringify_summary(item)
            if text:
                collected.append(text)
        return collected

    if raw_summary is None:
        return ""

    if isinstance(raw_summary, str):
        return raw_summary.strip()

    if isinstance(raw_summary, (list, tuple, set)):
        return " ".join(_collect_from_sequence(raw_summary)).strip()

    if isinstance(raw_summary, dict):
        preferred_keys = (
            "generated_text",
            "summary_text",
            "summary",
            "text",
        )
        structured_keys = (
            "highlights",
            "action_items",
            "decisions",
            "discussion_points",
        )

        candidates: list[str] = []

        for key in preferred_keys:
            value = raw_summary.get(key)
            if value is None:
                continue
            text = _stringify_summary(value)
            if text:
                candidates.append(text)

        for key in structured_keys:
            value = raw_summary.get(key)
            if value is None:
                continue
            text = _stringify_summary(value)
            if text:
                candidates.append(text)

        if not candidates:
            for value in raw_summary.values():
                text = _stringify_summary(value)
                if text:
                    candidates.append(text)

        return " ".join(candidates).strip()

    return str(raw_summary).strip()


def _load_pairs(base_dir: Path) -> Tuple[list[str], list[str]]:
    transcripts: list[str] = []
    summaries: list[str] = []

    for summary_file in _summary_files(base_dir):
        transcript_file = _find_transcript(summary_file)
        if transcript_file is None:
            continue

        transcript = transcript_file.read_text(encoding="utf-8").strip()
        if not transcript:
            continue

        summary_text = summary_file.read_text(encoding="utf-8").strip()
        if not summary_text:
            continue

        try:
            summary_payload = json.loads(summary_text)
        except json.JSONDecodeError:
            summary_payload = summary_text

        if isinstance(summary_payload, dict):
            raw_summary = summary_payload.get("summary", summary_payload)
        else:
            raw_summary = summary_payload

        target = _stringify_summary(raw_summary)

        if not target:
            continue

        transcripts.append(transcript)
        summaries.append(target)

    return transcripts, summaries


def _prepare_dataset(base_dir: Path) -> Dataset:
    transcripts, summaries = _load_pairs(base_dir)
    if not transcripts:
        raise RuntimeError(f"No transcript/summary pairs found under {base_dir}")

    dataset = Dataset.from_dict({"text": transcripts, "summary": summaries})
    return dataset


def _preprocess_fn(tokenizer, max_source_len: int, max_target_len: int):
    def _encode(examples: dict) -> dict:
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_source_len,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"],
                truncation=True,
                max_length=max_target_len,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _encode


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with transcript.txt and summary.json pairs")
    parser.add_argument("--model-name", default=os.getenv("MEETING_SUMMARY_MODEL", "gogamza/kobart-base-v2"))
    parser.add_argument("--output-dir", default=Path("./summariser_ft"), type=Path)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-split", type=float, default=0.1, help="Fraction of data for evaluation")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument(
        "--env-path",
        type=Path,
        default=Path(".env"),
        help=".env file to update with MEETING_SUMMARY_MODEL",
    )
    parser.add_argument(
        "--no-update-env",
        action="store_true",
        help="Skip writing MEETING_SUMMARY_MODEL to the .env file",
    )
    return parser


def _resolve_latest_checkpoint(output_dir: Path, trainer: Seq2SeqTrainer) -> Path:
    state = trainer.state
    preferred = [
        getattr(state, "best_model_checkpoint", None),
        getattr(state, "last_model_checkpoint", None),
    ]
    for candidate in preferred:
        if candidate:
            checkpoint_path = Path(candidate)
            if checkpoint_path.exists():
                return checkpoint_path

    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if checkpoints:
        return checkpoints[-1]

    return output_dir


def train(args: argparse.Namespace) -> None:
    dataset = _prepare_dataset(args.input_dir)
    split_ratio = min(max(args.eval_split, 0.0), 0.5)
    if split_ratio > 0:
        dataset_dict = dataset.train_test_split(test_size=split_ratio, seed=args.seed)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    preprocess = _preprocess_fn(tokenizer, args.max_source_length, args.max_target_length)
    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)

    mixed_precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch" if tokenized_eval is not None else "no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        save_total_limit=args.save_total_limit,
        logging_dir=str(args.output_dir / "logs"),
        logging_steps=args.logging_steps,
        fp16=mixed_precision == "fp16",
        bf16=mixed_precision == "bf16",
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    latest_checkpoint = _resolve_latest_checkpoint(args.output_dir, trainer).resolve()
    LOGGER.info("Training complete. Latest checkpoint: %s", latest_checkpoint)

    if args.no_update_env:
        return

    if set_key is None:
        LOGGER.warning("python-dotenv not installed; skipping .env update.")
        return

    env_path = args.env_path
    env_path.parent.mkdir(parents=True, exist_ok=True)
    set_key(str(env_path), "MEETING_SUMMARY_MODEL", str(latest_checkpoint))
    LOGGER.info("Updated %s with MEETING_SUMMARY_MODEL", env_path)


def main(argv: Iterable[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    train(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

"""
Training Pipeline for Schema Linking

Fine-tunes a causal language model (e.g., GPT-OSS-20B) to perform
schema linking: given a question and database schema, predict the
relevant tables and columns.

Uses:
- HuggingFace Transformers (AutoTokenizer, AutoModelForCausalLM)
- LoRA via PEFT for memory-efficient fine-tuning
- Trainer API for training loop
- JSONL dataset format
"""

import json
import argparse
import os
import math
import time
import threading
from pathlib import Path
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from config import (
    MODEL_NAME, OUTPUT_DIR, NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, LR_SCHEDULER_TYPE,
    FP16, BF16, MAX_SEQ_LENGTH, GRADIENT_CHECKPOINTING,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH,
)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_prompt(entry: Dict) -> str:
    """
    Format a training example into a full prompt string.

    Combines input instruction + output target into a single
    string for causal LM training.
    """
    return f"{entry['input']}\n\n{entry['output']}"


def tokenize_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = MAX_SEQ_LENGTH,
) -> List[Dict]:
    """
    Tokenize the dataset.

    For causal LM training, we tokenize the full combined string
    (input + output). The model learns to predict output tokens
    given the input prefix via standard next-token prediction.
    """
    tokenized = []
    for entry in data:
        prompt_text = entry["input"]
        full_text = format_prompt(entry)
        prompt_encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        full_encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        prompt_length = len(prompt_encoding["input_ids"])
        labels = list(full_encoding["input_ids"])
        labels[:prompt_length] = [-100] * min(prompt_length, len(labels))
        full_encoding["labels"] = labels
        tokenized.append(full_encoding)

    return tokenized


def load_model(model_name: str):
    """Load the base model and tokenizer."""
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",  # Causal LM: pad on right for generation
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if FP16 else torch.bfloat16 if BF16 else torch.float32,
        trust_remote_code=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )

    # Enable gradient checkpointing for memory efficiency
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Parameters: {model.num_parameters():,}")

    return model, tokenizer


def apply_lora(model):
    """Apply LoRA configuration to the model."""
    print(f"\nApplying LoRA configuration...")
    print(f"  r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class RankZeroProgressCallback:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(1, total_steps)
        self.start_time = None
        self.last_print = 0.0

    def on_train_begin(self) -> None:
        self.start_time = time.time()
        self.last_print = 0.0
        if _is_rank_zero():
            print(f"[Train] Total optimizer steps: {self.total_steps}")

    def on_step_end(self, step: int) -> None:
        if not _is_rank_zero():
            return
        now = time.time()
        if now - self.last_print < 30 and step < self.total_steps:
            return
        self.last_print = now
        elapsed = now - (self.start_time or now)
        completed = max(1, step)
        rate = elapsed / completed
        remaining_steps = max(0, self.total_steps - step)
        eta = rate * remaining_steps
        print(
            f"[Train] step {step}/{self.total_steps} | "
            f"elapsed {_format_seconds(elapsed)} | "
            f"eta {_format_seconds(eta)}"
        )


class _TrainingStatusCallback(TrainerCallback):
    def __init__(self, total_steps: int) -> None:
        self.progress = RankZeroProgressCallback(total_steps)

    def on_train_begin(self, args, state, control, **kwargs):
        self.progress.on_train_begin()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.progress.on_step_end(int(state.global_step))
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _is_rank_zero() or not logs:
            return control
        parts = []
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if parts:
            print(f"[Train] step {int(state.global_step)} | " + " | ".join(parts))
        return control


class TrainingStatusMonitor(threading.Thread):
    def __init__(self, trainer, total_steps: int, interval_seconds: int = 15) -> None:
        super().__init__(daemon=True)
        self.trainer = trainer
        self.total_steps = max(1, total_steps)
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.last_step = -1

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        if not _is_rank_zero():
            return
        print(f"[Train] Total optimizer steps: {self.total_steps}", flush=True)
        print(f"[Train] Status heartbeat every {self.interval_seconds}s", flush=True)
        while not self.stop_event.wait(self.interval_seconds):
            step = int(getattr(self.trainer.state, "global_step", 0))
            elapsed = time.time() - self.start_time
            if step <= 0:
                print(f"[Train] waiting for first step | elapsed {_format_seconds(elapsed)}", flush=True)
                continue
            if step == self.last_step:
                print(
                    f"[Train] step {step}/{self.total_steps} | "
                    f"elapsed {_format_seconds(elapsed)} | "
                    f"eta pending",
                    flush=True,
                )
                continue
            self.last_step = step
            rate = elapsed / max(1, step)
            remaining_steps = max(0, self.total_steps - step)
            eta = rate * remaining_steps
            print(
                f"[Train] step {step}/{self.total_steps} | "
                f"elapsed {_format_seconds(elapsed)} | "
                f"eta {_format_seconds(eta)}",
                flush=True,
            )


class SchemaLinkingTrainer:
    """Encapsulates the training pipeline."""

    def __init__(self, output_dir: str = OUTPUT_DIR, base_model: str = MODEL_NAME):
        self.output_dir = output_dir
        self.base_model = base_model

    def prepare_dataset(self, train_path: str, val_path: str, tokenizer):
        """Load and tokenize train/val datasets."""
        print(f"\nLoading dataset from: {train_path}")
        train_data = load_jsonl(train_path)
        print(f"  Train examples: {len(train_data)}")

        print(f"\nLoading validation set from: {val_path}")
        val_data = load_jsonl(val_path)
        print(f"  Val examples: {len(val_data)}")

        print(f"\nTokenizing training data...")
        train_tokenized = tokenize_dataset(train_data, tokenizer)

        print(f"Tokenizing validation data...")
        val_tokenized = tokenize_dataset(val_data, tokenizer)

        return train_tokenized, val_tokenized

    def create_trainer(self, model, train_data, val_data, tokenizer):
        """Create the HuggingFace Trainer instance."""
        steps_per_epoch = math.ceil(
            len(train_data)
            / max(1, PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        )
        total_steps = max(1, steps_per_epoch * NUM_TRAIN_EPOCHS)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            fp16=FP16,
            bf16=BF16,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            ddp_find_unused_parameters=False,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_num_workers=4,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=MAX_SEQ_LENGTH,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.add_callback(_TrainingStatusCallback(total_steps))

        return trainer

    def save_model(self, model, tokenizer):
        """Save the trained LoRA adapter.

        The adapter is lightweight (~50-200MB) and is loaded at inference
        time by attaching it to the base model via PeftModel.

        Output:
            <output_dir>/lora_adapter/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── tokenizer_config.json
            ├── training_config.json
        """
        lora_path = Path(self.output_dir) / "lora_adapter"
        print(f"\nSaving LoRA adapter to: {lora_path}")
        target_model = getattr(model, "module", model)
        target_model.save_pretrained(str(lora_path))
        tokenizer.save_pretrained(str(lora_path))

        # Save training config for reproducibility
        config_snapshot = {
            "base_model": self.base_model,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_TRAIN_EPOCHS,
            "batch_size": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
        }
        with open(lora_path / "training_config.json", "w") as f:
            json.dump(config_snapshot, f, indent=2)

        print(f"  Adapter saved to {lora_path}")
        print(f"  Load at inference with:")
        print(f"    from peft import PeftModel")
        print(f"    model = PeftModel.from_pretrained(base_model, '{lora_path}')")

    def train(self, train_path: str = OUTPUT_TRAIN_PATH,
              val_path: str = OUTPUT_VAL_PATH):
        """Run the full training pipeline."""
        print("=" * 60)
        print("Schema Linking Training Pipeline")
        print("=" * 60)

        # Load model
        model, tokenizer = load_model(self.base_model)

        # Apply LoRA
        model = apply_lora(model)

        # Prepare datasets
        train_data, val_data = self.prepare_dataset(
            train_path, val_path, tokenizer
        )

        # Create trainer
        trainer = self.create_trainer(model, train_data, val_data, tokenizer)
        status_monitor = None
        if _is_rank_zero():
            steps_per_epoch = math.ceil(
                len(train_data)
                / max(1, PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
            )
            total_steps = max(1, steps_per_epoch * NUM_TRAIN_EPOCHS)
            print(
                f"[Train] Planning for {NUM_TRAIN_EPOCHS} epochs | "
                f"train examples={len(train_data)} | "
                f"batch={PER_DEVICE_TRAIN_BATCH_SIZE} | "
                f"grad_accum={GRADIENT_ACCUMULATION_STEPS} | "
                f"steps/epoch≈{steps_per_epoch} | total_steps≈{total_steps}",
                flush=True,
            )
            status_monitor = TrainingStatusMonitor(trainer, total_steps)
            status_monitor.start()

        # Train
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}")

        try:
            trainer.train()
        finally:
            if status_monitor is not None:
                status_monitor.stop()
                status_monitor.join(timeout=5)

        # Save model
        if trainer.is_world_process_zero():
            self.save_model(model, tokenizer)

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")

        return trainer


def main():
    parser = argparse.ArgumentParser(description="Train the schema-linking LoRA adapter")
    parser.add_argument("--train-path", default=OUTPUT_TRAIN_PATH, help="Path to the training JSONL file")
    parser.add_argument("--val-path", default=OUTPUT_VAL_PATH, help="Path to the validation JSONL file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for trained adapter outputs")
    parser.add_argument("--base-model", default=MODEL_NAME, help="Base model name or local path")
    args = parser.parse_args()

    trainer = SchemaLinkingTrainer(output_dir=args.output_dir, base_model=args.base_model)
    trainer.train(train_path=args.train_path, val_path=args.val_path)


if __name__ == "__main__":
    main()

"""
Clean single-GPU training pipeline for Spider-style schema linking.

Goals:
- no torchrun / no DDP
- plain LoRA only
- eager attention, SDPA disabled
- prompt tokens masked from loss
- label padding uses -100
- preflight first-batch check before training
- progress logging at step level, not microbatch spam
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from config import (
    BF16,
    FP16,
    GRADIENT_CHECKPOINTING,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LR_SCHEDULER_TYPE,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    NUM_TRAIN_EPOCHS,
    OUTPUT_DIR,
    OUTPUT_TRAIN_PATH,
    OUTPUT_VAL_PATH,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
)


DEFAULT_MAX_SEQ_LENGTH = min(MAX_SEQ_LENGTH, 1024)
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_SAVE_STEPS = 250
DEFAULT_LOGGING_STEPS = 10
DEFAULT_HEARTBEAT_SECONDS = 30


if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if BF16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if FP16:
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def enforce_single_gpu_visibility() -> None:
    if not torch.cuda.is_available():
        return
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(
            "This trainer requires exactly one visible GPU. "
            f"Found {visible}. Re-run with CUDA_VISIBLE_DEVICES=<one_gpu_id>."
        )


def load_model(model_name: str, *, gradient_checkpointing: bool) -> Tuple[torch.nn.Module, AutoTokenizer]:
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
    )
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = DEFAULT_MAX_SEQ_LENGTH

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=load_dtype(),
        attn_implementation="eager",
        device_map={"": 0} if torch.cuda.is_available() else None,
    )

    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Parameters: {model.num_parameters():,}")
    return model, tokenizer


def apply_lora(model: torch.nn.Module) -> torch.nn.Module:
    print("\nApplying LoRA configuration...")
    print(f"  r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


class CausalLMDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = 8) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token.")
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of and max_length % self.pad_to_multiple_of:
            max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature["attention_mask"])
            labels = list(feature["labels"])

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids.extend([self.pad_token_id] * pad_len)
                attention_mask.extend([0] * pad_len)
                labels.extend([-100] * pad_len)

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


def format_prompt(entry: Dict) -> Tuple[str, str]:
    prompt = str(entry.get("input", "")).strip()
    output = str(entry.get("output", "")).strip()
    if not prompt or not output:
        raise ValueError("Each row must include non-empty 'input' and 'output' fields.")
    return prompt, output


def tokenize_dataset(data: List[Dict], tokenizer: AutoTokenizer, max_length: int) -> List[Dict]:
    tokenized: List[Dict] = []
    skipped = 0

    for index, entry in enumerate(data):
        try:
            prompt_text, output_text = format_prompt(entry)
        except ValueError:
            skipped += 1
            continue

        full_text = f"{prompt_text}\n\n{output_text}"

        prompt_encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        full_encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        input_ids = list(full_encoding["input_ids"])
        attention_mask = list(full_encoding["attention_mask"])
        prompt_length = len(prompt_encoding["input_ids"])

        if len(input_ids) <= prompt_length:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: skipped example {index} because the output was truncated away.")
            continue

        labels = list(input_ids)
        labels[: min(prompt_length, len(labels))] = [-100] * min(prompt_length, len(labels))

        if not any(label != -100 for label in labels):
            skipped += 1
            continue

        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    if skipped:
        print(f"  Tokenization skipped {skipped} example(s).")
    return tokenized


class ProgressCallback(TrainerCallback):
    def __init__(self, total_steps: int, heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS) -> None:
        self.total_steps = max(1, total_steps)
        self.heartbeat_seconds = heartbeat_seconds
        self.start_time = 0.0
        self.last_heartbeat = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        if is_rank_zero():
            self.start_time = time.time()
            self.last_heartbeat = self.start_time
            print(f"[Train] Total optimizer steps: {self.total_steps}", flush=True)
            print(f"[Train] Status heartbeat every {self.heartbeat_seconds}s", flush=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not is_rank_zero():
            return control
        now = time.time()
        if now - self.last_heartbeat < self.heartbeat_seconds and state.global_step < self.total_steps:
            return control
        self.last_heartbeat = now
        elapsed = now - self.start_time
        completed = max(1, int(state.global_step))
        rate = elapsed / completed
        remaining = max(0, self.total_steps - completed)
        eta = rate * remaining
        print(
            f"[Train] step {completed}/{self.total_steps} | "
            f"elapsed {fmt_seconds(elapsed)} | eta {fmt_seconds(eta)}",
            flush=True,
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_rank_zero() or not logs:
            return control
        parts = []
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={logs['grad_norm']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.3f}")
        if parts:
            print(f"[Train] log | " + " | ".join(parts), flush=True)
        return control


def preflight_first_batch(
    model: torch.nn.Module,
    dataset: List[Dict],
    collator: CausalLMDataCollator,
) -> None:
    if not dataset:
        raise ValueError("No training examples remain after tokenization.")

    device = get_model_device(model)
    batch = collator([dataset[0]])
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    model.train()
    model.zero_grad(set_to_none=True)
    start = time.time()

    outputs = model(**batch)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError("Preflight forward pass did not return a loss.")
    if not torch.isfinite(loss):
        raise RuntimeError(f"Preflight loss is not finite: {loss.item()}")

    loss.backward()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(f"[Preflight] First batch forward/backward passed in {time.time() - start:.2f}s.")


class SchemaLinkingTrainer:
    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        base_model: str = MODEL_NAME,
        *,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs: int = NUM_TRAIN_EPOCHS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        warmup_ratio: float = WARMUP_RATIO,
        lr_scheduler_type: str = LR_SCHEDULER_TYPE,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        save_steps: int = DEFAULT_SAVE_STEPS,
        logging_steps: int = DEFAULT_LOGGING_STEPS,
        gradient_checkpointing: bool = False,
        run_final_eval: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.base_model = base_model
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.max_seq_length = max_seq_length
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.run_final_eval = run_final_eval

    def prepare_dataset(self, train_path: str, val_path: str, tokenizer: AutoTokenizer):
        print(f"\nLoading dataset from: {train_path}")
        train_data = load_jsonl(train_path)
        print(f"  Train examples: {len(train_data)}")

        print(f"\nLoading validation set from: {val_path}")
        val_data = load_jsonl(val_path)
        print(f"  Val examples: {len(val_data)}")

        print("\nTokenizing training data...")
        train_tokenized = tokenize_dataset(train_data, tokenizer, self.max_seq_length)
        print("Tokenizing validation data...")
        val_tokenized = tokenize_dataset(val_data, tokenizer, self.max_seq_length)

        print(f"  Tokenized train examples: {len(train_tokenized)} | Tokenized val examples: {len(val_tokenized)}")
        return train_tokenized, val_tokenized

    def create_trainer(self, model, train_data, val_data, tokenizer):
        use_bf16 = torch.cuda.is_available() and BF16 and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and FP16 and not use_bf16

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=self.logging_steps,
            logging_first_step=True,
            save_steps=self.save_steps,
            save_strategy="steps",
            eval_strategy="no",
            load_best_model_at_end=False,
            save_total_limit=2,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
            group_by_length=False,
            optim="adamw_torch",
            gradient_checkpointing=self.gradient_checkpointing,
        )

        collator = CausalLMDataCollator(tokenizer, pad_to_multiple_of=8)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data if val_data else None,
            data_collator=collator,
        )
        return trainer, collator

    def save_model(self, model, tokenizer):
        lora_path = Path(self.output_dir) / "lora_adapter"
        lora_path.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving LoRA adapter to: {lora_path}")
        target_model = getattr(model, "module", model)
        target_model.save_pretrained(str(lora_path), safe_serialization=True)
        tokenizer.save_pretrained(str(lora_path))

        snapshot = {
            "base_model": self.base_model,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_seq_length": self.max_seq_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
        }
        with open(lora_path / "training_config.json", "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)

        print("  Adapter saved.")

    def train(self, train_path: str = OUTPUT_TRAIN_PATH, val_path: str = OUTPUT_VAL_PATH):
        enforce_single_gpu_visibility()

        print("=" * 60)
        print("Schema Linking Training Pipeline")
        print("=" * 60)
        print(f"Base model: {self.base_model}")
        print(f"Train path: {train_path}")
        print(f"Val path: {val_path}")
        print(f"Output dir: {self.output_dir}")
        print(f"Max seq length: {self.max_seq_length}")
        print(f"Per-device train batch size: {self.train_batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print("Gradient checkpointing: disabled")
        print("SDPA/Flash attention: disabled, eager attention forced")

        if self.train_batch_size != 1:
            print("[Warn] batch size is not 1; if you see stalls, try --train-batch-size 1.")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_model(self.base_model, gradient_checkpointing=self.gradient_checkpointing)
        model = apply_lora(model)

        train_data, val_data = self.prepare_dataset(train_path, val_path, tokenizer)
        trainer, collator = self.create_trainer(model, train_data, val_data, tokenizer)

        steps_per_epoch = math.ceil(
            max(1, len(train_data)) / max(1, self.train_batch_size * self.gradient_accumulation_steps)
        )
        total_steps = max(1, steps_per_epoch * self.num_train_epochs)
        print(
            f"[Train] Planned steps: {total_steps} "
            f"({steps_per_epoch} per epoch x {self.num_train_epochs} epochs)"
        )

        preflight_first_batch(model, train_data, collator)

        print(f"\n{'=' * 60}")
        print("Starting training...")
        print(f"{'=' * 60}")

        trainer.add_callback(ProgressCallback(total_steps))
        result = trainer.train()

        if self.run_final_eval and val_data:
            print("\nRunning final evaluation...")
            metrics = trainer.evaluate(eval_dataset=val_data)
            print(f"Final eval metrics: {metrics}")

        self.save_model(model, tokenizer)

        print(f"\n{'=' * 60}")
        print("Training complete!")
        print(f"{'=' * 60}")
        return result


def main():
    parser = argparse.ArgumentParser(description="Train the Spider schema-linking LoRA adapter")
    parser.add_argument("--train-path", default=OUTPUT_TRAIN_PATH)
    parser.add_argument("--val-path", default=OUTPUT_VAL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--lr-scheduler-type", default=LR_SCHEDULER_TYPE)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--run-final-eval", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    trainer = SchemaLinkingTrainer(
        output_dir=args.output_dir,
        base_model=args.base_model,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_checkpointing=False,
        run_final_eval=args.run_final_eval,
    )
    trainer.train(train_path=args.train_path, val_path=args.val_path)


if __name__ == "__main__":
    main()

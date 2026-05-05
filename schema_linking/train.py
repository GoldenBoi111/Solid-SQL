"""
Single-GPU training pipeline for Spider-style schema linking.

This version is intentionally conservative:
- no torchrun
- no DDP
- no multi-GPU device_map juggling
- prompt tokens are masked out of the loss
- labels are padded with -100
- plain LoRA only
- a preflight forward/backward check runs before full training

The trainer expects JSONL rows with:
- input
- output
- optionally db_id, question_id, difficulty, evidence
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from config import (
    BF16,
    FP16,
    GRADIENT_ACCUMULATION_STEPS,
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
DEFAULT_SAVE_STEPS = 250
DEFAULT_LOGGING_STEPS = 10
DEFAULT_EVAL_STEPS = 250


def load_jsonl(path: str) -> List[Dict]:
    entries: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_prompt(entry: Dict) -> Tuple[str, str]:
    prompt = entry.get("input", "").strip()
    output = entry.get("output", "").strip()
    if not prompt or not output:
        raise ValueError("Each training row must contain non-empty 'input' and 'output' fields.")
    return prompt, output


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if BF16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if FP16:
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(
    model_name: str,
    *,
    gradient_checkpointing: bool,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = DEFAULT_MAX_SEQ_LENGTH

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    compute_dtype = get_compute_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=compute_dtype,
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


class CausalLMDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token before training.")

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of and max_length % self.pad_to_multiple_of:
            max_length = (
                (max_length // self.pad_to_multiple_of) + 1
            ) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = list(feature["input_ids"])[:max_length]
            attention_mask = list(feature["attention_mask"])[:max_length]
            labels = list(feature["labels"])[:max_length]

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


def tokenize_dataset(
    data: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> List[Dict]:
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
                print(
                    f"  Warning: skipped example {index} because the output was truncated away."
                )
            continue

        labels = list(input_ids)
        mask_length = min(prompt_length, len(labels))
        labels[:mask_length] = [-100] * mask_length

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


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def preflight_first_batch(
    model: torch.nn.Module,
    dataset: List[Dict],
    collator: CausalLMDataCollator,
) -> None:
    if not dataset:
        raise ValueError("No training examples remain after tokenization.")

    device = get_model_device(model)
    sample = dataset[0]
    batch = collator([sample])
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    model.train()
    model.zero_grad(set_to_none=True)

    start_time = time.time()
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

    elapsed = time.time() - start_time
    print(f"[Preflight] First batch forward/backward passed in {elapsed:.2f}s.")


class DebugTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._microbatch_index = 0
        self._microbatch_start_time = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._microbatch_index += 1
        if is_rank_zero():
            input_shape = tuple(inputs["input_ids"].shape) if "input_ids" in inputs else ()
            print(
                f"[Batch] enter microbatch {self._microbatch_index} | "
                f"global_step={int(getattr(self.state, 'global_step', 0))} | "
                f"shape={input_shape}",
                flush=True,
            )
        self._microbatch_start_time = time.time()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if is_rank_zero():
            elapsed = time.time() - (self._microbatch_start_time or time.time())
            loss_value = loss.detach().float().item() if torch.is_tensor(loss) else float(loss)
            print(
                f"[Batch] exit microbatch {self._microbatch_index} | "
                f"loss={loss_value:.4f} | "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )
        return loss


class SchemaLinkingTrainer:
    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        base_model: str = MODEL_NAME,
        *,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        num_train_epochs: int = NUM_TRAIN_EPOCHS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        warmup_ratio: float = WARMUP_RATIO,
        lr_scheduler_type: str = LR_SCHEDULER_TYPE,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        save_steps: int = DEFAULT_SAVE_STEPS,
        logging_steps: int = DEFAULT_LOGGING_STEPS,
        eval_steps: int = DEFAULT_EVAL_STEPS,
        gradient_checkpointing: bool = GRADIENT_CHECKPOINTING,
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
        self.eval_steps = eval_steps
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

        print(
            f"  Tokenized train examples: {len(train_tokenized)} | "
            f"Tokenized val examples: {len(val_tokenized)}"
        )
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

        data_collator = CausalLMDataCollator(tokenizer, pad_to_multiple_of=8)

        trainer = DebugTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data if val_data else None,
            data_collator=data_collator,
        )
        return trainer, data_collator

    def save_model(self, model, tokenizer):
        lora_path = Path(self.output_dir) / "lora_adapter"
        lora_path.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving LoRA adapter to: {lora_path}")
        target_model = getattr(model, "module", model)
        target_model.save_pretrained(str(lora_path), safe_serialization=True)
        tokenizer.save_pretrained(str(lora_path))

        config_snapshot = {
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
            json.dump(config_snapshot, handle, indent=2)

        print(f"  Adapter saved to {lora_path}")
        print("  Load at inference with:")
        print("    from peft import PeftModel")
        print(f"    model = PeftModel.from_pretrained(base_model, '{lora_path}')")

    def train(self, train_path: str = OUTPUT_TRAIN_PATH, val_path: str = OUTPUT_VAL_PATH):
        print("=" * 60)
        print("Schema Linking Training Pipeline")
        print("=" * 60)
        print(f"Base model: {self.base_model}")
        print(f"Train path: {train_path}")
        print(f"Val path: {val_path}")
        print(f"Output dir: {self.output_dir}")
        print(f"Max seq length: {self.max_seq_length}")
        print("4-bit loading: disabled (plain LoRA)")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_model(
            self.base_model,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        model = apply_lora(model)

        train_data, val_data = self.prepare_dataset(train_path, val_path, tokenizer)
        trainer, data_collator = self.create_trainer(model, train_data, val_data, tokenizer)

        steps_per_epoch = math.ceil(
            max(1, len(train_data))
            / max(1, self.train_batch_size * self.gradient_accumulation_steps)
        )
        total_steps = max(1, steps_per_epoch * self.num_train_epochs)
        print(
            f"[Train] Planned steps: {total_steps} "
            f"({steps_per_epoch} per epoch x {self.num_train_epochs} epochs)"
        )

        preflight_first_batch(model, train_data, data_collator)

        print(f"\n{'=' * 60}")
        print("Starting training...")
        print(f"{'=' * 60}")

        train_result = trainer.train()

        if self.run_final_eval and val_data:
            print("\nRunning final evaluation...")
            eval_metrics = trainer.evaluate(eval_dataset=val_data)
            print(f"Final eval metrics: {eval_metrics}")

        self.save_model(model, tokenizer)

        print(f"\n{'=' * 60}")
        print("Training complete!")
        print(f"{'=' * 60}")
        return train_result


def main():
    parser = argparse.ArgumentParser(description="Train the Spider schema-linking LoRA adapter")
    parser.add_argument("--train-path", default=OUTPUT_TRAIN_PATH, help="Path to the training JSONL file")
    parser.add_argument("--val-path", default=OUTPUT_VAL_PATH, help="Path to the validation JSONL file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for trained adapter outputs")
    parser.add_argument("--base-model", default=MODEL_NAME, help="Base model name or local path")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Maximum token length")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Per-device evaluation batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO, help="Warmup ratio")
    parser.add_argument(
        "--lr-scheduler-type",
        default=LR_SCHEDULER_TYPE,
        help="Learning-rate scheduler type",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help="How often to save checkpoints",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=DEFAULT_LOGGING_STEPS,
        help="How often to log training metrics",
    )
    parser.add_argument(
        "--run-final-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run evaluation on the validation set after training",
    )
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
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        run_final_eval=args.run_final_eval,
    )
    trainer.train(train_path=args.train_path, val_path=args.val_path)


if __name__ == "__main__":
    main()

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
from pathlib import Path
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
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
) -> Dict:
    """
    Tokenize the dataset.

    For causal LM training, we tokenize the full combined string
    (input + output). The model learns to predict output tokens
    given the input prefix via standard next-token prediction.
    """
    texts = [format_prompt(entry) for entry in data]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Handled by data collator
        return_tensors=None,  # Return list of dicts for Dataset
    )

    # Add labels (same as input_ids for causal LM — loss is computed
    # only on the non-padded portion)
    tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]

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
        torch_dtype=torch.float16 if FP16 else torch.bfloat16 if BF16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
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


class SchemaLinkingTrainer:
    """Encapsulates the training pipeline."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir

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
        model.save_pretrained(str(lora_path))
        tokenizer.save_pretrained(str(lora_path))

        # Save training config for reproducibility
        config_snapshot = {
            "base_model": MODEL_NAME,
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
        model, tokenizer = load_model(MODEL_NAME)

        # Apply LoRA
        model = apply_lora(model)

        # Prepare datasets
        train_data, val_data = self.prepare_dataset(
            train_path, val_path, tokenizer
        )

        # Create trainer
        trainer = self.create_trainer(model, train_data, val_data, tokenizer)

        # Train
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}")

        trainer.train()

        # Save model
        self.save_model(model, tokenizer)

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")

        return trainer


def main():
    trainer = SchemaLinkingTrainer()
    trainer.train()


if __name__ == "__main__":
    main()

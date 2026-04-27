"""
Schema Linking Inference (Outlines + LoRA)

Uses Outlines with a single model instance that dynamically
loads/unloads the LoRA adapter for schema linking.

Usage:
    from schema_linking.inference import SchemaLinker

    linker = SchemaLinker(
        base_model="openai/gpt-oss-20b",
        adapter_path="./schema_linking_output/lora_adapter",
    )

    # Schema linking with LoRA
    result = linker.predict(
        question="How many singers are older than 20?",
        schema_text="Singer(id, name, age)\nAlbum(id, singer_id, title)",
    )

    # SQL generation without LoRA
    outputs = linker.generate_without_lora(
        ["Generate SQL for: ..."],
        max_new_tokens=512,
    )
"""

from pathlib import Path
from typing import List, Dict, Optional
import re

import outlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    MODEL_NAME,
    OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    INSTRUCTION_TEMPLATE,
    LORA_R,
)
from .schema_formatter import format_schema_compact, load_schemas_from_dir


def _resolve_generation_model(model: object) -> object:
    """Return the deepest nested model object suitable for HF-style `generate()`."""
    stack = [model]
    seen = set()
    best_candidate = None

    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        has_generate = hasattr(current, "generate")
        has_parameters = hasattr(current, "parameters")
        if has_generate and has_parameters:
            best_candidate = current

        for attr in ("model", "language_model", "base_model", "module"):
            nested = getattr(current, attr, None)
            if nested is not None and id(nested) not in seen:
                stack.append(nested)

    if best_candidate is not None:
        return best_candidate

    raise AttributeError(
        f"Could not find a model with both generate() and parameters() on type {type(model).__name__}"
    )


def _resolve_parameter_model(model: object) -> object:
    """Return the first nested model object that exposes `parameters()`."""
    current = model
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "parameters"):
            return current

        for attr in ("model", "language_model", "base_model", "module"):
            nested = getattr(current, attr, None)
            if nested is not None and id(nested) not in seen:
                current = nested
                break
        else:
            current = None

    raise AttributeError(f"Could not find a parameters() method on model type {type(model).__name__}")


def _clean_generated_text(text: str) -> str:
    """Normalize generated text and strip markdown fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_section_lines(text: str, heading: str) -> List[str]:
    """Extract bullet/content lines that belong to a named section."""
    lines = text.splitlines()
    collected = []
    in_section = False
    target = heading.lower()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_section and collected:
                continue
            continue

        if line.endswith(":"):
            if line[:-1].strip().lower() == target:
                in_section = True
                collected = []
                continue
            if in_section:
                break

        if in_section:
            collected.append(line)

    return collected


def _parse_schema_linking_response(text: str) -> Dict[str, object]:
    """Parse the schema-linking note into structured fields."""
    cleaned = _clean_generated_text(text)

    tables = []
    for line in _extract_section_lines(cleaned, "Relevant Tables"):
        item = line[1:].strip() if line.startswith("-") else line
        if item:
            tables.append(item)

    columns = {}
    for line in _extract_section_lines(cleaned, "Relevant Columns"):
        item = line[1:].strip() if line.startswith("-") else line
        if not item or ":" not in item:
            continue
        table_name, raw_columns = item.split(":", 1)
        column_names = [
            col.strip()
            for col in re.split(r",\s*", raw_columns.strip())
            if col.strip()
        ]
        if table_name.strip() and column_names:
            columns[table_name.strip()] = column_names

    join_relationships = []
    for line in _extract_section_lines(cleaned, "Join Relationships"):
        item = line[1:].strip() if line.startswith("-") else line
        if item:
            join_relationships.append(item)

    filters_constraints = []
    for line in _extract_section_lines(cleaned, "Filters / Constraints"):
        item = line[1:].strip() if line.startswith("-") else line
        if item:
            filters_constraints.append(item)

    question_intent_lines = _extract_section_lines(cleaned, "Question Intent")
    question_intent = ""
    for line in question_intent_lines:
        item = line[1:].strip() if line.startswith("-") else line
        if item:
            question_intent = item
            break

    return {
        "tables": tables,
        "columns": columns,
        "join_relationships": join_relationships,
        "filters_constraints": filters_constraints,
        "question_intent": question_intent,
        "raw_output": cleaned,
    }


class SchemaLinker:
    """Uses a single Outlines model with dynamic LoRA loading."""

    def __init__(
        self,
        base_model: str = MODEL_NAME,
        adapter_path: str = "",
        max_seq_length: int = MAX_SEQ_LENGTH,
    ):
        """
        Initialize the schema linker.

        Args:
            base_model: Hugging Face model name or local path
            adapter_path: Path to the fine-tuned LoRA adapter directory
            max_seq_length: Maximum sequence length (prompt + output)
        """
        if not adapter_path:
            adapter_path = str(Path(OUTPUT_DIR) / "lora_adapter")

        self.adapter_path = Path(adapter_path)
        self.has_adapter = (
            self.adapter_path.is_dir()
            and (self.adapter_path / "adapter_config.json").exists()
        )

        self.base_model_name = base_model
        self.max_seq_length = max_seq_length
        self._model = None
        self._tokenizer = None
        self._lora_loaded = False  # True once the adapter has been loaded (never resets)
        self._lora_active = False   # True when adapter is currently applied

        print(f"\n{'='*60}")
        print(f"Schema Linker Initialization (Outlines)")
        print(f"{'='*60}")
        print(f"  Base model: {base_model}")
        print(f"  Adapter:    {adapter_path if self.has_adapter else 'None'}")
        print(f"  Max seq len: {max_seq_length}")
        print(f"{'='*60}\n")

    def _load_model(self):
        """Load the base model and tokenizer."""
        if self._model is None:
            print("Loading base model...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
            )
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token

            if self.has_adapter and not self._lora_loaded:
                print("Loading LoRA adapter (first time)...")
                hf_model.load_adapter(
                    str(self.adapter_path),
                    adapter_name="lora_adapter",
                )
                self._lora_loaded = True
                self._lora_active = True

            self._model = outlines.from_transformers(hf_model, hf_tokenizer)
            self._tokenizer = hf_tokenizer
            self._model.model.eval()
            print("Base model loaded.")

    def _load_lora(self):
        """Activate the LoRA adapter (already loaded in _load_model)."""
        if not self.has_adapter:
            return
        
        # Adapter is already loaded in _load_model, just activate it
        self._model.model.enable_adapters()
        self._model.model.set_adapter("lora_adapter")
        self._lora_active = True

    def _unload_lora(self):
        """Disable the LoRA adapter (keep it loaded but don't use it)."""
        if self._lora_active:
            # Disable the active adapter without unloading its weights
            self._model.model.disable_adapters()
            self._lora_active = False
            print("LoRA adapter disabled (kept in memory).")

    def _format_prompt(self, question: str, schema_text: str) -> str:
        """Format the input prompt using the instruction template."""
        return INSTRUCTION_TEMPLATE.format(
            question=question,
            schema_text=schema_text,
        )

    def predict(
        self,
        question: str,
        schema_text: str,
        max_new_tokens: int = 4096,
    ) -> Dict:
        """
        Generate schema linking prediction for a single question.

        Loads the LoRA adapter, generates, then unloads it.
        """
        results = self.predict_batch(
            [{"question": question, "schema_text": schema_text}],
            max_new_tokens=max_new_tokens,
        )
        return results[0] if results else {}

    def predict_batch(
        self,
        inputs: List[Dict[str, str]],
        max_new_tokens: int = 4096,
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate predictions for multiple questions.

        Loads LoRA adapter once, generates all batches, then unloads.
        """
        # Load base model
        self._load_model()

        # Load LoRA adapter
        self._load_lora()

        # Format all prompts
        prompts = [
            self._format_prompt(item["question"], item["schema_text"])
            for item in inputs
        ]

        model = _resolve_generation_model(self._model)
        parameter_model = _resolve_parameter_model(self._model)
        device = next(parameter_model.parameters()).device
        all_results = []
        total = len(prompts)

        for i in range(0, total, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            tokenized = self._tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
            for output, prompt_length in zip(outputs, prompt_lengths):
                generated_tokens = output[int(prompt_length):]
                text = self._tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                ).strip()
                all_results.append(_parse_schema_linking_response(text))

            if show_progress:
                processed = min(i + batch_size, total)
                print(f"  Processed {processed}/{total}")

        # Keep LoRA adapter loaded - don't unload to avoid memory fragmentation
        # The adapter weights are small compared to the base model

        return all_results

    def generate_without_lora(
        self,
        prompts: List[str],
        max_new_tokens: int = 4096,
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> List[str]:
        """Generate raw text using the base model without the LoRA adapter."""
        # Load base model (without LoRA)
        self._load_model()

        # Ensure LoRA is not active
        if self._lora_active:
            self._unload_lora()

        all_outputs = []
        total = len(prompts)
        model = _resolve_generation_model(self._model)
        parameter_model = _resolve_parameter_model(self._model)
        device = next(parameter_model.parameters()).device

        for i in range(0, total, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = self._tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for output, prompt_length in zip(outputs, prompt_lengths):
                generated_tokens = output[int(prompt_length):]
                text = self._tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                ).strip()
                all_outputs.append(text)

            if show_progress:
                processed = min(i + batch_size, total)
                print(f"  Processed {processed}/{total}")

        return all_outputs

    # Warning: calling shutdown() resets _model to None, so a later
    # predict() or generate_without_lora() call will trigger a full reload.
    def shutdown(self):
        """Shut down the model to free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._lora_loaded = False
        self._lora_active = False

    def predict_from_db_id(
        self,
        question: str,
        db_id: str,
        db_root: str,
        max_new_tokens: int = 4096,
    ) -> Dict:
        """
        Load schema from db_id + db_root and predict.
        """
        schemas = load_schemas_from_dir(db_root)
        schema = schemas.get(db_id)

        if schema:
            schema_text = format_schema_compact(schema)
            return self.predict(question, schema_text, max_new_tokens)

        db_path = Path(db_root) / f"{db_id}.sqlite"
        if not db_path.exists():
            return {"error": f"No schema found for db_id '{db_id}' at {db_root}"}

        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = cursor.fetchall()
        parts = []
        for (tbl,) in tables:
            cursor.execute(f"PRAGMA table_info('{tbl}');")
            cols = cursor.fetchall()
            col_str = ", ".join(c[1] for c in cols)
            parts.append(f"{tbl}({col_str})")
        schema_text = "\n".join(parts)
        conn.close()

        return self.predict(question, schema_text, max_new_tokens)

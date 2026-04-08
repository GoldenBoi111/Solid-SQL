"""
Schema Linking Inference

Uses the fine-tuned LoRA adapter to predict relevant tables and columns
from a natural language question and database schema.

Usage:
    from schema_linking.inference import SchemaLinker

    linker = SchemaLinker(
        base_model="openai/gpt-oss-20b",
        adapter_path="./schema_linking_output/lora_adapter",
    )

    # Single prediction
    result = linker.predict(
        question="How many singers are older than 20?",
        schema_text="Singer(id, name, age)\nAlbum(id, singer_id, title)",
    )
    # Returns: {"tables": [...], "columns": [...]}

    # Batch prediction
    results = linker.predict_batch([
        {"question": "...", "schema_text": "..."},
        {"question": "...", "schema_text": "..."},
    ])
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    MODEL_NAME, OUTPUT_DIR, MAX_SEQ_LENGTH,
    INSTRUCTION_TEMPLATE,
)
from schema_formatter import format_schema_compact, load_schemas_from_dir


class SchemaLinker:
    """Loads a fine-tuned LoRA adapter and generates schema linking predictions."""

    def __init__(
        self,
        base_model: str = MODEL_NAME,
        adapter_path: str = "",
        device: str = "",
        max_seq_length: int = MAX_SEQ_LENGTH,
    ):
        """
        Initialize the schema linker.

        Args:
            base_model: Hugging Face model name or local path for the base model
            adapter_path: Path to the fine-tuned LoRA adapter directory
            device: Device override (e.g., "cuda:0"). Auto-detected if empty.
            max_seq_length: Maximum sequence length for tokenization
        """
        if not adapter_path:
            adapter_path = str(Path(OUTPUT_DIR) / "lora_adapter")

        self.adapter_path = Path(adapter_path)
        self.base_model_name = base_model
        self.max_seq_length = max_seq_length

        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"\n{'='*60}")
        print(f"Schema Linker Initialization")
        print(f"{'='*60}")
        print(f"  Base model: {base_model}")
        print(f"  Adapter:    {adapter_path}")
        print(f"  Device:     {self.device}")
        print(f"  Max seq len: {max_seq_length}")
        print(f"{'='*60}\n")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if "cuda" in self.device else None,
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
        self.model.eval()

        print(f"Model and adapter loaded successfully")

    def _format_prompt(self, question: str, schema_text: str) -> str:
        """Format the input prompt using the instruction template."""
        return INSTRUCTION_TEMPLATE.format(
            question=question,
            schema_text=schema_text,
        )

    def _parse_output(self, generated: str) -> Dict:
        """
        Parse the model output into structured JSON.

        Handles:
        - Raw JSON in the response
        - JSON wrapped in markdown fences
        - JSON followed by trailing text
        """
        # Trim to only the first complete JSON object
        generated = generated.strip()

        # Remove markdown fences
        if generated.startswith("```json"):
            generated = generated[7:]
        elif generated.startswith("```"):
            generated = generated[3:]
        if generated.endswith("```"):
            generated = generated[:-3].strip()

        # Find JSON boundaries
        start = generated.find("{")
        if start == -1:
            return {"error": "No JSON found", "raw": generated[:500]}

        end = generated.rfind("}") + 1
        if end <= start:
            return {"error": "Incomplete JSON", "raw": generated[:500]}

        json_str = generated[start:end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": json_str[:500]}

    @torch.no_grad()
    def predict(
        self,
        question: str,
        schema_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> Dict:
        """
        Generate schema linking prediction for a single question.

        Args:
            question: Natural language question
            schema_text: Database schema in text format
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (low for deterministic output)
            top_p: Nucleus sampling threshold

        Returns:
            Dict with structured output:
            {
                "tables": [{"name": "...", "reason": "..."}],
                "columns": [{"name": "...", "reason": "..."}]
            }
        """
        prompt = self._format_prompt(question, schema_text)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the generated tokens (skip the prompt)
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return self._parse_output(generated)

    @torch.no_grad()
    def predict_batch(
        self,
        inputs: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate schema linking predictions for multiple questions.

        Args:
            inputs: List of dicts with "question" and "schema_text" keys
            max_new_tokens: Maximum tokens to generate per item
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            batch_size: Number of items per batch
            show_progress: Print progress indicator

        Returns:
            List of structured output dicts (same order as input)
        """
        results = []
        total = len(inputs)

        for i in range(0, total, batch_size):
            batch = inputs[i : i + batch_size]

            for item in batch:
                question = item.get("question", "")
                schema_text = item.get("schema_text", "")
                result = self.predict(question, schema_text, max_new_tokens, temperature, top_p)
                results.append(result)

            if show_progress:
                processed = min(i + batch_size, total)
                print(f"  Processed {processed}/{total}")

        return results

    def predict_from_db_id(
        self,
        question: str,
        db_id: str,
        db_root: str,
        max_new_tokens: int = 512,
    ) -> Dict:
        """
        Convenience method: load schema from db_id + db_root and predict.

        Tries to load schema from JSON files first, then falls back to
        reading directly from SQLite if no JSON schema is found.

        Args:
            question: Natural language question
            db_id: Database identifier (e.g., "singer")
            db_root: Path to directory containing schema JSON files or SQLite databases
            max_new_tokens: Maximum tokens to generate

        Returns:
            Structured output dict
        """
        # Try loading from JSON schema files first
        schemas = load_schemas_from_dir(db_root)
        schema = schemas.get(db_id)

        if schema:
            schema_text = format_schema_compact(schema)
            return self.predict(question, schema_text, max_new_tokens)

        # Fall back to loading from SQLite database directly
        db_path = Path(db_root) / f"{db_id}.sqlite"
        if not db_path.exists():
            return {"error": f"No schema found for db_id '{db_id}' at {db_root}"}

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
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

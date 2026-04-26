"""
Schema Linking Inference (Transformers + LoRA)

Uses Hugging Face transformers with a single model instance that
dynamically loads/unloads LoRA adapter for schema linking.

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

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    MODEL_NAME,
    OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    INSTRUCTION_TEMPLATE,
    OUTPUT_SCHEMA,
    LORA_R,
)
from .schema_formatter import format_schema_compact, load_schemas_from_dir


class SchemaLinker:
    """Uses a single transformers model with dynamic LoRA loading."""

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
        self._lora_loaded = False

        print(f"\n{'='*60}")
        print(f"Schema Linker Initialization (Transformers)")
        print(f"{'='*60}")
        print(f"  Base model: {base_model}")
        print(f"  Adapter:    {adapter_path if self.has_adapter else 'None'}")
        print(f"  Max seq len: {max_seq_length}")
        print(f"{'='*60}\n")

    def _load_model(self):
        """Load the base model and tokenizer."""
        if self._model is None:
            print("Loading base model...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
            )

            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._model.eval()
            print("Base model loaded.")

    def _load_lora(self):
        """Load the LoRA adapter into the model."""
        if self.has_adapter:
            if not self._lora_loaded:
                print("Loading LoRA adapter...")
                from peft import PeftModel

                # Load adapter
                self._model.load_adapter(
                    str(self.adapter_path),
                    adapter_name="lora_adapter",
                )
                # Set active adapter to use LoRA
                self._model.set_adapter(["base_model", "lora_adapter"])
                self._lora_loaded = True
                print("LoRA adapter loaded.")
            else:
                # Adapter already loaded, just activate it
                print("LoRA adapter already loaded, activating...")
                self._model.set_adapter(["base_model", "lora_adapter"])

    def _unload_lora(self):
        """Unload the LoRA adapter by merging weights."""
        if self._lora_loaded:
            print("Merging LoRA weights into base model...")
            from peft import PeftModel

            # Merge LoRA weights into base model
            # This returns a new model, but we keep the same reference
            merged_model = self._model.merge_and_unload()

            # Copy state dict back to original model to maintain reference
            self._model.load_state_dict(merged_model.state_dict(), strict=False)
            del merged_model

            self._lora_loaded = False
            print("LoRA adapter merged into base model.")

    def _format_prompt(self, question: str, schema_text: str) -> str:
        """Format the input prompt using the instruction template."""
        return INSTRUCTION_TEMPLATE.format(
            question=question,
            schema_text=schema_text,
        )

    def _create_sampling_params(self, max_new_tokens: int):
        """Create generation parameters."""
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

    def predict(
        self,
        question: str,
        schema_text: str,
        max_new_tokens: int = 512,
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
        max_new_tokens: int = 512,
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

        # Tokenize
        inputs_tokenized = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self._model.device)

        # Generate
        all_results = []
        total = len(prompts)

        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_inputs = {
                    k: v[i : i + batch_size] for k, v in inputs_tokenized.items()
                }

                outputs = self._model.generate(
                    **batch_inputs,
                    **self._create_sampling_params(max_new_tokens),
                )

                for output in outputs:
                    response = self._tokenizer.decode(output, skip_special_tokens=True)
                    parsed = self._parse_json_response(response)
                    all_results.append(parsed)

                if show_progress:
                    processed = min(i + batch_size, total)
                    print(f"  Processed {processed}/{total}")

        # Keep LoRA adapter loaded - don't unload to avoid memory fragmentation
        # The adapter weights are small compared to the base model

        return all_results

    def generate_without_lora(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate text using the base model WITHOUT LoRA adapter.

        Args:
            prompts: List of prompt strings to generate from
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for generation
            show_progress: Whether to show progress

        Returns:
            List of generated text strings
        """
        # Load base model (without LoRA)
        self._load_model()

        # Ensure LoRA is not loaded
        if self._lora_loaded:
            self._unload_lora()

        # Tokenize
        inputs_tokenized = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self._model.device)

        # Generate
        all_outputs = []
        total = len(prompts)

        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_inputs = {
                    k: v[i : i + batch_size] for k, v in inputs_tokenized.items()
                }

                outputs = self._model.generate(
                    **batch_inputs,
                    **self._create_sampling_params(max_new_tokens),
                )

                for output in outputs:
                    all_outputs.append(
                        self._tokenizer.decode(output, skip_special_tokens=True)
                    )

                if show_progress:
                    processed = min(i + batch_size, total)
                    print(f"  Processed {processed}/{total}")

        return all_outputs

    def shutdown(self):
        """Shut down the model to free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._lora_loaded = False

    def predict_from_db_id(
        self,
        question: str,
        db_id: str,
        db_root: str,
        max_new_tokens: int = 512,
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

    @staticmethod
    def _parse_json_response(response: str) -> Dict:
        """Parse JSON from a model response."""
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3].strip()

            start_idx = response.find("{")
            if start_idx != -1:
                end_idx = response.rfind("}") + 1
                if end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)

            return json.loads(response)

        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": response[:500]}

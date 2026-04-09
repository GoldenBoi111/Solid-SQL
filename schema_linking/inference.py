"""
Schema Linking Inference (vLLM + LoRA)

Uses vLLM with forced JSON schema output and dynamic LoRA adapter loading
to predict relevant tables and columns from a question and database schema.

The LoRA adapter is loaded per-prediction-batch and unloaded after,
so the base model stays clean for other tasks.

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

    # Batch prediction
    results = linker.predict_batch([
        {"question": "...", "schema_text": "..."},
        {"question": "...", "schema_text": "..."},
    ])
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

from .config import (
    MODEL_NAME, OUTPUT_DIR, MAX_SEQ_LENGTH,
    INSTRUCTION_TEMPLATE, OUTPUT_SCHEMA,
    LORA_R,
)
from .schema_formatter import format_schema_compact, load_schemas_from_dir


class SchemaLinker:
    """Uses vLLM with a LoRA adapter and forced JSON schema for inference."""

    def __init__(
        self,
        base_model: str = MODEL_NAME,
        adapter_path: str = "",
        tensor_parallel_size: int = 1,
        max_seq_length: int = MAX_SEQ_LENGTH,
    ):
        """
        Initialize the schema linker.

        Args:
            base_model: Hugging Face model name or local path
            adapter_path: Path to the fine-tuned LoRA adapter directory
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_seq_length: Maximum sequence length (prompt + output)
        """
        if not adapter_path:
            adapter_path = str(Path(OUTPUT_DIR) / "lora_adapter")

        self.adapter_path = Path(adapter_path)
        if not self.adapter_path.is_dir() or not (self.adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Adapter not found at '{adapter_path}'. "
                "Run train.py first, or pass a valid adapter_path."
            )

        self.base_model_name = base_model
        self.max_seq_length = max_seq_length
        self.tensor_parallel_size = tensor_parallel_size

        # Import vLLM here to defer the dependency until use time
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for schema linking inference. "
                "Install with: pip install vllm"
            )

        self.LLM = LLM
        self.SamplingParams = SamplingParams

        print(f"\n{'='*60}")
        print(f"Schema Linker Initialization (vLLM)")
        print(f"{'='*60}")
        print(f"  Base model: {base_model}")
        print(f"  Adapter:    {adapter_path}")
        print(f"  GPUs:       {tensor_parallel_size}")
        print(f"  Max seq len: {max_seq_length}")
        print(f"{'='*60}\n")

    def _format_prompt(self, question: str, schema_text: str) -> str:
        """Format the input prompt using the instruction template."""
        return INSTRUCTION_TEMPLATE.format(
            question=question,
            schema_text=schema_text,
        )

    def _create_vllm(self):
        """Create a vLLM engine instance with LoRA support enabled."""
        llm = self.LLM(
            model=self.base_model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_seq_length,
            dtype="bfloat16",
            enable_lora=True,
            max_loras=4,
            max_lora_rank=LORA_R,
            enforce_eager=True,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        return llm

    def _create_sampling_params(self, max_new_tokens: int, lora_path: str):
        """Create SamplingParams with LoRA path and structured JSON output."""
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("schema_linking", 1, lora_path)

        return self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            lora_request=lora_request,
            guided_decode_json_schema=json.dumps(OUTPUT_SCHEMA),
        )

    def predict(
        self,
        question: str,
        schema_text: str,
        max_new_tokens: int = 512,
    ) -> Dict:
        """
        Generate schema linking prediction for a single question.

        Loads the LoRA adapter, generates with forced JSON schema,
        then unloads the adapter.
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

        Loads the LoRA adapter once for the entire batch,
        then unloads it when done.
        """
        lora_path = str(self.adapter_path)

        # Create vLLM engine with LoRA support
        llm = self._create_vllm()
        sampling_params = self._create_sampling_params(max_new_tokens, lora_path)

        # Format all prompts
        prompts = [
            self._format_prompt(item["question"], item["schema_text"])
            for item in inputs
        ]

        # Generate in sub-batches for memory efficiency
        all_results = []
        total = len(prompts)

        for i in range(0, total, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)

            for output in outputs:
                response = output.outputs[0].text
                parsed = self._parse_json_response(response)
                all_results.append(parsed)

            if show_progress:
                processed = min(i + batch_size, total)
                print(f"  Processed {processed}/{total}")

        # Shut down vLLM engine (unloads LoRA + frees VRAM)
        if hasattr(llm, "shutdown"):
            llm.shutdown()
        elif hasattr(llm, "llm_engine"):
            llm.llm_engine.shutdown()

        return all_results

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

"""
Data Enhancement Script using vLLM Model Manager

Reads a JSON file with fields like db_id, question, gold_sql and generates
2 question variations per entry using a local vLLM model with structured output.
"""

import csv
import json
import argparse
import sqlite3
import sys
from pathlib import Path
from vllm_model_manager import vLLMModelManager, HAS_STRUCTUREED_OUTPUTS


VARIATION_SCHEMA = {
    "type": "object",
    "properties": {
        "q1": {"type": "string"},
        "reasoning1": {"type": "string"},
        "q2": {"type": "string"},
        "reasoning2": {"type": "string"},
    },
    "required": ["q1", "reasoning1", "q2", "reasoning2"],
}


def _read_csv_descriptions(csv_dir: Path) -> str:
    """Read all CSV description files and return formatted column metadata."""
    if not csv_dir.is_dir():
        return ""

    parts = []
    for csv_file in sorted(csv_dir.glob("*.csv")):
        table_name = csv_file.stem
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                continue

            # Each row typically has: column_name, data_type, description (or similar)
            # Build a column description string
            col_info = []
            for row in rows:
                # Try common column name conventions
                col_name = row.get("column_name", row.get("Column", row.get("COLUMN_NAME", "")))
                col_type = row.get("data_type", row.get("Type", row.get("DATA_TYPE", "")))
                col_desc = row.get("description", row.get("Description", row.get("COMMENT", "")))

                if col_name:
                    info = f"  - {col_name}"
                    if col_type:
                        info += f" ({col_type})"
                    if col_desc:
                        info += f": {col_desc}"
                    col_info.append(info)

            if col_info:
                parts.append(f"Table: {table_name}\n" + "\n".join(col_info))
        except Exception as e:
            print(f"  Warning: Could not read {csv_file}: {e}")

    return "\n".join(parts) if parts else ""


def get_db_schema(db_path: str, db_id: str = "", db_root: str = "") -> str:
    """Extract table and column info from a SQLite database + CSV descriptions.

    Returns formatted schema text with table/column info and CSV descriptions.
    """
    db_path = Path(db_path)
    sections = []

    # 1. Read CSV descriptions if available
    if db_root and db_id:
        csv_dir = Path(db_root) / db_id / "database_description"
        csv_desc = _read_csv_descriptions(csv_dir)
        if csv_desc:
            sections.append(csv_desc)

    # 2. Read SQLite schema via PRAGMA
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()

        pragma_parts = []
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = cursor.fetchall()
            col_defs = ", ".join(f"{col[1]} {col[2]}" for col in columns)
            pragma_parts.append(f"Table: {table_name} ({col_defs})")

        conn.close()
        if pragma_parts:
            sections.append("SQLite Schema:\n" + "\n".join(pragma_parts))
    except Exception as e:
        sections.append(f"Error reading SQLite schema: {e}")

    return "\n\n".join(sections) if sections else "No schema found."


def build_prompt(question: str, schema_text: str = "") -> str:
    """Build a prompt asking the model to generate 2 variations of the question."""
    schema_section = ""
    if schema_text:
        schema_section = f"""Database Schema:
{schema_text}

"""

    return f"""Given the following question, generate 2 new variations.

{schema_section}Original Question: {question}

Instructions:
- Change the sentence structure significantly
- Use synonyms where possible
- Make each variation read naturally as a brand new question
- Preserve the original meaning and intent
- Reference real table and column names from the schema where appropriate

Return your response as a JSON object with these exact keys:
- q1: the first variation
- reasoning1: explain what changes you made and why for q1
- q2: the second variation
- reasoning2: explain what changes you made and why for q2

Do NOT include any text outside of the JSON object."""


def process_file(
    input_path: str,
    output_path: str,
    model_name: str,
    tensor_parallel_size: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    db_root: str = "",
):
    """Process each entry in the JSON file and generate question variations."""
    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data if isinstance(data, list) else [data]

    # Filter entries that have a question field
    valid_entries = []
    for i, entry in enumerate(entries):
        question = entry.get("question", "")
        if not question:
            print(f"Skipping entry {i}: no 'question' field found")
            continue
        valid_entries.append(entry)

    if not valid_entries:
        print("No valid entries with a 'question' field found. Exiting.")
        return

    print(f"\nLoaded {len(valid_entries)} entries from {input_path}")

    # Initialize vLLM model manager
    manager = vLLMModelManager(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Build prompts with optional schema injection
    prompts = []
    for entry in valid_entries:
        schema_text = ""
        db_id = entry.get("db_id", "")
        if db_root and db_id:
            db_path = Path(db_root) / f"{db_id}.sqlite"
            if db_path.exists():
                schema_text = get_db_schema(str(db_path), db_id=db_id, db_root=db_root)
                print(f"  Loaded schema for db_id '{db_id}' from {db_path}")
            else:
                print(f"  Warning: Database not found for db_id '{db_id}' at {db_path}")
        prompts.append(build_prompt(entry["question"], schema_text))

    print(f"\nGenerating variations for {len(prompts)} prompts (batch_size={batch_size})...")

    # Batch generation with structured JSON output
    results = manager.generate_json_batch(
        prompts=prompts,
        json_schema=VARIATION_SCHEMA,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        show_progress=True,
    )

    # Merge results back with original entries
    output_entries = []
    for i, (entry, variation) in enumerate(zip(valid_entries, results)):
        enhanced = {**entry}
        if "error" in variation:
            print(f"Entry {i} returned error: {variation.get('error', 'unknown')}")
            enhanced["error"] = variation.get("error", "unknown")
            enhanced["raw_response"] = variation.get("raw_response", "")
        else:
            enhanced["q1"] = variation.get("q1", "")
            enhanced["reasoning1"] = variation.get("reasoning1", "")
            enhanced["q2"] = variation.get("q2", "")
            enhanced["reasoning2"] = variation.get("reasoning2", "")
        output_entries.append(enhanced)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_entries, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Processed {len(output_entries)} entries. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate question variations using vLLM (local model, not API)"
    )
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("-o", "--output", default="enhanced_output.json", help="Path to output JSON file")
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Hugging Face model name or path (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation (default: 16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per entry (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--db-root",
        default="",
        help="Path to root directory containing SQLite databases (e.g. ./databases/)",
    )
    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        tensor_parallel_size=args.gpus,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        db_root=args.db_root,
    )


if __name__ == "__main__":
    main()

"""
Dataset Builder for Schema Linking

Loads dataset examples (question, SQL, db_id) and database schemas,
parses SQL queries to extract ground-truth tables and columns,
and generates instruction-style training prompts.

Output: JSONL file with {input, output} pairs where output is structured JSON.
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple

from sql_parser import extract_schema_labels
from schema_formatter import format_schema_compact, load_schemas_from_dir
from config import (
    RAW_DATA_DIR, DB_ROOT, OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH,
    VAL_SPLIT_RATIO, INSTRUCTION_TEMPLATE, SQL_DIALECT,
)


def build_reasoning_table(table: str, question: str, sql: str) -> str:
    """Generate a brief reasoning string for why a table is relevant."""
    return f"Used in the query to answer the question: {question[:80]}"


def build_reasoning_column(col: str, question: str, sql: str) -> str:
    """Generate a brief reasoning string for why a column is relevant."""
    return f"Referenced in the query to answer: {question[:80]}"


def build_training_example(
    question: str,
    schema_text: str,
    tables: set,
    columns: set,
    sql: str = "",
) -> Dict:
    """
    Create a single training example in instruction format.

    Input: Question + Schema text
    Output: Structured JSON with tables list and columns list, each with reasoning
    """
    input_text = INSTRUCTION_TEMPLATE.format(
        question=question,
        schema_text=schema_text,
    )

    # Build structured output
    output_obj = {
        "tables": [
            {"name": t, "reason": build_reasoning_table(t, question, sql)}
            for t in sorted(tables)
        ],
        "columns": [
            {"name": c, "reason": build_reasoning_column(c, question, sql)}
            for c in sorted(columns)
        ],
    }

    # Serialize to compact JSON string for training target
    output_text = json.dumps(output_obj, ensure_ascii=False)

    return {
        "input": input_text,
        "output": output_text,
        "output_structured": output_obj,  # Keep for inspection (stripped in JSONL)
    }


def load_dataset_examples(data_dir: str) -> List[Dict]:
    """
    Load dataset examples from JSON files.

    Supports:
    - Single file: train_spider.json, train_others.json, etc.
    - Files with keys: "data", "examples", or a top-level list
    """
    all_examples = []
    data_path = Path(data_dir)

    for json_file in data_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different file formats
        if isinstance(data, list):
            examples = data
        elif isinstance(data, dict):
            # Try common keys
            examples = data.get("data", data.get("examples", []))
            if not examples:
                # Check if it's a single schema file, skip
                if "table_names_original" in data:
                    continue
                examples = [data]
        else:
            continue

        all_examples.extend(examples)

    return all_examples


def process_dataset(
    examples: List[Dict],
    schemas: Dict[str, Dict],
    db_root: str,
    dialect: str,
) -> List[Dict[str, str]]:
    """
    Process all examples into training format.

    For each example:
    1. Parse SQL to extract tables and columns
    2. Format the database schema for the db_id
    3. Create input/output training pair
    """
    training_data = []
    skipped = 0

    for i, example in enumerate(examples):
        question = example.get("question", "")
        sql = example.get("query", example.get("SQL", ""))
        db_id = example.get("db_id", "")

        if not question or not sql:
            skipped += 1
            continue

        # Get schema for this database
        schema = schemas.get(db_id)
        if not schema:
            print(f"  Warning: No schema found for db_id '{db_id}' (example {i})")
            skipped += 1
            continue

        schema_text = format_schema_compact(schema)

        # Parse SQL to extract labels
        tables, columns = extract_schema_labels(sql, dialect=dialect)

        if not tables and not columns:
            # SQL parsing failed — skip or include with warning
            print(f"  Warning: No schema labels extracted from SQL (example {i}): {sql[:80]}...")
            skipped += 1
            continue

        # Build training example
        entry = build_training_example(question, schema_text, tables, columns, sql)
        entry["db_id"] = db_id  # Keep for reference
        training_data.append(entry)

    print(f"Processed {len(training_data)} examples, skipped {skipped}")
    return training_data


def split_dataset(
    data: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and validation sets."""
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


def save_jsonl(data: List[Dict], output_path: str) -> None:
    """
    Save data as JSONL file.
    Strips the 'output_structured' helper key — only 'input', 'output', 'db_id' are written.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            # Write clean keys only
            clean = {
                "input": entry["input"],
                "output": entry["output"],
                "db_id": entry.get("db_id", ""),
            }
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build schema linking training dataset")
    parser.add_argument("--data-dir", default=RAW_DATA_DIR, help="Directory with dataset JSON files")
    parser.add_argument("--schema-dir", default="", help="Directory with schema JSON files (optional)")
    parser.add_argument("--db-root", default=DB_ROOT, help="Root directory with SQLite databases")
    parser.add_argument("--train-output", default=OUTPUT_TRAIN_PATH, help="Output train JSONL path")
    parser.add_argument("--val-output", default=OUTPUT_VAL_PATH, help="Output val JSONL path")
    parser.add_argument("--val-ratio", type=float, default=VAL_SPLIT_RATIO, help="Validation split ratio")
    parser.add_argument("--dialect", default=SQL_DIALECT, help="SQL dialect for sqlglot")
    args = parser.parse_args()

    print("=" * 60)
    print("Building Schema Linking Dataset")
    print("=" * 60)

    # Load examples
    print(f"\nLoading dataset examples from: {args.data_dir}")
    examples = load_dataset_examples(args.data_dir)
    print(f"Loaded {len(examples)} examples")

    # Load schemas
    schema_dir = args.schema_dir if args.schema_dir else args.data_dir
    print(f"\nLoading schemas from: {schema_dir}")
    schemas = load_schemas_from_dir(schema_dir)
    print(f"Loaded {len(schemas)} schemas")

    # Process
    print(f"\nProcessing examples...")
    training_data = process_dataset(examples, schemas, args.db_root, args.dialect)

    # Split
    train_data, val_data = split_dataset(training_data, val_ratio=args.val_ratio)
    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")

    # Save
    save_jsonl(train_data, args.train_output)
    save_jsonl(val_data, args.val_output)

    print("\nDone!")


if __name__ == "__main__":
    main()

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

import sqlglot
from sqlglot import exp

from sql_parser import extract_schema_labels
from schema_formatter import format_schema_compact, load_schemas_from_dir, load_schema_from_sqlite
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


def _resolve_table_name(table_expr: exp.Table) -> str:
    return table_expr.name


def extract_join_relationships(sql: str, dialect: str) -> List[str]:
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError:
        return []

    joins: List[str] = []
    for join in parsed.find_all(exp.Join):
        left_table = join.parent and getattr(join.parent, "this", None)
        join_table = join.this
        on_expr = join.args.get("on")
        if join_table is None or on_expr is None:
            continue

        join_table_name = _resolve_table_name(join_table) if isinstance(join_table, exp.Table) else str(join_table)
        left_name = _resolve_table_name(left_table) if isinstance(left_table, exp.Table) else ""
        on_sql = on_expr.sql(dialect=dialect)
        if left_name:
            joins.append(f"{left_name} <-> {join_table_name} ON {on_sql}")
        else:
            joins.append(f"{join_table_name} ON {on_sql}")

    return sorted(set(joins))


def extract_filters(sql: str, dialect: str) -> List[str]:
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError:
        return []

    filters: List[str] = []
    for where in parsed.find_all(exp.Where):
        condition = where.this.sql(dialect=dialect).strip()
        if condition:
            filters.append(condition)
    for having in parsed.find_all(exp.Having):
        condition = having.this.sql(dialect=dialect).strip()
        if condition:
            filters.append(condition)

    return sorted(set(filters))


def extract_intent(question: str, sql: str) -> str:
    question_text = question.strip()
    if question_text:
        return question_text[:200]
    return "Spider-style schema linking"


def build_training_example(
    question: str,
    evidence: str,
    schema_text: str,
    tables: set,
    columns: set,
    sql: str = "",
) -> Dict:
    """
    Create a single training example in instruction format.

    Input: Question + optional evidence + Schema text
    Output: Structured JSON with tables list and columns list, each with reasoning
    """
    input_text = INSTRUCTION_TEMPLATE.format(
        question=question.strip(),
        evidence_block=evidence.strip() if evidence.strip() else "(none provided)",
        schema_text=schema_text,
    )

    relevant_tables = sorted(tables)
    relevant_columns = sorted(columns)
    join_relationships = extract_join_relationships(sql, SQL_DIALECT)
    filters = extract_filters(sql, SQL_DIALECT)
    intent = extract_intent(question, sql)

    output_lines = ["Relevant Tables:"]
    if relevant_tables:
        output_lines.extend([f"- {table}" for table in relevant_tables])
    else:
        output_lines.append("- (none)")

    output_lines.append("")
    output_lines.append("Relevant Columns:")
    if relevant_columns:
        table_to_columns = {}
        for column in relevant_columns:
            table_name, column_name = column.split(".", 1) if "." in column else ("?", column)
            table_to_columns.setdefault(table_name, []).append(column_name)
        for table_name in sorted(table_to_columns):
            columns_text = ", ".join(sorted(table_to_columns[table_name]))
            output_lines.append(f"- {table_name}: {columns_text}")
    else:
        output_lines.append("- (none)")

    output_lines.append("")
    output_lines.append("Join Relationships:")
    if join_relationships:
        output_lines.extend([f"- {join}" for join in join_relationships])
    else:
        output_lines.append("- (none)")

    output_lines.append("")
    output_lines.append("Filters / Constraints:")
    if filters:
        output_lines.extend([f"- {flt}" for flt in filters])
    else:
        output_lines.append("- (none)")

    output_lines.append("")
    output_lines.append("Question Intent:")
    output_lines.append(f"- {intent}")

    output_text = "\n".join(output_lines).strip()

    output_obj = {
        "tables": [
            {"name": t, "reason": build_reasoning_table(t, question, sql)}
            for t in relevant_tables
        ],
        "columns": [
            {"name": c, "reason": build_reasoning_column(c, question, sql)}
            for c in relevant_columns
        ],
        "joins": join_relationships,
        "filters": filters,
        "intent": intent,
    }

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
        evidence = example.get("evidence", "")
        sql = example.get("query", example.get("SQL", example.get("sql", "")))
        db_id = example.get("db_id", "")
        question_id = example.get("question_id", i)
        difficulty = example.get("difficulty", "unknown")

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
        entry = build_training_example(question, evidence, schema_text, tables, columns, sql)
        entry["db_id"] = db_id  # Keep for reference
        entry["question_id"] = question_id
        entry["difficulty"] = difficulty
        entry["evidence"] = evidence
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
                "question_id": entry.get("question_id", ""),
                "difficulty": entry.get("difficulty", "unknown"),
                "evidence": entry.get("evidence", ""),
            }
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} entries to {output_path}")


def load_schemas_from_databases(db_dir: str) -> Dict[str, Dict]:
    """
    Load schemas by introspecting SQLite databases in a directory tree.
    Recursively scans for *.sqlite files and extracts table/column metadata.
    """
    import sqlite3
    schemas = {}
    db_path = Path(db_dir)

    sqlite_files = list(db_path.rglob("*.sqlite"))
    for db_file in sqlite_files:
        db_id = db_file.stem
        if db_id not in schemas:
            try:
                schemas[db_id] = load_schema_from_sqlite(str(db_file))
            except (sqlite3.DatabaseError, ValueError) as e:
                print(f"  Warning: Skipping corrupted SQLite file: {db_file} ({e})")

    print(f"  Loaded {len(schemas)} schemas from SQLite databases")
    return schemas


def main():
    parser = argparse.ArgumentParser(description="Build schema linking training dataset")
    parser.add_argument("--train-json", required=True, help="Path to train.json file")
    parser.add_argument("--db-dir", required=True, help="Root directory with SQLite database files")
    parser.add_argument("--train-output", default=OUTPUT_TRAIN_PATH, help="Output train JSONL path")
    parser.add_argument("--val-output", default=OUTPUT_VAL_PATH, help="Output val JSONL path")
    parser.add_argument("--val-ratio", type=float, default=VAL_SPLIT_RATIO, help="Validation split ratio")
    parser.add_argument("--dialect", default=SQL_DIALECT, help="SQL dialect for sqlglot")
    args = parser.parse_args()

    print("=" * 60)
    print("Building Schema Linking Dataset")
    print("=" * 60)

    # Load examples from the specified train.json
    print(f"\nLoading dataset examples from: {args.train_json}")
    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = data if isinstance(data, list) else data.get("data", data.get("examples", [data]))
    print(f"Loaded {len(examples)} examples")

    # Load schemas from SQLite databases
    print(f"\nLoading schemas from: {args.db_dir}")
    schemas = load_schemas_from_databases(args.db_dir)

    # Process
    print(f"\nProcessing examples...")
    training_data = process_dataset(examples, schemas, args.db_dir, args.dialect)

    # Split
    train_data, val_data = split_dataset(training_data, val_ratio=args.val_ratio)
    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")

    # Save
    save_jsonl(train_data, args.train_output)
    save_jsonl(val_data, args.val_output)

    print("\nDone!")


if __name__ == "__main__":
    main()

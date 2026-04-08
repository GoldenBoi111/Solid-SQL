"""
Schema Formatter

Converts database schema JSON (Spider format) into text representations
suitable for inclusion in training prompts.

Supports two modes:
1. Compact: "Table(col1, col2, col3)" — used for inference
2. Detailed: includes column types and descriptions — used for richer prompts

Also supports loading schemas directly from SQLite database files.
"""

import sqlite3
from typing import Dict, List, Optional
from pathlib import Path
import json


def load_schema_from_sqlite(db_path: str) -> Dict:
    """
    Introspect a SQLite database and return a Spider-format schema dict.

    Output:
    {
        "db_id": "sales",
        "table_names_original": ["Customers", "Products", ...],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], ...]
    }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {
        "db_id": Path(db_path).stem,
        "table_names_original": tables,
        "column_names_original": [[-1, "*"]],
    }

    for table_idx, table_name in enumerate(tables):
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        for col in cursor.fetchall():
            col_name = col[1]  # Column name is at index 1
            schema["column_names_original"].append([table_idx, col_name])

    conn.close()
    return schema


def format_schema_compact(schema: Dict) -> str:
    """
    Format a schema into a compact text representation.

    Input schema follows Spider dataset format:
    {
        "db_id": "concert_singer",
        "table_names_original": ["Singer", "Album"],
        "column_names_original": [
            [-1, "*"],
            [0, "id"],
            [0, "name"],
            [0, "age"],
            [1, "id"],
            [1, "singer_id"],
            [1, "title"]
        ]
    }

    Output:
    Singer(id, name, age)
    Album(id, singer_id, title)
    """
    table_names = schema.get("table_names_original", [])
    column_names = schema.get("column_names_original", [])

    # Build table -> columns mapping
    table_columns: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}

    for col_entry in column_names:
        table_idx = col_entry[0]
        col_name = col_entry[1]

        # Skip the wildcard entry [-1, "*"]
        if table_idx == -1:
            continue

        if table_idx < len(table_names):
            table_columns[table_idx].append(col_name)

    # Format each table
    lines = []
    for i, table_name in enumerate(table_names):
        cols = ", ".join(table_columns.get(i, []))
        lines.append(f"{table_name}({cols})")

    return "\n".join(lines)


def format_schema_detailed(schema: Dict) -> str:
    """
    Format a schema with column types and descriptions if available.

    If the schema includes column_types or description fields, they are included.
    """
    table_names = schema.get("table_names_original", [])
    column_names = schema.get("column_names_original", [])
    column_types = schema.get("column_types", [])
    descriptions = schema.get("column_descriptions", [])

    # Build table -> columns mapping with optional metadata
    table_columns: Dict[int, List[Dict]] = {i: [] for i in range(len(table_names))}

    for idx, col_entry in enumerate(column_names):
        table_idx = col_entry[0]
        col_name = col_entry[1]

        if table_idx == -1:
            continue

        if table_idx < len(table_names):
            col_info = {"name": col_name}
            if idx < len(column_types):
                col_info["type"] = column_types[idx]
            if descriptions and idx < len(descriptions) and descriptions[idx]:
                col_info["description"] = descriptions[idx]
            table_columns[table_idx].append(col_info)

    # Format each table
    lines = []
    for i, table_name in enumerate(table_names):
        cols_info = table_columns.get(i, [])
        col_strs = []
        for ci in cols_info:
            parts = [ci["name"]]
            if "type" in ci:
                parts.append(ci["type"])
            if "description" in ci:
                parts.append(f'"{ci["description"]}"')
            col_strs.append(" ".join(parts))

        lines.append(f"{table_name}({', '.join(col_strs)})")

    return "\n".join(lines)


def load_schema(schema_path: str) -> Dict:
    """Load a single schema JSON file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_schemas_from_dir(schema_dir: str) -> Dict[str, Dict]:
    """
    Load all schemas from a directory.
    Returns dict mapping db_id -> schema.

    Expects files named like: concert_singer.json, dog_1.json, etc.
    Or a single combined file like: database.json with all schemas.
    """
    schemas = {}
    schema_path = Path(schema_dir)

    # Check for single combined file
    combined = schema_path / "database.json"
    if combined.exists():
        with open(combined, "r", encoding="utf-8") as f:
            all_schemas = json.load(f)
            for s in all_schemas:
                schemas[s["db_id"]] = s
        return schemas

    # Otherwise, load individual JSON files
    for json_file in schema_path.glob("*.json"):
        schema = load_schema(str(json_file))
        db_id = schema.get("db_id", json_file.stem)
        schemas[db_id] = schema

    return schemas

"""
SQL Parser for Schema Linking

Uses sqlglot to parse SQL queries and extract:
- Tables referenced (FROM, JOIN, subqueries)
- Columns referenced (SELECT, WHERE, JOIN ON, GROUP BY, ORDER BY, HAVING)

Handles edge cases:
- Table aliases (resolved back to original names)
- Nested/subqueries
- CTEs (Common Table Expressions)
- Star columns (COUNT(*), SELECT *)
- Duplicate deduplication
"""

from typing import Tuple, Set
import sqlglot
from sqlglot import exp


def extract_schema_labels(sql: str, dialect: str = "sqlite") -> Tuple[Set[str], Set[str]]:
    """
    Parse a SQL query and extract the tables and columns used.

    Args:
        sql: The SQL query string
        dialect: SQL dialect for parsing (default: sqlite)

    Returns:
        Tuple of (set of table names, set of table.column strings)

    Example:
        >>> sql = "SELECT COUNT(*) FROM Singer WHERE Age > 20"
        >>> tables, cols = extract_schema_labels(sql)
        >>> tables
        {'Singer'}
        >>> cols
        {'Singer.age'}
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError as e:
        print(f"  Warning: Failed to parse SQL: {sql[:100]}... Error: {e}")
        return set(), set()

    # Build alias -> table_name mapping
    alias_map = _build_alias_map(parsed)

    tables = set()
    columns = set()

    # Extract tables from FROM and JOIN clauses
    _extract_tables(parsed, tables, alias_map)

    # Extract columns from all expressions
    _extract_columns(parsed, columns, alias_map)

    return tables, columns


def _build_alias_map(ast: exp.Expression) -> dict:
    """
    Build a mapping from table aliases to their original names.

    Handles:
    - FROM table AS alias
    - FROM table alias (implicit alias)
    - JOIN table AS alias ON ...
    """
    alias_map = {}

    for table in ast.find_all(exp.Table):
        table_name = table.name
        table_alias = table.alias

        if table_alias:
            alias_map[table_alias] = table_name
        else:
            # Table without alias — register itself for lookups
            alias_map[table_name] = table_name

    return alias_map


def _extract_tables(ast: exp.Expression, tables: Set[str], alias_map: dict) -> None:
    """
    Recursively find all table references in the AST.
    Handles FROM, JOIN, and subquery tables.
    """
    for table in ast.find_all(exp.Table):
        table_name = table.name
        if table_name:
            tables.add(table_name)


def _extract_columns(ast: exp.Expression, columns: Set[str], alias_map: dict) -> None:
    """
    Extract all column references from the AST.

    Handles:
    - Column references with table qualifier: table.column
    - Column references without qualifier: resolve via alias map or mark as unknown
    - Star columns: ignore COUNT(*), SELECT * as they don't specify specific columns
    - Nested columns in subqueries
    """
    for col in ast.find_all(exp.Column):
        table_part = col.table  # The table/alias prefix
        col_name = col.name

        # Skip star columns (e.g., COUNT(*), SELECT *)
        if col_name == "*":
            continue

        if table_part:
            # Resolve alias to actual table name
            actual_table = alias_map.get(table_part, table_part)
            columns.add(f"{actual_table}.{col_name}")
        else:
            # No table qualifier — try to infer from context
            # In many text-to-SQL datasets, unqualified columns belong to
            # the most recently referenced table or the only table.
            # We'll mark these with a placeholder for now.
            if len(alias_map) == 1:
                actual_table = list(alias_map.values())[0]
                columns.add(f"{actual_table}.{col_name}")
            else:
                # Ambiguous — include as-is with warning during dataset build
                columns.add(f"?.{col_name}")


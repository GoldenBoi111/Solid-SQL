"""
SQL Skeleton Extractor

Extracts the structural skeleton (S⋆) from SQL statements by replacing table names,
column names, and values with placeholders while preserving keywords and logical structure.

This enables similarity matching based on SQL structure rather than specific schema elements,
using parse tree edit distance for more accurate structural comparison.

Usage:
    from schema_linking.sql_skeleton_extractor import SQLSkeletonExtractor

    extractor = SQLSkeletonExtractor()

    # Extract skeleton from a single SQL statement
    skeleton = extractor.extract("SELECT COUNT(*) FROM Singer WHERE Age > 20")

    # Batch extraction
    skeletons = extractor.extract_batch([
        "SELECT COUNT(*) FROM Singer WHERE Age > 20",
        "SELECT AVG(salary) FROM Employees WHERE dept = 'Engineering'",
    ])

    # Calculate edit distance between two skeletons
    from schema_linking.sql_skeleton_extractor import sql_skeleton_edit_distance
    distance = sql_skeleton_edit_distance(skeleton1, skeleton2)
"""

from typing import List, Optional, Tuple
import sqlglot
from sqlglot import exp


class SQLSkeletonExtractor:
    """
    Extracts structural skeletons from SQL statements by replacing schema-specific
    elements with placeholders while preserving SQL keywords and logical structure.
    """

    # Placeholders for different element types
    TABLE_PLACEHOLDER = "[TABLE]"
    COLUMN_PLACEHOLDER = "[COLUMN]"
    VALUE_PLACEHOLDER = "[VALUE]"

    def __init__(self, dialect: str = "sqlite"):
        """
        Initialize the SQL skeleton extractor.

        Args:
            dialect: SQL dialect for parsing (default: sqlite)
        """
        self.dialect = dialect

    def extract(self, sql: str) -> str:
        """
        Extract the skeleton (S⋆) from a single SQL statement.

        Args:
            sql: The SQL statement

        Returns:
            The extracted SQL skeleton with placeholders
        """
        if not isinstance(sql, str):
            sql = str(sql)

        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
        except sqlglot.errors.ParseError as e:
            print(f"  Warning: Failed to parse SQL: {sql[:100]}... Error: {e}")
            return sql  # Return original if parsing fails

        return self._transform_to_skeleton(parsed)

    def extract_batch(
        self,
        sql_statements: List[str],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Extract skeletons for multiple SQL statements.

        Args:
            sql_statements: List of SQL statements
            show_progress: Whether to show progress

        Returns:
            List of extracted SQL skeletons
        """
        skeletons = []
        total = len(sql_statements)

        for i, sql in enumerate(sql_statements):
            skeleton = self.extract(sql)
            skeletons.append(skeleton)

            if show_progress:
                print(f"  Processed {i + 1}/{total}")

        return skeletons

    def _transform_to_skeleton(self, ast: exp.Expression) -> str:
        """
        Transform a parsed SQL AST into its skeleton form by replacing
        table names, column names, and literal values with placeholders.
        """
        # Clone the AST to avoid modifying the original
        skeleton_ast = ast.copy()

        # Replace table names with placeholders
        for table in skeleton_ast.find_all(exp.Table):
            if table.name:
                table.set("this", exp.Identifier(this=self.TABLE_PLACEHOLDER))
            if table.alias:
                # Keep alias for structure preservation
                pass

        # Replace column names with placeholders
        for col in skeleton_ast.find_all(exp.Column):
            if col.name:
                col.set("this", exp.Identifier(this=self.COLUMN_PLACEHOLDER))

        # Replace literal values with placeholders
        self._replace_literals(skeleton_ast)

        # Convert back to SQL string
        return skeleton_ast.sql(dialect=self.dialect)

    def _replace_literals(self, ast: exp.Expression) -> None:
        """Replace literal values with [VALUE] placeholder."""
        for lit in ast.find_all(exp.Literal):
            if lit.is_string or lit.is_number or lit.is_int:
                lit.set("this", self.VALUE_PLACEHOLDER)
                if lit.is_string:
                    lit.set("is_string", False)

        # Also handle NULL, TRUE, FALSE
        for null_expr in ast.find_all(exp.Null):
            null_expr.set("this", self.VALUE_PLACEHOLDER)

        for boolean_expr in ast.find_all(exp.Boolean):
            boolean_expr.set("this", self.VALUE_PLACEHOLDER)


def sql_skeleton_edit_distance(skeleton1: str, skeleton2: str) -> float:
    """
    Calculate the normalized edit distance between two SQL skeletons.

    Uses parse tree structure to provide a more meaningful distance metric
    than simple string edit distance. Emphasizes logical framework over
    superficial textual similarity.

    Args:
        skeleton1: First SQL skeleton
        skeleton2: Second SQL skeleton

    Returns:
        Normalized edit distance in range [0, 1] where 0 means identical
    """
    # Tokenize both skeletons into structural tokens
    tokens1 = _tokenize_sql_skeleton(skeleton1)
    tokens2 = _tokenize_sql_skeleton(skeleton2)

    if not tokens1 and not tokens2:
        return 0.0
    if not tokens1 or not tokens2:
        return 1.0

    # Calculate Levenshtein distance on token sequences
    distance = _levenshtein_distance(tokens1, tokens2)
    max_len = max(len(tokens1), len(tokens2))

    return distance / max_len


def sql_skeleton_similarity(skeleton1: str, skeleton2: str) -> float:
    """
    Calculate similarity between two SQL skeletons.

    Args:
        skeleton1: First SQL skeleton
        skeleton2: Second SQL skeleton

    Returns:
        Similarity score in range [0, 1] where 1 means identical
    """
    distance = sql_skeleton_edit_distance(skeleton1, skeleton2)
    return 1.0 - distance


def _tokenize_sql_skeleton(skeleton: str) -> List[str]:
    """
    Tokenize a SQL skeleton into structural tokens.

    Separates SQL keywords, placeholders, operators, and punctuation
    while preserving the parse tree structure.
    """
    import re

    # Tokenize into keywords, placeholders, operators, and punctuation
    pattern = r'(SELECT|FROM|WHERE|JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|'
    pattern += r'ON|AND|OR|NOT|IN|BETWEEN|LIKE|IS\s+NULL|IS\s+NOT\s+NULL|'
    pattern += r'GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|'
    pattern += r'COUNT|SUM|AVG|MIN|MAX|DISTINCT|'
    pattern += r'UNION|UNION\s+ALL|INTERSECT|EXCEPT|'
    pattern += r'\[TABLE\]|\[COLUMN\]|\[VALUE\]|'
    pattern += r'[=<>!]+|,|;|\(|\)|\+|-|\*|/|'
    pattern += r'[A-Za-z_]+)'

    tokens = re.findall(pattern, skeleton, re.IGNORECASE)
    return [t.upper() for t in tokens if t.strip()]


def _levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Calculate Levenshtein edit distance between two sequences."""
    m = len(seq1)
    n = len(seq2)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]

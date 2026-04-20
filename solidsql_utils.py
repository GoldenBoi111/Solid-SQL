"""
SolidSQL Data Utilities

Helper functions for working with SolidSQL data formats and retrieval systems.
"""

import json
from typing import List, Dict, Any
from pathlib import Path


def load_spider_format(
    train_path: str, max_examples: int = None
) -> List[Dict[str, str]]:
    """
    Load candidate examples from Spider dataset format.

    Args:
        train_path: Path to Spider train.json file
        max_examples: Maximum number of examples to load (None for all)

    Returns:
        List of candidate examples with "question" and "sql" keys
    """
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    for i, entry in enumerate(data):
        if max_examples and i >= max_examples:
            break

        candidates.append(
            {
                "question": entry.get("question", ""),
                "sql": entry.get("query", ""),
                "db_id": entry.get("db_id", ""),
            }
        )

    return candidates


def save_candidate_examples(examples: List[Dict[str, str]], output_path: str) -> None:
    """
    Save candidate examples to JSON file.

    Args:
        examples: List of candidate examples
        output_path: Path to save the JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(examples)} examples to {output_path}")


def load_candidate_examples(input_path: str) -> List[Dict[str, str]]:
    """
    Load candidate examples from JSON file.

    Args:
        input_path: Path to JSON file containing examples

    Returns:
        List of candidate examples
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate format
    for i, entry in enumerate(data):
        if "question" not in entry or "sql" not in entry:
            raise ValueError(
                f"Entry {i} missing 'question' or 'sql' field. "
                f"Found keys: {list(entry.keys())}"
            )

    return data


def merge_candidate_sets(
    existing: List[Dict[str, str]], new: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Merge two candidate example sets, removing duplicates based on question+sql.

    Args:
        existing: Existing candidate examples
        new: New candidate examples to add

    Returns:
        Merged list of unique examples
    """
    # Create set of existing (question, sql) tuples
    existing_tuples = set()
    for ex in existing:
        existing_tuples.add((ex["question"], ex["sql"]))

    # Add new examples that aren't already present
    merged = existing.copy()
    for ex in new:
        if (ex["question"], ex["sql"]) not in existing_tuples:
            merged.append(ex)
            existing_tuples.add((ex["question"], ex["sql"]))

    return merged


def validate_candidate_examples(examples: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Validate candidate examples for correctness.

    Args:
        examples: List of candidate examples

    Returns:
        Validation report
    """
    report = {"total": len(examples), "valid": 0, "invalid": 0, "errors": []}

    for i, ex in enumerate(examples):
        valid = True

        if "question" not in ex:
            valid = False
            report["errors"].append(f"Example {i}: Missing 'question' field")

        if "sql" not in ex:
            valid = False
            report["errors"].append(f"Example {i}: Missing 'sql' field")

        if valid:
            report["valid"] += 1
        else:
            report["invalid"] += 1

    return report


# Example usage
if __name__ == "__main__":
    print("SolidSQL Data Utilities Demo")
    print("=" * 40)

    # Create sample data
    sample_examples = [
        {
            "question": "How many singers are older than 20?",
            "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20",
            "db_id": "concert_singer",
        },
        {
            "question": "What is the average salary?",
            "sql": "SELECT AVG(Salary) FROM Employee",
            "db_id": "employee_dept",
        },
    ]

    # Save examples
    save_candidate_examples(sample_examples, "sample_candidates.json")

    # Load examples
    loaded = load_candidate_examples("sample_candidates.json")
    print(f"Loaded {len(loaded)} examples")

    # Validate examples
    report = validate_candidate_examples(loaded)
    print(f"Validation: {report['valid']} valid, {report['invalid']} invalid")

    # Clean up
    Path("sample_candidates.json").unlink(missing_ok=True)

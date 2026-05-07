#!/usr/bin/env python3
"""
Generate a detailed benchmark report from per-question result JSON.

Supports result files shaped like:
- a top-level list of question result dicts
- or an object containing a `results` list

The script prints a console report and saves a structured JSON summary.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_results(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
    raise ValueError("Input JSON must be a list or an object with a 'results' list.")


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def fmt_percent(numerator: int, denominator: int, width: int = 6) -> str:
    return f"{pct(numerator, denominator):{width}.2f}%"


def safe_median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def safe_mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def safe_pstdev(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def extract_execution_times(record: Dict[str, Any]) -> List[float]:
    values: List[float] = []

    execution_times = record.get("execution_times")
    if isinstance(execution_times, list):
        for item in execution_times:
            if isinstance(item, (int, float)):
                values.append(float(item))

    execution_time = record.get("execution_time")
    if isinstance(execution_time, (int, float)):
        if not values:
            values.append(float(execution_time))

    return values


def extract_confidence(record: Dict[str, Any]) -> Optional[float]:
    for key in ("winner_confidence", "confidence"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    selection = record.get("selection")
    if isinstance(selection, dict):
        value = selection.get("confidence")
        if isinstance(value, (int, float)):
            return float(value)

    return None


def collect_sql_variations(record: Dict[str, Any]) -> List[str]:
    sqls: List[str] = []
    for key in ("round_1_sql", "round_2_sql", "generated_sql", "winner_sql"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            sqls.append(value.strip())

    selection = record.get("selection")
    if isinstance(selection, dict):
        value = selection.get("selected_sql")
        if isinstance(value, str) and value.strip():
            sqls.append(value.strip())

    alternatives = record.get("high_confidence_alternatives")
    if isinstance(alternatives, list):
        for alt in alternatives:
            if isinstance(alt, dict):
                value = alt.get("sql")
                if isinstance(value, str) and value.strip():
                    sqls.append(value.strip())
                all_sqls = alt.get("all_sqls_in_group")
                if isinstance(all_sqls, list):
                    for item in all_sqls:
                        if isinstance(item, str) and item.strip():
                            sqls.append(item.strip())

    seen = set()
    deduped: List[str] = []
    for sql in sqls:
        if sql not in seen:
            seen.add(sql)
            deduped.append(sql)
    return deduped


def compute_generation_counts(record: Dict[str, Any], sql_variations: List[str]) -> Tuple[int, int, int]:
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        generated = metrics.get("generated")
        valid_generations = metrics.get("valid_generations")
        execution_errors = metrics.get("execution_errors")
        if all(isinstance(v, int) for v in (generated, valid_generations, execution_errors)):
            return int(generated), int(valid_generations), int(execution_errors)

    generated_count = len(sql_variations)

    valid = record.get("valid")
    if isinstance(valid, bool):
        valid_count = 1 if valid else 0
    else:
        final_valid = record.get("final_valid")
        valid_count = 1 if final_valid is True else 0

    validation_error = record.get("validation_error")
    execution_error = record.get("execution_error")
    error_count = 1 if validation_error or execution_error else max(generated_count - valid_count, 0)
    return generated_count, valid_count, error_count


def build_report(results: List[Dict[str, Any]], source_path: Path) -> Dict[str, Any]:
    total_questions = len(results)
    correct = sum(1 for row in results if row.get("execution_match") is True or row.get("is_correct") is True)
    incorrect = total_questions - correct

    all_exec_times_s: List[float] = []
    all_confidences: List[float] = []

    total_sql_generated = 0
    total_valid_executions = 0
    total_execution_errors = 0
    unique_sqls_global = set()
    sql_variation_counts: List[int] = []

    difficulty_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    database_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    selection_made = 0
    majority_vote_used = 0

    for row in results:
        is_correct = row.get("execution_match") is True or row.get("is_correct") is True
        difficulty = str(row.get("difficulty", "Unknown") or "Unknown")
        db_id = str(row.get("db_id", "unknown") or "unknown")

        difficulty_stats[difficulty]["total"] += 1
        database_stats[db_id]["total"] += 1
        if is_correct:
            difficulty_stats[difficulty]["correct"] += 1
            database_stats[db_id]["correct"] += 1

        all_exec_times_s.extend(extract_execution_times(row))

        confidence = extract_confidence(row)
        if confidence is not None:
            all_confidences.append(confidence)

        sql_variations = collect_sql_variations(row)
        sql_variation_counts.append(len(sql_variations))
        unique_sqls_global.update(sql_variations)

        generated_count, valid_count, error_count = compute_generation_counts(row, sql_variations)
        total_sql_generated += generated_count
        total_valid_executions += valid_count
        total_execution_errors += error_count

        if row.get("generated_sql") or row.get("winner_sql"):
            selection_made += 1

        selection = row.get("selection")
        if isinstance(selection, dict):
            candidates_count = selection.get("candidates_count")
            if isinstance(candidates_count, int) and candidates_count > 1:
                majority_vote_used += 1

    exec_times_ms = [value * 1000.0 for value in all_exec_times_s]
    mean_ms = safe_mean(exec_times_ms)
    median_ms = safe_median(exec_times_ms)
    std_ms = safe_pstdev(exec_times_ms)
    min_ms = min(exec_times_ms) if exec_times_ms else None
    max_ms = max(exec_times_ms) if exec_times_ms else None

    mean_conf = safe_mean(all_confidences)
    median_conf = safe_median(all_confidences)
    high_conf = sum(1 for value in all_confidences if value > 0.5)
    low_conf = sum(1 for value in all_confidences if value < 0.2)

    report = {
        "source_file": str(source_path),
        "overall": {
            "total_questions": total_questions,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": pct(correct, total_questions),
        },
        "execution_time_statistics": {
            "total_queries_executed": len(exec_times_ms),
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "std_dev_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
        },
        "generation_statistics": {
            "total_sql_generated": total_sql_generated,
            "valid_executions": total_valid_executions,
            "execution_errors": total_execution_errors,
            "validation_success_rate": pct(total_valid_executions, max(total_sql_generated, 1)),
            "execution_error_rate": pct(total_execution_errors, max(total_sql_generated, 1)),
            "unique_sql_variations": len(unique_sqls_global),
            "avg_unique_sql_variations_per_question": safe_mean(sql_variation_counts) or 0.0,
        },
        "confidence_statistics": {
            "count": len(all_confidences),
            "mean_confidence": mean_conf,
            "median_confidence": median_conf,
            "high_confidence_count": high_conf,
            "low_confidence_count": low_conf,
            "high_confidence_rate": pct(high_conf, max(len(all_confidences), 1)),
            "low_confidence_rate": pct(low_conf, max(len(all_confidences), 1)),
        },
        "accuracy_by_difficulty": {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": pct(value["correct"], value["total"]),
            }
            for key, value in sorted(difficulty_stats.items())
        },
        "accuracy_by_database": {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": pct(value["correct"], value["total"]),
            }
            for key, value in sorted(
                database_stats.items(),
                key=lambda item: (-pct(item[1]["correct"], item[1]["total"]), item[0]),
            )
        },
        "selection_phase_statistics": {
            "selection_made": selection_made,
            "selection_rate": pct(selection_made, total_questions),
            "majority_vote_used": majority_vote_used,
            "majority_vote_rate": pct(majority_vote_used, total_questions),
        },
    }
    return report


def print_header(title: str) -> None:
    print(title)


def print_separator(char: str = "-", width: int = 100) -> None:
    print(char * width)


def fmt_ms(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def fmt_num(value: Optional[float], decimals: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}"


def console_text(preferred: str, fallback: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        preferred.encode(encoding)
        return preferred
    except UnicodeEncodeError:
        return fallback


def print_report(report: Dict[str, Any], output_path: Path) -> None:
    overall = report["overall"]
    execution = report["execution_time_statistics"]
    generation = report["generation_statistics"]
    confidence = report["confidence_statistics"]
    difficulty = report["accuracy_by_difficulty"]
    databases = report["accuracy_by_database"]
    selection = report["selection_phase_statistics"]

    print_separator("=")
    print(" " * 30 + "DETAILED BENCHMARK REPORT")
    print_separator("=")
    print()

    print(console_text("📊 OVERALL STATISTICS", "OVERALL STATISTICS"))
    print_separator("-")
    print(f"  Total Questions:     {overall['total_questions']}")
    print(f"  Correct:             {overall['correct']} ({overall['accuracy']:.2f}%)")
    print(f"  Incorrect:           {overall['incorrect']} ({100.0 - overall['accuracy']:.2f}%)")
    print()

    print("EXECUTION TIME STATISTICS (All SQL Queries)")
    print_separator("-")
    print(f"  Total Queries Executed:  {execution['total_queries_executed']}")
    print(f"  Mean Execution Time:     {fmt_ms(execution['mean_ms'])}")
    print(f"  Median Execution Time:   {fmt_ms(execution['median_ms'])}")
    print(f"  Std Dev:                 {fmt_ms(execution['std_dev_ms'])}")
    print(f"  Min Execution Time:      {fmt_ms(execution['min_ms'])}")
    print(f"  Max Execution Time:      {fmt_ms(execution['max_ms'])}")
    print()

    print(console_text("🔧 GENERATION STATISTICS", "GENERATION STATISTICS"))
    print_separator("-")
    print(f"  Total SQL Generated:     {generation['total_sql_generated']}")
    print(
        f"  Valid Executions:        {generation['valid_executions']} "
        f"({generation['validation_success_rate']:.1f}% success rate)"
    )
    print(
        f"  Execution Errors:        {generation['execution_errors']} "
        f"({generation['execution_error_rate']:.1f}%)"
    )
    print(
        f"  Unique SQL Variations:   {generation['unique_sql_variations']} "
        f"(avg {generation['avg_unique_sql_variations_per_question']:.1f} per question)"
    )
    print()

    print(console_text("📈 CONFIDENCE STATISTICS (Majority Voting)", "CONFIDENCE STATISTICS (Majority Voting)"))
    print_separator("-")
    print(f"  Mean Confidence:         {fmt_num(confidence['mean_confidence'])}")
    print(f"  Median Confidence:       {fmt_num(confidence['median_confidence'])}")
    print(
        f"  High Confidence (>0.5):  {confidence['high_confidence_count']} "
        f"({confidence['high_confidence_rate']:.1f}%)"
    )
    print(
        f"  Low Confidence (<0.2):   {confidence['low_confidence_count']} "
        f"({confidence['low_confidence_rate']:.1f}%)"
    )
    print()

    print(console_text("📚 ACCURACY BY DIFFICULTY", "ACCURACY BY DIFFICULTY"))
    print_separator("-")
    for key, value in difficulty.items():
        print(
            f"  {key:<15} {value['correct']:>4}/{value['total']:<4} "
            f"({value['accuracy']:>6.2f}%)"
        )
    print()

    print(console_text("🗄️  ACCURACY BY DATABASE", "ACCURACY BY DATABASE"))
    print_separator("-")
    print(f"  {'Database':<40} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 70}")
    for key, value in databases.items():
        print(
            f"  {key:<40} {value['correct']:>8} {value['total']:>8} "
            f"{value['accuracy']:>9.2f}%"
        )
    print()

    print(console_text("🎯 SQL SELECTION PHASE STATISTICS", "SQL SELECTION PHASE STATISTICS"))
    print_separator("-")
    print(
        f"  Selection Made:          {selection['selection_made']} "
        f"({selection['selection_rate']:.1f}%)"
    )
    print(
        f"  Majority Vote Used:      {selection['majority_vote_used']} "
        f"({selection['majority_vote_rate']:.1f}%)"
    )
    print()

    print(console_text(f"💾 Detailed report saved to: {output_path}", f"Detailed report saved to: {output_path}"))
    print_separator("=")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a detailed benchmark report from result JSON.")
    parser.add_argument("--input", required=True, help="Path to the per-question results JSON file")
    parser.add_argument(
        "--output",
        default="./detailed_report.json",
        help="Path to save the structured report JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    results = load_results(input_path)
    report = build_report(results, input_path)

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report, output_path)


if __name__ == "__main__":
    main()

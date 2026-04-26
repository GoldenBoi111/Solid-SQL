#!/usr/bin/env python3
"""
Evaluate the SolidSQL pipeline with pre-built indices and LoRA adapter.

This script processes a JSON file containing questions and database IDs,
generates SQL queries using the full SolidSQL pipeline, and computes
accuracy and statistics per database.

Usage:
    python evaluate_pipeline.py --questions path/to/questions.json --databases path/to/databases --indices path/to/index.json --adapter path/to/adapter --output path/to/results.json
"""

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from solidsql import SolidSQL


def load_schema_for_db(db_path: str) -> str:
    """
    Load schema for a SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        String representation of the database schema
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_lines = []
        for table in tables:
            table_name = table[0]
            
            # Get table info (columns)
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Format columns as "name(type)"
            column_defs = [f"{col[1]}({col[2]})" for col in columns]
            schema_lines.append(f"{table_name}({', '.join(column_defs)})")
        
        conn.close()
        return "\n".join(schema_lines)
    except Exception as e:
        print(f"Warning: Could not load schema for {db_path}: {e}")
        return ""


def execute_sql_and_fetch_results(db_path: str, sql: str) -> List[Tuple]:
    """
    Execute SQL query on database and return results.
    
    Args:
        db_path: Path to the SQLite database file
        sql: SQL query to execute
        
    Returns:
        Query results as list of tuples
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"Warning: Could not execute SQL on {db_path}: {e}")
        return []


def compare_results(generated_results: List[Tuple], ground_truth_results: List[Tuple]) -> bool:
    """
    Compare query results for equality.
    
    Args:
        generated_results: Results from generated SQL
        ground_truth_results: Results from ground truth SQL
        
    Returns:
        True if results match, False otherwise
    """
    # Execution accuracy: check if both queries executed successfully
    # If both executed, consider it correct (execution accuracy metric)
    # If you want exact result matching, uncomment the code below
    
    # For execution accuracy, we just need both to execute without error
    # The actual comparison is already done by checking if results are not empty
    # when they should not be, or checking specific values
    
    # Convert to sets for comparison (ignoring order)
    try:
        gen_set = set(generated_results)
        gt_set = set(ground_truth_results)
        return gen_set == gt_set
    except:
        # If conversion fails, compare as lists
        return generated_results == ground_truth_results


def evaluate_questions(
    questions_data: List[Dict],
    databases_dir: str,
    solidsql: SolidSQL,
    output_file: str
) -> Dict:
    """
    Evaluate all questions and compute statistics.
    
    Args:
        questions_data: List of question dictionaries
        databases_dir: Path to databases directory
        solidsql: Initialized SolidSQL system
        output_file: Path to output results file
        
    Returns:
        Evaluation statistics
    """
    results = []
    db_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "executed": 0,
        "errors": 0,
        "total_time": 0.0
    })
    
    # Cache database schemas and connections
    db_cache = {}
    
    print(f"Processing {len(questions_data)} questions...")
    
    for i, item in enumerate(questions_data):
        question_id = item.get("question_id", i)
        question = item["question"]
        db_id = item["db_id"]
        ground_truth_sql = item.get("SQL", item.get("sql", ""))
        
        print(f"Processing question {i+1}/{len(questions_data)}: {question_id}")
        
        # Find database file
        db_path = os.path.join(databases_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            # Try alternative naming
            db_path = os.path.join(databases_dir, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                print(f"Warning: Database not found for {db_id}")
                continue
        
        # Load schema from cache or database
        if db_id not in db_cache:
            schema_text = load_schema_for_db(db_path)
            if not schema_text:
                print(f"Warning: Could not load schema for {db_id}")
                continue
            db_cache[db_id] = {
                "schema": schema_text,
                "path": db_path
            }
        else:
            schema_text = db_cache[db_id]["schema"]
        
        # Track statistics
        db_stats[db_id]["total"] += 1
        start_time = time.time()
        
        try:
            # Generate SQL
            result = solidsql.generate_sql(
                question=question,
                schema_text=schema_text,
                top_n=3,
                round_2_refinement=True
            )
            
            # Use the refined SQL if available, otherwise fall back to round 1 SQL
            if result.get("refined_sql"):
                generated_sql = result["refined_sql"]
            else:
                generated_sql = result.get("round_1_sql", "")
            execution_time = time.time() - start_time
            db_stats[db_id]["total_time"] += execution_time
            
            # Execute both SQL queries
            generated_results = execute_sql_and_fetch_results(db_path, generated_sql)
            ground_truth_results = execute_sql_and_fetch_results(db_path, ground_truth_sql)
            
            db_stats[db_id]["executed"] += 1
            
            # Execution accuracy: check if both queries produce same results
            is_correct = compare_results(generated_results, ground_truth_results)
                
            if is_correct:
                db_stats[db_id]["correct"] += 1
            
            results.append({
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "generated_sql": generated_sql,
                "ground_truth_sql": ground_truth_sql,
                "generated_results_count": len(generated_results),
                "ground_truth_results_count": len(ground_truth_results),
                "is_correct": is_correct,
                "execution_time": execution_time,
                "retrieved_examples": len(result["retrieval_results"])
            })
            
        except Exception as e:
            execution_time = time.time() - start_time
            db_stats[db_id]["total_time"] += execution_time
            db_stats[db_id]["errors"] += 1
            
            results.append({
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "generated_sql": "",
                "ground_truth_sql": ground_truth_sql,
                "error": str(e),
                "execution_time": execution_time,
                "is_correct": False
            })
            
            print(f"Error processing question {question_id}: {e}")
    
    # Compute final statistics
    overall_stats = {
        "total_questions": len(questions_data),
        "processed_questions": sum(stats["total"] for stats in db_stats.values()),
        "correct_answers": sum(stats["correct"] for stats in db_stats.values()),
        "executed_queries": sum(stats["executed"] for stats in db_stats.values()),
        "errors": sum(stats["errors"] for stats in db_stats.values()),
        "average_execution_time": sum(stats["total_time"] for stats in db_stats.values()) / 
                                  max(sum(stats["total"] for stats in db_stats.values()), 1)
    }
    
    if overall_stats["executed_queries"] > 0:
        overall_stats["accuracy"] = overall_stats["correct_answers"] / overall_stats["executed_queries"]
    else:
        overall_stats["accuracy"] = 0.0
    
    # Per-database statistics
    for db_id in db_stats:
        stats = db_stats[db_id]
        if stats["executed"] > 0:
            stats["accuracy"] = stats["correct"] / stats["executed"]
        else:
            stats["accuracy"] = 0.0
        if stats["total"] > 0:
            stats["average_execution_time"] = stats["total_time"] / stats["total"]
        else:
            stats["average_execution_time"] = 0.0
    
    # Save results
    output_data = {
        "overall_statistics": overall_stats,
        "per_database_statistics": dict(db_stats),
        "detailed_results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_data


def print_summary(stats: Dict):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    overall = stats["overall_statistics"]
    print(f"Total questions: {overall['total_questions']}")
    print(f"Processed questions: {overall['processed_questions']}")
    print(f"Executed queries: {overall['executed_queries']}")
    print(f"Correct answers: {overall['correct_answers']}")
    print(f"Errors: {overall['errors']}")
    print(f"Accuracy: {overall['accuracy']:.4f}")
    print(f"Average execution time: {overall['average_execution_time']:.4f}s")
    
    print("\nPer-database statistics:")
    for db_id, db_stats in stats["per_database_statistics"].items():
        print(f"  {db_id}:")
        print(f"    Questions: {db_stats['total']}")
        print(f"    Accuracy: {db_stats['accuracy']:.4f}")
        print(f"    Average time: {db_stats['average_execution_time']:.4f}s")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate SolidSQL pipeline")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--databases", required=True, help="Path to databases directory")
    parser.add_argument("--metadata-index", required=True, help="Path to metadata JSON file (e.g., retrieval_index.metadata.json)")
    parser.add_argument("--question-index", required=False, help="Path to question FAISS index file (optional)")
    parser.add_argument("--sql-index", required=False, help="Path to SQL FAISS index file (optional)")
    parser.add_argument("--adapter", default="./schema_linking_output/lora_adapter", 
                        help="Path to LoRA adapter")
    parser.add_argument("--output", default="evaluation_results.json", 
                        help="Path to output results file")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.questions):
        print(f"Error: Questions file not found: {args.questions}")
        return
    
    if not os.path.exists(args.databases):
        print(f"Error: Databases directory not found: {args.databases}")
        return
    
    if not os.path.exists(args.metadata_index):
        print(f"Error: Metadata index file not found: {args.metadata_index}")
        return
    
    # Load questions
    print(f"Loading questions from {args.questions}...")
    with open(args.questions, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # Initialize SolidSQL with skeleton extraction skipped
    print("Initializing SolidSQL system...")
    solidsql = SolidSQL(
        candidate_examples=[],  # Empty since we'll load from indices
        adapter_path=args.adapter,
        build_index=False,  # Don't build new index
        skip_skeleton_extraction=True  # Skip question skeleton extraction for evaluation
    )
    
    # Load pre-built FAISS indices
    print(f"Loading FAISS indices from {args.metadata_index}...")
    solidsql.load_retrieval_index(
        args.metadata_index,
        question_index_path=args.question_index,
        sql_index_path=args.sql_index
    )
    
    # Evaluate questions
    print("Starting evaluation...")
    stats = evaluate_questions(questions_data, args.databases, solidsql, args.output)
    
    # Print summary
    print_summary(stats)
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
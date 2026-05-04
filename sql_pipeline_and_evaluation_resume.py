#!/usr/bin/env python3
"""
Resumable SQL generation pipeline plus evaluation.

This is a separate copy of the main evaluator with resume support:
- scans an existing question-logs/results folder
- skips question_ids already present there
- lets you choose which GPUs to use for the remaining questions
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from sql_pipeline_and_evaluation import (
    BaseModelSQLPipeline,
    load_schema_linking_results,
    merge_shard_outputs,
    run_evaluation_worker,
    split_questions_into_shards,
    write_combined_question_logs,
)


def load_records_from_results_dir(results_dir: str) -> List[Dict[str, Any]]:
    results_path = Path(results_dir)
    records: List[Dict[str, Any]] = []

    root_file = results_path / "all_outputs.json"
    if root_file.exists():
        try:
            loaded = json.loads(root_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records.extend(loaded)
        except Exception:
            pass

    for gpu_dir in sorted(path for path in results_path.iterdir() if path.is_dir() and path.name.startswith("gpu_")):
        log_file = gpu_dir / "all_outputs.json"
        if not log_file.exists():
            continue
        try:
            loaded = json.loads(log_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records.extend(loaded)
        except Exception:
            continue

    deduped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        if question_id is None:
            continue
        deduped[str(question_id)] = record
    return list(deduped.values())


def load_completed_question_ids(results_dir: str) -> set[str]:
    return {str(record.get("question_id")) for record in load_records_from_results_dir(results_dir)}


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_database = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    correct = 0
    errors = 0

    for record in records:
        db_id = record.get("db_id", "unknown")
        per_database[db_id]["total"] += 1
        if record.get("is_correct"):
            correct += 1
            per_database[db_id]["correct"] += 1
        if record.get("execution_error") or record.get("error") or record.get("cuda_device_assert"):
            errors += 1
            per_database[db_id]["errors"] += 1

    total = len(records)
    return {
        "overall": {
            "total_questions": total,
            "correct_answers": correct,
            "errors": errors,
            "execution_accuracy": correct / max(total, 1),
        },
        "per_database": dict(per_database),
    }


def print_resume_summary(records: List[Dict[str, Any]], completed: int, remaining: int) -> None:
    summary = summarize_records(records)
    overall = summary["overall"]
    print("\n" + "=" * 60)
    print("RESUME SUMMARY")
    print("=" * 60)
    print(f"Completed questions found: {completed}")
    print(f"Remaining questions to run: {remaining}")
    print(f"Total logged questions: {overall['total_questions']}")
    print(f"Correct answers: {overall['correct_answers']}")
    print(f"Errors: {overall['errors']}")
    print(f"Execution accuracy: {overall['execution_accuracy']:.4f}")


def build_gpu_ranges(
    questions: List[Dict[str, Any]],
    gpu_ids: List[int],
    gpu_starts: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    if not gpu_ids:
        return []

    total_questions = len(questions)
    if gpu_starts is None or not gpu_starts:
        chunk_size = (total_questions + len(gpu_ids) - 1) // len(gpu_ids)
        gpu_starts = [index * chunk_size for index in range(len(gpu_ids))]
    if len(gpu_starts) != len(gpu_ids):
        raise ValueError("--gpu-starts must have the same number of entries as --gpu-ids")

    ranges: List[Dict[str, Any]] = []
    for index, gpu_id in enumerate(gpu_ids):
        start_index = max(0, gpu_starts[index])
        next_start = gpu_starts[index + 1] if index + 1 < len(gpu_starts) else total_questions
        end_index = min(total_questions, next_start)
        if index + 1 >= len(gpu_starts):
            end_index = total_questions
        if start_index >= total_questions:
            continue
        if end_index <= start_index:
            end_index = total_questions
        ranges.append(
            {
                "gpu_id": gpu_id,
                "start_index": start_index,
                "end_index": end_index,
                "questions": questions[start_index:end_index],
            }
        )
    return ranges


def print_gpu_ranges(gpu_ranges: List[Dict[str, Any]]) -> None:
    print("\nGPU assignment plan:")
    for gpu_range in gpu_ranges:
        questions = gpu_range["questions"]
        first_qid = questions[0].get("question_id") if questions else None
        last_qid = questions[-1].get("question_id") if questions else None
        print(
            f"  GPU {gpu_range['gpu_id']}: indices {gpu_range['start_index']}:{gpu_range['end_index']} "
            f"({len(questions)} questions, question_id {first_qid} -> {last_qid})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable SQL pipeline and evaluation")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--databases", required=True, help="Path to databases directory")
    parser.add_argument("--metadata-index", help="Path to retrieval metadata JSON")
    parser.add_argument("--question-index", help="Path to question FAISS index")
    parser.add_argument("--sql-index", help="Path to SQL FAISS index")
    parser.add_argument("--schema-linking-results", help="Path to schema linking batch output JSON")
    parser.add_argument("--base-model", default="openai/gpt-oss-20b", help="Base model name or local path")
    parser.add_argument("--round-1-max-new-tokens", type=int, default=4096)
    parser.add_argument("--round-2-max-new-tokens", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, help="Number of parallel workers")
    parser.add_argument("--gpu-ids", default="0,1,2,3", help="Comma-separated GPU ids to use")
    parser.add_argument(
        "--gpu-starts",
        help="Comma-separated question start indices for each GPU id; next start becomes the upper bound",
    )
    parser.add_argument("--output", default="sql_pipeline_evaluation_results.json", help="Path to output results file")
    parser.add_argument(
        "--question-logs-dir",
        help="Root directory for GPU-sharded log folders and merged all_outputs.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip question_ids already present in the logs folder and continue the remainder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print GPU/question range assignment and exit without running workers",
    )

    args = parser.parse_args()

    if not os.path.exists(args.questions):
        raise FileNotFoundError(f"Questions file not found: {args.questions}")
    if not os.path.exists(args.databases):
        raise FileNotFoundError(f"Databases directory not found: {args.databases}")

    question_logs_dir = args.question_logs_dir
    if not question_logs_dir:
        output_path = Path(args.output)
        question_logs_dir = str(output_path.with_suffix("")) + "_question_logs"
    Path(question_logs_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Loading questions from {args.questions}...")
    questions_data = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    completed_question_ids: set[str] = set()
    if args.resume:
        print(f"[Setup] Resuming from logs directory: {question_logs_dir}")
        completed_question_ids = load_completed_question_ids(question_logs_dir)
        print(f"[Setup] Found {len(completed_question_ids)} completed question_ids")

    pending_questions: List[Dict[str, Any]] = []
    for index, item in enumerate(questions_data):
        question_id = str(item.get("question_id", index))
        if question_id not in completed_question_ids:
            pending_questions.append(item)

    print(f"[Setup] Pending questions to run: {len(pending_questions)}")

    if not pending_questions and args.resume:
        existing_records = load_records_from_results_dir(question_logs_dir)
        print_resume_summary(existing_records, len(completed_question_ids), 0)
        write_combined_question_logs(question_logs_dir, questions_data)
        return

    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("Provide at least one GPU id via --gpu-ids")

    gpu_starts = None
    if args.gpu_starts:
        gpu_starts = [int(item.strip()) for item in args.gpu_starts.split(",") if item.strip()]

    num_workers = args.num_workers or len(gpu_ids)
    if num_workers > len(gpu_ids):
        raise ValueError("--num-workers cannot exceed the number of GPU ids provided")

    selected_gpu_ids = gpu_ids[:num_workers]
    selected_gpu_starts = gpu_starts[:num_workers] if gpu_starts is not None else None
    print(f"[Setup] Using workers on GPU ids: {selected_gpu_ids}")
    if selected_gpu_starts is not None:
        print(f"[Setup] Using GPU start offsets: {selected_gpu_starts}")
    print(f"[Setup] Writing GPU-sharded logs to {question_logs_dir}...")

    gpu_ranges = build_gpu_ranges(
        pending_questions,
        selected_gpu_ids,
        selected_gpu_starts,
    )
    print_gpu_ranges(gpu_ranges)
    if args.dry_run:
        print("\n[Dry Run] No workers launched.")
        return
    temp_dir = Path(tempfile.mkdtemp(prefix="sql_eval_resume_shards_"))
    worker_outputs = [temp_dir / f"worker_{index}.json" for index in range(len(gpu_ranges))]

    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []

    for worker_index, (gpu_range, worker_output) in enumerate(zip(gpu_ranges, worker_outputs)):
        gpu_id = gpu_range["gpu_id"]
        shard = gpu_range["questions"]
        gpu_logs_dir = Path(question_logs_dir) / f"gpu_{gpu_id}"
        print(
            f"[Setup] GPU {gpu_id} will process question indices "
            f"{gpu_range['start_index']}:{gpu_range['end_index']}"
        )
        process = ctx.Process(
            target=run_evaluation_worker,
            args=(
                worker_index,
                gpu_id,
                shard,
                str(worker_output),
                str(gpu_logs_dir),
                args.base_model,
                args.databases,
                args.metadata_index,
                args.question_index,
                args.sql_index,
                args.schema_linking_results,
                args.round_1_max_new_tokens,
                args.round_2_max_new_tokens,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Evaluation worker failed with exit code {process.exitcode}")

    shard_outputs = [
        json.loads(Path(worker_output).read_text(encoding="utf-8"))
        for worker_output in worker_outputs
        if Path(worker_output).exists()
    ]
    processed_questions = []
    for gpu_range in gpu_ranges:
        processed_questions.extend(gpu_range["questions"])
    merged_new_stats = merge_shard_outputs(shard_outputs, processed_questions)
    Path(args.output).write_text(
        json.dumps(merged_new_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_combined_question_logs(question_logs_dir, questions_data)
    all_records = load_records_from_results_dir(question_logs_dir)
    print_resume_summary(all_records, len(completed_question_ids), len(pending_questions))
    print(f"\n[Run] Detailed results saved to: {args.output}")
    print(f"[Run] GPU-sharded logs saved to: {question_logs_dir}")


if __name__ == "__main__":
    main()

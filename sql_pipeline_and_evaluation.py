#!/usr/bin/env python3
"""
Base-model SQL generation pipeline plus evaluation.

This file intentionally does not load or use the LoRA adapter. It handles:
- Round 1 SQL generation
- Round 2 structural refinement
- SQL validation and optional repair
- Evaluation metrics and per-question logging
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sqlglot
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema_linking.skeleton_similarity import SkeletonSimilarity
from schema_linking.sql_skeleton_extractor import SQLSkeletonExtractor

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def summarize_text(text: str, limit: int = 240) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def summarize_sql(sql: str, limit: int = 180) -> str:
    return summarize_text(sql, limit=limit)


def load_schema_for_db(db_path: str) -> str:
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_lines = []
        for (table_name,) in tables:
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            columns = cursor.fetchall()
            column_defs = [f"{column[1]}({column[2]})" for column in columns]
            schema_lines.append(f'{table_name}({", ".join(column_defs)})')

        connection.close()
        return "\n".join(schema_lines)
    except Exception as exc:
        print(f"Warning: Could not load schema for {db_path}: {exc}")
        return ""


def schema_text_to_json(schema_text: str) -> Dict[str, Any]:
    tables: List[Dict[str, Any]] = []
    for raw_line in schema_text.splitlines():
        line = raw_line.strip()
        if not line or "(" not in line or not line.endswith(")"):
            continue

        table_name, raw_columns = line.split("(", 1)
        parsed_columns = []
        for raw_column in raw_columns[:-1].split(","):
            column_text = raw_column.strip()
            if not column_text or "(" not in column_text:
                continue
            column_name, column_type = column_text.split("(", 1)
            parsed_columns.append(
                {
                    "name": column_name.strip(),
                    "type": column_type.rstrip(")").strip(),
                }
            )

        tables.append({"name": table_name.strip(), "columns": parsed_columns})

    return {"tables": tables}


def format_schema_json(schema_json: Dict[str, Any]) -> str:
    return json.dumps(schema_json, indent=2, ensure_ascii=False)


def load_schema_linking_results(path: str) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_question_id = payload.get("by_question_id", {})
    return {str(key): value for key, value in by_question_id.items()}


def schema_linking_to_schema_json(schema_linking_record: Dict[str, Any]) -> Dict[str, Any]:
    linked = schema_linking_record.get("schema_linking", schema_linking_record)
    tables = linked.get("tables", [])
    columns = linked.get("columns", {})
    joins = linked.get("joins", [])
    filters = linked.get("filters", [])
    intent = linked.get("intent", "")

    schema_tables = []
    for table_name in tables:
        table_columns = columns.get(table_name, [])
        schema_tables.append(
            {
                "name": table_name,
                "columns": [{"name": column_name} for column_name in table_columns],
            }
        )

    return {
        "tables": schema_tables,
        "joins": joins,
        "filters": filters,
        "intent": intent,
    }


def execute_sql_and_fetch_results(db_path: str, sql: str) -> List[Tuple]:
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        connection.close()
        return results
    except Exception as exc:
        print(f"Warning: Could not execute SQL on {db_path}: {exc}")
        return []


def compare_execution_results(
    generated_results: List[Tuple],
    ground_truth_results: List[Tuple],
) -> bool:
    try:
        return set(generated_results) == set(ground_truth_results)
    except Exception:
        return generated_results == ground_truth_results


def parse_sql(sql: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        normalized = parsed.sql(dialect=dialect)
        return True, normalized, None
    except sqlglot.errors.ParseError as exc:
        return False, None, str(exc)


def clean_sql_output(sql: str) -> str:
    sql = (sql or "").strip()
    if not sql:
        return ""

    for prefix in ("SQL:", "SQLite:", "Query:", "Answer:"):
        if sql.upper().startswith(prefix.upper()):
            sql = sql[len(prefix):].strip()
            break

    if sql.startswith("```sql"):
        sql = sql[6:]
    elif sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    lines = sql.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped
            and (
                "SELECT" in stripped.upper()
                or "WITH" in stripped.upper()
                or not stripped.startswith(("#", "--"))
            )
        ):
            cleaned_lines.append(stripped)

    cleaned_sql = " ".join(cleaned_lines).strip()
    select_index = cleaned_sql.upper().find("SELECT")
    with_index = cleaned_sql.upper().find("WITH")

    if select_index != -1 and (with_index == -1 or select_index < with_index):
        cleaned_sql = cleaned_sql[select_index:]
    elif with_index != -1:
        cleaned_sql = cleaned_sql[with_index:]

    if ";" in cleaned_sql:
        cleaned_sql = cleaned_sql.split(";")[0] + ";"

    placeholder_sql = {
        "SELECT ...",
        "SELECT ...;",
        "SELECT ???",
        "SELECT ???;",
        "WITH ...",
        "WITH ...;",
    }
    if cleaned_sql.upper() in {item.upper() for item in placeholder_sql}:
        return ""

    return cleaned_sql.strip()


def exact_match_sql(predicted_sql: str, ground_truth_sql: str, dialect: str = "sqlite") -> bool:
    predicted_ok, predicted_normalized, _ = parse_sql(predicted_sql, dialect=dialect)
    ground_truth_ok, ground_truth_normalized, _ = parse_sql(ground_truth_sql, dialect=dialect)
    if not predicted_ok or not ground_truth_ok:
        return False
    return predicted_normalized == ground_truth_normalized


class SQLStructuralRetriever:
    """SQL-only structural retriever with optional FAISS-backed search."""

    def __init__(
        self,
        candidate_examples: List[Dict[str, str]],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sql_dialect: str = "sqlite",
    ) -> None:
        self.candidates = candidate_examples
        self.sql_dialect = sql_dialect
        self.sql_extractor = SQLSkeletonExtractor(dialect=sql_dialect)
        self.similarity = SkeletonSimilarity(embedding_model=embedding_model)
        self.sql_skeletons: List[str] = []
        self.sql_index = None

    def build_index(self, sql_skeletons: Optional[List[str]] = None) -> None:
        self.sql_skeletons = sql_skeletons or self.sql_extractor.extract_batch(
            [example["sql"] for example in self.candidates],
            show_progress=False,
        )

        if not FAISS_AVAILABLE or not self.sql_skeletons:
            return

        embeddings = self.similarity.get_sql_embeddings(self.sql_skeletons)
        embedding_array = np.array(embeddings, dtype=np.float32)
        self.sql_index = faiss.IndexFlatL2(embedding_array.shape[1])
        self.sql_index.add(embedding_array)

    def load_index(self, metadata_path: str, sql_index_path: Optional[str] = None) -> None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.candidates = metadata.get("candidates", [])
        self.sql_skeletons = metadata.get("sql_skeletons", [])

        if sql_index_path and FAISS_AVAILABLE and Path(sql_index_path).exists():
            self.sql_index = faiss.read_index(sql_index_path)
        elif self.candidates:
            self.build_index(sql_skeletons=self.sql_skeletons or None)

    def retrieve(self, sql: str, top_n: int = 3) -> List[Dict[str, Any]]:
        if not self.candidates:
            return []

        target_skeleton = self.sql_extractor.extract(sql)

        if FAISS_AVAILABLE and self.sql_index is not None and self.sql_skeletons:
            embedding = self.similarity.get_sql_embeddings([target_skeleton])[0]
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            distances, indices = self.sql_index.search(embedding_array, top_n)
            ranked = []
            for distance, index in zip(distances[0], indices[0]):
                if index < 0 or index >= len(self.candidates):
                    continue
                ranked.append(
                    {
                        "example": self.candidates[index],
                        "similarity_score": float(1.0 / (1.0 + distance)),
                        "candidate_sql_skeleton": self.sql_skeletons[index],
                    }
                )
            return ranked

        similarities = self.similarity.sql_similarity_batch(target_skeleton, self.sql_skeletons)
        indexed = sorted(enumerate(similarities), key=lambda item: item[1], reverse=True)[:top_n]
        return [
            {
                "example": self.candidates[index],
                "similarity_score": score,
                "candidate_sql_skeleton": self.sql_skeletons[index],
            }
            for index, score in indexed
        ]


class BaseModelSQLPipeline:
    """Two-stage SQL generation pipeline using only the base model."""

    def __init__(
        self,
        base_model: str = "openai/gpt-oss-20b",
        candidate_examples: Optional[List[Dict[str, str]]] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sql_dialect: str = "sqlite",
    ) -> None:
        self.base_model_name = base_model
        self.candidate_examples = candidate_examples or []
        self.embedding_model = embedding_model
        self.sql_dialect = sql_dialect
        self.max_seq_length = 2048
        self._model = None
        self._tokenizer = None
        self.sql_extractor = SQLSkeletonExtractor(dialect=sql_dialect)
        self.structural_retriever = SQLStructuralRetriever(
            candidate_examples=self.candidate_examples,
            embedding_model=embedding_model,
            sql_dialect=sql_dialect,
        )
        if self.candidate_examples:
            self.structural_retriever.build_index()

    def _load_model(self) -> None:
        if self._model is not None:
            return

        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def load_structural_index(self, metadata_path: str, sql_index_path: Optional[str] = None) -> None:
        self.structural_retriever.load_index(metadata_path, sql_index_path=sql_index_path)
        self.candidate_examples = self.structural_retriever.candidates

    def _generate_text(self, prompt: str, max_new_tokens: int = 768) -> str:
        self._load_model()
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_length = int(inputs["attention_mask"].sum(dim=1).tolist()[0])
        generated_tokens = outputs[0][prompt_length:]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _select_few_shot_examples(self, question: str, top_n: int = 3) -> List[Dict[str, str]]:
        if not self.candidate_examples:
            return []

        question_terms = set(question.lower().split())
        scored_examples = []
        for example in self.candidate_examples:
            example_terms = set(example.get("question", "").lower().split())
            overlap = len(question_terms & example_terms)
            scored_examples.append((overlap, example))

        scored_examples.sort(key=lambda item: item[0], reverse=True)
        return [example for _, example in scored_examples[:top_n]]

    def _format_few_shot_examples(self, examples: List[Dict[str, str]]) -> str:
        if not examples:
            return "(none provided)"
        return "\n\n".join(
            f"Question: {example['question']}\nSQL: {example['sql']}"
            for example in examples
        )

    def _format_structural_examples(self, examples: List[Dict[str, Any]]) -> str:
        if not examples:
            return "(none provided)"
        return "\n\n".join(
            "Question: {question}\nSQL: {sql}\nSQL Skeleton: {skeleton}".format(
                question=item["example"]["question"],
                sql=item["example"]["sql"],
                skeleton=item["candidate_sql_skeleton"],
            )
            for item in examples
        )

    def _round_1_prompt(
        self,
        question: str,
        schema_json: Dict[str, Any],
        few_shot_examples: List[Dict[str, str]],
    ) -> str:
        return (
            "You are a text-to-SQL generation model.\n\n"
            "Your task is to generate a SQL query based on:\n"
            "- the user question\n"
            "- the database schema\n"
            "- a set of example question-SQL pairs (few-shot demonstrations)\n\n"
            "IMPORTANT:\n"
            "- The SQL you generate is a DRAFT (Round 1 output)\n"
            "- It will be used for further refinement later\n"
            "- Do NOT try to be perfect; focus on following schema and example patterns\n"
            "- Do NOT output explanations\n\n"
            "---\n\n"
            "### DATABASE SCHEMA\n"
            f"{format_schema_json(schema_json)}\n\n"
            "---\n\n"
            "### USER QUESTION\n"
            f"{question}\n\n"
            "---\n\n"
            "### FEW-SHOT EXAMPLES\n"
            f"{self._format_few_shot_examples(few_shot_examples)}\n\n"
            "---\n\n"
            "### TASK\n\n"
            "Study the schema and examples carefully, then generate a SQL query that best answers the user question.\n\n"
            "Pay attention to:\n"
            "- correct table and column usage\n"
            "- similar query patterns from examples\n"
            "- joins, filters, and aggregations where needed\n\n"
            "---\n\n"
            "### OUTPUT FORMAT\n\n"
            "Return ONLY the SQL query.\n"
            "The answer must be a single executable SQLite query starting with SELECT or WITH.\n"
        )

    def _round_2_prompt(
        self,
        question: str,
        schema_json: Dict[str, Any],
        round_1_sql: str,
        structural_examples: List[Dict[str, Any]],
    ) -> str:
        sql_skeleton = self.sql_extractor.extract(round_1_sql)
        return (
            "You are a structure-aware text-to-SQL system.\n\n"
            "Your task is to generate the FINAL SQL query using:\n"
            "1. The database schema\n"
            "2. The user question\n"
            "3. The Round 1 draft SQL\n"
            "4. SQL examples selected based on structural similarity (SQL skeleton matching)\n\n"
            "---\n\n"
            "## IMPORTANT PRINCIPLE\n\n"
            "Unlike Round 1, which relies on semantic and example-based generation,\n"
            "Round 2 prioritizes STRUCTURAL similarity of SQL programs.\n\n"
            "You must:\n"
            "- Use the structure of the Round 1 SQL (not just the question)\n"
            "- Focus on SQL skeleton similarity (joins, filters, aggregations)\n"
            "- Ensure schema correctness and logical consistency\n\n"
            "---\n\n"
            "## DATABASE SCHEMA\n"
            f"{format_schema_json(schema_json)}\n\n"
            "---\n\n"
            "## USER QUESTION\n"
            f"{question}\n\n"
            "---\n\n"
            "## ROUND 1 DRAFT SQL\n"
            f"{round_1_sql}\n\n"
            "---\n\n"
            "## SQL SKELETON OF ROUND 1 SQL\n"
            f"{sql_skeleton}\n\n"
            "---\n\n"
            "## STRUCTURALLY SIMILAR SQL EXAMPLES\n"
            "These examples were retrieved using SQL skeleton edit distance (not text similarity):\n\n"
            f"{self._format_structural_examples(structural_examples)}\n\n"
            "---\n\n"
            "## TASK\n\n"
            "Step 1:\n"
            "Analyze the Round 1 SQL skeleton to understand the intended query structure.\n\n"
            "Step 2:\n"
            "Use the structurally similar SQL examples to refine understanding of:\n"
            "- correct join patterns\n"
            "- correct filtering conditions\n"
            "- correct aggregation logic\n"
            "- correct schema usage\n\n"
            "Step 3:\n"
            "Generate the FINAL SQL query that best answers the user question.\n\n"
            "---\n\n"
            "## RULES\n"
            "- Do NOT output explanations\n"
            "- Do NOT output multiple SQL queries\n"
            "- Use correct schema names exactly\n"
            "- Prioritize structural correctness over lexical similarity\n\n"
            "---\n\n"
            "## OUTPUT FORMAT\n\n"
            "Return ONLY the final SQL query.\n"
            "The answer must be a single executable SQLite query starting with SELECT or WITH.\n"
        )

    def _repair_prompt(
        self,
        question: str,
        schema_json: Dict[str, Any],
        invalid_sql: str,
        parse_error: str,
    ) -> str:
        return (
            "You repair invalid SQLite queries.\n\n"
            "Given the user question, schema JSON, invalid SQL, and parse error, "
            "return a repaired executable SQLite query.\n\n"
            "Rules:\n"
            "- Output SQL only\n"
            "- Return one executable query starting with SELECT or WITH\n\n"
            "## SCHEMA JSON\n"
            f"{format_schema_json(schema_json)}\n\n"
            "## USER QUESTION\n"
            f"{question}\n\n"
            "## INVALID SQL\n"
            f"{invalid_sql}\n\n"
            "## PARSE ERROR\n"
            f"{parse_error}\n"
        )

    def validate_and_repair_sql(
        self,
        question: str,
        schema_json: Dict[str, Any],
        sql: str,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        cleaned_sql = clean_sql_output(sql)
        parsed_ok, normalized_sql, parse_error = parse_sql(cleaned_sql, dialect=self.sql_dialect)

        if parsed_ok:
            return {
                "sql": cleaned_sql,
                "parsed": True,
                "normalized_sql": normalized_sql,
                "repair_attempted": False,
                "repair_error": None,
            }

        repair_prompt = self._repair_prompt(question, schema_json, cleaned_sql, parse_error or "Unknown parse error")
        repaired_text = self._generate_text(repair_prompt, max_new_tokens=max_new_tokens)
        repaired_sql = clean_sql_output(repaired_text)
        repaired_ok, repaired_normalized, repaired_error = parse_sql(repaired_sql, dialect=self.sql_dialect)

        return {
            "sql": repaired_sql if repaired_ok else cleaned_sql,
            "parsed": repaired_ok,
            "normalized_sql": repaired_normalized if repaired_ok else None,
            "repair_attempted": True,
            "repair_error": repaired_error if not repaired_ok else None,
            "original_parse_error": parse_error,
        }

    def generate_sql(
        self,
        question: str,
        schema_json: Dict[str, Any],
        round_1_examples: int = 3,
        round_2_examples: int = 3,
        max_new_tokens: int = 768,
    ) -> Dict[str, Any]:
        few_shot_examples = self._select_few_shot_examples(question, top_n=round_1_examples)
        round_1_prompt = self._round_1_prompt(question, schema_json, few_shot_examples)
        round_1_raw = self._generate_text(round_1_prompt, max_new_tokens=max_new_tokens)
        round_1_sql = clean_sql_output(round_1_raw)
        round_1_validation = self.validate_and_repair_sql(question, schema_json, round_1_sql)
        validated_round_1_sql = round_1_validation["sql"]

        structural_examples = self.structural_retriever.retrieve(
            validated_round_1_sql or round_1_sql,
            top_n=round_2_examples,
        )
        round_2_prompt = self._round_2_prompt(
            question,
            schema_json,
            validated_round_1_sql or round_1_sql,
            structural_examples,
        )
        round_2_raw = self._generate_text(round_2_prompt, max_new_tokens=max_new_tokens)
        round_2_sql = clean_sql_output(round_2_raw)
        round_2_validation = self.validate_and_repair_sql(question, schema_json, round_2_sql)

        final_sql = round_2_validation["sql"] or validated_round_1_sql or round_1_sql
        final_parse_ok, final_normalized, final_parse_error = parse_sql(final_sql, dialect=self.sql_dialect)

        return {
            "question": question,
            "schema_json": schema_json,
            "round_1_sql": round_1_sql,
            "round_1_validation": round_1_validation,
            "structural_examples": structural_examples,
            "round_2_sql": round_2_sql,
            "round_2_validation": round_2_validation,
            "final_sql": final_sql,
            "final_parsed": final_parse_ok,
            "final_normalized_sql": final_normalized,
            "final_parse_error": final_parse_error,
        }

    def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None


class SQLEvaluator:
    """Evaluation harness for the standalone SQL pipeline."""

    def __init__(self, pipeline: BaseModelSQLPipeline, sql_dialect: str = "sqlite") -> None:
        self.pipeline = pipeline
        self.sql_dialect = sql_dialect

    def evaluate_questions(
        self,
        questions_data: List[Dict[str, Any]],
        databases_dir: str,
        output_file: str,
        schema_linking_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        db_stats = defaultdict(
            lambda: {
                "total": 0,
                "correct": 0,
                "exact_match": 0,
                "parsed": 0,
                "executed": 0,
                "errors": 0,
                "total_time": 0.0,
            }
        )
        db_cache: Dict[str, Dict[str, str]] = {}

        print(f"Processing {len(questions_data)} questions...")

        for index, item in enumerate(questions_data):
            question_id = item.get("question_id", index)
            question = item["question"]
            db_id = item["db_id"]
            ground_truth_sql = item.get("SQL", item.get("sql", ""))

            print("\n" + "-" * 80)
            print(f"[Question {index + 1}/{len(questions_data)}] ID: {question_id}")
            print(f"Question: {question}")
            print(f"Database: {db_id}")

            db_path = os.path.join(databases_dir, db_id, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                db_path = os.path.join(databases_dir, f"{db_id}.sqlite")
                if not os.path.exists(db_path):
                    print(f"[Database Lookup] Missing database file for {db_id}")
                    continue

            print("[Stage 1: Schema Load] Loading schema text...")
            if db_id not in db_cache:
                schema_text = load_schema_for_db(db_path)
                if not schema_text:
                    print(f"[Stage 1: Schema Load] Failed for {db_id}")
                    continue
                db_cache[db_id] = {"schema": schema_text, "path": db_path}
                print("[Stage 1: Schema Load] Loaded from SQLite and cached.")
            else:
                schema_text = db_cache[db_id]["schema"]
                print("[Stage 1: Schema Load] Reused cached schema.")
            print(f"[Stage 1: Schema Load] Schema preview: {summarize_text(schema_text)}")

            linked_schema_record = None
            if schema_linking_lookup is not None:
                linked_schema_record = schema_linking_lookup.get(str(question_id))
            schema_json = (
                schema_linking_to_schema_json(linked_schema_record)
                if linked_schema_record is not None
                else schema_text_to_json(schema_text)
            )
            db_stats[db_id]["total"] += 1
            start_time = time.time()

            try:
                print("[Stage 2: Generation] Running base-model SQL pipeline...")
                generation = self.pipeline.generate_sql(
                    question=question,
                    schema_json=schema_json,
                )
                print("[Stage 2: Generation] Pipeline output received.")
                print(f"[Stage 2: Generation] Round 1 SQL: {summarize_sql(generation['round_1_sql'])}")
                print(f"[Stage 2: Generation] Round 2 SQL: {summarize_sql(generation['round_2_sql'])}")
                print(f"[Stage 2: Generation] Final SQL: {summarize_sql(generation['final_sql'])}")

                final_sql = generation["final_sql"]
                execution_time = time.time() - start_time
                db_stats[db_id]["total_time"] += execution_time

                if generation["final_parsed"]:
                    db_stats[db_id]["parsed"] += 1

                print("[Stage 3: Execution] Running generated SQL against SQLite...")
                generated_results = execute_sql_and_fetch_results(db_path, final_sql)
                print(f"[Stage 3: Execution] Generated SQL rows: {len(generated_results)}")
                print("[Stage 3: Execution] Running ground-truth SQL against SQLite...")
                ground_truth_results = execute_sql_and_fetch_results(db_path, ground_truth_sql)
                print(f"[Stage 3: Execution] Ground-truth rows: {len(ground_truth_results)}")

                db_stats[db_id]["executed"] += 1

                execution_match = compare_execution_results(generated_results, ground_truth_results)
                exact_match = exact_match_sql(final_sql, ground_truth_sql, dialect=self.sql_dialect)

                print("[Stage 4: Comparison] Comparing execution results...")
                if execution_match:
                    db_stats[db_id]["correct"] += 1
                    print("[Stage 4: Comparison] Execution result: MATCH")
                else:
                    print("[Stage 4: Comparison] Execution result: MISMATCH")

                if exact_match:
                    db_stats[db_id]["exact_match"] += 1

                results.append(
                    {
                        "question_id": question_id,
                        "db_id": db_id,
                        "question": question,
                        "generated_sql": final_sql,
                        "ground_truth_sql": ground_truth_sql,
                        "schema_linking_used": linked_schema_record is not None,
                        "round_1_sql": generation["round_1_sql"],
                        "round_2_sql": generation["round_2_sql"],
                        "execution_match": execution_match,
                        "exact_match": exact_match,
                        "parsed": generation["final_parsed"],
                        "parse_error": generation["final_parse_error"],
                        "execution_time": execution_time,
                        "retrieved_examples": len(generation["structural_examples"]),
                    }
                )
                print(f"[Stage 5: Outcome] Final status: {'correct' if execution_match else 'incorrect'}")
                print(f"[Stage 5: Outcome] Parsed successfully: {generation['final_parsed']}")
                print(f"[Stage 5: Outcome] Exact match: {exact_match}")
                print(f"[Stage 5: Outcome] Execution time: {execution_time:.4f}s")

            except Exception as exc:
                execution_time = time.time() - start_time
                db_stats[db_id]["total_time"] += execution_time
                db_stats[db_id]["errors"] += 1
                results.append(
                    {
                        "question_id": question_id,
                        "db_id": db_id,
                        "question": question,
                        "generated_sql": "",
                        "ground_truth_sql": ground_truth_sql,
                        "error": str(exc),
                        "execution_time": execution_time,
                        "execution_match": False,
                        "exact_match": False,
                        "parsed": False,
                    }
                )
                print(f"[Stage 5: Outcome] Error processing question {question_id}: {exc}")
                print(f"[Stage 5: Outcome] Execution time before failure: {execution_time:.4f}s")

        overall_stats = {
            "total_questions": len(questions_data),
            "processed_questions": sum(stats["total"] for stats in db_stats.values()),
            "correct_answers": sum(stats["correct"] for stats in db_stats.values()),
            "exact_matches": sum(stats["exact_match"] for stats in db_stats.values()),
            "executed_queries": sum(stats["executed"] for stats in db_stats.values()),
            "parsed_queries": sum(stats["parsed"] for stats in db_stats.values()),
            "errors": sum(stats["errors"] for stats in db_stats.values()),
            "average_execution_time": (
                sum(stats["total_time"] for stats in db_stats.values())
                / max(sum(stats["total"] for stats in db_stats.values()), 1)
            ),
        }
        overall_stats["execution_accuracy"] = (
            overall_stats["correct_answers"] / max(overall_stats["executed_queries"], 1)
        )
        overall_stats["exact_match_accuracy"] = (
            overall_stats["exact_matches"] / max(overall_stats["processed_questions"], 1)
        )
        overall_stats["parsing_success_rate"] = (
            overall_stats["parsed_queries"] / max(overall_stats["processed_questions"], 1)
        )

        for db_id, stats in db_stats.items():
            stats["accuracy"] = stats["correct"] / max(stats["executed"], 1)
            stats["exact_match_accuracy"] = stats["exact_match"] / max(stats["total"], 1)
            stats["parsing_success_rate"] = stats["parsed"] / max(stats["total"], 1)
            stats["average_execution_time"] = stats["total_time"] / max(stats["total"], 1)

        output_data = {
            "overall_statistics": overall_stats,
            "per_database_statistics": dict(db_stats),
            "detailed_results": results,
        }
        Path(output_file).write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_data


def print_summary(stats: Dict[str, Any]) -> None:
    overall = stats["overall_statistics"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions: {overall['total_questions']}")
    print(f"Processed questions: {overall['processed_questions']}")
    print(f"Executed queries: {overall['executed_queries']}")
    print(f"Correct answers: {overall['correct_answers']}")
    print(f"Exact matches: {overall['exact_matches']}")
    print(f"Parsed queries: {overall['parsed_queries']}")
    print(f"Errors: {overall['errors']}")
    print(f"Execution accuracy: {overall['execution_accuracy']:.4f}")
    print(f"Exact match accuracy: {overall['exact_match_accuracy']:.4f}")
    print(f"Parsing success rate: {overall['parsing_success_rate']:.4f}")
    print(f"Average execution time: {overall['average_execution_time']:.4f}s")

    print("\nPer-database statistics:")
    for db_id, db_stats in stats["per_database_statistics"].items():
        print(f"  {db_id}:")
        print(f"    Questions: {db_stats['total']}")
        print(f"    Execution accuracy: {db_stats['accuracy']:.4f}")
        print(f"    Exact match accuracy: {db_stats['exact_match_accuracy']:.4f}")
        print(f"    Parsing success rate: {db_stats['parsing_success_rate']:.4f}")
        print(f"    Average time: {db_stats['average_execution_time']:.4f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Base-model SQL pipeline and evaluation")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--databases", required=True, help="Path to databases directory")
    parser.add_argument("--metadata-index", help="Path to retrieval metadata JSON")
    parser.add_argument("--sql-index", help="Path to SQL FAISS index")
    parser.add_argument("--schema-linking-results", help="Path to schema linking batch output JSON")
    parser.add_argument(
        "--base-model",
        default="openai/gpt-oss-20b",
        help="Base model name or local path",
    )
    parser.add_argument(
        "--output",
        default="sql_pipeline_evaluation_results.json",
        help="Path to output results file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.questions):
        raise FileNotFoundError(f"Questions file not found: {args.questions}")
    if not os.path.exists(args.databases):
        raise FileNotFoundError(f"Databases directory not found: {args.databases}")

    print(f"[Setup] Loading questions from {args.questions}...")
    questions_data = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    print("[Setup] Initializing base-model SQL pipeline...")
    pipeline = BaseModelSQLPipeline(base_model=args.base_model)
    if args.metadata_index and os.path.exists(args.metadata_index):
        print(f"[Setup] Loading structural retrieval data from {args.metadata_index}...")
        pipeline.load_structural_index(args.metadata_index, sql_index_path=args.sql_index)
    print("[Setup] SQL pipeline ready.")

    evaluator = SQLEvaluator(pipeline)
    schema_linking_lookup = None
    if args.schema_linking_results and os.path.exists(args.schema_linking_results):
        print(f"[Setup] Loading schema-linking results from {args.schema_linking_results}...")
        schema_linking_lookup = load_schema_linking_results(args.schema_linking_results)
    print("[Run] Starting evaluation...")
    stats = evaluator.evaluate_questions(
        questions_data,
        args.databases,
        args.output,
        schema_linking_lookup=schema_linking_lookup,
    )
    print_summary(stats)
    print(f"\n[Run] Detailed results saved to: {args.output}")

    print("[Cleanup] Releasing resources...")
    pipeline.shutdown()


if __name__ == "__main__":
    main()

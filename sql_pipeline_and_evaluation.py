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
import multiprocessing as mp
import os
import sqlite3
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import outlines
except ImportError:  # pragma: no cover - optional dependency on the execution machine
    outlines = None

try:
    from pydantic import BaseModel, Field, constr
except ImportError:  # pragma: no cover - optional dependency on the execution machine
    BaseModel = None
    Field = None
    constr = None

from schema_linking.question_skeleton_extractor import SKELETON_EXTRACTION_PROMPT
from schema_linking.skeleton_similarity import SkeletonSimilarity
from schema_linking.sql_skeleton_extractor import SQLSkeletonExtractor

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


SQL_SKELETON_EXTRACTION_PROMPT = """Given a SQL query, extract its structural skeleton by replacing schema-specific details and literal values while preserving the overall SQL structure.

The skeleton (S*) should preserve:
- SQL keywords and clause order
- Join structure
- Aggregation structure
- Filter/operator structure
- Grouping and ordering structure

Replace:
- Table names with [TABLE]
- Column names with [COLUMN]
- Literal values with [VALUE]

Rules:
- Return ONLY the skeleton SQL
- Do NOT explain anything
- Keep the SQL structure intact

Examples:
Input:
SELECT COUNT(*) FROM Singer WHERE Age > 20
Output:
SELECT COUNT(*) FROM [TABLE] WHERE [COLUMN] > [VALUE]

Input:
SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'LAM' AND SUBSTR(T2.Date, 1, 4) = '2012' GROUP BY T1.CustomerID ORDER BY AVG(T2.Consumption) ASC LIMIT 1
Output:
SELECT [COLUMN] FROM [TABLE] AS T1 INNER JOIN [TABLE] AS T2 ON [COLUMN] = [COLUMN] WHERE [COLUMN] = [VALUE] AND SUBSTR([COLUMN], [VALUE], [VALUE]) = [VALUE] GROUP BY [COLUMN] ORDER BY AVG([COLUMN]) ASC LIMIT [VALUE]

SQL:
{sql}

Skeleton SQL:"""

if BaseModel is not None and constr is not None and Field is not None:
    SqlString = constr(
        pattern=r"(?is)^(?:WITH|SELECT)[\s\S]*FROM[\s\S]*$"
    )

    class SqlResponse(BaseModel):
        sql: SqlString = Field(
            description="A single executable SQLite query that starts with SELECT or WITH and contains a FROM clause."
        )
        reasoning: str = Field(description="Short explanation of how the SQL was derived.")
else:  # pragma: no cover - fallback when pydantic is unavailable locally
    class SqlResponse(object):
        sql: str
        reasoning: str


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


def execute_sql_with_metadata(db_path: str, sql: str) -> Tuple[List[Tuple], float, Optional[str]]:
    start_time = time.time()
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        connection.close()
        return results, time.time() - start_time, None
    except Exception as exc:
        print(f"Warning: Could not execute SQL on {db_path}: {exc}")
        return [], time.time() - start_time, str(exc)


def compare_execution_results(
    generated_results: List[Tuple],
    ground_truth_results: List[Tuple],
) -> bool:
    try:
        return set(generated_results) == set(ground_truth_results)
    except Exception:
        return generated_results == ground_truth_results


def summarize_execution_difference(
    generated_results: List[Tuple],
    ground_truth_results: List[Tuple],
) -> Dict[str, Any]:
    try:
        generated_set = set(generated_results)
        ground_truth_set = set(ground_truth_results)
        shared_rows = sorted(generated_set & ground_truth_set)
        generated_only = sorted(generated_set - ground_truth_set)
        ground_truth_only = sorted(ground_truth_set - generated_set)
        return {
            "generated_row_count": len(generated_results),
            "ground_truth_row_count": len(ground_truth_results),
            "shared_rows": shared_rows,
            "generated_only_rows": generated_only,
            "ground_truth_only_rows": ground_truth_only,
        }
    except Exception:
        return {
            "generated_row_count": len(generated_results),
            "ground_truth_row_count": len(ground_truth_results),
            "shared_rows": [],
            "generated_only_rows": [],
            "ground_truth_only_rows": [],
        }


def build_question_log_record(
    *,
    question_id: Any,
    question: str,
    db_id: str,
    difficulty: str,
    gold_sql: str,
    round_1_sql: str,
    round_1_reasoning: str,
    round_2_sql: str,
    round_2_reasoning: str,
    winner_sql: str,
    winner_reasoning: str,
    is_correct: bool,
    execution_times: List[float],
    generated_results: List[Tuple],
    gold_results: List[Tuple],
    round_1_valid: bool,
    round_2_valid: bool,
    final_valid: bool,
    execution_error: Optional[str] = None,
) -> Dict[str, Any]:
    has_sql = bool((winner_sql or "").strip())
    valid_generation = int(has_sql)
    execution_errors = 1 if execution_error else 0
    unique_valid_sqls = 1 if has_sql else 0
    execution_difference = summarize_execution_difference(generated_results, gold_results)
    alternatives = []
    if has_sql:
        best_exec_time = min(execution_times) if execution_times else None
        alternatives.append(
            {
                "sql": winner_sql,
                "confidence": 1.0,
                "count": 1,
                "best_exec_time": best_exec_time,
                "all_sqls_in_group": [winner_sql],
            }
        )

    return {
        "question_id": question_id,
        "question": question,
        "db_id": db_id,
        "difficulty": difficulty or "unknown",
        "gold_sql": gold_sql,
        "ground_truth": gold_sql,
        "round_1_sql": round_1_sql,
        "round_1_reasoning": round_1_reasoning,
        "round_2_sql": round_2_sql,
        "round_2_reasoning": round_2_reasoning,
        "winner_sql": winner_sql,
        "winner_reasoning": winner_reasoning,
        "generated_sql": winner_sql,
        "is_correct": is_correct,
        "winner_confidence": 1.0 if has_sql else 0.0,
        "execution_times": execution_times,
        "generated_results": generated_results,
        "gold_results": gold_results,
        "execution_difference": execution_difference,
        "execution_error": execution_error,
        "high_confidence_alternatives": alternatives,
        "selection": {
            "selected_sql": winner_sql,
            "selected_reasoning": winner_reasoning,
            "reasoning": winner_reasoning if has_sql else "No valid SQL candidate generated",
            "candidates_count": 1 if has_sql else 0,
        },
        "metrics": {
            "generated": 1,
            "execution_errors": execution_errors,
            "valid_generations": valid_generation,
            "unique_valid_sqls": unique_valid_sqls,
            "round_1_valid": int(round_1_valid),
            "round_2_valid": int(round_2_valid),
            "final_valid": int(final_valid),
        },
    }


def write_question_logs_array(logs_dir: str, payloads: List[Dict[str, Any]]) -> None:
    path = Path(logs_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_path = path / "all_outputs.json"
    log_path.write_text(json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8")


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

    max_sql_length = 4000
    if len(cleaned_sql) > max_sql_length:
        cleaned_sql = cleaned_sql[:max_sql_length].rstrip()
        if ";" not in cleaned_sql:
            cleaned_sql += ";"

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


class SolidSQLRetriever:
    """Question- and SQL-skeleton retriever with optional FAISS-backed search."""

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
        self.question_skeletons: List[str] = []
        self.question_index = None
        self.sql_skeletons: List[str] = []
        self.sql_index = None

    def build_index(
        self,
        question_skeletons: Optional[List[str]] = None,
        sql_skeletons: Optional[List[str]] = None,
    ) -> None:
        self.question_skeletons = question_skeletons or []
        self.sql_skeletons = sql_skeletons or self.sql_extractor.extract_batch(
            [example["sql"] for example in self.candidates],
            show_progress=False,
        )

        if FAISS_AVAILABLE and self.question_skeletons:
            question_embeddings = self.similarity.get_question_embeddings(self.question_skeletons)
            question_embedding_array = np.array(question_embeddings, dtype=np.float32)
            self.question_index = faiss.IndexFlatL2(question_embedding_array.shape[1])
            self.question_index.add(question_embedding_array)

        if FAISS_AVAILABLE and self.sql_skeletons:
            embeddings = self.similarity.get_sql_embeddings(self.sql_skeletons)
            embedding_array = np.array(embeddings, dtype=np.float32)
            self.sql_index = faiss.IndexFlatL2(embedding_array.shape[1])
            self.sql_index.add(embedding_array)

    def load_index(
        self,
        metadata_path: str,
        question_index_path: Optional[str] = None,
        sql_index_path: Optional[str] = None,
    ) -> None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.candidates = metadata.get("candidates", [])
        self.question_skeletons = metadata.get("question_skeletons", [])
        self.sql_skeletons = metadata.get("sql_skeletons", [])

        if question_index_path and FAISS_AVAILABLE and Path(question_index_path).exists():
            self.question_index = faiss.read_index(question_index_path)
        elif self.candidates and self.question_skeletons:
            self.build_index(question_skeletons=self.question_skeletons, sql_skeletons=self.sql_skeletons or None)

        if sql_index_path and FAISS_AVAILABLE and Path(sql_index_path).exists():
            self.sql_index = faiss.read_index(sql_index_path)
        elif self.candidates:
            self.build_index(question_skeletons=self.question_skeletons or None, sql_skeletons=self.sql_skeletons or None)

    def retrieve_by_question(self, question_skeleton: str, top_n: int = 3) -> List[Dict[str, Any]]:
        if not self.candidates or not self.question_skeletons:
            return []

        if FAISS_AVAILABLE and self.question_index is not None:
            embedding = self.similarity.get_question_embeddings([question_skeleton])[0]
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            distances, indices = self.question_index.search(embedding_array, top_n)
            ranked = []
            for distance, index in zip(distances[0], indices[0]):
                if index < 0 or index >= len(self.candidates):
                    continue
                ranked.append(
                    {
                        "example": self.candidates[index],
                        "similarity_score": float(1.0 / (1.0 + distance)),
                        "candidate_question_skeleton": self.question_skeletons[index],
                    }
                )
            return ranked

        similarities = self.similarity.question_similarity_batch(question_skeleton, self.question_skeletons)
        indexed = sorted(enumerate(similarities), key=lambda item: item[1], reverse=True)[:top_n]
        return [
            {
                "example": self.candidates[index],
                "similarity_score": score,
                "candidate_question_skeleton": self.question_skeletons[index],
            }
            for index, score in indexed
        ]

    def retrieve_by_sql_skeleton(self, target_skeleton: str, top_n: int = 3) -> List[Dict[str, Any]]:
        if not self.candidates:
            return []

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
        round_1_max_new_tokens: int = 4096,
        round_2_max_new_tokens: int = 4096,
        gpu_id: Optional[int] = None,
    ) -> None:
        self.base_model_name = base_model
        self.candidate_examples = candidate_examples or []
        self.embedding_model = embedding_model
        self.sql_dialect = sql_dialect
        self.max_seq_length = 2048
        self.round_1_max_new_tokens = round_1_max_new_tokens
        self.round_2_max_new_tokens = round_2_max_new_tokens
        self.gpu_id = gpu_id
        self._model = None
        self._tokenizer = None
        self._outlines_model = None
        self.sql_extractor = SQLSkeletonExtractor(dialect=sql_dialect)
        self.retriever = SolidSQLRetriever(
            candidate_examples=self.candidate_examples,
            embedding_model=embedding_model,
            sql_dialect=sql_dialect,
        )
        if self.candidate_examples:
            self.retriever.build_index()

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if self.gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            device_map: Any = {"": self.gpu_id}
        else:
            device_map = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.generation_config.use_cache = True
        self._model.eval()
        if outlines is not None:
            try:
                self._outlines_model = outlines.from_transformers(self._model, self._tokenizer)
                if BaseModel is not None:
                    print("[Setup] Outlines structured SQL generation enabled.")
                else:
                    print("[Setup] Outlines imported, but Pydantic is unavailable; using raw text fallback.")
            except Exception as exc:
                self._outlines_model = None
                print(f"[Setup] Outlines initialization failed; using raw text fallback. Reason: {exc}")
                print(traceback.format_exc())
        else:
            print("[Setup] Outlines not available; using raw text fallback.")

    def load_retrieval_index(
        self,
        metadata_path: str,
        question_index_path: Optional[str] = None,
        sql_index_path: Optional[str] = None,
    ) -> None:
        self.retriever.load_index(
            metadata_path,
            question_index_path=question_index_path,
            sql_index_path=sql_index_path,
        )
        self.candidate_examples = self.retriever.candidates

    def _generate_text(self, prompt: str, max_new_tokens: int = 4096) -> str:
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
                use_cache=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_length = int(inputs["attention_mask"].sum(dim=1).tolist()[0])
        generated_tokens = outputs[0][prompt_length:]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _generate_sql_response(self, prompt: str, max_new_tokens: int = 4096) -> Dict[str, str]:
        self._load_model()

        if self._outlines_model is not None and BaseModel is not None:
            try:
                response = self._outlines_model(prompt, SqlResponse, max_new_tokens=max_new_tokens)
                if hasattr(response, "model_dump"):
                    response = response.model_dump()
                elif isinstance(response, str):
                    response = json.loads(response)
                elif not isinstance(response, dict):
                    response = dict(response)
                sql_text = clean_sql_output(str(response.get("sql", "")))
                reasoning_text = str(response.get("reasoning", "")).strip()
                return {"sql": sql_text, "reasoning": reasoning_text}
            except Exception as exc:
                print(f"Warning: Outlines structured generation failed; falling back to raw text: {exc}")

        fallback_raw = self._generate_text(prompt, max_new_tokens=max_new_tokens)
        try:
            fallback_response = json.loads(fallback_raw)
            if isinstance(fallback_response, dict):
                sql_text = clean_sql_output(str(fallback_response.get("sql", "")))
                reasoning_text = str(fallback_response.get("reasoning", "")).strip()
                return {"sql": sql_text, "reasoning": reasoning_text}
        except Exception:
            pass
        return {"sql": clean_sql_output(fallback_raw), "reasoning": ""}

    def _clean_skeleton_response(self, response: str) -> str:
        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        elif response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        if "\n" in response:
            response = response.split("\n")[0].strip()
        return response

    def _extract_question_skeleton(self, question: str) -> str:
        skeleton_prompt = SKELETON_EXTRACTION_PROMPT.format(question=question)
        response = self._generate_text(skeleton_prompt, max_new_tokens=256)
        return self._clean_skeleton_response(response)

    def _extract_sql_skeleton(self, sql: str) -> str:
        skeleton_prompt = SQL_SKELETON_EXTRACTION_PROMPT.format(sql=sql)
        response = self._generate_text(skeleton_prompt, max_new_tokens=256)
        return self._clean_skeleton_response(response)

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
            "- The database is SQLite, so use SQLite syntax\n"
            "- Return a JSON object with exactly two keys: sql and reasoning\n\n"
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
            "Return ONLY a JSON object.\n"
            "The sql field must be a single executable SQLite query starting with SELECT or WITH.\n"
            "The reasoning field should briefly explain the choice.\n"
        )

    def _round_2_prompt(
        self,
        question: str,
        schema_json: Dict[str, Any],
        round_1_sql: str,
        round_1_sql_skeleton: str,
        structural_examples: List[Dict[str, Any]],
    ) -> str:
        return (
            "You are a structure-aware text-to-SQL system.\n\n"
            "Your task is to generate the FINAL SQL query using:\n"
            "1. The database schema\n"
            "2. The user question\n"
            "3. The Round 1 draft SQL\n"
            "4. SQL examples selected based on structural similarity (SQL skeleton matching)\n\n"
            "The database is SQLite, so use SQLite syntax.\n"
            "Return a JSON object with exactly two keys: sql and reasoning.\n\n"
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
            f"{round_1_sql_skeleton}\n\n"
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
            "Return ONLY a JSON object.\n"
            "The sql field must be a single executable SQLite query starting with SELECT or WITH.\n"
            "The reasoning field should briefly explain the final choice.\n"
        )

    def validate_sql(
        self,
        sql: str,
    ) -> Dict[str, Any]:
        cleaned_sql = clean_sql_output(sql)
        if not cleaned_sql:
            return {
                "sql": "",
                "valid": False,
                "validation_error": "Empty SQL after cleanup",
            }
        if not cleaned_sql.upper().startswith(("SELECT", "WITH")):
            return {
                "sql": "",
                "valid": False,
                "validation_error": "SQL must start with SELECT or WITH",
            }
        return {
            "sql": cleaned_sql,
            "valid": True,
            "validation_error": None,
        }

    def generate_sql(
        self,
        question: str,
        schema_json: Dict[str, Any],
        round_1_examples: int = 3,
        round_2_examples: int = 3,
    ) -> Dict[str, Any]:
        question_skeleton = self._extract_question_skeleton(question)
        question_retrieval_results = self.retriever.retrieve_by_question(
            question_skeleton,
            top_n=round_1_examples,
        )
        few_shot_examples = [item["example"] for item in question_retrieval_results]
        round_1_prompt = self._round_1_prompt(question, schema_json, few_shot_examples)
        print("[Stage 2.1] Generating Round 1 SQL...")
        round_1_response = self._generate_sql_response(
            round_1_prompt,
            max_new_tokens=self.round_1_max_new_tokens,
        )
        round_1_sql = round_1_response["sql"]
        round_1_reasoning = round_1_response["reasoning"]
        print("[Stage 2.2] Validating Round 1 SQL...")
        round_1_validation = self.validate_sql(round_1_sql)
        validated_round_1_sql = round_1_validation["sql"]

        print("[Stage 2.2b] Generating SQL skeleton for Round 1 draft...")
        round_1_sql_skeleton = self._extract_sql_skeleton(validated_round_1_sql or round_1_sql)

        print("[Stage 2.3] Retrieving structurally similar SQL examples...")
        structural_examples = self.retriever.retrieve_by_sql_skeleton(
            round_1_sql_skeleton,
            top_n=round_2_examples,
        )
        round_2_prompt = self._round_2_prompt(
            question,
            schema_json,
            validated_round_1_sql or round_1_sql,
            round_1_sql_skeleton,
            structural_examples,
        )
        print("[Stage 2.4] Generating Round 2 refined SQL...")
        round_2_response = self._generate_sql_response(
            round_2_prompt,
            max_new_tokens=self.round_2_max_new_tokens,
        )
        round_2_sql = round_2_response["sql"]
        round_2_reasoning = round_2_response["reasoning"]
        print("[Stage 2.5] Validating final SQL...")
        round_2_validation = self.validate_sql(round_2_sql)

        final_sql = round_2_validation["sql"] or validated_round_1_sql or round_1_sql
        final_validation = self.validate_sql(final_sql)
        final_sql = final_validation["sql"] or final_sql
        final_reasoning = round_2_reasoning or round_1_reasoning

        return {
            "question": question,
            "schema_json": schema_json,
            "question_skeleton": question_skeleton,
            "question_retrieval_results": question_retrieval_results,
            "round_1_sql": round_1_sql,
            "round_1_reasoning": round_1_reasoning,
            "round_1_sql_skeleton": round_1_sql_skeleton,
            "round_1_validation": round_1_validation,
            "structural_examples": structural_examples,
            "round_2_sql": round_2_sql,
            "round_2_reasoning": round_2_reasoning,
            "round_2_validation": round_2_validation,
            "final_sql": final_sql,
            "final_reasoning": final_reasoning,
            "final_valid": final_validation["valid"],
            "final_validation_error": final_validation["validation_error"],
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
        logs_dir: Optional[str] = None,
        schema_linking_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        question_logs: List[Dict[str, Any]] = []
        db_stats = defaultdict(
            lambda: {
                "total": 0,
                "correct": 0,
                "valid": 0,
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
            difficulty = item.get("difficulty", "unknown")
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
                    db_stats[db_id]["total"] += 1
                    db_stats[db_id]["errors"] += 1
                    if logs_dir:
                        question_logs.append(
                            build_question_log_record(
                                question_id=question_id,
                                question=question,
                                db_id=db_id,
                                difficulty=difficulty,
                                gold_sql=ground_truth_sql,
                                round_1_sql="",
                                round_2_sql="",
                                winner_sql="",
                                is_correct=False,
                                execution_times=[],
                                generated_results=[],
                                gold_results=[],
                                round_1_valid=False,
                                round_2_valid=False,
                                final_valid=False,
                                execution_error=f"Missing database file for {db_id}",
                            )
                        )
                        write_question_logs_array(logs_dir, question_logs)
                    continue

            print("[Stage 1: Schema Load] Loading schema text...")
            if db_id not in db_cache:
                schema_text = load_schema_for_db(db_path)
                if not schema_text:
                    print(f"[Stage 1: Schema Load] Failed for {db_id}")
                    db_stats[db_id]["total"] += 1
                    db_stats[db_id]["errors"] += 1
                    if logs_dir:
                        question_logs.append(
                            build_question_log_record(
                                question_id=question_id,
                                question=question,
                                db_id=db_id,
                                difficulty=difficulty,
                                gold_sql=ground_truth_sql,
                                round_1_sql="",
                                round_2_sql="",
                                winner_sql="",
                                is_correct=False,
                                execution_times=[],
                                generated_results=[],
                                gold_results=[],
                                round_1_valid=False,
                                round_2_valid=False,
                                final_valid=False,
                                execution_error=f"Schema load failed for {db_id}",
                            )
                        )
                        write_question_logs_array(logs_dir, question_logs)
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
            current_round_1_sql = ""
            current_round_1_reasoning = ""
            current_round_2_sql = ""
            current_round_2_reasoning = ""
            current_round_1_valid = False
            current_round_2_valid = False
            current_final_valid = False

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

                current_round_1_sql = generation["round_1_sql"]
                current_round_1_reasoning = generation["round_1_reasoning"]
                current_round_2_sql = generation["round_2_sql"]
                current_round_2_reasoning = generation["round_2_reasoning"]
                current_round_1_valid = generation["round_1_validation"]["valid"]
                current_round_2_valid = generation["round_2_validation"]["valid"]
                current_final_valid = generation["final_valid"]

                final_sql = generation["final_sql"]
                execution_time = time.time() - start_time
                db_stats[db_id]["total_time"] += execution_time

                if generation["final_valid"]:
                    db_stats[db_id]["valid"] += 1

                print("[Stage 3: Execution] Running generated SQL against SQLite...")
                generated_results, generated_exec_time, generated_exec_error = execute_sql_with_metadata(db_path, final_sql)
                print(f"[Stage 3: Execution] Generated SQL rows: {len(generated_results)}")
                print("[Stage 3: Execution] Running ground-truth SQL against SQLite...")
                ground_truth_results, _, _ = execute_sql_with_metadata(db_path, ground_truth_sql)
                print(f"[Stage 3: Execution] Ground-truth rows: {len(ground_truth_results)}")

                db_stats[db_id]["executed"] += 1

                execution_match = compare_execution_results(generated_results, ground_truth_results)

                print("[Stage 4: Comparison] Comparing execution results...")
                if execution_match:
                    db_stats[db_id]["correct"] += 1
                    print("[Stage 4: Comparison] Execution result: correct")
                else:
                    print("[Stage 4: Comparison] Execution result: incorrect")

                results.append(
                    {
                        "question_id": question_id,
                        "db_id": db_id,
                        "question": question,
                        "difficulty": difficulty,
                        "generated_sql": final_sql,
                        "ground_truth_sql": ground_truth_sql,
                        "schema_linking_used": linked_schema_record is not None,
                        "round_1_sql": generation["round_1_sql"],
                        "round_1_reasoning": generation["round_1_reasoning"],
                        "round_2_sql": generation["round_2_sql"],
                        "round_2_reasoning": generation["round_2_reasoning"],
                        "execution_match": execution_match,
                        "valid": generation["final_valid"],
                        "validation_error": generation["final_validation_error"],
                        "execution_time": execution_time,
                        "retrieved_examples": len(generation["structural_examples"]),
                    }
                )
                if logs_dir:
                    log_record = build_question_log_record(
                        question_id=question_id,
                        question=question,
                        db_id=db_id,
                        difficulty=difficulty,
                        gold_sql=ground_truth_sql,
                        round_1_sql=current_round_1_sql,
                        round_1_reasoning=current_round_1_reasoning,
                        round_2_sql=current_round_2_sql,
                        round_2_reasoning=current_round_2_reasoning,
                        winner_sql=final_sql,
                        winner_reasoning=generation["final_reasoning"],
                        is_correct=execution_match,
                        execution_times=[generated_exec_time],
                        generated_results=generated_results,
                        gold_results=ground_truth_results,
                        round_1_valid=current_round_1_valid,
                        round_2_valid=current_round_2_valid,
                        final_valid=current_final_valid,
                        execution_error=generated_exec_error,
                    )
                    question_logs.append(log_record)
                    write_question_logs_array(logs_dir, question_logs)
                print(f"[Stage 5: Outcome] Final status: {'correct' if execution_match else 'incorrect'}")
                print(f"[Stage 5: Outcome] Passed validation: {generation['final_valid']}")
                print(f"[Stage 5: Outcome] Execution time: {execution_time:.4f}s")

            except Exception as exc:
                execution_time = time.time() - start_time
                db_stats[db_id]["total_time"] += execution_time
                db_stats[db_id]["errors"] += 1
                traceback_text = traceback.format_exc()
                results.append(
                    {
                        "question_id": question_id,
                        "db_id": db_id,
                        "question": question,
                        "difficulty": difficulty,
                        "generated_sql": "",
                        "ground_truth_sql": ground_truth_sql,
                        "error": str(exc),
                        "traceback": traceback_text,
                        "execution_time": execution_time,
                        "execution_match": False,
                        "valid": False,
                    }
                )
                if logs_dir:
                    log_record = build_question_log_record(
                        question_id=question_id,
                        question=question,
                        db_id=db_id,
                        difficulty=difficulty,
                        gold_sql=ground_truth_sql,
                        round_1_sql=current_round_1_sql,
                        round_1_reasoning=current_round_1_reasoning,
                        round_2_sql=current_round_2_sql,
                        round_2_reasoning=current_round_2_reasoning,
                        winner_sql="",
                        winner_reasoning="",
                        is_correct=False,
                        execution_times=[execution_time],
                        generated_results=[],
                        gold_results=[],
                        round_1_valid=current_round_1_valid,
                        round_2_valid=current_round_2_valid,
                        final_valid=False,
                        execution_error=str(exc),
                    )
                    question_logs.append(log_record)
                    write_question_logs_array(logs_dir, question_logs)
                print(f"[Stage 5: Outcome] Error processing question {question_id}: {exc}")
                print(traceback_text)
                print(f"[Stage 5: Outcome] Execution time before failure: {execution_time:.4f}s")

        overall_stats = {
            "total_questions": len(questions_data),
            "processed_questions": sum(stats["total"] for stats in db_stats.values()),
            "correct_answers": sum(stats["correct"] for stats in db_stats.values()),
            "executed_queries": sum(stats["executed"] for stats in db_stats.values()),
            "valid_queries": sum(stats["valid"] for stats in db_stats.values()),
            "errors": sum(stats["errors"] for stats in db_stats.values()),
            "average_execution_time": (
                sum(stats["total_time"] for stats in db_stats.values())
                / max(sum(stats["total"] for stats in db_stats.values()), 1)
            ),
        }
        overall_stats["execution_accuracy"] = (
            overall_stats["correct_answers"] / max(overall_stats["executed_queries"], 1)
        )
        overall_stats["validation_success_rate"] = (
            overall_stats["valid_queries"] / max(overall_stats["processed_questions"], 1)
        )

        for db_id, stats in db_stats.items():
            stats["accuracy"] = stats["correct"] / max(stats["executed"], 1)
            stats["validation_success_rate"] = stats["valid"] / max(stats["total"], 1)
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
    print(f"Valid queries: {overall['valid_queries']}")
    print(f"Errors: {overall['errors']}")
    print(f"Execution accuracy: {overall['execution_accuracy']:.4f}")
    print(f"Validation success rate: {overall['validation_success_rate']:.4f}")
    print(f"Average execution time: {overall['average_execution_time']:.4f}s")

    print("\nPer-database statistics:")
    for db_id, db_stats in stats["per_database_statistics"].items():
        print(f"  {db_id}:")
        print(f"    Questions: {db_stats['total']}")
        print(f"    Execution accuracy: {db_stats['accuracy']:.4f}")
        print(f"    Validation success rate: {db_stats['validation_success_rate']:.4f}")
        print(f"    Average time: {db_stats['average_execution_time']:.4f}s")


def split_questions_into_shards(
    questions_data: List[Dict[str, Any]],
    num_shards: int,
) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    for index, item in enumerate(questions_data):
        shards[index % num_shards].append(item)
    return [shard for shard in shards if shard]


def merge_shard_outputs(
    shard_outputs: List[Dict[str, Any]],
    original_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    merged_results: List[Dict[str, Any]] = []
    merged_db_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "valid": 0,
            "executed": 0,
            "errors": 0,
            "total_time": 0.0,
        }
    )

    for shard_output in shard_outputs:
        merged_results.extend(shard_output.get("detailed_results", []))
        for db_id, stats in shard_output.get("per_database_statistics", {}).items():
            target = merged_db_stats[db_id]
            target["total"] += stats.get("total", 0)
            target["correct"] += stats.get("correct", 0)
            target["valid"] += stats.get("valid", 0)
            target["executed"] += stats.get("executed", 0)
            target["errors"] += stats.get("errors", 0)
            target["total_time"] += stats.get("total_time", 0.0)

    order = {str(item.get("question_id", index)): index for index, item in enumerate(original_questions)}
    merged_results.sort(key=lambda item: order.get(str(item.get("question_id")), 10**9))

    overall_stats = {
        "total_questions": len(original_questions),
        "processed_questions": sum(stats["total"] for stats in merged_db_stats.values()),
        "correct_answers": sum(stats["correct"] for stats in merged_db_stats.values()),
        "executed_queries": sum(stats["executed"] for stats in merged_db_stats.values()),
        "valid_queries": sum(stats["valid"] for stats in merged_db_stats.values()),
        "errors": sum(stats["errors"] for stats in merged_db_stats.values()),
        "average_execution_time": (
            sum(stats["total_time"] for stats in merged_db_stats.values())
            / max(sum(stats["total"] for stats in merged_db_stats.values()), 1)
        ),
    }
    overall_stats["execution_accuracy"] = (
        overall_stats["correct_answers"] / max(overall_stats["executed_queries"], 1)
    )
    overall_stats["validation_success_rate"] = (
        overall_stats["valid_queries"] / max(overall_stats["processed_questions"], 1)
    )

    for db_id, stats in merged_db_stats.items():
        stats["accuracy"] = stats["correct"] / max(stats["executed"], 1)
        stats["validation_success_rate"] = stats["valid"] / max(stats["total"], 1)
        stats["average_execution_time"] = stats["total_time"] / max(stats["total"], 1)

    return {
        "overall_statistics": overall_stats,
        "per_database_statistics": dict(merged_db_stats),
        "detailed_results": merged_results,
    }


def write_combined_question_logs(logs_dir: str, original_questions: List[Dict[str, Any]]) -> None:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    order = {str(item.get("question_id", index)): index for index, item in enumerate(original_questions)}
    records = []
    for gpu_dir in sorted(path for path in logs_path.iterdir() if path.is_dir() and path.name.startswith("gpu_")):
        log_file = gpu_dir / "all_outputs.json"
        if log_file.exists():
            records.extend(json.loads(log_file.read_text(encoding="utf-8")))
    records.sort(key=lambda item: order.get(str(item.get("question_id")), 10**9))
    (logs_path / "all_outputs.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_evaluation_worker(
    worker_index: int,
    gpu_id: int,
    questions_shard: List[Dict[str, Any]],
    worker_output_path: str,
    logs_dir: str,
    base_model: str,
    databases: str,
    metadata_index: Optional[str],
    question_index: Optional[str],
    sql_index: Optional[str],
    schema_linking_results: Optional[str],
    round_1_max_new_tokens: int,
    round_2_max_new_tokens: int,
) -> None:
    print(f"[Worker {worker_index}] Starting on GPU {gpu_id} with {len(questions_shard)} questions...")

    pipeline = BaseModelSQLPipeline(
        base_model=base_model,
        round_1_max_new_tokens=round_1_max_new_tokens,
        round_2_max_new_tokens=round_2_max_new_tokens,
        gpu_id=gpu_id,
    )
    if metadata_index and os.path.exists(metadata_index):
        pipeline.load_retrieval_index(
            metadata_index,
            question_index_path=question_index,
            sql_index_path=sql_index,
        )

    schema_linking_lookup = None
    if schema_linking_results and os.path.exists(schema_linking_results):
        schema_linking_lookup = load_schema_linking_results(schema_linking_results)

    evaluator = SQLEvaluator(pipeline)
    stats = evaluator.evaluate_questions(
        questions_shard,
        databases,
        worker_output_path,
        logs_dir=logs_dir,
        schema_linking_lookup=schema_linking_lookup,
    )
    pipeline.shutdown()

    Path(worker_output_path).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Worker {worker_index}] Finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Base-model SQL pipeline and evaluation")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--databases", required=True, help="Path to databases directory")
    parser.add_argument("--metadata-index", help="Path to retrieval metadata JSON")
    parser.add_argument("--question-index", help="Path to question FAISS index")
    parser.add_argument("--sql-index", help="Path to SQL FAISS index")
    parser.add_argument("--schema-linking-results", help="Path to schema linking batch output JSON")
    parser.add_argument(
        "--base-model",
        default="openai/gpt-oss-20b",
        help="Base model name or local path",
    )
    parser.add_argument("--round-1-max-new-tokens", type=int, default=4096, help="Max new tokens for Round 1 generation")
    parser.add_argument("--round-2-max-new-tokens", type=int, default=4096, help="Max new tokens for Round 2 generation")
    parser.add_argument("--num-workers", type=int, help="Number of parallel evaluation workers; defaults to the number of GPU ids provided")
    parser.add_argument("--gpu-ids", default="0,1,2,3", help="Comma-separated GPU ids to use, one per worker")
    parser.add_argument(
        "--output",
        default="sql_pipeline_evaluation_results.json",
        help="Path to output results file",
    )
    parser.add_argument(
        "--question-logs-dir",
        help="Root directory for GPU-sharded log folders and merged all_outputs.json",
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

    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("Provide at least one GPU id via --gpu-ids")

    num_workers = args.num_workers or len(gpu_ids)
    if num_workers > len(gpu_ids):
        raise ValueError("--num-workers cannot exceed the number of GPU ids provided")

    selected_gpu_ids = gpu_ids[:num_workers]
    print("[Setup] Initializing sharded base-model SQL evaluation...")
    if args.metadata_index and os.path.exists(args.metadata_index):
        print(f"[Setup] Loading structural retrieval data from {args.metadata_index}...")
    if args.schema_linking_results and os.path.exists(args.schema_linking_results):
        print(f"[Setup] Loading schema-linking results from {args.schema_linking_results}...")
    print(f"[Setup] Using {num_workers} workers on GPU ids: {selected_gpu_ids}")
    print(f"[Setup] Writing GPU-sharded logs to {question_logs_dir}...")

    shards = split_questions_into_shards(questions_data, num_workers)
    temp_dir = Path(tempfile.mkdtemp(prefix="sql_eval_shards_"))
    worker_outputs = [temp_dir / f"worker_{index}.json" for index in range(len(shards))]

    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    for worker_index, (gpu_id, shard, worker_output) in enumerate(zip(selected_gpu_ids, shards, worker_outputs)):
        gpu_logs_dir = Path(question_logs_dir) / f"gpu_{gpu_id}"
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
    stats = merge_shard_outputs(shard_outputs, questions_data)
    Path(args.output).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_combined_question_logs(question_logs_dir, questions_data)
    print_summary(stats)
    print(f"\n[Run] Detailed results saved to: {args.output}")
    print(f"[Run] GPU-sharded logs saved to: {question_logs_dir}")


if __name__ == "__main__":
    main()

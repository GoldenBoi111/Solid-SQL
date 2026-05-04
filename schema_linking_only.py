#!/usr/bin/env python3
"""
Standalone schema-linking module powered by the LoRA adapter.

This file is intentionally independent from SQL generation, retrieval, and
evaluation logic. It only performs schema grounding for a question/schema pair.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema_linking.config import INSTRUCTION_TEMPLATE


SCHEMA_LINKING_PROMPT = INSTRUCTION_TEMPLATE


def normalize_spider_record(item: Dict[str, object], fallback_index: int) -> Dict[str, object]:
    return {
        "question_id": item.get("question_id", fallback_index),
        "question": item.get("question", ""),
        "evidence": item.get("evidence", ""),
        "db_id": item.get("db_id", ""),
        "difficulty": item.get("difficulty", "unknown"),
        "gold_sql": item.get("SQL", item.get("sql", item.get("gold_sql", ""))),
    }


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
        raise RuntimeError(f"Could not load schema for {db_path}: {exc}") from exc


def _clean_generated_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_section_lines(text: str, heading: str) -> List[str]:
    lines = text.splitlines()
    collected: List[str] = []
    in_section = False
    target = heading.lower()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":"):
            current_heading = line[:-1].strip().lower()
            if current_heading == target:
                collected = []
                in_section = True
                continue
            if in_section:
                break

        if in_section:
            collected.append(line)

    return collected


def parse_schema_linking_response(text: str) -> Dict[str, object]:
    cleaned = _clean_generated_text(text)

    tables: List[str] = []
    for line in _extract_section_lines(cleaned, "Relevant Tables"):
        item = line[1:].strip() if line.startswith("-") else line
        if item:
            tables.append(item)

    columns: Dict[str, List[str]] = {}
    for line in _extract_section_lines(cleaned, "Relevant Columns"):
        item = line[1:].strip() if line.startswith("-") else line
        if not item or ":" not in item:
            continue
        table_name, raw_columns = item.split(":", 1)
        parsed_columns = [
            column.strip()
            for column in re.split(r",\s*", raw_columns.strip())
            if column.strip()
        ]
        if table_name.strip() and parsed_columns:
            columns[table_name.strip()] = parsed_columns

    joins: List[str] = []
    for line in _extract_section_lines(cleaned, "Join Relationships"):
        item = line[1:].strip() if line.startswith("-") else line
        if item:
            joins.append(item)

    filters: List[str] = []
    for line in _extract_section_lines(cleaned, "Filters / Constraints"):
        item = line[1:].strip() if line.startswith("-") else line
        if item:
            filters.append(item)

    intent = ""
    for line in _extract_section_lines(cleaned, "Question Intent"):
        item = line[1:].strip() if line.startswith("-") else line
        if item:
            intent = item
            break

    return {
        "tables": tables,
        "columns": columns,
        "joins": joins,
        "filters": filters,
        "intent": intent,
        "raw_output": cleaned,
    }


class LoRASchemaLinker:
    """Standalone LoRA-backed schema linker."""

    def __init__(
        self,
        base_model: str = "openai/gpt-oss-20b",
        adapter_path: str = "./schema_linking_output/lora_adapter",
        max_seq_length: int = 2048,
    ) -> None:
        self.base_model_name = base_model
        self.adapter_path = Path(adapter_path)
        self.max_seq_length = max_seq_length
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if not self.adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {self.adapter_path}")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.load_adapter(str(self.adapter_path), adapter_name="schema_linking")
        model.set_adapter("schema_linking")
        model.enable_adapters()
        model.eval()

        self._model = model
        self._tokenizer = tokenizer

    def _build_prompt(self, question: str, schema_text: str, evidence: str = "") -> str:
        prompt = SCHEMA_LINKING_PROMPT.format(question=question, schema_text=schema_text)
        if evidence:
            prompt += "\n\n## BENCHMARK EVIDENCE\n" + evidence
        return prompt

    def predict(
        self,
        question: str,
        schema_text: str,
        evidence: str = "",
        max_new_tokens: int = 768,
    ) -> Dict[str, object]:
        self._load_model()

        prompt = self._build_prompt(question, schema_text, evidence=evidence)
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
        generated_text = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return parse_schema_linking_response(generated_text)

    def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None


def run_batch_schema_linking(
    linker: LoRASchemaLinker,
    questions_path: str,
    databases_dir: str,
) -> Dict[str, object]:
    questions_data = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    results: List[Dict[str, object]] = []
    by_question_id: Dict[str, Dict[str, object]] = {}
    schema_cache: Dict[str, str] = {}

    for index, item in enumerate(questions_data):
        normalized = normalize_spider_record(item, index)
        question_id = normalized["question_id"]
        question = normalized["question"]
        evidence = normalized["evidence"]
        db_id = normalized["db_id"]
        difficulty = normalized["difficulty"]
        gold_sql = normalized["gold_sql"]

        db_path = Path(databases_dir) / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            alt_path = Path(databases_dir) / f"{db_id}.sqlite"
            if not alt_path.exists():
                raise FileNotFoundError(f"Database file not found for {db_id}")
            db_path = alt_path

        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema_for_db(str(db_path))
        schema_text = schema_cache[db_id]

        linked_schema = linker.predict(question=question, schema_text=schema_text, evidence=evidence)
        record = {
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "difficulty": difficulty,
            "gold_sql": gold_sql,
            "schema_text": schema_text,
            "schema_linking": linked_schema,
        }
        results.append(record)
        by_question_id[str(question_id)] = record

    return {
        "questions_file": questions_path,
        "databases_dir": databases_dir,
        "count": len(results),
        "results": results,
        "by_question_id": by_question_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone LoRA schema linker")
    parser.add_argument("--question", help="Natural language question")
    parser.add_argument("--evidence", default="", help="Optional benchmark evidence for single-question mode")
    parser.add_argument("--schema-file", help="Path to a text file containing schema text")
    parser.add_argument("--schema-text", help="Schema text provided directly")
    parser.add_argument("--questions", help="Path to question dataset JSON")
    parser.add_argument("--databases", help="Path to databases directory")
    parser.add_argument(
        "--adapter",
        default="./schema_linking_output/lora_adapter",
        help="Path to the LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        default="openai/gpt-oss-20b",
        help="Base model name or local path",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for the JSON result",
    )

    args = parser.parse_args()

    linker = LoRASchemaLinker(
        base_model=args.base_model,
        adapter_path=args.adapter,
    )

    if args.questions and args.databases:
        result = run_batch_schema_linking(
            linker=linker,
            questions_path=args.questions,
            databases_dir=args.databases,
        )
    else:
        schema_text = args.schema_text
        if not schema_text and args.schema_file:
            schema_text = Path(args.schema_file).read_text(encoding="utf-8")
        if not schema_text or not args.question:
            raise ValueError(
                "Single mode requires --question and either --schema-text or --schema-file; "
                "batch mode requires --questions and --databases"
            )
        if args.evidence:
            result = linker.predict(args.question, schema_text, evidence=args.evidence)
        else:
            result = linker.predict(args.question, schema_text)

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    linker.shutdown()


if __name__ == "__main__":
    main()

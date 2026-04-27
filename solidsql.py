"""
SolidSQL - Full End-to-End System

This module implements the complete SolidSQL system with:
1. Schema linking (original task)
2. Question skeleton extraction
3. SQL skeleton extraction
4. Skeleton-based example retrieval (Section 3.4.1 and 3.4.2)
5. Round-2 refinement pipeline

Usage:
    from solidsql import SolidSQL

    # Initialize with candidate examples
    solidsql = SolidSQL(
        candidate_examples=[
            {"question": "How many singers are older than 20?", "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20"},
            {"question": "What is the average salary?", "sql": "SELECT AVG(salary) FROM Employees"},
        ]
    )

    # Full end-to-end pipeline
    result = solidsql.generate_sql(
        question="How many actors are younger than 30?",
        schema_text="Actor(id, name, age)\nMovie(id, actor_id, title)",
        top_n=3
    )
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import all required components
from schema_linking.inference import SchemaLinker
from schema_linking.question_skeleton_extractor import QuestionSkeletonExtractor
from schema_linking.sql_skeleton_extractor import SQLSkeletonExtractor
from schema_linking.skeleton_retriever import SkeletonRetriever
from schema_linking.skeleton_similarity import SkeletonSimilarity


class SolidSQL:
    """
    Complete SolidSQL system with schema linking, skeleton-based retrieval,
    and round-2 refinement.
    """

    def __init__(
        self,
        candidate_examples: Optional[List[Dict[str, str]]] = None,
        base_model: str = "openai/gpt-oss-20b",
        adapter_path: str = "./schema_linking_output/lora_adapter",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sql_dialect: str = "sqlite",
        build_index: bool = True,
        skip_skeleton_extraction: bool = False,
    ):
        """
        Initialize the complete SolidSQL system.

        Args:
            candidate_examples: List of example questions and SQL statements
            base_model: Hugging Face model name for schema linking
            adapter_path: Path to trained LoRA adapter
            embedding_model: Sentence transformer for question embeddings
            sql_dialect: SQL dialect for parsing
            build_index: Whether to build skeleton index immediately
            skip_skeleton_extraction: Whether to skip question skeleton extraction (for evaluation with pre-built indices)
        """
        # Store configuration
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.embedding_model = embedding_model
        self.sql_dialect = sql_dialect
        
        # Initialize skeleton components
        self.skip_skeleton_extraction = skip_skeleton_extraction
        
        # Create SchemaLinker first (this will load the model)
        self.schema_linker = SchemaLinker(
            base_model=base_model,
            adapter_path=adapter_path,
        )
        self.schema_linker._load_model()  # eager load so shared_model is not None

        # Initialize question extractor with shared model
        if not skip_skeleton_extraction:
            self.q_extractor = QuestionSkeletonExtractor(
                model_name=base_model,
                shared_model=self.schema_linker._model,
                shared_tokenizer=self.schema_linker._tokenizer,
            )
        else:
            self.q_extractor = None
            
        self.sql_extractor = SQLSkeletonExtractor(dialect=sql_dialect)
        self.similarity = SkeletonSimilarity(embedding_model=embedding_model)

        # Initialize retrieval system
        self.retriever = SkeletonRetriever(
            candidate_examples=candidate_examples or [],
            embedding_model=embedding_model,
            sql_dialect=sql_dialect,
            shared_model=self.schema_linker._model,
            shared_tokenizer=self.schema_linker._tokenizer,
        )

        # Build index if provided
        if candidate_examples and build_index:
            self.retriever.build_index(show_progress=False)

    def generate_sql(
        self,
        question: str,
        schema_text: str,
        top_n: int = 5,
        round_2_refinement: bool = True,
        max_new_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Generate SQL using the complete SolidSQL pipeline.

        Args:
            question: Natural language question
            schema_text: Database schema text
            top_n: Number of similar examples to retrieve
            round_2_refinement: Whether to use round-2 refinement
            max_new_tokens: Maximum tokens for generation

        Returns:
            Dictionary with full generation results including intermediate steps
        """
        # Step 1: Schema linking with LoRA adapter
        schema_result = self.schema_linker.predict(
            question=question,
            schema_text=schema_text,
            max_new_tokens=max_new_tokens,
        )
        generated_schema_text = self._format_generated_schema(schema_result)
        
        # Generate SQL in round 1 using the base model without LoRA adapter
        round_1_sql = self._generate_sql_with_base_model(
            question=question,
            schema_text=generated_schema_text,
            examples_text=self._format_few_shot_examples(self.retriever.candidates),
            max_new_tokens=max_new_tokens,
        )

        # Step 2: Extract question skeleton for retrieval (if not skipped)
        question_skeleton = None
        if not self.skip_skeleton_extraction and self.q_extractor is not None:
            question_skeleton = self.q_extractor.extract(question)
        elif self.skip_skeleton_extraction:
            # Use the base model LLM for question skeleton extraction
            question_skeleton = self._extract_skeleton_with_base_model(question)

        # Step 3: Retrieve similar examples
        if self.retriever.question_skeletons:  # Only if index built
            retrieval_results = self.retriever.retrieve_by_question(
                question=question,
                top_n=top_n,
                show_progress=False,
            )
        else:
            retrieval_results = []

        # Step 4: Round 2 - Use round 1 SQL for retrieval, then refine with retrieved examples
        refined_sql = None

        if round_2_refinement and retrieval_results:
            structural_results = self.retriever.retrieve_by_sql(
                round_1_sql,
                top_n=top_n,
                show_progress=False,
            )
            structural_examples = self._format_structural_examples(structural_results[:3])
            draft_sql_skeleton = self.sql_extractor.extract(round_1_sql)
            
            # Create prompt for round 2 SQL generation
            round_2_prompt = (
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
                f"{generated_schema_text}\n\n"
                "---\n\n"
                "## USER QUESTION\n"
                f"{question}\n\n"
                "---\n\n"
                "## ROUND 1 DRAFT SQL\n"
                f"{round_1_sql}\n\n"
                "---\n\n"
                "## SQL SKELETON OF ROUND 1 SQL\n"
                f"{draft_sql_skeleton}\n\n"
                "---\n\n"
                "## STRUCTURALLY SIMILAR SQL EXAMPLES\n"
                "These examples were retrieved using SQL skeleton edit distance (not text similarity):\n\n"
                f"{structural_examples}\n\n"
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
            
            # Generate SQL using the base model without LoRA adapter
            try:
                outputs = self.schema_linker.generate_without_lora(
                    [round_2_prompt],
                    max_new_tokens=max_new_tokens,
                    show_progress=False,
                )
                
                if outputs:
                    refined_sql = outputs[0].strip()
                    # Clean up the SQL output
                    refined_sql = self._clean_sql_output(refined_sql)
                    
            except Exception as e:
                print(f"Warning: Failed to generate refined SQL: {e}")
                # Fallback to round 1 SQL if round 2 fails
                refined_sql = round_1_sql

        # Step 5: Return complete results
        result = {
            "question": question,
            "schema": schema_text,
            "generated_schema": generated_schema_text,
            "schema_linking_result": schema_result,
            "question_skeleton": question_skeleton,
            "retrieval_results": retrieval_results,
            "round_1_sql": round_1_sql,
            "refined_sql": refined_sql,
            "full_generation": {
                "schema_linking": schema_result,
                "retrieval": retrieval_results,
                "round_1": round_1_sql,
                "refinement": refined_sql,
            },
        }

        return result

    def generate_sql_with_context(
        self,
        question: str,
        schema_text: str,
        context_examples: List[Dict[str, str]],
        top_n: int = 5,
        max_new_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Generate SQL using provided context examples for retrieval.

        Args:
            question: Natural language question
            schema_text: Database schema text
            context_examples: List of example questions and SQL statements
            top_n: Number of similar examples to retrieve
            max_new_tokens: Maximum tokens for generation

        Returns:
            Dictionary with full generation results
        """
        # Rebuild index with new context
        self.retriever = SkeletonRetriever(
            candidate_examples=context_examples,
            embedding_model=self.embedding_model,
            sql_dialect=self.sql_dialect,
            shared_model=self.schema_linker._model,
            shared_tokenizer=self.schema_linker._tokenizer,
        )
        self.retriever.build_index(show_progress=False)

        # Generate using the full pipeline
        return self.generate_sql(
            question=question,
            schema_text=schema_text,
            top_n=top_n,
            round_2_refinement=True,
            max_new_tokens=max_new_tokens,
        )

    def add_candidate_examples(self, examples: List[Dict[str, str]]) -> None:
        """
        Add new candidate examples to the retrieval system.

        Args:
            examples: List of example dictionaries with "question" and "sql" keys
        """
        # Extend existing candidates
        self.retriever.candidates.extend(examples)

        # Rebuild index with new examples
        self.retriever.build_index(show_progress=False)

    def build_retrieval_index(self, show_progress: bool = True) -> None:
        """
        Build the skeleton index for retrieval.

        Args:
            show_progress: Whether to show progress messages
        """
        self.retriever.build_index(show_progress=show_progress)

    def save_retrieval_index(self, path: str) -> None:
        """
        Save the built retrieval index to disk.

        Args:
            path: Path to save the index
        """
        self.retriever.save_index(path)

    def load_retrieval_index(self, path: str, question_index_path: str = None, sql_index_path: str = None) -> None:
        """
        Load a previously saved retrieval index.

        Args:
            path: Path to the saved index or metadata JSON file
            question_index_path: Optional path to question FAISS index
            sql_index_path: Optional path to SQL FAISS index
        """
        self.retriever.load_index(path, question_index_path=question_index_path, sql_index_path=sql_index_path)
            
    def _extract_skeleton_with_base_model(self, question: str) -> str:
        """Extract question skeleton using the base model LLM."""
        from schema_linking.question_skeleton_extractor import SKELETON_EXTRACTION_PROMPT
        
        skeleton_prompt = SKELETON_EXTRACTION_PROMPT.format(question=question)
        
        # Use the base model LLM without LoRA adapter
        try:
            outputs = self.schema_linker.generate_without_lora(
                [skeleton_prompt],
                max_new_tokens=256,
                show_progress=False,
            )
            
            if outputs:
                skeleton = outputs[0].strip()
                # Clean up the response
                skeleton = self._clean_skeleton_response(skeleton)
                return skeleton
        except Exception as e:
            print(f"Warning: Failed to extract skeleton with base model: {e}")
        
        # Fallback to regular extractor if base model fails
        if self.q_extractor is not None:
            return self.q_extractor.extract(question)
        return ""
    
    def _clean_skeleton_response(self, response: str) -> str:
        """Clean up the LLM response for skeleton extraction."""
        response = response.strip()
        
        # Remove quotes if present
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        elif response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        
        # Remove any trailing explanations after the skeleton
        if "\n" in response:
            response = response.split("\n")[0].strip()
        
        return response
            
    def _generate_sql_with_base_model(
        self,
        question: str,
        schema_text: str,
        examples_text: str = "",
        max_new_tokens: int = 4096,
    ) -> str:
        """
        Generate SQL using the base model without LoRA adapter.
        
        Args:
            question: Natural language question
            schema_text: Database schema text
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated SQL string
        """
        # SQL generation prompt
        sql_prompt = (
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
            "{schema}\n\n"
            "---\n\n"
            "### USER QUESTION\n"
            "{question}\n\n"
            "---\n\n"
            "### FEW-SHOT EXAMPLES\n"
            "{examples}\n\n"
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
        ).format(question=question, schema=schema_text, examples=examples_text or "(none provided)")
        
        # Use the base model's generate_without_lora method
        try:
            outputs = self.schema_linker.generate_without_lora(
                [sql_prompt],
                max_new_tokens=max_new_tokens,
                show_progress=False,
            )
            
            if outputs:
                sql = outputs[0].strip()
                sql = self._clean_sql_output(sql)
            else:
                sql = ""
        except Exception as e:
            print(f"Warning: Failed to generate SQL with base model: {e}")
            sql = ""
        
        return sql

    def _format_generated_schema(self, schema_result: Dict[str, Any]) -> str:
        """Format the LoRA schema-linking output into a compact schema summary."""
        if not schema_result:
            return "(no schema predicted)"

        tables = schema_result.get("tables", [])
        columns = schema_result.get("columns", [])
        join_relationships = schema_result.get("join_relationships", [])
        filters_constraints = schema_result.get("filters_constraints", [])
        question_intent = schema_result.get("question_intent", "")
        raw_output = schema_result.get("raw_output", "")

        if raw_output:
            return raw_output

        lines = []
        if tables:
            lines.append("Relevant Tables:")
            for table in tables:
                if isinstance(table, dict):
                    table_name = table.get("name", "")
                else:
                    table_name = str(table)
                if table_name:
                    lines.append(f"- {table_name}")

        if columns:
            if lines:
                lines.append("")
            lines.append("Relevant Columns:")
            if isinstance(columns, dict):
                for table_name, table_columns in columns.items():
                    if table_columns:
                        lines.append(f"- {table_name}: {', '.join(table_columns)}")
            else:
                for column in columns:
                    if isinstance(column, dict):
                        column_name = column.get("name", "")
                    else:
                        column_name = str(column)
                    if column_name:
                        lines.append(f"- {column_name}")

        if join_relationships:
            if lines:
                lines.append("")
            lines.append("Join Relationships:")
            for relationship in join_relationships:
                lines.append(f"- {relationship}")

        if filters_constraints:
            if lines:
                lines.append("")
            lines.append("Filters / Constraints:")
            for item in filters_constraints:
                lines.append(f"- {item}")

        if question_intent:
            if lines:
                lines.append("")
            lines.append("Question Intent:")
            lines.append(f"- {question_intent}")

        return "\n".join(lines) if lines else "(no schema predicted)"

    def _format_few_shot_examples(self, candidates: List[Dict[str, str]], limit: int = 3) -> str:
        """Format a small few-shot block from candidate examples."""
        if not candidates:
            return "(none provided)"

        selected_examples = candidates[:limit]
        formatted_examples = []
        for example in selected_examples:
            formatted_examples.append(
                f"Question: {example['question']}\nSQL: {example['sql']}"
            )
        return "\n\n".join(formatted_examples)

    def _format_structural_examples(self, results: List[Dict[str, Any]]) -> str:
        """Format structurally similar SQL retrieval results for the round-2 prompt."""
        if not results:
            return "(none provided)"

        formatted_examples = []
        for result in results:
            example = result["example"]
            formatted_examples.append(
                f"Question: {example['question']}\n"
                f"SQL: {example['sql']}\n"
                f"SQL Skeleton: {result['candidate_sql_skeleton']}"
            )
        return "\n\n".join(formatted_examples)
    
    def _clean_sql_output(self, sql: str) -> str:
        """Clean up the SQL output from the LLM."""
        if not sql:
            return ""

        for prefix in ("SQL:", "SQLite:", "Query:", "Answer:"):
            if sql.upper().startswith(prefix.upper()):
                sql = sql[len(prefix):].strip()
                break

        # Remove any markdown code block markers
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
            
        # Remove any explanatory text before/after the SQL
        lines = sql.strip().split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that look like explanations
            if not line.startswith("#") and not line.startswith("--") or "SELECT" in line.upper() or "WITH" in line.upper():
                cleaned_lines.append(line)
                
        # Join lines and take only the first SQL statement if there are multiple
        cleaned_sql = ' '.join(cleaned_lines).strip()
        # Find the start of the SQL statement
        select_idx = cleaned_sql.upper().find("SELECT")
        with_idx = cleaned_sql.upper().find("WITH")
        
        if select_idx != -1 and (with_idx == -1 or select_idx < with_idx):
            cleaned_sql = cleaned_sql[select_idx:]
        elif with_idx != -1:
            cleaned_sql = cleaned_sql[with_idx:]
            
        # Take only the first statement (up to semicolon)
        if ";" in cleaned_sql:
            cleaned_sql = cleaned_sql.split(";")[0] + ";"

        cleaned_sql = cleaned_sql.strip()
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

        return cleaned_sql
            
    def shutdown(self):
        """Shut down the schema linker and question extractor to free resources."""
        if hasattr(self, 'schema_linker'):
            self.schema_linker.shutdown()
        if hasattr(self, 'q_extractor') and self.q_extractor is not None:
            self.q_extractor.shutdown()


# Convenience function for easy usage
def create_solidsql_system(
    candidate_examples: Optional[List[Dict[str, str]]] = None,
    base_model: str = "openai/gpt-oss-20b",
    adapter_path: str = "./schema_linking_output/lora_adapter",
    build_index: bool = True,
    skip_skeleton_extraction: bool = False,
) -> SolidSQL:
    """
    Create a SolidSQL system with the specified configuration.

    Args:
        candidate_examples: List of example questions and SQL statements
        base_model: Hugging Face model name for schema linking
        adapter_path: Path to trained LoRA adapter
        build_index: Whether to build skeleton index immediately
        skip_skeleton_extraction: Whether to skip question skeleton extraction (for evaluation with pre-built indices)

    Returns:
        Initialized SolidSQL system
    """
    return SolidSQL(
        candidate_examples=candidate_examples,
        base_model=base_model,
        adapter_path=adapter_path,
        build_index=build_index,
        skip_skeleton_extraction=skip_skeleton_extraction,
    )


# Example usage
if __name__ == "__main__":
    # Example of how to use the complete SolidSQL system
    print("=== SolidSQL End-to-End System Demo ===")

    # Sample candidate examples
    examples = [
        {
            "question": "How many singers are older than 20?",
            "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20",
        },
        {
            "question": "What is the average salary of employees?",
            "sql": "SELECT AVG(Salary) FROM Employee",
        },
        {
            "question": "List the names of students who scored above 90",
            "sql": "SELECT Name FROM Student WHERE Score > 90",
        },
    ]

    # Create SolidSQL system
    solidsql = SolidSQL(candidate_examples=examples, build_index=True)

    # Generate SQL for a new question
    result = solidsql.generate_sql(
        question="How many actors are younger than 30?",
        schema_text="Actor(id, name, age)\nMovie(id, actor_id, title)",
        top_n=3,
    )

    print("Generated result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Clean up
    solidsql.shutdown()

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
        # Initialize core components
        self.schema_linker = SchemaLinker(
            base_model=base_model,
            adapter_path=adapter_path,
        )

        # Initialize skeleton components
        self.skip_skeleton_extraction = skip_skeleton_extraction
        if not skip_skeleton_extraction:
            self.q_extractor = QuestionSkeletonExtractor()
        else:
            self.q_extractor = None
            
        self.sql_extractor = SQLSkeletonExtractor(dialect=sql_dialect)
        self.similarity = SkeletonSimilarity(embedding_model=embedding_model)
        
        # Initialize base model for SQL generation (shared across all questions)
        self.base_model_llm = None
        self.base_model = base_model
        
        # Initialize base model for SQL generation (shared across all questions)
        self.base_model_llm = None
        self.base_model = base_model

        # Initialize retrieval system
        self.retriever = SkeletonRetriever(
            candidate_examples=candidate_examples or [],
            embedding_model=embedding_model,
            sql_dialect=sql_dialect,
        )

        # Build index if provided
        if candidate_examples and build_index:
            self.retriever.build_index(show_progress=False)

        # Store configuration
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.embedding_model = embedding_model
        self.sql_dialect = sql_dialect

    def generate_sql(
        self,
        question: str,
        schema_text: str,
        top_n: int = 5,
        round_2_refinement: bool = True,
        max_new_tokens: int = 512,
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
        # Step 1: Round 1 - Generate SQL using schema linking LLM (with LoRA)
        # This uses the same LLM but with a different prompt that generates SQL directly
        self.schema_linker.set_keep_alive(True)
        schema_result = self.schema_linker.predict(
            question=question,
            schema_text=schema_text,
            max_new_tokens=max_new_tokens,
        )
        
        # Generate SQL in round 1 using the base model without LoRA adapter
        round_1_sql = self._generate_sql_with_base_model(
            question=question,
            schema_text=schema_text,
            max_new_tokens=max_new_tokens,
        )

        # Step 2: Extract question skeleton for retrieval (if not skipped)
        question_skeleton = None
        if not self.skip_skeleton_extraction and self.q_extractor is not None:
            question_skeleton = self.q_extractor.extract(question)
        elif self.skip_skeleton_extraction and hasattr(self.schema_linker, '_llm') and self.schema_linker._llm is not None:
            # Use the SchemaLinker's LLM for question skeleton extraction
            question_skeleton = self._extract_skeleton_with_shared_llm(question)

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
            # Get the most similar examples
            examples_context = []
            for result in retrieval_results[:3]:  # Use top 3 examples
                example = result["example"]
                examples_context.append(f"Question: {example['question']}\nSQL: {example['sql']}")
            
            examples_text = "\n\n".join(examples_context)
            
            # Create prompt for round 2 SQL generation
            # This uses the base model WITHOUT LoRA adapter
            round_2_prompt = (
                "Given a natural language question, database schema, examples of similar questions with their SQL queries, "
                "and a preliminary SQL query, generate a refined SQL query to answer the question.\n\n"
                f"Examples:\n{examples_text}\n\n"
                f"Question: {question}\n\n"
                f"Database Schema:\n{schema_text}\n\n"
                f"Preliminary SQL:\n{round_1_sql}\n\n"
                "Generate the refined SQL query:\n"
            )
            
            # Generate SQL using the base model without LoRA adapter
            try:
                from vllm import SamplingParams
                
                # Use shared LLM instance
                sql_generator = self._get_base_model_llm()
                
                sampling_params = SamplingParams(
                    max_tokens=512,
                    temperature=0.1,
                    top_p=0.95,
                )
                
                # Generate SQL without LoRA adapter
                outputs = sql_generator.generate([round_2_prompt], sampling_params)
                
                if outputs:
                    refined_sql = outputs[0].outputs[0].text.strip()
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
        max_new_tokens: int = 512,
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

    ['prompt], sampling_params, lora_request=lora_request)\n        \n        if outputs:\n            skeleton = outputs[0].outputs[0].text.strip()\n            # Clean up the response (same as QuestionSkeletonExtractor)\n            skeleton = self._clean_skeleton_response(skeleton)\n            return skeleton\n        return "', 'def _format_skeleton_prompt(self, question: str) -> str:\n        "', 'Format prompt for question skeleton extraction."', '\n        from schema_linking.question_skeleton_extractor import SKELETON_EXTRACTION_PROMPT\n        return SKELETON_EXTRACTION_PROMPT.format(question=question)\n        \n    def _clean_skeleton_response(self, response: str) -> str:\n        "', 'Clean up the LLM response (same as QuestionSkeletonExtractor)."', '\n        response = response.strip()\n\n        # Remove quotes if present\n        if response.startswith(\'"\') and response.endswith(\'', '):\n            response = response[1:-1]\n        elif response.startswith("\'") and response.endswith("', '):\n            response = response[1:-1]\n\n        # Remove any trailing explanations after the skeleton\n        # (in case the model didn\'t follow instructions perfectly)\n        if "\n" in response:\n            response = response.split("', [0], '.', 'strip()\n\n        return response']

    def _extract_skeleton_with_shared_llm(self, question: str) -> str:
        """Extract question skeleton using the SchemaLinker's shared LLM."""
        # Check if we have a shared LLM from schema linking
        if hasattr(self.schema_linker, '_llm') and self.schema_linker._llm is not None:
            return self._extract_skeleton_with_llm(self.schema_linker._llm, question)
        else:
            # Fallback to regular extractor if no shared LLM
            if self.q_extractor is not None:
                return self.q_extractor.extract(question)
            return ""
            
    def _get_base_model_llm(self):
        """Get or create the shared base model LLM instance."""
        if self.base_model_llm is None:
            from vllm import LLM
            self.base_model_llm = LLM(
                model=self.base_model,
                tensor_parallel_size=1,
                max_model_len=4096,
                dtype="bfloat16",
                enforce_eager=False,
                trust_remote_code=True,
                disable_log_stats=True,
            )
        return self.base_model_llm
    
    def _generate_sql_with_base_model(self, question: str, schema_text: str, max_new_tokens: int = 512) -> str:
        """
        Generate SQL using the base model without LoRA adapter.
        
        Uses a shared LLM instance to avoid reloading for each question.
        """
        from vllm import SamplingParams
        
        # SQL generation prompt
        sql_prompt = (
            "Given a natural language question and a database schema, "
            "generate a SQL query to answer the question.\n\n"
            "Rules:\n"
            "- Use only tables and columns from the provided schema\n"
            "- Generate valid SQL that answers the question\n"
            "- Do NOT include any text outside of the SQL query\n\n"
            "Question:\n{question}\n\n"
            "Database Schema:\n{schema_text}\n\n"
            "SQL:\n"
        ).format(question=question, schema_text=schema_text)
        
        # Use shared LLM instance
        llm = self._get_base_model_llm()
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
        )
        
        # Generate SQL without LoRA adapter
        outputs = llm.generate([sql_prompt], sampling_params)
        
        if outputs:
            sql = outputs[0].outputs[0].text.strip()
            sql = self._clean_sql_output(sql)
        else:
            sql = ""
        
        # vLLM v0.19+ manages engine lifecycle automatically
        # No manual shutdown needed
        
        return sql
    
    def _clean_sql_output(self, sql: str) -> str:
        """Clean up the SQL output from the LLM."""
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
        cleaned_sql = ' '.join(cleaned_lines)
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
            
        return cleaned_sql.strip()
            
    def shutdown(self):
        """Shut down the schema linker to free resources."""
        if hasattr(self, 'schema_linker'):
            self.schema_linker.shutdown()
    
    def shutdown(self):
        """Shut down the schema linker and base model to free resources."""
        if hasattr(self, 'schema_linker'):
            self.schema_linker.shutdown()
        if hasattr(self, 'base_model_llm') and self.base_model_llm is not None:
            # vLLM v0.19+ manages lifecycle automatically
            self.base_model_llm = None
    
    def __del__(self):
        """Ensure resources are freed on deletion."""
        try:
            self.shutdown()
        except:
            pass


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

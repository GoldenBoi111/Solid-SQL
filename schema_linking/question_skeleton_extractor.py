"""
Question Skeleton Extractor

Extracts the structural skeleton (Q⋆) from natural language questions by
removing domain-specific details and values, leaving only the core query pattern.

This enables similarity matching based on question structure rather than thematic content,
improving in-context example retrieval for SQL generation.

Usage:
    from schema_linking.question_skeleton_extractor import QuestionSkeletonExtractor

    extractor = QuestionSkeletonExtractor(model_name="openai/gpt-oss-20b")

    # Extract skeleton from a single question
    skeleton = extractor.extract("How many singers are older than 20?")

    # Batch extraction
    skeletons = extractor.extract_batch([
        "How many singers are older than 20?",
        "What is the average age of actors?",
    ])
"""

import json
from pathlib import Path
from typing import List, Optional

from .config import MODEL_NAME


# Prompt template for extracting question skeleton
SKELETON_EXTRACTION_PROMPT = """Given a natural language question, extract its structural skeleton by removing domain-specific details and concrete values.

The skeleton (Q⋆) should preserve:
- The query pattern (e.g., counting, aggregation, filtering, sorting)
- The logical structure (e.g., conditions, comparisons, joins)
- The type of answer expected (e.g., number, list, name)

Replace:
- Domain entities (e.g., "singers", "albums", "actors") with generic placeholders like [ENTITY]
- Concrete values (e.g., "20", "John", "2020") with [VALUE]
- Specific attributes (e.g., "age", "name", "title") with [ATTRIBUTE]

Rules:
- Keep the sentence structure intact
- Preserve logical operators (e.g., "greater than", "equal to", "between")
- Maintain the question type (e.g., "How many", "What is", "List all")
- Do NOT add any explanation - return ONLY the skeleton

Examples:
  Input: "How many singers are older than 20?"
  Output: "How many [ENTITY] have [ATTRIBUTE] greater than [VALUE]?"

  Input: "What is the average salary of employees in the Engineering department?"
  Output: "What is the average [ATTRIBUTE] of [ENTITY] in [ATTRIBUTE] equal to [VALUE]?"

  Input: "List the names of students who scored higher than 90 in Math"
  Output: "List the [ATTRIBUTE] of [ENTITY] who have [ATTRIBUTE] greater than [VALUE] in [ENTITY]"

Question:
{question}

Skeleton (Q⋆):"""


class QuestionSkeletonExtractor:
    """Extracts structural skeletons from natural language questions using LLM."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        tensor_parallel_size: int = 1,
        max_seq_length: int = 2048,
    ):
        """
        Initialize the skeleton extractor.

        Args:
            model_name: Hugging Face model name or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_seq_length: Maximum sequence length
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_seq_length = max_seq_length

        # Deferred import to avoid vLLM dependency at module load time
        self._llm = None
        self._sampling_params = None

    def _initialize_vllm(self):
        """Lazy initialization of vLLM engine."""
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for question skeleton extraction. "
                "Install with: pip install vllm"
            )

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_seq_length,
            dtype="bfloat16",
            enforce_eager=True,
            trust_remote_code=True,
            disable_log_stats=True,
        )

        self._sampling_params = SamplingParams(
            max_tokens=256,
            temperature=0.1,
            top_p=0.95,
        )

    def _format_prompt(self, question: str) -> str:
        """Format the prompt for skeleton extraction."""
        return SKELETON_EXTRACTION_PROMPT.format(question=question)

    def extract(
        self,
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Extract the skeleton (Q⋆) from a single question.

        Args:
            question: The natural language question
            max_new_tokens: Maximum tokens to generate

        Returns:
            The extracted question skeleton
        """
        results = self.extract_batch([question], max_new_tokens=max_new_tokens)
        return results[0] if results else ""

    def extract_batch(
        self,
        questions: List[str],
        max_new_tokens: int = 256,
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Extract skeletons for multiple questions.

        Args:
            questions: List of natural language questions
            max_new_tokens: Maximum tokens to generate per question
            batch_size: Batch size for generation
            show_progress: Whether to show progress bar

        Returns:
            List of extracted question skeletons
        """
        self._initialize_vllm()

        # Update sampling params with max_new_tokens
        from vllm import SamplingParams
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
        )

        # Format all prompts
        prompts = [self._format_prompt(q) for q in questions]

        # Generate in sub-batches for memory efficiency
        all_results = []
        total = len(prompts)

        for i in range(0, total, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            outputs = self._llm.generate(batch_prompts, self._sampling_params)

            for output in outputs:
                skeleton = output.outputs[0].text.strip()
                # Clean up the response
                skeleton = self._clean_response(skeleton)
                all_results.append(skeleton)

            if show_progress:
                processed = min(i + batch_size, total)
                print(f"  Processed {processed}/{total}")

        return all_results

    @staticmethod
    def _clean_response(response: str) -> str:
        """Clean up the LLM response."""
        response = response.strip()

        # Remove quotes if present
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        elif response.startswith("'") and response.endswith("'"):
            response = response[1:-1]

        # Remove any trailing explanations after the skeleton
        # (in case the model didn't follow instructions perfectly)
        if "\n" in response:
            response = response.split("\n")[0].strip()

        return response

    def shutdown(self):
        """Shut down the vLLM engine and free VRAM."""
        if self._llm is not None:
            if hasattr(self._llm, "shutdown"):
                self._llm.shutdown()
            elif hasattr(self._llm, "llm_engine"):
                self._llm.llm_engine.shutdown()
            self._llm = None
            self._sampling_params = None

    def __del__(self):
        """Ensure vLLM engine is shut down on deletion."""
        try:
            self.shutdown()
        except:
            pass

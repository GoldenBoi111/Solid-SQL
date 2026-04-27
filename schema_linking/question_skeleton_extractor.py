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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME
from .inference import _resolve_generation_model


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
    """Extracts structural skeletons from natural language questions using transformers."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_seq_length: int = 2048,
        shared_model: object = None,
        shared_tokenizer: object = None,
    ):
        """
        Initialize the skeleton extractor.

        Args:
            model_name: Hugging Face model name or local path
            max_seq_length: Maximum sequence length
            shared_model: Optional shared model instance to reuse
                (can be an Outlines wrapper or raw transformers model)
            shared_tokenizer: Optional shared tokenizer instance to reuse
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self._owns_model = False
        if shared_model is not None:
            if hasattr(shared_model, "model"):
                self._model = shared_model.model  # unwrap Outlines wrapper
            else:
                self._model = shared_model
        else:
            self._model = None

        if shared_tokenizer is not None and hasattr(shared_tokenizer, "tokenizer"):
            self._tokenizer = shared_tokenizer.tokenizer
        else:
            self._tokenizer = shared_tokenizer

    def _load_model(self):
        """Load the base model and tokenizer if not already loaded."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._model.eval()
            self._owns_model = True

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
        self._load_model()

        # Format all prompts
        prompts = [self._format_prompt(q) for q in questions]

        # Tokenize
        inputs = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        model = _resolve_generation_model(self._model)
        inputs = inputs.to(next(model.parameters()).device)

        # Generate
        all_results = []
        total = len(prompts)

        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_inputs = {
                    k: v[i : i + batch_size] for k, v in inputs.items()
                }
                
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.95,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
                
                batch_attention = batch_inputs["attention_mask"].sum(dim=1).tolist()
                for output, prompt_length in zip(outputs, batch_attention):
                    generated_tokens = output[int(prompt_length):]
                    skeleton = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
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
        """Shut down the model and free VRAM."""
        if self._model is not None and self._owns_model:
            del self._model
            self._model = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
        self._owns_model = False

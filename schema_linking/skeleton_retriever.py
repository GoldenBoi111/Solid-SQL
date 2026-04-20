"""
Skeleton-Based Example Retriever

Retrieves the top-N most similar examples from a candidate library based on:
1. Question skeleton similarity (cosine similarity on embeddings)
2. SQL skeleton similarity (edit distance on parse trees)

Supports both retrieval modes:
- Question-based: Find examples with similar question structure
- SQL-based: Find examples with similar SQL structure (for round-2 refinement)

Usage:
    from schema_linking.skeleton_retriever import SkeletonRetriever

    # Initialize retriever
    retriever = SkeletonRetriever(candidate_examples=[
        {"question": "How many singers are older than 20?", "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20"},
        {"question": "What is the average salary?", "sql": "SELECT AVG(salary) FROM Employees"},
    ])

    # Extract skeletons for all examples (done once during initialization)
    retriever.build_index()

    # Retrieve top-N similar examples for a new question
    results = retriever.retrieve_by_question(
        question="How many actors are younger than 30?",
        top_n=2,
    )

    # Retrieve top-N similar examples for a generated SQL
    results = retriever.retrieve_by_sql(
        sql="SELECT COUNT(*) FROM Actor WHERE age < 30",
        top_n=2,
    )
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from .question_skeleton_extractor import QuestionSkeletonExtractor
from .sql_skeleton_extractor import SQLSkeletonExtractor
from .skeleton_similarity import SkeletonSimilarity


class SkeletonRetriever:
    """
    Retrieves similar examples from a candidate library using skeleton-based similarity.

    Supports two retrieval modes:
    1. Question skeleton-based: Matches question structure (Q⋆)
    2. SQL skeleton-based: Matches SQL structure (S⋆)
    """

    def __init__(
        self,
        candidate_examples: List[Dict[str, str]],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sql_dialect: str = "sqlite",
    ):
        """
        Initialize the retriever.

        Args:
            candidate_examples: List of dicts with "question" and "sql" keys
            embedding_model: Sentence transformer model for question embeddings
            sql_dialect: SQL dialect for parsing
        """
        self.candidates = candidate_examples
        self.question_skeletons = []
        self.sql_skeletons = []

        # Initialize extractors and similarity calculator
        self.q_extractor = QuestionSkeletonExtractor()
        self.sql_extractor = SQLSkeletonExtractor(dialect=sql_dialect)
        self.similarity = SkeletonSimilarity(embedding_model=embedding_model)

    def build_index(
        self,
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> None:
        """
        Pre-compute skeletons for all candidate examples.

        This should be called once during initialization to build the
        example index for fast retrieval.

        Args:
            batch_size: Batch size for skeleton extraction
            show_progress: Whether to show progress
        """
        if not self.candidates:
            print("Warning: No candidate examples provided")
            return

        questions = [ex["question"] for ex in self.candidates]
        sqls = [ex["sql"] for ex in self.candidates]

        if show_progress:
            print(f"Extracting question skeletons for {len(questions)} examples...")
        self.question_skeletons = self.q_extractor.extract_batch(
            questions,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        if show_progress:
            print(f"Extracting SQL skeletons for {len(sqls)} examples...")
        self.sql_skeletons = self.sql_extractor.extract_batch(
            sqls,
            show_progress=show_progress,
        )

        if show_progress:
            print(f"Index built with {len(self.question_skeletons)} question skeletons "
                  f"and {len(self.sql_skeletons)} SQL skeletons")

    def retrieve_by_question(
        self,
        question: str,
        top_n: int = 5,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-N most similar examples based on question skeleton.

        Args:
            question: The target natural language question
            top_n: Number of examples to retrieve
            show_progress: Whether to show progress

        Returns:
            List of dicts with example, similarity score, and skeletons
        """
        if not self.question_skeletons:
            raise ValueError(
                "Skeleton index not built. Call build_index() first."
            )

        # Extract skeleton from target question
        if show_progress:
            print(f"Extracting question skeleton for target...")
        target_skeleton = self.q_extractor.extract(question)

        # Calculate similarities
        similarities = self.similarity.question_similarity_batch(
            target_skeleton,
            self.question_skeletons,
        )

        # Rank and return top-N
        return self._rank_and_return(similarities, top_n, target_skeleton, "question")

    def retrieve_by_sql(
        self,
        sql: str,
        top_n: int = 5,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-N most similar examples based on SQL skeleton.

        This is useful for round-2 SQL generation refinement.

        Args:
            sql: The generated SQL statement
            top_n: Number of examples to retrieve
            show_progress: Whether to show progress

        Returns:
            List of dicts with example, similarity score, and skeletons
        """
        if not self.sql_skeletons:
            raise ValueError(
                "Skeleton index not built. Call build_index() first."
            )

        # Extract skeleton from target SQL
        if show_progress:
            print(f"Extracting SQL skeleton for target...")
        target_skeleton = self.sql_extractor.extract(sql)

        # Calculate similarities
        similarities = self.similarity.sql_similarity_batch(
            target_skeleton,
            self.sql_skeletons,
        )

        # Rank and return top-N
        return self._rank_and_return(similarities, top_n, target_skeleton, "sql")

    def _rank_and_return(
        self,
        similarities: List[float],
        top_n: int,
        target_skeleton: str,
        skeleton_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Rank examples by similarity and return top-N.

        Args:
            similarities: List of similarity scores
            top_n: Number of examples to return
            target_skeleton: The target skeleton (for reference)
            skeleton_type: Either "question" or "sql"

        Returns:
            List of ranked examples with metadata
        """
        # Create (index, similarity) pairs
        indexed_sims = list(enumerate(similarities))

        # Sort by similarity (descending)
        indexed_sims.sort(key=lambda x: x[1], reverse=True)

        # Take top-N
        top_indices = indexed_sims[:top_n]

        # Build result list
        results = []
        for idx, sim_score in top_indices:
            example = self.candidates[idx]
            result = {
                "example": example,
                "similarity_score": sim_score,
                "candidate_question_skeleton": self.question_skeletons[idx],
                "candidate_sql_skeleton": self.sql_skeletons[idx],
            }
            results.append(result)

        return results

    def save_index(self, path: str) -> None:
        """
        Save the built index to disk for reuse.

        Args:
            path: Path to save the index (JSON file)
        """
        if not self.question_skeletons or not self.sql_skeletons:
            raise ValueError("No index to save. Call build_index() first.")

        index_data = {
            "candidates": self.candidates,
            "question_skeletons": self.question_skeletons,
            "sql_skeletons": self.sql_skeletons,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

        print(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        """
        Load a previously saved index from disk.

        Args:
            path: Path to the saved index (JSON file)
        """
        with open(path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.candidates = index_data["candidates"]
        self.question_skeletons = index_data["question_skeletons"]
        self.sql_skeletons = index_data["sql_skeletons"]

        print(f"Index loaded from {path} with {len(self.candidates)} examples")

    def shutdown(self):
        """Shut down vLLM engine and free resources."""
        self.q_extractor.shutdown()

    def __del__(self):
        """Ensure resources are freed on deletion."""
        try:
            self.shutdown()
        except:
            pass


def load_candidate_library_from_json(path: str) -> List[Dict[str, str]]:
    """
    Load candidate examples from a JSON file.

    Expected format:
    [
        {"question": "...", "sql": "...", ...},
        {"question": "...", "sql": "...", ...},
    ]

    Args:
        path: Path to JSON file

    Returns:
        List of candidate examples
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate format
    for i, entry in enumerate(data):
        if "question" not in entry or "sql" not in entry:
            raise ValueError(
                f"Entry {i} missing 'question' or 'sql' field. "
                f"Found keys: {list(entry.keys())}"
            )

    return data


def load_candidate_library_from_spider(
    train_path: str = "./train.json",
    max_examples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Load candidate examples from Spider dataset format.

    Args:
        train_path: Path to Spider train.json
        max_examples: Maximum number of examples to load (None for all)

    Returns:
        List of candidate examples with "question" and "sql" keys
    """
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    for entry in data:
        candidates.append({
            "question": entry.get("question", ""),
            "sql": entry.get("query", ""),
            "db_id": entry.get("db_id", ""),
        })
        if max_examples and len(candidates) >= max_examples:
            break

    return candidates
